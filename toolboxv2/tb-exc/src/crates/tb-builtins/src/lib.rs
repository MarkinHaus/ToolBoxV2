//! TB Language Built-in Functions
//!
//! High-performance, non-blocking built-in functions for:
//! - File I/O (real files + blob storage)
//! - Networking (HTTP/HTTPS, TCP, UDP)
//! - Utilities (JSON/YAML, time)
//!
//! All I/O operations are async and non-blocking for maximum efficiency.

pub mod file_io;
pub mod networking;
pub mod utils;
pub mod blob;
pub mod error;
pub mod builtins_impl;

use std::sync::Arc;
use tb_core::{Value, TBError};
use dashmap::DashMap;
use once_cell::sync::Lazy;

pub use error::{BuiltinError, BuiltinResult};

/// Global runtime for async operations
pub static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("tb-builtin-worker")
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
});

/// Global blob storage registry
pub static BLOB_STORAGES: Lazy<DashMap<String, Arc<blob::BlobStorage>>> = Lazy::new(DashMap::new);

/// Global network server registry
pub static NETWORK_SERVERS: Lazy<DashMap<String, Arc<networking::ServerHandle>>> = Lazy::new(DashMap::new);

/// Global HTTP session registry
pub static HTTP_SESSIONS: Lazy<DashMap<String, Arc<networking::HttpSession>>> = Lazy::new(DashMap::new);

/// Register all built-in functions
pub fn register_all_builtins() -> Vec<(&'static str, BuiltinFn)> {
    let mut builtins = Vec::new();

    // File I/O
    builtins.push(("open", builtin_open as BuiltinFn));
    builtins.push(("read_file", builtin_read_file as BuiltinFn));
    builtins.push(("write_file", builtin_write_file as BuiltinFn));
    builtins.push(("file_exists", builtin_file_exists as BuiltinFn));

    // Blob storage
    builtins.push(("blob_init", builtin_blob_init as BuiltinFn));
    builtins.push(("blob_create", builtin_blob_create as BuiltinFn));
    builtins.push(("blob_read", builtin_blob_read as BuiltinFn));
    builtins.push(("blob_update", builtin_blob_update as BuiltinFn));
    builtins.push(("blob_delete", builtin_blob_delete as BuiltinFn));

    // Networking
    builtins.push(("create_server", builtins_impl::builtin_create_server as BuiltinFn));
    builtins.push(("connect_to", builtins_impl::builtin_connect_to as BuiltinFn));
    builtins.push(("send_to", builtins_impl::builtin_send_to as BuiltinFn));
    builtins.push(("http_session", builtins_impl::builtin_http_session as BuiltinFn));
    builtins.push(("http_request", builtins_impl::builtin_http_request as BuiltinFn));

    // Utilities
    builtins.push(("json_parse", builtins_impl::builtin_json_parse as BuiltinFn));
    builtins.push(("json_stringify", builtins_impl::builtin_json_stringify as BuiltinFn));
    builtins.push(("yaml_parse", builtins_impl::builtin_yaml_parse as BuiltinFn));
    builtins.push(("yaml_stringify", builtins_impl::builtin_yaml_stringify as BuiltinFn));
    builtins.push(("time", builtins_impl::builtin_time as BuiltinFn));

    builtins
}

/// Built-in function signature
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, TBError>;

// ============================================================================
// FILE I/O BUILT-INS
// ============================================================================

/// open(path: str, mode: str, blob: bool = false, key: str = None, encoding: str = "utf-8") -> FileHandle
fn builtin_open(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 5 {
        return Err(TBError::RuntimeError {
            message: "open() takes 1-5 arguments: path, mode, blob, key, encoding".to_string(),
        });
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "open() path must be a string".to_string(),
        }),
    };

    let mode = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.to_string(),
            _ => "r".to_string(),
        }
    } else {
        "r".to_string()
    };

    let is_blob = if args.len() > 2 {
        match &args[2] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    let key = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.to_string()),
            Value::None => None,
            _ => return Err(TBError::RuntimeError {
                message: "open() key must be a string or None".to_string(),
            }),
        }
    } else {
        None
    };

    let encoding = if args.len() > 4 {
        match &args[4] {
            Value::String(s) => s.to_string(),
            _ => "utf-8".to_string(),
        }
    } else {
        "utf-8".to_string()
    };

    let handle = RUNTIME.block_on(async {
        file_io::open_file(path, mode, is_blob, key, encoding).await
    })?;

    Ok(Value::String(Arc::new(handle)))
}

/// read_file(path: str, blob: bool = false) -> str
fn builtin_read_file(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 2 {
        return Err(TBError::RuntimeError {
            message: "read_file() takes 1-2 arguments: path, blob".to_string(),
        });
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "read_file() path must be a string".to_string(),
        }),
    };

    let is_blob = if args.len() > 1 {
        match &args[1] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    let content = RUNTIME.block_on(async {
        file_io::read_file(path, is_blob).await
    })?;

    Ok(Value::String(Arc::new(content)))
}

/// write_file(path: str, content: str, blob: bool = false) -> None
fn builtin_write_file(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(TBError::RuntimeError {
            message: "write_file() takes 2-3 arguments: path, content, blob".to_string(),
        });
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "write_file() path must be a string".to_string(),
        }),
    };

    let content = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "write_file() content must be a string".to_string(),
        }),
    };

    let is_blob = if args.len() > 2 {
        match &args[2] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    RUNTIME.block_on(async {
        file_io::write_file(path, content, is_blob).await
    })?;

    Ok(Value::None)
}

/// file_exists(path: str, blob: bool = false) -> bool
fn builtin_file_exists(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 2 {
        return Err(TBError::RuntimeError {
            message: "file_exists() takes 1-2 arguments: path, blob".to_string(),
        });
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "file_exists() path must be a string".to_string(),
        }),
    };

    let is_blob = if args.len() > 1 {
        match &args[1] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    let exists = RUNTIME.block_on(async {
        file_io::file_exists(path, is_blob).await
    })?;

    Ok(Value::Bool(exists))
}

// ============================================================================
// BLOB STORAGE BUILT-INS
// ============================================================================

/// blob_init(servers: list[str], storage_dir: str = "./.data/blob_cache") -> str (storage_id)
fn builtin_blob_init(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 2 {
        return Err(TBError::RuntimeError {
            message: "blob_init() takes 1-2 arguments: servers, storage_dir".to_string(),
        });
    }

    let servers = match &args[0] {
        Value::List(list) => {
            list.iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.to_string()),
                    _ => Err(TBError::RuntimeError {
                        message: "blob_init() servers must be a list of strings".to_string(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err(TBError::RuntimeError {
            message: "blob_init() servers must be a list".to_string(),
        }),
    };

    let storage_dir = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.to_string(),
            _ => "./.data/blob_cache".to_string(),
        }
    } else {
        "./.data/blob_cache".to_string()
    };

    let storage_id = blob::init_blob_storage(servers, storage_dir)?;
    Ok(Value::String(Arc::new(storage_id)))
}

/// blob_create(storage_id: str, data: str, blob_id: str = None) -> str
fn builtin_blob_create(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(TBError::RuntimeError {
            message: "blob_create() takes 2-3 arguments: storage_id, data, blob_id".to_string(),
        });
    }

    let storage_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_create() storage_id must be a string".to_string(),
        }),
    };

    let data = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_create() data must be a string".to_string(),
        }),
    };

    let blob_id = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.to_string()),
            Value::None => None,
            _ => return Err(TBError::RuntimeError {
                message: "blob_create() blob_id must be a string or None".to_string(),
            }),
        }
    } else {
        None
    };

    let storage = BLOB_STORAGES.get(&storage_id)
        .ok_or_else(|| TBError::RuntimeError {
            message: format!("Blob storage not found: {}", storage_id),
        })?;

    let result_id = RUNTIME.block_on(async {
        storage.create_blob(&data, blob_id).await
    })?;

    Ok(Value::String(Arc::new(result_id)))
}

/// blob_read(storage_id: str, blob_id: str) -> str
fn builtin_blob_read(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::RuntimeError {
            message: "blob_read() takes 2 arguments: storage_id, blob_id".to_string(),
        });
    }

    let storage_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_read() storage_id must be a string".to_string(),
        }),
    };

    let blob_id = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_read() blob_id must be a string".to_string(),
        }),
    };

    let storage = BLOB_STORAGES.get(&storage_id)
        .ok_or_else(|| TBError::RuntimeError {
            message: format!("Blob storage not found: {}", storage_id),
        })?;

    let data = RUNTIME.block_on(async {
        storage.read_blob(&blob_id).await
    })?;

    let content = String::from_utf8(data)
        .map_err(|e| TBError::RuntimeError {
            message: format!("Invalid UTF-8 in blob: {}", e),
        })?;

    Ok(Value::String(Arc::new(content)))
}

/// blob_update(storage_id: str, blob_id: str, data: str) -> None
fn builtin_blob_update(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 3 {
        return Err(TBError::RuntimeError {
            message: "blob_update() takes 3 arguments: storage_id, blob_id, data".to_string(),
        });
    }

    let storage_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_update() storage_id must be a string".to_string(),
        }),
    };

    let blob_id = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_update() blob_id must be a string".to_string(),
        }),
    };

    let data = match &args[2] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_update() data must be a string".to_string(),
        }),
    };

    let storage = BLOB_STORAGES.get(&storage_id)
        .ok_or_else(|| TBError::RuntimeError {
            message: format!("Blob storage not found: {}", storage_id),
        })?;

    RUNTIME.block_on(async {
        storage.update_blob(&blob_id, &data).await
    })?;

    Ok(Value::None)
}

/// blob_delete(storage_id: str, blob_id: str) -> None
fn builtin_blob_delete(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::RuntimeError {
            message: "blob_delete() takes 2 arguments: storage_id, blob_id".to_string(),
        });
    }

    let storage_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_delete() storage_id must be a string".to_string(),
        }),
    };

    let blob_id = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "blob_delete() blob_id must be a string".to_string(),
        }),
    };

    let storage = BLOB_STORAGES.get(&storage_id)
        .ok_or_else(|| TBError::RuntimeError {
            message: format!("Blob storage not found: {}", storage_id),
        })?;

    RUNTIME.block_on(async {
        storage.delete_blob(&blob_id).await
    })?;

    Ok(Value::None)
}

