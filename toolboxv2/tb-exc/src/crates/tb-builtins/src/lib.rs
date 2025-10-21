//! TB Language Built-in Functions
//!
//! High-performance, non-blocking built-in functions for:
//! - File I/O (real files only)
//! - Networking (HTTP/HTTPS, TCP, UDP)
//! - Utilities (JSON/YAML, time)
//!
//! All I/O operations are async and non-blocking for maximum efficiency.

pub mod file_io;
pub mod networking;
pub mod utils;
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

/// open(path: str, mode: str, key: str = None, encoding: str = "utf-8") -> FileHandle
fn builtin_open(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 5 {
        return Err(TBError::runtime_error("open() takes 1-4 arguments: path, mode, key, encoding"));
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("open() path must be a string")),
    };

    let mode = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.to_string(),
            _ => "r".to_string(),
        }
    } else {
        "r".to_string()
    };

    let key = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.to_string()),
            Value::None => None,
            _ => return Err(TBError::runtime_error("open() key must be a string or None")),
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
        file_io::open_file(path, mode, key, encoding).await
    })?;

    Ok(Value::String(Arc::new(handle)))
}

/// read_file(path: str) -> str
fn builtin_read_file(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 1 {
        return Err(TBError::runtime_error("read_file() takes 1 argument: path"));
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("read_file() path must be a string")),
    };

    let content = RUNTIME.block_on(async {
        file_io::read_file(path, false).await
    })?;

    Ok(Value::String(Arc::new(content)))
}

/// write_file(path: str, content: str) -> None
fn builtin_write_file(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("write_file() takes 2 arguments: path, content"));
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("write_file() path must be a string")),
    };

    let content = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("write_file() content must be a string")),
    };

    RUNTIME.block_on(async {
        file_io::write_file(path, content, false).await
    })?;

    Ok(Value::None)
}

/// file_exists(path: str) -> bool
fn builtin_file_exists(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("file_exists() takes 1 argument: path"));
    }

    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("file_exists() path must be a string")),
    };

    let exists = RUNTIME.block_on(async {
        file_io::file_exists(path, false).await
    })?;

    Ok(Value::Bool(exists))
}

