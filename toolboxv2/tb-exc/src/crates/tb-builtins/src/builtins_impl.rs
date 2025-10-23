//! Implementation of remaining built-in functions

use crate::*;
use crate::error::BuiltinError;
use std::sync::Arc;
use tb_core::{Value, TBError, Program};
use std::collections::HashMap;
use im::HashMap as ImHashMap;
use tokio::process::Command;
use sha2::{Sha256, Sha512, Sha224, Sha384, Digest};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Convert TB Value to serde_json::Value
fn convert_tb_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::None => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::Value::Number((*i).into()),
        Value::Float(f) => {
            serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::List(items) => {
            let arr: Vec<serde_json::Value> = items.iter().map(convert_tb_to_json).collect();
            serde_json::Value::Array(arr)
        }
        Value::Dict(map) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in map.iter() {
                obj.insert(k.to_string(), convert_tb_to_json(v));
            }
            serde_json::Value::Object(obj)
        }
        _ => serde_json::Value::Null,
    }
}

// ============================================================================
// FILE I/O BUILT-INS
// ============================================================================

/// open(path: str, mode: str, key: str = None, encoding: str = "utf-8") -> FileHandle
pub fn builtin_open(args: Vec<Value>) -> Result<Value, TBError> {
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
pub fn builtin_read_file(args: Vec<Value>) -> Result<Value, TBError> {
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
pub fn builtin_write_file(args: Vec<Value>) -> Result<Value, TBError> {
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
pub fn builtin_file_exists(args: Vec<Value>) -> Result<Value, TBError> {
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


// ============================================================================
// INTROSPEKTION
// ============================================================================

/// type_of(value) -> str
pub fn builtin_type_of(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("type_of() takes 1 argument"));
    }
    Ok(Value::String(Arc::new(args[0].type_name().to_string())))
}

/// dir(object) -> list[str]
pub fn builtin_dir(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("dir() takes 1 argument"));
    }

    match &args[0] {
        Value::Dict(map) => {
            let keys: Vec<Value> = map.keys()
                .map(|k| Value::String(Arc::clone(k)))
                .collect();
            Ok(Value::List(Arc::new(keys)))
        }
        _ => Ok(Value::List(Arc::new(vec![]))) // Leere Liste für andere Typen
    }
}

/// has_attr(object, name: str) -> bool
pub fn builtin_has_attr(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("has_attr() takes 2 arguments"));
    }

    let attr_name = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TBError::runtime_error("Attribute name must be a string")),
    };

    match &args[0] {
        Value::Dict(map) => Ok(Value::Bool(map.contains_key(attr_name))),
        _ => Ok(Value::Bool(false)),
    }
}

// ============================================================================
// SYSTEM & I/O
// ============================================================================

pub fn builtin_list_dir(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("list_dir() takes 1 argument: path"));
    }
    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("path must be a string")),
    };

    let entries = RUNTIME.block_on(async {
        file_io::list_dir(path).await
    })?;

    let values: Vec<Value> = entries.into_iter().map(|s| Value::String(Arc::new(s))).collect();
    Ok(Value::List(Arc::new(values)))
}

pub fn builtin_create_dir(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() < 1 || args.len() > 2 {
        return Err(TBError::runtime_error("create_dir() takes 1-2 arguments: path, recursive=false"));
    }
    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("path must be a string")),
    };
    let recursive = if args.len() == 2 {
        match &args[1] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    RUNTIME.block_on(async {
        file_io::create_dir(path, recursive).await
    })?;

    Ok(Value::None)
}

pub fn builtin_delete_file(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("delete_file() takes 1 argument: path"));
    }
    let path = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("path must be a string")),
    };

    RUNTIME.block_on(async {
        file_io::delete_file(path, false).await
    })?;

    Ok(Value::None)
}

pub fn builtin_execute(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() {
        return Err(TBError::runtime_error("execute() requires at least one argument for the command"));
    }

    let command_args: Vec<String> = args.iter().map(|v| v.to_string()).collect();
    let (command, args) = command_args.split_first().unwrap();

    let output = RUNTIME.block_on(async {
        Command::new(command)
            .args(args)
            .output()
            .await
    })?;

    let mut result_map = ImHashMap::new();
    result_map.insert(Arc::new("status".to_string()), Value::Int(output.status.code().unwrap_or(-1) as i64));
    result_map.insert(Arc::new("stdout".to_string()), Value::String(Arc::new(String::from_utf8_lossy(&output.stdout).to_string())));
    result_map.insert(Arc::new("stderr".to_string()), Value::String(Arc::new(String::from_utf8_lossy(&output.stderr).to_string())));

    Ok(Value::Dict(Arc::new(result_map)))
}

pub fn builtin_get_env(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("get_env() takes 1 argument: var_name"));
    }
    let var_name = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("var_name must be a string")),
    };

    match std::env::var(var_name) {
        Ok(val) => Ok(Value::String(Arc::new(val))),
        Err(_) => Ok(Value::None),
    }
}

pub fn builtin_sleep(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("sleep() takes 1 argument: seconds"));
    }
    let seconds = match &args[0] {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        _ => return Err(TBError::runtime_error("seconds must be a number")),
    };

    RUNTIME.block_on(async {
        tokio::time::sleep(tokio::time::Duration::from_secs_f64(seconds)).await;
    });

    Ok(Value::None)
}


// ============================================================================
// CACHE MANAGEMENT
// ============================================================================

/// cache_stats() -> dict
pub fn builtin_cache_stats(_args: Vec<Value>) -> Result<Value, TBError> {
    let mut stats_map = ImHashMap::new();

    // Hot Cache
    let hot_stats = CACHE_MANAGER.stats();
    let mut hot_map = ImHashMap::new();
    hot_map.insert(Arc::new("entries".to_string()), Value::Int(hot_stats.hot_cache_entries as i64));
    hot_map.insert(Arc::new("size_bytes".to_string()), Value::Int(hot_stats.hot_cache_bytes as i64));
    stats_map.insert(Arc::new("hot_cache".to_string()), Value::Dict(Arc::new(hot_map)));

    // Import Cache
    let import_stats = CACHE_MANAGER.import_cache().stats();
    let mut import_map = ImHashMap::new();
    import_map.insert(Arc::new("entries".to_string()), Value::Int(import_stats.entries as i64));
    import_map.insert(Arc::new("size_bytes".to_string()), Value::Int(import_stats.total_bytes as i64));
    stats_map.insert(Arc::new("import_cache".to_string()), Value::Dict(Arc::new(import_map)));

    Ok(Value::Dict(Arc::new(stats_map)))
}

/// cache_clear(cache_name: str = "all")
pub fn builtin_cache_clear(args: Vec<Value>) -> Result<Value, TBError> {
    let cache_name = if args.is_empty() {
        "all".to_string()
    } else if let Value::String(s) = &args[0] {
        s.to_string()
    } else {
        return Err(TBError::runtime_error("cache_clear() argument must be a string"));
    };

    match cache_name.as_str() {
        "hot" => CACHE_MANAGER.clear_hot_cache(),
        "imports" => CACHE_MANAGER.import_cache().clear()?,
        // "artifacts" => CACHE_MANAGER.artifact_cache().clear()?, // Annahme: clear() wird implementiert
        "all" => {
            CACHE_MANAGER.clear_hot_cache();
            CACHE_MANAGER.import_cache().clear()?;
            // CACHE_MANAGER.artifact_cache().clear()?;
        }
        _ => return Err(TBError::runtime_error(format!("Unknown cache: {}", cache_name))),
    }

    Ok(Value::Bool(true))
}

/// cache_invalidate(module_path: str)
pub fn builtin_cache_invalidate(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() {
        return Err(TBError::runtime_error("cache_invalidate() requires a module path"));
    }
    let path = match &args[0] {
        Value::String(s) => std::path::PathBuf::from(s.as_ref()),
        _ => return Err(TBError::runtime_error("Path must be a string")),
    };

    CACHE_MANAGER.import_cache().invalidate(&path)?;
    Ok(Value::Bool(true))
}


// ============================================================================
// PLUGIN MANAGEMENT
// ============================================================================

/// list_plugins() -> list[str]
pub fn builtin_list_plugins(_args: Vec<Value>) -> Result<Value, TBError> {
    let loaded = PLUGIN_LOADER.list_loaded_libraries(); // Annahme: diese Methode wird implementiert
    let values: Vec<Value> = loaded.into_iter().map(|s| Value::String(Arc::new(s))).collect();
    Ok(Value::List(Arc::new(values)))
}

/// unload_plugin(name: str) -> bool
pub fn builtin_unload_plugin(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() { return Err(TBError::runtime_error("unload_plugin() requires a plugin name")); }
    let name = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("Plugin name must be a string")),
    };

    let result = PLUGIN_LOADER.unload_library(std::path::Path::new(&name));
    Ok(Value::Bool(result))
}

/// reload_plugin(name: str) -> bool
pub fn builtin_reload_plugin(args: Vec<Value>) -> Result<Value, TBError> {
    // Implementierung: zuerst unload, dann load.
    // Hier ist Vorsicht geboten, um Race Conditions zu vermeiden.
    // Fürs Erste:
    let name_val = args.get(0).ok_or_else(|| TBError::runtime_error("reload_plugin requires a name"))?.clone();
    builtin_unload_plugin(vec![name_val])?;
    // Der nächste Aufruf einer Funktion aus dem Plugin wird es neu laden (Lazy Loading).
    Ok(Value::Bool(true))
}

/// plugin_info(name: str) -> dict
pub fn builtin_plugin_info(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() {
        return Err(TBError::runtime_error("plugin_info() requires a plugin name"));
    }

    let plugin_name = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(TBError::runtime_error("Plugin name must be a string")),
    };

    // Get metadata from plugin loader
    if let Some(metadata) = PLUGIN_LOADER.get_plugin_info(plugin_name) {
        let mut dict = ImHashMap::new();
        dict.insert(Arc::new("path".to_string()), Value::String(Arc::new(metadata.path)));
        dict.insert(Arc::new("language".to_string()), Value::String(Arc::new(metadata.language)));

        let functions: Vec<Value> = metadata.functions
            .into_iter()
            .map(|f| Value::String(Arc::new(f)))
            .collect();
        dict.insert(Arc::new("functions".to_string()), Value::List(Arc::new(functions)));

        Ok(Value::Dict(Arc::new(dict)))
    } else {
        Err(TBError::runtime_error(&format!("Plugin '{}' not found", plugin_name)))
    }
}


// ============================================================================
// ASYNC TASK MANAGEMENT
// ============================================================================

/// spawn(func: fn, args: list) -> task_id
pub fn builtin_spawn(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("spawn() takes 2 arguments: function, args_list"));
    }

    let func = match &args[0] {
        Value::Function(f) => Arc::clone(f),
        _ => return Err(TBError::runtime_error("First argument to spawn() must be a function")),
    };

    let func_args = match &args[1] {
        Value::List(l) => (**l).clone(),
        _ => return Err(TBError::runtime_error("Second argument to spawn() must be a list")),
    };

    // Clone the current environment for the spawned task
    let env = TASK_ENVIRONMENT.read().clone();

    let handle = RUNTIME.spawn(async move {
        // Execute the function in a new task with the cloned environment
        use crate::task_runtime::TaskExecutor;

        let mut executor = TaskExecutor::new(env);
        executor.execute_function(&func, func_args)
    });

    let task_id = uuid::Uuid::new_v4().to_string();
    ACTIVE_TASKS.insert(task_id.clone(), handle);

    Ok(Value::String(Arc::new(task_id)))
}


/// await_task(task_id) -> value
pub fn builtin_await_task(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 { return Err(TBError::runtime_error("await_task() takes 1 argument")); }
    let task_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("Task ID must be a string")),
    };

    if let Some((_, handle)) = ACTIVE_TASKS.remove(&task_id) {
        let result = RUNTIME.block_on(handle)
            .map_err(|e| BuiltinError::TaskError(format!("Task panicked: {}", e)))??;
        Ok(result)
    } else {
        Err(BuiltinError::TaskError(format!("Task not found: {}", task_id)).into())
    }
}

/// cancel_task(task_id) -> bool
pub fn builtin_cancel_task(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 { return Err(TBError::runtime_error("cancel_task() takes 1 argument")); }
    let task_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("Task ID must be a string")),
    };

    if let Some(entry) = ACTIVE_TASKS.get(&task_id) {
        entry.value().abort();
        ACTIVE_TASKS.remove(&task_id);
        Ok(Value::Bool(true))
    } else {
        Ok(Value::Bool(false))
    }
}


// ============================================================================
// SERIALIZATION & HASHING
// ============================================================================

pub fn builtin_bincode_serialize(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("bincode_serialize() takes 1 argument"));
    }
    let value = &args[0];
    let encoded: Vec<u8> = bincode::serialize(value).map_err(|e| TBError::runtime_error(format!("bincode serialization failed: {}", e)))?;
    Ok(Value::List(Arc::new(encoded.into_iter().map(|b| Value::Int(b as i64)).collect())))
}

pub fn builtin_bincode_deserialize(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("bincode_deserialize() takes 1 argument"));
    }
    let bytes_val = match &args[0] {
        Value::List(l) => l,
        _ => return Err(TBError::runtime_error("argument must be a list of bytes (integers)"))
    };

    let bytes: Vec<u8> = bytes_val.iter().map(|v| match v {
        Value::Int(i) => *i as u8,
        _ => 0,
    }).collect();

    let decoded: Value = bincode::deserialize(&bytes).map_err(|e| TBError::runtime_error(format!("bincode deserialization failed: {}", e)))?;
    Ok(decoded)
}


pub fn builtin_hash(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("hash() requires 2 arguments: algorithm, data"));
    }
    let algo = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("algorithm must be a string")),
    };
    let data_val = args[1].clone();
    let data = match data_val {
        Value::String(s) => s.as_bytes().to_vec(),
        Value::List(l) => l.iter().map(|v| match v {
            Value::Int(i) => *i as u8,
            _ => 0,
        }).collect(),
        _ => return Err(TBError::runtime_error("data must be a string or a list of bytes")),
    };

    let hash_str = match algo.as_str() {
        "sha256" => {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let result = hasher.finalize();
            format!("{:x}", result)
        }
        _ => return Err(TBError::runtime_error(format!("unsupported hash algorithm: {}", algo))),
    };

    Ok(Value::String(Arc::new(hash_str)))
}

// ============================================================================
// NETWORKING BUILT-INS
// ============================================================================

/// create_server(protocol, address, callbacks) -> server_id
/// protocol: "tcp" or "udp"
/// address: "host:port"
/// callbacks: dict with on_connect, on_disconnect, on_message functions
pub fn builtin_create_server(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() < 2 {
        return Err(TBError::runtime_error(
            "create_server requires at least 2 arguments: protocol, address"
        ));
    }

    let protocol = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(TBError::runtime_error("create_server: protocol must be a string")),
    };

    let address = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err(TBError::runtime_error("create_server: address must be a string")),
    };

    // Parse address (format: "host:port")
    let parts: Vec<&str> = address.split(':').collect();
    if parts.len() != 2 {
        return Err(TBError::runtime_error(&format!(
            "create_server: invalid address format '{}', expected 'host:port'", address
        )));
    }

    let host = parts[0].to_string();
    let port: u16 = parts[1].parse().map_err(|_| {
        TBError::runtime_error(&format!("create_server: invalid port '{}'", parts[1]))
    })?;

    // Extract callbacks from args[2] (dictionary with on_connect, on_disconnect, on_message)
    let (on_connect, on_disconnect, on_message) = if args.len() > 2 {
        match &args[2] {
            Value::Dict(dict) => {
                let on_connect = dict.get(&Arc::new("on_connect".to_string()))
                    .and_then(|v| match v {
                        Value::Function(f) => Some((**f).clone()),
                        _ => None,
                    });

                let on_disconnect = dict.get(&Arc::new("on_disconnect".to_string()))
                    .and_then(|v| match v {
                        Value::Function(f) => Some((**f).clone()),
                        _ => None,
                    });

                let on_message = dict.get(&Arc::new("on_message".to_string()))
                    .and_then(|v| match v {
                        Value::Function(f) => Some((**f).clone()),
                        _ => None,
                    });

                (on_connect, on_disconnect, on_message)
            }
            _ => (None, None, None),
        }
    } else {
        (None, None, None)
    };

    // Create server based on protocol
    let server_id = uuid::Uuid::new_v4().to_string();
    let server_id_clone = server_id.clone();

    let server_handle = RUNTIME.block_on(async {
        match protocol {
            "tcp" => {
                networking::create_tcp_server(
                    host,
                    port,
                    on_connect,
                    on_disconnect,
                    on_message,
                ).await
            }
            "udp" => {
                networking::create_udp_server(
                    host,
                    port,
                    on_message,
                ).await
            }
            _ => Err(BuiltinError::InvalidArgument(
                format!("create_server: unsupported protocol '{}'", protocol)
            )),
        }
    }).map_err(|e| TBError::runtime_error(&format!("create_server failed: {}", e)))?;

    // Store server handle
    networking::NETWORK_SERVERS.insert(server_id_clone.clone(), server_handle);

    Ok(Value::String(Arc::new(server_id_clone)))
}

/// stop_server(server_id) -> bool
pub fn builtin_stop_server(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("stop_server requires 1 argument: server_id"));
    }

    let server_id = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(TBError::runtime_error("stop_server: server_id must be a string")),
    };

    // Remove server from registry and send shutdown signal
    if let Some((_, server_handle)) = networking::NETWORK_SERVERS.remove(server_id) {
        // Send shutdown signal (non-blocking)
        RUNTIME.block_on(async {
            let _ = server_handle.shutdown_tx.send(()).await;
        });
        Ok(Value::Bool(true))
    } else {
        Ok(Value::Bool(false))
    }
}

/// connect_to(on_connect, on_disconnect, on_msg, host, port, type) -> connection_id
pub fn builtin_connect_to(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 6 {
        return Err(TBError::runtime_error("connect_to() takes 6 arguments: on_connect, on_disconnect, on_msg, host, port, type"));
    }

    let host = match &args[3] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("connect_to() host must be a string")),
    };

    let port = match &args[4] {
        Value::Int(i) => *i as u16,
        _ => return Err(TBError::runtime_error("connect_to() port must be an integer")),
    };

    let conn_type = match &args[5] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("connect_to() type must be a string (tcp/udp)")),
    };

    let conn_id = format!("conn_{}:{}_{}", conn_type, host, port);

    RUNTIME.block_on(async {
        if conn_type.to_lowercase() == "tcp" {
            let client = networking::TcpClient::connect(host, port).await?;
            networking::TCP_CLIENTS.insert(conn_id.clone(), Arc::new(client));
        } else if conn_type.to_lowercase() == "udp" {
            let client = networking::UdpClient::connect(host, port).await?;
            networking::UDP_CLIENTS.insert(conn_id.clone(), Arc::new(client));
        } else {
            return Err(BuiltinError::InvalidArgument(
                format!("Unsupported connection type: {}", conn_type)
            ));
        }
        Ok::<(), BuiltinError>(())
    })?;

    Ok(Value::String(Arc::new(conn_id)))
}

/// send_to(connection_id, message) -> None
pub fn builtin_send_to(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("send_to() takes 2 arguments: connection_id, message"));
    }

    let conn_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("send_to() connection_id must be a string")),
    };

    let message = match &args[1] {
        Value::String(s) => s.to_string(),
        Value::Dict(_) => {
            // Convert dict to JSON
            serde_json::to_string(&args[1])
                .map_err(|e| TBError::runtime_error(format!("Failed to serialize message: {}", e)))?
        }
        _ => return Err(TBError::runtime_error("send_to() message must be a string or dict")),
    };

    RUNTIME.block_on(async {
        // Try TCP first
        if let Some(client) = networking::TCP_CLIENTS.get(&conn_id) {
            client.send(message).await?;
            return Ok::<(), BuiltinError>(());
        }

        // Try UDP
        if let Some(client) = networking::UDP_CLIENTS.get(&conn_id) {
            client.send(message).await?;
            return Ok::<(), BuiltinError>(());
        }

        Err(BuiltinError::NotFound(format!("Connection not found: {}", conn_id)))
    })?;

    Ok(Value::None)
}

/// http_session(base_url, headers, cookies_file) -> session_id
pub fn builtin_http_session(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() < 1 || args.len() > 3 {
        return Err(TBError::runtime_error("http_session() takes 1-3 arguments: base_url, headers, cookies_file"));
    }

    let base_url = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("http_session() base_url must be a string")),
    };

    let headers = if args.len() > 1 {
        match &args[1] {
            Value::Dict(map) => {
                let mut headers = HashMap::new();
                for (k, v) in map.iter() {
                    if let Value::String(val) = v {
                        headers.insert(k.to_string(), val.to_string());
                    }
                }
                headers
            }
            _ => HashMap::new(),
        }
    } else {
        HashMap::new()
    };

    let cookies_file = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.to_string()),
            Value::None => None,
            _ => None,
        }
    } else {
        None
    };

    let session = networking::HttpSession::new(base_url.clone(), headers, cookies_file)?;
    let session_id = format!("http_session_{}", base_url);

    HTTP_SESSIONS.insert(session_id.clone(), Arc::new(session));

    Ok(Value::String(Arc::new(session_id)))
}

/// http_request(session_id, url, method, data) -> response_dict
pub fn builtin_http_request(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() < 3 || args.len() > 4 {
        return Err(TBError::runtime_error("http_request() takes 3-4 arguments: session_id, url, method, data"));
    }

    let session_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("http_request() session_id must be a string")),
    };

    let url = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("http_request() url must be a string")),
    };

    let method = match &args[2] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("http_request() method must be a string")),
    };

    let data = if args.len() > 3 {
        match &args[3] {
            Value::Dict(_) => {
                // Convert TB Dict to JSON using helper function
                Some(convert_tb_to_json(&args[3]))
            }
            Value::String(s) => {
                // Try to parse string as JSON
                serde_json::from_str::<serde_json::Value>(s)
                    .ok()
                    .or_else(|| Some(serde_json::Value::String(s.to_string())))
            }
            Value::None => None,
            _ => return Err(TBError::runtime_error("http_request() data must be a dict, string, or None")),
        }
    } else {
        None
    };

    let session = HTTP_SESSIONS.get(&session_id)
        .ok_or_else(|| TBError::runtime_error(format!("HTTP session not found: {}", session_id)))?;

    let response = RUNTIME.block_on(async {
        session.request(url, method, data).await
    })?;

    // Convert response to TB dict
    use im::HashMap as ImHashMap;
    let mut response_dict = ImHashMap::new();
    response_dict.insert(
        Arc::new("status".to_string()),
        Value::Int(response.status as i64),
    );
    response_dict.insert(
        Arc::new("body".to_string()),
        Value::String(Arc::new(response.body)),
    );

    // Convert headers to dict
    let mut headers_dict = ImHashMap::new();
    for (k, v) in response.headers {
        headers_dict.insert(Arc::new(k), Value::String(Arc::new(v)));
    }
    response_dict.insert(
        Arc::new("headers".to_string()),
        Value::Dict(Arc::new(headers_dict)),
    );

    Ok(Value::Dict(Arc::new(response_dict)))
}

// ============================================================================
// UTILITY BUILT-INS
// ============================================================================

/// json_parse(json_str: str) -> dict
pub fn builtin_json_parse(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("json_parse() takes 1 argument: json_str"));
    }

    let json_str = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("json_parse() argument must be a string")),
    };

    let json_value: serde_json::Value = utils::json_parse(&json_str)?;

    // Convert serde_json::Value to TB Value
    fn convert_json_to_tb(val: serde_json::Value) -> Value {
        use im::HashMap as ImHashMap;
        match val {
            serde_json::Value::Null => Value::None,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::None
                }
            }
            serde_json::Value::String(s) => Value::String(Arc::new(s)),
            serde_json::Value::Array(arr) => {
                let items: Vec<Value> = arr.into_iter().map(convert_json_to_tb).collect();
                Value::List(Arc::new(items))
            }
            serde_json::Value::Object(obj) => {
                let mut map = ImHashMap::new();
                for (k, v) in obj {
                    map.insert(Arc::new(k), convert_json_to_tb(v));
                }
                Value::Dict(Arc::new(map))
            }
        }
    }

    Ok(convert_json_to_tb(json_value))
}

/// json_stringify(dict: dict, pretty: bool = false) -> str
pub fn builtin_json_stringify(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() || args.len() > 2 {
        return Err(TBError::runtime_error("json_stringify() takes 1-2 arguments: value, pretty"));
    }

    let pretty = if args.len() > 1 {
        match &args[1] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    // Convert TB Value to serde_json::Value
    fn convert_tb_to_json(val: &Value) -> serde_json::Value {
        match val {
            Value::None => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(i) => serde_json::Value::Number((*i).into()),
            Value::Float(f) => {
                serde_json::Number::from_f64(*f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
            Value::String(s) => serde_json::Value::String(s.to_string()),
            Value::List(items) => {
                let arr: Vec<serde_json::Value> = items.iter().map(convert_tb_to_json).collect();
                serde_json::Value::Array(arr)
            }
            Value::Dict(map) => {
                let mut obj = serde_json::Map::new();
                for (k, v) in map.iter() {
                    obj.insert(k.to_string(), convert_tb_to_json(v));
                }
                serde_json::Value::Object(obj)
            }
            _ => serde_json::Value::Null,
        }
    }

    let json_value = convert_tb_to_json(&args[0]);
    let json_str = utils::json_stringify(&json_value, pretty)?;

    Ok(Value::String(Arc::new(json_str)))
}

/// yaml_parse(yaml_str: str) -> dict
pub fn builtin_yaml_parse(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("yaml_parse() takes 1 argument: yaml_str"));
    }

    let yaml_str = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("yaml_parse() argument must be a string")),
    };

    let yaml_value: serde_yaml::Value = utils::yaml_parse(&yaml_str)?;

    // Convert serde_yaml::Value to TB Value (similar to JSON)
    fn convert_yaml_to_tb(val: serde_yaml::Value) -> Value {
        use im::HashMap as ImHashMap;
        match val {
            serde_yaml::Value::Null => Value::None,
            serde_yaml::Value::Bool(b) => Value::Bool(b),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::None
                }
            }
            serde_yaml::Value::String(s) => Value::String(Arc::new(s)),
            serde_yaml::Value::Sequence(arr) => {
                let items: Vec<Value> = arr.into_iter().map(convert_yaml_to_tb).collect();
                Value::List(Arc::new(items))
            }
            serde_yaml::Value::Mapping(obj) => {
                let mut map = ImHashMap::new();
                for (k, v) in obj {
                    if let serde_yaml::Value::String(key) = k {
                        map.insert(Arc::new(key), convert_yaml_to_tb(v));
                    }
                }
                Value::Dict(Arc::new(map))
            }
            _ => Value::None,
        }
    }

    Ok(convert_yaml_to_tb(yaml_value))
}

/// yaml_stringify(dict: dict) -> str
pub fn builtin_yaml_stringify(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("yaml_stringify() takes 1 argument: value"));
    }

    // Convert TB Value to serde_yaml::Value
    fn convert_tb_to_yaml(val: &Value) -> serde_yaml::Value {
        match val {
            Value::None => serde_yaml::Value::Null,
            Value::Bool(b) => serde_yaml::Value::Bool(*b),
            Value::Int(i) => serde_yaml::Value::Number((*i).into()),
            Value::Float(f) => {
                serde_yaml::Value::Number(serde_yaml::Number::from(*f))
            }
            Value::String(s) => serde_yaml::Value::String(s.to_string()),
            Value::List(items) => {
                let arr: Vec<serde_yaml::Value> = items.iter().map(convert_tb_to_yaml).collect();
                serde_yaml::Value::Sequence(arr)
            }
            Value::Dict(map) => {
                let mut obj = serde_yaml::Mapping::new();
                for (k, v) in map.iter() {
                    obj.insert(
                        serde_yaml::Value::String(k.to_string()),
                        convert_tb_to_yaml(v),
                    );
                }
                serde_yaml::Value::Mapping(obj)
            }
            _ => serde_yaml::Value::Null,
        }
    }

    let yaml_value = convert_tb_to_yaml(&args[0]);
    let yaml_str = utils::yaml_stringify(&yaml_value)?;

    Ok(Value::String(Arc::new(yaml_str)))
}

/// time(timezone: str = "auto") -> dict
pub fn builtin_time(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() > 1 {
        return Err(TBError::runtime_error("time() takes 0-1 arguments: timezone"));
    }

    let timezone = if args.is_empty() {
        None
    } else {
        match &args[0] {
            Value::String(s) => Some(s.to_string()),
            Value::None => None,
            _ => return Err(TBError::runtime_error("time() timezone must be a string or None")),
        }
    };

    let time_info = utils::get_time(timezone)?;

    // Convert to TB Dict with proper types
    use im::HashMap as ImHashMap;
    let mut tb_map = ImHashMap::new();
    tb_map.insert(Arc::new("year".to_string()), Value::Int(time_info.year as i64));
    tb_map.insert(Arc::new("month".to_string()), Value::Int(time_info.month as i64));
    tb_map.insert(Arc::new("day".to_string()), Value::Int(time_info.day as i64));
    tb_map.insert(Arc::new("hour".to_string()), Value::Int(time_info.hour as i64));
    tb_map.insert(Arc::new("minute".to_string()), Value::Int(time_info.minute as i64));
    tb_map.insert(Arc::new("second".to_string()), Value::Int(time_info.second as i64));
    tb_map.insert(Arc::new("microsecond".to_string()), Value::Int(time_info.microsecond as i64));
    tb_map.insert(Arc::new("weekday".to_string()), Value::Int(time_info.weekday as i64));
    tb_map.insert(Arc::new("timezone".to_string()), Value::String(Arc::new(time_info.timezone)));
    tb_map.insert(Arc::new("offset".to_string()), Value::Int(time_info.offset as i64));
    tb_map.insert(Arc::new("timestamp".to_string()), Value::Int(time_info.timestamp));
    tb_map.insert(Arc::new("iso8601".to_string()), Value::String(Arc::new(time_info.iso8601)));

    Ok(Value::Dict(Arc::new(tb_map)))
}

// ============================================================================
// TYPE CONVERSION BUILT-INS
// ============================================================================

/// int(value) -> int
pub fn builtin_int(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("int() takes exactly 1 argument"));
    }

    match &args[0] {
        Value::Int(i) => Ok(Value::Int(*i)),
        Value::Float(f) => Ok(Value::Int(*f as i64)),
        Value::String(s) => {
            s.parse::<i64>()
                .map(Value::Int)
                .map_err(|_| TBError::runtime_error(format!("Cannot convert '{}' to int", s)))
        }
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        _ => Err(TBError::runtime_error(format!("Cannot convert {} to int", args[0].type_name()))),
    }
}

/// str(value) -> str
pub fn builtin_str(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("str() takes exactly 1 argument"));
    }

    Ok(Value::String(Arc::new(args[0].to_string())))
}

/// float(value) -> float
pub fn builtin_float(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("float() takes exactly 1 argument"));
    }

    match &args[0] {
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Int(i) => Ok(Value::Float(*i as f64)),
        Value::String(s) => {
            s.parse::<f64>()
                .map(Value::Float)
                .map_err(|_| TBError::runtime_error(format!("Cannot convert '{}' to float", s)))
        }
        _ => Err(TBError::runtime_error(format!("Cannot convert {} to float", args[0].type_name()))),
    }
}

/// dict(value) -> dict
/// Converts JSON string or creates empty dict
pub fn builtin_dict(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() {
        // Create empty dict
        return Ok(Value::Dict(Arc::new(ImHashMap::new())));
    }

    if args.len() != 1 {
        return Err(TBError::runtime_error("dict() takes 0 or 1 argument"));
    }

    match &args[0] {
        Value::Dict(d) => Ok(Value::Dict(Arc::clone(d))),
        Value::String(s) => {
            // Try to parse as JSON
            let json_value: serde_json::Value = serde_json::from_str(s)
                .map_err(|e| TBError::runtime_error(format!("Cannot parse JSON: {}", e)))?;

            // Convert JSON object to TB Dict
            if let serde_json::Value::Object(obj) = json_value {
                let mut map = ImHashMap::new();
                for (k, v) in obj {
                    map.insert(Arc::new(k), json_to_tb_value(v));
                }
                Ok(Value::Dict(Arc::new(map)))
            } else {
                Err(TBError::runtime_error("JSON value is not an object"))
            }
        }
        _ => Err(TBError::runtime_error(format!("Cannot convert {} to dict", args[0].type_name()))),
    }
}

/// list(value) -> list
/// Converts JSON array string or creates empty list
pub fn builtin_list(args: Vec<Value>) -> Result<Value, TBError> {
    if args.is_empty() {
        // Create empty list
        return Ok(Value::List(Arc::new(vec![])));
    }

    if args.len() != 1 {
        return Err(TBError::runtime_error("list() takes 0 or 1 argument"));
    }

    match &args[0] {
        Value::List(l) => Ok(Value::List(Arc::clone(l))),
        Value::String(s) => {
            // Try to parse as JSON array
            let json_value: serde_json::Value = serde_json::from_str(s)
                .map_err(|e| TBError::runtime_error(format!("Cannot parse JSON: {}", e)))?;

            // Convert JSON array to TB List
            if let serde_json::Value::Array(arr) = json_value {
                let items: Vec<Value> = arr.into_iter().map(json_to_tb_value).collect();
                Ok(Value::List(Arc::new(items)))
            } else {
                Err(TBError::runtime_error("JSON value is not an array"))
            }
        }
        _ => Err(TBError::runtime_error(format!("Cannot convert {} to list", args[0].type_name()))),
    }
}

/// Helper function to convert serde_json::Value to TB Value
fn json_to_tb_value(val: serde_json::Value) -> Value {
    match val {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::None
            }
        }
        serde_json::Value::String(s) => Value::String(Arc::new(s)),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.into_iter().map(json_to_tb_value).collect();
            Value::List(Arc::new(items))
        }
        serde_json::Value::Object(obj) => {
            let mut map = ImHashMap::new();
            for (k, v) in obj {
                map.insert(Arc::new(k), json_to_tb_value(v));
            }
            Value::Dict(Arc::new(map))
        }
    }
}

// ============================================================================
// BASIC COLLECTION BUILT-INS
// ============================================================================

/// len(collection) -> int
pub fn builtin_len(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("len() takes exactly 1 argument"));
    }

    match &args[0] {
        Value::String(s) => Ok(Value::Int(s.len() as i64)),
        Value::List(l) => Ok(Value::Int(l.len() as i64)),
        Value::Dict(d) => Ok(Value::Int(d.len() as i64)),
        _ => Err(TBError::runtime_error(format!("len() not supported for {}", args[0].type_name()))),
    }
}

/// push(list, item) -> list
pub fn builtin_push(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("push() takes exactly 2 arguments"));
    }

    match &args[0] {
        Value::List(items) => {
            let mut new_items = (**items).clone();
            new_items.push(args[1].clone());
            Ok(Value::List(Arc::new(new_items)))
        }
        _ => Err(TBError::runtime_error("push() requires a list")),
    }
}

/// pop(list) -> list
pub fn builtin_pop(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("pop() takes exactly 1 argument"));
    }

    match &args[0] {
        Value::List(items) => {
            if items.is_empty() {
                return Err(TBError::runtime_error("Cannot pop from empty list"));
            }
            let mut new_items = (**items).clone();
            new_items.pop();
            Ok(Value::List(Arc::new(new_items)))
        }
        _ => Err(TBError::runtime_error("pop() requires a list")),
    }
}

/// keys(dict) -> list[str]
pub fn builtin_keys(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("keys() takes exactly 1 argument"));
    }

    match &args[0] {
        Value::Dict(map) => {
            let keys: Vec<Value> = map.keys()
                .map(|k| Value::String(Arc::clone(k)))
                .collect();
            Ok(Value::List(Arc::new(keys)))
        }
        _ => Err(TBError::runtime_error("keys() requires a dict")),
    }
}

/// values(dict) -> list
pub fn builtin_values(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("values() takes exactly 1 argument"));
    }

    match &args[0] {
        Value::Dict(map) => {
            let values: Vec<Value> = map.values().cloned().collect();
            Ok(Value::List(Arc::new(values)))
        }
        _ => Err(TBError::runtime_error("values() requires a dict")),
    }
}

/// range(start, end) or range(end) -> list[int]
pub fn builtin_range(args: Vec<Value>) -> Result<Value, TBError> {
    let (start, end) = match args.len() {
        1 => {
            match &args[0] {
                Value::Int(e) => (0, *e),
                _ => return Err(TBError::runtime_error("range() requires integer arguments")),
            }
        }
        2 => {
            match (&args[0], &args[1]) {
                (Value::Int(s), Value::Int(e)) => (*s, *e),
                _ => return Err(TBError::runtime_error("range() requires integer arguments")),
            }
        }
        _ => return Err(TBError::runtime_error("range() takes 1 or 2 arguments")),
    };

    let values: Vec<Value> = (start..end).map(Value::Int).collect();
    Ok(Value::List(Arc::new(values)))
}

/// print(...) -> None
pub fn builtin_print(args: Vec<Value>) -> Result<Value, TBError> {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", arg);
    }
    println!();
    Ok(Value::None)
}

// ============================================================================
// MODULE IMPORT BUILT-IN
// ============================================================================

/// import(path) -> dict
/// Loads and executes a TB module, returning its exported values as a dict
///
/// NOTE: This is a placeholder implementation. The actual import functionality
/// should be implemented at the JIT/Compiler level to avoid circular dependencies.
/// For now, this returns an error directing users to use the @import directive.
pub fn builtin_import(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 1 {
        return Err(TBError::runtime_error("import() takes exactly 1 argument"));
    }

    let _path_str = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(TBError::runtime_error("import() requires a string path")),
    };

    // TODO: Implement proper module loading
    // This requires access to the parser and executor, which creates circular dependencies
    // The proper solution is to implement this at the JIT executor level
    Err(TBError::runtime_error(
        "import() builtin is not yet implemented. Please use the @import directive instead:\n\
         @import { path: \"module.tb\" }"
    ))
}

// ============================================================================
// HIGHER-ORDER FUNCTIONS
// ============================================================================

/// map(fn, list) -> list
/// Applies a function to each element of a list and returns a new list
pub fn builtin_map(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("map() takes exactly 2 arguments: function, list"));
    }

    let func = match &args[0] {
        Value::Function(f) => f.clone(),
        Value::NativeFunction(_) => return Err(TBError::runtime_error("map() does not support native functions yet")),
        _ => return Err(TBError::runtime_error("map() first argument must be a function")),
    };

    let list = match &args[1] {
        Value::List(l) => l.clone(),
        _ => return Err(TBError::runtime_error("map() second argument must be a list")),
    };

    let mut result = Vec::new();
    for item in list.iter() {
        let mapped = crate::task_runtime::TaskExecutor::new(crate::TASK_ENVIRONMENT.read().clone())
            .execute_function(&func, vec![item.clone()])?;
        result.push(mapped);
    }

    Ok(Value::List(Arc::new(result)))
}

/// filter(fn, list) -> list
/// Filters a list based on a predicate function
pub fn builtin_filter(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("filter() takes exactly 2 arguments: function, list"));
    }

    let func = match &args[0] {
        Value::Function(f) => f.clone(),
        Value::NativeFunction(_) => return Err(TBError::runtime_error("filter() does not support native functions yet")),
        _ => return Err(TBError::runtime_error("filter() first argument must be a function")),
    };

    let list = match &args[1] {
        Value::List(l) => l.clone(),
        _ => return Err(TBError::runtime_error("filter() second argument must be a list")),
    };

    let mut result = Vec::new();
    for item in list.iter() {
        let keep = crate::task_runtime::TaskExecutor::new(crate::TASK_ENVIRONMENT.read().clone())
            .execute_function(&func, vec![item.clone()])?;

        if keep.is_truthy() {
            result.push(item.clone());
        }
    }

    Ok(Value::List(Arc::new(result)))
}

/// reduce(fn, list, initial) -> value
/// Reduces a list to a single value using an accumulator function
pub fn builtin_reduce(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 3 {
        return Err(TBError::runtime_error("reduce() takes exactly 3 arguments: function, list, initial"));
    }

    let func = match &args[0] {
        Value::Function(f) => f.clone(),
        Value::NativeFunction(_) => return Err(TBError::runtime_error("reduce() does not support native functions yet")),
        _ => return Err(TBError::runtime_error("reduce() first argument must be a function")),
    };

    let list = match &args[1] {
        Value::List(l) => l.clone(),
        _ => return Err(TBError::runtime_error("reduce() second argument must be a list")),
    };

    let mut accumulator = args[2].clone();

    for item in list.iter() {
        accumulator = crate::task_runtime::TaskExecutor::new(crate::TASK_ENVIRONMENT.read().clone())
            .execute_function(&func, vec![accumulator, item.clone()])?;
    }

    Ok(accumulator)
}

/// forEach(fn, list) -> None
/// Executes a function for each element in a list (side effects only)
pub fn builtin_forEach(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 2 {
        return Err(TBError::runtime_error("forEach() takes exactly 2 arguments: function, list"));
    }

    let func = match &args[0] {
        Value::Function(f) => f.clone(),
        Value::NativeFunction(_) => return Err(TBError::runtime_error("forEach() does not support native functions yet")),
        _ => return Err(TBError::runtime_error("forEach() first argument must be a function")),
    };

    let list = match &args[1] {
        Value::List(l) => l.clone(),
        _ => return Err(TBError::runtime_error("forEach() second argument must be a list")),
    };

    for item in list.iter() {
        let _ = crate::task_runtime::TaskExecutor::new(crate::TASK_ENVIRONMENT.read().clone())
            .execute_function(&func, vec![item.clone()])?;
    }

    Ok(Value::None)
}
