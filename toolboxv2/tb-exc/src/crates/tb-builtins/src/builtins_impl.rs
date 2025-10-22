//! Implementation of remaining built-in functions

use crate::*;
use std::sync::Arc;
use tb_core::{Value, TBError, Program};
use std::collections::HashMap;
use im::HashMap as ImHashMap;
use tokio::process::Command;
use sha2::{Sha256, Sha512, Sha224, Sha384, Digest};

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
    // Diese Funktion benötigt Zugriff auf die Plugin-Metadaten,
    // die idealerweise in einer globalen Registry gespeichert werden.
    Ok(Value::Dict(Arc::new(ImHashMap::new()))) // Placeholder
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

    // WICHTIG: Hier fehlt der Executor-Kontext (Environment).
    // Eine vollständige Implementierung benötigt Zugriff auf den aktuellen JitExecutor.
    // Als Vereinfachung gehen wir davon aus, dass die Funktion keine Umgebungsvariablen benötigt.

    let handle = RUNTIME.spawn(async move {
        // In einem echten Szenario würde hier ein neuer Executor mit geklontem Env erstellt.
        // Vereinfachung:
        if func.params.len() != func_args.len() {
            return Err(TBError::runtime_error("Argument count mismatch in spawned task"));
        }
        // Direkter Aufruf ist hier nicht trivial. Der Body der Funktion muss evaluiert werden.
        // Das ist ein komplexes Problem, das eine enge Integration mit dem Executor erfordert.
        // Als Platzhalter geben wir einen Fehler zurück.
        Err(TBError::runtime_error("Spawning functions is not fully implemented yet."))
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

/// create_server(on_connect, on_disconnect, on_msg, host, port, type) -> server_id
pub fn builtin_create_server(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 6 {
        return Err(TBError::runtime_error("create_server() takes 6 arguments: on_connect, on_disconnect, on_msg, host, port, type"));
    }

    // Extract callbacks (would need to be stored and called from TB runtime)
    let host = match &args[3] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("create_server() host must be a string")),
    };

    let port = match &args[4] {
        Value::Int(i) => *i as u16,
        _ => return Err(TBError::runtime_error("create_server() port must be an integer")),
    };

    let server_type = match &args[5] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::runtime_error("create_server() type must be a string (tcp/udp)")),
    };

    let server_id = format!("server_{}:{}_{}", server_type, host, port);

    // Note: Actual callback handling would require integration with TB runtime
    // This is a simplified implementation

    Ok(Value::String(Arc::new(server_id)))
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
            Value::Dict(_) | Value::String(_) => {
                Some(serde_json::to_value(&args[3])
                    .map_err(|e| TBError::runtime_error(format!("Failed to serialize data: {}", e)))?)
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
    let time_map = time_info.to_hashmap();

    // Convert to TB Dict
    use im::HashMap as ImHashMap;
    let mut tb_map = ImHashMap::new();
    for (k, v) in time_map {
        tb_map.insert(Arc::new(k), Value::String(Arc::new(v)));
    }

    Ok(Value::Dict(Arc::new(tb_map)))
}
