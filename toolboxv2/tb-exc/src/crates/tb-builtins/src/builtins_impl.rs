//! Implementation of remaining built-in functions

use crate::*;
use std::sync::Arc;
use tb_core::{Value, TBError};
use std::collections::HashMap;

// ============================================================================
// NETWORKING BUILT-INS
// ============================================================================

/// create_server(on_connect, on_disconnect, on_msg, host, port, type) -> server_id
pub fn builtin_create_server(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 6 {
        return Err(TBError::RuntimeError {
            message: "create_server() takes 6 arguments: on_connect, on_disconnect, on_msg, host, port, type".to_string(),
        });
    }

    // Extract callbacks (would need to be stored and called from TB runtime)
    let host = match &args[3] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "create_server() host must be a string".to_string(),
        }),
    };

    let port = match &args[4] {
        Value::Int(i) => *i as u16,
        _ => return Err(TBError::RuntimeError {
            message: "create_server() port must be an integer".to_string(),
        }),
    };

    let server_type = match &args[5] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "create_server() type must be a string (tcp/udp)".to_string(),
        }),
    };

    let server_id = format!("server_{}:{}_{}", server_type, host, port);

    // Note: Actual callback handling would require integration with TB runtime
    // This is a simplified implementation

    Ok(Value::String(Arc::new(server_id)))
}

/// connect_to(on_connect, on_disconnect, on_msg, host, port, type) -> connection_id
pub fn builtin_connect_to(args: Vec<Value>) -> Result<Value, TBError> {
    if args.len() != 6 {
        return Err(TBError::RuntimeError {
            message: "connect_to() takes 6 arguments: on_connect, on_disconnect, on_msg, host, port, type".to_string(),
        });
    }

    let host = match &args[3] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "connect_to() host must be a string".to_string(),
        }),
    };

    let port = match &args[4] {
        Value::Int(i) => *i as u16,
        _ => return Err(TBError::RuntimeError {
            message: "connect_to() port must be an integer".to_string(),
        }),
    };

    let conn_type = match &args[5] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "connect_to() type must be a string (tcp/udp)".to_string(),
        }),
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
        return Err(TBError::RuntimeError {
            message: "send_to() takes 2 arguments: connection_id, message".to_string(),
        });
    }

    let conn_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "send_to() connection_id must be a string".to_string(),
        }),
    };

    let message = match &args[1] {
        Value::String(s) => s.to_string(),
        Value::Dict(_) => {
            // Convert dict to JSON
            serde_json::to_string(&args[1])
                .map_err(|e| TBError::RuntimeError {
                    message: format!("Failed to serialize message: {}", e),
                })?
        }
        _ => return Err(TBError::RuntimeError {
            message: "send_to() message must be a string or dict".to_string(),
        }),
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
        return Err(TBError::RuntimeError {
            message: "http_session() takes 1-3 arguments: base_url, headers, cookies_file".to_string(),
        });
    }

    let base_url = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "http_session() base_url must be a string".to_string(),
        }),
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
        return Err(TBError::RuntimeError {
            message: "http_request() takes 3-4 arguments: session_id, url, method, data".to_string(),
        });
    }

    let session_id = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "http_request() session_id must be a string".to_string(),
        }),
    };

    let url = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "http_request() url must be a string".to_string(),
        }),
    };

    let method = match &args[2] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "http_request() method must be a string".to_string(),
        }),
    };

    let data = if args.len() > 3 {
        match &args[3] {
            Value::Dict(_) | Value::String(_) => {
                Some(serde_json::to_value(&args[3])
                    .map_err(|e| TBError::RuntimeError {
                        message: format!("Failed to serialize data: {}", e),
                    })?)
            }
            Value::None => None,
            _ => return Err(TBError::RuntimeError {
                message: "http_request() data must be a dict, string, or None".to_string(),
            }),
        }
    } else {
        None
    };

    let session = HTTP_SESSIONS.get(&session_id)
        .ok_or_else(|| TBError::RuntimeError {
            message: format!("HTTP session not found: {}", session_id),
        })?;

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
        return Err(TBError::RuntimeError {
            message: "json_parse() takes 1 argument: json_str".to_string(),
        });
    }

    let json_str = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "json_parse() argument must be a string".to_string(),
        }),
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
        return Err(TBError::RuntimeError {
            message: "json_stringify() takes 1-2 arguments: value, pretty".to_string(),
        });
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
        return Err(TBError::RuntimeError {
            message: "yaml_parse() takes 1 argument: yaml_str".to_string(),
        });
    }

    let yaml_str = match &args[0] {
        Value::String(s) => s.to_string(),
        _ => return Err(TBError::RuntimeError {
            message: "yaml_parse() argument must be a string".to_string(),
        }),
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
        return Err(TBError::RuntimeError {
            message: "yaml_stringify() takes 1 argument: value".to_string(),
        });
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
        return Err(TBError::RuntimeError {
            message: "time() takes 0-1 arguments: timezone".to_string(),
        });
    }

    let timezone = if args.is_empty() {
        None
    } else {
        match &args[0] {
            Value::String(s) => Some(s.to_string()),
            Value::None => None,
            _ => return Err(TBError::RuntimeError {
                message: "time() timezone must be a string or None".to_string(),
            }),
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

