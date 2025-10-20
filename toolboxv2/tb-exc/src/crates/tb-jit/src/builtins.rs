use im::HashMap as ImHashMap;
use std::sync::Arc;
use tb_core::{TBError, Value, NativeFunction};

pub fn register_builtins(env: &mut ImHashMap<Arc<String>, Value>) {
    // Core functions
    register_print(env);
    register_len(env);
    register_push(env);
    register_pop(env);
    register_keys(env);
    register_values(env);
    register_range(env);
    register_str(env);
    register_int(env);
    register_float(env);

    // File I/O functions
    register_file_exists(env);
    register_read_file(env);
    register_write_file(env);
    register_append_file(env);
    register_delete_file(env);

    // Utility functions
    register_json_parse(env);
    register_json_stringify(env);
    register_yaml_parse(env);
    register_yaml_stringify(env);
    register_time(env);
}

fn register_print(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("print".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    print!(" ");
                }
                print!("{}", arg);
            }
            println!();
            Ok(Value::None)
        }),
    }));
    env.insert(Arc::new("print".to_string()), func);
}

fn register_len(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("len".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "len() takes exactly 1 argument".to_string(),
                });
            }

            match &args[0] {
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::List(items) => Ok(Value::Int(items.len() as i64)),
                Value::Dict(map) => Ok(Value::Int(map.len() as i64)),
                _ => Err(TBError::RuntimeError {
                    message: format!("len() not supported for {}", args[0].type_name()),
                }),
            }
        }),
    }));
    env.insert(Arc::new("len".to_string()), func);
}

fn register_push(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("push".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 2 {
                return Err(TBError::RuntimeError {
                    message: "push() takes exactly 2 arguments".to_string(),
                });
            }

            match &args[0] {
                Value::List(items) => {
                    let mut new_items = (**items).clone();
                    new_items.push(args[1].clone());
                    Ok(Value::List(Arc::new(new_items)))
                }
                _ => Err(TBError::RuntimeError {
                    message: "push() requires a list".to_string(),
                }),
            }
        }),
    }));
    env.insert(Arc::new("push".to_string()), func);
}

fn register_pop(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("pop".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "pop() takes exactly 1 argument".to_string(),
                });
            }

            match &args[0] {
                Value::List(items) => {
                    if items.is_empty() {
                        return Err(TBError::RuntimeError {
                            message: "Cannot pop from empty list".to_string(),
                        });
                    }
                    let mut new_items = (**items).clone();
                    new_items.pop();
                    Ok(Value::List(Arc::new(new_items)))
                }
                _ => Err(TBError::RuntimeError {
                    message: "pop() requires a list".to_string(),
                }),
            }
        }),
    }));
    env.insert(Arc::new("pop".to_string()), func);
}

fn register_keys(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("keys".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "keys() takes exactly 1 argument".to_string(),
                });
            }

            match &args[0] {
                Value::Dict(map) => {
                    let keys: Vec<Value> = map.keys()
                        .map(|k| Value::String(Arc::clone(k)))
                        .collect();
                    Ok(Value::List(Arc::new(keys)))
                }
                _ => Err(TBError::RuntimeError {
                    message: "keys() requires a dict".to_string(),
                }),
            }
        }),
    }));
    env.insert(Arc::new("keys".to_string()), func);
}

fn register_values(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("values".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "values() takes exactly 1 argument".to_string(),
                });
            }

            match &args[0] {
                Value::Dict(map) => {
                    let values: Vec<Value> = map.values().cloned().collect();
                    Ok(Value::List(Arc::new(values)))
                }
                _ => Err(TBError::RuntimeError {
                    message: "values() requires a dict".to_string(),
                }),
            }
        }),
    }));
    env.insert(Arc::new("values".to_string()), func);
}

fn register_range(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("range".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            let (start, end) = match args.len() {
                1 => {
                    if let Value::Int(n) = args[0] {
                        (0, n)
                    } else {
                        return Err(TBError::RuntimeError {
                            message: "range() requires integer arguments".to_string(),
                        });
                    }
                }
                2 => {
                    if let (Value::Int(a), Value::Int(b)) = (&args[0], &args[1]) {
                        (*a, *b)
                    } else {
                        return Err(TBError::RuntimeError {
                            message: "range() requires integer arguments".to_string(),
                        });
                    }
                }
                _ => {
                    return Err(TBError::RuntimeError {
                        message: "range() takes 1 or 2 arguments".to_string(),
                    });
                }
            };

            let values: Vec<Value> = (start..end).map(Value::Int).collect();
            Ok(Value::List(Arc::new(values)))
        }),
    }));
    env.insert(Arc::new("range".to_string()), func);
}

fn register_str(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("str".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "str() takes exactly 1 argument".to_string(),
                });
            }

            Ok(Value::String(Arc::new(args[0].to_string())))
        }),
    }));
    env.insert(Arc::new("str".to_string()), func);
}

fn register_int(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("int".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "int() takes exactly 1 argument".to_string(),
                });
            }

            match &args[0] {
                Value::Int(i) => Ok(Value::Int(*i)),
                Value::Float(f) => Ok(Value::Int(*f as i64)),
                Value::String(s) => {
                    s.parse::<i64>()
                        .map(Value::Int)
                        .map_err(|_| TBError::RuntimeError {
                            message: format!("Cannot convert '{}' to int", s),
                        })
                }
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                _ => Err(TBError::RuntimeError {
                    message: format!("Cannot convert {} to int", args[0].type_name()),
                }),
            }
        }),
    }));
    env.insert(Arc::new("int".to_string()), func);
}

fn register_float(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("float".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "float() takes exactly 1 argument".to_string(),
                });
            }

            match &args[0] {
                Value::Float(f) => Ok(Value::Float(*f)),
                Value::Int(i) => Ok(Value::Float(*i as f64)),
                Value::String(s) => {
                    s.parse::<f64>()
                        .map(Value::Float)
                        .map_err(|_| TBError::RuntimeError {
                            message: format!("Cannot convert '{}' to float", s),
                        })
                }
                _ => Err(TBError::RuntimeError {
                    message: format!("Cannot convert {} to float", args[0].type_name()),
                }),
            }
        }),
    }));
    env.insert(Arc::new("float".to_string()), func);
}

// ============================================================================
// FILE I/O FUNCTIONS
// ============================================================================

fn register_file_exists(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("file_exists".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.is_empty() || args.len() > 2 {
                return Err(TBError::RuntimeError {
                    message: "file_exists() takes 1-2 arguments: path, blob=false".to_string(),
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

            if is_blob {
                // For blob files, always return false for now (not implemented)
                Ok(Value::Bool(false))
            } else {
                // Check real file
                Ok(Value::Bool(std::path::Path::new(&path).exists()))
            }
        }),
    }));
    env.insert(Arc::new("file_exists".to_string()), func);
}

fn register_read_file(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("read_file".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.is_empty() || args.len() > 2 {
                return Err(TBError::RuntimeError {
                    message: "read_file() takes 1-2 arguments: path, blob=false".to_string(),
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

            if is_blob {
                Err(TBError::RuntimeError {
                    message: "Blob file reading not yet implemented".to_string(),
                })
            } else {
                // Read real file
                std::fs::read_to_string(&path)
                    .map(|content| Value::String(Arc::new(content)))
                    .map_err(|e| TBError::RuntimeError {
                        message: format!("Failed to read file '{}': {}", path, e),
                    })
            }
        }),
    }));
    env.insert(Arc::new("read_file".to_string()), func);
}

fn register_write_file(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("write_file".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() < 2 || args.len() > 3 {
                return Err(TBError::RuntimeError {
                    message: "write_file() takes 2-3 arguments: path, content, blob=false".to_string(),
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

            if is_blob {
                Err(TBError::RuntimeError {
                    message: "Blob file writing not yet implemented".to_string(),
                })
            } else {
                // Write real file
                std::fs::write(&path, content)
                    .map(|_| Value::None)
                    .map_err(|e| TBError::RuntimeError {
                        message: format!("Failed to write file '{}': {}", path, e),
                    })
            }
        }),
    }));
    env.insert(Arc::new("write_file".to_string()), func);
}

fn register_append_file(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("append_file".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() < 2 || args.len() > 3 {
                return Err(TBError::RuntimeError {
                    message: "append_file() takes 2-3 arguments: path, content, blob=false".to_string(),
                });
            }

            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(TBError::RuntimeError {
                    message: "append_file() path must be a string".to_string(),
                }),
            };

            let content = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err(TBError::RuntimeError {
                    message: "append_file() content must be a string".to_string(),
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

            if is_blob {
                Err(TBError::RuntimeError {
                    message: "Blob file appending not yet implemented".to_string(),
                })
            } else {
                // Append to real file
                use std::io::Write;
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .and_then(|mut file| file.write_all(content.as_bytes()))
                    .map(|_| Value::None)
                    .map_err(|e| TBError::RuntimeError {
                        message: format!("Failed to append to file '{}': {}", path, e),
                    })
            }
        }),
    }));
    env.insert(Arc::new("append_file".to_string()), func);
}

fn register_delete_file(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("delete_file".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.is_empty() || args.len() > 2 {
                return Err(TBError::RuntimeError {
                    message: "delete_file() takes 1-2 arguments: path, blob=false".to_string(),
                });
            }

            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(TBError::RuntimeError {
                    message: "delete_file() path must be a string".to_string(),
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

            if is_blob {
                Err(TBError::RuntimeError {
                    message: "Blob file deletion not yet implemented".to_string(),
                })
            } else {
                // Delete real file
                std::fs::remove_file(&path)
                    .map(|_| Value::None)
                    .map_err(|e| TBError::RuntimeError {
                        message: format!("Failed to delete file '{}': {}", path, e),
                    })
            }
        }),
    }));
    env.insert(Arc::new("delete_file".to_string()), func);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn register_json_parse(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("json_parse".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "json_parse() takes exactly 1 argument: json_string".to_string(),
                });
            }

            let json_str = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(TBError::RuntimeError {
                    message: "json_parse() argument must be a string".to_string(),
                }),
            };

            // Parse JSON and convert to TB Value
            serde_json::from_str::<serde_json::Value>(&json_str)
                .map(|json_val| json_to_tb_value(&json_val))
                .map_err(|e| TBError::RuntimeError {
                    message: format!("JSON parse error: {}", e),
                })
        }),
    }));
    env.insert(Arc::new("json_parse".to_string()), func);
}

fn register_json_stringify(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("json_stringify".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.is_empty() || args.len() > 2 {
                return Err(TBError::RuntimeError {
                    message: "json_stringify() takes 1-2 arguments: value, pretty=false".to_string(),
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

            // Convert TB Value to JSON
            let json_val = tb_value_to_json(&args[0]);

            let result = if pretty {
                serde_json::to_string_pretty(&json_val)
            } else {
                serde_json::to_string(&json_val)
            };

            result
                .map(|s| Value::String(Arc::new(s)))
                .map_err(|e| TBError::RuntimeError {
                    message: format!("JSON stringify error: {}", e),
                })
        }),
    }));
    env.insert(Arc::new("json_stringify".to_string()), func);
}

fn register_yaml_parse(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("yaml_parse".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "yaml_parse() takes exactly 1 argument: yaml_string".to_string(),
                });
            }

            let yaml_str = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(TBError::RuntimeError {
                    message: "yaml_parse() argument must be a string".to_string(),
                }),
            };

            // Parse YAML and convert to TB Value
            serde_yaml::from_str::<serde_yaml::Value>(&yaml_str)
                .map(|yaml_val| yaml_to_tb_value(&yaml_val))
                .map_err(|e| TBError::RuntimeError {
                    message: format!("YAML parse error: {}", e),
                })
        }),
    }));
    env.insert(Arc::new("yaml_parse".to_string()), func);
}

fn register_yaml_stringify(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("yaml_stringify".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() != 1 {
                return Err(TBError::RuntimeError {
                    message: "yaml_stringify() takes exactly 1 argument: value".to_string(),
                });
            }

            // Convert TB Value to YAML
            let yaml_val = tb_value_to_yaml(&args[0]);

            serde_yaml::to_string(&yaml_val)
                .map(|s| Value::String(Arc::new(s)))
                .map_err(|e| TBError::RuntimeError {
                    message: format!("YAML stringify error: {}", e),
                })
        }),
    }));
    env.insert(Arc::new("yaml_stringify".to_string()), func);
}

fn register_time(env: &mut ImHashMap<Arc<String>, Value>) {
    let func = Value::NativeFunction(Arc::new(NativeFunction {
        name: Arc::new("time".to_string()),
        func: Arc::new(|args: Vec<Value>| {
            if args.len() > 1 {
                return Err(TBError::RuntimeError {
                    message: "time() takes 0-1 arguments: timezone".to_string(),
                });
            }

            use chrono::{Local, Datelike, Timelike};

            let now = Local::now();

            // Create time dictionary
            let mut time_dict = ImHashMap::new();
            time_dict.insert(Arc::new("year".to_string()), Value::Int(now.year() as i64));
            time_dict.insert(Arc::new("month".to_string()), Value::Int(now.month() as i64));
            time_dict.insert(Arc::new("day".to_string()), Value::Int(now.day() as i64));
            time_dict.insert(Arc::new("hour".to_string()), Value::Int(now.hour() as i64));
            time_dict.insert(Arc::new("minute".to_string()), Value::Int(now.minute() as i64));
            time_dict.insert(Arc::new("second".to_string()), Value::Int(now.second() as i64));
            time_dict.insert(Arc::new("timestamp".to_string()), Value::Int(now.timestamp()));
            time_dict.insert(Arc::new("iso8601".to_string()), Value::String(Arc::new(now.to_rfc3339())));

            Ok(Value::Dict(Arc::new(time_dict)))
        }),
    }));
    env.insert(Arc::new("time".to_string()), func);
}

// Helper functions for JSON/YAML conversion
fn json_to_tb_value(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::None
            }
        }
        serde_json::Value::String(s) => Value::String(Arc::new(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_tb_value).collect();
            Value::List(Arc::new(items))
        }
        serde_json::Value::Object(obj) => {
            let mut map = ImHashMap::new();
            for (k, v) in obj {
                map.insert(Arc::new(k.clone()), json_to_tb_value(v));
            }
            Value::Dict(Arc::new(map))
        }
    }
}

fn tb_value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::None => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::Value::Number((*i).into()),
        Value::Float(f) => serde_json::Value::Number(
            serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0))
        ),
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::List(items) => {
            let arr: Vec<serde_json::Value> = items.iter().map(tb_value_to_json).collect();
            serde_json::Value::Array(arr)
        }
        Value::Dict(map) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in map.iter() {
                obj.insert(k.to_string(), tb_value_to_json(v));
            }
            serde_json::Value::Object(obj)
        }
        _ => serde_json::Value::Null,
    }
}

fn yaml_to_tb_value(yaml: &serde_yaml::Value) -> Value {
    match yaml {
        serde_yaml::Value::Null => Value::None,
        serde_yaml::Value::Bool(b) => Value::Bool(*b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::None
            }
        }
        serde_yaml::Value::String(s) => Value::String(Arc::new(s.clone())),
        serde_yaml::Value::Sequence(arr) => {
            let items: Vec<Value> = arr.iter().map(yaml_to_tb_value).collect();
            Value::List(Arc::new(items))
        }
        serde_yaml::Value::Mapping(obj) => {
            let mut map = ImHashMap::new();
            for (k, v) in obj {
                if let serde_yaml::Value::String(key) = k {
                    map.insert(Arc::new(key.clone()), yaml_to_tb_value(v));
                }
            }
            Value::Dict(Arc::new(map))
        }
        _ => Value::None,
    }
}

fn tb_value_to_yaml(val: &Value) -> serde_yaml::Value {
    match val {
        Value::None => serde_yaml::Value::Null,
        Value::Bool(b) => serde_yaml::Value::Bool(*b),
        Value::Int(i) => serde_yaml::Value::Number((*i).into()),
        Value::Float(f) => serde_yaml::Value::Number(serde_yaml::Number::from(*f)),
        Value::String(s) => serde_yaml::Value::String(s.to_string()),
        Value::List(items) => {
            let arr: Vec<serde_yaml::Value> = items.iter().map(tb_value_to_yaml).collect();
            serde_yaml::Value::Sequence(arr)
        }
        Value::Dict(map) => {
            let mut obj = serde_yaml::Mapping::new();
            for (k, v) in map.iter() {
                obj.insert(
                    serde_yaml::Value::String(k.to_string()),
                    tb_value_to_yaml(v)
                );
            }
            serde_yaml::Value::Mapping(obj)
        }
        _ => serde_yaml::Value::Null,
    }
}

