use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use tb_core::Value;

/// FFI-safe value representation for cross-language communication
#[repr(C)]
pub struct FFIValue {
    tag: u8,
    data: FFIValueData,
}

#[repr(C)]
union FFIValueData {
    int_val: i64,
    float_val: f64,
    bool_val: u8,
    ptr: *mut c_void,
}

// Value type tags
const TAG_NONE: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_INT: u8 = 2;
const TAG_FLOAT: u8 = 3;
const TAG_STRING: u8 = 4;
const TAG_LIST: u8 = 5;
const TAG_DICT: u8 = 6;
const TAG_FUNCTION: u8 = 7;

impl FFIValue {
    /// Convert TB Value to FFI-safe representation (zero-copy where possible)
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::None => FFIValue {
                tag: TAG_NONE,
                data: FFIValueData { int_val: 0 },
            },
            Value::Bool(b) => FFIValue {
                tag: TAG_BOOL,
                data: FFIValueData { bool_val: if *b { 1 } else { 0 } },
            },
            Value::Int(i) => FFIValue {
                tag: TAG_INT,
                data: FFIValueData { int_val: *i },
            },
            Value::Float(f) => FFIValue {
                tag: TAG_FLOAT,
                data: FFIValueData { float_val: *f },
            },
            Value::String(s) => {
                // Allocate C string (caller must free)
                let c_str = CString::new(s.as_ref().clone()).unwrap();
                FFIValue {
                    tag: TAG_STRING,
                    data: FFIValueData { ptr: c_str.into_raw() as *mut c_void },
                }
            }
            Value::List(items) => {
                // Serialize list to JSON for FFI transfer
                let json = serde_json::to_string(items.as_ref()).unwrap_or_else(|_| "[]".to_string());
                let c_str = CString::new(json).unwrap();
                FFIValue {
                    tag: TAG_LIST,
                    data: FFIValueData { ptr: c_str.into_raw() as *mut c_void },
                }
            }
            Value::Dict(map) => {
                // Serialize dict to JSON for FFI transfer
                let json = serde_json::to_string(map.as_ref()).unwrap_or_else(|_| "{}".to_string());
                let c_str = CString::new(json).unwrap();
                FFIValue {
                    tag: TAG_DICT,
                    data: FFIValueData { ptr: c_str.into_raw() as *mut c_void },
                }
            }
            Value::Function(_) => {
                // Functions cannot be transferred via FFI - return None
                // TODO: Implement function handles/callbacks
                FFIValue {
                    tag: TAG_NONE,
                    data: FFIValueData { int_val: 0 },
                }
            }
            _ => FFIValue {
                tag: TAG_NONE,
                data: FFIValueData { int_val: 0 },
            },
        }
    }

    /// Convert FFI value back to TB Value (zero-copy where possible)
    pub unsafe fn to_value(&self) -> Value {
        use std::sync::Arc;
        use im::HashMap as ImHashMap;

        match self.tag {
            TAG_NONE => Value::None,
            TAG_BOOL => Value::Bool(self.data.bool_val != 0),
            TAG_INT => Value::Int(self.data.int_val),
            TAG_FLOAT => Value::Float(self.data.float_val),
            TAG_STRING => {
                let c_str = CStr::from_ptr(self.data.ptr as *const c_char);
                let s = c_str.to_str().unwrap().to_string();
                Value::String(Arc::new(s))
            }
            TAG_LIST => {
                // Deserialize JSON to list
                let c_str = CStr::from_ptr(self.data.ptr as *const c_char);
                let json_str = c_str.to_str().unwrap();
                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    Self::json_to_value(json_value)
                } else {
                    Value::List(Arc::new(vec![]))
                }
            }
            TAG_DICT => {
                // Deserialize JSON to dict
                let c_str = CStr::from_ptr(self.data.ptr as *const c_char);
                let json_str = c_str.to_str().unwrap();
                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    Self::json_to_value(json_value)
                } else {
                    Value::Dict(Arc::new(ImHashMap::new()))
                }
            }
            TAG_FUNCTION => {
                // Functions cannot be transferred via FFI
                Value::None
            }
            _ => Value::None,
        }
    }

    /// Helper to convert JSON to TB Value
    fn json_to_value(json: serde_json::Value) -> Value {
        use std::sync::Arc;
        use im::HashMap as ImHashMap;

        match json {
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
                let items: Vec<Value> = arr.into_iter().map(Self::json_to_value).collect();
                Value::List(Arc::new(items))
            }
            serde_json::Value::Object(obj) => {
                let mut map = ImHashMap::new();
                for (k, v) in obj {
                    map.insert(Arc::new(k), Self::json_to_value(v));
                }
                Value::Dict(Arc::new(map))
            }
        }
    }

    /// Free allocated resources
    pub unsafe fn free(&mut self) {
        match self.tag {
            TAG_STRING | TAG_LIST | TAG_DICT => {
                let _ = CString::from_raw(self.data.ptr as *mut c_char);
            }
            _ => {}
        }
    }
}

/// Function signature for plugin functions
pub type PluginFn = unsafe extern "C" fn(*const FFIValue, usize) -> FFIValue;

