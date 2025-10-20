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
            _ => FFIValue {
                tag: TAG_NONE,
                data: FFIValueData { int_val: 0 },
            },
        }
    }

    /// Convert FFI value back to TB Value (zero-copy where possible)
    pub unsafe fn to_value(&self) -> Value {
        match self.tag {
            TAG_NONE => Value::None,
            TAG_BOOL => Value::Bool(self.data.bool_val != 0),
            TAG_INT => Value::Int(self.data.int_val),
            TAG_FLOAT => Value::Float(self.data.float_val),
            TAG_STRING => {
                let c_str = CStr::from_ptr(self.data.ptr as *const c_char);
                let s = c_str.to_str().unwrap().to_string();
                Value::String(std::sync::Arc::new(s))
            }
            _ => Value::None,
        }
    }

    /// Free allocated resources
    pub unsafe fn free(&mut self) {
        if self.tag == TAG_STRING {
            let _ = CString::from_raw(self.data.ptr as *mut c_char);
        }
    }
}

/// Function signature for plugin functions
pub type PluginFn = unsafe extern "C" fn(*const FFIValue, usize) -> FFIValue;

