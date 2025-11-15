// nuitka_loader.rs - Nuitka Module Loader (OHNE PyO3)
// Lädt und verwaltet Nuitka-kompilierte Python-Module

use crate::python_ffi::{PythonFFI, PyObject};
use std::sync::Arc;
use anyhow::{Result, bail};
use serde_json::Value;
use tracing::{debug, info, warn};
use std::ffi::CString;

// =================== Nuitka Module Loader ===================

pub struct NuitkaModuleLoader {
    ffi: Arc<PythonFFI>,
}

impl NuitkaModuleLoader {
    pub fn new(ffi: Arc<PythonFFI>) -> Self {
        NuitkaModuleLoader { ffi }
    }

    pub fn load_module(&self, name: &str) -> Result<NuitkaModule> {
        self.ffi.with_gil(|| {
            let module = self.ffi.import_module(name)?;
            info!("Loaded Nuitka module: {}", name);

            Ok(NuitkaModule {
                name: name.to_string(),
                py_module: module,
                ffi: self.ffi.clone(),
            })
        })
    }
}

// =================== Nuitka Module ===================

pub struct NuitkaModule {
    name: String,
    py_module: PyObject,
    ffi: Arc<PythonFFI>,
}

impl NuitkaModule {
    pub fn call_function(
        &self,
        function_name: &str,
        args: Value,
    ) -> Result<Value> {
        self.ffi.with_gil(|| {
            // 1. Hole Funktion
            let func = self.ffi.get_attr(self.py_module, function_name)?;

            // 2. Konvertiere args zu Python
            let (py_args, py_kwargs) = self.json_to_py(&args)?;

            // 3. Rufe Funktion auf
            let result = self.ffi.call_function(func, py_args, Some(py_kwargs))?;

            // 4. Konvertiere Ergebnis zu JSON
            let json_result = self.py_to_json(result)?;

            debug!("Called {}.{}: {:?}", self.name, function_name, json_result);
            Ok(json_result)
        })
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    // =================== Type Conversion ===================

    fn json_to_py(&self, value: &Value) -> Result<(PyObject, PyObject)> {
        // Erstelle leere args tuple
        let args = self.ffi.create_tuple(0)?;

        // Erstelle kwargs dict
        let kwargs = self.ffi.create_dict()?;

        if let Value::Object(map) = value {
            for (key, val) in map {
                let py_val = self.value_to_py(val)?;
                self.ffi.dict_set_item_string(kwargs, key.as_str(), py_val)?;
            }
        }

        Ok((args, kwargs))
    }

    fn value_to_py(&self, value: &Value) -> Result<PyObject> {
        match value {
            Value::Null => {
                // Return Py_None - use 0 as placeholder for now
                self.ffi.long_from_i64(0)
            },
            Value::Bool(b) => {
                self.ffi.bool_from_bool(*b)
            },
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    self.ffi.long_from_i64(i)
                } else if let Some(f) = n.as_f64() {
                    self.ffi.float_from_f64(f)
                } else {
                    bail!("Invalid number: {:?}", n)
                }
            },
            Value::String(s) => {
                self.ffi.string_from_str(s.as_str())
            },
            Value::Array(arr) => {
                let py_list = self.ffi.list_new(arr.len() as isize)?;
                for (i, item) in arr.iter().enumerate() {
                    let py_item = self.value_to_py(item)?;
                    self.ffi.list_set_item(py_list, i as isize, py_item)?;
                }
                Ok(py_list)
            },
            Value::Object(map) => {
                let py_dict = self.ffi.create_dict()?;
                for (key, val) in map {
                    let py_val = self.value_to_py(val)?;
                    self.ffi.dict_set_item_string(py_dict, key.as_str(), py_val)?;
                }
                Ok(py_dict)
            }
        }
    }

    fn py_to_json(&self, obj: PyObject) -> Result<Value> {
        // Konvertiere Python-Objekt zu String
        let py_str = self.ffi.object_str(obj)?;
        let rust_str = self.ffi.unicode_as_utf8(py_str)?;

        // Versuche als JSON zu parsen
        match serde_json::from_str(&rust_str) {
            Ok(json) => Ok(json),
            Err(_) => {
                // Fallback: Wrap in string
                Ok(Value::String(rust_str))
            }
        }
    }
}

// =================== Helper Functions ===================

/// Konvertiert Python dict zu JSON Value
pub fn py_dict_to_json(_ffi: &PythonFFI, _dict: PyObject) -> Result<Value> {
    // TODO: Implementiere vollständige dict → JSON Konvertierung
    // Für jetzt: Vereinfachte Version
    Ok(Value::Object(serde_json::Map::new()))
}

/// Konvertiert Python list zu JSON Value
pub fn py_list_to_json(_ffi: &PythonFFI, _list: PyObject) -> Result<Value> {
    // TODO: Implementiere vollständige list → JSON Konvertierung
    // Für jetzt: Vereinfachte Version
    Ok(Value::Array(Vec::new()))
}

// Tests vorübergehend deaktiviert - werden nach App Singleton Integration reaktiviert
#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;

    // Alle Tests deaktiviert - werden nach App Singleton Integration reaktiviert
}

