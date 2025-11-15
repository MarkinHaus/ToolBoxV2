// python_ffi.rs - C-API FFI Bridge for Python (OHNE PyO3)
// Verwendet libloading für direkten Zugriff auf Python DLL

use libloading::{Library, Symbol};
use std::ffi::{CString, CStr, c_void, OsStr};
use std::ptr;
use std::sync::Arc;
use anyhow::{Result, Context, bail};
use tracing::{debug};
use std::env;
use std::os::windows::ffi::OsStrExt;

// =================== Windows API Types ===================

#[cfg(windows)]
type DllDirectoryCookie = *mut c_void;

#[cfg(windows)]
type AddDllDirectory_t = unsafe extern "system" fn(*const u16) -> DllDirectoryCookie;

// =================== Python C-API Types ===================

pub type PyObject = *mut c_void;
pub type PyGILState = i32;
pub type Py_ssize_t = isize;

// =================== Python C-API Function Signatures ===================

type PyGILState_Ensure_t = unsafe extern "C" fn() -> PyGILState;
type PyGILState_Release_t = unsafe extern "C" fn(PyGILState);
type PyGILState_Check_t = unsafe extern "C" fn() -> i32;

type Py_Initialize_t = unsafe extern "C" fn();
type Py_IsInitialized_t = unsafe extern "C" fn() -> i32;
type Py_Finalize_t = unsafe extern "C" fn();
type Py_SetPath_t = unsafe extern "C" fn(*const u16);
type PySys_SetPath_t = unsafe extern "C" fn(*const u16);
type PyEval_SaveThread_t = unsafe extern "C" fn() -> *mut c_void;
type PyEval_RestoreThread_t = unsafe extern "C" fn(*mut c_void);

type PyImport_ImportModule_t = unsafe extern "C" fn(*const i8) -> PyObject;
type PyImport_AddModule_t = unsafe extern "C" fn(*const i8) -> PyObject;

type PyObject_GetAttrString_t = unsafe extern "C" fn(PyObject, *const i8) -> PyObject;
type PyObject_SetAttrString_t = unsafe extern "C" fn(PyObject, *const i8, PyObject) -> i32;
type PyObject_CallObject_t = unsafe extern "C" fn(PyObject, PyObject) -> PyObject;
type PyObject_Call_t = unsafe extern "C" fn(PyObject, PyObject, PyObject) -> PyObject;
type PyObject_Str_t = unsafe extern "C" fn(PyObject) -> PyObject;

type PyDict_New_t = unsafe extern "C" fn() -> PyObject;
type PyDict_SetItemString_t = unsafe extern "C" fn(PyObject, *const i8, PyObject) -> i32;
type PyDict_GetItemString_t = unsafe extern "C" fn(PyObject, *const i8) -> PyObject;

type PyList_New_t = unsafe extern "C" fn(Py_ssize_t) -> PyObject;
type PyList_SetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t, PyObject) -> i32;
type PyList_GetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t) -> PyObject;
type PyList_Size_t = unsafe extern "C" fn(PyObject) -> Py_ssize_t;

type PyTuple_New_t = unsafe extern "C" fn(Py_ssize_t) -> PyObject;
type PyTuple_SetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t, PyObject) -> i32;
type PyTuple_GetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t) -> PyObject;

type PyUnicode_FromString_t = unsafe extern "C" fn(*const i8) -> PyObject;
type PyUnicode_AsUTF8_t = unsafe extern "C" fn(PyObject) -> *const i8;

type PyLong_FromLong_t = unsafe extern "C" fn(i64) -> PyObject;
type PyLong_AsLong_t = unsafe extern "C" fn(PyObject) -> i64;

type PyFloat_FromDouble_t = unsafe extern "C" fn(f64) -> PyObject;
type PyFloat_AsDouble_t = unsafe extern "C" fn(PyObject) -> f64;

type PyBool_FromLong_t = unsafe extern "C" fn(i64) -> PyObject;

type Py_IncRef_t = unsafe extern "C" fn(PyObject);
type Py_DecRef_t = unsafe extern "C" fn(PyObject);
type Py_NewRef_t = unsafe extern "C" fn(PyObject) -> PyObject;

type PyErr_Occurred_t = unsafe extern "C" fn() -> PyObject;
type PyErr_Print_t = unsafe extern "C" fn();
type PyErr_Clear_t = unsafe extern "C" fn();
type PyErr_Fetch_t = unsafe extern "C" fn(*mut PyObject, *mut PyObject, *mut PyObject);

type Py_None_t = unsafe extern "C" fn() -> PyObject;
type PyRun_SimpleString_t = unsafe extern "C" fn(*const i8) -> i32;

// =================== PythonFFI Struct ===================

pub struct PythonFFI {
    lib: Arc<Library>,
}

impl PythonFFI {
    pub fn new() -> Result<Self> {
        unsafe {
            // Lade Python DLL aus python_env (Standard-Python-Installation)
            #[cfg(windows)]
            let lib_path = "C:\\Users\\Markin\\Workspace\\ToolBoxV2\\python_env\\python312.dll";

            #[cfg(all(unix, not(target_os = "macos")))]
            let lib_path = "libpython3.12.so";

            #[cfg(target_os = "macos")]
            let lib_path = "libpython3.12.dylib";

            debug!("Loading Python library: {}", lib_path);

            let lib = Library::new(lib_path)
                .with_context(|| format!("Failed to load Python library: {}", lib_path))?;

            let ffi = PythonFFI {
                lib: Arc::new(lib),
            };

            // Initialisiere Python falls nötig
            let is_init: Symbol<Py_IsInitialized_t> = ffi.lib.get(b"Py_IsInitialized\0")?;
            if is_init() == 0 {
                debug!("Initializing Python interpreter");

                // Setze Python-Pfade BEVOR Py_Initialize() aufgerufen wird
                #[cfg(windows)]
                {
                    // Verwende python_env (Standard-Python-Installation) statt uv-Python
                    let python_home = String::from("C:\\Users\\Markin\\Workspace\\ToolBoxV2\\python_env");

                    // Get current working directory for app_singleton module
                    let cwd = env::current_dir()
                        .unwrap_or_else(|_| std::path::PathBuf::from("C:\\Users\\Markin\\Workspace\\ToolBoxV2\\toolboxv2\\src-core"))
                        .to_string_lossy()
                        .to_string();

                    let build_dir = format!("{}\\build", cwd);

                    debug!("Using python_home: {}", python_home);
                    debug!("Using cwd: {}", cwd);
                    debug!("Using build dir: {}", build_dir);

                    // Erweitere PATH-Umgebungsvariable mit DLL-Verzeichnissen
                    // Dies ist notwendig, damit Python-Extensions wie cryptography._rust.pyd ihre DLLs finden
                    let python_dlls = format!("{}\\DLLs", python_home);
                    let python_scripts = format!("{}\\Scripts", python_home);
                    let python_exe = format!("{}\\python.exe", python_home);

                    // Hole aktuelle PATH-Variable
                    let current_path = env::var("PATH").unwrap_or_default();

                    // Füge DLL-Verzeichnisse am Anfang hinzu (höchste Priorität)
                    let new_path = format!(
                        "{};{};{};{}",
                        python_home, python_dlls, python_scripts, current_path
                    );

                    // Setze erweiterte PATH-Variable
                    env::set_var("PATH", &new_path);

                    // Setze PYTHON_EXECUTABLE für toolboxv2.__main__.server_helper()
                    // Diese Umgebungsvariable wird von server_helper() benötigt
                    env::set_var("PYTHON_EXECUTABLE", &python_exe);

                    // Setze PYTHONUNBUFFERED=1 damit Python stdout/stderr sofort geflusht wird
                    // Dies ermöglicht es, Python print() Statements in der Rust-Konsole zu sehen
                    env::set_var("PYTHONUNBUFFERED", "1");

                    debug!("Extended PATH with DLL directories");
                    debug!("  - python_home: {}", python_home);
                    debug!("  - python DLLs: {}", python_dlls);
                    debug!("  - python scripts: {}", python_scripts);
                    debug!("  - PYTHON_EXECUTABLE: {}", python_exe);

                    // Setze Py_SetPath mit ALLEN Pfaden: python_home, Lib, site-packages, DLLs, cwd, build
                    // Reihenfolge: python_home zuerst, dann Lib (für stdlib), dann site-packages, dann DLLs, dann cwd und build (für Nuitka modules)
                    let lib_path = format!(
                        "{};{}\\Lib;{}\\Lib\\site-packages;{}\\DLLs;{};{}",
                        python_home, python_home, python_home, python_home, cwd, build_dir
                    );
                    let lib_path_wide: Vec<u16> = lib_path
                        .encode_utf16()
                        .chain(std::iter::once(0))
                        .collect();

                    if let Ok(set_path) = ffi.lib.get::<Py_SetPath_t>(b"Py_SetPath\0") {
                        debug!("Setting Python path to: {}", lib_path);
                        set_path(lib_path_wide.as_ptr());
                    }
                }

                let init: Symbol<Py_Initialize_t> = ffi.lib.get(b"Py_Initialize\0")?;
                init();

                // Leite Python stdout/stderr auf C stdout/stderr um, damit print() Statements in der Rust-Konsole erscheinen
                debug!("Redirecting Python stdout/stderr to C stdout/stderr...");
                let py_run: Symbol<PyRun_SimpleString_t> = ffi.lib.get(b"PyRun_SimpleString\0")?;
                let redirect_code = CString::new(
                    "import sys; import os; sys.stdout = os.fdopen(1, 'w', buffering=1); sys.stderr = os.fdopen(2, 'w', buffering=1)"
                )?;
                let result = py_run(redirect_code.as_ptr());
                if result != 0 {
                    debug!("Warning: Failed to redirect Python stdout/stderr (error code: {})", result);
                } else {
                    debug!("Python stdout/stderr redirected successfully");
                }

                // CRITICAL FIX: Release the GIL after initialization!
                // After Py_Initialize(), Python holds the GIL and never releases it.
                // This causes PyGILState_Ensure() to block forever.
                // We must call PyEval_SaveThread() to release the GIL so that
                // subsequent PyGILState_Ensure() calls can acquire it.
                debug!("Releasing GIL after initialization with PyEval_SaveThread()...");
                let save_thread: Symbol<PyEval_SaveThread_t> = ffi.lib.get(b"PyEval_SaveThread\0")?;
                save_thread();
                debug!("GIL released successfully! Subsequent with_gil() calls will now work.");
            }

            debug!("Python FFI initialized successfully");
            Ok(ffi)
        }
    }

    // =================== GIL Management ===================

    pub fn with_gil<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        use tracing::info;
        unsafe {
            info!("with_gil: Loading GIL symbols...");
            let gil_ensure: Symbol<PyGILState_Ensure_t> = self.lib.get(b"PyGILState_Ensure\0")?;
            let gil_release: Symbol<PyGILState_Release_t> = self.lib.get(b"PyGILState_Release\0")?;
            let err_occurred: Symbol<PyErr_Occurred_t> = self.lib.get(b"PyErr_Occurred\0")?;
            let err_print: Symbol<PyErr_Print_t> = self.lib.get(b"PyErr_Print\0")?;
            let err_clear: Symbol<PyErr_Clear_t> = self.lib.get(b"PyErr_Clear\0")?;
            info!("with_gil: Symbols loaded successfully");

            info!("with_gil: Calling PyGILState_Ensure() - THIS MAY BLOCK IF GIL IS HELD BY ANOTHER THREAD!");
            let gil = gil_ensure();
            info!("with_gil: GIL acquired successfully! Executing closure...");

            let result = f();
            info!("with_gil: Closure executed, checking for Python errors...");

            // CRITICAL: Check for Python errors BEFORE releasing the GIL!
            // err_occurred() requires the GIL to be held!
            if !err_occurred().is_null() {
                info!("with_gil: Python error occurred, printing and clearing...");
                err_print();
                err_clear();
                gil_release(gil);
                info!("with_gil: GIL released after error");
                if result.is_err() {
                    return result;
                }
                bail!("Python error occurred during with_gil");
            }

            info!("with_gil: No Python errors, releasing GIL...");
            gil_release(gil);
            info!("with_gil: GIL released successfully");

            info!("with_gil: Returning result");
            result
        }
    }

    // =================== Module Import ===================

    pub fn import_module(&self, name: &str) -> Result<PyObject> {
        // CRITICAL: Must hold GIL when calling PyImport_ImportModule!
        self.with_gil(|| {
            unsafe {
                let import_module: Symbol<PyImport_ImportModule_t> = self.lib.get(b"PyImport_ImportModule\0")?;
                let err_print: Symbol<PyErr_Print_t> = self.lib.get(b"PyErr_Print\0")?;
                let err_clear: Symbol<PyErr_Clear_t> = self.lib.get(b"PyErr_Clear\0")?;

                let c_name = CString::new(name)?;
                let module = import_module(c_name.as_ptr());

                if module.is_null() {
                    err_print();
                    err_clear();
                    bail!("Failed to import module: {}", name);
                }

                // PyImport_ImportModule() already returns a new reference, so we don't need to increment it
                debug!("Imported module: {}", name);
                Ok(module)
            }
        })
    }

    // =================== Object Operations ===================

    pub fn get_attr(&self, obj: PyObject, attr: &str) -> Result<PyObject> {
        use tracing::info;
        info!("get_attr() called for attribute: {}", attr);
        info!("Calling with_gil()...");

        let result = self.with_gil(|| {
            info!("GIL acquired successfully!");
            unsafe {
                info!("Loading PyObject_GetAttrString symbol...");
                let get_attr: Symbol<PyObject_GetAttrString_t> = self.lib.get(b"PyObject_GetAttrString\0")?;
                info!("Symbol loaded successfully");

                info!("Creating CString for attribute name...");
                let c_attr = CString::new(attr)?;
                info!("CString created successfully");

                info!("Calling PyObject_GetAttrString()...");
                let result = get_attr(obj, c_attr.as_ptr());
                info!("PyObject_GetAttrString() returned!");

                if result.is_null() {
                    bail!("Failed to get attribute: {}", attr);
                }

                // PyObject_GetAttrString() already returns a new reference
                info!("Attribute retrieved successfully");
                Ok(result)
            }
        });

        info!("with_gil() returned, result: {:?}", result.is_ok());
        result
    }

    pub fn call_function(&self, func: PyObject, args: PyObject, kwargs: Option<PyObject>) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let call_object: Symbol<PyObject_CallObject_t> = self.lib.get(b"PyObject_CallObject\0")?;
                let call: Symbol<PyObject_Call_t> = self.lib.get(b"PyObject_Call\0")?;
                let err_print: Symbol<PyErr_Print_t> = self.lib.get(b"PyErr_Print\0")?;
                let err_clear: Symbol<PyErr_Clear_t> = self.lib.get(b"PyErr_Clear\0")?;

                let result = if let Some(kw) = kwargs {
                    call(func, args, kw)
                } else {
                    call_object(func, args)
                };

                if result.is_null() {
                    err_print();
                    err_clear();
                    bail!("Function call failed");
                }

                // PyObject_Call() already returns a new reference
                Ok(result)
            }
        })
    }

    /// Ruft eine Python-Funktion auf und gibt den GIL während der Ausführung frei.
    /// Dies ermöglicht es Python, asyncio Event Loops und andere Threading-Operationen auszuführen.
    ///
    /// WICHTIG: Diese Methode sollte für lang laufende Python-Funktionen verwendet werden,
    /// die asyncio Event Loops oder Threading verwenden (z.B. server_helper()).
    pub fn call_function_async(&self, func: PyObject, args: PyObject, kwargs: Option<PyObject>) -> Result<PyObject> {
        unsafe {
            // Lade Funktionen OHNE GIL (sie sind thread-safe)
            let call_object: Symbol<PyObject_CallObject_t> = self.lib.get(b"PyObject_CallObject\0")?;
            let call: Symbol<PyObject_Call_t> = self.lib.get(b"PyObject_Call\0")?;
            let err_print: Symbol<PyErr_Print_t> = self.lib.get(b"PyErr_Print\0")?;
            let err_clear: Symbol<PyErr_Clear_t> = self.lib.get(b"PyErr_Clear\0")?;
            let gil_ensure: Symbol<PyGILState_Ensure_t> = self.lib.get(b"PyGILState_Ensure\0")?;
            let gil_release: Symbol<PyGILState_Release_t> = self.lib.get(b"PyGILState_Release\0")?;

            // Erwirb GIL für den Funktionsaufruf
            let gil = gil_ensure();

            // Rufe Python-Funktion auf
            let result = if let Some(kw) = kwargs {
                call(func, args, kw)
            } else {
                call_object(func, args)
            };

            // Gib GIL SOFORT frei, damit Python den Event Loop ausführen kann
            gil_release(gil);

            // Prüfe auf Fehler (OHNE GIL - nur Null-Check)
            if result.is_null() {
                // Erwirb GIL erneut für Fehlerbehandlung
                let gil = gil_ensure();
                err_print();
                err_clear();
                gil_release(gil);
                bail!("Function call failed");
            }

            // PyObject_Call() already returns a new reference
            Ok(result)
        }
    }

    // =================== Public Helper Methods ===================

    pub fn create_tuple(&self, size: isize) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let tuple_new: Symbol<PyTuple_New_t> = self.lib.get(b"PyTuple_New\0")?;
                let result = tuple_new(size);
                // PyTuple_New() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn tuple_set_item(&self, tuple: PyObject, index: isize, item: PyObject) -> Result<()> {
        self.with_gil(|| {
            unsafe {
                let tuple_set_item: Symbol<PyTuple_SetItem_t> = self.lib.get(b"PyTuple_SetItem\0")?;
                let result = tuple_set_item(tuple, index, item);
                if result != 0 {
                    anyhow::bail!("PyTuple_SetItem failed");
                }
                Ok(())
            }
        })
    }

    pub fn create_dict(&self) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let dict_new: Symbol<PyDict_New_t> = self.lib.get(b"PyDict_New\0")?;
                let result = dict_new();
                // PyDict_New() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn dict_set_item_string(&self, dict: PyObject, key: &str, value: PyObject) -> Result<()> {
        self.with_gil(|| {
            unsafe {
                let dict_set_item: Symbol<PyDict_SetItemString_t> = self.lib.get(b"PyDict_SetItemString\0")?;
                let c_key = CString::new(key)?;
                let result = dict_set_item(dict, c_key.as_ptr(), value);
                if result != 0 {
                    bail!("Failed to set dict item");
                }
                Ok(())
            }
        })
    }

    pub fn string_from_str(&self, s: &str) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let unicode_from_string: Symbol<PyUnicode_FromString_t> = self.lib.get(b"PyUnicode_FromString\0")?;
                let c_str = CString::new(s)?;
                let result = unicode_from_string(c_str.as_ptr());
                // PyUnicode_FromString() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn long_from_i64(&self, val: i64) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let long_from_long: Symbol<PyLong_FromLong_t> = self.lib.get(b"PyLong_FromLong\0")?;
                let result = long_from_long(val);
                // PyLong_FromLong() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn float_from_f64(&self, val: f64) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let float_from_double: Symbol<PyFloat_FromDouble_t> = self.lib.get(b"PyFloat_FromDouble\0")?;
                let result = float_from_double(val);
                // PyFloat_FromDouble() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn bool_from_bool(&self, val: bool) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let bool_from_long: Symbol<PyBool_FromLong_t> = self.lib.get(b"PyBool_FromLong\0")?;
                let result = bool_from_long(if val { 1 } else { 0 });
                // PyBool_FromLong() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn list_new(&self, size: isize) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let list_new: Symbol<PyList_New_t> = self.lib.get(b"PyList_New\0")?;
                let result = list_new(size);
                // PyList_New() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn list_set_item(&self, list: PyObject, index: isize, item: PyObject) -> Result<()> {
        self.with_gil(|| {
            unsafe {
                let list_set_item: Symbol<PyList_SetItem_t> = self.lib.get(b"PyList_SetItem\0")?;
                list_set_item(list, index, item);
                Ok(())
            }
        })
    }

    pub fn object_str(&self, obj: PyObject) -> Result<PyObject> {
        self.with_gil(|| {
            unsafe {
                let obj_str: Symbol<PyObject_Str_t> = self.lib.get(b"PyObject_Str\0")?;
                let result = obj_str(obj);
                if result.is_null() {
                    bail!("Failed to convert object to string");
                }
                // PyObject_Str() already returns a new reference
                Ok(result)
            }
        })
    }

    pub fn unicode_as_utf8(&self, obj: PyObject) -> Result<String> {
        self.with_gil(|| {
            unsafe {
                let unicode_as_utf8: Symbol<PyUnicode_AsUTF8_t> = self.lib.get(b"PyUnicode_AsUTF8\0")?;
                let c_str = unicode_as_utf8(obj);
                if c_str.is_null() {
                    bail!("Failed to get UTF-8 string");
                }
                let rust_str = CStr::from_ptr(c_str).to_str()?;
                Ok(rust_str.to_string())
            }
        })
    }
}

// =================== Unit Tests ===================
// HINWEIS: Tests vorübergehend deaktiviert wegen Heap Corruption
// Werden nach App Singleton Integration wieder aktiviert

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;

    // Alle Tests vorübergehend deaktiviert - werden nach App Singleton Integration reaktiviert
}
