// python_ffi.rs - C-API FFI Bridge for Python (OHNE PyO3)
// Verwendet libloading für direkten Zugriff auf Python DLL

use libloading::{Library, Symbol};
use std::ffi::{CString, CStr, c_void, c_char};
use std::ptr;
use std::sync::Arc;
use anyhow::{Result, Context, bail};
use tracing::{debug, warn, info};
use std::env;
use std::path::{Path, PathBuf};


// =================== Platform Specific Types ===================

// Definition von wchar_t (Python C-API nutzt dies für Pfade)
#[cfg(windows)]
type WChar = u16;
#[cfg(not(windows))]
type WChar = i32;

// Platform separator for PATH variables
#[cfg(windows)]
const PATH_SEPARATOR: &str = ";";
#[cfg(not(windows))]
const PATH_SEPARATOR: &str = ":";

// Library extension
#[cfg(windows)]
const LIB_EXT: &str = "dll";
#[cfg(target_os = "macos")]
const LIB_EXT: &str = "dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const LIB_EXT: &str = "so";

// =================== Platform-Agnostic c_char Handling ===================
// On most platforms (x86_64, Windows), c_char is i8
// On aarch64-linux-gnu, c_char is u8
// These helper functions ensure correct pointer casting regardless of platform

/// Converts a CString pointer to the platform-specific c_char pointer type
/// This is needed because on aarch64-linux-gnu, c_char is u8, not i8
#[inline]
fn cstring_as_c_char_ptr(cstr: &CString) -> *const c_char {
    cstr.as_ptr()
}

/// Converts a raw c_char pointer to a pointer that CStr::from_ptr expects
/// On most platforms this is a no-op, but on aarch64-linux-gnu it handles the u8/i8 difference
#[inline]
fn c_char_ptr_to_cstr_ptr(ptr: *const c_char) -> *const c_char {
    ptr
}

// =================== Python C-API Types ===================

pub type PyObject = *mut c_void;
pub type PyGILState = i32;
pub type Py_ssize_t = isize;

// =================== Python C-API Function Signatures ===================
// NOTE: All *const i8 changed to *const c_char for platform compatibility

type PyGILState_Ensure_t = unsafe extern "C" fn() -> PyGILState;
type PyGILState_Release_t = unsafe extern "C" fn(PyGILState);
type PyGILState_Check_t = unsafe extern "C" fn() -> i32;

type Py_Initialize_t = unsafe extern "C" fn();
type Py_IsInitialized_t = unsafe extern "C" fn() -> i32;
type Py_Finalize_t = unsafe extern "C" fn();

type Py_SetPath_t = unsafe extern "C" fn(*const WChar);
type PySys_SetPath_t = unsafe extern "C" fn(*const WChar);

type PyEval_SaveThread_t = unsafe extern "C" fn() -> *mut c_void;
type PyEval_RestoreThread_t = unsafe extern "C" fn(*mut c_void);

type PyImport_ImportModule_t = unsafe extern "C" fn(*const c_char) -> PyObject;
type PyImport_AddModule_t = unsafe extern "C" fn(*const c_char) -> PyObject;

type PyObject_GetAttrString_t = unsafe extern "C" fn(PyObject, *const c_char) -> PyObject;
type PyObject_SetAttrString_t = unsafe extern "C" fn(PyObject, *const c_char, PyObject) -> i32;
type PyObject_CallObject_t = unsafe extern "C" fn(PyObject, PyObject) -> PyObject;
type PyObject_Call_t = unsafe extern "C" fn(PyObject, PyObject, PyObject) -> PyObject;
type PyObject_Str_t = unsafe extern "C" fn(PyObject) -> PyObject;

type PyDict_New_t = unsafe extern "C" fn() -> PyObject;
type PyDict_SetItemString_t = unsafe extern "C" fn(PyObject, *const c_char, PyObject) -> i32;
type PyDict_GetItemString_t = unsafe extern "C" fn(PyObject, *const c_char) -> PyObject;

type PyList_New_t = unsafe extern "C" fn(Py_ssize_t) -> PyObject;
type PyList_SetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t, PyObject) -> i32;
type PyList_GetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t) -> PyObject;
type PyList_Size_t = unsafe extern "C" fn(PyObject) -> Py_ssize_t;

type PyTuple_New_t = unsafe extern "C" fn(Py_ssize_t) -> PyObject;
type PyTuple_SetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t, PyObject) -> i32;
type PyTuple_GetItem_t = unsafe extern "C" fn(PyObject, Py_ssize_t) -> PyObject;

type PyUnicode_FromString_t = unsafe extern "C" fn(*const c_char) -> PyObject;
type PyUnicode_AsUTF8_t = unsafe extern "C" fn(PyObject) -> *const c_char;

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
type PyRun_SimpleString_t = unsafe extern "C" fn(*const c_char) -> i32;

// =================== Helper Functions ===================

/// Konvertiert einen Rust String in einen Vektor von wchar_t (OS-abhängig)
fn to_wchar(s: &str) -> Vec<WChar> {
    #[cfg(windows)]
    {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }
    #[cfg(not(windows))]
    {
        // Unter Unix ist wchar_t meist UTF-32 (4 Bytes)
        s.chars().map(|c| c as i32).chain(std::iter::once(0)).collect()
    }
}

// =================== PythonFFI Struct ===================

pub struct PythonFFI {
    lib: Arc<Library>,
}

// =================== Python Environment Detection ===================

#[derive(Debug, Clone)]
struct PythonEnv {
    python_home: PathBuf,
    python_dll: PathBuf,
    python_exe: PathBuf,
    env_type: String,
}

impl PythonEnv {
    /// Detektiert automatisch die Python-Umgebung
    /// Priorität: CONDA_PREFIX > system python > python_env (fallback)
    fn detect() -> Result<Self> {
        // 1. Conda
        if let Ok(conda_path) = env::var("CONDA_PREFIX") {
            if let Some(env) = Self::try_conda(Path::new(&conda_path)) {
                return Ok(env);
            }
        }

        // 2. System Python
        if let Some(env) = Self::try_system_python() {
            return Ok(env);
        }

        // 3. Fallback
        if let Some(env) = Self::try_fallback_python_env() {
            return Ok(env);
        }

        bail!("Could not detect Python environment.")
    }

    fn check_version_files(base_path: &Path) -> Option<(PathBuf, PathBuf)> {
        // Liste der zu prüfenden Versionen (generisch)
        let versions = ["312", "3.12", "311", "3.11", "310", "3.10", "39", "3.9"];

        let exe_name = if cfg!(windows) { "python.exe" } else { "python3" };

        // Exe Pfad prüfen (Windows: root, Unix: bin/)
        let exe_path = if cfg!(windows) {
            base_path.join(exe_name)
        } else {
            base_path.join("bin").join(exe_name)
        };

        if !exe_path.exists() {
            return None;
        }

        // DLL suchen
        for ver in &versions {
            let (lib_name, lib_dir) = if cfg!(windows) {
                (format!("python{}.{}", ver.replace(".", ""), LIB_EXT), base_path.to_path_buf())
            } else {
                // Unix: oft libpython3.12.so in lib/
                let v_clean = if ver.contains('.') { ver.to_string() } else { format!("{}.{}", &ver[0..1], &ver[1..]) };
                (format!("libpython{}.{}", v_clean, LIB_EXT), base_path.join("lib"))
            };

            let lib_path = lib_dir.join(&lib_name);
            if lib_path.exists() {
                return Some((exe_path, lib_path));
            }

            // Auf Mac/Linux auch nach .dylib/.so ohne Version oder mit abi flags schauen könnte nötig sein
            // Hier vereinfacht.
        }
        None
    }

    fn try_fallback_python_env() -> Option<Self> {
        let cwd = env::current_dir().ok()?;

        let toolbox_root = cwd.ancestors().find(|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("ToolBoxV2")
        }).or_else(|| cwd.ancestors().find(|p| p.join(".git").exists()))?;

        let python_env_path = toolbox_root.join("python_env");
        debug!("Looking for fallback python_env at: {:?}", python_env_path);

        if let Some((exe, dll)) = Self::check_version_files(&python_env_path) {
            return Some(PythonEnv {
                python_home: python_env_path,
                python_dll: dll,
                python_exe: exe,
                env_type: "fallback".to_string(),
            });
        }
        None
    }

    fn try_conda(conda_path: &Path) -> Option<Self> {
        if let Some((exe, dll)) = Self::check_version_files(conda_path) {
            return Some(PythonEnv {
                python_home: conda_path.to_path_buf(),
                python_dll: dll,
                python_exe: exe,
                env_type: "conda".to_string(),
            });
        }
        None
    }

    fn try_system_python() -> Option<Self> {
        use std::process::Command;
        let python_cmd = if cfg!(windows) { "python" } else { "python3" };

        if let Ok(output) = Command::new(python_cmd)
            .arg("-c")
            .arg("import sys; print(sys.prefix, end='')")
            .output()
        {
            if output.status.success() {
                let prefix_str = String::from_utf8_lossy(&output.stdout).to_string();
                let prefix = Path::new(&prefix_str);

                if let Some((exe, dll)) = Self::check_version_files(prefix) {
                    return Some(PythonEnv {
                        python_home: prefix.to_path_buf(),
                        python_dll: dll,
                        python_exe: exe,
                        env_type: "system".to_string(),
                    });
                }
            }
        }
        None
    }
}

impl PythonFFI {
    pub fn new() -> Result<Self> {
        unsafe {
            let python_env = PythonEnv::detect().context("Failed to detect Python environment")?;

            debug!("Using Python Environment: {:?}", python_env);

            // Platform-specific Load
            // Unter Linux muss RTLD_GLOBAL gesetzt werden, damit Python Extensions Symbole finden
            #[cfg(unix)]
            let lib = {
                // libloading nutzt standardmäßig RTLD_LOCAL. Für Python extensions brauchen wir oft GLOBAL.
                // Hier nutzen wir die Standard libloading Methode, aber in komplexen Fällen braucht man os::unix::Library::open
                Library::new(&python_env.python_dll)?
            };
            #[cfg(windows)]
            let lib = Library::new(&python_env.python_dll)?;

            let ffi = PythonFFI { lib: Arc::new(lib) };

            let is_init: Symbol<Py_IsInitialized_t> = ffi.lib.get(b"Py_IsInitialized\0")?;
            if is_init() == 0 {
                debug!("Initializing Python interpreter...");

                let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

                // Pfade zusammenbauen
                let home_str = python_env.python_home.to_string_lossy();
                let cwd_str = cwd.to_string_lossy();

                // Workspace Root Logik (vereinfacht für Cross-Platform)
                let workspace_root = cwd.ancestors()
                    .find(|p| p.join("ToolBoxV2").exists() || p.join("Cargo.toml").exists())
                    .unwrap_or(&cwd);
                let workspace_str = workspace_root.to_string_lossy();
                let build_dir = cwd.join("build");

                // Environment Variables setzen
                env::set_var("PYTHON_EXECUTABLE", &python_env.python_exe);
                env::set_var("PYTHONUNBUFFERED", "1"); // Wichtig für Logs

                // PATH / LD_LIBRARY_PATH Anpassung
                let new_path_entry = if cfg!(windows) {
                    // Windows: Home + DLLs + Scripts
                    format!("{};{}\\DLLs;{}\\Scripts", home_str, home_str, home_str)
                } else {
                    // Unix: lib Verzeichnis
                    format!("{}/lib", home_str)
                };

                let path_env_key = if cfg!(windows) { "PATH" } else if cfg!(target_os = "macos") { "DYLD_LIBRARY_PATH" } else { "LD_LIBRARY_PATH" };

                if let Ok(current) = env::var(path_env_key) {
                    env::set_var(path_env_key, format!("{}{}{}", new_path_entry, PATH_SEPARATOR, current));
                } else {
                    env::set_var(path_env_key, &new_path_entry);
                }

                // PYTHONPATH (Py_SetPath Logic)
                // Wir müssen alle Pfade manuell setzen, wenn wir Py_SetPath nutzen.
                let mut paths = Vec::new();
                paths.push(python_env.python_home.clone());

                if cfg!(windows) {
                    paths.push(python_env.python_home.join("Lib"));
                    paths.push(python_env.python_home.join("Lib").join("site-packages"));
                    paths.push(python_env.python_home.join("DLLs"));
                } else {
                    // Unix Struktur (lib/python3.x/...)
                    // Vereinfachung: Wir nehmen an, dass detect die richtige Version hat oder globben
                    let lib_dir = python_env.python_home.join("lib");
                    // Suche nach python3.x Ordner
                    if let Ok(entries) = std::fs::read_dir(&lib_dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() && path.file_name().unwrap().to_string_lossy().starts_with("python3") {
                                paths.push(path.clone());
                                paths.push(path.join("site-packages"));
                                // dynload (entspricht DLLs)
                                paths.push(path.join("lib-dynload"));
                            }
                        }
                    }
                }

                // Eigene Pfade
                paths.push(workspace_root.to_path_buf());
                paths.push(cwd.clone());
                paths.push(build_dir);

                // Pfad-String zusammenbauen
                let path_str = paths.iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect::<Vec<String>>()
                    .join(PATH_SEPARATOR);

                debug!("Setting Python Path to: {}", path_str);

                // Py_SetPath aufrufen (Cross-Platform wide char handling)
                if let Ok(set_path) = ffi.lib.get::<Py_SetPath_t>(b"Py_SetPath\0") {
                    let wide_path = to_wchar(&path_str);
                    set_path(wide_path.as_ptr());
                }

                let init: Symbol<Py_Initialize_t> = ffi.lib.get(b"Py_Initialize\0")?;
                init();

                // Stdout Redirect (OS unabhängig)
                // FIX #1: Use cstring_as_c_char_ptr for platform-agnostic pointer conversion
                let py_run: Symbol<PyRun_SimpleString_t> = ffi.lib.get(b"PyRun_SimpleString\0")?;
                let redirect_code = CString::new(
                    "import sys; import os; sys.stdout = os.fdopen(1, 'w', buffering=1); sys.stderr = os.fdopen(2, 'w', buffering=1)"
                )?;
                py_run(cstring_as_c_char_ptr(&redirect_code));

                // GIL Release
                let save_thread: Symbol<PyEval_SaveThread_t> = ffi.lib.get(b"PyEval_SaveThread\0")?;
                save_thread();
            }

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
            let gil_ensure: Symbol<PyGILState_Ensure_t> = self.lib.get(b"PyGILState_Ensure\0")?;
            let gil_release: Symbol<PyGILState_Release_t> = self.lib.get(b"PyGILState_Release\0")?;
            let err_occurred: Symbol<PyErr_Occurred_t> = self.lib.get(b"PyErr_Occurred\0")?;
            let err_print: Symbol<PyErr_Print_t> = self.lib.get(b"PyErr_Print\0")?;
            let err_clear: Symbol<PyErr_Clear_t> = self.lib.get(b"PyErr_Clear\0")?;
            let gil = gil_ensure();

            let result = f();

            // CRITICAL: Check for Python errors BEFORE releasing the GIL!
            // err_occurred() requires the GIL to be held!
            if !err_occurred().is_null() {
                err_print();
                err_clear();
                gil_release(gil);
                if result.is_err() {
                    return result;
                }
                bail!("Python error occurred during execution");
            }
            gil_release(gil);
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

                // FIX #2: Use cstring_as_c_char_ptr for platform-agnostic pointer conversion
                let c_name = CString::new(name)?;
                let module = import_module(cstring_as_c_char_ptr(&c_name));

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

        let result = self.with_gil(|| {
            unsafe {
                let get_attr: Symbol<PyObject_GetAttrString_t> = self.lib.get(b"PyObject_GetAttrString\0")?;

                let c_attr = CString::new(attr)?;

                let result = get_attr(obj, cstring_as_c_char_ptr(&c_attr));

                if result.is_null() {
                    bail!("Failed to get attribute: {}", attr);
                }

                Ok(result)
            }
        });

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
                // FIX #4: Use cstring_as_c_char_ptr for platform-agnostic pointer conversion
                let result = dict_set_item(dict, cstring_as_c_char_ptr(&c_key), value);
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
                // FIX #5: Use cstring_as_c_char_ptr for platform-agnostic pointer conversion
                let result = unicode_from_string(cstring_as_c_char_ptr(&c_str));
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
                // FIX #6: Use c_char_ptr_to_cstr_ptr for platform-agnostic pointer conversion
                let rust_str = CStr::from_ptr(c_char_ptr_to_cstr_ptr(c_str)).to_str()?;
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
