use crate::ffi::{FFIValue, PluginFn};
use dashmap::DashMap;
use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::Arc;
use tb_core::{Result, TBError, Value, tb_debug_plugin};

/// Plugin loader with lazy loading and caching
pub struct PluginLoader {
    loaded_libraries: DashMap<String, Arc<Library>>,
    function_cache: DashMap<String, PluginFn>,
}

impl PluginLoader {
    pub fn new() -> Self {
        Self {
            loaded_libraries: DashMap::new(),
            function_cache: DashMap::new(),
        }
    }

    /// Load plugin library (lazy, cached)
    pub fn load_library(&self, path: &Path) -> Result<Arc<Library>> {
        let path_str = path.to_string_lossy().to_string();

        tb_debug_plugin!("load_library: Attempting to load: {}", path.display());

        // Check if file exists
        if !path.exists() {
            tb_debug_plugin!("load_library: File does not exist!");
            return Err(TBError::plugin_error(format!("Library file not found: {}", path.display())));
        }

        // Get file size for debugging
        if let Ok(metadata) = std::fs::metadata(path) {
            tb_debug_plugin!("load_library: File exists, size: {} bytes", metadata.len());
        }

        // Check cache first
        if let Some(lib) = self.loaded_libraries.get(&path_str) {
            tb_debug_plugin!("load_library: Using cached library");
            return Ok(Arc::clone(lib.value()));
        }

        // Load library
        tb_debug_plugin!("load_library: Loading library from disk...");
        let lib = unsafe {
            Library::new(path).map_err(|e| {
                tb_debug_plugin!("load_library: Failed to load: {}", e);
                TBError::plugin_error(format!("Failed to load plugin: {}", e))
            })?
        };

        tb_debug_plugin!("load_library: Successfully loaded library");

        let lib_arc = Arc::new(lib);
        self.loaded_libraries.insert(path_str, Arc::clone(&lib_arc));

        Ok(lib_arc)
    }

    /// Get function from plugin (cached)
    pub fn get_function(&self, library: &Arc<Library>, name: &str) -> Result<PluginFn> {
        let cache_key = format!("{:p}:{}", Arc::as_ptr(library), name);

        tb_debug_plugin!("get_function: Looking for function '{}'", name);

        // Check cache
        if let Some(func) = self.function_cache.get(&cache_key) {
            tb_debug_plugin!("get_function: Using cached function");
            return Ok(*func.value());
        }

        // Load function
        tb_debug_plugin!("get_function: Loading function from library...");
        let func: Symbol<PluginFn> = unsafe {
            library.get(name.as_bytes()).map_err(|e| {
                tb_debug_plugin!("get_function: Function '{}' not found: {}", name, e);
                tb_debug_plugin!("get_function: This usually means:");
                tb_debug_plugin!("  1. Function is not exported (missing #[no_mangle] or extern \"C\")");
                tb_debug_plugin!("  2. Function name is mangled");
                tb_debug_plugin!("  3. Function signature doesn't match PluginFn");
                TBError::plugin_error(format!("Function '{}' not found: {}", name, e))
            })?
        };

        tb_debug_plugin!("get_function: Successfully loaded function '{}'", name);

        let func_ptr = *func;
        self.function_cache.insert(cache_key, func_ptr);

        Ok(func_ptr)
    }

    /// Call plugin function with automatic FFI conversion
    pub fn call_function(
        &self,
        library: &Arc<Library>,
        name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        let func = self.get_function(library, name)?;

        // Convert arguments to FFI format
        let ffi_args: Vec<FFIValue> = args.iter().map(FFIValue::from_value).collect();

        // Call function
        let result = unsafe { func(ffi_args.as_ptr(), ffi_args.len()) };

        // Convert result back
        let value = unsafe { result.to_value() };

        // Cleanup
        for mut arg in ffi_args {
            unsafe { arg.free() };
        }

        Ok(value)
    }

    pub fn unload_library(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_string();
        tb_debug_plugin!("Unloading library: {}", path_str);

        if let Some((_, lib_arc)) = self.loaded_libraries.remove(&path_str) {
            let lib_ptr_str = format!("{:p}:", Arc::as_ptr(&lib_arc));
            self.function_cache.retain(|key, _| !key.starts_with(&lib_ptr_str));
            true
        } else {
            false
        }
    }

    pub fn list_loaded_libraries(&self) -> Vec<String> {
        self.loaded_libraries.iter().map(|r| r.key().clone()).collect()
    }

    /// Load and execute plugin with runtime execution support
    pub fn load_and_execute(
        &self,
        language: &tb_core::PluginLanguage,
        mode: &tb_core::PluginMode,
        library_path: &Path,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {        use tb_core::{PluginLanguage, PluginMode};

        tb_debug_plugin!("load_and_execute: {:?} {:?} {} {}", language, mode, library_path.display(), function_name);

        match (language, mode) {
            (PluginLanguage::Python, PluginMode::Jit) => {
                tb_debug_plugin!("Executing Python JIT from file: {}", library_path.display());
                self.execute_python_jit(library_path, function_name, args)
            }
            (PluginLanguage::JavaScript, PluginMode::Jit) => {
                tb_debug_plugin!("Executing JavaScript JIT from file: {}", library_path.display());
                self.execute_js_jit(library_path, function_name, args)
            }
            (PluginLanguage::Go, PluginMode::Jit) => {
                tb_debug_plugin!("Executing Go JIT from file: {}", library_path.display());
                self.execute_go_jit(library_path, function_name, args)
            }
            (PluginLanguage::Rust, PluginMode::Compile) => {
                // For Rust external files, we need to compile them first
                tb_debug_plugin!("Compiling Rust external file: {}", library_path.display());

                // Read the source code
                let source_code = std::fs::read_to_string(library_path).map_err(|e| TBError::plugin_error(format!("Failed to read Rust source file: {}", e)))?;

                // Compile to library
                let compiled_lib = self.compile_inline_to_library(language, &source_code)?;

                // Load and call
                tb_debug_plugin!("Loading compiled Rust library: {}", compiled_lib.display());
                let library = self.load_library(&compiled_lib)?;
                self.call_function(&library, function_name, args)
            }
            (PluginLanguage::Go, PluginMode::Compile) => {
                // For Go external files, we need to compile them first
                tb_debug_plugin!("Compiling Go external file: {}", library_path.display());

                // Read the source code
                let source_code = std::fs::read_to_string(library_path).map_err(|e| TBError::plugin_error(format!("Failed to read Go source file: {}", e)))?;

                // Compile to library
                let compiled_lib = self.compile_inline_to_library(language, &source_code)?;

                // Load and call
                tb_debug_plugin!("Loading compiled Go library: {}", compiled_lib.display());
                let library = self.load_library(&compiled_lib)?;
                self.call_function(&library, function_name, args)
            }
            _ => {
                // Compiled plugins use standard FFI
                tb_debug_plugin!("Loading compiled plugin library: {}", library_path.display());
                let library = self.load_library(library_path)?;
                self.call_function(&library, function_name, args)
            }
        }
    }

    /// Execute plugin from inline source code
    pub fn execute_inline(
        &self,
        language: &tb_core::PluginLanguage,
        mode: &tb_core::PluginMode,
        source_code: &str,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        use tb_core::{PluginLanguage, PluginMode};

        tb_debug_plugin!("execute_inline: {:?} {:?} function={}", language, mode, function_name);

        match (language, mode) {
            (PluginLanguage::Python, PluginMode::Jit) => {
                tb_debug_plugin!("Executing Python JIT inline");
                self.execute_python_jit_inline(source_code, function_name, args)
            }
            (PluginLanguage::JavaScript, PluginMode::Jit) => {
                tb_debug_plugin!("Executing JavaScript JIT inline");
                self.execute_js_jit_inline(source_code, function_name, args)
            }
            (PluginLanguage::Go, PluginMode::Jit) => {
                tb_debug_plugin!("Executing Go JIT inline");
                self.execute_go_jit_inline(source_code, function_name, args)
            }
            (_, PluginMode::Compile) => {
                tb_debug_plugin!("Compiling inline source code for {:?}", language);
                // Compile inline source to native library and execute via FFI
                let library_path = self.compile_inline_to_library(language, source_code)?;
                let library = self.load_library(&library_path)?;
                self.call_function(&library, function_name, args)
            }
            _ => {
                tb_debug_plugin!("Inline execution not supported for {:?} {:?}", language, mode);
                Err(TBError::plugin_error(format!("Unsupported plugin configuration: {:?} {:?}", language, mode)))
            }
        }
    }

    /// Compile inline source code to native library
    fn compile_inline_to_library(
        &self,
        language: &tb_core::PluginLanguage,
        source_code: &str,
    ) -> Result<std::path::PathBuf> {
        use std::io::Write;
        use std::process::Command;
        use tb_core::PluginLanguage;

        // Create cache directory
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("tb-lang")
            .join("plugins");

        std::fs::create_dir_all(&cache_dir).map_err(|e| TBError::plugin_error(format!("Failed to create cache directory: {}", e)))?;

        // Generate hash-based filename for caching
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        source_code.hash(&mut hasher);
        let hash = hasher.finish();

        match language {
            PluginLanguage::Python => {
                tb_debug_plugin!("Compiling Python inline code with Nuitka");

                // Write source to temporary file
                let source_file = cache_dir.join(format!("plugin_{}.py", hash));
                let mut file = std::fs::File::create(&source_file).map_err(|e| TBError::plugin_error(format!("Failed to create source file: {}", e)))?;
                file.write_all(source_code.as_bytes()).map_err(|e| TBError::plugin_error(format!("Failed to write source file: {}", e)))?;

                // Compile with Nuitka
                let output_name = format!("plugin_{}", hash);
                let output_path = if cfg!(windows) {
                    cache_dir.join(format!("{}.pyd", output_name))
                } else {
                    cache_dir.join(format!("{}.so", output_name))
                };

                // Check if already compiled
                if output_path.exists() {
                    tb_debug_plugin!("Using cached compiled plugin: {}", output_path.display());
                    return Ok(output_path);
                }

                tb_debug_plugin!("Compiling with Nuitka: {} -> {}", source_file.display(), output_path.display());

                let status = Command::new("python")
                    .args(&[
                        "-m", "nuitka",
                        "--module",
                        "--output-dir", cache_dir.to_str().unwrap(),
                        source_file.to_str().unwrap(),
                    ])
                    .status()
                    .map_err(|e| TBError::plugin_error(format!("Failed to run Nuitka: {}. Make sure Nuitka is installed (pip install nuitka)", e)))?;

                if !status.success() {
                    return Err(TBError::plugin_error("Nuitka compilation failed"));
                }

                Ok(output_path)
            }
            PluginLanguage::JavaScript => {
                Err(TBError::plugin_error("JavaScript inline compilation not yet implemented. Use JIT mode instead."))
            }
            PluginLanguage::Rust => {
                tb_debug_plugin!("Compiling Rust inline code");

                // Create temporary Cargo project
                let project_dir = cache_dir.join(format!("rust_plugin_{}", hash));
                std::fs::create_dir_all(&project_dir).map_err(|e| TBError::plugin_error(format!("Failed to create project directory: {}", e)))?;

                // Write Cargo.toml
                let cargo_toml = format!(r#"[package]
name = "plugin_{}"
version = "0.1.0"
edition = "2021"

[lib]
name = "plugin_{}"
crate-type = ["cdylib"]

[dependencies]
rayon = "1.8"
"#, hash, hash);
                std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).map_err(|e| TBError::plugin_error(format!("Failed to write Cargo.toml: {}", e)))?;

                // Create src directory
                let src_dir = project_dir.join("src");
                std::fs::create_dir_all(&src_dir).map_err(|e| TBError::plugin_error(format!("Failed to create src directory: {}", e)))?;

                // Write lib.rs with source code
                std::fs::write(src_dir.join("lib.rs"), source_code).map_err(|e| TBError::plugin_error(format!("Failed to write lib.rs: {}", e)))?;

                // Determine output library name
                let output_name = if cfg!(windows) {
                    format!("plugin_{}.dll", hash)
                } else if cfg!(target_os = "macos") {
                    format!("libplugin_{}.dylib", hash)
                } else {
                    format!("libplugin_{}.so", hash)
                };

                let output_path = cache_dir.join(&output_name);

                // Check if already compiled
                if output_path.exists() {
                    tb_debug_plugin!("Using cached Rust plugin: {}", output_path.display());
                    return Ok(output_path);
                }

                tb_debug_plugin!("Compiling Rust plugin: {}", project_dir.display());

                // Compile with cargo
                let status = Command::new("cargo")
                    .args(&["build", "--release"])
                    .current_dir(&project_dir)
                    .status()
                    .map_err(|e| TBError::plugin_error(format!("Failed to run cargo: {}. Make sure Rust is installed.", e)))?;

                if !status.success() {
                    return Err(TBError::plugin_error("Rust compilation failed"));
                }

                tb_debug_plugin!("Rust compilation succeeded");

                // List files in target/release to see what was actually created
                let target_dir = project_dir.join("target/release");
                tb_debug_plugin!("Checking target directory: {}", target_dir.display());

                if let Ok(entries) = std::fs::read_dir(&target_dir) {
                    tb_debug_plugin!("Files in target/release:");
                    for entry in entries.flatten() {
                        let file_name = entry.file_name();
                        let file_name_str = file_name.to_string_lossy();
                        // Only show DLL/SO/DYLIB files
                        if file_name_str.ends_with(".dll") ||
                           file_name_str.ends_with(".so") ||
                           file_name_str.ends_with(".dylib") {
                            tb_debug_plugin!("  - {}", file_name_str);
                        }
                    }
                }

                // Copy compiled library to cache
                let compiled_lib_name = if cfg!(windows) {
                    format!("plugin_{}.dll", hash)
                } else if cfg!(target_os = "macos") {
                    format!("libplugin_{}.dylib", hash)
                } else {
                    format!("libplugin_{}.so", hash)
                };

                let compiled_lib = target_dir.join(&compiled_lib_name);

                tb_debug_plugin!("Looking for compiled library: {}", compiled_lib.display());

                if !compiled_lib.exists() {
                    tb_debug_plugin!("ERROR: Compiled library not found!");
                    tb_debug_plugin!("Expected: {}", compiled_lib.display());
                    return Err(TBError::plugin_error(format!("Compiled library not found: {}", compiled_lib.display())));
                }

                tb_debug_plugin!("Found compiled library, copying to cache...");
                std::fs::copy(&compiled_lib, &output_path).map_err(|e| TBError::plugin_error(format!("Failed to copy compiled library: {}", e)))?;

                tb_debug_plugin!("Successfully copied to: {}", output_path.display());

                Ok(output_path)
            }
            PluginLanguage::Go => {
                tb_debug_plugin!("Compiling Go inline code to C-shared library");

                // Create temporary Go project
                let project_dir = cache_dir.join(format!("go_plugin_{}", hash));
                std::fs::create_dir_all(&project_dir).map_err(|e| TBError::plugin_error(format!("Failed to create Go project directory: {}", e)))?;

                // Write Go source file with CGO exports
                let go_file = project_dir.join("plugin.go");

                // Wrap functions with CGO exports
                let wrapped_code = self.wrap_go_code_for_cgo(source_code)?;

                std::fs::write(&go_file, wrapped_code).map_err(|e| TBError::plugin_error(format!("Failed to write Go source file: {}", e)))?;

                // Determine output library name
                let output_name = if cfg!(windows) {
                    format!("plugin_{}.dll", hash)
                } else if cfg!(target_os = "macos") {
                    format!("libplugin_{}.dylib", hash)
                } else {
                    format!("libplugin_{}.so", hash)
                };

                let output_path = cache_dir.join(&output_name);

                // Check if already compiled
                if output_path.exists() {
                    tb_debug_plugin!("Using cached Go plugin: {}", output_path.display());
                    return Ok(output_path);
                }

                tb_debug_plugin!("Compiling Go plugin: {}", project_dir.display());

                // Compile with go build -buildmode=c-shared
                let status = Command::new("go")
                    .args(&[
                        "build",
                        "-buildmode=c-shared",
                        "-o", output_path.to_str().unwrap(),
                        go_file.to_str().unwrap(),
                    ])
                    .current_dir(&project_dir)
                    .status()
                    .map_err(|e| TBError::plugin_error(format!("Failed to run go build: {}. Make sure Go is installed.", e)))?;

                if !status.success() {
                    return Err(TBError::plugin_error("Go compilation failed"));
                }

                tb_debug_plugin!("Go compilation succeeded: {}", output_path.display());

                Ok(output_path)
            }
        }
    }

    #[cfg(feature = "python")]
    fn execute_python_jit(
        &self,
        script_path: &Path,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;

        Python::with_gil(|py| {
            let code = std::fs::read_to_string(script_path)
                .map_err(|e| TBError::plugin_error(format!("Failed to read Python script: {}", e)))?;

            let module = PyModule::from_code(py, &code, "plugin", "plugin")
                .map_err(|e| TBError::plugin_error(format!("Python compilation error: {}", e)))?;

            let func = module.getattr(function_name)
                .map_err(|e| TBError::plugin_error(format!("Function '{}' not found: {}", function_name, e)))?;

            let py_args = self.values_to_python(py, args)?;
            let result = func.call(py_args.as_ref(py), None)
                .map_err(|e| TBError::plugin_error(format!("Python execution error: {}", e)))?;

            self.python_to_value(result)
        })
    }

    #[cfg(feature = "python")]
    pub fn execute_python_jit_inline(
        &self,
        source_code: &str,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;

        tb_debug_plugin!("Python inline JIT execution:");
        tb_debug_plugin!("Function: {}", function_name);
        tb_debug_plugin!("Source code:\n{}", source_code);
        tb_debug_plugin!("--- End of source code ---");

        Python::with_gil(|py| {
            let module = PyModule::from_code(py, source_code, "plugin", "plugin")
                .map_err(|e| TBError::plugin_error(format!("Python compilation error: {}", e)))?;

            let func = module.getattr(function_name)
                .map_err(|e| TBError::plugin_error(format!("Function '{}' not found: {}", function_name, e)))?;

            let py_args = self.values_to_python(py, args)?;
            let result = func.call(py_args.as_ref(py), None)
                .map_err(|e| TBError::plugin_error(format!("Python execution error: {}", e)))?;

            self.python_to_value(result)
        })
    }

    #[cfg(not(feature = "python"))]
    fn execute_python_jit(
        &self,
        _script_path: &Path,
        _function_name: &str,
        _args: Vec<Value>,
    ) -> Result<Value> {
        Err(TBError::plugin_error("Python support not enabled. Rebuild with --features python"))
    }

    #[cfg(not(feature = "python"))]
    pub fn execute_python_jit_inline(
        &self,
        _source_code: &str,
        _function_name: &str,
        _args: Vec<Value>,
    ) -> Result<Value> {
        Err(TBError::plugin_error("Python support not enabled. Rebuild with --features python"))
    }

    #[cfg(feature = "javascript")]
    fn execute_js_jit(
        &self,
        script_path: &Path,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        use boa_engine::{Context, JsValue, Source};

        let mut context = Context::default();
        let code = std::fs::read_to_string(script_path)
            .map_err(|e| TBError::plugin_error(format!("Failed to read JavaScript file: {}", e)))?;

        context.eval(Source::from_bytes(&code))
            .map_err(|e| TBError::plugin_error(format!("JavaScript compilation error: {}", e)))?;

        let func = context.eval(Source::from_bytes(function_name))
            .map_err(|e| TBError::plugin_error(format!("Function '{}' not found: {}", function_name, e)))?;

        let js_args = self.values_to_js(&mut context, args)?;

        let result = func
            .as_callable()
            .ok_or_else(|| TBError::plugin_error(format!("{} is not a function", function_name)))?
            .call(&JsValue::undefined(), &js_args, &mut context)
            .map_err(|e| TBError::plugin_error(format!("JavaScript execution error: {}", e)))?;

        self.js_to_value(result)
    }

    #[cfg(feature = "javascript")]
    pub fn execute_js_jit_inline(
        &self,
        source_code: &str,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        use boa_engine::{Context, JsValue, Source};

        let mut context = Context::default();

        context.eval(Source::from_bytes(source_code))
            .map_err(|e| TBError::plugin_error(format!("JavaScript compilation error: {}", e)))?;

        let func = context.eval(Source::from_bytes(function_name))
            .map_err(|e| TBError::plugin_error(format!("Function '{}' not found: {}", function_name, e)))?;

        let js_args = self.values_to_js(&mut context, args)?;

        let result = func
            .as_callable()
            .ok_or_else(|| TBError::plugin_error(format!("{} is not a function", function_name)))?
            .call(&JsValue::undefined(), &js_args, &mut context)
            .map_err(|e| TBError::plugin_error(format!("JavaScript execution error: {}", e)))?;

        self.js_to_value(result)
    }

    #[cfg(not(feature = "javascript"))]
    fn execute_js_jit(
        &self,
        _script_path: &Path,
        _function_name: &str,
        _args: Vec<Value>,
    ) -> Result<Value> {
        Err(TBError::plugin_error("JavaScript support not enabled. Rebuild with --features javascript"))
    }

    #[cfg(not(feature = "javascript"))]
    pub fn execute_js_jit_inline(
        &self,
        _source_code: &str,
        _function_name: &str,
        _args: Vec<Value>,
    ) -> Result<Value> {
        Err(TBError::plugin_error("JavaScript support not enabled. Rebuild with --features javascript"))
    }

    // Helper methods for Python conversion
    #[cfg(feature = "python")]
    fn values_to_python(&self, py: pyo3::Python, values: Vec<Value>) -> Result<pyo3::Py<pyo3::types::PyTuple>> {
        use pyo3::prelude::*;
        use pyo3::types::PyTuple;

        let py_values: Result<Vec<PyObject>> = values.iter().map(|v| self.value_to_python(py, v)).collect();
        Ok(PyTuple::new(py, py_values?).into())
    }

    #[cfg(feature = "python")]
    fn value_to_python(&self, py: pyo3::Python, value: &Value) -> Result<pyo3::PyObject> {
        use pyo3::prelude::*;
        use pyo3::types::{PyList, PyDict};

        match value {
            Value::None => Ok(py.None()),
            Value::Bool(b) => Ok(b.to_object(py)),
            Value::Int(i) => Ok(i.to_object(py)),
            Value::Float(f) => Ok(f.to_object(py)),
            Value::String(s) => Ok(s.as_ref().to_object(py)),
            Value::List(items) => {
                // Convert TB list to Python list
                let py_list = PyList::empty(py);
                for item in items.iter() {
                    let py_item = self.value_to_python(py, item)?;
                    py_list.append(py_item)
                        .map_err(|e| TBError::plugin_error(format!("Failed to append to Python list: {}", e)))?;
                }
                Ok(py_list.to_object(py))
            }
            Value::Dict(map) => {
                // Convert TB dict to Python dict
                let py_dict = PyDict::new(py);
                for (key, val) in map.iter() {
                    let py_val = self.value_to_python(py, val)?;
                    py_dict.set_item(key.as_ref(), py_val)
                        .map_err(|e| TBError::plugin_error(format!("Failed to set Python dict item: {}", e)))?;
                }
                Ok(py_dict.to_object(py))
            }
            _ => {
                tb_debug_plugin!("Unsupported value type for Python conversion: {:?}", value);
                Ok(py.None())
            }
        }
    }

    #[cfg(feature = "python")]
    fn python_to_value(&self, obj: &pyo3::PyAny) -> Result<Value> {
        use pyo3::types::{PyList, PyDict};
        use im::HashMap as ImHashMap;

        if obj.is_none() {
            Ok(Value::None)
        } else if let Ok(b) = obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = obj.extract::<i64>() {
            Ok(Value::Int(i))
        } else if let Ok(f) = obj.extract::<f64>() {
            // Check for NaN
            if f.is_nan() {
                tb_debug_plugin!("Python returned NaN, converting to None");
                Ok(Value::None)
            } else {
                Ok(Value::Float(f))
            }
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(Value::String(std::sync::Arc::new(s)))
        } else if let Ok(py_list) = obj.downcast::<PyList>() {
            // Convert Python list to TB list
            let mut items = Vec::new();
            for item in py_list.iter() {
                items.push(self.python_to_value(item)?);
            }
            Ok(Value::List(std::sync::Arc::new(items)))
        } else if let Ok(py_dict) = obj.downcast::<PyDict>() {
            // Convert Python dict to TB dict (using im::HashMap)
            let mut map = ImHashMap::new();
            for (key, val) in py_dict.iter() {
                if let Ok(key_str) = key.extract::<String>() {
                    map.insert(
                        std::sync::Arc::new(key_str),
                        self.python_to_value(val)?
                    );
                }
            }
            Ok(Value::Dict(std::sync::Arc::new(map)))
        } else {
            tb_debug_plugin!("Unsupported Python type, returning None");
            Ok(Value::None)
        }
    }

    // Helper methods for JavaScript conversion
    #[cfg(feature = "javascript")]
    fn values_to_js(&self, context: &mut boa_engine::Context, values: Vec<Value>) -> Result<Vec<boa_engine::JsValue>> {
        values.iter().map(|v| self.value_to_js(context, v)).collect()
    }

    #[cfg(feature = "javascript")]
    fn value_to_js(&self, context: &mut boa_engine::Context, value: &Value) -> Result<boa_engine::JsValue> {
        use boa_engine::JsValue;
        use boa_engine::object::builtins::JsArray;

        match value {
            Value::None => Ok(JsValue::null()),
            Value::Bool(b) => Ok(JsValue::from(*b)),
            Value::Int(i) => Ok(JsValue::from(*i)),
            Value::Float(f) => Ok(JsValue::from(*f)),
            Value::String(s) => Ok(JsValue::from(s.as_ref().clone())),
            Value::List(items) => {
                // Convert TB list to JS array
                let js_array = JsArray::new(context);
                for (i, item) in items.iter().enumerate() {
                    let js_item = self.value_to_js(context, item)?;
                    js_array.set(i, js_item, true, context)
                        .map_err(|e| TBError::plugin_error(format!("Failed to set array element: {}", e)))?;
                }
                Ok(js_array.into())
            }
            Value::Dict(map) => {
                // Convert TB dict to JS object
                let js_obj = boa_engine::object::JsObject::with_null_proto();
                for (key, val) in map.iter() {
                    let js_val = self.value_to_js(context, val)?;
                    js_obj.set(key.as_ref().clone(), js_val, true, context)
                        .map_err(|e| TBError::plugin_error(format!("Failed to set object property: {}", e)))?;
                }
                Ok(js_obj.into())
            }
            _ => {
                tb_debug_plugin!("Unsupported value type for JS conversion: {:?}", value);
                Ok(JsValue::null())
            }
        }
    }

    #[cfg(feature = "javascript")]
    fn js_to_value(&self, js_val: boa_engine::JsValue) -> Result<Value> {
        use std::collections::HashMap;

        if js_val.is_null() || js_val.is_undefined() {
            Ok(Value::None)
        } else if let Some(b) = js_val.as_boolean() {
            Ok(Value::Bool(b))
        } else if let Some(n) = js_val.as_number() {
            if n.fract() == 0.0 {
                Ok(Value::Int(n as i64))
            } else {
                Ok(Value::Float(n))
            }
        } else if let Some(s) = js_val.as_string() {
            Ok(Value::String(std::sync::Arc::new(s.to_std_string_escaped())))
        } else if let Some(js_obj) = js_val.as_object() {
            // Check if it's an array by checking for 'length' property
            let mut context = boa_engine::Context::default();

            if let Ok(length_val) = js_obj.get("length", &mut context) {
                if let Some(length) = length_val.as_number() {
                    // It's an array - convert to TB list
                    let length = length as usize;
                    let mut items = Vec::new();
                    for i in 0..length {
                        let item = js_obj.get(i, &mut context)
                            .map_err(|e| TBError::plugin_error(format!("Failed to get array element: {}", e)))?;
                        items.push(self.js_to_value(item)?);
                    }
                    return Ok(Value::List(std::sync::Arc::new(items)));
                }
            }

            // It's a regular object - convert to TB dict
            // For now, just return None as we don't have a good way to iterate object keys
            // in boa_engine 0.17 without using private APIs
            tb_debug_plugin!("JS object to dict conversion not fully implemented, returning None");
            Ok(Value::None)
        } else {
            tb_debug_plugin!("Unsupported JS value type, returning None");
            Ok(Value::None)
        }
    }

    pub fn unload_all(&self) {
        self.loaded_libraries.clear();
        self.function_cache.clear();
    }

    /// Execute Go code in JIT mode (from file)
    fn execute_go_jit(
        &self,
        script_path: &Path,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        let code = std::fs::read_to_string(script_path)
            .map_err(|e| TBError::plugin_error(format!("Failed to read Go script: {}", e)))?;

        self.execute_go_jit_inline(&code, function_name, args)
    }

    /// Execute Go code in JIT mode (inline)
    fn execute_go_jit_inline(
        &self,
        source_code: &str,
        function_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        use std::process::Command;

        tb_debug_plugin!("Executing Go JIT: function={}", function_name);

        // Create temporary directory
        let temp_dir = std::env::temp_dir().join(format!("tb_go_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).map_err(|e| TBError::plugin_error(format!("Failed to create temp directory: {}", e)))?;

        // Create Go file with main wrapper
        let go_file = temp_dir.join("main.go");

        // Convert args to Go format
        let go_args = self.values_to_go_args(&args);

        // Remove "package main" from source_code if it exists (we'll add it ourselves)
        let cleaned_code = source_code
            .lines()
            .filter(|line| !line.trim().starts_with("package "))
            .collect::<Vec<_>>()
            .join("\n");

        // Wrap code in main function
        let wrapped_code = format!(
            r#"package main

import "fmt"

{}

func main() {{
    result := {}({})
    fmt.Println(result)
}}
"#,
            cleaned_code,
            function_name,
            go_args
        );

        std::fs::write(&go_file, wrapped_code).map_err(|e| TBError::plugin_error(format!("Failed to write Go file: {}", e)))?;

        tb_debug_plugin!("Running: go run {}", go_file.display());

        // Execute with go run
        let output = Command::new("go")
            .args(&["run", go_file.to_str().unwrap()])
            .output()
            .map_err(|e| TBError::plugin_error(format!("Failed to run Go: {}. Make sure Go is installed.", e)))?;

        // Clean up temp directory
        let _ = std::fs::remove_dir_all(&temp_dir);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TBError::plugin_error(format!("Go execution failed: {}", stderr)));
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_str = stdout.trim();

        tb_debug_plugin!("Go output: {}", result_str);

        // Try to parse as different types
        if let Ok(i) = result_str.parse::<i64>() {
            Ok(Value::Int(i))
        } else if let Ok(f) = result_str.parse::<f64>() {
            Ok(Value::Float(f))
        } else if result_str == "true" {
            Ok(Value::Bool(true))
        } else if result_str == "false" {
            Ok(Value::Bool(false))
        } else {
            Ok(Value::String(std::sync::Arc::new(result_str.to_string())))
        }
    }

    /// Convert TB values to Go function arguments
    fn values_to_go_args(&self, args: &[Value]) -> String {
        args.iter()
            .map(|v| match v {
                Value::Int(i) => i.to_string(),
                Value::Float(f) => f.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::String(s) => format!("\"{}\"", s),
                Value::List(list) => {
                    let elements: Vec<String> = list.iter()
                        .map(|v| match v {
                            Value::Int(i) => i.to_string(),
                            Value::Float(f) => f.to_string(),
                            _ => "0".to_string(),
                        })
                        .collect();
                    format!("[]int{{{}}}", elements.join(", "))
                }
                _ => "nil".to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Wrap Go code with CGO exports for compilation
    fn wrap_go_code_for_cgo(&self, source_code: &str) -> Result<String> {
        // Remove "package main" from source_code if it exists (we'll add it ourselves)
        let cleaned_code = source_code
            .lines()
            .filter(|line| !line.trim().starts_with("package "))
            .collect::<Vec<_>>()
            .join("\n");

        // Add CGO exports for each function
        // For now, just add the necessary imports and package declaration
        let wrapped = format!(
            r#"package main

import "C"
import (
    "sync"
)

{}

func main() {{}}
"#,
            cleaned_code
        );

        Ok(wrapped)
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

