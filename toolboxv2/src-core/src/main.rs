use actix_web::{web, App,HttpRequest, HttpServer, HttpResponse, middleware};
use actix_files as fs;
use actix_session::{Session, SessionMiddleware, storage::CookieSessionStore};
use actix_web::cookie::{Key};
use actix_web::http::Method;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use config::{Config, File, FileFormat};
use env_logger;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use std::process::{Command, Child};
use std::thread::ThreadId;
use std::time::{Duration, Instant};
use futures::executor::block_on;
use futures::TryFutureExt;
use serde_json::json;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{timeout, sleep};
use uuid::Uuid;
use tracing::{info, warn, error, debug};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::RefCell;
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use base64::Engine;
use lazy_static::lazy_static;
use serde_json::Value;
use serde_json;
use std::future::Future;
use std::pin::Pin;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyList};
use tokio::task;
use pyo3::PyResult;
use std::env;
use std::path::PathBuf;
use pyo3_asyncio::{tokio::future_into_py, tokio::into_future}; // Added for async support

lazy_static! {
    static ref TOOLBOX_CLIENT: Mutex<Option<ToolboxClient>> = Mutex::new(None);
}

/// A Python toolbox instance that runs within the process
#[derive(Debug, Clone)]
struct PyToolboxInstance {
    id: String,
    module_cache: HashMap<String, bool>,
    py_app: PyObject,
    last_used: Instant,
    active_requests: usize,
}

/// Client for interacting with Python toolbox instances
#[derive(Debug, Clone)]
pub struct ToolboxClient {
    instances: Arc<Mutex<Vec<PyToolboxInstance>>>,
    max_instances: usize,
    timeout: Duration,
    maintenance_last_run: Arc<Mutex<Instant>>,
    pub client_prifix: String,
}

/// Errors that can occur when using the toolbox
#[derive(Debug, thiserror::Error)]
pub enum ToolboxError {
    #[error("Python error: {0}")]
    PyError(String),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Operation timeout")]
    Timeout,
    #[error("No available instances")]
    NoAvailableInstances,
    #[error("Instance not found: {0}")]
    InstanceNotFound(String),
    #[error("Maximum instances reached")]
    MaxInstancesReached,
    #[error("Module not found: {0}")]
    ModuleNotFound(String),
    #[error("Python initialization error: {0}")]
    PythonInitError(String),
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<PyErr> for ToolboxError {
    fn from(err: PyErr) -> Self {
        ToolboxError::PyError(err.to_string())
    }
}

/// Initialize Python properly in the current process
pub fn initialize_python_environment() -> Result<(), ToolboxError> {
    // First try to detect Python from conda environment
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_path = PathBuf::from(&conda_prefix);

        // Set PYTHONHOME to conda environment
        env::set_var("PYTHONHOME", &conda_prefix);

        // Create proper PYTHONPATH for Windows
        let lib_path = format!(
            "{0}\\Lib;{0}\\Lib\\site-packages;{0}\\DLLs",
            conda_prefix.replace("\\", "\\\\")
        );
        env::set_var("PYTHONPATH", &lib_path);

        info!("Using conda environment at: {}", conda_prefix);
        debug!("Set PYTHONHOME={}", conda_prefix);
        debug!("Set PYTHONPATH={}", lib_path);

        return Ok(());
    }

    // If conda not found, try to detect Python using the sys.executable
    // We need to run a Python process to discover this information
    let output = std::process::Command::new("python")
        .args(&["-c", r#"
import sys, json, os
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "sys_path": sys.path,
    "platform": sys.platform
}))
        "#])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            match serde_json::from_str::<serde_json::Value>(&stdout.trim()) {
                Ok(info) => {
                    if let Some(prefix) = info.get("prefix").and_then(|v| v.as_str()) {
                        let prefix_path = PathBuf::from(prefix);

                        // Set PYTHONHOME to the detected prefix
                        env::set_var("PYTHONHOME", prefix);

                        // Create PYTHONPATH based on platform
                        let is_windows = info.get("platform")
                            .and_then(|v| v.as_str())
                            .map(|s| s.contains("win"))
                            .unwrap_or(false);

                        let python_path = if is_windows {
                            format!("{0}\\Lib;{0}\\Lib\\site-packages;{0}\\DLLs",
                                    prefix.replace("\\", "\\\\"))
                        } else {
                            format!("{0}/lib/python3.10:{0}/lib/python3.10/site-packages", prefix)
                        };

                        env::set_var("PYTHONPATH", &python_path);

                        info!("Using Python at: {}", prefix);
                        debug!("Set PYTHONHOME={}", prefix);
                        debug!("Set PYTHONPATH={}", python_path);

                        return Ok(());
                    }
                },
                Err(e) => {
                    warn!("Failed to parse Python environment info: {}", e);
                }
            }
        },
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Python detection script failed: {}", stderr);
        },
        Err(e) => {
            warn!("Failed to execute Python detection script: {}", e);
        }
    }

    // As a last resort, try to use pyo3's auto-detection
    warn!("Falling back to PyO3 auto-detection for Python environment");

    Ok(())
}

/// Initialize the toolbox client and immediately create an instance
pub async fn initialize_toolbox_client(
    max_instances: usize,
    timeout_seconds: u64,
    client_prifix: String,
) -> Result<(), ToolboxError> {
    // Set up Python environment before creating the client
    initialize_python_environment()?;

    // Create the client
    let client = ToolboxClient::new(max_instances, timeout_seconds, client_prifix);

    // Immediately create a Python instance (don't wait for first request)
    client.create_python_instance().await?;

    // Store the client
    let mut client_mutex = TOOLBOX_CLIENT.lock().unwrap();
    *client_mutex = Some(client);

    info!("ToolboxClient initialized with initial instance created");
    Ok(())
}

/// Get the global toolbox client
pub fn get_toolbox_client() -> Result<ToolboxClient, ToolboxError> {
    let client = TOOLBOX_CLIENT.lock().unwrap();
    client.clone().ok_or_else(|| ToolboxError::Unknown("ToolboxClient not initialized".to_string()))
}

impl ToolboxClient {
    /// Create a new ToolboxClient
    pub fn new(max_instances: usize, timeout_seconds: u64, client_prifix: String) -> Self {
        ToolboxClient {
            instances: Arc::new(Mutex::new(Vec::new())),
            max_instances,
            timeout: Duration::from_secs(timeout_seconds),
            maintenance_last_run: Arc::new(Mutex::new(Instant::now())),
            client_prifix,
        }
    }

    /// Initialize Python instances and preload common modules
    pub async fn initialize(&self, modules: Vec<String>, attr_name: Option<&str>) -> Result<(), ToolboxError> {
        info!("Initializing ToolboxClient with modules: {:?}", modules);

        // Get or create an instance
        let instance_id = if self.instances.lock().unwrap().is_empty() {
            self.create_python_instance().await?
        } else {
            self.instances.lock().unwrap()[0].id.clone()
        };

        // Load each module
        for module in modules {
            info!("Preloading module: {}", module);
            if let Err(e) = self.load_module_into_instance(&instance_id, &module, attr_name).await {
                error!("Failed to preload module {}: {}", module, e);
                // Continue loading other modules even if one fails
            }
        }

        Ok(())
    }

    /// Create a new Python instance
    /// Create a new Python instance
    async fn create_python_instance(&self) -> Result<String, ToolboxError> {
        debug!("Creating new Python instance");

        // Check if we've reached max instances
        {
            let instances = self.instances.lock().unwrap();
            if instances.len() >= self.max_instances {
                return Err(ToolboxError::MaxInstancesReached);
            }
        }
        let cp = self.client_prifix.clone();
        // This needs to run in a separate thread because Python code might block
        let result = task::spawn_blocking(move || {

            Python::with_gil(|py| -> Result<(String, PyObject), ToolboxError> {
                // Import the toolboxv2 module
                let toolbox = match PyModule::import(py, "toolboxv2.__main__") {
                    Ok(module) => module,
                    Err(e) => {
                        error!("Failed to import toolboxv2: {}", e);
                        return Err(ToolboxError::PyError(format!("Failed to import toolboxv2: {}", e)));
                    }
                };

                // Get the server_helper function
                let server_helper = match toolbox.getattr("server_helper") {
                    Ok(func) => func,
                    Err(e) => {
                        error!("Failed to get server_helper function: {}", e);
                        return Err(ToolboxError::PyError(format!("Failed to get server_helper function: {}", e)));
                    }
                };

                // Call get_app to initialize the app
                let kwargs = PyDict::new(py);

                // Generate a unique ID for this instance
                let instance_id = format!("{}_{}", cp, Uuid::new_v4());
                kwargs.set_item("instance_id", instance_id.clone()).unwrap();

                let app = match server_helper.call((), Some(kwargs)) {
                    Ok(app) => app,
                    Err(e) => {
                        error!("Failed to call get_app: {}", e);
                        return Err(ToolboxError::PyError(format!("Failed to call get_app: {}", e)));
                    }
                };


                Ok((instance_id, app.into_py(py)))
            })
        }).await.map_err(|e| ToolboxError::Unknown(format!("Task join error: {}", e)))?;

        // Store the instance
        let clone_res = result?.clone();
        {
            let mut instances = self.instances.lock().unwrap();
            instances.push(PyToolboxInstance {
                id: clone_res.0.clone(),
                module_cache: HashMap::new(),
                py_app: clone_res.1,
                last_used: Instant::now(),
                active_requests: 0,
            });
        }

        info!("Created new Python instance: {}", clone_res.0);
        Ok(clone_res.0)
    }


    /// Load a module into an instance
    async fn load_module_into_instance(&self, instance_id: &str, module_name: &str, attr_name: Option<&str>) -> Result<(), ToolboxError> {
        debug!("Loading module {} into instance {}", module_name, instance_id);

        // Clone for async task
        let instance_id = instance_id.to_string();
        let module_name = module_name.to_string();
        let c_attr_name = attr_name.clone().map(String::from).unwrap_or_else(|| "init_mod".to_string());
        let instances = Arc::clone(&self.instances);

        // This needs to run in a separate thread because Python code might block
        task::spawn_blocking(move || {
            Python::with_gil(|py| -> Result<(), ToolboxError> {
                // Get the instance from the list
                let mut instance_opt = None;
                {
                    let instances_guard = instances.lock().unwrap();
                    for inst in instances_guard.iter() {
                        if inst.id == instance_id {
                            instance_opt = Some(inst.clone());
                            break;
                        }
                    }
                }

                let instance = instance_opt.ok_or_else(|| {
                    ToolboxError::InstanceNotFound(format!("Instance not found: {}", instance_id))
                })?;

                // Get the Python app object
                let app = instance.py_app.as_ref(py);

                // Call the init_mod method to load the module
                let init_mod = app.getattr(c_attr_name.as_str())?;
                init_mod.call1((module_name.clone(),))?;
                // Update module cache in the instance
                {
                    let mut instances_guard = instances.lock().unwrap();
                    for inst in instances_guard.iter_mut() {
                        if inst.id == instance_id {
                            inst.module_cache.insert(module_name.clone(), true);
                            break;
                        }
                    }
                }

                Ok(())
            })
        }).await.map_err(|e| ToolboxError::Unknown(format!("Task join error: {}", e)))?
    }

    /// Convert Rust value to Python
    fn value_to_py(&self, py: Python, value: &Value) -> PyResult<PyObject> {
        match value {
            Value::Null => Ok(py.None()),
            Value::Bool(b) => Ok(b.into_py(py)),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(i.into_py(py))
                } else if let Some(f) = n.as_f64() {
                    Ok(f.into_py(py))
                } else {
                    // Fallback to string if number can't be represented
                    Ok(n.to_string().into_py(py))
                }
            },
            Value::String(s) => Ok(s.into_py(py)),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    // Append each converted value to the list
                    py_list.append(self.value_to_py(py, item)?)?;
                }
                Ok(py_list.to_object(py))
            },
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (key, val) in obj {
                    py_dict.set_item(key, self.value_to_py(py, val)?)?;
                }
                Ok(py_dict.to_object(py))
            }
        }
    }

    /// Convert Python value to Rust
    fn py_to_value<'a>(&self, py: Python<'a>, value: &'a PyAny) -> PyResult<serde_json::Value> {
        if value.is_none() {
            return Ok(serde_json::Value::Null);
        }

        if let Ok(b) = value.extract::<bool>() {
            return Ok(serde_json::Value::Bool(b));
        }

        if let Ok(i) = value.extract::<i64>() {
            return Ok(serde_json::Value::Number(i.into()));
        }

        if let Ok(f) = value.extract::<f64>() {
            return Ok(serde_json::json!(f));
        }

        if let Ok(s) = value.extract::<String>() {
            return Ok(serde_json::Value::String(s));
        }

        // Check if it's a list/tuple
        if let Ok(seq) = value.downcast::<PyList>() {
            let mut arr = Vec::new();
            for item in seq.iter() {
                arr.push(self.py_to_value(py, item)?);
            }
            return Ok(serde_json::Value::Array(arr));
        }

        if let Ok(tup) = value.downcast::<PyTuple>() {
            let mut arr = Vec::new();
            for item in tup.iter() {
                arr.push(self.py_to_value(py, item)?);
            }
            return Ok(serde_json::Value::Array(arr));
        }

        // Check if it's a dict
        if let Ok(dict) = value.downcast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, val) in dict.iter() {
                let key_str = key.extract::<String>()?;
                map.insert(key_str, self.py_to_value(py, val)?);
            }
            return Ok(serde_json::Value::Object(map));
        }

        // Try to convert complex objects to string as fallback
        if let Ok(s) = value.str()?.extract::<String>() {
            return Ok(serde_json::Value::String(s));
        }

        // Default to null if we can't convert
        Ok(serde_json::Value::Null)
    }

    /// Make sure a module is loaded and get an instance that has it
    async fn ensure_module_loaded(&self, module_name: String) -> Result<String, ToolboxError> {
        debug!("Ensuring module {} is loaded", module_name);

        // Check if module is already loaded in an instance
        let instance_with_module = {
            let instances = self.instances.lock().unwrap();
            instances.iter()
                .filter(|i| i.module_cache.get(&module_name).copied().unwrap_or(false))
                .min_by_key(|i| i.active_requests)
                .map(|i| i.id.clone())
        };

        if let Some(instance_id) = instance_with_module {
            return Ok(instance_id);
        }

        // If no instance has this module, find an instance to load it into
        // or create a new instance if needed
        let instance_id = {
            let instances = self.instances.lock().unwrap();
            if instances.is_empty() {
                // No instances, need to create one
                None
            } else {
                // Find least loaded instance
                instances.iter()
                    .min_by_key(|i| i.active_requests)
                    .map(|i| i.id.clone())
            }
        };

        // Create new instance if needed
        let instance_id = if let Some(id) = instance_id {
            id
        } else {
            self.create_python_instance().await?
        };
        let instances = self.instances.lock().unwrap().clone();
        // Load the module into the instance
        if let Some(inst) = instances.iter().find(|inst| inst.id == instance_id) {
            // If the module is already loaded, return early
            if inst.module_cache.contains_key(&module_name) {
                return Ok(instance_id); // Module is already loaded, no need to load again
            }

            // Otherwise, load the module
            self.load_module_into_instance(&instance_id, &module_name, Option::from("init_mod")).await?;
        }

        Ok(instance_id)
    }

    /// Get a valid instance for a specific module
    async fn get_instance_for_module(&self, module_name: &str) -> Result<String, ToolboxError> {
        // Run maintenance if needed
        self.run_maintenance_if_needed().await;

        // Ensure module is loaded and get an instance
        let instance_id = self.ensure_module_loaded(module_name.to_string()).await?;

        // Check if instance is overloaded
        let needs_new_instance = {
            let instances = self.instances.lock().unwrap();
            let total_instances = instances.len();

            if let Some(instance) = instances.iter().find(|i| i.id == instance_id) {
                instance.active_requests > 5 && total_instances < self.max_instances
            } else {
                false
            }
        };

        // Create new instance if needed, but don't wait for it
        if needs_new_instance {
            let client_clone = self.clone();
            let module_name = module_name.to_string();
            tokio::spawn(async move {
                if let Err(e) = client_clone.ensure_module_loaded(module_name).await {
                    warn!("Failed to initialize additional instance: {}", e);
                }
            });
        }

        // Update instance stats
        {
            let mut instances = self.instances.lock().unwrap();
            if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
                instance.active_requests += 1;
                instance.last_used = Instant::now();
            }
        }

        Ok(instance_id)
    }

    /// Run maintenance tasks
    async fn run_maintenance_if_needed(&self) {
        let should_run = {
            let mut last_run = self.maintenance_last_run.lock().unwrap();
            if last_run.elapsed() > Duration::from_secs(30) {
                *last_run = Instant::now();
                true
            } else {
                false
            }
        };

        if should_run {
            self.cleanup_unused_instances().await;
        }
    }

    /// Clean up unused instances
    async fn cleanup_unused_instances(&self) {
        let mut to_remove = Vec::new();

        // Identify instances to remove
        {
            let instances = self.instances.lock().unwrap();
            for (idx, instance) in instances.iter().enumerate() {
                if instance.last_used.elapsed() > Duration::from_secs(600) && instance.active_requests == 0 {
                    to_remove.push(idx);
                }
            }
        }

        // Don't remove the last instance
        {
            let instances = self.instances.lock().unwrap();
            if instances.len() <= 1 && !to_remove.is_empty() {
                debug!("Not removing the last Python instance");
                to_remove.clear();
            }
        }

        // Remove instances (in reverse order to maintain indices)
        if !to_remove.is_empty() {
            let mut instances = self.instances.lock().unwrap();
            for idx in to_remove.iter().rev() {
                if *idx < instances.len() {
                    let instance = instances.remove(*idx);
                    debug!("Removed unused Python instance: {}", instance.id);
                }
            }
        }
    }

    /// Mark an instance as done with its current task
    fn mark_instance_done(&self, instance_id: &str) {
        let mut instances = self.instances.lock().unwrap();
        if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
            if instance.active_requests > 0 {
                instance.active_requests -= 1;
            }
            instance.last_used = Instant::now();
        }
    }

    /// Main function to run Python code - now properly handling Python's async nature
    pub async fn run_function(
        &self,
        module_name: &str,
        function_name: &str,
        spec: &str,
        args: Vec<String>,
        kwargs: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ToolboxError> {
        // Get a valid instance for this module.
        let instance_id = self.get_instance_for_module(module_name).await?;

        // Clone needed values.
        let instance_id_clone = instance_id.clone();
        let self_clone = self.clone();
        let module_name = module_name.to_string();
        let function_name = function_name.to_string();
        let spec = spec.to_string();
        let args_clone = args;
        let kwargs_clone = kwargs;

        // Prepare the Python coroutine and convert it into a Rust future.
        // (Do only the blocking work inside spawn_blocking.)
        let py_res = task::spawn_blocking(move || {
            Python::with_gil(|py| -> Result<_, ToolboxError> {
                // Get the instance.
                let instance = {
                    let instances = self_clone.instances.lock().unwrap();
                    instances
                        .iter()
                        .find(|i| i.id == instance_id_clone)
                        .cloned()
                        .ok_or_else(|| ToolboxError::InstanceNotFound(instance_id_clone.clone()))?
                };

                // Get the app object.
                let app = instance.py_app.as_ref(py);

                // Convert args to a Python list.
                let py_args = PyList::empty(py);
                for arg in args_clone {
                    py_args.append(arg.into_py(py))?;
                }

                // Convert kwargs to a Python dict.
                let py_kwargs = PyDict::new(py);
                for (key, value) in kwargs_clone {
                    py_kwargs.set_item(key, self_clone.value_to_py(py, &value)?)?;
                }

                // Add additional kwargs.
                py_kwargs.set_item("tb_run_with_specification", spec)?;
                py_kwargs.set_item("get_results", true)?;

                // Call a_run_function (an async Python method) on the app object.
                let run = app.getattr("run")?;

                // Prepare the function tuple ([module_name, function_name]).
                let function_tuple =
                    PyTuple::new(py, &[module_name.into_py(py), function_name.into_py(py)]);

                // Call the async function to get its coroutine.
                let py_result = run.call((function_tuple, py_args), Some(py_kwargs))?;

                let result = Python::with_gil(|py| self_clone.py_to_value(py, py_result))?;

                Ok(result)
            })
        }).await
            .map_err(|e| ToolboxError::Unknown(format!("Task join error: {}", e)))??;

        // Mark instance as done with this task.
        self.mark_instance_done(&instance_id);

        Ok(py_res)

    }



    /// Get information about all current instances
    pub fn get_instances_info(&self) -> Vec<HashMap<String, String>> {
        let instances = self.instances.lock().unwrap();

        let mut result = Vec::new();
        for instance in instances.iter() {
            let mut info = HashMap::new();
            info.insert("id".to_string(), instance.id.clone());
            info.insert("active_requests".to_string(), instance.active_requests.to_string());
            info.insert("loaded_modules".to_string(),
                        instance.module_cache.keys().cloned().collect::<Vec<_>>().join(", "));
            info.insert("last_used".to_string(), format!("{:?}", instance.last_used.elapsed()));

            result.push(info);
        }

        result
    }
}

// Configuration struct
#[derive(Debug, Deserialize, Clone)]
struct ServerConfig {
    server: ServerSettings,
    toolbox: ToolboxSettings,
    session: SessionSettings,
}

#[derive(Debug, Deserialize, Clone)]
struct ServerSettings {
    ip: String,
    port: u16,
    dist_path: String,
    open_modules: Vec<String>,
    init_modules: Vec<String>,
    watch_modules: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct ToolboxSettings {
    client_prifix: String,
    timeout_seconds: u64,
    max_instances: u16,
}

#[derive(Debug, Deserialize, Clone)]
struct SessionSettings {
    secret_key: String,
    duration_minutes: u64,
    cookie_name: String
}

// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionData {
    jwt_claim: Option<String>,
    validate: bool,
    live_data: HashMap<String, String>,
    exp: DateTime<Utc>,
    ip: String,
    port: String,
    count: u32,
    check: String,
    h_sid: String,
    user_name: Option<String>,
    new: bool,
}

impl Default for SessionData {
    fn default() -> Self {
        SessionData {
            jwt_claim: None,
            validate: false,
            live_data: HashMap::new(),
            exp: Utc::now(),
            ip: String::new(),
            port: String::new(),
            count: 0,
            check: String::new(),
            h_sid: String::new(),
            user_name: None,
            new: true,
        }
    }
}

// Session storage
type SessionStore = Arc<Mutex<HashMap<String, SessionData>>>;

// Request types
#[derive(Debug, Serialize, Deserialize)]
struct ValidateSessionRequest {
    jwt_claim: Option<String>,
    username: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiRequest {
    modul_name: String,
    function_name: String,
    args: Vec<String>,
    kwargs: HashMap<String, serde_json::Value>,
}

// Utility functions
fn generate_session_id() -> String {
    let random_value: u64 = thread_rng().gen();
    format!("0x{:x}", random_value)
}
// Session middleware
struct SessionManager {
    sessions: SessionStore,
    config: SessionSettings,
    client: Arc<ToolboxClient>,
    gray_list: Vec<String>,
    black_list: Vec<String>,
}

impl SessionManager {
    fn new(
        config: SessionSettings,
        client: Arc<ToolboxClient>,
    ) -> Self {
        SessionManager {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            config,
            client,
            gray_list: Vec::new(),
            black_list: Vec::new(),
        }
    }

    fn get_session(&self, session_id: &str) -> SessionData {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(session_id).cloned().unwrap_or_default()
    }

    fn save_session(&self, session_id: &str, data: SessionData) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(session_id.to_string(), data);
    }

    fn delete_session(&self, session_id: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(session_id);
    }

    async fn create_new_session(
        &self,
        ip: String,
        port: String,
        jwt_claim: Option<String>,
        username: Option<String>,
        h_session_id: Option<String>,
    ) -> String {
        let session_id = generate_session_id();
        let h_sid = h_session_id.unwrap_or_else(|| "#0".to_string());

        let session_data = SessionData {
            jwt_claim: jwt_claim.clone(),
            validate: false,
            live_data: HashMap::new(),
            exp: Utc::now(),
            ip: ip.clone(),
            port,
            count: 0,
            check: String::new(),
            h_sid,
            user_name: None,
            new: false,
        };

        self.save_session(&session_id, session_data);

        // Check if IP is in gray or black list
        if self.gray_list.contains(&ip) {
            return "#0".to_string();
        }

        if self.black_list.contains(&ip) {
            return "#0".to_string();
        }

        if let (Some(jwt), Some(user)) = (jwt_claim, username) {
            self.verify_session_id(&session_id, &user, &jwt).await
        } else {
            "#0".to_string()
        }
    }

    async fn verify_session_id(&self, session_id: &str, username: &str, jwt_claim: &str) -> String {
        info!("Verifying session ID: {}", session_id);
        let mut session = self.get_session(session_id);

        // Check JWT validity
        info!("Checking JWT validity for user: {}", username);
        let jwt_valid = match self.client.run_function(
            "CloudM.AuthManager",
            "jwt_check_claim_server_side",
            "",  // Using default spec initially
            vec![],
            {
                let mut map = HashMap::new();
                map.insert("username".to_string(), serde_json::json!(username));
                map.insert("jwt_claim".to_string(), serde_json::json!(jwt_claim));
                map
            },
        ).await {
            Ok(response) => {
                let is_valid = response
                    .get("result")
                    .and_then(|res| res.get("data"))
                    .and_then(|data| data.as_bool())
                    .unwrap_or(false);
                info!("JWT validation result: {}", is_valid);
                is_valid
            },
            Err(e) => {
                error!("JWT validation error: {:?}", e);
                false
            }
        };

        if !jwt_valid {
            info!("JWT validation failed for user: {}", username);
            session.check = "failed".to_string();
            session.count += 1;
            self.save_session(session_id, session);
            return "#0".to_string();
        }

        // Get user by name
        info!("Getting user information for: {}", username);
        let user_result = match self.client.run_function(
            "CloudM.AuthManager",
            "get_user_by_name",
            "",  // Using default spec initially
            vec![],
            {
                let mut map = HashMap::new();
                map.insert("username".to_string(), serde_json::json!(username));
                map
            },
        ).await {
            Ok(response) => response,
            Err(e) => {
                error!("Error getting user information: {}", e);
                session.check = e.to_string();
                session.count += 1;
                self.save_session(session_id, session);
                return "#0".to_string();
            }
        };
        // Ensure user is valid
        if user_result.get("error")
        .and_then(|err| err.as_str())
        .map(|err| err != "none")
        .unwrap_or(true) {
            info!("Invalid user: {}", username);
            session.check = "Invalid user".to_string();
            session.count += 1;
            self.save_session(session_id, session);
            return "#0".to_string();
        }
        let user = user_result.get("result").and_then(|res| res.get("data")).cloned().unwrap_or(serde_json::json!({}));
        let uid = user.get("uid").and_then(Value::as_str).unwrap_or("");
        info!("User UID: {}", uid);

        info!("Getting user instance for UID: {}", uid);
        let instance_result = match self.client.run_function(
                "CloudM.UserInstances",
                "get_user_instance",
                "",
                vec![],
                HashMap::from([
                    ("uid".to_string(), serde_json::json!(uid)),
                    ("hydrate".to_string(), serde_json::json!(false)),
                ]),
            ).await {
                Ok(response) => response,
                Err(e) => {
                    error!("Error getting user instance: {}", e);
                    return "#0".to_string();
                }
            };

            if instance_result.get("error").and_then(Value::as_str).map_or(true, |err| err != "none") {
                info!("Invalid user instance for UID: {}", uid);
                return "#0".to_string();
            }

            let instance = instance_result.get("result").and_then(|res| res.get("data")).cloned().unwrap_or(serde_json::json!({}));
            let mut live_data = HashMap::new();

            if let Some(si_id) = instance.get("SiID").and_then(Value::as_str) {
                live_data.insert("SiID".to_string(), si_id.to_string());
                info!("SiID for user instance: {}", si_id);
            }

            if let Some(level) = user.get("level").and_then(Value::as_u64) {
                let level = level.max(1);
                live_data.insert("level".to_string(), level.to_string());
                info!("User level: {}", level);
            }

            if let Some(vt_id) = instance.get("VtID").and_then(Value::as_str) {
                live_data.insert("spec".to_string(), vt_id.to_string());
                info!("VtID for user instance: {}", vt_id);
            }

            let encoded_username = format!("encoded:{}", username);
            live_data.insert("user_name".to_string(), encoded_username.clone());

            let updated_session = SessionData {
                jwt_claim: Some(jwt_claim.to_string()),
                validate: true,
                exp: Utc::now(),
                user_name: Some(encoded_username),
                count: 0,
                live_data,
                ..session
            };

            info!("Session verified successfully for user: {}", username);
            self.save_session(session_id, updated_session);
            session_id.to_string()
    }

    async fn validate_session(&self, session_id: &str) -> bool {
        info!("Validating session: {}", session_id);

        if session_id.is_empty() {
            info!("Session ID is empty, validation failed");
            return false;
        }

        let session = self.get_session(session_id);

        if session.new || !session.validate {
            info!("Session is new or not validated: new={}, validate={}", session.new, session.validate);
            if let (Some(user_name), Some(jwt)) = (&session.user_name, &session.jwt_claim) {
                // Extract username from encoded format
                let username = user_name.strip_prefix("encoded:").unwrap_or(user_name);
                info!("Verifying session ID for user: {}", username);
                let result = self.verify_session_id(session_id, username, jwt).await != "#0";
                info!("Session verification result: {}", result);
                return result;
            }
        }

        if session.user_name.is_none() || session.jwt_claim.is_none() {
            info!("Session missing user_name or jwt_claim, validation failed");
            return false;
        }

        // Check session expiration
        let session_duration = Duration::from_secs(self.config.duration_minutes * 60);
        let now = Utc::now();
        let session_age = now.signed_duration_since(session.exp);

        info!("Session age: {} seconds, Session duration: {} seconds",
              session_age.num_seconds(), session_duration.as_secs());

        if session_age.num_seconds() > session_duration.as_secs() as i64 {
            info!("Session expired, attempting to re-verify");
            // Session expired, need to verify again
            if let (Some(user_name), Some(jwt)) = (&session.user_name, &session.jwt_claim) {
                // Extract username from encoded format
                let username = user_name.strip_prefix("encoded:").unwrap_or(user_name);
                info!("Re-verifying session ID for user: {}", username);
                let result = self.verify_session_id(session_id, username, jwt).await != "#0";
                info!("Session re-verification result: {}", result);
                return result;
            }
            info!("Session expired and missing user_name or jwt_claim, validation failed");
            return false;
        }

        info!("Session validation successful");
        true
    }
}

// New structs to match Python ApiResult structure
#[derive(Serialize, Deserialize, Debug)]
pub struct ToolBoxInfoBM {
    pub exec_code: i32,
    pub help_text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolBoxResultBM {
    pub data_to: String,
    pub data_info: Option<String>,
    pub data: Option<serde_json::Value>,
    pub data_type: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiResult {
    pub error: Option<String>,
    pub origin: Option<serde_json::Value>,
    pub result: Option<ToolBoxResultBM>,
    pub info: Option<ToolBoxInfoBM>,
}
async fn validate_session_handler(
    manager: web::Data<SessionManager>,
    session: Session,
    body: Option<web::Json<serde_json::Value>>,  // Changed from web::Json<ValidateSessionRequest>
    req_info: HttpRequest,
) -> HttpResponse {
    // Extract client IP - try to get real IP even behind proxy
    let client_ip = req_info.connection_info().realip_remote_addr()
        .unwrap_or_else(|| {
            req_info.headers()
                .get("X-Forwarded-For")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.split(',').next())
                .unwrap_or("unknown")
        })
        .split(':')
        .next()
        .unwrap_or("unknown")
        .to_string();

    // Extract client port
    let client_port = req_info.connection_info().peer_addr()
        .unwrap_or("unknown")
        .split(':')
        .nth(1)
        .unwrap_or("unknown")
        .to_string();

    let current_session_id = session.get::<String>("ID").unwrap_or_else(|_| None);

    // Extract data from the request body
    let (username, jwt_claim) = if let Some(body) = &body {
        let username = body.get("Username").and_then(|u| u.as_str()).map(String::from);
        let jwt_claim = body.get("Jwt_claim").and_then(|j| j.as_str()).map(String::from);
        (username, jwt_claim)
    } else {
        (None, None)
    };

    info!(
        "Validating session - IP: {}, Port: {}, Current Session ID: {:?}, Username: {:?}, JWT Claim: {}",
        client_ip,
        client_port,
        current_session_id,
        username,
        jwt_claim.as_ref().map(|jwt| format!("{:.10}...", jwt)).unwrap_or_else(|| "None".to_string())
    );

    let session_id = manager.create_new_session(
        client_ip,
        client_port,
        jwt_claim,
        username,
        current_session_id,
    ).await;

    let valid = manager.validate_session(&session_id).await;

    // Update session
    if let Err(e) = session.insert("ID", &session_id) {
        error!("Failed to update session: {}", e);
    }

    if let Err(e) = session.insert("valid", valid) {
        error!("Failed to update session validity: {}", e);
    }

    if valid {
        let session_data = manager.get_session(&session_id);
        if let Err(e) = session.insert("live_data", session_data.live_data) {
            error!("Failed to update session live data: {}", e);
        }

        HttpResponse::Ok().json(ApiResult {
            error: Some("none".parse().unwrap()),
            origin: None,
            result: Some(ToolBoxResultBM {
                data_to: "API".to_string(),
                data_info: Some("Valid Session".to_string()),
                data: None,
                data_type: None,
            }),
            info: Some(ToolBoxInfoBM {
                exec_code: 0,
                help_text: "Valid Session".to_string(),
            }),
        })
    } else {
        HttpResponse::Unauthorized().json(ApiResult {
            error: Some("Invalid Auth data.".to_string()),
            origin: None,
            result: None,
            info: None,
        })
    }
}

async fn is_valid_session_handler(
    session: Session,
) -> HttpResponse {
    let valid = match session.get::<bool>("valid") {
        Ok(Some(true)) => true,
        _ => false,
    };
    if valid {
        HttpResponse::Ok().json(ApiResult {
            error: Some("none".parse().unwrap()),
            origin: None,
            result: Some(ToolBoxResultBM {
                data_to: "API".to_string(),
                data_info: Some("Valid Session".to_string()),
                data: None,
                data_type: None,
            }),
            info: Some(ToolBoxInfoBM {
                exec_code: 0,
                help_text: "Valid Session".to_string(),
            }),
        })
    } else {
        HttpResponse::Unauthorized().json(ApiResult {
            error: Some("Invalid Auth data.".to_string()),
            origin: None,
            result: None,
            info: None,
        })
    }
}

async fn logout_handler(
    manager: web::Data<SessionManager>,
    session: Session,
) -> HttpResponse {
    let valid = match session.get::<bool>("valid") {
        Ok(Some(true)) => true,
        _ => false,
    };
    let client = match get_toolbox_client() {
        Ok(client) => Arc::new(client),
        Err(e) => {
            panic!("{:?}", e)
        }
    };
    if valid {
        if let Ok(Some(live_data)) = session.get::<HashMap<String, String>>("live_data") {
            if let Some(si_id) = live_data.get("SiID") {
                // Get instance UID
                let instance_result = client.run_function(
                    "CloudM.UserInstances",
                    "get_instance_si_id",
                    live_data.get("spec").unwrap_or(&String::new()),
                    vec![],
                    {
                        let mut map = HashMap::new();
                        map.insert("si_id".to_string(), serde_json::json!(si_id));
                        map
                    },
                ).await.unwrap_or_else(|e| {
                    log::error!("Error getting instance by si_id: {}", e);
                    serde_json::json!({})
                });

                if let Some(uid) = instance_result.get("result")
                    .and_then(|r| r.get("save"))
                    .and_then(|s| s.get("uid"))
                    .and_then(|u| u.as_str())
                {
                    // Close user instance
                    let default_value = String::new();
                    let close_result = client.run_function(
                        "CloudM.UserInstances",
                        "close_user_instance",
                        live_data.get("spec").unwrap_or(&default_value),
                        vec![],
                        {
                            let mut map = HashMap::new();
                            map.insert("uid".to_string(), serde_json::json!(uid));
                            map
                        },
                    );

                    if let Err(e) = close_result.await {
                        log::warn!("Error closing user instance: {}", e);
                        // Continue with logout even if closing instance fails
                    }

                    // Delete session
                    if let Ok(Some(session_id)) = session.get::<String>("ID") {
                        manager.delete_session(&session_id);
                    }

                    // Clear session
                    session.purge();

                    return HttpResponse::Found()
                        .append_header(("Location", "/web/logout"))
                        .finish();
                }
            }
        }

        // Clear session if we couldn't properly log out
        session.purge();
    }

    HttpResponse::Forbidden().json(ApiResult {
        error: Some("Invalid Auth data.".to_string()),
        origin: None,
        result: None,
        info: None,
    })
}


async fn api_handler(
     req: HttpRequest,
    path: web::Path<(String, String)>,
    query: web::Query<HashMap<String, String>>,
    body: Option<web::Json<serde_json::Value>>,
    session: Session,
    open_modules: web::Data<Arc<Vec<String>>>,
) -> HttpResponse {
    let (module_name, function_name) = path.into_inner();

    // Check if the request is for a protected module or function
    let is_protected = !open_modules.contains(&module_name) && !function_name.starts_with("open");

    if is_protected {
        let valid = match session.get::<bool>("valid") {
            Ok(Some(true)) => true,
            _ => false,
        };
        if !valid {
            // Return unauthorized error with ApiResult format
            return HttpResponse::Unauthorized().json(ApiResult {
                error: Some("Unauthorized".to_string()),
                origin: None,
                result: None,
                info: None,
            });
        }
    }

    let live_data = session.get::<HashMap<String, String>>("live_data").unwrap_or_else(|_| None);
    info!("API FOR: {:?} {:?}", live_data, session.get::<HashMap<String, String>>("ip").unwrap_or_else(|_| None));

    // Get specification from live_data
    let spec = live_data
        .as_ref()
        .and_then(|data| data.get("spec"))
        .cloned()
        .unwrap_or_else(|| "app".to_string());

    // Convert query parameters
    let args: Vec<String> = Vec::new(); // Path params would go here if needed

    let mut kwargs: HashMap<String, serde_json::Value> = query.into_inner()
        .into_iter()
        .map(|(k, v)| (k, serde_json::json!(v)))
        .collect();

    // Check if this is a POST request and add data parameter only if it is
    if req.method() == Method::POST {
        // Extract the data from the request body (if it exists)
        if let Some(body_data) = body {
            let data = body_data.into_inner();
            kwargs.insert("data".to_string(), data);
        }
    }

   let request_metadata = serde_json::json!({
        "session": live_data.unwrap_or_default(),
        "request": {
            "path": req.path(),
            "headers": req.headers().iter().map(|(k, v)| (k.as_str(), v.to_str().unwrap_or(""))).collect::<HashMap<_, _>>(),
            "method": req.method().as_str(),
        }
    });

    info!("request_metadata: {:?}", request_metadata);

    kwargs.insert("request".to_string(), request_metadata);

    let client = match get_toolbox_client() {
        Ok(client) => Arc::new(client),
        Err(e) => {
            return HttpResponse::InternalServerError().json(ApiResult {
                error: Some(format!("Error getting toolbox client: {:?}", e)),
                origin: None,
                result: None,
                info: None,
            });
        }
    };

    // Run function via toolbox client
    match client.run_function(&module_name, &function_name, &spec, args, kwargs).await {
        Ok(response) => {
            // The Python client.run_function already returns an ApiResult object
            // which can be directly returned
            HttpResponse::Ok().json(response)
        },
        Err(e) => {
            HttpResponse::InternalServerError().json(ApiResult {
                error: Some(format!("Error: {:?}", e)),
                origin: None,
                result: None,
                info: None,
            })
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {

    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    // Load configuration
    let config_result = Config::builder()
        .add_source(File::new("config.toml", FileFormat::Toml))
        .build();

    let config: ServerConfig = match config_result {
        Ok(c) => c.try_deserialize().expect("Invalid configuration format"),
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            std::process::exit(1);
        }
    };

    info!("Configuration loaded: {:?}", config);

    let _ = initialize_toolbox_client(config.toolbox.max_instances as usize,  // Port range to use
                              config.toolbox.timeout_seconds,            // Timeout in seconds
                              config.toolbox.client_prifix,            // Timeout in seconds

    ).await;


    let client = match get_toolbox_client() {
        Ok(client) => Arc::new(client),
        Err(e) => {
            panic!("{:?}", e)
        }
    };

    info!("init_modules loaded: {:?} - {:?}", config.server.init_modules, client.initialize(config.server.init_modules.clone(), Option::from("init_mod")).await.map_err(|e| ToolboxError::from(e)));

    info!("watch_modules loaded: {:?} - {:?}", config.server.watch_modules, client.initialize(config.server.watch_modules.clone(), Option::from("watch_mod")).await.map_err(|e| ToolboxError::from(e)));

    // Create session manager
    let session_manager = web::Data::new(SessionManager::new(
        config.session.clone(),
        Arc::from(client.clone()),
    ));

    // Generate session key
    let key = Key::from(config.session.secret_key.as_bytes());

    // Start server
    info!("Starting server on {}:{}", config.server.ip, config.server.port);
    let dist_path = config.server.dist_path.clone(); // Clone the dist_path here
    let open_modules = Arc::new(config.server.open_modules.clone());
    HttpServer::new(move || {
        let dist_path = dist_path.clone(); // Move the cloned dist_path into the closure
        let open_modules = Arc::clone(&open_modules);
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(middleware::Compress::default())
            .wrap(SessionMiddleware::new(
                CookieSessionStore::default(),
                key.clone(),
            ))
            .app_data(web::Data::clone(&session_manager))
            // API routes
            .service(
                web::scope("/api")
                    .app_data(web::Data::new(open_modules.clone()))
                    .service(web::resource("/{module_name}/{function_name}")
                        .route(web::get().to(api_handler))
                        .route(web::post().to(api_handler))
                        .route(web::delete().to(api_handler))
                        .route(web::put().to(api_handler))
                    )
            )
            .service(web::resource("/validateSession")
                .route(web::post().to(validate_session_handler))
                )
            .service(web::resource("/IsValiSession")
                .route(web::get().to(is_valid_session_handler))
                )
            .service(web::resource("/web/logoutS")
                .route(web::post().to(logout_handler))
                )
            // Serve static files
            .service(fs::Files::new("/", &dist_path) // Use the moved dist_path
                .index_file("index.html"))
            // Default route - serve index.html
            .default_service(
                web::route().to(move || { // Move the dist_path into this closure as well
                    let dist_path = dist_path.clone();
                    async move {
                        HttpResponse::Ok()
                            .content_type("text/html; charset=utf-8")
                            .body(std::fs::read_to_string(
                                format!("{}/index.html", dist_path)
                            ).unwrap_or_else(|_| "404 Not Found".to_string()))
                    }
                })
            )
    })
        .bind(format!("{}:{}", config.server.ip, config.server.port))?
        .run()
        .await
}
