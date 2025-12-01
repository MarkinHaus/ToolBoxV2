// =================== Nuitka Client ===================
// Ersetzt ToolboxClient (PyO3) mit Nuitka-basierter Implementation
// Gleiche API wie ToolboxClient, aber nutzt AppSingleton + NuitkaModuleLoader

use crate::app_singleton::AppSingleton;
use crate::nuitka_loader::{NuitkaModuleLoader, NuitkaModule};
use crate::python_ffi::PythonFFI;
use anyhow::{Result, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

/// Nuitka Python Instance
#[derive(Debug, Clone)]
struct NuitkaPyInstance {
    id: String,
    module_cache: HashMap<String, bool>,
    last_used: Instant,
    active_requests: usize,
}

/// Nuitka Client - Ersetzt ToolboxClient
#[derive(Clone)]
pub struct NuitkaClient {
    ffi: Arc<PythonFFI>,
    app_singleton: Arc<Mutex<AppSingleton>>,
    module_loader: Arc<NuitkaModuleLoader>,
    instances: Arc<Mutex<Vec<NuitkaPyInstance>>>,
    max_instances: usize,
    timeout: Duration,
    maintenance_last_run: Arc<Mutex<Instant>>,
    pub client_prefix: String,
    app_initialized: Arc<Mutex<bool>>,  // Track if init_app() was called
}

/// Errors für NuitkaClient
#[derive(Debug, thiserror::Error)]
pub enum NuitkaClientError {
    #[error("Python error: {0}")]
    PyError(String),
    #[error("Python error: {0}")]
    PythonError(String),
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

impl From<anyhow::Error> for NuitkaClientError {
    fn from(err: anyhow::Error) -> Self {
        NuitkaClientError::Unknown(err.to_string())
    }
}

impl NuitkaClient {
    /// Erstellt neuen NuitkaClient
    pub fn new(
        max_instances: usize,
        timeout_seconds: u64,
        client_prefix: String,
    ) -> Result<Self> {
        info!("Creating NuitkaClient with max_instances={}, timeout={}s", max_instances, timeout_seconds);

        // Erstelle PythonFFI
        let ffi = Arc::new(PythonFFI::new()?);

        // Erstelle AppSingleton
        let app_singleton = Arc::new(Mutex::new(AppSingleton::new(ffi.clone())?));

        // Erstelle ModuleLoader
        let module_loader = Arc::new(NuitkaModuleLoader::new(ffi.clone()));

        Ok(Self {
            ffi,
            app_singleton,
            module_loader,
            instances: Arc::new(Mutex::new(Vec::new())),
            max_instances,
            timeout: Duration::from_secs(timeout_seconds),
            maintenance_last_run: Arc::new(Mutex::new(Instant::now())),
            client_prefix,
            app_initialized: Arc::new(Mutex::new(false)),
        })
    }

    /// Initialisiert Python-Instanzen und lädt Module
    pub async fn initialize(&self, modules: Vec<String>, attr_name: Option<&str>) -> Result<(), NuitkaClientError> {
        info!("Initializing NuitkaClient with modules: {:?}", modules);

        // Ensure App is initialized (only once)
        let mut app_init = self.app_initialized.lock().unwrap();
        if !*app_init {
            info!("First initialize() call - creating Python instance and calling init_app()");
            drop(app_init); // Release lock before async operation

            // Create first instance (this calls init_app())
            let instance_id = self.create_python_instance().await?;

            // Mark as initialized
            *self.app_initialized.lock().unwrap() = true;

            // Enable WebSocket bridge
            info!("Enabling Rust WebSocket bridge...");
            if let Err(e) = self.enable_ws_bridge().await {
                error!("Failed to enable WebSocket bridge: {}", e);
            } else {
                info!("WebSocket bridge enabled successfully");
            }

            // TEMPORARILY DISABLED: Load modules into this instance
            // TODO: Fix event loop deadlock issue in call_module_function
            info!("Module preloading DISABLED - modules will be loaded on-demand");
            // for module in modules {
            //     info!("Preloading module: {}", module);
            //     if let Err(e) = self.load_module_into_instance(&instance_id, &module, attr_name).await {
            //         error!("Failed to preload module {}: {}", module, e);
            //     }
            // }
        } else {
            info!("App already initialized - using existing instance");
            drop(app_init);

            // Get existing instance
            let instance_id = self.instances.lock().unwrap()[0].id.clone();

            // TEMPORARILY DISABLED: Load modules into existing instance
            // TODO: Fix event loop deadlock issue in call_module_function
            info!("Module preloading DISABLED - modules will be loaded on-demand");
            // for module in modules {
            //     info!("Preloading module: {}", module);
            //     if let Err(e) = self.load_module_into_instance(&instance_id, &module, attr_name).await {
            //         error!("Failed to preload module {}: {}", module, e);
            //     }
            // }
        }

        Ok(())
    }

    /// Erstellt neue Python-Instanz
    async fn create_python_instance(&self) -> Result<String, NuitkaClientError> {
        debug!("Creating new Python instance");

        // Check max instances
        {
            let instances = self.instances.lock().unwrap();
            if instances.len() >= self.max_instances {
                return Err(NuitkaClientError::MaxInstancesReached);
            }
        }

        let client_prefix = self.client_prefix.clone();
        let app_singleton = self.app_singleton.clone();

        // Spawn blocking task
        let result = tokio::task::spawn_blocking(move || -> Result<String, NuitkaClientError> {
            // Generate instance ID
            let instance_id = format!("{}_{}", client_prefix, Uuid::new_v4());

            // Initialize App Singleton
            let mut app = app_singleton.lock().unwrap();
            app.init_app(&instance_id)?;

            info!("Created Python instance: {}", instance_id);
            Ok(instance_id)
        }).await.map_err(|e| NuitkaClientError::Unknown(format!("Task join error: {}", e)))??;

        // Store instance
        {
            let mut instances = self.instances.lock().unwrap();
            instances.push(NuitkaPyInstance {
                id: result.clone(),
                module_cache: HashMap::new(),
                last_used: Instant::now(),
                active_requests: 0,
            });
        }

        Ok(result)
    }

    /// Lädt Modul in Instanz
    async fn load_module_into_instance(
        &self,
        instance_id: &str,
        module_name: &str,
        attr_name: Option<&str>,
    ) -> Result<(), NuitkaClientError> {
        debug!("Loading module {} into instance {}", module_name, instance_id);

        let instance_id = instance_id.to_string();
        let module_name = module_name.to_string();
        let attr_name = attr_name.map(String::from).unwrap_or_else(|| "init_mod".to_string());
        let app_singleton = self.app_singleton.clone();
        let instances = self.instances.clone();

        tokio::task::spawn_blocking(move || -> Result<(), NuitkaClientError> {
            // Get App instance
            let mut app = app_singleton.lock().unwrap();
            let app_obj = app.get_app()?;

            // Call init_mod(module_name)
            let kwargs = serde_json::json!({
                "module_name": module_name
            });

            app.call_module_function("", &attr_name, kwargs)?;

            // Update module cache
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
        }).await.map_err(|e| NuitkaClientError::Unknown(format!("Task join error: {}", e)))?
    }

    /// Stellt sicher, dass Modul geladen ist
    async fn ensure_module_loaded(&self, module_name: String) -> Result<String, NuitkaClientError> {
        debug!("Ensuring module {} is loaded", module_name);

        // Check if module is already loaded
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

        // Find instance to load into or create new one
        let instance_id = {
            let instances = self.instances.lock().unwrap();
            if instances.is_empty() {
                None
            } else {
                instances.iter()
                    .min_by_key(|i| i.active_requests)
                    .map(|i| i.id.clone())
            }
        };

        let instance_id = match instance_id {
            Some(id) => id,
            None => self.create_python_instance().await?,
        };

        // TEMPORARILY DISABLED: Load module
        // TODO: Fix event loop deadlock issue in call_module_function
        info!("On-demand module loading DISABLED - assuming module is already available");
        // self.load_module_into_instance(&instance_id, &module_name, None).await?;

        Ok(instance_id)
    }

    /// Ruft Modul-Funktion auf
    pub async fn call_module(
        &self,
        module: String,
        function: String,
        kwargs: Value,
    ) -> Result<Value, NuitkaClientError> {
        info!("Calling module function: {}::{}", module, function);

        // Ensure module is loaded
        let instance_id = self.ensure_module_loaded(module.clone()).await?;

        // Increment active requests
        {
            let mut instances = self.instances.lock().unwrap();
            for inst in instances.iter_mut() {
                if inst.id == instance_id {
                    inst.active_requests += 1;
                    inst.last_used = Instant::now();
                    break;
                }
            }
        }

        let app_singleton = self.app_singleton.clone();
        let instances = self.instances.clone();

        // Call function
        let result = tokio::task::spawn_blocking(move || -> Result<Value, NuitkaClientError> {
            let mut app = app_singleton.lock().unwrap();
            let result = app.call_module_function(&module, &function, kwargs)?;
            Ok(result)
        }).await.map_err(|e| NuitkaClientError::Unknown(format!("Task join error: {}", e)))??;

        // Decrement active requests
        {
            let mut instances = instances.lock().unwrap();
            for inst in instances.iter_mut() {
                if inst.id == instance_id {
                    inst.active_requests = inst.active_requests.saturating_sub(1);
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Gibt Statistiken zurück
    pub fn get_stats(&self) -> Value {
        let instances = self.instances.lock().unwrap();
        serde_json::json!({
            "total_instances": instances.len(),
            "max_instances": self.max_instances,
            "instances": instances.iter().map(|i| {
                serde_json::json!({
                    "id": i.id,
                    "active_requests": i.active_requests,
                    "loaded_modules": i.module_cache.len(),
                    "last_used_seconds_ago": i.last_used.elapsed().as_secs(),
                })
            }).collect::<Vec<_>>(),
        })
    }

    /// Aktiviert die Rust WebSocket Bridge in der Python App.
    async fn enable_ws_bridge(&self) -> Result<(), NuitkaClientError> {
        info!("Enabling Rust WebSocket bridge in Python App...");

        // Rufe set_rust_ws_bridge() in app_singleton.py auf
        // KORREKTUR: Funktionsname angepasst von "enable_rust_ws_bridge" zu "set_rust_ws_bridge"
        let app_singleton = self.app_singleton.lock().unwrap();
        let result = app_singleton.call_function_json("set_rust_ws_bridge", &serde_json::json!({}))
            .map_err(|e| NuitkaClientError::PythonError(format!("Failed to enable WebSocket bridge: {}", e)))?;

        info!("WebSocket bridge enabled: {:?}", result);
        Ok(())
    }
}

