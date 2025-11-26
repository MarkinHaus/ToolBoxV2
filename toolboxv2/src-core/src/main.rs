use actix_web::{web, App, HttpRequest, HttpServer, Error, HttpResponse, middleware, FromRequest};
use actix_web::cookie::{Key};
use actix_web::http::Method;
use actix_web::dev::Service;
use actix_files as fs;
use actix_session::{Session, SessionMiddleware, storage::CookieSessionStore, SessionExt};
use actix_web::dev::Server;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json;

use base64::{engine::general_purpose::STANDARD};
use chrono::{DateTime, Utc};
use config::{Config, File, FileFormat};
use env_logger;
use rand::{thread_rng, Rng};
use std::collections::{HashMap};

use std::time::{Duration, Instant};
use futures::executor::block_on;

use uuid::Uuid;
use tracing::{info, warn, error, debug};

use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;
// PyO3 removed - using Nuitka-only implementation
use simple_core_server::{NuitkaClient, NuitkaClientError};
use tokio::{task, time};
use std::env;
use std::process::Command;
use base64::Engine;
use futures::{Stream};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use bytes::Bytes;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use actix_multipart::Multipart;
use bytes::BytesMut; // To accumulate field data
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _}; // Use alias
use std::io::Write; // For writing bytes
use futures_util::{StreamExt, TryStreamExt};

use listenfd::ListenFd;
use std::path::Path;
use std::net::TcpListener;

use actix_web_actors::ws;
use dashmap::DashMap;
use actix::prelude::*;
use tokio::sync::broadcast;

#[cfg(unix)]
use std::os::unix::io::{FromRawFd, RawFd};

// Define a helper struct for file data representation
#[derive(Serialize, Deserialize, Debug, Clone)]
struct UploadedFile {
    filename: Option<String>,
    content_type: Option<String>,
    content_base64: String, // Store content as base64
}

// =================== PyO3 Helper Functions - REMOVED ===================
// py_object_to_bytes() and py_to_value_global() are no longer needed
// Type conversion is handled by NuitkaModuleLoader in lib.rs
/*
fn py_object_to_bytes(py: Python, obj: PyObject) -> PyResult<Bytes> { ... }
fn py_to_value_global(py: Python, value: &PyAny) -> PyResult<serde_json::Value> { ... }
*/


// =================== InstanceGuard - REMOVED ===================
// InstanceGuard is no longer needed with NuitkaClient
// Instance management is handled internally by NuitkaClient
/*
struct InstanceGuard { ... }
impl InstanceGuard { ... }
impl Drop for InstanceGuard { ... }
*/



/// Globale Senke für den Broadcast-Kanal.
/// Jede Nachricht, die hier gesendet wird, erreicht jeden aktiven WebSocket-Actor.
/// Dies ermöglicht die Kommunikation zwischen Python-Instanzen und 1-zu-N-Szenarien.
lazy_static! {
    static ref GLOBAL_WS_BROADCASTER: broadcast::Sender<WsMessage> = broadcast::channel(1024).0;
}

/// Speichert die Adressen der aktiven WebSocket-Actors, um gezielte 1-zu-1-Nachrichten zu ermöglichen.
/// Schlüssel: conn_id (String), Wert: Addr<WebSocketActor>
lazy_static! {
    static ref ACTIVE_CONNECTIONS: Arc<DashMap<String, Addr<WebSocketActor>>> = Arc::new(DashMap::new());
}

/// Nachrichtentyp für den internen Broadcast-Bus.
#[derive(Message, Clone, Debug)]
#[rtype(result = "()")]
struct WsMessage {
    /// Eindeutige ID der ursprünglichen Verbindung, die die Nachricht gesendet hat (kann leer sein).
    pub source_conn_id: String,
    /// JSON-serialisierter String der Nachricht.
    pub content: String,
    /// Optional: Gibt an, an welche spezifische Verbindung diese Nachricht gerichtet ist.
    pub target_conn_id: Option<String>,
    /// Optional: Gibt an, an welchen Kanal/welche Gruppe diese Nachricht gerichtet ist.
    pub target_channel_id: Option<String>,
}


// --- NEU: Der WebSocket Actor ---

struct WebSocketActor {
    conn_id: String,
    client:  Arc<NuitkaClient>,
    session: SessionData, // Annahme: SessionData wird aus der Session extrahiert
    channel_id: Option<String>, // Der "Raum" oder die Gruppe, der diese Verbindung angehört
    hb: Instant,
}

impl WebSocketActor {
    fn new(client: Arc<NuitkaClient>, session: Session, module: &str, function: &str) -> Self {
        let conn_id = Uuid::new_v4().to_string();
        let session_data = session.get("live_data").unwrap_or(None).unwrap_or_default(); // Vereinfachte Extraktion
        Self {
            conn_id,
            client,
            session: session_data,
            // Kanal-ID wird aus der URL abgeleitet, z.B. /ws/Chat/room123 -> "Chat/room123"
            channel_id: Some(format!("{}/{}", module, function)),
            hb: Instant::now(),
        }
    }

    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(45), |act, ctx| {
            if Instant::now().duration_since(act.hb) > Duration::from_secs(120) {
                warn!("WebSocket Client heartbeat failed, disconnecting!");
                ctx.close(Some(ws::CloseReason::from((ws::CloseCode::Away, "Heartbeat timeout"))));
                return;
            }
            ctx.ping(b"heartbeat");
        });
    }
}

impl Actor for WebSocketActor {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket Actor started: conn_id={}", self.conn_id);
        self.heartbeat(ctx);

        // Registriere den Actor global
        ACTIVE_CONNECTIONS.insert(self.conn_id.clone(), ctx.address());

        // Abonniere den globalen Broadcast-Kanal
        let mut rx = GLOBAL_WS_BROADCASTER.subscribe();
        let addr = ctx.address();

        // Erstelle einen Future, der auf Nachrichten vom Broadcast-Kanal lauscht
        let broadcast_listener = async move {
            while let Ok(msg) = rx.recv().await {
                addr.do_send(msg);
            }
        };
        // Führe den Future im Kontext des Actors aus
        ctx.spawn(broadcast_listener.into_actor(self));

        // Rufe den Python on_connect Handler auf
        let client = self.client.clone();
        let conn_id = self.conn_id.clone();
        let channel_id = self.channel_id.clone().unwrap_or_default();
        let session_data_json = serde_json::to_value(self.session.clone()).unwrap_or(Value::Null);

        tokio::spawn(async move {
            let mut kwargs = HashMap::new();
            kwargs.insert("conn_id".to_string(), Value::String(conn_id));
            kwargs.insert("session".to_string(), session_data_json);
            kwargs.insert("spec".to_string(), Value::String("ws_internal".to_string()));
            kwargs.insert("args".to_string(), Value::Array(vec![]));

            if let Err(e) = client.call_module(
                channel_id, // Der Kanalname dient zur Identifizierung des Handlers
                "on_connect".to_string(),
                serde_json::json!(kwargs),
            ).await {
                error!("Python on_connect handler failed: {:?}", e);
            }
        });
    }

    fn stopping(&mut self, _: &mut Self::Context) -> Running {
        info!("WebSocket Actor stopping: conn_id={}", self.conn_id);
        ACTIVE_CONNECTIONS.remove(&self.conn_id);

        // Rufe den Python on_disconnect Handler auf
        let client = self.client.clone();
        let conn_id = self.conn_id.clone();
        let channel_id = self.channel_id.clone().unwrap_or_default();

        tokio::spawn(async move {
            let mut kwargs = HashMap::new();
            kwargs.insert("conn_id".to_string(), Value::String(conn_id));
            kwargs.insert("spec".to_string(), Value::String("ws_internal".to_string()));
            kwargs.insert("args".to_string(), Value::Array(vec![]));
            if let Err(e) = client.call_module(
                channel_id,
                "on_disconnect".to_string(),
                serde_json::json!(kwargs),
            ).await {
                error!("Python on_disconnect handler failed: {:?}", e);
            }
        });

        Running::Stop
    }
}

/// Handler für Nachrichten vom Broadcast-Kanal
impl Handler<WsMessage> for WebSocketActor {
    type Result = ();

    fn handle(&mut self, msg: WsMessage, ctx: &mut Self::Context) {
        // Filtere Nachrichten:
        // 1. 1-to-1: Wenn target_conn_id gesetzt ist, sende nur, wenn es meine ID ist.
        if let Some(target_id) = &msg.target_conn_id {
            if *target_id == self.conn_id {
                ctx.text(msg.content);
            }
            return;
        }

        // 2. 1-to-n (Channel): Wenn target_channel_id gesetzt ist, sende nur, wenn ich in diesem Kanal bin.
        if let Some(target_channel) = &msg.target_channel_id {
            if self.channel_id.as_deref() == Some(target_channel) {
                // Sende nicht an den ursprünglichen Absender zurück
                if msg.source_conn_id != self.conn_id {
                    ctx.text(msg.content);
                }
            }
            return;
        }

        // 3. Global Broadcast: Wenn weder target_conn_id noch target_channel_id gesetzt ist.
        // Sende nicht an den ursprünglichen Absender zurück
        if msg.source_conn_id != self.conn_id {
            ctx.text(msg.content);
        }
    }
}


/// Handler für eingehende WebSocket-Nachrichten vom Client
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        self.hb = Instant::now();

        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.hb = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                // Leite die Nachricht an den Python on_message Handler weiter
                let client = self.client.clone();
                let conn_id = self.conn_id.clone();
                let channel_id = self.channel_id.clone().unwrap_or_default();
                let text_content = text.to_string();
                let session_data_json = serde_json::to_value(self.session.clone()).unwrap_or(Value::Null);

                tokio::spawn(async move {
                    let mut kwargs = HashMap::new();
                    kwargs.insert("conn_id".to_string(), Value::String(conn_id));
                    kwargs.insert("session".to_string(), session_data_json);

                    // Versuche, die Nachricht als JSON zu parsen
                    let payload = match serde_json::from_str::<Value>(&text_content) {
                        Ok(json_val) => json_val,
                        Err(_) => Value::String(text_content), // Sende als String, wenn kein JSON
                    };
                    kwargs.insert("payload".to_string(), payload);
                    kwargs.insert("spec".to_string(), Value::String("ws_internal".to_string()));
                    kwargs.insert("args".to_string(), Value::Array(vec![]));

                    if let Err(e) = client.call_module(
                        channel_id.clone(),
                        "on_message".to_string(),
                        serde_json::json!(kwargs),
                    ).await {
                        error!("Python on_message handler failed: {:?}", e);
                    }
                });
            }
            Ok(ws::Message::Binary(_bin)) => warn!("Binary WebSocket messages are not supported."),
            Err(e) => {
                error!("WebSocket error: {:?}", e);
                match e {
                    ws::ProtocolError::Io(_) => {
                        // Network errors - close gracefully
                        ctx.close(Some(ws::CloseReason::from((ws::CloseCode::Abnormal, "IO error"))));
                    }
                    _ => {
                        // Other protocol errors - maybe recoverable, just log
                        warn!("Non-fatal protocol error, continuing...");
                    }
                }
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => error!("Unexpected WebSocket condition, but keeping connection alive"), //ctx.stop(),
        }
    }
}


// --- NEU: WebSocket-Endpoint-Handler ---
async fn websocket_handler(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<(String, String)>,
    session: Session,
    open_modules: web::Data<Arc<Vec<String>>>,
    client: web::Data<Arc<NuitkaClient>>, // Add NuitkaClient from app_data
    init_modules: web::Data<Arc<Vec<String>>>, // Add init_modules for lazy initialization
    watch_modules: web::Data<Arc<Vec<String>>>, // Add watch_modules for lazy initialization
) -> Result<HttpResponse, Error> {
    let (module_name, function_name) = path.into_inner();

    // Lazy initialization: Ensure Python backend is initialized in this worker
    // Combine init_modules and watch_modules into a single list to avoid multiple initialize() calls
    let mut all_modules: Vec<String> = init_modules.as_ref().as_ref().clone();
    all_modules.extend(watch_modules.as_ref().as_ref().iter().cloned());
    if let Err(e) = client.initialize(all_modules, Some("init_mod")).await {
        error!("Failed to initialize Python backend: {:?}", e);
    }

    // Berechtigungsprüfung
    let is_protected = !open_modules.contains(&module_name) && !function_name.starts_with("open");
    if is_protected {
        let valid = session.get::<bool>("valid").unwrap_or(None).unwrap_or(false);
        if !valid {
            return Ok(HttpResponse::Unauthorized().finish());
        }
    }

    ws::WsResponseBuilder::new(
        WebSocketActor::new(Arc::clone(&client), session, &module_name, &function_name),
        &req,
        stream,
    )
    .frame_size(16 * 1024 * 1024)
    .start()
}

// =================== PyO3 WebSocket Bridge - REMOVED ===================
// RustWsBridge and rust_bridge_internal are no longer needed
// WebSocket functionality will be reimplemented with Nuitka if needed
/*
#[pyclass]
struct RustWsBridge;
#[pymethods]
impl RustWsBridge { ... }
#[pymodule]
fn rust_bridge_internal(_py: Python, m: &PyModule) -> PyResult<()> { ... }
*/

lazy_static! {
    // Nuitka-only implementation - no PyO3
    static ref NUITKA_CLIENT: Mutex<Option<Arc<NuitkaClient>>> = Mutex::new(None);
}

// =================== PyO3-based structures - REMOVED ===================
// Replaced by NuitkaClient from lib.rs
// PyToolboxInstance, ToolboxClient, ToolboxError are now in nuitka_client.rs


// =================== PyO3-based initialization - REMOVED ===================
// initialize_python_environment() is no longer needed with Nuitka
// Python environment is managed by PythonFFI in lib.rs

/// Initialize the Nuitka client
pub async fn initialize_and_get_nuitka_client(
    max_instances: usize,
    timeout_seconds: u64,
    client_prefix: String,
) -> Result<Arc<NuitkaClient>, NuitkaClientError> {
    info!("Initializing Nuitka client...");

    let client = NuitkaClient::new(max_instances, timeout_seconds, client_prefix)?;
    let client_arc = Arc::new(client);

    let mut client_mutex = NUITKA_CLIENT.lock().unwrap();
    *client_mutex = Some(client_arc.clone());

    info!("NuitkaClient initialized successfully.");
    Ok(client_arc)
}

/// Get the global Nuitka client
pub fn get_nuitka_client() -> Result<Arc<NuitkaClient>, NuitkaClientError> {
    let client_guard = NUITKA_CLIENT.lock().unwrap();
    client_guard.clone().ok_or_else(|| {
        NuitkaClientError::Unknown("NuitkaClient not initialized".to_string())
    })
}


// =================== PyO3 Debug Functions - REMOVED ===================
// check_python_paths() and check_toolboxv2() are no longer needed
/*
fn check_python_paths(py: Python) { ... }
fn check_toolboxv2(py: Python) -> PyResult<()> { ... }
*/

// =================== impl ToolboxClient - REMOVED ===================
// All ToolboxClient methods are now in NuitkaClient (nuitka_client.rs)
// This entire impl block (558-1396) has been replaced by NuitkaClient

/*
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
        debug!("Creating new Python instance at {:?}", env::var("PYO3_PYTHON"));

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
                debug!("this Python {:?}", check_python_paths(py));
                debug!("this tb {:?}", check_toolboxv2(py));
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

                // Bridge-Injektion
                let bridge_module = PyModule::new(py, "rust_bridge_internal")?;
                rust_bridge_internal(py, bridge_module)?;
                let bridge_class = bridge_module.getattr("RustWsBridge")?;
                let bridge_instance = bridge_class.call0()?; // Dies funktioniert jetzt

                if app.hasattr("_set_rust_ws_bridge")? {
                    app.call_method1("_set_rust_ws_bridge", (bridge_instance,))?;
                    info!("Successfully injected Rust WebSocket bridge into Python instance {}.", instance_id);
                } else {
                    warn!("Python App object is missing '_set_rust_ws_bridge' method.");
                }

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

    // Add this new method to support streaming
    pub async fn stream_generator(
        &self,
        module_name: &str,
        function_name: &str,
        spec: &str,
        args: Vec<String>,
        kwargs: HashMap<String, serde_json::Value>,
    ) -> Result<impl Stream<Item = Result<Bytes, io::Error>>, ToolboxError> {
        // Get instance and setup
        let instance_id = self.get_instance_for_module(module_name).await?;
        let self_clone = self.clone();
        let instance_guard = InstanceGuard::new(self_clone.clone(), instance_id);

        // Small buffer to prevent backpressure but not cause delays
        let (tx, rx) = mpsc::channel(1);

        // Clone variables for task
        let module_name = module_name.to_string();
        let function_name = function_name.to_string();
        let spec = spec.to_string();

        // Launch task
        tokio::spawn(async move {
            let guard = instance_guard;
            let instance_id = guard.instance_id.clone();

            // Run Python code in blocking task
            let _ = task::spawn_blocking(move || {
                Python::with_gil(|py| -> Result<(), ToolboxError> {
                    // Get Python instance
                    let instance = {
                        let instances = self_clone.instances.lock().unwrap();
                        match instances.iter().find(|i| i.id == instance_id).cloned() {
                            Some(inst) => inst,
                            None => return Err(ToolboxError::InstanceNotFound(instance_id))
                        }
                    };

                    // Prepare call
                    let app = instance.py_app.as_ref(py);
                    let py_args = PyList::empty(py);
                    for arg in args {
                        py_args.append(arg.into_py(py))?;
                    }

                    let py_kwargs = PyDict::new(py);
                    for (key, value) in kwargs {
                        py_kwargs.set_item(key, self_clone.value_to_py(py, &value)?)?;
                    }
                    py_kwargs.set_item("tb_run_with_specification", spec)?;

                    let func_tuple = PyTuple::new(py, &[module_name.into_py(py), function_name.into_py(py)]);

                    // Call Python function
                    let run_method = app.getattr("run")?;
                    let result = run_method.call((func_tuple, py_args), Some(py_kwargs))?;

                    // Handle start event
                    let start_event = serde_json::json!({"event": "stream_start", "id": "0"});
                    let _ = tx.blocking_send(Ok(Bytes::from(serde_json::to_vec(&start_event).unwrap())));

                    // DIRECT GENERATOR HANDLING
                    if let Ok(dict) = result.downcast::<PyDict>() {
                        // Dict result - check for stream/generator
                        if let Some(generator) = dict.get_item("generator")? {
                            // Process as generator
                            if generator.hasattr("__anext__")? {
                                // Async generator
                                let _asyncio  = py.import("asyncio")?;
                                let helper_code = r#"
def get_next_item(gen):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        return (True, loop.run_until_complete(gen.__anext__()))
    except StopAsyncIteration:
        return (False, None)
    except Exception as e:
        return (False, str(e))
"#;
                                py.run(helper_code, None, None)?;
                                let get_next = py.eval("get_next_item", None, None)?;

                                // Process items one by one - IMMEDIATE SENDING
                                loop {
                                    let result = get_next.call1((generator,))?;
                                    let tuple = result.downcast::<PyTuple>().unwrap();
                                    let has_more = tuple.get_item(0)?.extract::<bool>()?;

                                    if !has_more {
                                        break;
                                    }

                                    let item = tuple.get_item(1)?;
                                    let bytes = py_object_to_bytes(py, item.to_object(py))?;

                                    // SEND IMMEDIATELY
                                    if tx.blocking_send(Ok(bytes)).is_err() {
                                        break;
                                    }
                                }
                            } else {
                                // Sync generator
                                let next_fn = py.import("builtins")?.getattr("next")?;

                                loop {
                                    match next_fn.call1((generator,)) {
                                        Ok(item) => {
                                            let bytes = py_object_to_bytes(py, item.to_object(py))?;
                                            // SEND IMMEDIATELY
                                            if tx.blocking_send(Ok(bytes)).is_err() {
                                                break;
                                            }
                                        },
                                        Err(e) => {
                                            if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                                break;
                                            } else {
                                                let _ = tx.blocking_send(Err(io::Error::new(
                                                    io::ErrorKind::Other,
                                                    format!("Error: {}", e)
                                                )));
                                                return Err(ToolboxError::PyError(e.to_string()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if result.hasattr("__anext__")? {
                        // Direct async generator
                        let _asyncio = py.import("asyncio")?;

                        // Simplified helper
                        py.run(r#"
def process_async_gen(gen):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            item = loop.run_until_complete(gen.__anext__())
            yield item
        except StopAsyncIteration:
            break
"#, None, None)?;

                        let process_gen = py.eval("process_async_gen", None, None)?;
                        let sync_gen = process_gen.call1((result,))?;

                        // Process synchronously
                        let next_fn = py.import("builtins")?.getattr("next")?;
                        loop {
                            match next_fn.call1((sync_gen,)) {
                                Ok(item) => {
                                    let bytes = py_object_to_bytes(py, item.to_object(py))?;
                                    // SEND IMMEDIATELY
                                    if tx.blocking_send(Ok(bytes)).is_err() {
                                        break;
                                    }
                                },
                                Err(e) => {
                                    if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                        break;
                                    } else {
                                        let _ = tx.blocking_send(Err(io::Error::new(
                                            io::ErrorKind::Other,
                                            format!("Error: {}", e)
                                        )));
                                        break;
                                    }
                                }
                            }
                        }
                    } else if result.hasattr("__next__")? {
                        // Direct sync generator
                        let next_fn = py.import("builtins")?.getattr("next")?;

                        loop {
                            match next_fn.call1((result,)) {
                                Ok(item) => {
                                    let bytes = py_object_to_bytes(py, item.to_object(py))?;
                                    // SEND IMMEDIATELY
                                    if tx.blocking_send(Ok(bytes)).is_err() {
                                        break;
                                    }
                                },
                                Err(e) => {
                                    if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                        break;
                                    } else {
                                        let _ = tx.blocking_send(Err(io::Error::new(
                                            io::ErrorKind::Other,
                                            format!("Error: {}", e)
                                        )));
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        // Single item result
                        let bytes = py_object_to_bytes(py, result.to_object(py))?;
                        // SEND IMMEDIATELY
                        let _ = tx.blocking_send(Ok(bytes));
                    }

                    // Send end event
                    let end_event = serde_json::json!({"event": "stream_end", "id": "final"});
                    let _ = tx.blocking_send(Ok(Bytes::from(serde_json::to_vec(&end_event).unwrap())));

                    Ok(())
                })
            }).await;
        });

        // Return stream
        let rx_stream = ReceiverStream::new(rx)
            .inspect(|result| {
                match result {
                    Ok(bytes) => {
                        if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                            debug!("Sending SSE response: {}", text);
                        } else {
                            debug!("Sending binary SSE data: {} bytes", bytes.len());
                        }
                    },
                    Err(e) => debug!("Sending SSE error: {}", e),
                }
            });

        Ok(rx_stream)
    }

    fn py_to_value<'a>(&self, py: Python<'a>, value: &'a PyAny) -> PyResult<serde_json::Value> {
        // ... (your existing implementation) ...
        if value.is_none() { return Ok(serde_json::Value::Null); }
        if let Ok(b) = value.extract::<bool>() { return Ok(serde_json::Value::Bool(b)); }
        if let Ok(i) = value.extract::<i64>() { return Ok(serde_json::Value::Number(i.into())); }
        if let Ok(f) = value.extract::<f64>() { return Ok(serde_json::json!(f)); } // Use json! macro for potential NaN/Infinity
        if let Ok(s) = value.extract::<String>() { return Ok(serde_json::Value::String(s)); }
        if let Ok(seq) = value.downcast::<PyList>() {
            let mut arr = Vec::new();
            for item in seq.iter() { arr.push(self.py_to_value(py, item)?); }
            return Ok(serde_json::Value::Array(arr));
        }
        if let Ok(tup) = value.downcast::<PyTuple>() {
            let mut arr = Vec::new();
            for item in tup.iter() { arr.push(self.py_to_value(py, item)?); }
            return Ok(serde_json::Value::Array(arr));
        }
        if let Ok(dict) = value.downcast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, val) in dict.iter() {
                let key_str = key.extract::<String>()?;
                map.insert(key_str, self.py_to_value(py, val)?);
            }
            return Ok(serde_json::Value::Object(map));
        }
        if let Ok(s) = value.str()?.extract::<String>() { return Ok(serde_json::Value::String(s)); }
        Ok(serde_json::Value::Null) // Fallback
    }


    pub async fn stream_sse_events(&self,
                                   module_name: &str,
                                   function_name: &str,
                                   spec: &str,
                                   args: Vec<String>,
                                   kwargs: HashMap<String, serde_json::Value>,
    ) -> Result<impl Stream<Item = Result<Bytes, std::io::Error>>, ToolboxError> {
        // Get base stream
        let base_stream = self.stream_generator(module_name, function_name, spec, args, kwargs).await?;

        // IMPORTANT: Use fully qualified path to avoid ambiguity
        let mapped_stream = futures::stream::StreamExt::map(base_stream, move |result| {
            match result {
                Ok(data) => {
                    // Format as proper SSE
                    if let Ok(text) = String::from_utf8(data.to_vec()) {
                        let text = text.trim();

                        // Already in SSE format?
                        if (text.starts_with("data:") || text.starts_with("event:")) &&
                            (text.ends_with("\n\n") || text.contains("\n\ndata:") || text.contains("\n\nevent:")) {
                            debug!("Pre-formatted SSE: {}", text.replace("\n", "\\n"));
                            Ok(Bytes::from(text.to_string() + "\n"))
                        } else if text.starts_with("{") && text.ends_with("}") {
                            // JSON data
                            debug!("JSON as SSE: {}", text);
                            Ok(Bytes::from(format!("{}\n\n", text)))
                        } else {
                            // Plain text
                            debug!("Text as SSE: {}", text);
                            Ok(Bytes::from(format!("{}\n\n", text)))
                        }
                    } else {
                        // Binary data
                        debug!("Binary as SSE comment");
                        Ok(Bytes::from(format!(": [binary data: {} bytes]\n\n", data.len())))
                    }
                },
                Err(e) => {
                    error!("SSE stream error: {}", e);
                    Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Error: {}", e)))
                },
            }
        });

        // Add a heartbeat ping every 15 seconds
        struct HeartbeatStream {
            interval: tokio::time::Interval,
        }

        impl Stream for HeartbeatStream {
            type Item = Result<Bytes, std::io::Error>;

            fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                match Pin::new(&mut self.interval).poll_tick(cx) {
                    Poll::Ready(_) => Poll::Ready(Some(Ok(Bytes::from(":\n\n")))),
                    Poll::Pending => Poll::Pending,
                }
            }
        }

        let heartbeat = HeartbeatStream {
            interval: tokio::time::interval(std::time::Duration::from_secs(15)),
        };

        // Merge the streams using fully qualified path
        let combined = futures::stream::select(mapped_stream, heartbeat);

        Ok(combined)
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

    pub async fn send_ws_message(&self, conn_id: String, payload: Value) -> Result<(), ToolboxError> {
        let payload_str = serde_json::to_string(&payload)?;
        task::spawn_blocking(move || {
            Python::with_gil(|py| {
                // Hier würden wir eine in Rust definierte PyFunction aufrufen
                // Für dieses Beispiel simulieren wir den direkten Aufruf.
                if let Some(conn) = ACTIVE_CONNECTIONS.get(&conn_id) {
                    conn.value().do_send(WsMessage {
                        source_conn_id: "python_direct".to_string(),
                        content: payload_str,
                        target_conn_id: Some(conn_id),
                        target_channel_id: None,
                    });
                }
            });
        }).await.map_err(|e| ToolboxError::Unknown(e.to_string()))
    }

    /// Ruft eine Rust-Funktion auf, um eine Nachricht an einen WebSocket-Kanal zu senden.
    pub async fn broadcast_ws_message(&self, channel_id: String, payload: Value, source_conn_id: String) -> Result<(), ToolboxError> {
        let payload_str = serde_json::to_string(&payload)?;
        let msg = WsMessage {
            source_conn_id,
            content: payload_str,
            target_conn_id: None,
            target_channel_id: Some(channel_id),
        };
        if let Err(e) = GLOBAL_WS_BROADCASTER.send(msg) {
            error!("broadcast_ws_message: Failed to send broadcast message: {}", e);
            return Err(ToolboxError::Unknown(format!("Broadcast failed: {}", e)));
        }
        Ok(())
    }
}
*/
// =================== END OF REMOVED ToolboxClient ===================

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
}

// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionData {
    // Clerk-spezifische Felder (NEU)
    clerk_session_token: Option<String>,  // Ersetzt jwt_claim
    clerk_user_id: Option<String>,        // Clerk's User ID

    // Bestehende Felder
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
    anonymous: bool,
}

impl Default for SessionData {
    fn default() -> Self {
        SessionData {
            clerk_session_token: None,
            clerk_user_id: None,
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
            anonymous: true,
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
    let random_value: u64 = thread_rng().random();
    format!("0x{:x}", random_value)
}
// Session middleware
struct SessionManager {
    sessions: SessionStore,
    config: SessionSettings,
    client: Arc<NuitkaClient>,
    gray_list: Vec<String>,
    black_list: Vec<String>,
}


impl SessionManager {
    fn new(
        config: SessionSettings,
        client: Arc<NuitkaClient>,
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
        clerk_session_token: Option<String>,
        clerk_user_id: Option<String>,
        h_session_id: Option<String>,
    ) -> String {
        let session_id = generate_session_id();
        let h_sid = h_session_id.unwrap_or_else(|| "#0".to_string());

        let session_data = SessionData {
            clerk_session_token: clerk_session_token.clone(),
            clerk_user_id: clerk_user_id.clone(),
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
            anonymous: clerk_session_token.is_none() && clerk_user_id.is_none(),
        };

        self.save_session(&session_id, session_data);

        // Check if IP is in gray or black list
        if self.gray_list.contains(&ip) {
            return "#0X".to_string();
        }

        if self.black_list.contains(&ip) {
            return "#0X".to_string();
        }

        if let Some(token) = clerk_session_token {
            self.verify_session_id(&session_id, &token).await
        } else {
            session_id
        }
    }

    /// Verifiziert eine Session mit Clerk's verify_session Endpunkt
    async fn verify_session_id(&self, session_id: &str, clerk_session_token: &str) -> String {
        info!("Verifying session ID with Clerk: {}", session_id);
        let mut session = self.get_session(session_id);

        // Clerk Session Token validieren via CloudM.AuthClerk.verify_session
        info!("Checking Clerk session token validity");
        let verify_result = match self.client.call_module(
            "CloudM.AuthClerk".to_string(),
            "verify_session".to_string(),
            serde_json::json!({
                "session_token": clerk_session_token,
                "spec": "",
                "args": []
            }),
        ).await {
            Ok(response) => response,
            Err(e) => {
                error!("Clerk session verification error: {:?}", e);
                session.check = e.to_string();
                session.count += 1;
                self.save_session(session_id, session);
                return "#0".to_string();
            }
        };

        // Prüfe ob Session authentifiziert ist
        let is_authenticated = verify_result
            .get("result")
            .and_then(|res| res.get("data"))
            .and_then(|data| data.get("authenticated"))
            .and_then(|auth| auth.as_bool())
            .unwrap_or(false);

        if !is_authenticated {
            info!("Clerk session validation failed");
            session.check = "failed".to_string();
            session.count += 1;
            self.save_session(session_id, session);
            return "#0".to_string();
        }

        // Extrahiere User-Daten aus der Clerk-Antwort
        let result_data = verify_result
            .get("result")
            .and_then(|res| res.get("data"))
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let clerk_user_id = result_data
            .get("user_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let username = result_data
            .get("username")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let level = result_data
            .get("level")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1);

        info!("Clerk user authenticated: {} (ID: {})", username, clerk_user_id);

        // Optional: User Instance abfragen (falls noch benötigt)
        let mut live_data = HashMap::new();

        // Clerk User ID speichern
        live_data.insert("clerk_user_id".to_string(), clerk_user_id.clone());
        live_data.insert("level".to_string(), level.to_string());
        live_data.insert("user_name".to_string(), username.clone());

        // Optional: UserInstances Integration (falls noch verwendet)
        if !clerk_user_id.is_empty() {
            info!("Getting user instance for Clerk user ID: {}", clerk_user_id);
            if let Ok(instance_result) = self.client.call_module(
                "CloudM.UserInstances".to_string(),
                "get_user_instance".to_string(),
                serde_json::json!({
                    "uid": clerk_user_id,
                    "hydrate": false,
                    "spec": "",
                    "args": []
                }),
            ).await {
                if instance_result
                    .get("error")
                    .and_then(Value::as_str)
                    .map_or(false, |err| err == "none")
                {
                    let instance = instance_result
                        .get("result")
                        .and_then(|res| res.get("data"))
                        .cloned()
                        .unwrap_or(serde_json::json!({}));

                    if let Some(si_id) = instance.get("SiID").and_then(Value::as_str) {
                        live_data.insert("SiID".to_string(), si_id.to_string());
                        info!("SiID for user instance: {}", si_id);
                    }

                    if let Some(vt_id) = instance.get("VtID").and_then(Value::as_str) {
                        live_data.insert("spec".to_string(), vt_id.to_string());
                        info!("VtID for user instance: {}", vt_id);
                    }
                }
            }
        }

        let updated_session = SessionData {
            clerk_session_token: Some(clerk_session_token.to_string()),
            clerk_user_id: Some(clerk_user_id),
            validate: true,
            exp: Utc::now(),
            user_name: Some(username),
            count: 0,
            live_data,
            ..session
        };

        info!("Session verified successfully");
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

        if session.anonymous {
            info!("Anonymous session: {}", session_id);
            return false;  // Not valid for protected resources
        }

        if session.new || !session.validate {
            info!("Session is new or not validated: new={}, validate={}", session.new, session.validate);
            if let Some(token) = &session.clerk_session_token {
                info!("Verifying session with Clerk token");
                let result = self.verify_session_id(session_id, token).await != "#0";
                info!("Session verification result: {}", result);
                return result;
            }
        }

        if session.clerk_session_token.is_none() {
            info!("Session missing clerk_session_token, validation failed");
            return false;
        }

        // Check session expiration
        let session_duration = Duration::from_secs(self.config.duration_minutes * 60);
        let now = Utc::now();
        let session_age = now.signed_duration_since(session.exp);

        info!("Session age: {} seconds, Session duration: {} seconds",
              session_age.num_seconds(), session_duration.as_secs());

        if session_age.num_seconds() > session_duration.as_secs() as i64 {
            info!("Session expired, attempting to re-verify with Clerk");
            // Session expired, need to verify again with Clerk
            if let Some(token) = &session.clerk_session_token {
                info!("Re-verifying session with Clerk");
                let result = self.verify_session_id(session_id, token).await != "#0";
                info!("Session re-verification result: {}", result);
                return result;
            }
            info!("Session expired and missing clerk_session_token, validation failed");
            return false;
        }

        info!("Session validation successful");
        true
    }

    /// Holt User-Daten von Clerk (optional, falls zusätzliche Daten benötigt werden)
    async fn get_clerk_user_data(&self, clerk_user_id: &str) -> Option<serde_json::Value> {
        match self.client.call_module(
            "CloudM.AuthClerk".to_string(),
            "get_user_data".to_string(),
            serde_json::json!({
                "clerk_user_id": clerk_user_id,
                "spec": "",
                "args": []
            }),
        ).await {
            Ok(response) => {
                if response
                    .get("error")
                    .and_then(Value::as_str)
                    .map_or(false, |err| err == "none")
                {
                    response
                        .get("result")
                        .and_then(|res| res.get("data"))
                        .cloned()
                } else {
                    None
                }
            }
            Err(e) => {
                error!("Error getting Clerk user data: {:?}", e);
                None
            }
        }
    }

    /// Logout - Session beenden und Clerk benachrichtigen
    async fn logout(&self, session_id: &str) -> bool {
        let session = self.get_session(session_id);

        if let Some(clerk_user_id) = &session.clerk_user_id {
            // Clerk über Logout informieren
            let _ = self.client.call_module(
                "CloudM.AuthClerk".to_string(),
                "on_sign_out".to_string(),
                serde_json::json!({
                    "clerk_user_id": clerk_user_id,
                    "spec": "",
                    "args": []
                }),
            ).await;
        }

        self.delete_session(session_id);
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
    body: Option<web::Json<serde_json::Value>>,
    req_info: HttpRequest,
) -> HttpResponse {
    // Extract client IP
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

    let client_port = req_info.connection_info().peer_addr()
        .unwrap_or("unknown")
        .split(':')
        .nth(1)
        .unwrap_or("unknown")
        .to_string();

    let current_session_id = session.get::<String>("ID").unwrap_or_else(|_| None);

    // NEU: Clerk-spezifische Felder extrahieren
    let (clerk_session_token, clerk_user_id) = if let Some(body) = &body {
        let token = body.get("session_token")
            .or_else(|| body.get("Jwt_claim"))  // Fallback für Kompatibilität
            .and_then(|t| t.as_str())
            .map(String::from);
        let user_id = body.get("clerk_user_id")
            .or_else(|| body.get("Username"))  // Fallback für Kompatibilität
            .and_then(|u| u.as_str())
            .map(String::from);
        (token, user_id)
    } else {
        (None, None)
    };

    info!(
        "Validating session - IP: {}, Port: {}, Current Session ID: {:?}, Clerk User ID: {:?}",
        client_ip,
        client_port,
        current_session_id,
        clerk_user_id
    );

    let session_id = manager.create_new_session(
        client_ip,
        client_port,
        clerk_session_token,  // NEU: clerk_session_token statt jwt_claim
        clerk_user_id,        // NEU: clerk_user_id statt username
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
            error: Some("none".to_string()),
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

    if valid {
        if let Ok(Some(session_id)) = session.get::<String>("ID") {
            // Verwende die neue logout() Methode vom SessionManager
            // Diese ruft automatisch CloudM.AuthClerk.on_sign_out auf
            manager.logout(&session_id).await;
        }

        // Clear session
        session.purge();

        return HttpResponse::Found()
            .append_header(("Location", "/web/logout"))
            .finish();
    }

    HttpResponse::Forbidden().json(ApiResult {
        error: Some("Invalid Auth data.".to_string()),
        origin: None,
        result: None,
        info: None,
    })
}

// Füge diesen neuen Handler hinzu
async fn get_user_data_handler(
    manager: web::Data<SessionManager>,
    session: Session,
) -> HttpResponse {
    let valid = match session.get::<bool>("valid") {
        Ok(Some(true)) => true,
        _ => false,
    };

    if !valid {
        return HttpResponse::Unauthorized().json(ApiResult {
            error: Some("Unauthorized: Session invalid.".to_string()),
            origin: None,
            result: None,
            info: None,
        });
    }

    // Hole clerk_user_id aus der Session
    let clerk_user_id = match session.get::<HashMap<String, String>>("live_data") {
        Ok(Some(live_data)) => live_data.get("clerk_user_id").cloned(),
        _ => None,
    };

    let Some(clerk_user_id) = clerk_user_id else {
        return HttpResponse::BadRequest().json(ApiResult {
            error: Some("No Clerk user ID found in session.".to_string()),
            origin: None,
            result: None,
            info: None,
        });
    };

    // Hole User-Daten von Clerk
    match manager.get_clerk_user_data(&clerk_user_id).await {
        Some(user_data) => HttpResponse::Ok().json(ApiResult {
            error: Some("none".to_string()),
            origin: None,
            result: Some(ToolBoxResultBM {
                data_to: "API".to_string(),
                data_info: Some("User data retrieved".to_string()),
                data: Some(user_data),
                data_type: Some("json".to_string()),
            }),
            info: Some(ToolBoxInfoBM {
                exec_code: 0,
                help_text: "Success".to_string(),
            }),
        }),
        None => HttpResponse::NotFound().json(ApiResult {
            error: Some("User data not found.".to_string()),
            origin: None,
            result: None,
            info: None,
        }),
    }
}


fn parse_response(response: ApiResult, fall_back: Value) -> HttpResponse {
    match response.result {
        Some(result) => {
            let data_type = result.data_type.as_deref().unwrap_or("json"); // Default to JSON
            let data_value = result.data.unwrap_or(Value::Null);

            match data_type {
                // === JSON (default) ===
               "json" => HttpResponse::Ok().json(data_value),

                // === Plain Text ===
                "text" | "plain" => HttpResponse::Ok().content_type("text/plain; charset=utf-8").body(
                    data_value.as_str().unwrap_or("").to_string(), // Default to empty string
                ),

                // === HTML / CSS / JavaScript ===
                "html" => HttpResponse::Ok().content_type("text/html; charset=utf-8").body(
                    data_value.as_str().unwrap_or("").to_string(), // Default to empty string
                ),
                // === Special HTML with custom headers ===
                "special_html" => {
                    if let Value::Object(obj) = &data_value {
                        let html_content = obj.get("html")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""); // Default to empty string

                        // Create an empty map for default headers
                        let empty_map = serde_json::Map::new();
                        let headers = obj.get("headers")
                            .and_then(|v| v.as_object())
                            .unwrap_or(&empty_map); // Use empty map if headers are missing/invalid

                        let mut response_builder = HttpResponse::Ok();
                        response_builder.content_type("text/html; charset=utf-8");

                        // Add custom headers
                        for (key, value) in headers {
                            if let Some(header_value) = value.as_str() {
                                // Use try_from for header names for robustness
                                if let Ok(header_name) = actix_web::http::header::HeaderName::try_from(key.as_str()) {
                                     if let Ok(header_val) = actix_web::http::header::HeaderValue::from_str(header_value) {
                                        response_builder.append_header((header_name, header_val));
                                     } else {
                                         warn!("Invalid header value for {}: {}", key, header_value);
                                     }
                                } else {
                                     warn!("Invalid header name: {}", key);
                                }
                            } else {
                                 warn!("Non-string header value for key: {}", key);
                            }
                        }
                        response_builder.body(html_content.to_string())
                    } else {
                        // Fallback if the format isn't a JSON object
                        HttpResponse::BadRequest().content_type("text/plain")
                            .body("Invalid special_html data format: expected JSON object with 'html' and optional 'headers' keys.".to_string())
                    }
                },
                "css" => HttpResponse::Ok().content_type("text/css; charset=utf-8").body(
                    data_value.as_str().unwrap_or("").to_string(), // Default to empty string
                ),
                "js" | "javascript" => HttpResponse::Ok().content_type("application/javascript; charset=utf-8").body(
                    data_value.as_str().unwrap_or("").to_string(), // Default to empty string
                ),

                // === XML / YAML ===
                "xml" => HttpResponse::Ok().content_type("application/xml; charset=utf-8").body(
                    data_value.as_str().unwrap_or("").to_string(), // Default to empty string
                ),
                "yaml" => { // Handle potential serialization errors
                    match serde_yaml::to_string(&data_value) {
                        Ok(yaml_string) => HttpResponse::Ok().content_type("application/x-yaml; charset=utf-8").body(yaml_string),
                        Err(e) => HttpResponse::InternalServerError().body(format!("Failed to serialize data to YAML: {}", e)),
                    }
                },

                // === Images (PNG, JPEG, GIF, SVG, WebP) ===
                "png" => image_response("image/png", data_value),
                "jpg" | "jpeg" => image_response("image/jpeg", data_value),
                "gif" => image_response("image/gif", data_value),
                "svg" => image_response("image/svg+xml", data_value),
                "webp" => image_response("image/webp", data_value),

                // === Binary Data / Files ===
                "binary" | "bytes" | "file" => {
                     match data_value.as_str() {
                        Some(base64_data) => match STANDARD.decode(base64_data) {
                            Ok(decoded) => HttpResponse::Ok()
                                .content_type("application/octet-stream")
                                .body(decoded),
                            Err(_) => HttpResponse::BadRequest().body("Invalid base64 binary/file data"),
                        },
                        None => HttpResponse::BadRequest().body("Missing or invalid binary/file data (expected base64 string)"),
                    }
                },

                // === Streaming ===
                "stream" | "streaming" => streaming_response(data_value),

                // === Unsupported Type ===
                _ => HttpResponse::Ok().json(fall_back),
            }
        }
        None => HttpResponse::Ok().json(fall_back),
    }
}

/// 🖼️ **Helper for Image Responses**
fn image_response(content_type: &str, data: Value) -> HttpResponse {
    match data.as_str() {
        Some(base64_data) => match STANDARD.decode(base64_data.trim()) { // Trim whitespace just in case
            Ok(decoded) => HttpResponse::Ok()
                .content_type(content_type)
                .body(decoded),
            Err(e) => {
                warn!("Invalid base64 image data received: {}", e);
                HttpResponse::BadRequest().body("Invalid base64 image data")
            },
        },
        None => HttpResponse::BadRequest().body("Missing image data (expected base64 string)"),
    }
}

/// 📂 **Helper for File Responses**
fn file_response(data: Value) -> Vec<u8> {
    match data.as_str() {
        Some(base64_data) => STANDARD.decode(base64_data).unwrap_or_else(|_| vec![]),
        None => vec![],
    }
}

/// 🔄 **Helper for Binary Responses**
fn binary_response(data: Value) -> Vec<u8> {
    match data.as_array() {
        Some(arr) => arr.iter().filter_map(|v| v.as_u64().map(|b| b as u8)).collect(),
        None => vec![],
    }
}

/// 🔀 **Helper for Streaming Responses**
/// TODO: Implement streaming with NuitkaClient
fn streaming_response(_data: Value) -> HttpResponse {
    // Streaming is not yet implemented with NuitkaClient
    HttpResponse::NotImplemented()
        .body("Streaming responses are not yet implemented with NuitkaClient")
}

// --- WebSocket Bridge Endpoints ---

#[derive(Debug, Deserialize)]
struct WsSendRequest {
    conn_id: String,
    payload: String,
}

#[derive(Debug, Deserialize)]
struct WsBroadcastRequest {
    channel_id: String,
    payload: String,
    #[serde(default = "default_source_conn_id")]
    source_conn_id: String,
}

fn default_source_conn_id() -> String {
    "python_broadcast".to_string()
}

/// Internal endpoint for Python to send WebSocket messages to a single connection
async fn ws_send_internal(
    req_body: web::Json<WsSendRequest>,
) -> HttpResponse {
    let conn_id = &req_body.conn_id;
    let payload = &req_body.payload;

    info!("ws_send_internal: conn_id={}, payload_len={}", conn_id, payload.len());

    if let Some(conn) = ACTIVE_CONNECTIONS.get(conn_id) {
        conn.value().do_send(WsMessage {
            source_conn_id: "python_direct".to_string(),
            content: payload.clone(),
            target_conn_id: Some(conn_id.clone()),
            target_channel_id: None,
        });
        info!("Message sent to connection: {}", conn_id);
        HttpResponse::Ok().json(serde_json::json!({
            "status": "success",
            "message": format!("Message sent to connection {}", conn_id)
        }))
    } else {
        warn!("Connection ID '{}' not found for sending.", conn_id);
        HttpResponse::NotFound().json(serde_json::json!({
            "status": "error",
            "message": format!("Connection ID '{}' not found", conn_id)
        }))
    }
}

/// Internal endpoint for Python to broadcast WebSocket messages to a channel
async fn ws_broadcast_internal(
    req_body: web::Json<WsBroadcastRequest>,
) -> HttpResponse {
    let channel_id = &req_body.channel_id;
    let payload = &req_body.payload;
    let source_conn_id = &req_body.source_conn_id;

    info!("ws_broadcast_internal: channel_id={}, source_conn_id={}, payload_len={}",
          channel_id, source_conn_id, payload.len());

    let msg = WsMessage {
        source_conn_id: source_conn_id.clone(),
        content: payload.clone(),
        target_conn_id: None,
        target_channel_id: Some(channel_id.clone()),
    };

    if let Err(e) = GLOBAL_WS_BROADCASTER.send(msg) {
        error!("Failed to send broadcast message to channel {}: {}", channel_id, e);
        HttpResponse::InternalServerError().json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to send broadcast message: {}", e)
        }))
    } else {
        info!("Broadcast message sent to channel: {}", channel_id);
        HttpResponse::Ok().json(serde_json::json!({
            "status": "success",
            "message": format!("Broadcast message sent to channel {}", channel_id)
        }))
    }
}


async fn api_handler(
    req: HttpRequest,
    path: web::Path<(String, String)>,
    query: web::Query<HashMap<String, String>>,
    mut payload: web::Payload,
    session: Session,
    open_modules: web::Data<Arc<Vec<String>>>,
    client: web::Data<Arc<NuitkaClient>>, // Add NuitkaClient from app_data
    init_modules: web::Data<Arc<Vec<String>>>, // Add init_modules for lazy initialization
    watch_modules: web::Data<Arc<Vec<String>>>, // Add watch_modules for lazy initialization
) -> HttpResponse {
    let (module_name, function_name) = path.into_inner();
    let request_method = req.method().clone();

    // Lazy initialization: Ensure Python backend is initialized in this worker
    // This is called on every request, but initialize() is idempotent (only runs once per worker)
    // Combine init_modules and watch_modules into a single list to avoid multiple initialize() calls
    let mut all_modules: Vec<String> = init_modules.as_ref().as_ref().clone();
    all_modules.extend(watch_modules.as_ref().as_ref().iter().cloned());
    if let Err(e) = client.initialize(all_modules, Some("init_mod")).await {
        error!("Failed to initialize Python backend: {:?}", e);
    }

    // Session validation (unchanged)
    let session_id = match session.get::<String>("ID") {
        Ok(Some(id)) => id,
        _ => {
            let connection_info = req.connection_info().clone();
            let ip = connection_info.realip_remote_addr().unwrap_or("unknown").split(':').next().unwrap_or("unknown").to_string();
            let port = connection_info.peer_addr().unwrap_or("unknown").split(':').nth(1).unwrap_or("unknown").to_string();
            let session_manager = req.app_data::<web::Data<SessionManager>>().expect("SessionManager not found");
            let id = session_manager.create_new_session(ip, port, None, None, None).await;
            let _ = session.insert("ID", &id);
            let _ = session.insert("anonymous", true);
            id
        }
    };

    let is_protected = !open_modules.contains(&module_name) && !function_name.starts_with("open");
    if is_protected {
        let valid = match session.get::<bool>("valid") {
            Ok(Some(true)) => true,
            _ => false,
        };
        if !valid {
            return HttpResponse::Unauthorized().json(ApiResult {
                error: Some("Unauthorized: Session invalid or missing permissions.".to_string()),
                origin: None, result: None, info: None,
            });
        }
    }

    let live_data = session.get::<HashMap<String, String>>("live_data").unwrap_or_else(|_| None);
    info!("API FOR: {:?} SessionID: {}", live_data, session_id);
    let spec = live_data.as_ref().and_then(|data| data.get("spec")).cloned().unwrap_or_else(|| "app".to_string());
    let args: Vec<String> = Vec::new();

    // Initialize kwargs with query parameters
    let mut kwargs: HashMap<String, serde_json::Value> = query.into_inner()
        .into_iter()
        .map(|(k, v)| (k, serde_json::json!(v)))
        .collect();

    // Process payload based on HTTP method
    if request_method == Method::POST || request_method == Method::PUT || request_method == Method::PATCH {
        let content_type = req.headers().get(actix_web::http::header::CONTENT_TYPE)
            .and_then(|val| val.to_str().ok())
            .unwrap_or("");

        if content_type.starts_with("multipart/form-data") {
            debug!("Processing multipart/form-data request");
            let mut multipart_payload = Multipart::new(req.headers(), payload);
            let mut form_data_map: HashMap<String, serde_json::Value> = HashMap::new();

            while let Ok(Some(mut field)) = multipart_payload.try_next().await {
                let content_disposition_opt = field.content_disposition();
                if let Some(content_disposition) = content_disposition_opt {
                    let field_name = content_disposition.get_name().unwrap_or("").to_string();
                    if field_name.is_empty() {
                        warn!("Multipart field received without a name in Content-Disposition, skipping.");
                        continue;
                    }

                    if content_disposition.get_filename().is_some() {
                        // File Upload
                        let filename = content_disposition.get_filename().map(String::from);
                        let content_type_mime = field.content_type().map(|mime| mime.to_string());
                        debug!("Processing uploaded file: name='{}', filename='{:?}', content_type='{:?}'", field_name, filename, content_type_mime);
                        let mut file_bytes = BytesMut::new();
                        while let Ok(Some(chunk)) = field.try_next().await {
                            file_bytes.extend_from_slice(&chunk);
                        }
                        let base64_content = BASE64_STANDARD.encode(&file_bytes);
                        let file_data = UploadedFile { filename, content_type: content_type_mime, content_base64: base64_content };
                        form_data_map.insert(field_name, serde_json::to_value(file_data).unwrap_or(Value::Null));
                    } else {
                        // Regular Field
                        debug!("Processing form field: name='{}'", field_name);
                        let mut field_bytes = BytesMut::new();
                        while let Ok(Some(chunk)) = field.try_next().await {
                            field_bytes.extend_from_slice(&chunk);
                        }
                        match String::from_utf8(field_bytes.to_vec()) {
                            Ok(value_str) => { form_data_map.insert(field_name, serde_json::json!(value_str)); },
                            Err(e) => {
                                warn!("Failed to decode form field '{}' as UTF-8: {}. Storing as Null.", field_name, e);
                                form_data_map.insert(field_name, Value::Null);
                            }
                        }
                    }
                } else {
                    warn!("Multipart field received without Content-Disposition header, skipping.");
                    while let Ok(Some(_)) = field.try_next().await {} // Drain field
                    continue;
                }
            }
            if !form_data_map.is_empty() {
                // Use "form_data" key for consistency with multipart
                kwargs.insert("form_data".to_string(), serde_json::json!(form_data_map));
            }

        } else if content_type.starts_with("application/json") {
            debug!("Processing application/json request");
            // Collect payload bytes
            let mut body_bytes = BytesMut::new();
            while let Some(chunk_result) = payload.next().await {
                match chunk_result {
                    Ok(chunk) => body_bytes.extend_from_slice(&chunk),
                    Err(e) => {
                        warn!("Error reading JSON payload stream: {}", e);
                        break; // Stop reading on error
                    }
                }
            }

            if !body_bytes.is_empty() {
                // Try to parse as JSON
                match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                    Ok(json_value) => {
                        kwargs.insert("data".to_string(), json_value);
                    },
                    Err(e) => {
                        warn!("Failed to parse JSON body: {}", e);
                        // Optionally try to log the raw content for debugging
                        if let Ok(body_str) = String::from_utf8(body_bytes.to_vec()) {
                            debug!("Raw JSON body (potentially invalid): {}", body_str);
                        }
                    }
                }
            } else {
                debug!("Received empty body for application/json");
            }
        } else if content_type.starts_with("application/x-www-form-urlencoded") {
            debug!("Processing application/x-www-form-urlencoded request");
            let mut body_bytes = BytesMut::new();
            // Read the entire payload stream into bytes
            while let Some(chunk_result) = payload.next().await {
                match chunk_result {
                    Ok(chunk) => body_bytes.extend_from_slice(&chunk),
                    Err(e) => {
                        warn!("Error reading urlencoded payload stream: {}", e);
                        // Decide if this is fatal or if we try parsing what we got
                        break; // Stop reading on error
                    }
                }
            }

            if !body_bytes.is_empty() {
                // Attempt to parse the bytes as urlencoded string -> map -> JSON Value
                match serde_urlencoded::from_bytes::<HashMap<String, String>>(&body_bytes) {
                    Ok(parsed_form) => {
                        // Convert the HashMap<String, String> to serde_json::Value
                        match serde_json::to_value(parsed_form) {
                            Ok(json_value) => {
                                // Use "form_data" key for consistency
                                kwargs.insert("form_data".to_string(), json_value);
                            }
                            Err(e) => {
                                warn!("Failed to convert parsed urlencoded form data to JSON Value: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse application/x-www-form-urlencoded body: {}", e);
                        // Optionally try decoding as UTF-8 just for logging
                        if let Ok(body_str) = String::from_utf8(body_bytes.to_vec()) {
                            debug!("Raw urlencoded body (potentially invalid): {}", body_str);
                        }
                    }
                }
            } else {
                debug!("Received empty body for application/x-www-form-urlencoded");
            }

        } else if !content_type.is_empty() {
            debug!("Ignoring request body with Content-Type: {} for method {}", content_type, request_method);
            // Drain the payload if ignored, otherwise the connection might hang
            while let Some(_) = payload.next().await {}
        } else {
            debug!("Request method {} had no Content-Type header.", request_method);
            // No content type, no body processing needed. Payload is automatically drained on drop.
        }
    } else {
        debug!("Skipping payload processing for request method: {}", request_method);
    }


    // --- Prepare request metadata (unchanged) ---
    let request_metadata = serde_json::json!({
        // form_data or data are now directly in kwargs if processed
        "session": live_data.unwrap_or_default(),
        "session_id": session_id,
        "request": {
            "path": req.path(),
            "headers": req.headers().iter().map(|(k, v)| (k.as_str(), v.to_str().unwrap_or(""))).collect::<HashMap<_, _>>(),
            "method": request_method.as_str(),
            "query_params": kwargs.iter()
                         .filter(|(k, _)| *k != "data" && *k != "form_data" && *k != "request")
                         .map(|(k, v)| (k.clone(), v.clone()))
                         .collect::<HashMap<_,_>>(),
            "content_type": if request_method == Method::POST || request_method == Method::PUT || request_method == Method::PATCH {
                 req.headers().get(actix_web::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
            } else {
                "".to_string()
            }
        }
    });


    kwargs.insert("request".to_string(), request_metadata);
    // Add spec and args to kwargs for compatibility
    kwargs.insert("spec".to_string(), serde_json::json!(spec));
    kwargs.insert("args".to_string(), serde_json::json!(args));
    info!("Final kwargs keys before sending to Python: {:?}", kwargs.keys());

    // Process API request with Nuitka - client is now from app_data parameter
    // No need to call get_nuitka_client() anymore!

    match client.call_module(module_name.clone(), function_name.clone(), serde_json::json!(kwargs)).await {
        Ok(response_value) => {
            match serde_json::from_value::<ApiResult>(response_value.clone()) {
                Ok(parsed_api_result) => parse_response(parsed_api_result, response_value),
                Err(e) => {
                    error!("Failed to parse Python response into ApiResult. Error: {}. Raw response: {}", e, response_value);
                    HttpResponse::InternalServerError().json(ApiResult {
                        error: Some(format!("Internal Server Error: Unexpected response format from backend.")),
                        origin: Some(serde_json::json!({
                            "parsing_error": e.to_string(),
                            "raw_response_preview": format!("{:.200}", response_value)
                        })),
                        result: None, info: None,
                    })
                }
            }
        },
        Err(e) => {
            error!("Toolbox execution error for {}/{}: {:?}", module_name, function_name, e);
            HttpResponse::InternalServerError().json(ApiResult {
                error: Some(format!("Backend Execution Error: {}", e)),
                origin: None, result: None, info: None,
            })
        }
    }
}

async fn sse_handler(
    req: HttpRequest,
    path: web::Path<(String, String)>,
    query: web::Query<HashMap<String, String>>,
    session: Session,
) -> HttpResponse {
    let (module_name, function_name) = path.into_inner();

    // Session-ID prüfen
    let session_id = match session.get::<String>("ID") {
        Ok(Some(id)) => id,
        _ => {
            // Anonyme Session für öffentliche Streams erstellen
            let connection_info = req.connection_info().clone();
            let ip = connection_info.realip_remote_addr()
                .unwrap_or("unknown").split(':').next().unwrap_or("unknown").to_string();
            let port = connection_info.peer_addr()
                .unwrap_or("unknown").split(':').nth(1).unwrap_or("unknown").to_string();

            let session_manager = req.app_data::<web::Data<SessionManager>>()
                .expect("SessionManager not found");

            let id = block_on(session_manager.create_new_session(ip, port, None, None, None));
            session.insert("ID", &id).unwrap_or(());
            id
        }
    };

    // Berechtigungsprüfung für geschützte Streams
    let is_protected = !function_name.starts_with("open");
    let valid = match session.get::<bool>("valid") {
        Ok(Some(true)) => true,
        _ => false,
    };

    if is_protected && !valid {
        return HttpResponse::Unauthorized().finish();
    }

    // Session-Daten vorbereiten
    let live_data = session.get::<HashMap<String, String>>("live_data").unwrap_or_else(|_| None);
    let spec = live_data.as_ref().and_then(|data| data.get("spec")).cloned().unwrap_or_else(|| "app".to_string());

    // Parameter vorbereiten
    let mut kwargs: HashMap<String, serde_json::Value> = query.into_inner()
        .into_iter()
        .map(|(k, v)| (k, serde_json::json!(v)))
        .collect();

    // Session-Metadaten hinzufügen
    kwargs.insert("request".to_string(), serde_json::json!({
        "session": live_data.unwrap_or_default(),
        "session_id": session_id,
    }));

    // TODO: Implement SSE streaming with NuitkaClient
    // SSE streaming is not yet implemented with NuitkaClient
    HttpResponse::NotImplemented()
        .body(format!("SSE streaming for {}/{} is not yet implemented with NuitkaClient", module_name, function_name))
}

const PERSISTENT_FD_FILE: &str = "server_socket.fd"; // File to store the FD on POSIX

#[actix_web::main]
async fn main() -> std::io::Result<()> {

    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    // Load configuration
    let config: ServerConfig = Config::builder()
        .add_source(File::new("config.toml", FileFormat::Toml))
        .build()
        .expect("Failed to build config")
        .try_deserialize()
        .expect("Invalid configuration format");

    info!("Configuration loaded: {:?}", config);


    // Create NuitkaClient but DON'T initialize it yet!
    // Initialization will happen in each worker via on_worker_start hook
    let client = match NuitkaClient::new(
        config.toolbox.max_instances as usize,
        config.toolbox.timeout_seconds,
        config.toolbox.client_prifix.clone(),
    ) {
        Ok(client_instance) => Arc::new(client_instance),
        Err(e) => {
            error!("FATAL: NuitkaClient creation failed: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create Python backend: {:?}", e),
            ));
        }
    };

    // NOTE: Module initialization moved to on_worker_start hook
    // This ensures each worker process has its own Python interpreter instance

    // Create session manager
    let session_manager = web::Data::new(SessionManager::new(
        config.session.clone(),
        Arc::clone(&client),
    ));

    // Generate session key
    let key = Key::from(config.session.secret_key.as_bytes());

    // Start server
    info!("Starting server on {}:{}", config.server.ip, config.server.port);
    let dist_path = config.server.dist_path.clone(); // Clone the dist_path here
    let open_modules = Arc::new(config.server.open_modules.clone());

    // Clone module lists for lazy initialization in workers
    let init_modules = Arc::new(config.server.init_modules.clone());
    let watch_modules = Arc::new(config.server.watch_modules.clone());

    // NOTE: Python backend initialization moved to lazy initialization in api_handler
    // This ensures each worker process initializes its own Python interpreter instance
    info!("Python backend will be initialized lazily in each worker on first API request");

    let server_handle: Server = {
        let mut http_server  = HttpServer::new(move || {
        // This closure is called once per worker process
        // Python backend will be initialized lazily on first API request

        let dist_path = dist_path.clone(); // Move the cloned dist_path into the closure
        let open_modules = Arc::clone(&open_modules);
        let client_clone = Arc::clone(&client); // Clone the NuitkaClient for this worker
        let init_modules_clone = Arc::clone(&init_modules);
        let watch_modules_clone = Arc::clone(&watch_modules);
        App::new()
            .app_data(web::Data::new(client_clone)) // Add NuitkaClient to app_data
            .app_data(web::Data::new(init_modules_clone)) // Add init_modules for lazy initialization
            .app_data(web::Data::new(watch_modules_clone)) // Add watch_modules for lazy initialization
            .wrap(middleware::Logger::default())
            .wrap(middleware::Compress::default())
            .wrap(SessionMiddleware::new(
                CookieSessionStore::default(),
                key.clone(),
            ))
            .wrap_fn(|req, srv| {  // this middleware to ensure all requests have a session
                let fut = srv.call(req);
                async {
                    let res = fut.await?;

                    // Get the request's session
                    let session = res.request().get_session();
                    // Make sure we have a session ID
                    if session.get::<String>("ID").unwrap_or(None).is_none() {
                        let session_manager = res.request()
                            .app_data::<web::Data<SessionManager>>()
                            .expect("SessionManager not found");

                        // Extract IP and port
                        let connection_info = res.request().connection_info().clone();
                        let ip = connection_info.realip_remote_addr()
                            .unwrap_or("unknown")
                            .split(':')
                            .next()
                            .unwrap_or("unknown")
                            .to_string();

                        let port = connection_info.peer_addr()
                            .unwrap_or("unknown")
                            .split(':')
                            .nth(1)
                            .unwrap_or("unknown")
                            .to_string();

                        // Create anonymous session
                        let session_id = block_on(session_manager.create_new_session(
                            ip, port, None, None, None
                        ));

                        // Store session ID
                        let _ = session.insert("ID", session_id);
                        let _ = session.insert("anonymous", true);

                    }

                    Ok(res)
                }
            })
            // 6. Middleware to add session information to responses
            .wrap_fn(|req, srv| {
                let fut = srv.call(req);
                async {
                    let mut res = fut.await?;

                    // Add session ID to response headers for debugging/tracking
                    let session = res.request().get_session();
                    if let Ok(Some(session_id)) = session.get::<String>("ID") {
                        res.headers_mut().insert(
                            actix_web::http::header::HeaderName::from_static("x-session-id"),
                            actix_web::http::header::HeaderValue::from_str(&session_id).unwrap_or_else(|_|
                                actix_web::http::header::HeaderValue::from_static("unknown"))
                        );
                    }


                    Ok(res)
                }
            })
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
            .service(
                web::scope("/internal/ws")
                    .route("/send", web::post().to(ws_send_internal))
                    .route("/broadcast", web::post().to(ws_broadcast_internal))
            )
            .service(
                web::scope("/sse")
                    .route("/{module_name}/{function_name}", web::get().to(sse_handler))
            )
            .service(
                web::scope("/ws")
                    .app_data(web::Data::new(open_modules.clone()))
                    .route("/{module_name}/{function_name}", web::get().to(websocket_handler))
            )
            .service(web::resource("/validateSession")
                .route(web::post().to(validate_session_handler))
                )
            .service(web::resource("/IsValidSession")
                .route(web::get().to(is_valid_session_handler))
                )
            .service(web::resource("/web/logoutS")
                .route(web::post().to(logout_handler))
                )
            .service(web::resource("/api_user_data")
                .route(web::get().to(get_user_data_handler))
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
        .workers(1);  // Use 1 worker to avoid multi-process Python initialization issues
                      // Each worker needs its own Python interpreter, which is complex with Nuitka
                      // TODO: Implement proper per-worker initialization or use thread-based workers

        let bind_addr = format!("{}:{}", config.server.ip, config.server.port);
        info!("[Manual Bind] No inherited listener was successfully acquired. Binding to new TCP listener on {}.", bind_addr);
        // .bind() consumes http_server and returns a Result<HttpServer, std::io::Error>
        // The ? operator will get the HttpServer if Ok, or return Err early.
        // Then .run() is called on the HttpServer, which returns the Server handle.
        http_server.bind(bind_addr)?.run()

    }; // The semicolon here is important; server_handle is now the dev::Server

    info!("Server setup complete. Starting server (dev::Server handle obtained).");
    // server_handle is the `actix_web::dev::Server`
    // .await on it will run the server and return std::io::Result<()> upon completion/error
    server_handle.await
}
