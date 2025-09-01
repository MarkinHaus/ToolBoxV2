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
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyList};
use tokio::{task, time};
use pyo3::PyResult;
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

// For io::Error
// Helper function to convert PyObject to Bytes
fn py_object_to_bytes(py: Python, obj: PyObject) -> PyResult<Bytes> {
    // Prioritize raw bytes if available (e.g., from PyBytes)
    if let Ok(py_bytes) = obj.downcast::<pyo3::types::PyBytes>(py) {
        return Ok(Bytes::from(py_bytes.as_bytes().to_vec()));
    }
    // Then Vec<u8>
    if let Ok(bytes_vec) = obj.extract::<Vec<u8>>(py) {
        return Ok(Bytes::from(bytes_vec));
    }
    // Then string
    if let Ok(string) = obj.extract::<String>(py) {
        return Ok(Bytes::from(string));
    }

    // Fallback to JSON representation ONLY if other types fail
    // The 'py' parameter already represents the held GIL token.
    match py_to_value_global(py, obj.as_ref(py)) { // Use the 'py' passed into the function
        Ok(json_value) => {
            match serde_json::to_vec(&json_value) {
                Ok(vec) => Ok(Bytes::from(vec)),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialization error: {}", e))),
            }
        },
        Err(e) => {
             // Log the original Python error for better debugging
             error!("Error converting Python object to JSON Value: {}", e);
             Err(e) // Propagate the original PyErr
        }
    }
}
// Hypothetical global version of py_to_value for the helper
// You might need to adapt this based on where py_to_value is actually defined
fn py_to_value_global(py: Python, value: &PyAny) -> PyResult<serde_json::Value> {
    // ... (Implementation of py_to_value, assuming it's accessible here)
    // For demonstration, let's copy the logic (ideally, share it)
    if value.is_none() { return Ok(serde_json::Value::Null); }
    if let Ok(b) = value.extract::<bool>() { return Ok(serde_json::Value::Bool(b)); }
    if let Ok(i) = value.extract::<i64>() { return Ok(serde_json::Value::Number(i.into())); }
    if let Ok(f) = value.extract::<f64>() { return Ok(serde_json::json!(f)); } // Use json! macro for potential NaN/Infinity
    if let Ok(s) = value.extract::<String>() { return Ok(serde_json::Value::String(s)); }
    if let Ok(seq) = value.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in seq.iter() { arr.push(py_to_value_global(py, item)?); }
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(tup) = value.downcast::<PyTuple>() {
        let mut arr = Vec::new();
        for item in tup.iter() { arr.push(py_to_value_global(py, item)?); }
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, val) in dict.iter() {
            let key_str = key.extract::<String>()?;
            map.insert(key_str, py_to_value_global(py, val)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    if let Ok(s) = value.str()?.extract::<String>() { return Ok(serde_json::Value::String(s)); }
    Ok(serde_json::Value::Null) // Fallback
}


// Define a struct to hold the state needed in the Drop impl
struct InstanceGuard {
    client: ToolboxClient,
    instance_id: String,
    released: bool, // Flag to prevent double-release
}

impl InstanceGuard {
    fn new(client: ToolboxClient, instance_id: String) -> Self {
        InstanceGuard { client, instance_id, released: false }
    }

    // Explicitly release the instance (e.g., after successful stream completion)
    fn release(&mut self) {
        if !self.released {
            self.client.mark_instance_done(&self.instance_id);
            self.released = true;
            debug!("Instance {} released explicitly.", self.instance_id);
        }
    }
}

// Implement Drop to ensure the instance is marked done even on errors/panics
impl Drop for InstanceGuard {
    fn drop(&mut self) {
        if !self.released {
            self.client.mark_instance_done(&self.instance_id);
            debug!("Instance {} released via Drop.", self.instance_id);
        }
    }
}



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
    client:  Arc<ToolboxClient>,
    session: SessionData, // Annahme: SessionData wird aus der Session extrahiert
    channel_id: Option<String>, // Der "Raum" oder die Gruppe, der diese Verbindung angehört
    hb: Instant,
}

impl WebSocketActor {
    fn new(client: Arc<ToolboxClient>, session: Session, module: &str, function: &str) -> Self {
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
        ctx.run_interval(Duration::from_secs(5), |act, ctx| {
            if Instant::now().duration_since(act.hb) > Duration::from_secs(10) {
                warn!("WebSocket Client heartbeat failed, disconnecting!");
                ctx.stop();
                return;
            }
            ctx.ping(b"");
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

            if let Err(e) = client.run_function(
                &channel_id, // Der Kanalname dient zur Identifizierung des Handlers
                "on_connect",
                "ws_internal",
                vec![],
                kwargs,
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
            if let Err(e) = client.run_function(
                &channel_id,
                "on_disconnect",
                "ws_internal",
                vec![],
                kwargs,
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

                    if let Err(e) = client.run_function(
                        &channel_id,
                        "on_message",
                        "ws_internal",
                        vec![],
                        kwargs,
                    ).await {
                        error!("Python on_message handler failed: {:?}", e);
                    }
                });
            }
            Ok(ws::Message::Binary(_bin)) => warn!("Binary WebSocket messages are not supported."),
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => ctx.stop(),
        }
    }
}


// --- NEU: WebSocket-Endpoint-Handler ---
async fn websocket_handler(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<(String, String)>,
    session: Session,
    // Annahme: Open Modules und ToolboxClient sind als App-Daten verfügbar
    open_modules: web::Data<Arc<Vec<String>>>,
) -> Result<HttpResponse, Error> {
    let (module_name, function_name) = path.into_inner();

    // Berechtigungsprüfung
    let is_protected = !open_modules.contains(&module_name) && !function_name.starts_with("open");
    if is_protected {
        let valid = session.get::<bool>("valid").unwrap_or(None).unwrap_or(false);
        if !valid {
            return Ok(HttpResponse::Unauthorized().finish());
        }
    }

    let client = get_toolbox_client().map_err(|e| {
        error!("Could not get ToolboxClient for WebSocket: {:?}", e);
        actix_web::error::ErrorInternalServerError("Backend service unavailable")
    })?;

    ws::start(
        WebSocketActor::new(client, session, &module_name, &function_name),
        &req,
        stream,
    )
}

// --- NEU: Die Rust-zu-Python Bridge-Klasse ---

/// Diese Klasse wird an Python übergeben. Ihre Methoden können von Python aus aufgerufen werden.
/// Diese Klasse wird an Python übergeben. Ihre Methoden können von Python aus aufgerufen werden.
#[pyclass]
struct RustWsBridge;

#[pymethods]
impl RustWsBridge {
    /// KORREKTUR: Füge einen `#[new]` Konstruktor hinzu.
    /// Dieser wird aufgerufen, wenn Python `RustWsBridge()` ausführt.
    #[new]
    fn new() -> Self {
        RustWsBridge
    }

    /// Sendet eine Nachricht an eine einzelne WebSocket-Verbindung.
    #[pyo3(name = "send_message")]
    fn send_message_py<'p>(&self, py: Python<'p>, conn_id: String, payload: String) -> PyResult<&'p PyAny> {
        pyo3_asyncio::tokio::future_into_py(py, async move {
            if let Some(conn) = ACTIVE_CONNECTIONS.get(&conn_id) {
                conn.value().do_send(WsMessage {
                    source_conn_id: "python_direct".to_string(),
                    content: payload,
                    target_conn_id: Some(conn_id),
                    target_channel_id: None,
                });
            } else {
                warn!("RustWsBridge: Connection ID '{}' not found for sending.", conn_id);
            }
            Ok(())
        })
    }

    /// Sendet eine Nachricht an alle Clients in einem Kanal.
    #[pyo3(name = "broadcast_message")]
    fn broadcast_message_py<'p>(&self, py: Python<'p>, channel_id: String, payload: String, source_conn_id: String) -> PyResult<&'p PyAny> {
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let msg = WsMessage {
                source_conn_id,
                content: payload,
                target_conn_id: None,
                target_channel_id: Some(channel_id),
            };
            if let Err(e) = GLOBAL_WS_BROADCASTER.send(msg) {
                error!("RustWsBridge: Failed to send broadcast message: {}", e);
            }
            Ok(())
        })
    }
}

/// Ein internes Python-Modul, das in Rust erstellt wird.
#[pymodule]
fn rust_bridge_internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustWsBridge>()?;
    Ok(())
}

lazy_static! {
    // Der Option-Typ enthält nun einen Arc<ToolboxClient>, nicht den Client selbst.
    static ref TOOLBOX_CLIENT: Mutex<Option<Arc<ToolboxClient>>> = Mutex::new(None);
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


/// Initializes the Python environment by setting PYTHONHOME and PYTHONPATH
pub fn initialize_python_environment() -> Result<(), ToolboxError> {
    // First try to detect Python from conda environment
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        // Set PYTHONHOME to conda environment
        env::set_var("PYTHONHOME", &conda_prefix);

        // Determine Python executable path
        let python_executable = if cfg!(windows) {
            format!("{}\\python.exe", conda_prefix.replace("\\", "\\\\"))
        } else {
            format!("{}/bin/python", conda_prefix)
        };
        env::set_var("PYTHON_EXECUTABLE", &python_executable);

        // Create proper PYTHONPATH for Windows
        let lib_path = format!(
            "{0}\\Lib;{0}\\Lib\\site-packages;{0}\\DLLs",
            conda_prefix.replace("\\", "\\\\")
        );
        env::set_var("PYTHONPATH", &lib_path);

        info!("Using conda environment at: {}", conda_prefix);
        debug!(" PYTHONHOME={}", conda_prefix);
        debug!(" PYTHON_EXECUTABLE={}", python_executable);
        debug!(" PYTHONPATH={}", lib_path);

        return Ok(());
    }

    // Fallback to detecting Python normally
    let output = std::process::Command::new("python")
        .args(&["-c", "import sys; print(sys.executable)"])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            env::set_var("PYTHON_EXECUTABLE", stdout.trim());
            info!("Detected system Python at: {}", stdout.trim());
            return Ok(());
        }
        _ => {
            warn!("Could not detect Python automatically");
        }
    }

    // As a last resort, try to use pyo3's auto-detection
    warn!("Falling back to PyO3 auto-detection for Python environment");

    Ok(())
}


/// Initialize the toolbox client and immediately create an instance

pub async fn initialize_and_get_toolbox_client(
    max_instances: usize,
    timeout_seconds: u64,
    client_prifix: String,
) -> Result<Arc<ToolboxClient>, ToolboxError> {
    initialize_python_environment()?;
    let client = ToolboxClient::new(max_instances, timeout_seconds, client_prifix);

    if let Err(e) = client.create_python_instance().await {
        error!("Critical error during initial Python instance creation: {:?}", e);
        return Err(e);
    }

    let client_arc = Arc::new(client);

    let mut client_mutex = TOOLBOX_CLIENT.lock().unwrap();
    // KORREKTUR: Speichere den Arc direkt. .clone() erhöht nur den Zähler.
    *client_mutex = Some(client_arc.clone());

    info!("ToolboxClient initialized and first Python instance created successfully.");
    Ok(client_arc)
}

/// Get the global toolbox client
pub fn get_toolbox_client() -> Result<Arc<ToolboxClient>, ToolboxError> {
    let client_guard = TOOLBOX_CLIENT.lock().unwrap();
    // .clone() auf einer Option<Arc<T>> klont den Arc, was genau das ist, was wir wollen.
    client_guard.clone().ok_or_else(|| {
        ToolboxError::Unknown("ToolboxClient not initialized".to_string())
    })
}


fn check_python_paths(py: Python) {
    let sys = py.import("sys").unwrap();
    let sys_path: &PyList = sys.getattr("path").unwrap().downcast().unwrap();

    println!("Python sys.path:");
    for path in sys_path.iter() {
        println!("  {:?}", path);
    }

}

fn check_toolboxv2(py: Python) -> PyResult<()> {
    let toolbox = py.import("toolboxv2")?;
    let version: &PyAny = toolbox.getattr("__version__")?;
    println!("toolboxv2 version: {}", version.extract::<String>()?);

    Ok(())
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
    anonymous: bool,
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
            anonymous: jwt_claim.is_none() && username.is_none(),
        };

        self.save_session(&session_id, session_data);

        // Check if IP is in gray or black list
        if self.gray_list.contains(&ip) {
            return "#0X".to_string();
        }

        if self.black_list.contains(&ip) {
            return "#0X".to_string();
        }

        if let (Some(jwt), Some(user)) = (jwt_claim, username) {
            self.verify_session_id(&session_id, &user, &jwt).await
        } else {
            session_id
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

            let encoded_username = format!("{}", username);
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

        if session.anonymous {
            info!("Anonymous session: {}", session_id);
            return false;  // Not valid for protected resources
        }

        if session.new || !session.validate {
            info!("Session is new or not validated: new={}, validate={}", session.new, session.validate);
            if let (Some(user_name), Some(jwt)) = (&session.user_name, &session.jwt_claim) {
                // Extract username from encoded format
                let username = user_name;
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
                let username = user_name;
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
fn streaming_response(data: Value) -> HttpResponse {
    // Extract streaming parameters
    let module = data.get("module").and_then(Value::as_str).unwrap_or("default");
    let function = data.get("function").and_then(Value::as_str).unwrap_or("stream");
    let spec = data.get("spec").and_then(Value::as_str).unwrap_or("default");
    let content_type = data.get("content_type").and_then(Value::as_str).unwrap_or("text/plain");

    // Extract args and kwargs
    let args = data.get("args")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let kwargs = data.get("kwargs")
        .and_then(Value::as_object)
        .map(|obj| obj.clone())
        .unwrap_or_default()
        .into_iter()
        .collect();

    // Set up streaming
    match get_toolbox_client() {
        Ok(client) => {
            match tokio::runtime::Runtime::new() {
                Ok(rt) => {
                    match rt.block_on(client.stream_generator(module, function, spec, args, kwargs)) {
                        Ok(stream) => {
                            // Return a streaming response with the appropriate content type
                            HttpResponse::Ok()
                                .content_type(content_type)
                                .streaming(stream)
                        },
                        Err(e) => {
                            HttpResponse::InternalServerError()
                                .body(format!("Failed to create stream: {}", e))
                        }
                    }
                },
                Err(e) => {
                    HttpResponse::InternalServerError()
                        .body(format!("Failed to create runtime: {}", e))
                }
            }
        },
        Err(e) => {
            HttpResponse::InternalServerError()
                .body(format!("Failed to get toolbox client: {}", e))
        }
    }
}


async fn api_handler(
    req: HttpRequest,
    path: web::Path<(String, String)>,
    query: web::Query<HashMap<String, String>>,
    mut payload: web::Payload,
    session: Session,
    open_modules: web::Data<Arc<Vec<String>>>,
) -> HttpResponse {
    let (module_name, function_name) = path.into_inner();
    let request_method = req.method().clone();

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
    info!("Final kwargs keys before sending to Python: {:?}", kwargs.keys());

    // Process API request with Toolbox
    let client_result = get_toolbox_client();
    if client_result.is_err() {
        error!("Failed to get toolbox client: {:?}", client_result.err());
        return HttpResponse::InternalServerError().json(ApiResult {
            error: Some("Internal Server Error: Cannot access backend service.".to_string()),
            origin: None, result: None, info: None,
        });
    }

    let client = Arc::new(client_result.unwrap());

    match client.run_function(&module_name, &function_name, &spec, args, kwargs).await {
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

    // Stream starten
    match get_toolbox_client() {
        Ok(client) => {
            match client.stream_sse_events(&module_name, &function_name, &spec, vec![], kwargs).await {
                Ok(stream) => {
                    // Force immediate streaming with critical headers
                    HttpResponse::Ok()
                        .content_type("text/event-stream")
                        .insert_header(("Cache-Control", "no-cache, no-transform"))
                        .insert_header(("Connection", "keep-alive"))
                        .insert_header(("X-Accel-Buffering", "no"))
                        .insert_header(("Content-Encoding", "identity"))
                        .streaming(stream)
                },
                Err(e) => {
                    HttpResponse::InternalServerError().body(format!("Stream error: {:?}", e))
                }
            }
        },
        Err(e) => {
            HttpResponse::InternalServerError().body(format!("Toolbox error: {:?}", e))
        }
    }
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


    let client = match initialize_and_get_toolbox_client(
        config.toolbox.max_instances as usize,
        config.toolbox.timeout_seconds,
        config.toolbox.client_prifix,
    ).await {
        Ok(client_instance) => client_instance,
        Err(e) => {
            error!("FATAL: ToolboxClient initialization failed: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to initialize Python backend: {:?}", e),
            ));
        }
    };

    // Initialisiere die Module NACHDEM der Client erfolgreich erstellt wurde.
    if let Err(e) = client.initialize(config.server.init_modules.clone(), Some("init_mod")).await {
        warn!("Errors occurred during initial module loading: {:?}", e);
    }
    if let Err(e) = client.initialize(config.server.watch_modules.clone(), Some("watch_mod")).await {
        warn!("Errors occurred during watched module loading: {:?}", e);
    }

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

    let server_handle: Server = {
        let mut http_server  = HttpServer::new(move || {
        let dist_path = dist_path.clone(); // Move the cloned dist_path into the closure
        let open_modules = Arc::clone(&open_modules);
        App::new()
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
    });

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
