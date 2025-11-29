use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Error, middleware};
use actix_web::cookie::Key;
use actix_session::{Session, SessionMiddleware, storage::CookieSessionStore};
use actix_web_actors::ws;
use actix_files as fs;
use actix::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use uuid::Uuid;
use dashmap::DashMap;
use lazy_static::lazy_static;
use tokio::sync::broadcast;

use chrono::{DateTime, Utc};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Session Management
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub valid: bool,
    pub user_name: Option<String>,
    pub jwt_claim: Option<String>,
    pub ip: String,
    pub port: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub request_count: u32,
    pub anonymous: bool,
}

impl Default for SessionData {
    fn default() -> Self {
        let now = Utc::now();
        SessionData {
            id: Uuid::new_v4().to_string(),
            valid: false,
            user_name: None,
            jwt_claim: None,
            ip: String::new(),
            port: String::new(),
            created_at: now,
            expires_at: now + chrono::Duration::seconds(3600),
            request_count: 0,
            anonymous: true,
        }
    }
}

impl SessionData {
    pub fn new(ip: String, port: String) -> Self {
        let now = Utc::now();
        SessionData {
            id: Uuid::new_v4().to_string(),
            valid: false,
            user_name: None,
            jwt_claim: None,
            ip,
            port,
            created_at: now,
            expires_at: now + chrono::Duration::seconds(3600),
            request_count: 0,
            anonymous: true,
        }
    }

    fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    fn refresh(&mut self) {
        self.expires_at = Utc::now() + chrono::Duration::seconds(3600);
        self.request_count += 1;
    }
}

lazy_static! {
    static ref SESSIONS: Arc<DashMap<String, SessionData>> = Arc::new(DashMap::new());
}

// ============================================================================
// Rate Limiting
// ============================================================================

#[derive(Debug, Clone)]
struct RateLimitEntry {
    count: usize,
    window_start: Instant,
}

lazy_static! {
    static ref RATE_LIMITS: Arc<DashMap<String, RateLimitEntry>> = Arc::new(DashMap::new());
}

fn check_rate_limit(key: &str, max_requests: usize, window_secs: u64) -> bool {
    let now = Instant::now();
    let window = Duration::from_secs(window_secs);

    let mut entry = RATE_LIMITS.entry(key.to_string()).or_insert(RateLimitEntry {
        count: 0,
        window_start: now,
    });

    // Reset window if expired
    if now.duration_since(entry.window_start) > window {
        entry.count = 0;
        entry.window_start = now;
    }

    entry.count += 1;
    entry.count <= max_requests
}

// ============================================================================
// WebSocket Support
// ============================================================================

#[derive(Message, Clone, Debug)]
#[rtype(result = "()")]
struct WsMessage {
    pub source_conn_id: String,
    pub content: String,
    pub target_conn_id: Option<String>,
    pub target_channel_id: Option<String>,
}

lazy_static! {
    static ref GLOBAL_WS_BROADCASTER: broadcast::Sender<WsMessage> = broadcast::channel(1024).0;
    static ref ACTIVE_CONNECTIONS: Arc<DashMap<String, Addr<WebSocketActor>>> = Arc::new(DashMap::new());
}

struct WebSocketActor {
    conn_id: String,
    channel_id: Option<String>,
    session: SessionData,
    hb: Instant,
}

impl WebSocketActor {
    fn new(channel_id: String, session: SessionData) -> Self {
        Self {
            conn_id: Uuid::new_v4().to_string(),
            channel_id: Some(channel_id),
            session,
            hb: Instant::now(),
        }
    }

    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(45), |act, ctx| {
            if Instant::now().duration_since(act.hb) > Duration::from_secs(120) {
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
        tracing::info!("WebSocket Actor started: conn_id={}", self.conn_id);
        self.heartbeat(ctx);
        ACTIVE_CONNECTIONS.insert(self.conn_id.clone(), ctx.address());

        let mut rx = GLOBAL_WS_BROADCASTER.subscribe();
        let addr = ctx.address();

        let broadcast_listener = async move {
            while let Ok(msg) = rx.recv().await {
                addr.do_send(msg);
            }
        };
        ctx.spawn(broadcast_listener.into_actor(self));
    }

    fn stopping(&mut self, _: &mut Self::Context) -> Running {
        tracing::info!("WebSocket Actor stopping: conn_id={}", self.conn_id);
        ACTIVE_CONNECTIONS.remove(&self.conn_id);
        Running::Stop
    }
}

impl Handler<WsMessage> for WebSocketActor {
    type Result = ();

    fn handle(&mut self, msg: WsMessage, ctx: &mut Self::Context) {
        if let Some(target_id) = &msg.target_conn_id {
            if *target_id == self.conn_id {
                ctx.text(msg.content);
            }
            return;
        }

        if let Some(target_channel) = &msg.target_channel_id {
            if self.channel_id.as_deref() == Some(target_channel) {
                if msg.source_conn_id != self.conn_id {
                    ctx.text(msg.content);
                }
            }
            return;
        }

        if msg.source_conn_id != self.conn_id {
            ctx.text(msg.content);
        }
    }
}

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
                let text_str = text.to_string();
                // Parse message as JSON
                let payload: JsonValue = serde_json::from_str(&text_str).unwrap_or(serde_json::json!({"text": text_str}));

                // Build request JSON for TB Lang callback
                let request_json = serde_json::json!({
                    "type": "websocket",
                    "event": "message",
                    "conn_id": self.conn_id,
                    "channel": self.channel_id,
                    "payload": payload,
                    "session": {
                        "id": self.session.id,
                        "valid": self.session.valid,
                        "user": self.session.user_name,
                    }
                });

                // Call TB Lang callback if registered
                if let Some(callback) = TB_CALLBACK.lock().unwrap().as_ref() {
                    let request_str = CString::new(request_json.to_string()).unwrap();
                    let response_ptr = callback(request_str.as_ptr());

                    if !response_ptr.is_null() {
                        let response_str = unsafe { CStr::from_ptr(response_ptr).to_str().unwrap_or("{}") };
                        if let Ok(response_json) = serde_json::from_str::<JsonValue>(response_str) {
                            if let Some(body) = response_json.get("body") {
                                ctx.text(body.to_string());
                            }
                        }
                    }
                } else {
                    // No callback, echo back
                    ctx.text(text);
                }
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            Err(e) => {
                tracing::error!("WebSocket error: {:?}", e);
            }
            _ => {}
        }
    }
}

// ============================================================================
// Static File Serving with Path Traversal Protection
// ============================================================================

/// Validate that a path is safe (no path traversal)
fn is_safe_path(path: &str) -> bool {
    // Check for path traversal patterns
    if path.contains("..") || path.contains("~") {
        return false;
    }

    // Check for absolute paths (should be relative)
    if path.starts_with('/') && path.len() > 1 {
        // Allow single / for root
        return !path[1..].starts_with('/');
    }

    true
}

/// Normalize path for static file serving
fn normalize_static_path(path: &str) -> String {
    let mut normalized = path.trim_start_matches('/').to_string();

    // Default to index.html for directory requests
    if normalized.is_empty() || normalized.ends_with('/') {
        normalized.push_str("index.html");
    }

    normalized
}

/// Safely join base directory with requested path
fn safe_join(base: &Path, requested: &str) -> Option<PathBuf> {
    // Normalize the requested path
    let normalized = normalize_static_path(requested);

    // Check for path traversal
    if !is_safe_path(&normalized) {
        return None;
    }

    // Join paths
    let full_path = base.join(&normalized);

    // Canonicalize both paths to resolve symlinks and .. components
    let base_canonical = match base.canonicalize() {
        Ok(p) => p,
        Err(_) => return None,
    };

    let full_canonical = match full_path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // File might not exist yet, check parent directory
            if let Some(parent) = full_path.parent() {
                if let Ok(parent_canonical) = parent.canonicalize() {
                    // Ensure parent is within base
                    if !parent_canonical.starts_with(&base_canonical) {
                        return None;
                    }
                    return Some(full_path);
                }
            }
            return None;
        }
    };

    // Ensure the canonical path is within the base directory
    if !full_canonical.starts_with(&base_canonical) {
        return None;
    }

    Some(full_canonical)
}

// ============================================================================
// HTTP Handlers
// ============================================================================

async fn api_handler(
    req: HttpRequest,
    path: web::Path<(String, String)>,
    session: Session,
    body: web::Bytes,
) -> Result<HttpResponse, Error> {
    let (module, function) = path.into_inner();

    // Get client IP and port
    let conn_info = req.connection_info();
    let ip = conn_info.realip_remote_addr().unwrap_or("unknown").to_string();
    let port = conn_info.peer_addr().unwrap_or("unknown").to_string();

    // Rate limiting
    if !check_rate_limit(&ip, 100, 60) {
        return Ok(HttpResponse::TooManyRequests().json(serde_json::json!({
            "error": "Rate limit exceeded",
            "retry_after": 30
        })));
    }

    // Get or create session
    let mut session_data = if let Ok(Some(session_id)) = session.get::<String>("id") {
        if let Some(mut data) = SESSIONS.get_mut(&session_id) {
            if data.is_expired() {
                // Session expired, create new one
                SessionData::default()
            } else {
                data.refresh();
                data.clone()
            }
        } else {
            SessionData::default()
        }
    } else {
        SessionData::default()
    };

    // Update session data
    session_data.ip = ip.clone();
    session_data.port = port.clone();

    // Save session
    let _ = session.insert("id", session_data.id.clone());
    let _ = session.insert("valid", session_data.valid);
    SESSIONS.insert(session_data.id.clone(), session_data.clone());

    // Parse body
    let body_json: JsonValue = if body.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_slice(&body).unwrap_or(serde_json::json!({}))
    };

    // Build request JSON for TB Lang callback
    let request_json = serde_json::json!({
        "type": "http",
        "method": req.method().as_str(),
        "path": req.path(),
        "params": {
            "module": module,
            "function": function
        },
        "headers": {
            "Content-Type": req.headers().get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
        },
        "body": body_json,
        "session": {
            "id": session_data.id,
            "valid": session_data.valid,
            "user": session_data.user_name,
        },
        "client": {
            "ip": ip,
            "port": port
        }
    });

    // Call TB Lang callback if registered
    let response_json = if let Some(callback) = TB_CALLBACK.lock().unwrap().as_ref() {
        let request_str = CString::new(request_json.to_string()).unwrap();
        let response_ptr = callback(request_str.as_ptr());

        if response_ptr.is_null() {
            serde_json::json!({
                "status": 500,
                "body": {"error": "Callback returned null"}
            })
        } else {
            let response_str = unsafe { CStr::from_ptr(response_ptr).to_str().unwrap_or("{}") };
            serde_json::from_str(response_str).unwrap_or(serde_json::json!({
                "status": 500,
                "body": {"error": "Invalid callback response"}
            }))
        }
    } else {
        // No callback registered, return echo response
        serde_json::json!({
            "status": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "success": true,
                "module": module,
                "function": function,
                "session_id": session_data.id,
                "valid": session_data.valid,
                "data": body_json
            }
        })
    };

    // Extract status and body from response
    let status = response_json.get("status").and_then(|v| v.as_u64()).unwrap_or(200) as u16;
    let response_body = response_json.get("body").cloned().unwrap_or(serde_json::json!({}));

    Ok(HttpResponse::build(actix_web::http::StatusCode::from_u16(status).unwrap())
        .json(response_body))
}

async fn websocket_handler(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<(String, String)>,
    session: Session,
) -> Result<HttpResponse, Error> {
    let (module, function) = path.into_inner();
    let channel_id = format!("{}/{}", module, function);

    // Get or create session
    let session_data = if let Ok(Some(session_id)) = session.get::<String>("id") {
        SESSIONS.get(&session_id).map(|s| s.clone()).unwrap_or_default()
    } else {
        SessionData::default()
    };

    ws::WsResponseBuilder::new(
        WebSocketActor::new(channel_id, session_data),
        &req,
        stream,
    )
    .frame_size(16 * 1024 * 1024)
    .start()
}

// ============================================================================
// Plugin FFI Interface
// ============================================================================

// Callback function type for TB Lang
type TbLangCallback = extern "C" fn(*const c_char) -> *mut c_char;

lazy_static! {
    static ref SERVER_RUNNING: Arc<DashMap<u16, AtomicBool>> = Arc::new(DashMap::new());
    static ref TB_CALLBACK: Arc<Mutex<Option<TbLangCallback>>> = Arc::new(Mutex::new(None));
}

#[repr(C)]
pub struct PluginResult {
    pub success: bool,
    pub data: *mut c_char,
    pub error: *mut c_char,
}

impl PluginResult {
    fn success(data: String) -> Self {
        PluginResult {
            success: true,
            data: CString::new(data).unwrap().into_raw(),
            error: std::ptr::null_mut(),
        }
    }

    fn error(error: String) -> Self {
        PluginResult {
            success: false,
            data: std::ptr::null_mut(),
            error: CString::new(error).unwrap().into_raw(),
        }
    }
}

#[no_mangle]
pub extern "C" fn start_server(port_ptr: *const c_char) -> PluginResult {
    start_server_with_host(port_ptr, std::ptr::null())
}

#[no_mangle]
pub extern "C" fn start_server_with_host(port_ptr: *const c_char, host_ptr: *const c_char) -> PluginResult {
    let port_str = unsafe {
        if port_ptr.is_null() {
            "8080"
        } else {
            CStr::from_ptr(port_ptr).to_str().unwrap_or("8080")
        }
    };

    let host_str = unsafe {
        if host_ptr.is_null() {
            "0.0.0.0"  // Default: bind to all interfaces
        } else {
            CStr::from_ptr(host_ptr).to_str().unwrap_or("0.0.0.0")
        }
    };

    let port: u16 = match port_str.parse() {
        Ok(p) => p,
        Err(e) => return PluginResult::error(format!("Invalid port: {}", e)),
    };

    // Validate host address
    let host = host_str.to_string();
    if !is_valid_host(&host) {
        return PluginResult::error(format!("Invalid host address: {}", host));
    }

    // Check if server already running on this port
    if SERVER_RUNNING.contains_key(&port) {
        return PluginResult::error(format!("Server already running on port {}", port));
    }

    // Mark server as running
    SERVER_RUNNING.insert(port, AtomicBool::new(true));

    let host_clone = host.clone();
    // Start server in background
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            if let Err(e) = run_server(port, &host_clone).await {
                tracing::error!("Server error on {}:{}: {}", host_clone, port, e);
                SERVER_RUNNING.remove(&port);
            }
        });
    });

    // Wait a bit to ensure server started
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result_json = serde_json::json!({
        "success": true,
        "status": "started",
        "port": port,
        "host": host
    });

    PluginResult::success(result_json.to_string())
}

fn is_valid_host(host: &str) -> bool {
    // Check for common valid patterns
    if host == "0.0.0.0" || host == "127.0.0.1" || host == "localhost" {
        return true;
    }

    // Check for valid IPv4 address
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() == 4 {
        // Check if all parts are valid u8 (0-255)
        for part in &parts {
            match part.parse::<u8>() {
                Ok(_) => continue,
                Err(_) => {
                    // Not a valid IP, check if it's a hostname
                    return is_valid_hostname(host);
                }
            }
        }
        return true; // Valid IPv4
    }

    // Check for valid hostname
    is_valid_hostname(host)
}

fn is_valid_hostname(host: &str) -> bool {
    // Must not be empty or start/end with dots
    if host.is_empty() || host.starts_with('.') || host.ends_with('.') {
        return false;
    }

    // Check for valid hostname characters (alphanumeric + dots + hyphens)
    host.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '-')
}

async fn run_server(port: u16, host: &str) -> std::io::Result<()> {
    // Initialize tracing only once
    static TRACING_INIT: std::sync::Once = std::sync::Once::new();
    TRACING_INIT.call_once(|| {
        tracing_subscriber::fmt::init();
    });

    let key = Key::generate();

    // Get static directory from environment or use default
    let static_dir = std::env::var("STATIC_DIR").unwrap_or_else(|_| "toolboxv2/dist".to_string());
    let static_dir_clone = static_dir.clone();

    tracing::info!("Starting HTTP server on {}:{}", host, port);
    tracing::info!("Static files directory: {}", static_dir);

    HttpServer::new(move || {
        let static_dir_path = static_dir_clone.clone();

        App::new()
            .wrap(middleware::Logger::default())
            .wrap(SessionMiddleware::new(
                CookieSessionStore::default(),
                key.clone(),
            ))
            // API routes
            .service(
                web::scope("/api")
                    .route("/{module}/{function}", web::post().to(api_handler))
                    .route("/{module}/{function}", web::get().to(api_handler))
            )
            // WebSocket routes
            .service(
                web::scope("/ws")
                    .route("/{module}/{function}", web::get().to(websocket_handler))
            )
            // Health check
            .route("/health", web::get().to(health_check))
            // Static files with path traversal protection
            .service(
                fs::Files::new("/", &static_dir_path)
                    .index_file("index.html")
                    .use_last_modified(true)
                    .use_etag(true)
            )
            // Fallback to index.html for SPA routing
            .default_service(
                web::route().to(move |req: HttpRequest| {
                    let static_path = static_dir_path.clone();
                    async move {
                        let index_path = Path::new(&static_path).join("index.html");
                        if index_path.exists() {
                            match fs::NamedFile::open(index_path) {
                                Ok(file) => Ok::<_, Error>(file.into_response(&req)),
                                Err(_) => Ok(HttpResponse::NotFound().body("404 Not Found")),
                            }
                        } else {
                            Ok(HttpResponse::NotFound().body("404 Not Found"))
                        }
                    }
                })
            )
    })
    .bind((host, port))?
    .run()
    .await
}

async fn health_check() -> Result<HttpResponse, Error> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// Static file handler with path traversal protection
async fn static_file_handler(req: HttpRequest) -> Result<fs::NamedFile, Error> {
    let path: String = req.match_info().query("filename").parse().unwrap_or_default();

    // Get static directory from environment or use default
    let static_dir = std::env::var("STATIC_DIR").unwrap_or_else(|_| "toolboxv2/dist".to_string());
    let base_path = Path::new(&static_dir);

    // Safely join paths with traversal protection
    let safe_path = safe_join(base_path, &path)
        .ok_or_else(|| actix_web::error::ErrorForbidden("Path traversal detected"))?;

    // Check if file exists
    if !safe_path.exists() {
        // Try index.html for SPA routing
        let index_path = base_path.join("index.html");
        if index_path.exists() {
            return Ok(fs::NamedFile::open(index_path)?);
        }
        return Err(actix_web::error::ErrorNotFound("File not found"));
    }

    // Serve the file
    Ok(fs::NamedFile::open(safe_path)?)
}

#[no_mangle]
pub extern "C" fn free_plugin_result(result: PluginResult) {
    unsafe {
        if !result.data.is_null() {
            let _ = CString::from_raw(result.data);
        }
        if !result.error.is_null() {
            let _ = CString::from_raw(result.error);
        }
    }
}

/// Register TB Lang callback for request handling
#[no_mangle]
pub extern "C" fn register_callback(callback_ptr: TbLangCallback) -> PluginResult {
    let mut callback = TB_CALLBACK.lock().unwrap();
    *callback = Some(callback_ptr);

    let result_json = serde_json::json!({
        "success": true,
        "status": "callback_registered"
    });

    PluginResult::success(result_json.to_string())
}

/// Stop server on specified port
#[no_mangle]
pub extern "C" fn stop_server(port_ptr: *const c_char) -> PluginResult {
    let port_str = unsafe {
        if port_ptr.is_null() {
            "8080"
        } else {
            CStr::from_ptr(port_ptr).to_str().unwrap_or("8080")
        }
    };

    let port: u16 = match port_str.parse() {
        Ok(p) => p,
        Err(e) => return PluginResult::error(format!("Invalid port: {}", e)),
    };

    // Check if server is running
    if let Some(entry) = SERVER_RUNNING.get(&port) {
        entry.value().store(false, Ordering::Relaxed);
        SERVER_RUNNING.remove(&port);

        let result_json = serde_json::json!({
            "success": true,
            "status": "stopped",
            "port": port
        });

        PluginResult::success(result_json.to_string())
    } else {
        PluginResult::error(format!("No server running on port {}", port))
    }
}

/// Send message to specific WebSocket connection
#[no_mangle]
pub extern "C" fn send_websocket_message(conn_id_ptr: *const c_char, message_ptr: *const c_char) -> PluginResult {
    let conn_id = unsafe {
        if conn_id_ptr.is_null() {
            return PluginResult::error("Connection ID is null".to_string());
        }
        CStr::from_ptr(conn_id_ptr).to_str().unwrap_or("")
    };

    let message = unsafe {
        if message_ptr.is_null() {
            return PluginResult::error("Message is null".to_string());
        }
        CStr::from_ptr(message_ptr).to_str().unwrap_or("")
    };

    if let Some(conn) = ACTIVE_CONNECTIONS.get(conn_id) {
        let ws_msg = WsMessage {
            source_conn_id: "server".to_string(),
            content: message.to_string(),
            target_conn_id: Some(conn_id.to_string()),
            target_channel_id: None,
        };

        conn.value().do_send(ws_msg);

        let result_json = serde_json::json!({
            "success": true,
            "sent": true
        });

        PluginResult::success(result_json.to_string())
    } else {
        PluginResult::error(format!("Connection not found: {}", conn_id))
    }
}

/// Broadcast message to all connections in channel
#[no_mangle]
pub extern "C" fn broadcast_websocket_message(channel_ptr: *const c_char, message_ptr: *const c_char) -> PluginResult {
    let channel = unsafe {
        if channel_ptr.is_null() {
            return PluginResult::error("Channel is null".to_string());
        }
        CStr::from_ptr(channel_ptr).to_str().unwrap_or("")
    };

    let message = unsafe {
        if message_ptr.is_null() {
            return PluginResult::error("Message is null".to_string());
        }
        CStr::from_ptr(message_ptr).to_str().unwrap_or("")
    };

    let ws_msg = WsMessage {
        source_conn_id: "server".to_string(),
        content: message.to_string(),
        target_conn_id: None,
        target_channel_id: Some(channel.to_string()),
    };

    let sent_count = ACTIVE_CONNECTIONS.len();

    // Try to send, but don't fail if no receivers (RecvError means no receivers)
    let _ = GLOBAL_WS_BROADCASTER.send(ws_msg);

    let result_json = serde_json::json!({
        "success": true,
        "sent": sent_count,
        "channel": channel
    });

    PluginResult::success(result_json.to_string())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_data_default() {
        let session = SessionData::default();
        assert!(!session.valid);
        assert!(session.anonymous);
        assert_eq!(session.request_count, 0);
        assert!(!session.id.is_empty());
    }

    #[test]
    fn test_session_expiration() {
        let mut session = SessionData::default();
        assert!(!session.is_expired());

        // Set expiration to past
        session.expires_at = Utc::now() - chrono::Duration::seconds(10);
        assert!(session.is_expired());
    }

    #[test]
    fn test_session_refresh() {
        let mut session = SessionData::default();
        let old_count = session.request_count;
        let old_expires = session.expires_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        session.refresh();

        assert_eq!(session.request_count, old_count + 1);
        assert!(session.expires_at > old_expires);
    }

    #[test]
    fn test_rate_limiting() {
        let key = "test_ip_123";

        // First request should pass
        assert!(check_rate_limit(key, 5, 60));

        // Next 4 requests should pass
        for _ in 0..4 {
            assert!(check_rate_limit(key, 5, 60));
        }

        // 6th request should fail
        assert!(!check_rate_limit(key, 5, 60));
    }

    #[test]
    fn test_plugin_result_success() {
        let result = PluginResult::success("test data".to_string());
        assert!(result.success);
        assert!(!result.data.is_null());
        assert!(result.error.is_null());

        // Clean up
        free_plugin_result(result);
    }

    #[test]
    fn test_plugin_result_error() {
        let result = PluginResult::error("test error".to_string());
        assert!(!result.success);
        assert!(result.data.is_null());
        assert!(!result.error.is_null());

        // Clean up
        free_plugin_result(result);
    }

    #[test]
    fn test_websocket_message_creation() {
        let msg = WsMessage {
            source_conn_id: "conn1".to_string(),
            content: "test message".to_string(),
            target_conn_id: Some("conn2".to_string()),
            target_channel_id: None,
        };

        assert_eq!(msg.source_conn_id, "conn1");
        assert_eq!(msg.content, "test message");
        assert_eq!(msg.target_conn_id, Some("conn2".to_string()));
        assert_eq!(msg.target_channel_id, None);
    }

    #[actix_rt::test]
    async fn test_health_check() {
        let response = health_check().await.unwrap();
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[test]
    fn test_is_valid_host() {
        // Valid hosts
        assert!(is_valid_host("0.0.0.0"));
        assert!(is_valid_host("127.0.0.1"));
        assert!(is_valid_host("localhost"));
        assert!(is_valid_host("192.168.1.1"));
        assert!(is_valid_host("example.com"));
        assert!(is_valid_host("my-server.local"));

        // Invalid hosts
        assert!(!is_valid_host("")); // Empty
        assert!(!is_valid_host(".example.com")); // Starts with dot
        assert!(!is_valid_host("example.com.")); // Ends with dot
    }

    #[test]
    fn test_start_server_invalid_port() {
        let port = CString::new("invalid").unwrap();
        let result = start_server(port.as_ptr());

        assert!(!result.success);
        assert!(!result.error.is_null());

        unsafe {
            let error_str = CStr::from_ptr(result.error).to_str().unwrap();
            assert!(error_str.contains("Invalid port"));
        }

        free_plugin_result(result);
    }

    #[test]
    fn test_start_server_with_custom_host() {
        let port = CString::new("9999").unwrap();
        let host = CString::new("127.0.0.1").unwrap();

        let result = start_server_with_host(port.as_ptr(), host.as_ptr());

        assert!(result.success);
        assert!(!result.data.is_null());

        unsafe {
            let data_str = CStr::from_ptr(result.data).to_str().unwrap();
            assert!(data_str.contains("127.0.0.1"));
            assert!(data_str.contains("9999"));
        }

        free_plugin_result(result);

        // Stop server
        let stop_result = stop_server(port.as_ptr());
        free_plugin_result(stop_result);
    }

    #[test]
    fn test_start_server_invalid_host() {
        let port = CString::new("9998").unwrap();
        let host = CString::new(".invalid.host").unwrap();

        let result = start_server_with_host(port.as_ptr(), host.as_ptr());

        assert!(!result.success);
        assert!(!result.error.is_null());

        unsafe {
            let error_str = CStr::from_ptr(result.error).to_str().unwrap();
            assert!(error_str.contains("Invalid host"));
        }

        free_plugin_result(result);
    }

    #[test]
    fn test_register_callback() {
        extern "C" fn test_callback(_request: *const c_char) -> *mut c_char {
            let response = r#"{"status": 200, "body": {"message": "test"}}"#;
            CString::new(response).unwrap().into_raw()
        }

        let result = register_callback(test_callback);

        assert!(result.success);
        assert!(!result.data.is_null());

        unsafe {
            let data_str = CStr::from_ptr(result.data).to_str().unwrap();
            assert!(data_str.contains("registered"));
        }

        free_plugin_result(result);
    }

    #[test]
    fn test_send_websocket_message_no_connection() {
        let conn_id = CString::new("nonexistent").unwrap();
        let message = CString::new(r#"{"test": "data"}"#).unwrap();

        let result = send_websocket_message(conn_id.as_ptr(), message.as_ptr());

        assert!(!result.success);
        assert!(!result.error.is_null());

        unsafe {
            let error_str = CStr::from_ptr(result.error).to_str().unwrap();
            assert!(error_str.contains("not found"));
        }

        free_plugin_result(result);
    }

    #[test]
    fn test_broadcast_websocket_message() {
        let channel = CString::new("test_channel").unwrap();
        let message = CString::new(r#"{"test": "broadcast"}"#).unwrap();

        let result = broadcast_websocket_message(channel.as_ptr(), message.as_ptr());

        // Should succeed even with no connections
        assert!(result.success);
        assert!(!result.data.is_null());

        unsafe {
            let data_str = CStr::from_ptr(result.data).to_str().unwrap();
            assert!(data_str.contains("sent"));
        }

        free_plugin_result(result);
    }

    #[test]
    fn test_session_data_new() {
        let session = SessionData::new("192.168.1.1".to_string(), "54321".to_string());

        assert_eq!(session.ip, "192.168.1.1");
        assert_eq!(session.port, "54321");
        assert!(!session.valid);
        assert!(session.anonymous);
        assert!(!session.id.is_empty());
    }

    #[test]
    fn test_multiple_session_refresh() {
        let mut session = SessionData::default();

        for i in 0..5 {
            session.refresh();
            assert_eq!(session.request_count, i + 1);
        }

        assert_eq!(session.request_count, 5);
    }

    #[test]
    fn test_rate_limit_window_reset() {
        let key = "test_window_reset_unique";

        // Use up all tokens
        assert!(check_rate_limit(key, 1, 1));
        assert!(!check_rate_limit(key, 1, 1));

        // Wait for window to reset
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Should work again
        assert!(check_rate_limit(key, 1, 1));
    }
}

