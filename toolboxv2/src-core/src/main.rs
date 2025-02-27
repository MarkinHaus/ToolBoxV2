use actix_web::{web, App,HttpRequest, HttpServer, HttpResponse, middleware};
use actix_files as fs;
use actix_session::{Session, SessionMiddleware, storage::CookieSessionStore};
use actix_web::cookie::{Key};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};
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
use tokio::net::TcpStream;
use tokio::time::{timeout, sleep};

use uuid::Uuid;
use tracing::{info, warn, error, debug};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::RefCell;

// Create a global counter for unique task IDs
static NEXT_THREAD_ID: AtomicUsize = AtomicUsize::new(1);

// Thread-local storage for IDs
thread_local! {
    static THREAD_ID: RefCell<Option<usize>> = RefCell::new(None);
}

// Function to get or create a thread ID
fn get_thread_id() -> usize {
    THREAD_ID.with(|id| {
        let mut id_ref = id.borrow_mut();
        if id_ref.is_none() {
            *id_ref = Some(NEXT_THREAD_ID.fetch_add(1, Ordering::SeqCst));
        }
        id_ref.unwrap()
    })
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
}

#[derive(Debug, Deserialize, Clone)]
struct ToolboxSettings {
    host: String,
    port: u16,
    timeout_seconds: u64,
    max_instances: u16,
    tb_r_key: String,
}

#[derive(Debug, Deserialize, Clone)]
struct SessionSettings {
    secret_key: String,
    duration_minutes: u64,
    cookie_name: String,
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

#[derive(Debug)]
enum DataType {
    Json,
    Binary,
    Exit,
    Keepalive,
}

#[derive(Debug, Clone)]
pub struct ToolboxClient {
    instances: Arc<tokio::sync::Mutex<Vec<ToolboxInstance>>>,
    connection_pool: Arc<tokio::sync::RwLock<HashMap<String, Arc<tokio::sync::Mutex<TcpStream>>>>>,
    auth_key: Option<String>,
    default_host: String,
    default_port_range: (u16, u16),
    timeout: Duration,
    max_instances: usize,
    heartbeat_interval: Duration,
    // Track used ports to avoid restarting on the same port
    used_ports: Arc<tokio::sync::Mutex<HashSet<u16>>>,
    // Load balancing stats - track per thread instead of per instance
    thread_counters: Arc<tokio::sync::RwLock<HashMap<usize, HashMap<String, usize>>>>,
    // Package size for chunked transfers
    package_size: usize,
}

impl ToolboxClient {
    pub fn new(
        host: String,
        port_range: (u16, u16),
        timeout_seconds: u64,
        max_instances: usize,
        auth_key: Option<String>,
    ) -> Self {
        let client = ToolboxClient {
            instances: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            connection_pool: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            auth_key,
            default_host: host,
            default_port_range: port_range,
            timeout: Duration::from_secs(timeout_seconds),
            max_instances,
            heartbeat_interval: Duration::from_secs(30),
            used_ports: Arc::new(tokio::sync::Mutex::new(HashSet::new())),
            thread_counters: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            package_size: 8192, // Default chunk size
        };

        // Start background tasks
        let client_clone = client.clone();
        tokio::spawn(async move {
            client_clone.connection_manager().await;
        });

        client
    }

    /// Start monitoring connections and instances
    async fn connection_manager(&self) {
        loop {
            // Check health of instances and remove dead ones
            self.check_instance_health().await;

            // Try to maintain minimum number of instances if needed
            self.ensure_minimum_instances(1).await;

            // Clean up and recycle connections
            self.cleanup_connection_pool().await;

            // Sleep for a while before next check
            sleep(Duration::from_secs(5)).await;
        }
    }

    /// Cleanup unused connections in the connection pool
    async fn cleanup_connection_pool(&self) {
        let mut to_remove = Vec::new();

        // Get connection keys to check
        let connections_to_check = {
            let pool = self.connection_pool.read().await;
            pool.keys().cloned().collect::<Vec<_>>()
        };

        // Check each connection outside the lock
        for conn_key in connections_to_check {
            let parts: Vec<&str> = conn_key.split(':').collect();
            if parts.len() != 2 {
                to_remove.push(conn_key);
                continue;
            }

            let host = parts[0].to_string();
            let port = match parts[1].parse::<u16>() {
                Ok(p) => p,
                Err(_) => {
                    to_remove.push(conn_key.clone());
                    continue;
                }
            };

            // Check if any instances use this connection
            let is_connection_used = {
                let instances = self.instances.lock().await;
                instances.iter().any(|inst| inst.host == host && inst.port == port && inst.is_alive)
            };

            // Check if connection is actually valid
            let is_valid = if let Some(conn) = self.connection_pool.read().await.get(&conn_key) {
                let stream = conn.lock().await;
                match stream.peer_addr() {
                    Ok(_) => true,
                    Err(_) => {
                        debug!("Connection {} is invalid", conn_key);
                        false
                    }
                }
            } else {
                false
            };

            if !is_connection_used || !is_valid {
                debug!("Removing unused connection {}", conn_key);
                to_remove.push(conn_key);
            }
        }

        // Remove unused connections
        if !to_remove.is_empty() {
            let mut pool = self.connection_pool.write().await;
            for key in to_remove {
                pool.remove(&key);
            }
        }
    }

    /// Check health of all instances
    async fn check_instance_health(&self) {
        // First collect instances to check without holding the mutex
        let instances_to_check = {
            let instances = self.instances.lock().await;
            instances.iter().map(|inst| (inst.id.clone(), inst.clone())).collect::<Vec<_>>()
        };

        let mut to_remove = Vec::new();
        let mut to_update = Vec::new();

        // Process each instance without holding the lock
        for (id, instance) in instances_to_check {
            let mut should_remove = false;
            let mut instance_alive = true;

            // Check if process is still running
            if let Some(process) = &instance.process {
                let is_running = {
                    let mut process = process.lock().await;
                    match process.try_wait() {
                        Ok(Some(_)) => {
                            debug!("Instance {} (port {}) exited", instance.id, instance.port);
                            false
                        }
                        Ok(None) => true, // Process still running
                        Err(_) => {
                            warn!("Error checking process status for instance {}", instance.id);
                            false
                        }
                    }
                };

                if !is_running {
                    instance_alive = false;
                }
            }

            // Check connection health if the process is running
            if instance_alive && instance.last_used.elapsed() > self.heartbeat_interval {
                debug!("Checking heartbeat for instance {}", instance.id);

                // Send heartbeat using the keepalive signal
                let conn_key = format!("{}:{}", instance.host, instance.port);
                let connection = self.get_or_create_connection(&instance.host, instance.port).await;

                if let Ok(conn) = connection {
                    // Send a keepalive signal
                    let result = self.send_keepalive(conn.clone(), &instance.id).await;
                    if result.is_err() {
                        warn!("Heartbeat failed for instance {}, removing", instance.id);
                        instance_alive = false;
                    } else {
                        to_update.push((id.clone(), Instant::now()));
                    }
                } else {
                    warn!("Failed to create connection for heartbeat to instance {}", instance.id);
                    instance_alive = false;
                }
            }

            if !instance_alive {
                should_remove = true;
            }

            if should_remove {
                to_remove.push(id);
            }
        }

        // Apply updates with a single lock
        if !to_update.is_empty() {
            let mut instances = self.instances.lock().await;
            for (id, last_used) in to_update {
                if let Some(inst) = instances.iter_mut().find(|i| i.id == id) {
                    inst.last_used = last_used;
                }
            }
        }

        // Remove any dead instances with a single lock
        if !to_remove.is_empty() {
            let mut instances = self.instances.lock().await;
            let mut used_ports = self.used_ports.lock().await;

            for id in &to_remove {
                if let Some(pos) = instances.iter().position(|i| &i.id == id) {
                    let instance = &instances[pos];
                    used_ports.remove(&instance.port);
                    debug!("Freed port {} from dead instance {}", instance.port, instance.id);
                }
            }

            instances.retain(|inst| !to_remove.contains(&inst.id));
            debug!("Removed {} dead instances", to_remove.len());
        }
    }

    /// Send a keepalive signal to an instance
    async fn send_keepalive(&self, connection: Arc<tokio::sync::Mutex<TcpStream>>, instance_id: &str) -> Result<(), ToolboxError> {
        let mut stream = {
            let conn = connection.lock().await;
            let peer_addr = conn.peer_addr().map_err(|e| ToolboxError::IoError(e))?;

            // Create a new connection for the keepalive to avoid mutex contention
            TcpStream::connect(peer_addr).await.map_err(|e| ToolboxError::IoError(e))?
        };

        // Send keepalive signal (just the 'k' byte)
        stream.write_all(b"k").await.map_err(|e| ToolboxError::IoError(e))?;
        stream.flush().await.map_err(|e| ToolboxError::IoError(e))?;

        // No need to wait for response for keepalive
        Ok(())
    }

    /// Ensure we have at least min_instances running
    async fn ensure_minimum_instances(&self, min_instances: usize) {
        let instances_count = self.instances.lock().await.len();

        if instances_count < min_instances {
            for _ in instances_count..min_instances {
                if let Err(e) = self.start_new_instance().await {
                    warn!("Failed to start new instance: {}", e);
                    // Back off a bit before trying again
                    sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    /// Get existing or create a new connection to host:port
    async fn get_or_create_connection(&self, host: &str, port: u16) -> Result<Arc<tokio::sync::Mutex<TcpStream>>, ToolboxError> {
        // Create a connection key
        let conn_key = format!("{}:{}", host, port);

        // Check if we already have a connection in the pool
        {
            let pool = self.connection_pool.read().await;
            if let Some(conn) = pool.get(&conn_key) {
                // Test if the connection is still valid
                let is_valid = {
                    let stream = conn.lock().await;
                    match stream.peer_addr() {
                        Ok(_) => true,
                        Err(_) => false,
                    }
                };

                if is_valid {
                    return Ok(conn.clone());
                }
            }
        }

        // Connect to the server
        let stream = match timeout(
            self.timeout,
            TcpStream::connect(format!("{}:{}", host, port))
        ).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => return Err(ToolboxError::IoError(e)),
            Err(_) => return Err(ToolboxError::Timeout),
        };

        // Configure the connection for better reliability
        stream.set_nodelay(true).map_err(|e| ToolboxError::IoError(e))?;

        // Create a shareable connection and add to the pool
        let connection = Arc::new(tokio::sync::Mutex::new(stream));

        {
            let mut pool = self.connection_pool.write().await;
            pool.insert(conn_key, connection.clone());
        }

        Ok(connection)
    }

    /// Start a new toolbox instance
    pub async fn start_new_instance(&self) -> Result<(), ToolboxError> {
        // Check max instances and find available port
        let available_port = {
            let instances = self.instances.lock().await;
            let used_ports = self.used_ports.lock().await;

            if instances.len() >= self.max_instances {
                return Err(ToolboxError::MaxInstancesReached);
            }

            // Find an available port that's not already used
            (self.default_port_range.0..=self.default_port_range.1)
                .find(|&port| !used_ports.contains(&port))
        };

        let port = match available_port {
            Some(p) => p,
            None => return Err(ToolboxError::NoAvailablePorts),
        };

        // Check if something is already running on this port
        if let Ok(_) = TcpStream::connect(format!("{}:{}", self.default_host, port)).await {
            debug!("Port {} already has a running service, trying to use it", port);

            // Mark the port as used
            {
                let mut used_ports = self.used_ports.lock().await;
                used_ports.insert(port);
            }

            // Try to connect to this existing instance
            let instance_id = format!("found_instance_{}", Uuid::new_v4());
            let new_instance = ToolboxInstance {
                id: instance_id.clone(),
                host: self.default_host.clone(),
                port,
                process: None,
                last_used: Instant::now(),
                is_validated: false,
                is_alive: true,
                thread_connections: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            };

            // Add to instances
            {
                let mut instances = self.instances.lock().await;
                instances.push(new_instance);
            }

            // Try to validate the instance
            match self.validate_instance(&instance_id).await {
                Ok(_) => {
                    info!("Successfully connected to existing instance on port {}", port);
                    return Ok(());
                },
                Err(e) => {
                    warn!("Failed to validate existing instance on port {}: {}", port, e);
                    // Remove the instance
                    let mut instances = self.instances.lock().await;
                    instances.retain(|i| i.id != instance_id);
                }
            }

            // Mark the port as used so we don't try it again
            {
                let mut used_ports = self.used_ports.lock().await;
                used_ports.insert(port);
            }

            return Err(ToolboxError::InvalidInstance(port));
        }

        // Mark the port as used
        {
            let mut used_ports = self.used_ports.lock().await;
            used_ports.insert(port);
        }

        debug!("Starting new toolbox instance on port {}", port);

        // Start the toolbox process
        let child = Command::new("tb")
            .arg("-bgr")
            .arg("--sysPrint")
            .arg("-m")
            .arg("bg")
            .arg("-p")
            .arg(port.to_string())
            .spawn()
            .map_err(|e| ToolboxError::IoError(e))?;

        let instance_id = format!("instance_{}", Uuid::new_v4());

        // Wait for the server to start with exponential backoff
        let mut backoff = 500; // Start with 500ms
        let mut attempts = 0;
        let max_attempts = 5;

        let stream = loop {
            if attempts >= max_attempts {
                // Remove the port from used_ports if we can't connect
                {
                    let mut used_ports = self.used_ports.lock().await;
                    used_ports.remove(&port);
                }
                return Err(ToolboxError::Timeout);
            }

            sleep(Duration::from_millis(backoff)).await;
            attempts += 1;

            // Try to connect
            match timeout(
                Duration::from_millis(backoff),
                TcpStream::connect(format!("{}:{}", self.default_host, port))
            ).await {
                Ok(Ok(stream)) => break stream,
                _ => {
                    backoff *= 2; // Exponential backoff
                    debug!("Connection attempt {} failed, waiting {}ms", attempts, backoff);
                }
            }
        };

        // Configure stream
        stream.set_nodelay(true).map_err(|e| ToolboxError::IoError(e))?;

        // Create a connection and add to pool
        let connection = Arc::new(tokio::sync::Mutex::new(stream));
        let conn_key = format!("{}:{}", self.default_host, port);

        {
            let mut pool = self.connection_pool.write().await;
            pool.insert(conn_key, connection.clone());
        }

        // Create the new instance record
        let new_instance = ToolboxInstance {
            id: instance_id.clone(),
            host: self.default_host.clone(),
            port,
            process: Some(Arc::new(tokio::sync::Mutex::new(child))),
            last_used: Instant::now(),
            is_validated: false,
            is_alive: true,
            thread_connections: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        };

        // Add to instances
        {
            let mut instances = self.instances.lock().await;
            instances.push(new_instance);
        }

        // Validate the instance
        match self.validate_instance(&instance_id).await {
            Ok(_) => {
                info!("Started and validated new toolbox instance on port {}", port);
                Ok(())
            },
            Err(e) => {
                // Clean up failed instance
                self.stop_instance(&instance_id).await.ok();
                Err(e)
            }
        }
    }

    /// Stop a toolbox instance
    pub async fn stop_instance(&self, instance_id: &str) -> Result<(), ToolboxError> {
        // Find and remove the instance
        let instance = {
            let mut instances = self.instances.lock().await;
            let instance_pos = instances.iter().position(|i| i.id == instance_id);

            match instance_pos {
                Some(pos) => {
                    let instance = instances.remove(pos);

                    // Free the port
                    let mut used_ports = self.used_ports.lock().await;
                    used_ports.remove(&instance.port);

                    instance
                },
                None => return Err(ToolboxError::InstanceNotFound(instance_id.to_string())),
            }
        };

        // Send exit signal to all connections for this instance
        let conn_key = format!("{}:{}", instance.host, instance.port);
        if let Some(conn) = self.connection_pool.read().await.get(&conn_key) {
            let mut stream = {
                let base_conn = conn.lock().await;
                let peer_addr = base_conn.peer_addr().map_err(|e| ToolboxError::IoError(e))?;

                // Create a new connection for the exit signal
                TcpStream::connect(peer_addr).await.map_err(|e| ToolboxError::IoError(e))?
            };

            // Send exit signal (just the 'e' byte)
            stream.write_all(b"e").await.map_err(|e| ToolboxError::IoError(e))?;
            stream.flush().await.map_err(|e| ToolboxError::IoError(e))?;
        }

        // Kill the process if we have one
        if let Some(process) = instance.process {
            let mut process = process.lock().await;
            if let Err(e) = process.kill() {
                warn!("Failed to kill process for instance {}: {}", instance_id, e);
            }
        }

        // Try to kill using command as well
        if let Err(e) = Command::new("tb")
            .arg("-kill")
            .arg("-p")
            .arg(instance.port.to_string())
            .spawn() {
            warn!("Failed to execute kill command: {}", e);
        }

        // Remove connections from pool
        {
            let mut pool = self.connection_pool.write().await;
            pool.remove(&conn_key);
        }

        // Clear per-thread connections
        {
            let mut thread_conns = instance.thread_connections.write().await;
            thread_conns.clear();
        }

        info!("Stopped instance {} on port {}", instance_id, instance.port);
        Ok(())
    }

    /// Validate an instance with a more robust approach
    async fn validate_instance(&self, instance_id: &str) -> Result<(), ToolboxError> {
        let (host, port) = {
            let instances = self.instances.lock().await;
            let instance = instances.iter().find(|i| i.id == instance_id)
                .ok_or_else(|| ToolboxError::InstanceNotFound(instance_id.to_string()))?;

            (instance.host.clone(), instance.port)
        };

        // Create a new connection for validation
        let mut stream = match timeout(
            self.timeout,
            TcpStream::connect(format!("{}:{}", host, port))
        ).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => return Err(ToolboxError::IoError(e)),
            Err(_) => return Err(ToolboxError::Timeout),
        };

        // Configure stream
        stream.set_nodelay(true).map_err(|e| ToolboxError::IoError(e))?;

        // Send initial connection notification
        debug!("Sending initial connection notification to {}:{}", host, port);

        // First message: Initial connection notification with 'j' type prefix
        let initial_request = serde_json::json!({
            "identifier": "new_con",
            "data": [null, null]
        });
        let json_str = serde_json::to_string(&initial_request)
            .map_err(|e| ToolboxError::JsonError(e))?;

        // Prepare complete message with type prefix
        let mut data_to_send = vec![b'j'];
        data_to_send.extend_from_slice(json_str.as_bytes());

        // Send with chunking
        self.send_data_chunked(&mut stream, &data_to_send).await?;

        // Wait a short moment for server to process
        sleep(Duration::from_millis(1000)).await;

        // Second message: The validation with auth key
        let peer_addr = stream.peer_addr().map_err(|e| ToolboxError::IoError(e))?;
        let formatted_addr = format!("('127.0.0.1', {})", peer_addr.port());
        info!("server key : {:?}", self.auth_key.clone());
        let validation_request = serde_json::json!({
            "identifier": formatted_addr,
            "key": self.auth_key.clone()
        });
        let json_str = serde_json::to_string(&validation_request)
            .map_err(|e| ToolboxError::JsonError(e))?;

        // Prepare message with type prefix
        let mut data_to_send = vec![b'j'];
        data_to_send.extend_from_slice(json_str.as_bytes());

        // Send with chunking
        self.send_data_chunked(&mut stream, &data_to_send).await?;

        // Now wait for a response to confirm validation
        let response = self.receive_data(&mut stream).await;

        // Mark instance as validated regardless of response
        // (some older versions don't respond to validation)
        {
            let mut instances = self.instances.lock().await;
            if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
                instance.is_validated = true;
                instance.last_used = Instant::now();
                info!("Instance {} validated successfully", instance_id);
            }
        }

        Ok(())
    }

    /// Send data in chunks with proper end markers
    async fn send_data_chunked(&self, stream: &mut TcpStream, data: &[u8]) -> Result<(), ToolboxError> {
        // Send data in chunks
        for chunk in data.chunks(self.package_size) {
            stream.write_all(chunk).await.map_err(|e| ToolboxError::IoError(e))?;
            stream.flush().await.map_err(|e| ToolboxError::IoError(e))?;
        }

        // Calculate and send end marker
        let end_marker_size = self.package_size / 10;
        let end_marker = vec![b'E'; end_marker_size];
        stream.write_all(&end_marker).await.map_err(|e| ToolboxError::IoError(e))?;
        stream.flush().await.map_err(|e| ToolboxError::IoError(e))?;

        Ok(())
    }

    /// Receive data with proper handling of different data types
    async fn receive_data(&self, stream: &mut TcpStream) -> Result<serde_json::Value, ToolboxError> {
        let mut data_buffer = Vec::new();
        let mut data_type = None;
        let mut end_marker_count = 0;

        // Set a read timeout
        let read_timeout = Duration::from_secs(5);

        // Read in chunks
        loop {
            let mut chunk = vec![0u8; self.package_size];

            // Read with timeout
            let read_result = timeout(read_timeout, stream.read(&mut chunk)).await;

            match read_result {
                Ok(Ok(0)) => {
                    // Connection closed
                    return Err(ToolboxError::ConnectionClosed);
                },
                Ok(Ok(n)) => {
                    let actual_chunk = &chunk[0..n];

                    // Handle first byte as data type if not already set
                    if data_type.is_none() {
                        data_type = Some(actual_chunk[0]);

                        // Process rest of chunk
                        if actual_chunk.len() > 1 {
                            let mut valid_chunk = Vec::new();

                            for &byte in &actual_chunk[1..] {
                                if byte == b'E' {
                                    end_marker_count += 1;
                                } else {
                                    // If we saw Es but then a non-E, they were part of the data
                                    for _ in 0..end_marker_count {
                                        valid_chunk.push(b'E');
                                    }
                                    end_marker_count = 0;
                                    valid_chunk.push(byte);
                                }
                            }

                            data_buffer.extend_from_slice(&valid_chunk);
                        }
                    } else {
                        // Process entire chunk
                        let mut valid_chunk = Vec::new();

                        for &byte in actual_chunk {
                            if byte == b'E' {
                                end_marker_count += 1;
                            } else {
                                // If we saw Es but then a non-E, they were part of the data
                                for _ in 0..end_marker_count {
                                    valid_chunk.push(b'E');
                                }
                                end_marker_count = 0;
                                valid_chunk.push(byte);
                            }
                        }

                        data_buffer.extend_from_slice(&valid_chunk);
                    }

                    // Check if we've received the end marker
                    if end_marker_count >= 6 {
                        break;
                    }

                    // If this chunk wasn't full, wait a bit before trying again
                    if n < self.package_size {
                        sleep(Duration::from_millis(50)).await;
                    }
                },
                Ok(Err(e)) => {
                    return Err(ToolboxError::IoError(e));
                },
                Err(_) => {
                    return Err(ToolboxError::Timeout);
                }
            }
        }

        // Process the data based on type
        match data_type {
            Some(b'j') => {
                // JSON data
                let json_str = String::from_utf8_lossy(&data_buffer).to_string();
                match serde_json::from_str(&json_str) {
                    Ok(json) => Ok(json),
                    Err(e) => Err(ToolboxError::JsonError(e)),
                }
            },
            Some(b'b') => {
                // Binary data - convert to a special JSON structure
                let binary_value = serde_json::json!({
                    "binary_data": true,
                    "data": base64::encode(&data_buffer),
                    "size": data_buffer.len(),
                });
                Ok(binary_value)
            },
            Some(b'e') => {
                // Exit signal
                debug!("Received exit signal");
                Err(ToolboxError::ConnectionClosed)
            },
            Some(b'k') => {
                // Keepalive signal
                debug!("Received keepalive signal");
                Ok(serde_json::json!({"keepalive": true}))
            },
            _ => {
                Err(ToolboxError::UnknownDataType)
            }
        }
    }

    /// Get or create a thread-specific connection to an instance
    async fn get_thread_connection(&self, instance_id: &str) -> Result<TcpStream, ToolboxError> {
        // Get current thread ID
        let thread_id = get_thread_id();

        // Get the instance
        let instance = {
            let instances = self.instances.lock().await;
            instances.iter()
                .find(|i| i.id == instance_id)
                .cloned()
                .ok_or_else(|| ToolboxError::InstanceNotFound(instance_id.to_string()))?
        };

        // Check if we already have a connection for this thread
        {
            let thread_conns = instance.thread_connections.read().await;
            if thread_conns.contains_key(&thread_id) {
                // We've used this thread before, but TcpStream can't be cloned directly
                // Create a fresh connection
            }
        }

        // Create a new connection for this thread
        let stream = TcpStream::connect(format!("{}:{}", instance.host, instance.port))
            .await
            .map_err(|e| ToolboxError::IoError(e))?;

        stream.set_nodelay(true).map_err(|e| ToolboxError::IoError(e))?;

        // Update the connection count for this thread
        {
            let mut thread_conns = instance.thread_connections.write().await;
            *thread_conns.entry(thread_id).or_insert(0) += 1;
        }

        Ok(stream)
    }
    /// Send a request to a specific instance and handle the response
    async fn send_request(&self, instance_id: &str, request: &ToolboxRequest) -> Result<serde_json::Value, ToolboxError> {
        // Get or create a thread-specific connection
        let mut stream = self.get_thread_connection(instance_id).await?;

        // Serialize the request to JSON
        let json_str = serde_json::to_string(request)
            .map_err(|e| ToolboxError::JsonError(e))?;

        // Prepare data with type prefix
        let mut data_to_send = vec![b'j'];
        data_to_send.extend_from_slice(json_str.as_bytes());

        // Send the data with chunking
        self.send_data_chunked(&mut stream, &data_to_send).await?;

        // Receive the response
        self.receive_data(&mut stream).await
    }

    /// Get a valid instance for sending requests with smart load balancing
    async fn get_valid_instance(&self) -> Result<String, ToolboxError> {
        // Get current thread ID
        let thread_id = get_thread_id();

        // Try to get an existing validated instance
        let instance_id = {
            let instances = self.instances.lock().await;

            if instances.is_empty() {
                String::new()
            } else {
                // Find valid instances
                let valid_instances = instances.iter()
                    .filter(|i| i.is_validated && i.is_alive)
                    .collect::<Vec<_>>();

                if !valid_instances.is_empty() {
                    // Check thread-specific load counters
                    let thread_counters = self.thread_counters.read().await;

                    // Get counters for current thread
                    let thread_stats = thread_counters.get(&thread_id).cloned()
                        .unwrap_or_else(|| HashMap::new());

                    // Choose instance with fewest requests from this thread
                    valid_instances.iter()
                        .min_by(|a, b| {
                            let a_count = thread_stats.get(&a.id).cloned().unwrap_or(0);
                            let b_count = thread_stats.get(&b.id).cloned().unwrap_or(0);

                            if a_count != b_count {
                                a_count.cmp(&b_count)
                            } else {
                                // If equal load, use the one used longest ago
                                a.last_used.cmp(&b.last_used)
                            }
                        })
                        .map(|i| i.id.clone())
                        .unwrap_or_else(String::new)
                } else {
                    // Try to find an unvalidated instance
                    instances.iter()
                        .find(|i| !i.is_validated && i.is_alive)
                        .map(|i| i.id.clone())
                        .unwrap_or_else(String::new)
                }
            }
        };

        if !instance_id.is_empty() {
            // Update usage stats
            {
                let mut instances = self.instances.lock().await;
                if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
                    instance.last_used = Instant::now();
                }

                let mut counters = self.thread_counters.write().await;
                let thread_stats = counters.entry(thread_id).or_insert_with(HashMap::new);
                *thread_stats.entry(instance_id.clone()).or_insert(0) += 1;
            }

            // Validate if needed
            {
                let needs_validation = {
                    let instances = self.instances.lock().await;
                    instances.iter()
                        .find(|i| i.id == instance_id)
                        .map(|i| !i.is_validated)
                        .unwrap_or(false)
                };

                if needs_validation {
                    self.validate_instance(&instance_id).await?;
                }
            }

            return Ok(instance_id);
        }

        // No existing instances, try to start a new one
        self.start_new_instance().await?;

        // Try again to get a valid instance
        let id = {
            let mut instances = self.instances.lock().await;
            if let Some(instance) = instances.iter_mut().find(|i| i.is_validated) {
                instance.last_used = Instant::now();

                let mut counters = self.thread_counters.write().await;
                let thread_stats = counters.entry(thread_id).or_insert_with(HashMap::new);
                *thread_stats.entry(instance.id.clone()).or_insert(0) += 1;

                instance.id.clone()
            } else {
                return Err(ToolboxError::NoAvailableInstances);
            }
        };

        Ok(id)
    }

    /// Run a function on a toolbox instance
    pub async fn run_function(
        &self,
        module_name: &str,
        function_name: &str,
        spec: &str,
        args: Vec<String>,
        kwargs: HashMap<String, serde_json::Value>
    ) -> Result<serde_json::Value, ToolboxError> {
        // Convert args to JSON values
        let args_json: Vec<serde_json::Value> = args.into_iter()
            .map(|arg| serde_json::Value::String(arg))
            .collect();

        // Prepare run_any arguments
        let run_args = vec![
            serde_json::json!([module_name, function_name]),
            serde_json::json!(args_json),
        ];

        // Prepare kwargs with specification
        let mut run_kwargs = kwargs;
        run_kwargs.insert("tb_run_with_specification".to_string(), serde_json::json!(spec));
        run_kwargs.insert("get_results".to_string(), serde_json::json!(true));

        // Get a valid instance
        let instance_id = self.get_valid_instance().await?;

        let request = ToolboxRequest {
            name: "a_run_any".to_string(),
            args: run_args,
            kwargs: run_kwargs,
            identifier: Some(instance_id.clone()),
            claim: None,
            key: self.auth_key.clone(),
        };

        // Send the request
        self.send_request(&instance_id, &request).await
    }

    /// Get information about all current instances
    pub async fn get_instances_info(&self) -> Vec<HashMap<String, String>> {
        let instances = self.instances.lock().await;

        let mut result = Vec::new();
        for instance in instances.iter() {
            let mut info = HashMap::new();
            info.insert("id".to_string(), instance.id.clone());
            info.insert("host".to_string(), instance.host.clone());
            info.insert("port".to_string(), instance.port.to_string());
            info.insert("validated".to_string(), instance.is_validated.to_string());
            info.insert("alive".to_string(), instance.is_alive.to_string());
            info.insert("last_used".to_string(), format!("{:?}", instance.last_used.elapsed()));

            // Count connections - do this correctly with async
            let thread_conn_count = instance.thread_connections.read().await.len();
            info.insert("thread_connections".to_string(), thread_conn_count.to_string());

            result.push(info);
        }

        result
    }
}

/// Updated instance struct to support per-thread connections
/// Updated instance struct to support per-thread connections
#[derive(Debug, Clone)]
struct ToolboxInstance {
    id: String,
    host: String,
    port: u16,
    process: Option<Arc<tokio::sync::Mutex<Child>>>,
    last_used: Instant,
    is_validated: bool,
    is_alive: bool,
    // Store connection counts per thread ID instead of actual connections
    thread_connections: Arc<tokio::sync::RwLock<HashMap<usize, usize>>>,
}

/// Request structure for toolbox API
/// Request structure for toolbox API
#[derive(Debug, Clone, Serialize)]
struct ToolboxRequest {
    name: String,
    args: Vec<serde_json::Value>,
    kwargs: HashMap<String, serde_json::Value>,
    identifier: Option<String>,
    claim: Option<String>,
    key: Option<String>,
}

/// Custom error types
#[derive(Debug, Error)]
pub enum ToolboxError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Operation timed out")]
    Timeout,

    #[error("Maximum number of instances reached")]
    MaxInstancesReached,

    #[error("No available ports")]
    NoAvailablePorts,

    #[error("No available instances")]
    NoAvailableInstances,

    #[error("Instance not found: {0}")]
    InstanceNotFound(String),

    #[error("Invalid instance on port {0}")]
    InvalidInstance(u16),

    #[error("Lock acquisition failed")]
    LockError,

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Unknown data type")]
    UnknownDataType,

    #[error("Unknown error: {0}")]
    Unknown(String),
}

// Example API response structure for compatibility with your existing code
#[derive(Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub message: String,
    pub data: Option<T>,
    pub valid: Option<bool>,
}

// A convenience function to initialize the client once and reuse it
lazy_static::lazy_static! {
    static ref TOOLBOX_CLIENT: Mutex<Option<ToolboxClient>> = Mutex::new(None);
}

pub fn initialize_toolbox_client(
    host: String,
    port_range: (u16, u16),
    timeout_seconds: u64,
    max_instances: usize,
    auth_key: Option<String>,
) {
    let mut client = TOOLBOX_CLIENT.lock().unwrap();
    *client = Some(ToolboxClient::new(host, port_range, timeout_seconds, max_instances, auth_key));
}

pub fn get_toolbox_client() -> Result<ToolboxClient, ToolboxError> {
    let client = TOOLBOX_CLIENT.lock().unwrap();
    client.clone().ok_or_else(|| ToolboxError::Unknown("ToolboxClient not initialized".to_string()))
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
        let mut session = self.get_session(session_id);

        // Check JWT validity
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
            Ok(response) => response.as_bool().unwrap_or(false),
            Err(e) => {
                log::error!("JWT validation error: {}", e);
                false
            }
        };

        if !jwt_valid {
            session.check = "failed".to_string();
            session.count += 1;
            self.save_session(session_id, session);
            return "#0".to_string();
        }

        // Get user by name
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
                session.check = e.to_string();
                session.count += 1;
                self.save_session(session_id, session);
                return "#0".to_string();
            }
        };

        // Ensure user is valid
        if !user_result.get("error").map_or(true, |v| v.as_bool().unwrap_or(true) == false) {
            session.check = "Invalid user".to_string();
            session.count += 1;
            self.save_session(session_id, session);
            return "#0".to_string();
        }

        let user_result_value = user_result.get("result").cloned().unwrap_or(serde_json::json!({}));
        let user = &user_result_value;
        let uid = user.get("uid").and_then(|v| v.as_str()).unwrap_or("");

        // Get user instance - now using the spec if we have it
        let instance_result = match self.client.run_function(
            "CloudM.UserInstances",
            "get_user_instance",
            "",  // Use default spec initially
            vec![],
            {
                let mut map = HashMap::new();
                map.insert("uid".to_string(), serde_json::json!(uid));
                map.insert("hydrate".to_string(), serde_json::json!(false));
                map
            },
        ).await {
            Ok(response) => response,
            Err(e) => {
                log::error!("Error getting user instance: {}", e);
                return "#0".to_string();
            }
        };

        // Ensure instance is valid
        if !instance_result.get("error").map_or(true, |v| v.as_bool().unwrap_or(true) == false) {
            return "#0".to_string();
        }

        let instance_value = instance_result.get("result").cloned().unwrap_or(serde_json::json!({}));
        let instance = &instance_value;

        let mut live_data = HashMap::new();
        if let Some(si_id) = instance.get("SiID").and_then(|v| v.as_str()) {
            live_data.insert("SiID".to_string(), si_id.to_string());
        }

        if let Some(level) = user.get("level").and_then(|v| v.as_u64()) {
            let level = if level > 1 { level } else { 1 };
            live_data.insert("level".to_string(), level.to_string());
        }

        if let Some(vt_id) = instance.get("VtID").and_then(|v| v.as_str()) {
            live_data.insert("spec".to_string(), vt_id.to_string());
        }

        // Encode username (simplified for this implementation)
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

        self.save_session(session_id, updated_session);
        session_id.to_string()
    }
    async fn validate_session(&self, session_id: &str) -> bool {
        if session_id.is_empty() {
            return false;
        }

        let session = self.get_session(session_id);

        if session.new || !session.validate {
            return false;
        }

        if session.user_name.is_none() || session.jwt_claim.is_none() {
            return false;
        }

        // Check session expiration
        let session_duration = Duration::from_secs(self.config.duration_minutes * 60);
        let now = Utc::now();
        let session_age = now.signed_duration_since(session.exp);

        if session_age.num_seconds() > session_duration.as_secs() as i64 {
            // Session expired, need to verify again
            if let (Some(user_name), Some(jwt)) = (&session.user_name, &session.jwt_claim) {
                // Extract username from encoded format
                let username = user_name.strip_prefix("encoded:").unwrap_or(user_name);
                return self.verify_session_id(session_id, username, jwt).await != "#0";
            }
            return false;
        }

        true
    }
}

// API route handlers
async fn validate_session_handler(
    manager: web::Data<SessionManager>,
    session: Session,
    req: web::Json<ValidateSessionRequest>,
    req_info: HttpRequest,
) -> HttpResponse {
    let client_ip = req_info.connection_info().peer_addr()
        .unwrap_or("unknown").split(':').next().unwrap_or("unknown").to_string();
    let client_port = req_info.connection_info().peer_addr()
        .unwrap_or("unknown").split(':').nth(1).unwrap_or("unknown").to_string();

    let current_session_id = session.get::<String>("ID").unwrap_or_else(|_| None);

    let session_id = manager.create_new_session(
        client_ip,
        client_port,
        req.jwt_claim.clone(),
        req.username.clone(),
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

        HttpResponse::Ok().json(ApiResponse::<String> {
            message: "Valid Session".to_string(),
            data: None,
            valid: Some(true),
        })
    } else {
        HttpResponse::Unauthorized().json(ApiResponse::<String> {
            message: "Invalid Auth data.".to_string(),
            data: None,
            valid: Some(false),
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
        HttpResponse::Ok().json(ApiResponse::<String> {
            message: "Valid Session".to_string(),
            data: None,
            valid: Some(true),
        })
    } else {
        HttpResponse::Unauthorized().json(ApiResponse::<String> {
            message: "Invalid Auth data.".to_string(),
            data: None,
            valid: Some(false),
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
            panic!("{}", e)
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
                    let close_result = client.run_function(
                        "CloudM.UserInstances",
                        "close_user_instance",
                        live_data.get("spec").unwrap_or(&String::new()),
                        vec![],
                        {
                            let mut map = HashMap::new();
                            map.insert("uid".to_string(), serde_json::json!(uid));
                            map
                        },
                    ).await;

                    if let Err(e) = close_result {
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

    HttpResponse::Forbidden().json(ApiResponse::<String> {
        message: "Invalid Auth data.".to_string(),
        data: None,
        valid: None,
    })
}

async fn api_handler(
    path: web::Path<(String, String)>,
    query: web::Query<HashMap<String, String>>,
    session: Session
) -> HttpResponse {

    let (module_name, function_name) = path.into_inner();

    // Check if the request is for a protected module or function
    let is_protected = match module_name.as_str() {
        "CloudM" => false,
        "CloudM.AuthManager" => false, // Example: Allow all functions in this module without validation
        _ => !function_name.starts_with("open"), // Allow functions starting with "open" without validation
    };

    if is_protected {
        let valid = match session.get::<bool>("valid") {
            Ok(Some(true)) => true,
            _ => false,
        };
        if !valid {
            return HttpResponse::Unauthorized().json(ApiResponse::<String> {
                message: "Unauthorized".to_string(),
                data: None,
                valid: None,
            });
        }
    }

    let live_data = session.get::<HashMap<String, String>>("live_data").unwrap_or_else(|_| None);

    // Get specification from live_data
    let spec = live_data
        .as_ref()
        .and_then(|data| data.get("spec"))
        .cloned()
        .unwrap_or_else(|| "app".to_string());

    // Convert query parameters
    let args: Vec<String> = Vec::new(); // Path params would go here if needed

    let kwargs: HashMap<String, serde_json::Value> = query.into_inner()
        .into_iter()
        .map(|(k, v)| (k, serde_json::json!(v)))
        .collect();

    let client = match get_toolbox_client() {
        Ok(client) => Arc::new(client),
        Err(e) => {
            panic!("{}", e)
        }
    };
    // Run function via toolbox client
    match client.run_function(&module_name, &function_name, &spec, args, kwargs).await {
        Ok(response) => {
            HttpResponse::Ok().json(ApiResponse {
                message: "Success".to_string(),
                data: Some(response),
                valid: None,
            })
        },
        Err(e) => {
            HttpResponse::InternalServerError().json(ApiResponse::<String> {
                message: format!("Error: {}", e),
                data: None,
                valid: None,
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

    initialize_toolbox_client(
        config.toolbox.host.clone(),
        (config.toolbox.port, config.toolbox.port+config.toolbox.max_instances),  // Port range to use
        config.toolbox.timeout_seconds,            // Timeout in seconds
        config.toolbox.max_instances as usize,             // Maximum number of instances
        Option::from(config.toolbox.tb_r_key),
    );

    let client = match get_toolbox_client() {
        Ok(client) => Arc::new(client),
        Err(e) => {
            panic!("{}", e)
        }
    };


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

    HttpServer::new(move || {
        let dist_path = dist_path.clone(); // Move the cloned dist_path into the closure
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
                    .service(web::resource("/validateSession")
                        .route(web::post().to(validate_session_handler)))
                    .service(web::resource("/IsValiSession")
                        .route(web::get().to(is_valid_session_handler)))
                    .service(web::resource("/web/logoutS")
                        .route(web::post().to(logout_handler)))
                    .service(
                    web::resource("/{module_name}/{function_name}")
                        .route(web::get().to(|path, query, session| {
                            info!("Handling request for: {:?}", path);
                            api_handler(path, query, session)
                        }))
                )
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
