use actix_web::{web, App,HttpRequest, HttpServer, HttpResponse, middleware};
use actix_files as fs;
use actix_session::{Session, SessionMiddleware, storage::CookieSessionStore};
use actix_web::cookie::{Key};
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

lazy_static! {
    static ref TOOLBOX_CLIENT: Mutex<Option<ToolboxClient>> = Mutex::new(None);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolboxRequest {
    pub name: String,
    pub args: Vec<serde_json::Value>,
    pub kwargs: HashMap<String, serde_json::Value>,
    pub identifier: Option<String>,
    pub claim: Option<String>,
    pub key: Option<String>,
}

#[derive(Debug, Clone)]
struct Connection {
    stream: Arc<Mutex<Option<tokio::net::TcpStream>>>,
    last_used: Instant,
    is_validated: bool,
    in_use: bool,  // Track if connection is currently being used
}

#[derive(Debug, Clone)]
struct ToolboxInstance {
    id: String,
    host: String,
    port: u16,
    process: Option<Arc<Mutex<std::process::Child>>>,
    last_used: Instant,
    is_alive: bool,
    active_connections: usize,
}

#[derive(Debug, Clone)]
pub struct ToolboxClient {
    instances: Arc<Mutex<Vec<ToolboxInstance>>>,
    connection_pool: Arc<Mutex<HashMap<String, Connection>>>,
    auth_key: Option<String>,
    default_host: String,
    default_port_range: (u16, u16),
    timeout: Duration,
    max_instances: usize,
    heartbeat_interval: Duration,
    used_ports: Arc<Mutex<HashSet<u16>>>,
    package_size: usize,
    maintenance_last_run: Arc<Mutex<Instant>>,
}

// A wrapper for a connection that returns itself to the pool when dropped
struct PooledConnection {
    stream: tokio::net::TcpStream,
    key: String,
    client: ToolboxClient,
    instance_id: String,
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        // Mark the connection as available when dropped
        self.client.mark_connection_available(&self.key, &self.instance_id);
    }
}

impl std::ops::Deref for PooledConnection {
    type Target = tokio::net::TcpStream;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl std::ops::DerefMut for PooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stream
    }
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

impl ToolboxClient {
    pub fn new(
        host: String,
        port_range: (u16, u16),
        timeout_seconds: u64,
        max_instances: usize,
        auth_key: Option<String>,
    ) -> Self {
        ToolboxClient {
            instances: Arc::new(Mutex::new(Vec::new())),
            connection_pool: Arc::new(Mutex::new(HashMap::new())),
            auth_key,
            default_host: host,
            default_port_range: port_range,
            timeout: Duration::from_secs(timeout_seconds),
            max_instances,
            heartbeat_interval: Duration::from_secs(30),
            used_ports: Arc::new(Mutex::new(HashSet::new())),
            package_size: 8192,
            maintenance_last_run: Arc::new(Mutex::new(Instant::now())),
        }
    }

    // Mark a connection as available in the pool
    fn mark_connection_available(&self, key: &str, instance_id: &str) {
        // Update the connection in the pool
        {
            let mut pool = self.connection_pool.lock().unwrap();
            if let Some(conn) = pool.get_mut(key) {
                conn.in_use = false;
                conn.last_used = Instant::now();
            }
        }

        // Update instance stats
        {
            let mut instances = self.instances.lock().unwrap();
            if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
                if instance.active_connections > 0 {
                    instance.active_connections -= 1;
                }
            }
        }
    }

    // On-demand maintenance - called periodically during regular operations
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
            // Run maintenance tasks
            self.check_instance_health().await;
            self.ensure_minimum_instances(1).await;
            self.cleanup_connection_pool().await;
        }
    }

    // Clean up unused connections
    async fn cleanup_connection_pool(&self) {
        let mut to_remove = Vec::new();

        // Safely get keys from connection pool
        let keys = {
            let connection_pool = self.connection_pool.lock().unwrap();
            connection_pool.keys().cloned().collect::<Vec<_>>()
        };

        for key in keys {
            let should_remove = {
                let connection_pool = self.connection_pool.lock().unwrap();
                if let Some(conn) = connection_pool.get(&key) {
                    if conn.in_use {
                        false // Don't remove connections in use
                    } else if conn.last_used.elapsed() > Duration::from_secs(300) {
                        true  // Remove if unused for 5 minutes
                    } else {
                        // Check if the connection is actually valid
                        let is_valid = if let Some(stream) = &*conn.stream.lock().unwrap() {
                            stream.peer_addr().is_ok()
                        } else {
                            false
                        };

                        !is_valid
                    }
                } else {
                    false
                }
            };

            if should_remove {
                to_remove.push(key);
            }
        }

        // Remove dead connections
        if !to_remove.is_empty() {
            let mut connection_pool = self.connection_pool.lock().unwrap();
            for key in to_remove {
                connection_pool.remove(&key);
                debug!("Removed dead connection: {}", key);
            }
        }
    }

    // Check health of instances and remove dead ones
    async fn check_instance_health(&self) {
        // Get a snapshot of all instances to check
        let instances_to_check = {
            let instances = self.instances.lock().unwrap();
            instances.clone()
        };

        let mut dead_instances = Vec::new();

        for instance in &instances_to_check {
            let mut is_alive = true;

            // Check if process is running
            if let Some(process) = &instance.process {
                let process_running = {
                    let mut process_guard = process.lock().unwrap();
                    match process_guard.try_wait() {
                        Ok(Some(_)) => false,
                        Ok(None) => true,
                        Err(_) => false,
                    }
                };

                if !process_running {
                    is_alive = false;
                }
            }

            // Check connection via socket if it's been a while since last check
            if is_alive && instance.last_used.elapsed() > self.heartbeat_interval {
                let conn_key = format!("{}:{}", instance.host, instance.port);
                is_alive = self.check_connection(&conn_key).await;
            }

            // Mark for removal if dead
            if !is_alive {
                dead_instances.push(instance.id.clone());
            }
        }

        // Remove dead instances
        if !dead_instances.is_empty() {
            let mut instances = self.instances.lock().unwrap();
            let mut used_ports = self.used_ports.lock().unwrap();
            let mut connection_pool = self.connection_pool.lock().unwrap();

            for id in &dead_instances {
                if let Some(instance) = instances.iter().find(|i| &i.id == id) {
                    // Remove connections for this instance
                    let conn_key = format!("{}:{}", instance.host, instance.port);
                    connection_pool.remove(&conn_key);

                    // Free up the port
                    used_ports.remove(&instance.port);
                    debug!("Removed dead instance {} on port {}", instance.id, instance.port);
                }
            }

            instances.retain(|i| !dead_instances.contains(&i.id));
        }
    }

    // Verify if a connection is alive
    async fn check_connection(&self, conn_key: &str) -> bool {
        // First check if we have a connection in the pool
        let conn_exists = {
            let pool = self.connection_pool.lock().unwrap();
            if let Some(conn) = pool.get(conn_key) {
                if let Some(stream) = &*conn.stream.lock().unwrap() {
                    stream.peer_addr().is_ok()
                } else {
                    false
                }
            } else {
                false
            }
        };

        if conn_exists {
            return true;
        }

        // Otherwise try to connect
        tokio::net::TcpStream::connect(conn_key).await.is_ok()
    }

    // Ensure we have at least min_instances running
    async fn ensure_minimum_instances(&self, min_instances: usize) {
        let instances_count = {
            let instances = self.instances.lock().unwrap();
            instances.len()
        };

        if instances_count < min_instances {
            for _ in instances_count..min_instances {
                if let Err(e) = self.start_new_instance().await {
                    warn!("Failed to start new instance: {}", e);
                    sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    // Checkout a connection from the pool or create a new one
    async fn checkout_connection(&self, instance_id: &str) -> Result<PooledConnection, ToolboxError> {
        // Get instance details
        let (host, port) = {
            let instances = self.instances.lock().unwrap();
            let instance = instances.iter()
                .find(|i| i.id == instance_id)
                .ok_or_else(|| ToolboxError::InstanceNotFound(instance_id.to_string()))?;

            (instance.host.clone(), instance.port)
        };

        let conn_key = format!("{}:{}", host, port);

        // Try to get an existing idle connection
        let mut stream_available = false;
        let mut need_validation = false;

        {
            let mut pool = self.connection_pool.lock().unwrap();
            if let Some(conn) = pool.get_mut(&conn_key) {
                if !conn.in_use && conn.is_validated {
                    // We have a valid idle connection, check if it's still good
                    if let Some(ref mut stream) = *conn.stream.lock().unwrap() {
                        if stream.peer_addr().is_ok() {
                            // Mark connection as in-use
                            conn.in_use = true;
                            conn.last_used = Instant::now();
                            stream_available = true;
                        }
                    }
                }

                // If this instance hasn't been validated yet
                need_validation = !conn.is_validated;
            }
        }

        // Update instance stats
        {
            let mut instances = self.instances.lock().unwrap();
            if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
                instance.active_connections += 1;
                instance.last_used = Instant::now();
            }
        }

        // Connect to the instance
        let addr = format!("{}:{}", host, port);
        let stream = tokio::net::TcpStream::connect(&addr).await
            .map_err(|e| ToolboxError::IoError(e))?;

        stream.set_nodelay(true)
            .map_err(|e| ToolboxError::IoError(e))?;

        // If we don't have a connection in the pool yet, or need to replace it
        if !stream_available {
            // Store this connection in the pool for future use
            let mut pool = self.connection_pool.lock().unwrap();

            // Create a new stream for the pool
            let pool_stream = tokio::net::TcpStream::connect(&addr).await
                .map_err(|e| ToolboxError::IoError(e))?;

            pool_stream.set_nodelay(true)
                .map_err(|e| ToolboxError::IoError(e))?;

            // Update or create pool entry
            pool.insert(conn_key.clone(), Connection {
                stream: Arc::new(Mutex::new(Some(pool_stream))),
                last_used: Instant::now(),
                is_validated: !need_validation,  // Only set to true if it's already validated
                in_use: true,
            });
        }

        // If this instance needs validation
        if need_validation {
            // Validate connection with a new stream
            self.validate_with_new_stream(instance_id, &host, port).await?;

            // Mark as validated in the pool
            let mut pool = self.connection_pool.lock().unwrap();
            if let Some(conn) = pool.get_mut(&conn_key) {
                conn.is_validated = true;
            }
        }

        Ok(PooledConnection {
            stream,
            key: conn_key,
            client: self.clone(),
            instance_id: instance_id.to_string(),
        })
    }

    // Check if an instance has been validated already
    async fn is_instance_validated(&self, instance_id: &str) -> bool {
        let instances = self.instances.lock().unwrap();
        if let Some(_) = instances.iter().find(|i| i.id == instance_id) {
            return true;
        }
        false
    }

    // Start a new toolbox instance
    pub async fn start_new_instance(&self) -> Result<(), ToolboxError> {
        // Find available port
        let available_port = {
            let instances = self.instances.lock().unwrap();
            let used_ports = self.used_ports.lock().unwrap();

            if instances.len() >= self.max_instances {
                return Err(ToolboxError::MaxInstancesReached);
            }

            (self.default_port_range.0..=self.default_port_range.1)
                .find(|&port| !used_ports.contains(&port))
                .ok_or(ToolboxError::NoAvailablePorts)?
        };

        // Check if something is already running on this port
        let result = tokio::net::TcpStream::connect(
            format!("{}:{}", self.default_host, available_port)
        ).await;

        if result.is_ok() {
            // Mark port as used
            {
                let mut used_ports = self.used_ports.lock().unwrap();
                used_ports.insert(available_port);
            }

            // Try to use existing service
            let instance_id = format!("found_instance_{}", Uuid::new_v4());
            let new_instance = ToolboxInstance {
                id: instance_id.clone(),
                host: self.default_host.clone(),
                port: available_port,
                process: None,
                last_used: Instant::now(),
                is_alive: true,
                active_connections: 0,
            };

            // Add to instances
            {
                let mut instances = self.instances.lock().unwrap();
                instances.push(new_instance);
            }

            // Validate this instance
            if self.validate_instance(&instance_id).await.is_ok() {
                info!("Using existing service on port {}", available_port);
                return Ok(());
            } else {
                // Remove invalid instance
                let mut instances = self.instances.lock().unwrap();
                instances.retain(|i| i.id != instance_id);
                return Err(ToolboxError::InvalidInstance(available_port));
            }
        }

        // Mark port as used
        {
            let mut used_ports = self.used_ports.lock().unwrap();
            used_ports.insert(available_port);
        }

        // Start the toolbox process
        let child = Command::new("tb")
            .arg("-bgr")
            .arg("--sysPrint")
            .arg("-m")
            .arg("bgws")
            .arg("-p")
            .arg(available_port.to_string())
            .spawn()
            .map_err(|e| ToolboxError::IoError(e))?;

        let instance_id = format!("instance_{}", Uuid::new_v4());

        // Create instance
        let new_instance = ToolboxInstance {
            id: instance_id.clone(),
            host: self.default_host.clone(),
            port: available_port,
            process: Some(Arc::new(Mutex::new(child))),
            last_used: Instant::now(),
            is_alive: true,
            active_connections: 0,
        };

        // Add to instances
        {
            let mut instances = self.instances.lock().unwrap();
            instances.push(new_instance);
        }

        // Wait for server with exponential backoff
        let mut backoff = 500; // Start with 500ms
        let mut attempts = 0;
        let max_attempts = 5;

        while attempts < max_attempts {
            sleep(Duration::from_millis(backoff)).await;
            attempts += 1;

            // Try to connect
            if tokio::net::TcpStream::connect(
                format!("{}:{}", self.default_host, available_port)
            ).await.is_ok() {
                // Validate the instance
                match self.validate_instance(&instance_id).await {
                    Ok(_) => {
                        info!("Started and validated new instance on port {}", available_port);
                        return Ok(());
                    },
                    Err(e) => {
                        self.stop_instance(&instance_id).await.ok();
                        return Err(e);
                    }
                }
            }

            backoff *= 2; // Exponential backoff
        }

        // Failed after all attempts
        self.stop_instance(&instance_id).await.ok();
        Err(ToolboxError::Timeout)
    }

    // Stop a toolbox instance
    pub async fn stop_instance(&self, instance_id: &str) -> Result<(), ToolboxError> {
        // Find and remove the instance
        let instance = {
            let mut instances = self.instances.lock().unwrap();
            let instance_pos = instances.iter().position(|i| i.id == instance_id)
                .ok_or_else(|| ToolboxError::InstanceNotFound(instance_id.to_string()))?;

            let instance = instances.remove(instance_pos);

            let mut used_ports = self.used_ports.lock().unwrap();
            used_ports.remove(&instance.port);

            instance
        };

        // Send exit signal
        if let Ok(mut stream) = tokio::net::TcpStream::connect(
            format!("{}:{}", instance.host, instance.port)
        ).await {
            let _ = stream.write_all(b"e").await;
            let _ = stream.flush().await;
        }

        // Kill the process
        if let Some(process) = instance.process {
            let _ = process.lock().unwrap().kill();
        }

        // Also try the kill command
        let _ = Command::new("tb")
            .arg("-kill")
            .arg("-p")
            .arg(instance.port.to_string())
            .spawn();

        // Remove from connection pool
        let conn_key = format!("{}:{}", instance.host, instance.port);
        {
            let mut pool = self.connection_pool.lock().unwrap();
            pool.remove(&conn_key);
        }

        info!("Stopped instance {} on port {}", instance_id, instance.port);
        Ok(())
    }

    // Validate an instance
    async fn validate_instance(&self, instance_id: &str) -> Result<(), ToolboxError> {
        // Get instance details
        let (host, port) = {
            let instances = self.instances.lock().unwrap();
            let instance = instances.iter()
                .find(|i| i.id == instance_id)
                .ok_or_else(|| ToolboxError::InstanceNotFound(instance_id.to_string()))?;

            (instance.host.clone(), instance.port)
        };

        // Validate with a new stream
        self.validate_with_new_stream(instance_id, &host, port).await
    }

    // Validate an instance with a fresh connection
    async fn validate_with_new_stream(&self, instance_id: &str, host: &str, port: u16) -> Result<(), ToolboxError> {
        // Create a new connection for validation
        let addr = format!("{}:{}", host, port);
        let mut stream = tokio::net::TcpStream::connect(&addr).await
            .map_err(|e| ToolboxError::IoError(e))?;

        stream.set_nodelay(true).map_err(|e| ToolboxError::IoError(e))?;

        // Send initial connection notification with 'j' prefix
        let initial_request = serde_json::json!({
            "identifier": "new_con",
            "data": [null, null]
        });
        let json_str = serde_json::to_string(&initial_request)
            .map_err(|e| ToolboxError::JsonError(e))?;

        let mut data_to_send = vec![b'j'];
        data_to_send.extend_from_slice(json_str.as_bytes());

        self.send_data(&mut stream, &data_to_send).await?;

        // Wait briefly
        sleep(Duration::from_millis(500)).await;

        // Send validation with auth key
        let peer_addr = stream.peer_addr().map_err(|e| ToolboxError::IoError(e))?;
        let formatted_addr = format!("('127.0.0.1', {})", peer_addr.port());

        let validation_request = serde_json::json!({
            "identifier": formatted_addr,
            "key": self.auth_key.clone()
        });
        let json_str = serde_json::to_string(&validation_request)
            .map_err(|e| ToolboxError::JsonError(e))?;

        let mut data_to_send = vec![b'j'];
        data_to_send.extend_from_slice(json_str.as_bytes());

        self.send_data(&mut stream, &data_to_send).await?;

        // Mark instance as validated
        {
            let mut instances = self.instances.lock().unwrap();
            if let Some(instance) = instances.iter_mut().find(|i| i.id == instance_id) {
                instance.last_used = Instant::now();
                instance.is_alive = true;
            }
        }

        Ok(())
    }

    // Simplified data sending with end marker
    async fn send_data(&self, stream: &mut tokio::net::TcpStream, data: &[u8]) -> Result<(), ToolboxError> {
        // Send data in one go if possible
        stream.write_all(data).await.map_err(|e| ToolboxError::IoError(e))?;

        // Send end marker
        let end_marker = vec![b'E'; 10]; // Smaller end marker for efficiency
        stream.write_all(&end_marker).await.map_err(|e| ToolboxError::IoError(e))?;
        stream.flush().await.map_err(|e| ToolboxError::IoError(e))?;

        Ok(())
    }

    // Receive data with optimized buffer handling
    async fn receive_data(&self, stream: &mut tokio::net::TcpStream) -> Result<serde_json::Value, ToolboxError> {
        let mut buffer = Vec::with_capacity(self.package_size);
        let mut data_type = None;

        // First read to get data type and initial data
        let mut chunk = vec![0u8; self.package_size];
        let n = stream.read(&mut chunk).await.map_err(|e| ToolboxError::IoError(e))?;

        if n == 0 {
            return Err(ToolboxError::ConnectionClosed);
        }

        // Set data type from first byte
        data_type = Some(chunk[0]);

        // Process rest of chunk
        let mut end_marker_count = 0;
        for &byte in &chunk[1..n] {
            if byte == b'E' {
                end_marker_count += 1;
                if end_marker_count >= 5 {
                    break;
                }
            } else {
                // If we saw Es but then a non-E, they were part of the data
                for _ in 0..end_marker_count {
                    buffer.push(b'E');
                }
                end_marker_count = 0;
                buffer.push(byte);
            }
        }

        // If we haven't found the end marker yet, read more
        if end_marker_count < 5 {
            let mut more_data = vec![0u8; 1024];
            loop {
                match stream.read(&mut more_data).await {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        for &byte in &more_data[..n] {
                            if byte == b'E' {
                                end_marker_count += 1;
                                if end_marker_count >= 5 {
                                    break;
                                }
                            } else {
                                for _ in 0..end_marker_count {
                                    buffer.push(b'E');
                                }
                                end_marker_count = 0;
                                buffer.push(byte);
                            }
                        }
                        if end_marker_count >= 5 {
                            break;
                        }
                    },
                    Err(e) => return Err(ToolboxError::IoError(e)),
                }
            }
        }

        // Process the data based on type
        match data_type {
            Some(b'j') => {
                // JSON data
                let json_str = String::from_utf8_lossy(&buffer).to_string();
                serde_json::from_str(&json_str).map_err(|e| ToolboxError::JsonError(e))
            },
            Some(b'b') => {
                // Binary data
                Ok(serde_json::json!({
                    "binary_data": true,
                    "data": base64::engine::general_purpose::STANDARD.encode(&buffer),
                    "size": buffer.len(),
                }))
            },
            Some(b'e') => Err(ToolboxError::ConnectionClosed),
            Some(b'k') => Ok(serde_json::json!({"keepalive": true})),
            _ => Err(ToolboxError::UnknownDataType),
        }
    }

    // Get a valid instance with intelligent load balancing
    fn get_valid_instance(&self) -> Pin<Box<dyn Future<Output = Result<String, ToolboxError>> + Send + '_>> {
        Box::pin(async move {
            // Run maintenance if needed
            self.run_maintenance_if_needed().await;

            // Get all alive instances
            let instances = {
                let instances = self.instances.lock().unwrap();
                instances.iter()
                    .filter(|i| i.is_alive)
                    .map(|i| (i.id.clone(), i.active_connections))
                    .collect::<Vec<_>>()
            };

            if instances.is_empty() {
                // No instances, start a new one
                self.start_new_instance().await?;
                return self.get_valid_instance().await; // Boxed recursive call
            }

            // Load balancing: choose instance with least active connections
            let instance_id = instances.iter()
                .min_by_key(|(_, active_conns)| *active_conns)
                .map(|(id, _)| id.clone())
                .unwrap();

            // Check if instance is overloaded
            let needs_new_instance = {
                let instances = self.instances.lock().unwrap();
                let total_instances = instances.len();

                if let Some(instance) = instances.iter().find(|i| i.id == instance_id) {
                    instance.active_connections > 5 && total_instances < self.max_instances
                } else {
                    false
                }
            };

            // Start new instance if needed, but don't wait for it
            if needs_new_instance {
                let client_clone = self.clone();
                tokio::spawn(async move {
                    if let Err(e) = client_clone.start_new_instance().await {
                        warn!("Failed to start additional instance: {}", e);
                    }
                });
            }

            Ok(instance_id)
        })
    }

    // Run a function on a toolbox instance with connection pooling
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

        // Get a valid instance with load balancing
        let instance_id = self.get_valid_instance().await?;

        // Checkout a connection from the pool
        let mut conn = self.checkout_connection(&instance_id).await?;

        // Prepare the request
        let request = ToolboxRequest {
            name: "a_run_any".to_string(),
            args: run_args,
            kwargs: run_kwargs,
            identifier: Some(instance_id.clone()),
            claim: None,
            key: self.auth_key.clone(),
        };

        // Serialize the request
        let json_str = serde_json::to_string(&request)
            .map_err(|e| ToolboxError::JsonError(e))?;

        // Send the request
        let mut data_to_send = vec![b'j'];
        data_to_send.extend_from_slice(json_str.as_bytes());

        self.send_data(&mut conn, &data_to_send).await?;

        // Receive the response
        self.receive_data(&mut conn).await
    }

    // Get information about all current instances
    pub fn get_instances_info(&self) -> Vec<HashMap<String, String>> {
        let instances = self.instances.lock().unwrap();

        let mut result = Vec::new();
        for instance in instances.iter() {
            let mut info = HashMap::new();
            info.insert("id".to_string(), instance.id.clone());
            info.insert("host".to_string(), instance.host.clone());
            info.insert("port".to_string(), instance.port.to_string());
            info.insert("alive".to_string(), instance.is_alive.to_string());
            info.insert("active_connections".to_string(), instance.active_connections.to_string());
            info.insert("last_used".to_string(), format!("{:?}", instance.last_used.elapsed()));

            result.push(info);
        }

        result
    }
}

// A simplified error enum
#[derive(Debug, thiserror::Error)]
pub enum ToolboxError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Connection timeout")]
    Timeout,
    #[error("Connection closed")]
    ConnectionClosed,
    #[error("Unknown data type")]
    UnknownDataType,
    #[error("No available instances")]
    NoAvailableInstances,
    #[error("No available ports")]
    NoAvailablePorts,
    #[error("Instance not found: {0}")]
    InstanceNotFound(String),
    #[error("Invalid instance on port: {0}")]
    InvalidInstance(u16),
    #[error("Maximum instances reached")]
    MaxInstancesReached,
    #[error("Unknown error: {0}")]
    Unknown(String),
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

// Example API response structure for compatibility with your existing code
#[derive(Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub message: String,
    pub data: Option<T>,
    pub valid: Option<bool>,
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
                log::error!("JWT validation error: {:?}", e);
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
    info!("API FOR: {:?} {:?}", live_data, session.get::<HashMap<String, String>>("ip").unwrap_or_else(|_| None));
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
            panic!("{:?}", e)
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
                message: format!("Error: {:?}", e),
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
            panic!("{:?}", e)
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
                            info!("Handling GET request for: {:?}", path);
                            api_handler(path, query, session)
                        }))
                        .route(web::post().to(|path, query, session| {
                            info!("Handling POST request for: {:?}", path);
                            api_handler(path, query, session)
                        }))
                        .route(web::delete().to(|path, query, session| {
                            info!("Handling DEL request for: {:?}", path);
                            api_handler(path, query, session)
                        }))
                        .route(web::put().to(|path, query, session| {
                            info!("Handling PUT request for: {:?}", path);
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
