//! High-performance, non-blocking networking
//! 
//! Supports:
//! - HTTP/HTTPS clients and servers
//! - TCP sockets
//! - UDP sockets
//! - WebSockets

use crate::error::{BuiltinError, BuiltinResult};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde_json::Value as JsonValue;

// ============================================================================
// HTTP/HTTPS SESSION
// ============================================================================

/// HTTP session with persistent connections and cookie management
#[derive(Debug, Clone)]
pub struct HttpSession {
    base_url: String,
    client: reqwest::Client,
    headers: HashMap<String, String>,
    cookies_file: Option<String>,
}

impl HttpSession {
    pub fn new(
        base_url: String,
        headers: HashMap<String, String>,
        cookies_file: Option<String>,
    ) -> BuiltinResult<Self> {
        let mut client_builder = reqwest::Client::builder()
            .cookie_store(true)
            .timeout(std::time::Duration::from_secs(30));
        
        // Add default headers
        let mut header_map = reqwest::header::HeaderMap::new();
        for (key, value) in &headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                header_map.insert(name, val);
            }
        }
        client_builder = client_builder.default_headers(header_map);
        
        let client = client_builder.build()
            .map_err(|e| BuiltinError::Network(e))?;
        
        Ok(Self {
            base_url,
            client,
            headers,
            cookies_file,
        })
    }
    
    /// Send HTTP request
    pub async fn request(
        &self,
        url: String,
        method: String,
        data: Option<JsonValue>,
    ) -> BuiltinResult<HttpResponse> {
        let full_url = if url.starts_with("http") {
            url
        } else {
            format!("{}{}", self.base_url.trim_end_matches('/'), url)
        };
        
        let method = method.to_uppercase();
        let mut request = match method.as_str() {
            "GET" => self.client.get(&full_url),
            "POST" => self.client.post(&full_url),
            "PUT" => self.client.put(&full_url),
            "DELETE" => self.client.delete(&full_url),
            "PATCH" => self.client.patch(&full_url),
            "HEAD" => self.client.head(&full_url),
            _ => return Err(BuiltinError::InvalidArgument(
                format!("Unsupported HTTP method: {}", method)
            )),
        };
        
        // Add JSON body if provided
        if let Some(json_data) = data {
            request = request.json(&json_data);
        }
        
        let response = request.send().await?;
        
        let status = response.status().as_u16();
        let headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        
        let body = response.text().await?;
        
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

// ============================================================================
// TCP SERVER
// ============================================================================

/// TCP/UDP server handle
#[derive(Debug)]
pub struct ServerHandle {
    pub server_type: String,
    pub host: String,
    pub port: u16,
    pub shutdown_tx: tokio::sync::mpsc::Sender<()>,
}

/// Create TCP server
pub async fn create_tcp_server(
    host: String,
    port: u16,
    on_connect: Arc<dyn Fn(String, String) + Send + Sync>,
    on_disconnect: Arc<dyn Fn(String) + Send + Sync>,
    on_message: Arc<dyn Fn(String, String) + Send + Sync>,
) -> BuiltinResult<ServerHandle> {
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).await?;
    
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    
    // Spawn server task
    tokio::spawn(async move {
        loop {
            tokio::select! {
                Ok((stream, addr)) = listener.accept() => {
                    let client_addr = addr.to_string();
                    on_connect(client_addr.clone(), String::new());
                    
                    let on_msg = on_message.clone();
                    let on_disc = on_disconnect.clone();
                    let client_addr_clone = client_addr.clone();
                    
                    tokio::spawn(async move {
                        if let Err(e) = handle_tcp_client(stream, client_addr_clone.clone(), on_msg).await {
                            eprintln!("Client error: {}", e);
                        }
                        on_disc(client_addr_clone);
                    });
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
    });
    
    Ok(ServerHandle {
        server_type: "tcp".to_string(),
        host,
        port,
        shutdown_tx,
    })
}

async fn handle_tcp_client(
    mut stream: TcpStream,
    client_addr: String,
    on_message: Arc<dyn Fn(String, String) + Send + Sync>,
) -> BuiltinResult<()> {
    let mut buffer = vec![0u8; 4096];
    
    loop {
        let n = stream.read(&mut buffer).await?;
        if n == 0 {
            break; // Connection closed
        }
        
        let message = String::from_utf8_lossy(&buffer[..n]).to_string();
        on_message(client_addr.clone(), message);
    }
    
    Ok(())
}

// ============================================================================
// UDP SERVER
// ============================================================================

/// Create UDP server
pub async fn create_udp_server(
    host: String,
    port: u16,
    on_message: Arc<dyn Fn(String, String) + Send + Sync>,
) -> BuiltinResult<ServerHandle> {
    let addr = format!("{}:{}", host, port);
    let socket = UdpSocket::bind(&addr).await?;
    
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    
    tokio::spawn(async move {
        let mut buffer = vec![0u8; 4096];
        
        loop {
            tokio::select! {
                Ok((n, addr)) = socket.recv_from(&mut buffer) => {
                    let message = String::from_utf8_lossy(&buffer[..n]).to_string();
                    on_message(addr.to_string(), message);
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
    });
    
    Ok(ServerHandle {
        server_type: "udp".to_string(),
        host,
        port,
        shutdown_tx,
    })
}

// ============================================================================
// TCP CLIENT
// ============================================================================

/// TCP client connection
#[derive(Debug)]
pub struct TcpClient {
    stream: Arc<tokio::sync::Mutex<TcpStream>>,
    remote_addr: String,
}

impl TcpClient {
    pub async fn connect(host: String, port: u16) -> BuiltinResult<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr).await?;
        let remote_addr = stream.peer_addr()?.to_string();
        
        Ok(Self {
            stream: Arc::new(tokio::sync::Mutex::new(stream)),
            remote_addr,
        })
    }
    
    pub async fn send(&self, message: String) -> BuiltinResult<()> {
        let mut stream = self.stream.lock().await;
        stream.write_all(message.as_bytes()).await?;
        stream.flush().await?;
        Ok(())
    }
    
    pub async fn receive(&self) -> BuiltinResult<String> {
        let mut stream = self.stream.lock().await;
        let mut buffer = vec![0u8; 4096];
        let n = stream.read(&mut buffer).await?;
        
        if n == 0 {
            return Err(BuiltinError::Runtime("Connection closed".to_string()));
        }
        
        Ok(String::from_utf8_lossy(&buffer[..n]).to_string())
    }
}

// ============================================================================
// UDP CLIENT
// ============================================================================

/// UDP client
#[derive(Debug)]
pub struct UdpClient {
    socket: Arc<UdpSocket>,
    remote_addr: String,
}

impl UdpClient {
    pub async fn connect(host: String, port: u16) -> BuiltinResult<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        let remote_addr = format!("{}:{}", host, port);
        socket.connect(&remote_addr).await?;
        
        Ok(Self {
            socket: Arc::new(socket),
            remote_addr,
        })
    }
    
    pub async fn send(&self, message: String) -> BuiltinResult<()> {
        self.socket.send(message.as_bytes()).await?;
        Ok(())
    }
    
    pub async fn receive(&self) -> BuiltinResult<String> {
        let mut buffer = vec![0u8; 4096];
        let n = self.socket.recv(&mut buffer).await?;
        Ok(String::from_utf8_lossy(&buffer[..n]).to_string())
    }
}

// ============================================================================
// GLOBAL REGISTRIES
// ============================================================================

use dashmap::DashMap;
use once_cell::sync::Lazy;

pub static TCP_CLIENTS: Lazy<DashMap<String, Arc<TcpClient>>> = Lazy::new(DashMap::new);
pub static UDP_CLIENTS: Lazy<DashMap<String, Arc<UdpClient>>> = Lazy::new(DashMap::new);

