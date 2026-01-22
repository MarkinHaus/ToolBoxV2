//! Worker Manager for Tauri Desktop App
//! Manages the Python worker sidecar process for local backend operations.
//!
//! The worker can run in three modes:
//! 1. Local sidecar (bundled tb-worker binary)
//! 2. Remote API (simplecore.app)
//! 3. Home server (user-configured URL)
//!
//! If the sidecar is not available, the app automatically falls back to remote API.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::AppHandle;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri_plugin_shell::process::CommandChild;

const DEFAULT_HTTP_PORT: u16 = 5000;
const DEFAULT_WS_PORT: u16 = 5001;
const REMOTE_API_URL: &str = "https://simplecore.app";
const REMOTE_WS_URL: &str = "wss://simplecore.app";

#[derive(Debug, Clone, PartialEq)]
pub enum ApiEndpoint {
    Local,
    Remote,
    HomeServer(String),
}

impl Default for ApiEndpoint {
    fn default() -> Self { ApiEndpoint::Local }
}

impl ApiEndpoint {
    pub fn get_http_url(&self, port: u16) -> String {
        match self {
            ApiEndpoint::Local => format!("http://localhost:{}", port),
            ApiEndpoint::Remote => format!("{}/api", REMOTE_API_URL),
            ApiEndpoint::HomeServer(url) => format!("{}/api", url),
        }
    }

    pub fn get_ws_url(&self, port: u16) -> String {
        match self {
            ApiEndpoint::Local => format!("ws://localhost:{}", port),
            ApiEndpoint::Remote => REMOTE_WS_URL.to_string(),
            ApiEndpoint::HomeServer(url) => url.replace("https://", "wss://").replace("http://", "ws://"),
        }
    }
}

pub struct WorkerManager {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    child_process: Option<CommandChild>,
    app_handle: Option<AppHandle>,
    http_port: u16,
    ws_port: u16,
    endpoint: ApiEndpoint,
    running: Arc<AtomicBool>,
    data_dir: Option<std::path::PathBuf>,
    sidecar_available: bool,
}

impl Default for WorkerManager {
    fn default() -> Self { Self::new() }
}

impl WorkerManager {
    pub fn new() -> Self {
        let data_dir = dirs::data_dir().map(|p| p.join("toolboxv2"));
        WorkerManager {
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            child_process: None,
            app_handle: None,
            http_port: DEFAULT_HTTP_PORT,
            ws_port: DEFAULT_WS_PORT,
            endpoint: ApiEndpoint::default(),
            running: Arc::new(AtomicBool::new(false)),
            data_dir,
            sidecar_available: false,
        }
    }

    pub fn set_app_handle(&mut self, handle: AppHandle) {
        self.app_handle = Some(handle);
    }

    /// Check if the sidecar binary is available
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    fn check_sidecar_available(&self) -> bool {
        use tauri_plugin_shell::ShellExt;
        if let Some(app) = &self.app_handle {
            // Try to create the sidecar command - this will fail if binary not found
            app.shell().sidecar("tb-worker").is_ok()
        } else {
            false
        }
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    pub fn start(&mut self) -> Result<(), String> {
        use tauri_plugin_shell::ShellExt;
        if self.running.load(Ordering::SeqCst) { return Ok(()); }

        let app = self.app_handle.as_ref().ok_or("App handle not set")?;

        // Check if sidecar is available
        let sidecar_result = app.shell().sidecar("tb-worker");

        match sidecar_result {
            Ok(sidecar) => {
                let sidecar = sidecar.args([
                    "--http-port", &self.http_port.to_string(),
                    "--ws-port", &self.ws_port.to_string(),
                    "--mode", "tauri"
                ]);

                match sidecar.spawn() {
                    Ok((mut rx, child)) => {
                        self.child_process = Some(child);
                        self.running.store(true, Ordering::SeqCst);
                        self.sidecar_available = true;
                        self.endpoint = ApiEndpoint::Local;

                        let running = self.running.clone();
                        tauri::async_runtime::spawn(async move {
                            use tauri_plugin_shell::process::CommandEvent;
                            while let Some(event) = rx.recv().await {
                                match event {
                                    CommandEvent::Stdout(line) => {
                                        log::info!("[Worker] {}", String::from_utf8_lossy(&line));
                                    }
                                    CommandEvent::Stderr(line) => {
                                        log::warn!("[Worker] {}", String::from_utf8_lossy(&line));
                                    }
                                    CommandEvent::Terminated(p) => {
                                        log::info!("[Worker] Terminated: {:?}", p.code);
                                        running.store(false, Ordering::SeqCst);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        });
                        log::info!("Worker started on HTTP:{} WS:{}", self.http_port, self.ws_port);
                        Ok(())
                    }
                    Err(e) => {
                        log::warn!("Failed to spawn worker sidecar: {}. Falling back to remote API.", e);
                        self.fallback_to_remote()
                    }
                }
            }
            Err(e) => {
                log::warn!("Worker sidecar not available: {}. Using remote API.", e);
                self.fallback_to_remote()
            }
        }
    }

    /// Fallback to remote API when sidecar is not available
    fn fallback_to_remote(&mut self) -> Result<(), String> {
        self.endpoint = ApiEndpoint::Remote;
        self.sidecar_available = false;
        self.running.store(false, Ordering::SeqCst);
        log::info!("Using remote API: {}", REMOTE_API_URL);
        Ok(())
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    pub fn start(&mut self) -> Result<(), String> {
        self.endpoint = ApiEndpoint::Remote;
        self.sidecar_available = false;
        log::info!("Mobile platform, using remote API");
        Ok(())
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(child) = self.child_process.take() {
            child.kill().map_err(|e| format!("Failed to kill worker: {}", e))?;
            self.running.store(false, Ordering::SeqCst);
            log::info!("Worker stopped");
        }
        Ok(())
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    pub fn stop(&mut self) -> Result<(), String> { Ok(()) }

    pub fn get_status(&self) -> Value {
        let endpoint_str = match &self.endpoint {
            ApiEndpoint::Local => "local",
            ApiEndpoint::Remote => "remote",
            ApiEndpoint::HomeServer(url) => url.as_str(),
        };

        json!({
            "running": self.running.load(Ordering::SeqCst),
            "sidecar_available": self.sidecar_available,
            "http_port": self.http_port,
            "ws_port": self.ws_port,
            "endpoint": endpoint_str,
            "http_url": self.endpoint.get_http_url(self.http_port),
            "ws_url": self.endpoint.get_ws_url(self.ws_port),
            "remote_api_url": REMOTE_API_URL,
        })
    }

    pub fn set_endpoint(&mut self, endpoint: &str) {
        self.endpoint = match endpoint {
            "local" => {
                // Only allow local if sidecar is available
                if self.sidecar_available {
                    ApiEndpoint::Local
                } else {
                    log::warn!("Sidecar not available, staying on remote API");
                    ApiEndpoint::Remote
                }
            }
            "remote" => ApiEndpoint::Remote,
            url => ApiEndpoint::HomeServer(url.to_string()),
        };
    }

    pub fn get_data_paths(&self) -> Value {
        let base = self.data_dir.clone().unwrap_or_else(|| std::path::PathBuf::from("."));
        json!({
            "base_dir": base.to_string_lossy(),
            "user_data_enc": base.join("user-data-enc").to_string_lossy(),
            "tb_mods": base.join("tb-mods").to_string_lossy(),
        })
    }

    pub fn is_healthy(&self) -> bool {
        match &self.endpoint {
            ApiEndpoint::Local => {
                if !self.running.load(Ordering::SeqCst) {
                    return false;
                }
                #[cfg(not(any(target_os = "android", target_os = "ios")))]
                {
                    std::net::TcpStream::connect_timeout(
                        &format!("127.0.0.1:{}", self.http_port).parse().unwrap(),
                        std::time::Duration::from_millis(500)
                    ).is_ok()
                }
                #[cfg(any(target_os = "android", target_os = "ios"))]
                { false }
            }
            ApiEndpoint::Remote | ApiEndpoint::HomeServer(_) => {
                // For remote endpoints, we assume they're healthy
                // The frontend will handle connection errors
                true
            }
        }
    }

    /// Check if using remote API (no local worker)
    pub fn is_remote(&self) -> bool {
        matches!(self.endpoint, ApiEndpoint::Remote | ApiEndpoint::HomeServer(_))
    }

    /// Get the current API base URL
    pub fn get_api_url(&self) -> String {
        self.endpoint.get_http_url(self.http_port)
    }

    /// Get the current WebSocket URL
    pub fn get_ws_url(&self) -> String {
        self.endpoint.get_ws_url(self.ws_port)
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) { let _ = self.stop(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_manager_new() {
        let manager = WorkerManager::new();
        assert!(!manager.running.load(Ordering::SeqCst));
        assert_eq!(manager.http_port, DEFAULT_HTTP_PORT);
        assert_eq!(manager.ws_port, DEFAULT_WS_PORT);
        assert_eq!(manager.endpoint, ApiEndpoint::Local);
        assert!(!manager.sidecar_available);
    }

    #[test]
    fn test_set_endpoint_remote() {
        let mut manager = WorkerManager::new();
        manager.set_endpoint("remote");
        assert_eq!(manager.endpoint, ApiEndpoint::Remote);
    }

    #[test]
    fn test_set_endpoint_home_server() {
        let mut manager = WorkerManager::new();
        manager.set_endpoint("https://my-server.com");
        assert_eq!(manager.endpoint, ApiEndpoint::HomeServer("https://my-server.com".to_string()));
    }

    #[test]
    fn test_set_endpoint_local_without_sidecar() {
        let mut manager = WorkerManager::new();
        manager.sidecar_available = false;
        manager.set_endpoint("local");
        // Should fallback to remote since sidecar not available
        assert_eq!(manager.endpoint, ApiEndpoint::Remote);
    }

    #[test]
    fn test_set_endpoint_local_with_sidecar() {
        let mut manager = WorkerManager::new();
        manager.sidecar_available = true;
        manager.set_endpoint("local");
        assert_eq!(manager.endpoint, ApiEndpoint::Local);
    }

    #[test]
    fn test_get_status() {
        let manager = WorkerManager::new();
        let status = manager.get_status();
        assert_eq!(status["running"], false);
        assert_eq!(status["sidecar_available"], false);
        assert_eq!(status["http_port"], DEFAULT_HTTP_PORT);
        assert_eq!(status["ws_port"], DEFAULT_WS_PORT);
        assert_eq!(status["endpoint"], "local");
    }

    #[test]
    fn test_get_data_paths() {
        let manager = WorkerManager::new();
        let paths = manager.get_data_paths();
        assert!(paths["base_dir"].as_str().is_some());
        assert!(paths["user_data_enc"].as_str().is_some());
        assert!(paths["tb_mods"].as_str().is_some());
    }

    #[test]
    fn test_is_healthy_when_not_running_local() {
        let manager = WorkerManager::new();
        assert!(!manager.is_healthy());
    }

    #[test]
    fn test_is_healthy_when_remote() {
        let mut manager = WorkerManager::new();
        manager.endpoint = ApiEndpoint::Remote;
        // Remote endpoints are assumed healthy
        assert!(manager.is_healthy());
    }

    #[test]
    fn test_api_endpoint_default() {
        let endpoint = ApiEndpoint::default();
        assert_eq!(endpoint, ApiEndpoint::Local);
    }

    #[test]
    fn test_api_endpoint_urls() {
        assert_eq!(
            ApiEndpoint::Local.get_http_url(5000),
            "http://localhost:5000"
        );
        assert_eq!(
            ApiEndpoint::Remote.get_http_url(5000),
            format!("{}/api", REMOTE_API_URL)
        );
        assert_eq!(
            ApiEndpoint::HomeServer("https://my.server".to_string()).get_http_url(5000),
            "https://my.server/api"
        );
    }

    #[test]
    fn test_is_remote() {
        let mut manager = WorkerManager::new();
        assert!(!manager.is_remote());

        manager.endpoint = ApiEndpoint::Remote;
        assert!(manager.is_remote());

        manager.endpoint = ApiEndpoint::HomeServer("https://test.com".to_string());
        assert!(manager.is_remote());
    }
}
