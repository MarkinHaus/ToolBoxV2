//! Worker Manager for Tauri Desktop App
//! Manages the Python worker sidecar process for local backend operations.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::AppHandle;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri_plugin_shell::process::CommandChild;

const DEFAULT_HTTP_PORT: u16 = 5000;
const DEFAULT_WS_PORT: u16 = 5001;

#[derive(Debug, Clone, PartialEq)]
pub enum ApiEndpoint {
    Local,
    Remote,
    HomeServer(String),
}

impl Default for ApiEndpoint {
    fn default() -> Self { ApiEndpoint::Local }
}

pub struct WorkerManager {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    child_process: Option<CommandChild>,
    app_handle: Option<AppHandle>,
    http_port: u16,
    ws_port: u16,
    endpoint: ApiEndpoint,
    running: AtomicBool,
    data_dir: Option<std::path::PathBuf>,
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
            running: AtomicBool::new(false),
            data_dir,
        }
    }

    pub fn set_app_handle(&mut self, handle: AppHandle) {
        self.app_handle = Some(handle);
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    pub fn start(&mut self) -> Result<(), String> {
        use tauri_plugin_shell::ShellExt;
        if self.running.load(Ordering::SeqCst) { return Ok(()); }

        let app = self.app_handle.as_ref().ok_or("App handle not set")?;
        let sidecar = app.shell()
            .sidecar("tb-worker")
            .map_err(|e| format!("Failed to create sidecar: {}", e))?
            .args(["--http-port", &self.http_port.to_string(),
                   "--ws-port", &self.ws_port.to_string(), "--mode", "tauri"]);

        let (mut rx, child) = sidecar.spawn()
            .map_err(|e| format!("Failed to spawn worker: {}", e))?;

        self.child_process = Some(child);
        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();
        tauri::async_runtime::spawn(async move {
            use tauri_plugin_shell::process::CommandEvent;
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(line) => log::info!("[Worker] {}", String::from_utf8_lossy(&line)),
                    CommandEvent::Stderr(line) => log::warn!("[Worker] {}", String::from_utf8_lossy(&line)),
                    CommandEvent::Terminated(p) => { log::info!("[Worker] Terminated: {:?}", p.code); running.store(false, Ordering::SeqCst); break; }
                    _ => {}
                }
            }
        });
        log::info!("Worker started on HTTP:{} WS:{}", self.http_port, self.ws_port);
        Ok(())
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    pub fn start(&mut self) -> Result<(), String> {
        self.endpoint = ApiEndpoint::Remote;
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
        json!({
            "running": self.running.load(Ordering::SeqCst),
            "http_port": self.http_port, "ws_port": self.ws_port,
            "endpoint": match &self.endpoint {
                ApiEndpoint::Local => "local", ApiEndpoint::Remote => "remote",
                ApiEndpoint::HomeServer(url) => url.as_str(),
            },
            "http_url": format!("http://localhost:{}", self.http_port),
            "ws_url": format!("ws://localhost:{}", self.ws_port),
        })
    }

    pub fn set_endpoint(&mut self, endpoint: &str) {
        self.endpoint = match endpoint {
            "local" => ApiEndpoint::Local, "remote" => ApiEndpoint::Remote,
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
        if !self.running.load(Ordering::SeqCst) { return false; }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        { std::net::TcpStream::connect_timeout(&format!("127.0.0.1:{}", self.http_port).parse().unwrap(), std::time::Duration::from_millis(500)).is_ok() }
        #[cfg(any(target_os = "android", target_os = "ios"))]
        { false }
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
    }

    #[test]
    fn test_set_endpoint_local() {
        let mut manager = WorkerManager::new();
        manager.set_endpoint("local");
        assert_eq!(manager.endpoint, ApiEndpoint::Local);
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
    fn test_get_status() {
        let manager = WorkerManager::new();
        let status = manager.get_status();
        assert_eq!(status["running"], false);
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
    fn test_is_healthy_when_not_running() {
        let manager = WorkerManager::new();
        assert!(!manager.is_healthy());
    }

    #[test]
    fn test_api_endpoint_default() {
        let endpoint = ApiEndpoint::default();
        assert_eq!(endpoint, ApiEndpoint::Local);
    }
}
