// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
// toolboxv2/simple-core/src-tauri/src/lib.rs
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod worker_manager;

use std::sync::Mutex;
use tauri::State;
use worker_manager::WorkerManager;

/// Application state containing the worker manager
struct AppState {
    worker_manager: Mutex<WorkerManager>,
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Start the Python worker process
#[tauri::command]
async fn start_worker(state: State<'_, AppState>) -> Result<String, String> {
    let mut manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    manager.start().map_err(|e| e.to_string())?;
    Ok("Worker started successfully".to_string())
}

/// Stop the Python worker process
#[tauri::command]
async fn stop_worker(state: State<'_, AppState>) -> Result<String, String> {
    let mut manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    manager.stop().map_err(|e| e.to_string())?;
    Ok("Worker stopped successfully".to_string())
}

/// Get the current worker status
#[tauri::command]
fn get_worker_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    Ok(manager.get_status())
}

/// Set the API endpoint (local, remote, or home server)
#[tauri::command]
fn set_api_endpoint(endpoint: String, state: State<'_, AppState>) -> Result<String, String> {
    let mut manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    manager.set_endpoint(&endpoint);
    Ok(format!("API endpoint set to: {}", endpoint))
}

/// Get data paths for user-data-enc and tb-mods
#[tauri::command]
fn get_data_paths(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    Ok(manager.get_data_paths())
}

/// Check if local worker is available (health check)
#[tauri::command]
async fn check_worker_health(state: State<'_, AppState>) -> Result<bool, String> {
    let manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    Ok(manager.is_healthy())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize worker manager with default configuration
    let worker_manager = WorkerManager::new();

    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            worker_manager: Mutex::new(worker_manager),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            start_worker,
            stop_worker,
            get_worker_status,
            set_api_endpoint,
            get_data_paths,
            check_worker_health
        ])
        .setup(|app| {
            // Auto-start worker on desktop platforms
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            {
                let state = app.state::<AppState>();
                if let Ok(mut manager) = state.worker_manager.lock() {
                    // Set app handle for resource resolution
                    manager.set_app_handle(app.handle().clone());

                    // Try to start worker automatically
                    if let Err(e) = manager.start() {
                        eprintln!("Failed to auto-start worker: {}", e);
                        // Not fatal - user can start manually or use remote API
                    }
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // Graceful shutdown on window close
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<AppState>();
                if let Ok(mut manager) = state.worker_manager.lock() {
                    let _ = manager.stop();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
