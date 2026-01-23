// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
// toolboxv2/simple-core/src-tauri/src/lib.rs
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod hud_settings;
mod mode_manager;
mod worker_manager;

use std::sync::Mutex;
use tauri::{Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};
use worker_manager::WorkerManager;

// Re-export for use in other modules
use hud_settings::HudSettings;
use mode_manager::{switch_mode_animated, AppMode, ModeManager};

/// Application state containing all managers
struct AppState {
    worker_manager: Mutex<WorkerManager>,
    mode_manager: Mutex<ModeManager>,
    hud_settings: Mutex<HudSettings>,
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

/// Get the current API URLs (for frontend configuration)
#[tauri::command]
fn get_api_urls(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let manager = state.worker_manager.lock().map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "api_url": manager.get_api_url(),
        "ws_url": manager.get_ws_url(),
        "is_remote": manager.is_remote(),
    }))
}

/// Update system tray status
#[tauri::command]
fn update_tray_status(status: String) -> Result<(), String> {
    log::info!("[Tray] Status update: {}", status);
    Ok(())
}

/// Save settings to secure storage
#[tauri::command]
async fn save_settings(settings: serde_json::Value) -> Result<(), String> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| "Could not find config directory".to_string())?;
    let settings_path = config_dir.join("toolbox").join("settings.json");

    // Create directory if it doesn't exist
    if let Some(parent) = settings_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let settings_str = serde_json::to_string_pretty(&settings)
        .map_err(|e| e.to_string())?;
    std::fs::write(&settings_path, settings_str).map_err(|e| e.to_string())?;

    log::info!("[Settings] Saved to {:?}", settings_path);
    Ok(())
}

/// Load settings from storage
#[tauri::command]
async fn load_settings() -> Result<serde_json::Value, String> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| "Could not find config directory".to_string())?;
    let settings_path = config_dir.join("toolbox").join("settings.json");

    if settings_path.exists() {
        let settings_str = std::fs::read_to_string(&settings_path)
            .map_err(|e| e.to_string())?;
        let settings: serde_json::Value = serde_json::from_str(&settings_str)
            .map_err(|e| e.to_string())?;
        Ok(settings)
    } else {
        // Return empty object if no settings file exists
        Ok(serde_json::json!({}))
    }
}

// ============================================================================
// Mode Management Commands (Desktop only)
// ============================================================================

/// Switch between App and HUD mode (Desktop only)
#[tauri::command]
async fn switch_mode(
    mode: String,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let target = AppMode::from_str(&mode)
        .ok_or_else(|| format!("Invalid mode: {}. Use 'app' or 'hud'", mode))?;

    // Extract current mode and clone settings
    let (current, mut hud_settings) = {
        let mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        let settings = state.hud_settings.lock().map_err(|e| e.to_string())?;
        (mode_manager.get_current_mode(), settings.clone())
    };

    // Check if already animating
    {
        let mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        if mode_manager.is_animating() {
            return Err("Animation already running".to_string());
        }
    }

    // Set animating flag
    {
        let mut mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        mode_manager.set_animating(true);
    }

    // Perform the animation (modifies hud_settings if saving app state)
    let result = switch_mode_animated(target, current, &app, &mut hud_settings).await;

    // Update state after animation
    {
        let mut mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        mode_manager.set_animating(false);
        if result.is_ok() {
            mode_manager.set_current_mode(target);
            // Update hud_settings in state with any changes (like saved app state)
            if let Ok(mut settings) = state.hud_settings.lock() {
                *settings = hud_settings;
            }
        }
    }

    result?;
    Ok(format!("Switched to {} mode", mode))
}

/// Get current mode
#[tauri::command]
fn get_current_mode(state: State<'_, AppState>) -> Result<String, String> {
    let mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
    Ok(mode_manager.get_current_mode().as_str().to_string())
}

/// Toggle between App and HUD mode
#[tauri::command]
async fn toggle_mode(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    // Extract data and clone settings
    let (target, current, mut hud_settings) = {
        let mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        let settings = state.hud_settings.lock().map_err(|e| e.to_string())?;
        (
            mode_manager.get_toggle_target(),
            mode_manager.get_current_mode(),
            settings.clone(),
        )
    };

    // Check if already animating
    {
        let mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        if mode_manager.is_animating() {
            return Err("Animation already running".to_string());
        }
    }

    // Set animating flag
    {
        let mut mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        mode_manager.set_animating(true);
    }

    // Perform the animation
    let result = switch_mode_animated(target, current, &app, &mut hud_settings).await;

    // Update state after animation
    {
        let mut mode_manager = state.mode_manager.lock().map_err(|e| e.to_string())?;
        mode_manager.set_animating(false);
        if result.is_ok() {
            mode_manager.set_current_mode(target);
            // Update hud_settings in state
            if let Ok(mut settings) = state.hud_settings.lock() {
                *settings = hud_settings;
            }
        }
    }

    result?;
    Ok(format!("Toggled to {} mode", target.as_str()))
}

/// Get HUD settings
#[tauri::command]
fn get_hud_settings(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let settings = state.hud_settings.lock().map_err(|e| e.to_string())?;
    serde_json::to_value(&*settings).map_err(|e| e.to_string())
}

/// Save HUD position (from drag)
#[tauri::command]
async fn save_hud_position(
    x: i32,
    y: i32,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut settings = state.hud_settings.lock().map_err(|e| e.to_string())?;
    settings.set_position(x, y)?;
    log::info!("[HUD] Position saved: ({}, {})", x, y);
    Ok(())
}

/// Set HUD opacity (live update, 0.1-1.0)
#[tauri::command]
async fn set_hud_opacity(
    opacity: f32,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut settings = state.hud_settings.lock().map_err(|e| e.to_string())?;
    settings.set_opacity(opacity)?;

    // Emit event to HUD window to update CSS opacity
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    if let Some(hud_window) = app.get_webview_window("hud") {
        hud_window
            .emit("hud-opacity-changed", settings.opacity)
            .map_err(|e| format!("Failed to emit opacity event: {}", e))?;
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    let _ = app;

    Ok(())
}

/// Check if running on mobile
#[tauri::command]
fn is_mobile() -> bool {
    cfg!(any(target_os = "android", target_os = "ios"))
}

/// Check if HUD mode is available (Desktop only)
#[tauri::command]
fn is_hud_available() -> bool {
    cfg!(not(any(target_os = "android", target_os = "ios")))
}

// ============================================================================
// System Tray Setup (Desktop only)
// ============================================================================

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn setup_system_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder};

    let open_app = MenuItem::with_id(app, "open_app", "🚀 Open App", true, None::<&str>)?;
    let app_mode = MenuItem::with_id(app, "app_mode", "📺 App Mode", true, None::<&str>)?;
    let hud_mode = MenuItem::with_id(app, "hud_mode", "🎯 HUD Mode", true, None::<&str>)?;
    let separator = MenuItem::with_id(app, "sep", "─────────", false, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "❌ Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&open_app, &app_mode, &hud_mode, &separator, &quit])?;

    let _tray = TrayIconBuilder::new()
        .icon(app.default_window_icon().unwrap().clone())
        .menu(&menu)
        .tooltip("SimpleCore - ToolBoxV2")
        .on_menu_event(move |app, event| {
            match event.id.as_ref() {
                "quit" => {
                    log::info!("[Tray] Quit requested");
                    app.exit(0);
                }
                "open_app" => {
                    log::info!("[Tray] Open App requested");
                    // Try to show existing window or create new one
                    if let Some(main_window) = app.get_webview_window("main") {
                        let _ = main_window.show();
                        let _ = main_window.set_focus();
                    } else {
                        // Create main window if it doesn't exist (minimized start)
                        let mut window_builder = WebviewWindowBuilder::new(
                            app,
                            "main",
                            WebviewUrl::App("index.html".into()),
                        )
                        .title("SimpleCore - ToolBoxV2")
                        .inner_size(1200.0, 800.0)
                        .min_inner_size(800.0, 600.0)
                        .center()
                        .decorations(true)
                        .resizable(true);

                        #[cfg(target_os = "windows")]
                        {
                            window_builder = window_builder.use_https_scheme(true);
                        }

                        match window_builder.build() {
                            Ok(window) => {
                                log::info!("[Tray] Main window created");
                                let _ = window.set_focus();
                            }
                            Err(e) => log::error!("[Tray] Failed to create main window: {}", e),
                        }
                    }
                }
                "app_mode" => {
                    log::info!("[Tray] App mode requested");
                    let app_handle = app.clone();
                    tauri::async_runtime::spawn(async move {
                        let state = app_handle.state::<AppState>();

                        let (current, mut hud_settings, is_animating) = {
                            let mode_manager = match state.mode_manager.lock() {
                                Ok(m) => m,
                                Err(e) => {
                                    log::error!("[Tray] Lock error: {}", e);
                                    return;
                                }
                            };
                            let settings = match state.hud_settings.lock() {
                                Ok(s) => s.clone(),
                                Err(e) => {
                                    log::error!("[Tray] Lock error: {}", e);
                                    return;
                                }
                            };
                            (mode_manager.get_current_mode(), settings, mode_manager.is_animating())
                        };

                        if is_animating {
                            log::warn!("[Tray] Animation already running");
                            return;
                        }

                        if let Ok(mut mm) = state.mode_manager.lock() {
                            mm.set_animating(true);
                        }

                        let result = switch_mode_animated(AppMode::App, current, &app_handle, &mut hud_settings).await;

                        if let Ok(mut mm) = state.mode_manager.lock() {
                            mm.set_animating(false);
                            if result.is_ok() {
                                mm.set_current_mode(AppMode::App);
                                if let Ok(mut settings) = state.hud_settings.lock() {
                                    *settings = hud_settings;
                                }
                            }
                        }

                        if let Err(e) = result {
                            log::error!("[Tray] Switch failed: {}", e);
                        }
                    });
                }
                "hud_mode" => {
                    log::info!("[Tray] HUD mode requested");
                    let app_handle = app.clone();
                    tauri::async_runtime::spawn(async move {
                        let state = app_handle.state::<AppState>();

                        let (current, mut hud_settings, is_animating) = {
                            let mode_manager = match state.mode_manager.lock() {
                                Ok(m) => m,
                                Err(e) => {
                                    log::error!("[Tray] Lock error: {}", e);
                                    return;
                                }
                            };
                            let settings = match state.hud_settings.lock() {
                                Ok(s) => s.clone(),
                                Err(e) => {
                                    log::error!("[Tray] Lock error: {}", e);
                                    return;
                                }
                            };
                            (mode_manager.get_current_mode(), settings, mode_manager.is_animating())
                        };

                        if is_animating {
                            log::warn!("[Tray] Animation already running");
                            return;
                        }

                        if let Ok(mut mm) = state.mode_manager.lock() {
                            mm.set_animating(true);
                        }

                        let result = switch_mode_animated(AppMode::Hud, current, &app_handle, &mut hud_settings).await;

                        if let Ok(mut mm) = state.mode_manager.lock() {
                            mm.set_animating(false);
                            if result.is_ok() {
                                mm.set_current_mode(AppMode::Hud);
                                if let Ok(mut settings) = state.hud_settings.lock() {
                                    *settings = hud_settings;
                                }
                            }
                        }

                        if let Err(e) = result {
                            log::error!("[Tray] Switch failed: {}", e);
                        }
                    });
                }
                _ => {}
            }
        })
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(main_window) = app.get_webview_window("main") {
                    let _ = main_window.show();
                    let _ = main_window.set_focus();
                }
            }
        })
        .build(app)?;

    log::info!("[Tray] System tray initialized");
    Ok(())
}

// ============================================================================
// Global Hotkey Setup (Desktop only)
// ============================================================================

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn setup_global_hotkey(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

    let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
    let app_handle = app.handle().clone();

    app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, _event| {
        log::info!("[Hotkey] Ctrl+Shift+Space pressed");

        let handle = app_handle.clone();
        tauri::async_runtime::spawn(async move {
            let state = handle.state::<AppState>();

            let (target, current, mut hud_settings, is_animating) = {
                let mode_manager = match state.mode_manager.lock() {
                    Ok(m) => m,
                    Err(e) => {
                        log::error!("[Hotkey] Lock error: {}", e);
                        return;
                    }
                };
                let settings = match state.hud_settings.lock() {
                    Ok(s) => s.clone(),
                    Err(e) => {
                        log::error!("[Hotkey] Lock error: {}", e);
                        return;
                    }
                };
                (
                    mode_manager.get_toggle_target(),
                    mode_manager.get_current_mode(),
                    settings,
                    mode_manager.is_animating(),
                )
            };

            if is_animating {
                log::warn!("[Hotkey] Animation already running");
                return;
            }

            if let Ok(mut mm) = state.mode_manager.lock() {
                mm.set_animating(true);
            }

            let result = switch_mode_animated(target, current, &handle, &mut hud_settings).await;

            if let Ok(mut mm) = state.mode_manager.lock() {
                mm.set_animating(false);
                if result.is_ok() {
                    mm.set_current_mode(target);
                    if let Ok(mut settings) = state.hud_settings.lock() {
                        *settings = hud_settings;
                    }
                }
            }

            if let Err(e) = result {
                log::error!("[Hotkey] Mode switch failed: {}", e);
            }
        });
    })?;

    app.global_shortcut().register(shortcut)?;
    log::info!("[Hotkey] Ctrl+Shift+Space registered for mode toggle");
    Ok(())
}

// ============================================================================
// Startup Mode Detection
// ============================================================================

/// Check if app was started in minimized/silent mode (e.g., from autostart)
fn is_minimized_start() -> bool {
    std::env::args().any(|arg| arg == "--minimized" || arg == "--silent" || arg == "--background")
}

// ============================================================================
// Main Application Entry
// ============================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let worker_manager = WorkerManager::new();
    let start_minimized = is_minimized_start();

    if start_minimized {
        log::info!("[Setup] Starting in minimized/tray-only mode");
    }

    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build());

    // Add autostart plugin for desktop platforms (Windows, macOS, Linux)
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        use tauri_plugin_autostart::MacosLauncher;
        // Register with --minimized argument so autostart launches in tray-only mode
        // MacosLauncher::LaunchAgent is used on macOS, ignored on Windows/Linux
        builder = builder.plugin(
            tauri_plugin_autostart::init(
                MacosLauncher::LaunchAgent,
                Some(vec!["--minimized"]),
            )
        );
    }

    builder
        .manage(AppState {
            worker_manager: Mutex::new(worker_manager),
            mode_manager: Mutex::new(ModeManager::new()),
            hud_settings: Mutex::new(HudSettings::load()),
        })
        .invoke_handler(tauri::generate_handler![
            // Existing commands
            greet,
            start_worker,
            stop_worker,
            get_worker_status,
            set_api_endpoint,
            get_data_paths,
            check_worker_health,
            get_api_urls,
            update_tray_status,
            save_settings,
            load_settings,
            // Mode management commands
            switch_mode,
            get_current_mode,
            toggle_mode,
            get_hud_settings,
            save_hud_position,
            set_hud_opacity,
            is_mobile,
            is_hud_available,
            // Autostart commands
            is_autostart_enabled,
            set_autostart,
            is_started_minimized,
        ])
        .setup(move |app| {
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            {
                // Only create visible windows if NOT started minimized
                if !start_minimized {
                    // Create main window
                    let mut window_builder = WebviewWindowBuilder::new(
                        app,
                        "main",
                        WebviewUrl::App("index.html".into()),
                    )
                    .title("SimpleCore - ToolBoxV2")
                    .inner_size(1200.0, 800.0)
                    .min_inner_size(800.0, 600.0)
                    .center()
                    .decorations(true)
                    .resizable(true);

                    #[cfg(target_os = "windows")]
                    {
                        window_builder = window_builder.use_https_scheme(true);
                    }

                    match window_builder.build() {
                        Ok(_) => log::info!("[Setup] Main window created"),
                        Err(e) => log::error!("[Setup] Failed to create main window: {}", e),
                    }

                    // Create HUD window (hidden initially)
                    let hud_settings = HudSettings::load();
                    let hud_builder = WebviewWindowBuilder::new(
                        app,
                        "hud",
                        WebviewUrl::App("hud.html".into()),
                    )
                    .title("SimpleCore HUD")
                    .inner_size(hud_settings.width as f64, hud_settings.height as f64)
                    .decorations(false)
                    .transparent(true)
                    .always_on_top(true)
                    .skip_taskbar(true)
                    .visible(false);

                    match hud_builder.build() {
                        Ok(_) => log::info!("[Setup] HUD window created (hidden)"),
                        Err(e) => log::error!("[Setup] Failed to create HUD window: {}", e),
                    }
                } else {
                    log::info!("[Setup] Minimized start - no windows created, tray-only mode");
                }

                // Always setup system tray (needed for minimized mode)
                if let Err(e) = setup_system_tray(app) {
                    log::error!("[Setup] Failed to setup system tray: {}", e);
                }

                // Always setup global hotkey
                if let Err(e) = setup_global_hotkey(app) {
                    log::error!("[Setup] Failed to setup global hotkey: {}", e);
                }

                // Always auto-start worker (needed for background operation)
                let state = app.state::<AppState>();
                let mut manager = state.worker_manager.lock().unwrap();
                manager.set_app_handle(app.handle().clone());

                if let Err(e) = manager.start() {
                    log::warn!("[Setup] Failed to auto-start worker: {} (will use remote)", e);
                }
                drop(manager);
            }

            #[cfg(any(target_os = "android", target_os = "ios"))]
            {
                log::info!("[Setup] Mobile platform detected - no HUD mode");
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<AppState>();
                if let Ok(mut manager) = state.worker_manager.lock() {
                    let _ = manager.stop();
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ============================================================================
// Autostart Commands
// ============================================================================

/// Check if autostart is enabled (Desktop only: Windows, macOS, Linux)
#[tauri::command]
#[cfg(not(any(target_os = "android", target_os = "ios")))]
async fn is_autostart_enabled(app: tauri::AppHandle) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch()
        .is_enabled()
        .map_err(|e| format!("Failed to check autostart: {}", e))
}

#[tauri::command]
#[cfg(any(target_os = "android", target_os = "ios"))]
async fn is_autostart_enabled(_app: tauri::AppHandle) -> Result<bool, String> {
    Ok(false) // Not supported on mobile platforms
}

/// Enable or disable autostart (Desktop only: Windows, macOS, Linux)
#[tauri::command]
#[cfg(not(any(target_os = "android", target_os = "ios")))]
async fn set_autostart(app: tauri::AppHandle, enabled: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let autostart = app.autolaunch();

    if enabled {
        autostart.enable().map_err(|e| format!("Failed to enable autostart: {}", e))
    } else {
        autostart.disable().map_err(|e| format!("Failed to disable autostart: {}", e))
    }
}

#[tauri::command]
#[cfg(any(target_os = "android", target_os = "ios"))]
async fn set_autostart(_app: tauri::AppHandle, _enabled: bool) -> Result<(), String> {
    Err("Autostart not supported on mobile platforms".to_string())
}

/// Check if app was started in minimized mode
#[tauri::command]
fn is_started_minimized() -> bool {
    is_minimized_start()
}
