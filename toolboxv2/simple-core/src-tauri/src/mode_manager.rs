//! Mode Manager - Handles App/HUD mode switching with animation (Desktop only)

use std::time::Duration;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize};

use crate::hud_settings::HudSettings;

/// Application display modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    App,
    Hud,
}

impl AppMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            AppMode::App => "app",
            AppMode::Hud => "hud",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "app" => Some(AppMode::App),
            "hud" => Some(AppMode::Hud),
            _ => None,
        }
    }
}

impl Default for AppMode {
    fn default() -> Self {
        AppMode::App
    }
}

/// Manages mode switching between App and HUD modes
pub struct ModeManager {
    current_mode: AppMode,
    animation_running: bool,
}

impl Default for ModeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModeManager {
    pub fn new() -> Self {
        Self {
            current_mode: AppMode::App,
            animation_running: false,
        }
    }

    pub fn get_current_mode(&self) -> AppMode {
        self.current_mode
    }

    pub fn is_animating(&self) -> bool {
        self.animation_running
    }

    pub fn set_current_mode(&mut self, mode: AppMode) {
        self.current_mode = mode;
    }

    pub fn set_animating(&mut self, running: bool) {
        self.animation_running = running;
    }

    pub fn get_toggle_target(&self) -> AppMode {
        match self.current_mode {
            AppMode::App => AppMode::Hud,
            AppMode::Hud => AppMode::App,
        }
    }
}

/// Switch mode with morph animation (Desktop only)
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub async fn switch_mode_animated(
    target: AppMode,
    current: AppMode,
    app: &AppHandle,
    hud_settings: &mut HudSettings,
) -> Result<(), String> {
    if current == target {
        log::info!("[Mode] Already in {} mode", target.as_str());
        return Ok(());
    }

    log::info!("[Mode] Switching from {} to {}", current.as_str(), target.as_str());

    let result = match target {
        AppMode::Hud => animate_to_hud(app, hud_settings).await,
        AppMode::App => animate_to_app(app, hud_settings).await,
    };

    if result.is_ok() {
        log::info!("[Mode] Successfully switched to {} mode", target.as_str());
    }

    result
}

/// Animate from App to HUD mode
#[cfg(not(any(target_os = "android", target_os = "ios")))]
async fn animate_to_hud(app: &AppHandle, hud_settings: &mut HudSettings) -> Result<(), String> {
    let main_window = app.get_webview_window("main").ok_or("Main window not found")?;
    let hud_window = app.get_webview_window("hud").ok_or("HUD window not found")?;

    // Save current app position/size
    let start_pos = main_window.outer_position().map_err(|e| e.to_string())?;
    let start_size = main_window.outer_size().map_err(|e| e.to_string())?;

    hud_settings.save_app_state(start_pos.x, start_pos.y, start_size.width, start_size.height)?;

    // Target: saved HUD position or current position
    let (target_x, target_y) = hud_settings.get_position();
    let target_w = hud_settings.width;
    let target_h = hud_settings.height;

    log::info!("[Mode] HUD target position: ({}, {}) {}x{}", target_x, target_y, target_w, target_h);

    // Animate
    let steps = 20u32;
    let delay = Duration::from_millis(15);

    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let ease_t = 1.0 - (1.0 - t).powi(2);

        let x = lerp(start_pos.x as f64, target_x as f64, ease_t) as i32;
        let y = lerp(start_pos.y as f64, target_y as f64, ease_t) as i32;
        let w = lerp(start_size.width as f64, target_w as f64, ease_t).max(100.0) as u32;
        let h = lerp(start_size.height as f64, target_h as f64, ease_t).max(100.0) as u32;

        let _ = main_window.set_position(PhysicalPosition::new(x, y));
        let _ = main_window.set_size(PhysicalSize::new(w, h));

        tokio::time::sleep(delay).await;
    }

    // Switch windows
    main_window.hide().map_err(|e| e.to_string())?;
    hud_window.set_position(PhysicalPosition::new(target_x, target_y)).map_err(|e| e.to_string())?;
    hud_window.set_size(PhysicalSize::new(target_w, target_h)).map_err(|e| e.to_string())?;
    hud_window.show().map_err(|e| e.to_string())?;
    hud_window.set_focus().map_err(|e| e.to_string())?;

    Ok(())
}

/// Animate from HUD to App mode
#[cfg(not(any(target_os = "android", target_os = "ios")))]
async fn animate_to_app(app: &AppHandle, hud_settings: &HudSettings) -> Result<(), String> {
    let main_window = app.get_webview_window("main").ok_or("Main window not found")?;
    let hud_window = app.get_webview_window("hud").ok_or("HUD window not found")?;

    let start_pos = hud_window.outer_position().map_err(|e| e.to_string())?;
    let start_size = hud_window.outer_size().map_err(|e| e.to_string())?;

    let saved_state = hud_settings.get_app_state();
    let target_x = saved_state.x;
    let target_y = saved_state.y;
    let target_w = saved_state.width as i32;
    let target_h = saved_state.height as i32;

    log::info!("[Mode] Restoring app to: ({}, {}) {}x{}", target_x, target_y, target_w, target_h);

    // Position main window at HUD location
    main_window.set_position(PhysicalPosition::new(start_pos.x, start_pos.y)).map_err(|e| e.to_string())?;
    main_window.set_size(PhysicalSize::new(start_size.width, start_size.height)).map_err(|e| e.to_string())?;

    // Switch windows
    hud_window.hide().map_err(|e| e.to_string())?;
    main_window.show().map_err(|e| e.to_string())?;

    // Animate
    let steps = 20u32;
    let delay = Duration::from_millis(15);

    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let ease_t = 1.0 - (1.0 - t).powi(2);

        let x = lerp(start_pos.x as f64, target_x as f64, ease_t) as i32;
        let y = lerp(start_pos.y as f64, target_y as f64, ease_t) as i32;
        let w = lerp(start_size.width as f64, target_w as f64, ease_t).max(100.0) as u32;
        let h = lerp(start_size.height as f64, target_h as f64, ease_t).max(100.0) as u32;

        let _ = main_window.set_position(PhysicalPosition::new(x, y));
        let _ = main_window.set_size(PhysicalSize::new(w, h));

        tokio::time::sleep(delay).await;
    }

    main_window.set_focus().map_err(|e| e.to_string())?;
    Ok(())
}

/// Mobile: Mode switching not available
#[cfg(any(target_os = "android", target_os = "ios"))]
pub async fn switch_mode_animated(
    _target: AppMode,
    _current: AppMode,
    _app: &tauri::AppHandle,
    _hud_settings: &mut HudSettings,
) -> Result<(), String> {
    Err("HUD mode not available on mobile".to_string())
}

fn lerp(start: f64, end: f64, t: f64) -> f64 {
    start + (end - start) * t
}
