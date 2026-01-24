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

    // Animate with configurable parameters
    let (steps, delay_ms) = hud_settings.get_animation_params();
    let delay = Duration::from_millis(delay_ms as u64);

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

    // Animate with configurable parameters
    let (steps, delay_ms) = hud_settings.get_animation_params();
    let delay = Duration::from_millis(delay_ms as u64);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_mode_as_str() {
        assert_eq!(AppMode::App.as_str(), "app");
        assert_eq!(AppMode::Hud.as_str(), "hud");
    }

    #[test]
    fn test_app_mode_from_str() {
        assert_eq!(AppMode::from_str("app"), Some(AppMode::App));
        assert_eq!(AppMode::from_str("App"), Some(AppMode::App));
        assert_eq!(AppMode::from_str("APP"), Some(AppMode::App));
        assert_eq!(AppMode::from_str("hud"), Some(AppMode::Hud));
        assert_eq!(AppMode::from_str("Hud"), Some(AppMode::Hud));
        assert_eq!(AppMode::from_str("HUD"), Some(AppMode::Hud));
        assert_eq!(AppMode::from_str("invalid"), None);
        assert_eq!(AppMode::from_str(""), None);
    }

    #[test]
    fn test_app_mode_default() {
        assert_eq!(AppMode::default(), AppMode::App);
    }

    #[test]
    fn test_mode_manager_new() {
        let manager = ModeManager::new();
        assert_eq!(manager.get_current_mode(), AppMode::App);
        assert!(!manager.is_animating());
    }

    #[test]
    fn test_mode_manager_default() {
        let manager = ModeManager::default();
        assert_eq!(manager.get_current_mode(), AppMode::App);
        assert!(!manager.is_animating());
    }

    #[test]
    fn test_mode_manager_set_current_mode() {
        let mut manager = ModeManager::new();

        manager.set_current_mode(AppMode::Hud);
        assert_eq!(manager.get_current_mode(), AppMode::Hud);

        manager.set_current_mode(AppMode::App);
        assert_eq!(manager.get_current_mode(), AppMode::App);
    }

    #[test]
    fn test_mode_manager_set_animating() {
        let mut manager = ModeManager::new();

        manager.set_animating(true);
        assert!(manager.is_animating());

        manager.set_animating(false);
        assert!(!manager.is_animating());
    }

    #[test]
    fn test_mode_manager_get_toggle_target() {
        let mut manager = ModeManager::new();

        // Starting in App mode, toggle target should be Hud
        assert_eq!(manager.get_toggle_target(), AppMode::Hud);

        // Switch to Hud mode, toggle target should be App
        manager.set_current_mode(AppMode::Hud);
        assert_eq!(manager.get_toggle_target(), AppMode::App);
    }

    #[test]
    fn test_lerp() {
        // Test start value
        assert!((lerp(0.0, 100.0, 0.0) - 0.0).abs() < 0.001);

        // Test end value
        assert!((lerp(0.0, 100.0, 1.0) - 100.0).abs() < 0.001);

        // Test midpoint
        assert!((lerp(0.0, 100.0, 0.5) - 50.0).abs() < 0.001);

        // Test quarter
        assert!((lerp(0.0, 100.0, 0.25) - 25.0).abs() < 0.001);

        // Test with negative values
        assert!((lerp(-100.0, 100.0, 0.5) - 0.0).abs() < 0.001);

        // Test with same start and end
        assert!((lerp(50.0, 50.0, 0.5) - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_app_mode_equality() {
        assert_eq!(AppMode::App, AppMode::App);
        assert_eq!(AppMode::Hud, AppMode::Hud);
        assert_ne!(AppMode::App, AppMode::Hud);
    }

    #[test]
    fn test_app_mode_clone() {
        let mode = AppMode::Hud;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_app_mode_copy() {
        let mode = AppMode::App;
        let copied: AppMode = mode; // Copy trait
        assert_eq!(mode, copied);
    }
}
