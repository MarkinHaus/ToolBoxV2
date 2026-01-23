//! HUD Settings - Configuration for HUD window position and appearance (Desktop only)

use serde::{Deserialize, Serialize};

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri::AppHandle;

/// Saved App window state (position and size before switching to HUD)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedAppState {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Default for SavedAppState {
    fn default() -> Self {
        Self {
            x: 100,
            y: 100,
            width: 1200,
            height: 800,
        }
    }
}

/// HUD window configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HudSettings {
    /// Window X position (absolute screen coordinates)
    pub x: i32,
    /// Window Y position (absolute screen coordinates)
    pub y: i32,
    /// Window width in pixels
    pub width: u32,
    /// Window height in pixels
    pub height: u32,
    /// Window opacity (0.1 - 1.0)
    pub opacity: f32,
    /// Saved app window state for restoration
    #[serde(default)]
    pub saved_app_state: Option<SavedAppState>,
}

impl Default for HudSettings {
    fn default() -> Self {
        Self {
            x: 100,
            y: 100,
            width: 380,
            height: 500,
            opacity: 0.92,
            saved_app_state: None,
        }
    }
}

impl HudSettings {
    /// Get the saved position
    pub fn get_position(&self) -> (i32, i32) {
        (self.x, self.y)
    }

    /// Get settings file path
    fn get_settings_path() -> Option<std::path::PathBuf> {
        dirs::config_dir().map(|dir| dir.join("toolbox").join("hud_settings.json"))
    }

    /// Load settings from file
    pub fn load() -> Self {
        let Some(settings_path) = Self::get_settings_path() else {
            log::warn!("[HUD] Could not determine config directory, using defaults");
            return Self::default();
        };

        if !settings_path.exists() {
            log::info!("[HUD] No settings file found, using defaults");
            return Self::default();
        }

        match std::fs::read_to_string(&settings_path) {
            Ok(content) => match serde_json::from_str(&content) {
                Ok(settings) => {
                    log::info!("[HUD] Settings loaded from {:?}", settings_path);
                    settings
                }
                Err(e) => {
                    log::warn!("[HUD] Failed to parse settings: {}, using defaults", e);
                    Self::default()
                }
            },
            Err(e) => {
                log::warn!("[HUD] Failed to read settings: {}, using defaults", e);
                Self::default()
            }
        }
    }

    /// Save settings to file
    pub fn save(&self) -> Result<(), String> {
        let settings_path = Self::get_settings_path()
            .ok_or_else(|| "Could not determine config directory".to_string())?;

        if let Some(parent) = settings_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;

        std::fs::write(&settings_path, content)
            .map_err(|e| format!("Failed to write settings: {}", e))?;

        log::info!("[HUD] Settings saved to {:?}", settings_path);
        Ok(())
    }

    /// Update position and save
    pub fn set_position(&mut self, x: i32, y: i32) -> Result<(), String> {
        self.x = x;
        self.y = y;
        self.save()
    }

    /// Update dimensions and save
    pub fn set_dimensions(&mut self, width: u32, height: u32) -> Result<(), String> {
        self.width = width;
        self.height = height;
        self.save()
    }

    /// Update opacity and save (clamped to 0.1 - 1.0)
    pub fn set_opacity(&mut self, opacity: f32) -> Result<(), String> {
        self.opacity = opacity.clamp(0.1, 1.0);
        self.save()
    }

    /// Save current app window state
    pub fn save_app_state(&mut self, x: i32, y: i32, width: u32, height: u32) -> Result<(), String> {
        self.saved_app_state = Some(SavedAppState { x, y, width, height });
        self.save()
    }

    /// Get saved app state or default
    pub fn get_app_state(&self) -> SavedAppState {
        self.saved_app_state.clone().unwrap_or_default()
    }

    /// Clear saved app state
    pub fn clear_app_state(&mut self) -> Result<(), String> {
        self.saved_app_state = None;
        self.save()
    }
}
