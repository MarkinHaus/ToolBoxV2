//! HUD Settings - Configuration for HUD window position and appearance (Desktop only)

use serde::{Deserialize, Serialize};

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
    /// Number of animation steps for mode transitions (1-100)
    #[serde(default = "default_animation_steps")]
    pub animation_steps: u32,
    /// Delay between animation steps in milliseconds (5-100)
    #[serde(default = "default_animation_delay")]
    pub animation_delay_ms: u32,
    /// Selected MiniUI app name (persistent)
    #[serde(default)]
    pub selected_miniui_app: Option<String>,
    /// Saved app window state for restoration
    #[serde(default)]
    pub saved_app_state: Option<SavedAppState>,
}

fn default_animation_steps() -> u32 {
    20
}

fn default_animation_delay() -> u32 {
    15
}

impl Default for HudSettings {
    fn default() -> Self {
        Self {
            x: 100,
            y: 100,
            width: 380,
            height: 500,
            opacity: 0.92,
            animation_steps: default_animation_steps(),
            animation_delay_ms: default_animation_delay(),
            selected_miniui_app: None,
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

    /// Set animation steps (clamped to 1-100)
    pub fn set_animation_steps(&mut self, steps: u32) -> Result<(), String> {
        self.animation_steps = steps.clamp(1, 100);
        self.save()
    }

    /// Set animation delay in ms (clamped to 5-100)
    pub fn set_animation_delay(&mut self, delay_ms: u32) -> Result<(), String> {
        self.animation_delay_ms = delay_ms.clamp(5, 100);
        self.save()
    }

    /// Get animation parameters
    pub fn get_animation_params(&self) -> (u32, u32) {
        (self.animation_steps, self.animation_delay_ms)
    }

    /// Set selected MiniUI app
    pub fn set_selected_miniui_app(&mut self, app_name: Option<String>) -> Result<(), String> {
        self.selected_miniui_app = app_name;
        self.save()
    }

    /// Get selected MiniUI app
    pub fn get_selected_miniui_app(&self) -> Option<String> {
        self.selected_miniui_app.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_hud_settings() {
        let settings = HudSettings::default();
        assert_eq!(settings.x, 100);
        assert_eq!(settings.y, 100);
        assert_eq!(settings.width, 380);
        assert_eq!(settings.height, 500);
        assert!((settings.opacity - 0.92).abs() < 0.01);
        assert_eq!(settings.animation_steps, 20);
        assert_eq!(settings.animation_delay_ms, 15);
        assert!(settings.selected_miniui_app.is_none());
        assert!(settings.saved_app_state.is_none());
    }

    #[test]
    fn test_default_saved_app_state() {
        let state = SavedAppState::default();
        assert_eq!(state.x, 100);
        assert_eq!(state.y, 100);
        assert_eq!(state.width, 1200);
        assert_eq!(state.height, 800);
    }

    #[test]
    fn test_get_position() {
        let settings = HudSettings {
            x: 200,
            y: 300,
            ..Default::default()
        };
        assert_eq!(settings.get_position(), (200, 300));
    }

    #[test]
    fn test_get_animation_params() {
        let settings = HudSettings {
            animation_steps: 30,
            animation_delay_ms: 25,
            ..Default::default()
        };
        assert_eq!(settings.get_animation_params(), (30, 25));
    }

    #[test]
    fn test_get_app_state_default() {
        let settings = HudSettings::default();
        let state = settings.get_app_state();
        assert_eq!(state.x, 100);
        assert_eq!(state.y, 100);
        assert_eq!(state.width, 1200);
        assert_eq!(state.height, 800);
    }

    #[test]
    fn test_get_app_state_saved() {
        let settings = HudSettings {
            saved_app_state: Some(SavedAppState {
                x: 50,
                y: 60,
                width: 800,
                height: 600,
            }),
            ..Default::default()
        };
        let state = settings.get_app_state();
        assert_eq!(state.x, 50);
        assert_eq!(state.y, 60);
        assert_eq!(state.width, 800);
        assert_eq!(state.height, 600);
    }

    #[test]
    fn test_miniui_app_selection() {
        let mut settings = HudSettings::default();
        assert!(settings.get_selected_miniui_app().is_none());

        settings.selected_miniui_app = Some("test-app".to_string());
        assert_eq!(settings.get_selected_miniui_app(), Some("test-app".to_string()));
    }

    #[test]
    fn test_opacity_clamping() {
        let mut settings = HudSettings::default();

        // Test lower bound
        settings.opacity = 0.05;
        let clamped = settings.opacity.clamp(0.1, 1.0);
        assert!((clamped - 0.1).abs() < 0.01);

        // Test upper bound
        settings.opacity = 1.5;
        let clamped = settings.opacity.clamp(0.1, 1.0);
        assert!((clamped - 1.0).abs() < 0.01);

        // Test valid value
        settings.opacity = 0.5;
        let clamped = settings.opacity.clamp(0.1, 1.0);
        assert!((clamped - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_animation_steps_clamping() {
        // Test lower bound
        let clamped = 0u32.clamp(1, 100);
        assert_eq!(clamped, 1);

        // Test upper bound
        let clamped = 150u32.clamp(1, 100);
        assert_eq!(clamped, 100);

        // Test valid value
        let clamped = 50u32.clamp(1, 100);
        assert_eq!(clamped, 50);
    }

    #[test]
    fn test_animation_delay_clamping() {
        // Test lower bound
        let clamped = 2u32.clamp(5, 100);
        assert_eq!(clamped, 5);

        // Test upper bound
        let clamped = 200u32.clamp(5, 100);
        assert_eq!(clamped, 100);

        // Test valid value
        let clamped = 50u32.clamp(5, 100);
        assert_eq!(clamped, 50);
    }

    #[test]
    fn test_serialization() {
        let settings = HudSettings {
            x: 150,
            y: 250,
            width: 400,
            height: 600,
            opacity: 0.85,
            animation_steps: 25,
            animation_delay_ms: 20,
            selected_miniui_app: Some("my-app".to_string()),
            saved_app_state: Some(SavedAppState {
                x: 10,
                y: 20,
                width: 1000,
                height: 700,
            }),
        };

        let json = serde_json::to_string(&settings).expect("Failed to serialize");
        let deserialized: HudSettings = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.x, 150);
        assert_eq!(deserialized.y, 250);
        assert_eq!(deserialized.width, 400);
        assert_eq!(deserialized.height, 600);
        assert!((deserialized.opacity - 0.85).abs() < 0.01);
        assert_eq!(deserialized.animation_steps, 25);
        assert_eq!(deserialized.animation_delay_ms, 20);
        assert_eq!(deserialized.selected_miniui_app, Some("my-app".to_string()));

        let state = deserialized.saved_app_state.unwrap();
        assert_eq!(state.x, 10);
        assert_eq!(state.y, 20);
        assert_eq!(state.width, 1000);
        assert_eq!(state.height, 700);
    }

    #[test]
    fn test_deserialization_with_defaults() {
        // Test that missing fields get default values
        let json = r#"{"x": 100, "y": 200, "width": 300, "height": 400, "opacity": 0.9}"#;
        let settings: HudSettings = serde_json::from_str(json).expect("Failed to deserialize");

        assert_eq!(settings.x, 100);
        assert_eq!(settings.y, 200);
        assert_eq!(settings.animation_steps, 20); // default
        assert_eq!(settings.animation_delay_ms, 15); // default
        assert!(settings.selected_miniui_app.is_none()); // default
        assert!(settings.saved_app_state.is_none()); // default
    }
}
