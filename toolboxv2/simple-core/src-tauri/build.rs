// toolboxv2/simple-core/src-tauri/build.rs
fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(
            tauri_build::AppManifest::new().commands(&[
                // Worker management
                "greet",
                "start_worker",
                "stop_worker",
                "get_worker_status",
                "set_api_endpoint",
                "get_data_paths",
                "check_worker_health",
                "get_api_urls",
                "update_tray_status",
                // Settings
                "save_settings",
                "load_settings",
                // Mode management
                "switch_mode",
                "get_current_mode",
                "toggle_mode",
                "get_hud_settings",
                "save_hud_position",
                "set_hud_opacity",
                "is_mobile",
                "is_hud_available",
                // Animation settings
                "set_animation_steps",
                "set_animation_delay",
                // MiniUI app selection
                "set_selected_miniui_app",
                "get_selected_miniui_app",
                // Autostart
                "is_autostart_enabled",
                "set_autostart",
                "is_started_minimized",
            ]),
        ),
    )
    .expect("Failed to build Tauri application");
}
