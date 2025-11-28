// Prevents additional console window on Windows in release, DO NOT REMOVE!!
// toolboxv2/simple-core/src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    simple_core_lib::run()
}
