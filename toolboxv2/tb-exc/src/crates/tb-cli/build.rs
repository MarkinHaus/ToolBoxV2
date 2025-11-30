// ✅ PHASE 3.1: Build script to find tb-runtime path at compile time
// This makes the build process robust and independent of directory structure

use std::env;
use std::path::PathBuf;

fn main() {
    // Find the workspace root relative to the 'OUT_DIR'
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // The path from 'OUT_DIR' is typically .../target/{profile}/build/tb-cli-xxxx/out
    // We need to go up several levels to find the workspace root.
    let workspace_root = out_dir.ancestors()
        .find(|p| p.join("Cargo.lock").exists())
        .expect("Could not find workspace root");

    // Calculate path to tb-runtime
    // From workspace root (toolboxv2/tb-exc/src) -> crates/tb-runtime
    let runtime_path = workspace_root.join("crates/tb-runtime");

    if !runtime_path.exists() {
        panic!("tb-runtime directory not found at: {}", runtime_path.display());
    }

    // Set an environment variable that can be used in the code
    println!("cargo:rustc-env=TB_RUNTIME_PATH={}", runtime_path.display());
    
    // Also print for debugging
    println!("cargo:warning=TB_RUNTIME_PATH set to: {}", runtime_path.display());
}

