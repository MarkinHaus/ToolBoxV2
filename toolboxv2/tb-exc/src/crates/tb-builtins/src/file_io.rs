//! High-performance, non-blocking file I/O operations
//!
//! Supports real file operations with async I/O

use crate::error::{BuiltinError, BuiltinResult};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use std::path::Path;

// ✅ PHASE 1.3: FileHandle and open_file removed - no usable functionality
// /// File handle for tracking open files
// #[derive(Debug, Clone)]
// pub struct FileHandle { ... }
// pub async fn open_file(...) -> BuiltinResult<String> { ... }

/// Read entire file content (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn read_file(path: String) -> BuiltinResult<String> {
    let content = fs::read_to_string(&path).await?;
    Ok(content)
}

/// Write content to file (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn write_file(path: String, content: String) -> BuiltinResult<()> {
    // Create parent directories if needed
    if let Some(parent) = Path::new(&path).parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(&path, content).await?;
    Ok(())
}

/// Check if file exists (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn file_exists(path: String) -> BuiltinResult<bool> {
    Ok(Path::new(&path).exists())
}

/// Read file as bytes (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn read_bytes(path: String) -> BuiltinResult<Vec<u8>> {
    let bytes = fs::read(&path).await?;
    Ok(bytes)
}

/// Write bytes to file (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn write_bytes(path: String, data: Vec<u8>) -> BuiltinResult<()> {
    if let Some(parent) = Path::new(&path).parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(&path, data).await?;
    Ok(())
}

/// Append content to file (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn append_file(path: String, content: String) -> BuiltinResult<()> {
    use tokio::fs::OpenOptions;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await?;
    file.write_all(content.as_bytes()).await?;
    Ok(())
}

/// Delete file (async, non-blocking)
/// ✅ PHASE 1.1: Removed _is_blob parameter - blob storage not implemented
pub async fn delete_file(path: String) -> BuiltinResult<()> {
    fs::remove_file(&path).await?;
    Ok(())
}

/// List directory contents (async, non-blocking)
pub async fn list_dir(path: String) -> BuiltinResult<Vec<String>> {
    let mut entries = Vec::new();
    let mut dir = fs::read_dir(&path).await?;

    while let Some(entry) = dir.next_entry().await? {
        if let Ok(name) = entry.file_name().into_string() {
            entries.push(name);
        }
    }

    Ok(entries)
}

/// Create directory (async, non-blocking)
pub async fn create_dir(path: String, recursive: bool) -> BuiltinResult<()> {
    if recursive {
        fs::create_dir_all(&path).await?;
    } else {
        fs::create_dir(&path).await?;
    }
    Ok(())
}

// ✅ PHASE 1.3: FILE_HANDLES and FileHandleData removed - no usable functionality
// /// File handle data storage
// use dashmap::DashMap;
// use once_cell::sync::Lazy;
// pub static FILE_HANDLES: Lazy<DashMap<String, FileHandleData>> = Lazy::new(DashMap::new);
// #[derive(Debug, Clone)]
// pub struct FileHandleData { ... }

// ✅ PHASE 1.4: Custom UUID implementation will be removed in next phase
// mod uuid { ... }

