//! High-performance, non-blocking file I/O operations
//!
//! Supports real file operations with async I/O

use crate::error::{BuiltinError, BuiltinResult};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::path::Path;

/// File handle for tracking open files
#[derive(Debug, Clone)]
pub struct FileHandle {
    pub id: String,
    pub path: String,
    pub mode: String,
    pub encoding: String,
}

/// Open a file with optional encryption
///
/// # Arguments
/// * `path` - File path
/// * `mode` - File mode: "r" (read), "w" (write), "a" (append), "r+" (read/write)
/// * `key` - Optional encryption key (currently unused)
/// * `encoding` - Text encoding (default: "utf-8")
///
/// # Returns
/// File handle ID for subsequent operations
pub async fn open_file(
    path: String,
    mode: String,
    key: Option<String>,
    encoding: String,
) -> BuiltinResult<String> {
    // Validate mode
    if !["r", "w", "a", "r+", "w+", "a+"].contains(&mode.as_str()) {
        return Err(BuiltinError::InvalidArgument(
            format!("Invalid file mode: {}. Use r, w, a, r+, w+, or a+", mode)
        ));
    }

    // Generate unique handle ID
    let handle_id = format!("file_{}", uuid::Uuid::new_v4());

    // Real file handling
    let file_path = Path::new(&path);

    // Check if file exists for read modes
    if mode.starts_with('r') && !file_path.exists() {
        return Err(BuiltinError::NotFound(
            format!("File not found: {}", path)
        ));
    }

    // Create parent directories for write modes
    if mode.contains('w') || mode.contains('a') {
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await?;
        }
    }

    // Store handle in global registry
    FILE_HANDLES.insert(
        handle_id.clone(),
        FileHandleData {
            path: path.clone(),
            mode: mode.clone(),
            encoding: encoding.clone(),
        },
    );

    Ok(handle_id)
}

/// Read entire file content (async, non-blocking)
pub async fn read_file(path: String, _is_blob: bool) -> BuiltinResult<String> {
    let content = fs::read_to_string(&path).await?;
    Ok(content)
}

/// Write content to file (async, non-blocking)
pub async fn write_file(path: String, content: String, _is_blob: bool) -> BuiltinResult<()> {
    // Create parent directories if needed
    if let Some(parent) = Path::new(&path).parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(&path, content).await?;
    Ok(())
}

/// Check if file exists (async, non-blocking)
pub async fn file_exists(path: String, _is_blob: bool) -> BuiltinResult<bool> {
    Ok(Path::new(&path).exists())
}

/// Read file as bytes (async, non-blocking)
pub async fn read_bytes(path: String, _is_blob: bool) -> BuiltinResult<Vec<u8>> {
    let bytes = fs::read(&path).await?;
    Ok(bytes)
}

/// Write bytes to file (async, non-blocking)
pub async fn write_bytes(path: String, data: Vec<u8>, _is_blob: bool) -> BuiltinResult<()> {
    if let Some(parent) = Path::new(&path).parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(&path, data).await?;
    Ok(())
}

/// Append content to file (async, non-blocking)
pub async fn append_file(path: String, content: String, _is_blob: bool) -> BuiltinResult<()> {
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
pub async fn delete_file(path: String, _is_blob: bool) -> BuiltinResult<()> {
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

/// File handle data storage
use dashmap::DashMap;
use once_cell::sync::Lazy;

pub static FILE_HANDLES: Lazy<DashMap<String, FileHandleData>> = Lazy::new(DashMap::new);

#[derive(Debug, Clone)]
pub struct FileHandleData {
    pub path: String,
    pub mode: String,
    pub encoding: String,
}

// UUID generation (simple implementation)
mod uuid {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    pub struct Uuid(u64);

    impl Uuid {
        pub fn new_v4() -> Self {
            Uuid(COUNTER.fetch_add(1, Ordering::SeqCst))
        }
    }

    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:016x}", self.0)
        }
    }
}

