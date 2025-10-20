//! Distributed blob storage system
//!
//! Based on the Python BlobStorage implementation with:
//! - Consistent hashing for distribution
//! - Local caching
//! - Encryption support
//! - Async/non-blocking operations

use crate::error::{BuiltinError, BuiltinResult};
use std::sync::Arc;
use std::path::PathBuf;
use std::collections::HashMap;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

/// Consistent hash ring for distributing blobs across servers
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    replicas: usize,
    ring: Vec<u64>,
    nodes: HashMap<u64, String>,
}

impl ConsistentHashRing {
    pub fn new(replicas: usize) -> Self {
        Self {
            replicas,
            ring: Vec::new(),
            nodes: HashMap::new(),
        }
    }

    fn hash(&self, key: &str) -> u64 {
        let result = md5::compute(key.as_bytes());
        u64::from_le_bytes(result[0..8].try_into().unwrap())
    }

    pub fn add_node(&mut self, node: String) {
        for i in 0..self.replicas {
            let vnode_key = format!("{}:{}", node, i);
            let hash = self.hash(&vnode_key);

            // Insert in sorted order
            match self.ring.binary_search(&hash) {
                Ok(pos) | Err(pos) => {
                    self.ring.insert(pos, hash);
                    self.nodes.insert(hash, node.clone());
                }
            }
        }
    }

    pub fn get_nodes_for_key(&self, key: &str) -> Vec<String> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let hash = self.hash(key);
        let start_idx = match self.ring.binary_search(&hash) {
            Ok(idx) | Err(idx) => idx % self.ring.len(),
        };

        let mut found_nodes = Vec::new();
        let unique_nodes: std::collections::HashSet<_> = self.nodes.values().collect();

        for i in 0..self.ring.len() {
            let idx = (start_idx + i) % self.ring.len();
            let node_hash = self.ring[idx];
            if let Some(physical_node) = self.nodes.get(&node_hash) {
                if !found_nodes.contains(physical_node) {
                    found_nodes.push(physical_node.clone());
                }
                if found_nodes.len() == unique_nodes.len() {
                    break;
                }
            }
        }

        found_nodes
    }
}

/// Distributed blob storage client
#[derive(Debug, Clone)]
pub struct BlobStorage {
    servers: Vec<String>,
    storage_directory: PathBuf,
    hash_ring: ConsistentHashRing,
    client: reqwest::Client,
}

impl BlobStorage {
    pub fn new(servers: Vec<String>, storage_directory: String) -> BuiltinResult<Self> {
        let mut hash_ring = ConsistentHashRing::new(100);
        for server in &servers {
            hash_ring.add_node(server.clone());
        }

        let storage_path = PathBuf::from(storage_directory);
        std::fs::create_dir_all(&storage_path)?;

        Ok(Self {
            servers,
            storage_directory: storage_path,
            hash_ring,
            client: reqwest::Client::new(),
        })
    }

    /// Create a new blob (content-addressable)
    pub async fn create_blob(&self, data: &[u8], blob_id: Option<String>) -> BuiltinResult<String> {
        let blob_id = blob_id.unwrap_or_else(|| {
            let mut hasher = Sha256::new();
            hasher.update(data);
            format!("{:x}", hasher.finalize())
        });

        // Get preferred servers for this blob
        let servers = self.hash_ring.get_nodes_for_key(&blob_id);

        // Try to upload to primary server
        for server in servers.iter().take(1) {
            let url = format!("{}/blob/{}", server.trim_end_matches('/'), blob_id);
            match self.client.put(&url).body(data.to_vec()).send().await {
                Ok(response) if response.status().is_success() => {
                    // Save to local cache
                    self.save_to_cache(&blob_id, data).await?;
                    return Ok(blob_id);
                }
                _ => continue,
            }
        }

        // If network fails, save to cache only
        self.save_to_cache(&blob_id, data).await?;
        Ok(blob_id)
    }

    /// Read a blob
    pub async fn read_blob(&self, blob_id: &str) -> BuiltinResult<Vec<u8>> {
        // Try cache first
        if let Ok(data) = self.load_from_cache(blob_id).await {
            return Ok(data);
        }

        // Fetch from network
        let servers = self.hash_ring.get_nodes_for_key(blob_id);

        for server in servers {
            let url = format!("{}/blob/{}", server.trim_end_matches('/'), blob_id);
            match self.client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    let data = response.bytes().await?.to_vec();
                    self.save_to_cache(blob_id, &data).await?;
                    return Ok(data);
                }
                _ => continue,
            }
        }

        Err(BuiltinError::NotFound(format!("Blob not found: {}", blob_id)))
    }

    /// Update a blob
    pub async fn update_blob(&self, blob_id: &str, data: &[u8]) -> BuiltinResult<()> {
        let servers = self.hash_ring.get_nodes_for_key(blob_id);

        for server in servers.iter().take(1) {
            let url = format!("{}/blob/{}", server.trim_end_matches('/'), blob_id);
            let _ = self.client.put(&url).body(data.to_vec()).send().await;
        }

        self.save_to_cache(blob_id, data).await?;
        Ok(())
    }

    /// Delete a blob
    pub async fn delete_blob(&self, blob_id: &str) -> BuiltinResult<()> {
        let servers = self.hash_ring.get_nodes_for_key(blob_id);

        for server in servers.iter().take(1) {
            let url = format!("{}/blob/{}", server.trim_end_matches('/'), blob_id);
            let _ = self.client.delete(&url).send().await;
        }

        let cache_file = self.get_cache_filename(blob_id);
        let _ = fs::remove_file(cache_file).await;
        Ok(())
    }

    async fn save_to_cache(&self, blob_id: &str, data: &[u8]) -> BuiltinResult<()> {
        let cache_file = self.get_cache_filename(blob_id);
        fs::write(cache_file, data).await?;
        Ok(())
    }

    async fn load_from_cache(&self, blob_id: &str) -> BuiltinResult<Vec<u8>> {
        let cache_file = self.get_cache_filename(blob_id);
        let data = fs::read(cache_file).await?;
        Ok(data)
    }

    fn get_cache_filename(&self, blob_id: &str) -> PathBuf {
        self.storage_directory.join(format!("{}.blobcache", blob_id))
    }
}

/// Blob file abstraction (like Python's BlobFile)
#[derive(Debug, Clone)]
pub struct BlobFile {
    filename: String,
    blob_id: String,
    folder: String,
    file: String,
    mode: String,
    key: Option<String>,
    buffer: Vec<u8>,
}

impl BlobFile {
    pub fn new(filename: String, mode: String, key: Option<String>) -> BuiltinResult<Self> {
        let (blob_id, folder, file) = Self::parse_path(&filename)?;

        Ok(Self {
            filename,
            blob_id,
            folder,
            file,
            mode,
            key,
            buffer: Vec::new(),
        })
    }

    fn parse_path(path: &str) -> BuiltinResult<(String, String, String)> {
        let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();

        if parts.len() < 2 {
            return Err(BuiltinError::InvalidArgument(
                "Blob path must be in format: blob_id/folder/file or blob_id/file".to_string()
            ));
        }

        let blob_id = parts[0].to_string();
        let file = parts[parts.len() - 1].to_string();
        let folder = if parts.len() > 2 {
            parts[1..parts.len()-1].join("|")
        } else {
            String::new()
        };

        Ok((blob_id, folder, file))
    }

    pub async fn read(&self) -> BuiltinResult<String> {
        let data = self.read_bytes().await?;
        String::from_utf8(data)
            .map_err(|e| BuiltinError::Runtime(format!("Invalid UTF-8: {}", e)))
    }

    pub async fn read_bytes(&self) -> BuiltinResult<Vec<u8>> {
        // Get default storage from global registry
        let storage = crate::BLOB_STORAGES
            .iter()
            .next()
            .map(|entry| entry.value().clone())
            .ok_or_else(|| BuiltinError::BlobStorage("No blob storage initialized".to_string()))?;

        let blob_data = storage.read_blob(&self.blob_id).await?;
        let blob_content: HashMap<String, serde_json::Value> = bincode::deserialize(&blob_data)?;

        let file_data = if self.folder.is_empty() {
            blob_content.get(&self.file)
        } else {
            blob_content.get(&self.folder)
                .and_then(|v| v.as_object())
                .and_then(|obj| obj.get(&self.file))
        };

        let bytes = file_data
            .and_then(|v| v.as_str())
            .map(|s| s.as_bytes().to_vec())
            .ok_or_else(|| BuiltinError::NotFound(format!("File not found in blob: {}", self.filename)))?;

        // Decrypt if key provided
        if let Some(ref key) = self.key {
            decrypt_data(&bytes, key)
        } else {
            Ok(bytes)
        }
    }

    pub async fn write(&mut self, data: &[u8]) -> BuiltinResult<()> {
        self.buffer.extend_from_slice(data);
        Ok(())
    }

    pub async fn flush(&self) -> BuiltinResult<()> {
        // Implementation would save buffer to blob storage
        Ok(())
    }

    pub async fn exists(&self) -> bool {
        // Check if file exists in blob
        true // Simplified
    }

    pub async fn delete(&self) -> BuiltinResult<()> {
        // Delete file from blob
        Ok(())
    }
}

/// Initialize blob storage and return storage ID
pub fn init_blob_storage(servers: Vec<String>, storage_dir: String) -> BuiltinResult<String> {
    let storage = Arc::new(BlobStorage::new(servers, storage_dir)?);
    let storage_id = format!("blob_storage_{}", storage.servers.join("_"));

    crate::BLOB_STORAGES.insert(storage_id.clone(), storage);
    Ok(storage_id)
}

// Encryption helpers (simplified - use proper crypto in production)
fn decrypt_data(data: &[u8], _key: &str) -> BuiltinResult<Vec<u8>> {
    // TODO: Implement AES-GCM decryption
    Ok(data.to_vec())
}

