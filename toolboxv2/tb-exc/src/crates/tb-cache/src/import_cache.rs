use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tb_core::{Program, Result, TBError};

/// Import cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub source_hash: String,
    pub compiled_path: PathBuf,
    pub compiled_at: SystemTime,
    pub dependencies: Vec<PathBuf>,
    pub size_bytes: u64,
}

/// Import cache for compiled modules
/// Uses content-addressable storage with SHA256 hashing
pub struct ImportCache {
    cache_dir: PathBuf,
    manifest: DashMap<PathBuf, CacheEntry>,
    manifest_path: PathBuf,
}

impl ImportCache {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_dir)?;

        let manifest_path = cache_dir.join("manifest.bin");
        let manifest = Self::load_manifest(&manifest_path)?;

        Ok(Self {
            cache_dir,
            manifest,
            manifest_path,
        })
    }

    /// Check if source is cached and valid
    pub fn is_cached(&self, source_path: &Path) -> Result<bool> {
        if let Some(entry) = self.manifest.get(source_path) {
            // Verify source hasn't changed
            let current_hash = Self::hash_file(source_path)?;
            Ok(entry.source_hash == current_hash)
        } else {
            Ok(false)
        }
    }

    /// Get cached compiled module (zero-copy via mmap)
    pub fn get(&self, source_path: &Path) -> Result<Option<Arc<Program>>> {
        if !self.is_cached(source_path)? {
            return Ok(None);
        }

        let entry = self.manifest.get(source_path).unwrap();

        // Use memory-mapped file for zero-copy loading
        let file = fs::File::open(&entry.compiled_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Deserialize with bincode (very fast)
        let program: Program = bincode::deserialize(&mmap)
            .map_err(|e| TBError::CacheError {
                message: format!("Failed to deserialize cache: {}", e),
            })?;

        Ok(Some(Arc::new(program)))
    }

    /// Store compiled module in cache
    pub fn put(&self, source_path: &Path, program: &Program) -> Result<()> {
        let source_hash = Self::hash_file(source_path)?;

        // Generate cache file path from hash (content-addressable)
        let cache_filename = format!("{}.bin", &source_hash[..16]);
        let compiled_path = self.cache_dir.join(cache_filename);

        // Serialize with bincode (zero-copy where possible)
        let serialized = bincode::serialize(program)
            .map_err(|e| TBError::CacheError {
                message: format!("Failed to serialize: {}", e),
            })?;

        fs::write(&compiled_path, &serialized)?;

        let entry = CacheEntry {
            source_hash,
            compiled_path,
            compiled_at: SystemTime::now(),
            dependencies: Vec::new(), // TODO: track dependencies
            size_bytes: serialized.len() as u64,
        };

        self.manifest.insert(source_path.to_path_buf(), entry);
        self.save_manifest()?;

        Ok(())
    }

    /// Invalidate cache entry
    pub fn invalidate(&self, source_path: &Path) -> Result<()> {
        if let Some((_, entry)) = self.manifest.remove(source_path) {
            // Remove cached file
            if entry.compiled_path.exists() {
                fs::remove_file(entry.compiled_path)?;
            }
            self.save_manifest()?;
        }
        Ok(())
    }

    /// Clear entire cache
    pub fn clear(&self) -> Result<()> {
        for entry in self.manifest.iter() {
            let path = &entry.value().compiled_path;
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        self.manifest.clear();
        self.save_manifest()?;
        Ok(())
    }

    /// Hash file content using BLAKE3 (faster than SHA256)
    fn hash_file(path: &Path) -> Result<String> {
        let content = fs::read(path)?;
        let hash = blake3::hash(&content);
        Ok(hash.to_hex().to_string())
    }

    fn load_manifest(path: &Path) -> Result<DashMap<PathBuf, CacheEntry>> {
        if path.exists() {
            let data = fs::read(path)?;
            let manifest: Vec<(PathBuf, CacheEntry)> = bincode::deserialize(&data)
                .map_err(|e| TBError::CacheError {
                    message: format!("Failed to load manifest: {}", e),
                })?;

            Ok(manifest.into_iter().collect())
        } else {
            Ok(DashMap::new())
        }
    }

    fn save_manifest(&self) -> Result<()> {
        let entries: Vec<_> = self.manifest.iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();

        let serialized = bincode::serialize(&entries)
            .map_err(|e| TBError::CacheError {
                message: format!("Failed to serialize manifest: {}", e),
            })?;

        fs::write(&self.manifest_path, serialized)?;
        Ok(())
    }

    pub fn stats(&self) -> ImportCacheStats {
        let total_size: u64 = self.manifest.iter()
            .map(|e| e.value().size_bytes)
            .sum();

        ImportCacheStats {
            entries: self.manifest.len(),
            total_bytes: total_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImportCacheStats {
    pub entries: usize,
    pub total_bytes: u64,
}

