use dashmap::DashMap;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use tb_core::Result;

/// Central cache manager coordinating all caching strategies
pub struct CacheManager {
    #[allow(dead_code)]
    cache_dir: PathBuf,
    import_cache: Arc<crate::ImportCache>,
    artifact_cache: Arc<crate::ArtifactCache>,

    // In-memory hot cache for frequently accessed items
    hot_cache: DashMap<Arc<String>, Arc<Vec<u8>>>,
    max_hot_cache_mb: usize,
    current_hot_cache_bytes: Arc<RwLock<usize>>,
}

impl CacheManager {
    pub fn new(cache_dir: PathBuf, max_hot_cache_mb: usize) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;

        let import_cache = Arc::new(crate::ImportCache::new(cache_dir.join("imports"))?);
        let artifact_cache = Arc::new(crate::ArtifactCache::new(cache_dir.join("artifacts"))?);

        Ok(Self {
            cache_dir,
            import_cache,
            artifact_cache,
            hot_cache: DashMap::with_capacity(128),
            max_hot_cache_mb,
            current_hot_cache_bytes: Arc::new(RwLock::new(0)),
        })
    }

    /// Get from hot cache (memory) first, then cold cache (disk)
    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        let key_arc = Arc::new(key.to_string());

        // Hot path: memory cache
        if let Some(data) = self.hot_cache.get(&key_arc) {
            return Some(Arc::clone(data.value()));
        }

        None
    }

    /// Put into cache with automatic tier management
    pub fn put(&self, key: String, data: Vec<u8>) {
        let key_arc = Arc::new(key);
        let data_arc = Arc::new(data);
        let data_size = data_arc.len();

        // Check if we have room in hot cache
        let mut current_size = self.current_hot_cache_bytes.write();
        let max_bytes = self.max_hot_cache_mb * 1024 * 1024;

        if *current_size + data_size <= max_bytes {
            self.hot_cache.insert(Arc::clone(&key_arc), Arc::clone(&data_arc));
            *current_size += data_size;
        } else {
            // Evict oldest entries if needed (LRU-like)
            self.evict_from_hot_cache(*current_size + data_size - max_bytes);

            self.hot_cache.insert(Arc::clone(&key_arc), Arc::clone(&data_arc));
            *current_size = self.current_hot_cache_bytes.read().clone() + data_size;
        }
    }

    fn evict_from_hot_cache(&self, bytes_to_free: usize) {
        let mut freed = 0;
        let mut to_remove = Vec::new();

        // Simple FIFO eviction - in production, use LRU
        for entry in self.hot_cache.iter() {
            if freed >= bytes_to_free {
                break;
            }
            freed += entry.value().len();
            to_remove.push(Arc::clone(entry.key()));
        }

        for key in to_remove {
            if let Some((_, data)) = self.hot_cache.remove(&key) {
                let mut current = self.current_hot_cache_bytes.write();
                *current = current.saturating_sub(data.len());
            }
        }
    }

    pub fn import_cache(&self) -> &Arc<crate::ImportCache> {
        &self.import_cache
    }

    pub fn artifact_cache(&self) -> &Arc<crate::ArtifactCache> {
        &self.artifact_cache
    }

    pub fn clear_hot_cache(&self) {
        self.hot_cache.clear();
        *self.current_hot_cache_bytes.write() = 0;
    }

    /// ✅ PHASE 3.3: Clear all caches (import, artifact, hot cache)
    pub fn clear(&self) -> Result<()> {
        self.import_cache.clear()?;
        self.artifact_cache.clear()?;
        self.clear_hot_cache();
        Ok(())
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hot_cache_entries: self.hot_cache.len(),
            hot_cache_bytes: *self.current_hot_cache_bytes.read(),
            max_hot_cache_bytes: self.max_hot_cache_mb * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hot_cache_entries: usize,
    pub hot_cache_bytes: usize,
    pub max_hot_cache_bytes: usize,
}

