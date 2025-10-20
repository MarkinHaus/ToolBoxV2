use std::path::{Path, PathBuf};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::SystemTime;
use serde::{Serialize, Deserialize};

/// Plugin cache entry containing metadata about a compiled plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCacheEntry {
    /// Hash of the source code
    pub source_hash: u64,
    /// Path to the compiled library
    pub library_path: PathBuf,
    /// Programming language of the plugin
    pub language: String,
    /// Compilation mode (jit or compile)
    pub mode: String,
    /// When the plugin was compiled
    pub compiled_at: SystemTime,
    /// Optional path to source file (for external plugins)
    pub source_file: Option<PathBuf>,
    /// Function signatures exported by this plugin
    pub functions: Vec<FunctionSignature>,
}

/// Function signature for cache validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub param_types: Vec<String>,
    pub return_type: String,
}

/// Plugin compilation cache manager
pub struct PluginCache {
    cache_dir: PathBuf,
    manifest_path: PathBuf,
    entries: dashmap::DashMap<String, PluginCacheEntry>,
}

impl PluginCache {
    /// Create a new plugin cache
    pub fn new() -> std::io::Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("tb-lang")
            .join("plugin-cache");

        std::fs::create_dir_all(&cache_dir)?;

        let manifest_path = cache_dir.join("manifest.json");
        let entries = Self::load_manifest(&manifest_path)?;

        Ok(Self {
            cache_dir,
            manifest_path,
            entries,
        })
    }

    /// Generate cache key from source and metadata
    pub fn generate_key(
        source: &str,
        language: &str,
        mode: &str,
    ) -> String {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        language.hash(&mut hasher);
        mode.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{:016x}_{}_{}", hash, language, mode)
    }

    /// Get a cache entry by key
    pub fn get(&self, key: &str) -> Option<PluginCacheEntry> {
        self.entries.get(key).map(|e| e.value().clone())
    }

    /// Put a cache entry
    pub fn put(
        &self,
        key: String,
        entry: PluginCacheEntry,
    ) -> std::io::Result<()> {
        self.entries.insert(key, entry);
        self.save_manifest()?;
        Ok(())
    }

    /// Validate cache entry (check source hash and file modification time)
    pub fn is_valid(&self, entry: &PluginCacheEntry, current_source: &str) -> bool {
        // Check library exists
        if !entry.library_path.exists() {
            return false;
        }

        // Check source hash
        let mut hasher = DefaultHasher::new();
        current_source.hash(&mut hasher);
        let current_hash = hasher.finish();

        if entry.source_hash != current_hash {
            return false;
        }

        // Check file modification time if source is external
        if let Some(ref source_file) = entry.source_file {
            if let Ok(source_meta) = std::fs::metadata(source_file) {
                if let Ok(lib_meta) = std::fs::metadata(&entry.library_path) {
                    if let (Ok(source_time), Ok(lib_time)) = (
                        source_meta.modified(),
                        lib_meta.modified()
                    ) {
                        if source_time > lib_time {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Invalidate a cache entry
    pub fn invalidate(&self, key: &str) -> std::io::Result<()> {
        if let Some((_, entry)) = self.entries.remove(key) {
            if entry.library_path.exists() {
                std::fs::remove_file(entry.library_path)?;
            }
            self.save_manifest()?;
        }
        Ok(())
    }

    /// Clear all cache entries
    pub fn clear(&self) -> std::io::Result<()> {
        for entry in self.entries.iter() {
            if entry.value().library_path.exists() {
                let _ = std::fs::remove_file(&entry.value().library_path);
            }
        }
        self.entries.clear();
        self.save_manifest()?;
        Ok(())
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Load manifest from disk
    fn load_manifest(path: &Path) -> std::io::Result<dashmap::DashMap<String, PluginCacheEntry>> {
        if !path.exists() {
            return Ok(dashmap::DashMap::new());
        }

        let data = std::fs::read_to_string(path)?;
        let entries: Vec<(String, PluginCacheEntry)> = serde_json::from_str(&data)
            .unwrap_or_default();

        Ok(entries.into_iter().collect())
    }

    /// Save manifest to disk
    fn save_manifest(&self) -> std::io::Result<()> {
        let entries: Vec<_> = self.entries.iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();

        let data = serde_json::to_string_pretty(&entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&self.manifest_path, data)?;

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_size: u64 = self.entries.iter()
            .filter_map(|e| std::fs::metadata(&e.value().library_path).ok())
            .map(|m| m.len())
            .sum();

        CacheStats {
            entries: self.entries.len(),
            total_bytes: total_size,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub total_bytes: u64,
}

impl CacheStats {
    /// Format size in human-readable format
    pub fn format_size(&self) -> String {
        let bytes = self.total_bytes as f64;
        if bytes < 1024.0 {
            format!("{} B", bytes)
        } else if bytes < 1024.0 * 1024.0 {
            format!("{:.2} KB", bytes / 1024.0)
        } else if bytes < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.2} MB", bytes / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        let key1 = PluginCache::generate_key("fn test() {}", "rust", "jit");
        let key2 = PluginCache::generate_key("fn test() {}", "rust", "jit");
        let key3 = PluginCache::generate_key("fn test() {}", "rust", "compile");
        
        assert_eq!(key1, key2, "Same source should generate same key");
        assert_ne!(key1, key3, "Different mode should generate different key");
    }

    #[test]
    fn test_cache_stats_format() {
        let stats = CacheStats {
            entries: 5,
            total_bytes: 1024 * 1024 + 512 * 1024, // 1.5 MB
        };
        
        let formatted = stats.format_size();
        assert!(formatted.contains("MB"));
    }
}

