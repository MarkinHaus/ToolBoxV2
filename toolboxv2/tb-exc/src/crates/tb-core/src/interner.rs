use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free string interner using DashMap for concurrent access
/// Provides O(1) cloning through Arc and structural sharing
pub struct StringInterner {
    cache: DashMap<String, Arc<String>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    total_saved: AtomicUsize,
    config: InternerConfig,
}

#[derive(Debug, Clone)]
pub struct InternerConfig {
    pub max_entries: usize,
    pub max_memory_bytes: usize,
    pub eviction_threshold: f64,
    pub auto_cleanup: bool,
}

impl Default for InternerConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_memory_bytes: 10 * 1024 * 1024, // 10 MB
            eviction_threshold: 0.8,
            auto_cleanup: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InternerStats {
    pub hits: usize,
    pub misses: usize,
    pub total_entries: usize,
    pub estimated_memory: usize,
    pub hit_rate: f64,
    pub total_saved_bytes: usize,
}

impl StringInterner {
    pub fn new(config: InternerConfig) -> Self {
        Self {
            cache: DashMap::with_capacity(1024),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            total_saved: AtomicUsize::new(0),
            config,
        }
    }

    /// Intern a string with zero-copy when possible
    pub fn intern(&self, s: &str) -> Arc<String> {
        // Fast path: read-only lookup
        if let Some(entry) = self.cache.get(s) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.total_saved.fetch_add(s.len(), Ordering::Relaxed);
            return Arc::clone(entry.value());
        }

        // Slow path: insert new string
        self.misses.fetch_add(1, Ordering::Relaxed);
        let arc_str = Arc::new(s.to_string());
        self.cache.insert(s.to_string(), Arc::clone(&arc_str));

        // Auto-cleanup if needed
        if self.config.auto_cleanup && self.should_cleanup() {
            self.cleanup();
        }

        arc_str
    }

    /// Get interned string if it exists
    pub fn get(&self, s: &str) -> Option<Arc<String>> {
        self.cache.get(s).map(|entry| Arc::clone(entry.value()))
    }

    fn should_cleanup(&self) -> bool {
        self.cache.len() > self.config.max_entries
            || self.estimated_memory() > self.config.max_memory_bytes
    }

    fn estimated_memory(&self) -> usize {
        self.cache
            .iter()
            .map(|entry| {
                entry.key().len() + entry.value().len() + std::mem::size_of::<Arc<String>>()
            })
            .sum()
    }

    fn cleanup(&self) {
        let threshold = (self.config.max_entries as f64 * self.config.eviction_threshold) as usize;

        if self.cache.len() > threshold {
            // Remove strings with only one reference (not used anywhere else)
            self.cache.retain(|_, v| Arc::strong_count(v) > 1);
        }
    }

    pub fn stats(&self) -> InternerStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        InternerStats {
            hits,
            misses,
            total_entries: self.cache.len(),
            estimated_memory: self.estimated_memory(),
            hit_rate,
            total_saved_bytes: self.total_saved.load(Ordering::Relaxed),
        }
    }

    pub fn clear(&self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.total_saved.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_interning() {
        let interner = StringInterner::new(InternerConfig::default());

        let s1 = interner.intern("hello");
        let s2 = interner.intern("hello");

        assert!(Arc::ptr_eq(&s1, &s2));

        let stats = interner.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_concurrent_interning() {
        use std::thread;

        let interner = Arc::new(StringInterner::new(InternerConfig::default()));
        let mut handles = vec![];

        for _ in 0..10 {
            let interner = Arc::clone(&interner);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    interner.intern(&format!("string_{}", i % 10));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = interner.stats();
        assert!(stats.hit_rate > 0.9); // Should have high hit rate
    }
}

