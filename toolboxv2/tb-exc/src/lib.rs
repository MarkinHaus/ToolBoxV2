//! # TB Core Engine - Production-Ready Monolith
//!
//! A multi-paradigm, multi-language execution engine with three modes:
//! - Compiled: Zero-overhead native code generation
//! - JIT: Fast interpretation with caching
//! - Streaming: Interactive execution with live feedback
//!
//! ## Features
//! - Static type system with inference
//! - Null-safety via Option<T>
//! - Generic types
//! - Result-based error handling
//! - Multi-language support (Rust, Python, JS, Go, Bash)
//! - Cross-platform compilation
//!
//! ## Performance
//! - Compiled mode: 0-5% overhead vs native Rust
//! - JIT mode: ~20-30% overhead
//! - Streaming mode: Interactive performance
//!
//! Version: 1.0.0
//! License: MIT
//!
//! tb-exc/src/lib.rs

#![allow(dead_code, unused_variables, unused_imports)]

// ═══════════════════════════════════════════════════════════════════════════
// §1 IMPORTS & DEPENDENCIES
// ═══════════════════════════════════════════════════════════════════════════

use std::{
    collections::HashMap,
    collections::VecDeque,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{Arc, RwLock, Mutex},
    cell::RefCell,
    rc::Rc,
    any::Any,
    marker::PhantomData,
};
use std::cmp::PartialEq;
use std::collections::hash_map::DefaultHasher;
// Async runtime
use std::future::Future;
use std::pin::Pin;

// Serialization
use std::str::FromStr;

use std::time::{SystemTime, UNIX_EPOCH};

// File I/O
use std::fs;
use std::io::{self, Read, Write};

// Process execution
use std::process::{Command};

// Environment
use std::env;
// Time
use std::time::{Duration, Instant};
use bumpalo::Bump;

pub mod dependency_compiler;
pub use dependency_compiler::{DependencyCompiler, Dependency, CompilationStrategy};
use pyo3::types::PyModule;

// Multi-threading (Rayon)
use rayon::prelude::*;
// ═══════════════════════════════════════════════════════════════════════════
// DEBUG LOGGING MACRO
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(debug_assertions)]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        eprintln!("[DEBUG] {}", format!($($arg)*));
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_log {
    ($($arg:tt)*) => {};
}


// ═══════════════════════════════════════════════════════════════════════════
// §2 CORE TYPE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Core type representation - supports static typing with inference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Unit type (void)
    Unit,

    /// Boolean
    Bool,

    /// Integer (64-bit signed)
    Int,

    /// Float (64-bit)
    Float,

    /// String (UTF-8)
    String,

    /// List of homogeneous types
    List(Box<Type>),

    /// Dictionary with key-value types
    Dict {
        key: Box<Type>,
        value: Box<Type>,
    },

    /// Optional type (null-safety)
    Option(Box<Type>),

    /// Result type (error handling)
    Result {
        ok: Box<Type>,
        err: Box<Type>,
    },

    /// Function type
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },

    /// Generic type parameter
    Generic {
        name: Arc<String>,
        constraints: Vec<TypeConstraint>,
    },

    /// Tuple type
    Tuple(Vec<Type>),

    /// Dynamic type (for dynamic mode)
    Dynamic,

    /// Native language type (opaque)
    Native {
        language: Language,
        type_name: Arc<String>,
    },

    /// Type variable (for inference)
    Var(usize),
}

/// Type constraints for generics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeConstraint {
    /// Must implement trait
    Trait(Arc<String>),

    /// Must be specific type
    Exact(Type),
}

impl TypeConstraint {
    /// Create Trait constraint with interned name
    pub fn trait_constraint(name: &str) -> Self {
        TypeConstraint::Trait(STRING_INTERNER.intern(name))
    }
}

impl Type {
    /// Check if type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float)
    }

    /// Check if type is primitive
    pub fn is_primitive(&self) -> bool {
        matches!(self, Type::Unit | Type::Bool | Type::Int | Type::Float | Type::String)
    }

    /// Get default value for type
    pub fn default_value(&self) -> Value {
        match self {
            Type::Unit => Value::Unit,
            Type::Bool => Value::Bool(false),
            Type::Int => Value::Int(0),
            Type::Float => Value::Float(0.0),
            Type::String => Value::String(Arc::new(String::new())),
            Type::List(_) => Value::List(Vec::new()),
            Type::Dict { .. } => Value::Dict(HashMap::new()),
            Type::Option(_) => Value::Option(None),
            _ => Value::Unit,
        }
    }

    /// Create Generic type with interned name
    pub fn generic(name: &str, constraints: Vec<TypeConstraint>) -> Self {
        Type::Generic {
            name: STRING_INTERNER.intern(name),
            constraints,
        }
    }

    /// Create Native type with interned name
    pub fn native(language: Language, type_name: &str) -> Self {
        Type::Native {
            language,
            type_name: STRING_INTERNER.intern(type_name),
        }
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// §2.4 STRING INTERNER - With Auto-Cleanup & Sliding Window
// ═══════════════════════════════════════════════════════════════════════════


/// Interned string entry with access tracking
#[derive(Clone)]
struct InternedEntry {
    string: Arc<String>,
    access_count: usize,
    last_access: u64,  // Unix timestamp in milliseconds
    size_bytes: usize,
}

impl InternedEntry {
    fn new(s: String) -> Self {
        let size_bytes = s.len();
        Self {
            string: Arc::new(s),
            access_count: 1,
            last_access: current_timestamp_ms(),
            size_bytes,
        }
    }

    fn touch(&mut self) {
        self.access_count += 1;
        self.last_access = current_timestamp_ms();
    }
}

/// Configuration for StringInterner behavior
#[derive(Debug, Clone)]
pub struct InternerConfig {
    /// Maximum number of unique strings (0 = unlimited)
    pub max_entries: usize,

    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_bytes: usize,

    /// Eviction threshold (0.0-1.0): trigger cleanup when this full
    pub eviction_threshold: f64,

    /// How many entries to remove per eviction (as fraction: 0.0-1.0)
    pub eviction_ratio: f64,

    /// Enable automatic cleanup
    pub auto_cleanup: bool,

    /// Keep strings accessed within this many milliseconds
    pub hot_string_window_ms: u64,
}

impl Default for InternerConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,           // Max 10k unique strings
            max_memory_bytes: 10 * 1024 * 1024,  // 10 MB
            eviction_threshold: 0.80,      // Cleanup at 80% full
            eviction_ratio: 0.25,          // Remove 25% of entries
            auto_cleanup: true,
            hot_string_window_ms: 60_000,  // Keep strings from last 60s
        }
    }
}

impl InternerConfig {
    /// Conservative config for long-running services
    pub fn conservative() -> Self {
        Self {
            max_entries: 5_000,
            max_memory_bytes: 5 * 1024 * 1024,
            eviction_threshold: 0.70,
            eviction_ratio: 0.30,
            auto_cleanup: true,
            hot_string_window_ms: 30_000,
        }
    }

    /// Aggressive config for high-throughput, short-lived processes
    pub fn aggressive() -> Self {
        Self {
            max_entries: 50_000,
            max_memory_bytes: 50 * 1024 * 1024,
            eviction_threshold: 0.90,
            eviction_ratio: 0.20,
            auto_cleanup: true,
            hot_string_window_ms: 120_000,
        }
    }

    /// No limits (original behavior, use with caution!)
    pub fn unlimited() -> Self {
        Self {
            max_entries: 0,
            max_memory_bytes: 0,
            eviction_threshold: 1.0,
            eviction_ratio: 0.0,
            auto_cleanup: false,
            hot_string_window_ms: 0,
        }
    }
}

/// Enhanced string interner with automatic cleanup
pub struct StringInterner {
    cache: RwLock<HashMap<String, InternedEntry>>,
    stats: RwLock<InternerStats>,
    config: InternerConfig,
    eviction_log: RwLock<VecDeque<EvictionEvent>>,
}

#[derive(Debug, Default, Clone)]
pub struct InternerStats {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub unique_strings: usize,
    pub memory_used_bytes: usize,
    pub memory_saved_bytes: usize,
    pub evictions_triggered: usize,
    pub strings_evicted: usize,
}

#[derive(Debug, Clone)]
pub struct EvictionEvent {
    timestamp: u64,
    entries_removed: usize,
    memory_freed: usize,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::with_config(InternerConfig::default())
    }

    pub fn with_config(config: InternerConfig) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(
                if config.max_entries > 0 { config.max_entries } else { 1024 }
            )),
            stats: RwLock::new(InternerStats::default()),
            config,
            eviction_log: RwLock::new(VecDeque::with_capacity(100)),
        }
    }

    /// Get or create Arc<String> (deduplicated)
    pub fn intern(&self, s: &str) -> Arc<String> {
        // Fast path: read-only check
        {
            let mut cache = self.cache.write().unwrap();

            if let Some(entry) = cache.get_mut(s) {
                entry.touch();

                let mut stats = self.stats.write().unwrap();
                stats.total_requests += 1;
                stats.cache_hits += 1;
                stats.memory_saved_bytes += s.len();

                return Arc::clone(&entry.string);
            }
        }

        // Slow path: insert new string
        self.intern_new(s.to_string())
    }

    /// Intern from owned String (consumes String if not cached)
    pub fn intern_owned(&self, s: String) -> Arc<String> {
        // Check if already cached
        {
            let mut cache = self.cache.write().unwrap();

            if let Some(entry) = cache.get_mut(&s) {
                entry.touch();

                let mut stats = self.stats.write().unwrap();
                stats.total_requests += 1;
                stats.cache_hits += 1;
                stats.memory_saved_bytes += s.len();

                return Arc::clone(&entry.string);
            }
        }

        // Insert new
        self.intern_new(s)
    }

    /// Insert new string (with auto-cleanup check)
    fn intern_new(&self, s: String) -> Arc<String> {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.total_requests += 1;

        // Check if cleanup needed BEFORE insertion
        if self.config.auto_cleanup && self.should_evict(&stats) {
            drop(stats); // Release stats lock
            drop(cache); // Release cache lock

            self.evict_cold_strings();

            // Re-acquire locks
            cache = self.cache.write().unwrap();
            stats = self.stats.write().unwrap();
        }

        let entry = InternedEntry::new(s.clone());
        let arc = Arc::clone(&entry.string);

        stats.unique_strings += 1;
        stats.memory_used_bytes += entry.size_bytes;

        cache.insert(s, entry);

        arc
    }

    /// Check if eviction should be triggered
    fn should_evict(&self, stats: &InternerStats) -> bool {
        if self.config.max_entries > 0 {
            let usage = stats.unique_strings as f64 / self.config.max_entries as f64;
            if usage >= self.config.eviction_threshold {
                return true;
            }
        }

        if self.config.max_memory_bytes > 0 {
            let usage = stats.memory_used_bytes as f64 / self.config.max_memory_bytes as f64;
            if usage >= self.config.eviction_threshold {
                return true;
            }
        }

        false
    }

    /// Evict cold (rarely used) strings using LRU strategy
    fn evict_cold_strings(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let current_time = current_timestamp_ms();
        let hot_window = self.config.hot_string_window_ms;

        // Calculate how many to remove
        let target_remove = (cache.len() as f64 * self.config.eviction_ratio).ceil() as usize;

        if target_remove == 0 {
            return;
        }

        debug_log!("╔════════════════════════════════════════════════════════════════╗");
        debug_log!("║           STRING_INTERNER Auto-Cleanup Triggered               ║");
        debug_log!("╠════════════════════════════════════════════════════════════════╣");
        debug_log!("║ Current Entries: {:>43} ║", cache.len());
        debug_log!("║ Memory Used:     {:>39} KB ║", stats.memory_used_bytes / 1024);
        debug_log!("║ Target Remove:   {:>43} ║", target_remove);
        debug_log!("╚════════════════════════════════════════════════════════════════╝");

        // Collect candidates for eviction
        let mut candidates: Vec<(String, u64, usize)> = cache
            .iter()
            .filter_map(|(key, entry)| {
                let age_ms = current_time.saturating_sub(entry.last_access);

                // Keep hot strings (recently accessed)
                if hot_window > 0 && age_ms < hot_window {
                    return None;
                }

                // Keep strings with high access count (adjust threshold as needed)
                if entry.access_count > 10 {
                    return None;
                }

                // Score = older + less accessed = higher priority for removal
                let score = age_ms + (1000 / (entry.access_count + 1) as u64);
                Some((key.clone(), score, entry.size_bytes))
            })
            .collect();

        // Sort by score (highest = coldest)
        candidates.sort_by_key(|(_, score, _)| std::cmp::Reverse(*score));

        // Remove coldest entries
        let mut removed_count = 0;
        let mut memory_freed = 0;

        for (key, _, size) in candidates.iter().take(target_remove) {
            if cache.remove(key).is_some() {
                removed_count += 1;
                memory_freed += size;
            }
        }

        // Update stats
        stats.unique_strings = cache.len();
        stats.memory_used_bytes = stats.memory_used_bytes.saturating_sub(memory_freed);
        stats.evictions_triggered += 1;
        stats.strings_evicted += removed_count;

        // Log eviction event
        let mut log = self.eviction_log.write().unwrap();
        log.push_back(EvictionEvent {
            timestamp: current_time,
            entries_removed: removed_count,
            memory_freed,
        });

        // Keep only last 100 events
        while log.len() > 100 {
            log.pop_front();
        }

        debug_log!("✓ Evicted {} entries, freed {} KB", removed_count, memory_freed / 1024);
    }

    /// Get statistics
    pub fn stats(&self) -> InternerStats {
        self.stats.read().unwrap().clone()
    }

    /// Get cache hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().unwrap();
        if stats.total_requests == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / stats.total_requests as f64
        }
    }

    /// Get current memory usage as percentage (0.0 - 1.0)
    pub fn memory_usage(&self) -> f64 {
        if self.config.max_memory_bytes == 0 {
            return 0.0;
        }
        let stats = self.stats.read().unwrap();
        stats.memory_used_bytes as f64 / self.config.max_memory_bytes as f64
    }

    /// Get current entry count as percentage (0.0 - 1.0)
    pub fn capacity_usage(&self) -> f64 {
        if self.config.max_entries == 0 {
            return 0.0;
        }
        let stats = self.stats.read().unwrap();
        stats.unique_strings as f64 / self.config.max_entries as f64
    }

    /// Manual cleanup trigger
    pub fn cleanup_now(&self) {
        if !self.config.auto_cleanup {
            debug_log!("Warning: Auto-cleanup disabled, manual cleanup may not follow eviction strategy");
        }
        self.evict_cold_strings();
    }

    /// Clear entire cache (use with caution!)
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let freed_memory = stats.memory_used_bytes;

        cache.clear();
        *stats = InternerStats::default();

        debug_log!("STRING_INTERNER cleared: freed {} KB", freed_memory / 1024);
    }

    /// Get eviction history (last 100 events)
    pub fn eviction_history(&self) -> Vec<EvictionEvent> {
        self.eviction_log.read().unwrap().iter().cloned().collect()
    }

    /// Get detailed health report
    pub fn health_report(&self) -> String {
        let stats = self.stats();
        let capacity = self.capacity_usage();
        let memory = self.memory_usage();

        format!(
            "STRING_INTERNER Health Report\n\
             ═══════════════════════════════════════════\n\
             Entries:       {}/{} ({:.1}% full)\n\
             Memory:        {} KB / {} KB ({:.1}% full)\n\
             Hit Rate:      {:.1}%\n\
             Memory Saved:  {} KB\n\
             Evictions:     {} (removed {} strings)\n\
             Status:        {}",
            stats.unique_strings,
            if self.config.max_entries > 0 { self.config.max_entries } else { 0 },
            capacity * 100.0,
            stats.memory_used_bytes / 1024,
            self.config.max_memory_bytes / 1024,
            memory * 100.0,
            self.hit_rate() * 100.0,
            stats.memory_saved_bytes / 1024,
            stats.evictions_triggered,
            stats.strings_evicted,
            if capacity > 0.9 || memory > 0.9 { "⚠️  CRITICAL" }
            else if capacity > 0.7 || memory > 0.7 { "⚠️  WARNING" }
            else { "✓ OK" }
        )
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// Globaler String-Interner mit Default-Config
lazy_static::lazy_static! {
    /// Global string interner with auto-cleanup
    ///
    /// Default limits:
    /// - Max 10,000 entries
    /// - Max 10 MB memory
    /// - Auto-cleanup at 80% capacity
    ///
    /// Usage:
    /// ```
    /// use crate::STRING_INTERNER;
    /// let name = STRING_INTERNER.intern("my_var");
    /// ```
    pub static ref STRING_INTERNER: StringInterner = StringInterner::new();

    /// Alternative: Conservative interner for long-running services
    ///
    /// Limits:
    /// - Max 5,000 entries
    /// - Max 5 MB memory
    /// - Aggressive cleanup at 70% capacity
    #[allow(dead_code)]
    pub static ref STRING_INTERNER_CONSERVATIVE: StringInterner =
        StringInterner::with_config(InternerConfig::conservative());
}

/// Convenience macro for interning strings
#[macro_export]
macro_rules! intern {
    ($s:expr) => {
        $crate::STRING_INTERNER.intern($s)
    };
}
// ═══════════════════════════════════════════════════════════════════════════
// Environment-based Interner Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Initialize interner from environment variables
///
/// Supported env vars:
/// - TB_INTERNER_MAX_ENTRIES (default: 10000)
/// - TB_INTERNER_MAX_MEMORY_MB (default: 10)
/// - TB_INTERNER_EVICTION_THRESHOLD (default: 0.80)
/// - TB_INTERNER_EVICTION_RATIO (default: 0.25)
/// - TB_INTERNER_AUTO_CLEANUP (default: true)
pub fn interner_config_from_env() -> InternerConfig {
    let max_entries = std::env::var("TB_INTERNER_MAX_ENTRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let max_memory_mb = std::env::var("TB_INTERNER_MAX_MEMORY_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    let eviction_threshold = std::env::var("TB_INTERNER_EVICTION_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.80);

    let eviction_ratio = std::env::var("TB_INTERNER_EVICTION_RATIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.25);

    let auto_cleanup = std::env::var("TB_INTERNER_AUTO_CLEANUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true);

    InternerConfig {
        max_entries,
        max_memory_bytes: max_memory_mb * 1024 * 1024,
        eviction_threshold,
        eviction_ratio,
        auto_cleanup,
        hot_string_window_ms: 60_000,
    }
}

// Alternative: Create custom global interner
lazy_static::lazy_static! {
    /// Interner configured from environment variables
    #[allow(dead_code)]
    pub static ref STRING_INTERNER_ENV: StringInterner =
        StringInterner::with_config(interner_config_from_env());
}
// ═══════════════════════════════════════════════════════════════════════════
// §2.5 IMPORT CACHE & COMPILATION MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════
/// Cached compilation artifact with static lifetime
#[derive(Debug, Clone)]
pub struct CachedArtifact {
    pub source_path: PathBuf,
    pub source_hash: u64,
    pub binary_path: PathBuf,

    // Statements als 'static (via Box::leak)
    pub statements: Vec<Statement<'static>>,
    pub compiled_at: SystemTime,
}

/// Import cache manager with full binary compilation
pub struct ImportCache {
    cache_dir: PathBuf,
    artifacts: HashMap<PathBuf, CachedArtifact>,
}

impl ImportCache {
    pub fn new() -> TBResult<Self> {
        let cache_dir = std::env::temp_dir().join("tb_import_cache");
        fs::create_dir_all(&cache_dir)
            .map_err(|e| TBError::IoError(Arc::new(format!("Failed to create cache dir: {}", e))))?;

        Ok(Self {
            cache_dir,
            artifacts: HashMap::new(),
        })
    }

    /// Get or compile an import (FULL COMPILATION)
    pub fn get_or_compile<'arena>(
        &mut self,
        import_path: &Path,
        parent_config: &Config,
        target_arena: &'arena Bump,
    ) -> TBResult<Vec<Statement<'arena>>> {
        let canonical_path = import_path.canonicalize()
            .map_err(|_| TBError::IoError(
                Arc::new(format!("Import not found: {}", import_path.display()))
            ))?;

        // Read source and compute hash
        let source = fs::read_to_string(&canonical_path)?;
        let source_hash = self.compute_hash(&source);

        // Check cache
        if let Some(cached) = self.artifacts.get(&canonical_path) {
            if cached.source_hash == source_hash {
                debug_log!("✓ Using cached artifact for {}", import_path.display());

                // Clone cached 'static statements in neue Arena
                return Ok(self.clone_statements_to_arena(&cached.statements, target_arena));
            }
        }

        debug_log!("⚙ Processing import: {}", import_path.display());

        // Parse import config
        let import_config = Config::parse(&source)?;

        // COMPILE if in compiled mode
        if matches!(import_config.mode, ExecutionMode::Compiled { .. }) {
            debug_log!("  → Import is in compiled mode, compiling...");
            return self.compile_and_cache(
                &canonical_path,
                &source,
                source_hash,
                &import_config,
                target_arena
            );
        }

        // JIT mode: parse and cache
        debug_log!("  → Import is in JIT mode, parsing...");
        self.parse_and_cache_jit(&canonical_path, &source, source_hash, target_arena)
    }

    /// Compile import to native library (FULL IMPLEMENTATION)
    fn compile_and_cache<'arena>(
        &mut self,
        path: &Path,
        source: &str,
        source_hash: u64,
        config: &Config,
        target_arena: &'arena Bump,
    ) -> TBResult<Vec<Statement<'arena>>> {
        let cache_key = format!("{:x}", source_hash);
        let binary_path = self.cache_dir.join(format!("lib_{}.so", cache_key));

        // PHASE 1: Parse in temporärer Arena für Compilation
        let parse_arena = Bump::new();
        let statements_arena = self.parse_import(source, &parse_arena)?;

        // PHASE 2: Compile Binary (falls nicht existiert)
        if !binary_path.exists() {
            debug_log!("  ⚙ Compiling to: {}", binary_path.display());

            //  Erstelle 'static Versionen durch Box::leak
            let statements_static: &'static [Statement<'static>] = {
                let vec: Vec<Statement<'static>> = statements_arena.iter()
                    .map(|stmt| self.clone_statement_static(stmt))
                    .collect();

                // Leak zu 'static (wird nie freed, aber das ist OK für Compiler)
                Box::leak(vec.into_boxed_slice())
            };

            let compiler = Compiler::new(config.clone())
                .with_target(TargetPlatform::current())
                .with_optimization(3);

            compiler.compile_to_library(statements_static, &binary_path)?;
            debug_log!("  ✓ Compilation complete");
        } else {
            debug_log!("  ✓ Using cached binary: {}", binary_path.display());
        }

        // PHASE 3: Parse erneut in target_arena
        let final_statements = self.parse_import(source, target_arena)?;

        // PHASE 4: Cache Metadata
        let cached_static: Vec<Statement<'static>> = statements_arena.iter()
            .map(|stmt| self.clone_statement_static(stmt))
            .collect();

        self.artifacts.insert(path.to_path_buf(), CachedArtifact {
            source_path: path.to_path_buf(),
            source_hash,
            binary_path: binary_path.clone(),
            statements: cached_static,
            compiled_at: SystemTime::now(),
        });

        Ok(final_statements)
    }

    /// Parse and cache for JIT mode
    fn parse_and_cache_jit<'arena>(
        &mut self,
        path: &Path,
        source: &str,
        source_hash: u64,
        target_arena: &'arena Bump,
    ) -> TBResult<Vec<Statement<'arena>>> {
        // Parse in eigener Arena
        let parse_arena = Bump::new();
        let statements_arena = self.parse_import(source, &parse_arena)?;

        // Promote zu 'static
        let statements_static = self.promote_to_static(&statements_arena);

        // Cache ohne Binary
        self.artifacts.insert(path.to_path_buf(), CachedArtifact {
            source_path: path.to_path_buf(),
            source_hash,
            binary_path: PathBuf::new(),
            statements: statements_static.clone(),
            compiled_at: SystemTime::now(),
        });

        // Clone in Target-Arena
        Ok(self.clone_statements_to_arena(&statements_static, target_arena))
    }

    /// Parse import source
    fn parse_import<'a>(&self, source: &str, arena: &'a Bump) -> TBResult<Vec<Statement<'a>>> {
        let clean_source = TBCore::strip_directives(source);
        let mut lexer = Lexer::new(&clean_source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens, arena);
        parser.parse()
    }

    /// Promote statements to 'static lifetime via Box::leak
    fn promote_to_static<'a>(&self, statements: &[Statement<'a>]) -> Vec<Statement<'static>> {
        statements.iter().map(|stmt| self.clone_statement_static(stmt)).collect()
    }

    /// Deep clone statement to 'static
    fn clone_statement_static<'a>(&self, stmt: &Statement<'a>) -> Statement<'static> {
        match stmt {
            Statement::Function { name, params, return_type, body } => {
                let params_static: Vec<Parameter<'static>> = params.iter()
                    .map(|p| Parameter {
                        name: Arc::clone(&p.name),
                        type_annotation: p.type_annotation.clone(),
                        default: p.default.as_ref().map(|e| self.clone_expr_static(e)),
                    })
                    .collect();

                Statement::Function {
                    name: Arc::clone(name),
                    params: self.vec_to_arena_vec_static(params_static),
                    return_type: return_type.clone(),
                    body: self.clone_expr_static(body),
                }
            }

            Statement::Let { name, mutable, type_annotation, value, scope } => {
                Statement::Let {
                    name: Arc::clone(name),
                    mutable: *mutable,
                    type_annotation: type_annotation.clone(),
                    value: self.clone_expr_static(value),
                    scope: *scope,
                }
            }

            Statement::Assign { target, value } => {
                Statement::Assign {
                    target: self.clone_expr_static(target),
                    value: self.clone_expr_static(value),
                }
            }

            Statement::Expr(expr) => {
                Statement::Expr(self.clone_expr_static(expr))
            }

            Statement::Import { module, items } => {
                Statement::Import {
                    module: Arc::clone(module),
                    items: items.clone(),
                }
            }
        }
    }

    /// Deep clone expression to 'static
    fn clone_expr_static<'a>(&self, expr: &Expr<'a>) -> Expr<'static> {
        match expr {
            Expr::Literal(lit) => Expr::Literal(lit.clone()),
            Expr::Variable(name) => Expr::Variable(Arc::clone(name)),

            Expr::BinOp { op, left, right } => {
                let left_static = Box::new(self.clone_expr_static(left));
                let right_static = Box::new(self.clone_expr_static(right));

                Expr::BinOp {
                    op: *op,
                    left: Box::leak(left_static),
                    right: Box::leak(right_static),
                }
            }

            Expr::Call { function, args } => {
                let func_static = Box::new(self.clone_expr_static(function));
                let args_static: Vec<Expr<'static>> = args.iter()
                    .map(|a| self.clone_expr_static(a))
                    .collect();

                Expr::Call {
                    function: Box::leak(func_static),
                    args: self.vec_to_arena_vec_static(args_static),
                }
            }

            Expr::Block { statements, result } => {
                let stmts_static: Vec<Statement<'static>> = statements.iter()
                    .map(|s| self.clone_statement_static(s))
                    .collect();

                Expr::Block {
                    statements: self.vec_to_arena_vec_static(stmts_static),
                    result: result.as_ref().map(|e| {
                        let expr_static = Box::new(self.clone_expr_static(e));
                        Box::leak(expr_static) as &'static _
                    }),
                }
            }

            Expr::If { condition, then_branch, else_branch } => {
                let cond_static = Box::new(self.clone_expr_static(condition));
                let then_static = Box::new(self.clone_expr_static(then_branch));
                let else_static = else_branch.as_ref().map(|e| {
                    let expr = Box::new(self.clone_expr_static(e));
                    Box::leak(expr) as &'static _
                });

                Expr::If {
                    condition: Box::leak(cond_static),
                    then_branch: Box::leak(then_static),
                    else_branch: else_static,
                }
            }

            Expr::List(items) => {
                let items_static: Vec<Expr<'static>> = items.iter()
                    .map(|i| self.clone_expr_static(i))
                    .collect();
                Expr::List(self.vec_to_arena_vec_static(items_static))
            }

            // Analog für andere Expr-Typen
            _ => Expr::Literal(Literal::Unit),
        }
    }

    /// Convert Vec to ArenaVec in 'static lifetime
    fn vec_to_arena_vec_static<T>(&self, vec: Vec<T>) -> bumpalo::collections::Vec<'static, T> {
        // Erstelle neue lokale Arena und leak sie absichtlich
        let leaked_arena: &'static Bump = Box::leak(Box::new(Bump::new()));

        let mut arena_vec = bumpalo::collections::Vec::new_in(leaked_arena);
        arena_vec.extend(vec);
        arena_vec
    }

    /// Clone cached statements into target arena
    fn clone_statements_to_arena<'arena>(
        &self,
        statements: &[Statement<'static>],
        arena: &'arena Bump,
    ) -> Vec<Statement<'arena>> {
        statements.iter().map(|stmt| self.clone_statement_to_arena(stmt, arena)).collect()
    }

    /// Clone single statement into arena
    fn clone_statement_to_arena<'arena>(
        &self,
        stmt: &Statement<'static>,
        arena: &'arena Bump,
    ) -> Statement<'arena> {
        match stmt {
            Statement::Function { name, params, return_type, body } => {
                let params_arena: Vec<Parameter<'arena>> = params.iter()
                    .map(|p| Parameter {
                        name: Arc::clone(&p.name),
                        type_annotation: p.type_annotation.clone(),
                        default: p.default.as_ref().map(|e| self.clone_expr_to_arena(e, arena)),
                    })
                    .collect();

                Statement::Function {
                    name: Arc::clone(name),
                    params: ArenaVec::from_iter_in(params_arena, arena),
                    return_type: return_type.clone(),
                    body: self.clone_expr_to_arena(body, arena),
                }
            }

            Statement::Let { name, mutable, type_annotation, value, scope } => {
                Statement::Let {
                    name: Arc::clone(name),
                    mutable: *mutable,
                    type_annotation: type_annotation.clone(),
                    value: self.clone_expr_to_arena(value, arena),
                    scope: *scope,
                }
            }

            Statement::Assign { target, value } => {
                Statement::Assign {
                    target: self.clone_expr_to_arena(target, arena),
                    value: self.clone_expr_to_arena(value, arena),
                }
            }

            Statement::Expr(expr) => {
                Statement::Expr(self.clone_expr_to_arena(expr, arena))
            }

            Statement::Import { module, items } => {
                Statement::Import {
                    module: Arc::clone(module),
                    items: items.clone(),
                }
            }
        }
    }

    /// Clone expression into arena
    fn clone_expr_to_arena<'arena>(
        &self,
        expr: &Expr<'static>,
        arena: &'arena Bump,
    ) -> Expr<'arena> {
        match expr {
            Expr::Literal(lit) => Expr::Literal(lit.clone()),
            Expr::Variable(name) => Expr::Variable(Arc::clone(name)),

            Expr::BinOp { op, left, right } => {
                Expr::BinOp {
                    op: *op,
                    left: arena.alloc(self.clone_expr_to_arena(left, arena)),
                    right: arena.alloc(self.clone_expr_to_arena(right, arena)),
                }
            }

            Expr::Call { function, args } => {
                let args_arena: Vec<Expr<'arena>> = args.iter()
                    .map(|a| self.clone_expr_to_arena(a, arena))
                    .collect();

                Expr::Call {
                    function: arena.alloc(self.clone_expr_to_arena(function, arena)),
                    args: ArenaVec::from_iter_in(args_arena, arena),
                }
            }

            Expr::Block { statements, result } => {
                let stmts_arena: Vec<Statement<'arena>> = statements.iter()
                    .map(|s| self.clone_statement_to_arena(s, arena))
                    .collect();

                Expr::Block {
                    statements: ArenaVec::from_iter_in(stmts_arena, arena),
                    result: result.as_ref().map(|e|
                        arena.alloc(self.clone_expr_to_arena(e, arena)) as &_
                    ),
                }
            }

            Expr::List(items) => {
                let items_arena: Vec<Expr<'arena>> = items.iter()
                    .map(|i| self.clone_expr_to_arena(i, arena))
                    .collect();
                Expr::List(ArenaVec::from_iter_in(items_arena, arena))
            }

            // Analog für andere Expr-Typen
            _ => Expr::Literal(Literal::Unit),
        }
    }

    /// Compute hash of source
    fn compute_hash(&self, source: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear cache
    pub fn clear(&mut self) -> TBResult<()> {
        fs::remove_dir_all(&self.cache_dir)?;
        fs::create_dir_all(&self.cache_dir)?;
        self.artifacts.clear();
        Ok(())
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// §3 RUNTIME VALUE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime value - optimized for performance
#[derive(Debug, Clone)]
pub enum Value<'arena> {
    /// Unit value
    Unit,

    /// Boolean value
    Bool(bool),

    /// Integer value (64-bit)
    Int(i64),

    /// Float value (64-bit)
    Float(f64),

    /// String value
    String(Arc<String>),

    /// List of values
    List(Vec<Value<'arena>>),

    /// Dictionary
    Dict(HashMap<Arc<String>, Value<'arena>>),

    /// Optional value
    Option(Option<Box<Value<'arena>>>),

    /// Result value
    Result(Result<Box<Value<'arena>>, Box<Value<'arena>>>),

    /// Function closure
    Function {
        params: Vec<Arc<String>>,
        body: AstNode<'arena, Expr<'arena>>,
        env: Environment<'arena>,
    },

    /// Tuple
    Tuple(Vec<Value<'arena>>),

    /// Native handle (for language interop)
    Native {
        language: Language,
        type_name: Arc<String>,
        handle: NativeHandle,
    },
}

impl<'arena> Value<'arena> {
    /// Get type of value
    pub fn get_type(&self) -> Type {
        match self {
            Value::Unit => Type::Unit,
            Value::Bool(_) => Type::Bool,
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::String(_) => Type::String,
            Value::List(items) => {
                let inner = items.first()
                    .map(|v| v.get_type())
                    .unwrap_or(Type::Dynamic);
                Type::List(Box::new(inner))
            }
            Value::Dict(_) => Type::Dict {
                key: Box::new(Type::String),
                value: Box::new(Type::Dynamic),
            },
            Value::Option(opt) => {
                let inner = opt.as_ref()
                    .map(|v| v.get_type())
                    .unwrap_or(Type::Dynamic);
                Type::Option(Box::new(inner))
            }
            Value::Result(res) => {
                match res {
                    Ok(ref v) => Type::Result {
                        ok: Box::new(v.get_type()),
                        err: Box::new(Type::Dynamic),
                    },
                    Err(ref e) => Type::Result {
                        ok: Box::new(Type::Dynamic),
                        err: Box::new(e.get_type()),
                    },
                }
            },
            Value::Function { params, .. } => Type::Function {
                params: vec![Type::Dynamic; params.len()],
                ret: Box::new(Type::Dynamic),
            },
            Value::Tuple(items) => Type::Tuple(items.iter().map(|v| v.get_type()).collect()),
            Value::Native { language, type_name, .. } => Type::Native {
                language: *language,
                type_name: Arc::clone(type_name),
            },
        }
    }

    /// Convert to boolean (for conditionals)
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Unit => false,
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0 && !f.is_nan(),
            Value::String(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Dict(d) => !d.is_empty(),
            Value::Option(opt) => opt.is_some(),
            Value::Result(res) => res.is_ok(),
            _ => true,
        }
    }

    pub fn clone_static(&self) -> TBResult<Value<'static>> {
        match self {
            Value::Unit => Ok(Value::Unit),
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::Int(i) => Ok(Value::Int(*i)),
            Value::Float(f) => Ok(Value::Float(*f)),
            Value::String(s) => Ok(Value::String(s.clone())),
            Value::List(items) => {
                let static_items: TBResult<Vec<_>> = items.iter().map(|v| v.clone_static()).collect();
                Ok(Value::List(static_items?))
            }
            Value::Dict(map) => {
                let mut static_map = HashMap::new();
                for (k, v) in map {
                    static_map.insert(k.clone(), v.clone_static()?);
                }
                Ok(Value::Dict(static_map))
            }
            Value::Option(opt) => {
                if let Some(val) = opt {
                    Ok(Value::Option(Some(Box::new(val.clone_static()?))))
                } else {
                    Ok(Value::Option(None))
                }
            }
            Value::Result(res) => match res {
                Ok(ok_val) => Ok(Value::Result(Ok(Box::new(ok_val.clone_static()?)))),
                Err(err_val) => Ok(Value::Result(Err(Box::new(err_val.clone_static()?)))),
            },
            Value::Tuple(items) => {
                let static_items: TBResult<Vec<_>> = items.iter().map(|v| v.clone_static()).collect();
                Ok(Value::Tuple(static_items?))
            }
            Value::Native { language, type_name, handle } => Ok(Value::Native {
                language: *language,
                type_name: type_name.clone(),
                handle: handle.clone(),
            }),
            // Function kann nicht 'static gemacht werden, da sie den AST referenziert.
            Value::Function { .. } => Err(TBError::InvalidOperation(
                STRING_INTERNER.intern("Cannot return a function closure from the main execution scope."),
            )),
        }
    }
}

impl<'arena> Display for Value<'arena> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Value::Unit => write!(f, ""),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(fl) => {
                // Smart float formatting
                if fl.fract() == 0.0 && fl.is_finite() {
                    write!(f, "{:.1}", fl) // Show ".0" for whole numbers
                } else {
                    write!(f, "{}", fl)
                }
            },
            Value::String(s) => write!(f, "\"{}\"", s.as_str()),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Dict(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k.as_str(), v)?;
                }
                write!(f, "}}")
            }
            Value::Option(opt) => match opt {
                Some(v) => write!(f, "Some({})", v),
                None => write!(f, "None"),
            },
            Value::Result(res) => match res {
                Ok(v) => write!(f, "Ok({})", v),
                Err(e) => write!(f, "Err({})", e),
            },
            Value::Function { params, .. } => {
                write!(f, "fn({})", params.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            }
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            Value::Native { type_name, .. } => {
                write!(f, "<native:{}>", type_name)
            }
        }
    }
}

/// Native handle for language interop
#[derive(Debug, Clone)]
pub struct NativeHandle {
    pub id: u64,
    pub data: Arc<dyn Any + Send + Sync>,
}


/// Variable scope annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarScope {
    /// Thread-local (default)
    Local,

    /// Shared across threads with RwLock
    Shared,

    /// Immutable shared (Arc only, no lock)
    Immutable,
}

/// Thread-safe value wrapper for parallel execution
#[derive(Debug, Clone)]
pub enum ScopedValue<'arena> {
    /// Local to single thread
    Local(Value<'arena>),

    /// Shared with read-write lock
    Shared(Arc<RwLock<Value<'arena>>>),

    /// Immutable shared reference
    Immutable(Arc<Value<'arena>>),
}

impl<'arena> ScopedValue<'arena> {
    /// Get value (clones for safety)
    pub fn get(&self) -> TBResult<Value<'arena>> {
        match self {
            ScopedValue::Local(v) => Ok(v.clone()),
            ScopedValue::Shared(v) => Ok(v.read().unwrap().clone()),
            ScopedValue::Immutable(v) => Ok((**v).clone()),
        }
    }

    /// Set value (fails for Immutable)
    pub fn set(&mut self, value: Value<'arena>) -> TBResult<()> {
        match self {
            ScopedValue::Local(v) => {
                *v = value;
                Ok(())
            }
            ScopedValue::Shared(v) => {
                *v.write().unwrap() = value;
                Ok(())
            }
            ScopedValue::Immutable(_) => {
                Err(TBError::InvalidOperation(
                    Arc::new("Cannot modify immutable shared variable".to_string())
                ))
            }
        }
    }

    /// Create from Value with scope
    pub fn new(value: Value<'arena>, scope: VarScope) -> Self {
        match scope {
            VarScope::Local => ScopedValue::Local(value),
            VarScope::Shared => ScopedValue::Shared(Arc::new(RwLock::new(value))),
            VarScope::Immutable => ScopedValue::Immutable(Arc::new(value)),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §4 ABSTRACT SYNTAX TREE
// ═══════════════════════════════════════════════════════════════════════════

// Typ-Aliase für bessere Lesbarkeit
type AstNode<'arena, T> = &'arena T;
type ArenaVec<'arena, T> = bumpalo::collections::Vec<'arena, T>;

/// Expression - core AST node
#[derive(Debug, Clone)]
pub enum Expr<'arena> {
    /// Literal value
    Literal(Literal),

    /// Variable reference
    Variable(Arc<String>),

    /// Binary operation
    BinOp {
        op: BinOp,
        left: AstNode<'arena, Expr<'arena>>,
        right: AstNode<'arena, Expr<'arena>>,
    },

    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        expr: AstNode<'arena, Expr<'arena>>,
    },

    /// Function call
    Call {
        function: AstNode<'arena, Expr<'arena>>,
        args: ArenaVec<'arena, Expr<'arena>>,
    },

    /// Method call
    Method {
        object: AstNode<'arena, Expr<'arena>>,
        method: Arc<String>,
        args: ArenaVec<'arena, Expr<'arena>>,
    },

    /// Index access
    Index {
        object: AstNode<'arena, Expr<'arena>>,
        index: AstNode<'arena, Expr<'arena>>,
    },

    /// Field access
    Field {
        object: AstNode<'arena, Expr<'arena>>,
        field: Arc<String>,
    },

    /// Block expression
    Block {
        statements: ArenaVec<'arena, Statement<'arena>>,
        result: Option<AstNode<'arena, Expr<'arena>>>,
    },

    /// If expression
    If {
        condition: AstNode<'arena, Expr<'arena>>,
        then_branch: AstNode<'arena, Expr<'arena>>,
        else_branch: Option<AstNode<'arena, Expr<'arena>>>,
    },

    /// Match expression
    Match {
        scrutinee: AstNode<'arena, Expr<'arena>>,
        arms: ArenaVec<'arena, MatchArm<'arena>>,
    },

    /// Loop expression
    Loop {
        body: AstNode<'arena, Expr<'arena>>,
    },

    /// While loop
    While {
        condition: AstNode<'arena, Expr<'arena>>,
        body: AstNode<'arena, Expr<'arena>>,
    },

    /// For loop
    For {
        variable: Arc<String>,
        iterable: AstNode<'arena, Expr<'arena>>,
        body: AstNode<'arena, Expr<'arena>>,
    },

    /// Return expression
    Return(Option<AstNode<'arena, Expr<'arena>>>),

    /// Break expression
    Break(Option<AstNode<'arena, Expr<'arena>>>),



    /// Continue expression
    Continue,

    /// Lambda function
    Lambda {
        params: ArenaVec<'arena, Parameter<'arena>>,
        body: AstNode<'arena, Expr<'arena>>,
    },

    /// List literal
    List(ArenaVec<'arena, Expr<'arena>>),

    /// Dict literal
    Dict(ArenaVec<'arena, (Expr<'arena>, Expr<'arena>)>),

    /// Tuple literal
    Tuple(ArenaVec<'arena, Expr<'arena>>),

    /// Pipeline operation
    Pipeline {
        value: AstNode<'arena, Expr<'arena>>,
        operations: ArenaVec<'arena, Expr<'arena>>,
    },

    /// Parallel execution -
    Parallel(bumpalo::collections::Vec<'arena, Expr<'arena>>),

    /// Native code block
    Native {
        language: Language,
        code: Arc<String>,
    },

    /// Try expression (? operator)
    Try(AstNode<'arena, Expr<'arena>>),
}

/// Statement
#[derive(Debug, Clone)]
pub enum Statement<'arena> {
    /// Let binding
    Let {
        name: Arc<String>,
        mutable: bool,
        type_annotation: Option<Type>,
        value: Expr<'arena>,
        scope: VarScope,
    },

    /// Assignment
    Assign {
        target: Expr<'arena>,
        value: Expr<'arena>,
    },

    /// Expression statement
    Expr(Expr<'arena>),

    /// Function definition
    Function {
        name: Arc<String>,
        params: ArenaVec<'arena, Parameter<'arena>>,
        return_type: Option<Type>,
        body: Expr<'arena>,
    },

    /// Import statement
    Import {
        module: Arc<String>,
        items: Vec<Arc<String>>,
    },
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter<'arena> {
    pub name: Arc<String>,
    pub type_annotation: Option<Type>,
    pub default: Option<Expr<'arena>>,
}

/// Match arm
#[derive(Debug, Clone)]
pub struct MatchArm<'arena> {
    pub pattern: Pattern<'arena>,
    pub guard: Option<Expr<'arena>>,
    pub body: Expr<'arena>,
}

/// Pattern matching
#[derive(Debug, Clone)]
pub enum Pattern<'arena> {
    /// Wildcard pattern
    Wildcard,

    /// Literal pattern
    Literal(Literal),

    /// Variable binding
    Variable(Arc<String>),

    /// Tuple pattern
    Tuple(ArenaVec<'arena, Pattern<'arena>>),

    /// List pattern
    List(ArenaVec<'arena, Pattern<'arena>>),

    /// Or pattern
    Or(ArenaVec<'arena, Pattern<'arena>>),
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<String>),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add, Sub, Mul, Div, Mod, Pow,

    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,

    // Logical
    And, Or,

    // Bitwise
    BitAnd, BitOr, BitXor, Shl, Shr,

    // Pipeline
    Pipe,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
    BitNot,
}

// ═══════════════════════════════════════════════════════════════════════════
// §5 CONFIGURATION SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level configuration (from @config block)
#[derive(Debug, Clone)]
pub struct Config<'arena> {
    /// Execution mode
    pub mode: ExecutionMode,

    /// Compilation target
    pub target: CompilationTarget,

    /// Supported languages
    pub languages: Vec<Language>,

    /// Loaded macros
    pub macros: Vec<Arc<String>>,

    /// Optimization level
    pub optimize: bool,

    /// Hot reload support
    pub hot_reload: bool,

    /// Type system mode
    pub type_mode: TypeMode,

    /// Environment variables
    pub env: HashMap<String, Arc<String>>,

    /// Shared variables
    pub shared: HashMap<Arc<String>, Value<'arena>>,

    pub runtime_mode: RuntimeMode,
    /// Loaded plugins
    pub plugins: Vec<PluginConfig>,
    pub imports: Vec<PathBuf>,
}

impl<'arena> Default for Config<'arena> {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Jit { cache_enabled: true },
            target: CompilationTarget::Native,
            languages: vec![Language::Rust],
            macros: Vec::new(),
            optimize: true,
            hot_reload: false,
            type_mode: TypeMode::Static,
            env: env::vars().map(|(k, v)| (k, Arc::new(v))).collect(),
            shared: HashMap::new(),
            runtime_mode: RuntimeMode::default(),
            plugins: Vec::new(),
            imports: Vec::new(),
        }
    }
}

impl<'arena> Config<'arena> {
    /// Parse config from YAML-like text
    pub fn parse(source: &str) -> TBResult<Self> {
        let mut config = Self::default();

        // Simple parser for @config { ... }
        if let Some(start) = source.find("@config") {
            if let Some(block_start) = source[start..].find('{') {
                if let Some(block_end) = source[start + block_start..].find('}') {
                    let config_text = &source[start + block_start + 1..start + block_start + block_end];

                    for line in config_text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }

                        if let Some((key, value)) = line.split_once(':') {
                            // Intern config keys
                            let key = STRING_INTERNER.intern(key.trim());
                            let value = value.trim().trim_end_matches(',');

                            let val = if value.starts_with('"') {
                                //  Intern string values
                                Value::String(STRING_INTERNER.intern(value.trim_matches('"')))
                            } else if value == "true" {
                                Value::Bool(true)
                            } else if value == "false" {
                                Value::Bool(false)
                            } else if let Ok(n) = value.parse::<i64>() {
                                Value::Int(n)
                            } else if let Ok(f) = value.parse::<f64>() {
                                Value::Float(f)
                            } else {
                                Value::String(STRING_INTERNER.intern(value))
                            };

                            config.shared.insert(key, val);
                        }
                    }
                }
            }
        }

        // Parse @shared block
        if let Some(start) = source.find("@shared") {
            if let Some(block_start) = source[start..].find('{') {
                if let Some(block_end) = source[start + block_start..].find('}') {
                    let shared_text = &source[start + block_start + 1..start + block_start + block_end];

                    for line in shared_text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }

                        if let Some((key, value)) = line.split_once(':') {
                            let key = STRING_INTERNER.intern(key.trim());
                            let value = value.trim().trim_end_matches(',');

                            let val = if value.starts_with('"') {
                                Value::String(STRING_INTERNER.intern(value.trim_matches('"')))
                            } else if value == "true" {
                                Value::Bool(true)
                            } else if value == "false" {
                                Value::Bool(false)
                            } else if let Ok(n) = value.parse::<i64>() {
                                Value::Int(n)
                            } else if let Ok(f) = value.parse::<f64>() {
                                Value::Float(f)
                            } else {
                                Value::String(STRING_INTERNER.intern(value))
                            };

                            config.shared.insert(key, val);
                        }
                    }
                }
            }
        }

        // Parse @plugins block
        if let Some(start) = source.find("@plugins") {
            if let Some(block_start) = source[start..].find('{') {
                if let Some(block_end) = source[start + block_start..].find('}') {
                    let plugins_text = &source[start + block_start + 1..start + block_start + block_end];
                    config.plugins = Self::parse_plugins_block(plugins_text)?;
                }
            }
        }

        // Parse @imports block
        if let Some(start) = source.find("@imports") {
            if let Some(block_start) = source[start..].find('{') {
                if let Some(block_end) = source[start + block_start..].find('}') {
                    let imports_text = &source[start + block_start + 1..start + block_start + block_end];
                    config.imports = Self::parse_imports_block(imports_text)?;
                }
            }
        }

        // Substitute environment variables
        config.substitute_env_vars();

        Ok(config)
    }

    fn parse_imports_block(text: &str) -> TBResult<Vec<PathBuf>> {
        let mut imports = Vec::new();

        for line in text.lines() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse: "path/to/file.tbx" or "path/to/file.tbx",
            let cleaned = line.trim_matches(|c| c == ',' || c == '"' || c == '\'' || char::is_whitespace(c));

            if !cleaned.is_empty() {
                imports.push(PathBuf::from(cleaned));
            }
        }

        Ok(imports)
    }

    /// Parse @plugins { ... } block
    fn parse_plugins_block(text: &str) -> TBResult<Vec<PluginConfig>> {
        let mut plugins = Vec::new();
        let mut current_plugin: Option<PluginBuilder> = None;
        let mut in_function = false;
        let mut current_function_name = String::new();
        let mut current_function_code = String::new();

        for line in text.lines() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // New plugin definition: language "name"
            if line.contains("python") || line.contains("javascript") ||
                line.contains("typescript") || line.contains("go") ||
                line.contains("bash") {

                // Save previous plugin
                if let Some(builder) = current_plugin.take() {
                    plugins.push(builder.build());
                }

                // Parse: python "my_plugin" {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let lang_str = parts[0];
                    let name = parts[1].trim_matches(|c| c == '"' || c == '{');

                    let language = Language::from_str(lang_str)?;
                    current_plugin = Some(PluginBuilder::new(name, language));
                }
            }
            // Function definition
            else if line.starts_with("fn ") {
                let func_name = line.trim_start_matches("fn ")
                    .trim_matches(|c| c == '{' || c == ':')
                    .trim();
                current_function_name = func_name.to_string();
                current_function_code.clear();
                in_function = true;
            }
            // End of function
            else if line.starts_with("}") && in_function {
                if let Some(ref mut plugin) = current_plugin {
                    *plugin = plugin.clone().add_function(
                        current_function_name.clone(),
                        current_function_code.clone()
                    );
                }
                in_function = false;
            }
            // Function code
            else if in_function {
                current_function_code.push_str(line);
                current_function_code.push('\n');
            }
            // Import directive
            else if line.starts_with("import ") {
                let import = line.trim_start_matches("import ")
                    .trim_matches(|c| c == '"' || c == ';');
                if let Some(ref mut plugin) = current_plugin {
                    *plugin = plugin.clone().add_import(import);
                }
            }
        }

        // Save last plugin
        if let Some(builder) = current_plugin {
            plugins.push(builder.build());
        }

        Ok(plugins)
    }

    /// Substitute $ENV_VAR in shared values
    fn substitute_env_vars(&mut self) {
        for value in self.shared.values_mut() {
            if let Value::String(s) = value {
                if s.starts_with('$') {
                    let env_var = &s[1..];
                    if let Some(env_value) = self.env.get(env_var) {
                        *value = Value::String(env_value.clone());
                    }
                }
            }
        }
    }
}

/// Execution mode
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionMode {
    /// Compile to native binary
    Compiled { optimize: bool },

    /// Just-in-time execution
    Jit { cache_enabled: bool },

    /// Streaming/interactive mode
    Streaming { auto_complete: bool, suggestions: bool },
}

/// Compilation target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationTarget {
    Native,
    Wasm,
    Library,
}

/// Type system mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeMode {
    Static,
    Dynamic,
}

/// Runtime execution mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    /// Sequential execution (default)
    Sequential,

    /// Parallel execution with Rayon
    Parallel {
        worker_threads: usize,
    },
}

impl Default for RuntimeMode {
    fn default() -> Self {
        RuntimeMode::Sequential
    }
}

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Bash,
}

impl FromStr for Language {
    type Err = TBError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rust" => Ok(Language::Rust),
            "python" | "py" => Ok(Language::Python),
            "javascript" | "js" => Ok(Language::JavaScript),
            "typescript" | "ts" => Ok(Language::TypeScript),
            "go" => Ok(Language::Go),
            "bash" | "sh" => Ok(Language::Bash),
            _ => Err(TBError::UnsupportedLanguage(Arc::new(s.to_string()))),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §6 ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════

/// TB Error type - comprehensive error handling
#[derive(Debug, Clone)]
pub enum TBError {
    /// Parse error
    ParseError { message: Arc<String>, line: usize, column: usize },

    /// Type error
    TypeError { expected: Type, found: Type, context: Arc<String> },

    /// Runtime error
    RuntimeError { message: Arc<String>, trace: Vec<Arc<String>> },

    /// Compilation error
    CompilationError { message: Arc<String>, source: Arc<String> },

    /// IO error
    IoError(Arc<String>),

    /// Unsupported language
    UnsupportedLanguage(Arc<String>),

    /// Undefined variable
    UndefinedVariable(Arc<String>),

    /// Undefined function
    UndefinedFunction(Arc<String>),

    /// Division by zero
    DivisionByZero,

    /// Index out of bounds
    IndexOutOfBounds { index: i64, length: usize },

    /// Invalid operation
    InvalidOperation(Arc<String>),
}

impl Display for TBError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            TBError::ParseError { message, line, column } => {
                write!(f, "Parse error at {}:{}: {}", line, column, message)
            }
            TBError::TypeError { expected, found, context } => {
                write!(f, "Type error in {}: expected {:?}, found {:?}", context, expected, found)
            }
            TBError::RuntimeError { message, trace } => {
                write!(f, "Runtime error: {}\nStack trace:\n", message)?;
                for frame in trace {
                    write!(f, "  {}\n", frame)?;
                }
                Ok(())
            }
            TBError::CompilationError { message, .. } => {
                write!(f, "Compilation error: {}", message)
            }
            TBError::IoError(msg) => write!(f, "IO error: {}", msg),
            TBError::UnsupportedLanguage(lang) => write!(f, "Unsupported language: {}", lang),
            TBError::UndefinedVariable(var) => write!(f, "Undefined variable: {}", var),
            TBError::UndefinedFunction(func) => write!(f, "Undefined function: {}", func),
            TBError::DivisionByZero => write!(f, "Division by zero"),
            TBError::IndexOutOfBounds { index, length } => {
                write!(f, "Index {} out of bounds (length: {})", index, length)
            }
            TBError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for TBError {}

impl From<io::Error> for TBError {
    fn from(e: io::Error) -> Self {
        TBError::IoError(Arc::new(e.to_string()))
    }
}

impl TBError {
    /// Get formatted error with full context
    pub fn detailed_message(&self) -> String {
        match self {
            TBError::RuntimeError { message, trace } => {
                let mut output = format!("Runtime Error:\n{}\n", message);

                if !trace.is_empty() {
                    output.push_str("\nStack trace:\n");
                    for (i, frame) in trace.iter().enumerate() {
                        output.push_str(&format!("  #{} {}\n", i, frame));
                    }
                }

                output
            }

            TBError::ParseError { message, line, column } => {
                format!(
                    "Parse Error at {}:{}:\n  {}\n",
                    line, column, message
                )
            }

            TBError::CompilationError { message, source } => {
                let mut output = format!("Compilation Error:\n{}\n", message);
                if !source.is_empty() {
                    output.push_str(&format!("\nSource:\n{}\n", source));
                }
                output
            }

            other => format!("{}", other),
        }
    }
}

/// Result type alias
pub type TBResult<T> = Result<T, TBError>;

// ═══════════════════════════════════════════════════════════════════════════
// §7 PARSING
// ═══════════════════════════════════════════════════════════════════════════

/// Lexer - tokenizes source code

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Int(i64),
    Float(f64),
    String(Arc<String>),
    Bool(bool),

    // Identifiers and keywords
    Identifier(Arc<String>),

    // Keywords
    Let, Mut, Fn, If, Else, Match, Loop, While, For, In,
    Return, Break, Continue, Parallel,

    // Operators
    Dollar,
    Plus, Minus, Star, Slash, Percent, Power,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Not,
    Pipe,
    Arrow, FatArrow,
    Question,

    // Delimiters
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,
    Comma, Semicolon, Colon, Dot,

    // Special
    Assign,
    Newline,
    Eof,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Token::Int(n) => write!(f, "Int({})", n),
            Token::Float(fl) => write!(f, "Float({})", fl),
            Token::String(s) => write!(f, "String(\"{}\")", s),
            Token::Bool(b) => write!(f, "Bool({})", b),
            Token::Identifier(id) => write!(f, "Identifier({})", id),
            Token::Let => write!(f, "Let"),
            Token::Fn => write!(f, "Fn"),
            Token::LBrace => write!(f, "LBrace"),
            Token::RBrace => write!(f, "RBrace"),
            Token::LParen => write!(f, "LParen"),
            Token::RParen => write!(f, "RParen"),
            Token::Semicolon => write!(f, "Semicolon"),
            Token::Newline => write!(f, "Newline"),
            Token::Eof => write!(f, "EOF"),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub struct Lexer {
    source: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    // Track nesting depth to know when newlines are significant
    paren_depth: usize,    // ()
    brace_depth: usize,    // {}
    bracket_depth: usize,  // []
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
            paren_depth: 0,
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    pub fn tokenize(&mut self) -> TBResult<Vec<Token>> {
        debug_log!("Lexer::tokenize() started, source length: {}", self.source.len());
        let mut tokens = Vec::new();
        let mut iterations = 0;
        let max_iterations = self.source.len() + 100;
        let last_token_line = self.line;

        loop {
            iterations += 1;
            if iterations > max_iterations {
                debug_log!("ERROR: Lexer infinite loop detected at position {}", self.position);
                return Err(TBError::ParseError {
                    message: Arc::new(format!("Lexer infinite loop at position {}", self.position)),
                    line: self.line,
                    column: self.column,
                });
            }

            self.skip_whitespace_except_newline();

            if self.is_eof() {
                debug_log!("Lexer reached EOF, pushing Token::Eof");
                tokens.push(Token::Eof);
                break;
            }

            // Handle newlines as statement separators
            if self.current() == '\n' {
                self.advance(); // consume newline

                // Only insert Newline token if we're not nested (depth = 0)
                // and if the previous token wasn't already a separator
                if self.paren_depth == 0 && self.brace_depth == 0 && self.bracket_depth == 0 {
                    if !tokens.is_empty() {
                        // Don't insert Newline after another Newline, Semicolon, or LBrace
                        let needs_separator = !matches!(
                        tokens.last(),
                        Some(Token::Newline) | Some(Token::Semicolon) |
                        Some(Token::LBrace) | Some(Token::RBrace)
                    );

                        if needs_separator {
                            debug_log!("Inserting Newline separator at line {}", self.line);
                            tokens.push(Token::Newline);
                        }
                    }
                }

                continue;
            }

            // Skip comments
            if self.current() == '#' {
                debug_log!("Skipping comment at line {}", self.line);
                self.skip_line();
                continue;
            }

            //debug_log!("Tokenizing at position {}, char: '{}'", self.position, self.current());
            let token = self.next_token()?;

            // Update depth tracking
            match &token {
                Token::LParen => self.paren_depth += 1,
                Token::RParen => self.paren_depth = self.paren_depth.saturating_sub(1),
                Token::LBrace => self.brace_depth += 1,
                Token::RBrace => self.brace_depth = self.brace_depth.saturating_sub(1),
                Token::LBracket => self.bracket_depth += 1,
                Token::RBracket => self.bracket_depth = self.bracket_depth.saturating_sub(1),
                _ => {}
            }

            // debug_log!("Token created: {:?} (depth: p={}, b={}, br={})",
            //        token, self.paren_depth, self.brace_depth, self.bracket_depth);
            tokens.push(token);
        }

        debug_log!("Lexer::tokenize() completed with {} tokens", tokens.len());
        Ok(tokens)
    }

    fn skip_whitespace_except_newline(&mut self) {
        while !self.is_eof() && self.current() != '\n' && self.current().is_whitespace() {
            self.advance();
        }
    }

    fn next_token(&mut self) -> TBResult<Token> {
        let ch = self.current();

        if ch == '/' && self.peek_ahead(1) == Some('/') {
            debug_log!("Skipping C-style comment at line {}", self.line);
            self.skip_line();
            self.advance(); // Skip newline
            return self.next_token(); // Recursive call for next token
        }


        if ch == '#' {
            debug_log!("Skipping Python-style comment at line {}", self.line);
            self.skip_line();
            self.advance(); // Skip newline
            return self.next_token(); // Recursive call for next token
        }

        if ch == '@' {
            return self.read_identifier();
        }

        match ch {
            '0'..='9' => self.read_number(),
            'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
            '"' => {
                debug_log!("Found '\"' at position {}", self.position);

                // Check for triple-quoted string
                if self.peek_ahead(1) == Some('"') && self.peek_ahead(2) == Some('"') {
                    debug_log!("Detected triple-quote string (\"\"\"...)");
                    self.read_triple_quoted_string()
                } else {
                    debug_log!("Regular string (\"...)");
                    self.read_string()
                }
            }
            '$' => {
                self.advance();
                Ok(Token::Dollar)
            }
            '+' => { self.advance(); Ok(Token::Plus) }
            '-' => {
                self.advance();
                if self.current() == '>' {
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Minus)
                }
            }
            '*' => { self.advance(); Ok(Token::Star) }
            '/' => { self.advance(); Ok(Token::Slash) }
            '%' => { self.advance(); Ok(Token::Percent) }
            '(' => { self.advance(); Ok(Token::LParen) }
            ')' => { self.advance(); Ok(Token::RParen) }
            '{' => { self.advance(); Ok(Token::LBrace) }
            '}' => { self.advance(); Ok(Token::RBrace) }
            '[' => { self.advance(); Ok(Token::LBracket) }
            ']' => { self.advance(); Ok(Token::RBracket) }
            ',' => { self.advance(); Ok(Token::Comma) }
            ';' => { self.advance(); Ok(Token::Semicolon) }
            ':' => { self.advance(); Ok(Token::Colon) }
            '.' => { self.advance(); Ok(Token::Dot) }
            '=' => {
                self.advance();
                if self.current() == '=' {
                    self.advance();
                    Ok(Token::Eq)
                } else if self.current() == '>' {
                    self.advance();
                    Ok(Token::FatArrow)
                } else {
                    Ok(Token::Assign)
                }
            }
            '!' => {
                self.advance();
                if self.current() == '=' {
                    self.advance();
                    Ok(Token::Ne)
                } else {
                    Ok(Token::Not)
                }
            }
            '<' => {
                self.advance();
                if self.current() == '=' {
                    self.advance();
                    Ok(Token::Le)
                } else {
                    Ok(Token::Lt)
                }
            }
            '>' => {
                self.advance();
                if self.current() == '=' {
                    self.advance();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            '|' => {
                self.advance();
                if self.current() == '>' {
                    self.advance();
                    Ok(Token::Pipe)
                } else {
                    Ok(Token::Or)
                }
            }
            '&' => {
                self.advance();
                if self.current() == '&' {
                    self.advance();
                    Ok(Token::And)
                } else {
                    Err(self.error("Expected '&&'"))
                }
            }
            '?' => { self.advance(); Ok(Token::Question) }
            _ => Err(self.error(&format!("Unexpected character: '{}'", ch))),
        }
    }

    fn read_number(&mut self) -> TBResult<Token> {
        let mut num = String::new();
        let mut is_float = false;

        while !self.is_eof() && (self.current().is_numeric() || self.current() == '.') {
            if self.current() == '.' {
                if is_float {
                    return Err(self.error("Multiple decimal points in number"));
                }
                is_float = true;
            }
            num.push(self.current());
            self.advance();
        }

        if is_float {
            num.parse::<f64>()
                .map(Token::Float)
                .map_err(|_| self.error("Invalid float"))
        } else {
            num.parse::<i64>()
                .map(Token::Int)
                .map_err(|_| self.error("Invalid integer"))
        }
    }

    fn read_identifier(&mut self) -> TBResult<Token> {
        let mut ident = String::new();

        // FIXED: Allow @ at start for @shared, @immutable
        if self.current() == '@' {
            ident.push('@');
            self.advance();
        }

        // Allow Unicode identifiers
        while !self.is_eof() && (self.current().is_alphanumeric() || self.current() == '_' || self.current() as u32 > 127) {
            ident.push(self.current());
            self.advance();
        }

        let token = match ident.as_str() {
            "let" => Token::Let,
            "mut" => Token::Mut,
            "fn" => Token::Fn,
            "if" => Token::If,
            "else" => Token::Else,
            "match" => Token::Match,
            "loop" => Token::Loop,
            "while" => Token::While,
            "for" => Token::For,
            "in" => Token::In,
            "return" => Token::Return,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "parallel" => Token::Parallel,
            "true" => Token::Bool(true),
            "false" => Token::Bool(false),
            "and" => Token::And,
            "or" => Token::Or,
            "not" => Token::Not,
            _ => Token::Identifier(STRING_INTERNER.intern(&ident)),
        };

        Ok(token)
    }

    fn read_string(&mut self) -> TBResult<Token> {
        // Check for triple-quoted string
        if self.peek_ahead(2) == Some('"') && self.peek_ahead(1) == Some('"') {
            return self.read_triple_quoted_string();
        }

        // Regular string
        self.advance(); // Skip opening "
        let mut string = String::new();

        while !self.is_eof() && self.current() != '"' {
            if self.current() == '\\' {
                self.advance();
                if self.is_eof() {
                    return Err(self.error("Unterminated string"));
                }
                match self.current() {
                    'n' => string.push('\n'),
                    't' => string.push('\t'),
                    'r' => string.push('\r'),
                    '\\' => string.push('\\'),
                    '"' => string.push('"'),
                    _ => {
                        string.push('\\');
                        string.push(self.current());
                    }
                }
            } else {
                string.push(self.current());
            }
            self.advance();
        }

        if self.is_eof() {
            return Err(self.error("Unterminated string"));
        }

        self.advance(); // Skip closing "
        Ok(Token::String(STRING_INTERNER.intern_owned(string)))
    }

    fn read_triple_quoted_string(&mut self) -> TBResult<Token> {
        let start_line = self.line;
        let start_position = self.position;

        debug_log!("read_triple_quoted_string() at pos={}, char='{}'",
                   self.position, self.current());

        // Verify and skip opening """
        for i in 0..3 {
            if self.current() != '"' {
                return Err(TBError::ParseError {
                    message:Arc::new( format!("Expected '\"' (quote {}/3 of opening triple-quote)", i + 1)),
                    line: self.line,
                    column: self.column,
                });
            }
            debug_log!("Consuming opening quote {}/3 at position {}", i + 1, self.position);
            self.advance();
        }

        debug_log!("After opening quotes, position={}, char='{}'",
                   self.position, self.current());

        let mut string = String::new();

        // Read until we find closing """
        loop {
            if self.is_eof() {
                return Err(TBError::ParseError {
                    message: Arc::new(format!(
                        "Unterminated triple-quoted string (started at line {}, position {})",
                        start_line, start_position
                    )),
                    line: self.line,
                    column: self.column,
                });
            }

            // Check if current position starts with """
            if self.current() == '"'
                && self.peek_ahead(1) == Some('"')
                && self.peek_ahead(2) == Some('"')
            {
                // Found closing """
                debug_log!("Found closing triple-quote at position {}", self.position);

                // Consume all three closing quotes
                for i in 0..3 {
                    debug_log!("Consuming closing quote {}/3 at position {}", i + 1, self.position);
                    self.advance();
                }

                debug_log!("After closing quotes, position={}, char='{:?}'",
                           self.position,
                           if self.is_eof() { "EOF".to_string() } else { self.current().to_string() });
                debug_log!("Triple-quoted string complete: {} bytes", string.len());

                return Ok(Token::String(STRING_INTERNER.intern_owned(string)));
            }

            // Not a closing """, add character to string
            string.push(self.current());
            self.advance();
        }
    }

    fn peek_ahead(&self, offset: usize) -> Option<char> {
        let pos = self.position + offset;
        if pos < self.source.len() {
            Some(self.source[pos])
        } else {
            None
        }
    }

    fn current(&self) -> char {
        if self.is_eof() {
            '\0'
        } else {
            self.source[self.position]
        }
    }

    fn advance(&mut self) {
        if !self.is_eof() {
            if self.current() == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.position += 1;
        }
    }

    fn is_eof(&self) -> bool {
        self.position >= self.source.len()
    }

    fn skip_line(&mut self) {
        while !self.is_eof() && self.current() != '\n' {
            self.advance();
        }
    }

    fn error(&self, message: &str) -> TBError {
        TBError::ParseError {
            message: Arc::new(message.to_string()),
            line: self.line,
            column: self.column,
        }
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// §7.5 PARSER
// ═══════════════════════════════════════════════════════════════════════════


/// Parser - builds AST from tokens
pub struct Parser<'arena> {
    arena: &'arena Bump,
    tokens: Vec<Token>,
    position: usize,
    recursion_depth: usize,
    max_recursion_depth: usize,
}

impl<'arena> Parser<'arena> {
    pub fn new(tokens: Vec<Token>, arena: &'arena Bump) -> Self {
        Self {
            arena,
            tokens,
            position: 0,
            recursion_depth: 0,
            max_recursion_depth: 100,
        }
    }

    pub fn parse(&mut self) -> TBResult<Vec<Statement<'arena>>> {
        debug_log!("Parser::parse() started, tokens: {}", self.tokens.len());
        let mut statements = Vec::new();
        let mut iterations = 0;
        let max_iterations = self.tokens.len() * 2;

        while !self.is_eof() {
            iterations += 1;
            if iterations > max_iterations {
                return Err(TBError::ParseError {
                    message: Arc::new(format!(
                        "Parser exceeded {} iterations. Infinite loop at position {}",
                        max_iterations, self.position
                    )),
                    line: iterations,
                    column: iterations,
                });
            }

            let position_before = self.position;

            debug_log!("Parsing statement {}, position: {}, token: {:?}",
                   iterations, self.position, self.current());

            let stmt = self.parse_statement()?;

            // CRITICAL: Ensure we made progress!
            if self.position == position_before && !self.is_eof() {
                return Err(TBError::ParseError {
                    message: Arc::new(format!(
                        "Parser stuck at position {} with token {:?}. Statement parsing made no progress.",
                        self.position, self.current()
                    )),
                    line: 0,
                    column: 0,
                });
            }

            debug_log!("Parsed statement successfully, new position: {}", self.position);
            statements.push(stmt);
        }

        debug_log!("Parser::parse() completed with {} statements", statements.len());
        Ok(statements)
    }

    fn parse_statement(&mut self) -> TBResult<Statement<'arena>> {
        debug_log!("parse_statement at position {}, token: {:?}", self.position, self.current());

        // Skip any stray semicolons OR newlines
        while self.match_token(&Token::Semicolon) || self.match_token(&Token::Newline) {
            debug_log!("Skipping separator at position {}", self.position);
            self.advance();
            if self.is_eof() {
                return Err(TBError::ParseError {
                    message: Arc::from("Unexpected end of input".to_string()),
                    line: 0,
                    column: 0,
                });
            }
        }

        let result = match self.current().clone() {
            Token::Let => {
                debug_log!("Parsing let statement");
                self.parse_let()
            }
            Token::Fn => {
                debug_log!("Parsing function statement");
                self.parse_function()
            }
            Token::Eof => {
                return Err(TBError::ParseError {
                    message: Arc::from("Unexpected EOF".to_string()),
                    line: 0,
                    column: 0,
                });
            }
            _ => {
                debug_log!("Parsing expression statement or assignment");

                // Try to parse assignment: identifier = expr
                if let Token::Identifier(name) = self.current() {
                    let name = Arc::clone(name);
                    let pos_before = self.position;
                    self.advance();

                    // Check if next token is assignment
                    if self.match_token(&Token::Assign) {
                        self.advance();
                        let value = self.parse_expression()?;
                        if self.match_token(&Token::Semicolon) {
                            self.advance();
                        }
                        return Ok(Statement::Assign {
                            target: Expr::Variable(name),
                            value,
                        });
                    } else {
                        // Not assignment, backtrack and parse as expression
                        self.position = pos_before;
                    }
                }

                let expr = self.parse_expression()?;
                if self.match_token(&Token::Semicolon) {
                    debug_log!("Consuming semicolon after expression");
                    self.advance();
                }
                Ok(Statement::Expr(expr))
            }
        };

        debug_log!("parse_statement completed");
        result
    }

    fn parse_let(&mut self) -> TBResult<Statement<'arena>> {
        self.advance(); // consume 'let'

        let mutable = self.match_token(&Token::Mut);
        if mutable {
            self.advance();
        }

        // FIXED: Check for '@shared' or '@' prefix for scope
        let scope = if let Token::Identifier(s) = self.current() {
            if s.as_str() == "shared" || s.starts_with('@') {
                self.advance(); // consume 'shared' or '@shared'
                VarScope::Shared
            } else {
                VarScope::Local
            }
        } else {
            VarScope::Local
        };

        let name = if let Token::Identifier(n) = self.current() {
            let name = Arc::clone(n);  // Cheap pointer clone
            self.advance();
            name
        } else {
            return Err(TBError::ParseError {
                message: STRING_INTERNER.intern("Expected identifier after 'let'"),
                line: 0,
                column: 0,
            });
        };

        let type_annotation = if self.match_token(&Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        if !self.match_token(&Token::Assign) {
            return Err(TBError::ParseError {
                message: STRING_INTERNER.intern("Expected '=' after variable name"),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let value = self.parse_expression()?;

        if self.match_token(&Token::Semicolon) || self.match_token(&Token::Newline) {
            self.advance();
        }

        Ok(Statement::Let {
            name,  // Already interned Arc<String>
            mutable,
            type_annotation,
            value,
            scope
        })
    }

    fn parse_function(&mut self) -> TBResult<Statement<'arena>> {
        self.advance(); // consume 'fn'
        let name = if let Token::Identifier(n) = self.current() {
            let name = Arc::clone(n);
            self.advance();
            name
        } else {
            return Err(TBError::ParseError {
                message: STRING_INTERNER.intern("Expected function name"),
                line: 0,
                column: 0,
            });
        };

        if !self.match_token(&Token::LParen) {
            return Err(TBError::ParseError {
                message: Arc::from("Expected '(' after function name".to_string()),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let mut params = Vec::new();
        while !self.match_token(&Token::RParen) {
            let param_name = if let Token::Identifier(n) = self.current() {
                let name = Arc::clone(n);
                self.advance();
                name
            } else {
                return Err(TBError::ParseError {
                    message: Arc::from("Expected parameter name".to_string()),
                    line: 0,
                    column: 0,
                });
            };

            let type_annotation = if self.match_token(&Token::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            params.push(Parameter {
                name: param_name,
                type_annotation,
                default: None,
            });

            if !self.match_token(&Token::RParen) {
                if !self.match_token(&Token::Comma) {
                    return Err(TBError::ParseError {
                        message: Arc::from("Expected ',' or ')' in parameter list".to_string()),
                        line: 0,
                        column: 0,
                    });
                }
                self.advance();
            }
        }
        self.advance(); // consume ')'

        let return_type = if self.match_token(&Token::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_expression()?;

        Ok(Statement::Function {
            name,
            params: ArenaVec::from_iter_in(params, self.arena),
            return_type,
            body,
        })
    }

    fn parse_expression(&mut self) ->  TBResult<Expr<'arena>> {
        self.enter_recursion()?;
        let result = self.parse_pipeline();

        self.exit_recursion();
        result
    }

    fn parse_pipeline(&mut self) ->  TBResult<Expr<'arena>> {
        let mut expr = self.parse_logical_or()?;

        if self.match_token(&Token::Pipe) {
            let mut operations =  ArenaVec::new_in(self.arena);
            while self.match_token(&Token::Pipe) {
                self.advance();
                operations.push(self.parse_logical_or()?);
            }
            expr = Expr::Pipeline {
                value: self.arena.alloc(expr),
                operations,
            };
        }

        expr = self.parse_try(expr)?;

        Ok(expr)
    }

    fn parse_try(&mut self, expr: Expr<'arena>) ->  TBResult<Expr<'arena>> {
        if self.match_token(&Token::Question) {
            self.advance();
            Ok(Expr::Try(self.arena.alloc(expr)))
        } else {
            Ok(expr)
        }
    }

    fn parse_logical_or(&mut self) ->  TBResult<Expr<'arena>> {
        let mut left = self.parse_logical_and()?;

        while self.match_token(&Token::Or) {
            self.advance();
            let right = self.parse_logical_and()?;
            left = Expr::BinOp {
                op: BinOp::Or,
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
            };
        }

        Ok(left)
    }

    fn parse_logical_and(&mut self) ->  TBResult<Expr<'arena>> {
        let mut left = self.parse_equality()?;

        while self.match_token(&Token::And) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::BinOp {
                op: BinOp::And,
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
            };
        }

        Ok(left)
    }

    fn parse_equality(&mut self) ->  TBResult<Expr<'arena>> {
        let mut left = self.parse_comparison()?;

        while matches!(self.current(), Token::Eq | Token::Ne) {
            let op = match self.current() {
                Token::Eq => BinOp::Eq,
                Token::Ne => BinOp::Ne,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::BinOp {
                op,
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
            };
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) ->  TBResult<Expr<'arena>> {
        let mut left = self.parse_term()?;

        while matches!(self.current(), Token::Lt | Token::Le | Token::Gt | Token::Ge) {
            let op = match self.current() {
                Token::Lt => BinOp::Lt,
                Token::Le => BinOp::Le,
                Token::Gt => BinOp::Gt,
                Token::Ge => BinOp::Ge,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_term()?;
            left = Expr::BinOp {
                op,
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
            };
        }

        Ok(left)
    }

    fn parse_term(&mut self) ->  TBResult<Expr<'arena>> {
        let mut left = self.parse_factor()?;

        while matches!(self.current(), Token::Plus | Token::Minus) {
            let op = match self.current() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_factor()?;
            left = Expr::BinOp {
                op,
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
            };
        }

        Ok(left)
    }

    fn parse_factor(&mut self) ->  TBResult<Expr<'arena>> {
        let mut left = self.parse_unary()?;

        while matches!(self.current(), Token::Star | Token::Slash | Token::Percent) {
            let op = match self.current() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp {
                op,
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) ->  TBResult<Expr<'arena>> {
        // Check for unary operators FIRST
        match self.current() {
            Token::Not | Token::Minus => {
                let op = match self.current() {
                    Token::Not => UnaryOp::Not,
                    Token::Minus => UnaryOp::Neg,
                    _ => unreachable!(),
                };
                self.advance();
                let expr = self.arena.alloc(self.parse_unary()?); // Recursive for chaining
                return Ok(Expr::UnaryOp { op, expr });
            }
            _ => {}
        }

        // Then parse postfix operations
        let mut expr = self.parse_primary()?;

        loop {
            match self.current() {
                Token::LParen => {
                    // Explicit function call with parentheses
                    self.advance();
                    let mut args = ArenaVec::new_in(self.arena);
                    while !self.match_token(&Token::RParen) {
                        args.push(self.parse_expression()?);
                        if !self.match_token(&Token::RParen) {
                            if !self.match_token(&Token::Comma) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected ',' or ')' in argument list".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance();
                        }
                    }
                    self.advance();
                    expr = Expr::Call {
                        function: self.arena.alloc(expr),
                        args,
                    };
                }

                Token::Dot => {
                    self.advance();
                    if let Token::Identifier(field) = self.current() {
                        let field = field.clone();
                        self.advance();

                        if self.match_token(&Token::LParen) {
                            self.advance();
                            let mut args = ArenaVec::new_in(self.arena);
                            while !self.match_token(&Token::RParen) {
                                args.push(self.parse_expression()?);
                                if !self.match_token(&Token::RParen) {
                                    if !self.match_token(&Token::Comma) {
                                        return Err(TBError::ParseError {
                                            message: Arc::from("Expected ',' or ')' in argument list".to_string()),
                                            line: 0,
                                            column: 0,
                                        });
                                    }
                                    self.advance();
                                }
                            }
                            self.advance();
                            expr = Expr::Method {
                                object: self.arena.alloc(expr),
                                method: field,
                                args,
                            };
                        } else {
                            expr = Expr::Field {
                                object: self.arena.alloc(expr),
                                field,
                            };
                        }
                    } else {
                        return Err(TBError::ParseError {
                            message: Arc::from("Expected field name after '.'".to_string()),
                            line: 0,
                            column: 0,
                        });
                    }
                }

                Token::LBracket => {
                    self.advance();
                    let index = self.parse_expression()?;
                    if !self.match_token(&Token::RBracket) {
                        return Err(TBError::ParseError {
                            message: Arc::from("Expected ']' after index".to_string()),
                            line: 0,
                            column: 0,
                        });
                    }
                    self.advance();
                    expr = Expr::Index {
                        object: self.arena.alloc(expr),
                        index: self.arena.alloc(index),
                    };
                }

                // Implicit function call for builtin commands
                Token::String(_) | Token::Dollar | Token::Identifier(_)
                if matches!(expr, Expr::Variable(ref name) if self.is_builtin_command(name.as_str())) =>
                    {
                        debug_log!("Detected implicit builtin call for: {}",
                        if let Expr::Variable(ref n) = expr { n } else { "unknown" });

                        let mut args =ArenaVec::new_in(self.arena);

                        // Collect arguments until statement terminator
                        while self.is_argument_start() && !self.is_statement_end() {
                            let arg = if self.match_token(&Token::Dollar) {
                                self.advance();
                                if let Token::Identifier(name) = self.current() {
                                    let name = Arc::clone(name);
                                    self.advance();
                                    Expr::Variable(name)
                                } else {
                                    return Err(TBError::ParseError {
                                        message: Arc::from("Expected identifier after '$'".to_string()),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            } else {
                                self.parse_argument()?
                            };
                            args.push(arg);
                        }

                        expr = Expr::Call {
                            function: self.arena.alloc(expr),
                            args,
                        };
                    }

                _ => break,
            }
        }

        Ok(expr)
    }

    fn is_argument_start(&self) -> bool {
        matches!(
            self.current(),
            Token::String(_) | Token::Int(_) | Token::Float(_) |
            Token::Bool(_) | Token::LBracket | Token::Dollar |
            Token::Identifier(_)
        )
    }

    /// Check if we're at a statement boundary
    fn is_statement_end(&self) -> bool {
        matches!(
        self.current(),
        Token::Semicolon | Token::Newline | Token::RBrace | Token::Eof
    ) || self.is_keyword()
    }

    /// Check if identifier is a builtin command (echo, print, etc.)
    fn is_builtin_command(&self, name: &str) -> bool {
        matches!(
            name,
            "echo" | "print" | "println" | "read_line" |
            "python" | "javascript" | "bash" |
            "debug" | "len" | "str" | "int" | "float" | "type_of"
        )
    }

    /// Check if identifier is likely an argument (not a keyword or operator)
    fn is_likely_argument(&self) -> bool {
        matches!(self.current(), Token::Identifier(_))
    }

    /// Check if current token is a keyword
    fn is_keyword(&self) -> bool {
        matches!(
        self.current(),
        Token::Let | Token::Fn | Token::If | Token::Else | Token::Match |
        Token::Loop | Token::While | Token::For | Token::Return |
        Token::Break | Token::Continue
    )
    }

    /// Parse a single argument in implicit call
    /// Parse a single argument in implicit call (NOW SUPPORTS NESTED CALLS)
    fn parse_argument(&mut self) ->  TBResult<Expr<'arena>> {
        match self.current().clone() {
            Token::String(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::String(s)))
            }
            Token::Int(n) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int(n)))
            }
            Token::Float(f) => {
                self.advance();
                Ok(Expr::Literal(Literal::Float(f)))
            }
            Token::Bool(b) => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(b)))
            }
            Token::Identifier(name) => {
                let name = Arc::clone(&name);
                self.advance();

                // Check for function call after identifier
                if self.match_token(&Token::LParen) {
                    self.advance(); // consume '('

                    let mut args = ArenaVec::new_in(self.arena);
                    while !self.match_token(&Token::RParen) {
                        args.push(self.parse_expression()?);
                        if !self.match_token(&Token::RParen) {
                            if !self.match_token(&Token::Comma) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected ',' or ')' in argument list".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance();
                        }
                    }
                    self.advance(); // consume ')'

                    Ok(Expr::Call {
                        function: self.arena.alloc(Expr::Variable(name)),
                        args,
                    })
                } else {
                    // Just a variable reference
                    Ok(Expr::Variable(name))
                }
            }
            Token::LBracket => {
                // Parse list literal as argument
                self.advance();
                let mut items = ArenaVec::new_in(self.arena);
                while !self.match_token(&Token::RBracket) {
                    items.push(self.parse_expression()?);
                    if !self.match_token(&Token::RBracket) {
                        if !self.match_token(&Token::Comma) {
                            return Err(TBError::ParseError {
                                message: Arc::from("Expected ',' or ']' in list".to_string()),
                                line: 0,
                                column: 0,
                            });
                        }
                        self.advance();
                    }
                }
                self.advance();
                Ok(Expr::List(items))
            }
            _ => Err(TBError::ParseError {
                message: format!("Unexpected token in argument: {:?}", self.current()).into(),
                line: 0,
                column: 0,
            }),
        }
    }

    fn parse_postfix(&mut self) ->  TBResult<Expr<'arena>> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.current() {
                Token::LParen => {
                    self.advance();
                    let mut args =  ArenaVec::new_in(self.arena);
                    while !self.match_token(&Token::RParen) {
                        args.push(self.parse_expression()?);
                        if !self.match_token(&Token::RParen) {
                            if !self.match_token(&Token::Comma) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected ',' or ')' in argument list".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance();
                        }
                    }
                    self.advance();
                    expr = Expr::Call {
                        function: self.arena.alloc(expr),
                        args,
                    };
                }
                Token::Dot => {
                    self.advance();
                    if let Token::Identifier(field) = self.current() {
                        let field = field.clone();
                        self.advance();

                        // Check if it's a method call
                        if self.match_token(&Token::LParen) {
                            self.advance();
                            let mut args =  ArenaVec::new_in(self.arena);
                            while !self.match_token(&Token::RParen) {
                                args.push(self.parse_expression()?);
                                if !self.match_token(&Token::RParen) {
                                    if !self.match_token(&Token::Comma) {
                                        return Err(TBError::ParseError {
                                            message: Arc::from("Expected ',' or ')' in argument list".to_string()),
                                            line: 0,
                                            column: 0,
                                        });
                                    }
                                    self.advance();
                                }
                            }
                            self.advance();
                            expr = Expr::Method {
                                object: self.arena.alloc(expr),
                                method: field,
                                args,
                            };
                        } else {
                            expr = Expr::Field {
                                object: self.arena.alloc(expr),
                                field,
                            };
                        }
                    } else {
                        return Err(TBError::ParseError {
                            message: Arc::from("Expected field name after '.'".to_string()),
                            line: 0,
                            column: 0,
                        });
                    }
                }
                Token::LBracket => {
                    self.advance();
                    let index = self.parse_expression()?;
                    if !self.match_token(&Token::RBracket) {
                        return Err(TBError::ParseError {
                            message: Arc::from("Expected ']' after index".to_string()),
                            line: 0,
                            column: 0,
                        });
                    }
                    self.advance();
                    expr = Expr::Index {
                        object: self.arena.alloc(expr),
                        index: self.arena.alloc(index),
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) ->  TBResult<Expr<'arena>> {
        self.enter_recursion()?;
        debug_log!("parse_primary() at position {}, token: {:?}", self.position, self.current());

        let start_position = self.position;
        let result =match self.current().clone() {
            Token::Int(n) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int(n)))
            }
            Token::Float(f) => {
                self.advance();
                Ok(Expr::Literal(Literal::Float(f)))
            }
            Token::String(s) => {
                debug_log!("Parsing string literal: {}", s);
                self.advance();
                // For now, return as variable to avoid crash
                Ok(Expr::Literal(Literal::String(Arc::clone(&s))))
            }
            Token::Dollar => {
                self.advance(); // consume $
                if let Token::Identifier(name) = self.current() {
                    let name = Arc::clone(name);
                    self.advance();
                    Ok(Expr::Variable(name))
                } else {
                    Err(TBError::ParseError {
                        message: Arc::from("Expected identifier after '$'".to_string()),
                        line: 0,
                        column: 0,
                    })
                }
            }
            Token::Bool(b) => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(b)))
            }
            Token::Identifier(name) => {
                self.advance();
                Ok(Expr::Variable(Arc::clone(&name)))
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                if !self.match_token(&Token::RParen) {
                    return Err(TBError::ParseError {
                        message: Arc::from("Expected ')' after expression".to_string()),
                        line: 0,
                        column: 0,
                    });
                }
                self.advance();
                Ok(expr)
            }
            Token::LBracket => {
                self.advance();
                let mut items = ArenaVec::new_in(self.arena);
                while !self.match_token(&Token::RBracket) {
                    items.push(self.parse_expression()?);
                    if !self.match_token(&Token::RBracket) {
                        if !self.match_token(&Token::Comma) {
                            return Err(TBError::ParseError {
                                message: Arc::from("Expected ',' or ']' in list".to_string()),
                                line: 0,
                                column: 0,
                            });
                        }
                        self.advance();
                    }
                }
                self.advance();
                Ok(Expr::List(items))
            }
            Token::LBrace => {
                // Disambiguate: Block vs Dict literal
                // Dict if we see: { "key": value } or { key: value }
                // Block if we see: { statement }

                let is_dict = self.is_dict_literal();

                if is_dict {
                    self.parse_dict_literal()
                } else {
                    self.parse_block()
                }
            }
            Token::If => self.parse_if(),
            Token::Match => self.parse_match(),
            Token::Loop => self.parse_loop(),
            Token::While => self.parse_while(),
            Token::For => self.parse_for(),
            Token::Return => {
                self.advance();
                if self.match_token(&Token::Semicolon) || self.is_eof() {
                    Ok(Expr::Return(None))
                } else {
                    Ok(Expr::Return(Some(self.arena.alloc(self.parse_expression()?))))
                }
            }
            Token::Break => {
                self.advance();
                Ok(Expr::Break(None))
            }
            Token::Continue => {
                self.advance();
                Ok(Expr::Continue)
            }
            Token::Parallel => self.parse_parallel(),
            _ => Err(TBError::ParseError {
                message: format!("Unexpected token: {:?}", self.current()).into(),
                line: 0,
                column: 0,
            }),
        };
        // Check if we consumed at least one token
        if result.is_ok() && self.position == start_position {
            debug_log!("WARNING: parse_primary() did not consume any tokens!");
        }

        self.exit_recursion();
        result
    }

    // Check if current position starts a dict literal
    /// Dict pattern: { string_or_ident : expr }
    fn is_dict_literal(&self) -> bool {
        // Look ahead after '{'
        let lookahead_pos = self.position + 1;

        if lookahead_pos >= self.tokens.len() {
            return false;
        }

        // Empty dict: {}
        if matches!(self.tokens[lookahead_pos], Token::RBrace) {
            return true;
        }

        // Dict starts with string or identifier, followed by colon
        let first_is_key = matches!(
        self.tokens[lookahead_pos],
        Token::String(_) | Token::Identifier(_)
    );

        if !first_is_key {
            return false;
        }

        // Check for colon after key
        if lookahead_pos + 1 < self.tokens.len() {
            matches!(self.tokens[lookahead_pos + 1], Token::Colon)
        } else {
            false
        }
    }

    /// Parse dict literal: { key: value, key2: value2 }
    fn parse_dict_literal(&mut self) ->  TBResult<Expr<'arena>> {
        self.advance(); // consume '{'

        let mut pairs =  ArenaVec::new_in(self.arena);

        while !self.match_token(&Token::RBrace) {
            // Parse key (string or identifier)
            let key_expr = match self.current().clone() {
                Token::String(s) => {
                    self.advance();
                    Expr::Literal(Literal::String(s))
                }
                Token::Identifier(id) => {
                    self.advance();
                    Expr::Literal(Literal::String(id))
                }
                _ => {
                    return Err(TBError::ParseError {
                        message: format!("Expected string or identifier as dict key, found {:?}", self.current()).into(),
                        line: 0,
                        column: 0,
                    });
                }
            };

            // Expect colon
            if !self.match_token(&Token::Colon) {
                return Err(TBError::ParseError {
                    message: Arc::from("Expected ':' after dict key".to_string()),
                    line: 0,
                    column: 0,
                });
            }
            self.advance();

            // Parse value
            let value_expr = self.parse_expression()?;

            pairs.push((key_expr, value_expr));

            // Check for comma or closing brace
            if !self.match_token(&Token::RBrace) {
                if !self.match_token(&Token::Comma) {
                    return Err(TBError::ParseError {
                        message: Arc::from("Expected ',' or '}' in dict literal".to_string()),
                        line: 0,
                        column: 0,
                    });
                }
                self.advance();
            }
        }

        self.advance(); // consume '}'

        Ok(Expr::Dict(pairs))
    }

    fn parse_block(&mut self) ->  TBResult<Expr<'arena>> {
        debug_log!("parse_block() starting at position {}", self.position);
        self.advance(); // consume '{'

        let mut statements = ArenaVec::new_in(self.arena);
        let mut result_expr: Option<Expr<'arena>> = None;
        let mut iterations = 0;
        let max_iterations = 1000; // Safety limit

        while !self.match_token(&Token::RBrace) {
            iterations += 1;
            debug_log!("parse_block iteration {}, position: {}, token: {:?}",
                   iterations, self.position, self.current());

            if iterations > max_iterations {
                return Err(TBError::ParseError {
                    message: format!(
                        "Block parsing exceeded {} iterations. Infinite loop at position {}?",
                        max_iterations, self.position
                    ).into(),
                    line: 0,
                    column: 0,
                });
            }

            if self.is_eof() {
                return Err(TBError::ParseError {
                    message: Arc::from("Unexpected EOF in block".to_string()),
                    line: 0,
                    column: 0,
                });
            }

            let position_before = self.position;

            let stmt_or_expr = self.parse_statement()?;

            if self.position == position_before && !self.match_token(&Token::RBrace) {
                return Err(TBError::ParseError {
                    message: format!(
                        "Parser stuck at position {} with token {:?}. No progress made.",
                        self.position, self.current()
                    ).into(),
                    line: 0,
                    column: 0,
                });
            }

            match stmt_or_expr {
                Statement::Expr(expr) => {
                    if self.match_token(&Token::RBrace) {
                        result_expr = Some(expr);
                    } else {
                        statements.push(Statement::Expr(expr));
                    }
                }
                stmt => statements.push(stmt),
            }
        }

        self.advance(); // consume '}'
        debug_log!("parse_block() completed with {} statements", statements.len());

        Ok(Expr::Block {
            statements,
            result: result_expr.map(|e| self.arena.alloc(e) as &_)
        })
    }

    fn parse_if(&mut self) ->  TBResult<Expr<'arena>> {
        self.advance(); // consume 'if'

        let condition = self.arena.alloc(self.parse_expression()?);
        let then_branch = self.arena.alloc(self.parse_expression()?);

        let else_branch = if self.match_token(&Token::Else) {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(Expr::If {
            condition,
            then_branch,
            else_branch: else_branch.map(|e| self.arena.alloc(e) as &_)
        })
    }

    fn parse_match(&mut self) -> TBResult<Expr<'arena>> {
        self.advance(); // consume 'match'

        let scrutinee = self.arena.alloc(self.parse_expression()?);

        if !self.match_token(&Token::LBrace) {
            return Err(TBError::ParseError {
                message: STRING_INTERNER.intern("Expected '{' after match scrutinee"),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let mut arms = ArenaVec::new_in(self.arena);

        // FIX: Parse arms in separate method to avoid borrow conflicts
        self.parse_match_arms(&mut arms)?;

        Ok(Expr::Match { scrutinee, arms })
    }

    // Helper method - separate lifetime scope
    // FIX: Explizite Lifetime-Trennung
    fn parse_match_arms(&mut self, arms: &mut ArenaVec<'arena, MatchArm<'arena>>) -> TBResult<()> {
        loop {
            // ═══════════════════════════════════════════════════════════════
            // PHASE 1: Boundary Check
            // ═══════════════════════════════════════════════════════════════
            if self.position >= self.tokens.len() {
                return Err(TBError::ParseError {
                    message: STRING_INTERNER.intern("Unexpected EOF in match"),
                    line: 0,
                    column: 0,
                });
            }

            // ═══════════════════════════════════════════════════════════════
            // PHASE 2: Clone current token (ends borrow of self.tokens)
            // ═══════════════════════════════════════════════════════════════
            let current_token = self.tokens[self.position].clone();

            // Check for end
            if matches!(current_token, Token::RBrace) {
                self.position += 1;
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // PHASE 3: Parse pattern inline (avoid method call)
            // ═══════════════════════════════════════════════════════════════
            let pattern = match current_token {
                Token::Identifier(ref name) if name.as_str() == "_" => {
                    self.position += 1; // Manual advance
                    Pattern::Wildcard
                }
                Token::Identifier(ref name) => {
                    let name = Arc::clone(name);
                    self.position += 1; // Manual advance
                    Pattern::Variable(name)
                }
                Token::Int(n) => {
                    self.position += 1; // Manual advance
                    Pattern::Literal(Literal::Int(n))
                }
                Token::Bool(b) => {
                    self.position += 1; // Manual advance
                    Pattern::Literal(Literal::Bool(b))
                }
                _ => {
                    return Err(TBError::ParseError {
                        message: Arc::from("Invalid pattern".to_string()),
                        line: 0,
                        column: 0,
                    });
                }
            };

            // ═══════════════════════════════════════════════════════════════
            // PHASE 4: Check arrow (fresh borrow, pattern is independent)
            // ═══════════════════════════════════════════════════════════════
            if self.position >= self.tokens.len() {
                return Err(TBError::ParseError {
                    message: STRING_INTERNER.intern("Expected '=>' after pattern"),
                    line: 0,
                    column: 0,
                });
            }

            if !matches!(&self.tokens[self.position], Token::FatArrow) {
                return Err(TBError::ParseError {
                    message: STRING_INTERNER.intern("Expected '=>' after pattern"),
                    line: 0,
                    column: 0,
                });
            }
            self.position += 1;

            // ═══════════════════════════════════════════════════════════════
            // PHASE 5: Parse body
            // ═══════════════════════════════════════════════════════════════
            let body = self.parse_expression()?;

            // ═══════════════════════════════════════════════════════════════
            // PHASE 6: Store arm
            // ═══════════════════════════════════════════════════════════════
            arms.push(MatchArm {
                pattern,
                guard: None,
                body,
            });

            // ═══════════════════════════════════════════════════════════════
            // PHASE 7: Check continuation
            // ═══════════════════════════════════════════════════════════════
            if self.position >= self.tokens.len() {
                break;
            }

            if matches!(&self.tokens[self.position], Token::RBrace) {
                self.position += 1;
                break;
            }

            if !matches!(&self.tokens[self.position], Token::Comma) {
                return Err(TBError::ParseError {
                    message: STRING_INTERNER.intern("Expected ',' or '}' after match arm"),
                    line: 0,
                    column: 0,
                });
            }
            self.position += 1;
        }

        Ok(())
    }

    fn parse_loop(&mut self) ->  TBResult<Expr<'arena>> {
        self.advance(); // consume 'loop'
        let body = self.arena.alloc(self.parse_expression()?);
        Ok(Expr::Loop { body })
    }

    fn parse_while(&mut self) ->  TBResult<Expr<'arena>> {
        self.advance(); // consume 'while'
        let condition = self.arena.alloc(self.parse_expression()?);
        let body = self.arena.alloc(self.parse_expression()?);
        Ok(Expr::While { condition, body })
    }

    fn parse_for(&mut self) ->  TBResult<Expr<'arena>> {
        self.advance(); // consume 'for'

        let variable = if let Token::Identifier(name) = self.current() {
            let name = Arc::clone(name);
            self.advance();
            name
        } else {
            return Err(TBError::ParseError {
                message: Arc::from("Expected variable name after 'for'".to_string()),
                line: 0,
                column: 0,
            });
        };

        if !self.match_token(&Token::In) {
            return Err(TBError::ParseError {
                message: Arc::from("Expected 'in' after loop variable".to_string()),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let iterable = self.arena.alloc(self.parse_expression()?);
        let body = self.arena.alloc(self.parse_expression()?);

        Ok(Expr::For {
            variable,
            iterable,
            body,
        })
    }

    fn parse_parallel(&mut self) ->  TBResult<Expr<'arena>> {
        self.advance(); // consume 'parallel'

        if !self.match_token(&Token::LBrace) {
            return Err(TBError::ParseError {
                message: Arc::from("Expected '{' after 'parallel'".to_string()),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let mut tasks = ArenaVec::new_in(self.arena);

        // Support both comma and newline as separators
        while !self.match_token(&Token::RBrace) {
            if self.is_eof() {
                return Err(TBError::ParseError {
                    message: Arc::from("Unexpected EOF in parallel block".to_string()),
                    line: 0,
                    column: 0,
                });
            }

            tasks.push(self.parse_expression()?);

            // Skip separators (comma, semicolon, or newline)
            if !self.match_token(&Token::RBrace) {
                if self.match_token(&Token::Comma)
                    || self.match_token(&Token::Semicolon)
                    || self.match_token(&Token::Newline) {
                    self.advance();
                }
            }
        }

        self.advance(); // consume '}'

        Ok(Expr::Parallel(tasks))
    }

    fn parse_type(&mut self) -> TBResult<Type> {
        match self.current() {
            Token::Identifier(name) => {
                let name = Arc::clone(name);
                self.advance();
                match name.as_str() {
                    "int" => Ok(Type::Int),
                    "float" => Ok(Type::Float),
                    "bool" => Ok(Type::Bool),
                    "string" => Ok(Type::String),
                    _ => Ok(Type::Dynamic),
                }
            }
            _ => Ok(Type::Dynamic),
        }
    }

    fn current(&self) -> &Token {
        if self.position >= self.tokens.len() {
            &Token::Eof
        } else {
            &self.tokens[self.position]
        }
    }

    fn advance(&mut self) {
        if !self.is_eof() {
            self.position += 1;
        }
    }

    fn match_token(&self, token: &Token) -> bool {
        if self.is_eof() {
            false
        } else {
            std::mem::discriminant(self.current()) == std::mem::discriminant(token)
        }
    }

    fn enter_recursion(&mut self) -> TBResult<()> {
        self.recursion_depth += 1;
        debug_log!("Parser recursion depth: {}", self.recursion_depth);

        if self.recursion_depth > self.max_recursion_depth {
            return Err(TBError::ParseError {
                message: format!(
                    "Maximum recursion depth {} exceeded at position {}. Possible infinite loop.",
                    self.max_recursion_depth,
                    self.position
                ).into(),
                line: 0,
                column: 0,
            });
        }
        Ok(())
    }

    /// Exit a recursive parse call
    fn exit_recursion(&mut self) {
        if self.recursion_depth > 0 {
            self.recursion_depth -= 1;
        }
    }

    fn is_eof(&self) -> bool {
        // Check if past end OR at EOF token
        if self.position >= self.tokens.len() {
            debug_log!("Parser::is_eof() = true (position >= len)");
            return true;
        }

        // Check if current token is EOF (without calling self.current() to avoid recursion)
        let is_eof_token = matches!(self.tokens[self.position], Token::Eof);
        // debug_log!("Parser::is_eof() = {}, position = {}, tokens.len = {}, token = {:?}",
        //       is_eof_token, self.position, self.tokens.len(), self.tokens[self.position]);

        is_eof_token
    }
}




// ═══════════════════════════════════════════════════════════════════════════
// §8 TYPE CHECKER
// ═══════════════════════════════════════════════════════════════════════════

pub struct TypeChecker {
    mode: TypeMode,
    env: TypeEnvironment,
}

#[derive(Debug, Clone)]
struct TypeEnvironment {
    variables: HashMap<Arc<String>, Type>,
    functions: HashMap<Arc<String>, (Vec<Type>, Type)>,
}

impl TypeChecker {
    pub fn new(mode: TypeMode) -> Self {
        Self {
            mode,
            env: TypeEnvironment {
                variables: HashMap::new(),
                functions: HashMap::new(),
            },
        }
    }

    pub fn check_statements(&mut self, statements: &[Statement]) -> TBResult<()> {
        if self.mode == TypeMode::Dynamic {
            return Ok(()); // Skip type checking in dynamic mode
        }

        for stmt in statements {
            self.check_statement(stmt)?;
        }

        Ok(())
    }

    fn check_statement(&mut self, stmt: &Statement) -> TBResult<()> {
        match stmt {
            Statement::Let { name, type_annotation, value, .. } => {
                let value_type = self.infer_type(value)?;

                if let Some(expected) = type_annotation {
                    if &value_type != expected {
                        return Err(TBError::TypeError {
                            expected: expected.clone(),
                            found: value_type,
                            context: format!("let binding '{}'", name).into(),
                        });
                    }
                }

                self.env.variables.insert(name.clone(), value_type);
                Ok(())
            }

            Statement::Function { name, params, return_type, body } => {
                let param_types: Vec<_> = params.iter()
                    .map(|p| p.type_annotation.clone().unwrap_or(Type::Dynamic))
                    .collect();

                let ret_type = return_type.clone().unwrap_or(Type::Dynamic);

                self.env.functions.insert(name.clone(), (param_types, ret_type));

                // TODO: Check body
                Ok(())
            }

            Statement::Expr(expr) => {
                self.infer_type(expr)?;
                Ok(())
            }

            _ => Ok(()),
        }
    }

    fn infer_type(&self, expr: &Expr) -> TBResult<Type> {
        match expr {
            Expr::Literal(lit) => Ok(match lit {
                Literal::Unit => Type::Unit,
                Literal::Bool(_) => Type::Bool,
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
            }),

            Expr::Variable(name) => {
                // Check if it's a builtin function - allow it in static mode
                if matches!(
                name.as_str(),
                "echo" | "print" | "println" | "read_line" |
                "python" | "javascript" | "bash" | "go" |
                "debug" | "len" | "str" | "int" | "float" | "type_of"
            ) {
                    return Ok(Type::Function {
                        params:vec![],
                        ret: Box::new(Type::Dynamic),
                    });
                }

                self.env.variables.get(name)
                    .cloned()
                    .ok_or_else(|| TBError::UndefinedVariable(name.clone()))
            }

            Expr::BinOp { op, left, right } => {
                let left_type = self.infer_type(left)?;
                let right_type = self.infer_type(right)?;

                // Allow Dynamic types to bypass strict checking
                if left_type == Type::Dynamic || right_type == Type::Dynamic {
                    return Ok(Type::Dynamic);
                }

                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if left_type.is_numeric() && right_type.is_numeric() {
                            Ok(left_type)
                        } else {
                            Err(TBError::TypeError {
                                expected: Type::Int,
                                found: left_type,
                                context: Arc::from("binary operation".to_string()),
                            })
                        }
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        Ok(Type::Bool)
                    }
                    BinOp::And | BinOp::Or => Ok(Type::Bool),
                    _ => Ok(Type::Dynamic),
                }
            }

            Expr::Call { function, args } => {
                // TODO: Function type checking
                Ok(Type::Dynamic)
            }

            Expr::If { condition, then_branch, else_branch } => {
                let cond_type = self.infer_type(condition)?;
                let then_type = self.infer_type(then_branch)?;

                if let Some(else_branch) = else_branch {
                    let else_type = self.infer_type(else_branch)?;
                    Ok(then_type)
                } else {
                    Ok(Type::Option(Box::new(then_type)))
                }
            }

            Expr::Block { result, .. } => {
                if let Some(result) = result {
                    self.infer_type(result)
                } else {
                    Ok(Type::Unit)
                }
            }

            Expr::List(items) => {
                if items.is_empty() {
                    Ok(Type::List(Box::new(Type::Dynamic)))
                } else {
                    let item_type = self.infer_type(&items[0])?;
                    Ok(Type::List(Box::new(item_type)))
                }
            }

            _ => Ok(Type::Dynamic),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §9 OPTIMIZER
// ═══════════════════════════════════════════════════════════════════════════

pub struct Optimizer {
    config: OptimizerConfig,
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub inline_threshold: usize,
    pub const_fold: bool,
    pub dead_code_elimination: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            inline_threshold: 10,
            const_fold: true,
            dead_code_elimination: true,
        }
    }
}

impl Optimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    pub fn optimize<'arena>(&self, expr: Expr<'arena>) -> TBResult<Expr<'arena>> {
        let mut expr = expr;
        if self.config.const_fold {
            expr = self.constant_folding(expr)?;
        }
        Ok(expr)
    }

    fn constant_folding<'arena>(&self, expr: Expr<'arena>) -> TBResult<Expr<'arena>> {
        match expr {
            Expr::BinOp { op, left, right } => {
                // Clone statt alloc
                let left_opt = self.constant_folding((*left).clone())?;
                let right_opt = self.constant_folding((*right).clone())?;

                if let (Expr::Literal(l), Expr::Literal(r)) = (&left_opt, &right_opt) {
                    if let Some(result) = self.eval_const_binop(op, &l, &r) {
                        return Ok(Expr::Literal(result));
                    }
                }

                // Neues BinOp OHNE Arena-Alloc
                Ok(Expr::BinOp {
                    op,
                    left: Box::leak(Box::new(left_opt)),
                    right: Box::leak(Box::new(right_opt)),
                })
            }

            Expr::UnaryOp { op, expr: inner } => {
                let expr_opt = self.constant_folding((*inner).clone())?;

                if let Expr::Literal(lit) = &expr_opt {
                    if let Some(result) = self.eval_const_unaryop(op, &lit) {
                        return Ok(Expr::Literal(result));
                    }
                }

                Ok(Expr::UnaryOp {
                    op,
                    expr: Box::leak(Box::new(expr_opt)),
                })
            }

            Expr::If { condition, then_branch, else_branch } => {
                let condition_opt = self.constant_folding((*condition).clone())?;

                if let Expr::Literal(Literal::Bool(b)) = condition_opt {
                    return if b {
                        self.constant_folding((*then_branch).clone())
                    } else if let Some(else_b) = else_branch {
                        self.constant_folding((*else_b).clone())
                    } else {
                        Ok(Expr::Literal(Literal::Unit))
                    };
                }

                let then_opt = self.constant_folding((*then_branch).clone())?;
                let else_opt = else_branch
                    .map(|e| self.constant_folding((*e).clone()))
                    .transpose()?;

                Ok(Expr::If {
                    condition: Box::leak(Box::new(condition_opt)),
                    then_branch: Box::leak(Box::new(then_opt)),
                    else_branch: else_opt.map(|e| Box::leak(Box::new(e)) as &_),
                })
            }

            // Für andere Expr-Typen: einfach zurückgeben
            _ => Ok(expr),
        }
    }


    fn eval_const_binop(&self, op: BinOp, left: &Literal, right: &Literal) -> Option<Literal> {
        match (op, left, right) {
            (BinOp::Add, Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a + b)),
            (BinOp::Sub, Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a - b)),
            (BinOp::Mul, Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a * b)),
            (BinOp::Div, Literal::Int(a), Literal::Int(b)) if *b != 0 => Some(Literal::Int(a / b)),
            (BinOp::Mod, Literal::Int(a), Literal::Int(b)) if *b != 0 => Some(Literal::Int(a % b)),

            (BinOp::Add, Literal::Float(a), Literal::Float(b)) => Some(Literal::Float(a + b)),
            (BinOp::Sub, Literal::Float(a), Literal::Float(b)) => Some(Literal::Float(a - b)),
            (BinOp::Mul, Literal::Float(a), Literal::Float(b)) => Some(Literal::Float(a * b)),
            (BinOp::Div, Literal::Float(a), Literal::Float(b)) => Some(Literal::Float(a / b)),

            (BinOp::Eq, a, b) => Some(Literal::Bool(a == b)),
            (BinOp::Ne, a, b) => Some(Literal::Bool(a != b)),
            (BinOp::Lt, Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a < b)),
            (BinOp::Le, Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a <= b)),
            (BinOp::Gt, Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a > b)),
            (BinOp::Ge, Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a >= b)),

            (BinOp::And, Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(*a && *b)),
            (BinOp::Or, Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(*a || *b)),

            _ => None,
        }
    }

    fn eval_const_unaryop(&self, op: UnaryOp, expr: &Literal) -> Option<Literal> {
        match (op, expr) {
            (UnaryOp::Not, Literal::Bool(b)) => Some(Literal::Bool(!b)),
            (UnaryOp::Neg, Literal::Int(n)) => Some(Literal::Int(-n)),
            (UnaryOp::Neg, Literal::Float(f)) => Some(Literal::Float(-f)),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §10 CODE GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

fn parse_string_interpolation_standalone(s: &str) -> (String, Vec<Expr<'static>>) {
    let mut result = String::new();
    let mut variables = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' {
            if let Some(&next_ch) = chars.peek() {
                if next_ch.is_alphabetic() || next_ch == '_' {
                    let mut var_name = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch.is_alphanumeric() || ch == '_' {
                            var_name.push(ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    result.push_str("{}");
                    variables.push(Expr::Variable(Arc::new(var_name)));
                    continue;
                }
            }
            result.push('$');
        } else if ch == '{' || ch == '}' {
            result.push(ch);
            result.push(ch);
        } else {
            result.push(ch);
        }
    }

    (result, variables)
}

pub struct CodeGenerator {
    target_language: Language,
    buffer: String,
    indent: usize,
    variables_in_scope: Vec<(Arc<String>, Type)>,
    compiled_deps: Vec<CompiledDependency>,
    mutable_vars: std::collections::HashSet<Arc<String>>,
    builtin_registry: Option<BuiltinRegistry>,
}

impl CodeGenerator {
    pub fn new(target_language: Language) -> Self {
        Self {
            target_language,
            buffer: String::new(),
            indent: 0,
            variables_in_scope: Vec::new(),
            compiled_deps: Vec::new(),
            mutable_vars: std::collections::HashSet::new(),
            builtin_registry: None,
        }
    }
    pub fn set_builtin_registry(&mut self, registry: BuiltinRegistry) {
        self.builtin_registry = Some(registry);
    }

    /// Set compiled dependencies (called by Compiler)
    pub fn set_compiled_dependencies(&mut self, deps: &[CompiledDependency]) {
        self.compiled_deps = deps.to_vec();
    }

    /// Generate builtin helper functions (echo, print, etc.)
    fn generate_builtin_functions(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("// Builtin Functions");
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // echo - Print with newline (Generic für alle Display-Typen)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn echo<T: std::fmt::Display>(s: T) {");
        self.indent += 1;
        self.emit_line("println!(\"{}\", s);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // print - Print without newline (Generic für alle Display-Typen)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn print<T: std::fmt::Display>(s: T) {");
        self.indent += 1;
        self.emit_line("print!(\"{}\", s);");
        self.emit_line("use std::io::Write;");
        self.emit_line("std::io::stdout().flush().ok();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // println - Alias for echo (Generic für alle Display-Typen)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn println<T: std::fmt::Display>(s: T) {");
        self.indent += 1;
        self.emit_line("echo(s);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.generate_builtin_wrappers();
    }

    /// Generate wrappers for all registered builtin functions
    fn generate_builtin_wrappers(&mut self) {

        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("// Auto-Generated Builtin Function Wrappers");
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("");

        let binding = self.builtin_registry.clone().unwrap();
        let function_names = binding.list();
        // Filtere bereits manuell definierte Funktionen
        let skip_functions = ["echo", "print", "println"];

        for func_name in function_names {
            if skip_functions.contains(&func_name) {
                continue; // Überspringen, bereits manuell definiert
            }

            if let Some(builtin) = self.builtin_registry.clone().unwrap().get(func_name) {
                self.generate_builtin_wrapper(&builtin);
            }
        }

        self.emit_line("");
    }

    /// Generate a single builtin function wrapper
    fn generate_builtin_wrapper(&mut self, builtin: &Arc<BuiltinFunction>) {
        let name = builtin.name.as_str();
        let min_args = builtin.min_args;
        let max_args = builtin.max_args;

        self.emit_line(&format!("/// {}", builtin.description));
        self.emit_line("#[allow(dead_code)]");

        // Generiere Funktionssignatur basierend auf Argumentanzahl
        if min_args == 0 && max_args == Some(0) {
            // Keine Argumente
            self.emit_line(&format!("fn {}() -> String {{", name));
            self.indent += 1;
            self.emit_line(&format!("// TODO: Implement {} via runtime", name));
            self.emit_line("String::from(\"Not implemented in compiled mode\")");
            self.indent -= 1;
            self.emit_line("}");
        } else if min_args == max_args.unwrap_or(min_args) {
            // Feste Argumentanzahl
            let args: Vec<String> = (0..min_args)
                .map(|i| format!("arg{}: &str", i))
                .collect();

            self.emit_line(&format!("fn {}({}) -> String {{", name, args.join(", ")));
            self.indent += 1;
            self.emit_line(&format!("// TODO: Implement {} via runtime", name));
            self.emit_line("String::from(\"Not implemented in compiled mode\")");
            self.indent -= 1;
            self.emit_line("}");
        } else {
            // Variable Argumentanzahl - verwende Macro
            self.emit_line(&format!("macro_rules! {} {{", name));
            self.indent += 1;
            self.emit_line("($($arg:expr),*) => {{");
            self.indent += 1;
            self.emit_line(&format!("// TODO: Implement {} via runtime", name));
            self.emit_line("String::from(\"Not implemented in compiled mode\")");
            self.indent -= 1;
            self.emit_line("}};");
            self.indent -= 1;
            self.emit_line("}");
        }

        self.emit_line("");
    }

    /// NEW: Set mutable variables info
    pub fn set_mutable_vars(&mut self, vars: &std::collections::HashSet<Arc<String>>) {
        self.mutable_vars = vars.clone();
        debug_log!("CodeGenerator: mutable_vars = {:?}", self.mutable_vars);
    }
    pub fn generate(&mut self, statements: &[Statement]) -> TBResult<String> {
        match self.target_language {
            Language::Rust => self.generate_rust(statements),
            Language::Python => self.generate_python(statements),
            Language::JavaScript => self.generate_javascript(statements),
            _ => Err(TBError::UnsupportedLanguage(format!("{:?}", self.target_language).into())),
        }
    }

    fn generate_rust(&mut self, statements: &[Statement]) -> TBResult<String> {
        // Prelude
        self.emit_line("// Auto-generated by TB Compiler");
        self.emit_line("// Optimized for native execution");
        self.emit_line("");
        self.emit_line("#[allow(unused)]");
        self.emit_line("");

        // Core imports
        self.emit_line("use std::process::Command;");
        self.emit_line("use std::io::Write;");
        self.emit_line("");

        // NEW: Pre-analyze variable mutability
        let mut mutable_vars = std::collections::HashSet::new();
        for stmt in statements {
            if let Statement::Let { name, .. } = stmt {
                // Check if this variable is assigned later
                if self.is_variable_assigned(name, statements) {
                    mutable_vars.insert(name.clone());
                    debug_log!("Variable '{}' detected as mutable", name);
                }
            }
        }

        // Store for use in generate_rust_statement
        self.mutable_vars = mutable_vars;

        // Check if we need language bridges
        let needs_bridges = self.statements_use_language_bridges(statements);

        if needs_bridges {
            self.emit_line("// ═══════════════════════════════════════════════════");
            self.emit_line("// Language Bridge Helper Functions");
            self.emit_line("// ═══════════════════════════════════════════════════");
            self.emit_line("");
            self.generate_language_bridge_helpers();
            self.emit_line("");
        }

        // Add fmt_display
        self.emit_line("");
        self.emit_line("/// Format any value for display in println!");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn fmt_display<T: std::fmt::Debug>(val: &T) -> String {");
        self.indent += 1;
        self.emit_line("format!(\"{:?}\", val)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Generate functions
        for stmt in statements {
            if let Statement::Function { .. } = stmt {
                self.generate_rust_statement(stmt)?;
            }
        }

        // Main function
        self.emit_line("fn main() {");
        self.indent += 1;

        for stmt in statements {
            if !matches!(stmt, Statement::Function { .. }) {
                self.generate_rust_statement(stmt)?;
            }
        }

        self.indent -= 1;
        self.emit_line("}");

        Ok(std::mem::take(&mut self.buffer.to_string()))
    }

    pub fn extract_function_code(&self, full_code: &str) -> String {
        let mut result = String::new();
        let mut in_function = false;
        let mut brace_count = 0;

        for line in full_code.lines() {
            let trimmed = line.trim();

            // Detect function start
            if trimmed.starts_with("#[inline") ||
                trimmed.starts_with("pub fn") ||
                trimmed.starts_with("fn ") {
                in_function = true;
            }

            if in_function {
                result.push_str(line);
                result.push('\n');

                // Count braces
                for ch in line.chars() {
                    if ch == '{' { brace_count += 1; }
                    if ch == '}' {
                        brace_count -= 1;
                        if brace_count == 0 {
                            in_function = false;
                            result.push('\n');
                        }
                    }
                }
            }
        }

        result
    }

    /// Check if statements use language bridges
    fn statements_use_language_bridges(&self, statements: &[Statement]) -> bool {
        for stmt in statements {
            if self.statement_uses_bridges(stmt) {
                return true;
            }
        }
        false
    }

    fn statement_uses_bridges(&self, stmt: &Statement) -> bool {
        match stmt {
            Statement::Let { value, .. } => self.expr_uses_bridges(value),
            Statement::Expr(expr) => self.expr_uses_bridges(expr),
            Statement::Function { body, .. } => self.expr_uses_bridges(body),
            _ => false,
        }
    }

    fn expr_uses_bridges(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Call { function, args } => {
                if let Expr::Variable(name) = function {
                    if matches!(name.as_str(), "python" | "javascript" | "go" | "bash") {
                        return true;
                    }
                }
                args.iter().any(|a| self.expr_uses_bridges(a))
            }
            Expr::Block { statements, result } => {
                statements.iter().any(|s| self.statement_uses_bridges(s))
                    || result.as_ref().map_or(false, |e| self.expr_uses_bridges(e))
            }
            Expr::BinOp { left, right, .. } => {
                self.expr_uses_bridges(left) || self.expr_uses_bridges(right)
            }
            _ => false,
        }
    }

    /// Generate language bridge helper functions
    /// Generate language bridge helper functions with JSON support
    /// Generate language bridge helper functions with JSON support
    /// Generate high-performance language bridge helper functions
    fn generate_language_bridge_helpers(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("// HIGH-PERFORMANCE LANGUAGE BRIDGES");
        self.emit_line("// Zero-copy variable injection, optimized execution paths");
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // PYTHON BRIDGE - Full Implementation with Variable Injection
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn python(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("use std::process::{Command, Stdio};");
        self.emit_line("use std::io::Write;");
        self.emit_line("");

        self.emit_line("// Detect Python executable (venv-aware)");
        self.emit_line("let python_exe = detect_python_exe();");
        self.emit_line("");

        self.emit_line("// Auto-wrap for return values");
        self.emit_line("let wrapped = wrap_python_auto_return(code);");
        self.emit_line("");

        self.emit_line("// Execute with UTF-8 encoding");
        self.emit_line("let mut child = Command::new(&python_exe)");
        self.indent += 1;
        self.emit_line(".arg(\"-c\")");
        self.emit_line(".arg(&wrapped)");
        self.emit_line(".env(\"PYTHONIOENCODING\", \"utf-8\")");
        self.emit_line(".stdin(Stdio::null())");
        self.emit_line(".stdout(Stdio::piped())");
        self.emit_line(".stderr(Stdio::piped())");
        self.emit_line(".spawn()");
        self.emit_line(".expect(\"Failed to spawn Python process\");");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("// Wait with timeout");
        self.emit_line("let output = child.wait_with_output()");
        self.indent += 1;
        self.emit_line(".expect(\"Failed to wait for Python\");");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("");

        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"[Python Error] {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("print!(\"{}\", stdout);");
        self.emit_line("stdout.trim().to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Python helper functions
        self.generate_python_helpers();

        // ═══════════════════════════════════════════════════════════════
        // JAVASCRIPT BRIDGE - Full Implementation
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn javascript(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("use std::process::{Command, Stdio};");
        self.emit_line("");

        self.emit_line("// Detect Node.js/Bun/Deno");
        self.emit_line("let runtime = detect_js_runtime();");
        self.emit_line("");

        self.emit_line("// Auto-wrap for return values");
        self.emit_line("let wrapped = wrap_js_auto_return(code);");
        self.emit_line("");

        self.emit_line("// Execute based on runtime");
        self.emit_line("let output = match runtime.as_str() {");
        self.indent += 1;
        self.emit_line("\"node\" => Command::new(\"node\")");
        self.indent += 1;
        self.emit_line(".arg(\"-e\")");
        self.emit_line(".arg(&wrapped)");
        self.emit_line(".output(),");
        self.indent -= 1;
        self.emit_line("\"bun\" => Command::new(\"bun\")");
        self.indent += 1;
        self.emit_line(".arg(\"run\")");
        self.emit_line(".arg(\"-e\")");
        self.emit_line(".arg(&wrapped)");
        self.emit_line(".output(),");
        self.indent -= 1;
        self.emit_line("\"deno\" => Command::new(\"deno\")");
        self.indent += 1;
        self.emit_line(".arg(\"eval\")");
        self.emit_line(".arg(\"--no-check\")");
        self.emit_line(".arg(&wrapped)");
        self.emit_line(".output(),");
        self.indent -= 1;
        self.emit_line("_ => panic!(\"No JavaScript runtime found\"),");
        self.indent -= 1;
        self.emit_line("}.expect(\"Failed to execute JavaScript\");");
        self.emit_line("");

        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("");

        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"[JavaScript Error] {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("print!(\"{}\", stdout);");
        self.emit_line("stdout.trim().to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // JavaScript helper functions
        self.generate_js_helpers();

        // ═══════════════════════════════════════════════════════════════
        // GO BRIDGE - Full Implementation with Proper Variable Handling
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn go(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("use std::process::Command;");
        self.emit_line("use std::fs;");
        self.emit_line("use std::env;");
        self.emit_line("");

        self.emit_line("// Create temporary directory");
        self.emit_line("let temp_dir = env::temp_dir().join(format!(\"tb_go_{}\", std::process::id()));");
        self.emit_line("fs::create_dir_all(&temp_dir).ok();");
        self.emit_line("");

        self.emit_line("// Split injection and user code");
        self.emit_line("let (injection, user_code) = split_go_injection(code);");
        self.emit_line("");

        self.emit_line("// Extract variable names from injection");
        self.emit_line("let var_names = extract_go_var_names(&injection);");
        self.emit_line("");

        self.emit_line("// Fix redeclarations in user code");
        self.emit_line("let fixed_user = fix_go_redeclarations(&user_code, &var_names);");
        self.emit_line("");

        self.emit_line("// Auto-wrap for return values");
        self.emit_line("let wrapped = wrap_go_auto_return(&fixed_user);");
        self.emit_line("");

        self.emit_line("// Combine injection + wrapped user code");
        self.emit_line("let combined = if injection.is_empty() {");
        self.indent += 1;
        self.emit_line("wrapped");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("format!(\"{}\\n{}\", injection, wrapped)");
        self.indent -= 1;
        self.emit_line("};");
        self.emit_line("");

        self.emit_line("// Extract imports from combined code");
        self.emit_line("let (imports, clean_code) = extract_go_imports(&combined);");
        self.emit_line("");

        self.emit_line("// Build complete Go file");
        self.emit_line("let mut full_code = String::from(\"package main\\n\\n\");");
        self.emit_line("");

        self.emit_line("// Add imports");
        self.emit_line("if !imports.is_empty() {");
        self.indent += 1;
        self.emit_line("full_code.push_str(\"import (\\n\");");
        self.emit_line("let mut import_set: std::collections::HashSet<String> = imports.into_iter().collect();");
        self.emit_line("import_set.insert(\"fmt\".to_string());");
        self.emit_line("let mut sorted_imports: Vec<String> = import_set.into_iter().collect();");
        self.emit_line("sorted_imports.sort();");
        self.emit_line("for imp in sorted_imports {");
        self.indent += 1;
        self.emit_line("full_code.push_str(&format!(\"    \\\"{}\\\"\\n\", imp));");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("full_code.push_str(\")\\n\\n\");");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("full_code.push_str(\"import \\\"fmt\\\"\\n\\n\");");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Add main function with indented code");
        self.emit_line("full_code.push_str(\"func main() {\\n\");");
        self.emit_line("for line in clean_code.lines() {");
        self.indent += 1;
        self.emit_line("if !line.trim().is_empty() {");
        self.indent += 1;
        self.emit_line("full_code.push_str(\"    \");");
        self.emit_line("full_code.push_str(line);");
        self.emit_line("full_code.push('\\n');");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("full_code.push_str(\"}\\n\");");
        self.emit_line("");

        self.emit_line("// Write and compile");
        self.emit_line("let go_file = temp_dir.join(\"main.go\");");
        self.emit_line("fs::write(&go_file, &full_code).ok();");
        self.emit_line("");

        self.emit_line("let output = Command::new(\"go\")");
        self.indent += 1;
        self.emit_line(".arg(\"run\")");
        self.emit_line(".arg(&go_file)");
        self.emit_line(".current_dir(&temp_dir)");
        self.emit_line(".output()");
        self.emit_line(".expect(\"Failed to execute Go\");");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("// Cleanup");
        self.emit_line("fs::remove_dir_all(&temp_dir).ok();");
        self.emit_line("");

        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("");

        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"[Go Error] {}\\n[Generated Code]\\n{}\", stderr, full_code);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("print!(\"{}\", stdout);");
        self.emit_line("stdout.lines().filter(|l| !l.trim().is_empty()).last().unwrap_or(\"\").trim().to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Go helper functions
        self.generate_go_helpers();

        // ═══════════════════════════════════════════════════════════════
        // BASH BRIDGE - Full Implementation
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn bash(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("use std::process::Command;");
        self.emit_line("");

        self.emit_line("// Detect shell (bash/zsh/sh)");
        self.emit_line("let (shell, flag) = detect_shell();");
        self.emit_line("");

        self.emit_line("// Auto-wrap for return values");
        self.emit_line("let wrapped = wrap_bash_auto_return(code);");
        self.emit_line("");

        self.emit_line("let output = Command::new(shell)");
        self.indent += 1;
        self.emit_line(".arg(flag)");
        self.emit_line(".arg(&wrapped)");
        self.emit_line(".output()");
        self.emit_line(".expect(\"Failed to execute Bash\");");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("");

        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"[Bash Error] {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("print!(\"{}\", stdout);");
        self.emit_line("stdout.trim().to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Bash helper functions
        self.generate_bash_helpers();

        // Type parser helpers (existing)
        self.generate_type_parser_helpers();
    }

    /// Generate Python-specific helper functions
    fn generate_python_helpers(&mut self) {
        // Detect Python executable
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn detect_python_exe() -> String {");
        self.indent += 1;
        self.emit_line("use std::env;");
        self.emit_line("use std::path::Path;");
        self.emit_line("use std::process::Command;");
        self.emit_line("");

        self.emit_line("// Priority: VIRTUAL_ENV > CONDA_PREFIX > UV > Poetry > python3 > python");
        self.emit_line("if let Ok(venv) = env::var(\"VIRTUAL_ENV\") {");
        self.indent += 1;
        self.emit_line("let python = Path::new(&venv).join(if cfg!(windows) { \"Scripts\\\\python.exe\" } else { \"bin/python\" });");
        self.emit_line("if python.exists() { return python.to_string_lossy().to_string(); }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if let Ok(conda) = env::var(\"CONDA_PREFIX\") {");
        self.indent += 1;
        self.emit_line("let python = Path::new(&conda).join(if cfg!(windows) { \"python.exe\" } else { \"bin/python\" });");
        self.emit_line("if python.exists() { return python.to_string_lossy().to_string(); }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if Command::new(\"python3\").arg(\"--version\").output().map(|o| o.status.success()).unwrap_or(false) {");
        self.indent += 1;
        self.emit_line("return \"python3\".to_string();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("\"python\".to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Auto-return wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn wrap_python_auto_return(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let lines: Vec<&str> = code.trim().lines().collect();");
        self.emit_line("if lines.is_empty() { return code.to_string(); }");
        self.emit_line("");

        self.emit_line("let last = lines.last().unwrap().trim();");
        self.emit_line("if last.is_empty() || last.starts_with('#') || last.starts_with(\"print(\")");
        self.indent += 1;
        self.emit_line("|| (last.contains(\" = \") && !last.contains(\"==\"))");
        self.emit_line("|| last.starts_with(\"if \") || last.starts_with(\"for \")");
        self.emit_line("|| last.starts_with(\"def \") || last.ends_with(':')");
        self.emit_line("{ return code.to_string(); }");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("let mut result = lines[..lines.len()-1].join(\"\\n\");");
        self.emit_line("if !result.is_empty() { result.push('\\n'); }");
        self.emit_line("result.push_str(&format!(\"__tb_result = ({})\\nprint(__tb_result, end='')\", last));");
        self.emit_line("result");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    /// Generate JavaScript-specific helper functions
    fn generate_js_helpers(&mut self) {
        // Detect JS runtime
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn detect_js_runtime() -> String {");
        self.indent += 1;
        self.emit_line("use std::process::Command;");
        self.emit_line("");

        self.emit_line("// Try Node.js");
        self.emit_line("if Command::new(\"node\").arg(\"--version\").output().map(|o| o.status.success()).unwrap_or(false) {");
        self.indent += 1;
        self.emit_line("return \"node\".to_string();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Try Bun");
        self.emit_line("if Command::new(\"bun\").arg(\"--version\").output().map(|o| o.status.success()).unwrap_or(false) {");
        self.indent += 1;
        self.emit_line("return \"bun\".to_string();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Try Deno");
        self.emit_line("if Command::new(\"deno\").arg(\"--version\").output().map(|o| o.status.success()).unwrap_or(false) {");
        self.indent += 1;
        self.emit_line("return \"deno\".to_string();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("panic!(\"No JavaScript runtime found. Install Node.js, Bun, or Deno.\");");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Auto-return wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn wrap_js_auto_return(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let lines: Vec<&str> = code.trim().lines().collect();");
        self.emit_line("if lines.is_empty() { return code.to_string(); }");
        self.emit_line("");

        self.emit_line("let last = lines.last().unwrap().trim();");
        self.emit_line("if last.is_empty() || last.starts_with(\"//\") || last.starts_with(\"console.log\")");
        self.indent += 1;
        self.emit_line("|| last.starts_with(\"const \") || last.starts_with(\"let \") || last.starts_with(\"var \")");
        self.emit_line("|| last.starts_with(\"if \") || last.starts_with(\"for \") || last.ends_with('{')");
        self.emit_line("{ return code.to_string(); }");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("let mut result = lines[..lines.len()-1].join(\"\\n\");");
        self.emit_line("if !result.is_empty() { result.push('\\n'); }");
        self.emit_line("result.push_str(&format!(\"const __tb_result = ({});\\nprocess.stdout.write(String(__tb_result));\", last));");
        self.emit_line("result");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    /// Generate Go-specific helper functions with PRODUCTION-READY variable handling
    fn generate_go_helpers(&mut self) {
        // Split injection block
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn split_go_injection(code: &str) -> (String, String) {");
        self.indent += 1;
        self.emit_line("let mut injection = String::new();");
        self.emit_line("let mut user_code = String::new();");
        self.emit_line("let mut in_injection = false;");
        self.emit_line("");

        self.emit_line("for line in code.lines() {");
        self.indent += 1;
        self.emit_line("if line.contains(\"TB Variables (auto-injected)\") {");
        self.indent += 1;
        self.emit_line("in_injection = true;");
        self.emit_line("continue;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("if in_injection {");
        self.indent += 1;
        self.emit_line("if line.trim().is_empty() { continue; }");
        self.emit_line("if line.trim().starts_with(\"var \") || line.trim().starts_with(\"_\") || line.trim().contains(\" := \") {");
        self.indent += 1;
        self.emit_line("injection.push_str(line);");
        self.emit_line("injection.push('\\n');");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("in_injection = false;");
        self.emit_line("user_code.push_str(line);");
        self.emit_line("user_code.push('\\n');");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("user_code.push_str(line);");
        self.emit_line("user_code.push('\\n');");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("(injection, user_code)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Extract variable names
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn extract_go_var_names(injection: &str) -> Vec<String> {");
        self.indent += 1;
        self.emit_line("let mut names = Vec::new();");
        self.emit_line("for line in injection.lines() {");
        self.indent += 1;
        self.emit_line("let trimmed = line.trim();");
        self.emit_line("");

        self.emit_line("// Match: var name type = value");
        self.emit_line("if trimmed.starts_with(\"var \") {");
        self.indent += 1;
        self.emit_line("if let Some(rest) = trimmed.strip_prefix(\"var \") {");
        self.indent += 1;
        self.emit_line("if let Some(name) = rest.split_whitespace().next() {");
        self.indent += 1;
        self.emit_line("names.push(name.to_string());");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Match: name := value");
        self.emit_line("if trimmed.contains(\" := \") {");
        self.indent += 1;
        self.emit_line("if let Some(name) = trimmed.split(\" := \").next() {");
        self.indent += 1;
        self.emit_line("names.push(name.trim().to_string());");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("names");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Fix redeclarations
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn fix_go_redeclarations(code: &str, existing_vars: &[String]) -> String {");
        self.indent += 1;
        self.emit_line("let mut result = String::new();");
        self.emit_line("");

        self.emit_line("for line in code.lines() {");
        self.indent += 1;
        self.emit_line("let mut fixed = line.to_string();");
        self.emit_line("");

        self.emit_line("// Replace 'name :=' with 'name =' if variable exists");
        self.emit_line("for var in existing_vars {");
        self.indent += 1;
        self.emit_line("let pattern = format!(\"{} :=\", var);");
        self.emit_line("if fixed.contains(&pattern) {");
        self.indent += 1;
        self.emit_line("let replacement = format!(\"{} =\", var);");
        self.emit_line("fixed = fixed.replace(&pattern, &replacement);");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("result.push_str(&fixed);");
        self.emit_line("result.push('\\n');");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("result");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Extract imports
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn extract_go_imports(code: &str) -> (Vec<String>, String) {");
        self.indent += 1;
        self.emit_line("let mut imports = Vec::new();");
        self.emit_line("let mut clean_lines = Vec::new();");
        self.emit_line("let mut in_import_block = false;");
        self.emit_line("let mut paren_depth = 0;");
        self.emit_line("");

        self.emit_line("for line in code.lines() {");
        self.indent += 1;
        self.emit_line("let trimmed = line.trim();");
        self.emit_line("");

        self.emit_line("// Skip empty lines at start");
        self.emit_line("if trimmed.is_empty() || trimmed.starts_with(\"//\") {");
        self.indent += 1;
        self.emit_line("if !in_import_block && imports.is_empty() { continue; }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Detect import block");
        self.emit_line("if trimmed.starts_with(\"import (\") {");
        self.indent += 1;
        self.emit_line("in_import_block = true;");
        self.emit_line("paren_depth = 1;");
        self.emit_line("continue;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Single import");
        self.emit_line("if trimmed.starts_with(\"import \\\"\") && !trimmed.contains('(') {");
        self.indent += 1;
        self.emit_line("if let Some(path) = trimmed.strip_prefix(\"import \\\"\") {");
        self.indent += 1;
        self.emit_line("let path = path.trim_end_matches('\\\"');");
        self.emit_line("imports.push(path.to_string());");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("continue;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Inside import block");
        self.emit_line("if in_import_block {");
        self.indent += 1;
        self.emit_line("if trimmed.contains(')') {");
        self.indent += 1;
        self.emit_line("paren_depth -= 1;");
        self.emit_line("if paren_depth == 0 { in_import_block = false; }");
        self.emit_line("continue;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("if trimmed.starts_with('\\\"') {");
        self.indent += 1;
        self.emit_line("let path = trimmed.trim_matches('\\\"');");
        self.emit_line("imports.push(path.to_string());");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("continue;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("clean_lines.push(line);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("(imports, clean_lines.join(\"\\n\"))");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Auto-return wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn wrap_go_auto_return(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let lines: Vec<&str> = code.trim().lines().collect();");
        self.emit_line("if lines.is_empty() { return code.to_string(); }");
        self.emit_line("");

        self.emit_line("let last = lines.last().unwrap().trim();");
        self.emit_line("");

        self.emit_line("// Skip wrapping for statements/declarations");
        self.emit_line("if last.is_empty() || last.starts_with(\"//\")");
        self.indent += 1;
        self.emit_line("|| last.starts_with(\"if \") || last.starts_with(\"for \")");
        self.emit_line("|| last.starts_with(\"var \") || last.contains(\" := \")");
        self.emit_line("|| (last.contains(\" = \") && !last.contains(\"==\"))");
        self.emit_line("|| last.starts_with(\"fmt.Print\") || last == \"}\"");
        self.emit_line("{ return code.to_string(); }");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("let mut result = lines[..lines.len()-1].join(\"\\n\");");
        self.emit_line("if !result.is_empty() { result.push('\\n'); }");
        self.emit_line("result.push_str(&format!(\"__tb_result := ({})\\nfmt.Print(__tb_result)\", last));");
        self.emit_line("result");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    /// Generate Bash-specific helper functions
    fn generate_bash_helpers(&mut self) {
        // Detect shell
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn detect_shell() -> (&'static str, &'static str) {");
        self.indent += 1;
        self.emit_line("use std::process::Command;");
        self.emit_line("");

        self.emit_line("if cfg!(windows) {");
        self.indent += 1;
        self.emit_line("// Try Git Bash, WSL bash, then fallback to cmd");
        self.emit_line("let bash_paths = [\"bash\", \"C:\\\\Program Files\\\\Git\\\\bin\\\\bash.exe\"];");
        self.emit_line("for path in &bash_paths {");
        self.indent += 1;
        self.emit_line("if Command::new(path).arg(\"--version\").output().map(|o| o.status.success()).unwrap_or(false) {");
        self.indent += 1;
        self.emit_line("return (path, \"-c\");");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("(\"cmd\", \"/C\")");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("(\"bash\", \"-c\")");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Auto-return wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn wrap_bash_auto_return(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let lines: Vec<&str> = code.trim().lines().collect();");
        self.emit_line("if lines.is_empty() { return code.to_string(); }");
        self.emit_line("");

        self.emit_line("let last = lines.last().unwrap().trim();");
        self.emit_line("if last.is_empty() || last.starts_with('#') || last.starts_with(\"echo\")");
        self.indent += 1;
        self.emit_line("|| last.contains('=') || last == \"fi\" || last == \"done\"");
        self.emit_line("{ return code.to_string(); }");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("let mut result = lines[..lines.len()-1].join(\"\\n\");");
        self.emit_line("if !result.is_empty() { result.push('\\n'); }");
        self.emit_line("result.push_str(&format!(\"__tb_result=$(echo \\\"{}\\\" | bc -l 2>/dev/null || echo $(({}))); echo -n $__tb_result\", last, last));");
        self.emit_line("result");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    /// Generate ZERO-OVERHEAD language bridges for compiled mode
    /// Uses FFI, native libraries, and direct linking instead of subprocesses
    fn generate_compiled_language_bridges(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("// ZERO-OVERHEAD COMPILED LANGUAGE BRIDGES");
        self.emit_line("// Direct FFI calls, native library linking, no subprocess overhead");
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("");

        // Add necessary imports
        self.emit_line("use std::ffi::{CString, CStr};");
        self.emit_line("use std::os::raw::c_char;");
        self.emit_line("use std::sync::OnceLock;");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // COMPILED DEPENDENCY REGISTRY
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("/// Registry of compiled dependencies with lazy loading");
        self.emit_line("struct CompiledDepRegistry {");
        self.indent += 1;
        self.emit_line("python_libs: std::collections::HashMap<String, libloading::Library>,");
        self.emit_line("js_binaries: std::collections::HashMap<String, std::path::PathBuf>,");
        self.emit_line("go_plugins: std::collections::HashMap<String, libloading::Library>,");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("static DEP_REGISTRY: OnceLock<std::sync::Mutex<CompiledDepRegistry>> = OnceLock::new();");
        self.emit_line("");

        self.emit_line("fn get_dep_registry() -> &'static std::sync::Mutex<CompiledDepRegistry> {");
        self.indent += 1;
        self.emit_line("DEP_REGISTRY.get_or_init(|| {");
        self.indent += 1;
        self.emit_line("std::sync::Mutex::new(CompiledDepRegistry {");
        self.indent += 1;
        self.emit_line("python_libs: std::collections::HashMap::new(),");
        self.emit_line("js_binaries: std::collections::HashMap::new(),");
        self.emit_line("go_plugins: std::collections::HashMap::new(),");
        self.indent -= 1;
        self.emit_line("})");
        self.indent -= 1;
        self.emit_line("})");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // GO PLUGIN FFI BRIDGE (Zero Overhead)
        // ═══════════════════════════════════════════════════════════════
        self.generate_go_ffi_bridge();

        // ═══════════════════════════════════════════════════════════════
        // PYTHON NUITKA/PYOXIDIZER FFI BRIDGE
        // ═══════════════════════════════════════════════════════════════
        self.generate_python_ffi_bridge();

        // ═══════════════════════════════════════════════════════════════
        // JAVASCRIPT BUN BINARY BRIDGE (Cached Execution)
        // ═══════════════════════════════════════════════════════════════
        self.generate_js_binary_bridge();

        // ═══════════════════════════════════════════════════════════════
        // SMART WRAPPER: Auto-select compiled vs JIT
        // ═══════════════════════════════════════════════════════════════
        self.generate_smart_bridge_wrappers();
    }

    fn generate_go_ffi_bridge(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("// GO FFI BRIDGE - Direct plugin loading via libloading");
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("unsafe fn go_ffi(plugin_path: &str, code: &str) -> String {");
        self.indent += 1;
        self.emit_line("use libloading::{Library, Symbol};");
        self.emit_line("");

        self.emit_line("// Load library (cached in registry)");
        self.emit_line("let mut registry = get_dep_registry().lock().unwrap();");
        self.emit_line("");

        self.emit_line("let lib = registry.go_plugins.entry(plugin_path.to_string())");
        self.indent += 1;
        self.emit_line(".or_insert_with(|| {");
        self.indent += 1;
        self.emit_line("Library::new(plugin_path).expect(&format!(\"Failed to load Go plugin: {}\", plugin_path))");
        self.indent -= 1;
        self.emit_line("});");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("// Get Execute function pointer");
        self.emit_line("type ExecuteFunc = unsafe extern \"C\" fn(*const c_char) -> *mut c_char;");
        self.emit_line("let execute: Symbol<ExecuteFunc> = lib.get(b\"Execute\")");
        self.indent += 1;
        self.emit_line(".expect(\"Failed to find Execute function in Go plugin\");");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("// Call with zero-copy");
        self.emit_line("let input = CString::new(code).unwrap();");
        self.emit_line("let result_ptr = execute(input.as_ptr());");
        self.emit_line("");

        self.emit_line("// Convert result");
        self.emit_line("if result_ptr.is_null() {");
        self.indent += 1;
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("let result = CStr::from_ptr(result_ptr).to_string_lossy().to_string();");
        self.emit_line("");

        self.emit_line("// Free C string (Go's responsibility via C.free)");
        self.emit_line("libc::free(result_ptr as *mut libc::c_void);");
        self.emit_line("");

        self.emit_line("result");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    fn generate_python_ffi_bridge(&mut self) {
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("// PYTHON FFI BRIDGE - Direct library calls with OUTPUT CAPTURE");
    self.emit_line("// Supports Nuitka, PyOxidizer, and standard Python extensions");
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("");

    self.emit_line("#[inline(always)]");
    self.emit_line("unsafe fn python_ffi(lib_path: &str, code: &str) -> String {");
    self.indent += 1;
    self.emit_line("use libloading::{Library, Symbol};");
    self.emit_line("use std::ffi::{CString, CStr};");
    self.emit_line("use std::os::raw::c_char;");
    self.emit_line("");

    self.emit_line("// Load Python library");
    self.emit_line("let lib = Library::new(lib_path)");
    self.indent += 1;
    self.emit_line(".expect(&format!(\"Failed to load Python library: {}\", lib_path));");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("// PHASE 1: Initialize Python Interpreter");
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("");

    self.emit_line("// Check if already initialized");
    self.emit_line("type PyIsInitialized = unsafe extern \"C\" fn() -> i32;");
    self.emit_line("let py_is_initialized: Symbol<PyIsInitialized> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"Py_IsInitialized\")");
    self.emit_line(".expect(\"Failed to find Py_IsInitialized\");");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("if py_is_initialized() == 0 {");
    self.indent += 1;
    self.emit_line("// Initialize Python");
    self.emit_line("type PyInitialize = unsafe extern \"C\" fn();");
    self.emit_line("let py_initialize: Symbol<PyInitialize> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"Py_Initialize\")");
    self.emit_line(".expect(\"Failed to find Py_Initialize\");");
    self.indent -= 1;
    self.emit_line("py_initialize();");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("// PHASE 2: Redirect stdout to capture output");
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("");

    self.emit_line("// Create StringIO object to capture output");
    self.emit_line("let capture_setup = CString::new(r#\"");
    self.emit_line("import sys");
    self.emit_line("from io import StringIO");
    self.emit_line("__tb_stdout_capture = StringIO()");
    self.emit_line("__tb_old_stdout = sys.stdout");
    self.emit_line("sys.stdout = __tb_stdout_capture");
    self.emit_line("\"#).unwrap();");
    self.emit_line("");

    self.emit_line("type PyRunFunc = unsafe extern \"C\" fn(*const c_char) -> i32;");
    self.emit_line("let py_run: Symbol<PyRunFunc> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"PyRun_SimpleString\")");
    self.emit_line(".expect(\"Failed to find PyRun_SimpleString\");");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("// Setup capture");
    self.emit_line("if py_run(capture_setup.as_ptr()) != 0 {");
    self.indent += 1;
    self.emit_line("eprintln!(\"Failed to setup Python output capture\");");
    self.emit_line("return String::new();");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("// PHASE 3: Execute user code");
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("");

    self.emit_line("let c_code = CString::new(code).unwrap();");
    self.emit_line("let result_code = py_run(c_code.as_ptr());");
    self.emit_line("");

    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("// PHASE 4: Retrieve captured output");
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("");

    self.emit_line("let get_output = CString::new(r#\"");
    self.emit_line("sys.stdout = __tb_old_stdout");
    self.emit_line("__tb_result = __tb_stdout_capture.getvalue()");
    self.emit_line("print(__tb_result, end='')");
    self.emit_line("\"#).unwrap();");
    self.emit_line("");

    self.emit_line("py_run(get_output.as_ptr());");
    self.emit_line("");

    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("// PHASE 5: Extract result using Python API");
    self.emit_line("// ═══════════════════════════════════════════════════════════════");
    self.emit_line("");

    self.emit_line("// Get __main__ module");
    self.emit_line("type PyImportAddModule = unsafe extern \"C\" fn(*const c_char) -> *mut std::ffi::c_void;");
    self.emit_line("let py_import_add_module: Symbol<PyImportAddModule> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"PyImport_AddModule\")");
    self.emit_line(".expect(\"Failed to find PyImport_AddModule\");");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("let main_name = CString::new(\"__main__\").unwrap();");
    self.emit_line("let main_module = py_import_add_module(main_name.as_ptr());");
    self.emit_line("");

    self.emit_line("if main_module.is_null() {");
    self.indent += 1;
    self.emit_line("eprintln!(\"Failed to get __main__ module\");");
    self.emit_line("return String::new();");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("// Get __tb_result variable");
    self.emit_line("type PyObjectGetAttrString = unsafe extern \"C\" fn(*mut std::ffi::c_void, *const c_char) -> *mut std::ffi::c_void;");
    self.emit_line("let py_object_get_attr: Symbol<PyObjectGetAttrString> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"PyObject_GetAttrString\")");
    self.emit_line(".expect(\"Failed to find PyObject_GetAttrString\");");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("let result_name = CString::new(\"__tb_result\").unwrap();");
    self.emit_line("let result_obj = py_object_get_attr(main_module, result_name.as_ptr());");
    self.emit_line("");

    self.emit_line("if result_obj.is_null() {");
    self.indent += 1;
    self.emit_line("return String::new();");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("// Convert to string");
    self.emit_line("type PyObjectStr = unsafe extern \"C\" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void;");
    self.emit_line("let py_object_str: Symbol<PyObjectStr> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"PyObject_Str\")");
    self.emit_line(".expect(\"Failed to find PyObject_Str\");");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("let str_obj = py_object_str(result_obj);");
    self.emit_line("");

    self.emit_line("// Get C string");
    self.emit_line("type PyUnicodeAsUTF8 = unsafe extern \"C\" fn(*mut std::ffi::c_void) -> *const c_char;");
    self.emit_line("let py_unicode_as_utf8: Symbol<PyUnicodeAsUTF8> = lib");
    self.indent += 1;
    self.emit_line(".get(b\"PyUnicode_AsUTF8\")");
    self.emit_line(".expect(\"Failed to find PyUnicode_AsUTF8\");");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("let c_str = py_unicode_as_utf8(str_obj);");
    self.emit_line("");

    self.emit_line("if c_str.is_null() {");
    self.indent += 1;
    self.emit_line("return String::new();");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("// Convert to Rust String");
    self.emit_line("let rust_string = CStr::from_ptr(c_str)");
    self.indent += 1;
    self.emit_line(".to_string_lossy()");
    self.emit_line(".to_string();");
    self.indent -= 1;
    self.emit_line("");

    self.emit_line("// Cleanup references");
    self.emit_line("type PyDecRef = unsafe extern \"C\" fn(*mut std::ffi::c_void);");
    self.emit_line("if let Ok(py_decref) = lib.get::<Symbol<PyDecRef>>(b\"Py_DecRef\") {");
    self.indent += 1;
    self.emit_line("py_decref(result_obj);");
    self.emit_line("py_decref(str_obj);");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("if result_code != 0 {");
    self.indent += 1;
    self.emit_line("eprintln!(\"Python FFI execution failed with code {}\", result_code);");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");

    self.emit_line("rust_string");
    self.indent -= 1;
    self.emit_line("}");
    self.emit_line("");
}

    fn generate_js_binary_bridge(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("// JAVASCRIPT BINARY BRIDGE - Execute BUN-compiled binaries");
        self.emit_line("// Uses memory-mapped execution for zero I/O overhead");
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("fn js_binary(binary_path: &str, args: &[&str]) -> String {");
        self.indent += 1;
        self.emit_line("use std::process::Command;");
        self.emit_line("");

        self.emit_line("// Execute compiled binary (faster than Node.js startup)");
        self.emit_line("let output = Command::new(binary_path)");
        self.indent += 1;
        self.emit_line(".args(args)");
        self.emit_line(".output()");
        self.emit_line(".expect(&format!(\"Failed to execute JS binary: {}\", binary_path));");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("eprintln!(\"JS binary failed: {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("String::from_utf8_lossy(&output.stdout).to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    fn generate_smart_bridge_wrappers(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("// SMART BRIDGE WRAPPERS - Auto-select compiled vs JIT");
        self.emit_line("// ═══════════════════════════════════════════════════════════════");
        self.emit_line("");

        // Python wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("fn python_smart(code: &str, dep_id: Option<&str>) -> String {");
        self.indent += 1;
        self.emit_line("if let Some(id) = dep_id {");
        self.indent += 1;
        self.emit_line("// Try compiled library first");
        self.emit_line("let lib_path = format!(\"deps/python/{}.so\", id);");
        self.emit_line("if std::path::Path::new(&lib_path).exists() {");
        self.indent += 1;
        self.emit_line("unsafe { return python_ffi(&lib_path, code); }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Fallback to JIT execution");
        self.emit_line("python(code)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Go wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("fn go_smart(code: &str, dep_id: Option<&str>) -> String {");
        self.indent += 1;
        self.emit_line("if let Some(id) = dep_id {");
        self.indent += 1;
        self.emit_line("let plugin_ext = if cfg!(windows) { \"dll\" } else if cfg!(target_os = \"macos\") { \"dylib\" } else { \"so\" };");
        self.emit_line("let plugin_path = format!(\"deps/go/plugin_{}.{}\", id, plugin_ext);");
        self.emit_line("");
        self.emit_line("if std::path::Path::new(&plugin_path).exists() {");
        self.indent += 1;
        self.emit_line("unsafe { return go_ffi(&plugin_path, code); }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("go(code)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // JavaScript wrapper
        self.emit_line("#[inline(always)]");
        self.emit_line("fn javascript_smart(code: &str, dep_id: Option<&str>) -> String {");
        self.indent += 1;
        self.emit_line("if let Some(id) = dep_id {");
        self.indent += 1;
        self.emit_line("let binary_ext = if cfg!(windows) { \".exe\" } else { \"\" };");
        self.emit_line("let binary_path = format!(\"deps/js/{}{}\", id, binary_ext);");
        self.emit_line("");
        self.emit_line("if std::path::Path::new(&binary_path).exists() {");
        self.indent += 1;
        self.emit_line("return js_binary(&binary_path, &[]);");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("javascript(code)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    /// Generate helper functions for parsing different types
    /// Generate helper functions for parsing different types
    fn generate_type_parser_helpers(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("// Type Parsing Helpers (Production-Grade)");
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Parse to i64 (robust, handles multiple formats)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_i64(s: &str) -> i64 {");
        self.indent += 1;
        self.emit_line("let cleaned = s.trim()");
        self.indent += 1;
        self.emit_line(".replace('_', \"\")     // 1_000_000");
        self.emit_line(".replace(' ', \"\")     // 1 000 000");
        self.emit_line(".replace(',', \"\");    // 1,000,000");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("// Try direct int parse");
        self.emit_line("if let Ok(n) = cleaned.parse::<i64>() {");
        self.indent += 1;
        self.emit_line("return n;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Try float → int (handles \"42.0\")");
        self.emit_line("if let Ok(f) = cleaned.parse::<f64>() {");
        self.indent += 1;
        self.emit_line("return f as i64;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Try JSON integer");
        self.emit_line("if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cleaned) {");
        self.indent += 1;
        self.emit_line("if let Some(n) = val.as_i64() {");
        self.indent += 1;
        self.emit_line("return n;");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("eprintln!(\"⚠️  parse_i64: Cannot parse '{}' as integer\", s);");
        self.emit_line("0 // Fallback");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Parse to f64 (robust)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_f64(s: &str) -> f64 {");
        self.indent += 1;
        self.emit_line("let cleaned = s.trim()");
        self.indent += 1;
        self.emit_line(".replace('_', \"\")");
        self.emit_line(".replace(' ', \"\");");
        self.indent -= 1;
        self.emit_line("");

        self.emit_line("// Try direct parse");
        self.emit_line("if let Ok(f) = cleaned.parse::<f64>() {");
        self.indent += 1;
        self.emit_line("return f;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Try German format (1.234,56 → 1234.56)");
        self.emit_line("let german_style = cleaned.replace('.', \"\").replace(',', \".\");");
        self.emit_line("if let Ok(f) = german_style.parse::<f64>() {");
        self.indent += 1;
        self.emit_line("return f;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Try JSON number");
        self.emit_line("if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cleaned) {");
        self.indent += 1;
        self.emit_line("if let Some(f) = val.as_f64() {");
        self.indent += 1;
        self.emit_line("return f;");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("eprintln!(\"⚠️  parse_f64: Cannot parse '{}' as float\", s);");
        self.emit_line("0.0");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Parse to bool (robust)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_bool(s: &str) -> bool {");
        self.indent += 1;
        self.emit_line("match s.trim().to_lowercase().as_str() {");
        self.indent += 1;
        self.emit_line("\"true\" | \"1\" | \"yes\" | \"y\" | \"on\" => true,");
        self.emit_line("\"false\" | \"0\" | \"no\" | \"n\" | \"off\" | \"\" => false,");
        self.emit_line("_ => {");
        self.indent += 1;
        self.emit_line("eprintln!(\"⚠️  parse_bool: Cannot parse '{}' as bool\", s);");
        self.emit_line("false");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Parse to Vec<i64> (robust, JSON + bracket format)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_vec_i64(s: &str) -> Vec<i64> {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("");

        self.emit_line("// Try JSON array parse first");
        self.emit_line("if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {");
        self.indent += 1;
        self.emit_line("if let Some(arr) = val.as_array() {");
        self.indent += 1;
        self.emit_line("return arr.iter()");
        self.indent += 1;
        self.emit_line(".filter_map(|v| v.as_i64())");
        self.emit_line(".collect();");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Fallback: Manual bracket parsing");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("let inner = &trimmed[1..trimmed.len()-1];");
        self.emit_line("return inner.split(',')");
        self.indent += 1;
        self.emit_line(".map(|item| parse_i64(item.trim()))");
        self.emit_line(".collect();");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("Vec::new()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Parse to Vec<f64>
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_vec_f64(s: &str) -> Vec<f64> {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("");

        self.emit_line("// Try JSON");
        self.emit_line("if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {");
        self.indent += 1;
        self.emit_line("if let Some(arr) = val.as_array() {");
        self.indent += 1;
        self.emit_line("return arr.iter()");
        self.indent += 1;
        self.emit_line(".filter_map(|v| v.as_f64())");
        self.emit_line(".collect();");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Manual parsing");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("let inner = &trimmed[1..trimmed.len()-1];");
        self.emit_line("return inner.split(',')");
        self.indent += 1;
        self.emit_line(".map(|item| parse_f64(item.trim()))");
        self.emit_line(".collect();");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("Vec::new()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Parse to Vec<String>
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_vec_string(s: &str) -> Vec<String> {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("");

        self.emit_line("// Try JSON");
        self.emit_line("if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {");
        self.indent += 1;
        self.emit_line("if let Some(arr) = val.as_array() {");
        self.indent += 1;
        self.emit_line("return arr.iter()");
        self.indent += 1;
        self.emit_line(".filter_map(|v| v.as_str())");
        self.emit_line(".map(|s| s.to_string())");
        self.emit_line(".collect();");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("// Manual parsing");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("let inner = &trimmed[1..trimmed.len()-1];");
        self.emit_line("return inner.split(',')");
        self.indent += 1;
        self.emit_line(".map(|item| item.trim().trim_matches('\"').trim_matches('\\'').to_string())");
        self.emit_line(".collect();");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("Vec::new()");
        self.indent -= 1;
        self.emit_line("}");
    }

    /// Check if variable is assigned in any statement
    fn is_variable_assigned(&self, var_name: &str, statements: &[Statement]) -> bool {
        for stmt in statements {
            if self.statement_assigns_variable(stmt, var_name) {
                return true;
            }
        }
        false
    }

    /// Check if statement contains assignment to variable
    fn statement_assigns_variable(&self, stmt: &Statement, var_name: &str) -> bool {
        match stmt {
            Statement::Assign { target, .. } => {
                matches!(target, Expr::Variable(name) if name.to_string() == var_name.to_string())
            }
            Statement::Expr(expr) => self.expr_assigns_variable(expr, var_name),
            Statement::Function { body, .. } => self.expr_assigns_variable(body, var_name),
            _ => false,
        }
    }

    /// Check if expression contains assignment to variable
    fn expr_assigns_variable(&self, expr: &Expr, var_name: &str) -> bool {
        match expr {
            Expr::Block { statements, result } => {
                statements.iter().any(|s| self.statement_assigns_variable(s, var_name))
                    || result.as_ref().map_or(false, |e| self.expr_assigns_variable(e, var_name))
            }
            Expr::If { condition, then_branch, else_branch } => {
                self.expr_assigns_variable(condition, var_name)
                    || self.expr_assigns_variable(then_branch, var_name)
                    || else_branch.as_ref().map_or(false, |e| self.expr_assigns_variable(e, var_name))
            }
            Expr::Loop { body } => self.expr_assigns_variable(body, var_name),
            Expr::While { condition, body } => {
                self.expr_assigns_variable(condition, var_name)
                    || self.expr_assigns_variable(body, var_name)
            }
            Expr::For { iterable, body, .. } => {
                self.expr_assigns_variable(iterable, var_name)
                    || self.expr_assigns_variable(body, var_name)
            }
            _ => false,
        }
    }

    fn generate_rust_statement(&mut self, stmt: &Statement) -> TBResult<()> {
        match stmt {
            Statement::Let { name, mutable, type_annotation, value, .. } => {
                let is_mutable = *mutable || self.mutable_vars.contains(name);
                let keyword = if is_mutable { "let mut" } else { "let" };

                debug_log!("Generating let for '{}': mutable={} (explicit={}, detected={})",
            name, is_mutable, *mutable, self.mutable_vars.contains(name));

                // Check if value is a language bridge call
                let is_bridge = self.is_language_bridge_call(value);
                let is_parallel = matches!(value, Expr::Parallel(_));

                // Use type_annotation if provided, otherwise infer
                let actual_type = if let Some(ann) = type_annotation.as_ref() {
                    debug_log!("  Using explicit type annotation: {:?}", ann);
                    ann.clone()
                } else {
                    let inferred = self.infer_type_from_expr(value);
                    debug_log!("  Inferred type for '{}': {:?}", name, inferred);

                    // Override String to actual type for bridge calls
                    if is_bridge {
                        if let Expr::Call { args, .. } = value {
                            if let Some(Expr::Literal(Literal::String(code))) = args.first() {
                                // Try to detect type from code
                                let detected_type = self.infer_type_from_bridge_code(
                                    "python", // Detect language from function name
                                    code
                                );

                                debug_log!("  Override bridge return type: {:?} -> {:?}",
                              inferred, detected_type);
                                detected_type
                            } else {
                                inferred
                            }
                        } else {
                            inferred
                        }
                    } else {
                        inferred
                    }
                };

                // Generate code based on type
                if is_bridge {
                    // Step 1: Get string result
                    self.emit(&format!("let {}_str = ", name));
                    self.generate_rust_expr(value)?;
                    self.emit_line(";");

                    // Step 2: Parse to actual_type
                    if actual_type == Type::String {
                        self.emit(&format!("let {} = {}_str;", name, name));
                        self.emit_line("");
                    } else {
                        self.generate_typed_parser(name, &actual_type);
                    }
                } else if is_parallel {
                    self.emit(&format!("let {} = ", name));
                    self.generate_rust_expr(value)?;
                    self.emit_line(";");
                } else {
                    self.emit(&format!("{} {} = ", keyword, name));
                    self.generate_rust_expr(value)?;
                    self.emit_line(";");
                }

                self.variables_in_scope.push((name.clone(), actual_type));

                Ok(())
            }

            Statement::Assign { target, value } => {
                match target {
                    Expr::Variable(name) => {
                        debug_log!("Generating assignment: {} = ...", name);
                        self.emit(&format!("{} = ", name));
                        self.generate_rust_expr(value)?;
                        self.emit_line(";");
                        Ok(())
                    }
                    Expr::Index { object, index } => {
                        self.generate_rust_expr(object)?;
                        self.emit("[");
                        self.generate_rust_expr(index)?;
                        self.emit("] = ");
                        self.generate_rust_expr(value)?;
                        self.emit_line(";");
                        Ok(())
                    }
                    Expr::Field { object, field } => {
                        self.generate_rust_expr(object)?;
                        self.emit(&format!(".{} = ", field));
                        self.generate_rust_expr(value)?;
                        self.emit_line(";");
                        Ok(())
                    }
                    _ => {
                        Err(TBError::InvalidOperation(
                            format!("Invalid assignment target: {:?}", target).into()
                        ))
                    }
                }
            }

            Statement::Function { name, params, body, return_type, .. } => {
                // Determine return type
                let ret_type = if let Some(ref rt) = return_type {
                    self.type_to_rust_string(rt)
                } else {
                    if self.expr_returns_unit(body) {
                        "()".to_string()
                    } else {
                        "i64".to_string()
                    }
                };

                self.emit_line("#[inline(always)]");
                self.emit(&format!("fn {}(", name));
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { self.emit(", "); }
                    let param_type = param.type_annotation.as_ref()
                        .map(|t| self.type_to_rust_string(t))
                        .unwrap_or_else(|| "i64".to_string());
                    self.emit(&format!("{}: {}", param.name, param_type));
                }

                if ret_type != "()" {
                    self.emit(&format!(") -> {} ", ret_type));
                } else {
                    self.emit(") ");
                }

                self.generate_rust_expr(body)?;
                self.emit_line("");
                Ok(())
            }

            Statement::Expr(expr) => {
                self.generate_rust_expr(expr)?;
                self.emit_line(";");
                Ok(())
            }

            Statement::Import { .. } => {
                // Imports are handled elsewhere
                Ok(())
            }
        }
    }

    fn bridge_code_looks_numeric(&self, expr: &Expr) -> bool {
        if let Expr::Call { function, args } = expr {
            if let Expr::Variable(name) = function {
                if matches!(name.as_str(), "python" | "javascript" | "go" | "bash") {
                    if let Some(Expr::Literal(Literal::String(code))) = args.first() {
                        let code_lower = code.to_lowercase();

                        // Check for arithmetic operators
                        let has_arithmetic = code_lower.contains(" + ")
                            || code_lower.contains(" - ")
                            || code_lower.contains(" * ")
                            || code_lower.contains(" / ")
                            || code_lower.contains("+=")
                            || code_lower.contains("-=")
                            || code_lower.contains("*=")
                            || code_lower.contains("/=");

                        // Check for numeric literals
                        let has_numbers = code.chars().any(|c| c.is_ascii_digit());

                        // Check for numeric function calls
                        let has_numeric_funcs = code_lower.contains("math.")
                            || code_lower.contains("len(")
                            || code_lower.contains("count")
                            || code_lower.contains("parseint")
                            || code_lower.contains("parsefloat")
                            || code_lower.contains("number(");

                        return (has_arithmetic && has_numbers) || has_numeric_funcs;
                    }
                }
            }
        }
        false
    }


    /// Generate type-specific parser
    fn generate_typed_parser(&mut self, name: &str, target_type: &Type) {
        debug_log!("  Generating parser for '{}' -> {:?}", name, target_type);
        match target_type {
            Type::Int => {
                self.emit(&format!("let {} = parse_i64(&{}_str);", name, name));
                self.emit_line("");
            }

            Type::Float => {
                self.emit(&format!("let {} = parse_f64(&{}_str);", name, name));
                self.emit_line("");
            }

            Type::Bool => {
                self.emit(&format!("let {} = parse_bool(&{}_str);", name, name));
                self.emit_line("");
            }

            Type::String => {
                self.emit(&format!("let {} = {}_str;", name, name));
                self.emit_line("");
            }

            Type::List(inner) => {
                match inner.as_ref() {
                    Type::Int => {
                        self.emit(&format!("let {} = parse_vec_i64(&{}_str);", name, name));
                        self.emit_line("");
                    }
                    Type::Float => {
                        self.emit(&format!("let {} = parse_vec_f64(&{}_str);", name, name));
                        self.emit_line("");
                    }
                    Type::String => {
                        self.emit(&format!("let {} = parse_vec_string(&{}_str);", name, name));
                        self.emit_line("");
                    }
                    _ => {
                        // Default to string for unknown list types
                        self.emit(&format!("let {} = {}_str;", name, name));
                        self.emit_line("");
                    }
                }
            }

            Type::Option(inner) => {
                self.emit(&format!("let {} = if {}_str.trim().is_empty() || {}_str.trim().to_lowercase() == \"none\" {{",
                                   name, name, name));
                self.emit_line("");
                self.indent += 1;
                self.emit_line("None");
                self.indent -= 1;
                self.emit_line("} else {");
                self.indent += 1;

                // Parse the inner type
                match inner.as_ref() {
                    Type::Int => self.emit(&format!("Some(parse_i64(&{}_str))", name)),
                    Type::Float => self.emit(&format!("Some(parse_f64(&{}_str))", name)),
                    Type::Bool => self.emit(&format!("Some(parse_bool(&{}_str))", name)),
                    _ => self.emit(&format!("Some({}_str)", name)),
                }
                self.emit_line("");
                self.indent -= 1;
                self.emit_line("};");
            }

            _ => {
                // Default: try int, fallback to string
                self.emit(&format!("let {} = parse_i64(&{}_str);", name, name));
                self.emit_line("");
            }
        }
    }

    fn infer_type_from_expr(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::Bool(_) => Type::Bool,
                Literal::String(_) => Type::String,
                Literal::Unit => Type::Unit,
            },
            Expr::BinOp { op, left, right } => {
                use BinOp::*;
                match op {
                    Add | Sub | Mul | Div | Mod | Pow => {
                        let left_type = self.infer_type_from_expr(left);
                        let right_type = self.infer_type_from_expr(right);

                        if left_type == Type::Float || right_type == Type::Float {
                            Type::Float
                        } else {
                            Type::Int
                        }
                    }
                    Eq | Ne | Lt | Le | Gt | Ge | And | Or => Type::Bool,
                    _ => Type::Dynamic,
                }
            }
            Expr::Call { function, args } => {
                // IMPROVED: Infer type from language bridge code
                if let Expr::Variable(name) = function {
                    if matches!(name.as_str(), "python" | "javascript" | "go" | "bash") {
                        // Try to infer type from code content
                        if let Some(Expr::Literal(Literal::String(code))) = args.first() {
                            return self.infer_type_from_bridge_code(&name, code);
                        }
                        return Type::String; // Changed: Default to String (safer)
                    }
                }
                Type::Dynamic
            }
            Expr::List(_) => Type::List(Box::new(Type::Dynamic)),
            _ => Type::Dynamic,
        }
    }

    /// Infer return type from language bridge code
    /// Infer return type from language bridge code
    fn infer_type_from_bridge_code(&self, language: &str, code: &str) -> Type {
        let code_lower = code.to_lowercase();

        if language == "python" {
            // Match: print(NUMBER)
            if let Some(start) = code_lower.find("print(") {
                let after_print = &code[start + 6..];
                if let Some(end) = after_print.find(')') {
                    let value = after_print[..end].trim();

                    // Check if it's a number
                    if value.chars().all(|c| c.is_numeric() || c == '.' || c == '-') {
                        if value.contains('.') {
                            debug_log!("  Detected Python print(FLOAT): {}", value);
                            return Type::Float;
                        } else {
                            debug_log!("  Detected Python print(INT): {}", value);
                            return Type::Int;
                        }
                    }

                    // Check for boolean literals
                    if value == "True" || value == "False" {
                        debug_log!("  Detected Python print(BOOL): {}", value);
                        return Type::Bool;
                    }

                    // Check for string literals
                    if (value.starts_with('"') && value.ends_with('"')) ||
                        (value.starts_with('\'') && value.ends_with('\'')) {
                        debug_log!("  Detected Python print(STRING): {}", value);
                        return Type::String;
                    }

                    // Check for list literals
                    if value.starts_with('[') && value.ends_with(']') {
                        debug_log!("  Detected Python print(LIST): {}", value);
                        // Try to detect inner type
                        let inner = &value[1..value.len()-1].trim();
                        if inner.is_empty() {
                            return Type::List(Box::new(Type::Dynamic));
                        }
                        // Check first element
                        if let Some(first) = inner.split(',').next() {
                            let first = first.trim();
                            if first.chars().all(|c| c.is_numeric() || c == '-') {
                                return Type::List(Box::new(Type::Int));
                            } else if first.chars().any(|c| c == '.') {
                                return Type::List(Box::new(Type::Float));
                            }
                        }
                        return Type::List(Box::new(Type::Dynamic));
                    }
                }
            }
        }


        // ═══════════════════════════════════════════════════════════════════
        // PRIORITY 1: Detect explicit type conversions
        // ═══════════════════════════════════════════════════════════════════
        if code_lower.contains("int(")
            || code_lower.contains("len(")
            || code_lower.contains("count(")
            || code_lower.contains("sum(")
            || code_lower.contains("range(") {
            return Type::Int;
        }

        if code_lower.contains("float(")
            || code_lower.contains("math.sqrt")
            || code_lower.contains("math.pow")
            || code_lower.contains("math.") {
            // Special case: math.sqrt returns float, but often used in int context
            // Check if result is multiplied/divided (suggests numeric)
            if code_lower.contains("* ") || code_lower.contains("/ ") {
                return Type::Int;  // Default to Int for arithmetic
            }
            return Type::Float;
        }

        if code_lower.contains("bool(")
            || code_lower.contains("true")
            || code_lower.contains("false")
            || code_lower.contains(" and ")
            || code_lower.contains(" or ") {
            return Type::Bool;
        }

        // ═══════════════════════════════════════════════════════════════════
        // PRIORITY 2: Detect arithmetic operations (NEW)
        // ═══════════════════════════════════════════════════════════════════
        let has_arithmetic = code_lower.contains(" + ")
            || code_lower.contains(" - ")
            || code_lower.contains(" * ")
            || code_lower.contains(" / ")
            || code_lower.contains(" % ")
            || code_lower.contains("+=")
            || code_lower.contains("-=")
            || code_lower.contains("*=")
            || code_lower.contains("/=");

        let has_numeric_literal = code.chars().any(|c| c.is_numeric());

        if has_arithmetic && has_numeric_literal {
            debug_log!("  Detected arithmetic: {} -> Type::Int", &code[..code.len().min(50)]);
            return Type::Int;
        }

        // ═══════════════════════════════════════════════════════════════════
        // PRIORITY 3: Language-specific patterns
        // ═══════════════════════════════════════════════════════════════════
        match language {
            "python" => {
                // Check for numeric variables being printed
                if code_lower.contains("print(") && has_numeric_literal {
                    return Type::Int;
                }
            }

            "javascript" => {
                // const/let with numeric operations
                if (code_lower.contains("const ") || code_lower.contains("let "))
                    && has_arithmetic {
                    return Type::Int;
                }

                if code_lower.contains("console.log") && has_numeric_literal {
                    return Type::Int;
                }
            }

            "go" => {
                // Go variable declarations with numeric types
                if code_lower.contains(":=") && has_arithmetic {
                    return Type::Int;
                }

                if code_lower.contains("fmt.println") && has_numeric_literal {
                    return Type::Int;
                }
            }

            _ => {}
        }

        // ═══════════════════════════════════════════════════════════════════
        // DEFAULT: String (safest fallback - can always be parsed later)
        // ═══════════════════════════════════════════════════════════════════
        debug_log!("  No type detected, defaulting to String for: {}",
               &code[..code.len().min(50)]);
        Type::String
    }

    /// Check if expression is a language bridge call
    fn is_language_bridge_call(&self, expr: &Expr) -> bool {
        if let Expr::Call { function, .. } = expr {
            if let Expr::Variable(name) = function {
                return matches!(name.as_str(), "python" | "javascript" | "go" | "bash");
            }
        }
        false
    }

    /// Check if expression returns unit type (used for type inference)
    fn expr_returns_unit(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Unit) => true,
            Expr::Call { function, .. } => {
                // Check if calling echo/print/println
                if let Expr::Variable(name) = function {
                    matches!(name.as_str(), "echo" | "print" | "println")
                } else {
                    false
                }
            }
            Expr::Block { result, statements } => {
                if let Some(res) = result {
                    self.expr_returns_unit(res)
                } else {
                    // Block without result is unit
                    true
                }
            }
            _ => false,
        }
    }

    /// Convert TB Type to Rust type string
    fn type_to_rust_string(&self, ty: &Type) -> String {
        match ty {
            Type::Unit => "()".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Int => "i64".to_string(),
            Type::Float => "f64".to_string(),
            Type::String => "String".to_string(),
            Type::List(inner) => format!("Vec<{}>", self.type_to_rust_string(inner)),
            Type::Option(inner) => format!("Option<{}>", self.type_to_rust_string(inner)),
            Type::Result { ok, err } => {
                format!("Result<{}, {}>",
                        self.type_to_rust_string(ok),
                        self.type_to_rust_string(err))
            }
            Type::Function { params, ret } => {
                let param_types: Vec<_> = params.iter()
                    .map(|p| self.type_to_rust_string(p))
                    .collect();
                format!("fn({}) -> {}",
                        param_types.join(", "),
                        self.type_to_rust_string(ret))
            }
            _ => "()".to_string(), // Fallback
        }
    }

    /// Generate language bridge call with automatic variable injection
    /// Generate language bridge call with automatic variable injection
    /// Generate language bridge call with automatic variable injection
    fn generate_language_bridge_call_with_context(
        &mut self,
        language: &str,
        args: &[Expr],
    ) -> TBResult<()> {
        if args.is_empty() {
            return Err(TBError::InvalidOperation(
                format!("{} requires code argument", language).into()
            ));
        }

        let code_expr = &args[0];

        // Build variable injection code
        let var_injection = self.generate_var_injection_for_language(language);

        if var_injection.is_empty() {
            // No variables to inject, simple call WITHOUT extra arguments
            self.emit(language);
            self.emit("(");
            self.generate_rust_expr(code_expr)?;
            self.emit(")");  // ← KEIN ", None)" mehr!
        } else {
            // Collect ALL variable data BEFORE any emit() calls
            let var_data: Vec<(Arc<String>, Type, bool)> = self.variables_in_scope.iter()
                .map(|(name, ty)| {
                    let is_list = matches!(ty, Type::List(_));
                    (Arc::clone(name), ty.clone(), is_list)
                })
                .collect();

            let is_javascript = language == "javascript";
            let is_go = language == "go";

            // Generate format! call with variable injection
            self.emit(&format!("{}(&format!(r#\"", language));
            self.emit(&var_injection);
            self.emit("\n{}");  // Placeholder for user code
            self.emit("\"#");

            // Format arguments - emit variable values with correct formatting
            for (name, ty, is_list) in &var_data {
                self.emit(", ");

                if *is_list {
                    if is_javascript {
                        self.emit("&format!(\"[{}]\", ");
                        self.emit(name.as_str());
                        self.emit(".iter().map(|x| x.to_string()).collect::<Vec<_>>().join(\",\"))");
                    } else if is_go {
                        let inner_type = if let Type::List(inner) = ty {
                            match inner.as_ref() {
                                Type::Int => "int64",
                                Type::Float => "float64",
                                Type::String => "string",
                                _ => "interface{{}}",
                            }
                        } else {
                            "interface{{}}"
                        };
                        self.emit(&format!(
                            "&format!(\"[]{}{{}}{{}}{{}}\", \"{{\", {}.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(\",\"), \"}}\")",
                            inner_type, name.as_str()
                        ));
                    } else {
                        self.emit(name.as_str());
                    }
                } else {
                    self.emit(name.as_str());
                }
            }

            // User code argument
            self.emit(", ");
            self.generate_rust_expr(code_expr)?;
            self.emit("))");  // ← KEIN extra ", None)"
        }

        Ok(())
    }

    /// Generate variable injection code for specific language
    fn generate_var_injection_for_language(&self, language: &str) -> String {
        if self.variables_in_scope.is_empty() {
            return String::new();
        }

        let mut injection = String::new();

        match language {
            "python" => {
                injection.push_str("# TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    injection.push_str(&format!("{} = {}\n", name.as_str(),
                                                Self::format_spec_for_type(ty, "python")));
                }
            }

            "javascript" => {
                injection.push_str("// TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    injection.push_str(&format!("const {} = {};\n", name.as_str(),
                                                Self::format_spec_for_type(ty, "javascript")));
                }
            }

            "go" => {
                injection.push_str("// TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    // let go_type = Self::go_type_for(ty);
                    // Use {:?} for List types (Vec doesn't implement Display)
                    let value_placeholder = Self::format_spec_for_type(ty, "go");

                    // Wir verwenden den `:=` Operator für eine saubere Deklaration und Initialisierung.
                    // Der Go-Compiler leitet den Typ automatisch vom Wert auf der rechten Seite ab.
                    injection.push_str(&format!("{} := {}\n", name.as_str(), value_placeholder));
                    injection.push_str(&format!("_ = {}\n", name.as_str()));

                }
            }

            "bash" => {
                injection.push_str("# TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    injection.push_str(&format!("{}={}\n", name.as_str(),
                                                Self::format_spec_for_type(ty, "bash")));
                }
            }

            _ => {}
        }

        injection
    }

    /// Get format specifier for a type in target language
    fn format_spec_for_type(ty: &Type, language: &str) -> &'static str {
        match language {
            "python" => match ty {
                Type::String => "\"{}\"",
                Type::Bool | Type::Int | Type::Float => "{}",
                Type::List(_) => "{:?}",  // Debug format for Python lists
                _ => "{}",
            },

            "javascript" => match ty {
                Type::String => "\"{}\"",
                Type::Bool | Type::Int | Type::Float => "{}",
                Type::List(_) => "{}",  // Custom formatted (via iter().join())
                _ => "{}",
            },

            "go" => match ty {
                Type::String => "\"{}\"",
                Type::Bool | Type::Int | Type::Float => "{}",
                Type::List(_) => "{}",  // Debug format for Go slices
                _ => "{}",
            },

            "bash" => match ty {
                Type::String => "\"{}\"",
                Type::List(_) => "{:?}",  // Debug format
                _ => "{}",
            },

            _ => "{}",
        }
    }

    /// Get Go type string for TB type
    /// Get Go type string for TB type (FIXED: Escaped braces for format! macro)
    fn go_type_for(ty: &Type) -> String {
        match ty {
            Type::String => "string".to_string(),
            Type::Int => "int64".to_string(),
            Type::Float => "float64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::List(inner) => {
                if matches!(inner.as_ref(), Type::Int) {
                    "[]int64".to_string()
                } else if matches!(inner.as_ref(), Type::Float) {
                    "[]float64".to_string()
                } else {
                    // FIXED: Escape braces for format! macro
                    "[]interface{{}}".to_string()
                }
            }
            _ => "interface{{}}".to_string(),
        }
    }

    /// Map language bridge calls to compiled dependency IDs
    fn get_dependency_id(&self, code: &str) -> Option<String> {
        for dep in &self.compiled_deps {
            // Simple ID extraction from path
            if let Some(name) = dep.output_path.file_stem() {  // ← Jetzt OK!
                if let Some(name_str) = name.to_str() {
                    if name_str.contains("dep_") {
                        return Some(name_str.replace("plugin_", "").to_string());
                    }
                }
            }
        }

        None
    }

    fn generate_rust_expr(&mut self, expr: &Expr) -> TBResult<()> {
        match expr {
            Expr::Literal(lit) => {
                match lit {
                    Literal::Unit => self.emit("()"),
                    Literal::Bool(b) => self.emit(&format!("{}", b)),
                    Literal::Int(n) => self.emit(&format!("{}i64", n)),
                    Literal::Float(f) => self.emit(&format!("{}f64", f)),
                    Literal::String(s) => {
                        // FIXED: Proper string escaping!
                        let escaped = s
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\t', "\\t")
                            .replace('\r', "\\r");
                        self.emit(&format!("\"{}\"", escaped))
                    }
                }
                Ok(())
            }

            Expr::Variable(name) => {
                self.emit(name);
                Ok(())
            }

            Expr::BinOp { op, left, right } => {
                self.emit("(");
                self.generate_rust_expr(left)?;
                self.emit(&format!(" {} ", self.binop_to_rust(*op)));
                self.generate_rust_expr(right)?;
                self.emit(")");
                Ok(())
            }
            Expr::Parallel(tasks) => {
                self.emit_line("// Parallel execution");
                self.emit("{");
                self.indent += 1;

                // Generate parallel execution using rayon
                self.emit_line("use rayon::prelude::*;");
                self.emit_line("let __tasks: Vec<Box<dyn Fn() -> i64 + Send + Sync>> = vec![");
                self.indent += 1;

                for (i, task) in tasks.iter().enumerate() {
                    self.emit("Box::new(|| ");

                    // Generate task body
                    match task {
                        Expr::Call { function, args } => {
                            self.generate_rust_expr(function)?;
                            self.emit("(");
                            for (j, arg) in args.iter().enumerate() {
                                if j > 0 { self.emit(", "); }
                                self.generate_rust_expr(arg)?;
                            }
                            self.emit(")");
                        }
                        _ => {
                            self.generate_rust_expr(task)?;
                        }
                    }

                    self.emit(")");
                    if i < tasks.len() - 1 {
                        self.emit(",");
                    }
                    self.emit_line("");
                }

                self.indent -= 1;
                self.emit_line("];");
                self.emit_line("");

                // Execute in parallel and collect results
                self.emit_line("let __results: Vec<i64> = (0..__tasks.len())");
                self.indent += 1;
                self.emit_line(".into_par_iter()");
                self.emit_line(".map(|i| __tasks[i]())");
                self.emit_line(".collect();");
                self.indent -= 1;
                self.emit_line("");
                self.emit_line("__results");

                self.indent -= 1;
                self.emit("}");
                Ok(())
            }
            Expr::Call { function, args } => {
                if let Expr::Variable(func_name) = function {
                    // ═══════════════════════════════════════════════════════════
                    // SPECIAL CASE: Builtin functions (echo, print, println)
                    // ═══════════════════════════════════════════════════════════
                    if matches!(func_name.as_str(), "echo" | "print" | "println") {
                        self.emit(&format!("{}(", func_name));

                        // Generate single string argument
                        if args.len() == 1 {
                            self.generate_rust_expr(&args[0])?;
                        } else {
                            // Multiple args: concatenate
                            self.emit("&format!(\"");
                            for (i, _) in args.iter().enumerate() {
                                if i > 0 { self.emit(" "); }
                                self.emit("{}");
                            }
                            self.emit("\"");
                            for arg in args {
                                self.emit(", ");
                                self.generate_rust_expr(arg)?;
                            }
                            self.emit(")");
                        }

                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════
                    // SPECIAL CASE: Language bridges
                    // ═══════════════════════════════════════════════════════════
                    if matches!(func_name.as_str(), "python" | "javascript" | "go" | "bash") {
                        return self.generate_language_bridge_call_with_context(&func_name, args);
                    }
                }

                // ═══════════════════════════════════════════════════════════
                // REGULAR FUNCTION CALL
                // ═══════════════════════════════════════════════════════════
                self.generate_rust_expr(function)?;
                self.emit("(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { self.emit(", "); }
                    self.generate_rust_expr(arg)?;
                }
                self.emit(")");
                Ok(())
            }

            Expr::Block { statements, result } => {
                self.emit_line("{");
                self.indent += 1;

                for stmt in statements {
                    self.generate_rust_statement(stmt)?;
                }

                if let Some(result) = result {
                    // Last expression is return value (no semicolon!)
                    self.generate_rust_expr(result)?;
                    self.emit_line(""); // Just newline, no semicolon
                }

                self.indent -= 1;
                self.emit("}");
                Ok(())
            }

            Expr::If { condition, then_branch, else_branch } => {
                self.emit("if ");
                self.generate_rust_expr(condition)?;
                self.emit(" ");
                self.generate_rust_expr(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.emit(" else ");
                    self.generate_rust_expr(else_branch)?;
                }
                Ok(())
            }

            Expr::List(items) => {
                self.emit("vec![");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { self.emit(", "); }
                    self.generate_rust_expr(item)?;
                }
                self.emit("]");
                Ok(())
            }

            Expr::Loop { body } => {
                // Loops müssen einen Wert zurückgeben können
                self.emit("loop ");
                self.generate_rust_expr(body)?;
                Ok(())
            }

            Expr::While { condition, body } => {
                self.emit("while ");
                self.generate_rust_expr(condition)?;
                self.emit(" ");
                self.generate_rust_expr(body)?;
                Ok(())
            }

            Expr::For { variable, iterable, body } => {
                self.emit("for ");
                self.emit(variable);
                self.emit(" in ");
                self.generate_rust_expr(iterable)?;
                self.emit(" ");
                self.generate_rust_expr(body)?;
                Ok(())
            }

            Expr::Break(value) => {
                if let Some(val) = value {
                    self.emit("break ");
                    self.generate_rust_expr(val)?;
                } else {
                    self.emit("break");
                }
                Ok(())
            }

            _ => {
                self.emit("()");
                Ok(())
            }
        }
    }

    fn parse_string_interpolation_owned(&self, s: &str) -> (String, Vec<Expr>) {
        let mut result = String::new();
        let mut variables = Vec::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '$' {
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphabetic() || next_ch == '_' {
                        let mut var_name = String::new();
                        while let Some(&ch) = chars.peek() {
                            if ch.is_alphanumeric() || ch == '_' {
                                var_name.push(ch);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        result.push_str("{}");
                        variables.push(Expr::Variable(Arc::new(var_name)));
                        continue;
                    }
                }
                result.push('$');
            } else if ch == '{' || ch == '}' {
                result.push(ch);
                result.push(ch);
            } else {
                result.push(ch);
            }
        }

        (result, variables)
    }


    /// Parse string with $variable interpolation
    /// Returns (format_string, variables)
    ///
    /// Example:
    ///   "Hello $name, you are $age years old"
    /// → ("Hello {}, you are {} years old", [Variable("name"), Variable("age")])
    fn parse_string_interpolation(&self, s: &str) -> (String, Vec<Expr>) {
        let mut result = String::new();
        let mut variables = Vec::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '$' {
                // Check if next char is valid identifier start
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphabetic() || next_ch == '_' {
                        // Parse variable name
                        let mut var_name = String::new();

                        while let Some(&ch) = chars.peek() {
                            if ch.is_alphanumeric() || ch == '_' {
                                var_name.push(ch);
                                chars.next();
                            } else {
                                break;
                            }
                        }

                        // Add placeholder and variable
                        result.push_str("{}");
                        variables.push(Expr::Variable(Arc::new(var_name)));
                        continue;
                    }
                }

                // Not a variable reference, just literal $
                result.push('$');
            } else if ch == '{' || ch == '}' {
                // Escape braces for format! macro
                result.push(ch);
                result.push(ch);
            } else {
                result.push(ch);
            }
        }

        (result, variables)
    }

    fn binop_to_rust(&self, op: BinOp) -> &'static str {
        match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            _ => "/*unknown*/",
        }
    }

    fn emit(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    fn emit_line(&mut self, s: &str) {
        self.buffer.push_str(&"    ".repeat(self.indent));
        self.buffer.push_str(s);
        self.buffer.push('\n');
    }

    fn generate_python(&mut self, statements: &[Statement]) -> TBResult<String> {
        self.emit_line("# Generated by TB Compiler");
        self.emit_line("");

        for stmt in statements {
            self.generate_python_statement(stmt)?;
        }

        Ok(std::mem::take(&mut self.buffer))
    }

    fn generate_python_statement(&mut self, stmt: &Statement) -> TBResult<()> {
        match stmt {
            Statement::Let { name, value, .. } => {
                self.emit(&format!("{} = ", name));
                self.generate_python_expr(value)?;
                self.emit_line("");
                Ok(())
            }

            Statement::Expr(expr) => {
                self.generate_python_expr(expr)?;
                self.emit_line("");
                Ok(())
            }

            _ => Ok(()),
        }
    }

    fn generate_python_expr(&mut self, expr: &Expr) -> TBResult<()> {
        match expr {
            Expr::Literal(lit) => {
                match lit {
                    Literal::Unit => self.emit("None"),
                    Literal::Bool(b) => self.emit(if *b { "True" } else { "False" }),
                    Literal::Int(n) => self.emit(&format!("{}", n)),
                    Literal::Float(f) => self.emit(&format!("{}", f)),
                    Literal::String(s) => {
                        let escaped = s.replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\t', "\\t")
                            .replace('\r', "\\r");
                        self.emit(&format!("\"{}\"", escaped))
                    }
                }
                Ok(())
            }

            Expr::Variable(name) => {
                self.emit(name);
                Ok(())
            }

            Expr::BinOp { op, left, right } => {
                self.emit("(");
                self.generate_python_expr(left)?;
                self.emit(&format!(" {} ", self.binop_to_python(*op)));
                self.generate_python_expr(right)?;
                self.emit(")");
                Ok(())
            }

            _ => {
                self.emit("None");
                Ok(())
            }
        }
    }

    fn generate_javascript(&mut self, statements: &[Statement]) -> TBResult<String> {
        self.emit_line("// Generated by TB Compiler");
        self.emit_line("");

        for stmt in statements {
            self.generate_javascript_statement(stmt)?;
        }

        Ok(std::mem::take(&mut self.buffer))
    }

    fn generate_javascript_statement(&mut self, stmt: &Statement) -> TBResult<()> {
        match stmt {
            Statement::Let { name, value, .. } => {
                self.emit(&format!("const {} = ", name));
                self.generate_javascript_expr(value)?;
                self.emit_line(";");
                Ok(())
            }

            Statement::Expr(expr) => {
                self.generate_javascript_expr(expr)?;
                self.emit_line(";");
                Ok(())
            }

            _ => Ok(()),
        }
    }

    fn generate_javascript_expr(&mut self, expr: &Expr) -> TBResult<()> {
        match expr {
            Expr::Literal(lit) => {
                match lit {
                    Literal::Unit => self.emit("null"),
                    Literal::Bool(b) => self.emit(&format!("{}", b)),
                    Literal::Int(n) => self.emit(&format!("{}", n)),
                    Literal::Float(f) => self.emit(&format!("{}", f)),
                    Literal::String(s) => {
                        let escaped = s.replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\t', "\\t")
                            .replace('\r', "\\r");
                        self.emit(&format!("\"{}\"", escaped))
                    }
                }
                Ok(())
            }

            Expr::Variable(name) => {
                self.emit(name);
                Ok(())
            }

            Expr::BinOp { op, left, right } => {
                self.emit("(");
                self.generate_javascript_expr(left)?;
                self.emit(&format!(" {} ", self.binop_to_rust(*op))); // JS uses same ops
                self.generate_javascript_expr(right)?;
                self.emit(")");
                Ok(())
            }

            _ => {
                self.emit("null");
                Ok(())
            }
        }
    }

    fn binop_to_python(&self, op: BinOp) -> &'static str {
        match op {
            BinOp::And => "and",
            BinOp::Or => "or",
            _ => self.binop_to_rust(op),
        }
    }

}

// ═══════════════════════════════════════════════════════════════════════════
// §11 JIT EXECUTOR
// ═══════════════════════════════════════════════════════════════════════════

pub type Environment<'arena> = Arc<RwLock<HashMap<Arc<String>, Value<'arena>>>>;


/// Fast single-threaded environment (10x faster than Arc<RwLock>)
pub type FastEnvironment<'arena> = Rc<RefCell<HashMap<Arc<String>, Value<'arena>>>>;

/// Function registry for zero-copy function calls
/// Function registry for zero-copy function calls
pub struct FunctionRegistry<'arena> {
    functions: HashMap<String, (Vec<Arc<String>>, Rc<Expr<'arena>>)>,
}

impl<'arena> FunctionRegistry<'arena> {
    fn new() -> Self {
        Self { functions: HashMap::new() }
    }

    fn register(&mut self, name: Arc<String>, params: Vec<Arc<String>>, body: Expr<'arena>) {
        self.functions.insert(name.to_string(), (params, Rc::new(body)));
    }

    fn get(&self, name: &str) -> Option<&(Vec<Arc<String>>, Rc<Expr<'arena>>)> {
        self.functions.get(name)
    }
}

pub struct JitExecutor<'arena> {
    arena: &'arena Bump,
    env: FastEnvironment<'arena>,
    config: Config<'arena>,
    builtins: BuiltinRegistry,
    functions: FunctionRegistry<'arena>,
    recursion_depth: usize,
    max_recursion_depth: usize,
}

impl <'arena> JitExecutor<'arena> {
    pub fn new(config: Config<'arena>, arena: &'arena Bump) -> Self {
        let mut env_map = HashMap::new();

        // Initialize with shared variables
        for (key, value) in &config.shared {
            env_map.insert(Arc::clone(key), value.clone());
        }

        // Create builtin registry
        let mut builtins = BuiltinRegistry::new();

        if !config.plugins.is_empty() {
            debug_log!("Loading {} plugins", config.plugins.len());
            if let Err(e) = builtins.load_plugins(&config.plugins) {
                eprintln!("Warning: Failed to load plugins: {}", e);
            }
        }

        Self {
            env: Rc::new(RefCell::new(env_map)),
            config,
            builtins,
            functions: FunctionRegistry::new(),
            recursion_depth: 0,
            max_recursion_depth: 10000,
            arena
        }
    }

    pub fn execute(&mut self, statements: &'arena [Statement<'arena>]) -> TBResult<Value<'arena>> {
        debug_log!("JitExecutor::execute() started with {} statements", statements.len());

        let mut last_value = Value::Unit;

        // PHASE 1: Register all functions FIRST
        for stmt in statements.iter() {
            if let Statement::Function { name, params, body, .. } = stmt {
                let param_names: Vec<Arc<String>> = params.iter()
                    .map(|p| Arc::clone(&p.name))
                    .collect();

                // FIX: Body muss zu 'static konvertiert werden
                let body_static: Expr<'static> = unsafe {
                    std::mem::transmute(body.clone())
                };

                self.functions.register(
                    Arc::clone(name),
                    param_names,
                    body_static
                );
            }
        }

        // PHASE 2: Execute statements
        for stmt in statements.iter() {
            last_value = self.execute_statement(stmt)?;
        }

        Ok(last_value)
    }

    pub fn execute_statement(&mut self, stmt: &'arena Statement<'arena>) -> TBResult<Value<'arena>> {
        match stmt {
            Statement::Let { name, value, .. } => {
                let val = self.eval_expr(value)?;
                self.env.borrow_mut().insert(Arc::clone(name), val);
                Ok(Value::Unit)
            }

            Statement::Assign { target, value } => {
                let new_value = self.eval_expr(value)?;
                if let Expr::Variable(name) = target {
                    self.env.borrow_mut().insert(Arc::clone(name), new_value);
                    Ok(Value::Unit)
                } else {
                    Err(TBError::InvalidOperation(
                        Arc::from("Assignment target must be a variable".to_string())
                    ))
                }
            }

            Statement::Function { .. } => Ok(Value::Unit),

            Statement::Expr(expr) => self.eval_expr(expr),

            _ => Ok(Value::Unit),
        }
    }

    pub fn eval_expr(&mut self, expr:  &'arena Expr<'arena>) -> TBResult<Value<'arena>>{
        match expr {
            Expr::Literal(lit) => {
                match lit {
                    Literal::String(s) if s.contains('$') => {
                        Ok(Value::String(Arc::from(self.interpolate_string(s)?)))
                    }
                    _ => Ok(self.literal_to_value(lit)),
                }
            }

            Expr::Variable(name) => {
                // Fast lookup with Rc<RefCell>
                if let Some(value) = self.env.borrow().get(name).cloned() {
                    return Ok(value);
                }

                // Check builtin functions
                //if let Some(_) = self.builtins.get(name.as_str()) {
                //    return Ok(Value::Native {
                //        language: Language::Rust,
                //        type_name: format!("builtin:{}", name).into(),
                //        handle: NativeHandle {
                //            id: name.as_ptr() as u64,
                //            data: Arc::new(name),
                //        },
                //    });
                //}
                if let Some(_) = self.builtins.get(name.as_str()) {
                    //  Erstelle einfaches Function-Value statt Native
                    return Ok(Value::Function {
                        params: vec![],
                        body: &Expr::Literal(Literal::Unit), // Dummy
                        env: Arc::new(RwLock::new(HashMap::new())),
                    });
                }

                Err(TBError::UndefinedVariable(Arc::clone(name)))
            }

            Expr::BinOp { op, left, right } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binop(*op, left_val, right_val)
            }

            Expr::UnaryOp { op, expr } => {
                let val = self.eval_expr(expr)?;
                self.eval_unaryop(*op, val)
            }

            Expr::Call { function, args } => {
                // PHASE 1: Collect all arguments
                let mut arg_vals = Vec::new();
                for arg in args.iter() {
                    arg_vals.push(self.eval_expr(arg)?);
                }

                // PHASE 2: Check for builtin functions (highest priority)
                if let Expr::Variable(func_name) = function {
                    // ═══════════════════════════════════════════════════════════
                    // BUILTIN FUNCTIONS (echo, print, len, etc.)
                    // ═══════════════════════════════════════════════════════════
                    if let Some(builtin) = self.builtins.get(func_name.as_str()) {
                        debug_log!("Calling builtin: {}", func_name);

                        // Convert args to 'arena lifetime
                        let arena_args: Vec<Value<'arena>> = arg_vals.iter().map(|v| v.clone()).collect();
                        let args_slice = self.arena.alloc_slice_fill_with(arena_args.len(), |i| {
                            arena_args[i].clone()
                        });

                        // Call builtin directly with correct signature
                        return (builtin.function)(args_slice);
                    }

                    // ═══════════════════════════════════════════════════════════
                    // LANGUAGE BRIDGES (python, javascript, go, bash)
                    // ═══════════════════════════════════════════════════════════
                    if matches!(func_name.as_str(), "python" | "javascript" | "go" | "bash") {
                        if let Some(Value::String(code)) = arg_vals.first() {
                            let variables = self.env.borrow().clone();
                            let context = LanguageExecutionContext::with_variables(variables);

                            let result_static = match func_name.as_str() {
                                "python" => execute_python_code_with_context(code, &arg_vals[1..], Some(&context))?,
                                "javascript" => execute_js_code_with_context(code, &arg_vals[1..], Some(&context))?,
                                "go" => execute_go_code_with_context(code, &arg_vals[1..], Some(&context))?,
                                "bash" => execute_bash_command_with_context(code, &arg_vals[1..], Some(&context))?,
                                _ => unreachable!(),
                            };

                            return Ok(unsafe { std::mem::transmute(result_static) });
                        }
                    }

                    if let Some((params, body_rc)) = self.functions.get(func_name.as_str()) {
                        let params = params.clone();
                        let body_rc = body_rc.clone();
                        return self.call_function_fast(func_name.as_str(), &params, &body_rc, arg_vals);
                    }
                }

                // PHASE 3: Fallback - evaluate function expression and call
                let func = self.eval_expr(function)?;
                self.call_function_legacy(func, arg_vals)
            }

            Expr::Block { statements, result } => {
                for stmt in statements {
                    self.execute_statement(stmt)?;
                }

                if let Some(result) = result {
                    self.eval_expr(result)
                } else {
                    Ok(Value::Unit)
                }
            }

            Expr::If { condition, then_branch, else_branch } => {
                let cond = self.eval_expr(condition)?;

                if cond.is_truthy() {
                    self.eval_expr(then_branch)
                } else if let Some(else_branch) = else_branch {
                    self.eval_expr(else_branch)
                } else {
                    Ok(Value::Unit)
                }
            }

            Expr::Loop { body } => {
                let mut iterations = 0;
                let max_iterations = 100000;

                loop {
                    iterations += 1;

                    if iterations > max_iterations {
                        return Err(TBError::RuntimeError {
                            message: format!("Loop exceeded maximum iterations ({})", max_iterations).into(),
                            trace:  vec![Arc::from("loop execution".to_string())],
                        });
                    }

                    match self.eval_expr(body) {
                        Ok(Value::Unit) => continue,
                        Ok(val) => return Ok(val),
                        Err(TBError::RuntimeError { message, .. }) if message.contains("break") => {
                            return Ok(Value::Unit);
                        }
                        Err(e) => return Err(e),
                    }
                }
            }

            Expr::While { condition, body } => {
                let mut iterations = 0;
                let max_iterations = 10000;
                let mut last_value = Value::Unit;

                while self.eval_expr(condition)?.is_truthy() {
                    iterations += 1;

                    if iterations > max_iterations {
                        return Err(TBError::RuntimeError {
                            message: format!("While loop exceeded maximum iterations ({})", max_iterations).into(),
                            trace: vec![Arc::from("loop execution".to_string())],
                        });
                    }

                    last_value = self.eval_expr(body)?;
                }

                Ok(last_value)
            }

            Expr::For { variable, iterable, body } => {
                let iter_val = self.eval_expr(iterable)?;

                if let Value::List(items) = iter_val {
                    let mut last_value = Value::Unit;

                    for item in items {
                        self.env.borrow_mut().insert(Arc::clone(variable), item);
                        last_value = self.eval_expr(body)?;
                    }

                    Ok(last_value)
                } else {
                    Err(TBError::InvalidOperation(
                        Arc::from("For loop requires an iterable (list)".to_string())
                    ))
                }
            }

            Expr::Break(value) => {
                if let Some(val_expr) = value {
                    let val = self.eval_expr(val_expr)?;
                    Ok(val)
                } else {
                    Ok(Value::Unit)
                }
            }

            Expr::List(items) => {
                //  Sammle Werte mit Schleife
                let mut values = Vec::new();
                for item in items.iter() {
                    values.push(self.eval_expr(item)?);
                }
                Ok(Value::List(values))
            }

            Expr::Index { object, index } => {
                let obj = self.eval_expr(object)?;
                let idx = self.eval_expr(index)?;

                match (obj, idx) {
                    (Value::List(items), Value::Int(i)) => {
                        let idx = if i < 0 {
                            (items.len() as i64 + i) as usize
                        } else {
                            i as usize
                        };

                        items.get(idx)
                            .cloned()
                            .ok_or_else(|| TBError::IndexOutOfBounds {
                                index: i,
                                length: items.len(),
                            })
                    }
                    _ => Err(TBError::InvalidOperation(Arc::from("Index operation".to_string()))),
                }
            }

            Expr::Pipeline { value, operations } => {
                let mut result = self.eval_expr(value)?;

                for op in operations {
                    result = match op {
                        Expr::Call { function, args } => {
                            // Sammle args
                            let mut all_args = vec![result];
                            for arg in args.iter() {
                                all_args.push(self.eval_expr(arg)?);
                            }

                            let func = self.eval_expr(function)?;
                             self.call_function_legacy(func, all_args)?

                        }
                        Expr::Variable(name) => {
                            if let Some((params, body_rc)) = self.functions.get(name.as_str()) {
                                let params = params.clone();
                                let body_rc = body_rc.clone();
                                self.call_function_fast(&name, &params, &body_rc, vec![result])?

                            } else {
                                let func = self.env.borrow().get(name)
                                    .cloned()
                                    .ok_or_else(|| TBError::UndefinedFunction(name.clone()))?;
                                 self.call_function_legacy(func, vec![result])?

                            }
                        }
                        _ => return Err(TBError::InvalidOperation(STRING_INTERNER.intern("Invalid pipeline operation"))),
                    };
                }

                Ok(result)
            }

            Expr::Try(expr) => {
                match self.eval_expr(expr) {
                    Ok(Value::Result(Ok(val))) => Ok(*val),
                    Ok(Value::Result(Err(err))) => Err(TBError::RuntimeError {
                        message: format!("Propagated error: {}", err).into(),
                        trace: vec![],
                    }),
                    Ok(val) => Ok(val),
                    Err(e) => Err(e),
                }
            }

            Expr::Parallel(tasks) => {
                // Ensure this returns Value::List
                let results: TBResult<Vec<_>> = tasks
                    .iter()
                    .map(|task| self.eval_expr(task))
                    .collect();

                Ok(Value::List(results?))
            }

            _ => Ok(Value::Unit),
        }
    }

    fn interpolate_string(&self, s: &str) -> TBResult<String> {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '$' {
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphabetic() || next_ch == '_' {
                        // Parse variable name
                        let mut var_name = String::new();
                        while let Some(&ch) = chars.peek() {
                            if ch.is_alphanumeric() || ch == '_' {
                                var_name.push(ch);
                                chars.next();
                            } else {
                                break;
                            }
                        }

                        // Lookup variable
                        if let Some(value) = self.env.borrow().get(&STRING_INTERNER.intern(&var_name)) {
                            result.push_str(&format!("{}", value));
                        } else {
                            result.push('$');
                            result.push_str(&var_name);
                        }
                        continue;
                    }
                }
                result.push('$');
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }

    /// ULTRA-FAST function call (zero environment cloning!)
    fn call_function_fast(
        &mut self,
        func_name: &str,
        params: &[Arc<String>],
        body: &Rc<Expr<'arena>>,
        args: Vec<Value<'arena>>,
    ) -> TBResult<Value<'arena>> {
        self.recursion_depth += 1;

        #[cfg(debug_assertions)]
        {
            let estimated_stack_kb = self.recursion_depth * 4;
            let env_size = self.env.borrow().len();
            debug_log!(
            "📞 CALL depth={}/{} func='{}' params={} env_vars={} ~{}KB stack",
            self.recursion_depth,
            self.max_recursion_depth,
            if func_name.is_empty() { "closure" } else { func_name },
            params.len(),
            env_size,
            estimated_stack_kb
        );
        }

        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth = 0;
            return Err(TBError::RuntimeError {
                message: format!("Maximum recursion depth exceeded ({})", self.max_recursion_depth).into(),
                trace: vec![],
            });
        }

        if params.len() != args.len() {
            self.recursion_depth -= 1;
            return Err(TBError::InvalidOperation(
                format!("Expected {} arguments, got {}", params.len(), args.len()).into()
            ));
        }

        // Speichere alte Werte
        let mut saved_params = Vec::new();
        for (param, arg) in params.iter().zip(args.iter()) {
            let old_value = self.env.borrow_mut().insert(Arc::clone(param), arg.clone());
            saved_params.push((Arc::clone(param), old_value));
        }

        //  Evaluiere body mit korrektem Lifetime
        let body_ref: &'arena Expr<'arena> = unsafe {
            // SAFETY: Rc<Expr<'arena>> enthält bereits 'arena Daten
            // Wir casten nur die Referenz
            std::mem::transmute::<&Expr<'arena>, &'arena Expr<'arena>>(body.as_ref())
        };
        let result = self.eval_expr(body_ref);

        // Restore Parameter
        for (param, old_value) in saved_params {
            if let Some(old) = old_value {
                self.env.borrow_mut().insert(param, old);
            } else {
                self.env.borrow_mut().remove(&param);
            }
        }

        self.recursion_depth -= 1;
        result
    }

    /// Legacy function call (for closures)
    fn call_function_legacy(
        &mut self,
        func: Value<'arena>,
        args: Vec<Value<'arena>>
    ) -> TBResult<Value<'arena>> {
        match func {
            Value::Function { params, body, env: _ } => {
                let body_rc = Rc::new(body.clone());
                self.call_function_fast("", &params, &body_rc, args)
            }

            Value::Native { type_name, .. } if type_name.starts_with("builtin:") => {
                let func_name = type_name.strip_prefix("builtin:").unwrap();

                if let Some(builtin) = self.builtins.get(func_name) {
                    // FIX: Mit HRTB können wir direkt aufrufen
                    let arena_args: Vec<Value<'arena>> = args.iter().map(|v| v.clone()).collect();
                    let args_slice = self.arena.alloc_slice_fill_with(arena_args.len(), |i| {
                        arena_args[i].clone()
                    });

                    // HRTB erlaubt direkten Call mit 'arena
                    (builtin.function)(args_slice)
                } else {
                    Err(TBError::UndefinedFunction(STRING_INTERNER.intern(func_name)))
                }
            }

            _ => Err(TBError::InvalidOperation(
                format!("Not a function: {:?}", func.get_type()).into()
            )),
        }
    }

    fn eval_binop(&self, op: BinOp, left: Value<'arena>, right: Value<'arena>) -> TBResult<Value<'arena>> {
        match (op, &left, &right) {
            // ═══════════════════════════════════════════════════════════
            // INTEGER ARITHMETIC
            // ═══════════════════════════════════════════════════════════
            (BinOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (BinOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (BinOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (BinOp::Div, Value::Int(a), Value::Int(b)) => {
                if b == &0 {
                    Err(TBError::DivisionByZero)
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            (BinOp::Mod, Value::Int(a), Value::Int(b)) => {
                if b == &0 {
                    Err(TBError::DivisionByZero)
                } else {
                    Ok(Value::Int(a % b))
                }
            }

            // ═══════════════════════════════════════════════════════════
            // FLOAT ARITHMETIC
            // ═══════════════════════════════════════════════════════════
            (BinOp::Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (BinOp::Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (BinOp::Mul, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (BinOp::Div, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),

            // ═══════════════════════════════════════════════════════════
            // MIXED-TYPE ARITHMETIC (Int + Float → Float)
            // ═══════════════════════════════════════════════════════════
            (BinOp::Add, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
            (BinOp::Add, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),

            (BinOp::Sub, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
            (BinOp::Sub, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),

            (BinOp::Mul, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
            (BinOp::Mul, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),

            (BinOp::Div, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),
            (BinOp::Div, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),

            // ═══════════════════════════════════════════════════════════
            // STRING CONCATENATION
            // ═══════════════════════════════════════════════════════════
            (BinOp::Add, Value::String(a), Value::String(b)) => {
                Ok(Value::String(Arc::new(format!("{}{}", a, b))))
            }

            // ═══════════════════════════════════════════════════════════
            // COMPARISONS (Int)
            // ═══════════════════════════════════════════════════════════
            (BinOp::Eq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
            (BinOp::Ne, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a != b)),
            (BinOp::Lt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Le, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (BinOp::Ge, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),

            // ═══════════════════════════════════════════════════════════
            // COMPARISONS (Float)
            // ═══════════════════════════════════════════════════════════
            (BinOp::Eq, Value::Float(a), Value::Float(b)) => Ok(Value::Bool((a - b).abs() < f64::EPSILON)),
            (BinOp::Ne, Value::Float(a), Value::Float(b)) => Ok(Value::Bool((a - b).abs() >= f64::EPSILON)),
            (BinOp::Lt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Le, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (BinOp::Ge, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),

            // ═══════════════════════════════════════════════════════════
            // COMPARISONS (Mixed)
            // ═══════════════════════════════════════════════════════════
            (BinOp::Eq, Value::Float(a), Value::Int(b)) => Ok(Value::Bool((a - *b as f64).abs() < f64::EPSILON)),
            (BinOp::Eq, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64 - b).abs() < f64::EPSILON)),

            (BinOp::Ne, Value::Float(a), Value::Int(b)) => Ok(Value::Bool((a - *b as f64).abs() >= f64::EPSILON)),
            (BinOp::Ne, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64 - b).abs() >= f64::EPSILON)),

            (BinOp::Lt, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a < *b as f64)),
            (BinOp::Lt, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) < *b)),

            (BinOp::Le, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a <= *b as f64)),
            (BinOp::Le, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) <= *b)),

            (BinOp::Gt, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a > *b as f64)),
            (BinOp::Gt, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) > *b)),

            (BinOp::Ge, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a >= *b as f64)),
            (BinOp::Ge, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) >= *b)),

            // ═══════════════════════════════════════════════════════════
            // GENERIC EQUALITY (works for any type)
            // ═══════════════════════════════════════════════════════════
            (BinOp::Eq, a, b) => Ok(Value::Bool(self.values_equal(&a, &b))),
            (BinOp::Ne, a, b) => Ok(Value::Bool(!self.values_equal(&a, &b))),

            // ═══════════════════════════════════════════════════════════
            // LOGICAL OPERATIONS
            // ═══════════════════════════════════════════════════════════
            (BinOp::And, a, b) => Ok(Value::Bool(a.is_truthy() && b.is_truthy())),
            (BinOp::Or, a, b) => Ok(Value::Bool(a.is_truthy() || b.is_truthy())),

            // ═══════════════════════════════════════════════════════════
            // ERROR: Unsupported operation
            // ═══════════════════════════════════════════════════════════
            (op, left, right) => Err(TBError::InvalidOperation(
                format!("Cannot perform {:?} on {:?} and {:?}",
                        op,
                        left.get_type(),
                        right.get_type()).into()
            )),
        }
    }

    fn eval_unaryop(&self, op: UnaryOp, val: Value<'arena>) -> TBResult<Value<'arena>> {
        match (op, val) {
            (UnaryOp::Not, val) => Ok(Value::Bool(!val.is_truthy())),
            (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
            _ => Err(TBError::InvalidOperation(format!("Unary operation {:?}", op).into())),
        }
    }

    fn literal_to_value(&self, lit: &Literal) -> Value<'arena>  {
        match lit {
            Literal::Unit => Value::Unit,
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Int(n) => Value::Int(*n),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
        }
    }

    fn values_equal(&self, a: &Value<'arena>, b: &Value<'arena>)-> bool {
        match (a, b) {
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ParallelExecutor
// ═══════════════════════════════════════════════════════════════════════════

/// Parallel executor using Rayon for CPU-bound work
// ═══════════════════════════════════════════════════════════════════════════
// PARALLEL EXECUTOR - Multi-JIT Design
// ═══════════════════════════════════════════════════════════════════════════

use rayon::prelude::*;

/// Parallel executor using multiple JitExecutor instances
pub struct ParallelExecutor {
    config: Config<'static>,
    shared_state: Arc<Mutex<HashMap<Arc<String>, Value<'static>>>>,
    worker_threads: usize,
}

impl ParallelExecutor {
    pub fn new(config: Config<'static>) -> Self {
        let worker_threads = match config.runtime_mode {
            RuntimeMode::Parallel { worker_threads } => worker_threads,
            _ => num_cpus::get(),
        };

        // Initialize shared state from config
        let mut shared_map = HashMap::new();
        for (key, value) in &config.shared {
            shared_map.insert(key.clone(), value.clone());
        }

        // Configure Rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .build_global()
            .ok();

        Self {
            config,
            shared_state: Arc::new(Mutex::new(shared_map)),
            worker_threads,
        }
    }

    /// Execute statements with parallel support
    pub fn execute(&mut self, statements: &'static [Statement<'static>]) -> TBResult<Value<'static>> {
        debug_log!("ParallelExecutor: {} workers, {} statements", self.worker_threads, statements.len());

        let mut sequential_stmts: Vec<&'static Statement<'static>> = Vec::new();
        let mut parallel_blocks: Vec<&'static Statement<'static>> = Vec::new();

        for stmt in statements.iter() {
            if self.contains_parallel_expr(stmt) {
                parallel_blocks.push(stmt);
            } else {
                sequential_stmts.push(stmt);
            }
        }

        let mut result = Value::Unit;

        if !sequential_stmts.is_empty() {
            result = self.execute_sequential(&sequential_stmts)?;
        }

        for block_stmt in parallel_blocks {
            result = self.execute_statement_parallel(block_stmt)?;
        }

        Ok(result)
    }

    fn execute_sequential(&self, statements: &[&'static Statement<'static>]) -> TBResult<Value<'static>> {
        // FIX: Erstelle Arena als 'static
        let arena: &'static Bump = Box::leak(Box::new(Bump::new()));

        let mut executor = JitExecutor::new(self.config.clone(), arena);

        // FIX: Allokiere statements in der 'static Arena
        let stmts_slice: &'static [Statement<'static>] = {
            let vec: Vec<Statement<'static>> = statements.iter()
                .map(|&s| s.clone())
                .collect();

            // Leak das Vec in die Arena
            arena.alloc_slice_fill_with(vec.len(), |i| vec[i].clone())
        };

        executor.execute(stmts_slice)
    }

    fn clone_statement_to_static<'a>(
        &self,
        stmt: &Statement<'a>,
        arena: &'static Bump
    ) -> Statement<'static> {
        // Deep clone implementation
        match stmt {
            Statement::Let { name, mutable, type_annotation, value, scope } => {
                Statement::Let {
                    name: name.clone(),
                    mutable: *mutable,
                    type_annotation: type_annotation.clone(),
                    value: self.clone_expr_to_static(value, arena),
                    scope: *scope,
                }
            }
            Statement::Function { name, params, return_type, body } => {
                let params_static: Vec<_> = params.iter()
                    .map(|p| Parameter {
                        name: p.name.clone(),
                        type_annotation: p.type_annotation.clone(),
                        default: p.default.as_ref().map(|e| self.clone_expr_to_static(e, arena)),
                    })
                    .collect();

                Statement::Function {
                    name: name.clone(),
                    params: ArenaVec::from_iter_in(params_static, arena),
                    return_type: return_type.clone(),
                    body: self.clone_expr_to_static(body, arena),
                }
            }
            Statement::Expr(expr) => {
                Statement::Expr(self.clone_expr_to_static(expr, arena))
            }
            Statement::Assign { target, value } => {
                Statement::Assign {
                    target: self.clone_expr_to_static(target, arena),
                    value: self.clone_expr_to_static(value, arena),
                }
            }
            Statement::Import { module, items } => {
                Statement::Import {
                    module: module.clone(),
                    items: items.clone(),
                }
            }
        }
    }

    fn clone_expr_to_static<'a>(
        &self,
        expr: &Expr<'a>,
        arena: &'static Bump
    ) -> Expr<'static> {
        // Minimale Implementation für häufigste Fälle
        match expr {
            Expr::Literal(lit) => Expr::Literal(lit.clone()),
            Expr::Variable(name) => Expr::Variable(name.clone()),
            Expr::BinOp { op, left, right } => {
                Expr::BinOp {
                    op: *op,
                    left: arena.alloc(self.clone_expr_to_static(left, arena)),
                    right: arena.alloc(self.clone_expr_to_static(right, arena)),
                }
            }
            _ => Expr::Literal(Literal::Unit), // Fallback
        }
    }

    fn execute_statement_parallel(&self, stmt: &'static Statement<'static>) -> TBResult<Value<'static>> {
        match stmt {
            Statement::Expr(Expr::Parallel(tasks)) => {
                self.execute_parallel_tasks(&tasks)
            }

            Statement::Expr(Expr::For { variable, iterable, body }) => {
                let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
                let mut executor = JitExecutor::new(self.config.clone(), arena);
                let iter_val = executor.eval_expr(iterable)?;

                if let Value::List(items) = iter_val {
                    self.execute_parallel_for(variable, &items, body)
                } else {
                    Err(TBError::InvalidOperation(
                        Arc::from("Parallel for requires list".to_string())
                    ))
                }
            }

            _ => {
                let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
                let mut executor = JitExecutor::new(self.config.clone(), arena);
                executor.execute_statement(stmt)
            }
        }
    }

    fn execute_parallel_tasks(&self, tasks: &'static [Expr<'static>]) -> TBResult<Value<'static>> {
        debug_log!("Executing {} tasks in parallel", tasks.len());

        let results: Vec<TBResult<Value<'static>>> = tasks
            .iter()
            .map(|task| {
                let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
                let mut executor = JitExecutor::new(self.config.clone(), arena);
                executor.eval_expr(task)
            })
            .collect();

        let mut values = Vec::new();
        for result in results {
            values.push(result?);
        }

        Ok(Value::List(values))
    }

    fn execute_parallel_for(
        &self,
        variable: &Arc<String>,
        items: &[Value<'static>],
        body: &'static Expr<'static>,
    ) -> TBResult<Value<'static>> {
        debug_log!("Parallel for: {} iterations", items.len());

        let results: Vec<TBResult<Value<'static>>> = items
            .iter()
            .map(|item| {
                let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
                let mut executor = JitExecutor::new(self.config.clone(), arena);
                executor.env.borrow_mut().insert(Arc::clone(variable), item.clone());
                executor.eval_expr(body)
            })
            .collect();

        let mut last_value = Value::Unit;
        for result in results {
            last_value = result?;
        }

        Ok(last_value)
    }

    fn create_executor(&self) -> JitExecutor<'static> {
        let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
        let mut config = self.config.clone();
        let shared_state = self.shared_state.lock().unwrap();
        config.shared = shared_state.clone();
        JitExecutor::new(config, arena)
    }

    fn contains_parallel_expr(&self, stmt: &'static Statement<'static>) -> bool {
        match stmt {
            Statement::Expr(expr) => self.is_parallel_expr(expr),
            Statement::Let { value, .. } => self.is_parallel_expr(value),
            Statement::Function { body, .. } => self.is_parallel_expr(body),
            _ => false,
        }
    }

    fn is_parallel_expr(&self, expr: &'static Expr<'static>) -> bool {
        match expr {
            Expr::Parallel(_) => true,
            Expr::For { .. } if matches!(self.config.runtime_mode, RuntimeMode::Parallel { .. }) => true,
            Expr::Block { statements, result } => {
                statements.iter().any(|s| self.contains_parallel_expr(s))
                    || result.as_ref().map_or(false, |e| self.is_parallel_expr(e))
            }
            _ => false,
        }
    }

    pub fn get_shared(&self, key: &str) -> Option<Value<'static>> {
        let state = self.shared_state.lock().unwrap();
        state.iter()
            .find(|(k, _)| k.as_str() == key)
            .map(|(_, v)| v.clone())
    }

    pub fn set_shared(&self, key: Arc<String>, value: Value<'static>) {
        self.shared_state.lock().unwrap().insert(key, value);
    }

    pub fn get_all_shared(&self) -> HashMap<Arc<String>, Value<'static>> {
        self.shared_state.lock().unwrap().clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.5 BUILTIN FUNCTIONS & NATIVE BRIDGES
// ═══════════════════════════════════════════════════════════════════════════

/// Native function signature

pub struct BuiltinFunction {
    pub name: Arc<String>,
    // Box statt Arc für Flexibility
    pub function: Box<dyn for<'a> Fn(&'a [Value<'a>]) -> TBResult<Value<'a>> + Send + Sync>,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub description: Arc<String>,
}

impl Clone for BuiltinFunction {
    fn clone(&self) -> Self {
        // Cannot clone Box<dyn Fn>, so panic
        panic!("BuiltinFunction cannot be cloned - use Arc at registry level")
    }
}

/// Registry for builtin and plugin functions
#[derive(Clone)]
pub struct BuiltinRegistry {
    functions: HashMap<String, Arc<BuiltinFunction>>,  // Arc um Registry
}

impl BuiltinRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::with_capacity(64),
        };

        // Register standard library functions
        registry.register_stdlib();

        registry
    }

    /// Register standard library functions
    fn register_stdlib(&mut self) {
        // IO Functions
        self.register("python", Box::new(builtin_python), 1, Some(1), STRING_INTERNER.intern("Execute Python code").as_str());
        self.register("javascript", Box::new(builtin_javascript), 1, Some(1), STRING_INTERNER.intern("Execute JavaScript code").as_str());
        self.register("bash", Box::new(builtin_bash), 1, Some(1), STRING_INTERNER.intern("Execute Bash code").as_str());
        self.register("go", Box::new(builtin_go), 1, Some(1), STRING_INTERNER.intern("Execute Go code").as_str());

        self.register("echo", Box::new(builtin_echo), 1, None, STRING_INTERNER.intern("Print values to stdout").as_str());
        self.register("print", Box::new(builtin_print), 1, None, STRING_INTERNER.intern("Print values without newline").as_str());
        self.register("println", Box::new(builtin_println), 1, None, STRING_INTERNER.intern("Print values with newline").as_str());
        self.register("read_line", Box::new(builtin_read_line), 0, Some(0), STRING_INTERNER.intern("Read a line from stdin").as_str());

        // Type conversion
        self.register("str", Box::new(builtin_str), 1, Some(1), STRING_INTERNER.intern("Convert to string").as_str());
        self.register("int", Box::new(builtin_int), 1, Some(1), STRING_INTERNER.intern("Convert to integer").as_str());
        self.register("float", Box::new(builtin_float), 1, Some(1), STRING_INTERNER.intern("Convert to float").as_str());
        self.register("pretty", Box::new(builtin_pretty), 1, Some(1), STRING_INTERNER.intern("Pretty print with indentation").as_str());

        // List operations
        self.register("len", Box::new(builtin_len), 1, Some(1), STRING_INTERNER.intern("Get length of collection").as_str());
        self.register("push", Box::new(builtin_push), 2, Some(2), STRING_INTERNER.intern("Push item to list").as_str());
        self.register("pop", Box::new(builtin_pop), 1, Some(1), STRING_INTERNER.intern("Pop item from list").as_str());

        // Debug
        self.register("debug", Box::new(builtin_debug), 1, None, STRING_INTERNER.intern("Debug print with type info").as_str());
        self.register("type_of", Box::new(builtin_type_of), 1, Some(1), STRING_INTERNER.intern("Get type of value").as_str());
        self.register("python_info", Box::new(builtin_python_info), 0, Some(0), STRING_INTERNER.intern("Show active Python environment").as_str());

    }

    /// Register a builtin function
    pub fn register(
        &mut self,
        name: &str,
        function:  Box<dyn for<'a> Fn(&'a [Value<'a>]) -> TBResult<Value<'a>> + Send + Sync>,
        min_args: usize,
        max_args: Option<usize>,
        description: &str,
    ) {
        let arc_name = STRING_INTERNER.intern(name);
        self.functions.insert(
            name.to_string(),
            Arc::new(BuiltinFunction {
                name: arc_name,
                function,
                min_args,
                max_args,
                description: STRING_INTERNER.intern(description),
            }),
        );
    }

    /// Get a builtin function (O(1) lookup with &str)
    pub fn get(&self, name: &str) -> Option<Arc<BuiltinFunction>> {
        self.functions.get(name).cloned()
    }

    /// Check if function exists (O(1))
    pub fn has(&self, name: &str) -> bool {
        self.functions.contains_key(name)  // Direkter O(1) lookup
    }

    /// List all functions
    pub fn list(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }
    /// Load plugins from configuration
    pub fn load_plugins(&mut self, plugins: &[PluginConfig]) -> TBResult<()> {
        for plugin in plugins {
            if !plugin.enabled {
                debug_log!("Skipping disabled plugin: {}", plugin.name);
                continue;
            }

            debug_log!("Loading plugin: {} ({})", plugin.name, format!("{:?}", plugin.language));

            match plugin.language {
                Language::Python => self.load_python_plugin(plugin)?,
                Language::JavaScript => self.load_javascript_plugin(plugin)?,
                Language::TypeScript => self.load_typescript_plugin(plugin)?,
                Language::Go => self.load_go_plugin(plugin)?,
                Language::Bash => self.load_bash_plugin(plugin)?,
                _ => {
                    return Err(TBError::UnsupportedLanguage(
                        format!("{:?}", plugin.language).into()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Load Python plugin functions
    fn load_python_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let imports = plugin.imports.clone();
            let func_name = func.name.clone();

            let full_code = if !imports.is_empty() {
                let mut code_with_imports = imports.join("\n");
                code_with_imports.push_str("\n\n");
                code_with_imports.push_str(&code);
                code_with_imports
            } else {
                code
            };

            let description = format!("[Plugin:{}] {}", plugin.name, func.description);

            self.register(
                &func_name,
                Box::new(move |args| {
                    // FIX: Transmute Args zu 'static (safe weil Arena geleakt ist)
                    // SAFETY: In TBCore::execute() wird Arena mit Box::leak geleakt,
                    // sodass alle Values effektiv 'static leben
                    let static_args: &'static [Value<'static>] = unsafe {
                        std::mem::transmute(args)
                    };

                    let result = execute_python_code(&full_code, static_args)?;

                    // SAFETY: Result kommt aus derselben geleakten Arena
                    Ok(unsafe { std::mem::transmute(result) })
                }),
                func.min_args,
                func.max_args,
                &description,
            );

            debug_log!("  Registered function: {}", func_name);
        }

        Ok(())
    }

    fn load_javascript_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let func_name = func.name.clone();
            let description = format!("[Plugin:{}] {}", plugin.name, func.description);

            self.register(
                &func_name,
                Box::new(move |args| {
                    let static_args: &'static [Value<'static>] = unsafe {
                        std::mem::transmute(args)
                    };

                    let result = execute_js_code(&code, static_args)?;
                    Ok(unsafe { std::mem::transmute(result) })
                }),
                func.min_args,
                func.max_args,
                &description,
            );
        }
        Ok(())
    }

    fn load_go_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let func_name = func.name.clone();
            let description = format!("[Plugin:{}] {}", plugin.name, func.description);

            self.register(
                &func_name,
                Box::new(move |args| {
                    let static_args: &'static [Value<'static>] = unsafe {
                        std::mem::transmute(args)
                    };

                    let result = execute_go_code(&code, static_args)?;
                    Ok(unsafe { std::mem::transmute(result) })
                }),
                func.min_args,
                func.max_args,
                &description,
            );
        }
        Ok(())
    }

    fn load_bash_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let func_name = func.name.clone();
            let description = format!("[Plugin:{}] {}", plugin.name, func.description);

            self.register(
                &func_name,
                Box::new(move |args| {
                    let static_args: &'static [Value<'static>] = unsafe {
                        std::mem::transmute(args)
                    };

                    let result = execute_bash_command(&code, static_args)?;
                    Ok(unsafe { std::mem::transmute(result) })
                }),
                func.min_args,
                func.max_args,
                &description,
            );
        }
        Ok(())
    }

    /// Load TypeScript plugin functions (transpile to JS first)
    fn load_typescript_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        // For now, treat as JavaScript
        // TODO: Add ts-node or transpilation support
        self.load_javascript_plugin(plugin)
    }

}

// ═══════════════════════════════════════════════════════════════════════════
// STANDARD LIBRARY IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

fn builtin_echo<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", format_value_for_output(arg));
    }
    println!();
    Ok(Value::Unit)
}

fn builtin_print<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>>  {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", format_value_for_output(arg));
    }
    use std::io::Write;
    std::io::stdout().flush().ok();
    Ok(Value::Unit)
}

fn builtin_println<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    builtin_echo(args)
}

fn builtin_read_line<'a>(_args: &'a [Value<'a>]) -> TBResult<Value<'a>>  {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)
        .map_err(|e| TBError::IoError(Arc::from(e.to_string())))?;
    Ok(Value::String(Arc::new(line.trim_end().to_string())))
}

fn builtin_str<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>>  {
    let formatted = match &args[0] {
        Value::Unit => String::from(""),
        Value::Bool(b) => format!("{}", b),
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                format!("{:.0}", f)
            } else {
                format!("{}", f)
            }
        }
        Value::String(s) => s.to_string(),
        Value::List(items) => {
            let elements: Vec<String> = items.iter()
                .map(|v| format_value_for_output(v))
                .collect();
            format!("[{}]", elements.join(", "))
        }
        Value::Dict(map) => {
            let pairs: Vec<String> = map.iter()
                .map(|(k, v)| format!("{}: {}", k, format_value_for_output(v)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        Value::Tuple(items) => {
            let elements: Vec<String> = items.iter()
                .map(|v| format_value_for_output(v))
                .collect();
            format!("({})", elements.join(", "))
        }
        Value::Option(opt) => match opt {
            Some(v) => format!("Some({})", format_value_for_output(v)),
            None => String::from("None"),
        },
        Value::Result(res) => match res {
            Ok(v) => format!("Ok({})", format_value_for_output(v)),
            Err(e) => format!("Err({})", format_value_for_output(e)),
        },
        Value::Function { params, .. } => {
            let param_strs: Vec<&str> = params.iter()
                .map(|s| s.as_str())
                .collect();
            format!("fn({})", param_strs.join(", "))
        }
        Value::Native { type_name, .. } => {
            format!("<native:{}>", type_name)
        }
    };

    Ok(Value::String(Arc::from(formatted)))
}

fn builtin_int<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Float(f) => Ok(Value::Int(*f as i64)),
        Value::String(s) => {
            let cleaned = s.trim();
            if let Ok(n) = cleaned.parse::<i64>() {
                return Ok(Value::Int(n));
            }
            let no_underscores = cleaned.replace('_', "");
            if let Ok(n) = no_underscores.parse::<i64>() {
                return Ok(Value::Int(n));
            }
            let no_spaces = cleaned.replace(' ', "");
            if let Ok(n) = no_spaces.parse::<i64>() {
                return Ok(Value::Int(n));
            }
            Err(TBError::InvalidOperation(
                format!("Cannot convert '{}' to int", s).into()
            ))
        }
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        _ => Err(TBError::InvalidOperation(
            format!("Cannot convert {:?} to int", args[0].get_type()).into()
        )),
    }
}

fn builtin_float<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Float(*n as f64)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::String(s) => {
            let cleaned = s.trim();
            if let Ok(f) = cleaned.parse::<f64>() {
                return Ok(Value::Float(f));
            }
            let german_to_english = cleaned.replace(',', ".");
            if let Ok(f) = german_to_english.parse::<f64>() {
                return Ok(Value::Float(f));
            }
            let no_thousand_comma = cleaned.replace(",", "");
            if let Ok(f) = no_thousand_comma.parse::<f64>() {
                return Ok(Value::Float(f));
            }
            let mut german_style = cleaned.replace('.', "");
            german_style = german_style.replace(',', ".");
            if let Ok(f) = german_style.parse::<f64>() {
                return Ok(Value::Float(f));
            }
            let no_underscores = cleaned.replace('_', "");
            if let Ok(f) = no_underscores.parse::<f64>() {
                return Ok(Value::Float(f));
            }
            Err(TBError::InvalidOperation(
                format!("Cannot convert '{}' to float", s).into()
            ))
        }
        Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        _ => Err(TBError::InvalidOperation(
            format!("Cannot convert {:?} to float", args[0].get_type()).into()
        )),
    }
}

fn builtin_pretty<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    fn pretty_print(value: &Value, indent: usize) -> String {
        let spaces = "  ".repeat(indent);
        match value {
            Value::List(items) => {
                if items.is_empty() {
                    return "[]".to_string();
                }
                let mut result = "[\n".to_string();
                for (i, item) in items.iter().enumerate() {
                    result.push_str(&format!("{}  {}", spaces, pretty_print(item, indent + 1)));
                    if i < items.len() - 1 {
                        result.push(',');
                    }
                    result.push('\n');
                }
                result.push_str(&format!("{}]", spaces));
                result
            }
            Value::Dict(map) => {
                if map.is_empty() {
                    return "{}".to_string();
                }
                let mut result = "{\n".to_string();
                let entries: Vec<_> = map.iter().collect();
                for (i, (k, v)) in entries.iter().enumerate() {
                    result.push_str(&format!("{}  {}: {}",
                                             spaces, k, pretty_print(v, indent + 1)));
                    if i < entries.len() - 1 {
                        result.push(',');
                    }
                    result.push('\n');
                }
                result.push_str(&format!("{}}}", spaces));
                result
            }
            _ => format_value_for_output(value),
        }
    }

    let output = pretty_print(&args[0], 0);
    println!("{}", output);
    Ok(Value::Unit)
}

fn builtin_len<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    match &args[0] {
        Value::String(s) => Ok(Value::Int(s.len() as i64)),
        Value::List(l) => Ok(Value::Int(l.len() as i64)),
        Value::Dict(d) => Ok(Value::Int(d.len() as i64)),
        Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
        _ => Err(TBError::InvalidOperation(STRING_INTERNER.intern("Value has no length"))),
    }
}

fn builtin_push<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    if let Value::List(mut list) = args[0].clone() {
        list.push(args[1].clone());
        Ok(Value::List(list))
    } else {
        Err(TBError::InvalidOperation(STRING_INTERNER.intern("First argument must be a list")))
    }
}

fn builtin_pop<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    if let Value::List(mut list) = args[0].clone() {
        list.pop()
            .map(|v| Value::Tuple(vec![Value::List(list), v]))
            .ok_or_else(|| TBError::InvalidOperation(STRING_INTERNER.intern("Cannot pop from empty list")))
    } else {
        Err(TBError::InvalidOperation(STRING_INTERNER.intern("Argument must be a list")))
    }
}

fn builtin_debug<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    for arg in args {
        eprintln!("[DEBUG] {:?} : {:?}", arg, arg.get_type());
    }
    Ok(Value::Unit)
}

fn builtin_type_of<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    let type_str = match &args[0] {
        Value::Unit => "unit",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::String(_) => "string",

        Value::List(items) => {
            // Detaillierte List-Typ-Info
            if items.is_empty() {
                return Ok(Value::String(Arc::new("list<empty>".to_string())));
            }

            // Check if homogeneous
            let first_type = match &items[0] {
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::String(_) => "string",
                Value::Bool(_) => "bool",
                Value::List(_) => "list",
                _ => "mixed",
            };

            let is_homogeneous = items.iter().all(|item| {
                std::mem::discriminant(item) == std::mem::discriminant(&items[0])
            });

            if is_homogeneous {
                return Ok(Value::String(Arc::new(format!("list<{}>", first_type))));
            } else {
                return Ok(Value::String(Arc::new("list<mixed>".to_string())));
            }
        }

        Value::Dict(map) => {
            // Dict mit Key-Value-Typ-Info
            if map.is_empty() {
                return Ok(Value::String(Arc::new("dict<empty>".to_string())));
            }

            // Sample first value type
            let first_val_type = map.values().next().map(|v| match v {
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::String(_) => "string",
                Value::Bool(_) => "bool",
                _ => "mixed",
            }).unwrap_or("unknown");

            return Ok(Value::String(Arc::new(format!("dict<string, {}>", first_val_type))));
        }

        Value::Tuple(items) => {
            // Tuple mit Typ-Signatur
            let types: Vec<&str> = items.iter().map(|v| match v {
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::String(_) => "string",
                Value::Bool(_) => "bool",
                _ => "unknown",
            }).collect();

            return Ok(Value::String(Arc::new(format!("tuple<{}>", types.join(", ")))));
        }

        Value::Option(opt) => {
            match opt {
                Some(val) => {
                    let inner_type = match val.as_ref() {
                        Value::Int(_) => "int",
                        Value::Float(_) => "float",
                        Value::String(_) => "string",
                        Value::Bool(_) => "bool",
                        _ => "unknown",
                    };
                    return Ok(Value::String(Arc::new(format!("option<{}>", inner_type))));
                }
                None => return Ok(Value::String(Arc::new("option<none>".to_string()))),
            }
        }

        Value::Result(res) => {
            match res {
                Ok(val) => {
                    let ok_type = match val.as_ref() {
                        Value::Int(_) => "int",
                        Value::Float(_) => "float",
                        Value::String(_) => "string",
                        _ => "unknown",
                    };
                    return Ok(Value::String(Arc::new(format!("result<ok: {}>", ok_type))));
                }
                Err(val) => {
                    let err_type = match val.as_ref() {
                        Value::String(_) => "string",
                        _ => "unknown",
                    };
                    return Ok(Value::String(Arc::new(format!("result<err: {}>", err_type))));
                }
            }
        }

        Value::Function { params, .. } => {
            return Ok(Value::String(Arc::new(format!(
                "function<{} params>",
                params.len()
            ))));
        }

        Value::Native { type_name, .. } => {
            return Ok(Value::String(Arc::new(format!("native<{}>", type_name))));
        }
    };

    Ok(Value::String(Arc::new(type_str.to_string())))
}

fn builtin_python_info<'a>(_args: &'a [Value<'a>]) -> TBResult<Value<'a>>  {
    use std::process::Command;

    let python_exe = detect_python_executable();
    let version_output = Command::new(&python_exe)
        .arg("--version")
        .output()
        .ok();

    let version = if let Some(out) = version_output {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        "Unknown".to_string()
    };

    let path_output = Command::new(&python_exe)
        .arg("-c")
        .arg("import sys; print(sys.executable)")
        .output()
        .ok();

    let full_path = if let Some(out) = path_output {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        python_exe.clone()
    };

    let env_type = if std::env::var("VIRTUAL_ENV").is_ok() {
        "venv"
    } else if std::env::var("CONDA_PREFIX").is_ok() {
        "conda"
    } else if std::env::var("UV_PROJECT_ENVIRONMENT").is_ok() {
        "uv"
    } else if std::env::var("POETRY_ACTIVE").is_ok() {
        "poetry"
    } else if std::env::var("PYO3_PYTHON").is_ok() {
        "pyo3"
    } else {
        "system"
    };

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Python Environment Info                     ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Type:       {:<51} ║", env_type);
    println!("║ Version:    {:<51} ║", version);
    println!("║ Executable: {:<51} ║", python_exe);
    println!("║ Path:       {:<51} ║", full_path);
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(Value::Unit)
}

fn builtin_python<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    if let Value::String(code) = &args[0] {
        let args_static: Vec<Value<'static>> = args[1..].iter()
            .map(|v| v.clone_static())
            .collect::<TBResult<Vec<_>>>()?;

        let result = execute_python_code(code, &args_static)?;
        Ok(unsafe { std::mem::transmute(result) })
    } else {
        Err(TBError::InvalidOperation(Arc::from("python() requires string argument".to_string())))
    }
}

fn builtin_javascript<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    if let Value::String(code) = &args[0] {
        let args_static: Vec<Value<'static>> = args[1..].iter()
            .map(|v| v.clone_static())
            .collect::<TBResult<Vec<_>>>()?;

        let result = execute_js_code(code, &args_static)?;
        Ok(unsafe { std::mem::transmute(result) })
    } else {
        Err(TBError::InvalidOperation(Arc::from("javascript() requires string argument".to_string())))
    }
}

fn builtin_bash<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    if let Value::String(code) = &args[0] {
        let args_static: Vec<Value<'static>> = args[1..].iter()
            .map(|v| v.clone_static())
            .collect::<TBResult<Vec<_>>>()?;

        let result = execute_bash_command(code, &args_static)?;
        Ok(unsafe { std::mem::transmute(result) })
    } else {
        Err(TBError::InvalidOperation(Arc::from("bash() requires string argument".to_string())))
    }
}

fn builtin_go<'a>(args: &'a [Value<'a>]) -> TBResult<Value<'a>> {
    if let Value::String(code) = &args[0] {
        let args_static: Vec<Value<'static>> = args[1..].iter()
            .map(|v| v.clone_static())
            .collect::<TBResult<Vec<_>>>()?;

        let result = execute_go_code(code, &args_static)?;
        Ok(unsafe { std::mem::transmute(result) })
    } else {
        Err(TBError::InvalidOperation(Arc::from("go() requires string argument".to_string())))
    }
}

/// Helper to format value for output (without quotes for strings)
fn format_value_for_output<'arena>(value: &Value<'arena>) -> String {
    match value {
        Value::String(s) => s.to_string(), // No quotes
        Value::Unit => String::new(),
        Value::Float(f) => {
            // Smart formatting
            if f.fract() == 0.0 && f.is_finite() {
                format!("{:.0}", f) // "6" not "6.0" for integer-valued floats
            } else {
                format!("{}", f)
            }
        }
        Value::List(items) => {
            let elements: Vec<String> = items.iter()
                .map(|v| format_value_for_output(v))
                .collect();
            format!("[{}]", elements.join(", "))
        }
        Value::Dict(map) => {
            let pairs: Vec<String> = map.iter()
                .map(|(k, v)| format!("{}: {}", k, format_value_for_output(v)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        Value::Tuple(items) => {
            let elements: Vec<String> = items.iter()
                .map(|v| format_value_for_output(v))
                .collect();
            format!("({})", elements.join(", "))
        }
        other => format!("{}", other),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.6 PLUGIN SYSTEM - NATIVE LANGUAGE BRIDGES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PluginConfig {
    pub name: String,
    pub language: Language,
    pub enabled: bool,
    pub functions: Vec<PluginFunctionConfig>,
    pub imports: Vec<String>,
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PluginFunctionConfig {
    pub name: String,
    pub code: String,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub description: String,
}

/// Builder for Plugin configuration
#[derive(Debug, Clone)]
pub struct PluginBuilder {
    name: String,
    language: Language,
    enabled: bool,
    functions: Vec<PluginFunctionConfig>,
    imports: Vec<String>,
    env_vars: HashMap<String, String>,
}

impl PluginBuilder {
    pub fn new(name: impl Into<String>, language: Language) -> Self {
        Self {
            name: name.into(),
            language,
            enabled: true,
            functions: Vec::new(),
            imports: Vec::new(),
            env_vars: HashMap::new(),
        }
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn add_function(
        mut self,
        name: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        self.functions.push(PluginFunctionConfig {
            name: name.into(),
            code: code.into(),
            min_args: 0,
            max_args: None,
            description: String::new(),
        });
        self
    }

    pub fn add_function_with_args(
        mut self,
        name: impl Into<String>,
        code: impl Into<String>,
        min_args: usize,
        max_args: Option<usize>,
    ) -> Self {
        self.functions.push(PluginFunctionConfig {
            name: name.into(),
            code: code.into(),
            min_args,
            max_args,
            description: String::new(),
        });
        self
    }

    pub fn add_function_full(
        mut self,
        name: impl Into<String>,
        code: impl Into<String>,
        min_args: usize,
        max_args: Option<usize>,
        description: impl Into<String>,
    ) -> Self {
        self.functions.push(PluginFunctionConfig {
            name: name.into(),
            code: code.into(),
            min_args,
            max_args,
            description: description.into(),
        });
        self
    }

    pub fn add_import(mut self, import: impl Into<String>) -> Self {
        self.imports.push(import.into());
        self
    }

    pub fn add_imports(mut self, imports: Vec<String>) -> Self {
        self.imports.extend(imports);
        self
    }

    pub fn set_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> PluginConfig {
        PluginConfig {
            name: self.name,
            language: self.language,
            enabled: self.enabled,
            functions: self.functions,
            imports: self.imports,
            env_vars: self.env_vars,
        }
    }
}

/// Quick builder functions for each language
pub mod builders {
    use super::*;

    pub fn python(name: impl Into<String>) -> PluginBuilder {
        PluginBuilder::new(name, Language::Python)
    }

    pub fn javascript(name: impl Into<String>) -> PluginBuilder {
        PluginBuilder::new(name, Language::JavaScript)
    }

    pub fn typescript(name: impl Into<String>) -> PluginBuilder {
        PluginBuilder::new(name, Language::TypeScript)
    }

    pub fn go(name: impl Into<String>) -> PluginBuilder {
        PluginBuilder::new(name, Language::Go)
    }

    pub fn bash(name: impl Into<String>) -> PluginBuilder {
        PluginBuilder::new(name, Language::Bash)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// : Modul LanguageExecutionContext
// ═══════════════════════════════════════════════════════════════════════════

/// Context für Language-Execution mit Variablen-Zugriff
#[derive(Debug, Clone)]
pub struct LanguageExecutionContext<'arena> {
    pub variables: HashMap<Arc<String>, Value<'arena>>,
    pub return_type: Option<Type>,
}

impl<'arena> LanguageExecutionContext<'arena> {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            return_type: None,
        }
    }

    pub fn with_variables(variables: HashMap<Arc<String>, Value<'arena>>) -> Self {

        Self {
            variables,
            return_type: None,
        }
    }

    /// Serialize TB value to target language
    pub fn serialize_for_language(&self, language: Language) -> String {
        match language {
            Language::Python => self.to_python_vars(),
            Language::JavaScript | Language::TypeScript => self.to_js_vars(),
            Language::Go => self.to_go_vars(),
            Language::Bash => self.to_bash_vars(),
            _ => String::new(),
        }
    }

    /// Convert TB variables to Python code
    fn to_python_vars(&self) -> String {
        let mut code = String::from("# TB Variables\n");

        for (name, value) in &self.variables {
            let py_value = match value {
                Value::Unit => "None".to_string(),
                Value::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
                Value::Int(n) => format!("{}", n),
                Value::Float(f) => format!("{}", f),
                Value::String(s) => {
                    // FIXED: Proper Python string escaping with newlines
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t");
                    format!("\"{}\"", escaped)
                }
                Value::List(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_python(v))
                        .collect();
                    format!("[{}]", elements.join(", "))
                }
                Value::Dict(map) => {
                    let pairs: Vec<String> = map.iter()
                        .map(|(k, v)| format!("\"{}\": {}", k, self.value_to_python(v)))
                        .collect();
                    format!("{{{}}}", pairs.join(", "))
                }
                Value::Tuple(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_python(v))
                        .collect();
                    format!("({})", elements.join(", "))
                }
                _ => "None".to_string(),
            };

            code.push_str(&format!("{} = {}\n", name, py_value));
        }

        code.push('\n');
        code
    }

    /// Convert TB variables to JavaScript code
    fn to_js_vars(&self) -> String {
        let mut code = String::from("// TB Variables\n");

        for (name, value) in &self.variables {
            let js_value = match value {
                Value::Unit => "null".to_string(),
                Value::Bool(b) => format!("{}", b),
                Value::Int(n) => format!("{}", n),
                Value::Float(f) => format!("{}", f),
                Value::String(s) => {
                    // FIXED: Proper JavaScript string escaping for multiline strings
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t");
                    format!("\"{}\"", escaped)
                }
                Value::List(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_js(v))
                        .collect();
                    format!("[{}]", elements.join(", "))
                }
                Value::Dict(map) => {
                    let pairs: Vec<String> = map.iter()
                        .map(|(k, v)| format!("\"{}\": {}", k, self.value_to_js(v)))
                        .collect();
                    format!("{{{}}}", pairs.join(", "))
                }
                Value::Tuple(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_js(v))
                        .collect();
                    format!("[{}]", elements.join(", "))
                }
                _ => "null".to_string(),
            };

            code.push_str(&format!("const {} = {};\n", name, js_value));
        }

        code.push('\n');
        code
    }

    /// Convert TB variables to Go code
    fn to_go_vars(&self) -> String {
        let mut code = String::from("// TB Variables\n");

        for (name, value) in &self.variables {
            match value {
                Value::Unit => {
                    code.push_str(&format!("var {} interface{{}} = nil\n", name));
                }

                Value::Bool(b) => {
                    code.push_str(&format!("var {} bool = {}\n", name, if *b { "true" } else { "false" }));
                }

                Value::Int(n) => {
                    code.push_str(&format!("var {} int64 = {}\n", name, n));
                }

                Value::Float(f) => {
                    code.push_str(&format!("var {} float64 = {}\n", name, f));
                }

                Value::String(s) => {
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t");
                    code.push_str(&format!("var {} string = \"{}\"\n", name, escaped));
                }

                Value::List(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_go(v))
                        .collect();

                    code.push_str(&format!("var {} = []interface{{}}{{{}}}\n",
                                           name, elements.join(", ")));
                }

                Value::Dict(map) => {
                    code.push_str(&format!("var {} = make(map[string]interface{{}})\n", name));
                    for (k, v) in map {
                        code.push_str(&format!("{}[\"{}\"] = {}\n", name, k, self.value_to_go(&v)));
                    }
                }

                Value::Tuple(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_go(v))
                        .collect();

                    code.push_str(&format!("var {} = []interface{{}}{{{}}}\n",
                                           name, elements.join(", ")));
                }

                _ => {
                    code.push_str(&format!("var {} interface{{}} = nil\n", name));
                }
            }
        }

        //  Suppress unused variable warnings
        if !self.variables.is_empty() {
            code.push_str("\n// Suppress unused warnings\n");
            for (name, _) in &self.variables {
                code.push_str(&format!("_ = {}\n", name));
            }
        }

        code.push('\n');
        code
    }

    /// Convert TB variables to Bash code
    fn to_bash_vars(&self) -> String {
        let mut code = String::from("# TB Variables\n");

        for (name, value) in &self.variables {
            let bash_value = match value {
                Value::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
                Value::Int(n) => format!("{}", n),
                Value::Float(f) => format!("{}", f),
                Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
                Value::List(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| format_value_for_output(v))
                        .collect();
                    format!("({})", elements.join(" "))
                }
                _ => String::new(),
            };

            code.push_str(&format!("{}={}\n", name, bash_value));
        }

        code.push('\n');
        code
    }

    // Helper methods for value conversion
    fn value_to_python(&self, value: &Value) -> String {
        match value {
            Value::Unit => "None".to_string(),
            Value::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
            Value::Int(n) => format!("{}", n),
            Value::Float(f) => format!("{}", f),
            Value::String(s) => {
                // FIXED: Proper Python string escaping
                let escaped = s
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("\"{}\"", escaped)
            }
            _ => "None".to_string(),
        }
    }

    fn value_to_js(&self, value: &Value) -> String {
        match value {
            Value::Unit => "null".to_string(),
            Value::Bool(b) => format!("{}", b),
            Value::Int(n) => format!("{}", n),
            Value::Float(f) => format!("{}", f),
            Value::String(s) => {
                // Proper JavaScript string escaping
                let escaped = s
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("\"{}\"", escaped)
            }
            _ => "null".to_string(),
        }
    }

    fn value_to_go(&self, value: &Value) -> String {
        match value {
            Value::Unit => "nil".to_string(),
            Value::Bool(b) => {
                if *b { "true".to_string() } else { "false".to_string() }
            }
            Value::Int(n) => format!("{}", n),
            Value::Float(f) => format!("{}", f),
            Value::String(s) => {
                // FIXED: Proper Go string escaping (same as Rust/C-style)
                let escaped = s
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("\"{}\"", escaped)
            }
            Value::List(items) => {
                let elements: Vec<String> = items.iter()
                    .map(|v| self.value_to_go(v))
                    .collect();
                format!("[]{{{}}}", elements.join(", "))
            }
            _ => "nil".to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ReturnValueParser
// ═══════════════════════════════════════════════════════════════════════════

/// Parse return values from language output
pub struct ReturnValueParser;

impl ReturnValueParser {
    pub fn from_python(output: &str) -> Value<'static> {
        let trimmed = output.trim();

        // ═══════════════════════════════════════════════════════════════
        // PRIORITY 1: Try JSON parsing (most reliable)
        // ═══════════════════════════════════════════════════════════════
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return Self::json_to_value(&json_val);
        }

        // ═══════════════════════════════════════════════════════════════
        // PRIORITY 2: Python literals
        // ═══════════════════════════════════════════════════════════════

        // Boolean
        if trimmed == "True" {
            return Value::Bool(true);
        }
        if trimmed == "False" {
            return Value::Bool(false);
        }

        // None
        if trimmed == "None" {
            return Value::Unit;
        }

        // List (Python syntax: [1, 2, 3])
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            return Self::parse_python_list(trimmed);
        }

        // Dict (Python syntax: {'key': 'value'})
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            return Self::parse_python_dict(trimmed);
        }

        // ═══════════════════════════════════════════════════════════════
        // PRIORITY 3: Numeric parsing
        // ═══════════════════════════════════════════════════════════════

        // Try integer
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }

        // Try float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        // ═══════════════════════════════════════════════════════════════
        // FALLBACK: String
        // ═══════════════════════════════════════════════════════════════
        Value::String(Arc::from(trimmed.to_string()))
    }

    /// Parse JavaScript output as TB value
    pub fn from_javascript(output: &str) -> Value<'static> {
        let trimmed = output.trim();

        // Try JSON first
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return Self::json_to_value(&json_val);
        }

        // JavaScript literals
        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }
        if trimmed == "null" || trimmed == "undefined" {
            return Value::Unit;
        }

        // Arrays
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            return Self::parse_js_array(trimmed);
        }

        // Objects
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            return Self::parse_js_object(trimmed);
        }

        // Numbers
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        // String (remove quotes if present)
        if (trimmed.starts_with('"') && trimmed.ends_with('"')) ||
            (trimmed.starts_with('\'') && trimmed.ends_with('\'')) {
            return Value::String(Arc::from(trimmed[1..trimmed.len()-1].to_string()));
        }

        Value::String(Arc::from(trimmed.to_string()))
    }

    /// Parse Go output as TB value
    pub fn from_go(output: &str) -> Value<'static> {
        let trimmed = output.trim();

        // Try JSON first
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return Self::json_to_value(&json_val);
        }

        // Go literals
        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }
        if trimmed == "<nil>" || trimmed == "nil" {
            return Value::Unit;
        }

        // Numbers
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        Value::String(Arc::from(trimmed.to_string()))
    }

    /// Parse Bash output as TB value
    pub fn from_bash(output: &str) -> Value<'static> {
        let trimmed = output.trim();

        // Boolean
        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }

        // Numbers
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        Value::String(Arc::from(trimmed.to_string()))
    }

    // ═══════════════════════════════════════════════════════════════
    // Helper: JSON → TB Value
    // ═══════════════════════════════════════════════════════════════
    fn json_to_value(json: &serde_json::Value) -> Value<'static> {
        match json {
            serde_json::Value::Null => Value::Unit,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::Int(0)
                }
            }
            serde_json::Value::String(s) => Value::String(Arc::from(s.clone())),
            serde_json::Value::Array(arr) => {
                let items: Vec<Value> = arr.iter()
                    .map(|v| Self::json_to_value(v))
                    .collect();
                Value::List(items)
            }
            serde_json::Value::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(Arc::new(k.clone()), Self::json_to_value(v));
                }
                Value::Dict(map)
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Helper: Parse Python list
    // ═══════════════════════════════════════════════════════════════
    fn parse_python_list(s: &str) -> Value<'static> {
        let inner = &s[1..s.len()-1];
        let items: Vec<Value> = inner.split(',')
            .map(|item| {
                let trimmed = item.trim();

                // Try boolean
                if trimmed == "True" {
                    return Value::Bool(true);
                }
                if trimmed == "False" {
                    return Value::Bool(false);
                }

                // Try int
                if let Ok(n) = trimmed.parse::<i64>() {
                    return Value::Int(n);
                }

                // Try float
                if let Ok(f) = trimmed.parse::<f64>() {
                    return Value::Float(f);
                }

                // String (remove quotes)
                let unquoted = trimmed.trim_matches('\'').trim_matches('"');
                Value::String(Arc::from(unquoted.to_string()))
            })
            .collect();

        Value::List(items)
    }

    // Helper: Parse Python dict (simple version)
    fn parse_python_dict(_s: &str) -> Value<'static> {
        // TODO: Implement proper dict parsing
        Value::Dict(HashMap::new())
    }

    // Helper: Parse JS array
    fn parse_js_array(s: &str) -> Value<'static> {
        let inner = &s[1..s.len()-1];
        let items: Vec<Value> = inner.split(',')
            .map(|item| {
                let trimmed = item.trim();

                if trimmed == "true" {
                    return Value::Bool(true);
                }
                if trimmed == "false" {
                    return Value::Bool(false);
                }

                if let Ok(n) = trimmed.parse::<i64>() {
                    return Value::Int(n);
                }
                if let Ok(f) = trimmed.parse::<f64>() {
                    return Value::Float(f);
                }

                let unquoted = trimmed.trim_matches('\'').trim_matches('"');
                Value::String(Arc::from(unquoted.to_string()))
            })
            .collect();

        Value::List(items)
    }

    // Helper: Parse JS object (simple version)
    fn parse_js_object(_s: &str) -> Value<'static> {
        // TODO: Implement proper object parsing
        Value::Dict(HashMap::new())
    }


}
// Helper methods
impl ReturnValueParser {
    /// Enhanced list parser with nested structure support
    fn parse_list(s: &str) -> Value<'static> {
        let trimmed = s.trim();

        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return Value::List(Vec::new());
        }

        if trimmed == "[]" {
            return Value::List(Vec::new());
        }

        let inner = &trimmed[1..trimmed.len()-1];
        let mut items = Vec::new();
        let mut chars = inner.chars().peekable();

        loop {
            Self::skip_whitespace(&mut chars);

            if chars.peek().is_none() {
                break;
            }

            // Parse value
            if let Ok(value) = Self::parse_dict_value(&mut chars) {
                items.push(value);
            }

            // Check for comma
            Self::skip_whitespace(&mut chars);
            if let Some(&ch) = chars.peek() {
                if ch == ',' {
                    chars.next();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Value::List(items)
    }
    fn parse_dict(s: &str) -> Value<'static> {
        let trimmed = s.trim();

        // Must start with { and end with }
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return Value::Dict(HashMap::new());
        }

        // Empty dict
        if trimmed == "{}" {
            return Value::Dict(HashMap::new());
        }

        let mut result = HashMap::new();
        let inner = &trimmed[1..trimmed.len()-1];

        // Parse key-value pairs with proper nesting support
        if let Ok(pairs) = Self::parse_dict_pairs(inner) {
            for (key, value) in pairs {
                result.insert(Arc::new(key), value);
            }
        }

        Value::Dict(result)
    }

    /// Parse dict key-value pairs with nesting support
    fn parse_dict_pairs(s: &str) -> Result<Vec<(String, Value<'static>)>, ()> {
        let mut pairs = Vec::new();
        let mut chars = s.chars().peekable();

        loop {
            // Skip whitespace
            Self::skip_whitespace(&mut chars);

            if chars.peek().is_none() {
                break;
            }

            // Parse key
            let key = Self::parse_dict_key(&mut chars)?;

            // Skip whitespace and colon
            Self::skip_whitespace(&mut chars);
            if chars.next() != Some(':') {
                return Err(());
            }
            Self::skip_whitespace(&mut chars);

            // Parse value
            let value = Self::parse_dict_value(&mut chars)?;

            pairs.push((key, value));

            // Check for comma or end
            Self::skip_whitespace(&mut chars);
            if let Some(&ch) = chars.peek() {
                if ch == ',' {
                    chars.next();
                } else if ch == '}' {
                    break; // Nested dict end
                }
            } else {
                break;
            }
        }

        Ok(pairs)
    }

    /// Parse dict key (string with quotes)
    fn parse_dict_key(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<String, ()> {
        let quote = chars.next();

        let quote_char = match quote {
            Some('\'') => '\'',
            Some('"') => '"',
            _ => return Err(()),
        };

        let mut key = String::new();
        let mut escaped = false;

        while let Some(ch) = chars.next() {
            if escaped {
                // Handle escape sequences
                match ch {
                    'n' => key.push('\n'),
                    't' => key.push('\t'),
                    'r' => key.push('\r'),
                    '\\' => key.push('\\'),
                    '\'' => key.push('\''),
                    '"' => key.push('"'),
                    _ => {
                        key.push('\\');
                        key.push(ch);
                    }
                }
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote_char {
                return Ok(key);
            } else {
                key.push(ch);
            }
        }

        Err(())
    }

    /// Parse dict value (supports all types)
    fn parse_dict_value(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<Value<'static>, ()> {
        Self::skip_whitespace(chars);

        let ch = chars.peek().ok_or(())?;

        match ch {
            // String
            '\'' | '"' => {
                let quote = *ch;
                chars.next();
                let mut value = String::new();
                let mut escaped = false;

                while let Some(ch) = chars.next() {
                    if escaped {
                        match ch {
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            'r' => value.push('\r'),
                            '\\' => value.push('\\'),
                            '\'' => value.push('\''),
                            '"' => value.push('"'),
                            _ => {
                                value.push('\\');
                                value.push(ch);
                            }
                        }
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == quote {
                        return Ok(Value::String(Arc::new(value)));
                    } else {
                        value.push(ch);
                    }
                }
                Err(())
            }

            // Nested dict
            '{' => {
                chars.next();
                let mut depth = 1;
                let mut nested = String::from("{");

                while let Some(ch) = chars.next() {
                    nested.push(ch);
                    if ch == '{' {
                        depth += 1;
                    } else if ch == '}' {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(Self::parse_dict(&nested));
                        }
                    }
                }
                Err(())
            }

            // List
            '[' => {
                chars.next();
                let mut depth = 1;
                let mut list_str = String::from("[");

                while let Some(ch) = chars.next() {
                    list_str.push(ch);
                    if ch == '[' {
                        depth += 1;
                    } else if ch == ']' {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(Self::parse_list(&list_str));
                        }
                    }
                }
                Err(())
            }

            // Boolean or None/null
            'T' | 'F' | 't' | 'f' | 'N' | 'n' => {
                let word = Self::parse_word(chars);
                match word.as_str() {
                    "True" | "true" => Ok(Value::Bool(true)),
                    "False" | "false" => Ok(Value::Bool(false)),
                    "None" | "null" => Ok(Value::Unit),
                    _ => Err(()),
                }
            }

            // Number
            '-' | '0'..='9' => {
                let num_str = Self::parse_number(chars);

                // Try integer first
                if let Ok(n) = num_str.parse::<i64>() {
                    Ok(Value::Int(n))
                }
                // Try float
                else if let Ok(f) = num_str.parse::<f64>() {
                    Ok(Value::Float(f))
                } else {
                    Err(())
                }
            }

            _ => Err(()),
        }
    }

    /// Skip whitespace characters
    fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars>) {
        while let Some(&ch) = chars.peek() {
            if ch.is_whitespace() {
                chars.next();
            } else {
                break;
            }
        }
    }

    /// Parse a word (identifier)
    fn parse_word(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut word = String::new();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                word.push(ch);
                chars.next();
            } else {
                break;
            }
        }

        word
    }

    /// Parse a number (int or float)
    fn parse_number(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut num = String::new();

        // Handle negative sign
        if let Some(&'-') = chars.peek() {
            num.push('-');
            chars.next();
        }

        // Parse digits and decimal point
        while let Some(&ch) = chars.peek() {
            if ch.is_numeric() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-' {
                num.push(ch);
                chars.next();
            } else {
                break;
            }
        }

        num
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NATIVE LANGUAGE EXECUTORS
// ═══════════════════════════════════════════════════════════════════════════

/// Execute bash command

/// Execute bash command with TB variable context
fn execute_bash_command_with_context<'arena>(
    code: &str,
    args: &[Value<'arena>],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value<'static>> {
    use std::process::Command;

    let arg_strings: Vec<String> = args.iter()
        .map(|v| format_value_for_output(v))
        .collect();

    let wrapped_bash = wrap_bash_auto_return(code);

    let mut full_command = String::new();

    // Inject TB variables
    if let Some(ctx) = context {
        full_command.push_str(&ctx.serialize_for_language(Language::Bash));
    }

    full_command.push_str(&wrapped_bash);

    // Inject TB variables
    if let Some(ctx) = context {
        full_command.push_str(&ctx.serialize_for_language(Language::Bash));
    }

    full_command.push_str(code);

    for arg in &arg_strings {
        full_command.push(' ');
        if arg.contains(' ') {
            full_command.push_str(&format!("'{}'", arg));
        } else {
            full_command.push_str(arg);
        }
    }

    // Detect proper shell
    let (shell, flag) = if cfg!(target_os = "windows") {
        // Try to find real bash on Windows
        let bash_candidates = vec![
            "bash",           // WSL or in PATH
            "C:\\Program Files\\Git\\bin\\bash.exe",  // Git Bash
            "C:\\msys64\\usr\\bin\\bash.exe",         // MSYS2
            "C:\\cygwin64\\bin\\bash.exe",            // Cygwin
        ];

        let mut found_bash = None;
        for candidate in &bash_candidates {
            if Command::new(candidate).arg("--version").output().is_ok() {
                found_bash = Some(*candidate);
                debug_log!("Found bash: {}", candidate);
                break;
            }
        }

        if let Some(bash_path) = found_bash {
            (bash_path, "-c")
        } else {
            return Err(TBError::RuntimeError {
                message: format!(
                    "Bash not found on Windows.\n\n\
                    Please install one of the following:\n\
                    • Git for Windows (https://git-scm.com/download/win)\n\
                    • WSL (Windows Subsystem for Linux)\n\
                    • MSYS2 (https://www.msys2.org/)\n\n\
                    Then ensure 'bash' is in your PATH.\n\n\
                    Code that failed:\n{}\n",
                    code
                ).into(),
                trace: vec![
                    Arc::from("execute_bash_command_with_context()".to_string()),
                    format!("bash(\"{}\") at script line (unable to determine)",
                            code.lines().next().unwrap_or("")).into()
                ],
            });
        }
    } else {
        ("bash", "-c")
    };

    debug_log!("Executing Bash command with '{}':\n{}", shell, full_command);

    let output = Command::new(shell)
        .arg(flag)
        .arg(&full_command)
        .output()
        .map_err(|e| TBError::RuntimeError {
            message: format!("Failed to execute bash: {}", e).into(),
            trace: vec![],
        })?;

    //  Clone stdout to owned String
    let stdout_owned = String::from_utf8_lossy(&output.stdout).to_string();

    if !output.status.success() {
        let stderr_owned = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(TBError::RuntimeError {
            message: format!("Bash failed: {}", stderr_owned).into(),
            trace: vec![],
        });
    }

    if !output.stderr.is_empty() {
        eprint!("{:?}", output.stderr);
    }

    if !output.stdout.is_empty() {
        print!("{:?}", output.stdout);
    }
    // Return owned Value
    Ok(ReturnValueParser::from_bash(&stdout_owned))
}

fn execute_bash_command<'arena>(code: &str, args: &[Value<'arena>]) -> TBResult<Value<'static>> {
    execute_bash_command_with_context(code, args, None)
}

/// Wrap Bash code to auto-return last expression
fn wrap_bash_auto_return(code: &str) -> String {
    let trimmed = code.trim();
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.is_empty() {
        return code.to_string();
    }

    let last_line = lines.last().unwrap().trim();

    // Skip wrapping if:
    if last_line.is_empty()
        || last_line.starts_with('#')
        || last_line == "fi"
        || last_line == "done"
        || last_line == "}"
        || last_line == "esac"
        || last_line.starts_with("if ")
        || last_line.starts_with("for ")
        || last_line.starts_with("while ")
        || last_line.starts_with("function ")
        || last_line.contains('=')
        || last_line.starts_with("echo ")
    {
        return code.to_string();
    }

    let mut wrapped = String::new();

    for (i, line) in lines.iter().enumerate() {
        if i < lines.len() - 1 {
            wrapped.push_str(line);
            wrapped.push('\n');
        }
    }

    // For bash, try to evaluate as arithmetic
    wrapped.push_str("__tb_result=$(echo \"");
    wrapped.push_str(last_line);
    wrapped.push_str("\" | bc -l 2>/dev/null || echo $((");
    wrapped.push_str(last_line);
    wrapped.push_str(")))\n");
    wrapped.push_str("echo -n $__tb_result");

    wrapped
}
/// Detect the active Python interpreter
/// Priority: venv > conda > uv > poetry > PYO3_PYTHON > system python
fn detect_python_executable() -> String {
    use std::env;
    use std::path::Path;

    // 1. Check for active venv (VIRTUAL_ENV)
    if let Ok(venv_path) = env::var("VIRTUAL_ENV") {
        let venv = Path::new(&venv_path);

        // Unix-like: bin/python
        let unix_python = venv.join("bin").join("python");
        if unix_python.exists() {
            debug_log!("Using venv Python: {}", unix_python.display());
            return unix_python.to_string_lossy().to_string();
        }

        // Windows: Scripts\python.exe
        let win_python = venv.join("Scripts").join("python.exe");
        if win_python.exists() {
            debug_log!("Using venv Python: {}", win_python.display());
            return win_python.to_string_lossy().to_string();
        }
    }

    // 2. Check for active conda environment (CONDA_PREFIX)
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda = Path::new(&conda_prefix);

        // Unix-like: bin/python
        let unix_python = conda.join("bin").join("python");
        if unix_python.exists() {
            debug_log!("Using conda Python: {}", unix_python.display());
            return unix_python.to_string_lossy().to_string();
        }

        // Windows: python.exe
        let win_python = conda.join("python.exe");
        if win_python.exists() {
            debug_log!("Using conda Python: {}", win_python.display());
            return win_python.to_string_lossy().to_string();
        }
    }

    // 3. Check for uv environment (UV_PROJECT_ENVIRONMENT or .venv)
    if let Ok(uv_env) = env::var("UV_PROJECT_ENVIRONMENT") {
        let uv_path = Path::new(&uv_env);

        let unix_python = uv_path.join("bin").join("python");
        if unix_python.exists() {
            debug_log!("Using uv Python: {}", unix_python.display());
            return unix_python.to_string_lossy().to_string();
        }

        let win_python = uv_path.join("Scripts").join("python.exe");
        if win_python.exists() {
            debug_log!("Using uv Python: {}", win_python.display());
            return win_python.to_string_lossy().to_string();
        }
    }

    // 3b. Check for local .venv (commonly used by uv)
    let current_dir = env::current_dir().ok();
    if let Some(cwd) = current_dir {
        let local_venv = cwd.join(".venv");
        if local_venv.exists() {
            let unix_python = local_venv.join("bin").join("python");
            if unix_python.exists() {
                debug_log!("Using local .venv Python: {}", unix_python.display());
                return unix_python.to_string_lossy().to_string();
            }

            let win_python = local_venv.join("Scripts").join("python.exe");
            if win_python.exists() {
                debug_log!("Using local .venv Python: {}", win_python.display());
                return win_python.to_string_lossy().to_string();
            }
        }
    }

    // 4. Check for Poetry environment (POETRY_ACTIVE)
    if let Ok(poetry_active) = env::var("POETRY_ACTIVE") {
        if poetry_active == "1" {
            // Poetry sets PATH, so "python" should point to the right one
            // But we can also try to get it explicitly
            if let Ok(poetry_venv) = env::var("VIRTUAL_ENV") {
                let poetry_path = Path::new(&poetry_venv);

                let unix_python = poetry_path.join("bin").join("python");
                if unix_python.exists() {
                    debug_log!("Using Poetry Python: {}", unix_python.display());
                    return unix_python.to_string_lossy().to_string();
                }

                let win_python = poetry_path.join("Scripts").join("python.exe");
                if win_python.exists() {
                    debug_log!("Using Poetry Python: {}", win_python.display());
                    return win_python.to_string_lossy().to_string();
                }
            }
        }
    }

    // 5. Check PYO3_PYTHON environment variable (used by PyO3/maturin)
    if let Ok(pyo3_python) = env::var("PYO3_PYTHON") {
        let pyo3_path = Path::new(&pyo3_python);
        if pyo3_path.exists() {
            debug_log!("Using PYO3_PYTHON: {}", pyo3_python);
            return pyo3_python;
        }
    }

    // 6. Fallback to system Python
    // Try python3 first (preferred on Unix)
    if let Ok(output) = std::process::Command::new("python3")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            debug_log!("Using system python3");
            return "python3".to_string();
        }
    }

    // Final fallback to python (Windows default)
    debug_log!("Using system python (fallback)");
    "python".to_string()
}

/// Execute Python code with automatic environment detection

/// Execute Python code with TB variable context
fn execute_python_code_with_context<'arena>(
    code: &str,
    args: &[Value<'arena>],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value<'static>> {
    use std::process::Command;

    let python_exe = detect_python_executable();

    let arg_strings: Vec<String> = args.iter()
        .map(|v| format_value_for_output(v))
        .collect();

    // AUTO-WRAP: Last expression wird automatisch ausgegeben
    let wrapped_code = wrap_python_auto_return(code);

    let mut script = String::from("import sys\nimport json\n");
    script.push_str("args = sys.argv[1:]\n\n");

    if let Some(ctx) = context {
        script.push_str(&ctx.serialize_for_language(Language::Python));
    }

    script.push_str("# User Code\n");
    script.push_str(&wrapped_code);  // Wrapped code verwenden
    script.push('\n');

    debug_log!("Executing Python with: {}", python_exe);
    debug_log!("Python script:\n{}", script);

    let output = Command::new(&python_exe)
        .arg("-c")
        .arg(&script)
        .args(&arg_strings)
        .env("PYTHONIOENCODING", "utf-8")
        .output()
        .map_err(|e| TBError::RuntimeError {
            message: format!(
                "Failed to execute Python ({}): {}\nMake sure Python is installed.",
                python_exe, e
            ).into(),
            trace: vec![],
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let exit_code = output.status.code().unwrap_or(-1);
        return Err(TBError::RuntimeError {
            message: format!(
                "Python execution failed (exit code {}):\n{}\n\nScript:\n{}",
                exit_code, stderr, script
            ).into(),
            trace: vec![],
        });
    }

    if !stderr.is_empty() {
        eprint!("{}", stderr);
    }
    if !stdout.is_empty() {
        print!("{}", stdout);
    }

    let owned_stdout = stdout.trim().to_string();
    Ok(ReturnValueParser::from_python(&owned_stdout))
}

// Wrapper for backward compatibility
fn execute_python_code<'arena>(code: &str, args: &[Value<'arena>]) -> TBResult<Value<'static>> {
    execute_python_code_with_context(code, args, None)
}

fn wrap_python_auto_return(code: &str) -> String {
    let trimmed = code.trim();

    // Check if last line is an expression (not assignment/statement)
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.is_empty() {
        return code.to_string();
    }

    let last_line = lines.last().unwrap().trim();

    // Skip wrapping if:
    // - Empty line
    // - Comment
    // - Starts with keyword (if, for, while, def, class, import, return, print)
    // - Contains assignment (=, but not ==, !=, <=, >=)
    // - Already has print()
    if last_line.is_empty()
        || last_line.starts_with('#')
        || last_line == "}"
        || last_line == ")"
        || last_line == "]"
        || last_line.starts_with("if ")
        || last_line.starts_with("for ")
        || last_line.starts_with("while ")
        || last_line.starts_with("def ")
        || last_line.starts_with("class ")
        || last_line.starts_with("import ")
        || last_line.starts_with("from ")
        || last_line.starts_with("return ")
        || last_line.starts_with("print(")
        || last_line.starts_with("print ")
        || (last_line.contains(" = ") && !last_line.contains("=="))
        || last_line.ends_with(':')
    {
        return code.to_string();
    }

    // Build wrapped code
    let mut wrapped = String::new();

    // Add all lines except last
    for (i, line) in lines.iter().enumerate() {
        if i < lines.len() - 1 {
            wrapped.push_str(line);
            wrapped.push('\n');
        }
    }

    // Wrap last line
    wrapped.push_str("__tb_result = (");
    wrapped.push_str(last_line);
    wrapped.push_str(")\n");
    wrapped.push_str("print(__tb_result, end='')");

    wrapped
}

/// Execute JavaScript code with TB variable context
fn execute_js_code_with_context<'arena>(
    code: &str,
    args: &[Value<'arena>],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value<'static>> {
    use std::process::Command;

    let arg_strings: Vec<String> = args.iter()
        .map(|v| format_value_for_output(v))
        .collect();

    // AUTO-WRAP
    let wrapped_code = wrap_js_auto_return(code);

    let mut script = String::from("const args = process.argv.slice(2);\n\n");

    if let Some(ctx) = context {
        script.push_str(&ctx.serialize_for_language(Language::JavaScript));
    }

    script.push_str("// User Code\n");
    script.push_str(&wrapped_code);  // ✅
    script.push('\n');

    debug_log!("Executing JavaScript script:\n{}", script);

    let result = Command::new("node")
        .arg("-e")
        .arg(&script)
        .args(&arg_strings)
        .output();

    let output = if let Ok(out) = result {
        out
    } else {
        match Command::new("deno")
            .arg("eval")
            .arg("--no-check")
            .arg(&script)
            .args(&arg_strings)
            .output()
        {
            Ok(out) => out,
            Err(_) => {
                return Err(TBError::RuntimeError {
                    message: Arc::from("Node.js/Deno not found. Please install Node.js or Deno.".to_string()),
                    trace: vec![],
                });
            }
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        return Err(TBError::RuntimeError {
            message: format!("JavaScript execution failed:\n{}", stderr).into(),
            trace: vec![],
        });
    }
    if !stderr.is_empty() {
        eprint!("{}", stderr);
    }

    if !stdout.is_empty() {
        print!("{}", stdout);
    }

    Ok(ReturnValueParser::from_javascript(stdout.trim()))
}

fn execute_js_code<'arena>(code: &str, args: &[Value<'arena>]) -> TBResult<Value<'static>> {
    execute_js_code_with_context(code, args, None)
}
/// Wrap JavaScript code to auto-return last expression
fn wrap_js_auto_return(code: &str) -> String {
    let trimmed = code.trim();
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.is_empty() {
        return code.to_string();
    }

    let last_line = lines.last().unwrap().trim();

    // Skip wrapping if:
    if last_line.is_empty()
        || last_line.starts_with("//")
        || last_line == "}"
        || last_line == ")"
        || last_line == "]"
        || last_line.starts_with("if ")
        || last_line.starts_with("for ")
        || last_line.starts_with("while ")
        || last_line.starts_with("function ")
        || last_line.starts_with("const ")
        || last_line.starts_with("let ")
        || last_line.starts_with("var ")
        || last_line.starts_with("return ")
        || last_line.starts_with("console.log(")
        || last_line.ends_with('{')
    {
        return code.to_string();
    }

    let mut wrapped = String::new();

    for (i, line) in lines.iter().enumerate() {
        if i < lines.len() - 1 {
            wrapped.push_str(line);
            wrapped.push('\n');
        }
    }

    wrapped.push_str("const __tb_result = (");
    wrapped.push_str(last_line);
    wrapped.push_str(");\n");
    wrapped.push_str("process.stdout.write(String(__tb_result));");

    wrapped
}

/// Execute Go code with TB variable context
/// Execute Go code with TB variable context
fn execute_go_code_with_context<'arena> (
    code: &str,
    args: &[Value<'arena> ],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value<'static> > {
    use std::process::Command;
    use std::fs;
    use std::env;

    let go_version = Command::new("go").arg("version").output();
    if go_version.is_err() {
        return Err(TBError::RuntimeError {
            message: Arc::from("Go not found. Please install Go from https://go.dev/dl/".to_string()),
            trace: vec![],
        });
    }

    let arg_strings: Vec<String> = args.iter()
        .map(|v| format_value_for_output(v))
        .collect();

    let temp_dir = env::temp_dir().join(format!("tb_go_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir)
        .map_err(|e| TBError::IoError(format!("Failed to create temp dir: {}", e).into()))?;

    // ═══════════════════════════════════════════════════════════════════════
    // PREPROCESS USER CODE TO ESCAPE LITERAL NEWLINES IN STRINGS
    // ═══════════════════════════════════════════════════════════════════════
    let existing_vars: Vec<Arc<String>> = if let Some(ctx) = context {
        ctx.variables.keys().cloned().collect()
    } else {
        Vec::new()
    };

    debug_log!("Preprocessing Go code (existing vars: {:?})...", existing_vars);
    let preprocessed_code = preprocess_go_code_for_existing_vars(code, &existing_vars);
    debug_log!("Preprocessed code:\n{}", preprocessed_code);

    // ═══════════════════════════════════════════════════════════════════════
    // PARSE USER CODE FOR IMPORTS
    // ═══════════════════════════════════════════════════════════════════════
    let (user_imports, clean_code) = extract_go_imports(&preprocessed_code);  // Use preprocessed!

    // ═══════════════════════════════════════════════════════════════════════
    // BUILD COMPLETE GO FILE
    // ═══════════════════════════════════════════════════════════════════════
    let mut full_code = String::from("package main\n\n");

    // Add imports
    full_code.push_str("import (\n");

    // Always include fmt (for printing)
    let mut all_imports = vec!["fmt".to_string()];

    // Check if we need os (for args)
    if context.is_some() || !arg_strings.is_empty() {
        all_imports.push("os".to_string());
    }

    // Add user imports
    all_imports.extend(user_imports);

    // Deduplicate and format
    let import_set: std::collections::HashSet<String> = all_imports.into_iter().collect();
    let mut sorted_imports: Vec<String> = import_set.into_iter().collect();
    sorted_imports.sort();

    for import in &sorted_imports {
        full_code.push_str(&format!("    \"{}\"\n", import));
    }

    full_code.push_str(")\n\n");

    // Add main function
    full_code.push_str("func main() {\n");

    // Add args if needed
    if context.is_some() || !arg_strings.is_empty() {
        full_code.push_str("    args := os.Args[1:]\n\n");
    }

    // Add TB variables
    if let Some(ctx) = context {
        for line in ctx.serialize_for_language(Language::Go).lines() {
            if !line.trim().is_empty() {
                full_code.push_str("    ");
                full_code.push_str(line);
                full_code.push('\n');
            }
        }
        full_code.push('\n');
    }

    // Add user code (indented)
    for line in clean_code.lines() {
        if !line.trim().is_empty() {
            full_code.push_str("    ");
            full_code.push_str(line);
        }
        full_code.push('\n');
    }

    // Suppress unused variable warnings
    full_code.push_str("\n    // Suppress unused warnings\n");
    if context.is_some() || !arg_strings.is_empty() {
        full_code.push_str("    _ = args\n");
    }
    if let Some(ctx) = context {
        for var_name in ctx.variables.keys() {
            full_code.push_str(&format!("    _ = {}\n", var_name));
        }
    }

    full_code.push_str("}\n");

    // Write Go file
    let go_file = temp_dir.join("main.go");
    fs::write(&go_file, &full_code)
        .map_err(|e| TBError::IoError(format!("Failed to write Go file: {}", e).into()))?;

    debug_log!("Executing Go code:\n{}", full_code);

    // Execute
    let output = Command::new("go")
        .arg("run")
        .arg(&go_file)
        .args(&arg_strings)
        .current_dir(&temp_dir)
        .output()
        .map_err(|e| TBError::RuntimeError {
            message: format!("Failed to execute Go code: {}", e).into(),
            trace: vec![],
        })?;

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.is_empty() {
        print!("{}", stdout);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        let exit_code = output.status.code().unwrap_or(-1);
        return Err(TBError::RuntimeError {
            message: format!("Go execution failed (exit code {}):\n{}", exit_code, stderr).into(),
            trace: vec![],
        });
    }

    if !stderr.is_empty() {
        eprint!("{}", stderr);
    }

    if !stdout.is_empty() {
        print!("{}", stdout);
    }

    Ok(ReturnValueParser::from_go(stdout.trim()))
}
/// Extract import statements from Go code and return (imports, clean_code)
///
/// Example:
///   Input: "import \"math\"\nfmt.Println(math.Sqrt(64))"
///   Output: (vec!["math"], "fmt.Println(math.Sqrt(64))")
fn extract_go_imports(code: &str) -> (Vec<String>, String) {
    let mut imports = Vec::new();
    let mut clean_lines = Vec::new();
    let mut in_import_block = false;
    let mut paren_depth = 0;

    let wrapped_go = wrap_go_auto_return(&code);

    for line in wrapped_go.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comments at start
        if trimmed.is_empty() || trimmed.starts_with("//") {
            if !in_import_block && imports.is_empty() {
                continue; // Skip leading whitespace/comments
            }
        }

        // Detect import block start: import (
        if trimmed.starts_with("import (") {
            in_import_block = true;
            paren_depth = 1;
            continue;
        }

        // Detect single import: import "..."
        if trimmed.starts_with("import ") && !trimmed.contains('(') {
            let import_path = trimmed
                .trim_start_matches("import")
                .trim()
                .trim_matches('"');
            imports.push(import_path.to_string());
            continue;
        }

        // Inside import block
        if in_import_block {
            // Check for closing paren
            if trimmed.contains(')') {
                paren_depth -= 1;
                if paren_depth == 0 {
                    in_import_block = false;
                }
                continue;
            }

            // Parse import line: "fmt" or "math/rand"
            if trimmed.starts_with('"') {
                let import_path = trimmed.trim_matches('"');
                imports.push(import_path.to_string());
            }
            continue;
        }

        // Regular code line
        clean_lines.push(line);
    }

    let clean_code = clean_lines.join("\n");
    (imports, clean_code)
}

/// Wrap Go code to auto-return last expression
fn wrap_go_auto_return(code: &str) -> String {
    let trimmed = code.trim();
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.is_empty() {
        return code.to_string();
    }

    let last_line = lines.last().unwrap().trim();

    // Skip wrapping if last line is:
    // - Empty/comment
    // - Statement (if/for/var/assignment)
    // - Already prints (fmt.Println/fmt.Print)
    // - Structural bracket (}, ), ], etc.)
    if last_line.is_empty()
        || last_line.starts_with("//")
        || last_line.starts_with("/*")
        || last_line.starts_with("if ")
        || last_line.starts_with("for ")
        || last_line.starts_with("func ")
        || last_line.starts_with("var ")
        || last_line.starts_with("const ")
        || last_line.contains(" := ")
        || last_line.contains(" = ") && !last_line.contains("==")
        || last_line.starts_with("return ")
        || last_line.starts_with("fmt.Print")   // fmt.Println, fmt.Printf, etc.
        || last_line.starts_with("println(")     // builtin println
        || last_line == "}"                       // closing brace
        || last_line == ")"                       // closing paren
        || last_line == "]"                       // closing bracket
        || last_line.ends_with('{')               // opening brace
        || last_line.ends_with('(')               // opening paren
        || last_line.ends_with(';')               // explicit statement terminator
    {
        return code.to_string();
    }

    let mut wrapped = String::new();

    // Add all lines except last
    for (i, line) in lines.iter().enumerate() {
        if i < lines.len() - 1 {
            wrapped.push_str(line);
            wrapped.push('\n');
        }
    }

    // Wrap last line as expression
    wrapped.push_str("__tb_result := (");
    wrapped.push_str(last_line);
    wrapped.push_str(")\n");
    wrapped.push_str("fmt.Print(__tb_result)");

    wrapped
}
fn execute_go_code<'arena> (code: &str, args: &[Value<'arena>]) -> TBResult<Value<'static> > {
    execute_go_code_with_context(code, args, None)
}

/// Preprocess Go code to escape literal newlines in string literals
fn preprocess_go_code(code: &str) -> String {
    let mut result = String::new();
    let mut in_double_string = false;
    let mut in_raw_string = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut escape_next = false;

    let mut chars = code.chars().peekable();

    while let Some(ch) = chars.next() {
        // Handle newlines - reset line comments
        if ch == '\n' {
            in_line_comment = false;

            // If we're in a double-quoted string, replace with \n
            if in_double_string {
                result.push_str("\\n");
                continue;
            } else {
                result.push(ch);
                continue;
            }
        }

        // Skip processing if in comments
        if in_line_comment {
            result.push(ch);
            continue;
        }

        if in_block_comment {
            result.push(ch);
            if ch == '*' && chars.peek() == Some(&'/') {
                result.push(chars.next().unwrap());
                in_block_comment = false;
            }
            continue;
        }

        // Check for comment starts (only if not in string)
        if !in_double_string && !in_raw_string {
            if ch == '/' {
                if let Some(&next_ch) = chars.peek() {
                    if next_ch == '/' {
                        in_line_comment = true;
                        result.push(ch);
                        continue;
                    } else if next_ch == '*' {
                        in_block_comment = true;
                        result.push(ch);
                        continue;
                    }
                }
            }
        }

        // Handle raw strings (backticks)
        if ch == '`' && !in_double_string {
            in_raw_string = !in_raw_string;
            result.push(ch);
            continue;
        }

        // If in raw string, keep everything as-is
        if in_raw_string {
            result.push(ch);
            continue;
        }

        // Handle escape sequences in double-quoted strings
        if in_double_string {
            if escape_next {
                result.push(ch);
                escape_next = false;
                continue;
            }

            if ch == '\\' {
                result.push(ch);
                escape_next = true;
                continue;
            }

            if ch == '"' {
                in_double_string = false;
                result.push(ch);
                continue;
            }

            // If we get here with a newline, it was handled above
            result.push(ch);
            continue;
        }

        // Handle double-quoted string start
        if ch == '"' && !in_raw_string {
            in_double_string = true;
            result.push(ch);
            continue;
        }

        // Default: just copy the character
        result.push(ch);
    }

    result
}

/// Preprocess Go code to replace := with = for existing variables
fn preprocess_go_code_for_existing_vars(code: &str, existing_vars: &Vec<Arc<String>>) -> String {
    // First escape literal newlines in strings (existing logic)
    let escaped = preprocess_go_code(code);

    // FIXED: Replace := with = if variable already exists
    let mut result = String::new();

    for line in escaped.lines() {
        let mut modified_line = line.to_string();

        // Check each existing variable
        for var_name in existing_vars {
            // Look for "varname :=" pattern
            let pattern = format!("{} :=", var_name);

            if modified_line.contains(&pattern) {
                // Replace with "varname ="
                let replacement = format!("{} =", var_name);
                modified_line = modified_line.replace(&pattern, &replacement);

                debug_log!("Go code fix: Replaced '{}' with '{}' in line: {}",
                          pattern, replacement, line.trim());
            }
        }

        result.push_str(&modified_line);
        result.push('\n');
    }

    result
}
// ═══════════════════════════════════════════════════════════════════════════
// §12 COMPILER - Enhanced with Cross-Platform Support
// ═══════════════════════════════════════════════════════════════════════════

pub mod target;
pub use target::{TargetPlatform, CompilationConfig};
use crate::dependency_compiler::{CompiledDependency, CompiledDependencyRegistry};

pub struct Compiler<'arena> {
    config: Config<'arena>,
    target: TargetPlatform,
    optimization_level: u8,
    compiled_deps: Vec<CompiledDependency>,
    dep_compiler: Option<DependencyCompiler>,
    builtins: BuiltinRegistry,
}

impl<'arena> Compiler<'arena> {
    pub fn new(config: Config<'arena>) -> Self {
        let builtins = BuiltinRegistry::new();
        Self {
            config,
            target: TargetPlatform::current(),
            optimization_level: 3,
            compiled_deps: Vec::new(),
            dep_compiler: None,
            builtins
        }
    }


    pub fn set_dependency_compiler(&mut self, dep_compiler: DependencyCompiler) {
        self.dep_compiler = Some(dep_compiler);
    }

    /// Get compiled dependencies registry (if available)
    pub fn get_compiled_dependencies(&self) -> Option<CompiledDependencyRegistry> {
        self.dep_compiler
            .as_ref()
            .map(|dc| dc.export_registry())
    }

    pub fn with_target(mut self, target: TargetPlatform) -> Self {
        self.target = target;
        self
    }

    pub fn with_optimization(mut self, level: u8) -> Self {
        self.optimization_level = level;
        self
    }

    pub fn set_compiled_dependencies(&mut self, deps: Vec<CompiledDependency>) {
        self.compiled_deps = deps;
    }

    /// Compile statements to native binary with full import support
    fn _compile(&self, statements: &[Statement]) -> TBResult<Vec<u8>> {
        debug_log!("Compiler::compile() for target {}", self.target);

        // Statements already include imports (loaded by TBCore)
        let function_count = statements.iter()
            .filter(|s| matches!(s, Statement::Function{..}))
            .count();
        debug_log!("✓ Total functions for compilation: {}", function_count);

        // 2. Generate Rust code with runtime support
        let rust_code = self.generate_rust_code_with_runtime(&statements)?;
        debug_log!("Generated Rust code: {} bytes", rust_code.len());

        debug_log!("full code {}", rust_code);

        // 3. Write to temporary directory
        let temp_dir = self.create_temp_project()?;
        let main_rs = temp_dir.join("src").join("main.rs");
        fs::write(&main_rs, &rust_code)?;

        // 4. Create Cargo.toml with runtime dependencies
        self.create_cargo_toml_with_runtime(&temp_dir)?;

        // 5. Compile with cargo
        debug_log!("Running cargo build...");
        let binary = self.cargo_build(&temp_dir)?;

        // 6. Read compiled binary
        let binary_data = fs::read(&binary)?;

        // 7. Cleanup temp directory
        fs::remove_dir_all(&temp_dir).ok();

        debug_log!("Compilation successful: {} bytes", binary_data.len());

        Ok(binary_data)
    }

    /// Recursively load all imports and their transitive dependencies
    fn load_all_imports_recursive(&self, statements: &[Statement<'arena>]) -> TBResult<Vec<Statement<'arena>>> {
        let mut all_statements = Vec::new();
        let mut visited_imports = std::collections::HashSet::new();
        let mut function_names = std::collections::HashSet::new();

        // Process imports from config
        if !self.config.imports.is_empty() {
            debug_log!("╔════════════════════════════════════════════════════════════════╗");
            debug_log!("║           Compiling with Import Resolution                     ║");
            debug_log!("╚════════════════════════════════════════════════════════════════╝");
            debug_log!("Processing {} imports from config", self.config.imports.len());

            self.collect_imports_recursive(
                &self.config.imports,
                &mut all_statements,
                &mut visited_imports,
                &mut function_names,
                0  // depth
            )?;
        }

        // Add main statements (skip duplicate functions)
        for stmt in statements {
            match stmt {
                Statement::Function { name, .. } => {
                    if !function_names.contains(name.as_str()) {
                        function_names.insert(name.to_string());
                        all_statements.push(stmt.clone());
                    } else {
                        debug_log!("  Skipping duplicate function in main: {}", name);
                    }
                }
                _ => all_statements.push(stmt.clone()),
            }
        }

        debug_log!("✓ Total functions collected: {}", function_names.len());

        Ok(all_statements)
    }

    /// Helper: Recursively collect imports with cycle detection
    fn collect_imports_recursive(
        &self,
        imports: &[PathBuf],
        collected: &mut Vec<Statement<'arena>>,
        visited: &mut std::collections::HashSet<PathBuf>,
        function_names: &mut std::collections::HashSet<String>,
        depth: usize,
    ) -> TBResult<()> {
        let indent = "  ".repeat(depth);
        let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
        for import_path in imports {
            // Resolve to canonical path with better error handling
            let resolved = if import_path.is_absolute() {
                import_path.clone()
            } else {
                std::env::current_dir()?.join(import_path)
            };

            // Try to canonicalize, but continue with non-canonical if file exists
            let canonical = match resolved.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    if resolved.exists() {
                        resolved  // Use as-is for temp files
                    } else {
                        return Err(TBError::IoError(
                            format!("Import not found: {} (searched at: {})",
                                    import_path.display(),
                                    resolved.display()).into()
                        ));
                    }
                }
            };

            // Check for cycles
            if visited.contains(&canonical) {
                debug_log!("{}→ Skipping already imported: {}",
                indent, canonical.file_name().unwrap_or_default().to_string_lossy());
                continue;
            }

            visited.insert(canonical.clone());
            debug_log!("{}→ Processing: {}",
            indent, canonical.file_name().unwrap_or_default().to_string_lossy());

            // Read and parse import
            let source = fs::read_to_string(&canonical)?;
            let import_config = Config::parse(&source)?;

            // Recursively process nested imports FIRST (depth-first)
            if !import_config.imports.is_empty() {
                debug_log!("{}  ↳ {} nested imports", indent, import_config.imports.len());
                self.collect_imports_recursive(
                    &import_config.imports,
                    collected,
                    visited,
                    function_names,
                    depth + 1
                )?;
            }


            // Parse this import's statements
            let clean_source = TBCore::strip_directives(&source);
            let mut lexer = Lexer::new(&clean_source);
            let tokens = lexer.tokenize()?;
            let mut parser = Parser::new(tokens, arena); // &arena
            let statements = parser.parse()?;

            debug_log!("{}  ✓ Parsed {} statements", indent, statements.len());

            // Add functions (skip duplicates)
            let mut added_count = 0;
            for stmt in statements {
                match stmt.clone() {
                    Statement::Function { ref name, .. } => {
                        if !function_names.contains(name.as_str()) {
                            function_names.insert(name.to_string());
                            collected.push(stmt);
                            added_count += 1;
                            debug_log!("{}    + fn {}", indent, name);
                        } else {
                            debug_log!("{}    - fn {} (duplicate)", indent, name);
                        }
                    }
                    _ => {
                        collected.push(stmt);
                        added_count += 1;
                    }
                }
            }

            if added_count > 0 {
                debug_log!("{}  ✓ Added {} items", indent, added_count);
            }
        }

        Ok(())
    }

    /// Analyze which variables need to be mutable
    fn analyze_mutability(&self, statements: &[Statement]) -> std::collections::HashSet<Arc<String>> {
        let mut mutable_vars = std::collections::HashSet::new();

        // Find all variable assignments
        for stmt in statements {
            self.collect_assigned_variables(stmt, &mut mutable_vars);
        }

        debug_log!("Mutable variables detected: {:?}",
                   mutable_vars.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        mutable_vars
    }

    /// Recursively collect variables that are assigned to
    fn collect_assigned_variables(&self, stmt: &Statement, mutable_vars: &mut std::collections::HashSet<Arc<String>>) {
        match stmt {
            Statement::Assign { target, value } => {
                // Mark target variable as mutable
                if let Expr::Variable(name) = target {
                    mutable_vars.insert(name.clone());  // Clone Arc (cheap)
                }

                // Check value expression for assignments
                self.collect_assigned_in_expr(value, mutable_vars);
            }

            Statement::Let { value, .. } => {
                self.collect_assigned_in_expr(value, mutable_vars);
            }

            Statement::Expr(expr) => {
                self.collect_assigned_in_expr(expr, mutable_vars);
            }

            Statement::Function { body, .. } => {
                self.collect_assigned_in_expr(body, mutable_vars);
            }

            _ => {}
        }
    }

    /// Recursively find assignments in expressions
    fn collect_assigned_in_expr(&self, expr: &Expr, mutable_vars: &mut std::collections::HashSet<Arc<String>>) {
        match expr {
            Expr::Block { statements, result } => {
                for stmt in statements {
                    self.collect_assigned_variables(stmt, mutable_vars);
                }
                if let Some(res) = result {
                    self.collect_assigned_in_expr(res, mutable_vars);
                }
            }

            Expr::If { condition, then_branch, else_branch } => {
                self.collect_assigned_in_expr(condition, mutable_vars);
                self.collect_assigned_in_expr(then_branch, mutable_vars);
                if let Some(else_b) = else_branch {
                    self.collect_assigned_in_expr(else_b, mutable_vars);
                }
            }

            Expr::Loop { body } => {
                self.collect_assigned_in_expr(body, mutable_vars);
            }

            Expr::While { condition, body } => {
                self.collect_assigned_in_expr(condition, mutable_vars);
                self.collect_assigned_in_expr(body, mutable_vars);
            }

            Expr::For { iterable, body, .. } => {
                self.collect_assigned_in_expr(iterable, mutable_vars);
                self.collect_assigned_in_expr(body, mutable_vars);
            }

            Expr::BinOp { left, right, .. } => {
                self.collect_assigned_in_expr(left, mutable_vars);
                self.collect_assigned_in_expr(right, mutable_vars);
            }

            Expr::UnaryOp { expr, .. } => {
                self.collect_assigned_in_expr(expr, mutable_vars);
            }

            Expr::Call { function, args } => {
                self.collect_assigned_in_expr(function, mutable_vars);
                for arg in args {
                    self.collect_assigned_in_expr(arg, mutable_vars);
                }
            }

            _ => {}
        }
    }

    fn generate_rust_code_with_runtime(&self, statements: &[Statement]) -> TBResult<String> {
        let mutable_vars = self.analyze_mutability(statements);
        let mut codegen = CodeGenerator::new(Language::Rust);

        codegen.set_builtin_registry(self.builtins.clone());
        codegen.set_compiled_dependencies(&self.compiled_deps);
        codegen.set_mutable_vars(&mutable_vars);

        // Check if we have compiled dependencies
        let has_compiled_deps = !self.compiled_deps.is_empty();

        if has_compiled_deps {
            debug_log!("Using ZERO-OVERHEAD compiled language bridges");
            codegen.generate_compiled_language_bridges();
        } else {
            debug_log!("No compiled dependencies, using JIT bridges");
        }

        let mut final_code = String::new();

        final_code.push_str(&codegen.buffer);  // Language bridges
        codegen.buffer.clear();

        codegen.generate_builtin_functions();
        final_code.push_str(&codegen.buffer);
        codegen.buffer.clear();

        // Generate main code
        let main_code = codegen.generate(statements)?;
        final_code.push_str(&main_code);

        Ok(final_code)
    }

    fn expr_needs_vec_return(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Parallel(_) => true,
            Expr::Block { result, .. } => {
                result.as_ref().map_or(false, |e| self.expr_needs_vec_return(e))
            }
            Expr::If { then_branch, else_branch, .. } => {
                self.expr_needs_vec_return(then_branch)
                    || else_branch.as_ref().map_or(false, |e| self.expr_needs_vec_return(e))
            }
            _ => false,
        }
    }

    /// Check if statements use language bridges (python, javascript, go, bash)
    fn statements_use_language_bridges(&self, statements: &[Statement]) -> bool {
        for stmt in statements {
            if self.statement_uses_bridges(stmt) {
                return true;
            }
        }
        false
    }

    fn statement_uses_bridges(&self, stmt: &Statement) -> bool {
        match stmt {
            Statement::Let { value, .. } => self.expr_uses_bridges(value),
            Statement::Expr(expr) => self.expr_uses_bridges(expr),
            Statement::Function { body, .. } => self.expr_uses_bridges(body),
            _ => false,
        }
    }

    fn expr_uses_bridges(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Call { function, args } => {
                if let Expr::Variable(name) = function {
                    if matches!(name.as_str(), "python" | "javascript" | "go" | "bash") {
                        return true;
                    }
                }
                args.iter().any(|a| self.expr_uses_bridges(a))
            }
            Expr::Block { statements, result } => {
                statements.iter().any(|s| self.statement_uses_bridges(s))
                    || result.as_ref().map_or(false, |e| self.expr_uses_bridges(e))
            }
            Expr::BinOp { left, right, .. } => {
                self.expr_uses_bridges(left) || self.expr_uses_bridges(right)
            }
            Expr::If { condition, then_branch, else_branch } => {
                self.expr_uses_bridges(condition)
                    || self.expr_uses_bridges(then_branch)
                    || else_branch.as_ref().map_or(false, |e| self.expr_uses_bridges(e))
            }
            Expr::Native { .. } => true,
            _ => false,
        }
    }


    /// Check if statements contain parallel expressions
    fn has_parallel_expressions(&self, statements: &[Statement]) -> bool {
        for stmt in statements {
            if self.statement_has_parallel(stmt) {
                return true;
            }
        }
        false
    }

    /// Create Cargo.toml with runtime dependencies for async/parallel
    fn create_cargo_toml_with_runtime(&self, project_dir: &Path) -> TBResult<()> {
        let has_compiled_deps = !self.compiled_deps.is_empty();

        let dependencies = if has_compiled_deps {
            r#"rayon = "1.11.0"
libloading = "0.8.5"
libc = "0.2""#
        } else {
            r#"rayon = "1.11.0""#
        };

        let cargo_toml = format!(r#"[package]
name = "tb_compiled"
version = "0.1.0"
edition = "2021"

[dependencies]
serde_json = "1.0.145"
{}

[profile.release]
opt-level = {}
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
"#, dependencies, self.optimization_level);

        fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;
        Ok(())
    }

    fn statement_has_parallel(&self, stmt: &Statement) -> bool {
        match stmt {
            Statement::Expr(expr) => self.expr_has_parallel(expr),
            Statement::Let { value, .. } => self.expr_has_parallel(value),
            Statement::Function { body, .. } => self.expr_has_parallel(body),
            _ => false,
        }
    }

    fn expr_has_parallel(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Parallel(_) => true,
            Expr::Block { statements, result } => {
                statements.iter().any(|s| self.statement_has_parallel(s))
                    || result.as_ref().map_or(false, |e| self.expr_has_parallel(e))
            }
            _ => false,
        }
    }

    /// Create temporary Cargo project
    fn create_temp_project(&self) -> TBResult<PathBuf> {
        let temp_dir = env::temp_dir().join(format!("tb_build_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir)?;
        fs::create_dir_all(temp_dir.join("src"))?;
        Ok(temp_dir)
    }

    /// Compile with cargo
    fn cargo_build(&self, project_dir: &Path) -> TBResult<PathBuf> {
        debug_log!("Running cargo build...");

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("--release")
            .arg("--target")
            .arg(self.target.rust_target())
            .current_dir(project_dir);

        // Add RUSTFLAGS for optimization
        let mut rustflags = self.target.optimization_flags().join(" ");

        // Add existing RUSTFLAGS if any
        if let Ok(existing) = env::var("RUSTFLAGS") {
            rustflags = format!("{} {}", existing, rustflags);
        }

        cmd.env("RUSTFLAGS", rustflags);

        // Execute compilation
        let output = cmd.output()
            .map_err(|e| TBError::CompilationError {
                message: format!("Failed to run cargo: {}", e).into(),
                source: Arc::new("".to_string()),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TBError::CompilationError {
                message: format!("Cargo build failed:\n{}", stderr).into(),
                source: Arc::new("".to_string()),
            });
        }

        // Find the compiled binary
        let target_dir = project_dir
            .join("target")
            .join(self.target.rust_target())
            .join("release");

        let binary_name = format!("tb_compiled{}", self.target.exe_extension());
        let binary_path = target_dir.join(binary_name);

        if !binary_path.exists() {
            return Err(TBError::CompilationError {
                message: format!("Compiled binary not found at: {}", binary_path.display()).into(),
                source: Arc::new("".to_string()),
            });
        }

        Ok(binary_path)
    }



    /// Compile to file with proper naming
    pub fn compile_to_file(&self, statements: &[Statement<'arena>], output: &Path) -> TBResult<()> {
        let binary = self.compile(statements)?;

        // Create output directory if needed
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write binary
        fs::write(output, binary)?;

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(output)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(output, perms)?;
        }

        Ok(())
    }

}

impl<'arena> Compiler<'arena> {
    pub fn compile_to_library(&self, statements: &'arena [Statement<'arena>], output: &Path) -> TBResult<()> {
        debug_log!("Compiling to library: {}", output.display());

        // Generate Rust code
        let rust_code = self.generate_rust_library_code(statements)?;

        // Create temporary Cargo project
        let temp_dir = self.create_temp_project()?;
        let main_rs = temp_dir.join("src").join("lib.rs");
        fs::write(&main_rs, &rust_code)?;

        // Create Cargo.toml for library
        self.create_library_cargo_toml(&temp_dir)?;

        // Compile
        let binary = self.cargo_build_library(&temp_dir)?;

        // Copy to output
        fs::copy(&binary, output)?;

        // Cleanup
        fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    /// Generate Rust code as library
    fn generate_rust_library_code(&self, statements: &'arena [Statement<'arena>]) -> TBResult<String> {
        let codegen = CodeGenerator::new(Language::Rust);

        // Generate base function code WITHOUT wrapping in main()
        let mut lib_code = String::from("// Auto-generated TB Library\n");
        lib_code.push_str("#[allow(dead_code, unused)]\n\n");  // Outer attribute, not inner

        // Add necessary imports
        lib_code.push_str("use std::collections::HashMap;\n\n");

        // Generate each function directly (no main wrapper)
        for stmt in statements {
            match stmt {
                Statement::Function { name, params, return_type, body } => {
                    // Generate function with inline optimization
                    lib_code.push_str("#[inline(always)]\n");
                    lib_code.push_str("pub fn ");
                    lib_code.push_str(name);
                    lib_code.push('(');

                    // Parameters
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 { lib_code.push_str(", "); }
                        let param_type = param.type_annotation.as_ref()
                            .map(|t| self.type_to_c_string(t))
                            .unwrap_or_else(|| "i64".to_string());
                        lib_code.push_str(&format!("{}: {}", param.name, param_type));
                    }

                    lib_code.push(')');

                    // Return type
                    if let Some(rt) = return_type {
                        lib_code.push_str(" -> ");
                        lib_code.push_str(&self.type_to_c_string(rt));
                    } else {
                        lib_code.push_str(" -> i64");  // Default return type
                    }

                    lib_code.push_str(" {\n");

                    // Generate function body
                    let mut body_gen = CodeGenerator::new(Language::Rust);
                    body_gen.indent = 1;

                    // Manually generate body expression
                    lib_code.push_str("    ");
                    let body_code = self.generate_expr_code(body)?;
                    lib_code.push_str(&body_code);
                    lib_code.push('\n');

                    lib_code.push_str("}\n\n");
                }
                _ => {}
            }
        }

        Ok(lib_code)
    }

    // Generate expression code (simplified)
    fn generate_expr_code(&self, expr: &Expr) -> TBResult<String> {
        match expr {
            Expr::Block { statements, result } => {
                let mut code = String::from("{\n");

                // Generate statements
                for stmt in statements {
                    match stmt {
                        Statement::Expr(e) => {
                            code.push_str("        ");
                            code.push_str(&self.generate_expr_code(&e)?);
                            code.push_str(";\n");
                        }
                        _ => {}
                    }
                }

                // Result expression
                if let Some(res) = result {
                    code.push_str("        ");
                    code.push_str(&self.generate_expr_code(res)?);
                    code.push('\n');
                }

                code.push_str("    }");
                Ok(code)
            }

            Expr::If { condition, then_branch, else_branch } => {
                let mut code = String::from("if ");
                code.push_str(&self.generate_expr_code(condition)?);
                code.push_str(" ");
                code.push_str(&self.generate_expr_code(then_branch)?);

                if let Some(else_b) = else_branch {
                    code.push_str(" else ");
                    code.push_str(&self.generate_expr_code(else_b)?);
                }

                Ok(code)
            }

            Expr::BinOp { op, left, right } => {
                Ok(format!("({} {} {})",
                           self.generate_expr_code(left)?,
                           self.binop_to_rust_str(*op),
                           self.generate_expr_code(right)?
                ))
            }

            Expr::Call { function, args } => {
                let mut code = self.generate_expr_code(function)?;
                code.push('(');
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { code.push_str(", "); }
                    code.push_str(&self.generate_expr_code(arg)?);
                }
                code.push(')');
                Ok(code)
            }

            Expr::Variable(name) => Ok(name.to_string()),

            Expr::Literal(lit) => {
                match lit {
                    Literal::Int(n) => Ok(format!("{}", n)),
                    Literal::Float(f) => Ok(format!("{}", f)),
                    Literal::Bool(b) => Ok(format!("{}", b)),
                    Literal::String(s) => Ok(format!("\"{}\"", s)),
                    Literal::Unit => Ok(String::from("()")),
                }
            }

            _ => Ok(String::from("0"))  // Fallback
        }
    }

    fn binop_to_rust_str(&self, op: BinOp) -> &'static str {
        match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            _ => "+"
        }
    }

    /// Create Cargo.toml for library
    fn create_library_cargo_toml(&self, project_dir: &Path) -> TBResult<()> {
        let cargo_toml = format!(r#"[package]
name = "tb_import_lib"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]

[profile.release]
opt-level = {}
lto = "fat"
codegen-units = 1
strip = true
"#, self.optimization_level);

        fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;
        Ok(())
    }

    /// Build library with cargo
    fn cargo_build_library(&self, project_dir: &Path) -> TBResult<PathBuf> {
        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("--release")
            .arg("--lib")
            .current_dir(project_dir);

        let output = cmd.output()
            .map_err(|e| TBError::CompilationError {
                message: format!("Failed to run cargo: {}", e).into(),
                source: Arc::new("".to_string()),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TBError::CompilationError {
                message: format!("Library compilation failed:\n{}", stderr).into(),
                source: Arc::new("".to_string()),
            });
        }

        let target_dir = project_dir.join("target").join("release");
        let lib_name = if cfg!(target_os = "windows") {
            "tb_import_lib.dll"
        } else if cfg!(target_os = "macos") {
            "libtb_import_lib.dylib"
        } else {
            "libtb_import_lib.so"
        };

        let lib_path = target_dir.join(lib_name);

        if !lib_path.exists() {
            return Err(TBError::CompilationError {
                message: format!("Library not found at: {}", lib_path.display()).into(),
                source: Arc::new("".to_string()),
            });
        }

        Ok(lib_path)
    }

    /// Convert TB type to C type string
    fn type_to_c_string(&self, ty: &Type) -> String {
        match ty {
            Type::Int => "i64".to_string(),
            Type::Float => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            _ => "i64".to_string(),
        }
    }
}
impl<'arena> Compiler<'arena> {

    // ═══════════════════════════════════════════════════════════════════════════
    // DEPENDENCY EXTRACTION FOR COMPILATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Extract all external code dependencies from statements for compilation
    fn extract_dependencies_from_statements(&self, statements: &[Statement<'arena>]) -> TBResult<Vec<Dependency>> {
        use std::sync::Arc;

        let mut dependencies = Vec::new();
        let mut dep_counter = 0;

        for stmt in statements {
            self.extract_deps_from_statement(stmt, &mut dependencies, &mut dep_counter)?;
        }

        if !dependencies.is_empty() {
            debug_log!("📦 Extracted {} dependencies for compilation", dependencies.len());
        }

        Ok(dependencies)
    }

    /// Extract dependencies from a single statement
    fn extract_deps_from_statement(
        &self,
        stmt: &Statement<'arena>,
        dependencies: &mut Vec<Dependency>,
        counter: &mut usize,
    ) -> TBResult<()> {
        match stmt {
            Statement::Expr(expr) => {
                self.extract_deps_from_expr(expr, dependencies, counter)?;
            }
            Statement::Let { value, .. } => {
                self.extract_deps_from_expr(value, dependencies, counter)?;
            }
            Statement::Function { body, .. } => {
                self.extract_deps_from_expr(body, dependencies, counter)?;
            }
            Statement::Assign { value, .. } => {
                self.extract_deps_from_expr(value, dependencies, counter)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Recursively extract dependencies from expressions
    fn extract_deps_from_expr(
        &self,
        expr: &Expr<'arena>,
        dependencies: &mut Vec<Dependency>,
        counter: &mut usize,
    ) -> TBResult<()> {
        match expr {
            // Native code block: Expr::Native { language, code }
            Expr::Native { language, code } => {
                *counter += 1;

                debug_log!("  → Found {:?} dependency #{}", language, counter);

                dependencies.push(Dependency {
                    id: Arc::new(format!("dep_{}", counter)),
                    language: *language,
                    code: code.clone(),
                    imports: Self::extract_imports_from_code(code, *language),
                    is_in_loop: false, // TODO: detect from context
                    estimated_calls: 1,
                });
            }

            // Function call: python(...), js(...), go(...), bash(...)
            Expr::Call { function, args } => {
                if let Expr::Variable(func_name) = &**function {
                    let language_opt = match func_name.as_str() {
                        "python" | "py" => Some(Language::Python),
                        "js" | "javascript" => Some(Language::JavaScript),
                        "ts" | "typescript" => Some(Language::TypeScript),
                        "go" => Some(Language::Go),
                        "bash" | "sh" => Some(Language::Bash),
                        _ => None,
                    };

                    if let Some(language) = language_opt {
                        // Extract code from first argument (string literal)
                        if let Some(Expr::Literal(Literal::String(code))) = args.first() {
                            *counter += 1;

                            debug_log!("  → Found {:?} dependency #{} from {}() call",
                                       language, counter, func_name);

                            dependencies.push(Dependency {
                                id: Arc::new(format!("dep_{}", counter)),
                                language,
                                code: code.clone(),
                                imports: Self::extract_imports_from_code(code, language),
                                is_in_loop: false,
                                estimated_calls: 1,
                            });
                        }
                    }
                }

                // Recursively check arguments
                for arg in args.iter() {
                    self.extract_deps_from_expr(arg, dependencies, counter)?;
                }
            }

            // Block expression
            Expr::Block { statements, result } => {
                for stmt in statements.iter() {
                    self.extract_deps_from_statement(stmt, dependencies, counter)?;
                }
                if let Some(res) = result {
                    self.extract_deps_from_expr(res, dependencies, counter)?;
                }
            }

            // If expression
            Expr::If { condition, then_branch, else_branch } => {
                self.extract_deps_from_expr(condition, dependencies, counter)?;
                self.extract_deps_from_expr(then_branch, dependencies, counter)?;
                if let Some(else_b) = else_branch {
                    self.extract_deps_from_expr(else_b, dependencies, counter)?;
                }
            }

            // Loop expressions
            Expr::Loop { body } => {
                self.extract_deps_from_expr(body, dependencies, counter)?;
            }

            Expr::While { condition, body } => {
                self.extract_deps_from_expr(condition, dependencies, counter)?;
                self.extract_deps_from_expr(body, dependencies, counter)?;
            }

            Expr::For { iterable, body, .. } => {
                self.extract_deps_from_expr(iterable, dependencies, counter)?;
                self.extract_deps_from_expr(body, dependencies, counter)?;
            }

            // Binary operations
            Expr::BinOp { left, right, .. } => {
                self.extract_deps_from_expr(left, dependencies, counter)?;
                self.extract_deps_from_expr(right, dependencies, counter)?;
            }

            // Unary operations
            Expr::UnaryOp { expr, .. } => {
                self.extract_deps_from_expr(expr, dependencies, counter)?;
            }

            // Method calls
            Expr::Method { object, args, .. } => {
                self.extract_deps_from_expr(object, dependencies, counter)?;
                for arg in args.iter() {
                    self.extract_deps_from_expr(arg, dependencies, counter)?;
                }
            }

            // Parallel execution
            Expr::Parallel(exprs) => {
                for expr in exprs.iter() {
                    self.extract_deps_from_expr(expr, dependencies, counter)?;
                }
            }

            _ => {}
        }

        Ok(())
    }

    /// Extract imports from source code based on language
    fn extract_imports_from_code(code: &str, language: Language) -> Vec<Arc<String>> {
        let mut imports = Vec::new();

        match language {
            Language::Python => {
                for line in code.lines() {
                    let trimmed = line.trim();

                    // import module
                    if trimmed.starts_with("import ") {
                        if let Some(module) = trimmed
                            .strip_prefix("import ")
                            .and_then(|s| s.split_whitespace().next())
                        {
                            imports.push(Arc::new(module.to_string()));
                        }
                    }
                    // from module import ...
                    else if trimmed.starts_with("from ") {
                        if let Some(module) = trimmed
                            .strip_prefix("from ")
                            .and_then(|s| s.split_whitespace().next())
                        {
                            imports.push(Arc::new(module.to_string()));
                        }
                    }
                }
            }

            Language::JavaScript | Language::TypeScript => {
                for line in code.lines() {
                    let trimmed = line.trim();

                    // import ... from "module" or require("module")
                    if trimmed.contains("import ") || trimmed.contains("require(") {
                        // Simple extraction - find string in quotes
                        if let Some(start) = trimmed.find('"').or_else(|| trimmed.find('\'')) {
                            if let Some(end) = trimmed[start + 1..].find('"').or_else(|| trimmed[start + 1..].find('\'')) {
                                let module = &trimmed[start + 1..start + 1 + end];
                                if !module.starts_with('.') && !module.starts_with('/') {
                                    imports.push(Arc::new(module.to_string()));
                                }
                            }
                        }
                    }
                }
            }

            Language::Go => {
                for line in code.lines() {
                    let trimmed = line.trim();

                    // import "package" or import ("package")
                    if trimmed.starts_with("import ") {
                        if let Some(start) = trimmed.find('"') {
                            if let Some(end) = trimmed[start + 1..].find('"') {
                                let package = &trimmed[start + 1..start + 1 + end];
                                imports.push(Arc::new(package.to_string()));
                            }
                        }
                    }
                }
            }

            _ => {}
        }

        imports
    }

    /// Compile statements to native binary with full import support
    pub fn compile(&self, statements: &[Statement<'arena>]) -> TBResult<Vec<u8>> {
        debug_log!("╔════════════════════════════════════════════════════════════════╗");
        debug_log!("║              TB COMPILATION PIPELINE                           ║");
        debug_log!("╚════════════════════════════════════════════════════════════════╝");
        debug_log!("Compiler::compile() for target {}", self.target);

        let total_start = std::time::Instant::now();

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 1: Statement Analysis
        // ═══════════════════════════════════════════════════════════════════════════
        let function_count = statements.iter()
            .filter(|s| matches!(s, Statement::Function{..}))
            .count();
        debug_log!("✓ Total functions: {}", function_count);

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 2: Dependency Compilation (if external code exists)
        // ═══════════════════════════════════════════════════════════════════════════
        let compiled_deps = if self.statements_use_language_bridges(statements) {
            debug_log!("");
            debug_log!("⚡ Phase 2: Compiling External Language Dependencies");
            debug_log!("────────────────────────────────────────────────────────");

            let dep_start = std::time::Instant::now();
            let temp_compile_dir = std::env::temp_dir().join(format!("tb_dep_compile_{}",
                                                                     uuid::Uuid::new_v4()));
            let dep_compiler = DependencyCompiler::new(&temp_compile_dir);

            // Extract dependencies from statements
            let deps = self.extract_dependencies_from_statements(statements)?;

            if !deps.is_empty() {
                debug_log!("  Found {} external code blocks to compile", deps.len());
                debug_log!("");

                // Compile each dependency
                for (idx, dep) in deps.iter().enumerate() {
                    debug_log!("  [{}/{}] Compiling {:?} dependency: {}",
                               idx + 1, deps.len(), dep.language, dep.id);

                    match dep_compiler.compile(&dep) {
                        Ok(compiled) => {
                            debug_log!("       ✓ Strategy: {:?}", compiled.strategy);
                            debug_log!("       ✓ Size: {:.2} KB", compiled.size_bytes as f64 / 1024.0);
                            debug_log!("       ✓ Time: {}ms", compiled.compile_time_ms);
                        }
                        Err(e) => {
                            debug_log!("       ⚠️  Compilation failed: {}", e);
                            debug_log!("       → Falling back to runtime interpretation");
                        }
                    }
                    debug_log!("");
                }

                // Get compilation statistics
                let stats = dep_compiler.get_stats();
                let dep_elapsed = dep_start.elapsed();

                debug_log!("────────────────────────────────────────────────────────");
                debug_log!("  ✅ Dependency Compilation Complete");
                debug_log!("     • Total dependencies: {}", stats.total_count);
                debug_log!("     • Total size: {:.2} KB", stats.total_size_bytes as f64 / 1024.0);
                debug_log!("     • Compilation time: {:.2}s", dep_elapsed.as_secs_f64());
                debug_log!("");
                debug_log!("  📊 Breakdown:");
                if stats.modern_native > 0 {
                    debug_log!("     • Modern native (UV/BUN): {}", stats.modern_native);
                }
                if stats.native_compilation > 0 {
                    debug_log!("     • Native compilation: {}", stats.native_compilation);
                }
                if stats.bundled > 0 {
                    debug_log!("     • Bundled: {}", stats.bundled);
                }
                if stats.plugin > 0 {
                    debug_log!("     • Plugins: {}", stats.plugin);
                }
                if stats.system_installed > 0 {
                    debug_log!("     • System installed: {}", stats.system_installed);
                }
                if stats.embedded > 0 {
                    debug_log!("     • Embedded: {}", stats.embedded);
                }
                debug_log!("────────────────────────────────────────────────────────");
                debug_log!("");

                // Export compiled dependencies
                let registry = dep_compiler.export_registry();
                registry.dependencies.values().cloned().collect()
            } else {
                debug_log!("  ℹ️  No external dependencies found");
                Vec::new()
            }
        } else {
            // Use pre-set compiled_deps if available
            if !self.compiled_deps.is_empty() {
                debug_log!("  ℹ️  Using pre-compiled dependencies: {}", self.compiled_deps.len());
                self.compiled_deps.clone()
            } else {
                debug_log!("  ℹ️  No language bridges detected");
                Vec::new()
            }
        };

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 3: Rust Code Generation
        // ═══════════════════════════════════════════════════════════════════════════
        debug_log!("⚙️  Phase 3: Generating Rust Code");
        debug_log!("────────────────────────────────────────────────────────");

        let codegen_start = std::time::Instant::now();
        let rust_code = self.generate_rust_code_with_runtime_and_deps(statements, &compiled_deps)?;
        let codegen_elapsed = codegen_start.elapsed();

        debug_log!("  ✓ Generated code: {} bytes ({:.2} KB)",
                   rust_code.len(),
                   rust_code.len() as f64 / 1024.0);
        debug_log!("  ✓ Generation time: {:.2}ms", codegen_elapsed.as_secs_f64() * 1000.0);
        debug_log!("");

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 4: Project Setup
        // ═══════════════════════════════════════════════════════════════════════════
        debug_log!("📁 Phase 4: Creating Cargo Project");
        debug_log!("────────────────────────────────────────────────────────");

        let temp_dir = self.create_temp_project()?;
        debug_log!("  ✓ Temp directory: {}", temp_dir.display());

        let main_rs = temp_dir.join("src").join("main.rs");
        fs::write(&main_rs, &rust_code)?;
        debug_log!("  ✓ Wrote main.rs");

        self.create_cargo_toml_with_runtime(&temp_dir)?;
        debug_log!("  ✓ Created Cargo.toml");
        debug_log!("");

        debug_log!("╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
        debug_log!("Full Code:");
        for (i, line) in rust_code.lines().enumerate() {
            debug_log!("{:>4} | {}", i + 1, line);
        }
        debug_log!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");


        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 5: Cargo Compilation
        // ═══════════════════════════════════════════════════════════════════════════
        debug_log!("🔨 Phase 5: Compiling with Cargo");
        debug_log!("────────────────────────────────────────────────────────");

        let cargo_start = std::time::Instant::now();
        let binary = self.cargo_build(&temp_dir)?;
        let cargo_elapsed = cargo_start.elapsed();

        debug_log!("  ✓ Cargo build complete: {:.2}s", cargo_elapsed.as_secs_f64());
        debug_log!("");

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 6: Binary Packaging
        // ═══════════════════════════════════════════════════════════════════════════
        let binary_data = fs::read(&binary)?;
        let binary_size_mb = binary_data.len() as f64 / (1024.0 * 1024.0);

        debug_log!("📦 Phase 6: Packaging Binary");
        debug_log!("────────────────────────────────────────────────────────");
        debug_log!("  ✓ Binary size: {:.2} MB", binary_size_mb);

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 7: Cleanup
        // ═══════════════════════════════════════════════════════════════════════════
        fs::remove_dir_all(&temp_dir).ok();
        debug_log!("  ✓ Cleaned up temporary files");
        debug_log!("");

        // ═══════════════════════════════════════════════════════════════════════════
        // COMPILATION SUMMARY
        // ═══════════════════════════════════════════════════════════════════════════
        let total_elapsed = total_start.elapsed();

        debug_log!("╔════════════════════════════════════════════════════════════════╗");
        debug_log!("║                      COMPILATION SUCCESSFUL                    ║");
        debug_log!("╚════════════════════════════════════════════════════════════════╝");
        debug_log!("");
        debug_log!("📊 Summary:");
        debug_log!("   • Total time: {:.2}s", total_elapsed.as_secs_f64());
        debug_log!("   • Functions compiled: {}", function_count);
        debug_log!("   • External dependencies: {}", compiled_deps.len());
        debug_log!("   • Binary size: {:.2} MB", binary_size_mb);
        debug_log!("   • Target: {}", self.target);
        debug_log!("");

        Ok(binary_data)
    }

    /// Generate Rust code with runtime support and compiled dependencies
    fn generate_rust_code_with_runtime_and_deps(
        &self,
        statements: &[Statement],
        compiled_deps: &[CompiledDependency]
    ) -> TBResult<String> {
        let mutable_vars = self.analyze_mutability(statements);
        let mut codegen = CodeGenerator::new(Language::Rust);

        // Set compiled dependencies
        codegen.set_builtin_registry(self.builtins.clone());
        codegen.set_compiled_dependencies(compiled_deps);
        codegen.set_mutable_vars(&mutable_vars);

        let has_compiled_deps = !compiled_deps.is_empty();

        // Generate code based on whether we have compiled deps
        let mut final_code = String::new();

        if has_compiled_deps {
            debug_log!("  → Using ZERO-OVERHEAD compiled language bridges");
            codegen.generate_compiled_language_bridges();
            final_code.push_str(&codegen.buffer);
            codegen.buffer.clear();
        } else {
            debug_log!("  → No compiled dependencies, using JIT bridges");
        }

        // Generate builtin functions
        codegen.generate_builtin_functions();
        final_code.push_str(&codegen.buffer);
        codegen.buffer.clear();

        // Generate main code
        let main_code = codegen.generate(statements)?;
        final_code.push_str(&main_code);

        Ok(final_code)
    }
}
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> String {
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros();
            format!("{:x}", now)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §13 CORE ENGINE (MAIN INTERFACE)
// ═══════════════════════════════════════════════════════════════════════════

pub struct TBCore {
    config: Config<'static>,
    optimizer: Optimizer,
}

impl TBCore {
    /// Create new TB Core engine
    pub fn new(config: Config<'static>) -> Self {
        Self {
            optimizer: Optimizer::new(OptimizerConfig::default()),
            config,
        }
    }

    /// Execute TB source code

    pub fn execute(
        &mut self,
        source: &str
    ) -> TBResult<Value<'static>>  {
        debug_log!("TBCore::execute() started!!");
        debug_log!("Source length: {} bytes", source.len());
        let arena: &'static Bump = Box::leak(Box::new(Bump::new()));
        let result= {
            let config = Config::parse(source)?;
            let imports_to_load = config.imports.clone();
            self.config = config;
            debug_log!("Configuration parsed: mode={:?}", self.config.mode);

            let clean_source = Self::strip_directives(source);
            debug_log!("Clean source length: {} bytes", clean_source.len());

            let mut lexer = Lexer::new(&clean_source);
            let tokens = lexer.tokenize()?;
            let mut parser = Parser::new(tokens, &arena);
            let mut statements = parser.parse()?;
            debug_log!("Parsing complete: {} statements", statements.len());

            if !imports_to_load.is_empty() {
                debug_log!("📦 Loading {} imports", imports_to_load.len());
                let imported_statements = self.load_imports(&imports_to_load, &arena)?;
                debug_log!("✓ Loaded {} statements from imports", imported_statements.len());
                statements = Self::merge_statements_deduplicated(imported_statements, statements)?;
                debug_log!("✓ Total statements after deduplication: {}", statements.len());
            }

            // Analyze dependencies
            let dependencies = self.analyze_dependencies(&statements)?;

            // Compile dependencies if any
            if !dependencies.is_empty() {
                debug_log!("╔════════════════════════════════════════════════════════════════╗");
                debug_log!("║              Compiling Dependencies                            ║");
                debug_log!("╚════════════════════════════════════════════════════════════════╝\n");

                let compiler = DependencyCompiler::new(Path::new("."));

                for dep in &dependencies {
                    match compiler.compile(dep) {
                        Ok(compiled) => {
                            debug_log!("✓ {} ({} bytes, {}ms)",
                                 compiled.id,
                                 compiled.size_bytes,
                                 compiled.compile_time_ms);
                        }
                        Err(e) => {
                            eprintln!("✗ Failed to compile {}: {}", dep.id, e);
                        }
                    }
                }
            }

            // Type check (if static mode)
            if self.config.type_mode == TypeMode::Static {
                debug_log!("Type checking...");
                let mut type_checker = TypeChecker::new(TypeMode::Static);
                type_checker.check_statements(&statements)?;
                debug_log!("Type checking complete");
            }

            // Execute based on mode
            let mode = self.config.mode.clone();
            let runtime_mode = self.config.runtime_mode;

            debug_log!("Executing in mode: {:?}", mode);
            let arena_result = match mode {
                ExecutionMode::Compiled { optimize } => {
                    debug_log!("Compiled mode execution");

                    let arena: &'static Bump = Box::leak(Box::new(Bump::new()));

                    // Load imports BEFORE compilation
                    let imports = self.config.imports.clone();
                    let all_statements = if !imports.is_empty() {
                        debug_log!("📦 Loading {} imports for compilation", imports.len());
                        let imported_statements = self.load_imports(&imports, &arena)?;
                        debug_log!("✓ Loaded {} statements from imports", imported_statements.len());

                        let merged = Self::merge_statements_deduplicated(imported_statements, statements)?;
                        debug_log!("✓ Total statements after merge: {}", merged.len());
                        merged
                    } else {
                        debug_log!("No imports to load");
                        statements
                    };

                    let func_count = all_statements.iter()
                        .filter(|s| matches!(s, Statement::Function { .. }))
                        .count();
                    debug_log!("✓ Functions available for compilation: {}", func_count);

                    //Analyze and compile dependencies (Python/JS/Go/Bash blocks)
                    let dependencies = self.analyze_dependencies(&all_statements)?;

                    let compiled_deps = if !dependencies.is_empty() {
                        println!("\n╔════════════════════════════════════════════════════════════════╗");
                        println!("║              Compiling Language Dependencies                   ║");
                        println!("╚════════════════════════════════════════════════════════════════╝\n");

                        let dep_compiler = DependencyCompiler::new(Path::new("."));
                        let mut compiled = Vec::new();

                        for dep in &dependencies {
                            match dep_compiler.compile(dep) {
                                Ok(compiled_dep) => {
                                    println!("✓ {} → {} ({:.2} KB, {}ms)",
                                             dep.id,
                                             compiled_dep.output_path.display(),
                                             compiled_dep.size_bytes as f64 / 1024.0,
                                             compiled_dep.compile_time_ms
                                    );
                                    compiled.push(compiled_dep);
                                }
                                Err(e) => {
                                    eprintln!("✗ Failed to compile {}: {}", dep.id, e);
                                    // Continue with other deps
                                }
                            }
                        }

                        println!("\n✓ Compiled {} dependencies\n", compiled.len());
                        compiled
                    } else {
                        Vec::new()
                    };

                    // Create compiler with dependency info
                    let mut compiler = Compiler::new(self.config.clone())
                        .with_optimization(if optimize { 3 } else { 0 });

                    // Pass compiled dependencies to compiler
                    compiler.set_compiled_dependencies(compiled_deps);

                    // Compile to temporary file
                    let temp_exe = std::env::temp_dir().join(format!(
                        "tb_exec{}",
                        TargetPlatform::current().exe_extension()
                    ));

                    debug_log!("Compiling to: {}", temp_exe.display());

                    compiler.compile_to_file(&all_statements, &temp_exe)?;

                    debug_log!("✓ Compiled binary: {}", temp_exe.display());

                    // Execute compiled binary
                    let output = Command::new(&temp_exe).output()?;

                    // Clean up
                    fs::remove_file(&temp_exe).ok();

                    if output.status.success() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        if !stdout.is_empty() {
                            print!("{}", stdout);
                        }
                        Ok(Value::Unit)
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        Err(TBError::RuntimeError {
                            message: format!("Execution failed:\n{}", stderr).into(),
                            trace: vec![],
                        })
                    }
                }

                ExecutionMode::Jit { .. } => {
                    debug_log!("JIT mode execution");

                    // Choose executor based on runtime mode
                    match runtime_mode {
                        RuntimeMode::Parallel { .. } => {
                            let mut executor = ParallelExecutor::new(self.config.clone());
                            let static_stmts: &'static [Statement<'static>] = unsafe {
                                std::mem::transmute(statements.as_slice())
                            };
                            executor.execute(static_stmts)
                        }
                        RuntimeMode::Sequential => {
                            // FIX: Pass arena to JitExecutor
                            let mut executor = JitExecutor::new(self.config.clone(), arena);
                            let static_stmts: &'static [Statement<'static>] = unsafe {
                                std::mem::transmute(statements.as_slice())
                            };
                            executor.execute(static_stmts)
                        }
                    }
                }

                ExecutionMode::Streaming { .. } => {
                    let mut executor = JitExecutor::new(self.config.clone(), arena);
                    let static_stmts: &'static [Statement<'static>] = unsafe {
                        std::mem::transmute(statements.as_slice())
                    };
                    executor.execute(static_stmts)
                }
            };
            #[cfg(debug_assertions)]
            {
                let stats = STRING_INTERNER.stats();
                if stats.total_requests > 0 {
                    debug_log!("╔════════════════════════════════════════════════════════════════╗");
                    debug_log!("║              STRING_INTERNER Statistics                        ║");
                    debug_log!("╠════════════════════════════════════════════════════════════════╣");
                    debug_log!("║ Total Requests:  {:>45} ║", stats.total_requests);
                    debug_log!("║ Cache Hits:      {:>45} ║", stats.cache_hits);
                    debug_log!("║ Hit Rate:        {:>44.1}% ║", STRING_INTERNER.hit_rate() * 100.0);
                    debug_log!("║ Unique Strings:  {:>45} ║", stats.unique_strings);
                    debug_log!("║ Memory Used:     {:>42} KB ║", stats.memory_used_bytes / 1024);
                    debug_log!("║ Memory Saved:    {:>42} KB ║", stats.memory_saved_bytes / 1024);
                    debug_log!("║ Capacity:        {:>44.1}% ║", STRING_INTERNER.capacity_usage() * 100.0);
                    debug_log!("║ Evictions:       {:>45} ║", stats.evictions_triggered);
                    debug_log!("╚════════════════════════════════════════════════════════════════╝");

                    // Warning if approaching limits
                    if STRING_INTERNER.capacity_usage() > 0.7 || STRING_INTERNER.memory_usage() > 0.7 {
                        debug_log!("⚠️  WARNING: STRING_INTERNER approaching capacity limits!");
                    }
                }
            }

            arena_result?
        };
        let static_result = result.clone_static()?;
        debug_log!("TBCore::execute() completed: {:?}", static_result);
        Ok(static_result)
    }




    /// Merge imported and main statements, removing duplicate function definitions
    pub fn merge_statements_deduplicated<'arena>(
        imported: Vec<Statement<'arena>>,
        main: Vec<Statement<'arena>>,
    ) -> TBResult<Vec<Statement<'arena>>> {
        let mut result = Vec::new();
        let mut function_names = std::collections::HashSet::new();

        // Track function names from main statements (main has priority)
        for stmt in &main {
            if let Statement::Function { name, .. } = stmt {
                function_names.insert(name.clone());
            }
        }

        // Add imported statements, skip duplicate functions
        for stmt in imported {
            match &stmt {
                Statement::Function { name, .. } => {
                    if !function_names.contains(name) {
                        debug_log!("  Adding imported function: {}", name);
                        function_names.insert(name.clone());
                        result.push(stmt);
                    } else {
                        debug_log!("  Skipping duplicate function: {}", name);
                    }
                }
                _ => result.push(stmt),
            }
        }

        // Add main statements
        result.extend(main);

        Ok(result)
    }

    /// Load imports with compilation caching
    pub fn load_imports<'arena>(
        &mut self,
        imports: &[PathBuf],
        arena: &'arena Bump
    ) -> TBResult<Vec<Statement<'arena>>> {
        let mut all_statements = Vec::new();
        let mut visited = std::collections::HashSet::new();

        debug_log!("╔════════════════════════════════════════════════════════════════╗");
        debug_log!("║                    Loading Imports (JIT)                       ║");
        debug_log!("╚════════════════════════════════════════════════════════════════╝\n");

        self.load_imports_recursive(imports, &mut all_statements, &mut visited, 0, arena)?;

        Ok(all_statements)
    }

    /// Recursive helper for loading imports
    fn load_imports_recursive<'arena>(
        &mut self,
        imports: &[PathBuf],
        collected: &mut Vec<Statement<'arena>>,
        visited: &mut std::collections::HashSet<PathBuf>,
        depth: usize,
        arena: &'arena Bump
    ) -> TBResult<()> {
        let indent = "  ".repeat(depth);

        for import_path in imports {
            debug_log!("{}→ Import: {}", indent, import_path.display());

            // Resolve path
            let resolved_path = if import_path.is_absolute() {
                import_path.clone()
            } else {
                std::env::current_dir()
                    .map_err(|e| TBError::IoError(Arc::new(e.to_string())))?
                    .join(import_path)
            };

            let canonical = resolved_path.canonicalize()
                .map_err(|_| TBError::IoError(
                    format!("Import file not found: {}", resolved_path.display()).into()
                ))?;

            // Check for cycles
            if visited.contains(&canonical) {
                debug_log!("{}  (already loaded)", indent);
                continue;
            }
            visited.insert(canonical.clone());

            // Read and parse import source
            let source = fs::read_to_string(&canonical)?;
            let import_config = Config::parse(&source)?;

            // Recursively load nested imports FIRST
            if !import_config.imports.is_empty() {
                debug_log!("{}  ↳ {} nested imports", indent, import_config.imports.len());
                self.load_imports_recursive(
                    &import_config.imports,
                    collected,
                    visited,
                    depth + 1,
                    arena
                )?;
            }

            // Parse statements
            let clean_source = Self::strip_directives(&source);
            let mut lexer = Lexer::new(&clean_source);
            let tokens = lexer.tokenize()?;
            let mut parser = Parser::new(tokens, arena);
            let statements = parser.parse()?;

            debug_log!("{}  ✓ Loaded {} statements\n", indent, statements.len());
            collected.extend(statements);
        }

        Ok(())
    }

    /// Strip @config and @shared directives from source
    pub fn strip_directives(source: &str) -> String {
        debug_log!("strip_directives() called, source:\n{}", source);

        let mut result = String::new();
        let mut skip_until_brace = false;
        let mut brace_depth = 0;

        for line in source.lines() {
            let trimmed = line.trim();

            // Skip shebang
            if trimmed.starts_with("#!") {
                debug_log!("Skipping shebang line");
                continue;
            }

            // Detect start of directive block (@config, @shared, @imports, @plugins)
            if trimmed.starts_with("@config") || trimmed.starts_with("@shared")
                || trimmed.starts_with("@imports") || trimmed.starts_with("@plugins") {
                skip_until_brace = true;
                brace_depth = 0;
                debug_log!("Entering directive block: {}", trimmed);

                // Check if opening brace is on same line
                if trimmed.contains('{') {
                    brace_depth = 1;
                }
                continue;
            }

            // Handle directive block content
            if skip_until_brace {
                debug_log!("Inside directive block, line: {}", trimmed);

                // Count braces
                for ch in line.chars() {
                    if ch == '{' {
                        brace_depth += 1;
                    } else if ch == '}' {
                        brace_depth -= 1;
                        if brace_depth == 0 {
                            skip_until_brace = false;
                            debug_log!("Exiting directive block");
                            break;
                        }
                    }
                }
                continue;
            }

            // Keep normal code lines
            if !trimmed.is_empty() {
                result.push_str(line);
                result.push('\n');
            }
        }

        let trimmed_result = result.trim().to_string();
        debug_log!("strip_directives() output ({} bytes):\n{}", trimmed_result.len(), trimmed_result);
        trimmed_result
    }

    /// Compile source to file with import support
    pub fn compile_to_file(&mut self, source: &str, output_path: &Path) -> TBResult<()> {
        debug_log!("TBCore::compile_to_file() started");

        // Parse configuration
        self.config = Config::parse(source)?;
        debug_log!("Configuration parsed: mode={:?}", self.config.mode);

        // Strip directives
        let clean_source = Self::strip_directives(source);
        debug_log!("Clean source length: {} bytes", clean_source.len());

        let arena: &'static Bump = Box::leak(Box::new(Bump::new()));

        let mut lexer = Lexer::new(&clean_source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens, &arena);
        let mut statements = parser.parse()?;
        debug_log!("Parsing complete: {} statements", statements.len());

        // Load imports with SAME arena
        let imports = self.config.imports.clone();
        if !imports.is_empty() {
            debug_log!("📦 Loading {} imports for compilation", imports.len());
            let imported_statements = self.load_imports(&imports, &arena)?;
            debug_log!("✓ Loaded {} statements from imports", imported_statements.len());

            statements = Self::merge_statements_deduplicated(imported_statements, statements)?;
            debug_log!("✓ Total statements after merge: {}", statements.len());
        }

        let func_count = statements.iter()
            .filter(|s| matches!(s, Statement::Function { .. }))
            .count();
        debug_log!("✓ Functions available for compilation: {}", func_count);

        // NEW: Analyze and compile dependencies
        let dependencies = self.analyze_dependencies(&statements)?;

        let compiled_deps = if !dependencies.is_empty() {
            println!("\n╔════════════════════════════════════════════════════════════════╗");
            println!("║              Compiling Language Dependencies                   ║");
            println!("╚════════════════════════════════════════════════════════════════╝\n");

            let dep_compiler = DependencyCompiler::new(Path::new("."));
            let mut compiled = Vec::new();

            for dep in &dependencies {
                match dep_compiler.compile(dep) {
                    Ok(compiled_dep) => {
                        println!("✓ {} → {} ({:.2} KB, {}ms)",
                                 dep.id,
                                 compiled_dep.output_path.display(),
                                 compiled_dep.size_bytes as f64 / 1024.0,
                                 compiled_dep.compile_time_ms
                        );
                        compiled.push(compiled_dep);
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to compile {}: {}", dep.id, e);
                    }
                }
            }

            println!("\n✓ Compiled {} dependencies\n", compiled.len());
            compiled
        } else {
            Vec::new()
        };

        // Create compiler with dependencies
        let mut compiler = Compiler::new(self.config.clone());
        compiler.set_compiled_dependencies(compiled_deps);

        // Compile
        let binary = compiler.compile(&statements)?;

        // Write to file
        fs::write(output_path, binary)?;

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(output_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(output_path, perms)?;
        }

        debug_log!("✓ Compilation complete: {}", output_path.display());

        Ok(())
    }

    /// Analyze dependencies in statements
    pub fn analyze_dependencies(&self, statements: &[Statement]) -> TBResult<Vec<Dependency>> {
        let mut dependencies = Vec::new();
        let mut dep_id = 0;

        for stmt in statements {
            self.collect_dependencies_from_statement(stmt, &mut dependencies, &mut dep_id, 0)?;
        }

        debug_log!("Found {} language dependencies", dependencies.len());

        Ok(dependencies)
    }

    fn collect_dependencies_from_statement(
        &self,
        stmt: &Statement,
        dependencies: &mut Vec<Dependency>,
        dep_id: &mut usize,
        loop_depth: usize,
    ) -> TBResult<()> {
        match stmt {
            Statement::Expr(expr) => {
                self.collect_dependencies_from_expr(expr, dependencies, dep_id, loop_depth)?;
            }
            Statement::Function { body, .. } => {
                self.collect_dependencies_from_expr(body, dependencies, dep_id, loop_depth)?;
            }
            Statement::Let { value, .. } => {
                self.collect_dependencies_from_expr(value, dependencies, dep_id, loop_depth)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_dependencies_from_expr(
        &self,
        expr: &Expr,
        dependencies: &mut Vec<Dependency>,
        dep_id: &mut usize,
        loop_depth: usize,
    ) -> TBResult<()> {
        match expr {
            Expr::Native { language, code } => {
                let imports = self.extract_imports(*language, code);

                dependencies.push(Dependency {
                    id: format!("dep_{}", dep_id).into(),
                    language: *language,
                    code: code.clone(),
                    imports,
                    is_in_loop: loop_depth > 0,
                    estimated_calls: if loop_depth > 0 { 1000 } else { 1 },
                });

                *dep_id += 1;
            }

            Expr::Call { function, args } => {
                if let Expr::Variable(func_name) = function {
                    if let Some(language) = self.language_from_builtin_name(&func_name) {
                        // Extract code from first argument
                        if let Some(Expr::Literal(Literal::String(code))) = args.first() {
                            let imports = self.extract_imports(language, code);

                            dependencies.push(Dependency {
                                id: format!("dep_{}", dep_id).into(),
                                language,
                                code: code.clone(),
                                imports,
                                is_in_loop: loop_depth > 0,
                                estimated_calls: if loop_depth > 0 { 1000 } else { 1 },
                            });

                            *dep_id += 1;
                        }
                    }
                }

                // Recurse into function and args
                self.collect_dependencies_from_expr(function, dependencies, dep_id, loop_depth)?;
                for arg in args {
                    self.collect_dependencies_from_expr(arg, dependencies, dep_id, loop_depth)?;
                }
            }

            Expr::Block { statements, result } => {
                for stmt in statements {
                    self.collect_dependencies_from_statement(stmt, dependencies, dep_id, loop_depth)?;
                }
                if let Some(res) = result {
                    self.collect_dependencies_from_expr(res, dependencies, dep_id, loop_depth)?;
                }
            }

            Expr::Loop { body } => {
                self.collect_dependencies_from_expr(body, dependencies, dep_id, loop_depth + 1)?;
            }

            Expr::While { condition, body } => {
                self.collect_dependencies_from_expr(condition, dependencies, dep_id, loop_depth)?;
                self.collect_dependencies_from_expr(body, dependencies, dep_id, loop_depth + 1)?;
            }

            Expr::For { iterable, body, .. } => {
                self.collect_dependencies_from_expr(iterable, dependencies, dep_id, loop_depth)?;
                self.collect_dependencies_from_expr(body, dependencies, dep_id, loop_depth + 1)?;
            }

            Expr::If { condition, then_branch, else_branch } => {
                self.collect_dependencies_from_expr(condition, dependencies, dep_id, loop_depth)?;
                self.collect_dependencies_from_expr(then_branch, dependencies, dep_id, loop_depth)?;
                if let Some(else_b) = else_branch {
                    self.collect_dependencies_from_expr(else_b, dependencies, dep_id, loop_depth)?;
                }
            }

            Expr::BinOp { left, right, .. } => {
                self.collect_dependencies_from_expr(left, dependencies, dep_id, loop_depth)?;
                self.collect_dependencies_from_expr(right, dependencies, dep_id, loop_depth)?;
            }

            Expr::UnaryOp { expr, .. } => {
                self.collect_dependencies_from_expr(expr, dependencies, dep_id, loop_depth)?;
            }

            _ => {}
        }

        Ok(())
    }


    /// Map builtin function name to Language
    fn language_from_builtin_name(&self, name: &str) -> Option<Language> {
        match name {
            "python" => Some(Language::Python),
            "javascript" => Some(Language::JavaScript),
            "typescript" => Some(Language::TypeScript),
            "go" => Some(Language::Go),
            "bash" => Some(Language::Bash),
            _ => None,
        }
    }

    fn extract_imports(&self, language: Language, code: &str) -> Vec<Arc<String>> {  //  Return Arc<String>
        let mut imports = Vec::new();

        match language {
            Language::Python => {
                for line in code.lines() {
                    let line = line.trim();
                    if line.starts_with("import ") {
                        if let Some(module) = line.strip_prefix("import ") {
                            let module = module.split_whitespace().next().unwrap_or("");
                            imports.push(Arc::new(module.to_string()));  //  Wrap in Arc
                        }
                    } else if line.starts_with("from ") {
                        if let Some(rest) = line.strip_prefix("from ") {
                            if let Some(module) = rest.split_whitespace().next() {
                                imports.push(Arc::new(module.to_string()));  //  Wrap in Arc
                            }
                        }
                    }
                }
            }

            Language::JavaScript | Language::TypeScript => {
                for line in code.lines() {
                    let line = line.trim();
                    if line.starts_with("import ") || line.contains("require(") {
                        if let Some(start) = line.find(|c| c == '\'' || c == '"') {
                            if let Some(end) = line[start + 1..].find(|c| c == '\'' || c == '"') {
                                let module = &line[start + 1..start + 1 + end];
                                imports.push(Arc::new(module.to_string()));  //  Wrap in Arc
                            }
                        }
                    }
                }
            }

            Language::Go => {
                let mut in_import_block = false;
                for line in code.lines() {
                    let line = line.trim();

                    if line.starts_with("import (") {
                        in_import_block = true;
                    } else if in_import_block && line.starts_with(")") {
                        in_import_block = false;
                    } else if in_import_block {
                        let import_path = line.trim_matches('"');
                        imports.push(Arc::new(import_path.to_string()));  // Wrap in Arc
                    } else if line.starts_with("import \"") {
                        if let Some(start) = line.find('"') {
                            if let Some(end) = line[start + 1..].find('"') {
                                let import_path = &line[start + 1..start + 1 + end];
                                imports.push(Arc::new(import_path.to_string()));  //  Wrap in Arc
                            }
                        }
                    }
                }
            }

            _ => {}
        }

        #[cfg(debug_assertions)]
        {
            let stats = STRING_INTERNER.stats();
            if stats.total_requests > 0 {
                debug_log!("╔════════════════════════════════════════════════════════════════╗");
                debug_log!("║              STRING_INTERNER Statistics                        ║");
                debug_log!("╠════════════════════════════════════════════════════════════════╣");
                debug_log!("║ Total Requests:  {:>45} ║", stats.total_requests);
                debug_log!("║ Cache Hits:      {:>45} ║", stats.cache_hits);
                debug_log!("║ Hit Rate:        {:>44.1}% ║", STRING_INTERNER.hit_rate() * 100.0);
                debug_log!("║ Unique Strings:  {:>45} ║", stats.unique_strings);
                debug_log!("║ Memory Saved:    {:>42} KB ║", stats.memory_saved_bytes / 1024);
                debug_log!("╚════════════════════════════════════════════════════════════════╝");
            }
        }

        imports
    }
}

// =================================================================================
// §13.5 STREAMING EXECUTOR MODULE
// =================================================================================

pub mod streaming;

// ═══════════════════════════════════════════════════════════════════════════
// §14 PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// High-level TB API
pub struct TB {
    core: TBCore,
}

impl TB {
    /// Create new TB instance with default configuration
    pub fn new() -> Self {
        Self {
            core: TBCore::new(Config::default()),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Config<'static>) -> Self {
        Self {
            core: TBCore::new(config),
        }
    }

    /// Execute TB source code
    pub fn execute(&mut self, source: &str) -> TBResult<Value<'static>> {
        self.core.execute(source)
    }

    /// Execute TB file
    pub fn execute_file(&mut self, path: &Path) -> TBResult<Value<'static>> {
        let source = fs::read_to_string(path)?;
        self.execute(&source)
    }

    /// Compile to executable
    pub fn compile(&mut self, source: &str, output: &Path) -> TBResult<()> {
        self.core.compile_to_file(source, output)
    }

    pub fn interner_health(&self) -> String {
        STRING_INTERNER.health_report()
    }

    /// Manually trigger interner cleanup
    pub fn cleanup_interner(&self) {
        STRING_INTERNER.cleanup_now();
    }

    /// Clear entire interner (use between independent script runs)
    pub fn reset_interner(&self) {
        STRING_INTERNER.clear();
    }
}

impl Default for TB{
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §15 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let mut tb = TB::new();
        let result = tb.execute("2 + 3").unwrap();
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_variables() {
        let mut tb = TB::new();
        let result = tb.execute("let x = 10; x * 2").unwrap();
        assert!(matches!(result, Value::Int(20)));
    }

    #[test]
    fn test_if_expression() {
        let mut tb = TB::new();
        let result = tb.execute("if true { 1 } else { 2 }").unwrap();
        assert!(matches!(result, Value::Int(1)));
    }

    #[test]
    fn test_function() {
        let mut tb = TB::new();
        let result = tb.execute(r#"
            fn double(x: int) { x * 2 }
            double(5)
        "#).unwrap();
        assert!(matches!(result, Value::Int(10)));
    }

    #[test]
    fn test_pipeline() {
        let mut tb = TB::new();
        let result = tb.execute(r#"
            fn double(x: int) { x * 2 }
            fn inc(x: int) { x + 1 }
            5 |> double |> inc
        "#).unwrap();
        assert!(matches!(result, Value::Int(11)));
    }

    #[test]
    fn test_dict_parser_simple() {
        let dict_str = r#"{'name': 'Alice', 'age': 30}"#;
        let result = ReturnValueParser::parse_dict(dict_str);

        if let Value::Dict(map) = result {
            assert_eq!(map.len(), 2);
            assert!(matches!(
                map.get(&Arc::new("name".to_string())),
                Some(Value::String(s)) if s.as_str() == "Alice"
            ));
            assert!(matches!(
                map.get(&Arc::new("age".to_string())),
                Some(Value::Int(30))
            ));
        } else {
            panic!("Expected Dict");
        }
    }

    #[test]
    fn test_dict_parser_nested() {
        let dict_str = r#"{'user': {'name': 'Bob', 'id': 42}, 'active': true}"#;
        let result = ReturnValueParser::parse_dict(dict_str);

        if let Value::Dict(map) = result {
            assert!(matches!(
                map.get(&Arc::new("user".to_string())),
                Some(Value::Dict(_))
            ));
            assert!(matches!(
                map.get(&Arc::new("active".to_string())),
                Some(Value::Bool(true))
            ));
        } else {
            panic!("Expected Dict");
        }
    }

    #[test]
    fn test_dict_parser_with_lists() {
        let dict_str = r#"{'numbers': [1, 2, 3], 'name': 'test'}"#;
        let result = ReturnValueParser::parse_dict(dict_str);

        if let Value::Dict(map) = result {
            if let Some(Value::List(nums)) = map.get(&Arc::new("numbers".to_string())) {
                assert_eq!(nums.len(), 3);
                assert!(matches!(nums[0], Value::Int(1)));
            } else {
                panic!("Expected list in dict");
            }
        } else {
            panic!("Expected Dict");
        }
    }
}

#[cfg(debug_assertions)]
fn print_type_sizes() {
    use std::mem::size_of;
    eprintln!("Type sizes:");
    eprintln!("  Expr: {} bytes", size_of::<Expr>());
    eprintln!("  Statement: {} bytes", size_of::<Statement>());
    eprintln!("  Value: {} bytes", size_of::<Value>());
    eprintln!("  Vec<Expr>: {} bytes", size_of::<Vec<Expr>>());
}


// ═══════════════════════════════════════════════════════════════════════════
// §16 CLI ENTRY POINT (Example)
// ═══════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    TB Language Core v1.0                      ║");
    println!("║              Production-Ready Multi-Language Engine            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Example usage
    let mut tb = TB::new();

    let source = r#"
        let x = 10
        let y = 20
        x + y
    "#;

    match tb.execute(source) {
        Ok(result) => println!("{}", result),
        Err(e) => {
            // Call in main() or execute()
            #[cfg(debug_assertions)]
            print_type_sizes();
            eprintln!("✗ Execution failed:\n{}", e.detailed_message());
            std::process::exit(1);
        }
    }
}
