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
//! ## Supports
//! - Compiled mode
//! - JIT mode
//! - Streaming mode
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
pub mod dependency_compiler;
pub use dependency_compiler::{DependencyCompiler, Dependency, CompilationStrategy};


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
struct EvictionEvent {
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


/// Cached compilation artifact
#[derive(Debug, Clone)]
pub struct CachedArtifact {
    pub source_path: PathBuf,
    pub source_hash: u64,
    pub binary_path: PathBuf,
    pub statements: Vec<Statement>,
    pub compiled_at: std::time::SystemTime,
}

/// Import cache manager
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

    /// Get or compile an import
    pub fn get_or_compile(
        &mut self,
        import_path: &Path,
        parent_config: &Config,
    ) -> TBResult<Vec<Statement>> {
        let canonical_path = import_path.canonicalize()
            .map_err(|e| TBError::IoError(Arc::new(format!("Import not found: {}", import_path.display()))))?;

        // Read source and compute hash
        let source = fs::read_to_string(&canonical_path)?;
        let source_hash = self.compute_hash(&source);

        // Check cache
        if let Some(cached) = self.artifacts.get(&canonical_path) {
            if cached.source_hash == source_hash {
                debug_log!("✓ Using cached artifact for {}", import_path.display());
                return Ok(cached.statements.clone());
            }
        }

        debug_log!("⚙ Processing import: {}", import_path.display());

        // Parse import config
        let import_config = Config::parse(&source)?;

        // Check if import requires compilation
        if matches!(import_config.mode, ExecutionMode::Compiled { .. }) {
            debug_log!("  → Import is in compiled mode, compiling...");
            return self.compile_and_cache(&canonical_path, &source, source_hash, &import_config);
        }

        // JIT mode: just parse and cache statements
        debug_log!("  → Import is in JIT mode, parsing...");
        let statements = self.parse_import(&source)?;

        self.artifacts.insert(canonical_path, CachedArtifact {
            source_path: import_path.to_path_buf(),
            source_hash,
            binary_path: PathBuf::new(),
            statements: statements.clone(),
            compiled_at: std::time::SystemTime::now(),
        });

        Ok(statements)
    }

    /// Compile import to native library and cache
    fn compile_and_cache(
        &mut self,
        path: &Path,
        source: &str,
        source_hash: u64,
        config: &Config,
    ) -> TBResult<Vec<Statement>> {
        let cache_key = format!("{:x}", source_hash);
        let binary_path = self.cache_dir.join(format!("lib_{}.so", cache_key));

        // Check if binary exists
        if binary_path.exists() {
            debug_log!("  ✓ Using cached binary: {}", binary_path.display());
        } else {
            debug_log!("  ⚙ Compiling to: {}", binary_path.display());

            // Parse statements
            let statements = self.parse_import(source)?;

            // Compile to shared library
            let compiler = Compiler::new(config.clone())
                .with_target(TargetPlatform::current())
                .with_optimization(3);

            compiler.compile_to_library(&statements, &binary_path)?;

            debug_log!("  ✓ Compilation complete");
        }

        // Parse statements for metadata (functions, types)
        let statements = self.parse_import(source)?;

        self.artifacts.insert(path.to_path_buf(), CachedArtifact {
            source_path: path.to_path_buf(),
            source_hash,
            binary_path: binary_path.clone(),
            statements: statements.clone(),
            compiled_at: std::time::SystemTime::now(),
        });

        Ok(statements)
    }

    /// Parse import source
    fn parse_import(&self, source: &str) -> TBResult<Vec<Statement>> {
        let clean_source = TBCore::strip_directives(source);
        let mut lexer = Lexer::new(&clean_source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        parser.parse()
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
pub enum Value {
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
    List(Vec<Value>),

    /// Dictionary
    Dict(HashMap<Arc<String>, Value>),

    /// Optional value
    Option(Option<Box<Value>>),

    /// Result value
    Result(Result<Box<Value>, Box<Value>>),

    /// Function closure
    Function {
        params: Vec<Arc<String>>,
        body: Box<Expr>,
        env: Environment,
    },

    /// Tuple
    Tuple(Vec<Value>),

    /// Native handle (for language interop)
    Native {
        language: Language,
        type_name: Arc<String>,
        handle: NativeHandle,
    },
}

impl Value {
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
}

impl Display for Value {
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
pub enum ScopedValue {
    /// Local to single thread
    Local(Value),

    /// Shared with read-write lock
    Shared(Arc<RwLock<Value>>),

    /// Immutable shared reference
    Immutable(Arc<Value>),
}

impl ScopedValue {
    /// Get value (clones for safety)
    pub fn get(&self) -> TBResult<Value> {
        match self {
            ScopedValue::Local(v) => Ok(v.clone()),
            ScopedValue::Shared(v) => Ok(v.read().unwrap().clone()),
            ScopedValue::Immutable(v) => Ok((**v).clone()),
        }
    }

    /// Set value (fails for Immutable)
    pub fn set(&mut self, value: Value) -> TBResult<()> {
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
    pub fn new(value: Value, scope: VarScope) -> Self {
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

/// Expression - core AST node
#[derive(Debug, Clone)]
pub enum Expr {
    /// Literal value
    Literal(Literal),

    /// Variable reference
    Variable(Arc<String>),

    /// Binary operation
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },

    /// Function call
    Call {
        function: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Method call
    Method {
        object: Box<Expr>,
        method: Arc<String>,
        args: Vec<Expr>,
    },

    /// Index access
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },

    /// Field access
    Field {
        object: Box<Expr>,
        field: Arc<String>,
    },

    /// Block expression
    Block {
        statements: Box<Vec<Statement>>,
        result: Option<Box<Expr>>,
    },

    /// If expression
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },

    /// Match expression
    Match {
        scrutinee: Box<Expr>,
        arms: Box<Vec<MatchArm>>,
    },

    /// Loop expression
    Loop {
        body: Box<Expr>,
    },

    /// While loop
    While {
        condition: Box<Expr>,
        body: Box<Expr>,
    },

    /// For loop
    For {
        variable: Arc<String>,
        iterable: Box<Expr>,
        body: Box<Expr>,
    },

    /// Return expression
    Return(Option<Box<Expr>>),

    /// Break expression
    Break(Option<Box<Expr>>),

    /// Continue expression
    Continue,

    /// Lambda function
    Lambda {
        params: Box<Vec<Parameter>>,
        body: Box<Expr>,
    },

    /// List literal
    List(Vec<Expr>),

    /// Dict literal
    Dict(Vec<(Expr, Expr)>),

    /// Tuple literal
    Tuple(Vec<Expr>),

    /// Pipeline operation
    Pipeline {
        value: Box<Expr>,
        operations: Box<Vec<Expr>>,
    },

    /// Parallel execution -
    Parallel(Box<Vec<Expr>>),

    /// Native code block
    Native {
        language: Language,
        code: Arc<String>,
    },

    /// Try expression (? operator)
    Try(Box<Expr>),
}

/// Statement
#[derive(Debug, Clone)]
pub enum Statement {
    /// Let binding
    Let {
        name: Arc<String>,
        mutable: bool,
        type_annotation: Option<Type>,
        value: Expr,
        scope: VarScope,
    },

    /// Assignment
    Assign {
        target: Expr,
        value: Expr,
    },

    /// Expression statement
    Expr(Expr),

    /// Function definition
    Function {
        name: Arc<String>,
        params: Box<Vec<Parameter>>,
        return_type: Option<Type>,
        body: Expr,
    },

    /// Import statement
    Import {
        module: Arc<String>,
        items: Vec<Arc<String>>,
    },
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Arc<String>,
    pub type_annotation: Option<Type>,
    pub default: Option<Expr>,
}

/// Match arm
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

/// Pattern matching
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Wildcard pattern
    Wildcard,

    /// Literal pattern
    Literal(Literal),

    /// Variable binding
    Variable(Arc<String>),

    /// Tuple pattern
    Tuple(Vec<Pattern>),

    /// List pattern
    List(Vec<Pattern>),

    /// Or pattern
    Or(Vec<Pattern>),
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
    Not, Neg, BitNot,
}

// ═══════════════════════════════════════════════════════════════════════════
// §5 CONFIGURATION SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level configuration (from @config block)
#[derive(Debug, Clone)]
pub struct Config {
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
    pub shared: HashMap<Arc<String>, Value>,

    pub runtime_mode: RuntimeMode,

    /// Tokio runtime configuration (for compiled mode)
    pub tokio_runtime: TokioRuntimeConfig,

    /// Enable networking features (auto-detected or manual)
    pub networking_enabled: bool,

    /// Loaded plugins
    pub plugins: Vec<PluginConfig>,
    pub imports: Vec<PathBuf>,
}

impl Default for Config {
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
            tokio_runtime: TokioRuntimeConfig::Auto,
            networking_enabled: false,
            plugins: Vec::new(),
            imports: Vec::new(),
        }
    }
}

impl Config {
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
                            let key_str = key.trim();
                            let value = value.trim().trim_end_matches(',');

                            // Handle special config keys
                            match key_str {
                                "threads" => {
                                    if let Ok(n) = value.parse::<usize>() {
                                        config.tokio_runtime = TokioRuntimeConfig::Minimal { worker_threads: n };
                                    }
                                }
                                "networking" => {
                                    if value == "true" || value == "\"true\"" {
                                        config.networking_enabled = true;
                                        if matches!(config.tokio_runtime, TokioRuntimeConfig::None) {
                                            config.tokio_runtime = TokioRuntimeConfig::Minimal { worker_threads: 2 };
                                        }
                                    } else if value == "false" || value == "\"false\"" {
                                        config.networking_enabled = false;
                                    } else if value == "\"auto\"" || value == "auto" {
                                        // Will be auto-detected later
                                        config.tokio_runtime = TokioRuntimeConfig::Auto;
                                    }
                                }
                                "runtime" => {
                                    match value.trim_matches('"') {
                                        "none" => config.tokio_runtime = TokioRuntimeConfig::None,
                                        "auto" => config.tokio_runtime = TokioRuntimeConfig::Auto,
                                        "minimal" => config.tokio_runtime = TokioRuntimeConfig::Minimal { worker_threads: 2 },
                                        "full" => config.tokio_runtime = TokioRuntimeConfig::Full { worker_threads: 4 },
                                        _ => {}
                                    }
                                }
                                _ => {
                                    // Store in shared for other config values
                                    let key = STRING_INTERNER.intern(key_str);
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

/// Tokio runtime configuration for compiled mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokioRuntimeConfig {
    /// No Tokio runtime (minimal, fastest startup)
    None,

    /// Automatic detection based on code analysis
    Auto,

    /// Minimal runtime with specified thread count
    Minimal {
        worker_threads: usize,
    },

    /// Full runtime with all features
    Full {
        worker_threads: usize,
    },
}

impl Default for TokioRuntimeConfig {
    fn default() -> Self {
        TokioRuntimeConfig::Auto
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
// §7 PARSER
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

                // Only insert Newline token if we're not nested
                if self.paren_depth == 0 && self.brace_depth == 0 && self.bracket_depth == 0 {
                    if !tokens.is_empty() {
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

            // ✅ NEW: Skip Python-style comments (#)
            if self.current() == '#' {
                debug_log!("Skipping Python-style comment at line {}", self.line);
                self.skip_line();
                continue;
            }

            // ✅ NEW: Skip C-style comments (//)
            if self.current() == '/' && self.peek_ahead(1) == Some('/') {
                debug_log!("Skipping C-style comment at line {}", self.line);
                self.skip_line();
                continue;  // ✅ Don't advance, let main loop handle \n
            }

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

/// Parser - builds AST from tokens
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
    recursion_depth: usize,
    max_recursion_depth: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
            recursion_depth: 0,
            max_recursion_depth: 100,
        }
    }

    pub fn parse(&mut self) -> TBResult<Vec<Statement>> {
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
                    line: 0,
                    column: 0,
                });
            }

            // ✅ CRITICAL FIX: Skip separators BEFORE parse_statement
            while (self.match_token(&Token::Semicolon)
                || self.match_token(&Token::Newline))
                && !self.is_eof()
            {
                debug_log!("Skipping separator at position {}", self.position);
                self.advance();
            }

            // ✅ Check EOF after skipping separators
            if self.is_eof() {
                debug_log!("Reached EOF after skipping separators");
                break;
            }

            let position_before = self.position;

            debug_log!("Parsing statement {}, position: {}, token: {:?}",
               iterations, self.position, self.current());

            let stmt = self.parse_statement()?;

            // Ensure we made progress
            if self.position == position_before && !self.is_eof() {
                return Err(TBError::ParseError {
                    message: Arc::new(format!(
                        "Parser stuck at position {} with token {:?}.",
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

    fn parse_statement(&mut self) -> TBResult<Statement> {
        debug_log!("parse_statement at position {}, token: {:?}", self.position, self.current());

        // Skip any stray semicolons OR newlines
        // while self.match_token(&Token::Semicolon) || self.match_token(&Token::Newline) {
        //     debug_log!("Skipping separator at position {}", self.position);
        //     self.advance();
        //     if self.is_eof() {
        //         return Err(TBError::ParseError {
        //             message: Arc::from("Unexpected end of input".to_string()),
        //             line: 0,
        //             column: 0,
        //         });
        //     }
        // }

        let result = match self.current().clone() {
            Token::Let => {
                debug_log!("Parsing let statement");
                self.parse_let()
            }
            Token::Fn => {
                debug_log!("Parsing function statement");
                self.parse_function()
            }
            // Token::Eof => {
            //     return Err(TBError::ParseError {
            //         message: Arc::from("Unexpected EOF".to_string()),
            //         line: 0,
            //         column: 0,
            //     });
            // }
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

    fn parse_let(&mut self) -> TBResult<Statement> {
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
                message: Arc::from("Expected identifier after 'let'".to_string()),
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
                message: Arc::from("Expected '=' after variable name".to_string()),
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

    fn parse_function(&mut self) -> TBResult<Statement> {
        self.advance(); // consume 'fn'
        let name = if let Token::Identifier(n) = self.current() {
            let name = Arc::clone(n);
            self.advance();
            name
        } else {
            return Err(TBError::ParseError {
                message: Arc::from("Expected function name".to_string()),
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
            params: Box::new(params),
            return_type,
            body,
        })
    }

    fn parse_expression(&mut self) -> TBResult<Expr> {
        self.enter_recursion()?;
        let result = self.parse_pipeline();

        self.exit_recursion();
        result
    }

    fn parse_pipeline(&mut self) -> TBResult<Expr> {
        let mut expr = self.parse_logical_or()?;

        if self.match_token(&Token::Pipe) {
            let mut operations = Vec::new();
            while self.match_token(&Token::Pipe) {
                self.advance();
                operations.push(self.parse_logical_or()?);
            }
            expr = Expr::Pipeline {
                value: Box::new(expr),
                operations: Box::new(operations),
            };
        }

        expr = self.parse_try(expr)?;

        Ok(expr)
    }

    fn parse_try(&mut self, expr: Expr) -> TBResult<Expr> {
        if self.match_token(&Token::Question) {
            self.advance();
            Ok(Expr::Try(Box::new(expr)))
        } else {
            Ok(expr)
        }
    }

    fn parse_logical_or(&mut self) -> TBResult<Expr> {
        let mut left = self.parse_logical_and()?;

        while self.match_token(&Token::Or) {
            self.advance();
            let right = self.parse_logical_and()?;
            left = Expr::BinOp {
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_logical_and(&mut self) -> TBResult<Expr> {
        let mut left = self.parse_equality()?;

        while self.match_token(&Token::And) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::BinOp {
                op: BinOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_equality(&mut self) -> TBResult<Expr> {
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
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> TBResult<Expr> {
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
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> TBResult<Expr> {
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
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> TBResult<Expr> {
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
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> TBResult<Expr> {
        // Check for unary operators FIRST
        match self.current() {
            Token::Not | Token::Minus => {
                let op = match self.current() {
                    Token::Not => UnaryOp::Not,
                    Token::Minus => UnaryOp::Neg,
                    _ => unreachable!(),
                };
                self.advance();
                let expr = Box::new(self.parse_unary()?); // Recursive for chaining
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
                    let mut args = Vec::new();
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
                        function: Box::new(expr),
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
                            let mut args = Vec::new();
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
                                object: Box::new(expr),
                                method: field,
                                args,
                            };
                        } else {
                            expr = Expr::Field {
                                object: Box::new(expr),
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
                        object: Box::new(expr),
                        index: Box::new(index),
                    };
                }

                // Implicit function call for builtin commands
                Token::String(_) | Token::Dollar | Token::Identifier(_)
                if matches!(expr, Expr::Variable(ref name) if self.is_builtin_command(name.as_str())) =>
                    {
                        debug_log!("Detected implicit builtin call for: {}",
                        if let Expr::Variable(ref n) = expr { n } else { "unknown" });

                        let mut args = Vec::new();

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
                            function: Box::new(expr),
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
    fn parse_argument(&mut self) -> TBResult<Expr> {
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

                    let mut args = Vec::new();
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
                        function: Box::new(Expr::Variable(name)),
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
                let mut items = Vec::new();
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

    fn parse_postfix(&mut self) -> TBResult<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.current() {
                Token::LParen => {
                    self.advance();
                    let mut args = Vec::new();
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
                        function: Box::new(expr),
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
                            let mut args = Vec::new();
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
                                object: Box::new(expr),
                                method: field,
                                args,
                            };
                        } else {
                            expr = Expr::Field {
                                object: Box::new(expr),
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
                        object: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> TBResult<Expr> {
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
                let mut items = Vec::new();
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
                    Ok(Expr::Return(Some(Box::new(self.parse_expression()?))))
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
    fn parse_dict_literal(&mut self) -> TBResult<Expr> {
        self.advance(); // consume '{'

        let mut pairs = Vec::new();

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

    fn parse_block(&mut self) -> TBResult<Expr> {
        debug_log!("parse_block() starting at position {}", self.position);
        self.advance(); // consume '{'

        let mut statements = Vec::new();
        let mut result = None;
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

            // Check if this is the last expression (no semicolon)
            let stmt_or_expr = self.parse_statement()?;

            // Ensure we made progress
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
                        result = Some(Box::new(expr));
                    } else {
                        statements.push(Statement::Expr(expr));
                    }
                }
                stmt => statements.push(stmt),
            }
        }

        self.advance(); // consume '}'
        debug_log!("parse_block() completed with {} statements", statements.len());

        Ok(Expr::Block {  statements: Box::new(statements) , result })
    }

    fn parse_if(&mut self) -> TBResult<Expr> {
        self.advance(); // consume 'if'

        let condition = Box::new(self.parse_expression()?);
        let then_branch = Box::new(self.parse_expression()?);

        let else_branch = if self.match_token(&Token::Else) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        Ok(Expr::If {
            condition,
            then_branch,
            else_branch,
        })
    }

    fn parse_match(&mut self) -> TBResult<Expr> {
        self.advance(); // consume 'match'

        let scrutinee = Box::new(self.parse_expression()?);

        if !self.match_token(&Token::LBrace) {
            return Err(TBError::ParseError {
                message: Arc::from("Expected '{' after match scrutinee".to_string()),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let mut arms = Vec::new();
        while !self.match_token(&Token::RBrace) {
            let pattern = self.parse_pattern()?;

            if !self.match_token(&Token::FatArrow) {
                return Err(TBError::ParseError {
                    message: Arc::from("Expected '=>' after pattern".to_string()),
                    line: 0,
                    column: 0,
                });
            }
            self.advance();

            let body = self.parse_expression()?;

            arms.push(MatchArm {
                pattern,
                guard: None,
                body,
            });

            if !self.match_token(&Token::RBrace) {
                if !self.match_token(&Token::Comma) {
                    return Err(TBError::ParseError {
                        message: Arc::from("Expected ',' or '}' after match arm".to_string()),
                        line: 0,
                        column: 0,
                    });
                }
                self.advance();
            }
        }
        self.advance(); // consume '}'

        Ok(Expr::Match { scrutinee, arms: Box::new(arms) })
    }

    fn parse_pattern(&mut self) -> TBResult<Pattern> {
        match self.current() {
            Token::Identifier(name) if name.as_str() == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            Token::Identifier(name) => {
                let name = Arc::clone(name);
                self.advance();
                Ok(Pattern::Variable(name))
            }
            Token::Int(n) => {
                let n = *n;
                self.advance();
                Ok(Pattern::Literal(Literal::Int(n)))
            }
            Token::Bool(b) => {
                let b = *b;
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(b)))
            }
            _ => Err(TBError::ParseError {
                message: Arc::from("Invalid pattern".to_string()),
                line: 0,
                column: 0,
            }),
        }
    }

    fn parse_loop(&mut self) -> TBResult<Expr> {
        self.advance(); // consume 'loop'
        let body = Box::new(self.parse_expression()?);
        Ok(Expr::Loop { body })
    }

    fn parse_while(&mut self) -> TBResult<Expr> {
        self.advance(); // consume 'while'
        let condition = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_expression()?);
        Ok(Expr::While { condition, body })
    }

    fn parse_for(&mut self) -> TBResult<Expr> {
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

        let iterable = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_expression()?);

        Ok(Expr::For {
            variable,
            iterable,
            body,
        })
    }

    fn parse_parallel(&mut self) -> TBResult<Expr> {
        self.advance(); // consume 'parallel'

        if !self.match_token(&Token::LBrace) {
            return Err(TBError::ParseError {
                message: Arc::from("Expected '{' after 'parallel'".to_string()),
                line: 0,
                column: 0,
            });
        }
        self.advance();

        let mut tasks = Vec::new();

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

        Ok(Expr::Parallel(Box::new(tasks)))
    }

    fn parse_type(&mut self) -> TBResult<Type> {
        debug_log!("parse_type() at position {}, token: {:?}", self.position, self.current());

        match self.current() {
            Token::Identifier(name) => {
                let name = Arc::clone(name);
                self.advance();

                match name.as_str() {
                    // ═══════════════════════════════════════════════════════════
                    // PRIMITIVE TYPES
                    // ═══════════════════════════════════════════════════════════
                    "int" => Ok(Type::Int),
                    "float" => Ok(Type::Float),
                    "bool" => Ok(Type::Bool),
                    "string" => Ok(Type::String),
                    "unit" => Ok(Type::Unit),

                    // ═══════════════════════════════════════════════════════════
                    // GENERIC TYPES: list<T>
                    // ═══════════════════════════════════════════════════════════
                    "list" => {
                        if self.match_token(&Token::Lt) {
                            debug_log!("  Parsing generic list type");
                            self.advance(); // consume '<'

                            let inner = Box::new(self.parse_type()?);

                            if !self.match_token(&Token::Gt) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected '>' after list type parameter".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume '>'

                            debug_log!("  ✓ Parsed list<{:?}>", inner);
                            Ok(Type::List(inner))
                        } else {
                            // list without type parameter → list<dynamic>
                            debug_log!("  List without type parameter, using Dynamic");
                            Ok(Type::List(Box::new(Type::Dynamic)))
                        }
                    }

                    // ═══════════════════════════════════════════════════════════
                    // GENERIC TYPES: dict<K, V>
                    // ═══════════════════════════════════════════════════════════
                    "dict" => {
                        if self.match_token(&Token::Lt) {
                            debug_log!("  Parsing generic dict type");
                            self.advance(); // consume '<'

                            let key = Box::new(self.parse_type()?);

                            if !self.match_token(&Token::Comma) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected ',' between dict type parameters".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume ','

                            let value = Box::new(self.parse_type()?);

                            if !self.match_token(&Token::Gt) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected '>' after dict type parameters".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume '>'

                            debug_log!("  ✓ Parsed dict<{:?}, {:?}>", key, value);
                            Ok(Type::Dict { key, value })
                        } else {
                            Ok(Type::Dict {
                                key: Box::new(Type::String),
                                value: Box::new(Type::Dynamic),
                            })
                        }
                    }

                    // ═══════════════════════════════════════════════════════════
                    // GENERIC TYPES: option<T>
                    // ═══════════════════════════════════════════════════════════
                    "option" => {
                        if self.match_token(&Token::Lt) {
                            debug_log!("  Parsing generic option type");
                            self.advance(); // consume '<'

                            let inner = Box::new(self.parse_type()?);

                            if !self.match_token(&Token::Gt) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected '>' after option type parameter".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume '>'

                            debug_log!("  ✓ Parsed option<{:?}>", inner);
                            Ok(Type::Option(inner))
                        } else {
                            Ok(Type::Option(Box::new(Type::Dynamic)))
                        }
                    }

                    // ═══════════════════════════════════════════════════════════
                    // GENERIC TYPES: result<T, E>
                    // ═══════════════════════════════════════════════════════════
                    "result" => {
                        if self.match_token(&Token::Lt) {
                            debug_log!("  Parsing generic result type");
                            self.advance(); // consume '<'

                            let ok = Box::new(self.parse_type()?);

                            if !self.match_token(&Token::Comma) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected ',' between result type parameters".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume ','

                            let err = Box::new(self.parse_type()?);

                            if !self.match_token(&Token::Gt) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected '>' after result type parameters".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume '>'

                            debug_log!("  ✓ Parsed result<{:?}, {:?}>", ok, err);
                            Ok(Type::Result { ok, err })
                        } else {
                            Ok(Type::Result {
                                ok: Box::new(Type::Dynamic),
                                err: Box::new(Type::Dynamic),
                            })
                        }
                    }

                    // ═══════════════════════════════════════════════════════════
                    // TUPLE TYPES: (T1, T2, ...)
                    // ═══════════════════════════════════════════════════════════
                    "tuple" => {
                        if self.match_token(&Token::Lt) {
                            debug_log!("  Parsing generic tuple type");
                            self.advance(); // consume '<'

                            let mut types = vec![self.parse_type()?];

                            while self.match_token(&Token::Comma) {
                                self.advance();
                                types.push(self.parse_type()?);
                            }

                            if !self.match_token(&Token::Gt) {
                                return Err(TBError::ParseError {
                                    message: Arc::from("Expected '>' after tuple type parameters".to_string()),
                                    line: 0,
                                    column: 0,
                                });
                            }
                            self.advance(); // consume '>'

                            debug_log!("  ✓ Parsed tuple with {} elements", types.len());
                            Ok(Type::Tuple(types))
                        } else {
                            Ok(Type::Tuple(vec![]))
                        }
                    }

                    // ═══════════════════════════════════════════════════════════
                    // UNKNOWN TYPE → Dynamic
                    // ═══════════════════════════════════════════════════════════
                    _ => {
                        debug_log!("  Unknown type '{}', using Dynamic", name);
                        Ok(Type::Dynamic)
                    }
                }
            }

            // ═══════════════════════════════════════════════════════════
            // PARENTHESIZED TUPLE TYPES: (int, string)
            // ═══════════════════════════════════════════════════════════
            Token::LParen => {
                debug_log!("  Parsing tuple type with parentheses");
                self.advance(); // consume '('

                let mut types = vec![self.parse_type()?];

                while self.match_token(&Token::Comma) {
                    self.advance();
                    types.push(self.parse_type()?);
                }

                if !self.match_token(&Token::RParen) {
                    return Err(TBError::ParseError {
                        message: Arc::from("Expected ')' after tuple types".to_string()),
                        line: 0,
                        column: 0,
                    });
                }
                self.advance(); // consume ')'

                debug_log!("  ✓ Parsed tuple with {} elements", types.len());
                Ok(Type::Tuple(types))
            }

            // ═══════════════════════════════════════════════════════════
            // FALLBACK: Dynamic
            // ═══════════════════════════════════════════════════════════
            _ => {
                debug_log!("  No type annotation, using Dynamic");
                Ok(Type::Dynamic)
            }
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
                    // ✅ NEW: Wenn User explizit annotiert hat, vertraue ihm!
                    // Er übernimmt Verantwortung für Type Safety zur Laufzeit
                    // (z.B. python() gibt String zurück, aber User parst zu Int)

                    // Nur warnen bei offensichtlichem Konflikt (nicht Dynamic)
                    if value_type != Type::Dynamic && &value_type != expected {
                        // Nur Fehler wenn BEIDE Typen bekannt sind UND unterschiedlich
                        // Erlaube Dynamic → konkret (User weiß was er tut)
                        return Err(TBError::TypeError {
                            expected: expected.clone(),
                            found: value_type,
                            context: format!("let binding '{}'", name).into(),
                        });
                    }

                    // ✅ Use explicit annotation (User knows best)
                    self.env.variables.insert(name.clone(), expected.clone());
                } else {
                    // No annotation, use inferred type
                    self.env.variables.insert(name.clone(), value_type);
                }

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
                        params: vec![Type::Dynamic],
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
                    // TODO: Unify types
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

    pub fn optimize(&self, expr: Expr) -> TBResult<Expr> {
        let mut expr = expr;

        if self.config.const_fold {
            expr = self.constant_folding(expr)?;
        }

        Ok(expr)
    }

    fn constant_folding(&self, expr: Expr) -> TBResult<Expr> {
        match expr {
            Expr::BinOp { op, left, right } => {
                let left = self.constant_folding(*left)?;
                let right = self.constant_folding(*right)?;

                if let (Expr::Literal(l), Expr::Literal(r)) = (&left, &right) {
                    if let Some(result) = self.eval_const_binop(op, l, r) {
                        return Ok(Expr::Literal(result));
                    }
                }

                Ok(Expr::BinOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }

            Expr::UnaryOp { op, expr } => {
                let expr = self.constant_folding(*expr)?;

                if let Expr::Literal(lit) = &expr {
                    if let Some(result) = self.eval_const_unaryop(op, lit) {
                        return Ok(Expr::Literal(result));
                    }
                }

                Ok(Expr::UnaryOp {
                    op,
                    expr: Box::new(expr),
                })
            }

            Expr::If { condition, then_branch, else_branch } => {
                let condition = self.constant_folding(*condition)?;

                if let Expr::Literal(Literal::Bool(b)) = condition {
                    return if b {
                        self.constant_folding(*then_branch)
                    } else if let Some(else_branch) = else_branch {
                        self.constant_folding(*else_branch)
                    } else {
                        Ok(Expr::Literal(Literal::Unit))
                    };
                }

                Ok(Expr::If {
                    condition: Box::new(condition),
                    then_branch: Box::new(self.constant_folding(*then_branch)?),
                    else_branch: else_branch.map(|e| self.constant_folding(*e)).transpose()?.map(Box::new),
                })
            }

            Expr::Block { statements, result } => {
                // TODO: Optimize statements
                Ok(Expr::Block {
                    statements,
                    result: result.map(|e| self.constant_folding(*e)).transpose()?.map(Box::new),
                })
            }

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

pub struct CodeGenerator {
    target_language: Language,
    buffer: String,
    indent: usize,
    variables_in_scope: Vec<(Arc<String>, Type)>,
    compiled_deps: HashMap<Arc<String>, PathBuf>,
    mutable_vars: std::collections::HashSet<Arc<String>>,
}

impl CodeGenerator {
    pub fn new(target_language: Language) -> Self {
        Self {
            target_language,
            buffer: String::new(),
            indent: 0,
            variables_in_scope: Vec::new(),
            compiled_deps: HashMap::new(),
            mutable_vars: std::collections::HashSet::new(),
        }
    }

    /// Set compiled dependencies (called by Compiler)
    pub fn set_compiled_dependencies(&mut self, deps: &[CompiledDependency]) {
        for dep in deps {
            self.compiled_deps.insert(dep.id.clone(), dep.output_path.clone());
        }
    }

    ///  NEW: Set mutable variables info
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
        self.emit_line("#![allow(unused)]");
        self.emit_line("");

        // Core imports
        self.emit_line("use std::process::Command;");
        self.emit_line("use std::io::Write;");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // STEP 1: Analyze what helpers we need
        // ═══════════════════════════════════════════════════════════════
        let mut mutable_vars = std::collections::HashSet::new();
        for stmt in statements {
            if let Statement::Let { name, .. } = stmt {
                if self.is_variable_assigned(name, statements) {
                    mutable_vars.insert(name.clone());
                    debug_log!("Variable '{}' detected as mutable", name);
                }
            }
        }
        self.mutable_vars = mutable_vars;

        let needs_bridges = self.statements_use_language_bridges(statements);

        // ═══════════════════════════════════════════════════════════════
        // STEP 2: Generate Language Bridge Helpers (if needed)
        // ═══════════════════════════════════════════════════════════════
        if needs_bridges {
            self.emit_line("// ═══════════════════════════════════════════════════");
            self.emit_line("// Language Bridge Helper Functions");
            self.emit_line("// ═══════════════════════════════════════════════════");
            self.emit_line("");
            self.generate_language_bridge_helpers();
            self.emit_line("");
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 3: Generate Type Parser Helpers (always needed)
        // ═══════════════════════════════════════════════════════════════
        // Note: These are separate from builtin helpers!
        // They are used internally by the generated code for parsing
        self.generate_type_parser_helpers();
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // STEP 4: Generate Builtin Function Helpers (ONLY ONCE!)
        // ═══════════════════════════════════════════════════════════════
        self.generate_builtin_helpers();
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // STEP 5: Generate user-defined functions
        // ═══════════════════════════════════════════════════════════════
        for stmt in statements {
            if let Statement::Function { .. } = stmt {
                self.generate_rust_statement(stmt)?;
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 6: Generate main function
        // ═══════════════════════════════════════════════════════════════
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
                if let Expr::Variable(name) = function.as_ref() {
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
    fn generate_language_bridge_helpers(&mut self) {
        self.emit_line("// Note: For complex types, ensure serde_json is available");
        self.emit_line("// or results will be returned as strings");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Python executor WITH variable detection
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn python(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let wrapped = wrap_python_auto_return(code);");
        self.emit_line("let output = Command::new(\"python\")");
        self.indent += 1;
        self.emit_line(".arg(\"-c\").arg(&wrapped).output().expect(\"Failed to execute Python\");");
        self.indent -= 1;
        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"Python error: {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("");
        //self.emit_line("print!(\"{}\", stdout);");
        self.emit_line("// ✅ NEW: Split output into display (stdout) and return value (last line)");
        self.emit_line("let lines: Vec<&str> = stdout.lines().collect();");
        self.emit_line("if lines.is_empty() {");
        self.indent += 1;
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Print all lines EXCEPT last (visible output)");
        self.emit_line("for line in &lines[..lines.len().saturating_sub(1)] {");
        self.indent += 1;
        self.emit_line("println!(\"{}\", line);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Return ONLY last NON-EMPTY line as value");
        self.emit_line("lines.iter()");
        self.indent += 1;
        self.emit_line(".filter(|l| !l.trim().is_empty())");
        self.emit_line(".last()");
        self.emit_line(".unwrap_or(&\"\")");
        self.emit_line(".trim()");
        self.emit_line(".to_string()");
        self.indent -= 1;
        self.indent -= 1;
        self.emit_line("}");

        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn wrap_python_auto_return(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let trimmed = code.trim();");
        self.emit_line("let lines: Vec<&str> = trimmed.lines().collect();");
        self.emit_line("if lines.is_empty() { return code.to_string(); }");
        self.emit_line("let last_line = lines.last().unwrap().trim();");
        self.emit_line("");
        self.emit_line("// Skip wrapping if:");
        self.emit_line("// - Empty line");
        self.emit_line("// - Comment");
        self.emit_line("// - Already has print() → IT PRINTS, no return needed!");
        self.emit_line("// - Assignment");
        self.emit_line("// - Block start");
        self.emit_line("if last_line.is_empty()");
        self.indent += 1;
        self.emit_line("|| last_line.starts_with('#')");
        self.emit_line("|| last_line.starts_with(\"print(\")");
        self.emit_line("|| (last_line.contains(\" = \") && !last_line.contains(\"==\"))");
        self.emit_line("|| last_line.ends_with(':')");
        self.indent -= 1;
        self.emit_line("{");
        self.indent += 1;
        self.emit_line("return code.to_string();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Extract defined variables");
        self.emit_line("let defined_vars = extract_python_variables(&lines);");
        self.emit_line("let is_variable = defined_vars.contains(&last_line.to_string());");
        self.emit_line("let is_bare_string = !is_variable && is_bare_string_literal(last_line);");
        self.emit_line("");
        self.emit_line("let mut wrapped = lines.iter().take(lines.len() - 1).map(|&s| s).collect::<Vec<&str>>().join(\"\\n\");");
        self.emit_line("if !wrapped.is_empty() { wrapped.push('\\n'); }");
        self.emit_line("");
        self.emit_line("if is_bare_string {");
        self.indent += 1;
        self.emit_line("wrapped.push_str(&format!(\"__tb_result = (\\\"{}\\\")\\nprint(__tb_result, end='')\", last_line));");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("wrapped.push_str(&format!(\"__tb_result = ({})\\nprint(__tb_result, end='')\", last_line));");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("wrapped");
        self.indent -= 1;
        self.emit_line("}");

        // --- Helper: Extract Python variables ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn extract_python_variables(lines: &[&str]) -> Vec<String> {");
        self.indent += 1;
        self.emit_line("let mut vars = Vec::new();");
        self.emit_line("for line in lines {");
        self.indent += 1;
        self.emit_line("let trimmed = line.trim();");
        self.emit_line("if trimmed.is_empty() || trimmed.starts_with('#') { continue; }");
        self.emit_line("if let Some(eq_pos) = trimmed.find('=') {");
        self.indent += 1;
        self.emit_line("let before = &trimmed[..eq_pos];");
        self.emit_line("if before.ends_with('!') || before.ends_with('<') || before.ends_with('>') || before.ends_with('=') { continue; }");
        self.emit_line("let var_part = before.trim();");
        self.emit_line("if var_part.contains(',') {");
        self.indent += 1;
        self.emit_line("for var in var_part.split(',') {");
        self.indent += 1;
        self.emit_line("let clean_var = var.trim().to_string();");
        self.emit_line("if !clean_var.is_empty() && is_valid_python_identifier(&clean_var) { vars.push(clean_var); }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("let clean_var = var_part.to_string();");
        self.emit_line("if is_valid_python_identifier(&clean_var) { vars.push(clean_var); }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("vars");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // --- Helper: Validate Python identifier ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn is_valid_python_identifier(s: &str) -> bool {");
        self.indent += 1;
        self.emit_line("if s.is_empty() { return false; }");
        self.emit_line("let first = s.chars().next().unwrap();");
        self.emit_line("if !first.is_alphabetic() && first != '_' { return false; }");
        self.emit_line("s.chars().all(|c| c.is_alphanumeric() || c == '_')");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // --- Helper: Check if bare string literal ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn is_bare_string_literal(s: &str) -> bool {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("if (trimmed.starts_with('\"') && trimmed.ends_with('\"')) || (trimmed.starts_with('\\'') && trimmed.ends_with('\\'')) { return false; }");
        self.emit_line("if trimmed.contains('+') || trimmed.contains('-') || trimmed.contains('*') || trimmed.contains('/') || trimmed.contains('(') || trimmed.contains('[') || trimmed.contains('.') || trimmed.chars().any(|c| c.is_numeric()) { return false; }");
        self.emit_line("if matches!(trimmed, \"True\" | \"False\" | \"None\" | \"and\" | \"or\" | \"not\" | \"if\" | \"else\") { return false; }");
        self.emit_line("trimmed.chars().all(|c| c.is_alphabetic() || c == '_')");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // JavaScript executor
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn javascript(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let output = Command::new(\"node\").arg(\"-e\").arg(code).output().expect(\"Failed to execute JavaScript\");");
        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"JavaScript error: {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("");
        self.emit_line("// ✅ FIX: Split output into display (stdout) and return value (last line)");
        self.emit_line("let lines: Vec<&str> = stdout.lines().collect();");
        self.emit_line("if lines.is_empty() {");
        self.indent += 1;
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Print all lines EXCEPT last (visible output)");
        self.emit_line("for line in &lines[..lines.len().saturating_sub(1)] {");
        self.indent += 1;
        self.emit_line("println!(\"{}\", line);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Return ONLY last line (for variable assignment)");
        self.emit_line("lines.last().unwrap_or(&\"\").trim().to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");


        // ═══════════════════════════════════════════════════════════════
        // Go executor WITH CORRECT parseInt PLACEMENT
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn go(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("use std::env;");
        self.emit_line("use std::fs;");
        self.emit_line("let temp_dir = env::temp_dir().join(format!(\"tb_go_{}\", std::process::id()));");
        self.emit_line("fs::create_dir_all(&temp_dir).ok();");

        // ✅ FIX: Split injection and user code FIRST
        self.emit_line("let (injection, user_code) = split_go_injection(code);");
        self.emit_line("let var_names = extract_go_var_names(&injection);");
        self.emit_line("let fixed_user = fix_go_redeclarations(&user_code, &var_names);");
        self.emit_line("let wrapped = wrap_go_auto_return(&fixed_user);");

        // ✅ FIX: Build complete Go file with proper structure
        self.emit_line("let mut full_code = String::from(\"package main\\n\\n\");");

        // Add imports
        self.emit_line("full_code.push_str(\"import (\\n\");");
        self.emit_line("full_code.push_str(\"    \\\"fmt\\\"\\n\");");
        self.emit_line("full_code.push_str(\"    \\\"strconv\\\"\\n\");");
        self.emit_line("full_code.push_str(\"    \\\"strings\\\"\\n\");");
        self.emit_line("full_code.push_str(\")\\n\\n\");");

        // ✅ CRITICAL: Define helper functions BEFORE main()
        self.emit_line("full_code.push_str(\"// Helper functions\\n\");");
        self.emit_line("full_code.push_str(\"func parseInt(s string) int64 {\\n\");");
        self.emit_line("full_code.push_str(\"    i, _ := strconv.ParseInt(strings.TrimSpace(s), 10, 64)\\n\");");
        self.emit_line("full_code.push_str(\"    return i\\n\");");
        self.emit_line("full_code.push_str(\"}\\n\\n\");");

        // Add main function
        self.emit_line("full_code.push_str(\"func main() {\\n\");");

        // Add variable injection
        self.emit_line("if !injection.is_empty() {");
        self.indent += 1;
        self.emit_line("for line in injection.lines() {");
        self.indent += 1;
        self.emit_line("if !line.trim().is_empty() {");
        self.indent += 1;
        self.emit_line("full_code.push_str(&format!(\"    {}\\n\", line));");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("full_code.push_str(\"\\n\");");
        self.indent -= 1;
        self.emit_line("}");

        // Add user code
        self.emit_line("for line in wrapped.lines() {");
        self.indent += 1;
        self.emit_line("if !line.trim().is_empty() {");
        self.indent += 1;
        self.emit_line("full_code.push_str(&format!(\"    {}\\n\", line));");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");

        // Close main function
        self.emit_line("full_code.push_str(\"}\\n\");");

        // Write and execute
        self.emit_line("let go_file = temp_dir.join(\"main.go\");");
        self.emit_line("fs::write(&go_file, &full_code).ok();");
        self.emit_line("let output = Command::new(\"go\").arg(\"run\").arg(&go_file).current_dir(&temp_dir).output().expect(\"Failed to execute Go\");");
        self.emit_line("fs::remove_dir_all(&temp_dir).ok();");

        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"[Go Error] {}\\n[Generated Code]\\n{}\", stderr, full_code);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("");

        // ✅ FIX: Apply the same return value logic as Python and JavaScript
        self.emit_line("// ✅ FIX: Split output into display and return value");
        self.emit_line("let lines: Vec<&str> = stdout.lines().collect();");
        self.emit_line("if lines.is_empty() {");
        self.indent += 1;
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Print all lines EXCEPT last");
        self.emit_line("for line in &lines[..lines.len().saturating_sub(1)] {");
        self.indent += 1;
        self.emit_line("println!(\"{}\", line);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Return ONLY last NON-EMPTY line as value");
        self.emit_line("lines.iter()");
        self.indent += 1;
        self.emit_line(".filter(|l| !l.trim().is_empty())");
        self.emit_line(".last()");
        self.emit_line(".unwrap_or(&\"\")");
        self.emit_line(".trim()");
        self.emit_line(".to_string()");
        self.indent -= 1;

        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // --- Go helper: Split injection ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn split_go_injection(code: &str) -> (String, String) {");
        self.indent += 1;
        self.emit_line("let mut injection = String::new();");
        self.emit_line("let mut user_code = String::new();");
        self.emit_line("let mut in_injection = false;");
        self.emit_line("for line in code.lines() {");
        self.indent += 1;
        self.emit_line("if line.contains(\"TB Variables (auto-injected)\") { in_injection = true; continue; }");
        self.emit_line("if in_injection {");
        self.indent += 1;
        self.emit_line("if line.trim().is_empty() { continue; }");
        self.emit_line("if line.trim().starts_with(\"var \") || line.trim().starts_with(\"_\") || line.trim().contains(\" := \") { injection.push_str(line); injection.push('\\n'); } else { in_injection = false; user_code.push_str(line); user_code.push('\\n'); }");
        self.indent -= 1;
        self.emit_line("} else { user_code.push_str(line); user_code.push('\\n'); }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("(injection, user_code)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // --- Go helper: Extract var names ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn extract_go_var_names(injection: &str) -> Vec<String> {");
        self.indent += 1;
        self.emit_line("injection.lines().filter_map(|line| { if line.trim().starts_with(\"var \") { line.trim().strip_prefix(\"var \").and_then(|r| r.split_whitespace().next().map(String::from)) } else if line.contains(\" := \") { line.split(\" := \").next().map(|s| s.trim().to_string()) } else { None } }).collect()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // --- Go helper: Fix redeclarations ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn fix_go_redeclarations(code: &str, existing_vars: &[String]) -> String {");
        self.indent += 1;
        self.emit_line("code.lines().map(|line| { let mut fixed = line.to_string(); for var in existing_vars { let pattern = format!(\"{} :=\", var); if fixed.contains(&pattern) { fixed = fixed.replace(&pattern, &format!(\"{} =\", var)); } } fixed }).collect::<Vec<_>>().join(\"\\n\")");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // --- Go helper: Wrap auto-return ---
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn wrap_go_auto_return(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("let trimmed = code.trim();");
        self.emit_line("let lines: Vec<&str> = trimmed.lines().collect();");
        self.emit_line("if lines.is_empty() { return code.to_string(); }");
        self.emit_line("let last_line = lines.last().unwrap().trim();");
        self.emit_line("let skip_wrap = last_line.is_empty() || last_line.starts_with(\"//\") || last_line.starts_with(\"if \") || last_line.starts_with(\"for \") || last_line.starts_with(\"var \") || last_line.contains(\" := \") || (last_line.contains(\" = \") && !last_line.contains(\"==\")) || last_line.starts_with(\"fmt.Print\") || last_line == \"}\" || last_line.ends_with('{');");
        self.emit_line("if skip_wrap { return code.to_string(); }");
        self.emit_line("let mut wrapped = lines.iter().take(lines.len() - 1).map(|&s| s).collect::<Vec<&str>>().join(\"\\n\");");
        self.emit_line("if !wrapped.is_empty() { wrapped.push('\\n'); }");
        self.emit_line("wrapped.push_str(&format!(\"__tb_result := ({})\\nfmt.Print(__tb_result)\", last_line));");
        self.emit_line("wrapped");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Bash executor
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn bash(code: &str) -> String {");
        self.indent += 1;
        self.emit_line("// ✅ NEW: Find real bash on Windows");
        self.emit_line("let (shell, flag) = if cfg!(windows) {");
        self.indent += 1;
        self.emit_line("// Try to find bash.exe (Git Bash, WSL, MSYS2)");
        self.emit_line("let bash_candidates = vec![");
        self.indent += 1;
        self.emit_line("\"bash\",  // In PATH");
        self.emit_line("\"C:\\\\Program Files\\\\Git\\\\bin\\\\bash.exe\",");
        self.emit_line("\"C:\\\\msys64\\\\usr\\\\bin\\\\bash.exe\",");
        self.emit_line("\"C:\\\\cygwin64\\\\bin\\\\bash.exe\",");
        self.indent -= 1;
        self.emit_line("];");
        self.emit_line("");
        self.emit_line("let mut found_bash = None;");
        self.emit_line("for candidate in &bash_candidates {");
        self.indent += 1;
        self.emit_line("if Command::new(candidate).arg(\"--version\").output().is_ok() {");
        self.indent += 1;
        self.emit_line("found_bash = Some(*candidate);");
        self.emit_line("break;");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("if let Some(bash_path) = found_bash {");
        self.indent += 1;
        self.emit_line("(bash_path, \"-c\")");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("eprintln!(\"Warning: Bash not found on Windows, using cmd (limited compatibility)\");");
        self.emit_line("(\"cmd\", \"/C\")");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("(\"bash\", \"-c\")");
        self.indent -= 1;
        self.emit_line("};");
        self.emit_line("");
        self.emit_line("let output = Command::new(shell).arg(flag).arg(code).output().expect(\"Failed to execute Bash\");");
        self.emit_line("let stdout = String::from_utf8_lossy(&output.stdout);");
        self.emit_line("let stderr = String::from_utf8_lossy(&output.stderr);");
        self.emit_line("if !output.status.success() {");
        self.indent += 1;
        self.emit_line("eprintln!(\"Bash error: {}\", stderr);");
        self.emit_line("return String::new();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("if !stderr.is_empty() { eprint!(\"{}\", stderr); }");
        self.emit_line("print!(\"{}\", stdout);");
        self.emit_line("stdout.trim().to_string()");
        self.indent -= 1;
        self.emit_line("}");
    }

    /// Generate helper functions for parsing different types
    fn generate_type_parser_helpers(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("// Type Parsing Helpers");
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("");

        // Parse to i64
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_i64(s: &str) -> i64 { s.trim().parse::<i64>().or_else(|_| s.trim().parse::<f64>().map(|f| f as i64)).unwrap_or(0) }");
        self.emit_line("");

        // Parse to f64
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_f64(s: &str) -> f64 { s.trim().parse::<f64>().unwrap_or(0.0) }");
        self.emit_line("");

        // Parse to bool
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_bool(s: &str) -> bool { matches!(s.trim().to_lowercase().as_str(), \"true\" | \"True\" | \"1\" | \"yes\" | \"on\") }");
        self.emit_line("");

        // Parse to Vec<i64>
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_vec_i64(s: &str) -> Vec<i64> {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("let inner = &trimmed[1..trimmed.len()-1];");
        self.emit_line("inner.split(',').map(|item| parse_i64(item.trim())).collect()");
        self.indent -= 1;
        self.emit_line("} else { Vec::new() }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Parse to Vec<f64>
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_vec_f64(s: &str) -> Vec<f64> {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("let inner = &trimmed[1..trimmed.len()-1];");
        self.emit_line("inner.split(',').map(|item| parse_f64(item.trim())).collect()");
        self.indent -= 1;
        self.emit_line("} else { Vec::new() }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // Parse to Vec<String>
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_vec_string(s: &str) -> Vec<String> {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("let inner = &trimmed[1..trimmed.len()-1];");
        self.emit_line("inner.split(',').map(|item| item.trim().trim_matches('\"').trim_matches('\\'').to_string()).collect()");
        self.indent -= 1;
        self.emit_line("} else { Vec::new() }");
        self.indent -= 1;
        self.emit_line("}");
    }

    fn generate_builtin_helpers(&mut self) {
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("// Builtin Functions");
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // SMART MATH OPERATIONS - Polymorphic for all types
        // Supports: i64, f64, String, &str with automatic type conversion
        // ═══════════════════════════════════════════════════════════════

        // ───────────────────────────────────────────────────────────────
        // Trait Definitions
        // ───────────────────────────────────────────────────────────────
        self.emit_line("// Smart arithmetic traits for polymorphic operations");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("trait SmartAdd<Rhs = Self> {");
        self.indent += 1;
        self.emit_line("type Output;");
        self.emit_line("fn smart_add(&self, other: &Rhs) -> Self::Output;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[allow(dead_code)]");
        self.emit_line("trait SmartSub<Rhs = Self> {");
        self.indent += 1;
        self.emit_line("type Output;");
        self.emit_line("fn smart_sub(&self, other: &Rhs) -> Self::Output;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[allow(dead_code)]");
        self.emit_line("trait SmartMul<Rhs = Self> {");
        self.indent += 1;
        self.emit_line("type Output;");
        self.emit_line("fn smart_mul(&self, other: &Rhs) -> Self::Output;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[allow(dead_code)]");
        self.emit_line("trait SmartDiv<Rhs = Self> {");
        self.indent += 1;
        self.emit_line("type Output;");
        self.emit_line("fn smart_div(&self, other: &Rhs) -> Self::Output;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // i64 Implementations
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// i64 operations");
        self.emit_line("impl SmartAdd for i64 {");
        self.indent += 1;
        self.emit_line("type Output = i64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_add(&self, other: &i64) -> i64 { self + other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartSub for i64 {");
        self.indent += 1;
        self.emit_line("type Output = i64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_sub(&self, other: &i64) -> i64 { self - other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartMul for i64 {");
        self.indent += 1;
        self.emit_line("type Output = i64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_mul(&self, other: &i64) -> i64 { self * other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartDiv for i64 {");
        self.indent += 1;
        self.emit_line("type Output = i64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_div(&self, other: &i64) -> i64 {");
        self.indent += 1;
        self.emit_line("if *other == 0 { 0 } else { self / other }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // f64 Implementations
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// f64 operations");
        self.emit_line("impl SmartAdd for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_add(&self, other: &f64) -> f64 { self + other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartSub for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_sub(&self, other: &f64) -> f64 { self - other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartMul for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_mul(&self, other: &f64) -> f64 { self * other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartDiv for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_div(&self, other: &f64) -> f64 {");
        self.indent += 1;
        self.emit_line("if *other == 0.0 { 0.0 } else { self / other }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // String Implementations (with numeric parsing)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// String operations (with smart numeric detection)");
        self.emit_line("impl SmartAdd for String {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_add(&self, other: &String) -> String {");
        self.indent += 1;
        self.emit_line("if let (Ok(a), Ok(b)) = (self.trim().parse::<f64>(), other.trim().parse::<f64>()) {");
        self.indent += 1;
        self.emit_line("let result = a + b;");
        self.emit_line("if result.fract() == 0.0 { (result as i64).to_string() } else { result.to_string() }");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("format!(\"{}{}\", self, other)");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartSub for String {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_sub(&self, other: &String) -> String {");
        self.indent += 1;
        self.emit_line("if let (Ok(a), Ok(b)) = (self.trim().parse::<f64>(), other.trim().parse::<f64>()) {");
        self.indent += 1;
        self.emit_line("let result = a - b;");
        self.emit_line("if result.fract() == 0.0 { (result as i64).to_string() } else { result.to_string() }");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("\"0\".to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartMul for String {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_mul(&self, other: &String) -> String {");
        self.indent += 1;
        self.emit_line("if let (Ok(a), Ok(b)) = (self.trim().parse::<f64>(), other.trim().parse::<f64>()) {");
        self.indent += 1;
        self.emit_line("let result = a * b;");
        self.emit_line("if result.fract() == 0.0 { (result as i64).to_string() } else { result.to_string() }");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("\"0\".to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartDiv for String {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_div(&self, other: &String) -> String {");
        self.indent += 1;
        self.emit_line("if let (Ok(a), Ok(b)) = (self.trim().parse::<f64>(), other.trim().parse::<f64>()) {");
        self.indent += 1;
        self.emit_line("if b == 0.0 { return \"0\".to_string(); }");
        self.emit_line("let result = a / b;");
        self.emit_line("if result.fract() == 0.0 { (result as i64).to_string() } else { result.to_string() }");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("\"0\".to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // &str Implementations (delegate to String)
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// &str operations (delegate to String)");
        self.emit_line("impl SmartAdd for &str {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_add(&self, other: &&str) -> String {");
        self.indent += 1;
        self.emit_line("self.to_string().smart_add(&other.to_string())");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartSub for &str {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_sub(&self, other: &&str) -> String {");
        self.indent += 1;
        self.emit_line("self.to_string().smart_sub(&other.to_string())");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartMul for &str {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_mul(&self, other: &&str) -> String {");
        self.indent += 1;
        self.emit_line("self.to_string().smart_mul(&other.to_string())");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartDiv for &str {");
        self.indent += 1;
        self.emit_line("type Output = String;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_div(&self, other: &&str) -> String {");
        self.indent += 1;
        self.emit_line("self.to_string().smart_div(&other.to_string())");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Cross-type implementations: i64 ↔ f64
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// Cross-type operations: i64 <-> f64");
        self.emit_line("impl SmartAdd<f64> for i64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_add(&self, other: &f64) -> f64 { (*self as f64) + other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartAdd<i64> for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_add(&self, other: &i64) -> f64 { self + (*other as f64) }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartSub<f64> for i64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_sub(&self, other: &f64) -> f64 { (*self as f64) - other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartSub<i64> for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_sub(&self, other: &i64) -> f64 { self - (*other as f64) }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartMul<f64> for i64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_mul(&self, other: &f64) -> f64 { (*self as f64) * other }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartMul<i64> for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_mul(&self, other: &i64) -> f64 { self * (*other as f64) }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartDiv<f64> for i64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_div(&self, other: &f64) -> f64 {");
        self.indent += 1;
        self.emit_line("if *other == 0.0 { 0.0 } else { (*self as f64) / other }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl SmartDiv<i64> for f64 {");
        self.indent += 1;
        self.emit_line("type Output = f64;");
        self.emit_line("#[inline(always)]");
        self.emit_line("fn smart_div(&self, other: &i64) -> f64 {");
        self.indent += 1;
        self.emit_line("if *other == 0 { 0.0 } else { self / (*other as f64) }");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // Helper functions for convenient calling
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// Helper functions for ergonomic usage");
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn smart_add<T, U>(left: &T, right: &U) -> T::Output");
        self.emit_line("where T: SmartAdd<U> {");
        self.indent += 1;
        self.emit_line("left.smart_add(right)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn smart_sub<T, U>(left: &T, right: &U) -> T::Output");
        self.emit_line("where T: SmartSub<U> {");
        self.indent += 1;
        self.emit_line("left.smart_sub(right)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn smart_mul<T, U>(left: &T, right: &U) -> T::Output");
        self.emit_line("where T: SmartMul<U> {");
        self.indent += 1;
        self.emit_line("left.smart_mul(right)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn smart_div<T, U>(left: &T, right: &U) -> T::Output");
        self.emit_line("where T: SmartDiv<U> {");
        self.indent += 1;
        self.emit_line("left.smart_div(right)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // TYPE_OF - Trait-based polymorphic implementation
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("trait TypeName {");
        self.indent += 1;
        self.emit_line("fn type_name(&self) -> &'static str;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl TypeName for i64 { fn type_name(&self) -> &'static str { \"int\" } }");
        self.emit_line("impl TypeName for f64 { fn type_name(&self) -> &'static str { \"float\" } }");
        self.emit_line("impl TypeName for bool { fn type_name(&self) -> &'static str { \"bool\" } }");
        self.emit_line("impl TypeName for String { fn type_name(&self) -> &'static str { \"string\" } }");
        self.emit_line("impl TypeName for &str { fn type_name(&self) -> &'static str { \"string\" } }");

        // ✅ Specific Vec implementations with proper type names
        self.emit_line("impl TypeName for Vec<i64> { fn type_name(&self) -> &'static str { \"list<int>\" } }");
        self.emit_line("impl TypeName for Vec<f64> { fn type_name(&self) -> &'static str { \"list<float>\" } }");
        self.emit_line("impl TypeName for Vec<String> { fn type_name(&self) -> &'static str { \"list<string>\" } }");
        self.emit_line("impl TypeName for Vec<bool> { fn type_name(&self) -> &'static str { \"list<bool>\" } }");
        self.emit_line("impl TypeName for Vec<&str> { fn type_name(&self) -> &'static str { \"list<string>\" } }");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn type_of<T: TypeName>(val: &T) -> &'static str {");
        self.indent += 1;
        self.emit_line("val.type_name()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // LEN - Trait-based polymorphic implementation
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("trait Len {");
        self.indent += 1;
        self.emit_line("fn len_tb(&self) -> usize;");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("impl Len for String { fn len_tb(&self) -> usize { self.len() } }");
        self.emit_line("impl Len for &str { fn len_tb(&self) -> usize { self.len() } }");
        self.emit_line("impl Len for Vec<i64> { fn len_tb(&self) -> usize { self.len() } }");
        self.emit_line("impl Len for Vec<f64> { fn len_tb(&self) -> usize { self.len() } }");
        self.emit_line("impl Len for Vec<String> { fn len_tb(&self) -> usize { self.len() } }");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn len<T: Len>(val: &T) -> usize {");
        self.indent += 1;
        self.emit_line("val.len_tb()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // STR - Convert to string
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn str<T: std::fmt::Display>(val: T) -> String {");
        self.indent += 1;
        self.emit_line("format!(\"{}\", val)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // INT - Convert to i64
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn int_from_str(s: &str) -> i64 {");
        self.indent += 1;
        self.emit_line("s.trim().parse::<i64>().unwrap_or(0)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn int_from_f64(f: f64) -> i64 {");
        self.indent += 1;
        self.emit_line("f as i64");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn int_from_bool(b: bool) -> i64 {");
        self.indent += 1;
        self.emit_line("if b { 1 } else { 0 }");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // FLOAT - Convert to f64
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn float_from_str(s: &str) -> f64 {");
        self.indent += 1;
        self.emit_line("s.trim().parse::<f64>().unwrap_or(0.0)");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn float_from_i64(i: i64) -> f64 {");
        self.indent += 1;
        self.emit_line("i as f64");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // PRETTY - Pretty print with indentation
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn pretty<T: std::fmt::Debug>(val: &T) {");
        self.indent += 1;
        self.emit_line("println!(\"[PRETTY] {:#?}\", val);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // PYTHON_INFO - Show Python environment
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn python_info() {");
        self.indent += 1;
        self.emit_line("use std::process::Command;");
        self.emit_line("");
        self.emit_line("let version = Command::new(\"python\")");
        self.indent += 1;
        self.emit_line(".arg(\"--version\")");
        self.emit_line(".output()");
        self.emit_line(".ok();");
        self.indent -= 1;
        self.emit_line("");
        self.emit_line("if let Some(out) = version {");
        self.indent += 1;
        self.emit_line("let ver = String::from_utf8_lossy(&out.stdout);");
        self.emit_line("println!(\"╔════════════════════════════════════════════════════════════════╗\");");
        self.emit_line("println!(\"║                Python Environment Info                         ║\");");
        self.emit_line("println!(\"╠════════════════════════════════════════════════════════════════╣\");");
        self.emit_line("println!(\"║ Version: {:<51} ║\", ver.trim());");
        self.emit_line("println!(\"╚════════════════════════════════════════════════════════════════╝\");");
        self.indent -= 1;
        self.emit_line("} else {");
        self.indent += 1;
        self.emit_line("eprintln!(\"Python not found\");");
        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // DEBUG - Debug print
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("#[inline(always)]");
        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn debug<T: std::fmt::Debug>(val: &T) {");
        self.indent += 1;
        self.emit_line("eprintln!(\"[DEBUG] {:?}\", val);");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        // ═══════════════════════════════════════════════════════════════
        // DYNAMIC TYPE PARSER - Detect type from string
        // ═══════════════════════════════════════════════════════════════
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("// Dynamic Type Parser");
        self.emit_line("// ═══════════════════════════════════════════════════");
        self.emit_line("");

        self.emit_line("#[allow(dead_code)]");
        self.emit_line("fn parse_dynamic_type(s: &str) -> String {");
        self.indent += 1;
        self.emit_line("let trimmed = s.trim();");
        self.emit_line("");
        self.emit_line("// Empty string");
        self.emit_line("if trimmed.is_empty() { return String::new(); }");
        self.emit_line("");
        self.emit_line("// Try parse as list");
        self.emit_line("if trimmed.starts_with('[') && trimmed.ends_with(']') {");
        self.indent += 1;
        self.emit_line("return trimmed.to_string();  // Keep as string representation");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Try parse as dict");
        self.emit_line("if trimmed.starts_with('{') && trimmed.ends_with('}') {");
        self.indent += 1;
        self.emit_line("return trimmed.to_string();");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
        self.emit_line("// Default: return as-is");
        self.emit_line("trimmed.to_string()");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
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

                let is_bridge = self.is_language_bridge_call(value);

                if is_bridge {

                    if type_annotation.is_none() {
                        self.emit(&format!("{} {} = ", keyword, name));
                        self.generate_rust_expr(value)?;
                        self.emit_line(";");
                    } else {
                        let actual_type = type_annotation.as_ref().unwrap().clone();

                        self.emit(&format!("let {}_str = ", name));
                        self.generate_rust_expr(value)?;
                        self.emit_line(";");

                        if actual_type == Type::String {
                            self.emit(&format!("let {} = {}_str;", name, name));
                            self.emit_line("");
                        } else {
                            self.generate_typed_parser(name, &actual_type);
                        }
                    }

                   } else {
                    // Regular assignment
                    self.emit(&format!("{} {} = ", keyword, name));
                    self.generate_rust_expr(value)?;
                    self.emit_line(";");

                }
                // ✅ Infer type from value if no annotation
                let var_type = type_annotation.clone().unwrap_or_else(|| {
                    self.infer_type_from_expr(value)
                });

                // ✅ Add to scope
                self.variables_in_scope.push((name.clone(), var_type.clone()));
                // ✅ CRITICAL: Generate _formatted version IMMEDIATELY
                self.generate_single_formatted_var(name, &var_type);
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

    /// Generate formatted version of a single variable (for language bridge injection)
    fn generate_single_formatted_var(&mut self, name: &Arc<String>, ty: &Type) {
        self.emit_line(&format!("let {}_formatted =", name));
        self.indent += 1;

        match ty {
            Type::List(inner) => {
                match inner.as_ref() {
                    Type::Int | Type::Float | Type::Bool => {
                        // [1, 2, 3]
                        self.emit_line(&format!(
                            "format!(\"[{{}}]\", {}.iter().map(|x| format!(\"{{}}\", x)).collect::<Vec<_>>().join(\", \"));",
                            name
                        ));
                    }
                    Type::String => {
                        // ["a", "b", "c"]
                        self.emit_line(&format!(
                            "format!(\"[{{}}]\", {}.iter().map(|x| format!(r#\"\\\"{{}}\\\"\"#, x)).collect::<Vec<_>>().join(\", \"));",
                            name
                        ));
                    }
                    _ => {
                        self.emit_line(&format!("format!(\"{{:?}}\", {});", name));
                    }
                }
            }

            Type::Dict { .. } | Type::Tuple(_) => {
                self.emit_line(&format!("format!(\"{{:?}}\", {});", name));
            }

            Type::Bool => {
                self.emit_line(&format!(
                    "if {} {{ \"true\".to_string() }} else {{ \"false\".to_string() }};",
                    name
                ));
            }

            // ═══════════════════════════════════════════════════════════
            // STRING - NO QUOTES HERE! Quotes added in injection template
            // ═══════════════════════════════════════════════════════════
            Type::String => {
                // ✅ NEW: Escape for multi-line strings
                self.emit_line(&format!(
                    "format!(\"{{}}\", {}.replace('\\n', \"\\\\n\").replace('\\r', \"\\\\r\").replace('\\t', \"\\\\t\"));",
                    name
                ));
            }

            Type::Int | Type::Float => {
                self.emit_line(&format!("format!(\"{{}}\", {});", name));
            }

            _ => {
                self.emit_line(&format!("format!(\"{{:?}}\", {});", name));
            }
        }

        self.indent -= 1;
    }

    /// Generate type-specific parser
    fn generate_typed_parser(&mut self, name: &str, target_type: &Type) {
        debug_log!("Generating parser for '{}' -> {:?}", name, target_type);

        match target_type {
            Type::Int => {
                // ✅ Enhanced: Parse "42", "42.0", "42\n", " 42 ", "" (empty)
                self.emit(&format!("let {} = {{", name));
                self.emit_line("");
                self.indent += 1;

                // ✅ NEW: Check for empty string first
                self.emit_line(&format!("let trimmed = {}_str.trim();", name));
                self.emit_line("if trimmed.is_empty() {");
                self.indent += 1;
                self.emit_line(&format!("eprintln!(\"Warning: Empty string for '{}', using default 0\");", name));
                self.emit_line("0");
                self.indent -= 1;
                self.emit_line("}");

                // Try direct i64 parse
                self.emit_line("else if let Ok(i) = trimmed.parse::<i64>() {");
                self.indent += 1;
                self.emit_line("i");
                self.indent -= 1;
                self.emit_line("}");

                // Fallback: Try f64 parse and cast to i64
                self.emit_line("else if let Ok(f) = trimmed.parse::<f64>() {");
                self.indent += 1;
                self.emit_line("f as i64");
                self.indent -= 1;
                self.emit_line("}");

                // Last resort: 0
                self.emit_line("else {");
                self.indent += 1;
                self.emit_line(&format!("eprintln!(\"Failed to parse '{{}}' as int: not a number\", trimmed);"));
                self.emit_line("0");
                self.indent -= 1;
                self.emit_line("}");

                self.indent -= 1;
                self.emit_line("};");
            }

            Type::Float => {
                self.emit(&format!("let {} = {}_str.trim().parse::<f64>().unwrap_or_else(|e| {{", name, name));
                self.emit_line("");
                self.indent += 1;
                self.emit_line(&format!("eprintln!(\"Failed to parse '{{}}' as float: {{}}\", {}_str, e);", name));
                self.emit_line("0.0");
                self.indent -= 1;
                self.emit_line("});");
            }

            Type::Bool => {
                // ✅ Enhanced: "True"/"true"/"1" -> true
                self.emit(&format!("let {} = match {}_str.trim().to_lowercase().as_str() {{", name, name));
                self.emit_line("");
                self.indent += 1;
                self.emit_line("\"true\" | \"1\" | \"yes\" => true,");
                self.emit_line("_ => false,");
                self.indent -= 1;
                self.emit_line("};");
            }

            Type::String => {
                // ✅ Just trim
                self.emit(&format!("let {} = {}_str.trim().to_string();", name, name));
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
                        self.emit(&format!("let {} = {}_str.trim().to_string();", name, name));
                        self.emit_line("");
                    }
                }
            }

            _ => {
                self.emit(&format!("let {} = {}_str.trim().to_string();", name, name));
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

            // ✅ LIST - Infer element type from first element
            Expr::List(items) => {
                if items.is_empty() {
                    Type::List(Box::new(Type::Dynamic))
                } else {
                    let first_type = self.infer_type_from_expr(&items[0]);
                    Type::List(Box::new(first_type))
                }
            }

            // ✅ DICT
            Expr::Dict(_) => Type::Dict {
                key: Box::new(Type::String),
                value: Box::new(Type::Dynamic),
            },

            // ✅ TUPLE
            Expr::Tuple(items) => {
                let types: Vec<Type> = items.iter()
                    .map(|e| self.infer_type_from_expr(e))
                    .collect();
                Type::Tuple(types)
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
                if let Expr::Variable(name) = function.as_ref() {
                    if matches!(name.as_str(), "python" | "javascript" | "go" | "bash") {
                        return Type::String;
                    }
                }
                Type::Dynamic
            }

            _ => Type::Dynamic,
        }
    }


    /// Check if expression is a language bridge call
    fn is_language_bridge_call(&self, expr: &Expr) -> bool {
        if let Expr::Call { function, .. } = expr {
            if let Expr::Variable(name) = function.as_ref() {
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
                if let Expr::Variable(name) = function.as_ref() {
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

    // Generate language bridge call with automatic variable injection
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
        let var_injection = self.generate_var_injection_for_language(language);

        if var_injection.is_empty() {
            self.emit(language);
            self.emit("(");
            self.generate_rust_expr(code_expr)?;
            self.emit(")");
            return Ok(());
        }

        // ✅ Use normal strings for Go, raw for others
        if language == "go" || language == "javascript" {
            self.emit(&format!("{}(&format!(\"", language));

            // Escape für normale Strings
            let escaped_injection = var_injection
                .replace('\\', "\\\\")
                .replace('\n', "\\n")
                .replace('"', "\\\"");

            self.emit(&escaped_injection);
            self.emit("\\n{}");
            self.emit("\"");
        } else {
            // Python und Bash nutzen raw strings
            self.emit(&format!("{}(&format!(r#\"", language));
            self.emit(&var_injection);
            self.emit("\n{}");
            self.emit("\"#");
        }

        // ✅ FIX: Pre-format ALL variables to strings
        let var_data: Vec<(Arc<String>, Type)> = self.variables_in_scope.iter()
            .map(|(name, ty)| (Arc::clone(name), ty.clone()))
            .collect();

        // Insert formatted variables
        for (name, ty) in var_data {
            self.emit(", ");

            match (language, ty) {
                ("go", Type::List(_)) => {
                    // For Go slices, provide just the comma-separated values,
                    // without the brackets or extra braces.
                    self.emit(&format!(
                        "{}_formatted.trim_start_matches('[').trim_end_matches(']')",
                        name
                    ));
                }
                ("bash", Type::List(_)) => {
                    self.emit(&format!(
                        "{}_formatted.trim_start_matches('[').trim_end_matches(']').replace(\", \", \" \")",
                        name
                    ));
                }
                _ => {
                    self.emit(&format!("{}_formatted", name));
                }
            }
        }

        self.emit(", ");
        self.generate_rust_expr(code_expr)?;
        self.emit("))");

        Ok(())
    }
    fn generate_var_injection_for_language(&self, language: &str) -> String {
        if self.variables_in_scope.is_empty() {
            return String::new();
        }

        let mut injection = String::new();

        match language {
            "python" => {
                injection.push_str("# TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    match ty {
                        Type::String => {
                            injection.push_str(&format!("{} = \"{{}}\"\n", name.as_str()));
                        }
                        Type::List(_) => {
                            injection.push_str(&format!("{} = {}\n", name.as_str(), "{}"));
                        }
                        Type::Bool => {
                            // Python needs capitalized booleans. The formatted string is "true" or "false".
                            // The expression `'{}' == 'true'` correctly evaluates to True or False in Python.
                            injection.push_str(&format!("{} = '{{}}' == 'true'\n", name.as_str()));
                        }
                        _ => { // Int, Float
                            injection.push_str(&format!("{} = {}\n", name.as_str(), "{}"));
                        }
                    }
                }
            }

            "javascript" => {
                injection.push_str("// TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    match ty {
                        Type::String => {
                            injection.push_str(&format!("const {} = \"{{}}\";\n", name.as_str()));
                        }
                        // The `_formatted` variable already contains "true" or "false",
                        // which are valid JavaScript literals.
                        _ => {
                            injection.push_str(&format!("const {} = {{}};\n", name.as_str()));
                        }
                    }
                }
            }

            "go" => {
                injection.push_str("// TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    match ty {
                        Type::String => {
                            injection.push_str(&format!(
                                "{} := `{{}}`\n_ = {}\n",
                                name.as_str(),
                                name.as_str()
                            ));
                        }
                        Type::Bool => {
                            // Go requires parsing the string "true" or "false"
                            injection.push_str(&format!(
                                "{} := `{{}}` == \"true\"\n_ = {}\n",
                                name.as_str(),
                                name.as_str()
                            ));
                        }
                        Type::Int => {
                            injection.push_str(&format!(
                                "{}Str := `{{}}`\n",
                                name.as_str()
                            ));
                            injection.push_str(&format!(
                                "{} := parseInt({}Str)\n",
                                name.as_str(),
                                name.as_str()
                            ));
                            injection.push_str(&format!("_ = {}\n", name.as_str()));
                        }
                        Type::Float => {
                            injection.push_str(&format!(
                                "{}Str := `{{}}`\n",
                                name.as_str()
                            ));
                            injection.push_str(&format!(
                                "{}, _ := strconv.ParseFloat(strings.TrimSpace({}Str), 64)\n",
                                name.as_str(),
                                name.as_str()
                            ));
                            injection.push_str(&format!("_ = {}\n", name.as_str()));
                        }
                        Type::List(inner) => {
                            let go_type = match inner.as_ref() {
                                Type::Int => "[]int64",
                                Type::Float => "[]float64",
                                Type::String => "[]string",
                                Type::Bool => "[]bool",
                                _ => "[]interface{}",
                            };
                            // Manually construct the string to produce `{{{}}}`.
                            // This ensures the outer format! macro sees a placeholder
                            // inside literal braces for the Go slice definition.
                            injection.push_str(&format!("{} := {}{{{{{}}}}}\n_ = {}\n",
                                                        name.as_str(),
                                                        go_type,
                                                        "{}", // This placeholder is for the list values
                                                        name.as_str()
                            ));
                        }
                        _ => {
                            injection.push_str(&format!(
                                "{} := `{{}}`\n_ = {}\n",
                                name.as_str(),
                                name.as_str()
                            ));
                        }
                    }
                }
            }

            "bash" => {
                injection.push_str("# TB Variables (auto-injected)\n");
                for (name, ty) in &self.variables_in_scope {
                    match ty {
                        Type::List(_) => {
                            // Array assignment is handled differently and remains correct.
                            injection.push_str(&format!("{}=({})\n", name.as_str(), "{}"));
                        }
                        _ => {
                            // For scalar variables (strings, numbers, etc.), use the safe `printf -v` method.
                            // This correctly interprets escaped newlines (`\n`) and handles special characters.
                            injection.push_str(&format!("printf -v {} %b '{{}}'\n", name.as_str()));
                        }
                    }
                }
            }

            _ => {}
        }

        injection
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
                let left_type = self.infer_type_from_expr(left);
                let right_type = self.infer_type_from_expr(right);

                // ✅ CRITICAL: Use smart operations for String types
                let needs_smart_op =
                    matches!(left_type, Type::String | Type::Dynamic)
                        || matches!(right_type, Type::String | Type::Dynamic)
                        || (left_type.is_numeric() && right_type.is_numeric() && left_type != right_type);

                if needs_smart_op && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
                    let op_name = match op {
                        BinOp::Add => "smart_add",
                        BinOp::Sub => "smart_sub",
                        BinOp::Mul => "smart_mul",
                        BinOp::Div => "smart_div",
                        _ => unreachable!(),
                    };

                    self.emit(&format!("{}(&", op_name));
                    self.generate_rust_expr(left)?;
                    self.emit(", &");
                    self.generate_rust_expr(right)?;
                    self.emit(")");
                    return Ok(());
                }

                // Normal binary operation (for primitives)
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
                // Special handling for builtin functions
                if let Expr::Variable(func_name) = function.as_ref() {
                    if matches!(func_name.as_str(), "echo" | "print" | "println") {
                        if args.is_empty() {
                            self.emit("println!()");
                            return Ok(());
                        }

                        let mut all_parts = Vec::new();
                        let mut all_args = Vec::new();

                        for arg in args {
                            match arg {
                                Expr::Literal(Literal::String(s)) => {
                                    let (fmt_str, vars) = self.parse_string_interpolation(s);
                                    all_parts.push(fmt_str);
                                    all_args.extend(vars);
                                }
                                _ => {
                                    // ✅ FIX: Dynamically choose {} or {:?} based on type
                                    let needs_debug = self.expr_needs_debug_format(arg);
                                    let fmt_str = if needs_debug { "{:?}" } else { "{}" };
                                    all_parts.push(fmt_str.to_string());
                                    all_args.push(arg.clone());
                                }
                            }
                        }

                        // Build println! call
                        self.emit("println!(\"");
                        self.emit(&all_parts.join(""));
                        self.emit("\"");

                        for arg_expr in &all_args {
                            self.emit(", ");
                            self.generate_rust_expr(arg_expr)?;
                        }

                        self.emit(")");
                        return Ok(());
                    }

                    if func_name.as_str() == "type_of" {
                        self.emit("type_of(&");
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("type_of requires an argument")
                            ));
                        }
                        self.generate_rust_expr(&args[0])?;
                        self.emit(")");
                        return Ok(());
                    }


                    // ✅ Special handling für len (pass by reference)
                    if func_name.as_str() == "len" {
                        self.emit("len(&");
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("len requires an argument")
                            ));
                        }
                        self.generate_rust_expr(&args[0])?;
                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ STR - Convert to string
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "str" {
                        self.emit("str(");
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("str requires an argument")
                            ));
                        }
                        self.generate_rust_expr(&args[0])?;
                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ INT - Convert to int (polymorphic)
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "int" {
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("int requires an argument")
                            ));
                        }

                        // Generate type-aware conversion
                        self.emit("(");
                        self.emit("if let Ok(i) = (");
                        self.generate_rust_expr(&args[0])?;
                        self.emit(").to_string().parse::<i64>() { i } else { 0 }");
                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ FLOAT - Convert to float (polymorphic)
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "float" {
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("float requires an argument")
                            ));
                        }

                        self.emit("(");
                        self.emit("if let Ok(f) = (");
                        self.generate_rust_expr(&args[0])?;
                        self.emit(").to_string().parse::<f64>() { f } else { 0.0 }");
                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ DEBUG - Debug print (pass by reference)
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "debug" {
                        self.emit("debug(&");
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("debug requires an argument")
                            ));
                        }
                        self.generate_rust_expr(&args[0])?;
                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ READ_LINE - Read from stdin
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "read_line" {
                        self.emit("({");
                        self.emit("use std::io::{self, BufRead};");
                        self.emit("let stdin = io::stdin();");
                        self.emit("let mut line = String::new();");
                        self.emit("stdin.lock().read_line(&mut line).ok();");
                        self.emit("line.trim_end().to_string()");
                        self.emit("})");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ PUSH - Append to list (mutable operation)
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "push" {
                        if args.len() < 2 {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("push requires 2 arguments (list, item)")
                            ));
                        }

                        self.emit("({");
                        self.emit("let mut tmp = ");
                        self.generate_rust_expr(&args[0])?;
                        self.emit(".clone();");
                        self.emit("tmp.push(");
                        self.generate_rust_expr(&args[1])?;
                        self.emit(");");
                        self.emit("tmp");
                        self.emit("})");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ POP - Remove from list (mutable operation)
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "pop" {
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("pop requires an argument")
                            ));
                        }

                        self.emit("({");
                        self.emit("let mut tmp = ");
                        self.generate_rust_expr(&args[0])?;
                        self.emit(".clone();");
                        self.emit("tmp.pop();");
                        self.emit("tmp");
                        self.emit("})");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ PRETTY - Pretty print with indentation
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "pretty" {
                        self.emit("pretty(&");
                        if args.is_empty() {
                            return Err(TBError::InvalidOperation(
                                STRING_INTERNER.intern("pretty requires an argument")
                            ));
                        }
                        self.generate_rust_expr(&args[0])?;
                        self.emit(")");
                        return Ok(());
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // ✅ PYTHON_INFO - Show Python environment info
                    // ═══════════════════════════════════════════════════════════════
                    if func_name.as_str() == "python_info" {
                        self.emit("python_info()");
                        return Ok(());
                    }


                    // ═══════════════════════════════════════════════════════════════
                    // SPECIAL CASE: Language bridges (python, javascript, go, bash)
                    // ═══════════════════════════════════════════════════════════════
                    if matches!(func_name.as_str(), "python" | "javascript" | "go" | "bash") {
                        return self.generate_language_bridge_call_with_context(func_name, args);
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // REGULAR FUNCTION CALL
                // ═══════════════════════════════════════════════════════════════
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

                for stmt in statements.as_ref() {
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

    /// Check if expression needs Debug formatting ({:?}) instead of Display ({})
    fn expr_needs_debug_format(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Variable(name) => {
                // Check variable type from scope
                let var_type = self.get_variable_type(name.as_str());
                self.needs_debug_format(&var_type)
            }

            Expr::List(_) => true,           // Lists always need {:?}
            Expr::Dict(_) => true,           // Dicts always need {:?}
            Expr::Tuple(_) => true,          // Tuples always need {:?}

            Expr::Call { function, .. } => {
                // Check if function returns complex type
                if let Expr::Variable(func_name) = function.as_ref() {
                    // Language bridges return strings (safe with {})
                    if matches!(func_name.as_str(), "python" | "javascript" | "go" | "bash") {
                        return false;
                    }

                    // Builtin functions that return complex types
                    if matches!(func_name.as_str(), "push" | "pop") {
                        return true;
                    }
                }
                false
            }

            _ => false  // Literals, BinOps etc. use Display
        }
    }


    /// Parse string with $variable interpolation (TYPE-AWARE)
    fn parse_string_interpolation(&self, s: &str) -> (String, Vec<Expr>) {
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

                        // ✅ Get variable type to decide format specifier
                        let var_type = self.get_variable_type(&var_name);

                        if self.needs_debug_format(&var_type) {
                            // Use {:?} for complex types
                            result.push_str("{:?}");
                        } else {
                            // Use {} for primitives
                            result.push_str("{}");
                        }

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

    /// Get type of variable from scope
    fn get_variable_type(&self, name: &str) -> Type {
        self.variables_in_scope
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, t)| t.clone())
            .unwrap_or(Type::Dynamic)
    }

    /// Check if type needs Debug format ({:?})
    fn needs_debug_format(&self, ty: &Type) -> bool {
        match ty {
            Type::Unit | Type::Bool | Type::Int | Type::Float | Type::String => false,
            Type::List(_) | Type::Dict { .. } | Type::Tuple(_)
            | Type::Option(_) | Type::Result { .. } => true,
            Type::Dynamic => false,
            _ => true,
        }
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

pub type Environment = Arc<RwLock<HashMap<Arc<String>, Value>>>;


/// Fast single-threaded environment (10x faster than Arc<RwLock>)
pub type FastEnvironment = Rc<RefCell<HashMap<Arc<String>, Value>>>;

/// Function registry for zero-copy function calls
pub struct FunctionRegistry {
    functions: HashMap<String, (Vec<Arc<String>>, Rc<Expr>)>, // name → (params, body)
}

impl FunctionRegistry {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    fn register(&mut self, name: Arc<String>, params: Vec<Arc<String>>, body: Expr) {
        self.functions.insert(name.to_string(), (params, Rc::new(body)));  //  Konvertiere einmal
    }

    fn get(&self, name: &str) -> Option<&(Vec<Arc<String>>, Rc<Expr>)> {
        self.functions.get(name)
    }
}
pub struct JitExecutor {
    env: FastEnvironment,
    config: Config,
    builtins: BuiltinRegistry,
    functions: FunctionRegistry,
    recursion_depth: usize,
    max_recursion_depth: usize,
}

impl JitExecutor {
    pub fn new(config: Config) -> Self {
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
            max_recursion_depth: 1000000,
        }
    }

    pub fn execute(&mut self, statements: &[Statement]) -> TBResult<Value> {
        debug_log!("JitExecutor::execute() started with {} statements", statements.len());
        //  TEMPORARY: Check if we have enough stack space
        #[cfg(debug_assertions)]
        {
            // Try to allocate a large-ish array on stack to test available space
            fn check_stack_space() -> bool {
                let _test: [u8; 100_000] = [0; 100_000]; // 100KB test
                true
            }

            if !std::panic::catch_unwind(|| check_stack_space()).is_ok() {
                eprintln!("⚠️  WARNING: Limited stack space detected!");
                eprintln!("   Consider running with: RUST_MIN_STACK=8388608 cargo run");
            }
        }
        let mut last_value = Value::Unit;

        //  PHASE 1: Register all functions FIRST (no cloning!)
        for stmt in statements {
            if let Statement::Function { name, params, body, .. } = stmt {
                let param_names: Vec<Arc<String>> = params.iter()
                    .map(|p| Arc::clone(&p.name))
                    .collect();

                self.functions.register(
                    Arc::clone(name),
                    param_names,
                    body.clone()
                );

                debug_log!("Registered function: {}", name);
            }
        }

        //  PHASE 2: Execute non-function statements
        for (i, stmt) in statements.iter().enumerate() {
            debug_log!("Executing statement {}/{}", i + 1, statements.len());
            last_value = self.execute_statement(stmt)?;
            debug_log!("Statement {} result: {:?}", i + 1, last_value);
        }

        debug_log!("JitExecutor::execute() completed");
        Ok(last_value)
    }

    pub fn execute_statement(&mut self, stmt: &Statement) -> TBResult<Value> {
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

            Statement::Function { .. } => {
                // Already registered in execute()
                Ok(Value::Unit)
            }

            Statement::Expr(expr) => self.eval_expr(expr),

            _ => Ok(Value::Unit),
        }
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> TBResult<Value> {
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
                //  Fast lookup with Rc<RefCell>
                if let Some(value) = self.env.borrow().get(name).cloned() {
                    return Ok(value);
                }

                // Check builtin functions
                if let Some(_) = self.builtins.get(name.as_str()) {
                    return Ok(Value::Native {
                        language: Language::Rust,
                        type_name: format!("builtin:{}", name).into(),
                        handle: NativeHandle {
                            id: name.as_ptr() as u64,
                            data: Arc::new(name.clone()),
                        },
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
                //  OPTIMIZED: Check if it's a registered function FIRST
                if let Expr::Variable(func_name) = function.as_ref() {
                    // Check language bridges first
                    if matches!(func_name.as_str(), "python" | "javascript" | "go" | "bash") {
                        let arg_vals: TBResult<Vec<_>> = args.iter().map(|a| self.eval_expr(a)).collect();
                        let arg_vals = arg_vals?;

                        if let Some(Value::String(code)) = arg_vals.first() {
                            let variables = self.env.borrow().clone();
                            let context = LanguageExecutionContext::with_variables(variables);

                            return match func_name.as_str() {
                                "python" => execute_python_code_with_context(code, &arg_vals[1..], Some(&context)),
                                "javascript" => execute_js_code_with_context(code, &arg_vals[1..], Some(&context)),
                                "go" => execute_go_code_with_context(code, &arg_vals[1..], Some(&context)),
                                "bash" => execute_bash_command_with_context(code, &arg_vals[1..], Some(&context)),
                                _ => unreachable!(),
                            };
                        }
                    }

                    //  ZERO-COPY function call from registry (NO CLONING!)
                    if let Some((params, body_rc)) = self.functions.get(func_name.as_str()) {
                        //  Clone only the Vec<String>, not the Expr!
                        let params = params.clone();
                        let body_rc = body_rc.clone(); //  Rc::clone is cheap (just increments ref count)

                        // Evaluate arguments
                        let arg_vals: TBResult<Vec<_>> = args.iter().map(|a| self.eval_expr(a)).collect();
                        let arg_vals = arg_vals?;

                        return self.call_function_fast(func_name.as_str(), &params, &body_rc, arg_vals);
                    }
                }

                // Fallback: regular function call
                let func = self.eval_expr(function.as_ref())?;
                let arg_vals: TBResult<Vec<_>> = args.iter().map(|a| self.eval_expr(a)).collect();
                let arg_vals = arg_vals?;
                self.call_function_legacy(func, arg_vals)
            }

            Expr::Block { statements, result } => {
                for stmt in statements.as_ref() {
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
                let max_iterations = 10000000;

                loop {
                    iterations += 1;

                    if iterations > max_iterations {
                        return Err(TBError::RuntimeError {
                            message: format!("Loop exceeded maximum iterations ({})", max_iterations).into(),
                            trace: vec![Arc::from("loop execution".to_string())],
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
                let max_iterations = 1000000;
                let mut last_value = Value::Unit;

                while self.eval_expr(condition)?.is_truthy() {
                    iterations += 1;

                    if iterations > max_iterations {
                        return Err(TBError::RuntimeError {
                            message: format!("While loop exceeded maximum iterations ({})", max_iterations).into(),
                            trace: vec![Arc::from("while loop execution".to_string())],
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
                let values: TBResult<Vec<_>> = items.iter().map(|e| self.eval_expr(e)).collect();
                Ok(Value::List(values?))
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

                for op in operations.as_ref() {
                    result = match op {
                        Expr::Call { function, args } => {
                            if let Expr::Variable(func_name) = function.as_ref() {
                                if matches!(func_name.as_str(), "python" | "javascript" | "go" | "bash") {
                                    let arg_vals: TBResult<Vec<_>> = args.iter().map(|a| self.eval_expr(a)).collect();
                                    let arg_vals = arg_vals?;

                                    if let Some(Value::String(code)) = arg_vals.first() {
                                        let variables = self.env.borrow().clone();
                                        let context = LanguageExecutionContext::with_variables(variables);

                                        match func_name.as_str() {
                                            "python" => execute_python_code_with_context(code, &arg_vals[1..], Some(&context))?,
                                            "javascript" => execute_js_code_with_context(code, &arg_vals[1..], Some(&context))?,
                                            "go" => execute_go_code_with_context(code, &arg_vals[1..], Some(&context))?,
                                            "bash" => execute_bash_command_with_context(code, &arg_vals[1..], Some(&context))?,
                                            _ => unreachable!(),
                                        }
                                    } else {
                                        result
                                    }
                                } else {
                                    let func = self.eval_expr(function.as_ref())?;
                                    let arg_vals: TBResult<Vec<_>> = args.iter().map(|a| self.eval_expr(a)).collect();
                                    let arg_vals = arg_vals?;
                                    self.call_function_legacy(func, arg_vals)?
                                }
                            } else {
                                let func = self.eval_expr(function.as_ref())?;
                                let arg_vals: TBResult<Vec<_>> = args.iter().map(|a| self.eval_expr(a)).collect();
                                let arg_vals = arg_vals?;
                                self.call_function_legacy(func, arg_vals)?
                            }
                        }
                        Expr::Variable(name) => {
                            if let Some((params, body_rc)) = self.functions.get(name) {
                                let params = params.clone();
                                let body_rc = body_rc.clone(); //  Rc::clone is cheap
                                self.call_function_fast(name, &params, &body_rc, vec![result])?
                            } else {
                                let func = self.env.borrow().get(name)
                                    .cloned()
                                    .ok_or_else(|| TBError::UndefinedFunction(name.clone()))?;
                                self.call_function_legacy(func, vec![result])?
                            }
                        }
                        _ => return Err(TBError::InvalidOperation(Arc::from("Pipeline".to_string()))),
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
                //  Ensure this returns Value::List
                let results: TBResult<Vec<_>> = tasks
                    .iter()
                    .map(|task| self.eval_expr(task))
                    .collect();

                Ok(Value::List(results?))
            }

            _ => Ok(Value::Unit),
        }
    }

    ///  ULTRA-FAST function call (zero environment cloning!)
    fn call_function_fast(
        &mut self,
        func_name: &str,  //  Changed from _func_name to use it
        params: &[Arc<String>],
        body: &Rc<Expr>,
        args: Vec<Value>,
    ) -> TBResult<Value> {
        // Check recursion depth
        self.recursion_depth += 1;

        //  DEBUG: Print current recursion depth and estimated stack usage
        #[cfg(debug_assertions)]
        {
            let estimated_stack_kb = self.recursion_depth * 4; // Rough estimate: ~4KB per call
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
                message: format!(
                    "Maximum recursion depth exceeded ({})",
                    self.max_recursion_depth
                ).into(),
                trace: vec![Arc::from("function call".to_string())],
            });
        }

        if params.len() != args.len() {
            self.recursion_depth -= 1;
            return Err(TBError::InvalidOperation(
                format!("Expected {} arguments, got {}", params.len(), args.len()).into()
            ));
        }

        //  CRITICAL OPTIMIZATION: Temporarily push parameters into environment
        let mut saved_params = Vec::new();
        for (param, arg) in params.iter().zip(args.iter()) {
            let old_value = self.env.borrow_mut().insert(Arc::clone(param), arg.clone());
            saved_params.push((Arc::clone(param), old_value));
        }

        //  Execute body (dereference Rc to get &Expr)
        let result = self.eval_expr(body);  // body: &Rc<Expr> derefs to &Expr automatically

        // Restore old parameter values
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
    fn call_function_legacy(&mut self, func: Value, args: Vec<Value>) -> TBResult<Value> {
        match func {
            Value::Function { params, body, env: _ } => {
                //  Wrap body in Rc for consistency
                let body_rc = Rc::new(*body);
                self.call_function_fast("", &params, &body_rc, args)
            }

            Value::Native { type_name, .. } if type_name.starts_with("builtin:") => {
                let func_name = type_name.strip_prefix("builtin:").unwrap();

                if let Some(builtin) = self.builtins.get(func_name) {
                    if args.len() < builtin.min_args {
                        return Err(TBError::InvalidOperation(
                            format!("{} requires at least {} arguments, got {}",
                                    func_name, builtin.min_args, args.len()).into()
                        ));
                    }

                    if let Some(max) = builtin.max_args {
                        if args.len() > max {
                            return Err(TBError::InvalidOperation(
                                format!("{} accepts at most {} arguments, got {}",
                                        func_name, max, args.len()).into()
                            ));
                        }
                    }

                    (builtin.function)(&args)
                } else {
                    Err(TBError::UndefinedFunction(Arc::from(func_name.to_string())))
                }
            }

            _ => Err(TBError::InvalidOperation(format!(
                "Not a function: {:?}", func.get_type()
            ).into())),
        }
    }

    fn interpolate_string(&self, template: &str) -> TBResult<String> {
        let mut result = String::new();
        let mut chars = template.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '$' {
                // Found variable reference
                let mut var_name = String::new();

                // Read variable name (alphanumeric + underscore)
                while let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        var_name.push(next_ch);
                        chars.next();
                    } else {
                        break;
                    }
                }

                if var_name.is_empty() {
                    // Just a dollar sign, keep it
                    result.push('$');
                } else {
                    // Look up variable and format it
                    if let Some(value) = self.env.borrow().get(&var_name).cloned() {
                        result.push_str(&format_value_for_output(&value));
                    } else {
                        // Variable not found, keep original
                        result.push('$');
                        result.push_str(&var_name);
                    }
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }

    fn eval_binop(&self, op: BinOp, left: Value, right: Value) -> TBResult<Value> {
        match (op, left, right) {
            // ═══════════════════════════════════════════════════════════
            // INTEGER ARITHMETIC
            // ═══════════════════════════════════════════════════════════
            (BinOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (BinOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (BinOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (BinOp::Div, Value::Int(a), Value::Int(b)) => {
                if b == 0 {
                    Err(TBError::DivisionByZero)
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            (BinOp::Mod, Value::Int(a), Value::Int(b)) => {
                if b == 0 {
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
            (BinOp::Add, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
            (BinOp::Add, Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),

            (BinOp::Sub, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
            (BinOp::Sub, Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),

            (BinOp::Mul, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
            (BinOp::Mul, Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),

            (BinOp::Div, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
            (BinOp::Div, Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),

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
            (BinOp::Eq, Value::Float(a), Value::Int(b)) => Ok(Value::Bool((a - b as f64).abs() < f64::EPSILON)),
            (BinOp::Eq, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64 - b).abs() < f64::EPSILON)),

            (BinOp::Ne, Value::Float(a), Value::Int(b)) => Ok(Value::Bool((a - b as f64).abs() >= f64::EPSILON)),
            (BinOp::Ne, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64 - b).abs() >= f64::EPSILON)),

            (BinOp::Lt, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a < b as f64)),
            (BinOp::Lt, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64) < b)),

            (BinOp::Le, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a <= b as f64)),
            (BinOp::Le, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64) <= b)),

            (BinOp::Gt, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a > b as f64)),
            (BinOp::Gt, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64) > b)),

            (BinOp::Ge, Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a >= b as f64)),
            (BinOp::Ge, Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64) >= b)),

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

    fn eval_unaryop(&self, op: UnaryOp, val: Value) -> TBResult<Value> {
        match (op, val) {
            (UnaryOp::Not, val) => Ok(Value::Bool(!val.is_truthy())),
            (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
            _ => Err(TBError::InvalidOperation(format!("Unary operation {:?}", op).into())),
        }
    }

    fn literal_to_value(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Unit => Value::Unit,
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Int(n) => Value::Int(*n),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
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
    config: Config,
    shared_state: Arc<Mutex<HashMap<Arc<String>, Value>>>,
    worker_threads: usize,
}

impl ParallelExecutor {
    pub fn new(config: Config) -> Self {
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
    pub fn execute(&mut self, statements: &[Statement]) -> TBResult<Value> {
        debug_log!("ParallelExecutor: {} workers, {} statements",
            self.worker_threads, statements.len());

        // Separate parallel blocks from sequential statements
        let mut sequential_stmts = Vec::new();
        let mut parallel_blocks = Vec::new();

        for stmt in statements {
            if self.contains_parallel_expr(stmt) {
                parallel_blocks.push(stmt.clone());
            } else {
                sequential_stmts.push(stmt.clone());
            }
        }

        // Execute sequential statements first
        let mut result = Value::Unit;
        if !sequential_stmts.is_empty() {
            result = self.execute_sequential(&sequential_stmts)?;
        }

        // Execute parallel blocks
        for block_stmt in parallel_blocks {
            result = self.execute_statement_parallel(&block_stmt)?;
        }

        Ok(result)
    }

    /// Execute sequential statements using single JitExecutor
    fn execute_sequential(&self, statements: &[Statement]) -> TBResult<Value> {
        let mut executor = self.create_executor();
        executor.execute(statements)
    }

    /// Execute statement that contains parallel operations
    fn execute_statement_parallel(&self, stmt: &Statement) -> TBResult<Value> {
        match stmt {
            Statement::Expr(Expr::Parallel(tasks)) => {
                self.execute_parallel_tasks(tasks)
            }

            Statement::Expr(Expr::For { variable, iterable, body }) => {
                // Evaluate iterable first
                let mut executor = self.create_executor();
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
                // Fallback to sequential
                let mut executor = self.create_executor();
                executor.execute_statement(stmt)
            }
        }
    }

    /// Execute parallel tasks using Rayon
    fn execute_parallel_tasks(&self, tasks: &[Expr]) -> TBResult<Value> {
        debug_log!("Executing {} tasks in parallel", tasks.len());

        // Execute each task in parallel with its own JitExecutor
        let results: Vec<TBResult<Value>> = tasks
            .par_iter()
            .map(|task| {
                let mut executor = self.create_executor();
                executor.eval_expr(task)
            })
            .collect();

        // Collect ALL results, not just last
        let mut values = Vec::new();
        for result in results {
            values.push(result?);
        }

        //  CRITICAL: Return as List, not Unit
        Ok(Value::List(values))
    }

    /// Execute parallel for loop
    fn execute_parallel_for(
        &self,
        variable: &Arc<String>,
        items: &[Value],
        body: &Expr,
    ) -> TBResult<Value> {
        debug_log!("Parallel for: {} iterations", items.len());

        // Process items in parallel
        let results: Vec<TBResult<Value>> = items
            .par_iter()
            .map(|item| {
                let mut executor = self.create_executor();

                // Set loop variable
                executor.env.borrow_mut().insert(Arc::clone(variable), item.clone());

                // Execute body
                executor.eval_expr(body)
            })
            .collect();

        // Return last result or first error
        let mut last_value = Value::Unit;
        for result in results {
            last_value = result?;
        }

        Ok(last_value)
    }

    /// Create new JitExecutor with shared config and state
    fn create_executor(&self) -> JitExecutor {
        let mut config = self.config.clone();

        // Inject current shared state
        let shared_state = self.shared_state.lock().unwrap();
        config.shared = shared_state.clone();

        JitExecutor::new(config)
    }

    /// Check if statement contains parallel expressions
    fn contains_parallel_expr(&self, stmt: &Statement) -> bool {
        match stmt {
            Statement::Expr(expr) => self.is_parallel_expr(expr),
            Statement::Let { value, .. } => self.is_parallel_expr(value),
            Statement::Function { body, .. } => self.is_parallel_expr(body),
            _ => false,
        }
    }

    fn is_parallel_expr(&self, expr: &Expr) -> bool {
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

    /// Get value from shared state
    pub fn get_shared(&self, key: &str) -> Option<Value> {
        let state = self.shared_state.lock().unwrap();
        state.iter()
            .find(|(k, _)| k.as_str() == key)
            .map(|(_, v)| v.clone())
    }

    /// Set value in shared state
    pub fn set_shared(&self, key: Arc<String>, value: Value) {
        self.shared_state.lock().unwrap().insert(key, value);
    }

    /// Get all shared state
    pub fn get_all_shared(&self) -> HashMap<Arc<String>, Value> {
        self.shared_state.lock().unwrap().clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.5 BUILTIN FUNCTIONS & NATIVE BRIDGES
// ═══════════════════════════════════════════════════════════════════════════

/// Native function signature
type NativeFunction = Arc<dyn Fn(&[Value]) -> TBResult<Value> + Send + Sync>;

/// Native function with metadata
pub struct BuiltinFunction {
    pub name: Arc<String>,
    pub function: Arc<dyn Fn(&[Value]) -> TBResult<Value> + Send + Sync>,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub description: Arc<String>,
}

impl Clone for BuiltinFunction {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            function: self.function.clone(),
            min_args: self.min_args,
            max_args: self.max_args,
            description: self.description.clone(),
        }
    }
}

/// Registry for builtin and plugin functions
#[derive(Clone)]
pub struct BuiltinRegistry {
    functions: HashMap<String, BuiltinFunction>,
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
        self.register("python", Arc::new(builtin_python), 1, Some(1), "Execute Python code");
        self.register("javascript", Arc::new(builtin_javascript), 1, Some(1), "Execute JavaScript code");
        self.register("bash", Arc::new(builtin_bash), 1, Some(1), "Execute Bash code");
        self.register("go", Arc::new(builtin_go), 1, Some(1), "Execute Go code");

        self.register("echo", Arc::new(builtin_echo), 1, None, "Print values to stdout");
        self.register("print", Arc::new(builtin_print), 1, None, "Print values without newline");
        self.register("println", Arc::new(builtin_println), 1, None, "Print values with newline");
        self.register("read_line", Arc::new(builtin_read_line), 0, Some(0), "Read a line from stdin");

        // Type conversion
        self.register("str", Arc::new(builtin_str), 1, Some(1), "Convert to string");
        self.register("int", Arc::new(builtin_int), 1, Some(1), "Convert to integer");
        self.register("float", Arc::new(builtin_float), 1, Some(1), "Convert to float");
        self.register("pretty", Arc::new(builtin_pretty), 1, Some(1), "Pretty print with indentation");

        // List operations
        self.register("len", Arc::new(builtin_len), 1, Some(1), "Get length of collection");
        self.register("push", Arc::new(builtin_push), 2, Some(2), "Push item to list");
        self.register("pop", Arc::new(builtin_pop), 1, Some(1), "Pop item from list");

        // Debug
        self.register("debug", Arc::new(builtin_debug), 1, None, "Debug print with type info");
        self.register("type_of", Arc::new(builtin_type_of), 1, Some(1), "Get type of value");
        self.register("python_info", Arc::new(builtin_python_info), 0, Some(0), "Show active Python environment");

    }

    /// Register a builtin function
    pub fn register(
        &mut self,
        name: &str,
        function: NativeFunction,
        min_args: usize,
        max_args: Option<usize>,
        description: &str,
    ) {
        let arc_name = STRING_INTERNER.intern(name);
        self.functions.insert(
            name.to_string(),  //
            BuiltinFunction {
                name: arc_name,
                function,
                min_args,
                max_args,
                description: STRING_INTERNER.intern(description),
            },
        );
    }

    /// Get a builtin function (O(1) lookup with &str)
    pub fn get(&self, name: &str) -> Option<&BuiltinFunction> {
        self.functions.get(name)  //  Direkter O(1) lookup
    }

    /// Check if function exists (O(1))
    pub fn has(&self, name: &str) -> bool {
        self.functions.contains_key(name)  //  Direkter O(1) lookup
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

            // Build code with imports
            let full_code = if !imports.is_empty() {
                let mut code_with_imports = imports.join("\n");
                code_with_imports.push_str("\n\n");
                code_with_imports.push_str(&code);
                code_with_imports
            } else {
                code
            };
            let description = Arc::new(format!("[Plugin:{}] {}", plugin.name, func.description));
            self.register(
                &func_name,
                Arc::new(move |args: &[Value]| -> TBResult<Value> {
                    execute_python_code(&full_code, args)
                }),
                func.min_args,
                func.max_args,
                &description,
            );

            debug_log!("  Registered function: {}", func_name);
        }

        Ok(())
    }

    /// Load JavaScript plugin functions
    fn load_javascript_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let imports = plugin.imports.clone();
            let func_name = func.name.clone();

            // Build code with imports (require statements)
            let full_code = if !imports.is_empty() {
                let mut code_with_imports = String::new();
                for import in &imports {
                    code_with_imports.push_str(&format!("const {} = require('{}');\n",
                                                        import.split('/').last().unwrap_or(import), import));
                }
                code_with_imports.push('\n');
                code_with_imports.push_str(&code);
                code_with_imports
            } else {
                code
            };
            let description = Arc::new(format!("[Plugin:{}] {}", plugin.name, func.description));
            self.register(
                &func_name,
                Arc::new(move |args: &[Value]| -> TBResult<Value> {
                    execute_js_code(&full_code, args)
                }),
                func.min_args,
                func.max_args,
                &description,
            );

            debug_log!("  Registered function: {}", func_name);
        }

        Ok(())
    }

    /// Load TypeScript plugin functions (transpile to JS first)
    fn load_typescript_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        // For now, treat as JavaScript
        // TODO: Add ts-node or transpilation support
        self.load_javascript_plugin(plugin)
    }

    /// Load Go plugin functions
    fn load_go_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let imports = plugin.imports.clone();
            let func_name = func.name.clone();
            let description = Arc::new(format!("[Plugin:{}] {}", plugin.name, func.description));
            self.register(
                &func_name,
                Arc::new(move |args: &[Value]| -> TBResult<Value> {
                    // Build complete Go code with imports
                    let mut full_code = String::new();

                    // Add custom imports
                    for import in &imports {
                        full_code.push_str(&format!("    \"{}\"\n", import));
                    }

                    full_code.push_str(&code);

                    execute_go_code(&full_code, args)
                }),
                func.min_args,
                func.max_args,
                &description,
            );

            debug_log!("  Registered function: {}", func_name);
        }

        Ok(())
    }

    /// Load Bash plugin functions
    fn load_bash_plugin(&mut self, plugin: &PluginConfig) -> TBResult<()> {
        for func in &plugin.functions {
            let code = func.code.clone();
            let func_name = func.name.clone();
            let description = Arc::new(format!("[Plugin:{}] {}", plugin.name, func.description));
            self.register(
                &func_name,
                Arc::new(move |args: &[Value]| -> TBResult<Value> {
                    execute_bash_command(&code, args)
                }),
                func.min_args,
                func.max_args,
                &description,
            );

            debug_log!("  Registered function: {}", func_name);
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STANDARD LIBRARY IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// echo - Print values to stdout with newline
fn builtin_echo(args: &[Value]) -> TBResult<Value> {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", format_value_for_output(arg));
    }
    println!();
    Ok(Value::Unit)
}

/// print - Print values without newline
fn builtin_print(args: &[Value]) -> TBResult<Value> {
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

/// println - Print values with newline
fn builtin_println(args: &[Value]) -> TBResult<Value> {
    builtin_echo(args)
}

/// read_line - Read a line from stdin
fn builtin_read_line(_args: &[Value]) -> TBResult<Value> {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)
        .map_err(|e| TBError::IoError(Arc::from(e.to_string())))?;
    Ok(Value::String(Arc::new(line.trim_end().to_string())))
}

/// str - Convert to string
fn builtin_str(args: &[Value]) -> TBResult<Value> {
    let formatted = match &args[0] {
        Value::Unit => String::from(""),
        Value::Bool(b) => format!("{}", b),
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => {
            // Smart float formatting
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

/// int - Convert to integer with enhanced parsing
fn builtin_int(args: &[Value]) -> TBResult<Value> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Float(f) => Ok(Value::Int(*f as i64)),
        Value::String(s) => {
            let cleaned = s.trim();

            // Try direct parse
            if let Ok(n) = cleaned.parse::<i64>() {
                return Ok(Value::Int(n));
            }

            // Try parsing with underscores (1_000_000)
            let no_underscores = cleaned.replace('_', "");
            if let Ok(n) = no_underscores.parse::<i64>() {
                return Ok(Value::Int(n));
            }

            // Try parsing with spaces (1 000 000)
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

/// float - Convert to float with German notation support
fn builtin_float(args: &[Value]) -> TBResult<Value> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Float(*n as f64)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::String(s) => {
            let cleaned = s.trim();

            // Try direct parse (English notation: 3.14)
            if let Ok(f) = cleaned.parse::<f64>() {
                return Ok(Value::Float(f));
            }

            // Try German notation: 3,14 → 3.14
            let german_to_english = cleaned.replace(',', ".");
            if let Ok(f) = german_to_english.parse::<f64>() {
                return Ok(Value::Float(f));
            }

            // Try parsing with thousand separators
            // English: 1,234.56 → 1234.56
            let no_thousand_comma = cleaned.replace(",", "");
            if let Ok(f) = no_thousand_comma.parse::<f64>() {
                return Ok(Value::Float(f));
            }

            // German thousand separator: 1.234,56 → 1234.56
            // First remove dots (thousand separator)
            let mut german_style = cleaned.replace('.', "");
            // Then replace comma with dot (decimal separator)
            german_style = german_style.replace(',', ".");
            if let Ok(f) = german_style.parse::<f64>() {
                return Ok(Value::Float(f));
            }

            // Try with underscores (Rust style: 3.14_159)
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

/// pretty - Pretty print with indentation
fn builtin_pretty(args: &[Value]) -> TBResult<Value> {
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

/// len - Get length of collection
fn builtin_len(args: &[Value]) -> TBResult<Value> {
    match &args[0] {
        Value::String(s) => Ok(Value::Int(s.len() as i64)),
        Value::List(l) => Ok(Value::Int(l.len() as i64)),
        Value::Dict(d) => Ok(Value::Int(d.len() as i64)),
        Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
        _ => Err(TBError::InvalidOperation(Arc::from("Value has no length".to_string()))),
    }
}

/// push - Push item to list (modifies in place)
fn builtin_push(args: &[Value]) -> TBResult<Value> {
    if let Value::List(mut list) = args[0].clone() {
        list.push(args[1].clone());
        Ok(Value::List(list))
    } else {
        Err(TBError::InvalidOperation(Arc::from("First argument must be a list".to_string())))
    }
}

/// pop - Pop item from list
fn builtin_pop(args: &[Value]) -> TBResult<Value> {
    if let Value::List(mut list) = args[0].clone() {
        list.pop()
            .map(|v| Value::Tuple(vec![Value::List(list), v]))
            .ok_or_else(|| TBError::InvalidOperation(Arc::from("Cannot pop from empty list".to_string())))
    } else {
        Err(TBError::InvalidOperation(Arc::from("Argument must be a list".to_string())))
    }
}

/// debug - Debug print with type info
fn builtin_debug(args: &[Value]) -> TBResult<Value> {
    for arg in args {
        eprintln!("[DEBUG] {:?} : {:?}", arg, arg.get_type());
    }
    Ok(Value::Unit)
}

/// type_of - Get type of value
fn builtin_type_of(args: &[Value]) -> TBResult<Value> {
    let type_str = match &args[0] {
        Value::Unit => "unit",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::String(_) => "string",
        Value::List(_) => "list",
        Value::Dict(_) => "dict",
        Value::Tuple(_) => "tuple",
        Value::Option(_) => "option",
        Value::Result(_) => "result",
        Value::Function { .. } => "function",
        Value::Native { .. } => "native",
    };
    Ok(Value::String(Arc::new(type_str.to_string())))
}
/// python_info - Show information about the active Python environment
fn builtin_python_info(_args: &[Value]) -> TBResult<Value> {
    use std::process::Command;

    let python_exe = detect_python_executable();

    // Get Python version
    let version_output = Command::new(&python_exe)
        .arg("--version")
        .output()
        .ok();

    let version = if let Some(out) = version_output {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        "Unknown".to_string()
    };

    // Get Python path
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

    // Detect environment type
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

/// python - Execute Python code
fn builtin_python(args: &[Value]) -> TBResult<Value> {
    if let Value::String(code) = &args[0] {
        execute_python_code(code, &args[1..])
    } else {
        Err(TBError::InvalidOperation(Arc::from("python() requires string argument".to_string())))
    }
}

/// javascript - Execute JavaScript code
fn builtin_javascript(args: &[Value]) -> TBResult<Value> {
    if let Value::String(code) = &args[0] {
        execute_js_code(code, &args[1..])
    } else {
        Err(TBError::InvalidOperation(Arc::from("javascript() requires string argument".to_string())))
    }
}

/// bash - Execute Bash code
fn builtin_bash(args: &[Value]) -> TBResult<Value> {
    if let Value::String(code) = &args[0] {
        execute_bash_command(code, &args[1..])
    } else {
        Err(TBError::InvalidOperation(Arc::from("bash() requires string argument".to_string())))
    }
}

/// go - Execute Go code
fn builtin_go(args: &[Value]) -> TBResult<Value> {
    if let Value::String(code) = &args[0] {
        execute_go_code(code, &args[1..])
    } else {
        Err(TBError::InvalidOperation(Arc::from("go() requires string argument".to_string())))
    }
}

/// Helper to format value for output (without quotes for strings)
fn format_value_for_output(value: &Value) -> String {
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
pub struct LanguageExecutionContext {
    pub variables: HashMap<Arc<String>, Value>,
    pub return_type: Option<Type>,
}

impl LanguageExecutionContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            return_type: None,
        }
    }

    pub fn with_variables(variables: HashMap<Arc<String>, Value>) -> Self {

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
                    // ✅ FIX: Proper Python string escaping with triple quotes for multiline
                    if s.contains('\n') {
                        let escaped = s.replace("\\", "\\\\").replace("\"\"\"", "\\\"\\\"\\\"");
                        format!("\"\"\"{}\"\"\"", escaped)
                    } else {
                        let escaped = s
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\r', "\\r")
                            .replace('\t', "\\t");
                        format!("\"{}\"", escaped)
                    }
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
                    // ✅ FIX: Proper JavaScript string escaping
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t")
                        .replace('\x08', "\\b")  // backspace
                        .replace('\x0C', "\\f"); // form feed
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
        let mut code = String::from("// TB Variables (auto-injected)\n");

        for (name, value) in &self.variables {
            match value {
                Value::Unit => {
                    code.push_str(&format!("var {} interface{{}} = nil\n", name));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::Bool(b) => {
                    code.push_str(&format!("{} := {}\n", name, if *b { "true" } else { "false" }));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::Int(n) => {
                    code.push_str(&format!("{} := int64({})\n", name, n));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::Float(f) => {
                    code.push_str(&format!("{} := float64({})\n", name, f));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::String(s) => {
                    // ✅ FIX: Use raw string literal (backticks) for Go
                    let escaped = s.replace('`', "` + \"`\" + `");
                    code.push_str(&format!("{} := `{}`\n", name, escaped));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::List(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_go(v))
                        .collect();

                    code.push_str(&format!("{} := []interface{{}}{{{}}}\n",
                                           name, elements.join(", ")));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::Dict(_) => {
                    code.push_str(&format!("{} := make(map[string]interface{{}})\n", name));
                    code.push_str(&format!("_ = {}\n", name));
                }

                Value::Tuple(items) => {
                    let elements: Vec<String> = items.iter()
                        .map(|v| self.value_to_go(v))
                        .collect();

                    code.push_str(&format!("{} := []interface{{}}{{{}}}\n",
                                           name, elements.join(", ")));
                    code.push_str(&format!("_ = {}\n", name));
                }

                _ => {
                    code.push_str(&format!("var {} interface{{}} = nil\n", name));
                    code.push_str(&format!("_ = {}\n", name));
                }
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
                Value::String(s) => {
                    // ✅ FIX: Proper Bash string quoting
                    // Replace single quotes with '\''
                    let escaped = s.replace('\'', "'\\''");
                    format!("'{}'", escaped)
                }
                Value::Int(n) => format!("{}", n),
                Value::Float(f) => format!("{}", f),
                Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
                Value::List(items) => {
                    // ✅ FIX: Bash array syntax
                    let elements: Vec<String> = items.iter()
                        .map(|v| {
                            match v {
                                Value::String(s) => {
                                    let escaped = s.replace('\'', "'\\''");
                                    format!("'{}'", escaped)
                                }
                                Value::Int(n) => format!("{}", n),
                                Value::Float(f) => format!("{}", f),
                                _ => format!("'{}'", format_value_for_output(v))
                            }
                        })
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
                // ✅ FIX: For Go, use raw strings but escape backticks
                if s.contains('\n') || s.contains('\t') {
                    // Multi-line: use regular string with escapes
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t");
                    format!("\"{}\"", escaped)
                } else {
                    // Single line: use raw string
                    let escaped = s.replace('`', "` + \"`\" + `");
                    format!("`{}`", escaped)
                }
            }
            Value::List(items) => {
                let elements: Vec<String> = items.iter()
                    .map(|v| self.value_to_go(v))
                    .collect();
                format!("[]interface{{{}}}", elements.join(", "))
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
    /// Parse Python output as TB value
    pub fn from_python(output: &str) -> Value {
        let trimmed = output.trim();

        // Try to parse as JSON-like structure
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            // List
            return Self::parse_list(trimmed);
        }

        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            // Dict
            return Self::parse_dict(trimmed);
        }

        // Try boolean
        if trimmed == "True" {
            return Value::Bool(true);
        }
        if trimmed == "False" {
            return Value::Bool(false);
        }

        // Try integer
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }

        // Try float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        // Default to string
        Value::String(Arc::from(trimmed.to_string()))
    }

    /// Parse JavaScript output as TB value
    pub fn from_javascript(output: &str) -> Value {
        let trimmed = output.trim();

        // Similar to Python but with different boolean syntax
        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }

        if trimmed == "null" || trimmed == "undefined" {
            return Value::Unit;
        }

        // Try integer
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }

        // Try float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        // Default to string
        Value::String(Arc::from(trimmed.to_string()))
    }

    /// Parse Go output as TB value
    pub fn from_go(output: &str) -> Value {
        let trimmed = output.trim();

        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }

        // Try integer
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }

        // Try float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        // Default to string
        Value::String(Arc::from(trimmed.to_string()))
    }

    /// Parse Bash output as TB value
    pub fn from_bash(output: &str) -> Value {
        let trimmed = output.trim();

        // Bash returns everything as strings, try to infer type
        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }

        // Try integer
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Int(n);
        }

        // Try float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Value::Float(f);
        }

        // Default to string
        Value::String(Arc::from(trimmed.to_string()))
    }

    // Helper methods
    fn parse_list(s: &str) -> Value {
        // Simple list parser - just split by comma for now
        let inner = &s[1..s.len()-1];
        let items: Vec<Value> = inner.split(',')
            .map(|item| {
                let trimmed = item.trim();
                if let Ok(n) = trimmed.parse::<i64>() {
                    Value::Int(n)
                } else if let Ok(f) = trimmed.parse::<f64>() {
                    Value::Float(f)
                } else {
                    Value::String(Arc::from(trimmed.trim_matches('"').to_string()))
                }
            })
            .collect();
        Value::List(items)
    }

    fn parse_dict(_s: &str) -> Value {
        // TODO: Implement dict parsing
        Value::Dict(HashMap::new())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NATIVE LANGUAGE EXECUTORS
// ═══════════════════════════════════════════════════════════════════════════

/// Execute bash command

/// Execute bash command with TB variable context
fn execute_bash_command_with_context(
    code: &str,
    args: &[Value],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value> {
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
            message: format!(
                "Failed to execute bash command: {}\n\n\
                Shell: {}\n\
                Code:\n{}\n",
                e, shell, code
            ).into(),
            trace: vec![
                Arc::from("execute_bash_command_with_context()".to_string()),
                format!("bash(...) call").into(),
            ],
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        let exit_code = output.status.code().unwrap_or(-1);
        let code_preview = code.lines()
            .take(3)
            .collect::<Vec<_>>()
            .join("\n");

        return Err(TBError::RuntimeError {
            message: format!(
                "Bash command failed (exit code {}):\n\n\
            Error output:\n{}\n\n\
            Code executed:\n{}\n\n\
            Shell: {}\n",
                exit_code,
                stderr.trim(),
                code_preview,
                shell
            ).into(),
            trace: vec![
                Arc::from("execute_bash_command_with_context()".to_string()),
                format!("bash() call").into(),
                format!("Exit code: {}", exit_code).into(),
            ],
        });
    }

    // ✅ FIX: Split output
    let stdout_str = stdout.trim();
    let lines: Vec<&str> = stdout_str.lines().collect();

    if lines.len() > 1 {
        for line in &lines[..lines.len() - 1] {
            println!("{}", line);
        }
        let return_value = lines.last().unwrap_or(&"");
        Ok(ReturnValueParser::from_bash(return_value))
    } else {
        Ok(ReturnValueParser::from_bash(stdout_str))
    }
}

fn execute_bash_command(code: &str, args: &[Value]) -> TBResult<Value> {
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
fn execute_python_code_with_context(
    code: &str,
    args: &[Value],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value> {
    use std::process::Command;

    let python_exe = detect_python_executable();

    let arg_strings: Vec<String> = args.iter()
        .map(|v| format_value_for_output(v))
        .collect();

    //  AUTO-WRAP: Last expression wird automatisch ausgegeben
    let wrapped_code = wrap_python_auto_return(code);

    let mut script = String::from("import sys\nimport json\n");
    script.push_str("args = sys.argv[1:]\n\n");

    if let Some(ctx) = context {
        script.push_str(&ctx.serialize_for_language(Language::Python));
    }

    script.push_str("# User Code\n");
    script.push_str(&wrapped_code);  //  Wrapped code verwenden
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

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);


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

    // ✅ FIX: Split stdout into display and return value
    let stdout_str = stdout.trim();
    let lines: Vec<&str> = stdout_str.lines().collect();

    // If multiple lines: Print all except last, return last
    if lines.len() > 1 {
        // Print visible output (all except last line)
        for line in &lines[..lines.len() - 1] {
            println!("{}", line);
        }

        // Return ONLY last line as value
        let return_value = lines.last().unwrap_or(&"");
        Ok(ReturnValueParser::from_python(return_value))
    } else {
        // Single line: return as-is
        Ok(ReturnValueParser::from_python(stdout_str))
    }
}

// Wrapper for backward compatibility
fn execute_python_code(code: &str, args: &[Value]) -> TBResult<Value> {
    execute_python_code_with_context(code, args, None)
}

/// Wrap Python code to auto-return last expression
fn wrap_python_auto_return(code: &str) -> String {
    let trimmed = code.trim();
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.is_empty() {
        return code.to_string();
    }

    let last_line = lines.last().unwrap().trim();

    // Skip wrapping if:
    if last_line.is_empty()
        || last_line.starts_with('#')
        || last_line.starts_with("print(")
        || (last_line.contains(" = ") && !last_line.contains("=="))
        || last_line.ends_with(':')
    {
        return code.to_string();
    }

    // ✅ NEW: Extract all defined variables from code
    let defined_vars = extract_python_variables(&lines);

    // ✅ NEW: Check if last line is a variable reference or bare string
    let is_variable = defined_vars.contains(&last_line.to_string());
    let is_bare_string = !is_variable && is_bare_string_literal(last_line);

    let mut wrapped = String::new();

    // Add all lines except last
    for (i, line) in lines.iter().enumerate() {
        if i < lines.len() - 1 {
            wrapped.push_str(line);
            wrapped.push('\n');
        }
    }

    // Wrap last line
    if is_bare_string {
        // ✅ Bare string literal → Add quotes
        wrapped.push_str(&format!("__tb_result = (\"{}\")\n", last_line));
    } else {
        // ✅ Variable reference OR expression → No quotes
        wrapped.push_str(&format!("__tb_result = ({})\n", last_line));
    }

    wrapped.push_str("print(__tb_result, end='')");

    wrapped
}

/// Extract all variable names defined in Python code
fn extract_python_variables(lines: &[&str]) -> Vec<String> {
    let mut vars = Vec::new();

    for line in lines {
        let trimmed = line.trim();

        // Skip comments and empty lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Match: var_name = ...
        if let Some(eq_pos) = trimmed.find('=') {
            // Check it's not ==, !=, <=, >=
            let before = &trimmed[..eq_pos];
            if before.ends_with('!') || before.ends_with('<')
                || before.ends_with('>') || before.ends_with('=') {
                continue;
            }

            // Extract variable name(s)
            let var_part = before.trim();

            // Handle multiple assignment: a, b = ...
            if var_part.contains(',') {
                for var in var_part.split(',') {
                    let clean_var = var.trim().to_string();
                    if !clean_var.is_empty() && is_valid_python_identifier(&clean_var) {
                        vars.push(clean_var);
                    }
                }
            } else {
                // Single variable
                let clean_var = var_part.to_string();
                if is_valid_python_identifier(&clean_var) {
                    vars.push(clean_var);
                }
            }
        }

        // Match: for var in ...
        if trimmed.starts_with("for ") {
            if let Some(in_pos) = trimmed.find(" in ") {
                let var_part = trimmed[4..in_pos].trim();

                // Handle: for x, y in ...
                if var_part.contains(',') {
                    for var in var_part.split(',') {
                        let clean_var = var.trim().to_string();
                        if is_valid_python_identifier(&clean_var) {
                            vars.push(clean_var);
                        }
                    }
                } else {
                    if is_valid_python_identifier(var_part) {
                        vars.push(var_part.to_string());
                    }
                }
            }
        }

        // Match: def function_name(...):
        if trimmed.starts_with("def ") {
            if let Some(paren_pos) = trimmed.find('(') {
                let func_name = trimmed[4..paren_pos].trim();
                if is_valid_python_identifier(func_name) {
                    vars.push(func_name.to_string());
                }
            }
        }
    }

    vars
}

/// Check if string is a valid Python identifier
fn is_valid_python_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // Must start with letter or underscore
    let first = s.chars().next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Rest must be alphanumeric or underscore
    s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Check if a string is a bare string literal (not a variable or expression)
fn is_bare_string_literal(s: &str) -> bool {
    let trimmed = s.trim();

    // Already quoted? → Not bare
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        return false;
    }

    // Contains operators/brackets/numbers? → Expression, not bare string
    if trimmed.contains('+')
        || trimmed.contains('-')
        || trimmed.contains('*')
        || trimmed.contains('/')
        || trimmed.contains('(')
        || trimmed.contains('[')
        || trimmed.contains('.')
        || trimmed.chars().any(|c| c.is_numeric())
    {
        return false;
    }

    // Is it a Python keyword?
    if matches!(
        trimmed,
        "True" | "False" | "None" | "and" | "or" | "not" | "if" | "else"
            | "elif" | "while" | "for" | "def" | "class" | "return" | "break"
            | "continue" | "pass" | "import" | "from" | "as" | "with"
    ) {
        return false;
    }

    // ✅ Single word with only letters/underscores → Bare string!
    trimmed.chars().all(|c| c.is_alphabetic() || c == '_')
}



/// Execute JavaScript code with TB variable context
fn execute_js_code_with_context(
    code: &str,
    args: &[Value],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value> {
    use std::process::Command;

    let arg_strings: Vec<String> = args.iter()
        .map(|v| format_value_for_output(v))
        .collect();

    //  AUTO-WRAP
    let wrapped_code = wrap_js_auto_return(code);

    let mut script = String::from("const args = process.argv.slice(2);\n\n");

    if let Some(ctx) = context {
        script.push_str(&ctx.serialize_for_language(Language::JavaScript));
    }

    script.push_str("// User Code\n");
    script.push_str(&wrapped_code);  //
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

    // ✅ FIX: Same as Python - split output
    let stdout_str = stdout.trim();
    let lines: Vec<&str> = stdout_str.lines().collect();

    if lines.len() > 1 {
        for line in &lines[..lines.len() - 1] {
            println!("{}", line);
        }
        let return_value = lines.last().unwrap_or(&"");
        Ok(ReturnValueParser::from_javascript(return_value))
    } else {
        Ok(ReturnValueParser::from_javascript(stdout_str))
    }
}

fn execute_js_code(code: &str, args: &[Value]) -> TBResult<Value> {
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
    let last_line = last_line.trim_end_matches(';').trim();

    // Skip wrapping if:
    if last_line.ends_with(';')
        || last_line.is_empty()
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
fn execute_go_code_with_context(
    code: &str,
    args: &[Value],
    context: Option<&LanguageExecutionContext>,
) -> TBResult<Value> {
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
    //  PREPROCESS USER CODE TO ESCAPE LITERAL NEWLINES IN STRINGS
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
    let (user_imports, clean_code) = extract_go_imports(&preprocessed_code);  //  Use preprocessed!

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

    // Add helper functions
    full_code.push_str("// Helper functions for TB\n");
    full_code.push_str("func parseInt(s string) int64 {\n");
    full_code.push_str("    i, _ := strconv.ParseInt(strings.TrimSpace(s), 10, 64)\n");
    full_code.push_str("    return i\n");
    full_code.push_str("}\n\n");
    full_code.push_str("func parseFloat(s string) float64 {\n");
    full_code.push_str("    f, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)\n");
    full_code.push_str("    return f\n");
    full_code.push_str("}\n\n");

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

    Ok(ReturnValueParser::from_go(stdout.trim()))
}
/// Extract import statements from Go code and return (imports, clean_code)
///
/// Example:
///   Input: "import \"math\"\nfmt.Println(math.Sqrt(64))"
///   Output: (vec!["math"], "fmt.Println(math.Sqrt(64))")
/// Extract import statements from Go code and return (imports, clean_code)
fn extract_go_imports(code: &str) -> (Vec<String>, String) {
    let mut imports = Vec::new();
    let mut clean_lines = Vec::new();
    let mut in_import_block = false;
    let mut paren_depth = 0;

    let wrapped_go = wrap_go_auto_return(&code);

    for line in wrapped_go.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with("//") {
            if !in_import_block && imports.is_empty() {
                continue;
            }
        }

        if trimmed.starts_with("import (") {
            in_import_block = true;
            paren_depth = 1;
            continue;
        }

        if trimmed.starts_with("import ") && !trimmed.contains('(') {
            let import_path = trimmed
                .trim_start_matches("import")
                .trim()
                .trim_matches('"');
            imports.push(import_path.to_string());
            continue;
        }

        if in_import_block {
            if trimmed.contains(')') {
                paren_depth -= 1;
                if paren_depth == 0 {
                    in_import_block = false;
                }
                continue;
            }

            if trimmed.starts_with('"') {
                let import_path = trimmed.trim_matches('"');
                imports.push(import_path.to_string());
            }
            continue;
        }

        clean_lines.push(line);
    }

    // ✅ ADD: Always include strconv and strings for TB variable conversions
    if !imports.contains(&"strconv".to_string()) {
        imports.push("strconv".to_string());
    }
    if !imports.contains(&"strings".to_string()) {
        imports.push("strings".to_string());
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
        || last_line == "}"                       //  closing brace
        || last_line == ")"                       //  closing paren
        || last_line == "]"                       //  closing bracket
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
fn execute_go_code(code: &str, args: &[Value]) -> TBResult<Value> {
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
use crate::dependency_compiler::CompiledDependency;

pub struct Compiler {
    config: Config,
    target: TargetPlatform,
    optimization_level: u8,
    compiled_deps: Vec<CompiledDependency>,
}

impl Compiler {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            target: TargetPlatform::current(),
            optimization_level: 3,
            compiled_deps: Vec::new(),
        }
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
    /// NOTE: This method is currently unused. Compilation is handled by runner.rs::compile_file()
    pub fn compile(&mut self, statements: &[Statement]) -> TBResult<Vec<u8>> {
        debug_log!("Compiler::compile() for target {}", self.target);

        // Statements already include imports (loaded by TBCore)
        let function_count = statements.iter()
            .filter(|s| matches!(s, Statement::Function{..}))
            .count();
        debug_log!("✓ Total functions for compilation: {}", function_count);

        // Generate Rust code with runtime support
        let rust_code = self.generate_rust_code_with_runtime(&statements)?;
        debug_log!("Generated Rust code: {} bytes", rust_code.len());

        debug_log!("full code {}", rust_code);

        // Write to temporary directory
        let temp_dir = self.create_temp_project()?;
        let main_rs = temp_dir.join("src").join("main.rs");
        fs::write(&main_rs, &rust_code)?;

        // Create Cargo.toml with runtime dependencies
        self.create_cargo_toml_with_runtime(&temp_dir)?;

        // Compile with cargo
        debug_log!("Running cargo build...");
        let binary = self.cargo_build(&temp_dir)?;

        // Read compiled binary
        let binary_data = fs::read(&binary)?;

        // Cleanup temp directory
        fs::remove_dir_all(&temp_dir).ok();

        debug_log!("Compilation successful: {} bytes", binary_data.len());

        Ok(binary_data)
    }

    /// Recursively load all imports and their transitive dependencies
    fn load_all_imports_recursive(&self, statements: &[Statement]) -> TBResult<Vec<Statement>> {
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
        collected: &mut Vec<Statement>,
        visited: &mut std::collections::HashSet<PathBuf>,
        function_names: &mut std::collections::HashSet<String>,
        depth: usize,
    ) -> TBResult<()> {
        let indent = "  ".repeat(depth);

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
            let mut parser = Parser::new(tokens);
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
                for stmt in statements.as_ref() {
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
        // NOTE: This method is currently unused. Code generation is handled by runner.rs::compile_file()
        // which uses RustCodeGenerator directly.

        // Analysiere, welche Variablen tatsächlich veränderbar sein müssen.
        let mutable_vars = self.analyze_mutability(statements);
        debug_log!("Variables requiring mut: {:?}", mutable_vars);

        // Erstelle eine einzige CodeGenerator-Instanz.
        let mut codegen = CodeGenerator::new(Language::Rust);

        // Konfiguriere den Generator mit allen notwendigen Kontext-Informationen.
        codegen.set_compiled_dependencies(&self.compiled_deps);
        codegen.set_mutable_vars(&mutable_vars);

        // Generiere den gesamten Code
        let code = codegen.generate(statements)?;

        Ok(code)
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
                if let Expr::Variable(name) = function.as_ref() {
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
        // Determine which features to enable based on config
        let mut features = Vec::new();

        // Check if networking is needed
        if self.config.networking_enabled {
            features.push("networking");
        }

        // Build features string
        let features_str = if features.is_empty() {
            String::new()
        } else {
            format!(r#"tb-runtime = {{ path = "../../../tb-exc/src/crates/tb-runtime", features = [{}] }}"#,
                features.iter().map(|f| format!("\"{}\"", f)).collect::<Vec<_>>().join(", "))
        };

        let cargo_toml = format!(r#"[package]
name = "tb_compiled"
version = "0.1.0"
edition = "2021"

[dependencies]
rayon = "1.11.0"
{}

[profile.release]
opt-level = {}
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
"#, features_str, self.optimization_level);

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
    pub fn compile_to_file(&self, statements: &[Statement], output: &Path) -> TBResult<()> {
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

impl Compiler {
    pub fn compile_to_library(&self, statements: &[Statement], output: &Path) -> TBResult<()> {
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
    fn generate_rust_library_code(&self, statements: &[Statement]) -> TBResult<String> {
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
                for stmt in statements.as_ref() {
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
    config: Config,
    optimizer: Optimizer,
}

impl TBCore {
    /// Create new TB Core engine
    pub fn new(config: Config) -> Self {
        Self {
            optimizer: Optimizer::new(OptimizerConfig::default()),
            config,
        }
    }

    /// Execute TB source code
    pub fn execute(&mut self, source: &str) -> TBResult<Value> {
        debug_log!("TBCore::execute() started!!");
        debug_log!("Source length: {} bytes", source.len());

        // Parse configuration
        self.config = Config::parse(source)?;
        debug_log!("Configuration parsed: mode={:?}", self.config.mode);

        // Strip directives
        let clean_source = Self::strip_directives(source);
        debug_log!("Clean source length: {} bytes", clean_source.len());

        // Tokenize and parse main source
        let mut lexer = Lexer::new(&clean_source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let mut statements = parser.parse()?;
        debug_log!("Parsing complete: {} statements", statements.len());

        // Load imports BEFORE executing (for ALL modes)
        let imports = self.config.imports.clone();
        if !imports.is_empty() {
            debug_log!("📦 Loading {} imports", imports.len());
            let imported_statements = self.load_imports(&imports)?;
            debug_log!("✓ Loaded {} statements from imports", imported_statements.len());

            // Deduplicate function definitions
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
        let result = match &mode {
            ExecutionMode::Compiled { optimize } => {
                debug_log!("Compiled mode execution");

                // Load imports BEFORE compilation
                let imports = self.config.imports.clone();
                let all_statements = if !imports.is_empty() {
                    debug_log!("📦 Loading {} imports for compilation", imports.len());
                    let imported_statements = self.load_imports(&imports)?;
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
                    .with_optimization(if *optimize { 3 } else { 0 });

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
                        debug_log!("Using ParallelExecutor");
                        let mut executor = ParallelExecutor::new(self.config.clone());
                        executor.execute(&statements)
                    }

                    RuntimeMode::Sequential => {
                        debug_log!("Using standard JitExecutor");
                        let mut executor = JitExecutor::new(self.config.clone());
                        executor.execute(&statements)
                    }
                }
            }

            ExecutionMode::Streaming { .. } => {
                debug_log!("Streaming mode execution (using JIT)");
                let mut executor = JitExecutor::new(self.config.clone());
                executor.execute(&statements)
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

        debug_log!("TBCore::execute() completed: {:?}", result);
        result
    }




    /// Merge imported and main statements, removing duplicate function definitions
    pub fn merge_statements_deduplicated(
        imported: Vec<Statement>,
        main: Vec<Statement>,
    ) -> TBResult<Vec<Statement>> {
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
    /// Takes owned Vec to avoid borrowing issues
    /// Load imports with compilation caching
    pub fn load_imports(&mut self, imports: &[PathBuf]) -> TBResult<Vec<Statement>> {
        let mut all_statements = Vec::new();
        let mut visited = std::collections::HashSet::new();

        debug_log!("╔════════════════════════════════════════════════════════════════╗");
        debug_log!("║                    Loading Imports (JIT)                       ║");
        debug_log!("╚════════════════════════════════════════════════════════════════╝\n");

        self.load_imports_recursive(imports, &mut all_statements, &mut visited, 0)?;

        Ok(all_statements)
    }

    /// Recursive helper for loading imports
    fn load_imports_recursive(
        &mut self,
        imports: &[PathBuf],
        collected: &mut Vec<Statement>,
        visited: &mut std::collections::HashSet<PathBuf>,
        depth: usize,
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
                    depth + 1
                )?;
            }

            // Parse statements
            let clean_source = Self::strip_directives(&source);
            let mut lexer = Lexer::new(&clean_source);
            let tokens = lexer.tokenize()?;
            let mut parser = Parser::new(tokens);
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

            // ✅ FIX: Detect start of directive block
            if trimmed.starts_with("@config") || trimmed.starts_with("@shared")
                || trimmed.starts_with("@imports") || trimmed.starts_with("@plugins") {
                skip_until_brace = true;
                brace_depth = 0;
                debug_log!("Entering directive block: {}", trimmed);

                // ✅ NEW: Count ALL braces on this line
                for ch in trimmed.chars() {
                    if ch == '{' {
                        brace_depth += 1;
                    } else if ch == '}' {
                        brace_depth -= 1;
                    }
                }

                // ✅ NEW: If block complete on same line, exit directive mode
                if brace_depth == 0 {
                    skip_until_brace = false;
                    debug_log!("Exiting directive block (complete on same line)");
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

    /// Compile source to binary
    /// Compile source to file with import support
    /// Compile source to file with import support
    pub fn compile_to_file(&mut self, source: &str, output_path: &Path) -> TBResult<()> {
        debug_log!("TBCore::compile_to_file() started");

        // Parse configuration
        self.config = Config::parse(source)?;
        debug_log!("Configuration parsed: mode={:?}", self.config.mode);

        // Strip directives
        let clean_source = Self::strip_directives(source);
        debug_log!("Clean source length: {} bytes", clean_source.len());

        // Tokenize and parse main source
        let mut lexer = Lexer::new(&clean_source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let mut statements = parser.parse()?;
        debug_log!("Parsing complete: {} statements", statements.len());

        //  Load imports before compilation
        let imports = self.config.imports.clone();
        if !imports.is_empty() {
            debug_log!("📦 Loading {} imports for compilation", imports.len());
            let imported_statements = self.load_imports(&imports)?;
            debug_log!("✓ Loaded {} statements from imports", imported_statements.len());

            statements = Self::merge_statements_deduplicated(imported_statements, statements)?;
            debug_log!("✓ Total statements after merge: {}", statements.len());
        }

        let func_count = statements.iter()
            .filter(|s| matches!(s, Statement::Function { .. }))
            .count();
        debug_log!("✓ Functions available for compilation: {}", func_count);

        //  NEW: Analyze and compile dependencies
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
                if let Expr::Variable(func_name) = function.as_ref() {
                    if let Some(language) = self.language_from_builtin_name(func_name) {
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
                for stmt in statements.as_ref() {
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
    pub fn with_config(config: Config) -> Self {
        Self {
            core: TBCore::new(config),
        }
    }

    /// Execute TB source code
    pub fn execute(&mut self, source: &str) -> TBResult<Value> {
        self.core.execute(source)
    }

    /// Execute TB file
    pub fn execute_file(&mut self, path: &Path) -> TBResult<Value> {
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

impl Default for TB {
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
    println!("║                    TB Language Core v1.0                       ║");
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
