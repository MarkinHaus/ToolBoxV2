//! TB Language Built-in Functions
//!
//! High-performance, non-blocking built-in functions for:
//! - File I/O (real files only)
//! - Networking (HTTP/HTTPS, TCP, UDP)
//! - Utilities (JSON/YAML, time)
//!
//! All I/O operations are async and non-blocking for maximum efficiency.

pub mod file_io;
pub mod networking;
pub mod utils;
pub mod error;
pub mod builtins_impl;
pub mod task_runtime;

use std::sync::Arc;
use tb_core::{Value, TBError, Program};

use tb_cache::CacheManager;
use tb_plugin::PluginLoader;
use tokio::task::JoinHandle;
use dashmap::DashMap;
use once_cell::sync::Lazy;

pub use error::{BuiltinError, BuiltinResult};

use crate::builtins_impl::*;

// Import all built-in functions
use crate::builtins_impl::*;

/// Global runtime for async operations
pub static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("tb-builtin-worker")
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
});

/// Global HTTP session registry
pub static HTTP_SESSIONS: Lazy<DashMap<String, Arc<networking::HttpSession>>> = Lazy::new(DashMap::new);

// NEUE GLOBALE ZUSTÄNDE HINZUFÜGEN
pub static CACHE_MANAGER: Lazy<Arc<CacheManager>> = Lazy::new(|| {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("tb-lang");
    Arc::new(CacheManager::new(cache_dir, 100).expect("Failed to create CacheManager"))
});

pub static PLUGIN_LOADER: Lazy<Arc<PluginLoader>> = Lazy::new(|| {
    Arc::new(PluginLoader::new())
});

pub static ACTIVE_TASKS: Lazy<DashMap<String, JoinHandle<Result<Value, TBError>>>> =
    Lazy::new(DashMap::new);

/// Global environment snapshot for spawned tasks
/// This is updated by the JIT executor before calling spawn
pub static TASK_ENVIRONMENT: Lazy<Arc<parking_lot::RwLock<im::HashMap<Arc<String>, Value>>>> =
    Lazy::new(|| Arc::new(parking_lot::RwLock::new(im::HashMap::new())));

/// Register all built-in functions
pub fn register_all_builtins() -> Vec<(&'static str, BuiltinFn)> {
    let mut builtins = Vec::new();

    // Type Conversions
    builtins.push(("int", builtins_impl::builtin_int as BuiltinFn));
    builtins.push(("str", builtins_impl::builtin_str as BuiltinFn));
    builtins.push(("float", builtins_impl::builtin_float as BuiltinFn));
    builtins.push(("dict", builtins_impl::builtin_dict as BuiltinFn));
    builtins.push(("list", builtins_impl::builtin_list as BuiltinFn));

    // Basic Collections
    builtins.push(("len", builtins_impl::builtin_len as BuiltinFn));
    builtins.push(("push", builtins_impl::builtin_push as BuiltinFn));
    builtins.push(("pop", builtins_impl::builtin_pop as BuiltinFn));
    builtins.push(("keys", builtins_impl::builtin_keys as BuiltinFn));
    builtins.push(("values", builtins_impl::builtin_values as BuiltinFn));
    builtins.push(("range", builtins_impl::builtin_range as BuiltinFn));
    builtins.push(("print", builtins_impl::builtin_print as BuiltinFn));

    // Module Import
    builtins.push(("import", builtins_impl::builtin_import as BuiltinFn));

    // File I/O
    builtins.push(("open", builtin_open as BuiltinFn));
    builtins.push(("read_file", builtin_read_file as BuiltinFn));
    builtins.push(("write_file", builtin_write_file as BuiltinFn));
    builtins.push(("file_exists", builtin_file_exists as BuiltinFn));
    builtins.push(("list_dir", builtins_impl::builtin_list_dir as BuiltinFn));
    builtins.push(("create_dir", builtins_impl::builtin_create_dir as BuiltinFn));
    builtins.push(("delete_file", builtins_impl::builtin_delete_file as BuiltinFn));

    // System
    builtins.push(("execute", builtins_impl::builtin_execute as BuiltinFn));
    builtins.push(("get_env", builtins_impl::builtin_get_env as BuiltinFn));
    builtins.push(("sleep", builtins_impl::builtin_sleep as BuiltinFn));

    // Introspection
    builtins.push(("type_of", builtins_impl::builtin_type_of as BuiltinFn));
    builtins.push(("dir", builtins_impl::builtin_dir as BuiltinFn));
    builtins.push(("has_attr", builtins_impl::builtin_has_attr as BuiltinFn));

    // Networking
    builtins.push(("create_server", builtins_impl::builtin_create_server as BuiltinFn));
    builtins.push(("stop_server", builtins_impl::builtin_stop_server as BuiltinFn));
    builtins.push(("connect_to", builtins_impl::builtin_connect_to as BuiltinFn));
    builtins.push(("send_to", builtins_impl::builtin_send_to as BuiltinFn));
    builtins.push(("http_session", builtins_impl::builtin_http_session as BuiltinFn));
    builtins.push(("http_request", builtins_impl::builtin_http_request as BuiltinFn));

    // Utilities
    builtins.push(("json_parse", builtins_impl::builtin_json_parse as BuiltinFn));
    builtins.push(("json_stringify", builtins_impl::builtin_json_stringify as BuiltinFn));
    builtins.push(("yaml_parse", builtins_impl::builtin_yaml_parse as BuiltinFn));
    builtins.push(("yaml_stringify", builtins_impl::builtin_yaml_stringify as BuiltinFn));
    builtins.push(("time", builtins_impl::builtin_time as BuiltinFn));

    // Cache Management
    builtins.push(("cache_stats", builtins_impl::builtin_cache_stats as BuiltinFn));
    builtins.push(("cache_clear", builtins_impl::builtin_cache_clear as BuiltinFn));
    builtins.push(("cache_invalidate", builtins_impl::builtin_cache_invalidate as BuiltinFn));

    // Plugin Management
    builtins.push(("list_plugins", builtins_impl::builtin_list_plugins as BuiltinFn));
    builtins.push(("reload_plugin", builtins_impl::builtin_reload_plugin as BuiltinFn));
    builtins.push(("unload_plugin", builtins_impl::builtin_unload_plugin as BuiltinFn));
    builtins.push(("plugin_info", builtins_impl::builtin_plugin_info as BuiltinFn));

    // Async Task Management
    builtins.push(("spawn", builtins_impl::builtin_spawn as BuiltinFn));
    builtins.push(("await_task", builtins_impl::builtin_await_task as BuiltinFn));
    builtins.push(("cancel_task", builtins_impl::builtin_cancel_task as BuiltinFn));

    // Serialisierung & Hashing
    builtins.push(("bincode_serialize", builtins_impl::builtin_bincode_serialize as BuiltinFn));
    builtins.push(("bincode_deserialize", builtins_impl::builtin_bincode_deserialize as BuiltinFn));
    builtins.push(("hash", builtins_impl::builtin_hash as BuiltinFn));

    // Higher-Order Functions
    builtins.push(("map", builtins_impl::builtin_map as BuiltinFn));
    builtins.push(("filter", builtins_impl::builtin_filter as BuiltinFn));
    builtins.push(("reduce", builtins_impl::builtin_reduce as BuiltinFn));
    builtins.push(("forEach", builtins_impl::builtin_forEach as BuiltinFn));

    builtins
}

/// Built-in function signature
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, TBError>;



