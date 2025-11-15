//! FFI Interface for Compiled Mode
//!
//! This module provides a C-compatible FFI interface that allows compiled TB programs
//! to call into the JIT runtime without code duplication.
//!
//! Architecture:
//! - Compiled programs link against tb-runtime.dll/so
//! - All built-in functions are called via FFI
//! - No code duplication between JIT and compiled modes
//! - Fast compilation (rustc only, no cargo build)

#![cfg(feature = "full")]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tb_core::Value;
use tb_builtins;

// ============================================================================
// VALUE SERIALIZATION (for FFI)
// ============================================================================

/// Serialize a Value to JSON for FFI transfer
fn value_to_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

/// Deserialize a Value from JSON
fn json_to_value(json: &str) -> Value {
    serde_json::from_str(json).unwrap_or(Value::None)
}

// ============================================================================
// FFI HELPER FUNCTIONS
// ============================================================================

/// Convert C string to Rust string
unsafe fn c_str_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

/// Convert Rust string to C string (caller must free!)
fn string_to_c_str(s: String) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

// ============================================================================
// CORE FFI FUNCTIONS
// ============================================================================

/// Call a built-in function by name with JSON-serialized arguments
/// Returns JSON-serialized result
#[no_mangle]
pub extern "C" fn tb_call_builtin(
    name_ptr: *const c_char,
    args_json_ptr: *const c_char,
) -> *mut c_char {
    unsafe {
        let name = c_str_to_string(name_ptr);
        let args_json = c_str_to_string(args_json_ptr);

        // Deserialize arguments
        let args: Vec<Value> = serde_json::from_str(&args_json).unwrap_or_default();

        // Call the built-in function
        let result = match name.as_str() {
            // Type conversions
            "int" => tb_builtins::builtins_impl::builtin_int(args),
            "str" => tb_builtins::builtins_impl::builtin_str(args),
            "float" => tb_builtins::builtins_impl::builtin_float(args),
            "dict" => tb_builtins::builtins_impl::builtin_dict(args),
            "list" => tb_builtins::builtins_impl::builtin_list(args),

            // Collections
            "len" => tb_builtins::builtins_impl::builtin_len(args),
            "push" => tb_builtins::builtins_impl::builtin_push(args),
            "pop" => tb_builtins::builtins_impl::builtin_pop(args),
            "keys" => tb_builtins::builtins_impl::builtin_keys(args),
            "values" => tb_builtins::builtins_impl::builtin_values(args),
            "range" => tb_builtins::builtins_impl::builtin_range(args),

            // I/O
            "print" => tb_builtins::builtins_impl::builtin_print(args),
            // ✅ PHASE 1.3: open() removed - no usable functionality
            // "open" => tb_builtins::builtins_impl::builtin_open(args),
            "read_file" => tb_builtins::builtins_impl::builtin_read_file(args),
            "write_file" => tb_builtins::builtins_impl::builtin_write_file(args),
            "file_exists" => tb_builtins::builtins_impl::builtin_file_exists(args),
            "list_dir" => tb_builtins::builtins_impl::builtin_list_dir(args),
            "create_dir" => tb_builtins::builtins_impl::builtin_create_dir(args),
            "delete_file" => tb_builtins::builtins_impl::builtin_delete_file(args),

            // Module Import
            "import" => tb_builtins::builtins_impl::builtin_import(args),

            // System
            "execute" => tb_builtins::builtins_impl::builtin_execute(args),
            "get_env" => tb_builtins::builtins_impl::builtin_get_env(args),
            "sleep" => tb_builtins::builtins_impl::builtin_sleep(args),

            // Introspection
            "type_of" => tb_builtins::builtins_impl::builtin_type_of(args),
            "dir" => tb_builtins::builtins_impl::builtin_dir(args),
            "has_attr" => tb_builtins::builtins_impl::builtin_has_attr(args),

            // Networking
            "create_server" => tb_builtins::builtins_impl::builtin_create_server(args),
            "stop_server" => tb_builtins::builtins_impl::builtin_stop_server(args),
            "connect_to" => tb_builtins::builtins_impl::builtin_connect_to(args),
            "send_to" => tb_builtins::builtins_impl::builtin_send_to(args),
            "http_session" => tb_builtins::builtins_impl::builtin_http_session(args),
            "http_request" => tb_builtins::builtins_impl::builtin_http_request(args),

            // JSON/YAML
            "json_parse" => tb_builtins::builtins_impl::builtin_json_parse(args),
            "json_stringify" => tb_builtins::builtins_impl::builtin_json_stringify(args),
            "yaml_parse" => tb_builtins::builtins_impl::builtin_yaml_parse(args),
            "yaml_stringify" => tb_builtins::builtins_impl::builtin_yaml_stringify(args),

            // Time
            "time" => tb_builtins::builtins_impl::builtin_time(args),

            // Cache Management
            "cache_stats" => tb_builtins::builtins_impl::builtin_cache_stats(args),
            "cache_clear" => tb_builtins::builtins_impl::builtin_cache_clear(args),
            "cache_invalidate" => tb_builtins::builtins_impl::builtin_cache_invalidate(args),

            // Plugin Management
            "list_plugins" => tb_builtins::builtins_impl::builtin_list_plugins(args),
            "reload_plugin" => tb_builtins::builtins_impl::builtin_reload_plugin(args),
            "unload_plugin" => tb_builtins::builtins_impl::builtin_unload_plugin(args),
            "plugin_info" => tb_builtins::builtins_impl::builtin_plugin_info(args),

            // Async Task Management
            "spawn" => tb_builtins::builtins_impl::builtin_spawn(args),
            "await_task" => tb_builtins::builtins_impl::builtin_await_task(args),
            "cancel_task" => tb_builtins::builtins_impl::builtin_cancel_task(args),

            // Serialization & Hashing
            "bincode_serialize" => tb_builtins::builtins_impl::builtin_bincode_serialize(args),
            "bincode_deserialize" => tb_builtins::builtins_impl::builtin_bincode_deserialize(args),
            "hash" => tb_builtins::builtins_impl::builtin_hash(args),

            // ✅ PHASE 1.2: Higher-order functions removed - now implemented natively in JIT executor
            // These functions are not available in FFI mode
            // "map" => tb_builtins::builtins_impl::builtin_map(args),
            // "filter" => tb_builtins::builtins_impl::builtin_filter(args),
            // "reduce" => tb_builtins::builtins_impl::builtin_reduce(args),
            // "forEach" => tb_builtins::builtins_impl::builtin_for_each(args),

            _ => Err(tb_core::TBError::runtime_error(format!("Unknown built-in function: {}", name))),
        };

        // Serialize result
        let result_json = match result {
            Ok(value) => value_to_json(&value),
            Err(err) => format!(r#"{{"error":"{}"}}"#, err),
        };

        string_to_c_str(result_json)
    }
}

/// Free a C string allocated by tb_call_builtin
#[no_mangle]
pub extern "C" fn tb_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

// ============================================================================
// OPTIMIZED DIRECT FFI FUNCTIONS (no JSON overhead)
// ============================================================================
// Note: tb_print_int, tb_print_float, tb_print_string are already defined in lib.rs

/// Print a boolean (optimized, no JSON)
#[no_mangle]
pub extern "C" fn tb_print_bool(value: bool) {
    println!("{}", value);
}

/// Get length of a list (optimized, no JSON)
/// For now, we'll use JSON for complex types, but this shows the pattern
#[no_mangle]
pub extern "C" fn tb_len_list(list_json_ptr: *const c_char) -> i64 {
    unsafe {
        let json = c_str_to_string(list_json_ptr);
        let value: Value = json_to_value(&json);
        match value {
            Value::List(l) => l.len() as i64,
            _ => -1,
        }
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/// Initialize the TB runtime (must be called before any other FFI functions)
#[no_mangle]
pub extern "C" fn tb_runtime_init() {
    // Initialize any global state if needed
    // For now, this is a no-op
}

/// Shutdown the TB runtime
#[no_mangle]
pub extern "C" fn tb_runtime_shutdown() {
    // Cleanup any global state if needed
    // For now, this is a no-op
}

