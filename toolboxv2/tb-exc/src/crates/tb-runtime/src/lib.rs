//! TB Runtime Library
//! Provides core runtime functions for compiled TB programs

use std::sync::Arc;
use dashmap::DashMap;
use tb_plugin::PluginLoader;
use tb_core::{PluginLanguage, PluginMode, Value};

/// Global plugin loader instance
static PLUGIN_LOADER: once_cell::sync::Lazy<Arc<PluginLoader>> =
    once_cell::sync::Lazy::new(|| Arc::new(PluginLoader::new()));

/// Global plugin function cache: (module_name, function_name) -> function_id
static PLUGIN_FUNCTIONS: once_cell::sync::Lazy<DashMap<(String, String), usize>> =
    once_cell::sync::Lazy::new(|| DashMap::new());

/// Print to stdout
#[no_mangle]
pub extern "C" fn tb_print_int(value: i64) {
    println!("{}", value);
}

#[no_mangle]
pub extern "C" fn tb_print_float(value: f64) {
    println!("{}", value);
}

#[no_mangle]
pub extern "C" fn tb_print_string(ptr: *const u8, len: usize) {
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    if let Ok(s) = std::str::from_utf8(slice) {
        println!("{}", s);
    }
}

/// Memory allocation
#[no_mangle]
pub extern "C" fn tb_alloc(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

#[no_mangle]
pub extern "C" fn tb_dealloc(ptr: *mut u8, size: usize) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) }
}

/// Array operations
#[no_mangle]
pub extern "C" fn tb_array_len(ptr: *const u8) -> usize {
    // Assuming first 8 bytes are length
    unsafe { *(ptr as *const usize) }
}

/// String operations
#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn tb_string_concat(
    s1_ptr: *const u8,
    s1_len: usize,
    s2_ptr: *const u8,
    s2_len: usize,
) -> (*mut u8, usize) {
    let s1 = unsafe { std::slice::from_raw_parts(s1_ptr, s1_len) };
    let s2 = unsafe { std::slice::from_raw_parts(s2_ptr, s2_len) };

    let mut result = Vec::with_capacity(s1_len + s2_len);
    result.extend_from_slice(s1);
    result.extend_from_slice(s2);

    let len = result.len();
    let ptr = result.as_mut_ptr();
    std::mem::forget(result);

    (ptr, len)
}
pub struct Runtime;

impl Runtime {
    pub fn new() -> Self {
        Self
    }
}

/// Plugin runtime functions
/// These are called from generated code to execute plugin functions

/// Call a Python plugin function (JIT mode)
#[no_mangle]
pub extern "C" fn tb_plugin_call_python_jit(
    source_ptr: *const u8,
    source_len: usize,
    func_name_ptr: *const u8,
    func_name_len: usize,
    arg: i64,
) -> i64 {
    let source = unsafe {
        let slice = std::slice::from_raw_parts(source_ptr, source_len);
        std::str::from_utf8(slice).unwrap_or("")
    };

    let func_name = unsafe {
        let slice = std::slice::from_raw_parts(func_name_ptr, func_name_len);
        std::str::from_utf8(slice).unwrap_or("")
    };

    // Execute Python function
    let args = vec![Value::Int(arg)];
    match PLUGIN_LOADER.execute_python_jit_inline(source, func_name, args) {
        Ok(Value::Int(result)) => result,
        _ => 0, // Error fallback
    }
}

/// Call a JavaScript plugin function (JIT mode)
#[no_mangle]
pub extern "C" fn tb_plugin_call_js_jit(
    source_ptr: *const u8,
    source_len: usize,
    func_name_ptr: *const u8,
    func_name_len: usize,
    arg: i64,
) -> i64 {
    let source = unsafe {
        let slice = std::slice::from_raw_parts(source_ptr, source_len);
        std::str::from_utf8(slice).unwrap_or("")
    };

    let func_name = unsafe {
        let slice = std::slice::from_raw_parts(func_name_ptr, func_name_len);
        std::str::from_utf8(slice).unwrap_or("")
    };

    // Execute JavaScript function
    let args = vec![Value::Int(arg)];
    match PLUGIN_LOADER.execute_js_jit_inline(source, func_name, args) {
        Ok(Value::Int(result)) => result,
        _ => 0, // Error fallback
    }
}

