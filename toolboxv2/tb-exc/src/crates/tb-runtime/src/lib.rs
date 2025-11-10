//! TB Runtime Library
//! Provides core runtime functions for compiled TB programs
//!
//! This crate re-exports all built-in functions from tb-builtins
//! to avoid code duplication and ensure consistency between JIT and compiled modes.

use std::collections::HashMap;
use std::collections::HashMap as StdHashMap;
use std::sync::{Arc, RwLock};
use std::fmt;
use serde::{Serialize, Deserialize};

// Conditional imports based on features
#[cfg(feature = "networking")]
use once_cell::sync::Lazy;
#[cfg(feature = "networking")]
use dashmap::DashMap;
#[cfg(feature = "networking")]
use tokio::net::{TcpStream, UdpSocket};
#[cfg(feature = "networking")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(feature = "networking")]
use tokio::task::JoinHandle;
#[cfg(feature = "networking-full")]
use sha2::{Sha256, Sha512, Digest};

// FFI interface for compiled mode
#[cfg(feature = "full")]
pub mod ffi;

// Re-export built-in functions only when "full" feature is enabled
#[cfg(feature = "full")]
use tb_core::Value;

#[cfg(feature = "full")]
pub use tb_builtins::builtins_impl::{
    // Type conversions
    builtin_int as int_from_value,
    builtin_str as str_from_value,
    builtin_float as float_from_value,

    // Collections
    builtin_len as len_from_value,
    builtin_push as push_from_value,
    builtin_pop as pop_from_value,
    builtin_keys as keys_from_value,
    builtin_values as values_from_value,
    builtin_range as range_from_value,

    // I/O
    builtin_print as print_from_value,
    builtin_read_file as read_file_from_value,
    builtin_write_file as write_file_from_value,
    builtin_file_exists as file_exists_from_value,

    // Utilities
    builtin_json_parse as json_parse_from_value,
    builtin_json_stringify as json_stringify_from_value,
    builtin_yaml_parse as yaml_parse_from_value,
    builtin_yaml_stringify as yaml_stringify_from_value,
    builtin_time as time_from_value,

    // ✅ PHASE 1.2: Higher-order functions removed - now implemented natively in JIT executor
    // builtin_map as map_from_value,
    // builtin_filter as filter_from_value,
    // builtin_reduce as reduce_from_value,
    // builtin_for_each as forEach_from_value,
};

// Plugin support (only when "plugins" feature is enabled)
#[cfg(feature = "plugins")]
use tb_plugin::PluginLoader;
#[cfg(feature = "plugins")]
use tb_core::PluginLanguage;

/// Global plugin loader instance (only with "plugins" feature)
#[cfg(feature = "plugins")]
static PLUGIN_LOADER: once_cell::sync::Lazy<Arc<PluginLoader>> =
    once_cell::sync::Lazy::new(|| Arc::new(PluginLoader::new()));

/// Global plugin function cache: (module_name, function_name) -> function_id (only with "plugins" feature)
#[cfg(feature = "plugins")]
static PLUGIN_FUNCTIONS: once_cell::sync::Lazy<DashMap<(String, String), usize>> =
    once_cell::sync::Lazy::new(|| DashMap::new());

// ============================================================================
// DICTVALUE - Heterogeneous Dictionary Support
// ============================================================================

/// DictValue enum for heterogeneous dictionaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DictValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    List(Vec<DictValue>),
    Dict(HashMap<String, DictValue>),
}

impl DictValue {
    pub fn as_int(&self) -> i64 {
        match self {
            DictValue::Int(i) => *i,
            DictValue::Float(f) => *f as i64, // Toleranz für Typ-Mismatches
            _ => 0, // Sicherer Standardwert
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            DictValue::String(s) => s.clone(),
            DictValue::Int(i) => i.to_string(),
            DictValue::Float(f) => f.to_string(),
            DictValue::Bool(b) => b.to_string(),
            _ => String::new(), // Sicherer Standardwert
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            DictValue::Float(f) => *f,
            DictValue::Int(i) => *i as f64, // Toleranz für Typ-Mismatches
            _ => 0.0, // Sicherer Standardwert
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            DictValue::Bool(b) => *b,
            DictValue::Int(i) => *i != 0,
            _ => false, // Sicherer Standardwert
        }
    }

    pub fn get(&self, key: &str) -> Option<&DictValue> {
        match self {
            DictValue::Dict(map) => map.get(key),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> &HashMap<String, DictValue> {
        match self {
            DictValue::Dict(map) => map,
            _ => {
                // Erstellt und leakt eine statische leere HashMap, um eine sichere Referenz zurückzugeben
                static EMPTY_MAP: std::sync::OnceLock<HashMap<String, DictValue>> = std::sync::OnceLock::new();
                EMPTY_MAP.get_or_init(HashMap::new)
            }
        }
    }

    pub fn as_list(&self) -> &Vec<DictValue> {
        match self {
            DictValue::List(v) => v,
            _ => {
                // Erstellt und leakt einen statischen leeren Vektor
                static EMPTY_VEC: std::sync::OnceLock<Vec<DictValue>> = std::sync::OnceLock::new();
                EMPTY_VEC.get_or_init(Vec::new)
            }
        }
    }
}

impl fmt::Display for DictValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DictValue::Int(i) => write!(f, "{}", i),
            DictValue::Float(fl) => {
                // ✅ FIX: Format floats with .1 precision if they are whole numbers
                if fl.fract() == 0.0 && fl.is_finite() {
                    write!(f, "{:.1}", fl)
                } else {
                    write!(f, "{}", fl)
                }
            },
            DictValue::String(s) => write!(f, "{}", s),
            DictValue::Bool(b) => write!(f, "{}", b),
            DictValue::List(_) => write!(f, "[...]"),
            DictValue::Dict(_) => write!(f, "{{...}}"),
        }
    }
}

impl PartialOrd for DictValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (DictValue::Int(a), DictValue::Int(b)) => a.partial_cmp(b),
            (DictValue::Float(a), DictValue::Float(b)) => a.partial_cmp(b),
            (DictValue::String(a), DictValue::String(b)) => a.partial_cmp(b),
            (DictValue::Bool(a), DictValue::Bool(b)) => a.partial_cmp(b),
            (DictValue::Int(a), DictValue::Float(b)) => (*a as f64).partial_cmp(b),
            (DictValue::Float(a), DictValue::Int(b)) => a.partial_cmp(&(*b as f64)),
            _ => None,
        }
    }
}

impl PartialEq<i64> for DictValue {
    fn eq(&self, other: &i64) -> bool {
        match self { DictValue::Int(i) => i == other, _ => false }
    }
}

impl PartialOrd<i64> for DictValue {
    fn partial_cmp(&self, other: &i64) -> Option<std::cmp::Ordering> {
        match self { DictValue::Int(i) => i.partial_cmp(other), _ => None }
    }
}

impl PartialEq<f64> for DictValue {
    fn eq(&self, other: &f64) -> bool {
        match self { DictValue::Float(f) => f == other, _ => false }
    }
}

impl PartialOrd<f64> for DictValue {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        match self { DictValue::Float(f) => f.partial_cmp(other), _ => None }
    }
}

impl PartialEq<&str> for DictValue {
    fn eq(&self, other: &&str) -> bool {
        match self { DictValue::String(s) => s == other, _ => false }
    }
}

impl PartialEq for DictValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DictValue::Int(a), DictValue::Int(b)) => a == b,
            (DictValue::Float(a), DictValue::Float(b)) => a == b,
            (DictValue::String(a), DictValue::String(b)) => a == b,
            (DictValue::Bool(a), DictValue::Bool(b)) => a == b,
            _ => false,
        }
    }
}

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

// ============================================================================
// PLUGIN RUNTIME FUNCTIONS (only with "plugins" feature)
// ============================================================================

#[cfg(feature = "plugins")]
/// Plugin runtime functions
/// These are called from generated code to execute plugin functions

/// Call a Python plugin function (JIT mode)
#[cfg(feature = "plugins")]
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
#[cfg(feature = "plugins")]
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

// ============================================================================
// BUILT-IN FUNCTIONS FOR COMPILED CODE
// ============================================================================

/// Print functions
pub fn print_float_formatted(value: f64) {
    if value.fract() == 0.0 && value.is_finite() {
        println!("{:.1}", value);
    } else {
        println!("{}", value);
    }
}

pub fn print_value<T: fmt::Display>(value: &T) {
    println!("{}", value);
}

// Alias for print_value to avoid conflict with Rust's print! macro
pub fn print<T: fmt::Display>(value: &T) {
    println!("{}", value);
}

/// ✅ FIX 18: Print multiple values with space separator - for multi-argument print()
pub fn print_multi(values: Vec<String>) {
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", value);
    }
    println!();
}

/// ✅ FIX 18: Convert values to String for multi-argument print()
pub fn to_string_unit(_value: &()) -> String {
    "None".to_string()
}

pub fn to_string_vec_dictvalue(vec: &Vec<DictValue>) -> String {
    let mut result = String::from("[");
    for (i, item) in vec.iter().enumerate() {
        if i > 0 {
            result.push_str(", ");
        }
        result.push_str(&to_string_dictvalue(item));
    }
    result.push(']');
    result
}

pub fn to_string_hashmap_dictvalue(map: &HashMap<String, DictValue>) -> String {
    let mut result = String::from("{");
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            result.push_str(", ");
        }
        result.push_str(&format!("{}: {}", k, to_string_dictvalue(v)));
    }
    result.push('}');
    result
}

pub fn to_string_dictvalue(value: &DictValue) -> String {
    match value {
        DictValue::Int(i) => i.to_string(),
        DictValue::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                format!("{:.1}", f)
            } else {
                f.to_string()
            }
        }
        DictValue::String(s) => s.clone(),
        DictValue::Bool(b) => b.to_string(),
        DictValue::List(l) => to_string_vec_dictvalue(l),
        DictValue::Dict(d) => to_string_hashmap_dictvalue(d),
    }
}

pub fn print_hashmap_i64(map: HashMap<String, i64>) {
    println!("{:?}", map);
}

pub fn print_hashmap_f64(map: HashMap<String, f64>) {
    println!("{:?}", map);
}

pub fn print_hashmap_string(map: HashMap<String, String>) {
    println!("{:?}", map);
}

pub fn print_hashmap_bool(map: HashMap<String, bool>) {
    println!("{:?}", map);
}

/// Print function for heterogeneous dictionaries (HashMap<String, DictValue>)
pub fn print_hashmap_dictvalue(map: &HashMap<String, DictValue>) {
    print!("{{");
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}: {}", k, v);
    }
    println!("}}");
}

/// ✅ NEW: Print functions for vectors
pub fn print_vec_i64(vec: Vec<i64>) {
    print!("[");
    for (i, item) in vec.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", item);
    }
    println!("]");
}

pub fn print_vec_f64(vec: Vec<f64>) {
    print!("[");
    for (i, item) in vec.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        if item.fract() == 0.0 && item.is_finite() {
            print!("{:.1}", item);
        } else {
            print!("{}", item);
        }
    }
    println!("]");
}

pub fn print_vec_string(vec: Vec<String>) {
    print!("[");
    for (i, item) in vec.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", item);
    }
    println!("]");
}

pub fn print_vec_bool(vec: Vec<bool>) {
    print!("[");
    for (i, item) in vec.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", item);
    }
    println!("]");
}

/// ✅ NEW: Print function for Vec<DictValue> - used for heterogeneous lists and pop() results
pub fn print_vec_dictvalue(vec: &Vec<DictValue>) {
    print!("[");
    for (i, item) in vec.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        match item {
            DictValue::Int(n) => print!("{}", n),
            DictValue::Float(f) => {
                if f.fract() == 0.0 && f.is_finite() {
                    print!("{:.1}", f);
                } else {
                    print!("{}", f);
                }
            }
            DictValue::String(s) => print!("{}", s),
            DictValue::Bool(b) => print!("{}", b),
            DictValue::List(items) => {
                // Print nested list
                print!("[");
                for (j, nested_item) in items.iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    match nested_item {
                        DictValue::Int(n) => print!("{}", n),
                        DictValue::Float(f) => {
                            if f.fract() == 0.0 && f.is_finite() {
                                print!("{:.1}", f);
                            } else {
                                print!("{}", f);
                            }
                        }
                        DictValue::String(s) => print!("{}", s),
                        DictValue::Bool(b) => print!("{}", b),
                        _ => print!("[...]"),
                    }
                }
                print!("]");
            }
            DictValue::Dict(_) => print!("{{...}}"),
        }
    }
    println!("]");
}

/// Print function for Option types
pub fn print_option<T: fmt::Display>(opt: &Option<T>) {
    match opt {
        Some(v) => println!("{}", v),
        None => println!("None"),
    }
}

/// Print function for unit type ()
pub fn print_unit(_value: &()) {
    println!("None");
}

/// Generic print_debug for types that don't implement Display
pub fn print_debug<T: fmt::Debug>(value: &T) {
    println!("{:?}", value);
}

/// ✅ NEW: Print function for DictValue - properly prints lists and dicts
pub fn print_dictvalue(value: &DictValue) {
    match value {
        DictValue::Int(i) => println!("{}", i),
        DictValue::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                println!("{:.1}", f);
            } else {
                println!("{}", f);
            }
        }
        DictValue::String(s) => println!("{}", s),
        DictValue::Bool(b) => println!("{}", b),
        DictValue::List(items) => {
            print!("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                match item {
                    DictValue::Int(n) => print!("{}", n),
                    DictValue::Float(f) => {
                        if f.fract() == 0.0 && f.is_finite() {
                            print!("{:.1}", f);
                        } else {
                            print!("{}", f);
                        }
                    }
                    DictValue::String(s) => print!("{}", s),
                    DictValue::Bool(b) => print!("{}", b),
                    DictValue::List(_) => print!("[...]"),
                    DictValue::Dict(_) => print!("{{...}}"),
                }
            }
            println!("]");
        }
        DictValue::Dict(map) => {
            print!("{{");
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}: ", k);
                match v {
                    DictValue::Int(n) => print!("{}", n),
                    DictValue::Float(f) => {
                        if f.fract() == 0.0 && f.is_finite() {
                            print!("{:.1}", f);
                        } else {
                            print!("{}", f);
                        }
                    }
                    DictValue::String(s) => print!("{}", s),
                    DictValue::Bool(b) => print!("{}", b),
                    DictValue::List(_) => print!("[...]"),
                    DictValue::Dict(_) => print!("{{...}}"),
                }
            }
            println!("}}");
        }
    }
}

/// Print function for tuple (Vec<i64>, i64) - used by pop()
pub fn print_tuple_vec_i64_i64(value: &(Vec<i64>, i64)) {
    print!("[[");
    for (i, item) in value.0.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", item);
    }
    print!("], {}]", value.1);
    println!();
}

// ============================================================================
// POLYMORPHIC TRAITS
// ============================================================================

/// Polymorphic len() function
pub trait Len {
    fn tb_len(&self) -> i64;
}

impl<T> Len for Vec<T> {
    fn tb_len(&self) -> i64 {
        self.len() as i64
    }
}

impl<T> Len for &[T] {
    fn tb_len(&self) -> i64 {
        self.len() as i64
    }
}

impl Len for String {
    fn tb_len(&self) -> i64 {
        self.len() as i64
    }
}

impl Len for &str {
    fn tb_len(&self) -> i64 {
        self.len() as i64
    }
}

impl Len for DictValue {
    fn tb_len(&self) -> i64 {
        match self {
            DictValue::List(v) => v.len() as i64,
            DictValue::Dict(m) => m.len() as i64,
            DictValue::String(s) => s.len() as i64,
            _ => 0,
        }
    }
}

impl Len for &DictValue {
    fn tb_len(&self) -> i64 {
        match self {
            DictValue::List(v) => v.len() as i64,
            DictValue::Dict(m) => m.len() as i64,
            DictValue::String(s) => s.len() as i64,
            _ => 0,
        }
    }
}

impl<K, V> Len for HashMap<K, V> {
    fn tb_len(&self) -> i64 {
        self.len() as i64
    }
}

pub fn len<T: Len>(collection: &T) -> i64 {
    collection.tb_len()
}

/// Polymorphic is_truthy() function for Python-like truthiness
pub trait IsTruthy {
    fn is_truthy(&self) -> bool;
}

impl IsTruthy for bool {
    fn is_truthy(&self) -> bool {
        *self
    }
}

impl IsTruthy for i64 {
    fn is_truthy(&self) -> bool {
        *self != 0
    }
}

impl IsTruthy for f64 {
    fn is_truthy(&self) -> bool {
        *self != 0.0
    }
}

impl IsTruthy for &str {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl IsTruthy for String {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl<T> IsTruthy for Vec<T> {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl<K, V> IsTruthy for HashMap<K, V> {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl IsTruthy for DictValue {
    fn is_truthy(&self) -> bool {
        match self {
            DictValue::Bool(b) => *b,
            DictValue::Int(i) => *i != 0,
            DictValue::Float(f) => *f != 0.0,
            DictValue::String(s) => !s.is_empty(),
            DictValue::List(l) => !l.is_empty(),
            DictValue::Dict(d) => !d.is_empty(),
        }
    }
}

impl<T> IsTruthy for Option<T> {
    fn is_truthy(&self) -> bool {
        self.is_some()
    }
}

/// ✅ FIX: Unit type () is falsy (represents None in TB Language)
impl IsTruthy for () {
    fn is_truthy(&self) -> bool {
        false
    }
}

pub fn is_truthy<T: IsTruthy>(value: &T) -> bool {
    value.is_truthy()
}



// ============================================================================
// TYPE CONVERSION TRAITS
// ============================================================================

/// Polymorphic int() function
pub trait ToInt {
    fn to_int(&self) -> i64;
}

impl ToInt for bool {
    fn to_int(&self) -> i64 {
        if *self { 1 } else { 0 }
    }
}

impl ToInt for i64 {
    fn to_int(&self) -> i64 {
        *self
    }
}

impl ToInt for f64 {
    fn to_int(&self) -> i64 {
        *self as i64
    }
}

impl ToInt for &str {
    fn to_int(&self) -> i64 {
        self.parse().unwrap_or(0)
    }
}

impl ToInt for String {
    fn to_int(&self) -> i64 {
        self.parse().unwrap_or(0)
    }
}

pub fn int<T: ToInt>(value: T) -> i64 {
    value.to_int()
}

/// Polymorphic float() function
pub trait ToFloat {
    fn to_float(&self) -> f64;
}

impl ToFloat for i64 {
    fn to_float(&self) -> f64 {
        *self as f64
    }
}

impl ToFloat for &str {
    fn to_float(&self) -> f64 {
        self.parse().unwrap_or(0.0)
    }
}

impl ToFloat for String {
    fn to_float(&self) -> f64 {
        self.parse().unwrap_or(0.0)
    }
}

pub fn float<T: ToFloat>(value: T) -> f64 {
    value.to_float()
}

/// String conversion
pub fn str_from<T: fmt::Display>(value: T) -> String {
    format!("{}", value)
}

// ============================================================================
// TYPE INTROSPECTION
// ============================================================================

/// type_of() function - returns the type name of a value
pub fn type_of_i64(_value: &i64) -> String {
    "int".to_string()
}

pub fn type_of_f64(_value: &f64) -> String {
    "float".to_string()
}

pub fn type_of_string(_value: &String) -> String {
    "string".to_string()
}

pub fn type_of_bool(_value: &bool) -> String {
    "bool".to_string()
}

pub fn type_of_vec_i64(_value: &Vec<i64>) -> String {
    "list".to_string()
}

pub fn type_of_vec_f64(_value: &Vec<f64>) -> String {
    "list".to_string()
}

pub fn type_of_vec_string(_value: &Vec<String>) -> String {
    "list".to_string()
}

pub fn type_of_vec_bool(_value: &Vec<bool>) -> String {
    "list".to_string()
}

pub fn type_of_hashmap<K, V>(_value: &HashMap<K, V>) -> String {
    "dict".to_string()
}

pub fn type_of_option<T>(_value: &Option<T>) -> String {
    "None".to_string()
}

pub fn type_of_unit(_value: &()) -> String {
    "None".to_string()
}

// Generic type_of for any type
pub fn type_of<T>(_value: &T) -> String {
    std::any::type_name::<T>().to_string()
}

// ✅ FIX 11: type_of for Vec<DictValue>
pub fn type_of_vec_dictvalue(_value: &Vec<DictValue>) -> String {
    "list".to_string()
}

// type_of for DictValue
pub fn type_of_dict_value(value: &DictValue) -> String {
    match value {
        DictValue::Int(_) => "int".to_string(),
        DictValue::Float(_) => "float".to_string(),
        DictValue::String(_) => "string".to_string(),
        DictValue::Bool(_) => "bool".to_string(),
        DictValue::List(_) => "list".to_string(),
        DictValue::Dict(_) => "dict".to_string(),
    }
}

// ============================================================================
// COLLECTION FUNCTIONS
// ============================================================================

/// Range function with optional step
pub fn range(start: i64, end: Option<i64>, step: Option<i64>) -> Vec<i64> {
    let step_val = step.unwrap_or(1);

    if step_val == 0 {
        panic!("range() step cannot be zero");
    }

    match end {
        Some(e) => {
            if step_val > 0 {
                (start..e).step_by(step_val as usize).collect()
            } else {
                // Negative step: count down
                let mut result = Vec::new();
                let mut current = start;
                while current > e {
                    result.push(current);
                    current += step_val; // step_val is negative
                }
                result
            }
        }
        None => (0..start).collect(),
    }
}

/// Push function
pub fn push<T>(mut vec: Vec<T>, item: T) -> Vec<T> {
    vec.push(item);
    vec
}

/// Pop function - returns Vec<DictValue> containing [new_list, popped_value]
/// Example: pop([1, 2, 3, 4]) -> [[1, 2, 3], 4]
/// This matches the JIT mode behavior where pop() returns a list, not a tuple
pub fn pop_i64(mut vec: Vec<i64>) -> Vec<DictValue> {
    let popped = vec.pop().expect("Cannot pop from empty list");
    vec![
        DictValue::List(vec.into_iter().map(DictValue::Int).collect()),
        DictValue::Int(popped),
    ]
}

/// Pop function for generic types - returns Vec<T> containing [new_list, popped_value]
pub fn pop_generic<T: Clone + Into<DictValue>>(mut vec: Vec<T>) -> Vec<DictValue> {
    let popped = vec.pop().expect("Cannot pop from empty list");
    vec![
        DictValue::List(vec.into_iter().map(|v| v.into()).collect()),
        popped.into(),
    ]
}

/// Keys function
pub fn keys<K: Clone, V>(map: &HashMap<K, V>) -> Vec<K> {
    map.keys().cloned().collect()
}

/// Values function
pub fn values<K, V: Clone>(map: &HashMap<K, V>) -> Vec<V> {
    map.values().cloned().collect()
}

/// Keys function for DictValue
pub fn keys_dictvalue(dv: &DictValue) -> Vec<String> {
    dv.as_dict().keys().cloned().collect()
}

/// Values function for DictValue
pub fn values_dictvalue(dv: &DictValue) -> Vec<DictValue> {
    dv.as_dict().values().cloned().collect()
}

// ============================================================================
// TIME FUNCTIONS
// ============================================================================

/// time() function - no args, uses Local timezone
pub fn time() -> HashMap<String, DictValue> {
    time_with_tz("Local".to_string())
}

/// time(timezone: String) -> HashMap
pub fn time_with_tz(timezone: String) -> HashMap<String, DictValue> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap();
    let timestamp = duration.as_secs() as i64;

    // Simple date/time calculation (UTC)
    let days_since_epoch = timestamp / 86400;
    let seconds_today = timestamp % 86400;
    let hour = (seconds_today / 3600) as i64;
    let minute = ((seconds_today % 3600) / 60) as i64;
    let second = (seconds_today % 60) as i64;

    // Approximate year/month/day (simplified)
    let year = 1970 + (days_since_epoch / 365) as i64;
    let day_of_year = (days_since_epoch % 365) as i64;
    let month = (day_of_year / 30 + 1).min(12) as i64;
    let day = (day_of_year % 30 + 1) as i64;

    let mut map = HashMap::new();
    map.insert("year".to_string(), DictValue::Int(year));
    map.insert("month".to_string(), DictValue::Int(month));
    map.insert("day".to_string(), DictValue::Int(day));
    map.insert("hour".to_string(), DictValue::Int(hour));
    map.insert("minute".to_string(), DictValue::Int(minute));
    map.insert("second".to_string(), DictValue::Int(second));
    map.insert("timestamp".to_string(), DictValue::Int(timestamp));
    map.insert("timezone".to_string(), DictValue::String(timezone));
    map.insert("iso8601".to_string(), DictValue::String(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hour, minute, second)));
    map
}

// ============================================================================
// DICT CONVERSION FUNCTIONS
// ============================================================================

/// dict() function - copy existing dict (DictValue)
pub fn dict_from_dict_value(d: HashMap<String, DictValue>) -> HashMap<String, DictValue> {
    d.clone()
}

/// dict() function - convert from homogeneous dict to DictValue dict
pub fn dict_from_int(d: HashMap<String, i64>) -> HashMap<String, DictValue> {
    d.into_iter().map(|(k, v)| (k, DictValue::Int(v))).collect()
}

pub fn dict_from_float(d: HashMap<String, f64>) -> HashMap<String, DictValue> {
    d.into_iter().map(|(k, v)| (k, DictValue::Float(v))).collect()
}

pub fn dict_from_string_map(d: HashMap<String, String>) -> HashMap<String, DictValue> {
    d.into_iter().map(|(k, v)| (k, DictValue::String(v))).collect()
}

pub fn dict_from_bool(d: HashMap<String, bool>) -> HashMap<String, DictValue> {
    d.into_iter().map(|(k, v)| (k, DictValue::Int(if v { 1 } else { 0 }))).collect()
}

/// ✅ FIX: dict() function - parse JSON string to HashMap<String, DictValue>
#[cfg(feature = "json")]
pub fn dict_from_string(json_str: Option<String>) -> HashMap<String, DictValue> {
    if let Some(s) = json_str {
        json_parse(s)
    } else {
        HashMap::new()
    }
}

#[cfg(not(feature = "json"))]
pub fn dict_from_string(_json_str: Option<String>) -> HashMap<String, DictValue> {
    HashMap::new()
}

// ============================================================================
// LIST CONVERSION FUNCTIONS
// ============================================================================

/// list() function - copy existing list
pub fn list_from_list(list: Vec<DictValue>) -> Vec<DictValue> {
    list.clone()
}

/// ✅ FIX: list() function - parse JSON array string to Vec<DictValue>
#[cfg(feature = "json")]
pub fn list_from_string(json_str: Option<String>) -> Vec<DictValue> {
    if let Some(s) = json_str {
        // Parse JSON array
        let value: serde_json::Value = serde_json::from_str(&s)
            .unwrap_or(serde_json::Value::Array(vec![]));

        if let serde_json::Value::Array(arr) = value {
            arr.iter().map(json_value_to_dict_value).collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    }
}

#[cfg(not(feature = "json"))]
pub fn list_from_string(_json_str: Option<String>) -> Vec<DictValue> {
    Vec::new()
}

// ============================================================================
// WRAPPER FUNCTIONS FOR COMPILED CODE
// ============================================================================
// These functions wrap tb-builtins functions to work with native Rust types
// instead of Value types, avoiding code duplication.

/// String operations - split function
pub fn split(s: &str, delimiter: &str) -> Vec<String> {
    s.split(delimiter).map(|s| s.to_string()).collect()
}

/// String operations - join function
pub fn join(list: &[String], delimiter: &str) -> String {
    list.join(delimiter)
}

/// String operations - trim function
pub fn trim(s: &str) -> String {
    s.trim().to_string()
}

/// String operations - replace function
pub fn replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

/// String operations - to_upper function
pub fn to_upper(s: &str) -> String {
    s.to_uppercase()
}

/// String operations - to_lower function
pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

/// String operations - contains function
pub fn contains(s: &str, substring: &str) -> bool {
    s.contains(substring)
}

/// String operations - starts_with function
pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

/// String operations - ends_with function
pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

/// Math operations - abs function
pub fn abs_i64(x: i64) -> i64 {
    x.abs()
}

pub fn abs_f64(x: f64) -> f64 {
    x.abs()
}

/// Math operations - min/max functions
pub fn min_i64(a: i64, b: i64) -> i64 {
    a.min(b)
}

pub fn max_i64(a: i64, b: i64) -> i64 {
    a.max(b)
}

pub fn min_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}

pub fn max_f64(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Math operations - pow function
pub fn pow_i64(base: i64, exp: u32) -> i64 {
    base.pow(exp)
}

pub fn pow_f64(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Math operations - sqrt function
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Math operations - floor/ceil/round functions
pub fn floor(x: f64) -> f64 {
    x.floor()
}

pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

pub fn round(x: f64) -> f64 {
    x.round()
}

// ============================================================================
// JSON/YAML FUNCTIONS
// ============================================================================

#[cfg(feature = "json")]
/// Helper: Convert DictValue to serde_json::Value
fn dict_value_to_json_value(val: &DictValue) -> serde_json::Value {
    match val {
        DictValue::Int(i) => serde_json::Value::Number((*i).into()),
        DictValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        DictValue::String(s) => serde_json::Value::String(s.clone()),
        DictValue::Bool(b) => serde_json::Value::Bool(*b),
        DictValue::List(list) => {
            serde_json::Value::Array(list.iter().map(dict_value_to_json_value).collect())
        },
        DictValue::Dict(map) => {
            let mut json_map = serde_json::Map::new();
            for (k, v) in map {
                json_map.insert(k.clone(), dict_value_to_json_value(v));
            }
            serde_json::Value::Object(json_map)
        },
    }
}

#[cfg(feature = "json")]
/// Helper: Convert serde_json::Value to DictValue
fn json_value_to_dict_value(val: &serde_json::Value) -> DictValue {
    match val {
        serde_json::Value::Null => DictValue::Int(0),
        serde_json::Value::Bool(b) => DictValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                DictValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                DictValue::Float(f)
            } else {
                DictValue::Int(0)
            }
        },
        serde_json::Value::String(s) => DictValue::String(s.clone()),
        serde_json::Value::Array(arr) => {
            DictValue::List(arr.iter().map(json_value_to_dict_value).collect())
        },
        serde_json::Value::Object(obj) => {
            let mut map = HashMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), json_value_to_dict_value(v));
            }
            DictValue::Dict(map)
        },
    }
}

#[cfg(feature = "yaml")]
/// Helper: Convert serde_yaml::Value to DictValue
fn yaml_value_to_dict_value(val: &serde_yaml::Value) -> DictValue {
    match val {
        serde_yaml::Value::Null => DictValue::Int(0),
        serde_yaml::Value::Bool(b) => DictValue::Int(if *b { 1 } else { 0 }),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                DictValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                DictValue::Float(f)
            } else {
                DictValue::Int(0)
            }
        },
        serde_yaml::Value::String(s) => DictValue::String(s.clone()),
        serde_yaml::Value::Sequence(arr) => {
            DictValue::List(arr.iter().map(yaml_value_to_dict_value).collect())
        },
        serde_yaml::Value::Mapping(obj) => {
            let mut map = HashMap::new();
            for (k, v) in obj {
                if let serde_yaml::Value::String(key) = k {
                    map.insert(key.clone(), yaml_value_to_dict_value(v));
                }
            }
            DictValue::Dict(map)
        },
        _ => DictValue::Int(0),
    }
}

#[cfg(feature = "json")]
/// JSON parse function
pub fn json_parse(json_str: String) -> HashMap<String, DictValue> {
    let value: serde_json::Value = serde_json::from_str(&json_str)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    if let serde_json::Value::Object(obj) = value {
        let mut map = HashMap::new();
        for (k, v) in obj {
            map.insert(k, json_value_to_dict_value(&v));
        }
        map
    } else {
        HashMap::new()  // Return empty HashMap on error
    }
}

#[cfg(not(feature = "json"))]
/// JSON parse function (stub when json feature is disabled)
pub fn json_parse(_json_str: String) -> HashMap<String, DictValue> {
    HashMap::new()
}

#[cfg(feature = "json")]
/// JSON stringify function
pub fn json_stringify(value: &HashMap<String, DictValue>) -> String {
    let mut json_map = serde_json::Map::new();
    for (k, v) in value {
        json_map.insert(k.clone(), dict_value_to_json_value(v));
    }
    let json_value = serde_json::Value::Object(json_map);
    serde_json::to_string(&json_value).unwrap_or_default()
}

#[cfg(not(feature = "json"))]
/// JSON stringify function (stub when json feature is disabled)
pub fn json_stringify(_value: &HashMap<String, DictValue>) -> String {
    String::new()
}

#[cfg(feature = "yaml")]
/// YAML parse function
pub fn yaml_parse(yaml_str: String) -> HashMap<String, DictValue> {
    let value: serde_yaml::Value = serde_yaml::from_str(&yaml_str)
        .unwrap_or(serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));

    if let serde_yaml::Value::Mapping(obj) = value {
        let mut map = HashMap::new();
        for (k, v) in obj {
            if let serde_yaml::Value::String(key) = k {
                map.insert(key, yaml_value_to_dict_value(&v));
            }
        }
        map
    } else {
        HashMap::new()  // Return empty HashMap on error
    }
}

#[cfg(not(feature = "yaml"))]
/// YAML parse function (stub when yaml feature is disabled)
pub fn yaml_parse(_yaml_str: String) -> HashMap<String, DictValue> {
    HashMap::new()
}

#[cfg(feature = "yaml")]
/// Helper: Convert DictValue to serde_yaml::Value
fn dict_value_to_yaml_value(val: &DictValue) -> serde_yaml::Value {
    match val {
        DictValue::Int(i) => serde_yaml::Value::Number((*i).into()),
        DictValue::Float(f) => serde_yaml::Value::Number(serde_yaml::Number::from(*f)),
        DictValue::String(s) => serde_yaml::Value::String(s.clone()),
        DictValue::Bool(b) => serde_yaml::Value::Bool(*b),
        DictValue::List(list) => {
            serde_yaml::Value::Sequence(list.iter().map(dict_value_to_yaml_value).collect())
        },
        DictValue::Dict(map) => {
            let mut yaml_map = serde_yaml::Mapping::new();
            for (k, v) in map {
                yaml_map.insert(
                    serde_yaml::Value::String(k.clone()),
                    dict_value_to_yaml_value(v)
                );
            }
            serde_yaml::Value::Mapping(yaml_map)
        },
    }
}

#[cfg(feature = "yaml")]
/// YAML stringify function
pub fn yaml_stringify(value: &HashMap<String, DictValue>) -> String {
    // Convert HashMap<String, DictValue> to serde_yaml::Value
    let mut yaml_map = serde_yaml::Mapping::new();
    for (k, v) in value {
        yaml_map.insert(
            serde_yaml::Value::String(k.clone()),
            dict_value_to_yaml_value(v)
        );
    }
    let yaml_value = serde_yaml::Value::Mapping(yaml_map);
    serde_yaml::to_string(&yaml_value).unwrap_or_default()
}

#[cfg(not(feature = "yaml"))]
/// YAML stringify function (stub when yaml feature is disabled)
pub fn yaml_stringify(_value: &HashMap<String, DictValue>) -> String {
    String::new()
}

// ============================================================================
// HIGHER-ORDER FUNCTIONS
// ============================================================================

/// for_each function - executes a function for each element in a vector
pub fn for_each<T, F>(func: F, vec: Vec<T>)
where
    F: Fn(&T)
{
    for item in vec.iter() {
        func(item);
    }
}

/// Alias for backwards compatibility - forEach is the same as for_each
#[allow(non_snake_case)]
pub fn forEach<T, F>(func: F, vec: Vec<T>)
where
    F: Fn(&T)
{
    for_each(func, vec)
}

/// reduce function for i64
pub fn reduce_i64<F>(func: F, vec: Vec<i64>, initial: i64) -> i64
where
    F: Fn(i64, &i64) -> i64
{
    vec.iter().fold(initial, func)
}

/// reduce function for f64
pub fn reduce_f64<F>(func: F, vec: Vec<f64>, initial: f64) -> f64
where
    F: Fn(f64, &f64) -> f64
{
    vec.iter().fold(initial, func)
}

/// reduce function for String
pub fn reduce_string<F>(func: F, vec: Vec<String>, initial: String) -> String
where
    F: Fn(String, &String) -> String
{
    vec.iter().fold(initial, func)
}

// ============================================================================
// STRING CONVERSION FUNCTIONS
// ============================================================================

/// str() function for i64
pub fn str_i64(value: i64) -> String {
    value.to_string()
}

/// str() function for f64
pub fn str_f64(value: f64) -> String {
    value.to_string()
}

/// str() function for bool
pub fn str_bool(value: bool) -> String {
    value.to_string()
}

// ============================================================================
// HTTP FUNCTIONS
// ============================================================================

#[cfg(feature = "networking")]
/// HTTP session creation
pub fn http_session(base_url: String) -> String {
    // Return a session ID (simplified - just return the base URL)
    base_url
}

#[cfg(not(feature = "networking"))]
/// HTTP session creation (stub)
pub fn http_session(base_url: String) -> String {
    base_url
}

#[cfg(feature = "networking")]
/// HTTP request function
pub fn http_request(
    session_id: String,
    path: String,
    method: String,
    data: Option<HashMap<String, DictValue>>
) -> HashMap<String, DictValue> {
    let url = format!("{}{}", session_id, path);

    let client = reqwest::blocking::Client::new();
    let mut response_map = HashMap::new();

    let result = match method.to_uppercase().as_str() {
        "GET" => client.get(&url).send(),
        "POST" => {
            let mut req = client.post(&url);
            if let Some(body) = data {
                #[cfg(feature = "json")]
                {
                    let json_body = serde_json::to_string(&body).unwrap_or_default();
                    req = req.header("Content-Type", "application/json").body(json_body);
                }
                #[cfg(not(feature = "json"))]
                {
                    let _ = body; // Suppress unused warning
                    req = req.header("Content-Type", "application/json").body("{}");
                }
            }
            req.send()
        },
        _ => {
            response_map.insert("error".to_string(), DictValue::String("Unsupported method".to_string()));
            return response_map;
        }
    };

    match result {
        Ok(resp) => {
            response_map.insert("status".to_string(), DictValue::Int(resp.status().as_u16() as i64));
            if let Ok(text) = resp.text() {
                response_map.insert("body".to_string(), DictValue::String(text));
            }
        },
        Err(e) => {
            response_map.insert("error".to_string(), DictValue::String(e.to_string()));
        }
    }

    response_map
}

#[cfg(not(feature = "networking"))]
/// HTTP request function (stub)
pub fn http_request(
    _session_id: String,
    _path: String,
    _method: String,
    _data: Option<HashMap<String, DictValue>>
) -> HashMap<String, DictValue> {
    let mut response_map = HashMap::new();
    response_map.insert("error".to_string(), DictValue::String("Networking not enabled".to_string()));
    response_map
}

// ============================================================================
// NETWORKING - REAL IMPLEMENTATIONS (CONDITIONAL)
// ============================================================================

#[cfg(feature = "networking")]
use std::sync::OnceLock;

#[cfg(feature = "networking")]
/// Global Tokio runtime for async operations in compiled mode
/// Uses OnceLock for lazy initialization - only created when first used!
static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

#[cfg(feature = "networking")]
/// Get or create the Tokio runtime with configurable thread count
fn get_runtime(worker_threads: usize) -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| {
        #[cfg(feature = "tokio-multi-thread")]
        {
            // Multi-threaded runtime (when threads > 1 or networking with multiple threads)
            let threads = if worker_threads == 0 { 2 } else { worker_threads };
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(threads)
                .thread_name("tb-compiled-worker")
                .enable_io()
                .enable_time()
                .build()
                .expect("Failed to create Tokio runtime")
        }
        #[cfg(not(feature = "tokio-multi-thread"))]
        {
            // Single-threaded runtime (default, fastest startup)
            let _ = worker_threads; // Suppress unused warning
            tokio::runtime::Builder::new_current_thread()
                .thread_name("tb-compiled-worker")
                .enable_io()
                .enable_time()
                .build()
                .expect("Failed to create Tokio runtime")
        }
    })
}

#[cfg(feature = "networking")]
/// Global TCP client registry
static TCP_CLIENTS: Lazy<DashMap<String, Arc<tokio::sync::Mutex<TcpStream>>>> = Lazy::new(|| DashMap::new());

#[cfg(feature = "networking")]
/// Global UDP client registry
static UDP_CLIENTS: Lazy<DashMap<String, Arc<UdpSocket>>> = Lazy::new(|| DashMap::new());

#[cfg(feature = "networking")]
/// TCP connection function - complex version with callbacks
pub fn connect_to<F1, F2, F3>(
    _on_connect: F1,
    _on_disconnect: F2,
    _on_message: F3,
    host: String,
    port: i64,
    protocol: String
) -> String
where
    F1: Fn(String, String),
    F2: Fn(String),
    F3: Fn(String, String)
{
    // For compiled mode, we use the simplified version
    connect_to_simple(host, port, protocol, 2)
}

#[cfg(feature = "networking")]
/// Simplified TCP/UDP connection function - REAL IMPLEMENTATION
pub fn connect_to_simple(
    host: String,
    port: i64,
    protocol: String,
    worker_threads: usize,
) -> String
{
    let conn_id = format!("conn_{}:{}_{}", protocol, host, port);
    let conn_id_clone = conn_id.clone();

    get_runtime(worker_threads).block_on(async move {
        match protocol.to_lowercase().as_str() {
            "tcp" => {
                let addr = format!("{}:{}", host, port);
                match TcpStream::connect(&addr).await {
                    Ok(stream) => {
                        TCP_CLIENTS.insert(conn_id_clone.clone(), Arc::new(tokio::sync::Mutex::new(stream)));
                    }
                    Err(e) => {
                        eprintln!("Failed to connect to TCP {}:{}: {}", host, port, e);
                    }
                }
            }
            "udp" => {
                match UdpSocket::bind("0.0.0.0:0").await {
                    Ok(socket) => {
                        let remote_addr = format!("{}:{}", host, port);
                        if let Err(e) = socket.connect(&remote_addr).await {
                            eprintln!("Failed to connect UDP socket: {}", e);
                        } else {
                            UDP_CLIENTS.insert(conn_id_clone.clone(), Arc::new(socket));
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to create UDP socket: {}", e);
                    }
                }
            }
            _ => {
                eprintln!("Unsupported protocol: {}", protocol);
            }
        }
    });

    conn_id
}

#[cfg(feature = "networking")]
/// Send message to connection - REAL IMPLEMENTATION
pub fn send_to(conn_id: String, message: String) {
    send_to_with_threads(conn_id, message, 2);
}

#[cfg(feature = "networking")]
pub fn send_to_with_threads(conn_id: String, message: String, worker_threads: usize) {
    get_runtime(worker_threads).block_on(async move {
        // Try TCP first
        if let Some(stream_ref) = TCP_CLIENTS.get(&conn_id) {
            let mut stream = stream_ref.lock().await;
            if let Err(e) = stream.write_all(message.as_bytes()).await {
                eprintln!("Failed to send TCP message: {}", e);
            } else {
                let _ = stream.flush().await;
            }
            return;
        }

        // Try UDP
        if let Some(socket_ref) = UDP_CLIENTS.get(&conn_id) {
            if let Err(e) = socket_ref.send(message.as_bytes()).await {
                eprintln!("Failed to send UDP message: {}", e);
            }
            return;
        }

        eprintln!("Connection not found: {}", conn_id);
    });
}

#[cfg(feature = "networking")]
/// Receive message from connection - REAL IMPLEMENTATION
pub fn receive_from(conn_id: String) -> String {
    receive_from_with_threads(conn_id, 2)
}

#[cfg(feature = "networking")]
pub fn receive_from_with_threads(conn_id: String, worker_threads: usize) -> String {
    get_runtime(worker_threads).block_on(async move {
        // Try TCP first
        if let Some(stream_ref) = TCP_CLIENTS.get(&conn_id) {
            let mut stream = stream_ref.lock().await;
            let mut buffer = vec![0u8; 4096];
            match stream.read(&mut buffer).await {
                Ok(n) if n > 0 => {
                    return String::from_utf8_lossy(&buffer[..n]).to_string();
                }
                Ok(_) => {
                    eprintln!("Connection closed");
                    return String::new();
                }
                Err(e) => {
                    eprintln!("Failed to receive TCP message: {}", e);
                    return String::new();
                }
            }
        }

        // Try UDP
        if let Some(socket_ref) = UDP_CLIENTS.get(&conn_id) {
            let mut buffer = vec![0u8; 4096];
            match socket_ref.recv(&mut buffer).await {
                Ok(n) => {
                    return String::from_utf8_lossy(&buffer[..n]).to_string();
                }
                Err(e) => {
                    eprintln!("Failed to receive UDP message: {}", e);
                    return String::new();
                }
            }
        }

        eprintln!("Connection not found: {}", conn_id);
        String::new()
    })
}

#[cfg(feature = "networking")]
/// Close connection - REAL IMPLEMENTATION
pub fn close_connection(conn_id: String) -> bool {
    let removed_tcp = TCP_CLIENTS.remove(&conn_id).is_some();
    let removed_udp = UDP_CLIENTS.remove(&conn_id).is_some();
    removed_tcp || removed_udp
}

// ============================================================================
// NETWORKING STUBS (when feature is disabled)
// ============================================================================

#[cfg(not(feature = "networking"))]
pub fn connect_to<F1, F2, F3>(
    _on_connect: F1,
    _on_disconnect: F2,
    _on_message: F3,
    _host: String,
    _port: i64,
    _protocol: String
) -> String
where
    F1: Fn(String, String),
    F2: Fn(String),
    F3: Fn(String, String)
{
    panic!("Networking not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn connect_to_simple(_host: String, _port: i64, _protocol: String, _worker_threads: usize) -> String {
    panic!("Networking not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn send_to(_conn_id: String, _message: String) {
    panic!("Networking not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn send_to_with_threads(_conn_id: String, _message: String, _worker_threads: usize) {
    panic!("Networking not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn receive_from(_conn_id: String) -> String {
    panic!("Networking not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn receive_from_with_threads(_conn_id: String, _worker_threads: usize) -> String {
    panic!("Networking not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn close_connection(_conn_id: String) -> bool {
    panic!("Networking not enabled! Recompile with --features networking");
}


// ============================================================================
// CACHE MANAGEMENT - REAL IMPLEMENTATIONS
// ============================================================================

#[cfg(feature = "networking")]
/// Simple in-memory cache for compiled mode
static CACHE_STORAGE: Lazy<RwLock<StdHashMap<String, Vec<u8>>>> = Lazy::new(|| RwLock::new(StdHashMap::new()));

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub total_bytes: usize,
}

#[cfg(feature = "networking")]
/// Get cache statistics
pub fn cache_stats() -> HashMap<String, DictValue> {
    let cache = CACHE_STORAGE.read().unwrap();
    let entries = cache.len();
    let total_bytes: usize = cache.values().map(|v| v.len()).sum();

    let mut stats = HashMap::new();
    stats.insert("entries".to_string(), DictValue::Int(entries as i64));
    stats.insert("total_bytes".to_string(), DictValue::Int(total_bytes as i64));
    stats
}

#[cfg(not(feature = "networking"))]
/// Get cache statistics (stub)
pub fn cache_stats() -> HashMap<String, DictValue> {
    HashMap::new()
}

#[cfg(feature = "networking")]
/// Clear cache
pub fn cache_clear() -> bool {
    let mut cache = CACHE_STORAGE.write().unwrap();
    cache.clear();
    true
}

#[cfg(not(feature = "networking"))]
/// Clear cache (stub)
pub fn cache_clear() -> bool {
    false
}

#[cfg(feature = "networking")]
/// Invalidate specific cache entry
pub fn cache_invalidate(key: String) -> bool {
    let mut cache = CACHE_STORAGE.write().unwrap();
    cache.remove(&key).is_some()
}

#[cfg(not(feature = "networking"))]
/// Invalidate cache entry (stub)
pub fn cache_invalidate(_key: String) -> bool {
    false
}

#[cfg(feature = "networking")]
/// Set cache value
pub fn cache_set(key: String, value: Vec<u8>) {
    let mut cache = CACHE_STORAGE.write().unwrap();
    cache.insert(key, value);
}

#[cfg(not(feature = "networking"))]
/// Set cache value (stub)
pub fn cache_set(_key: String, _value: Vec<u8>) {
}

#[cfg(feature = "networking")]
/// Get cache value
pub fn cache_get(key: String) -> Option<Vec<u8>> {
    let cache = CACHE_STORAGE.read().unwrap();
    cache.get(&key).cloned()
}

#[cfg(not(feature = "networking"))]
/// Get cache value (stub)
pub fn cache_get(_key: String) -> Option<Vec<u8>> {
    None
}

// ============================================================================
// ASYNC TASK MANAGEMENT - REAL IMPLEMENTATIONS
// ============================================================================

#[cfg(feature = "networking")]
/// Global task registry
static ACTIVE_TASKS: Lazy<DashMap<String, JoinHandle<String>>> = Lazy::new(|| DashMap::new());

#[cfg(feature = "networking")]
/// Spawn async task - REAL IMPLEMENTATION
/// Returns task ID
pub fn spawn_task<F>(task_fn: F, worker_threads: usize) -> String
where
    F: std::future::Future<Output = String> + Send + 'static,
{
    let task_id = uuid::Uuid::new_v4().to_string();
    let task_id_clone = task_id.clone();

    let runtime = get_runtime(worker_threads);
    let handle = runtime.spawn(task_fn);
    ACTIVE_TASKS.insert(task_id_clone, handle);

    task_id
}

#[cfg(feature = "networking")]
/// Await task completion - REAL IMPLEMENTATION
pub fn await_task(task_id: String, worker_threads: usize) -> String {
    if let Some((_, handle)) = ACTIVE_TASKS.remove(&task_id) {
        let runtime = get_runtime(worker_threads);
        runtime.block_on(handle).unwrap_or_else(|e| {
            format!("Task panicked: {}", e)
        })
    } else {
        format!("Task not found: {}", task_id)
    }
}

#[cfg(feature = "networking")]
/// Cancel task - REAL IMPLEMENTATION
pub fn cancel_task(task_id: String) -> bool {
    if let Some(entry) = ACTIVE_TASKS.get(&task_id) {
        entry.value().abort();
        ACTIVE_TASKS.remove(&task_id);
        true
    } else {
        false
    }
}

#[cfg(feature = "networking")]
/// List active tasks
pub fn list_tasks() -> Vec<String> {
    ACTIVE_TASKS.iter().map(|entry| entry.key().clone()).collect()
}

#[cfg(not(feature = "networking"))]
pub fn spawn_task<F>(_task_fn: F, _worker_threads: usize) -> String
where
    F: std::future::Future<Output = String> + Send + 'static,
{
    panic!("Async tasks not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn await_task(_task_id: String, _worker_threads: usize) -> String {
    panic!("Async tasks not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn cancel_task(_task_id: String) -> bool {
    panic!("Async tasks not enabled! Recompile with --features networking");
}

#[cfg(not(feature = "networking"))]
pub fn list_tasks() -> Vec<String> {
    panic!("Async tasks not enabled! Recompile with --features networking");
}

// ============================================================================
// SERIALIZATION & HASHING - REAL IMPLEMENTATIONS
// ============================================================================

#[cfg(feature = "networking-full")]
/// Serialize value to bincode - REAL IMPLEMENTATION
pub fn bincode_serialize<T: serde::Serialize>(value: &T) -> Vec<u8> {
    bincode::serialize(value).unwrap_or_default()
}

#[cfg(not(feature = "networking-full"))]
/// Serialize value to bincode - STUB
pub fn bincode_serialize<T: serde::Serialize>(_value: &T) -> Vec<u8> {
    Vec::new()
}

#[cfg(feature = "networking-full")]
/// Deserialize value from bincode - REAL IMPLEMENTATION
pub fn bincode_deserialize<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    bincode::deserialize(bytes).ok()
}

#[cfg(not(feature = "networking-full"))]
/// Deserialize value from bincode - STUB
pub fn bincode_deserialize<T: serde::de::DeserializeOwned>(_bytes: &[u8]) -> Option<T> {
    None
}

#[cfg(feature = "networking-full")]
/// Hash data with specified algorithm - REAL IMPLEMENTATION
pub fn hash(algorithm: String, data: Vec<u8>) -> String {
    match algorithm.to_lowercase().as_str() {
        "sha256" => {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            format!("{:x}", hasher.finalize())
        }
        "sha512" => {
            let mut hasher = Sha512::new();
            hasher.update(&data);
            format!("{:x}", hasher.finalize())
        }
        _ => {
            eprintln!("Unsupported hash algorithm: {}", algorithm);
            String::new()
        }
    }
}

#[cfg(not(feature = "networking-full"))]
/// Hash data with specified algorithm - STUB
pub fn hash(_algorithm: String, _data: Vec<u8>) -> String {
    String::new()
}

/// Hash string - convenience function
pub fn hash_string(algorithm: String, data: String) -> String {
    hash(algorithm, data.into_bytes())
}

// ============================================================================
// PLUGIN MANAGEMENT - REAL IMPLEMENTATIONS
// ============================================================================

#[cfg(feature = "networking")]
/// Global plugin registry (simplified for compiled mode)
static LOADED_PLUGINS: Lazy<DashMap<String, String>> = Lazy::new(|| DashMap::new());

#[cfg(feature = "networking")]
/// List loaded plugins
pub fn list_plugins() -> Vec<String> {
    LOADED_PLUGINS.iter().map(|entry| entry.key().clone()).collect()
}

#[cfg(not(feature = "networking"))]
/// List loaded plugins (stub)
pub fn list_plugins() -> Vec<String> {
    Vec::new()
}

#[cfg(feature = "networking")]
/// Register plugin (for compiled mode)
pub fn register_plugin(name: String, path: String) {
    LOADED_PLUGINS.insert(name, path);
}

#[cfg(not(feature = "networking"))]
/// Register plugin (stub)
pub fn register_plugin(_name: String, _path: String) {
}

#[cfg(feature = "networking")]
/// Unload plugin
pub fn unload_plugin(name: String) -> bool {
    LOADED_PLUGINS.remove(&name).is_some()
}

#[cfg(not(feature = "networking"))]
/// Unload plugin (stub)
pub fn unload_plugin(_name: String) -> bool {
    false
}

#[cfg(feature = "networking")]
/// Reload plugin (simplified - just marks for reload)
pub fn reload_plugin(name: String) -> bool {
    LOADED_PLUGINS.contains_key(&name)
}

#[cfg(not(feature = "networking"))]
/// Reload plugin (stub)
pub fn reload_plugin(_name: String) -> bool {
    false
}

#[cfg(feature = "networking")]
/// Get plugin info
pub fn plugin_info(name: String) -> Option<HashMap<String, DictValue>> {
    if let Some(entry) = LOADED_PLUGINS.get(&name) {
        let mut info = HashMap::new();
        info.insert("name".to_string(), DictValue::String(name));
        info.insert("path".to_string(), DictValue::String(entry.value().clone()));
        Some(info)
    } else {
        None
    }
}

#[cfg(not(feature = "networking"))]
/// Get plugin info (stub)
pub fn plugin_info(_name: String) -> Option<HashMap<String, DictValue>> {
    None
}
