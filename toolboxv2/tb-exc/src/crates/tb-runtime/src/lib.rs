//! TB Runtime Library
//! Provides core runtime functions for compiled TB programs
//!
//! This crate re-exports all built-in functions from tb-builtins
//! to avoid code duplication and ensure consistency between JIT and compiled modes.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

// FFI interface for compiled mode
#[cfg(feature = "full")]
pub mod ffi;

// Re-export built-in functions only when "full" feature is enabled
#[cfg(feature = "full")]
use tb_core::{Value, TBError};

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

    // Higher-order functions
    builtin_map as map_from_value,
    builtin_filter as filter_from_value,
    builtin_reduce as reduce_from_value,
    builtin_forEach as forEach_from_value,
};

// Plugin support (only when "plugins" feature is enabled)
#[cfg(feature = "plugins")]
use std::sync::Arc;
#[cfg(feature = "plugins")]
use dashmap::DashMap;
#[cfg(feature = "plugins")]
use tb_plugin::PluginLoader;
#[cfg(feature = "plugins")]
use tb_core::{PluginLanguage, PluginMode};

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
            _ => panic!("Expected Int"),
        }
    }

    pub fn as_string(&self) -> &str {
        match self {
            DictValue::String(s) => s,
            _ => panic!("Expected String"),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            DictValue::Float(f) => *f,
            _ => panic!("Expected Float"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            DictValue::Bool(b) => *b,
            _ => panic!("Expected Bool"),
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
            _ => panic!("Expected Dict"),
        }
    }

    pub fn as_list(&self) -> &Vec<DictValue> {
        match self {
            DictValue::List(v) => v,
            _ => panic!("Expected List"),
        }
    }
}

impl fmt::Display for DictValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DictValue::Int(i) => write!(f, "{}", i),
            DictValue::Float(fl) => write!(f, "{}", fl),
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
// COLLECTION FUNCTIONS
// ============================================================================

/// Range function
pub fn range(start: i64, end: Option<i64>) -> Vec<i64> {
    match end {
        Some(e) => (start..e).collect(),
        None => (0..start).collect(),
    }
}

/// Push function
pub fn push<T>(mut vec: Vec<T>, item: T) -> Vec<T> {
    vec.push(item);
    vec
}

/// Pop function
pub fn pop<T: Clone>(mut vec: Vec<T>) -> Vec<T> {
    vec.pop();
    vec
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

/// Helper: Convert serde_json::Value to DictValue
fn json_value_to_dict_value(val: &serde_json::Value) -> DictValue {
    match val {
        serde_json::Value::Null => DictValue::Int(0),
        serde_json::Value::Bool(b) => DictValue::Int(if *b { 1 } else { 0 }),
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

/// JSON parse function
pub fn json_parse(json_str: String) -> Option<HashMap<String, DictValue>> {
    let value: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    if let serde_json::Value::Object(obj) = value {
        let mut map = HashMap::new();
        for (k, v) in obj {
            map.insert(k, json_value_to_dict_value(&v));
        }
        Some(map)
    } else {
        None
    }
}

/// JSON stringify function
pub fn json_stringify(value: &HashMap<String, DictValue>) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// YAML parse function
pub fn yaml_parse(yaml_str: String) -> Option<HashMap<String, DictValue>> {
    let value: serde_yaml::Value = serde_yaml::from_str(&yaml_str).ok()?;
    if let serde_yaml::Value::Mapping(obj) = value {
        let mut map = HashMap::new();
        for (k, v) in obj {
            if let serde_yaml::Value::String(key) = k {
                map.insert(key, yaml_value_to_dict_value(&v));
            }
        }
        Some(map)
    } else {
        None
    }
}

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

// ============================================================================
// HIGHER-ORDER FUNCTIONS
// ============================================================================

/// forEach function - executes a function for each element in a vector
pub fn forEach<T, F>(func: F, vec: Vec<T>)
where
    F: Fn(&T)
{
    for item in vec.iter() {
        func(item);
    }
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

/// HTTP session creation
pub fn http_session(base_url: String) -> String {
    // Return a session ID (simplified - just return the base URL)
    base_url
}

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
                let json_body = serde_json::to_string(&body).unwrap_or_default();
                req = req.header("Content-Type", "application/json").body(json_body);
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

/// TCP connection function (placeholder)
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
    // Placeholder - returns error message
    "TCP connection not implemented in compiled mode".to_string()
}
