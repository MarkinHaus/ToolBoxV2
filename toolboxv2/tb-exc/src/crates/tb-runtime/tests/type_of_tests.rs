use tb_runtime::*;
use std::collections::HashMap;

#[test]
fn test_type_of_i64() {
    let value: i64 = 42;
    assert_eq!(type_of_i64(&value), "int");
}

#[test]
fn test_type_of_f64() {
    let value: f64 = 3.14;
    assert_eq!(type_of_f64(&value), "float");
}

#[test]
fn test_type_of_string() {
    let value = "hello".to_string();
    assert_eq!(type_of_string(&value), "string");
}

#[test]
fn test_type_of_bool() {
    let value = true;
    assert_eq!(type_of_bool(&value), "bool");
}

#[test]
fn test_type_of_vec_i64() {
    let value = vec![1, 2, 3];
    assert_eq!(type_of_vec_i64(&value), "list");
}

#[test]
fn test_type_of_vec_f64() {
    let value = vec![1.0, 2.0, 3.0];
    assert_eq!(type_of_vec_f64(&value), "list");
}

#[test]
fn test_type_of_vec_string() {
    let value = vec!["a".to_string(), "b".to_string()];
    assert_eq!(type_of_vec_string(&value), "list");
}

#[test]
fn test_type_of_vec_bool() {
    let value = vec![true, false];
    assert_eq!(type_of_vec_bool(&value), "list");
}

#[test]
fn test_type_of_hashmap() {
    let mut value: HashMap<String, i64> = HashMap::new();
    value.insert("key".to_string(), 42);
    assert_eq!(type_of_hashmap(&value), "dict");
}

#[test]
fn test_type_of_option() {
    let value: Option<i64> = None;
    assert_eq!(type_of_option(&value), "none");
    
    let value: Option<i64> = Some(42);
    assert_eq!(type_of_option(&value), "none");
}

#[test]
fn test_type_of_unit() {
    let value = ();
    assert_eq!(type_of_unit(&value), "none");
}

#[test]
fn test_type_of_dict_value() {
    let value = DictValue::Int(42);
    assert_eq!(type_of_dict_value(&value), "int");
    
    let value = DictValue::String("hello".to_string());
    assert_eq!(type_of_dict_value(&value), "string");
    
    let value = DictValue::Bool(true);
    assert_eq!(type_of_dict_value(&value), "bool");
    
    let value = DictValue::Float(3.14);
    assert_eq!(type_of_dict_value(&value), "float");
    
    let value = DictValue::List(vec![]);
    assert_eq!(type_of_dict_value(&value), "list");
    
    let value = DictValue::Dict(HashMap::new());
    assert_eq!(type_of_dict_value(&value), "dict");
}

