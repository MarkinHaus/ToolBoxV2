use tb_core::*;
use tb_builtins::builtins_impl::builtin_range;
use std::sync::Arc;

#[test]
fn test_range_single_arg() {
    // range(5) should produce [0, 1, 2, 3, 4]
    let result = builtin_range(vec![Value::Int(5)]).unwrap();
    
    if let Value::List(list) = result {
        assert_eq!(list.len(), 5);
        assert_eq!(list[0], Value::Int(0));
        assert_eq!(list[1], Value::Int(1));
        assert_eq!(list[2], Value::Int(2));
        assert_eq!(list[3], Value::Int(3));
        assert_eq!(list[4], Value::Int(4));
    } else {
        panic!("Expected list result");
    }
}

#[test]
fn test_range_two_args() {
    // range(2, 7) should produce [2, 3, 4, 5, 6]
    let result = builtin_range(vec![Value::Int(2), Value::Int(7)]).unwrap();
    
    if let Value::List(list) = result {
        assert_eq!(list.len(), 5);
        assert_eq!(list[0], Value::Int(2));
        assert_eq!(list[1], Value::Int(3));
        assert_eq!(list[2], Value::Int(4));
        assert_eq!(list[3], Value::Int(5));
        assert_eq!(list[4], Value::Int(6));
    } else {
        panic!("Expected list result");
    }
}

#[test]
fn test_range_three_args_positive_step() {
    // range(0, 10, 2) should produce [0, 2, 4, 6, 8]
    let result = builtin_range(vec![Value::Int(0), Value::Int(10), Value::Int(2)]).unwrap();
    
    if let Value::List(list) = result {
        assert_eq!(list.len(), 5);
        assert_eq!(list[0], Value::Int(0));
        assert_eq!(list[1], Value::Int(2));
        assert_eq!(list[2], Value::Int(4));
        assert_eq!(list[3], Value::Int(6));
        assert_eq!(list[4], Value::Int(8));
    } else {
        panic!("Expected list result");
    }
}

#[test]
fn test_range_three_args_negative_step() {
    // range(10, 0, -2) should produce [10, 8, 6, 4, 2]
    let result = builtin_range(vec![Value::Int(10), Value::Int(0), Value::Int(-2)]).unwrap();
    
    if let Value::List(list) = result {
        assert_eq!(list.len(), 5);
        assert_eq!(list[0], Value::Int(10));
        assert_eq!(list[1], Value::Int(8));
        assert_eq!(list[2], Value::Int(6));
        assert_eq!(list[3], Value::Int(4));
        assert_eq!(list[4], Value::Int(2));
    } else {
        panic!("Expected list result");
    }
}

#[test]
fn test_range_zero_step_error() {
    // range(0, 10, 0) should return an error
    let result = builtin_range(vec![Value::Int(0), Value::Int(10), Value::Int(0)]);
    assert!(result.is_err());
    
    if let Err(e) = result {
        assert!(e.to_string().contains("step cannot be zero"));
    }
}

#[test]
fn test_range_empty() {
    // range(5, 5) should produce []
    let result = builtin_range(vec![Value::Int(5), Value::Int(5)]).unwrap();
    
    if let Value::List(list) = result {
        assert_eq!(list.len(), 0);
    } else {
        panic!("Expected list result");
    }
}

#[test]
fn test_range_negative_range() {
    // range(5, 2) with default step 1 should produce []
    let result = builtin_range(vec![Value::Int(5), Value::Int(2)]).unwrap();
    
    if let Value::List(list) = result {
        assert_eq!(list.len(), 0);
    } else {
        panic!("Expected list result");
    }
}

#[test]
fn test_range_invalid_arg_count() {
    // range() with 0 or 4+ arguments should return an error
    let result = builtin_range(vec![]);
    assert!(result.is_err());
    
    let result = builtin_range(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]);
    assert!(result.is_err());
}

#[test]
fn test_range_non_integer_args() {
    // range("hello") should return an error
    let result = builtin_range(vec![Value::String(Arc::new("hello".to_string()))]);
    assert!(result.is_err());
    
    if let Err(e) = result {
        assert!(e.to_string().contains("integer arguments"));
    }
}

