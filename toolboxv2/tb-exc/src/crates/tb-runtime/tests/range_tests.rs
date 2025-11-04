use tb_runtime::range;

#[test]
fn test_range_single_arg() {
    // range(5, None, None) should produce [0, 1, 2, 3, 4]
    let result = range(5, None, None);
    assert_eq!(result, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_range_two_args() {
    // range(2, Some(7), None) should produce [2, 3, 4, 5, 6]
    let result = range(2, Some(7), None);
    assert_eq!(result, vec![2, 3, 4, 5, 6]);
}

#[test]
fn test_range_three_args_positive_step() {
    // range(0, Some(10), Some(2)) should produce [0, 2, 4, 6, 8]
    let result = range(0, Some(10), Some(2));
    assert_eq!(result, vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_range_three_args_negative_step() {
    // range(10, Some(0), Some(-2)) should produce [10, 8, 6, 4, 2]
    let result = range(10, Some(0), Some(-2));
    assert_eq!(result, vec![10, 8, 6, 4, 2]);
}

#[test]
#[should_panic(expected = "step cannot be zero")]
fn test_range_zero_step_panics() {
    // range(0, Some(10), Some(0)) should panic
    range(0, Some(10), Some(0));
}

#[test]
fn test_range_empty() {
    // range(5, Some(5), None) should produce []
    let result = range(5, Some(5), None);
    assert_eq!(result, Vec::<i64>::new());
}

#[test]
fn test_range_negative_range() {
    // range(5, Some(2), None) with default step 1 should produce []
    let result = range(5, Some(2), None);
    assert_eq!(result, Vec::<i64>::new());
}

#[test]
fn test_range_large_step() {
    // range(0, Some(10), Some(5)) should produce [0, 5]
    let result = range(0, Some(10), Some(5));
    assert_eq!(result, vec![0, 5]);
}

#[test]
fn test_range_step_larger_than_range() {
    // range(0, Some(5), Some(10)) should produce [0]
    let result = range(0, Some(5), Some(10));
    assert_eq!(result, vec![0]);
}

#[test]
fn test_range_negative_start_positive_end() {
    // range(-5, Some(5), None) should produce [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    let result = range(-5, Some(5), None);
    assert_eq!(result, vec![-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]);
}

#[test]
fn test_range_negative_start_negative_end() {
    // range(-10, Some(-5), None) should produce [-10, -9, -8, -7, -6]
    let result = range(-10, Some(-5), None);
    assert_eq!(result, vec![-10, -9, -8, -7, -6]);
}

#[test]
fn test_range_negative_step_descending() {
    // range(5, Some(-5), Some(-1)) should produce [5, 4, 3, 2, 1, 0, -1, -2, -3, -4]
    let result = range(5, Some(-5), Some(-1));
    assert_eq!(result, vec![5, 4, 3, 2, 1, 0, -1, -2, -3, -4]);
}

