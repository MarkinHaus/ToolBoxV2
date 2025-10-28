//! Task Executor for async task management
//!
//! Provides functionality to spawn, await, and cancel TB functions in separate tasks

use std::sync::Arc;
use tb_core::{Value, Result, TBError, Function};
use im::HashMap as ImHashMap;

/// Execute a TB function in a new environment
/// This is used by the spawn builtin to run functions asynchronously
pub fn execute_function_in_task(
    func: Arc<Function>,
    args: Vec<Value>,
    base_env: ImHashMap<Arc<String>, Value>,
) -> Result<Value> {
    use crate::executor::JitExecutor;

    // Create a new executor with the base environment
    let mut executor = JitExecutor::new();

    // Copy the base environment (this includes all built-ins and global variables)
    for (key, value) in base_env.iter() {
        executor.set_variable(key.clone(), value.clone());
    }

    // Bind function parameters to arguments
    if func.params.len() != args.len() {
        return Err(TBError::runtime_error(format!(
            "Function '{}' expects {} arguments, got {}",
            func.name,
            func.params.len(),
            args.len()
        )));
    }

    for (param, arg) in func.params.iter().zip(args.iter()) {
        executor.set_variable(param.clone(), arg.clone());
    }

    // Execute function body
    let mut last_value = Value::None;
    for stmt in &func.body {
        last_value = executor.execute_statement(stmt)?;

        // Check for early return
        if let Some(return_val) = executor.take_return_value() {
            return Ok(return_val);
        }
    }

    Ok(last_value)
}

