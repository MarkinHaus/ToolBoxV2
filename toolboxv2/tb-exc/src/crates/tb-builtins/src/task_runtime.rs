//! Minimal task runtime for executing TB functions in spawned tasks
//!
//! This module provides a lightweight executor for running TB functions
//! asynchronously without creating circular dependencies with tb-jit.

use std::sync::Arc;
use tb_core::{Value, Result, TBError, Function, Statement, Expression, Literal, BinaryOp, UnaryOp};
use im::HashMap as ImHashMap;

/// Minimal task executor for spawned functions
pub struct TaskExecutor {
    env: ImHashMap<Arc<String>, Value>,
    return_value: Option<Value>,
}

impl TaskExecutor {
    pub fn new(env: ImHashMap<Arc<String>, Value>) -> Self {
        Self {
            env,
            return_value: None,
        }
    }

    pub fn execute_function(
        &mut self,
        func: &Function,
        args: Vec<Value>,
    ) -> Result<Value> {
        // Bind parameters
        if func.params.len() != args.len() {
            return Err(TBError::runtime_error(format!(
                "Function '{}' expects {} arguments, got {}",
                func.name,
                func.params.len(),
                args.len()
            )));
        }

        for (param, arg) in func.params.iter().zip(args.iter()) {
            self.env.insert(param.clone(), arg.clone());
        }

        // Execute body
        let mut last_value = Value::None;
        for stmt in &func.body {
            last_value = self.eval_statement(stmt)?;

            if let Some(return_val) = self.return_value.take() {
                return Ok(return_val);
            }
        }

        Ok(last_value)
    }

    fn eval_statement(&mut self, stmt: &Statement) -> Result<Value> {
        match stmt {
            Statement::Let { name, value, .. } => {
                let val = self.eval_expression(value)?;
                self.env.insert(name.clone(), val);
                Ok(Value::None)
            }

            Statement::Assign { target, value, .. } => {
                // For now, only support simple variable assignment
                if let Expression::Ident(name, _) = target {
                    let val = self.eval_expression(value)?;
                    self.env.insert(name.clone(), val);
                    Ok(Value::None)
                } else {
                    Err(TBError::runtime_error("Complex assignment not supported in task runtime"))
                }
            }

            Statement::Return { value, .. } => {
                let val = if let Some(expr) = value {
                    self.eval_expression(expr)?
                } else {
                    Value::None
                };
                self.return_value = Some(val.clone());
                Ok(val)
            }

            Statement::Expression { expr, .. } => {
                self.eval_expression(expr)
            }

            Statement::If { condition, then_block, else_block, .. } => {
                let cond = self.eval_expression(condition)?;
                if cond.is_truthy() {
                    self.eval_block(then_block)
                } else if let Some(else_stmts) = else_block {
                    self.eval_block(else_stmts)
                } else {
                    Ok(Value::None)
                }
            }

            Statement::For { variable, iterable, body, .. } => {
                let iter_value = self.eval_expression(iterable)?;
                match iter_value {
                    Value::List(items) => {
                        for item in items.iter() {
                            self.env.insert(variable.clone(), item.clone());
                            self.eval_block(body)?;
                            if self.return_value.is_some() {
                                break;
                            }
                        }
                        Ok(Value::None)
                    }
                    _ => Err(TBError::runtime_error(format!(
                        "Cannot iterate over {}",
                        iter_value.type_name()
                    ))),
                }
            }

            Statement::While { condition, body, .. } => {
                while self.eval_expression(condition)?.is_truthy() {
                    self.eval_block(body)?;
                    if self.return_value.is_some() {
                        break;
                    }
                }
                Ok(Value::None)
            }

            _ => {
                // For other statement types, return None
                // This is a simplified executor for spawned tasks
                Ok(Value::None)
            }
        }
    }

    fn eval_block(&mut self, stmts: &[Statement]) -> Result<Value> {
        let mut last = Value::None;
        for stmt in stmts {
            last = self.eval_statement(stmt)?;
            if self.return_value.is_some() {
                break;
            }
        }
        Ok(last)
    }

    fn eval_expression(&mut self, expr: &Expression) -> Result<Value> {
        match expr {
            Expression::Literal(lit, _) => Ok(match lit {
                Literal::Int(i) => Value::Int(*i),
                Literal::Float(f) => Value::Float(*f),
                Literal::String(s) => Value::String(s.clone()),
                Literal::Bool(b) => Value::Bool(*b),
                Literal::None => Value::None,
            }),

            Expression::Ident(name, _) => {
                self.env.get(name).cloned().ok_or_else(|| {
                    TBError::runtime_error(format!("Undefined variable: {}", name))
                })
            }

            Expression::Binary { left, op, right, .. } => {
                let l = self.eval_expression(left)?;
                let r = self.eval_expression(right)?;
                self.eval_binary_op(&l, op, &r)
            }

            Expression::Unary { op, operand, .. } => {
                let val = self.eval_expression(operand)?;
                self.eval_unary_op(op, &val)
            }

            Expression::Call { callee, args, .. } => {
                let func_val = self.eval_expression(callee)?;
                let arg_vals: Result<Vec<_>> = args.iter()
                    .map(|a| self.eval_expression(a))
                    .collect();
                let arg_vals = arg_vals?;

                match func_val {
                    Value::Function(f) => {
                        // ✅ CLOSURE FIX: Use closure environment if available
                        let base_env = if let Some(ref closure_env) = f.closure_env {
                            // Read from the RwLock
                            closure_env.read().unwrap().clone()
                        } else {
                            self.env.clone()
                        };
                        let mut sub_executor = TaskExecutor::new(base_env);
                        sub_executor.execute_function(&f, arg_vals)
                    }
                    Value::NativeFunction(nf) => {
                        (nf.func)(arg_vals)
                    }
                    _ => Err(TBError::runtime_error(format!(
                        "Cannot call {}",
                        func_val.type_name()
                    ))),
                }
            }

            Expression::List { elements, .. } => {
                let vals: Result<Vec<_>> = elements.iter()
                    .map(|e| self.eval_expression(e))
                    .collect();
                Ok(Value::List(Arc::new(vals?)))
            }

            Expression::Dict { entries, .. } => {
                let mut map = ImHashMap::new();
                for (k, v) in entries {
                    let val = self.eval_expression(v)?;
                    map.insert(k.clone(), val);
                }
                Ok(Value::Dict(Arc::new(map)))
            }

            Expression::Index { object, index, .. } => {
                let obj = self.eval_expression(object)?;
                let idx = self.eval_expression(index)?;

                match (&obj, &idx) {
                    (Value::List(items), Value::Int(i)) => {
                        let index = if *i < 0 {
                            (items.len() as i64 + i) as usize
                        } else {
                            *i as usize
                        };
                        items.get(index).cloned().ok_or_else(|| {
                            TBError::runtime_error("Index out of bounds")
                        })
                    }
                    (Value::Dict(map), Value::String(key)) => {
                        map.get(key).cloned().ok_or_else(|| {
                            TBError::runtime_error(format!("Key not found: {}", key))
                        })
                    }
                    _ => Err(TBError::runtime_error("Invalid index operation")),
                }
            }

            Expression::Range { start, end, inclusive, .. } => {
                let start_val = self.eval_expression(start)?;
                let end_val = self.eval_expression(end)?;

                match (start_val, end_val) {
                    (Value::Int(s), Value::Int(e)) => {
                        let range_end = if *inclusive { e + 1 } else { e };
                        let values: Vec<Value> = if s <= range_end {
                            (s..range_end).map(Value::Int).collect()
                        } else {
                            (range_end..s).rev().map(Value::Int).collect()
                        };
                        Ok(Value::List(Arc::new(values)))
                    }
                    _ => Err(TBError::runtime_error(
                        "Range expressions require integer start and end values"
                    )),
                }
            }

            _ => {
                // For other expression types, return None
                Ok(Value::None)
            }
        }
    }

    fn eval_binary_op(&self, left: &Value, op: &BinaryOp, right: &Value) -> Result<Value> {
        use BinaryOp::*;

        match (left, op, right) {
            (Value::Int(l), Add, Value::Int(r)) => Ok(Value::Int(l + r)),
            (Value::Int(l), Sub, Value::Int(r)) => Ok(Value::Int(l - r)),
            (Value::Int(l), Mul, Value::Int(r)) => Ok(Value::Int(l * r)),
            (Value::Int(l), Div, Value::Int(r)) => {
                if *r == 0 {
                    Err(TBError::runtime_error("Division by zero"))
                } else {
                    Ok(Value::Int(l / r))
                }
            }
            (Value::Int(l), Mod, Value::Int(r)) => {
                if *r == 0 {
                    Err(TBError::runtime_error("Modulo by zero"))
                } else {
                    Ok(Value::Int(l % r))
                }
            }
            (Value::Int(l), Eq, Value::Int(r)) => Ok(Value::Bool(l == r)),
            (Value::Int(l), NotEq, Value::Int(r)) => Ok(Value::Bool(l != r)),
            (Value::Int(l), Lt, Value::Int(r)) => Ok(Value::Bool(l < r)),
            (Value::Int(l), LtEq, Value::Int(r)) => Ok(Value::Bool(l <= r)),
            (Value::Int(l), Gt, Value::Int(r)) => Ok(Value::Bool(l > r)),
            (Value::Int(l), GtEq, Value::Int(r)) => Ok(Value::Bool(l >= r)),

            (Value::String(l), Add, Value::String(r)) => {
                Ok(Value::String(Arc::new(format!("{}{}", l, r))))
            }

            _ => Err(TBError::runtime_error(format!(
                "Unsupported operation: {} {:?} {}",
                left.type_name(),
                op,
                right.type_name()
            ))),
        }
    }

    fn eval_unary_op(&self, op: &UnaryOp, operand: &Value) -> Result<Value> {
        match (op, operand) {
            (UnaryOp::Not, val) => Ok(Value::Bool(!val.is_truthy())),
            (UnaryOp::Neg, Value::Int(i)) => Ok(Value::Int(-i)),
            (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
            _ => Err(TBError::runtime_error(format!(
                "Unsupported unary operation: {:?} {}",
                op,
                operand.type_name()
            ))),
        }
    }
}

