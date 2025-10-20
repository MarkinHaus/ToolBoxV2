use crate::{TypeEnvironment, TypeInference};
use std::sync::Arc;
use tb_core::*;

/// Type checker with inference and validation
pub struct TypeChecker {
    env: TypeEnvironment,
    errors: Vec<TBError>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut env = TypeEnvironment::new();

        // Register built-in types
        Self::register_builtins(&mut env);

        Self {
            env,
            errors: Vec::new(),
        }
    }

    fn register_builtins(env: &mut TypeEnvironment) {
        // Built-in functions with Type::Any for polymorphism
        let builtins = vec![
            ("print", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::None),
            }),
            ("len", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Int),
            }),
            ("push", Type::Function {
                params: vec![Type::Any, Type::Any],
                return_type: Box::new(Type::Any),
            }),
            ("pop", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Any),
            }),
            ("keys", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::List(Box::new(Type::String))),
            }),
            ("values", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::List(Box::new(Type::Any))),
            }),
            ("range", Type::Function {
                params: vec![Type::Any],  // Variadic - accepts 1 or 2 args, checked at runtime
                return_type: Box::new(Type::List(Box::new(Type::Int))),
            }),
            ("str", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::String),
            }),
            ("int", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Int),
            }),
            ("float", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Float),
            }),
            // File I/O functions
            ("file_exists", Type::Function {
                params: vec![Type::String, Type::Any],  // path, blob=false
                return_type: Box::new(Type::Bool),
            }),
            ("read_file", Type::Function {
                params: vec![Type::String, Type::Any],  // path, blob=false
                return_type: Box::new(Type::String),
            }),
            ("write_file", Type::Function {
                params: vec![Type::String, Type::String, Type::Any],  // path, content, blob=false
                return_type: Box::new(Type::None),
            }),
            ("append_file", Type::Function {
                params: vec![Type::String, Type::String, Type::Any],  // path, content, blob=false
                return_type: Box::new(Type::None),
            }),
            ("delete_file", Type::Function {
                params: vec![Type::String, Type::Any],  // path, blob=false
                return_type: Box::new(Type::None),
            }),
            // Utility functions
            ("json_parse", Type::Function {
                params: vec![Type::String],
                return_type: Box::new(Type::Any),
            }),
            ("json_stringify", Type::Function {
                params: vec![Type::Any, Type::Any],  // value, pretty=false
                return_type: Box::new(Type::String),
            }),
            ("yaml_parse", Type::Function {
                params: vec![Type::String],
                return_type: Box::new(Type::Any),
            }),
            ("yaml_stringify", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::String),
            }),
            ("time", Type::Function {
                params: vec![Type::Any],  // timezone (optional)
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
        ];

        for (name, ty) in builtins {
            env.define(Arc::new(name.to_string()), ty);
        }
    }

    /// Type check an entire program
    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        for stmt in &program.statements {
            self.check_statement(stmt)?;
        }

        if !self.errors.is_empty() {
            return Err(self.errors[0].clone());
        }

        Ok(())
    }

    /// Check a statement and update environment
    pub fn check_statement(&mut self, stmt: &Statement) -> Result<Type> {
        match stmt {
            Statement::Let { name, type_annotation, value, .. } => {
                let inferred_type = self.check_expression(value)?;

                // Validate against annotation if provided
                if let Some(annotated) = type_annotation {
                    if !self.is_assignable(&inferred_type, annotated) {
                        return Err(TBError::TypeError {
                            message: format!(
                                "Type mismatch: expected {:?}, got {:?}",
                                annotated, inferred_type
                            ),
                        });
                    }
                }

                self.env.define(Arc::clone(name), inferred_type.clone());
                Ok(Type::None)
            }

            Statement::Assign { name, value, .. } => {
                // Check if variable exists
                let existing_type = self.env.lookup(name).ok_or_else(|| TBError::UndefinedVariable {
                    name: name.to_string(),
                })?.clone();

                let value_type = self.check_expression(value)?;

                // Check type compatibility
                if !self.is_assignable(&value_type, &existing_type) {
                    return Err(TBError::TypeError {
                        message: format!(
                            "Cannot assign {:?} to variable of type {:?}",
                            value_type, existing_type
                        ),
                    });
                }

                Ok(Type::None)
            }

            Statement::Function { name, params, return_type, body, .. } => {
                // Create function type
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| p.type_annotation.clone().unwrap_or(Type::Generic(Arc::clone(&p.name))))
                    .collect();

                let func_type = Type::Function {
                    params: param_types.clone(),
                    return_type: Box::new(return_type.clone().unwrap_or(Type::None)),
                };

                // Register function in environment
                self.env.define(Arc::clone(name), func_type);

                // Check function body in new scope
                let mut body_env = self.env.child();

                // Bind parameters
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    body_env.define(Arc::clone(&param.name), ty.clone());
                }

                // Temporarily swap environment
                let old_env = std::mem::replace(&mut self.env, body_env);

                let mut body_type = Type::None;
                let mut has_return = false;

                for stmt in body {
                    let stmt_type = self.check_statement(stmt)?;
                    // If we find a return statement, use its type
                    if matches!(stmt, Statement::Return { .. }) {
                        body_type = stmt_type;
                        has_return = true;
                    } else if !has_return {
                        // Only update body_type if we haven't seen a return yet
                        body_type = stmt_type;
                    }
                }

                // Restore environment
                self.env = old_env;

                // Validate return type if specified
                if let Some(expected_return) = return_type {
                    // More flexible return type checking for Generic types
                    let types_match = match (expected_return, &body_type) {
                        // Exact match
                        (a, b) if a == b => true,

                        // Generic matches anything
                        (Type::Generic(_), _) => true,
                        (_, Type::Generic(_)) => true,

                        // List(Generic) matches List(T)
                        (Type::List(a), Type::List(b)) => {
                            matches!(a.as_ref(), Type::Generic(_)) ||
                            matches!(b.as_ref(), Type::Generic(_)) ||
                            a == b
                        }

                        // Any matches anything
                        (Type::Any, _) | (_, Type::Any) => true,

                        // Use is_assignable for other cases
                        _ => self.is_assignable(&body_type, expected_return),
                    };

                    // If function has explicit return type but no return statement, that's ok if return type is None
                    if !has_return && expected_return != &Type::None {
                        // Allow it - runtime will check
                    } else if !types_match {
                        return Err(TBError::TypeError {
                            message: format!(
                                "Function return type mismatch: expected {:?}, got {:?}",
                                expected_return, body_type
                            ),
                        });
                    }
                }

                Ok(Type::None)
            }

            Statement::If { condition, then_block, else_block, .. } => {
                let cond_type = self.check_expression(condition)?;
                // Allow Type::Any for dynamic conditions (e.g., from dict lookups)
                if cond_type != Type::Bool && cond_type != Type::Any {
                    return Err(TBError::TypeError {
                        message: format!("If condition must be bool, got {:?}", cond_type),
                    });
                }

                // Check branches in child scopes
                let old_env = self.env.clone();

                for stmt in then_block {
                    self.check_statement(stmt)?;
                }

                self.env = old_env.clone();

                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        self.check_statement(stmt)?;
                    }
                }

                self.env = old_env;
                Ok(Type::None)
            }

            Statement::For { variable, iterable, body, .. } => {
                let iter_type = self.check_expression(iterable)?;

                // Extract element type from iterable
                let elem_type = match iter_type {
                    Type::List(elem) => *elem,
                    // Allow iteration over Generic types (like "list", "dict")
                    Type::Generic(ref name) if name.as_str() == "list" => Type::Any,
                    Type::Generic(ref name) if name.as_str() == "dict" => Type::Any,
                    // Allow iteration over ANY Generic type (not just "list" or "dict")
                    Type::Generic(_) => Type::Any,
                    // Allow iteration over Type::Any
                    Type::Any => Type::Any,
                    _ => return Err(TBError::TypeError {
                        message: format!("Cannot iterate over {:?}", iter_type),
                    }),
                };

                // Check body in child scope with loop variable
                let mut body_env = self.env.child();
                body_env.define(Arc::clone(variable), elem_type);

                let old_env = std::mem::replace(&mut self.env, body_env);

                for stmt in body {
                    self.check_statement(stmt)?;
                }

                self.env = old_env;
                Ok(Type::None)
            }

            Statement::While { condition, body, .. } => {
                let cond_type = self.check_expression(condition)?;
                if cond_type != Type::Bool {
                    return Err(TBError::TypeError {
                        message: format!("While condition must be bool, got {:?}", cond_type),
                    });
                }

                let old_env = self.env.clone();
                for stmt in body {
                    self.check_statement(stmt)?;
                }
                self.env = old_env;

                Ok(Type::None)
            }

            Statement::Return { value, .. } => {
                if let Some(expr) = value {
                    self.check_expression(expr)
                } else {
                    Ok(Type::None)
                }
            }

            Statement::Expression { expr, .. } => self.check_expression(expr),

            Statement::Match { value, arms, .. } => {
                let value_type = self.check_expression(value)?;

                let mut result_type: Option<Type> = None;

                for arm in arms {
                    // Check pattern matches value type
                    self.check_pattern(&arm.pattern, &value_type)?;

                    // Check arm body
                    let arm_type = self.check_expression(&arm.body)?;

                    // Unify with previous arms
                    if let Some(ref prev) = result_type {
                        result_type = Some(TypeInference::unify(prev.clone(), arm_type)?);
                    } else {
                        result_type = Some(arm_type);
                    }
                }

                Ok(result_type.unwrap_or(Type::None))
            }

            Statement::Plugin { definitions, .. } => {
                // Register plugin modules as dictionaries in the environment
                for def in definitions {
                    // Create a generic type for the plugin module
                    // In a full implementation, we would extract function signatures
                    let module_type = Type::Generic(Arc::new(format!("Plugin<{}>", def.name)));
                    self.env.define(Arc::clone(&def.name), module_type);
                }
                Ok(Type::None)
            }

            Statement::Import { .. } => {
                // TODO: Implement import type checking
                Ok(Type::None)
            }

            Statement::Config { .. } => {
                // Config blocks don't affect types
                Ok(Type::None)
            }

            _ => Ok(Type::None),
        }
    }

    /// Check an expression and infer its type
    pub fn check_expression(&mut self, expr: &Expression) -> Result<Type> {
        match expr {
            Expression::Literal(lit, _) => Ok(TypeInference::infer_literal(lit)),

            Expression::Ident(name, _span) => {
                self.env.lookup(name.as_ref())
                    .cloned()
                    .ok_or_else(|| TBError::UndefinedVariable {
                        name: name.to_string(),
                    })
            }

            Expression::Binary { op, left, right, .. } => {
                let left_type = self.check_expression(left)?;
                let right_type = self.check_expression(right)?;

                // Handle generic types gracefully
                match (&left_type, &right_type) {
                    (Type::Generic(_), _) | (_, Type::Generic(_)) => {
                        // For generic types, assume operation is valid and return Any
                        Ok(Type::Any)
                    }
                    _ => TypeInference::infer_binary_op(op, &left_type, &right_type),
                }
            }

            Expression::Unary { op, operand, .. } => {
                let operand_type = self.check_expression(operand)?;

                // Handle generic types
                if matches!(operand_type, Type::Generic(_)) {
                    Ok(Type::Any)
                } else {
                    TypeInference::infer_unary_op(op, &operand_type)
                }
            }

            Expression::Call { callee, args, .. } => {
                let func_type = self.check_expression(callee)?;

                match func_type {
                    Type::Function { params, return_type } => {
                        // If params contains Type::Any, allow variadic arguments
                        let is_variadic = params.iter().any(|p| matches!(p, Type::Any));

                        if !is_variadic && args.len() != params.len() {
                            return Err(TBError::TypeError {
                                message: format!(
                                    "Expected {} arguments, got {}",
                                    params.len(),
                                    args.len()
                                ),
                            });
                        }

                        // Check argument types
                        if is_variadic {
                            // For variadic functions, just check all args are valid expressions
                            for arg in args {
                                self.check_expression(arg)?;
                            }
                        } else {
                            // For non-variadic, check types match
                            for (arg, param_type) in args.iter().zip(params.iter()) {
                                let arg_type = self.check_expression(arg)?;
                                // Allow Generic types to match with their expected types
                                let types_match = self.is_assignable(&arg_type, param_type) ||
                                    matches!(param_type, Type::Generic(_)) ||
                                    matches!(arg_type, Type::Generic(_));

                                if !types_match {
                                    return Err(TBError::TypeError {
                                        message: format!(
                                            "Argument type mismatch: expected {:?}, got {:?}",
                                            param_type, arg_type
                                        ),
                                    });
                                }
                            }
                        }

                        Ok(*return_type)
                    }
                    Type::Any => {
                        // Allow calling Any type (e.g., plugin functions)
                        // Just check that all arguments are valid expressions
                        for arg in args {
                            self.check_expression(arg)?;
                        }
                        Ok(Type::Any)
                    }
                    _ => Err(TBError::TypeError {
                        message: format!("Cannot call non-function type {:?}", func_type),
                    }),
                }
            }

            Expression::List { elements, .. } => {
                if elements.is_empty() {
                    return Ok(Type::List(Box::new(Type::Generic(Arc::new("T".to_string())))));
                }

                let first_type = self.check_expression(&elements[0])?;

                // Ensure all elements have compatible types
                for elem in &elements[1..] {
                    let elem_type = self.check_expression(elem)?;
                    // Use is_assignable instead of types_compatible for better Dict(String, Any) handling
                    if !self.is_assignable(&elem_type, &first_type) && !self.is_assignable(&first_type, &elem_type) {
                        return Err(TBError::TypeError {
                            message: format!(
                                "List elements must have compatible types, got {:?} and {:?}",
                                first_type, elem_type
                            ),
                        });
                    }
                }

                Ok(Type::List(Box::new(first_type)))
            }

            Expression::Dict { entries, .. } => {
                if entries.is_empty() {
                    return Ok(Type::Dict(
                        Box::new(Type::String),
                        Box::new(Type::Any),
                    ));
                }

                // Allow heterogeneous dictionaries
                let mut value_types = Vec::new();
                for (_, value) in entries {
                    let value_type = self.check_expression(value)?;
                    value_types.push(value_type);
                }

                // Check if all types are the same
                let first_type = &value_types[0];
                let all_same = value_types.iter().all(|t| t == first_type);

                if all_same {
                    // Homogeneous dict
                    Ok(Type::Dict(
                        Box::new(Type::String),
                        Box::new(first_type.clone()),
                    ))
                } else {
                    // Heterogeneous dict - use Any
                    Ok(Type::Dict(
                        Box::new(Type::String),
                        Box::new(Type::Any),
                    ))
                }
            }

            Expression::Index { object, index, .. } => {
                let obj_type = self.check_expression(object)?;
                let idx_type = self.check_expression(index)?;

                match obj_type {
                    Type::List(elem_type) => {
                        if idx_type != Type::Int && !matches!(idx_type, Type::Any) {
                            return Err(TBError::TypeError {
                                message: "List index must be int".to_string(),
                            });
                        }
                        Ok(*elem_type)
                    }
                    Type::Dict(key_type, value_type) => {
                        if !self.is_assignable(&idx_type, &key_type) && !matches!(idx_type, Type::Any) {
                            return Err(TBError::TypeError {
                                message: format!(
                                    "Dict key type mismatch: expected {:?}, got {:?}",
                                    key_type, idx_type
                                ),
                            });
                        }
                        Ok(*value_type)
                    }
                    // Allow indexing on Type::Any - runtime will check
                    Type::Any => Ok(Type::Any),
                    // Allow indexing on Generic types (like "list", "dict")
                    Type::Generic(ref name) => {
                        if name.as_str() == "list" || name.as_str() == "dict" {
                            Ok(Type::Any)
                        } else {
                            Err(TBError::TypeError {
                                message: format!("Cannot index type Generic({})", name),
                            })
                        }
                    }
                    ref other => Err(TBError::TypeError {
                        message: format!("Cannot index type {:?}", other),
                    }),
                }
            }

            Expression::Match { value, arms, .. } => {
                let value_type = self.check_expression(value)?;

                if arms.is_empty() {
                    return Err(TBError::TypeError {
                        message: "Match expression must have at least one arm".to_string(),
                    });
                }

                // Check all patterns are compatible with value type
                for arm in arms {
                    self.check_pattern(&arm.pattern, &value_type)?;
                }

                // All arms should return the same type
                let first_arm_type = self.check_expression(&arms[0].body)?;
                for arm in &arms[1..] {
                    let arm_type = self.check_expression(&arm.body)?;
                    if !TypeInference::types_compatible(&first_arm_type, &arm_type) {
                        return Err(TBError::TypeError {
                            message: format!(
                                "Match arms have incompatible types: {:?} vs {:?}",
                                first_arm_type, arm_type
                            ),
                        });
                    }
                }

                Ok(first_arm_type)
            }

            Expression::Member { object, member, .. } => {
                let obj_type = self.check_expression(object)?;

                match obj_type {
                    Type::Dict(_, value_type) => {
                        // Member access on dict returns the value type
                        Ok(*value_type)
                    }
                    Type::Generic(name) if name.starts_with("Plugin<") => {
                        // Plugin module member access - return Any to allow any function call
                        // In a full implementation, we would look up the actual function signature
                        Ok(Type::Any)
                    }
                    _ => {
                        // For other types, return Unknown
                        Ok(Type::Generic(Arc::new("Unknown".to_string())))
                    }
                }
            }

            _ => Ok(Type::Generic(Arc::new("Unknown".to_string()))),
        }
    }

    fn check_pattern(&self, pattern: &Pattern, value_type: &Type) -> Result<()> {
        match pattern {
            Pattern::Literal(lit) => {
                let pattern_type = TypeInference::infer_literal(lit);
                if !TypeInference::types_compatible(&pattern_type, value_type) {
                    return Err(TBError::TypeError {
                        message: format!(
                            "Pattern type {:?} doesn't match value type {:?}",
                            pattern_type, value_type
                        ),
                    });
                }
                Ok(())
            }
            Pattern::Range { .. } => {
                // Range patterns only work with integers
                if !matches!(value_type, Type::Int) && !matches!(value_type, Type::Any) {
                    return Err(TBError::TypeError {
                        message: format!(
                            "Range pattern requires Int type, got {:?}",
                            value_type
                        ),
                    });
                }
                Ok(())
            }
            Pattern::Wildcard | Pattern::Ident(_) => Ok(()),
        }
    }

    fn is_assignable(&self, from: &Type, to: &Type) -> bool {
        if from == to {
            return true;
        }

        // Type::Any is always assignable
        if matches!(to, Type::Any) || matches!(from, Type::Any) {
            return true;
        }

        // Dict compatibility with Any values
        match (from, to) {
            (Type::Dict(k1, v1), Type::Dict(k2, v2)) => {
                let keys_match = k1 == k2 || matches!(k1.as_ref(), Type::Any) || matches!(k2.as_ref(), Type::Any);
                let values_match = v1 == v2 || matches!(v1.as_ref(), Type::Any) || matches!(v2.as_ref(), Type::Any);
                return keys_match && values_match;
            }
            (Type::List(t1), Type::List(t2)) => {
                return t1 == t2 || matches!(t1.as_ref(), Type::Any) || matches!(t2.as_ref(), Type::Any);
            }
            _ => {}
        }

        // Check type compatibility
        TypeInference::types_compatible(from, to)
    }

    pub fn environment(&self) -> &TypeEnvironment {
        &self.env
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

