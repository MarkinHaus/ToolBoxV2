use crate::{TypeEnvironment, TypeInference};
use std::sync::Arc;
use std::path::PathBuf;
use tb_core::*;

/// Type checker with inference and validation
pub struct TypeChecker {
    env: TypeEnvironment,
    errors: Vec<TBError>,
    source_context: Option<SourceContext>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut env = TypeEnvironment::new();

        // Register built-in types
        Self::register_builtins(&mut env);

        Self {
            env,
            errors: Vec::new(),
            source_context: None,
        }
    }

    /// Create a new TypeChecker with source context for better error messages
    pub fn new_with_source(source: String, file_path: Option<PathBuf>) -> Self {
        let mut checker = Self::new();
        checker.source_context = Some(SourceContext::new(source, file_path));
        checker
    }

    fn register_builtins(env: &mut TypeEnvironment) {
        // Built-in constants
        env.define(Arc::new("None".to_string()), Type::None);
        env.define(Arc::new("true".to_string()), Type::Bool);
        env.define(Arc::new("false".to_string()), Type::Bool);

        // Built-in functions with Type::Any for polymorphism
        let builtins = vec![
            ("print", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::None),
            }),
            // len: accepts List<T>, Dict<K,V>, or String
            ("len", Type::Function {
                params: vec![Type::Generic(Arc::new("T".to_string()))],
                return_type: Box::new(Type::Int),
            }),
            // push: List<T> + T -> List<T>
            ("push", Type::Function {
                params: vec![
                    Type::List(Box::new(Type::Generic(Arc::new("T".to_string())))),
                    Type::Generic(Arc::new("T".to_string()))
                ],
                return_type: Box::new(Type::List(Box::new(Type::Generic(Arc::new("T".to_string()))))),
            }),
            // pop: List<T> -> List<Any>
            // Returns [[new_list], popped_value] to match JIT mode behavior
            ("pop", Type::Function {
                params: vec![Type::List(Box::new(Type::Generic(Arc::new("T".to_string()))))],
                return_type: Box::new(Type::List(Box::new(Type::Any))),
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
            ("dict", Type::Function {
                params: vec![Type::Any],  // Accepts 0 or 1 arg
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
            ("list", Type::Function {
                params: vec![Type::Any],  // Accepts 0 or 1 arg
                return_type: Box::new(Type::List(Box::new(Type::Any))),
            }),
            ("import", Type::Function {
                params: vec![Type::String],
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
            // File I/O functions
            ("open", Type::Function {
                params: vec![Type::String, Type::Any, Type::Any, Type::Any],  // path, mode, key, encoding
                return_type: Box::new(Type::String),  // file handle
            }),
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
            ("list_dir", Type::Function {
                params: vec![Type::String],  // path
                return_type: Box::new(Type::List(Box::new(Type::String))),
            }),
            ("create_dir", Type::Function {
                params: vec![Type::String],  // path
                return_type: Box::new(Type::None),
            }),
            // System functions
            ("execute", Type::Function {
                params: vec![Type::String, Type::Any],  // command, args
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
            ("get_env", Type::Function {
                params: vec![Type::String],  // var name
                return_type: Box::new(Type::String),
            }),
            ("sleep", Type::Function {
                params: vec![Type::Any],  // duration (int or float)
                return_type: Box::new(Type::None),
            }),
            // Introspection functions
            ("type_of", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::String),
            }),
            ("dir", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::List(Box::new(Type::String))),
            }),
            ("has_attr", Type::Function {
                params: vec![Type::Any, Type::String],  // object, attribute name
                return_type: Box::new(Type::Bool),
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
            // Async task management
            // spawn: (T -> U, List<T>) -> String (task ID)
            ("spawn", Type::Function {
                params: vec![
                    Type::Function {
                        params: vec![Type::Generic(Arc::new("T".to_string()))],
                        return_type: Box::new(Type::Generic(Arc::new("U".to_string()))),
                    },
                    Type::List(Box::new(Type::Generic(Arc::new("T".to_string())))),
                ],
                return_type: Box::new(Type::String),  // task ID
            }),
            // await_task: String -> T (generic result)
            ("await_task", Type::Function {
                params: vec![Type::String],  // task ID
                return_type: Box::new(Type::Generic(Arc::new("T".to_string()))),  // task result
            }),
            ("cancel_task", Type::Function {
                params: vec![Type::String],  // task ID
                return_type: Box::new(Type::Bool),  // success
            }),
            // Network functions
            ("create_server", Type::Function {
                params: vec![Type::String, Type::String, Type::Any],  // protocol, address, callback
                return_type: Box::new(Type::String),  // server ID
            }),
            ("stop_server", Type::Function {
                params: vec![Type::String],  // server ID
                return_type: Box::new(Type::Bool),  // success
            }),
            ("send_message", Type::Function {
                params: vec![Type::String, Type::String],  // connection ID, message
                return_type: Box::new(Type::Bool),  // success
            }),
            ("connect_to", Type::Function {
                params: vec![Type::Any, Type::Any, Type::Any, Type::String, Type::Int, Type::String],  // on_connect, on_disconnect, on_msg, host, port, type
                return_type: Box::new(Type::String),  // connection ID
            }),
            ("send_to", Type::Function {
                params: vec![Type::String, Type::Any],  // connection ID, message
                return_type: Box::new(Type::None),
            }),
            ("http_session", Type::Function {
                params: vec![Type::String, Type::Any, Type::Any],  // base_url, headers, cookies_file
                return_type: Box::new(Type::String),  // session ID
            }),
            ("http_request", Type::Function {
                params: vec![Type::String, Type::String, Type::String, Type::Any],  // session_id, url, method, data
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),  // response dict
            }),
            // Cache Management
            ("cache_stats", Type::Function {
                params: vec![],
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
            ("cache_clear", Type::Function {
                params: vec![],
                return_type: Box::new(Type::None),
            }),
            ("cache_invalidate", Type::Function {
                params: vec![Type::String],  // key
                return_type: Box::new(Type::Bool),
            }),
            // Plugin functions
            ("load_plugin", Type::Function {
                params: vec![Type::String, Type::String],  // path, language
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
            ("plugin_info", Type::Function {
                params: vec![Type::String],  // plugin name
                return_type: Box::new(Type::Dict(Box::new(Type::String), Box::new(Type::Any))),
            }),
            ("list_plugins", Type::Function {
                params: vec![],
                return_type: Box::new(Type::List(Box::new(Type::String))),
            }),
            ("reload_plugin", Type::Function {
                params: vec![Type::String],  // plugin name
                return_type: Box::new(Type::Bool),
            }),
            ("unload_plugin", Type::Function {
                params: vec![Type::String],  // plugin name
                return_type: Box::new(Type::Bool),
            }),
            // Serialization & Hashing
            ("bincode_serialize", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::String),  // base64 encoded
            }),
            ("bincode_deserialize", Type::Function {
                params: vec![Type::String],  // base64 encoded
                return_type: Box::new(Type::Any),
            }),
            ("hash", Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::String),  // SHA-256 hash
            }),
            // Higher-order functions with proper type signatures
            // map: ( (T -> U), List<T> ) -> List<U>
            ("map", Type::Function {
                params: vec![
                    Type::Function {
                        params: vec![Type::Generic(Arc::new("T".to_string()))],
                        return_type: Box::new(Type::Generic(Arc::new("U".to_string()))),
                    },
                    Type::List(Box::new(Type::Generic(Arc::new("T".to_string())))),
                ],
                return_type: Box::new(Type::List(Box::new(Type::Generic(Arc::new("U".to_string()))))),
            }),
            // filter: ( (T -> bool), List<T> ) -> List<T>
            ("filter", Type::Function {
                params: vec![
                    Type::Function {
                        params: vec![Type::Generic(Arc::new("T".to_string()))],
                        return_type: Box::new(Type::Bool),
                    },
                    Type::List(Box::new(Type::Generic(Arc::new("T".to_string())))),
                ],
                return_type: Box::new(Type::List(Box::new(Type::Generic(Arc::new("T".to_string()))))),
            }),
            // reduce: ( (A, B -> A), List<B>, A ) -> A
            ("reduce", Type::Function {
                params: vec![
                    Type::Function {
                        params: vec![
                            Type::Generic(Arc::new("A".to_string())),
                            Type::Generic(Arc::new("B".to_string())),
                        ],
                        return_type: Box::new(Type::Generic(Arc::new("A".to_string()))),
                    },
                    Type::List(Box::new(Type::Generic(Arc::new("B".to_string())))),
                    Type::Generic(Arc::new("A".to_string())),
                ],
                return_type: Box::new(Type::Generic(Arc::new("A".to_string()))),
            }),
            // forEach: (fn, list) -> None
            ("forEach", Type::Function {
                params: vec![Type::Any, Type::Any],
                return_type: Box::new(Type::None),
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
            Statement::Let { name, type_annotation, value, span } => {
                let inferred_type = self.check_expression(value)?;

                // Validate against annotation if provided
                if let Some(annotated) = type_annotation {
                    if !self.is_assignable(&inferred_type, annotated) {
                        return Err(self.type_error_with_context(
                            format!("Type mismatch: expected {:?}, got {:?}", annotated, inferred_type),
                            Some(*span)
                        ));
                    }
                }

                self.env.define(Arc::clone(name), inferred_type.clone());
                Ok(Type::None)
            }

            Statement::Assign { target, value, span } => {
                let value_type = self.check_expression(value)?;

                match target {
                    Expression::Ident(name, _) => {
                        // Simple variable assignment
                        let existing_type = self.env.lookup(name)
                            .ok_or_else(|| TBError::undefined_variable(name.to_string()))?
                            .clone();

                        // Check type compatibility
                        if !self.is_assignable(&value_type, &existing_type) {
                            return Err(self.type_error_with_context(
                                format!("Cannot assign {:?} to variable of type {:?}", value_type, existing_type),
                                Some(*span)
                            ));
                        }

                        Ok(Type::None)
                    }
                    Expression::Member { object, member, .. } => {
                        // Object property assignment: obj.member = value
                        let obj_type = self.check_expression(object)?;

                        match obj_type {
                            Type::Dict(_, val_type) => {
                                // Check if value type is compatible with dict value type
                                if !self.is_assignable(&value_type, &val_type) {
                                    return Err(self.type_error_with_context(
                                        format!("Cannot assign {:?} to dict field of type {:?}", value_type, val_type),
                                        Some(*span)
                                    ));
                                }
                                Ok(Type::None)
                            }
                            _ => Err(self.type_error_with_context(
                                format!("Cannot assign property to non-dict type: {:?}", obj_type),
                                Some(*span)
                            ))
                        }
                    }
                    Expression::Index { object, index, .. } => {
                        // Array/dict index assignment
                        let obj_type = self.check_expression(object)?;
                        let idx_type = self.check_expression(index)?;

                        match obj_type {
                            Type::List(elem_type) => {
                                // Check index is int
                                if idx_type != Type::Int && !matches!(idx_type, Type::Any) {
                                    return Err(self.type_error_with_context(
                                        "List index must be int".to_string(),
                                        Some(*span)
                                    ));
                                }
                                // Check value type matches element type
                                if !self.is_assignable(&value_type, &elem_type) {
                                    return Err(self.type_error_with_context(
                                        format!("Cannot assign {:?} to list of {:?}", value_type, elem_type),
                                        Some(*span)
                                    ));
                                }
                                Ok(Type::None)
                            }
                            Type::Dict(key_type, val_type) => {
                                // Check key type
                                if !self.is_assignable(&idx_type, &key_type) {
                                    return Err(self.type_error_with_context(
                                        format!("Dict key type mismatch: expected {:?}, got {:?}", key_type, idx_type),
                                        Some(*span)
                                    ));
                                }
                                // Check value type
                                if !self.is_assignable(&value_type, &val_type) {
                                    return Err(self.type_error_with_context(
                                        format!("Cannot assign {:?} to dict of {:?}", value_type, val_type),
                                        Some(*span)
                                    ));
                                }
                                Ok(Type::None)
                            }
                            _ => Err(self.type_error_with_context(
                                format!("Cannot index into type: {:?}", obj_type),
                                Some(*span)
                            ))
                        }
                    }
                    _ => Err(self.type_error_with_context(
                        "Invalid assignment target".to_string(),
                        Some(*span)
                    ))
                }
            }

            Statement::Function { name, params, return_type, body, span } => {
                // ✅ FIX 7: Infer parameter types from usage in function body
                // This handles recursive functions where params are Generic("n") etc.
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| {
                        if let Some(ty) = &p.type_annotation {
                            ty.clone()
                        } else {
                            // ✅ FIX 7: Infer from usage in function body
                            self.infer_param_type_from_body(&p.name, body)
                                .unwrap_or(Type::Generic(Arc::clone(&p.name)))
                        }
                    })
                    .collect();

                // ✅ FIX 7: Register function in environment FIRST (before type inference)
                // This is critical for recursive functions to be able to call themselves
                // Use Int as placeholder return type for recursive functions
                let placeholder_func_type = Type::Function {
                    params: param_types.clone(),
                    return_type: Box::new(return_type.clone().unwrap_or(Type::Int)),
                };
                self.env.define(Arc::clone(name), placeholder_func_type);

                // ✅ FIX: Infer return type from function body if not explicitly specified
                // First, we need to check the body to infer the return type
                // But we do this WITHOUT modifying self.env to avoid side effects

                // Create a temporary checker with a child environment for type inference
                let mut temp_env = self.env.child();

                // ✅ FIX 7: Register function in temp environment for recursive calls
                temp_env.define(Arc::clone(name), Type::Function {
                    params: param_types.clone(),
                    return_type: Box::new(return_type.clone().unwrap_or(Type::Int)),
                });

                for (param, ty) in params.iter().zip(param_types.iter()) {
                    temp_env.define(Arc::clone(&param.name), ty.clone());
                }

                // Create a temporary type checker to infer the return type
                let mut temp_checker = TypeChecker {
                    env: temp_env,
                    errors: Vec::new(),
                    source_context: self.source_context.clone(),
                };

                let mut inferred_return_type = Type::None;
                for stmt in body {
                    if let Ok(stmt_type) = temp_checker.check_statement(stmt) {
                        if matches!(stmt, Statement::Return { .. }) {
                            inferred_return_type = stmt_type;
                            break;
                        }
                    }
                }

                // Use explicit return type if provided, otherwise use inferred type
                let final_return_type = return_type.clone().unwrap_or(inferred_return_type);

                let func_type = Type::Function {
                    params: param_types.clone(),
                    return_type: Box::new(final_return_type),
                };

                // Update function in environment with final type
                self.env.define(Arc::clone(name), func_type);

                // Now check the function body in a proper child scope for validation
                let mut body_env = self.env.child();
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    body_env.define(Arc::clone(&param.name), ty.clone());
                }

                let old_env = std::mem::replace(&mut self.env, body_env);

                let mut body_type = Type::None;
                let mut has_return = false;
                for stmt in body {
                    let stmt_type = self.check_statement(stmt)?;
                    if matches!(stmt, Statement::Return { .. }) {
                        body_type = stmt_type;
                        has_return = true;
                    } else if !has_return {
                        body_type = stmt_type;
                    }
                }

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
                        return Err(self.type_error_with_context(
                            format!("Function return type mismatch: expected {:?}, got {:?}", expected_return, body_type),
                            Some(*span)
                        ));
                    }
                }

                Ok(Type::None)
            }

            Statement::If { condition, then_block, else_block, span } => {
                let _cond_type = self.check_expression(condition)?;
                // Accept any type for conditions - the runtime will use is_truthy()
                // This matches Python-like truthiness rules:
                // - None, 0, 0.0, "", [], {} are falsy
                // - Everything else is truthy
                // No type checking needed here since all types have defined truthiness

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

            Statement::For { variable, iterable, body, span } => {
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
                    _ => return Err(self.type_error_with_context(
                        format!("Cannot iterate over {:?}", iter_type),
                        Some(*span)
                    )),
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

            Statement::While { condition, body, span } => {
                let _cond_type = self.check_expression(condition)?;
                // Accept any type for conditions - the runtime will use is_truthy()
                // This matches Python-like truthiness rules
                // No type checking needed here since all types have defined truthiness

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

            Statement::Match { value, arms, span } => {
                let value_type = self.check_expression(value)?;

                let mut result_type: Option<Type> = None;

                for arm in arms {
                    // Check pattern matches value type
                    self.check_pattern(&arm.pattern, &value_type, *span)?;

                    // ✅ FIX: Bind pattern variables in a new scope
                    // Save the old environment
                    let old_env = self.env.clone();

                    // Bind pattern variable if it's an Ident
                    if let Pattern::Ident(var_name) = &arm.pattern {
                        self.env.define(Arc::clone(var_name), value_type.clone());
                    }

                    // Check arm body (with pattern variable in scope)
                    let arm_type = self.check_expression(&arm.body)?;

                    // Restore the old environment (remove pattern variable)
                    self.env = old_env;

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
                    .ok_or_else(|| {
                        #[cfg(debug_assertions)]
                        {
                            // Debug output: Show available types in current scope
                            eprintln!("\n[TB TYPE CHECK] Undefined variable '{}' during type checking", name);
                            eprintln!("[TB TYPE CHECK] Available variables in type environment:");
                            let bindings = self.env.bindings();
                            let mut var_names: Vec<_> = bindings.keys().map(|k| k.as_ref()).collect();
                            var_names.sort();
                            for var_name in var_names.iter().take(20) {
                                if let Some(ty) = bindings.get(&Arc::new(var_name.to_string())) {
                                    eprintln!("  - {}: {:?}", var_name, ty);
                                }
                            }
                            if var_names.len() > 20 {
                                eprintln!("  ... and {} more", var_names.len() - 20);
                            }
                        }
                        TBError::undefined_variable(name.to_string())
                    })
            }

            Expression::Binary { op, left, right, .. } => {
                use tb_core::BinaryOp;

                let left_type = self.check_expression(left)?;

                // ✅ FIX (Priority 2): Short-circuit evaluation for AND/OR
                // For AND/OR operators, we don't check the right operand's type
                // because it might not be evaluated at runtime due to short-circuiting.
                // This allows expressions like: false && undefined_function()
                match op {
                    BinaryOp::And | BinaryOp::Or => {
                        // For short-circuit operators, only check left operand
                        // Right operand might not be evaluated, so we don't type-check it
                        return Ok(Type::Bool);
                    }
                    _ => {
                        // For all other operators, check both operands
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

            Expression::Call { callee, args, span } => {
                let func_type = self.check_expression(callee)?;

                match func_type {
                    Type::Function { params, return_type } => {
                        // If params contains Type::Any, allow variadic arguments
                        let is_variadic = params.iter().any(|p| matches!(p, Type::Any));

                        if !is_variadic && args.len() != params.len() {
                            return Err(self.type_error_with_context(
                                format!("Expected {} arguments, got {}", params.len(), args.len()),
                                Some(*span)
                            ));
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
                                    return Err(self.type_error_with_context(
                                        format!("Argument type mismatch: expected {:?}, got {:?}", param_type, arg_type),
                                        Some(*span)
                                    ));
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
                    _ => Err(self.type_error_with_context(
                        format!("Cannot call non-function type {:?}", func_type),
                        Some(*span)
                    )),
                }
            }

            Expression::List { elements, span } => {
                if elements.is_empty() {
                    return Ok(Type::List(Box::new(Type::Generic(Arc::new("T".to_string())))));
                }

                let first_type = self.check_expression(&elements[0])?;

                // Ensure all elements have compatible types
                for elem in &elements[1..] {
                    let elem_type = self.check_expression(elem)?;
                    // Use is_assignable instead of types_compatible for better Dict(String, Any) handling
                    if !self.is_assignable(&elem_type, &first_type) && !self.is_assignable(&first_type, &elem_type) {
                        return Err(self.type_error_with_context(
                            format!("List elements must have compatible types, got {:?} and {:?}", first_type, elem_type),
                            Some(*span)
                        ));
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

                // Collect all value types
                let mut value_types = Vec::new();
                for (_, value) in entries {
                    let value_type = self.check_expression(value)?;
                    value_types.push(value_type);
                }

                // Find the Least Upper Bound (LUB) of all value types
                // This allows Int+Float -> Float instead of falling back to Any
                let lub_type = TypeInference::least_upper_bound(&value_types);

                Ok(Type::Dict(
                    Box::new(Type::String),
                    Box::new(lub_type),
                ))
            }

            Expression::Index { object, index, span } => {
                let obj_type = self.check_expression(object)?;
                let idx_type = self.check_expression(index)?;

                match obj_type {
                    Type::List(elem_type) => {
                        if idx_type != Type::Int && !matches!(idx_type, Type::Any) && !matches!(idx_type, Type::Generic(_)) {
                            return Err(self.type_error_with_context(
                                "List index must be int".to_string(),
                                Some(*span)
                            ));
                        }
                        Ok(*elem_type)
                    }
                    Type::Dict(key_type, value_type) => {
                        if !self.is_assignable(&idx_type, &key_type) && !matches!(idx_type, Type::Any) {
                            return Err(self.type_error_with_context(
                                format!("Dict key type mismatch: expected {:?}, got {:?}", key_type, idx_type),
                                Some(*span)
                            ));
                        }
                        Ok(*value_type)
                    }
                    // Allow indexing on Type::Any - runtime will check
                    Type::Any => Ok(Type::Any),
                    // Allow indexing on ALL Generic types - runtime will check
                    // This is important for function parameters with generic types
                    Type::Generic(_) => Ok(Type::Any),
                    ref other => Err(self.type_error_with_context(
                        format!("Cannot index type {:?}", other),
                        Some(*span)
                    )),
                }
            }

            Expression::Match { value, arms, span } => {
                let value_type = self.check_expression(value)?;

                if arms.is_empty() {
                    return Err(self.type_error_with_context(
                        "Match expression must have at least one arm".to_string(),
                        Some(*span)
                    ));
                }

                // Check all patterns are compatible with value type and collect arm types
                let mut arm_types = Vec::new();

                for arm in arms {
                    self.check_pattern(&arm.pattern, &value_type, *span)?;

                    // ✅ FIX: Bind pattern variables in a new scope
                    // Save the old environment
                    let old_env = self.env.clone();

                    // Bind pattern variable if it's an Ident
                    if let Pattern::Ident(var_name) = &arm.pattern {
                        self.env.define(Arc::clone(var_name), value_type.clone());
                    }

                    // Check arm body (with pattern variable in scope)
                    let arm_type = self.check_expression(&arm.body)?;
                    arm_types.push(arm_type);

                    // Restore the old environment (remove pattern variable)
                    self.env = old_env;
                }

                // All arms should return the same type
                let first_arm_type = &arm_types[0];
                for arm_type in &arm_types[1..] {
                    if !TypeInference::types_compatible(first_arm_type, arm_type) {
                        return Err(self.type_error_with_context(
                            format!("Match arms have incompatible types: {:?} vs {:?}", first_arm_type, arm_type),
                            Some(*span)
                        ));
                    }
                }

                Ok(first_arm_type.clone())
            }

            Expression::Member { object, .. } => {
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

            Expression::Lambda { params, body, .. } => {
                // Create a child environment for lambda parameters
                let old_env = self.env.clone();
                self.env = self.env.child();

                // ✅ FIX: Give each parameter a unique generic type (T0, T1, T2, ...)
                // This allows type inference to work correctly for each parameter
                let mut param_types = Vec::new();
                for (i, param) in params.iter().enumerate() {
                    let param_type = param.type_annotation.clone()
                        .unwrap_or_else(|| Type::Generic(Arc::new(format!("T{}", i))));
                    self.env.define(Arc::clone(&param.name), param_type.clone());
                    param_types.push(param_type);
                }

                // Check the body
                let return_type = self.check_expression(body)?;

                // Restore the old environment
                self.env = old_env;

                // Return a function type with unique generic parameters
                Ok(Type::Function {
                    params: param_types,
                    return_type: Box::new(return_type),
                })
            }

            Expression::Block { statements, .. } => {
                // Create a child environment for block scope
                let old_env = self.env.clone();
                self.env = self.env.child();

                // Check all statements and get the type of the last one
                let mut last_type = Type::None;
                for stmt in statements {
                    match stmt {
                        Statement::Expression { expr, .. } => {
                            last_type = self.check_expression(expr)?;
                        }
                        Statement::Return { value: Some(expr), .. } => {
                            last_type = self.check_expression(expr)?;
                        }
                        _ => {
                            self.check_statement(stmt)?;
                            last_type = Type::None;
                        }
                    }
                }

                // Restore the old environment
                self.env = old_env;

                Ok(last_type)
            }

            _ => Ok(Type::Generic(Arc::new("Unknown".to_string()))),
        }
    }

    fn check_pattern(&self, pattern: &Pattern, value_type: &Type, span: Span) -> Result<()> {
        match pattern {
            Pattern::Literal(lit) => {
                let pattern_type = TypeInference::infer_literal(lit);
                if !TypeInference::types_compatible(&pattern_type, value_type) {
                    return Err(self.type_error_with_context(
                        format!("Pattern type {:?} doesn't match value type {:?}", pattern_type, value_type),
                        Some(span)
                    ));
                }
                Ok(())
            }
            Pattern::Range { .. } => {
                // Range patterns only work with integers
                if !matches!(value_type, Type::Int) && !matches!(value_type, Type::Any) {
                    return Err(self.type_error_with_context(
                        format!("Range pattern requires Int type, got {:?}", value_type),
                        Some(span)
                    ));
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

        // Generic types are always assignable
        if matches!(to, Type::Generic(_)) || matches!(from, Type::Generic(_)) {
            return true;
        }

        // Dict compatibility with Any values
        match (from, to) {
            (Type::Dict(k1, v1), Type::Dict(k2, v2)) => {
                let keys_match = k1 == k2 || matches!(k1.as_ref(), Type::Any) || matches!(k2.as_ref(), Type::Any) || matches!(k1.as_ref(), Type::Generic(_)) || matches!(k2.as_ref(), Type::Generic(_));
                let values_match = v1 == v2 || matches!(v1.as_ref(), Type::Any) || matches!(v2.as_ref(), Type::Any) || matches!(v1.as_ref(), Type::Generic(_)) || matches!(v2.as_ref(), Type::Generic(_));
                return keys_match && values_match;
            }
            (Type::List(t1), Type::List(t2)) => {
                return t1 == t2 || matches!(t1.as_ref(), Type::Any) || matches!(t2.as_ref(), Type::Any) || matches!(t1.as_ref(), Type::Generic(_)) || matches!(t2.as_ref(), Type::Generic(_));
            }
            // Function type compatibility with generics
            (Type::Function { params: p1, return_type: r1 }, Type::Function { params: p2, return_type: r2 }) => {
                // Check if parameter counts match
                if p1.len() != p2.len() {
                    return false;
                }

                // Check if all parameters are assignable (with generic support)
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    if !self.is_assignable(param1, param2) {
                        return false;
                    }
                }

                // Check if return types are assignable
                return self.is_assignable(r1, r2);
            }
            _ => {}
        }

        // Check type compatibility
        TypeInference::types_compatible(from, to)
    }

    pub fn environment(&self) -> &TypeEnvironment {
        &self.env
    }


    /// ✅ FIX 7: Infer parameter type from usage in function body
    /// This handles recursive functions where params need type inference
    fn infer_param_type_from_body(&self, param_name: &Arc<String>, body: &[Statement]) -> Option<Type> {
        for stmt in body {
            if let Some(ty) = self.infer_param_type_from_stmt(param_name, stmt) {
                return Some(ty);
            }
        }
        None
    }

    /// Helper to infer parameter type from statement
    fn infer_param_type_from_stmt(&self, param_name: &Arc<String>, stmt: &Statement) -> Option<Type> {
        match stmt {
            Statement::Expression { expr, .. } => self.infer_param_type_from_expr(param_name, expr),
            Statement::Return { value: Some(expr), .. } => self.infer_param_type_from_expr(param_name, expr),
            Statement::If { condition, then_block, else_block, .. } => {
                self.infer_param_type_from_expr(param_name, condition)
                    .or_else(|| {
                        for stmt in then_block {
                            if let Some(ty) = self.infer_param_type_from_stmt(param_name, stmt) {
                                return Some(ty);
                            }
                        }
                        None
                    })
                    .or_else(|| {
                        if let Some(else_stmts) = else_block {
                            for stmt in else_stmts {
                                if let Some(ty) = self.infer_param_type_from_stmt(param_name, stmt) {
                                    return Some(ty);
                                }
                            }
                        }
                        None
                    })
            }
            _ => None,
        }
    }

    /// Helper to infer parameter type from expression
    fn infer_param_type_from_expr(&self, param_name: &Arc<String>, expr: &Expression) -> Option<Type> {
        match expr {
            // ✅ FIX 10: Index access suggests List type
            Expression::Index { object, index, .. } => {
                // If parameter is being indexed, it's a List
                if let Expression::Ident(name, _) = object.as_ref() {
                    if name == param_name {
                        // lst[idx] means lst is a List
                        // We can't infer element type without more context, default to Int
                        return Some(Type::List(Box::new(Type::Int)));
                    }
                }
                // Recursively check object and index
                self.infer_param_type_from_expr(param_name, object)
                    .or_else(|| self.infer_param_type_from_expr(param_name, index))
            }

            // Binary operations give us type hints
            Expression::Binary { op, left, right, .. } => {
                use tb_core::BinaryOp;

                // Check if parameter is used in this binary operation
                let left_is_param = matches!(left.as_ref(), Expression::Ident(name, _) if name == param_name);
                let right_is_param = matches!(right.as_ref(), Expression::Ident(name, _) if name == param_name);

                if left_is_param || right_is_param {
                    // Infer type from operation
                    match op {
                        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                            return Some(Type::Int);
                        }
                        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
                            return Some(Type::Int);
                        }
                        BinaryOp::Eq | BinaryOp::NotEq => {
                            // Could be any type, check the other operand
                            if left_is_param {
                                if let Expression::Literal(lit, _) = right.as_ref() {
                                    return Some(Self::type_from_literal(lit));
                                }
                            } else if right_is_param {
                                if let Expression::Literal(lit, _) = left.as_ref() {
                                    return Some(Self::type_from_literal(lit));
                                }
                            }
                        }
                        BinaryOp::And | BinaryOp::Or => {
                            return Some(Type::Bool);
                        }
                        _ => {}
                    }
                }

                // Recursively check both sides
                self.infer_param_type_from_expr(param_name, left)
                    .or_else(|| self.infer_param_type_from_expr(param_name, right))
            }

            // ✅ FIX 10: Function calls - check for len(), which suggests List type
            Expression::Call { callee, args, .. } => {
                // Check if the parameter itself is being called as a function
                if let Expression::Ident(func_name, _) = callee.as_ref() {
                    if func_name == param_name {
                        // param(x) means param is a Function
                        // Use Any for parameter types since we can't infer them without full type checking
                        let param_types: Vec<Type> = args.iter()
                            .map(|_| Type::Any)
                            .collect();
                        return Some(Type::Function {
                            params: param_types,
                            return_type: Box::new(Type::Any),
                        });
                    }

                    // Check if this is len(param) - suggests param is a List
                    if func_name.as_str() == "len" && args.len() == 1 {
                        if let Expression::Ident(arg_name, _) = &args[0] {
                            if arg_name == param_name {
                                // len(lst) means lst is a List
                                return Some(Type::List(Box::new(Type::Int)));
                            }
                        }
                    }
                }

                // Recursively check arguments
                for arg in args {
                    if let Some(ty) = self.infer_param_type_from_expr(param_name, arg) {
                        return Some(ty);
                    }
                }
                None
            }

            // ✅ FIX 10: Match expression - infer from value
            Expression::Match { value, .. } => {
                // If parameter is the match value, check pattern types
                if let Expression::Ident(name, _) = value.as_ref() {
                    if name == param_name {
                        // Match on parameter suggests it's an Int (for range patterns)
                        return Some(Type::Int);
                    }
                }
                self.infer_param_type_from_expr(param_name, value)
            }

            _ => None,
        }
    }

    /// Helper to get type from literal
    fn type_from_literal(lit: &Literal) -> Type {
        match lit {
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::String(_) => Type::String,
            Literal::Bool(_) => Type::Bool,
            Literal::None => Type::None,
        }
    }

    /// Create a type error with source context and span
    fn type_error_with_context(&self, message: String, span: Option<Span>) -> TBError {
        if let Some(ctx) = &self.source_context {
            TBError::TypeError {
                message,
                span,
                source_context: Some(ctx.clone()),
            }
        } else {
            TBError::type_error(message)
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

