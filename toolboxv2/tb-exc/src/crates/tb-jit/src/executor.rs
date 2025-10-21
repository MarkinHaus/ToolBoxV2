use crate::builtins;
use im::HashMap as ImHashMap;
use std::sync::Arc;
use std::path::PathBuf;
use tb_core::*;
use tb_plugin::PluginLoader;

/// JIT executor using tree-walking interpreter with persistent data structures
pub struct JitExecutor {
    env: ImHashMap<Arc<String>, Value>,
    return_value: Option<Value>,
    should_break: bool,
    should_continue: bool,
    plugin_loader: Arc<PluginLoader>,
    source_context: Option<SourceContext>,
}

impl JitExecutor {
    pub fn new() -> Self {
        let mut env = ImHashMap::new();
        builtins::register_builtins(&mut env);

        Self {
            env,
            return_value: None,
            should_break: false,
            should_continue: false,
            plugin_loader: Arc::new(PluginLoader::new()),
            source_context: None,
        }
    }

    /// Create a new JitExecutor with source context for better error messages
    pub fn new_with_source(source: String, file_path: Option<PathBuf>) -> Self {
        let mut executor = Self::new();
        executor.source_context = Some(SourceContext::new(source, file_path));
        executor
    }

    pub fn execute(&mut self, program: &Program) -> Result<Value> {
        tb_debug_jit!("Starting JIT execution with {} statements", program.statements.len());
        let mut last = Value::None;

        for (i, stmt) in program.statements.iter().enumerate() {
            // Get source code snippet for this statement
            let debug_info = self.get_statement_debug_info(stmt);
            tb_debug_jit!("Statement {}: {} | {}", i + 1, debug_info.location, debug_info.source_line);

            last = self.eval_statement(stmt)?;

            if self.return_value.is_some() {
                tb_debug_jit!("Early return with value: {:?}", self.return_value);
                return Ok(self.return_value.take().unwrap());
            }
        }

        tb_debug_jit!("Execution completed successfully");
        Ok(last)
    }

    fn eval_statement(&mut self, stmt: &Statement) -> Result<Value> {
        match stmt {
            Statement::Let { name, value, .. } => {
                let val = self.eval_expression(value)?;
                self.env.insert(Arc::clone(name), val);
                Ok(Value::None)
            }

            Statement::Assign { name, value, span } => {
                if !self.env.contains_key(name) {
                    return Err(self.undefined_variable_with_context(name.to_string(), Some(*span)));
                }

                let val = self.eval_expression(value)?;
                self.env.insert(Arc::clone(name), val);
                Ok(Value::None)
            }

            Statement::Function { name, params, body, return_type, .. } => {
                let func = Value::Function(Arc::new(Function {
                    name: Arc::clone(name),
                    params: params.iter().map(|p| Arc::clone(&p.name)).collect(),
                    body: body.clone(),
                    return_type: return_type.clone(),
                }));

                self.env.insert(Arc::clone(name), func);
                Ok(Value::None)
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

            Statement::For { variable, iterable, body, span } => {
                let iter_value = self.eval_expression(iterable)?;

                match iter_value {
                    Value::List(items) => {
                        for item in items.iter() {
                            self.env.insert(Arc::clone(variable), item.clone());

                            self.eval_block(body)?;

                            if self.should_break {
                                self.should_break = false;
                                break;
                            }

                            if self.should_continue {
                                self.should_continue = false;
                                continue;
                            }

                            if self.return_value.is_some() {
                                break;
                            }
                        }
                        Ok(Value::None)
                    }
                    _ => Err(self.runtime_error_with_context(
                        format!("Cannot iterate over {}", iter_value.type_name()),
                        Some(*span)
                    )),
                }
            }

            Statement::While { condition, body, .. } => {
                loop {
                    let cond = self.eval_expression(condition)?;
                    if !cond.is_truthy() {
                        break;
                    }

                    self.eval_block(body)?;

                    if self.should_break {
                        self.should_break = false;
                        break;
                    }

                    if self.should_continue {
                        self.should_continue = false;
                        continue;
                    }

                    if self.return_value.is_some() {
                        break;
                    }
                }
                Ok(Value::None)
            }

            Statement::Match { value, arms, span } => {
                let match_value = self.eval_expression(value)?;

                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &match_value)? {
                        return self.eval_expression(&arm.body);
                    }
                }

                Err(self.runtime_error_with_context(
                    "No matching pattern in match expression".to_string(),
                    Some(*span)
                ))
            }

            Statement::Return { value, .. } => {
                let return_val = if let Some(expr) = value {
                    self.eval_expression(expr)?
                } else {
                    Value::None
                };

                self.return_value = Some(return_val);
                Ok(Value::None)
            }

            Statement::Break { .. } => {
                self.should_break = true;
                Ok(Value::None)
            }

            Statement::Continue { .. } => {
                self.should_continue = true;
                Ok(Value::None)
            }

            Statement::Expression { expr, .. } => self.eval_expression(expr),

            Statement::Plugin { definitions, .. } => {
                // For each plugin definition, create a module (Dict) with functions
                for def in definitions {
                    let module_name = Arc::clone(&def.name);
                    let language = def.language.clone();
                    let mode = def.mode.clone();

                    // Create module dict with plugin functions
                    let module = match &def.source {
                        PluginSource::Inline(code) => {
                            self.create_plugin_module_inline(&language, &mode, code)?
                        }
                        PluginSource::File(path) => {
                            self.create_plugin_module_file(&language, &mode, path)?
                        }
                    };

                    self.env.insert(module_name, module);
                }
                Ok(Value::None)
            }

            Statement::Import { .. } => {
                // Imports are handled at the CLI level before execution
                // This is just a no-op placeholder
                Ok(Value::None)
            }

            Statement::Config { .. } => {
                // Config blocks don't affect runtime
                Ok(Value::None)
            }
        }
    }

    fn eval_block(&mut self, statements: &[Statement]) -> Result<Value> {
        let mut last = Value::None;

        for stmt in statements {
            last = self.eval_statement(stmt)?;

            if self.should_break || self.should_continue || self.return_value.is_some() {
                break;
            }
        }

        Ok(last)
    }

    fn eval_expression(&mut self, expr: &Expression) -> Result<Value> {
        match expr {
            Expression::Literal(lit, _) => Ok(self.eval_literal(lit)),

            Expression::Ident(name, span) => {
                self.env.get(name)
                    .cloned()
                    .ok_or_else(|| self.undefined_variable_with_context(name.to_string(), Some(*span)))
            }

            Expression::Binary { op, left, right, span } => {
                let left_val = self.eval_expression(left)?;
                let right_val = self.eval_expression(right)?;
                self.eval_binary_op(*op, left_val, right_val, *span)
            }

            Expression::Unary { op, operand, .. } => {
                let val = self.eval_expression(operand)?;
                self.eval_unary_op(*op, val)
            }

            Expression::Call { callee, args, span } => {
                let func = self.eval_expression(callee)?;
                let arg_values: Result<Vec<_>> = args.iter()
                    .map(|arg| self.eval_expression(arg))
                    .collect();
                let arg_values = arg_values?;

                self.call_function(func, arg_values, *span)
            }

            Expression::List { elements, .. } => {
                let values: Result<Vec<_>> = elements.iter()
                    .map(|elem| self.eval_expression(elem))
                    .collect();
                Ok(Value::List(Arc::new(values?)))
            }

            Expression::Dict { entries, .. } => {
                let mut map = ImHashMap::new();
                for (key, value_expr) in entries {
                    let value = self.eval_expression(value_expr)?;
                    map.insert(Arc::clone(key), value);
                }
                Ok(Value::Dict(Arc::new(map)))
            }

            Expression::Index { object, index, span } => {
                let obj = self.eval_expression(object)?;
                let idx = self.eval_expression(index)?;
                self.eval_index(obj, idx, *span)
            }

            Expression::Member { object, member, span } => {
                let obj = self.eval_expression(object)?;
                match obj {
                    Value::Dict(map) => {
                        map.get(member)
                            .cloned()
                            .ok_or_else(|| self.runtime_error_with_context(
                                format!("Key '{}' not found", member),
                                Some(*span)
                            ))
                    }
                    _ => Err(self.runtime_error_with_context(
                        format!("Cannot access member on {}", obj.type_name()),
                        Some(*span)
                    )),
                }
            }

            Expression::Match { value, arms, .. } => {
                let val = self.eval_expression(value)?;

                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val)? {
                        return self.eval_expression(&arm.body);
                    }
                }

                Err(TBError::invalid_operation("No matching pattern in match expression".to_string()))
            }

            Expression::Lambda { params, body, span } => {
                // Create anonymous function
                let param_names: Vec<Arc<String>> = params.iter()
                    .map(|p| Arc::clone(&p.name))
                    .collect();

                // Convert expression body to statement block
                let body_stmt = Statement::Expression {
                    expr: (**body).clone(),
                    span: *span,
                };

                Ok(Value::Function(Arc::new(Function {
                    name: Arc::new("<lambda>".to_string()),
                    params: param_names,
                    body: vec![body_stmt],
                    return_type: None,
                })))
            }

            _ => Ok(Value::None),
        }
    }

    fn pattern_matches(&self, pattern: &Pattern, value: &Value) -> Result<bool> {
        match pattern {
            Pattern::Literal(lit) => {
                let pattern_val = self.eval_literal(lit);
                Ok(self.values_equal(&pattern_val, value))
            }
            Pattern::Wildcard => Ok(true),
            Pattern::Ident(_) => Ok(true), // Binds to any value
            Pattern::Range { start, end, inclusive } => {
                if let Value::Int(val) = value {
                    if *inclusive {
                        Ok(*val >= *start && *val <= *end)
                    } else {
                        Ok(*val >= *start && *val < *end)
                    }
                } else {
                    Ok(false)
                }
            }
        }
    }

    fn eval_literal(&self, lit: &Literal) -> Value {
        match lit {
            Literal::None => Value::None,
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Int(i) => Value::Int(*i),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(Arc::clone(s)),
        }
    }



    fn eval_binary_op(&self, op: BinaryOp, left: Value, right: Value, span: Span) -> Result<Value> {
        match op {
            BinaryOp::Add => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                (Value::String(a), Value::String(b)) => {
                    Ok(Value::String(Arc::new(format!("{}{}", a, b))))
                }
                _ => Err(TBError::invalid_operation("Invalid addition".to_string())),
            },

            BinaryOp::Sub => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
                _ => Err(TBError::invalid_operation("Invalid subtraction".to_string())),
            },

            BinaryOp::Mul => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
                _ => Err(TBError::invalid_operation("Invalid multiplication".to_string())),
            },

            BinaryOp::Div => match (left, right) {
                (Value::Int(a), Value::Int(b)) if b != 0 => Ok(Value::Int(a / b)),
                (Value::Float(a), Value::Float(b)) if b != 0.0 => Ok(Value::Float(a / b)),
                (Value::Int(a), Value::Float(b)) if b != 0.0 => Ok(Value::Float(a as f64 / b)),
                (Value::Float(a), Value::Int(b)) if b != 0 => Ok(Value::Float(a / b as f64)),
                _ => Err(self.runtime_error_with_context("Division by zero".to_string(), Some(span))),
            },

            BinaryOp::Mod => match (left, right) {
                (Value::Int(a), Value::Int(b)) if b != 0 => Ok(Value::Int(a % b)),
                _ => Err(self.runtime_error_with_context("Invalid modulo operation".to_string(), Some(span))),
            },

            BinaryOp::Eq => Ok(Value::Bool(self.values_equal(&left, &right))),
            BinaryOp::NotEq => Ok(Value::Bool(!self.values_equal(&left, &right))),

            BinaryOp::Lt => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64) < b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a < b as f64)),
                _ => Err(TBError::invalid_operation("Invalid comparison".to_string())),
            },

            BinaryOp::Gt => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((a as f64) > b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a > b as f64)),
                _ => Err(TBError::invalid_operation("Invalid comparison".to_string())),
            },

            BinaryOp::LtEq => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                _ => Err(TBError::invalid_operation("Invalid comparison".to_string())),
            },

            BinaryOp::GtEq => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                _ => Err(TBError::invalid_operation("Invalid comparison".to_string())),
            },

            BinaryOp::And => Ok(Value::Bool(left.is_truthy() && right.is_truthy())),
            BinaryOp::Or => Ok(Value::Bool(left.is_truthy() || right.is_truthy())),

            BinaryOp::In => {
                // Membership test: "x" in list, "key" in dict, "sub" in string
                match (&left, &right) {
                    // String in String (substring check)
                    (Value::String(needle), Value::String(haystack)) => {
                        Ok(Value::Bool(haystack.contains(needle.as_str())))
                    }
                    // Value in List
                    (_, Value::List(list)) => {
                        Ok(Value::Bool(list.iter().any(|v| v == &left)))
                    }
                    // Key in Dict
                    (Value::String(key), Value::Dict(dict)) => {
                        Ok(Value::Bool(dict.contains_key(key)))
                    }
                    _ => Err(TBError::invalid_operation(
                        format!("'in' operator not supported for {} in {}", left.type_name(), right.type_name())
                    )),
                }
            }
        }
    }

    fn eval_unary_op(&self, op: UnaryOp, operand: Value) -> Result<Value> {
        match op {
            UnaryOp::Neg => match operand {
                Value::Int(i) => Ok(Value::Int(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(TBError::invalid_operation("Cannot negate non-numeric value".to_string())),
            },
            UnaryOp::Not => Ok(Value::Bool(!operand.is_truthy())),
        }
    }

    fn eval_index(&self, object: Value, index: Value, span: Span) -> Result<Value> {
        match (object, index) {
            (Value::List(items), Value::Int(i)) => {
                let idx = if i < 0 {
                    (items.len() as i64 + i) as usize
                } else {
                    i as usize
                };

                items.get(idx)
                    .cloned()
                    .ok_or_else(|| self.runtime_error_with_context(
                        format!("Index {} out of bounds", i),
                        Some(span)
                    ))
            }
            (Value::Dict(map), Value::String(key)) => {
                map.get(&key)
                    .cloned()
                    .ok_or_else(|| self.runtime_error_with_context(
                        format!("Key '{}' not found", key),
                        Some(span)
                    ))
            }
            _ => Err(self.runtime_error_with_context("Invalid index operation".to_string(), Some(span))),
        }
    }

    fn call_function(&mut self, func: Value, args: Vec<Value>, span: Span) -> Result<Value> {
        match func {
            Value::Function(f) => {
                // Check argument count
                if args.len() != f.params.len() {
                    return Err(self.runtime_error_with_context(
                        format!("Expected {} arguments, got {}", f.params.len(), args.len()),
                        Some(span)
                    ));
                }

                // Create new environment with O(1) clone
                let mut new_executor = JitExecutor {
                    env: self.env.clone(), // O(1) structural sharing!
                    return_value: None,
                    should_break: false,
                    should_continue: false,
                    plugin_loader: Arc::clone(&self.plugin_loader),
                    source_context: self.source_context.clone(),
                };

                // Bind parameters
                for (param, arg) in f.params.iter().zip(args.iter()) {
                    new_executor.env.insert(Arc::clone(param), arg.clone());
                }

                // Execute function body
                new_executor.eval_block(&f.body)?;

                Ok(new_executor.return_value.unwrap_or(Value::None))
            }

            Value::NativeFunction(f) => {
                (f.func)(args)
            }

            _ => Err(self.runtime_error_with_context(
                format!("Cannot call {}", func.type_name()),
                Some(span)
            )),
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::None, Value::None) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }

    fn create_plugin_module_inline(
        &self,
        language: &PluginLanguage,
        mode: &PluginMode,
        source_code: &str,
    ) -> Result<Value> {
        tb_debug_plugin!("Creating inline plugin module for {:?} in {:?} mode", language, mode);
        tb_debug_plugin!("Source code length: {} bytes", source_code.len());

        // For JIT mode, we need to extract function names from the source
        // and create wrapper functions that call the plugin loader

        let function_names = self.extract_function_names(language, source_code)?;

        tb_debug_plugin!("Extracted {} functions: {:?}", function_names.len(), function_names);

        let mut module_dict = ImHashMap::new();

        for func_name in function_names {
            let func_name_arc = Arc::new(func_name.clone());
            let source_clone = source_code.to_string();
            let language_clone = language.clone();
            let mode_clone = mode.clone();
            let loader = Arc::clone(&self.plugin_loader);

            tb_debug_plugin!("Registering function: {}", func_name);

            // Create a native function that calls the plugin
            let native_func = Value::NativeFunction(Arc::new(NativeFunction {
                name: Arc::clone(&func_name_arc),
                func: Arc::new(move |args: Vec<Value>| {
                    tb_debug_plugin!("Executing inline function: {}", func_name);
                    loader.execute_inline(
                        &language_clone,
                        &mode_clone,
                        &source_clone,
                        &func_name,
                        args,
                    )
                }),
            }));

            module_dict.insert(func_name_arc, native_func);
        }

        tb_debug_plugin!("Created plugin module with {} functions", module_dict.len());

        Ok(Value::Dict(Arc::new(module_dict)))
    }

    fn create_plugin_module_file(
        &self,
        language: &PluginLanguage,
        mode: &PluginMode,
        file_path: &str,
    ) -> Result<Value> {
        use std::path::{Path, PathBuf};

        tb_debug_plugin!("Loading plugin file: {}", file_path);

        // ✅ PASS 23 FIX #1: Try multiple path resolution strategies
        let canonical_path = self.resolve_plugin_file_path(file_path)?;

        tb_debug_plugin!("Resolved path: {}", canonical_path.display());

        // Read the file content
        let source_code = std::fs::read_to_string(&canonical_path)
            .map_err(|e| TBError::plugin_error(format!("Failed to read plugin file '{}': {}", canonical_path.display(), e)))?;

        tb_debug_plugin!("Read {} bytes from plugin file", source_code.len());

        // Extract function names from the file
        let function_names = self.extract_function_names(language, &source_code)?;

        tb_debug_plugin!("Extracted {} functions: {:?}", function_names.len(), function_names);

        let mut module_dict = ImHashMap::new();

        for func_name in function_names {
            let func_name_arc = Arc::new(func_name.clone());
            // FIX: Use canonical path instead of original file_path
            let path_clone = canonical_path.to_string_lossy().to_string();
            let language_clone = language.clone();
            let mode_clone = mode.clone();
            let loader = Arc::clone(&self.plugin_loader);

            // Create a native function that calls the plugin loader with file path
            let native_func = Value::NativeFunction(Arc::new(NativeFunction {
                name: Arc::clone(&func_name_arc),
                func: Arc::new(move |args: Vec<Value>| {
                    loader.load_and_execute(
                        &language_clone,
                        &mode_clone,
                        Path::new(&path_clone),
                        &func_name,
                        args,
                    )
                }),
            }));

            module_dict.insert(func_name_arc, native_func);
        }

        Ok(Value::Dict(Arc::new(module_dict)))
    }

    /// ✅ PASS 23 FIX #1: Helper method to resolve plugin file paths
    /// Tries multiple strategies to find the file
    fn resolve_plugin_file_path(&self, file_path: &str) -> Result<std::path::PathBuf> {
        use std::path::PathBuf;

        // Strategy 1: Try as absolute path or direct path
        let path_buf = PathBuf::from(file_path);
        if path_buf.exists() {
            return Ok(path_buf.canonicalize().unwrap_or(path_buf));
        }

        // Strategy 2: Try relative to current working directory
        if let Ok(cwd) = std::env::current_dir() {
            let cwd_relative = cwd.join(file_path);
            if cwd_relative.exists() {
                return Ok(cwd_relative.canonicalize().unwrap_or(cwd_relative));
            }
        }

        // Strategy 3: Try relative to test directory (for test execution)
        let test_relative = PathBuf::from("toolboxv2/utils/tbx/test").join(file_path);
        if test_relative.exists() {
            return Ok(test_relative.canonicalize().unwrap_or(test_relative));
        }

        // Strategy 4: Try with platform-specific path normalization
        #[cfg(windows)]
        let normalized = file_path.replace("/", "\\");
        #[cfg(not(windows))]
        let normalized = file_path.replace("\\", "/");

        let normalized_path = PathBuf::from(&normalized);
        if normalized_path.exists() {
            return Ok(normalized_path.canonicalize().unwrap_or(normalized_path));
        }

        // All strategies failed - return error with helpful message
        Err(TBError::plugin_error(format!(
                "Plugin file not found: '{}'\nTried:\n  1. Direct path: {}\n  2. CWD relative: {:?}\n  3. Test relative: {:?}\n  4. Normalized: {}",
                file_path,
                path_buf.display(),
                std::env::current_dir().ok().map(|cwd| cwd.join(file_path)),
                test_relative,
                normalized
            )))
    }

    fn extract_function_names(
        &self,
        language: &PluginLanguage,
        source_code: &str,
    ) -> Result<Vec<String>> {
        match language {
            PluginLanguage::Rust => {
                // For Rust, look for "pub extern "C" fn" or just "fn" after #[no_mangle]
                let mut functions = Vec::new();
                let lines: Vec<&str> = source_code.lines().collect();

                for (i, line) in lines.iter().enumerate() {
                    let trimmed = line.trim();

                    // Check if this line or previous line has #[no_mangle]
                    let has_no_mangle = trimmed.contains("#[no_mangle]") ||
                        (i > 0 && lines[i-1].trim().contains("#[no_mangle]"));

                    // Look for function definitions
                    if let Some(fn_pos) = trimmed.find(" fn ") {
                        // Extract function name
                        let after_fn = &trimmed[fn_pos + 4..];
                        if let Some(paren_pos) = after_fn.find('(') {
                            let func_name = after_fn[..paren_pos].trim().to_string();
                            if has_no_mangle || trimmed.contains("pub extern") {
                                tb_debug_plugin!("Found Rust function: {}", func_name);
                                functions.push(func_name);
                            }
                        }
                    }
                }

                Ok(functions)
            }
            _ => {
                // Generic function extraction for other languages
                let prefix = match language {
                    PluginLanguage::Python => "def ",
                    PluginLanguage::JavaScript => "function ",
                    PluginLanguage::Go => "func ",
                    PluginLanguage::Rust => unreachable!(),
                };

                Ok(source_code
                    .lines()
                    .filter_map(|line| {
                        let trimmed = line.trim();
                        trimmed.strip_prefix(prefix).and_then(|name_part| {
                            name_part.find('(').map(|pos| name_part[..pos].trim().to_string())
                        })
                    })
                    .collect())
            }
        }
    }

    /// Create a runtime error with source context and span
    fn runtime_error_with_context(&self, message: String, span: Option<Span>) -> TBError {
        if let Some(ctx) = &self.source_context {
            TBError::RuntimeError {
                message,
                span,
                source_context: Some(ctx.clone()),
                call_stack: vec![],
            }
        } else {
            TBError::runtime_error(message)
        }
    }

    /// Create an undefined variable error with source context and span
    fn undefined_variable_with_context(&self, name: String, span: Option<Span>) -> TBError {
        if let Some(ctx) = &self.source_context {
            TBError::UndefinedVariable {
                name,
                span,
                source_context: Some(ctx.clone()),
            }
        } else {
            TBError::undefined_variable(name)
        }
    }

    /// Get debug information for a statement (source code + location)
    fn get_statement_debug_info(&self, stmt: &Statement) -> StatementDebugInfo {
        let span = self.get_statement_span(stmt);

        if let Some(ctx) = &self.source_context {
            let source_line = ctx.get_line_content(span.line);
            let location = format!("Line {}", span.line);

            StatementDebugInfo {
                location,
                source_line: source_line.trim().to_string(),
                statement_type: self.get_statement_type_name(stmt),
            }
        } else {
            StatementDebugInfo {
                location: format!("Line {}", span.line),
                source_line: self.get_statement_type_name(stmt).to_string(),
                statement_type: self.get_statement_type_name(stmt),
            }
        }
    }

    /// Get the span for a statement
    fn get_statement_span(&self, stmt: &Statement) -> Span {
        match stmt {
            Statement::Let { span, .. } => *span,
            Statement::Assign { span, .. } => *span,
            Statement::Function { span, .. } => *span,
            Statement::If { span, .. } => *span,
            Statement::For { span, .. } => *span,
            Statement::While { span, .. } => *span,
            Statement::Match { span, .. } => *span,
            Statement::Return { span, .. } => *span,
            Statement::Break { span } => *span,
            Statement::Continue { span } => *span,
            Statement::Expression { span, .. } => *span,
            Statement::Import { span, .. } => *span,
            Statement::Config { span, .. } => *span,
            Statement::Plugin { span, .. } => *span,
        }
    }

    /// Get a human-readable name for the statement type
    fn get_statement_type_name(&self, stmt: &Statement) -> &'static str {
        match stmt {
            Statement::Let { .. } => "let",
            Statement::Assign { .. } => "assign",
            Statement::Function { .. } => "fn",
            Statement::If { .. } => "if",
            Statement::For { .. } => "for",
            Statement::While { .. } => "while",
            Statement::Match { .. } => "match",
            Statement::Return { .. } => "return",
            Statement::Break { .. } => "break",
            Statement::Continue { .. } => "continue",
            Statement::Expression { .. } => "expr",
            Statement::Import { .. } => "import",
            Statement::Config { .. } => "config",
            Statement::Plugin { .. } => "plugin",
        }
    }
}

/// Debug information for a statement
struct StatementDebugInfo {
    location: String,
    source_line: String,
    statement_type: &'static str,
}

impl Default for JitExecutor {
    fn default() -> Self {
        Self::new()
    }
}

