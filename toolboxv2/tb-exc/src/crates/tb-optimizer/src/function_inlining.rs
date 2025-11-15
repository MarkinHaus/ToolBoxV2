use std::collections::HashMap;
use std::sync::Arc;
use tb_core::*;

/// Function inlining optimization pass
/// Inlines small functions to reduce call overhead
pub struct FunctionInlining {
    threshold: usize,
    changes: usize,
    functions: HashMap<Arc<String>, (Vec<Parameter>, Vec<Statement>)>,
}

impl FunctionInlining {
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            changes: 0,
            functions: HashMap::new(),
        }
    }

    fn collect_functions(&mut self, statements: &[Statement]) {
        for stmt in statements {
            if let Statement::Function { name, params, body, .. } = stmt {
                // Only collect small functions
                if self.estimate_size(body) <= self.threshold {
                    self.functions.insert(Arc::clone(name), (params.clone(), body.clone()));
                }
            }
        }
    }

    fn estimate_size(&self, statements: &[Statement]) -> usize {
        statements.iter().map(|s| self.statement_size(s)).sum()
    }

    fn statement_size(&self, stmt: &Statement) -> usize {
        match stmt {
            Statement::Let { .. } => 1,
            Statement::Return { .. } => 1,
            Statement::Expression { .. } => 1,
            Statement::Function { body, .. } => self.estimate_size(body) + 1,
            Statement::If { then_block, else_block, .. } => {
                1 + self.estimate_size(then_block)
                    + else_block.as_ref().map_or(0, |b| self.estimate_size(b))
            }
            Statement::For { body, .. } | Statement::While { body, .. } => 1 + self.estimate_size(body),
            _ => 1,
        }
    }

    fn try_inline_call(&mut self, expr: &mut Expression) -> bool {
        match expr {
            Expression::Call { callee, args, span: _ } => {
                if let Expression::Ident(func_name, _) = callee.as_ref() {
                    if let Some((params, body)) = self.functions.get(func_name).cloned() {
                        // Simple inlining: replace with body if single return statement
                        if body.len() == 1 {
                            if let Statement::Return { value: Some(return_expr), .. } = &body[0] {
                                // ✅ FIX: Don't inline functions that return lambdas
                                // Lambdas need to capture their environment at runtime, not compile time
                                if matches!(return_expr, Expression::Lambda { .. }) {
                                    return false;
                                }

                                // Substitute parameters with arguments
                                let mut inlined = return_expr.clone();
                                self.substitute_params(&mut inlined, &params, args);
                                *expr = inlined;
                                self.changes += 1;
                                return true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn substitute_params(&self, expr: &mut Expression, params: &[Parameter], args: &[Expression]) {
        match expr {
            Expression::Ident(name, _) => {
                // Find parameter and replace with argument
                for (param, arg) in params.iter().zip(args.iter()) {
                    if param.name == *name {
                        *expr = arg.clone();
                        return;
                    }
                }
            }
            Expression::Binary { left, right, .. } => {
                self.substitute_params(left, params, args);
                self.substitute_params(right, params, args);
            }
            Expression::Unary { operand, .. } => {
                self.substitute_params(operand, params, args);
            }
            Expression::Call { callee, args: call_args, .. } => {
                self.substitute_params(callee, params, args);
                for arg in call_args {
                    self.substitute_params(arg, params, args);
                }
            }
            Expression::List { elements, .. } => {
                for elem in elements {
                    self.substitute_params(elem, params, args);
                }
            }
            Expression::Block { statements, .. } => {
                for stmt in statements {
                    match stmt {
                        Statement::Expression { expr, .. } => {
                            self.substitute_params(expr, params, args);
                        }
                        Statement::Return { value: Some(expr), .. } => {
                            self.substitute_params(expr, params, args);
                        }
                        Statement::Assign { value, .. } => {
                            self.substitute_params(value, params, args);
                        }
                        _ => {}
                    }
                }
            }
            Expression::Match { value, arms, .. } => {
                // Substitute in the match value
                self.substitute_params(value, params, args);
                // Substitute in all match arm bodies
                for arm in arms.iter_mut() {
                    self.substitute_params(&mut arm.body, params, args);
                }
            }
            _ => {}
        }
    }

    fn inline_in_expression(&mut self, expr: &mut Expression) {
        self.try_inline_call(expr);

        match expr {
            Expression::Binary { left, right, .. } => {
                self.inline_in_expression(left);
                self.inline_in_expression(right);
            }
            Expression::Unary { operand, .. } => {
                self.inline_in_expression(operand);
            }
            Expression::Call { callee, args, .. } => {
                self.inline_in_expression(callee);
                for arg in args {
                    self.inline_in_expression(arg);
                }
            }
            Expression::List { elements, .. } => {
                for elem in elements {
                    self.inline_in_expression(elem);
                }
            }
            Expression::Block { statements, .. } => {
                for stmt in statements {
                    self.inline_in_statement(stmt);
                }
            }
            _ => {}
        }
    }

    fn inline_in_statement(&mut self, stmt: &mut Statement) {
        match stmt {
            Statement::Let { value, .. } => {
                self.inline_in_expression(value);
            }
            Statement::Expression { expr, .. } => {
                self.inline_in_expression(expr);
            }
            Statement::Return { value, .. } => {
                if let Some(expr) = value {
                    self.inline_in_expression(expr);
                }
            }
            Statement::If { condition, then_block, else_block, .. } => {
                self.inline_in_expression(condition);
                for stmt in then_block {
                    self.inline_in_statement(stmt);
                }
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        self.inline_in_statement(stmt);
                    }
                }
            }
            Statement::For { iterable, body, .. } => {
                self.inline_in_expression(iterable);
                for stmt in body {
                    self.inline_in_statement(stmt);
                }
            }
            Statement::While { condition, body, .. } => {
                self.inline_in_expression(condition);
                for stmt in body {
                    self.inline_in_statement(stmt);
                }
            }
            Statement::Function { body, .. } => {
                for stmt in body {
                    self.inline_in_statement(stmt);
                }
            }
            _ => {}
        }
    }
}

impl super::optimizer::OptimizationPass for FunctionInlining {
    fn name(&self) -> &str {
        "FunctionInlining"
    }

    fn run(&mut self, statements: &mut Vec<Statement>) -> Result<usize> {
        self.changes = 0;

        // First pass: collect inlineable functions
        self.collect_functions(statements);

        // Second pass: inline function calls
        for stmt in statements {
            self.inline_in_statement(stmt);
        }

        Ok(self.changes)
    }
}

