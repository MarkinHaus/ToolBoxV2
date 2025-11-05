use tb_core::*;
use std::sync::Arc;

/// Constant folding optimization pass
/// Evaluates constant expressions at compile time
pub struct ConstantFolding {
    changes: usize,
}

impl ConstantFolding {
    pub fn new() -> Self {
        Self { changes: 0 }
    }

    fn fold_expression(&mut self, expr: &mut Expression) -> bool {
        let mut changed = false;

        match expr {
            Expression::Binary { op, left, right, span } => {
                // First fold children
                changed |= self.fold_expression(left);
                changed |= self.fold_expression(right);

                // Try to fold this operation
                if let Some(folded) = self.try_fold_binary(op, left, right, *span) {
                    *expr = folded;
                    self.changes += 1;
                    changed = true;
                }
            }

            Expression::Unary { op, operand, span } => {
                changed |= self.fold_expression(operand);

                if let Some(folded) = self.try_fold_unary(op, operand, *span) {
                    *expr = folded;
                    self.changes += 1;
                    changed = true;
                }
            }

            Expression::List { elements, .. } => {
                for elem in elements {
                    changed |= self.fold_expression(elem);
                }
            }

            Expression::Dict { entries, .. } => {
                for (_, value) in entries {
                    changed |= self.fold_expression(value);
                }
            }

            Expression::Block { statements, .. } => {
                for stmt in statements {
                    match stmt {
                        Statement::Expression { expr, .. } => {
                            changed |= self.fold_expression(expr);
                        }
                        Statement::Return { value: Some(expr), .. } => {
                            changed |= self.fold_expression(expr);
                        }
                        Statement::Assign { value, .. } => {
                            changed |= self.fold_expression(value);
                        }
                        _ => {}
                    }
                }
            }

            Expression::Call { args, .. } => {
                for arg in args {
                    changed |= self.fold_expression(arg);
                }
            }

            _ => {}
        }

        changed
    }

    fn try_fold_binary(
        &self,
        op: &BinaryOp,
        left: &Expression,
        right: &Expression,
        span: Span,
    ) -> Option<Expression> {
        // Extract literal values
        let left_lit = self.extract_literal(left)?;
        let right_lit = self.extract_literal(right)?;

        let result = match (op, &left_lit, &right_lit) {
            // Integer arithmetic
            (BinaryOp::Add, Literal::Int(a), Literal::Int(b)) => Literal::Int(a + b),
            (BinaryOp::Sub, Literal::Int(a), Literal::Int(b)) => Literal::Int(a - b),
            (BinaryOp::Mul, Literal::Int(a), Literal::Int(b)) => Literal::Int(a * b),
            // Division always returns Float (Python-like behavior)
            (BinaryOp::Div, Literal::Int(a), Literal::Int(b)) if *b != 0 => Literal::Float(*a as f64 / *b as f64),
            (BinaryOp::Mod, Literal::Int(a), Literal::Int(b)) if *b != 0 => Literal::Int(a % b),

            // Float arithmetic
            (BinaryOp::Add, Literal::Float(a), Literal::Float(b)) => Literal::Float(a + b),
            (BinaryOp::Sub, Literal::Float(a), Literal::Float(b)) => Literal::Float(a - b),
            (BinaryOp::Mul, Literal::Float(a), Literal::Float(b)) => Literal::Float(a * b),
            (BinaryOp::Div, Literal::Float(a), Literal::Float(b)) if *b != 0.0 => Literal::Float(a / b),

            // Mixed int/float arithmetic
            (BinaryOp::Add, Literal::Int(a), Literal::Float(b)) => Literal::Float(*a as f64 + b),
            (BinaryOp::Add, Literal::Float(a), Literal::Int(b)) => Literal::Float(a + *b as f64),
            (BinaryOp::Div, Literal::Int(a), Literal::Float(b)) if *b != 0.0 => Literal::Float(*a as f64 / b),
            (BinaryOp::Div, Literal::Float(a), Literal::Int(b)) if *b != 0 => Literal::Float(a / *b as f64),

            // String concatenation
            (BinaryOp::Add, Literal::String(a), Literal::String(b)) => {
                Literal::String(Arc::new(format!("{}{}", a, b)))
            }

            // Boolean operations
            (BinaryOp::And, Literal::Bool(a), Literal::Bool(b)) => Literal::Bool(*a && *b),
            (BinaryOp::Or, Literal::Bool(a), Literal::Bool(b)) => Literal::Bool(*a || *b),

            // Comparisons - integers
            (BinaryOp::Eq, Literal::Int(a), Literal::Int(b)) => Literal::Bool(a == b),
            (BinaryOp::NotEq, Literal::Int(a), Literal::Int(b)) => Literal::Bool(a != b),
            (BinaryOp::Lt, Literal::Int(a), Literal::Int(b)) => Literal::Bool(a < b),
            (BinaryOp::Gt, Literal::Int(a), Literal::Int(b)) => Literal::Bool(a > b),
            (BinaryOp::LtEq, Literal::Int(a), Literal::Int(b)) => Literal::Bool(a <= b),
            (BinaryOp::GtEq, Literal::Int(a), Literal::Int(b)) => Literal::Bool(a >= b),

            // Comparisons - floats
            (BinaryOp::Eq, Literal::Float(a), Literal::Float(b)) => Literal::Bool((a - b).abs() < f64::EPSILON),
            (BinaryOp::Lt, Literal::Float(a), Literal::Float(b)) => Literal::Bool(a < b),
            (BinaryOp::Gt, Literal::Float(a), Literal::Float(b)) => Literal::Bool(a > b),

            _ => return None,
        };

        Some(Expression::Literal(result, span))
    }

    fn try_fold_unary(&self, op: &UnaryOp, operand: &Expression, span: Span) -> Option<Expression> {
        let lit = self.extract_literal(operand)?;

        let result = match (op, &lit) {
            (UnaryOp::Neg, Literal::Int(n)) => Literal::Int(-n),
            (UnaryOp::Neg, Literal::Float(f)) => Literal::Float(-f),
            (UnaryOp::Not, Literal::Bool(b)) => Literal::Bool(!b),
            _ => return None,
        };

        Some(Expression::Literal(result, span))
    }

    fn extract_literal<'a>(&self, expr: &'a Expression) -> Option<&'a Literal> {
        match expr {
            Expression::Literal(lit, _) => Some(lit),
            _ => None,
        }
    }
}

impl super::optimizer::OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn run(&mut self, statements: &mut Vec<Statement>) -> Result<usize> {
        self.changes = 0;

        for stmt in statements {
            self.fold_statement(stmt);
        }

        Ok(self.changes)
    }
}

impl ConstantFolding {
    fn fold_statement(&mut self, stmt: &mut Statement) {
        match stmt {
            Statement::Let { value, .. } => {
                self.fold_expression(value);
            }
            Statement::Function { body, .. } => {
                for stmt in body {
                    self.fold_statement(stmt);
                }
            }
            Statement::If { condition, then_block, else_block, .. } => {
                self.fold_expression(condition);
                for stmt in then_block {
                    self.fold_statement(stmt);
                }
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        self.fold_statement(stmt);
                    }
                }
            }
            Statement::For { iterable, body, .. } => {
                self.fold_expression(iterable);
                for stmt in body {
                    self.fold_statement(stmt);
                }
            }
            Statement::While { condition, body, .. } => {
                self.fold_expression(condition);
                for stmt in body {
                    self.fold_statement(stmt);
                }
            }
            Statement::Return { value, .. } => {
                if let Some(expr) = value {
                    self.fold_expression(expr);
                }
            }
            Statement::Expression { expr, .. } => {
                self.fold_expression(expr);
            }
            _ => {}
        }
    }
}

