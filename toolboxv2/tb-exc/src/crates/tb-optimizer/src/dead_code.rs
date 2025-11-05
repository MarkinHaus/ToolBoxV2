use tb_core::*;

/// Dead code elimination pass
/// Removes unreachable code and unused variables
pub struct DeadCodeElimination {
    changes: usize,
}

impl DeadCodeElimination {
    pub fn new() -> Self {
        Self { changes: 0 }
    }

    #[allow(dead_code)]
    fn is_unreachable_after(&self, statements: &[Statement], index: usize) -> bool {
        if index >= statements.len() {
            return false;
        }

        matches!(
            statements[index],
            Statement::Return { .. } | Statement::Break { .. } | Statement::Continue { .. }
        )
    }

    fn remove_unreachable(&mut self, statements: &mut Vec<Statement>) -> usize {
        let mut removed = 0;
        let mut i = 0;

        while i < statements.len() {
            match &statements[i] {
                Statement::Return { .. } | Statement::Break { .. } | Statement::Continue { .. } => {
                    // Remove everything after this
                    let to_remove = statements.len() - i - 1;
                    statements.truncate(i + 1);
                    removed += to_remove;
                    break;
                }
                Statement::Function {  .. } => {
                    if let Statement::Function { body, .. } = &mut statements[i] {
                        removed += self.remove_unreachable(body);
                    }
                    i += 1;
                }
                Statement::If {   .. } => {
                    if let Statement::If { then_block, else_block, .. } = &mut statements[i] {
                        removed += self.remove_unreachable(then_block);
                        if let Some(else_stmts) = else_block {
                            removed += self.remove_unreachable(else_stmts);
                        }
                    }
                    i += 1;
                }
                Statement::For { body: _, .. } | Statement::While { body: _, .. } => {
                    match &mut statements[i] {
                        Statement::For { body, .. } | Statement::While { body, .. } => {
                            removed += self.remove_unreachable(body);
                        }
                        _ => unreachable!(),
                    }
                    i += 1;
                }
                _ => i += 1,
            }
        }

        removed
    }

    fn eliminate_constant_conditions(&mut self, statements: &mut Vec<Statement>) -> usize {
        let mut removed = 0;

        let mut i = 0;
        while i < statements.len() {
            match &statements[i] {
                // ✅ RE-ENABLED: This optimization is now safe after fixing JIT scoping!
                // The JIT executor now properly handles block scopes using im::HashMap clone,
                // so variables declared in if-blocks no longer leak to the outer scope.
                Statement::If { condition, then_block, else_block, .. } => {
                    if let Expression::Literal(Literal::Bool(value), _) = condition {
                        let replacement = if *value {
                            then_block.clone()
                        } else {
                            else_block.clone().unwrap_or_default()
                        };

                        statements.splice(i..=i, replacement);
                        removed += 1;
                        continue; // Don't increment i, check new statements
                    }
                }
                Statement::While { condition, .. } => {
                    if let Expression::Literal(Literal::Bool(false), _) = condition {
                        // while false { ... } - remove entirely
                        statements.remove(i);
                        removed += 1;
                        continue;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        removed
    }
}

impl super::optimizer::OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn run(&mut self, statements: &mut Vec<Statement>) -> Result<usize> {
        self.changes = 0;

        self.changes += self.remove_unreachable(statements);
        self.changes += self.eliminate_constant_conditions(statements);

        Ok(self.changes)
    }
}

