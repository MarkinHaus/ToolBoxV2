use crate::constant_folding::ConstantFolding;
use crate::dead_code::DeadCodeElimination;
use crate::function_inlining::FunctionInlining;
use tb_core::{Program, Result, Statement};

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub max_passes: usize,
    pub optimization_level: u8, // 0-3
    pub inline_threshold: usize,
    pub enable_constant_folding: bool,
    pub enable_dead_code_elimination: bool,
    pub enable_function_inlining: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_passes: 5,
            optimization_level: 2,
            inline_threshold: 50,
            enable_constant_folding: true,
            enable_dead_code_elimination: true,
            enable_function_inlining: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    pub total_passes: usize,
    pub constants_folded: usize,
    pub dead_code_removed: usize,
    pub functions_inlined: usize,
    pub total_time_ms: u128,
}

pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&mut self, statements: &mut Vec<Statement>) -> Result<usize>;
}

pub struct Optimizer {
    config: OptimizerConfig,
    stats: OptimizationStats,
}

impl Optimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats::default(),
        }
    }

    pub fn optimize(&mut self, program: &mut Program) -> Result<OptimizationStats> {
        let start = std::time::Instant::now();

        for _pass_num in 0..self.config.max_passes {
            let mut changes = 0;

            // Pass 1: Constant Folding
            if self.config.enable_constant_folding {
                let mut pass = ConstantFolding::new();
                let folded = pass.run(&mut program.statements)?;
                changes += folded;
                self.stats.constants_folded += folded;
            }

            // Pass 2: Dead Code Elimination
            if self.config.enable_dead_code_elimination {
                let mut pass = DeadCodeElimination::new();
                let removed = pass.run(&mut program.statements)?;
                changes += removed;
                self.stats.dead_code_removed += removed;
            }

            // Pass 3: Function Inlining (only at higher opt levels)
            if self.config.enable_function_inlining && self.config.optimization_level >= 2 {
                let mut pass = FunctionInlining::new(self.config.inline_threshold);
                let inlined = pass.run(&mut program.statements)?;
                changes += inlined;
                self.stats.functions_inlined += inlined;
            }

            self.stats.total_passes += 1;

            // If no changes, we're done
            if changes == 0 {
                break;
            }
        }

        self.stats.total_time_ms = start.elapsed().as_millis();
        Ok(self.stats.clone())
    }

    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }
}

