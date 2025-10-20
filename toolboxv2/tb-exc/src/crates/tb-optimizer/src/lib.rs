pub mod constant_folding;
pub mod dead_code;
pub mod function_inlining;
pub mod optimizer;

pub use optimizer::{Optimizer, OptimizerConfig, OptimizationStats};

