pub mod app;
pub mod guard;
pub mod logic;

pub use app::{format_tool_line, run, Mode, ToolLineView, UiState};
pub use guard::TerminalGuard;
pub use logic::{project_dag_tree, render_token_meter};

/// Legacy no-op init hook.
pub fn init() {}
