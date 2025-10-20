use std::path::PathBuf;
use thiserror::Error;

#[cfg(debug_assertions)]
use colored::Colorize;

#[derive(Error, Debug, Clone)]
pub enum TBError {
    #[error("Syntax error at {location}: {message}")]
    SyntaxError { location: String, message: String },

    #[error("Type error: {message}")]
    TypeError { message: String },

    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    #[error("Import error: failed to load {path}: {reason}")]
    ImportError { path: PathBuf, reason: String },

    #[error("Plugin error: {message}")]
    PluginError { message: String },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Compilation error: {message}")]
    CompilationError { message: String },

    #[error("Undefined variable: {name}")]
    UndefinedVariable { name: String },

    #[error("Undefined function: {name}")]
    UndefinedFunction { name: String },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("Format error: {0}")]
    FormatError(String),
}

pub type Result<T> = std::result::Result<T, TBError>;

impl TBError {
    /// Get a detailed error message with stack trace (debug builds only)
    pub fn detailed_message(&self) -> String {
        #[cfg(debug_assertions)]
        {
            self.detailed_message_debug()
        }
        #[cfg(not(debug_assertions))]
        {
            format!("{}", self)
        }
    }

    #[cfg(debug_assertions)]
    fn detailed_message_debug(&self) -> String {
        use std::backtrace::Backtrace;

        let mut output = String::new();

        // Error header
        output.push_str(&format!("\n{}\n", "═".repeat(80).bright_red()));
        output.push_str(&format!("{}\n", "ERROR".bright_red().bold()));
        output.push_str(&format!("{}\n\n", "═".repeat(80).bright_red()));

        // Error message
        output.push_str(&format!("{}: {}\n\n", "Message".bright_yellow().bold(), format!("{}", self).bright_white()));

        // Rust backtrace
        let bt = Backtrace::capture();
        if bt.status() == std::backtrace::BacktraceStatus::Captured {
            output.push_str(&format!("{}\n", "Rust Backtrace:".bright_cyan().bold()));
            output.push_str(&format!("{:?}\n", bt));
        }

        output.push_str(&format!("{}\n", "═".repeat(80).bright_red()));
        output
    }
}

impl From<std::io::Error> for TBError {
    fn from(err: std::io::Error) -> Self {
        TBError::IoError(err.to_string())
    }
}

impl From<std::fmt::Error> for TBError {
    fn from(err: std::fmt::Error) -> Self {
        TBError::FormatError(err.to_string())
    }
}

