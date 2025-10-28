use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use crate::Span;

#[cfg(debug_assertions)]
use colored::Colorize;

/// Source context for better error messages
#[derive(Debug, Clone)]
pub struct SourceContext {
    pub source: Arc<String>,
    pub file_path: Option<PathBuf>,
}

impl SourceContext {
    pub fn new(source: String, file_path: Option<PathBuf>) -> Self {
        Self {
            source: Arc::new(source),
            file_path,
        }
    }

    pub fn get_line(&self, line_num: usize) -> Option<String> {
        self.source.lines().nth(line_num.saturating_sub(1)).map(|s| s.to_string())
    }

    /// Get line content for debug output (returns empty string if not found)
    pub fn get_line_content(&self, line_num: usize) -> String {
        self.get_line(line_num).unwrap_or_else(|| String::from("<source unavailable>"))
    }

    pub fn get_context(&self, span: &Span, context_lines: usize) -> Vec<(usize, String)> {
        let start_line = span.line.saturating_sub(context_lines).max(1);
        let end_line = span.line + context_lines;

        self.source
            .lines()
            .enumerate()
            .skip(start_line - 1)
            .take(end_line - start_line + 1)
            .map(|(i, line)| (i + 1, line.to_string()))
            .collect()
    }
}

#[derive(Error, Debug, Clone)]
pub enum TBError {
    #[error("Syntax error at {location}: {message}")]
    SyntaxError {
        location: String,
        message: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
    },

    #[error("Type error: {message}")]
    TypeError {
        message: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
    },

    #[error("Runtime error: {message}")]
    RuntimeError {
        message: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
        call_stack: Vec<String>,
    },

    #[error("Import error: failed to load {path}: {reason}")]
    ImportError {
        path: PathBuf,
        reason: String,
    },

    #[error("Plugin error: {message}")]
    PluginError {
        message: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
    },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Compilation error: {message}")]
    CompilationError {
        message: String,
        compiler_output: Option<String>,
    },

    #[error("Undefined variable: {name}")]
    UndefinedVariable {
        name: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
    },

    #[error("Undefined function: {name}")]
    UndefinedFunction {
        name: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
    },

    #[error("Invalid operation: {message}")]
    InvalidOperation {
        message: String,
        span: Option<Span>,
        source_context: Option<SourceContext>,
    },

    #[error("Format error: {0}")]
    FormatError(String),
}

pub type Result<T> = std::result::Result<T, TBError>;

impl TBError {
    /// Helper: Create RuntimeError without context (for backward compatibility)
    pub fn runtime_error(message: impl Into<String>) -> Self {
        TBError::RuntimeError {
            message: message.into(),
            span: None,
            source_context: None,
            call_stack: Vec::new(),
        }
    }

    /// Helper: Create TypeError without context (for backward compatibility)
    pub fn type_error(message: impl Into<String>) -> Self {
        TBError::TypeError {
            message: message.into(),
            span: None,
            source_context: None,
        }
    }

    /// Helper: Create UndefinedVariable without context (for backward compatibility)
    pub fn undefined_variable(name: impl Into<String>) -> Self {
        TBError::UndefinedVariable {
            name: name.into(),
            span: None,
            source_context: None,
        }
    }

    /// Helper: Create UndefinedFunction without context (for backward compatibility)
    pub fn undefined_function(name: impl Into<String>) -> Self {
        TBError::UndefinedFunction {
            name: name.into(),
            span: None,
            source_context: None,
        }
    }

    /// Helper: Create PluginError without context (for backward compatibility)
    pub fn plugin_error(message: impl Into<String>) -> Self {
        TBError::PluginError {
            message: message.into(),
            span: None,
            source_context: None,
        }
    }

    /// Helper: Create InvalidOperation without context (for backward compatibility)
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        TBError::InvalidOperation {
            message: message.into(),
            span: None,
            source_context: None,
        }
    }

    /// Helper: Create CompilationError without context (for backward compatibility)
    pub fn compilation_error(message: impl Into<String>) -> Self {
        TBError::CompilationError {
            message: message.into(),
            compiler_output: None,
        }
    }

    /// Get a detailed error message with source context and helpful hints
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
        let mut output = String::new();

        // Error header
        output.push_str(&format!("\n{}\n", "═".repeat(80).bright_red()));
        output.push_str(&format!("{}\n", "ERROR".bright_red().bold()));
        output.push_str(&format!("{}\n\n", "═".repeat(80).bright_red()));

        // Error type and message
        output.push_str(&format!("{}: {}\n",
            self.error_type().bright_yellow().bold(),
            self.main_message().bright_white()));

        // Source location and context
        if let Some((span, ctx)) = self.get_span_and_context() {
            output.push_str(&format!("\n{}\n", "Location:".bright_cyan().bold()));

            if let Some(path) = &ctx.file_path {
                output.push_str(&format!("  File: {}\n", path.display()));
            }

            output.push_str(&format!("  Line: {}, Column: {}\n\n", span.line, span.column));

            // Show source code with error marker
            if let Some(source_line) = ctx.get_line(span.line) {
                let line_num_width = format!("{}", span.line).len().max(4);

                // Show context lines (1 before, error line, 1 after)
                let context = ctx.get_context(&span, 1);
                for (line_num, line_text) in context {
                    if line_num == span.line {
                        // Error line with marker
                        output.push_str(&format!("{:>width$} | {}\n",
                            line_num.to_string().bright_white().bold(),
                            line_text,
                            width = line_num_width));

                        // Error marker (^^^)
                        let marker_start = span.column.saturating_sub(1);
                        let marker_len = (span.end - span.start).max(1);
                        output.push_str(&format!("{:>width$} | {}{}\n",
                            "",
                            " ".repeat(marker_start),
                            "^".repeat(marker_len).bright_red().bold(),
                            width = line_num_width));
                    } else {
                        // Context line
                        output.push_str(&format!("{:>width$} | {}\n",
                            line_num.to_string().bright_black(),
                            line_text.bright_black(),
                            width = line_num_width));
                    }
                }
            }
        }

        // TB Call Stack (for RuntimeError)
        if let TBError::RuntimeError { call_stack, .. } = self {
            if !call_stack.is_empty() {
                output.push_str(&format!("\n{}\n", "TB Call Stack:".bright_cyan().bold()));
                for (i, frame) in call_stack.iter().enumerate() {
                    output.push_str(&format!("  {} {}\n", i, frame));
                }
            }
        }

        // Compiler output (for CompilationError)
        if let TBError::CompilationError { compiler_output: Some(output_text), .. } = self {
            output.push_str(&format!("\n{}\n", "Compiler Output:".bright_cyan().bold()));
            output.push_str(&format!("{}\n", output_text.bright_black()));
        }

        // Helpful hints
        if let Some(hint) = self.get_hint() {
            output.push_str(&format!("\n{}: {}\n",
                "Hint".bright_green().bold(),
                hint.bright_white()));
        }

        output.push_str(&format!("\n{}\n", "═".repeat(80).bright_red()));
        output
    }

    #[cfg(debug_assertions)]
    fn error_type(&self) -> &str {
        match self {
            TBError::SyntaxError { .. } => "Syntax Error",
            TBError::TypeError { .. } => "Type Error",
            TBError::RuntimeError { .. } => "Runtime Error",
            TBError::ImportError { .. } => "Import Error",
            TBError::PluginError { .. } => "Plugin Error",
            TBError::CacheError { .. } => "Cache Error",
            TBError::IoError(_) => "I/O Error",
            TBError::CompilationError { .. } => "Compilation Error",
            TBError::UndefinedVariable { .. } => "Undefined Variable",
            TBError::UndefinedFunction { .. } => "Undefined Function",
            TBError::InvalidOperation { .. } => "Invalid Operation",
            TBError::FormatError(_) => "Format Error",
        }
    }

    #[cfg(debug_assertions)]
    fn main_message(&self) -> String {
        match self {
            TBError::SyntaxError { message, .. } => message.clone(),
            TBError::TypeError { message, .. } => message.clone(),
            TBError::RuntimeError { message, .. } => message.clone(),
            TBError::ImportError { path, reason, .. } => {
                format!("Failed to load '{}': {}", path.display(), reason)
            }
            TBError::PluginError { message, .. } => message.clone(),
            TBError::CacheError { message } => message.clone(),
            TBError::IoError(msg) => msg.clone(),
            TBError::CompilationError { message, .. } => message.clone(),
            TBError::UndefinedVariable { name, .. } => format!("Variable '{}' is not defined", name),
            TBError::UndefinedFunction { name, .. } => format!("Function '{}' is not defined", name),
            TBError::InvalidOperation { message, .. } => message.clone(),
            TBError::FormatError(msg) => msg.clone(),
        }
    }

    #[cfg(debug_assertions)]
    fn get_span_and_context(&self) -> Option<(Span, SourceContext)> {
        match self {
            TBError::SyntaxError { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            TBError::TypeError { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            TBError::RuntimeError { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            TBError::PluginError { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            TBError::UndefinedVariable { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            TBError::UndefinedFunction { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            TBError::InvalidOperation { span: Some(s), source_context: Some(ctx), .. } => Some((*s, ctx.clone())),
            _ => None,
        }
    }

    #[cfg(debug_assertions)]
    fn get_hint(&self) -> Option<String> {
        match self {
            TBError::UndefinedVariable { name, .. } => {
                Some(format!("Did you forget to declare '{}'? Use: let {} = ...", name, name))
            }
            TBError::UndefinedFunction { name, .. } => {
                Some(format!("Did you forget to define '{}'? Use: fn {}(...) {{ ... }}", name, name))
            }
            TBError::TypeError { message, .. } if message.contains("Int") && message.contains("Float") => {
                Some("Try converting types explicitly with int() or float()".to_string())
            }
            TBError::SyntaxError { message, .. } if message.contains("Expected") => {
                Some("Check for missing brackets, parentheses, or semicolons".to_string())
            }
            TBError::CompilationError { .. } => {
                Some("Check the compiler output above for details. This is usually a code generation issue.".to_string())
            }
            _ => None,
        }
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

