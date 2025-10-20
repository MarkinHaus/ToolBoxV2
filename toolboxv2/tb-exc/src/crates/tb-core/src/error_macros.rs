//! Macros for convenient error creation with automatic location tracking

/// Create a runtime error with automatic location tracking (debug only)
///
/// # Examples
/// ```
/// runtime_error!("Division by zero");
/// runtime_error!("Invalid value: {}", value);
/// ```
#[macro_export]
macro_rules! runtime_error {
    ($msg:expr) => {{
        #[cfg(debug_assertions)]
        {
            $crate::error::TBError::runtime_error_at(
                $msg.to_string(),
                $crate::error::SourceLocation::new(
                    Some(file!().to_string()),
                    line!() as usize,
                    column!() as usize,
                    None,
                ),
            )
        }
        #[cfg(not(debug_assertions))]
        {
            $crate::error::TBError::runtime_error($msg.to_string())
        }
    }};
    ($fmt:expr, $($arg:tt)*) => {{
        runtime_error!(format!($fmt, $($arg)*))
    }};
}

/// Create a syntax error with location
#[macro_export]
macro_rules! syntax_error {
    ($location:expr, $msg:expr) => {{
        $crate::error::TBError::syntax_error($location.to_string(), $msg.to_string())
    }};
    ($location:expr, $fmt:expr, $($arg:tt)*) => {{
        syntax_error!($location, format!($fmt, $($arg)*))
    }};
}

/// Create a type error with location
#[macro_export]
macro_rules! type_error {
    ($msg:expr) => {{
        $crate::error::TBError::type_error($msg.to_string())
    }};
    ($fmt:expr, $($arg:tt)*) => {{
        type_error!(format!($fmt, $($arg)*))
    }};
}

/// Create a compilation error with Rust backtrace (debug only)
#[macro_export]
macro_rules! compilation_error {
    ($msg:expr) => {{
        $crate::error::TBError::compilation_error($msg.to_string())
    }};
    ($fmt:expr, $($arg:tt)*) => {{
        compilation_error!(format!($fmt, $($arg)*))
    }};
}

/// Create an undefined variable error
#[macro_export]
macro_rules! undefined_var {
    ($name:expr) => {{
        $crate::error::TBError::undefined_variable($name.to_string())
    }};
}

/// Create an undefined function error
#[macro_export]
macro_rules! undefined_fn {
    ($name:expr) => {{
        $crate::error::TBError::undefined_function($name.to_string())
    }};
}

/// Add a stack frame to an error (debug only)
#[macro_export]
macro_rules! with_frame {
    ($error:expr, $fn_name:expr) => {{
        #[cfg(debug_assertions)]
        {
            $error.with_frame(
                $fn_name.to_string(),
                $crate::error::SourceLocation::new(
                    Some(file!().to_string()),
                    line!() as usize,
                    column!() as usize,
                    None,
                ),
            )
        }
        #[cfg(not(debug_assertions))]
        {
            $error
        }
    }};
}

/// Trace a function call (debug only) - adds frame on error
#[macro_export]
macro_rules! trace_call {
    ($fn_name:expr, $expr:expr) => {{
        #[cfg(debug_assertions)]
        {
            $expr.map_err(|e| with_frame!(e, $fn_name))
        }
        #[cfg(not(debug_assertions))]
        {
            $expr
        }
    }};
}

