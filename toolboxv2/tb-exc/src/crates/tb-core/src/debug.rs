// Debug logging utilities for TB Language
// Only active in debug builds

/// Debug log macro - only active in debug mode
#[macro_export]
macro_rules! tb_debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB DEBUG] {}", format!($($arg)*));
        }
    };
}

/// Debug log with location info
#[macro_export]
macro_rules! tb_debug_loc {
    ($loc:expr, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB DEBUG @ {}] {}", $loc, format!($($arg)*));
        }
    };
}

/// Debug log for JIT execution
#[macro_export]
macro_rules! tb_debug_jit {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB JIT] {}", format!($($arg)*));
        }
    };
}

/// Debug log for type checking
#[macro_export]
macro_rules! tb_debug_type {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB TYPE] {}", format!($($arg)*));
        }
    };
}

/// Debug log for compilation
#[macro_export]
macro_rules! tb_debug_compile {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB COMPILE] {}", format!($($arg)*));
        }
    };
}

/// Debug log for plugins
#[macro_export]
macro_rules! tb_debug_plugin {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB PLUGIN] {}", format!($($arg)*));
        }
    };
}

/// Debug log for FFI
#[macro_export]
macro_rules! tb_debug_ffi {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB FFI] {}", format!($($arg)*));
        }
    };
}

/// Trace execution flow
#[macro_export]
macro_rules! tb_trace {
    ($fn_name:expr) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB TRACE] Entering: {}", $fn_name);
        }
    };
}

/// Trace with value
#[macro_export]
macro_rules! tb_trace_val {
    ($fn_name:expr, $val:expr) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[TB TRACE] {}: {:?}", $fn_name, $val);
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_debug_macros() {
        tb_debug!("Test debug message");
        tb_debug_loc!("test.rs:10", "Test with location");
        tb_debug_jit!("JIT test");
        tb_debug_type!("Type test");
        tb_trace!("test_function");
    }
}

