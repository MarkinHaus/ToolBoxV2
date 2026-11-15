use thiserror::Error;

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Security violation: path {0:?} is outside workspace root")]
    SecurityViolation(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Patch failed: context mismatch or invalid hunk")]
    PatchFailed(String),
    #[error("Command timed out after {0} ms")]
    Timeout(u32),
    #[error("Subprocess forbidden by policy: {0}")]
    PolicyViolation(String),
    #[error("Encoding error: non-UTF-8 payload")]
    InvalidEncoding,
}
