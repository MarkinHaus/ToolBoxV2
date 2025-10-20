//! Error types for built-in functions

use thiserror::Error;
use tb_core::TBError;

#[derive(Error, Debug)]
pub enum BuiltinError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Blob storage error: {0}")]
    BlobStorage(String),
    
    #[error("Encryption error: {0}")]
    Encryption(String),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Runtime error: {0}")]
    Runtime(String),
}

pub type BuiltinResult<T> = Result<T, BuiltinError>;

impl From<BuiltinError> for TBError {
    fn from(err: BuiltinError) -> Self {
        TBError::RuntimeError {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for BuiltinError {
    fn from(err: serde_json::Error) -> Self {
        BuiltinError::Serialization(err.to_string())
    }
}

impl From<serde_yaml::Error> for BuiltinError {
    fn from(err: serde_yaml::Error) -> Self {
        BuiltinError::Serialization(err.to_string())
    }
}

impl From<bincode::Error> for BuiltinError {
    fn from(err: bincode::Error) -> Self {
        BuiltinError::Serialization(err.to_string())
    }
}

