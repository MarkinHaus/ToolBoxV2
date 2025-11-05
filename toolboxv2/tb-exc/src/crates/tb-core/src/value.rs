use im::HashMap as ImHashMap;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, RwLock};

/// Runtime value representation - optimized for zero-copy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<String>),
    List(Arc<Vec<Value>>),
    Dict(Arc<ImHashMap<Arc<String>, Value>>),
    Function(Arc<Function>),
    #[serde(skip)]
    NativeFunction(Arc<NativeFunction>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: Arc<String>,
    pub params: Vec<Arc<String>>,
    pub body: Vec<crate::ast::Statement>,
    pub return_type: Option<crate::ast::Type>,
    /// Captured environment for closures (None for regular functions)
    /// Uses RwLock to allow mutable closures (closures that modify captured variables)
    #[serde(skip)]
    pub closure_env: Option<Arc<RwLock<ImHashMap<Arc<String>, Value>>>>,
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        // Compare everything except closure_env (RwLock can't be compared)
        self.name == other.name
            && self.params == other.params
            && self.body == other.body
            && self.return_type == other.return_type
    }
}

#[derive(Clone)]
pub struct NativeFunction {
    pub name: Arc<String>,
    pub func: Arc<dyn Fn(Vec<Value>) -> crate::Result<Value> + Send + Sync>,
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeFunction")
            .field("name", &self.name)
            .finish()
    }
}

impl PartialEq for NativeFunction {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Function(_) => "function",
            Value::NativeFunction(_) => "native_function",
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Dict(d) => !d.is_empty(),
            _ => true,
        }
    }

    pub fn to_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    pub fn to_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::None => write!(f, "None"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => {
                // Always show decimal point for floats
                if fl.fract() == 0.0 && fl.is_finite() {
                    // Integer-like float: show .0
                    write!(f, "{:.1}", fl)
                } else {
                    // Regular float: use default formatting
                    write!(f, "{}", fl)
                }
            }
            Value::String(s) => write!(f, "{}", s),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Dict(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Function(func) => write!(f, "<function {}>", func.name),
            Value::NativeFunction(func) => write!(f, "<native function {}>", func.name),
        }
    }
}

