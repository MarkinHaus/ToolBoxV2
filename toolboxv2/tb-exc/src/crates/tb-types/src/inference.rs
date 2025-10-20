use tb_core::{BinaryOp, Literal, Result, TBError, Type, UnaryOp};

/// Type inference engine with constraint solving
pub struct TypeInference;

impl TypeInference {
    /// Infer the type of a literal
    pub fn infer_literal(lit: &Literal) -> Type {
        match lit {
            Literal::None => Type::None,
            Literal::Bool(_) => Type::Bool,
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::String(_) => Type::String,
        }
    }

    /// Infer binary operation result type with automatic promotion
    pub fn infer_binary_op(op: &BinaryOp, left: &Type, right: &Type) -> Result<Type> {
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                // Arithmetic operations with type promotion
                match (left, right) {
                    (Type::Int, Type::Int) => Ok(Type::Int),
                    (Type::Float, Type::Float) => Ok(Type::Float),

                    // Automatic promotion: Int + Float = Float
                    (Type::Int, Type::Float) | (Type::Float, Type::Int) => Ok(Type::Float),

                    // Handle Type::Any
                    (Type::Any, _) | (_, Type::Any) => Ok(Type::Any),

                    // String concatenation
                    (Type::String, Type::String) if matches!(op, BinaryOp::Add) => Ok(Type::String),

                    _ => Err(TBError::TypeError {
                        message: format!(
                            "Cannot apply {:?} to types {:?} and {:?}",
                            op, left, right
                        ),
                    }),
                }
            }
            BinaryOp::Eq | BinaryOp::NotEq => {
                // Equality works on same types
                if Self::types_compatible(left, right) ||
                   matches!(left, Type::Any) ||
                   matches!(right, Type::Any) {
                    Ok(Type::Bool)
                } else {
                    Err(TBError::TypeError {
                        message: format!("Cannot compare {:?} and {:?}", left, right),
                    })
                }
            }
            BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
                // Comparison for numeric types
                match (left, right) {
                    (Type::Int, Type::Int)
                    | (Type::Float, Type::Float)
                    | (Type::Int, Type::Float)
                    | (Type::Float, Type::Int)
                    | (Type::Any, _)
                    | (_, Type::Any) => Ok(Type::Bool),
                    _ => Err(TBError::TypeError {
                        message: format!("Cannot compare {:?} and {:?}", left, right),
                    }),
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                // Logical operations require bool
                match (left, right) {
                    (Type::Bool, Type::Bool) => Ok(Type::Bool),
                    (Type::Any, _) | (_, Type::Any) => Ok(Type::Bool),
                    _ => Err(TBError::TypeError {
                        message: format!("Logical operations require bool, got {:?} and {:?}", left, right),
                    }),
                }
            }
        }
    }

    /// Infer unary operation result type
    pub fn infer_unary_op(op: &UnaryOp, operand: &Type) -> Result<Type> {
        match op {
            UnaryOp::Neg => match operand {
                Type::Int => Ok(Type::Int),
                Type::Float => Ok(Type::Float),
                _ => Err(TBError::TypeError {
                    message: format!("Cannot negate type {:?}", operand),
                }),
            },
            UnaryOp::Not => match operand {
                Type::Bool => Ok(Type::Bool),
                _ => Err(TBError::TypeError {
                    message: format!("Cannot apply 'not' to type {:?}", operand),
                }),
            },
        }
    }

    /// Check if two types are compatible (for comparisons)
    pub fn types_compatible(a: &Type, b: &Type) -> bool {
        match (a, b) {
            (Type::Int, Type::Int) => true,
            (Type::Float, Type::Float) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::String, Type::String) => true,
            (Type::None, Type::None) => true,

            // Numeric compatibility
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,

            _ => false,
        }
    }

    /// Unify two types (find common type)
    pub fn unify(a: Type, b: Type) -> Result<Type> {
        match (&a, &b) {
            _ if a == b => Ok(a),

            // Int/Float promotion
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => Ok(Type::Float),

            _ => Err(TBError::TypeError {
                message: format!("Cannot unify types {:?} and {:?}", a, b),
            }),
        }
    }

    /// Promote type for operations (Int -> Float when needed)
    pub fn promote_type(ty: &Type) -> Type {
        match ty {
            Type::Int => Type::Float,
            other => other.clone(),
        }
    }
}

