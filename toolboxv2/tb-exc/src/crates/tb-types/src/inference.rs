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

                    _ => Err(TBError::type_error(format!(
                        "Cannot apply {:?} to types {:?} and {:?}",
                        op, left, right
                    ))),
                }
            }
            BinaryOp::Eq | BinaryOp::NotEq => {
                // Equality works on same types
                if Self::types_compatible(left, right) ||
                   matches!(left, Type::Any) ||
                   matches!(right, Type::Any) {
                    Ok(Type::Bool)
                } else {
                    Err(TBError::type_error(format!("Cannot compare {:?} and {:?}", left, right)))
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
                    _ => Err(TBError::type_error(format!("Cannot compare {:?} and {:?}", left, right))),
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                // Logical operations require bool
                match (left, right) {
                    (Type::Bool, Type::Bool) => Ok(Type::Bool),
                    (Type::Any, _) | (_, Type::Any) => Ok(Type::Bool),
                    _ => Err(TBError::type_error(format!(
                        "Logical operations require bool, got {:?} and {:?}", left, right
                    ))),
                }
            }
            BinaryOp::In => {
                // Membership test: left in right
                // right can be: String, List, Dict
                // Always returns Bool
                match right {
                    Type::String | Type::List(_) | Type::Dict(_, _) | Type::Any => Ok(Type::Bool),
                    _ => Err(TBError::type_error(format!(
                        "'in' operator requires String, List, or Dict on right side, got {:?}", right
                    ))),
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
                _ => Err(TBError::type_error(format!("Cannot negate type {:?}", operand))),
            },
            UnaryOp::Not => match operand {
                Type::Bool => Ok(Type::Bool),
                _ => Err(TBError::type_error(format!("Cannot apply 'not' to type {:?}", operand))),
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

            _ => Err(TBError::type_error(format!("Cannot unify types {:?} and {:?}", a, b))),
        }
    }

    /// Promote type for operations (Int -> Float when needed)
    pub fn promote_type(ty: &Type) -> Type {
        match ty {
            Type::Int => Type::Float,
            other => other.clone(),
        }
    }

    /// Find the Least Upper Bound (LUB) of multiple types
    /// This is the most general type that all input types can be converted to
    ///
    /// Examples:
    /// - LUB(Int, Int) = Int
    /// - LUB(Int, Float) = Float (Int can be promoted to Float)
    /// - LUB(Int, String) = Any (no common supertype)
    /// - LUB(Int, Float, Int) = Float
    pub fn least_upper_bound(types: &[Type]) -> Type {
        if types.is_empty() {
            return Type::Any;
        }

        if types.len() == 1 {
            return types[0].clone();
        }

        // Start with the first type
        let mut result = types[0].clone();

        // Try to unify with each subsequent type
        for ty in &types[1..] {
            result = match Self::try_lub_pair(&result, ty) {
                Some(lub) => lub,
                None => return Type::Any, // No common supertype, fall back to Any
            };
        }

        result
    }

    /// Try to find the LUB of two types, return None if no common supertype exists
    fn try_lub_pair(a: &Type, b: &Type) -> Option<Type> {
        // Same types
        if a == b {
            return Some(a.clone());
        }

        match (a, b) {
            // Int/Float promotion: Int can be promoted to Float
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => Some(Type::Float),

            // Any is the top type
            (Type::Any, _) | (_, Type::Any) => Some(Type::Any),

            // List types: LUB of element types
            (Type::List(elem_a), Type::List(elem_b)) => {
                Self::try_lub_pair(elem_a, elem_b).map(|elem_lub| Type::List(Box::new(elem_lub)))
            }

            // Dict types: LUB of key and value types
            (Type::Dict(key_a, val_a), Type::Dict(key_b, val_b)) => {
                let key_lub = Self::try_lub_pair(key_a, key_b)?;
                let val_lub = Self::try_lub_pair(val_a, val_b)?;
                Some(Type::Dict(Box::new(key_lub), Box::new(val_lub)))
            }

            // No common supertype
            _ => None,
        }
    }
}

