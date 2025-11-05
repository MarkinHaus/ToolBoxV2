use crate::span::Span;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Literal(Literal),
    Ident(Arc<String>),
    Wildcard,
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Let {
        name: Arc<String>,
        type_annotation: Option<Type>,
        value: Expression,
        span: Span,
    },
    Assign {
        target: Expression,  // Can be Ident, Member, or Index
        value: Expression,
        span: Span,
    },
    Function {
        name: Arc<String>,
        params: Vec<Parameter>,
        return_type: Option<Type>,
        body: Vec<Statement>,
        span: Span,
    },
    If {
        condition: Expression,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
        span: Span,
    },
    For {
        variable: Arc<String>,
        iterable: Expression,
        body: Vec<Statement>,
        span: Span,
    },
    While {
        condition: Expression,
        body: Vec<Statement>,
        span: Span,
    },
    Match {
        value: Expression,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Return {
        value: Option<Expression>,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    Expression {
        expr: Expression,
        span: Span,
    },
    Import {
        items: Vec<ImportItem>,
        span: Span,
    },
    Config {
        entries: Vec<ConfigEntry>,
        span: Span,
    },
    Plugin {
        definitions: Vec<PluginDefinition>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: Arc<String>,
    pub type_annotation: Option<Type>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportItem {
    pub path: Arc<String>,
    pub alias: Option<Arc<String>>,
    pub condition: Option<Arc<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigEntry {
    pub key: Arc<String>,
    pub value: ConfigValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    Int(i64),
    String(Arc<String>),
    Dict(Vec<ConfigEntry>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginDefinition {
    pub language: PluginLanguage,
    pub name: Arc<String>,
    pub mode: PluginMode,
    pub requires: Vec<Arc<String>>,
    pub source: PluginSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginLanguage {
    Python,
    JavaScript,
    Go,
    Rust,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginMode {
    Jit,
    Compile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginSource {
    Inline(Arc<String>),
    File(Arc<String>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    Literal(Literal, Span),
    Ident(Arc<String>, Span),
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expression>,
        span: Span,
    },
    Call {
        callee: Box<Expression>,
        args: Vec<Expression>,
        span: Span,
    },
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
        span: Span,
    },
    Member {
        object: Box<Expression>,
        member: Arc<String>,
        span: Span,
    },
    List {
        elements: Vec<Expression>,
        span: Span,
    },
    Dict {
        entries: Vec<(Arc<String>, Expression)>,
        span: Span,
    },
    Lambda {
        params: Vec<Parameter>,
        body: Box<Expression>,
        span: Span,
    },
    Match {
        value: Box<Expression>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Range {
        start: Box<Expression>,
        end: Box<Expression>,
        inclusive: bool,
        span: Span,
    },
    If {
        condition: Box<Expression>,
        then_branch: Box<Expression>,
        else_branch: Box<Expression>,
        span: Span,
    },
    Block {
        statements: Vec<Statement>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    In,  // Membership test: "x" in list, "key" in dict, "sub" in string
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    None,
    Any,  // Accept any type (no type checking)
    List(Box<Type>),
    Dict(Box<Type>, Box<Type>),
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Generic(Arc<String>),
}

impl Expression {
    pub fn span(&self) -> &Span {
        match self {
            Expression::Literal(_, span) => span,
            Expression::Ident(_, span) => span,
            Expression::Binary { span, .. } => span,
            Expression::Unary { span, .. } => span,
            Expression::Call { span, .. } => span,
            Expression::Index { span, .. } => span,
            Expression::Member { span, .. } => span,
            Expression::List { span, .. } => span,
            Expression::Dict { span, .. } => span,
            Expression::Lambda { span, .. } => span,
            Expression::Match { span, .. } => span,
            Expression::Range { span, .. } => span,
            Expression::If { span, .. } => span,
            Expression::Block { span, .. } => span,
        }
    }
}

