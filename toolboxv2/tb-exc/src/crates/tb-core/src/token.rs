use crate::span::Span;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    Float(f64),
    String(Arc<String>),
    True,
    False,
    None,

    // Identifiers
    Ident(Arc<String>),

    // Keywords
    Let,
    Fn,
    If,
    Else,
    For,
    In,
    While,
    Match,
    Return,
    Break,
    Continue,
    Import,
    Plugin,
    Config,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Bang,
    Arrow,      // ->
    FatArrow,   // =>
    DotDot,     // ..
    DotDotEq,   // ..=

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    At,         // @

    // Special
    Newline,
    Eof,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn dummy(kind: TokenKind) -> Self {
        Self {
            kind,
            span: Span::dummy(),
        }
    }
}

impl TokenKind {
    pub fn keyword(s: &str) -> Option<Self> {
        match s {
            "let" => Some(TokenKind::Let),
            "fn" => Some(TokenKind::Fn),
            "if" => Some(TokenKind::If),
            "else" => Some(TokenKind::Else),
            "for" => Some(TokenKind::For),
            "in" => Some(TokenKind::In),
            "while" => Some(TokenKind::While),
            "match" => Some(TokenKind::Match),
            "return" => Some(TokenKind::Return),
            "break" => Some(TokenKind::Break),
            "continue" => Some(TokenKind::Continue),
            "true" => Some(TokenKind::True),
            "false" => Some(TokenKind::False),
            "None" => Some(TokenKind::None),
            "import" => Some(TokenKind::Import),
            "plugin" => Some(TokenKind::Plugin),
            "config" => Some(TokenKind::Config),
            "and" => Some(TokenKind::And),
            "or" => Some(TokenKind::Or),
            "not" => Some(TokenKind::Bang),
            _ => None,
        }
    }
}

