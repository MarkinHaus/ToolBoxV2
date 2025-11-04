use tb_core::{Span, StringInterner, Token, TokenKind};
use std::sync::Arc;

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
    interner: Arc<StringInterner>,
}

impl Lexer {
    pub fn new(source: &str, interner: Arc<StringInterner>) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            interner,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(self.source.len() / 4);

        loop {
            self.skip_whitespace_and_comments();

            if self.is_at_end() {
                tokens.push(Token::new(
                    TokenKind::Eof,
                    Span::new(self.pos, self.pos, self.line, self.column),
                ));
                break;
            }

            if let Some(token) = self.next_token() {
                tokens.push(token);
            }
        }

        tokens
    }

    fn next_token(&mut self) -> Option<Token> {
        let start = self.pos;
        let start_line = self.line;
        let start_column = self.column;

        let c = self.current()?;
        let kind = match c {
            '(' => {
                self.advance();
                TokenKind::LParen
            }
            ')' => {
                self.advance();
                TokenKind::RParen
            }
            '{' => {
                self.advance();
                TokenKind::LBrace
            }
            '}' => {
                self.advance();
                TokenKind::RBrace
            }
            '[' => {
                self.advance();
                TokenKind::LBracket
            }
            ']' => {
                self.advance();
                TokenKind::RBracket
            }
            ',' => {
                self.advance();
                TokenKind::Comma
            }
            ':' => {
                self.advance();
                TokenKind::Colon
            }
            ';' => {
                self.advance();
                TokenKind::Semicolon
            }
            '@' => {
                self.advance();
                TokenKind::At
            }
            '+' => {
                self.advance();
                TokenKind::Plus
            }
            '*' => {
                self.advance();
                TokenKind::Star
            }
            '/' => {
                self.advance();
                TokenKind::Slash
            }
            '%' => {
                self.advance();
                TokenKind::Percent
            }
            '-' => {
                self.advance();
                if self.current() == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '=' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    TokenKind::EqEq
                } else if self.current() == Some('>') {
                    self.advance();
                    TokenKind::FatArrow
                } else {
                    TokenKind::Eq
                }
            }
            '!' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    TokenKind::BangEq
                } else {
                    TokenKind::Bang
                }
            }
            '<' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            '.' => {
                self.advance();
                if self.current() == Some('.') {
                    self.advance();
                    if self.current() == Some('=') {
                        self.advance();
                        TokenKind::DotDotEq
                    } else {
                        TokenKind::DotDot
                    }
                } else {
                    TokenKind::Dot
                }
            }
            '"' => return Some(self.read_string()),
            '\n' => {
                self.advance();
                return self.next_token(); // Skip newlines for now
            }
            _ if c.is_ascii_digit() => return Some(self.read_number()),
            _ if c.is_alphabetic() || c == '_' => return Some(self.read_identifier()),
            _ => {
                self.advance();
                return self.next_token(); // Skip unknown characters
            }
        };

        Some(Token::new(
            kind,
            Span::new(start, self.pos, start_line, start_column),
        ))
    }

    fn read_string(&mut self) -> Token {
        let start = self.pos;
        let start_line = self.line;
        let start_column = self.column;

        self.advance(); // Skip opening quote

        let mut value = String::new();
        while let Some(c) = self.current() {
            if c == '"' {
                break;
            }
            if c == '\\' {
                self.advance();
                if let Some(escaped) = self.current() {
                    value.push(match escaped {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '"' => '"',
                        _ => escaped,
                    });
                    self.advance();
                }
            } else {
                value.push(c);
                self.advance();
            }
        }

        self.advance(); // Skip closing quote

        Token::new(
            TokenKind::String(self.interner.intern(&value)),
            Span::new(start, self.pos, start_line, start_column),
        )
    }

    fn read_number(&mut self) -> Token {
        let start = self.pos;
        let start_line = self.line;
        let start_column = self.column;

        let mut raw_value = String::new();
        let mut is_float = false;
        let mut has_comma_separator = false;
        let mut has_dot_separator = false;

        // Read all digits, dots, commas, and underscores
        while let Some(c) = self.current() {
            if c.is_ascii_digit() {
                raw_value.push(c);
                self.advance();
            } else if c == '.' {
                // Could be decimal point or thousand separator
                if let Some(next) = self.peek() {
                    if next.is_ascii_digit() {
                        raw_value.push(c);
                        has_dot_separator = true;
                        self.advance();
                    } else {
                        break; // End of number (e.g., "3.method()")
                    }
                } else {
                    break;
                }
            } else if c == ',' {
                // Could be decimal point (German) or thousand separator (English)
                if let Some(next) = self.peek() {
                    if next.is_ascii_digit() {
                        raw_value.push(c);
                        has_comma_separator = true;
                        self.advance();
                    } else {
                        break; // End of number
                    }
                } else {
                    break;
                }
            } else if c == '_' {
                // Rust-style separator - skip it
                self.advance();
            } else {
                break;
            }
        }

        // Normalize the number to standard format
        let normalized = self.normalize_number(&raw_value, has_dot_separator, has_comma_separator, &mut is_float);

        let kind = if is_float {
            TokenKind::Float(normalized.parse().unwrap_or(0.0))
        } else {
            TokenKind::Integer(normalized.parse().unwrap_or(0))
        };

        Token::new(
            kind,
            Span::new(start, self.pos, start_line, start_column),
        )
    }

    /// Normalize number from various formats to standard format
    /// Supports:
    /// - English: 3.14, 1,234.56 (comma = thousand, dot = decimal)
    /// - German: 3,14, 1.234,56 (dot = thousand, comma = decimal)
    /// - Rust: 3.14_159 (underscore = separator, ignored)
    fn normalize_number(&self, raw: &str, has_dot: bool, has_comma: bool, is_float: &mut bool) -> String {
        if !has_dot && !has_comma {
            // Simple integer
            *is_float = false;
            return raw.to_string();
        }

        // Determine format based on separators
        if has_dot && has_comma {
            // Both separators present - determine which is decimal point
            let last_dot = raw.rfind('.');
            let last_comma = raw.rfind(',');

            if let (Some(dot_pos), Some(comma_pos)) = (last_dot, last_comma) {
                if dot_pos > comma_pos {
                    // English format: 1,234.56 -> dot is decimal
                    *is_float = true;
                    raw.replace(',', "")
                } else {
                    // German format: 1.234,56 -> comma is decimal
                    *is_float = true;
                    raw.replace('.', "").replace(',', ".")
                }
            } else {
                *is_float = false;
                raw.to_string()
            }
        } else if has_comma {
            // Only comma - could be German decimal or English thousand separator
            let comma_count = raw.matches(',').count();
            let parts: Vec<&str> = raw.split(',').collect();

            if comma_count == 1 && parts.len() == 2 {
                // Single comma - check if it's decimal point (German) or thousand separator
                let after_comma = parts[1];
                if after_comma.len() <= 2 || after_comma.len() == 3 && parts[0].len() <= 3 {
                    // Likely German decimal: 3,14 or 123,45
                    *is_float = true;
                    raw.replace(',', ".")
                } else {
                    // Likely English thousand: 1,234
                    *is_float = false;
                    raw.replace(',', "")
                }
            } else {
                // Multiple commas - thousand separators (English)
                *is_float = false;
                raw.replace(',', "")
            }
        } else if has_dot {
            // Only dot - could be English decimal or German thousand separator
            let dot_count = raw.matches('.').count();
            let parts: Vec<&str> = raw.split('.').collect();

            if dot_count == 1 && parts.len() == 2 {
                // Single dot - check if it's decimal point (English) or thousand separator
                let after_dot = parts[1];
                if after_dot.len() <= 2 || after_dot.len() == 3 && parts[0].len() <= 3 {
                    // Likely English decimal: 3.14 or 123.45
                    *is_float = true;
                    raw.to_string()
                } else {
                    // Likely German thousand: 1.234
                    *is_float = false;
                    raw.replace('.', "")
                }
            } else {
                // Multiple dots - thousand separators (German)
                *is_float = false;
                raw.replace('.', "")
            }
        } else {
            *is_float = false;
            raw.to_string()
        }
    }

    fn read_identifier(&mut self) -> Token {
        let start = self.pos;
        let start_line = self.line;
        let start_column = self.column;

        let mut value = String::new();
        while let Some(c) = self.current() {
            if c.is_alphanumeric() || c == '_' {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        let kind = TokenKind::keyword(&value)
            .unwrap_or_else(|| TokenKind::Ident(self.interner.intern(&value)));

        Token::new(
            kind,
            Span::new(start, self.pos, start_line, start_column),
        )
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(c) = self.current() {
            if c.is_whitespace() {
                if c == '\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                self.pos += 1;
            } else if c == '#' {
                // Skip Python-style comment (#) until end of line
                while let Some(c) = self.current() {
                    self.pos += 1;
                    if c == '\n' {
                        self.line += 1;
                        self.column = 1;
                        break;
                    }
                }
            } else if c == '/' && self.peek() == Some('/') {
                // Skip JavaScript-style comment (//) until end of line
                self.pos += 2; // Skip //
                self.column += 2;
                while let Some(c) = self.current() {
                    self.pos += 1;
                    if c == '\n' {
                        self.line += 1;
                        self.column = 1;
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn current(&self) -> Option<char> {
        self.source.get(self.pos).copied()
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.pos + 1).copied()
    }

    fn advance(&mut self) {
        if self.pos < self.source.len() {
            self.pos += 1;
            self.column += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.source.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let interner = Arc::new(StringInterner::new(Default::default()));
        let mut lexer = Lexer::new("let x = 42", interner);
        let tokens = lexer.tokenize();

        assert_eq!(tokens.len(), 5); // let, x, =, 42, EOF
        assert!(matches!(tokens[0].kind, TokenKind::Let));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(_)));
        assert!(matches!(tokens[2].kind, TokenKind::Eq));
        assert!(matches!(tokens[3].kind, TokenKind::Integer(42)));
        assert!(matches!(tokens[4].kind, TokenKind::Eof));
    }
}

