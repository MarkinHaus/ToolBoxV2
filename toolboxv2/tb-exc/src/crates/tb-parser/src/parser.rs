use tb_core::*;
use std::sync::Arc;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    source: Arc<String>,  // NEW: Original source code for extracting plugin bodies
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            source: Arc::new(String::new()),
        }
    }

    pub fn new_with_source(tokens: Vec<Token>, source: String) -> Self {
        Self {
            tokens,
            pos: 0,
            source: Arc::new(source),
        }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }

        Ok(Program { statements })
    }

    fn parse_statement(&mut self) -> Result<Option<Statement>> {
        match &self.current().kind {
            TokenKind::Let => Ok(Some(self.parse_let()?)),
            TokenKind::Fn => Ok(Some(self.parse_function()?)),
            TokenKind::If => Ok(Some(self.parse_if()?)),
            TokenKind::For => Ok(Some(self.parse_for()?)),
            TokenKind::While => Ok(Some(self.parse_while()?)),
            TokenKind::Match => Ok(Some(self.parse_match()?)),
            TokenKind::Return => Ok(Some(self.parse_return()?)),
            TokenKind::Break => {
                let span = self.current().span;
                self.advance();
                Ok(Some(Statement::Break { span }))
            }
            TokenKind::Continue => {
                let span = self.current().span;
                self.advance();
                Ok(Some(Statement::Continue { span }))
            }
            TokenKind::At => Ok(Some(self.parse_special_block()?)),

            // Check for variable reassignment
            TokenKind::Ident(_) => {
                if self.peek_ahead(1).map(|t| &t.kind) == Some(&TokenKind::Eq) {
                    Ok(Some(self.parse_assignment()?))
                } else {
                    let expr = self.parse_expression()?;
                    let span = *expr.span();
                    Ok(Some(Statement::Expression { expr, span }))
                }
            }

            TokenKind::Eof => Ok(None),
            _ => {
                let expr = self.parse_expression()?;
                let span = *expr.span();
                Ok(Some(Statement::Expression { expr, span }))
            }
        }
    }

    fn parse_let(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::Let)?;

        let name = self.expect_ident()?;

        let type_annotation = if self.check(&TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::Eq)?;
        let value = self.parse_expression()?;

        let span = start_span.merge(value.span());

        Ok(Statement::Let {
            name,
            type_annotation,
            value,
            span,
        })
    }

    fn parse_function(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::Fn)?;

        let name = self.expect_ident()?;

        self.expect(TokenKind::LParen)?;
        let params = self.parse_parameter_list()?;
        self.expect(TokenKind::RParen)?;

        let return_type = if self.check(&TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_block()?;
        let end_span = self.expect(TokenKind::RBrace)?;

        Ok(Statement::Function {
            name,
            params,
            return_type,
            body,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_if(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::If)?;

        let condition = self.parse_expression()?;

        self.expect(TokenKind::LBrace)?;
        let then_block = self.parse_block()?;
        let mut end_span = self.expect(TokenKind::RBrace)?;

        let else_block = if self.check(&TokenKind::Else) {
            self.advance();
            self.expect(TokenKind::LBrace)?;
            let block = self.parse_block()?;
            end_span = self.expect(TokenKind::RBrace)?;
            Some(block)
        } else {
            None
        };

        Ok(Statement::If {
            condition,
            then_block,
            else_block,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_for(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::For)?;

        let variable = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        let iterable = self.parse_expression()?;

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_block()?;
        let end_span = self.expect(TokenKind::RBrace)?;

        Ok(Statement::For {
            variable,
            iterable,
            body,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_while(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::While)?;

        let condition = self.parse_expression()?;

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_block()?;
        let end_span = self.expect(TokenKind::RBrace)?;

        Ok(Statement::While {
            condition,
            body,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_match(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::Match)?;

        let value = self.parse_expression()?;

        self.expect(TokenKind::LBrace)?;
        let mut arms = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let pattern = self.parse_pattern()?;
            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expression()?;

            // Optional comma
            if self.check(&TokenKind::Comma) {
                self.advance();
            }

            arms.push(MatchArm { pattern, body });
        }

        let end_span = self.expect(TokenKind::RBrace)?;

        Ok(Statement::Match {
            value,
            arms,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_return(&mut self) -> Result<Statement> {
        let span = self.current().span;
        self.expect(TokenKind::Return)?;

        let value = if self.check(&TokenKind::RBrace) || self.is_at_end() {
            None
        } else {
            Some(self.parse_expression()?)
        };

        Ok(Statement::Return { value, span })
    }

    fn parse_special_block(&mut self) -> Result<Statement> {
        self.expect(TokenKind::At)?;

        match &self.current().kind {
            TokenKind::Config => self.parse_config_block(),
            TokenKind::Import => self.parse_import_block(),
            TokenKind::Plugin => self.parse_plugin_block(),
            _ => Err(TBError::SyntaxError {
                location: format!("{}:{}", self.current().span.line, self.current().span.column),
                message: "Expected config, import, or plugin after @".to_string(),
            }),
        }
    }

    fn parse_config_block(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::Config)?;
        self.expect(TokenKind::LBrace)?;

        let mut entries = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let key = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_config_value()?;

            // Optional comma
            if self.check(&TokenKind::Comma) {
                self.advance();
            }

            entries.push(ConfigEntry { key, value });
        }

        let end_span = self.expect(TokenKind::RBrace)?;

        Ok(Statement::Config {
            entries,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_import_block(&mut self) -> Result<Statement> {
        let start_span = self.current().span;
        self.expect(TokenKind::Import)?;
        self.expect(TokenKind::LBrace)?;

        let mut items = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let path = if let TokenKind::String(s) = &self.current().kind {
                let s = Arc::clone(s);
                self.advance();
                s
            } else {
                return Err(TBError::SyntaxError {
                    location: format!("{}:{}", self.current().span.line, self.current().span.column),
                    message: "Expected string path in import".to_string(),
                });
            };

            let alias = if self.check(&TokenKind::Ident(Arc::new("as".to_string()))) {
                self.advance();
                Some(self.expect_ident()?)
            } else {
                None
            };

            items.push(ImportItem {
                path,
                alias,
                condition: None,
            });

            // Optional comma after each import item
            if self.check(&TokenKind::Comma) {
                self.advance();
            }
        }

        let end_span = self.expect(TokenKind::RBrace)?;

        Ok(Statement::Import {
            items,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_plugin_block(&mut self) -> Result<Statement> {
        tb_debug!("Parsing plugin block");
        let start_span = self.current().span;
        self.expect(TokenKind::Plugin)?;
        self.expect(TokenKind::LBrace)?;

        let mut definitions = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            tb_debug!("Parsing plugin definition, current token: {:?}", self.current().kind);

            // Parse language: python, javascript, go, rust
            let language = match &self.current().kind {
                TokenKind::Ident(lang) => {
                    let lang_str = lang.as_str();
                    let plugin_lang = match lang_str {
                        "python" => PluginLanguage::Python,
                        "javascript" => PluginLanguage::JavaScript,
                        "go" => PluginLanguage::Go,
                        "rust" => PluginLanguage::Rust,
                        _ => return Err(TBError::SyntaxError {
                            location: format!("{}:{}", self.current().span.line, self.current().span.column),
                            message: format!("Unknown plugin language: {}", lang_str),
                        }),
                    };
                    self.advance();
                    plugin_lang
                }
                _ => return Err(TBError::SyntaxError {
                    location: format!("{}:{}", self.current().span.line, self.current().span.column),
                    message: "Expected plugin language (python, javascript, go, rust)".to_string(),
                }),
            };

            // Parse name (string)
            let name = if let TokenKind::String(s) = &self.current().kind {
                let n = Arc::clone(s);
                self.advance();
                n
            } else {
                return Err(TBError::SyntaxError {
                    location: format!("{}:{}", self.current().span.line, self.current().span.column),
                    message: "Expected plugin name (string)".to_string(),
                });
            };

            // Expect opening brace for plugin body
            self.expect(TokenKind::LBrace)?;

            // Parse plugin body (mode, requires, source code)
            let mut mode = PluginMode::Jit;
            let mut requires = Vec::new();
            let mut source_code = String::new();
            let mut has_file = false;
            let mut file_path = Arc::new(String::new());

            while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
                // Check for mode: "jit" or mode: "compile"
                if let TokenKind::Ident(key) = &self.current().kind {
                    let key_str = key.as_str().to_string();

                    if key_str == "mode" {
                        self.advance();
                        self.expect(TokenKind::Colon)?;

                        if let TokenKind::String(m) = &self.current().kind {
                            mode = if m.as_str() == "compile" {
                                PluginMode::Compile
                            } else {
                                PluginMode::Jit
                            };
                            self.advance();
                        }

                        if self.check(&TokenKind::Comma) {
                            self.advance();
                        }
                        continue;
                    }

                    if key_str == "requires" {
                        self.advance();
                        self.expect(TokenKind::Colon)?;
                        self.expect(TokenKind::LBracket)?;

                        while !self.check(&TokenKind::RBracket) && !self.is_at_end() {
                            if let TokenKind::String(s) = &self.current().kind {
                                requires.push(Arc::clone(s));
                                self.advance();
                                if self.check(&TokenKind::Comma) {
                                    self.advance();
                                }
                            }
                        }

                        self.expect(TokenKind::RBracket)?;

                        if self.check(&TokenKind::Comma) {
                            self.advance();
                        }
                        continue;
                    }

                    if key_str == "file" {
                        self.advance();
                        self.expect(TokenKind::Colon)?;

                        if let TokenKind::String(f) = &self.current().kind {
                            file_path = Arc::clone(f);
                            has_file = true;
                            self.advance();
                        }

                        if self.check(&TokenKind::Comma) {
                            self.advance();
                        }
                        continue;
                    }
                }

                // Everything else is inline source code
                // Extract raw source from original text using span information
                if !has_file {
                    // FIX: Find the start of the line, not just the token start
                    // This preserves the indentation of the first line
                    let token_start = self.current().span.start;
                    let mut source_start = token_start;

                    // Walk backwards to find the start of the line (after the last newline)
                    while source_start > 0 {
                        let ch = self.source.as_bytes()[source_start - 1] as char;
                        if ch == '\n' || ch == '\r' {
                            break;
                        }
                        source_start -= 1;
                    }

                    // Find the closing brace by counting braces
                    let mut brace_depth = 0;
                    let mut source_end = token_start;

                    while !self.is_at_end() {
                        if self.check(&TokenKind::LBrace) {
                            brace_depth += 1;
                        } else if self.check(&TokenKind::RBrace) {
                            if brace_depth == 0 {
                                // This is the closing brace of the plugin definition
                                source_end = self.current().span.start;
                                break;
                            }
                            brace_depth -= 1;
                        }
                        self.advance();
                    }

                    // Extract source code from original text
                    if source_start < self.source.len() && source_end <= self.source.len() {
                        let raw_source = &self.source[source_start..source_end];

                        // FIX: Dedent while preserving relative indentation
                        source_code = dedent_source(raw_source);
                    }
                }
                break;
            }

            // If no file specified, the remaining content is inline source
            let source = if has_file {
                PluginSource::File(file_path)
            } else {
                PluginSource::Inline(Arc::new(source_code))
            };

            self.expect(TokenKind::RBrace)?;

            definitions.push(PluginDefinition {
                language,
                name,
                mode,
                requires,
                source,
            });

            tb_debug!("Parsed plugin definition: {:?}", definitions.last());

            // Optional comma
            if self.check(&TokenKind::Comma) {
                self.advance();
            }
        }

        let end_span = self.expect(TokenKind::RBrace)?;

        tb_debug!("Plugin block parsed with {} definitions", definitions.len());

        Ok(Statement::Plugin {
            definitions,
            span: start_span.merge(&end_span),
        })
    }

    fn parse_config_value(&mut self) -> Result<ConfigValue> {
        match &self.current().kind {
            TokenKind::True => {
                self.advance();
                Ok(ConfigValue::Bool(true))
            }
            TokenKind::False => {
                self.advance();
                Ok(ConfigValue::Bool(false))
            }
            TokenKind::Integer(i) => {
                let val = *i;
                self.advance();
                Ok(ConfigValue::Int(val))
            }
            TokenKind::String(s) => {
                let val = Arc::clone(s);
                self.advance();
                Ok(ConfigValue::String(val))
            }
            TokenKind::LBrace => {
                self.advance();
                let mut entries = Vec::new();
                while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
                    let key = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_config_value()?;

                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }

                    entries.push(ConfigEntry { key, value });
                }
                self.expect(TokenKind::RBrace)?;
                Ok(ConfigValue::Dict(entries))
            }
            _ => Err(TBError::SyntaxError {
                location: format!("{}:{}", self.current().span.line, self.current().span.column),
                message: "Expected config value".to_string(),
            }),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>> {
        let mut statements = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }

        Ok(statements)
    }

    fn parse_parameter_list(&mut self) -> Result<Vec<Parameter>> {
        let mut params = Vec::new();

        while !self.check(&TokenKind::RParen) && !self.is_at_end() {
            let name = self.expect_ident()?;

            let type_annotation = if self.check(&TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            params.push(Parameter {
                name,
                type_annotation,
            });

            if !self.check(&TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
            }
        }

        Ok(params)
    }

    fn parse_type(&mut self) -> Result<Type> {
        match &self.current().kind {
            TokenKind::Ident(name) => {
                let type_name = name.as_str().to_string();
                self.advance();

                match type_name.as_str() {
                    "int" => Ok(Type::Int),
                    "float" => Ok(Type::Float),
                    "bool" => Ok(Type::Bool),
                    "string" => Ok(Type::String),
                    _ => Ok(Type::Generic(Arc::new(type_name))),
                }
            }
            _ => Err(TBError::SyntaxError {
                location: format!("{}:{}", self.current().span.line, self.current().span.column),
                message: "Expected type".to_string(),
            }),
        }
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expression> {
        let mut left = self.parse_and()?;

        while self.check(&TokenKind::Or) {
            let _op_span = self.current().span;
            self.advance();
            let right = self.parse_and()?;
            let span = left.span().merge(right.span());

            left = Expression::Binary {
                op: BinaryOp::Or,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression> {
        let mut left = self.parse_equality()?;

        while self.check(&TokenKind::And) {
            self.advance();
            let right = self.parse_equality()?;
            let span = left.span().merge(right.span());

            left = Expression::Binary {
                op: BinaryOp::And,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expression> {
        let mut left = self.parse_comparison()?;

        while let Some(op) = self.match_binary_op(&[TokenKind::EqEq, TokenKind::BangEq]) {
            self.advance();
            let right = self.parse_comparison()?;
            let span = left.span().merge(right.span());

            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expression> {
        let mut left = self.parse_term()?;

        while let Some(op) = self.match_binary_op(&[
            TokenKind::Lt,
            TokenKind::Gt,
            TokenKind::LtEq,
            TokenKind::GtEq,
        ]) {
            self.advance();
            let right = self.parse_term()?;
            let span = left.span().merge(right.span());

            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expression> {
        let mut left = self.parse_factor()?;

        while let Some(op) = self.match_binary_op(&[TokenKind::Plus, TokenKind::Minus]) {
            self.advance();
            let right = self.parse_factor()?;
            let span = left.span().merge(right.span());

            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expression> {
        let mut left = self.parse_unary()?;

        while let Some(op) = self.match_binary_op(&[
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
        ]) {
            self.advance();
            let right = self.parse_unary()?;
            let span = left.span().merge(right.span());

            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression> {
        if let Some(op) = self.match_unary_op(&[TokenKind::Bang, TokenKind::Minus]) {
            let op_span = self.current().span;
            self.advance();
            let operand = self.parse_unary()?;
            let span = op_span.merge(operand.span());

            Ok(Expression::Unary {
                op,
                operand: Box::new(operand),
                span,
            })
        } else {
            self.parse_postfix()
        }
    }

    fn parse_postfix(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary()?;

        loop {
            match &self.current().kind {
                TokenKind::LParen => {
                    self.advance();
                    let args = self.parse_argument_list()?;
                    let end_span = self.expect(TokenKind::RParen)?;
                    let span = expr.span().merge(&end_span);

                    expr = Expression::Call {
                        callee: Box::new(expr),
                        args,
                        span,
                    };
                }
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expression()?;
                    let end_span = self.expect(TokenKind::RBracket)?;
                    let span = expr.span().merge(&end_span);

                    expr = Expression::Index {
                        object: Box::new(expr),
                        index: Box::new(index),
                        span,
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    let member = self.expect_ident()?;
                    let span = expr.span().merge(&self.previous().span);

                    expr = Expression::Member {
                        object: Box::new(expr),
                        member,
                        span,
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression> {
        let span = self.current().span;

        match &self.current().kind {
            TokenKind::True => {
                self.advance();
                Ok(Expression::Literal(Literal::Bool(true), span))
            }
            TokenKind::False => {
                self.advance();
                Ok(Expression::Literal(Literal::Bool(false), span))
            }
            TokenKind::Integer(i) => {
                let val = *i;
                self.advance();
                Ok(Expression::Literal(Literal::Int(val), span))
            }
            TokenKind::Float(f) => {
                let val = *f;
                self.advance();
                Ok(Expression::Literal(Literal::Float(val), span))
            }
            TokenKind::String(s) => {
                let s = Arc::clone(s);
                self.advance();
                Ok(Expression::Literal(Literal::String(s), span))
            }
            TokenKind::Ident(name) => {
                let name = Arc::clone(name);
                self.advance();
                Ok(Expression::Ident(name, span))
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => {
                self.advance();
                let elements = self.parse_expression_list(TokenKind::RBracket)?;
                let end_span = self.expect(TokenKind::RBracket)?;
                Ok(Expression::List {
                    elements,
                    span: span.merge(&end_span),
                })
            }
            TokenKind::LBrace => {
                self.advance();
                let mut entries = Vec::new();

                while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
                    let key = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_expression()?;

                    entries.push((key, value));

                    if !self.check(&TokenKind::RBrace) {
                        self.expect(TokenKind::Comma)?;
                    }
                }

                let end_span = self.expect(TokenKind::RBrace)?;
                Ok(Expression::Dict {
                    entries,
                    span: span.merge(&end_span),
                })
            }

            TokenKind::Match => {
                self.advance();
                let value = self.parse_expression()?;

                self.expect(TokenKind::LBrace)?;
                let mut arms = Vec::new();

                while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
                    let pattern = self.parse_pattern()?;
                    self.expect(TokenKind::FatArrow)?;
                    let body = self.parse_expression()?;

                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }

                    arms.push(MatchArm { pattern, body });
                }

                let end_span = self.expect(TokenKind::RBrace)?;

                Ok(Expression::Match {
                    value: Box::new(value),
                    arms,
                    span: span.merge(&end_span),
                })
            }

            _ => Err(TBError::SyntaxError {
                location: format!("{}:{}", span.line, span.column),
                message: format!("Unexpected token: {:?}", self.current().kind),
            }),
        }
    }

    fn parse_argument_list(&mut self) -> Result<Vec<Expression>> {
        self.parse_expression_list(TokenKind::RParen)
    }

    fn parse_expression_list(&mut self, terminator: TokenKind) -> Result<Vec<Expression>> {
        let mut exprs = Vec::new();

        while !self.check(&terminator) && !self.is_at_end() {
            exprs.push(self.parse_expression()?);

            if !self.check(&terminator) {
                self.expect(TokenKind::Comma)?;
            }
        }

        Ok(exprs)
    }

    fn match_binary_op(&self, kinds: &[TokenKind]) -> Option<BinaryOp> {
        for kind in kinds {
            if self.check(kind) {
                return Some(match kind {
                    TokenKind::Plus => BinaryOp::Add,
                    TokenKind::Minus => BinaryOp::Sub,
                    TokenKind::Star => BinaryOp::Mul,
                    TokenKind::Slash => BinaryOp::Div,
                    TokenKind::Percent => BinaryOp::Mod,
                    TokenKind::EqEq => BinaryOp::Eq,
                    TokenKind::BangEq => BinaryOp::NotEq,
                    TokenKind::Lt => BinaryOp::Lt,
                    TokenKind::Gt => BinaryOp::Gt,
                    TokenKind::LtEq => BinaryOp::LtEq,
                    TokenKind::GtEq => BinaryOp::GtEq,
                    _ => return None,
                });
            }
        }
        None
    }

    fn match_unary_op(&self, kinds: &[TokenKind]) -> Option<UnaryOp> {
        for kind in kinds {
            if self.check(kind) {
                return Some(match kind {
                    TokenKind::Bang => UnaryOp::Not,
                    TokenKind::Minus => UnaryOp::Neg,
                    _ => return None,
                });
            }
        }
        None
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn previous(&self) -> &Token {
        &self.tokens[(self.pos - 1).max(0)]
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.pos += 1;
        }
    }

    fn check(&self, kind: &TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        std::mem::discriminant(&self.current().kind) == std::mem::discriminant(kind)
    }

    fn is_at_end(&self) -> bool {
        matches!(self.current().kind, TokenKind::Eof)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Span> {
        if self.check(&kind) {
            let span = self.current().span;
            self.advance();
            Ok(span)
        } else {
            Err(TBError::SyntaxError {
                location: format!(
                    "{}:{}",
                    self.current().span.line,
                    self.current().span.column
                ),
                message: format!("Expected {:?}, found {:?}", kind, self.current().kind),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<Arc<String>> {
        match &self.current().kind {
            TokenKind::Ident(name) => {
                let name = Arc::clone(name);
                self.advance();
                Ok(name)
            }
            _ => Err(TBError::SyntaxError {
                location: format!(
                    "{}:{}",
                    self.current().span.line,
                    self.current().span.column
                ),
                message: format!("Expected identifier, found {:?}", self.current().kind),
            }),
        }
    }

    fn parse_assignment(&mut self) -> Result<Statement> {
        let name = self.expect_ident()?;
        let start_span = self.previous().span;

        self.expect(TokenKind::Eq)?;
        let value = self.parse_expression()?;

        let span = start_span.merge(value.span());

        Ok(Statement::Assign {
            name,
            value,
            span,
        })
    }

    fn peek_ahead(&self, offset: usize) -> Option<&Token> {
        let pos = self.pos + offset;
        if pos < self.tokens.len() {
            Some(&self.tokens[pos])
        } else {
            None
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        match &self.current().kind {
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::Integer(i) => {
                let val = *i;
                self.advance();

                // Check for range operators BEFORE returning
                if self.check(&TokenKind::DotDot) || self.check(&TokenKind::DotDotEq) {
                    let inclusive = self.check(&TokenKind::DotDotEq);
                    self.advance();

                    if let TokenKind::Integer(end) = &self.current().kind {
                        let end_val = *end;
                        self.advance();
                        Ok(Pattern::Range {
                            start: val,
                            end: end_val,
                            inclusive,
                        })
                    } else {
                        Err(TBError::SyntaxError {
                            location: format!("{}:{}", self.current().span.line, self.current().span.column),
                            message: "Expected integer after range operator".to_string(),
                        })
                    }
                } else {
                    Ok(Pattern::Literal(Literal::Int(val)))
                }
            }
            TokenKind::Float(f) => {
                let val = *f;
                self.advance();
                Ok(Pattern::Literal(Literal::Float(val)))
            }
            TokenKind::String(s) => {
                let s = Arc::clone(s);
                self.advance();
                Ok(Pattern::Literal(Literal::String(s)))
            }
            TokenKind::Ident(name) => {
                if name.as_ref() == "_" {
                    self.advance();
                    Ok(Pattern::Wildcard)
                } else {
                    let name = Arc::clone(name);
                    self.advance();
                    Ok(Pattern::Ident(name))
                }
            }
            _ => Err(TBError::SyntaxError {
                location: format!(
                    "{}:{}",
                    self.current().span.line,
                    self.current().span.column
                ),
                message: format!("Expected pattern, found {:?}", self.current().kind),
            }),
        }
    }
}

/// Helper function to dedent source code while preserving relative indentation
fn dedent_source(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Find minimum indentation (excluding empty lines)
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
        .min()
        .unwrap_or(0);

    // Remove minimum indentation from all lines
    lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                "" // Keep empty lines empty
            } else {
                &line[min_indent.min(line.len())..]
            }
        })
        .collect::<Vec<&str>>()
        .join("\n")
}

