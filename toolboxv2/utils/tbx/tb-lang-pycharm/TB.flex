// file: tb-lang-pycharm/TB.flex

package dev.tblang;

import com.intellij.lexer.FlexLexer;
import com.intellij.psi.tree.IElementType;
import static dev.tblang.psi.TBTypes.*;

%%

%class TBLexer
%implements FlexLexer
%unicode
%function advance
%type IElementType
%eof{  return;
%eof}

CRLF=\R
WHITE_SPACE=[\ \n\t\f]
COMMENT="#".*

IDENTIFIER=[a-zA-Z_][a-zA-Z0-9_]*
INTEGER=[0-9]+
FLOAT=[0-9]+\.[0-9]+
STRING=\"([^\\\"]|\\.)*\"

%%

<YYINITIAL> {
    {WHITE_SPACE}+              { return WHITE_SPACE; }
    {COMMENT}                   { return COMMENT; }

    // Keywords
    "fn"                        { return FN; }
    "let"                       { return LET; }
    "mut"                       { return MUT; }
    "if"                        { return IF; }
    "else"                      { return ELSE; }
    "match"                     { return MATCH; }
    "loop"                      { return LOOP; }
    "while"                     { return WHILE; }
    "for"                       { return FOR; }
    "in"                        { return IN; }
    "return"                    { return RETURN; }
    "break"                     { return BREAK; }
    "continue"                  { return CONTINUE; }
    "async"                     { return ASYNC; }
    "await"                     { return AWAIT; }
    "parallel"                  { return PARALLEL; }

    // Literals
    "true"                      { return TRUE; }
    "false"                     { return FALSE; }

    // Types
    "int"                       { return INT_TYPE; }
    "float"                     { return FLOAT_TYPE; }
    "bool"                      { return BOOL_TYPE; }
    "string"                    { return STRING_TYPE; }

    // Operators
    "+"                         { return PLUS; }
    "-"                         { return MINUS; }
    "*"                         { return STAR; }
    "/"                         { return SLASH; }
    "%"                         { return PERCENT; }
    "="                         { return EQ; }
    "=="                        { return EQEQ; }
    "!="                        { return NEQ; }
    "<"                         { return LT; }
    "<="                        { return LEQ; }
    ">"                         { return GT; }
    ">="                        { return GEQ; }
    "&&"                        { return ANDAND; }
    "||"                        { return OROR; }
    "|>"                        { return PIPE; }
    "->"                        { return ARROW; }
    "=>"                        { return FAT_ARROW; }
    "?"                         { return QUESTION; }

    // Delimiters
    "("                         { return LPAREN; }
    ")"                         { return RPAREN; }
    "{"                         { return LBRACE; }
    "}"                         { return RBRACE; }
    "["                         { return LBRACKET; }
    "]"                         { return RBRACKET; }
    ","                         { return COMMA; }
    ";"                         { return SEMICOLON; }
    ":"                         { return COLON; }
    "."                         { return DOT; }

    // Config
    "@config"                   { return CONFIG; }
    "@shared"                   { return SHARED; }

    // Literals
    {INTEGER}                   { return INTEGER_LITERAL; }
    {FLOAT}                     { return FLOAT_LITERAL; }
    {STRING}                    { return STRING_LITERAL; }
    {IDENTIFIER}                { return IDENTIFIER; }
}

[^]                             { return BAD_CHARACTER; }
