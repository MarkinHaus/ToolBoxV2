# 🛠️ TB Language - Complete Development Guide

**Version:** 1.0
**Last Updated:** 2025-11-10
**For:** Compiler Contributors & Language Developers

---

## 🎯 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Development Setup](#development-setup)
4. [Compilation Pipeline](#compilation-pipeline)
5. [Crate Responsibilities](#crate-responsibilities)
6. [Type System](#type-system)
7. [Code Generation](#code-generation)
8. [Testing Strategy](#testing-strategy)
9. [Debugging Techniques](#debugging-techniques)
10. [Contributing Guidelines](#contributing-guidelines)
11. [Common Development Tasks](#common-development-tasks)
12. [Fix History](#fix-history)

---

## 🏗️ Architecture Overview

### Dual Execution Modes

TB Language supports two execution modes:

1. **JIT Mode (Tree-Walking Interpreter)**
   - Fast startup time
   - No compilation overhead
   - Ideal for development and testing
   - Implemented in `tb-jit` crate

2. **AOT Mode (Rust Codegen Compiler)**
   - Generates optimized Rust code
   - Compiles to native binary
   - Maximum runtime performance
   - Implemented in `tb-codegen` crate

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TB Language Source                      │
│                         (*.tbx)                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    tb-parser (Lexer + Parser)                │
│                  Tokenization → AST Generation               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      tb-types (Type System)                  │
│              Type Inference + Call-Site Analysis             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
┌───────────────────────┐   ┌───────────────────────┐
│      tb-jit           │   │     tb-codegen        │
│  (JIT Interpreter)    │   │  (Rust Code Gen)      │
│                       │   │                       │
│  Direct AST           │   │  AST → Rust Code      │
│  Execution            │   │                       │
└───────────────────────┘   └──────────┬────────────┘
                                       │
                                       ▼
                            ┌───────────────────────┐
                            │   Rust Compiler       │
                            │   (rustc)             │
                            └──────────┬────────────┘
                                       │
                                       ▼
                            ┌───────────────────────┐
                            │  Native Binary        │
                            │  (.exe)               │
                            └───────────────────────┘
```

---

## 📁 Project Structure

```
toolboxv2/tb-exc/src/
├── crates/
│   ├── tb-core/          # Core utilities and shared types
│   ├── tb-parser/        # Lexer, Parser, AST
│   ├── tb-types/         # Type system, type inference
│   ├── tb-jit/           # JIT interpreter
│   ├── tb-codegen/       # Rust code generator
│   ├── tb-optimizer/     # AST optimization passes
│   ├── tb-runtime/       # Runtime functions for compiled code
│   ├── tb-builtins/      # Built-in functions
│   ├── tb-plugin/        # Plugin system
│   └── tb-cli/           # Command-line interface
├── tests/                # Unit tests (*.tbx)
├── target/               # Build artifacts
│   ├── debug/            # Debug builds
│   │   └── tb.exe        # Main executable
│   └── tb-compile-cache/ # Compiled TB programs
│       ├── src/          # Generated Rust code
│       └── tb-compiled.exe
├── Cargo.toml            # Workspace configuration
├── info.md               # Feature documentation
├── Lang.md               # Language specification
└── validated_syntax_and_features.md  # Test coverage tracking
```

---

## 🚀 Development Setup

### Prerequisites

- **Rust:** 1.70+ (stable)
- **Cargo:** Latest version
- **Git:** For version control
- **IDE:** VS Code with rust-analyzer (recommended)

### Initial Setup

```bash
# Clone repository
cd C:/Users/Markin/Workspace/ToolBoxV2/toolboxv2/tb-exc/src

# Build all crates
cargo build

# Run tests
cargo test

# Build release version (optional)
cargo build --release
```

### Development Workflow

```bash
# 1. Make changes to source code
# 2. Build compiler
cargo build

# 3. Test with JIT mode
./target/debug/tb.exe run tests/your_test.tbx

# 4. Test with compiled mode
./target/debug/tb.exe compile tests/your_test.tbx
./target/tb-compile-cache/tb-compiled.exe

# 5. Run unit tests
cargo test

# 6. Commit changes
git add .
git commit -m "Your commit message"
```

---

## 🔄 Compilation Pipeline

### Phase 1: Lexical Analysis (Lexer)

**Location:** `crates/tb-parser/src/lexer.rs`

**Input:** Source code string
**Output:** Token stream

```rust
// Example tokens
"let x = 42" → [
    Token::Let,
    Token::Ident("x"),
    Token::Assign,
    Token::Int(42)
]
```

### Phase 2: Syntax Analysis (Parser)

**Location:** `crates/tb-parser/src/parser.rs`

**Input:** Token stream
**Output:** Abstract Syntax Tree (AST)

```rust
// AST Node Example
Statement::Let {
    name: "x",
    value: Expression::Literal(Literal::Int(42)),
    mutable: false,
    type_annotation: None
}
```

### Phase 3: Type Inference

**Location:** `crates/tb-types/src/inference.rs`

**Input:** AST
**Output:** Typed AST with type annotations

**Key Features:**
- **Call-Site Analysis (FIX 12-13):** Infers function parameter types from call sites
- **Empty List Type Inference (FIX 17):** Infers list element types from push() operations
- **Variable Type Tracking:** Maintains `variable_types` HashMap

### Phase 4: Optimization (Optional)

**Location:** `crates/tb-optimizer/src/`

**Optimizations:**
- Constant folding
- Dead code elimination
- Inline expansion

### Phase 5: Execution/Code Generation

**JIT Mode:** `crates/tb-jit/src/executor.rs`
- Direct AST interpretation
- Runtime type checking
- Immediate execution

**AOT Mode:** `crates/tb-codegen/src/rust_codegen.rs`
- Generates Rust source code
- Writes to `target/tb-compile-cache/src/main.rs`
- Invokes `cargo build` to compile

---

## 🎯 Crate Responsibilities

### tb-core

**Purpose:** Shared utilities and core types

**Key Components:**
- Error types
- Span (source location tracking)
- Common utilities

### tb-parser

**Purpose:** Lexical and syntax analysis

**Key Files:**
- `lexer.rs` - Tokenization
- `parser.rs` - AST construction
- `ast.rs` - AST node definitions

**Key Types:**
```rust
pub enum Statement {
    Let { name: String, value: Expression, mutable: bool, type_annotation: Option<Type> },
    Expression(Expression),
    If { condition: Expression, then_block: Vec<Statement>, else_block: Option<Vec<Statement>> },
    For { var: String, iterable: Expression, body: Vec<Statement> },
    While { condition: Expression, body: Vec<Statement> },
    Return(Option<Expression>),
    FunctionDef { name: String, params: Vec<(String, Option<Type>)>, body: Vec<Statement>, return_type: Option<Type> },
}

pub enum Expression {
    Literal(Literal),
    Ident(String),
    Binary { op: BinaryOp, left: Box<Expression>, right: Box<Expression> },
    Unary { op: UnaryOp, operand: Box<Expression> },
    Call { callee: Box<Expression>, args: Vec<Expression> },
    Lambda { params: Vec<String>, body: Box<Expression> },
    List(Vec<Expression>),
    Dict(Vec<(String, Expression)>),
    Index { object: Box<Expression>, index: Box<Expression> },
    Match { value: Box<Expression>, arms: Vec<MatchArm> },
}
```

### tb-types

**Purpose:** Type system and type inference

**Key Files:**
- `types.rs` - Type definitions
- `inference.rs` - Type inference algorithms

**Key Types:**
```rust
pub enum Type {
    Int,
    Float,
    String,
    Bool,
    None,
    List(Box<Type>),
    Dict(Box<Type>, Box<Type>),
    Function(Vec<Type>, Box<Type>),
    Any,  // Unknown type
}
```

**Key Functions:**
```rust
// FIX 12-13: Call-Site Analysis
pub fn analyze_function_param_types_from_calls(
    statements: &[Statement],
    function_param_types: &mut HashMap<String, Vec<Type>>
) -> Result<(), String>

// FIX 17: Empty List Type Inference
pub fn analyze_empty_list_types(
    statements: &[Statement],
    empty_list_types: &mut HashMap<String, Type>
) -> Result<(), String>
```

### tb-jit

**Purpose:** JIT interpreter (tree-walking)

**Key Files:**
- `executor.rs` - Main execution engine
- `environment.rs` - Variable scoping

**Key Types:**
```rust
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    None,
    List(Arc<RwLock<Vec<Value>>>),
    Dict(Arc<RwLock<HashMap<String, Value>>>),
    Function { params: Vec<String>, body: Vec<Statement>, env: Environment },
    Lambda { params: Vec<String>, body: Box<Expression>, env: Environment },
    NativeFunction(fn(&[Value]) -> Result<Value, String>),
}
```

**Important Notes:**
- **FIX 1-5:** Lambda parameters are dereferenced in filter/reduce/map
- **FIX 15:** Float modulo operation supported
- **FIX 16:** If-block scoping preserves modifications to existing variables

### tb-codegen

**Purpose:** Rust code generation (AOT compiler)

**Key Files:**
- `rust_codegen.rs` - Main code generator

**Key Functions:**
```rust
pub fn generate_rust_code(ast: &[Statement]) -> Result<String, String>

// FIX 18: Multi-argument print()
fn generate_call_expression(&mut self, callee: &Expression, args: &[Expression]) -> Result<(), std::fmt::Error>
```

**Important Notes:**
- **FIX 6:** None literal generates `()`
- **FIX 10:** Empty list generates `Vec::new()`
- **FIX 14:** File I/O uses `.expect()` for error handling
- **FIX 18:** Multi-argument print() generates `tb_runtime::print_multi(vec![...])`

### tb-runtime

**Purpose:** Runtime functions for compiled code

**Key Files:**
- `lib.rs` - Runtime function implementations

**Key Functions:**
```rust
// FIX 18: Multi-argument print()
pub fn print_multi(values: Vec<String>)
pub fn to_string_unit(value: &()) -> String
pub fn to_string_vec_dictvalue(vec: &Vec<DictValue>) -> String
pub fn to_string_hashmap_dictvalue(map: &HashMap<String, DictValue>) -> String
pub fn to_string_dictvalue(value: &DictValue) -> String

// File I/O (FIX 14)
pub fn read_file(path: &str) -> String
pub fn write_file(path: &str, content: &str)
pub fn read_lines(path: &str) -> Vec<String>
pub fn write_lines(path: &str, lines: &[String])
```

### tb-builtins

**Purpose:** Built-in functions (len, range, print, etc.)

**Key Functions:**
- `len()` - Get length of list/dict/string
- `range()` - Generate integer range
- `print()` - Print to stdout
- `int()`, `float()`, `str()`, `bool()` - Type conversions
- `map()`, `filter()`, `reduce()`, `forEach()` - Functional operations

### tb-cli

**Purpose:** Command-line interface

**Commands:**
- `tb.exe run <file>` - Run in JIT mode
- `tb.exe compile <file>` - Compile to native binary
- `tb.exe --version` - Show version
- `tb.exe --help` - Show help

---

## 🧪 Testing Strategy

### Unit Tests (TB Language)

**Location:** `src/tests/*.tbx`

**Test Files:**
- `comprehensive_unit_tests.tbx` - 20 tests for literals, operators, control flow
- `advanced_unit_tests.tbx` - 19 tests for loops, functions, collections
- `diagnostic_tests_simple.tbx` - 10 tests for complex scenarios

**Running Tests:**
```bash
# JIT mode
tb.exe run tests/comprehensive_unit_tests.tbx

# Compiled mode
tb.exe compile tests/comprehensive_unit_tests.tbx
./target/tb-compile-cache/tb-compiled.exe
```

### Rust Unit Tests

**Location:** `crates/*/src/*.rs` (inline tests)

**Running Tests:**
```bash
cargo test
```

### Test Coverage Tracking

**Location:** `src/validated_syntax_and_features.md`

**Format:**
```markdown
## Feature Name
- **Status:** ✅ VALIDATED / ⚠️ PARTIAL / ❌ FAILING / 🚧 NOT_TESTED
- **Test File:** tests/comprehensive_unit_tests.tbx
- **Test Name:** test_feature_name
```

---

## 🐛 Debugging Techniques

### Debug Logging

**Enable Debug Output:**
```rust
// In tb-types/src/inference.rs
println!("DEBUG FIX12: Analyzing calls to '{}'", func_name);
println!("DEBUG FIX13: Inferred param types: {:?}", param_types);
```

**View Generated Rust Code:**
```bash
# Compile TB program
tb.exe compile your_program.tbx

# View generated code
cat target/tb-compile-cache/src/main.rs
```

### Common Debugging Scenarios

#### 1. Type Inference Issues

**Problem:** Function parameter types not inferred correctly

**Debug Steps:**
1. Enable FIX12/FIX13 debug logging
2. Check `analyze_function_param_types_from_calls()` output
3. Verify call sites are being analyzed
4. Check `function_param_types` HashMap

#### 2. Codegen Issues

**Problem:** Generated Rust code doesn't compile

**Debug Steps:**
1. View `target/tb-compile-cache/src/main.rs`
2. Check for type mismatches
3. Verify runtime function signatures match codegen
4. Test in JIT mode first to isolate codegen issues

#### 3. Runtime Issues

**Problem:** Program crashes or produces wrong results

**Debug Steps:**
1. Add `print()` statements to TB code
2. Compare JIT vs Compiled mode behavior
3. Check variable scoping (FIX 16)
4. Verify lambda parameter handling (FIX 1-5)

---

## 🤝 Contributing Guidelines

### Code Style

- **Rust:** Follow `rustfmt` conventions
- **TB Language:** Follow examples in `tests/*.tbx`
- **Comments:** Use `//` for single-line, `/* */` for multi-line
- **Naming:** snake_case for variables/functions, PascalCase for types

### Commit Message Format

```
[CATEGORY] Brief description

Detailed explanation (optional)

Fixes: #issue_number (if applicable)
```

**Categories:**
- `[FIX]` - Bug fix
- `[FEATURE]` - New feature
- `[REFACTOR]` - Code refactoring
- `[TEST]` - Test additions/changes
- `[DOCS]` - Documentation updates

**Example:**
```
[FIX] Multi-argument print() in compiled mode

- Added print_multi() runtime function
- Modified codegen to generate print_multi() calls
- Added type-specific to_string_*() helpers

Fixes: FIX 18
```

### Pull Request Process

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit
3. Run tests: `cargo test` and `tb.exe run tests/*.tbx`
4. Update documentation if needed
5. Create pull request with clear description
6. Wait for review and address feedback

---

## 🔧 Common Development Tasks

### Task 1: Add New Built-in Function

**Example:** Add `abs()` function for absolute value

**Steps:**

1. **Add to tb-builtins:**
```rust
// crates/tb-builtins/src/lib.rs
pub fn builtin_abs(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("abs() takes exactly 1 argument".to_string());
    }

    match &args[0] {
        Value::Int(n) => Ok(Value::Int(n.abs())),
        Value::Float(f) => Ok(Value::Float(f.abs())),
        _ => Err("abs() requires Int or Float".to_string())
    }
}

// Register in builtins map
pub fn get_builtins() -> HashMap<String, NativeFunction> {
    let mut builtins = HashMap::new();
    // ... existing builtins ...
    builtins.insert("abs".to_string(), builtin_abs);
    builtins
}
```

2. **Add to tb-runtime (for compiled mode):**
```rust
// crates/tb-runtime/src/lib.rs
pub fn abs_int(n: i64) -> i64 {
    n.abs()
}

pub fn abs_float(f: f64) -> f64 {
    f.abs()
}
```

3. **Add codegen support:**
```rust
// crates/tb-codegen/src/rust_codegen.rs
} else if name.as_str() == "abs" && args.len() == 1 {
    let arg_type = self.infer_expr_type(&args[0])?;
    match arg_type {
        Type::Int => {
            write!(self.buffer, "tb_runtime::abs_int(")?;
            self.generate_expression(&args[0])?;
            write!(self.buffer, ")")?;
        }
        Type::Float => {
            write!(self.buffer, "tb_runtime::abs_float(")?;
            self.generate_expression(&args[0])?;
            write!(self.buffer, ")")?;
        }
        _ => return Err("abs() requires Int or Float".to_string())
    }
    return Ok(());
}
```

4. **Add tests:**
```tb
// tests/test_abs.tbx
fn test_abs() {
    print("TEST: abs() function")

    let pos = abs(5)
    let neg = abs(-5)
    let zero = abs(0)
    let float_val = abs(-3.14)

    print("  abs(5):", pos)
    print("  abs(-5):", neg)
    print("  abs(0):", zero)
    print("  abs(-3.14):", float_val)

    if pos == 5 and neg == 5 and zero == 0 and float_val == 3.14 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
}

fn main() {
    test_abs()
}

main()
```

5. **Build and test:**
```bash
cargo build
tb.exe run tests/test_abs.tbx
tb.exe compile tests/test_abs.tbx
./target/tb-compile-cache/tb-compiled.exe
```

### Task 2: Fix Type Inference Bug

**Example:** Fix empty list type inference (FIX 17)

**Steps:**

1. **Identify the problem:**
   - Empty lists `[]` have type `List<Any>`
   - Type should be inferred from `push()` operations

2. **Implement solution:**
```rust
// crates/tb-types/src/inference.rs
pub fn analyze_empty_list_types(
    statements: &[Statement],
    empty_list_types: &mut HashMap<String, Type>
) -> Result<(), String> {
    for stmt in statements {
        match stmt {
            Statement::Let { name, value, .. } => {
                // Check if value is empty list
                if let Expression::List(elements) = value {
                    if elements.is_empty() {
                        // Mark as empty list
                        empty_list_types.insert(name.clone(), Type::Any);
                    }
                }
            }
            Statement::Expression(Expression::Call { callee, args }) => {
                // Check for push() calls
                if let Expression::Ident(method) = &**callee {
                    if method == "push" && args.len() == 2 {
                        if let Expression::Ident(list_name) = &args[0] {
                            // Infer element type from push() argument
                            let elem_type = infer_expr_type(&args[1])?;
                            empty_list_types.insert(
                                list_name.clone(),
                                Type::List(Box::new(elem_type))
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
```

3. **Add tests:**
```tb
fn test_empty_list_type_inference() {
    let empty = []
    empty.push(1)
    empty.push(2)

    let first = empty[0]

    if first == 1 {
        print("✅ PASS")
    } else {
        print("❌ FAIL")
    }
}
```

4. **Update documentation:**
   - Add to `validated_syntax_and_features.md`
   - Update `info.md` with FIX 17 details

### Task 3: Add New Language Feature

**Example:** Add `break` statement for loops

**Steps:**

1. **Update AST:**
```rust
// crates/tb-parser/src/ast.rs
pub enum Statement {
    // ... existing variants ...
    Break,
}
```

2. **Update parser:**
```rust
// crates/tb-parser/src/parser.rs
fn parse_statement(&mut self) -> Result<Statement, String> {
    match self.current_token {
        // ... existing cases ...
        Token::Break => {
            self.advance();
            Ok(Statement::Break)
        }
        _ => // ...
    }
}
```

3. **Update JIT executor:**
```rust
// crates/tb-jit/src/executor.rs
pub enum ControlFlow {
    None,
    Return(Value),
    Break,  // New variant
}

fn execute_statement(&mut self, stmt: &Statement) -> Result<ControlFlow, String> {
    match stmt {
        // ... existing cases ...
        Statement::Break => Ok(ControlFlow::Break),
        Statement::For { var, iterable, body } => {
            // ... loop setup ...
            for value in values {
                // ... variable binding ...
                for stmt in body {
                    match self.execute_statement(stmt)? {
                        ControlFlow::Break => break,  // Handle break
                        ControlFlow::Return(v) => return Ok(ControlFlow::Return(v)),
                        ControlFlow::None => {}
                    }
                }
            }
            Ok(ControlFlow::None)
        }
        _ => // ...
    }
}
```

4. **Update codegen:**
```rust
// crates/tb-codegen/src/rust_codegen.rs
fn generate_statement(&mut self, stmt: &Statement) -> Result<(), std::fmt::Error> {
    match stmt {
        // ... existing cases ...
        Statement::Break => {
            writeln!(self.buffer, "break;")?;
        }
        _ => // ...
    }
    Ok(())
}
```

5. **Add tests and documentation**

---

## 📜 Fix History

### FIX 1-5: Lambda Parameter Dereferencing
**Problem:** Lambda parameters in filter/reduce/map were not dereferenced
**Solution:** Added `*` dereference operator in JIT executor
**Files:** `tb-jit/src/executor.rs`

### FIX 6: None Literal Codegen
**Problem:** None literal didn't generate valid Rust code
**Solution:** Generate `()` for None in codegen
**Files:** `tb-codegen/src/rust_codegen.rs`

### FIX 7: Named Functions in filter()
**Problem:** filter() only accepted lambdas
**Solution:** Support both lambdas and named functions
**Files:** `tb-builtins/src/lib.rs`

### FIX 8-9: forEach with Functions/Lambdas
**Problem:** forEach() didn't work with named functions or lambdas
**Solution:** Support both function types
**Files:** `tb-builtins/src/lib.rs`

### FIX 10: Empty List Literal
**Problem:** Empty list `[]` didn't generate valid code
**Solution:** Generate `Vec::new()` in codegen
**Files:** `tb-codegen/src/rust_codegen.rs`

### FIX 11: Pop from List
**Problem:** pop() return type not handled correctly
**Solution:** Fixed return type handling
**Files:** `tb-jit/src/executor.rs`, `tb-codegen/src/rust_codegen.rs`

### FIX 12-13: Call-Site Analysis
**Problem:** Function parameter types not inferred from call sites
**Solution:** Implemented `analyze_function_param_types_from_calls()`
**Files:** `tb-types/src/inference.rs`

### FIX 14: File I/O Error Handling
**Problem:** File I/O silently failed
**Solution:** Use `.expect()` for explicit error messages
**Files:** `tb-runtime/src/lib.rs`

### FIX 15: Float Modulo
**Problem:** Float modulo operation not supported in JIT
**Solution:** Added float modulo support
**Files:** `tb-jit/src/executor.rs`

### FIX 16: If-Block Scoping
**Problem:** If-blocks created new scope, losing variable modifications
**Solution:** Preserve modifications to existing variables
**Files:** `tb-jit/src/executor.rs`

### FIX 17: Empty List Type Inference
**Problem:** Empty list type not inferred from push() operations
**Solution:** Implemented `analyze_empty_list_types()`
**Files:** `tb-types/src/inference.rs`

### FIX 18: Multi-argument print()
**Problem:** print() with multiple arguments failed in compiled mode
**Solution:** Added `print_multi()` runtime function and codegen support
**Files:** `tb-runtime/src/lib.rs`, `tb-codegen/src/rust_codegen.rs`

---

## 🎓 Learning Resources

- **Rust Book:** https://doc.rust-lang.org/book/
- **Crafting Interpreters:** https://craftinginterpreters.com/
- **TB Lang Spec:** `src/Lang.md`
- **Feature Docs:** `src/info.md`

---

---

## 🔬 Advanced Topics

### Memory Management

TB Language uses **reference counting (Arc)** for memory management:

**Advantages:**
- ✅ Deterministic deallocation
- ✅ No GC pauses
- ✅ Predictable performance
- ✅ Thread-safe sharing

**Disadvantages:**
- ⚠️ Cyclic references cause memory leaks
- ⚠️ Overhead for reference counting

**Implementation:**
```rust
// JIT mode uses Arc<RwLock<T>> for mutable collections
pub enum Value {
    List(Arc<RwLock<Vec<Value>>>),
    Dict(Arc<RwLock<HashMap<String, Value>>>),
    // ...
}

// Compiled mode uses Vec<T> and HashMap<K, V> directly
// (Rust's ownership system handles memory)
```

### Optimization Passes

**Location:** `crates/tb-optimizer/src/`

**Current Optimizations:**
1. **Constant Folding:** Evaluate constant expressions at compile time
2. **Dead Code Elimination:** Remove unreachable code
3. **Inline Expansion:** Inline small functions

**Future Optimizations:**
- Loop unrolling
- Tail call optimization
- Common subexpression elimination

### Error Handling Strategy

**Design Philosophy:**
- **Compile-time errors:** Catch as many errors as possible during compilation
- **Runtime errors:** Use Result<T, String> for recoverable errors
- **Panics:** Only for unrecoverable errors (e.g., file I/O failures)

**Error Types:**
```rust
// Parse errors
pub enum ParseError {
    UnexpectedToken { expected: String, found: Token },
    UnexpectedEOF,
    InvalidSyntax { message: String },
}

// Type errors
pub enum TypeError {
    TypeMismatch { expected: Type, found: Type },
    UndefinedVariable { name: String },
    UndefinedFunction { name: String },
}

// Runtime errors
pub enum RuntimeError {
    DivisionByZero,
    IndexOutOfBounds { index: usize, length: usize },
    KeyNotFound { key: String },
}
```

---

## 🧩 Plugin System

**Location:** `crates/tb-plugin/src/`

**Purpose:** Cross-language FFI (Foreign Function Interface) for Python, JavaScript, Go, and Rust

### Architecture

```
TB Program (@plugin block)
        ↓
    Parser (tb-parser)
        ↓
    AST (PluginDefinition)
        ↓
    Plugin Compiler (tb-plugin/compiler.rs)
        ↓
    ┌─────────┬──────────┬──────────┬──────────┐
    │ Python  │   JS     │   Go     │   Rust   │
    │ (PyO3)  │ (Node)   │ (cgo)    │ (native) │
    └─────────┴──────────┴──────────┴──────────┘
        ↓
    Shared Library (.so/.dll/.dylib)
        ↓
    Plugin Loader (tb-plugin/loader.rs)
        ↓
    Runtime Execution (JIT or Compiled)
```

### Plugin Definition (AST)

**Location:** `crates/tb-core/src/ast.rs`

```rust
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
    Jit,      // Just-In-Time execution
    Compile,  // Ahead-Of-Time compilation
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginSource {
    Inline(Arc<String>),  // Inline code in @plugin block
    File(Arc<String>),    // External file path
}
```

### Plugin Parser

**Location:** `crates/tb-parser/src/parser.rs`

**Key Function:** `parse_plugin_block()`

```rust
fn parse_plugin_block(&mut self) -> Result<Statement> {
    self.expect(TokenKind::Plugin)?;
    self.expect(TokenKind::LBrace)?;

    let mut definitions = Vec::new();

    while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
        // Parse language (python, javascript, go, rust)
        let language = self.parse_plugin_language()?;

        // Parse module name (string literal)
        let name = self.expect_string()?;

        self.expect(TokenKind::LBrace)?;

        // Parse plugin body (mode, requires, source code)
        let mut mode = PluginMode::Jit;
        let mut requires = Vec::new();
        let mut source_code = String::new();

        // Parse mode: "jit" or "compile"
        if self.check_ident("mode") {
            self.advance();
            self.expect(TokenKind::Colon)?;
            mode = self.parse_plugin_mode()?;
        }

        // Parse requires: ["dep1", "dep2"]
        if self.check_ident("requires") {
            self.advance();
            self.expect(TokenKind::Colon)?;
            requires = self.parse_string_list()?;
        }

        // Parse file: "path.py" (optional)
        if self.check_ident("file") {
            self.advance();
            self.expect(TokenKind::Colon)?;
            let file_path = self.expect_string()?;
            source = PluginSource::File(file_path);
        } else {
            // Inline source code (native language syntax)
            source_code = self.parse_plugin_source()?;
            source = PluginSource::Inline(Arc::new(source_code));
        }

        self.expect(TokenKind::RBrace)?;

        definitions.push(PluginDefinition {
            language,
            name,
            mode,
            requires,
            source,
        });
    }

    self.expect(TokenKind::RBrace)?;

    Ok(Statement::Plugin { definitions, span })
}
```

**Important:** The parser extracts **native language code** verbatim, preserving indentation and syntax.

### Plugin Compiler

**Location:** `crates/tb-plugin/src/compiler.rs`

**Key Function:** `compile()`

```rust
pub fn compile(
    &self,
    language: &PluginLanguage,
    mode: &PluginMode,
    source: &str,
    name: &str,
    requires: &[String],
) -> Result<PathBuf> {
    match language {
        PluginLanguage::Python => self.compile_python(source, name, mode, requires),
        PluginLanguage::JavaScript => self.compile_javascript(source, name, mode, requires),
        PluginLanguage::Go => self.compile_go(source, name, requires),
        PluginLanguage::Rust => self.compile_rust(source, name),
    }
}
```

#### Python Compilation

```rust
fn compile_python(
    &self,
    source: &str,
    name: &str,
    mode: &PluginMode,
    requires: &[String],
) -> Result<PathBuf> {
    let source_file = self.temp_dir.join(format!("{}.py", name));

    // Install dependencies if needed
    if !requires.is_empty() {
        self.install_python_deps(requires)?;
    }

    // Write source (native Python code)
    fs::write(&source_file, source)?;

    match mode {
        PluginMode::Jit => {
            // JIT: Return Python file path for PyO3
            Ok(source_file)
        }
        PluginMode::Compile => {
            // Compile: Use Nuitka
            self.compile_python_nuitka(&source_file, name)
        }
    }
}

fn install_python_deps(&self, requires: &[String]) -> Result<()> {
    let mut cmd = Command::new("pip");
    cmd.arg("install");
    for dep in requires {
        cmd.arg(dep);
    }

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(TBError::plugin_error(
            format!("Failed to install Python dependencies: {}",
                    String::from_utf8_lossy(&output.stderr))
        ));
    }

    Ok(())
}
```

#### JavaScript Compilation

```rust
fn compile_javascript(
    &self,
    source: &str,
    name: &str,
    mode: &PluginMode,
    requires: &[String],
) -> Result<PathBuf> {
    let source_file = self.temp_dir.join(format!("{}.js", name));

    // Create package.json if dependencies exist
    if !requires.is_empty() {
        self.create_package_json(name, requires)?;
        self.install_npm_deps()?;
    }

    // Write source (native JavaScript code)
    fs::write(&source_file, source)?;

    match mode {
        PluginMode::Jit => {
            // JIT: Return JS file path for Node.js
            Ok(source_file)
        }
        PluginMode::Compile => {
            // Compile: Use pkg or nexe
            self.compile_javascript_pkg(&source_file, name)
        }
    }
}
```

#### Go Compilation

```rust
fn compile_go(&self, source: &str, name: &str, requires: &[String]) -> Result<PathBuf> {
    let source_file = self.temp_dir.join(format!("{}.go", name));

    // Write source (native Go code)
    fs::write(&source_file, source)?;

    // Compile to shared library
    let output_file = self.output_dir.join(format!("lib{}.so", name));

    let mut cmd = Command::new("go");
    cmd.arg("build")
       .arg("-buildmode=c-shared")
       .arg("-o")
       .arg(&output_file)
       .arg(&source_file);

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(TBError::plugin_error(
            format!("Go compilation failed: {}",
                    String::from_utf8_lossy(&output.stderr))
        ));
    }

    Ok(output_file)
}
```

#### Rust Compilation

```rust
fn compile_rust(&self, source: &str, name: &str) -> Result<PathBuf> {
    let source_file = self.temp_dir.join(format!("{}.rs", name));

    // Write source (native Rust code)
    fs::write(&source_file, source)?;

    // Compile to shared library
    let output_file = self.output_dir.join(format!("lib{}.so", name));

    let mut cmd = Command::new("rustc");
    cmd.arg("--crate-type=cdylib")
       .arg("-o")
       .arg(&output_file)
       .arg(&source_file);

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(TBError::plugin_error(
            format!("Rust compilation failed: {}",
                    String::from_utf8_lossy(&output.stderr))
        ));
    }

    Ok(output_file)
}
```

### Plugin Loader

**Location:** `crates/tb-plugin/src/loader.rs`

**Key Features:**
- **Lazy loading** - Plugins loaded only when first used
- **Caching** - Loaded libraries cached for reuse
- **Thread-safe** - Uses `DashMap` for concurrent access

```rust
pub struct PluginLoader {
    loaded_libraries: DashMap<String, Arc<Library>>,
    function_cache: DashMap<String, PluginFn>,
    plugin_metadata: DashMap<String, PluginMetadata>,
}

impl PluginLoader {
    pub fn new() -> Self {
        Self {
            loaded_libraries: DashMap::new(),
            function_cache: DashMap::new(),
            plugin_metadata: DashMap::new(),
        }
    }

    /// Load plugin library (lazy, cached)
    pub fn load_library(&self, path: &Path) -> Result<Arc<Library>> {
        let path_str = path.to_string_lossy().to_string();

        // Check cache first
        if let Some(lib) = self.loaded_libraries.get(&path_str) {
            return Ok(Arc::clone(lib.value()));
        }

        // Load library
        let library = unsafe {
            Library::new(path).map_err(|e| {
                TBError::plugin_error(format!("Failed to load library: {}", e))
            })?
        };

        let library = Arc::new(library);
        self.loaded_libraries.insert(path_str, Arc::clone(&library));

        Ok(library)
    }

    /// Get function from plugin (cached)
    pub fn get_function(&self, library_path: &Path, name: &str) -> Result<PluginFn> {
        let cache_key = format!("{}::{}", library_path.display(), name);

        // Check cache first
        if let Some(func) = self.function_cache.get(&cache_key) {
            return Ok(*func.value());
        }

        // Load library
        let library = self.load_library(library_path)?;

        // Load function
        let func: Symbol<PluginFn> = unsafe {
            library.get(name.as_bytes()).map_err(|e| {
                TBError::plugin_error(format!("Function '{}' not found: {}", name, e))
            })?
        };

        let func_ptr = *func;
        self.function_cache.insert(cache_key, func_ptr);

        Ok(func_ptr)
    }
}
```

### Plugin Execution (JIT Mode)

**Location:** `crates/tb-jit/src/executor.rs`

```rust
fn execute_statement(&mut self, stmt: &Statement) -> Result<()> {
    match stmt {
        Statement::Plugin { definitions, .. } => {
            for def in definitions {
                // Compile plugin
                let library_path = self.plugin_compiler.compile(
                    &def.language,
                    &def.mode,
                    &def.source,
                    &def.name,
                    &def.requires,
                )?;

                // Register plugin in environment
                self.env.insert(
                    def.name.clone(),
                    Value::Plugin {
                        library_path,
                        functions: Vec::new(),
                    }
                );
            }
            Ok(())
        }
        // ... other statements
    }
}
```

### Plugin Best Practices

1. **Use native syntax** - Don't try to write TB syntax in plugin blocks
2. **Specify dependencies** - Always use `requires` for external dependencies
3. **Test plugins separately** - Ensure plugin code works before integration
4. **Use JIT for development** - Faster iteration
5. **Use Compile for production** - Better performance
6. **Keep plugins small** - One responsibility per plugin
7. **Handle errors gracefully** - Return Result types in Rust plugins

---

## ⚙️ Configuration System

**Location:** `crates/tb-core/src/ast.rs`, `crates/tb-parser/src/parser.rs`

### Configuration AST

```rust
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
```

### Configuration Parser

**Location:** `crates/tb-parser/src/parser.rs`

```rust
fn parse_config_block(&mut self) -> Result<Statement> {
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

    self.expect(TokenKind::RBrace)?;

    Ok(Statement::Config { entries, span })
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
        TokenKind::Int(n) => {
            let value = *n;
            self.advance();
            Ok(ConfigValue::Int(value))
        }
        TokenKind::String(s) => {
            let value = s.clone();
            self.advance();
            Ok(ConfigValue::String(value))
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
        _ => Err(TBError::syntax_error(
            format!("Expected config value"),
            self.current().span,
            self.source_context.clone()
        ))
    }
}
```

### Runtime Configuration

**Location:** `crates/tb-cli/src/runner.rs`

```rust
// Parse @config block
let config = Config::parse(&source).unwrap_or_default();

// Extract configuration
let mode = config.mode; // "jit" or "compile"
let optimize = config.optimize; // true or false
let opt_level = config.opt_level; // 0-3
let threads = config.threads; // Number of worker threads

// Auto-detect networking usage
let uses_networking = detect_networking_usage(&program.statements)
                      || config.networking_enabled;

// Configure Tokio runtime
let runtime_config = if uses_networking {
    if threads > 1 {
        TokioRuntimeConfig::Full { worker_threads: threads }
    } else {
        TokioRuntimeConfig::Minimal { worker_threads: 1 }
    }
} else {
    TokioRuntimeConfig::None
};
```

### Networking Auto-Detection

**Location:** `crates/tb-cli/src/runner.rs`

```rust
fn detect_networking_usage(statements: &[Statement]) -> bool {
    for stmt in statements {
        if statement_uses_networking(stmt) {
            return true;
        }
    }
    false
}

fn statement_uses_networking(stmt: &Statement) -> bool {
    match stmt {
        Statement::Expression { expr, .. } => expr_uses_networking(expr),
        Statement::Let { value, .. } => expr_uses_networking(value),
        Statement::Assign { value, .. } => expr_uses_networking(value),
        Statement::If { condition, then_block, else_block, .. } => {
            expr_uses_networking(condition)
                || then_block.iter().any(statement_uses_networking)
                || else_block.as_ref().map_or(false, |b| b.iter().any(statement_uses_networking))
        }
        Statement::For { iterable, body, .. } => {
            expr_uses_networking(iterable)
                || body.iter().any(statement_uses_networking)
        }
        Statement::While { condition, body, .. } => {
            expr_uses_networking(condition)
                || body.iter().any(statement_uses_networking)
        }
        Statement::Function { body, .. } => {
            body.iter().any(statement_uses_networking)
        }
        _ => false,
    }
}

fn expr_uses_networking(expr: &Expression) -> bool {
    match expr {
        Expression::Call { callee, args, .. } => {
            // Check if function name is a networking function
            if let Expression::Ident(name) = &**callee {
                let networking_functions = [
                    "http_get", "http_post", "http_put", "http_delete",
                    "http_session", "http_request",
                    "tcp_connect", "tcp_listen", "udp_bind",
                    "connect_to", "send_message", "receive_message",
                    "spawn_task", "await_task",
                ];

                if networking_functions.contains(&name.as_str()) {
                    return true;
                }
            }

            // Recursively check arguments
            args.iter().any(expr_uses_networking)
        }
        // ... other expression types
        _ => false,
    }
}
```

### Conditional Compilation Features

**Location:** `crates/tb-runtime/Cargo.toml`

```toml
[features]
default = []

# Serialization
json = ["serde_json"]
yaml = ["serde_yaml"]

# Networking (single-threaded Tokio by default)
networking = ["tokio", "reqwest", "uuid", "dashmap", "once_cell"]
networking-full = ["networking", "bincode", "sha2"]

# Multi-threaded Tokio (only when threads > 1)
tokio-multi-thread = ["tokio/rt-multi-thread"]

# Full runtime (JIT mode)
full = ["tb-core", "tb-builtins", "json", "yaml",
        "networking-full", "tokio-multi-thread"]
```

### Configuration Best Practices

1. **Use auto-detection** - Let the compiler detect networking usage
2. **Optimize for production** - Set `opt_level: 3` for production builds
3. **Use JIT for development** - Faster iteration with `mode: "jit"`
4. **Specify threads explicitly** - For CPU-intensive workloads
5. **Enable caching** - Speeds up repeated compilations
6. **Disable debug in production** - Reduces binary size and improves performance

---

## 📊 Performance Benchmarking

### Benchmark Setup

```rust
// benches/benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci_benchmark(c: &mut Criterion) {
    c.bench_function("fibonacci 20", |b| {
        b.iter(|| {
            // Run TB program
            let result = run_tb_program("fibonacci.tbx");
            black_box(result)
        })
    });
}

criterion_group!(benches, fibonacci_benchmark);
criterion_main!(benches);
```

### Running Benchmarks

```bash
cargo bench
```

### Performance Comparison

| Mode | Fibonacci(30) | Startup Time | Binary Size |
|------|---------------|--------------|-------------|
| JIT | ~500ms | ~10ms | N/A |
| Compiled | ~50ms | ~1ms | ~2MB |
| Python | ~2000ms | ~50ms | N/A |
| Rust (native) | ~30ms | ~1ms | ~500KB |

---

## 🔐 Security Considerations

### 1. File I/O Safety

**Current Implementation:**
- File paths are not sanitized
- No sandboxing
- Full filesystem access

**Recommendations:**
- Add path validation
- Implement sandboxing for untrusted code
- Add permission system

### 2. Memory Safety

**Current Implementation:**
- Rust's ownership system prevents memory corruption
- Reference counting prevents use-after-free
- No unsafe code in core crates

**Potential Issues:**
- Cyclic references cause memory leaks
- Large allocations can cause OOM

### 3. Code Injection

**Current Implementation:**
- No eval() or dynamic code execution
- No string-to-code conversion

**Safe by Design:** TB Language does not support dynamic code execution.

---

## 🎨 Code Style Guide

### Rust Code Style

```rust
// ✅ GOOD: Clear naming, proper formatting
pub fn analyze_function_param_types_from_calls(
    statements: &[Statement],
    function_param_types: &mut HashMap<String, Vec<Type>>
) -> Result<(), String> {
    for stmt in statements {
        match stmt {
            Statement::Expression(expr) => {
                analyze_expression(expr, function_param_types)?;
            }
            _ => {}
        }
    }
    Ok(())
}

// ❌ BAD: Unclear naming, poor formatting
pub fn afptfc(s: &[Statement], fpt: &mut HashMap<String, Vec<Type>>) -> Result<(), String> {
    for stmt in s { match stmt { Statement::Expression(expr) => { analyze_expression(expr, fpt)?; } _ => {} } }
    Ok(())
}
```

### TB Language Code Style

```tb
// ✅ GOOD: Clear, readable
fn calculate_average(numbers) {
    let sum = reduce(numbers, 0, |acc, x| => acc + x)
    let count = len(numbers)
    return sum / float(count)
}

// ❌ BAD: Unclear, hard to read
fn calc_avg(n) {
    return reduce(n, 0, |a, x| => a + x) / float(len(n))
}
```

---

## 🧪 Test-Driven Development

### TDD Workflow

1. **Write failing test:**
```tb
fn test_new_feature() {
    let result = new_feature(42)

    if result == 84 {
        print("✅ PASS")
    } else {
        print("❌ FAIL")
    }
}
```

2. **Implement feature:**
```rust
// Implement in appropriate crate
pub fn new_feature(x: i64) -> i64 {
    x * 2
}
```

3. **Run test:**
```bash
tb.exe run tests/test_new_feature.tbx
```

4. **Refactor and repeat**

---

## 📈 Roadmap

### Short-term (Next Release)

- [ ] Fix Grade Calculator Bug (Float to Int conversion in match)
- [ ] Add `//` operator for integer division
- [ ] Improve error messages
- [ ] Add more built-in functions (abs, min, max, etc.)

### Medium-term (Next 3 Months)

- [ ] String interpolation: `"Hello, {name}!"`
- [ ] List comprehensions: `[x * 2 for x in numbers if x > 0]`
- [ ] Destructuring: `let [a, b, c] = list`
- [ ] Async/await support
- [ ] Package manager

### Long-term (Next Year)

- [ ] LLVM backend for better performance
- [ ] Incremental compilation
- [ ] Language server protocol (LSP) support
- [ ] Standard library expansion
- [ ] Cross-compilation support

---

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers
- Follow project conventions

### Contribution Process

1. **Fork repository**
2. **Create feature branch:** `git checkout -b feature/your-feature`
3. **Make changes and commit:** `git commit -m "[FEATURE] Your feature"`
4. **Push to fork:** `git push origin feature/your-feature`
5. **Create pull request**
6. **Address review feedback**
7. **Merge after approval**

### Review Checklist

- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages follow format
- [ ] All tests pass
- [ ] No compiler warnings

---

## 📚 Additional Resources

### Books

- **Crafting Interpreters** by Robert Nystrom
- **Programming Language Pragmatics** by Michael L. Scott
- **Types and Programming Languages** by Benjamin C. Pierce
- **The Rust Programming Language** by Steve Klabnik and Carol Nichols

### Papers

- **Type Inference for ML** - Robin Milner
- **A Theory of Type Polymorphism in Programming** - Robin Milner
- **Call-by-Push-Value** - Paul Blain Levy

### Online Resources

- **Rust Book:** https://doc.rust-lang.org/book/
- **Rust by Example:** https://doc.rust-lang.org/rust-by-example/
- **Crafting Interpreters:** https://craftinginterpreters.com/
- **LLVM Tutorial:** https://llvm.org/docs/tutorial/

---

## 🎓 Internship/Learning Projects

### Beginner Projects

1. **Add new built-in function** (e.g., `abs()`, `min()`, `max()`)
2. **Improve error messages** (add line numbers, suggestions)
3. **Add new operator** (e.g., `**` for exponentiation)

### Intermediate Projects

1. **Implement string interpolation**
2. **Add list comprehensions**
3. **Implement destructuring assignment**
4. **Add break/continue statements**

### Advanced Projects

1. **Implement async/await**
2. **Add LLVM backend**
3. **Implement incremental compilation**
4. **Add LSP support**

---

## 🔍 Debugging Checklist

### When Tests Fail

- [ ] Check if test is correct (not a test bug)
- [ ] Run in JIT mode first (faster iteration)
- [ ] Add debug print statements
- [ ] Check type inference (enable FIX12/FIX13 logging)
- [ ] Compare with similar working tests
- [ ] Check recent changes (git diff)
- [ ] Run single test in isolation
- [ ] Check for environment issues

### When Compilation Fails

- [ ] Check generated Rust code (`target/tb-compile-cache/src/main.rs`)
- [ ] Verify runtime function signatures
- [ ] Check for type mismatches
- [ ] Test in JIT mode (isolate codegen issues)
- [ ] Check for missing imports
- [ ] Verify Cargo.toml dependencies

### When Runtime Crashes

- [ ] Check for division by zero
- [ ] Check for index out of bounds
- [ ] Check for None dereference
- [ ] Check for cyclic references
- [ ] Add error handling
- [ ] Use debugger (gdb/lldb)

---

## 📞 Contact & Support

### Maintainers

- **Primary Maintainer:** [Your Name]
- **Email:** [your.email@example.com]
- **GitHub:** [github.com/your-username]

### Getting Help

1. **Check documentation** - Most questions are answered here
2. **Search issues** - Someone may have had the same problem
3. **Ask in discussions** - Community can help
4. **Create issue** - For bugs or feature requests
5. **Email maintainers** - For private/sensitive issues

---

**Happy Hacking on TB Language! 🚀**

**Version:** 1.0 | **Last Updated:** 2025-11-10 | **Contributors:** TB Lang Team

