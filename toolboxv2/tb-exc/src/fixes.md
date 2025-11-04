# TB Language Compiler - Fixes und Implementierungsstatus

**Datum:** 2025-10-28
**Status:** E2E-Tests: 17/92 bestanden (18.5%)
**Cargo-Tests:** 46/46 bestanden (100%)

---

## Zusammenfassung der aktuellen Situation

### ✅ Erfolgreich implementierte Fixes:

1. **Short-circuit Evaluation für AND/OR** (`tb-jit/src/executor.rs` Zeilen 325-353)
   - AND: Rechter Operand wird nicht evaluiert, wenn linker `false` ist
   - OR: Rechter Operand wird nicht evaluiert, wenn linker `true` ist
   - **Problem:** Tests schlagen trotzdem fehl wegen Typ-Fehler (siehe unten)

2. **Empty Return Detection** (`tb-codegen/src/rust_codegen.rs` Zeilen 1696-1770)
   - `has_empty_return()` erkennt `return;` ohne Wert in Funktionen
   - Setzt Rückgabetyp auf `Type::None` wenn gefunden
   - **Problem:** Codegen generiert trotzdem falschen Code (siehe unten)

3. **Scoping für Blöcke** (`tb-jit/src/executor.rs`)
   - If-Blöcke (Zeilen 144-155): `eval_block_with_scope()`
   - For-Loops (Zeilen 157-194): Environment save/restore
   - While-Loops (Zeilen 196-226): Environment save/restore
   - **Problem:** Falsch implementiert - löscht ALLE Variablen statt nur neue (siehe unten)

### ❌ Kritische Probleme (nach Priorität):

---

## 1. KRITISCH: Scoping-Bug (Variable Shadowing)

### Problem:
Die aktuelle Implementierung löscht **alle Variablen** im Block, nicht nur die **neuen Variablen**.

### Erwartetes Verhalten:
```tb
let x = 1
if true {
    let x = 2  // Neues x im if-Block (shadowing)
    print(x)   // Sollte 2 ausgeben
}
print(x)       // Sollte 1 ausgeben (äußeres x wiederhergestellt)
```

### Aktuelles Verhalten:
```
2
2  // ❌ Falsch! Sollte 1 sein
```

### Ursache:
```rust
// tb-jit/src/executor.rs, Zeilen 454-466
fn eval_block_with_scope(&mut self, stmts: &[Statement]) -> Result<Value> {
    let saved_env = self.env.clone();  // Speichert ALLE Variablen
    let result = self.eval_block(stmts);
    self.env = saved_env;  // ❌ Stellt ALLE Variablen wieder her
    result
}
```

Das Problem: Wenn `let x = 2` im Block ausgeführt wird, überschreibt es das äußere `x` in `self.env`. Beim Wiederherstellen wird das äußere `x` wiederhergestellt, aber das ist **zufällig korrekt** nur wenn keine Änderungen am äußeren `x` gemacht wurden.

### Korrekte Lösung:

**Option A: Scope-Stack (empfohlen)**
```rust
// In Executor struct
scopes: Vec<ImHashMap<Arc<String>, Value>>,  // Stack von Scopes

// Bei Block-Eintritt
fn enter_scope(&mut self) {
    self.scopes.push(self.env.clone());
}

// Bei Block-Austritt
fn exit_scope(&mut self) {
    if let Some(parent_scope) = self.scopes.pop() {
        self.env = parent_scope;
    }
}

// Bei Variable-Lookup
fn get_variable(&self, name: &Arc<String>) -> Option<&Value> {
    // Suche von innen nach außen
    self.env.get(name)
        .or_else(|| {
            self.scopes.iter().rev()
                .find_map(|scope| scope.get(name))
        })
}
```

**Option B: Tracking von neuen Variablen**
```rust
fn eval_block_with_scope(&mut self, stmts: &[Statement]) -> Result<Value> {
    let keys_before: HashSet<_> = self.env.keys().cloned().collect();
    let result = self.eval_block(stmts);

    // Nur neue Variablen löschen
    let keys_after: HashSet<_> = self.env.keys().cloned().collect();
    for new_key in keys_after.difference(&keys_before) {
        self.env.remove(new_key);
    }

    result
}
```

### Betroffene Tests:
- `Variable shadowing in scope` (jit + compiled)
- `Variable Declaration and Scope` (jit + compiled)
- `Scope - nested blocks` (jit + compiled)

---

## 2. KRITISCH: Short-Circuit Evaluation Typ-Fehler

### Problem:
Short-circuit ist implementiert, aber Tests schlagen fehl mit:
```
Type error: Logical operations require bool, got Bool and None
```

### Ursache:
Der Typ-Checker in `tb-types/src/checker.rs` prüft den Typ des rechten Operanden, **bevor** die Short-Circuit-Logik greift.

### Beispiel-Test:
```tb
fn should_not_be_called() {
    print("ERROR: Function was called!")
    return false
}

let result = false && should_not_be_called()
```

### Erwartetes Verhalten:
- `false &&` → Short-circuit, Funktion wird nicht aufgerufen
- Ausgabe: (nichts)

### Aktuelles Verhalten:
- Typ-Checker prüft `should_not_be_called()` → Rückgabetyp `None` (weil `return false` nicht erkannt wird)
- Fehler: `Logical operations require bool, got Bool and None`

### Lösung:
1. **Typ-Checker anpassen** (`tb-types/src/checker.rs`):
   - Bei AND/OR: Rechten Operanden nur prüfen wenn nötig
   - Oder: Typ-Fehler nur als Warning, nicht als Error

2. **Return-Type-Inference fixen** (`tb-codegen/src/rust_codegen.rs`):
   - `has_empty_return()` ist implementiert, aber wird nicht korrekt verwendet
   - Funktionen mit `return false` sollten Typ `Bool` haben, nicht `None`

### Betroffene Tests:
- `Short-circuit AND evaluation` (jit + compiled)
- `Short-circuit OR evaluation` (jit + compiled)

---

## 3. KRITISCH: Closure Capturing

### Problem:
Lambdas erfassen keine äußeren Variablen.

### Beispiel:
```tb
let outer = 100
let f = || {
    return outer  // ❌ `outer` ist nicht verfügbar
}
print(f())  // Fehler: Undefined variable: outer
```

### Ursache:
Die `Function`-Struktur hat ein `closure_env`-Feld, aber:
1. Es wird beim Erstellen von Lambdas nicht gesetzt
2. Es wird beim Aufrufen nicht verwendet

### Lösung:
**In `tb-jit/src/executor.rs`:**

```rust
// Bei Lambda-Erstellung (Expression::Lambda)
Expression::Lambda { params, body, .. } => {
    Ok(Value::Function(Function {
        name: Arc::new("<lambda>".to_string()),
        params: params.clone(),
        body: body.clone(),
        closure_env: Some(self.env.clone()),  // ✅ Erfasse aktuelle Umgebung
    }))
}

// Bei Funktionsaufruf (call_function)
fn call_function(&mut self, func: &Function, args: Vec<Value>) -> Result<Value> {
    // Speichere aktuelle Umgebung
    let saved_env = self.env.clone();

    // Verwende Closure-Umgebung als Basis, falls vorhanden
    if let Some(closure_env) = &func.closure_env {
        self.env = closure_env.clone();  // ✅ Starte mit Closure-Umgebung
    } else {
        self.env = ImHashMap::new();  // Leere Umgebung für normale Funktionen
    }

    // Füge Parameter hinzu
    for (param, arg) in func.params.iter().zip(args.iter()) {
        self.env.insert(param.name.clone(), arg.clone());
    }

    // Führe Funktion aus
    let result = self.eval_block(&func.body);

    // Stelle Umgebung wieder her
    self.env = saved_env;

    result
}
```

### Betroffene Tests:
- `Closure capturing variable` (jit + compiled)
- `Function - returning function` (jit + compiled)
- `Functions and Closures` (jit + compiled)
- `Scope - closure captures outer scope` (jit + compiled)
- `Complex program - closure with state` (jit + compiled)

---

## Detaillierte Fehleranalyse und Lösungsvorschläge

Hier sind die priorisierten Korrekturvorschläge, geordnet nach den von Ihnen genannten Schwerpunkten.

#### 1. Typen und deren Laufzeit-Repräsentation

##### Problem 1.1: Typinferenz von `None` im kompilierten Code

*   **Fehler:** `Builtin - type_of for all types [compiled]` schlägt fehl mit `error[E0282]: type annotations needed` bei `type_of(&None)`.
*   **Hypothese:** Der Rust-Compiler kann den generischen Typ `T` in `Option<T>` für einen literalen `None`-Wert nicht ohne Kontext ableiten. Der Codegenerator erzeugt `type_of(&None)`, was zu diesem Kompilierungsfehler führt.
*   **Vorgeschlagene Lösung:** Passen Sie den Codegenerator in `tb-codegen/src/rust_codegen.rs` an, um für `None`-Literale einen expliziten Typ zu deklarieren, der für den Kontext irrelevant ist, aber die Kompilierung ermöglicht.
    ```rust
    // Generierter Code für type_of(None) sollte so aussehen:
    type_of(&Option::<()>::None) // Option mit dem leeren Typ ()
    ```
    Alternativ kann die `type_of`-Funktion in `tb-runtime` so überladen werden, dass sie `Option` speziell behandelt.

##### Problem 1.2: Falsche Typ-Darstellung bei `type_of`

*   **Fehler:** `List constructor [compiled]` gibt den internen Rust-Typ `alloc::vec::Vec<tb_runtime::DictValue>` anstelle des erwarteten Strings `"list"` aus.
*   **Hypothese:** Die `type_of`-Funktion im `tb-runtime` verwendet `std::any::type_name::<T>()`, was für die Fehlersuche nützlich ist, aber nicht die vom Benutzer erwarteten Typnamen der TB-Sprache liefert.
*   **Vorgeschlagene Lösung:** Implementieren Sie eine robustere `type_of`-Funktion in `tb-runtime/src/lib.rs`, die den Wert zur Laufzeit prüft und die korrekten TB-Typnamen zurückgibt.
    ```rust
    // In tb-runtime/src/lib.rs
    pub fn type_of(value: &DictValue) -> String {
        match value {
            DictValue::Int(_) => "int".to_string(),
            DictValue::Float(_) => "float".to_string(),
            DictValue::String(_) => "string".to_string(),
            DictValue::Bool(_) => "bool".to_string(),
            DictValue::List(_) => "list".to_string(),
            DictValue::Dict(_) => "dict".to_string(),
        }
    }
    ```
    Der Codegenerator muss sicherstellen, dass diese Funktion für `type_of`-Aufrufe verwendet wird.

---

#### 2. Funktionen und Lambdas

##### Problem 2.1: Fehlerhafte Closure-Umgebung (Capturing)

*   **Fehler:** `Closure capturing variable [jit]` und `Function - returning function [jit]` schlagen fehl mit `Type error: Cannot call non-function type None`.
*   **Hypothese:** Dies ist ein kritisches Problem mit dem lexikalischen Gültigkeitsbereich (lexical scoping). Wenn eine Lambda-Funktion erstellt wird, erfasst (`capturing`) sie nicht die Umgebung (die zu diesem Zeitpunkt sichtbaren Variablen), in der sie definiert wurde. Wenn sie später außerhalb dieses Bereichs aufgerufen wird, sind die Variablen "verloren".
*   **Vorgeschlagene Lösung:**
    1.  **AST-Anpassung:** Erweitern Sie die `Function`-Struktur in `tb-core/src/ast.rs` um ein optionales Feld für die Closure-Umgebung: `closure_env: Option<ImHashMap<Arc<String>, Value>>`.
    2.  **JIT-Executor anpassen (`tb-jit/src/executor.rs`):**
        *   Beim Auswerten eines `Expression::Lambda`-Knotens, klonen Sie die *aktuelle Umgebung* (`self.env.clone()`) und speichern Sie sie im `closure_env`-Feld des neuen `Value::Function`.
        *   In der `call_function`-Methode: Wenn die aufgerufene Funktion ein `closure_env` besitzt, verwenden Sie **diese Umgebung** als Basis für den neuen Ausführungskontext, anstatt der Umgebung des Aufrufers.

##### Problem 2.2: Falscher Rückgabetyp für `None`

*   **Fehler:** `Function returning None [compiled]` schlägt fehl mit `mismatched types: expected 'String', found 'Option<_>'`.
*   **Hypothese:** Der Codegenerator leitet fälschlicherweise den Rückgabetyp `String` für eine Funktion ab, die `None` zurückgibt. Ein `None` in TB sollte in Rust zu `()` (dem "unit type") oder einem `Option`-Typ führen, aber nicht zu `String`.
*   **Vorgeschlagene Lösung:** Korrigieren Sie die Typherleitung für Rückgabewerte in `tb-codegen/src/rust_codegen.rs`. Der Codegenerator muss erkennen, wenn eine Funktion implizit oder explizit `None` zurückgibt, und den Rust-Funktions-Rückgabetyp korrekt als `()` oder einen passenden `Option`-Typ deklarieren.

---

## 4. HOCH: Range-Syntax in For-Loops

### Problem:
Parser unterstützt `for i in 1..5` und `for i in 1..=5` nicht.

### Beispiel:
```tb
for i in 1..5 {  // ❌ Syntax error: Expected LBrace, found DotDot
    print(i)
}
```

### Ursache:
Der Lexer tokenisiert `..` als `DotDot` und `..=` als `DotDotEq`, aber der Parser erwartet in `parse_for` einen allgemeinen Ausdruck, nicht speziell eine Range.

### Lösung:

**Option A: Range-Expression im AST**
```rust
// In tb-core/src/ast.rs, Expression enum
Range {
    start: Box<Expression>,
    end: Box<Expression>,
    inclusive: bool,
    span: Span,
}
```

**Option B: Desugaring zu range()-Funktion**
```rust
// In tb-parser/src/parser.rs, parse_for
fn parse_for(&mut self) -> Result<Statement> {
    // ...
    let iterable = if self.peek_token() == Some(&Token::DotDot)
                   || self.peek_token() == Some(&Token::DotDotEq) {
        // Parse range syntax: start..end oder start..=end
        let start = self.parse_expression()?;
        let inclusive = self.consume_token() == Token::DotDotEq;
        let end = self.parse_expression()?;

        // Desugar zu range(start, end) oder range(start, end, 1)
        Expression::Call {
            function: Box::new(Expression::Ident(Arc::new("range".to_string()), span)),
            args: vec![start, end],
            span,
        }
    } else {
        self.parse_expression()?
    };
    // ...
}
```

### Betroffene Tests:
- `Range - exclusive end` (jit + compiled)
- `Range - inclusive end` (jit + compiled)

---

## 5. HOCH: Codegen Type Inference Probleme

### Problem 5.1: `type_of(&None)` Kompilierungsfehler

**Fehler:**
```rust
error[E0282]: type annotations needed
  --> src\main.rs:19:21
   |
19 |     print(&type_of(&None));
   |                     ^^^^ cannot infer type of the type parameter `T`
```

**Lösung:**
```rust
// In tb-codegen/src/rust_codegen.rs, generate_expression
Expression::Literal(Literal::None, _) => {
    // Wenn in type_of()-Kontext, generiere Option::<()>::None
    if self.in_type_of_context {
        write!(self.buffer, "Option::<()>::None")?;
    } else {
        write!(self.buffer, "None")?;
    }
}
```

### Problem 5.2: `type_of` gibt Rust-Typen zurück statt TB-Typen

**Fehler:**
```
Expected: "list"
Actual: "alloc::vec::Vec<tb_runtime::DictValue>"
```

**Lösung:**
```rust
// In tb-runtime/src/lib.rs
pub fn type_of_int(_: &i64) -> String { "int".to_string() }
pub fn type_of_float(_: &f64) -> String { "float".to_string() }
pub fn type_of_string(_: &String) -> String { "string".to_string() }
pub fn type_of_bool(_: &bool) -> String { "bool".to_string() }
pub fn type_of_list<T>(_: &Vec<T>) -> String { "list".to_string() }
pub fn type_of_dict<K, V>(_: &HashMap<K, V>) -> String { "dict".to_string() }
pub fn type_of_none<T>(_: &Option<T>) -> String { "none".to_string() }
pub fn type_of_unit(_: &()) -> String { "none".to_string() }
```

**Codegen muss typ-spezifische Funktionen verwenden:**
```rust
// Statt: type_of(&value)
// Generiere: type_of_int(&value) oder type_of_list(&value) etc.
```

### Problem 5.3: Funktionen mit `return;` generieren falschen Code

**Fehler:**
```rust
error[E0069]: `return;` in a function whose return type is not `()`
  --> src\main.rs:16:13
   |
13 |     fn countdown(n: i64) -> String {
   |                             ------ expected `String` because of this return type
...
16 |             return;
   |             ^^^^^^ return type is not `()`
```

**Ursache:**
`has_empty_return()` ist implementiert, aber die Funktion hat trotzdem einen Rückgabetyp `String` (vom letzten Statement inferiert).

**Lösung:**
```rust
// In tb-codegen/src/rust_codegen.rs, generate_statement für Function
let ret_ty = if let Some(ty) = return_type {
    // Expliziter Rückgabetyp
    ty.clone()
} else {
    // ✅ FIX: Prüfe auf empty return ZUERST
    if self.has_empty_return(body) {
        Type::None  // Funktion gibt nichts zurück
    } else {
        self.infer_return_type(body)?  // Inferiere von letztem Statement
    }
};
```

### Betroffene Tests:
- `Builtin - type_of for all types` (compiled)
- `List constructor` (compiled)
- `Recursion - countdown` (compiled)
- `Function returning None` (compiled)

---

## 6. MITTEL: Parser-Erweiterungen

### Problem 6.1: `if` als Expression

**Fehler:**
```
Syntax error at 2:9: Unexpected token: If
```

**Beispiel:**
```tb
let x = if true { 1 } else { 2 }  // ❌ Parser erkennt if nicht als Expression
```

**Lösung:**
```rust
// In tb-core/src/ast.rs, Expression enum
IfElse {
    condition: Box<Expression>,
    then_branch: Box<Expression>,
    else_branch: Box<Expression>,  // MUSS vorhanden sein!
    span: Span,
}

// In tb-parser/src/parser.rs, parse_primary
fn parse_primary(&mut self) -> Result<Expression> {
    match self.current_token() {
        Token::If => self.parse_if_expression(),
        // ...
    }
}

fn parse_if_expression(&mut self) -> Result<Expression> {
    self.consume(Token::If)?;
    let condition = self.parse_expression()?;
    let then_branch = self.parse_block_expression()?;
    self.consume(Token::Else)?;  // else ist PFLICHT für if-expression
    let else_branch = self.parse_block_expression()?;

    Ok(Expression::IfElse { condition, then_branch, else_branch, span })
}
```

### Problem 6.2: Dict-Mutation

**Fehler:**
```
Syntax error at 3:9: Unexpected token: Eq
```

**Beispiel:**
```tb
let d = {"a": 1}
d["a"] = 2  // ❌ Parser erkennt Index-Assignment nicht
```

**Lösung:**
Erfordert neue Statement-Variante:
```rust
// In tb-core/src/ast.rs, Statement enum
IndexAssignment {
    object: Expression,
    index: Expression,
    value: Expression,
    span: Span,
}
```

### Betroffene Tests:
- `Expression - if returns value` (jit + compiled)
- `Control Flow` (jit + compiled)
- `Dict modification` (jit + compiled)

---

## 7. MITTEL: Listenoperationen (push/pop)

### Problem:
`push` und `pop` sind pure Funktionen (geben neue Liste zurück), aber Tests erwarten In-Place-Mutation.

### Beispiel:
```tb
let my_list = [1, 2]
push(my_list, 3)
print(my_list)  // Erwartet: [1, 2, 3], Aktuell: [1, 2]
```

### Lösung:

**Option A: Tests anpassen (empfohlen)**
```tb
let my_list = [1, 2]
my_list = push(my_list, 3)  // ✅ Ergebnis zuweisen
print(my_list)  // [1, 2, 3]
```

**Option B: In-Place-Mutation implementieren**
- Erfordert mutable References in TB
- Komplexe Änderung am Typ-System

### Betroffene Tests:
- `Push to list` (jit + compiled)
- `Pop from list` (jit + compiled)

---

## 8. NIEDRIG: Weitere Probleme

### 8.1: Float Modulo
**Fehler:** `Runtime error: Invalid modulo operation`
**Lösung:** Implementiere Modulo für Float-Typen

### 8.2: Import-System
**Fehler:** `Undefined variable: mymod`
**Lösung:** Import-System ist nicht implementiert

### 8.3: Pattern Matching
**Fehler:** `Type error: Pattern type Int doesn't match value type Generic("n")`
**Lösung:** Pattern-Matching-Typ-Inferenz verbessern

---

## Implementierungsplan (Priorität)

### Phase 1: Kritische Fixes (1-2 Tage)
1. ✅ Short-circuit Evaluation (implementiert, aber Typ-Fehler)
2. ❌ Scoping-Bug fixen (Option A: Scope-Stack)
3. ❌ Closure Capturing implementieren
4. ❌ Short-circuit Typ-Fehler fixen

### Phase 2: Codegen-Fixes (1 Tag)
5. ❌ `type_of(&None)` fixen
6. ❌ `type_of` TB-Typen zurückgeben
7. ❌ `return;` in Funktionen fixen

### Phase 3: Parser-Erweiterungen (2-3 Tage)
8. ❌ Range-Syntax in for-loops
9. ❌ `if` als Expression
10. ❌ Dict-Mutation

### Phase 4: Weitere Fixes (1-2 Tage)
11. ❌ Float Modulo
12. ❌ Import-System
13. ❌ Pattern Matching

---

## Test-Statistiken

### Aktuelle E2E-Test-Ergebnisse (17/92 bestanden):

**Bestanden (17):**
- Builtin - type_of for all types (jit)
- Complex program - nested data structures (jit)
- Complex program - string manipulation (jit)
- Edge case - empty function body (jit)
- Empty list literal (jit)
- File I/O (jit)
- Float arithmetic (compiled)
- Function returning None (jit)
- List constructor (jit)
- Literals and Basic Types (jit)
- Nested lists (jit)
- None literal (jit)
- Real program - grade calculator (jit)
- Real program - text processing (jit)
- Recursion - countdown (jit)
- Truthiness - None is falsy (jit)

**Fehlgeschlagen (75):**
- Alle Closure/Lambda-Tests (Capturing nicht implementiert)
- Alle Scoping-Tests (Scoping-Bug)
- Alle Short-circuit-Tests (Typ-Fehler)
- Alle Range-Tests (Parser unterstützt nicht)
- Alle if-as-expression-Tests (Parser unterstützt nicht)
- Alle Dict-Mutation-Tests (Parser unterstützt nicht)
- Viele Codegen-Tests (Typ-Inferenz-Probleme)

---

## Alte Analyse (Referenz)
