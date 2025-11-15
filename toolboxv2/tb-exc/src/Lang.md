# ToolBox Language Specification

**Complete Language Reference for Humans and LLMs**

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Syntax](#basic-syntax)
3. [Data Types](#data-types)
4. [Variables](#variables)
5. [Operators](#operators)
6. [Control Flow](#control-flow)
7. [Functions](#functions)
8. [Data Structures](#data-structures)
9. [Pattern Matching](#pattern-matching)
10. [Type System](#type-system)
11. [Builtin Functions](#builtin-functions)
12. [Plugin System](#plugin-system)
13. [Import System](#import-system)
14. [Configuration](#configuration)
15. [Error Handling](#error-handling)
16. [Comments](#comments)
17. [Execution Modes](#execution-modes)
18. [Standard Library](#standard-library)

---

## Overview

ToolBox (TB) is a modern, multi-paradigm scripting language with:
- **Type safety** with annotations and inference
- **Multi-language integration** via plugin system
- **Two execution modes**: JIT (fast development) and Compiled (production performance)
- **Pattern matching** for elegant control flow
- **Functional and imperative** programming styles
- **Zero-dependency deployment** as single binary

### File Extension
`.tbx` or `.tb`

### Basic Example
```tb
fn greet(name: string) -> string {
    return "Hello, " + name
}

print(greet("World"))
```

---

## Basic Syntax

### Case Sensitivity
ToolBox is **case-sensitive**. `variable` and `Variable` are different identifiers.

### Whitespace
- Whitespace is **significant** for readability but not for parsing
- Indentation is **optional** (uses braces `{}` for blocks)
- Semicolons are **optional** (newline terminates statements)

### Statement Termination
```tb
// Both valid:
let x = 10
let y = 20;

// Multi-statement lines need semicolons:
let a = 1; let b = 2
```

### Blocks
Use curly braces `{}` to define code blocks:
```tb
if condition {
    // block
}

fn example() {
    // block
}
```

---

## Data Types

### Primitive Types

#### Integer (`int`)
64-bit signed integer
```tb
let count: int = 42
let negative: int = -100
let zero: int = 0
```

#### Float (`float`)
64-bit floating point
```tb
let pi: float = 3.14159
let negative: float = -2.5
let scientific: float = 1.5e10
```

#### String (`string`)
UTF-8 encoded text
```tb
let name: string = "Alice"
let emoji: string = "🚀"
let empty: string = ""
```

**String Operations:**
```tb
// Concatenation
let full = "Hello" + " " + "World"

// Length
let len = len("hello")  // 5

// Escape sequences
let escaped = "Line 1\nLine 2\tTab"
```

#### Boolean (`bool`)
True or false values
```tb
let is_active: bool = true
let is_disabled: bool = false
```

### Composite Types

#### List (`list`)
Ordered collection of elements
```tb
let numbers: list = [1, 2, 3, 4, 5]
let mixed: list = [1, "two", 3.0, true]
let empty: list = []
```

#### Dictionary (`dict`)
Key-value pairs
```tb
let person: dict = {
    name: "Alice",
    age: 30,
    active: true
}

// Nested dictionaries
let config: dict = {
    server: {
        host: "localhost",
        port: 8080
    }
}
```

### Type Hierarchy
```
Value
├── Int (int)
├── Float (float)
├── String (string)
├── Bool (bool)
├── List (list)
├── Dict (dict)
├── Function
└── None
```

---

## Variables

### Declaration and Assignment

#### Immutable by Default (let)
```tb
let x = 10
// x = 20  // ERROR: cannot reassign

// But variables can be shadowed:
let x = 20  // New binding, old x is shadowed
```

#### Mutable Variables
Variables are mutable after first assignment:
```tb
let x = 10
x = 20  // Valid: reassignment
x = x + 5
```

### Type Annotations
```tb
// Explicit type annotation
let name: string = "Alice"
let age: int = 30
let score: float = 95.5
let active: bool = true

// Type inference (preferred)
let name = "Alice"      // inferred as string
let age = 30            // inferred as int
let score = 95.5        // inferred as float
```

### Scope
```tb
let global = 100

fn example() {
    let local = 200
    print(global)  // OK: can access global
    print(local)   // OK: local scope
}

// print(local)  // ERROR: local not in scope
```

---

## Operators

### Arithmetic Operators
```tb
let a = 10
let b = 3

let sum = a + b         // 13
let diff = a - b        // 7
let product = a * b     // 30
let quotient = a / b    // 3 (integer division)
let remainder = a % b   // 1

// Float arithmetic
let x = 10.0
let y = 3.0
let division = x / y    // 3.333...
```

### Comparison Operators
```tb
5 == 5      // true (equal)
5 != 3      // true (not equal)
5 > 3       // true (greater than)
5 < 3       // false (less than)
5 >= 5      // true (greater or equal)
5 <= 5      // true (less or equal)
```

### Logical Operators
```tb
true and true    // true
true and false   // false
true or false    // true
false or false   // false
not true         // false
not false        // true

// Short-circuit evaluation
let result = expensive_check() and another_check()
```

### String Operators
```tb
"Hello" + " " + "World"  // "Hello World"
```

### Operator Precedence
```
Highest to Lowest:
1. () (parentheses)
2. not (logical NOT)
3. *, /, % (multiplication, division, modulo)
4. +, - (addition, subtraction)
5. ==, !=, <, >, <=, >= (comparison)
6. and (logical AND)
7. or (logical OR)
```

### Type Coercion
```tb
// Automatic promotion int -> float
let x = 10      // int
let y = 2.5     // float
let z = x + y   // 12.5 (float)

// String concatenation requires explicit conversion
let num = 42
let text = "The answer is " + str(num)
```

---

## Control Flow

### If Statements

#### Basic If
```tb
if condition {
    // code
}
```

#### If-Else
```tb
if x > 10 {
    print("big")
} else {
    print("small")
}
```

#### If-Else-If Chain NO USE MATCHING INSTEAD


#### Nested If
```tb
if outer_condition {
    if inner_condition {
        // nested code
    }
}
```

### Loops

#### While Loop
```tb
let i = 0
while i < 5 {
    print(i)
    i = i + 1
}
// Output: 0 1 2 3 4
```

#### For Loop with Range
```tb
// range(n) generates 0..n-1
for i in range(5) {
    print(i)
}
// Output: 0 1 2 3 4

// range(start, end) generates start..end-1
for i in range(2, 7) {
    print(i)
}
// Output: 2 3 4 5 6

// range(start, end, step)
for i in range(0, 10, 2) {
    print(i)
}
// Output: 0 2 4 6 8
```

#### For Loop with List
```tb
let items = [10, 20, 30]
for item in items {
    print(item)
}
// Output: 10 20 30
```

#### Break Statement
```tb
for i in range(10) {
    if i == 5 {
        break  // Exit loop immediately
    }
    print(i)
}
// Output: 0 1 2 3 4
```

#### Continue Statement
```tb
for i in range(5) {
    if i == 2 {
        continue  // Skip to next iteration
    }
    print(i)
}
// Output: 0 1 3 4
```

### Loop Control Best Practices
```tb
// Infinite loop (use with break)
let running = true
while running {
    // ... do work ...
    if should_stop() {
        running = false
    }
}

// Early exit
for item in items {
    if not is_valid(item) {
        continue
    }
    process(item)
}
```

---

## Functions

### Function Definition

#### Basic Function
```tb
fn add(a: int, b: int) -> int {
    return a + b
}

let result = add(5, 3)  // 8
```

#### Function Without Return Type
```tb
fn greet(name: string) {
    print("Hello, " + name)
}

greet("Alice")
```

#### Function Without Parameters
```tb
fn get_constant() -> int {
    return 42
}

print(get_constant())
```

### Return Statements

#### Explicit Return
```tb
fn abs(x: int) -> int {
    if x < 0 {
        return -x
    }
    return x
}
```

#### Early Return Pattern
```tb
fn validate(x: int) -> bool {
    if x < 0 {
        return false
    }
    if x > 100 {
        return false
    }
    return true
}
```

### Recursion

#### Basic Recursion
```tb
fn factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

print(factorial(5))  // 120
```

#### Tail Recursion
```tb
fn fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

print(fibonacci(10))  // 55
```

### Higher-Order Functions

#### Functions as Values
```tb
fn apply_twice(f: function, x: int) -> int {
    return f(f(x))
}

fn double(x: int) -> int {
    return x * 2
}

let result = apply_twice(double, 5)  // 20
```

### Closures
```tb
fn make_counter() -> function {
    let count = 0

    fn increment() -> int {
        count = count + 1
        return count
    }

    return increment
}

let counter = make_counter()
print(counter())  // 1
print(counter())  // 2
print(counter())  // 3
```

### Function Best Practices
```tb
// Good: Type annotations for clarity
fn process_data(input: list, threshold: float) -> dict {
    // ...
}

// Good: Descriptive names
fn calculate_average_score(scores: list) -> float {
    // ...
}

// Good: Early validation
fn divide(a: int, b: int) -> float {
    if b == 0 {
        return 0.0  // or handle error
    }
    return float(a) / float(b)
}
```

---

## Data Structures

### Lists

#### List Creation
```tb
let empty = []
let numbers = [1, 2, 3, 4, 5]
let mixed = [1, "two", 3.0, true]
let nested = [[1, 2], [3, 4]]
```

#### List Access
```tb
let items = [10, 20, 30, 40, 50]

let first = items[0]    // 10
let third = items[2]    // 30
let last = items[4]     // 50

// Negative indices not supported
// items[-1]  // ERROR
```

#### List Operations
```tb
let items = [1, 2, 3]

// Length
let size = len(items)  // 3

// Append
let more = push(items, 4)  // [1, 2, 3, 4]

// Remove last
let less = pop(items)  // [1, 2]

// Lists are immutable - operations return new lists
print(items)  // [1, 2, 3] (unchanged)
```

#### List Iteration
```tb
let numbers = [1, 2, 3, 4, 5]

for num in numbers {
    print(num)
}

// With index
for i in range(len(numbers)) {
    print(i, numbers[i])
}
```

### Dictionaries

#### Dictionary Creation
```tb
let empty = {}

let person = {
    name: "Alice",
    age: 30,
    active: true
}

let config = {
    server: {
        host: "localhost",
        port: 8080
    },
    database: {
        name: "mydb",
        user: "admin"
    }
}
```

#### Dictionary Access
```tb
let person = {
    name: "Alice",
    age: 30
}

// Dot notation
let name = person.name   // "Alice"
let age = person.age     // 30

// Bracket notation
let key = "name"
let value = person[key]  // "Alice"
```

#### Dictionary Operations
```tb
let data = {
    a: 1,
    b: 2,
    c: 3
}

// Get keys
let key_list = keys(data)  // ["a", "b", "c"]

// Get values
let val_list = values(data)  // [1, 2, 3]

// Length
let size = len(data)  // 3

// Check existence (use pattern matching or try-catch)
```

#### Dictionary Iteration
```tb
let person = {
    name: "Alice",
    age: 30,
    city: "NYC"
}

// Iterate over keys
for key in keys(person) {
    print(key, person[key])
}

// Iterate over values
for value in values(person) {
    print(value)
}
```

### Nested Structures
```tb
let company = {
    name: "TechCorp",
    employees: [
        {name: "Alice", role: "Engineer"},
        {name: "Bob", role: "Designer"}
    ],
    locations: {
        hq: "San Francisco",
        branch: "New York"
    }
}

// Deep access
let first_employee = company.employees[0].name  // "Alice"
let hq_location = company.locations.hq          // "San Francisco"
```

---

## Pattern Matching

### Match Expression

#### Basic Match
```tb
let x = 2

let result = match x {
    0 => "zero",
    1 => "one",
    2 => "two",
    _ => "many"
}

print(result)  // "two"
```

#### Match with Blocks
```tb
let status = match code {
    200 => {
        log("Success")
        "OK"
    },
    404 => {
        log("Not Found")
        "Error"
    },
    _ => "Unknown"
}
```

#### Range Patterns
```tb
let score = 85

let grade = match score {
    0..60 => "F",
    60..70 => "D",
    70..80 => "C",
    80..90 => "B",
    90..100 => "A",
    _ => "Invalid"
}

print(grade)  // "B"
```

#### Multiple Patterns
```tb
let value = 42

let category = match value {
    0 => "zero",
    1 | 2 | 3 => "small",
    4..10 => "medium",
    _ => "large"
}
```

#### Type Patterns (if supported)
```tb
let data = [1, 2, 3]

let result = match type_of(data) {
    "int" => "number",
    "string" => "text",
    "list" => "array",
    _ => "unknown"
}
```

### Wildcard Pattern
The underscore `_` matches anything:
```tb
match x {
    1 => "one",
    2 => "two",
    _ => "other"  // catches everything else
}
```

---

## Type System

### Type Annotations

#### Variable Types
```tb
let name: string = "Alice"
let age: int = 30
let score: float = 95.5
let active: bool = true
let items: list = [1, 2, 3]
let data: dict = {key: "value"}
```

#### Function Types
```tb
fn process(
    input: string,
    count: int,
    threshold: float
) -> dict {
    // function body
    return {}
}
```

### Type Inference

ToolBox can infer types automatically:
```tb
// Type inference
let name = "Alice"       // inferred: string
let age = 30             // inferred: int
let score = 95.5         // inferred: float
let active = true        // inferred: bool

fn add(a, b) {           // inferred from usage
    return a + b
}
```

### Type Checking

#### Compile-Time Checking
```tb
let x: int = 42
// x = "hello"  // ERROR: type mismatch

fn add(a: int, b: int) -> int {
    return a + b
}

// add("hello", "world")  // ERROR: wrong types
```

#### Runtime Type Checking
```tb
let value = get_input()

match type_of(value) {
    "int" => process_number(value),
    "string" => process_text(value),
    _ => print("Unsupported type")
}
```

### Type Conversion

#### Explicit Conversion
```tb
// To string
let text = str(42)         // "42"
let text2 = str(3.14)      // "3.14"
let text3 = str(true)      // "true"

// To int
let num = int(3.14)        // 3 (truncates)
let num2 = int("42")       // 42
let num3 = int(true)       // 1
let num4 = int(false)      // 0

// To float
let f = float(42)          // 42.0
let f2 = float("3.14")     // 3.14

// To bool (any non-zero is true)
let b = bool(1)            // true
let b2 = bool(0)           // false
```

### Type Compatibility

#### Automatic Promotion
```tb
let x: int = 10
let y: float = 2.5
let z = x + y  // z is float (10 + 2.5 = 12.5)
```

#### Type Annotations in Collections
```tb
// List of integers
let numbers: list = [1, 2, 3, 4, 5]

// Mixed-type list
let mixed: list = [1, "two", 3.0, true]

// Dictionary with known structure
let person: dict = {
    name: "Alice",
    age: 30
}
```

---

## Builtin Functions

### Input/Output

#### print
Print values to stdout
```tb
print("Hello")           // Hello
print(42)                // 42
print(3.14)              // 3.14
print(true)              // true

// Multiple arguments
print("x =", 42)         // x = 42
```

#### echo
Similar to print, returns the value
```tb
let x = echo("Hello")    // Prints and returns "Hello"
```

### Type Conversion

#### str
Convert to string
```tb
str(42)          // "42"
str(3.14)        // "3.14"
str(true)        // "true"
str([1, 2, 3])   // "[1, 2, 3]"
```

#### int
Convert to integer
```tb
int(3.14)        // 3
int("42")        // 42
int(true)        // 1
int(false)       // 0
```

#### float
Convert to float
```tb
float(42)        // 42.0
float("3.14")    // 3.14
```

#### bool
Convert to boolean
```tb
bool(1)          // true
bool(0)          // false
bool("")         // false
bool("text")     // true
```

### Type Inspection

#### type_of
Get type as string
```tb
type_of(42)          // "int"
type_of(3.14)        // "float"
type_of("hello")     // "string"
type_of(true)        // "bool"
type_of([1, 2])      // "list"
type_of({a: 1})      // "dict"
```

### Collection Functions

#### len
Get length/size
```tb
len("hello")         // 5
len([1, 2, 3])       // 3
len({a: 1, b: 2})    // 2
```

#### range
Generate integer sequence
```tb
range(5)             // [0, 1, 2, 3, 4]
range(2, 7)          // [2, 3, 4, 5, 6]
range(0, 10, 2)      // [0, 2, 4, 6, 8]
```

#### push
Append to list (returns new list)
```tb
let items = [1, 2, 3]
let more = push(items, 4)    // [1, 2, 3, 4]
// items is unchanged: [1, 2, 3]
```

#### pop
Remove last element (returns new list)
```tb
let items = [1, 2, 3, 4]
let less = pop(items)        // [1, 2, 3]
// items is unchanged: [1, 2, 3, 4]
```

#### keys
Get dictionary keys as list
```tb
let data = {a: 1, b: 2, c: 3}
let key_list = keys(data)    // ["a", "b", "c"]
```

#### values
Get dictionary values as list
```tb
let data = {a: 1, b: 2, c: 3}
let val_list = values(data)  // [1, 2, 3]
```

### Mathematical Functions

#### abs
Absolute value
```tb
abs(-5)          // 5
abs(5)           // 5
abs(-3.14)       // 3.14
```

#### min
Minimum value
```tb
min(5, 3)        // 3
min(10, 20, 5)   // 5
```

#### max
Maximum value
```tb
max(5, 3)        // 5
max(10, 20, 5)   // 20
```

---

### File I/O Functions

#### open
Open a file
```tb
// Real file
let file = open("data.txt", "r")

// Modes: "r" (read), "w" (write), "a" (append), "r+" (read/write)
```

#### read_file
Read entire file content (async, non-blocking)
```tb
// Read real file
let content = read_file("data.txt")
```

#### write_file
Write content to file (async, non-blocking)
```tb
// Write to real file
write_file("output.txt", "Hello, World!")
```

#### file_exists
Check if file exists
```tb
if file_exists("config.json") {
    print("Config found")
}
```

---

### Networking Functions

#### create_server
Create TCP or UDP server
```tb
// TCP server
let server = create_server(
    on_connect,      // Callback: fn(client_addr, client_msg)
    on_disconnect,   // Callback: fn(client_addr)
    on_message,      // Callback: fn(client_addr, msg)
    "0.0.0.0",       // Host
    8080,            // Port
    "tcp"            // Type: "tcp" or "udp"
)

// UDP server
let udp_server = create_server(
    on_connect,
    on_disconnect,
    on_message,
    "0.0.0.0",
    9000,
    "udp"
)
```

#### connect_to
Connect to remote server
```tb
// TCP client
let conn = connect_to(
    on_connect,
    on_disconnect,
    on_message,
    "localhost",
    8080,
    "tcp"
)

// UDP client
let udp_conn = connect_to(
    on_connect,
    on_disconnect,
    on_message,
    "localhost",
    9000,
    "udp"
)
```

#### send_to
Send message to connection
```tb
// Send string
send_to(conn, "Hello, Server!")

// Send JSON (dict auto-converted)
send_to(conn, {"type": "message", "data": "Hello"})
```

#### http_session
Create HTTP session with persistent connections
```tb
// Basic session
let session = http_session("https://api.example.com")

// With headers
let session = http_session(
    "https://api.example.com",
    {"Authorization": "Bearer token123", "User-Agent": "TB/1.0"}
)

// With cookies file (blob or real)
let session = http_session(
    "https://api.example.com",
    {"User-Agent": "TB/1.0"},
    "blob_id/cookies.json"
)
```

#### http_request
Send HTTP request
```tb
// GET request
let response = http_request(session, "/users", "GET")
print(response.status)  // 200
print(response.body)    // Response body

// POST with JSON data
let response = http_request(
    session,
    "/api/data",
    "POST",
    {"name": "John", "age": 30}
)

// Access response
print(response.status)           // 201
print(response.headers["Content-Type"])  // "application/json"
print(response.body)             // Response body
```

---

### Utility Functions

#### json_parse
Parse JSON string to dictionary
```tb
let json_str = '{"name": "Alice", "age": 25, "active": true}'
let data = json_parse(json_str)

print(data["name"])   // "Alice"
print(data["age"])    // 25
print(data["active"]) // true
```

#### json_stringify
Convert dictionary to JSON string
```tb
let data = {"name": "Bob", "scores": [95, 87, 92]}

// Compact JSON
let json = json_stringify(data)
print(json)  // {"name":"Bob","scores":[95,87,92]}

// Pretty-printed JSON
let pretty = json_stringify(data, true)
print(pretty)
// {
//   "name": "Bob",
//   "scores": [95, 87, 92]
// }
```

#### yaml_parse
Parse YAML string to dictionary
```tb
let yaml_str = "
name: Alice
age: 25
hobbies:
  - reading
  - coding
"
let data = yaml_parse(yaml_str)
print(data["name"])      // "Alice"
print(data["hobbies"])   // ["reading", "coding"]
```

#### yaml_stringify
Convert dictionary to YAML string
```tb
let data = {
    "name": "Bob",
    "config": {
        "debug": true,
        "port": 8080
    }
}

let yaml = yaml_stringify(data)
print(yaml)
// name: Bob
// config:
//   debug: true
//   port: 8080
```

#### time
Get current time information
```tb
// Auto-detect local timezone
let now = time()
print(now["year"])       // 2024
print(now["month"])      // 10
print(now["day"])        // 19
print(now["hour"])       // 14
print(now["minute"])     // 30
print(now["second"])     // 45
print(now["timezone"])   // "Local"
print(now["iso8601"])    // "2024-10-19T14:30:45+00:00"
print(now["timestamp"])  // 1729348245

// Specific timezone
let ny_time = time("America/New_York")
print(ny_time["timezone"])  // "America/New_York"

let tokyo_time = time("Asia/Tokyo")
print(tokyo_time["hour"])   // Adjusted for Tokyo timezone
```

---

## Plugin System

The plugin system allows embedding Python, JavaScript, Rust, and Go code directly in TB programs.

### Plugin Declaration

#### Basic Syntax
```tb
@plugin {
    <language> "<plugin_name>" {
        mode: "<jit|compile>",
        requires: ["<dependencies>"],
        file: "<optional_file_path>",

        <code>
    }
}
```

### Python Plugin

#### Inline Python (JIT)
```tb
@plugin {
    python "math_helpers" {
        mode: "jit",

        def square(x: int) -> int:
            return x * x

        def cube(x: int) -> int:
            return x * x * x
    }
}

print(math_helpers.square(5))  // 25
print(math_helpers.cube(3))    // 27
```

#### Python with Dependencies
```tb
@plugin {
    python "data_analysis" {
        mode: "jit",
        requires: ["numpy", "pandas"],

        import numpy as np

        def mean(data: list) -> float:
            return float(np.mean(data))

        def std(data: list) -> float:
            return float(np.std(data))
    }
}

let numbers = [1, 2, 3, 4, 5]
print(data_analysis.mean(numbers))  // 3.0
```

#### Python External File
```tb
@plugin {
    python "utilities" {
        mode: "jit",
        file: "path/to/utils.py"
    }
}

// All functions from utils.py are available
utilities.some_function()
```

### JavaScript Plugin

#### Inline JavaScript (JIT)
```tb
@plugin {
    javascript "string_ops" {
        mode: "jit",

        function reverse(s) {
            return s.split('').reverse().join('');
        }

        function uppercase(s) {
            return s.toUpperCase();
        }
    }
}

print(string_ops.reverse("hello"))     // "olleh"
print(string_ops.uppercase("world"))   // "WORLD"
```

#### JavaScript with Arrays
```tb
@plugin {
    javascript "array_ops" {
        mode: "jit",

        function sum(arr) {
            return arr.reduce((a, b) => a + b, 0);
        }

        function product(arr) {
            return arr.reduce((a, b) => a * b, 1);
        }
    }
}

let numbers = [1, 2, 3, 4, 5]
print(array_ops.sum(numbers))      // 15
print(array_ops.product(numbers))  // 120
```

### Rust Plugin

#### Inline Rust (Compiled)
```tb
@plugin {
    rust "fast_math" {
        mode: "compile",

        use std::os::raw::c_void;

        #[repr(C)]
        pub struct FFIValue {
            tag: u8,
            data: FFIValueData,
        }

        #[repr(C)]
        union FFIValueData {
            int_val: i64,
            float_val: f64,
            bool_val: u8,
            ptr: *mut c_void,
        }

        const TAG_FLOAT: u8 = 3;

        #[no_mangle]
        pub unsafe extern "C" fn fast_sqrt(args: *const FFIValue, _len: usize) -> FFIValue {
            let x = (*args).data.float_val;
            let result = x.sqrt();
            FFIValue {
                tag: TAG_FLOAT,
                data: FFIValueData { float_val: result },
            }
        }
    }
}

print(fast_math.fast_sqrt(16.0))  // 4.0
```

#### Rust External File
```tb
@plugin {
    rust "performance" {
        mode: "compile",
        file: "path/to/fast_ops.rs"
    }
}

performance.fast_function()
```

### Go Plugin

#### Inline Go (JIT)
```tb
@plugin {
    go "concurrent_ops" {
        mode: "jit",

        package main

        import "fmt"

        func Fibonacci(n int) int {
            if n <= 1 {
                return n
            }
            return Fibonacci(n-1) + Fibonacci(n-2)
        }

        func Sum(arr []int) int {
            sum := 0
            for _, v := range arr {
                sum += v
            }
            return sum
        }
    }
}

print(concurrent_ops.Fibonacci(10))  // 55
```

#### Go External File
```tb
@plugin {
    go "utils" {
        mode: "jit",
        file: "path/to/helpers.go"
    }
}

utils.SomeGoFunction()
```

### Multi-Language Integration

#### Combining Multiple Languages
```tb
@plugin {
    python "preprocessor" {
        mode: "jit",

        def normalize(data: list) -> list:
            max_val = max(data)
            return [x / max_val for x in data]
    }

    javascript "processor" {
        mode: "jit",

        function sum(data) {
            return data.reduce((a, b) => a + b, 0);
        }
    }

    rust "optimizer" {
        mode: "compile",

        // Rust code here
    }
}

// Use all plugins together
let raw_data = [10, 20, 30, 40, 50]
let normalized = preprocessor.normalize(raw_data)
let total = processor.sum(normalized)
print(total)
```

### Plugin Best Practices

#### When to Use Each Language
- **Python**: Data analysis, ML, scientific computing
- **JavaScript**: String/array manipulation, JSON processing
- **Rust**: Performance-critical code, system programming
- **Go**: Concurrent operations, networking, microservices

#### Mode Selection
- **JIT Mode**: Fast development, quick iteration
- **Compile Mode**: Production performance, static linking

```tb
// Development (fast iteration)
@plugin {
    python "dev_tool" {
        mode: "jit",  // <- Use JIT for development
        // code
    }
}

// Production (optimized)
@plugin {
    rust "prod_service" {
        mode: "compile",  // <- Use compile for production
        // code
    }
}
```

---

## Import System

### Basic Import

#### Single File Import
```tb
@import {
    "path/to/module.tbx"
}

// Now all functions from module.tbx are available
module_function()
```

#### Multiple File Import
```tb
@import {
    "path/to/math_utils.tbx",
    "path/to/string_utils.tbx",
    "path/to/helpers.tbx"
}

// Use functions from all imported modules
math_utils.add(5, 3)
string_utils.reverse("hello")
```

### Import Paths

#### Absolute Path
```tb
@import {
    "/full/path/to/module.tbx"
}
```

#### Relative Path
```tb
@import {
    "./local_module.tbx",
    "../parent_module.tbx",
    "../../grandparent_module.tbx"
}
```

### Module Structure

#### Creating a Module
**math_utils.tbx:**
```tb
fn add(a: int, b: int) -> int {
    return a + b
}

fn multiply(a: int, b: int) -> int {
    return a * b
}

fn square(x: int) -> int {
    return x * x
}
```

#### Using the Module
**main.tbx:**
```tb
@import {
    "math_utils.tbx"
}

let result = add(5, 3)
let product = multiply(4, 7)
print(result)   // 8
print(product)  // 28
```

### Import Best Practices

#### Organize Imports at Top
```tb
// Good: All imports at the top
@import {
    "utils/math.tbx",
    "utils/string.tbx",
    "helpers/validation.tbx"
}

fn main() {
    // Your code here
}
```

#### Avoid Circular Dependencies
```tb
// BAD: Don't do this
// file_a.tbx imports file_b.tbx
// file_b.tbx imports file_a.tbx

// GOOD: Extract common code to a third module
// file_a.tbx imports common.tbx
// file_b.tbx imports common.tbx
```

### Module Caching

ToolBox automatically caches imported modules:
- SHA256 content-based cache keys
- Automatic invalidation on file changes
- 90%+ cache hit rate in typical workflows

---

## Configuration

### Config Block

The `@config` block sets compilation options:

```tb
@config {
    mode: "jit",           // "jit" or "compile"
    optimize: true,        // Enable optimizations
    opt_level: 3,          // 0-3 (higher = more optimization)
    cache: true,           // Enable caching
    debug: false           // Debug mode
}
```

### Configuration Options

#### mode
Execution mode:
- `"jit"`: Just-In-Time execution (fast startup, ~92ms avg)
- `"compile"`: Compile to native binary (slower startup, faster execution)

```tb
@config {
    mode: "jit"
}
```

#### optimize
Enable/disable optimizations:
```tb
@config {
    optimize: true   // Enable all optimizations
}
```

#### opt_level
Optimization level (0-3):
- `0`: No optimization
- `1`: Basic optimizations
- `2`: Standard optimizations (default)
- `3`: Aggressive optimizations

```tb
@config {
    opt_level: 3
}
```

#### cache
Enable/disable caching:
```tb
@config {
    cache: true   // Cache compiled results
}
```

#### debug
Enable debug output:
```tb
@config {
    debug: true   // Print debug information
}
```

### Complete Config Example
```tb
@config {
    mode: "compile",
    optimize: true,
    opt_level: 3,
    cache: true,
    debug: false
}

// Your code here
```

---

## Error Handling

### Runtime Errors

ToolBox provides comprehensive error messages:

#### Division by Zero
```tb
let x = 10 / 0
// ERROR: Division by zero
```

#### Type Mismatch
```tb
let x: int = 42
x = "hello"
// ERROR: Type mismatch: expected int, got string
```

#### Undefined Variable
```tb
print(undefined_variable)
// ERROR: Undefined variable: undefined_variable
```

#### Undefined Function
```tb
non_existent_function()
// ERROR: Undefined function: non_existent_function
```

#### Index Out of Bounds
```tb
let items = [1, 2, 3]
print(items[10])
// ERROR: Index out of bounds: 10 (list length: 3)
```

### Error Handling Patterns

#### Validation Pattern
```tb
fn divide(a: int, b: int) -> float {
    if b == 0 {
        print("ERROR: Division by zero")
        return 0.0
    }
    return float(a) / float(b)
}
```

#### Guard Clauses
```tb
fn process_list(items: list) -> int {
    if len(items) == 0 {
        return 0
    }

    let sum = 0
    for item in items {
        sum = sum + item
    }
    return sum
}
```

#### Type Checking
```tb
fn process_value(value) {
    let t = type_of(value)

    match t {
        "int" => print("Processing integer"),
        "string" => print("Processing string"),
        _ => print("Unsupported type")
    }
}
```

---

## Comments

### Single-Line Comments
```tb
// This is a single-line comment
let x = 42  // Comment after code
```

### Multi-Line Comments
```tb
/*
This is a multi-line comment
spanning multiple lines
*/

let y = 100
```

### Documentation Comments
```tb
// Function to calculate factorial
// Args:
//   n: integer value
// Returns:
//   factorial of n
fn factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}
```

---

## Execution Modes

ToolBox supports two execution modes:

### JIT Mode (Just-In-Time)

**Characteristics:**
- Fast startup (~45ms cold, <1ms warm)
- Immediate execution
- No compilation step
- Average execution: ~92ms
- Ideal for development and scripts

**Usage:**
```bash
tb run script.tbx
tb run script.tbx --mode jit
```

**In Code:**
```tb
@config {
    mode: "jit"
}
```

**Best For:**
- Development and testing
- Quick scripts
- Interactive REPL
- Rapid iteration

### Compiled Mode

**Characteristics:**
- Native binary output
- Slower startup (~250ms first compile)
- Fast execution (2-5x faster than JIT)
- Requires compilation step
- Production performance

**Usage:**
```bash
tb compile script.tbx -o output
./output

# Or with tb run
tb run script.tbx --mode compile
```

**In Code:**
```tb
@config {
    mode: "compile"
}
```

**Best For:**
- Production deployments
- Performance-critical code
- Long-running processes
- Distribution as binary

### Mode Comparison

| Feature | JIT Mode | Compiled Mode |
|---------|----------|---------------|
| Startup Time | 45ms (cold), <1ms (warm) | 250ms (first compile), <1ms (cached) |
| Execution Speed | ~92ms average | 18-45ms (2-5x faster) |
| Memory Usage | 12-28MB | 8-20MB |
| Best Use | Development, scripts | Production, performance |
| Output | None | Native binary |

---

## Standard Library

### Math Functions

#### abs
Absolute value
```tb
abs(-5)      // 5
abs(5)       // 5
abs(-3.14)   // 3.14
```

#### min
Minimum of values
```tb
min(5, 3)           // 3
min(10, 20, 5, 15)  // 5
```

#### max
Maximum of values
```tb
max(5, 3)           // 5
max(10, 20, 5, 15)  // 20
```

#### pow
Power function
```tb
pow(2, 8)    // 256
pow(10, 3)   // 1000
```

#### sqrt
Square root
```tb
sqrt(16)     // 4.0
sqrt(2)      // 1.414...
```

### String Functions

#### len
String length
```tb
len("hello")     // 5
len("")          // 0
len("🚀")        // 1 (counts UTF-8 characters)
```

#### str
Convert to string
```tb
str(42)          // "42"
str(3.14)        // "3.14"
str(true)        // "true"
```

#### Concatenation
```tb
"Hello" + " " + "World"  // "Hello World"
```

### List Functions

#### len
List length
```tb
len([1, 2, 3])   // 3
len([])          // 0
```

#### push
Append element (returns new list)
```tb
let items = [1, 2, 3]
let more = push(items, 4)  // [1, 2, 3, 4]
```

#### pop
Remove last element (returns new list)
```tb
let items = [1, 2, 3, 4]
let less = pop(items)      // [1, 2, 3]
```

#### range
Generate integer sequence
```tb
range(5)             // [0, 1, 2, 3, 4]
range(2, 7)          // [2, 3, 4, 5, 6]
range(0, 10, 2)      // [0, 2, 4, 6, 8]
```

### Dictionary Functions

#### keys
Get all keys
```tb
let data = {a: 1, b: 2, c: 3}
let k = keys(data)  // ["a", "b", "c"]
```

#### values
Get all values
```tb
let data = {a: 1, b: 2, c: 3}
let v = values(data)  // [1, 2, 3]
```

#### len
Dictionary size
```tb
len({a: 1, b: 2})  // 2
```

### Type Functions

#### type_of
Get type as string
```tb
type_of(42)          // "int"
type_of(3.14)        // "float"
type_of("hello")     // "string"
type_of(true)        // "bool"
type_of([1, 2])      // "list"
type_of({a: 1})      // "dict"
```

### I/O Functions

#### print
Print to stdout
```tb
print("Hello")           // Hello
print(42)                // 42
print("x =", x)          // x = <value of x>
```

#### echo
Print and return value
```tb
let x = echo("Hello")    // Prints "Hello", returns "Hello"
```

---

## Complete Examples

### Example 1: Fibonacci Sequence
```tb
@config {
    mode: "jit",
    optimize: true,
    opt_level: 2
}

// Iterative fibonacci (efficient)
fn fib(n: int) -> int {
    if n <= 1 {
        return n
    }

    let a = 0
    let b = 1

    for i in range(2, n + 1) {
        let temp = a + b
        a = b
        b = temp
    }

    return b
}

// Main execution
fn main() {
    for i in range(0, 20) {
        let result = fib(i)
        print("fib(" + str(i) + ") = " + str(result))
    }
}

main()
```

### Example 2: Data Processing
```tb
@config {
    mode: "jit"
}

// Process list of numbers
fn analyze_data(numbers: list) -> dict {
    let sum = 0
    let min_val = numbers[0]
    let max_val = numbers[0]

    for num in numbers {
        sum = sum + num

        if num < min_val {
            min_val = num
        }

        if num > max_val {
            max_val = num
        }
    }

    let count = len(numbers)
    let avg = float(sum) / float(count)

    return {
        sum: sum,
        min: min_val,
        max: max_val,
        avg: avg,
        count: count
    }
}

// Main
let data = [10, 25, 15, 30, 20, 35, 5]
let stats = analyze_data(data)

print("Sum:", stats.sum)
print("Min:", stats.min)
print("Max:", stats.max)
print("Avg:", stats.avg)
print("Count:", stats.count)
```

### Example 3: Multi-Language Plugin
```tb
@config {
    mode: "jit"
}

@plugin {
    python "data_processor" {
        mode: "jit",
        requires: ["numpy"],

        import numpy as np

        def normalize(data: list) -> list:
            arr = np.array(data)
            max_val = np.max(arr)
            normalized = arr / max_val
            return normalized.tolist()

        def calculate_stats(data: list) -> dict:
            arr = np.array(data)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr))
            }
    }

    javascript "formatter" {
        mode: "jit",

        function format_results(stats) {
            return `Mean: ${stats.mean.toFixed(2)}, ` +
                   `Std: ${stats.std.toFixed(2)}, ` +
                   `Median: ${stats.median.toFixed(2)}`;
        }
    }
}

// Main program
let raw_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print("Original data:", raw_data)

let normalized = data_processor.normalize(raw_data)
print("Normalized:", normalized)

let stats = data_processor.calculate_stats(raw_data)
let formatted = formatter.format_results(stats)
print("Statistics:", formatted)
```

### Example 4: Module System
**math_utils.tbx:**
```tb
// Mathematical utilities module

fn factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

fn fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

fn is_prime(n: int) -> bool {
    if n < 2 {
        return false
    }

    for i in range(2, n) {
        if n % i == 0 {
            return false
        }
    }

    return true
}
```

**string_utils.tbx:**
```tb
// String utilities module

fn reverse(s: string) -> string {
    // Implementation would need string indexing
    // This is a placeholder
    return s
}

fn repeat(s: string, count: int) -> string {
    let result = ""
    for i in range(count) {
        result = result + s
    }
    return result
}
```

**main.tbx:**
```tb
@config {
    mode: "jit"
}

@import {
    "math_utils.tbx",
    "string_utils.tbx"
}

fn main() {
    // Use math utilities
    print("factorial(5) =", factorial(5))
    print("fibonacci(10) =", fibonacci(10))
    print("is_prime(17) =", is_prime(17))

    // Use string utilities
    let greeting = repeat("Hello! ", 3)
    print(greeting)
}

main()
```

---

## Best Practices

### 1. Use Type Annotations
```tb
// Good: Clear types
fn process_data(input: list, threshold: float) -> dict {
    // ...
}

// Avoid: Unclear types
fn process_data(input, threshold) {
    // ...
}
```

### 2. Prefer Iterative Over Recursive
```tb
// Good: Iterative (faster, no stack overflow)
fn fib(n: int) -> int {
    if n <= 1 { return n }
    let a = 0
    let b = 1
    for i in range(2, n + 1) {
        let temp = a + b
        a = b
        b = temp
    }
    return b
}

// Avoid: Deep recursion
fn fib(n: int) -> int {
    if n <= 1 { return n }
    return fib(n - 1) + fib(n - 2)  // Exponential time
}
```

### 3. Early Returns
```tb
// Good: Early validation
fn divide(a: int, b: int) -> float {
    if b == 0 {
        return 0.0
    }
    return float(a) / float(b)
}

// Avoid: Nested conditions
fn divide(a: int, b: int) -> float {
    if b != 0 {
        return float(a) / float(b)
    } else {
        return 0.0
    }
}
```

### 4. Use Match for Multiple Conditions
```tb
// Good: Pattern matching
let grade = match score {
    0..60 => "F",
    60..70 => "D",
    70..80 => "C",
    80..90 => "B",
    90..100 => "A",
    _ => "Invalid"
}

// Avoid: Long if-else chains
let grade = ""
if score < 60 {
match score {
    0..60 => "F",
    60..70 => "D",
    70..80 => "C",
    80..90 => "B",
    90..100 => "A",
    _ => "Invalid"
}
// ... more conditions
```

### 5. Organize Imports
```tb
// Good: All imports at top
@import {
    "utils/math.tbx",
    "utils/string.tbx",
    "helpers/validation.tbx"
}

@config {
    mode: "jit"
}

fn main() {
    // code
}
```

### 6. Use Descriptive Names
```tb
// Good
fn calculate_average_score(scores: list) -> float {
    // ...
}

// Avoid
fn calc(s: list) -> float {
    // ...
}
```

### 7. Keep Functions Small
```tb
// Good: Single responsibility
fn validate_input(x: int) -> bool {
    return x >= 0 and x <= 100
}

fn process_input(x: int) -> int {
    if not validate_input(x) {
        return 0
    }
    return x * 2
}

// Avoid: Doing too much
fn do_everything(x: int) -> int {
    if x < 0 or x > 100 {
        return 0
    }
    let result = x * 2
    print("Processed:", result)
    // ... more logic
    return result
}
```

### 8. Choose Right Execution Mode
```tb
// Development: JIT mode
@config {
    mode: "jit",
    debug: true
}

// Production: Compiled mode
@config {
    mode: "compile",
    optimize: true,
    opt_level: 3
}
```

---

## Performance Tips

### 1. Use JIT for Development
```tb
@config {
    mode: "jit"  // Fast iteration
}
```

### 2. Compile for Production
```tb
@config {
    mode: "compile",
    opt_level: 3  // Maximum performance
}
```

### 3. Enable Caching
```tb
@config {
    cache: true  // 90%+ cache hit rate
}
```

### 4. Avoid Deep Recursion
```tb
// Prefer iteration over deep recursion
// Iterative uses O(1) space, recursive uses O(n) stack
```

### 5. Use Appropriate Data Structures
```tb
// Lists: Sequential access, iteration
let items = [1, 2, 3, 4, 5]

// Dictionaries: Key-value lookup
let data = {key1: "value1", key2: "value2"}
```

### 6. Minimize String Concatenation in Loops
```tb
// Less efficient
let result = ""
for i in range(1000) {
    result = result + str(i)  // Creates new string each time
}

// More efficient (use list and join if available)
let parts = []
for i in range(1000) {
    parts = push(parts, str(i))
}
// Then join parts
```

---

## Common Patterns

### Factory Pattern
```tb
fn create_person(name: string, age: int) -> dict {
    return {
        name: name,
        age: age,
        type: "Person"
    }
}

let person1 = create_person("Alice", 30)
let person2 = create_person("Bob", 25)
```

### Strategy Pattern
```tb
fn apply_operation(op: string, a: int, b: int) -> int {
    return match op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => a / b,
        _ => 0
    }
}
```

### Builder Pattern
```tb
fn build_config(base: dict) -> dict {
    let config = base

    if not has_key(config, "optimize") {
        config.optimize = true
    }

    if not has_key(config, "cache") {
        config.cache = true
    }

    return config
}
```

### Iterator Pattern
```tb
fn process_items(items: list, processor: function) {
    for item in items {
        processor(item)
    }
}

fn print_item(item) {
    print(item)
}

process_items([1, 2, 3, 4, 5], print_item)
```

---

## Tooling

### Command Line Interface

#### Run Script
```bash
# JIT mode (default)
tb run script.tbx

# Compiled mode
tb run script.tbx --mode compile

# With optimization
tb run script.tbx --opt-level 3
```

#### Compile to Binary
```bash
# Basic compilation
tb compile script.tbx -o output

# With optimization level
tb compile script.tbx -o output --opt-level 3

# With debug symbols
tb compile script.tbx -o output --debug
```

#### Interactive REPL
```bash
# Start REPL
tb repl

# REPL with specific mode
tb repl --mode jit
```

#### Cache Management
```bash
# Show cache statistics
tb cache stats

# Clear cache
tb cache clear

# Show cache location
tb cache info
```

#### Version Info
```bash
tb version
tb --version
tb -v
```

### Environment Variables

```bash
# Enable debug logging
export TB_DEBUG=1

# Set cache directory
export TB_CACHE_DIR=/path/to/cache

# Disable caching
export TB_NO_CACHE=1

# Set optimization level
export TB_OPT_LEVEL=3
```

---

## Troubleshooting

### Common Errors

#### "Undefined variable"
```tb
// ERROR
print(x)

// FIX
let x = 42
print(x)
```

#### "Type mismatch"
```tb
// ERROR
let x: int = "hello"

// FIX
let x: string = "hello"
// OR
let x: int = 42
```

#### "Division by zero"
```tb
// ERROR
let result = 10 / 0

// FIX
fn safe_divide(a: int, b: int) -> float {
    if b == 0 {
        return 0.0
    }
    return float(a) / float(b)
}
```

#### "Index out of bounds"
```tb
// ERROR
let items = [1, 2, 3]
print(items[10])

// FIX
let items = [1, 2, 3]
if 10 < len(items) {
    print(items[10])
} else {
    print("Index out of range")
}
```

### Performance Issues

#### Slow Startup
- Use JIT mode instead of compiled for development
- Enable caching: `@config { cache: true }`

#### Slow Execution
- Use compiled mode: `@config { mode: "compile" }`
- Enable optimizations: `@config { optimize: true, opt_level: 3 }`

#### High Memory Usage
- Avoid creating large lists/dictionaries
- Use iteration instead of building large intermediate structures

---

## Appendix

### Grammar (Simplified BNF)

```bnf
program        ::= statement*

statement      ::= config_block
                 | import_block
                 | plugin_block
                 | fn_decl
                 | let_decl
                 | assignment
                 | if_stmt
                 | while_stmt
                 | for_stmt
                 | match_stmt
                 | return_stmt
                 | break_stmt
                 | continue_stmt
                 | expr_stmt

config_block   ::= "@config" "{" config_item* "}"
import_block   ::= "@import" "{" string_literal ("," string_literal)* "}"
plugin_block   ::= "@plugin" "{" plugin_def+ "}"

fn_decl        ::= "fn" identifier "(" params? ")" ("->" type)? block
let_decl       ::= "let" identifier (":" type)? "=" expr
assignment     ::= identifier "=" expr

if_stmt        ::= "if" expr block ("else" (if_stmt | block))?
while_stmt     ::= "while" expr block
for_stmt       ::= "for" identifier "in" expr block
match_stmt     ::= "match" expr "{" match_arm+ "}"
match_arm      ::= pattern "=>" (expr | block) ","?

return_stmt    ::= "return" expr?
break_stmt     ::= "break"
continue_stmt  ::= "continue"

expr           ::= literal
                 | identifier
                 | fn_call
                 | binary_op
                 | unary_op
                 | list_literal
                 | dict_literal

type           ::= "int" | "float" | "string" | "bool" | "list" | "dict" | "function"
```

### Operator Precedence Table

| Precedence | Operator | Description | Associativity |
|------------|----------|-------------|---------------|
| 1 (highest) | `()` | Parentheses | N/A |
| 2 | `not` | Logical NOT | Right |
| 3 | `*` `/` `%` | Multiplication, Division, Modulo | Left |
| 4 | `+` `-` | Addition, Subtraction | Left |
| 5 | `==` `!=` `<` `>` `<=` `>=` | Comparison | Left |
| 6 | `and` | Logical AND | Left |
| 7 (lowest) | `or` | Logical OR | Left |

### Reserved Keywords

```
and       break     continue  else      false
fn        for       if        import    in
let       match     not       or        return
true      while     config    plugin
```

### File Extensions

- `.tbx` - ToolBox source file (preferred)
- `.tb` - ToolBox source file (alternative)

### Version History

- **v0.1.24** - Current version
  - Plugin system (Python, JS, Rust, Go)
  - Cache system (92% hit rate)
  - JIT and Compiled modes
  - 99.2% test success rate

---

## Quick Reference Card

### Variable Declaration
```tb
let x = 42                    // Inferred type
let name: string = "Alice"    // Explicit type
x = 100                       // Reassignment
```

### Functions
```tb
fn add(a: int, b: int) -> int {
    return a + b
}
```

### Control Flow
```tb
if condition { }              // If
while condition { }           // While
for i in range(10) { }        // For
match x { 0 => "zero" }       // Match
```

### Data Structures
```tb
let list = [1, 2, 3]          // List
let dict = {key: "value"}     // Dictionary
```

### Operators
```tb
+ - * / %                     // Arithmetic
== != < > <= >=               // Comparison
and or not                    // Logical
```

### Builtins
```tb
print(x)                      // Print
len(list)                     // Length
type_of(x)                    // Get type
str(42) int("42") float(3)    // Convert
```

### Blocks
```tb
@config { mode: "jit" }       // Configuration
@import { "module.tbx" }      // Import modules
@plugin { python "name" { } } // Plugins
```

---

**ToolBox Language Specification v0.1.24**
*Complete reference for humans and LLMs*
*Built with Rust 🦀 | Production Ready ✅*
