# TB Language - LLM Programming Reference

## Quick Syntax Matrix

```
LITERALS: 42 | 3.14 | "str" | true | false | [] | {}
KEYWORDS: let fn if else for in while match return break continue and or not import plugin config
OPERATORS: + - * / % == != < > <= >= and or not in . [] .. ..=
BLOCKS: @config @import @plugin
COMMENTS: // single-line
```

## Type System (Dynamic + Optional Static)

```
None Bool Int(i64) Float(f64) String List Dict Function
let x = 42          // inferred int
let x: int = 42     // annotated
```

## Variable Declaration

```
let name = value
let name: type = value
name = new_value    // reassignment
```

## Functions

```
fn name(p1, p2) { body }                    // basic
fn name(p1: type) -> type { body }          // typed
x => expr                                   // lambda 1-param
(x, y) => expr                              // lambda multi-param
fn(x, y) { expr }                           // lambda traditional
```

## Control Flow

```
if cond { } else { }
for item in iterable { }
while cond { }
match val { pat1 => expr1, _ => default }
break | continue | return [value]
```

## Patterns (match)

```
42              // literal
x               // binding
_               // wildcard
1..10           // exclusive range
1..=10          // inclusive range
```

## Collections

```
LIST: [1, 2, 3] | list[idx] | push(list, val) | pop(list) | len(list)
DICT: {key: val, k2: v2} | dict.key | dict["key"] | keys(dict) | values(dict)
```

## Operators Precedence (High→Low)

```
1. . [] ()
2. not -unary
3. * / %
4. + -
5. < > <= >=
6. == !=
7. in
8. and
9. or
10. =
```

## Built-ins Essential

```
TYPE: int() float() str() dict() list() type_of()
COLLECTION: len() push() pop() keys() values() range(n) range(start,end)
IO: print(...) read_file(p) write_file(p,c) file_exists(p) open(p,mode)
SYSTEM: execute(cmd,...) get_env(v) sleep(s)
JSON: json_parse(s) json_stringify(v,pretty) yaml_parse(s) yaml_stringify(v)
TIME: time(tz?)
HTTP: http_session(url,h?,c?) http_request(sid,url,method,data?)
ASYNC: spawn(fn,args) await_task(tid) cancel_task(tid)
FUNCTIONAL: map(fn,list) filter(fn,list) reduce(fn,list,init) forEach(fn,list)
```

## Truthiness

```
FALSE: None 0 0.0 "" [] {}
TRUE: everything else
```

## Type Coercion

```
Int + Float → Float (auto)
```

## String Operations

```
"a" + "b" → "ab"
"sub" in "substring" → true
str(val) → string conversion
```

## Imports

```
@import {
    "path/to/mod.tb"
    "path/other.tb" as alias
}
```

## Plugins (Python/JS/Go/Rust)

```
@plugin {
    python "modname" {
        mode: "jit"                     // or "compile"
        requires: ["dep1", "dep2"]
        file: "path.py"                 // or inline code
        def func(x): return x*2
    }
}
// Use: modname.func(5)
```

## Config

```
@config {
    threads: 4
    optimize: true
    networking: true
}
```

## CLI Commands

```
tb run script.tb                        // JIT execute
tb run --mode=jit script.tb             // explicit JIT
tb compile -o binary script.tb          // AOT compile
tb compile --optimize script.tb         // optimized build
```

## Memory Model

- Reference counting (Arc)
- No GC pauses
- Persistent data structures (structural sharing)
- Zero-copy where possible

## Error Handling

- No try/catch
- Errors propagate with detailed messages
- Stack traces included
- Source context shown

## Common Patterns

### Iteration

```
for i in range(10) { print(i) }
for item in [1,2,3] { print(item) }
```

### Closures

```
fn make_adder(n) { return x => x + n }
let add5 = make_adder(5)
print(add5(10))  // 15
```

### Higher-Order

```
let nums = [1,2,3,4,5]
let doubled = map(x => x * 2, nums)
let evens = filter(x => x % 2 == 0, nums)
let sum = reduce((a,x) => a + x, nums, 0)
```

### File I/O

```
write_file("f.txt", "data")
let data = read_file("f.txt")
if file_exists("f.txt") { /* ... */ }
```

### JSON

```
let obj = {name: "Alice", age: 30}
let json = json_stringify(obj, true)
let parsed = json_parse(json)
```

### HTTP

```
let sess = http_session("https://api.com", {}, None)
let resp = http_request(sess, "/endpoint", "GET", None)
print(resp.status, resp.body)
```

### Async Tasks

```
fn work(n) { sleep(1); return n * 2 }
let tid = spawn(work, [42])
let result = await_task(tid)
```

### Pattern Matching

```
match value {
    0 => "zero",
    1..=10 => "small",
    _ => "other"
}
```

## Type Annotations (Optional)

```
fn add(a: int, b: int) -> int { a + b }
let x: string = "hello"
let y: list = [1, 2, 3]
let z: dict = {key: "value"}
```

## Scoping

- Block scope
- Function scope
- Closures capture environment
- No hoisting

## Key Constraints

- Semicolons optional
- Braces required for blocks
- Case-sensitive
- No implicit type conversion except Int+Float→Float
- Dict keys must be strings
- Zero-indexed arrays
- String/list/dict are reference-counted

## Expression vs Statement

Most constructs are expressions (return values):
- if/else returns value of taken branch
- match returns matched expression
- Functions return last expression (or None)
- Blocks return last expression

## Special Syntax

```
@prefix for special declarations
. for member access
[] for indexing
.. for exclusive range
..= for inclusive range
=> for lambda/match arms
-> for function return types
```

## Compilation Modes

**JIT (Interpret):** Fast startup, development
**AOT (Compile):** Native binary, production

## Plugin Modes

**jit:** Interpret at runtime (fast compilation)
**compile:** Native code (fast execution)

## Networking

```
create_server(proto, addr, callbacks) → server_id
connect_to(on_conn, on_disc, on_msg, host, port, type)
send_to(conn_id, msg)
stop_server(server_id)
```

## Cache Management

```
cache_stats() cache_clear(name?) cache_invalidate(path)
```

## Plugin Management

```
list_plugins() unload_plugin(name) reload_plugin(name) plugin_info(name)
```

## Complete Example

```
@config { threads: 4 }

fn factorial(n) {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}

let nums = [1, 2, 3, 4, 5]
let doubled = map(x => x * 2, nums)
let sum = reduce((a, x) => a + x, doubled, 0)

print("Factorial(5):", factorial(5))
print("Sum:", sum)

match sum {
    0 => print("zero"),
    1..=20 => print("small"),
    _ => print("large")
}
```

## Performance Tips

- Use persistent data structures (automatic)
- Compile for production (`tb compile`)
- Enable optimization (`@config { optimize: true }`)
- Use plugins for CPU-intensive tasks
- Spawn tasks for parallelism
- Cache results when appropriate

## Language Philosophy

- Performance + Developer Experience
- Gradual typing (optional annotations)
- Minimal boilerplate
- Expression-oriented
- Multi-paradigm (imperative + functional)
- Zero-cost abstractions where possible

## Common Gotchas

- Dict keys must be strings (not ints)
- Range `1..10` excludes 10 (use `1..=10` to include)
- Variables are reassignable (no const keyword)
- No automatic string→number conversion
- Empty collections are falsy
- Function calls need parens even with no args: `func()`

## Quick Reference Card

```
VARIABLE: let x = val
FUNCTION: fn f(p) { body } | x => expr
IF: if c { } else { }
LOOP: for x in iter { } | while c { }
MATCH: match v { pat => expr, _ => def }
LIST: [a, b, c] | lst[i] | push(lst, v)
DICT: {k: v} | d.k | d["k"] | keys(d)
LAMBDA: x => x*2 | (x,y) => x+y
IMPORT: @import { "path.tb" }
PLUGIN: @plugin { lang "name" { code } }
IO: print(...) read_file(p) write_file(p, c)
JSON: json_parse(s) json_stringify(v)
ASYNC: spawn(f, args) await_task(tid)
FUNC: map filter reduce forEach
```

---

TB is a dynamic, multi-paradigm language with optional typing, reference-counted memory, pattern matching, closures, async tasks, and multi-language plugins. Syntax: `let x = 42`, `fn f(p) { }`, `if c { }`, `for x in lst { }`, `match v { p => e }`. Built-ins: print, range, map, filter, json_parse, http_request, spawn. Compile: `tb compile` or run: `tb run`. Plugins: `@plugin { python "m" { code } }`. Zero-indexed, expression-oriented, no semicolons needed.
