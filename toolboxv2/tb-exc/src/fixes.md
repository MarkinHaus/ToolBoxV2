
[----------------------------------------] 6/332 (1.8%)
[Arrow Functions]
  Testing: arrow function - with block [     jit] OK (20ms)
  Testing: arrow function - with block [compiled] FAIL (compile: 2114ms/2115ms)
[=---------------------------------------] 16/332 (4.8%)
[Builtins]
  Testing: Builtin - type_of for all types [     jit] OK (20ms)
  Testing: Builtin - type_of for all types [compiled] FAIL (compile: 144ms/145ms)
[=====-----------------------------------] 43/332 (13.0%)
[Cache - String Interning]
  Testing: Cache: String interning statistics [     jit] FAIL (20ms)
  Testing: Cache: String interning statistics [compiled] OK (669ms/934ms)
[=====-----------------------------------] 46/332 (13.9%)
[Functions]
  Testing: Closure capturing variable [     jit] OK (21ms)
  Testing: Closure capturing variable [compiled] FAIL (compile: 342ms/342ms)
[======----------------------------------] 50/332 (15.1%)
[Integration]
  Testing: Complex program - closure with state [     jit] FAIL (21ms)
  Testing: Complex program - closure with state [compiled] FAIL (compile: 437ms/438ms)
  Testing: Complex program - fibonacci [     jit] FAIL (468ms)
  Testing: Complex program - fibonacci [compiled] FAIL (compile: 14ms/15ms)
  Testing: Complex program - match with ranges [     jit] FAIL (123ms)
  Testing: Complex program - match with ranges [compiled] FAIL (compile: 14ms/16ms)
  Testing: Complex program - nested data structures [     jit] OK (22ms)
  Testing: Complex program - nested data structures [compiled] FAIL (compile: 711ms/712ms)
  Testing: Complex program - recursive list sum [     jit] FAIL (118ms)
  Testing: Complex program - recursive list sum [compiled] FAIL (compile: 13ms/14ms)
  Testing: Complex program - string manipulation [     jit] OK (21ms)
  Testing: Complex program - string manipulation [compiled] FAIL (compile: 601ms/601ms)
[=======---------------------------------] 63/332 (19.0%)
[Control]
  Testing: Control Flow [     jit] FAIL (183ms)
  Testing: Control Flow [compiled] FAIL (compile: 13ms/2907ms)
[========--------------------------------] 72/332 (21.7%)
[Dictionaries]
  Testing: Dict iteration over keys [     jit] FAIL (19ms)
  Testing: Dict iteration over keys [compiled] FAIL (1031ms/1440ms)
[=========-------------------------------] 80/332 (24.1%)
[Error Handling]
  Testing: Division by zero [     jit] OK (162ms)
  Testing: Division by zero [compiled] FAIL (409ms/881ms)
[=========-------------------------------] 82/332 (24.7%)
[EdgeCases]
  Testing: Edge case - empty function body [     jit] OK (21ms)
  Testing: Edge case - empty function body [compiled] FAIL (compile: 182ms/183ms)
[===========-----------------------------] 94/332 (28.3%)
[Literals]
  Testing: Empty list literal [     jit] OK (21ms)
  Testing: Empty list literal [compiled] FAIL (compile: 160ms/161ms)
[===========-----------------------------] 98/332 (29.5%)
[Errors]
  Testing: Error - division by zero [     jit] OK (153ms)
  Testing: Error - division by zero [compiled] FAIL (593ms/1132ms)
  Testing: Error Handling [     jit] OK (682ms)32 (29.8%)
  Testing: Error Handling [compiled] FAIL (500ms/1126ms)
[============----------------------------] 107/332 (32.2%)
[IO]
  Testing: File I/O [     jit] OK (218ms)
  Testing: File I/O [compiled] FAIL (971ms/5767ms)
[=============---------------------------] 113/332 (34.0%)
[Basic]
  Testing: Float arithmetic [     jit] FAIL (158ms)
  Testing: Float arithmetic [compiled] OK (515ms/1007ms)
[===============-------------------------] 125/332 (37.7%)
[Functions]
  Testing: Function returning None [     jit] OK (23ms)
  Testing: Function returning None [compiled] FAIL (compile: 212ms/213ms)
[===============-------------------------] 128/332 (38.6%)
[AdvancedFunctions]
  Testing: Function - returning function [     jit] OK (23ms)
  Testing: Function - returning function [compiled] FAIL (compile: 207ms/208ms)
[===============-------------------------] 130/332 (39.2%)
[Functions]
  Testing: Functions and Closures [     jit] OK (63ms)
  Testing: Functions and Closures [compiled] FAIL (compile: 306ms/2568ms)
[=================-----------------------] 147/332 (44.3%)
[Arrow Functions]
  Testing: inline function syntax - with map [     jit] FAIL (25ms)
  Testing: inline function syntax - with map [compiled] FAIL (compile: 254ms/255ms)
[===================---------------------] 158/332 (47.6%)
[Integration]
  Testing: Integration: Quicksort algorithm [     jit] FAIL (26ms)
  Testing: Integration: Quicksort algorithm [compiled] OK (889ms/1390ms)
[====================--------------------] 166/332 (50.0%)
[Serialization]
  Testing: JSON and YAML Operations [     jit] OK (167ms)
  Testing: JSON and YAML Operations [compiled] FAIL (535ms/6345ms)
[====================--------------------] 168/332 (50.6%)
[AdvancedFunctions]
  Testing: function - as function argument [     jit] FAIL (166ms)
  Testing: function - as function argument [compiled] FAIL (compile: 18ms/19ms)
[====================--------------------] 171/332 (51.5%)
[Functions]
  Testing: Lambda traditional syntax [     jit] OK (22ms)
  Testing: Lambda traditional syntax [compiled] FAIL (compile: 1063ms/1064ms)
[=====================-------------------] 176/332 (53.0%)
[Types]
  Testing: List constructor [     jit] OK (22ms)
  Testing: List constructor [compiled] FAIL (398ms/708ms)
[======================------------------] 186/332 (56.0%)
[Lists]
  Testing: Pop from list [     jit] OK (21ms)
  Testing: Pop from list [compiled] FAIL (361ms/646ms)
[======================------------------] 188/332 (56.6%)
[Fundamentals]
  Testing: Literals and Basic Types [     jit] OK (194ms)
  Testing: Literals and Basic Types [compiled] FAIL (compile: 161ms/162ms)
[==========================--------------] 216/332 (65.1%)
[Lists]
  Testing: Nested lists [     jit] OK (20ms)
  Testing: Nested lists [compiled] FAIL (compile: 185ms/187ms)
[==========================--------------] 219/332 (66.0%)
[Literals]
  Testing: None literal [     jit] OK (20ms)
  Testing: None literal [compiled] FAIL (compile: 147ms/148ms)
[==============================----------] 254/332 (76.5%)
[Builtins]
  Testing: pop function [     jit] OK (22ms)
  Testing: pop function [compiled] FAIL (compile: 171ms/172ms)
[=================================-------] 274/332 (82.5%)
[RealWorld]
  Testing: Real program - grade calculator [     jit] FAIL (145ms)
  Testing: Real program - grade calculator [compiled] FAIL (compile: 193ms/194ms)
  Testing: Real program - text processing [     jit] OK (21ms)
  Testing: Real program - text processing [compiled] FAIL (compile: 164ms/165ms)
[=================================-------] 277/332 (83.4%)
[Recursion]
  Testing: Recursion - sum of list [     jit] OK (19ms)
  Testing: Recursion - sum of list [compiled] FAIL (compile: 171ms/172ms)
[==================================------] 285/332 (85.8%)
[Scope]
  Testing: Scope - nested blocks [     jit] FAIL (20ms)
  Testing: Scope - nested blocks [compiled] FAIL (389ms/730ms)
[====================================----] 307/332 (92.5%)
[Truthiness]
  Testing: Truthiness - None is falsy [     jit] OK (20ms)
  Testing: Truthiness - None is falsy [compiled] FAIL (compile: 166ms/167ms)
[=======================================-] 325/332 (97.9%)
[Variables]
  Testing: Variable shadowing in scope [     jit] FAIL (20ms)
  Testing: Variable shadowing in scope [compiled] FAIL (472ms/821ms)
[=======================================-] 327/332 (98.5%)
[Fundamentals]
  Testing: Variable Declaration and Scope [     jit] FAIL (42ms)
  Testing: Variable Declaration and Scope [compiled] FAIL (787ms/1961ms)
[========================================] 332/332 (100.0%)

================================================================================
TEST SUMMARY
================================================================================
FAILED - 52 of 80 tests failed
OK - 28 passed

Total time: 41805.14ms
JIT avg time: 80.68ms
Compiled avg time: 1110.50ms (compile: 418.46ms, exec: 690.97ms)

Failed tests:
  - arrow function - with block (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
   Compiling tb-runtime v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime)
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0277]: `()` doesn't implement `std::fmt::Display`
   --> src\main.rs:17:11
    |
 17 |     print(&triple(4));
    |     ----- ^^^^^^^^^^ the trait `std::fmt::Display` is not implemented for `()`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

error[E0277]: `()` doesn't implement `std::fmt::Display`
   --> src\main.rs:18:11
    |
 18 |     print(&triple(7));
    |     ----- ^^^^^^^^^^ the trait `std::fmt::Display` is not implemented for `()`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 2 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Builtin - type_of for all types (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0282]: type annotations needed
  --> src\main.rs:19:21
   |
19 |     print(&type_of(&None));
   |                     ^^^^ cannot infer type of the type parameter `T` declared on the enum `Option`
   |
help: consider specifying the generic argument
   |
19 |     print(&type_of(&None::<T>));
   |                         +++++

For more information about this error, try `rustc --explain E0282`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Cache: String interning statistics (jit)
    Output mismatch:
Expected: '3'
Got: '0'
  - Closure capturing variable (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:17:17
   |
17 |     print(&add5(10));
   |            ---- ^^ expected `String`, found integer
   |            |
   |            arguments to this function are incorrect
   |
help: try using a conversion method
   |
17 |     print(&add5(10.to_string()));
   |                   ++++++++++++

error[E0308]: mismatched types
  --> src\main.rs:14:25
   |
14 |         return |x| (x + n);
   |                         ^ expected `&str`, found `i64`

error[E0308]: mismatched types
  --> src\main.rs:14:20
   |
14 |         return |x| (x + n);
   |                    ^^^^^^^ expected `i64`, found `String`

error[E0308]: mismatched types
  --> src\main.rs:14:16
   |
13 |     fn make_adder(n: i64) -> fn(String) -> i64 {
   |                              ----------------- expected `fn(std::string::String) -> i64` because of return type
14 |         return |x| (x + n);
   |                ^^^^^^^^^^^ expected fn pointer, found closure
   |
   = note: expected fn pointer `fn(std::string::String) -> i64`
                 found closure `{closure@src\main.rs:14:16: 14:19}`
note: closures can only be coerced to `fn` types if they do not capture any variables
  --> src\main.rs:14:25
   |
14 |         return |x| (x + n);
   |                         ^ `n` captured here

For more information about this error, try `rustc --explain E0308`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 4 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Complex program - closure with state (jit)
    Output does not contain '1
2
3':
1
1
1

  - Complex program - closure with state (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:17:32
   |
17 |                         return count;
   |                                ^^^^^ expected `String`, found integer
   |
note: return type inferred to be `std::string::String` here
  --> src\main.rs:17:32
   |
17 |                         return count;
   |                                ^^^^^
help: try using a conversion method
   |
17 |                         return count.to_string();
   |                                     ++++++++++++

error[E0308]: mismatched types
  --> src\main.rs:15:16
   |
13 |       fn make_counter() -> fn() -> String {
   |                            -------------- expected `fn() -> std::string::String` because of return type
14 |           let count = 0;
15 |           return || {
   |  ________________^
16 | |                         count = (count + 1);
17 | |                         return count;
18 | |         };
   | |_________^ expected fn pointer, found closure
   |
   = note: expected fn pointer `fn() -> std::string::String`
                 found closure `{closure@src\main.rs:15:16: 15:18}`
note: closures can only be coerced to `fn` types if they do not capture any variables
  --> src\main.rs:16:34
   |
16 |                         count = (count + 1);
   |                                  ^^^^^ `count` captured here

For more information about this error, try `rustc --explain E0308`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 2 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Complex program - fibonacci (jit)
    Execution failed:
Error: Type error: Cannot apply Add to types None and None

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - Complex program - fibonacci (compiled)
    Execution failed:

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Type Error: Cannot apply Add to types None and None

════════════════════════════════════════════════════════════════════════════════


  - Complex program - match with ranges (jit)
    Execution failed:
Error: Type error: Pattern type Int doesn't match value type Generic("n")

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - Complex program - match with ranges (compiled)
    Execution failed:

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Type Error: Pattern type Int doesn't match value type Generic("n")

Location:
  File: C:\Users\Markin\AppData\Local\Temp\tmp5w2z8jfg.tbx
  Line: 3, Column: 12

   2 | fn classify(n) {
   3 |     return match n {
     |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   4 |         0 => "zero",

════════════════════════════════════════════════════════════════════════════════


  - Complex program - nested data structures (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0599]: no method named `iter` found for enum `tb_runtime::DictValue` in the current scope
  --> src\main.rs:15:78
   |
15 | ...   let avg = ((user.get("scores").cloned().unwrap_or(DictValue::Int(0)).iter().fold(0, |a, &x| (a + x)) as f64) / (len(user.get("score...
   |                   ---- ------------- --------                              ^^^^ method not found in `tb_runtime::DictValue`
   |                   |    |             |
   |                   |    |             method `iter` is available on `Option<tb_runtime::DictValue>`
   |                   |    method `iter` is available on `Option<&tb_runtime::DictValue>`
   |                   method `iter` is available on `&HashMap<std::string::String, tb_runtime::DictValue>`

For more information about this error, try `rustc --explain E0599`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Complex program - recursive list sum (jit)
    Execution failed:
Error: Syntax error at 6:38: Expected Comma, found For

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - Complex program - recursive list sum (compiled)
    Execution failed:

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Syntax Error: Expected Comma, found For

Location:
  File: C:\Users\Markin\AppData\Local\Temp\tmp3m43mjmd.tbx
  Line: 6, Column: 38

   5 |     }
   6 |     return lst[0] + sum_list([lst[i] for i in range(1, len(lst))])
     |                                      ^^^
   7 | }

Hint: Check for missing brackets, parentheses, or semicolons

════════════════════════════════════════════════════════════════════════════════


  - Complex program - string manipulation (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0507]: cannot move out of a shared reference
  --> src\main.rs:14:37
   |
14 |     let lengths = words.iter().map(|&x| len(&x)).collect::<Vec<_>>();
   |                                     ^-
   |                                      |
   |                                      data moved here
   |                                      move occurs because `x` has type `std::string::String`, which does not implement the `Copy` trait
   |
help: consider removing the borrow
   |
14 -     let lengths = words.iter().map(|&x| len(&x)).collect::<Vec<_>>();
14 +     let lengths = words.iter().map(|x| len(&x)).collect::<Vec<_>>();
   |

For more information about this error, try `rustc --explain E0507`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Control Flow (jit)
    Execution failed:
Error: Undefined variable: x

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - Control Flow (compiled)
    Execution failed:

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Undefined Variable: Variable 'x' is not defined

Hint: Did you forget to declare 'x'? Use: let x = ...

════════════════════════════════════════════════════════════════════════════════


  - Dict iteration over keys (jit)
    Output does not contain 'a
b':
b
a

  - Dict iteration over keys (compiled)
    Output does not contain 'a
b':
b
a

  - Division by zero (compiled)
    Expected failure but succeeded:

  - Edge case - empty function body (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0277]: `()` doesn't implement `std::fmt::Display`
   --> src\main.rs:15:11
    |
 15 |     print(&empty());
    |     ----- ^^^^^^^^ the trait `std::fmt::Display` is not implemented for `()`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Empty list literal (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0277]: `Vec<i64>` doesn't implement `std::fmt::Display`
   --> src\main.rs:13:11
    |
 13 |     print(&Vec::<i64>::new());
    |     ----- ^^^^^^^^^^^^^^^^^^ the trait `std::fmt::Display` is not implemented for `Vec<i64>`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Error - division by zero (compiled)
    Expected failure but succeeded:
inf

  - Error Handling (compiled)
    Expected failure but succeeded:
inf

  - File I/O (compiled)
    Expected failure but succeeded:

  - Float arithmetic (jit)
    Execution failed:
[TB JIT] Starting JIT execution with 7 statements
[TB JIT] Statement 1: Line 2 | let a = 10.5
[TB JIT] Statement 2: Line 3 | let b = 2.5
[TB JIT] Statement 3: Line 4 | print(a + b)
[TB JIT] Statement 4: Line 5 | print(a - b)
[TB JIT] Statement 5: Line 6 | print(a * b)
[TB JIT] Statement 6: Line 7 | print(a / b)
[TB JIT] Statement 7: Line 8 | print(a % b)
Error: Runtime error: Invalid modulo operation

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - Function returning None (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:14:9
   |
13 |     fn do_nothing() -> String {
   |                        ------ expected `std::string::String` because of return type
14 |         None
   |         ^^^^ expected `String`, found `Option<_>`
   |
   = note: expected struct `std::string::String`
                found enum `Option<_>`

For more information about this error, try `rustc --explain E0308`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Function - returning function (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:17:19
   |
17 |     print(&times3(7));
   |            ------ ^ expected `String`, found integer
   |            |
   |            arguments to this function are incorrect
   |
help: try using a conversion method
   |
17 |     print(&times3(7.to_string()));
   |                    ++++++++++++

error[E0369]: cannot multiply `std::string::String` by `i64`
   --> src\main.rs:14:23
    |
 14 |         return |x| (x * factor);
    |                     - ^ ------ i64
    |                     |
    |                     std::string::String
    |
note: the foreign item type `std::string::String` doesn't implement `Mul<i64>`
   --> C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\alloc\src\string.rs:360:1
    |
360 | pub struct String {
    | ^^^^^^^^^^^^^^^^^ not implement `Mul<i64>`

error[E0308]: mismatched types
  --> src\main.rs:14:16
   |
13 |     fn multiplier(factor: i64) -> fn(String) -> i64 {
   |                                   ----------------- expected `fn(std::string::String) -> i64` because of return type
14 |         return |x| (x * factor);
   |                ^^^^^^^^^^^^^^^^ expected fn pointer, found closure
   |
   = note: expected fn pointer `fn(std::string::String) -> i64`
                 found closure `{closure@src\main.rs:14:16: 14:19}`
note: closures can only be coerced to `fn` types if they do not capture any variables
  --> src\main.rs:14:25
   |
14 |         return |x| (x * factor);
   |                         ^^^^^^ `factor` captured here

Some errors have detailed explanations: E0308, E0369.
For more information about an error, try `rustc --explain E0308`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 3 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Functions and Closures (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:17:17
   |
17 |     print(&add5(10));
   |            ---- ^^ expected `String`, found integer
   |            |
   |            arguments to this function are incorrect
   |
help: try using a conversion method
   |
17 |     print(&add5(10.to_string()));
   |                   ++++++++++++

error[E0308]: mismatched types
  --> src\main.rs:14:25
   |
14 |         return |x| (x + n);
   |                         ^ expected `&str`, found `i64`

error[E0308]: mismatched types
  --> src\main.rs:14:20
   |
14 |         return |x| (x + n);
   |                    ^^^^^^^ expected `i64`, found `String`

error[E0308]: mismatched types
  --> src\main.rs:14:16
   |
13 |     fn make_adder(n: i64) -> fn(String) -> i64 {
   |                              ----------------- expected `fn(std::string::String) -> i64` because of return type
14 |         return |x| (x + n);
   |                ^^^^^^^^^^^ expected fn pointer, found closure
   |
   = note: expected fn pointer `fn(std::string::String) -> i64`
                 found closure `{closure@src\main.rs:14:16: 14:19}`
note: closures can only be coerced to `fn` types if they do not capture any variables
  --> src\main.rs:14:25
   |
14 |         return |x| (x + n);
   |                         ^ `n` captured here

For more information about this error, try `rustc --explain E0308`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 4 previous errors


════════════════════════════════════════════════════════════════════════════════


  - inline function syntax - with map (jit)
    Output mismatch:
Expected: '5\n3\n15'
Got: '5\nNone\nNone'
  - inline function syntax - with map (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0277]: `()` doesn't implement `std::fmt::Display`
   --> src\main.rs:19:11
    |
 19 |     print(&tripled[(0 as usize)].clone());
    |     ----- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `std::fmt::Display` is not implemented for `()`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

error[E0277]: `()` doesn't implement `std::fmt::Display`
   --> src\main.rs:20:11
    |
 20 |     print(&tripled[(4 as usize)].clone());
    |     ----- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `std::fmt::Display` is not implemented for `()`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 2 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Integration: Quicksort algorithm (jit)
    Output mismatch:
Expected: '9'
Got: '1'
  - JSON and YAML Operations (compiled)
    Expected failure but succeeded:

  - function - as function argument (jit)
    Execution failed:
Error: Type error: Cannot call non-function type Generic("f")

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - function - as function argument (compiled)
    Execution failed:

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Type Error: Cannot call non-function type Generic("f")

Location:
  File: C:\Users\Markin\AppData\Local\Temp\tmpyay79izn.tbx
  Line: 3, Column: 12

   2 | fn apply(f, x) {
   3 |     return f(x)
     |            ^^^^
   4 | }

════════════════════════════════════════════════════════════════════════════════


  - Lambda traditional syntax (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
   Compiling tb-runtime v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime)
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0277]: `()` doesn't implement `std::fmt::Display`
   --> src\main.rs:17:11
    |
 17 |     print(&triple(4));
    |     ----- ^^^^^^^^^^ the trait `std::fmt::Display` is not implemented for `()`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - List constructor (compiled)
    Output does not contain 'list':
alloc::vec::Vec<tb_runtime::DictValue>

  - Pop from list (compiled)
    Output does not contain '3
[1, 2, 3]
[1, 2]':
2
[1, 2, 3]
1

  - Literals and Basic Types (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0282]: type annotations needed
  --> src\main.rs:18:21
   |
18 |     print(&type_of(&None));
   |                     ^^^^ cannot infer type of the type parameter `T` declared on the enum `Option`
   |
help: consider specifying the generic argument
   |
18 |     print(&type_of(&None::<T>));
   |                         +++++

For more information about this error, try `rustc --explain E0282`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Nested lists (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0599]: no method named `as_list` found for struct `Vec<{integer}>` in the current scope
    --> src\main.rs:14:41
     |
  14 |     print(&matrix[(1 as usize)].clone().as_list()[(0 as usize)].clone());
     |                                         ^^^^^^^
     |
help: there is a method `split` with a similar name, but with different arguments
    --> C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\slice\mod.rs:2238:5
     |
2238 | /     pub fn split<F>(&self, pred: F) -> Split<'_, T, F>
2239 | |     where
2240 | |         F: FnMut(&T) -> bool,
     | |_____________________________^

For more information about this error, try `rustc --explain E0599`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - None literal (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0277]: `Option<_>` doesn't implement `std::fmt::Display`
   --> src\main.rs:13:11
    |
 13 |     print(&None);
    |     ----- ^^^^^ the trait `std::fmt::Display` is not implemented for `Option<_>`
    |     |
    |     required by a bound introduced by this call
    |
note: required by a bound in `tb_runtime::print`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:393:17
    |
393 | pub fn print<T: fmt::Display>(value: &T) {
    |                 ^^^^^^^^^^^^ required by this bound in `print`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - pop function (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0599]: no method named `as_list` found for type `{integer}` in the current scope
  --> src\main.rs:15:43
   |
15 |     print(&len(less[(0 as usize)].clone().as_list()));
   |                                           ^^^^^^^ method not found in `{integer}`

For more information about this error, try `rustc --explain E0599`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Real program - grade calculator (jit)
    Execution failed:
[TB JIT] Starting JIT execution with 3 statements
[TB JIT] Statement 1: Line 2 | fn calculate_grade(scores) {
[TB JIT] Statement 2: Line 15 | let students = [
[TB JIT] Statement 3: Line 21 | forEach(student => print(student.name + ": " + calculate_grade(student.scores)), students)
Error: Runtime error: Unsupported operation: None Add string

Stack backtrace:
   0: std::backtrace_rs::backtrace::win64::trace
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1: std::backtrace_rs::backtrace::trace_unsynchronized
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2: std::backtrace::Backtrace::create
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:331
   3: std::backtrace::Backtrace::capture
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\backtrace.rs:296
   4: anyhow::error::impl$1::from<enum2$<tb_core::error::TBError> >
             at C:\Users\Markin\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\anyhow-1.0.100\src\backtrace.rs:27
   5: core::result::impl$28::from_residual<tuple$<>,enum2$<tb_core::error::TBError>,anyhow::Error>
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\result.rs:2087
   6: tb::main
             at .\tb-exc\src\crates\tb-cli\src\main.rs:93
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),tuple$<> >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\core\src\ops\function.rs:253
   8: std::sys::backtrace::__rust_begin_short_backtrace<enum2$<core::result::Result<tuple$<>,anyhow::Error> > (*)(),enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\sys\backtrace.rs:158
   9: std::rt::lang_start::closure$0<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:206
  10: std::rt::lang_start_internal::closure$0
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:175
  11: std::panicking::catch_unwind::do_call
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:589
  12: std::panicking::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panicking.rs:552
  13: std::panic::catch_unwind
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\panic.rs:359
  14: std::rt::lang_start_internal
             at /rustc/1159e78c4747b02ef996e55082b704c09b970588/library\std\src\rt.rs:171
  15: std::rt::lang_start<enum2$<core::result::Result<tuple$<>,anyhow::Error> > >
             at C:\Users\Markin\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib\rustlib\src\rust\library\std\src\rt.rs:205
  16: main
  17: invoke_main
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:78
  18: __scrt_common_main_seh
             at D:\a\_work\1\s\src\vctools\crt\vcstartup\src\startup\exe_common.inl:288
  19: BaseThreadInitThunk
  20: RtlUserThreadStart

  - Real program - grade calculator (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:25:158
   |
25 | ..._string()), calculate_grade(student.get("scores").cloned().unwrap_or(DictValue::Int(0))))), students);
   |                --------------- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected `i64`, found `DictValue`
   |                |
   |                arguments to this function are incorrect
   |
note: function defined here
  --> src\main.rs:13:8
   |
13 |     fn calculate_grade(scores: i64) -> String {
   |        ^^^^^^^^^^^^^^^ -----------

error[E0599]: no method named `iter` found for type `i64` in the current scope
  --> src\main.rs:14:26
   |
14 |         let sum = scores.iter().fold(0, |a, &x| (a + x));
   |                          ^^^^ method not found in `i64`

error[E0277]: the trait bound `i64: Len` is not satisfied
   --> src\main.rs:15:40
    |
 15 |         let avg = ((sum as f64) / (len(&scores) as f64));
    |                                    --- ^^^^^^^ the trait `Len` is not implemented for `i64`
    |                                    |
    |                                    required by a bound introduced by this call
    |
    = help: the following other types implement trait `Len`:
              &[T]
              &str
              &tb_runtime::DictValue
              HashMap<K, V>
              Vec<T>
              std::string::String
              tb_runtime::DictValue
note: required by a bound in `tb_runtime::len`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:553:15
    |
553 | pub fn len<T: Len>(collection: &T) -> i64 {
    |               ^^^ required by this bound in `len`

Some errors have detailed explanations: E0277, E0308, E0599.
For more information about an error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 3 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Real program - text processing (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0507]: cannot move out of a shared reference
  --> src\main.rs:14:43
   |
14 |     let long_words = words.iter().filter(|&&w| (len(&w) > 4)).cloned().collect::<Vec<_>>();
   |                                           ^^-
   |                                             |
   |                                             data moved here
   |                                             move occurs because `w` has type `std::string::String`, which does not implement the `Copy` trait
   |
help: consider removing the borrow
   |
14 -     let long_words = words.iter().filter(|&&w| (len(&w) > 4)).cloned().collect::<Vec<_>>();
14 +     let long_words = words.iter().filter(|&w| (len(&w) > 4)).cloned().collect::<Vec<_>>();
   |

error[E0507]: cannot move out of a shared reference
  --> src\main.rs:15:50
   |
15 |     let uppercase_first = long_words.iter().map(|&w| (w).to_string()).collect::<Vec<_>>();
   |                                                  ^-
   |                                                   |
   |                                                   data moved here
   |                                                   move occurs because `w` has type `std::string::String`, which does not implement the `Copy` trait
   |
help: consider removing the borrow
   |
15 -     let uppercase_first = long_words.iter().map(|&w| (w).to_string()).collect::<Vec<_>>();
15 +     let uppercase_first = long_words.iter().map(|w| (w).to_string()).collect::<Vec<_>>();
   |

error[E0507]: cannot move out of a shared reference
  --> src\main.rs:18:15
   |
18 |     for_each(|&w| print(&w), long_words);
   |               ^-
   |                |
   |                data moved here
   |                move occurs because `w` has type `std::string::String`, which does not implement the `Copy` trait
   |
help: consider removing the borrow
   |
18 -     for_each(|&w| print(&w), long_words);
18 +     for_each(|w| print(&w), long_words);
   |

For more information about this error, try `rustc --explain E0507`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 3 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Recursion - sum of list (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime
DEBUG: Analyzing function body with params

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0308]: mismatched types
  --> src\main.rs:19:26
   |
19 |     print(&sum_recursive(vec![1, 2, 3, 4, 5], 0));
   |            ------------- ^^^^^^^^^^^^^^^^^^^ expected `i64`, found `Vec<{integer}>`
   |            |
   |            arguments to this function are incorrect
   |
   = note: expected type `i64`
            found struct `Vec<{integer}>`
note: function defined here
  --> src\main.rs:13:8
   |
13 |     fn sum_recursive(lst: i64, idx: i64) -> i64 {
   |        ^^^^^^^^^^^^^ --------

error[E0277]: the trait bound `i64: Len` is not satisfied
   --> src\main.rs:14:24
    |
 14 |         if (idx >= len(&lst)) {
    |                    --- ^^^^ the trait `Len` is not implemented for `i64`
    |                    |
    |                    required by a bound introduced by this call
    |
    = help: the following other types implement trait `Len`:
              &[T]
              &str
              &tb_runtime::DictValue
              HashMap<K, V>
              Vec<T>
              std::string::String
              tb_runtime::DictValue
note: required by a bound in `tb_runtime::len`
   --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:553:15
    |
553 | pub fn len<T: Len>(collection: &T) -> i64 {
    |               ^^^ required by this bound in `len`

error[E0599]: no method named `get` found for type `i64` in the current scope
  --> src\main.rs:17:21
   |
17 |         return (lst.get(&idx).cloned().unwrap_or(DictValue::Int(0)) + sum_recursive(lst, (idx + 1)));
   |                     ^^^
   |
help: there is a method `ge` with a similar name
   |
17 -         return (lst.get(&idx).cloned().unwrap_or(DictValue::Int(0)) + sum_recursive(lst, (idx + 1)));
17 +         return (lst.ge(&idx).cloned().unwrap_or(DictValue::Int(0)) + sum_recursive(lst, (idx + 1)));
   |

Some errors have detailed explanations: E0277, E0308, E0599.
For more information about an error, try `rustc --explain E0277`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 3 previous errors


════════════════════════════════════════════════════════════════════════════════


  - Scope - nested blocks (jit)
    Output does not contain '3
2
1':
3
3
3

  - Scope - nested blocks (compiled)
    Output does not contain '3
2
1':
3
3
3

  - Truthiness - None is falsy (compiled)
    Execution failed:
[TB Compiler] ✓ No networking usage detected - using minimal single-threaded runtime

════════════════════════════════════════════════════════════════════════════════
ERROR
════════════════════════════════════════════════════════════════════════════════

Runtime Error: Cargo compilation failed:
warning: unused import: `std::collections::HashMap as StdHashMap`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:8:5
  |
8 | use std::collections::HashMap as StdHashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused imports: `Arc` and `RwLock`
 --> C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\crates\tb-runtime\src\lib.rs:9:17
  |
9 | use std::sync::{Arc, RwLock};
  |                 ^^^  ^^^^^^

warning: `tb-runtime` (lib) generated 2 warnings (run `cargo fix --lib -p tb-runtime` to apply 2 suggestions)
   Compiling tb-compiled v0.1.0 (C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tb-exc\src\target\tb-compile-cache)
error[E0282]: type annotations needed
  --> src\main.rs:13:19
   |
13 |     if is_truthy(&(None)) {
   |                   ^^^^^^ cannot infer type of the type parameter `T` declared on the enum `Option`
   |
help: consider specifying the generic argument
   |
13 |     if is_truthy(&(None::<T>)) {
   |                        +++++

For more information about this error, try `rustc --explain E0282`.
error: could not compile `tb-compiled` (bin "tb-compiled") due to 1 previous error


════════════════════════════════════════════════════════════════════════════════


  - Variable shadowing in scope (jit)
    Output does not contain '2
1':
2
2

  - Variable shadowing in scope (compiled)
    Output does not contain '2
1':
2
2

  - Variable Declaration and Scope (jit)
    Output does not contain '2
1':
2
2

  - Variable Declaration and Scope (compiled)
    Output does not contain '2
1':
2
2
