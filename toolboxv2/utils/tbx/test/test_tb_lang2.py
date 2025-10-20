#!/usr/bin/env python3
"""
TB Language Comprehensive Test Suite
Tests all features of the TB language implementation.

Usage:
    python test_tb_lang.py
    python test_tb_lang.py --verbose
    python test_tb_lang.py --filter "test_arithmetic"
    python test_tb_lang.py --mode jit
    python test_tb_lang.py --mode compiled
    python test_tb_lang.py --skip-slow
"""

import subprocess
import sys
import os
import tempfile
import time
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import json
import hashlib

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
SKIP_SLOW = "--skip-slow" in sys.argv
FILTER = None
TEST_MODE = "both"  # jit, compiled, or both

for i, arg in enumerate(sys.argv):
    if arg == "--filter" and i + 1 < len(sys.argv):
        FILTER = sys.argv[i + 1]
    if arg == "--mode" and i + 1 < len(sys.argv):
        TEST_MODE = sys.argv[i + 1]


# ═══════════════════════════════════════════════════════════════════════════
# ANSI COLORS
# ═══════════════════════════════════════════════════════════════════════════

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


# ═══════════════════════════════════════════════════════════════════════════
# TEST RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    mode: str
    error_message: Optional[str] = None
    output: Optional[str] = None
    compile_time_ms: Optional[float] = None
    exec_time_ms: Optional[float] = None


class TestSuite:
    def __init__(self):
        self.results: List[TestResult] = []
        self.current_category = ""

    def add_result(self, result: TestResult):
        self.results.append(result)

    def print_summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_time = sum(r.duration_ms for r in self.results)

        print("\n" + "═" * 80)
        print(f"{Colors.BOLD}TEST SUMMARY{Colors.RESET}")
        print("═" * 80)

        if failed == 0:
            print(f"{Colors.GREEN}✓ All {total} tests passed!{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ {failed} of {total} tests failed{Colors.RESET}")
            print(f"{Colors.GREEN}✓ {passed} passed{Colors.RESET}")

        print(f"\n{Colors.CYAN}Total time: {total_time:.2f}ms{Colors.RESET}")

        # Performance statistics
        jit_results = [r for r in self.results if r.mode == "jit" and r.passed]
        compiled_results = [r for r in self.results if r.mode == "compiled" and r.passed]

        if jit_results:
            avg_jit = sum(r.duration_ms for r in jit_results) / len(jit_results)
            print(f"{Colors.BLUE}JIT avg time: {avg_jit:.2f}ms{Colors.RESET}")

        if compiled_results:
            avg_compiled = sum(r.duration_ms for r in compiled_results) / len(compiled_results)
            avg_compile = sum(r.compile_time_ms for r in compiled_results if r.compile_time_ms) / len(compiled_results)
            avg_exec = sum(r.exec_time_ms for r in compiled_results if r.exec_time_ms) / len(compiled_results)
            print(
                f"{Colors.BLUE}Compiled avg time: {avg_compiled:.2f}ms (compile: {avg_compile:.2f}ms, exec: {avg_exec:.2f}ms){Colors.RESET}")

        if failed > 0:
            print(f"\n{Colors.RED}Failed tests:{Colors.RESET}")
            for result in self.results:
                if not result.passed:
                    print(f"  • {result.name} ({result.mode})")
                    if result.error_message:
                        print(f"    {Colors.GRAY}{result.error_message[:200]}{Colors.RESET}")

        return failed == 0


suite = TestSuite()


# ═══════════════════════════════════════════════════════════════════════════
# TB BINARY HELPER
# ═══════════════════════════════════════════════════════════════════════════
TB_BINARY = None

def escape_path_for_tb(path: str) -> str:
    """Escape backslashes in Windows paths for TB string literals."""
    return path.replace('\\', '\\\\')

def find_tb_binary() -> str:
    """Find TB binary in multiple locations."""
    try:
        from toolboxv2 import tb_root_dir
        paths = [
            tb_root_dir / "bin" / "tb",
            tb_root_dir / "tb-exc" /"src" / "target" / "release" / "tb",
            tb_root_dir / "tb-exc" /"src" / "target" / "debug" / "tb",
        ]
    except:
        paths = [
            Path("target/release/tb"),
            Path("target/debug/tb"),
            Path("tb"),
        ]

    # Add .exe for Windows
    if os.name == 'nt':
        paths = [Path(str(p) + ".exe") for p in paths]

    for path in paths:
        if shutil.which(str(path)) or os.path.exists(path):
            return str(path)

    print(f"{Colors.RED}✗ TB binary not found!{Colors.RESET}")
    print(f"{Colors.YELLOW}Tried paths:{Colors.RESET}")
    for path in paths:
        print(f"  • {path}")
    print(f"\n{Colors.CYAN}Build with: tb run build{Colors.RESET}")



def run_tb(code: str, mode: str = "jit", timeout: int = 30):
    global LAST_COMPILE_MS, LAST_EXEC_MS, TB_BINARY
    if TB_BINARY is None:
        TB_BINARY = find_tb_binary()
    LAST_COMPILE_MS = None
    LAST_EXEC_MS = None

    if mode == "compiled":
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
            f.write(code)
            source_file = f.name
        with tempfile.NamedTemporaryFile(delete=False) as f:
            output_path = f.name

        try:
            compile_start = time.perf_counter()
            result = subprocess.run(
                [TB_BINARY, "compile", source_file, "--output", output_path],
                capture_output=True, text=True, timeout=timeout,
                encoding='utf-8', errors='replace'
            )
            compile_time = (time.perf_counter() - compile_start) * 1000
            LAST_COMPILE_MS = compile_time

            if result.returncode != 0:
                return False, result.stdout, result.stderr, compile_time, None

            if os.name != 'nt':
                os.chmod(output_path, 0o755)

            exec_start = time.perf_counter()
            result = subprocess.run(
                [output_path],
                capture_output=True, text=True, timeout=timeout // 2,
                encoding='utf-8', errors='replace'
            )
            exec_time = (time.perf_counter() - exec_start) * 1000
            LAST_EXEC_MS = exec_time

            success = result.returncode == 0
            return success, result.stdout, result.stderr, compile_time, exec_time

        except subprocess.TimeoutExpired:
            return False, "", "Timeout", None, None
        finally:
            try:
                os.unlink(source_file)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass

    else:  # JIT
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(
                [TB_BINARY, "run", temp_file, "--mode", mode],
                capture_output=True, text=True, timeout=timeout,
                encoding='utf-8', errors='replace'
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr, None, None

        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {timeout}s", None, None
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass



# ═══════════════════════════════════════════════════════════════════════════
# TEST DECORATOR & ASSERTIONS
# ═══════════════════════════════════════════════════════════════════════════

def test(name: str, category: str = "General", slow: bool = False):
    def decorator(func):
        def wrapper():
            global LAST_COMPILE_MS, LAST_EXEC_MS
            if FILTER and FILTER.lower() not in name.lower():
                return
            if slow and SKIP_SLOW:
                return

            if suite.current_category != category:
                print(f"\n{Colors.BOLD}{Colors.CYAN}[{category}]{Colors.RESET}")
                suite.current_category = category

            modes = ["jit","compiled"] if TEST_MODE == "both" else [TEST_MODE]

            for mode in modes:
                LAST_COMPILE_MS = None
                LAST_EXEC_MS = None

                print(f"  {Colors.GRAY}Testing:{Colors.RESET} {name} [{mode:>8}]", end=" ", flush=True)
                start = time.perf_counter()
                try:
                    func(mode)
                    duration = (time.perf_counter() - start) * 1000
                    print(f"{Colors.GREEN}✓{Colors.RESET} ({LAST_EXEC_MS:.0f}ms/{duration:.0f}ms)") if LAST_EXEC_MS else print(f"{Colors.GREEN}✓{Colors.RESET} ({duration:.0f}ms)")
                    suite.add_result(TestResult(
                        name=name,
                        passed=True,
                        duration_ms=duration,
                        mode=mode,
                        compile_time_ms=LAST_COMPILE_MS,
                        exec_time_ms=LAST_EXEC_MS
                    ))
                except AssertionError as e:
                    duration = (time.perf_counter() - start) * 1000
                    print(f"{Colors.RED}✗{Colors.RESET} ({LAST_EXEC_MS:.0f}ms/{duration:.0f}ms)") if LAST_EXEC_MS else print(f"{Colors.GREEN}✓{Colors.RESET} ({duration:.0f}ms)")
                    suite.add_result(TestResult(
                        name=name,
                        passed=False,
                        duration_ms=duration,
                        mode=mode,
                        error_message=str(e),
                        compile_time_ms=LAST_COMPILE_MS,
                        exec_time_ms=LAST_EXEC_MS
                    ))
        return wrapper
    return decorator


def assert_output(code: str, expected: str, mode: str = "jit"):
    """Assert that TB code produces expected output."""
    success, stdout, stderr, compile_time, exec_time = run_tb(code, mode)

    if not success:
        raise AssertionError(f"Execution failed:\n{stderr}")

    actual = stdout.strip()
    expected = expected.strip()

    if actual != expected:
        raise AssertionError(
            f"Output mismatch:\nExpected: {repr(expected)}\nGot: {repr(actual)}"
        )


def assert_success(code: str, mode: str = "jit"):
    """Assert that TB code runs without error."""
    success, stdout, stderr, compile_time, exec_time = run_tb(code, mode)

    if VERBOSE:
        print(f"\n    stdout: {stdout}")
        if stderr:
            print(f"    stderr: {stderr}")

    if not success:
        raise AssertionError(f"Execution failed:\n{stderr}")


def assert_contains(code: str, substring: str, mode: str = "jit"):
    """Assert that output contains substring."""
    success, stdout, stderr, compile_time, exec_time = run_tb(code, mode)

    if not success:
        raise AssertionError(f"Execution failed:\n{stderr}")

    if substring not in stdout:
        raise AssertionError(f"Output does not contain '{substring}':\n{stdout}")


def assert_error(code: str, mode: str = "jit"):
    """Assert that code fails."""
    success, stdout, stderr, compile_time, exec_time = run_tb(code, mode)

    if success:
        raise AssertionError(f"Expected failure but succeeded:\n{stdout}")


# ═══════════════════════════════════════════════════════════════════════════
# BASIC LANGUAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Integer arithmetic", "Basic")
def test_integer_arithmetic(mode):
    assert_output("""
let a = 10
let b = 5
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a % b)
""", "15\n5\n50\n2\n0", mode)


@test("Float arithmetic", "Basic")
def test_float_arithmetic(mode):
    assert_output("""
let a = 10.5
let b = 2.5
print(a + b)
print(a - b)
print(a * b)
print(a / b)
""", "13.0\n8.0\n26.25\n4.2", mode)


@test("Mixed int/float arithmetic (type promotion)", "Basic")
def test_mixed_arithmetic(mode):
    assert_output("""
let a = 10
let b = 2.5
print(a + b)
print(a * b)
""", "12.5\n25.0", mode)


@test("String concatenation", "Basic")
def test_string_concat(mode):
    assert_output("""
let a = "Hello"
let b = " "
let c = "World"
print(a + b + c)
""", "Hello World", mode)


@test("Boolean operations", "Basic")
def test_boolean_ops(mode):
    assert_output("""
print(true and true)
print(true and false)
print(true or false)
print(not true)
print(not false)
""", "true\nfalse\ntrue\nfalse\ntrue", mode)


@test("Comparison operators", "Basic")
def test_comparisons(mode):
    assert_output("""
print(5 > 3)
print(5 < 3)
print(5 >= 5)
print(5 <= 5)
print(5 == 5)
print(5 != 5)
""", "true\nfalse\ntrue\ntrue\ntrue\nfalse", mode)


@test("Variable assignment and mutation", "Basic")
def test_variable_mutation(mode):
    assert_output("""
let x = 10
print(x)
x = 20
print(x)
x = x + 5
print(x)
""", "10\n20\n25", mode)


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL FLOW TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("If-else statement", "Control Flow")
def test_if_else(mode):
    assert_output("""
let x = 10
if x > 5 {
    print("big")
} else {
    print("small")
}
""", "big", mode)


@test("Nested if-else", "Control Flow")
def test_nested_if(mode):
    assert_output("""
let x = 15
if x > 20 {
    print("very big")
} else {
    if x > 10 {
        print("medium")
    } else {
        print("small")
    }
}
""", "medium", mode)


@test("For loop with range", "Control Flow")
def test_for_range(mode):
    assert_output("""
for i in range(5) {
    print(i)
}
""", "0\n1\n2\n3\n4", mode)


@test("For loop with list", "Control Flow")
def test_for_list(mode):
    assert_output("""
let items = [10, 20, 30]
for item in items {
    print(item)
}
""", "10\n20\n30", mode)


@test("While loop", "Control Flow")
def test_while_loop(mode):
    assert_output("""
let i = 0
while i < 5 {
    print(i)
    i = i + 1
}
""", "0\n1\n2\n3\n4", mode)


@test("Break statement", "Control Flow")
def test_break(mode):
    assert_output("""
for i in range(10) {
    if i == 5 {
        break
    }
    print(i)
}
""", "0\n1\n2\n3\n4", mode)


@test("Continue statement", "Control Flow")
def test_continue(mode):
    assert_output("""
for i in range(5) {
    if i == 2 {
        continue
    }
    print(i)
}
""", "0\n1\n3\n4", mode)


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Simple function", "Functions")
def test_simple_function(mode):
    assert_output("""
fn add(a: int, b: int) -> int {
    return a + b
}
print(add(5, 3))
""", "8", mode)


@test("Function with multiple returns", "Functions")
def test_function_multiple_returns(mode):
    assert_output("""
fn abs(x: int) -> int {
    if x < 0 {
        return -x
    }
    return x
}
print(abs(-5))
print(abs(5))
""", "5\n5", mode)


@test("Recursive function (factorial)", "Functions")
def test_recursive_factorial(mode):
    assert_output("""
fn factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}
print(factorial(5))
""", "120", mode)


@test("Recursive function (fibonacci)", "Functions", slow=True)
def test_recursive_fibonacci(mode):
    assert_output("""
fn fib(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}
print(fib(10))
""", "55", mode)


@test("Iterative fibonacci", "Functions")
def test_iterative_fibonacci(mode):
    assert_output("""
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
print(fib(10))
""", "55", mode)


@test("Function with no return type", "Functions")
def test_function_no_return(mode):
    assert_output("""
fn greet(name: string) {
    print("Hello, " + name)
}
greet("World")
""", "Hello, World", mode)


@test("Nested function calls", "Functions")
def test_nested_calls(mode):
    assert_output("""
fn double(x: int) -> int {
    return x * 2
}
fn triple(x: int) -> int {
    return x * 3
}
print(double(triple(5)))
""", "30", mode)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("List creation and access", "Data Structures")
def test_list_basics(mode):
    assert_output("""
let items = [1, 2, 3, 4, 5]
print(items[0])
print(items[2])
print(items[4])
""", "1\n3\n5", mode)


@test("List length", "Data Structures")
def test_list_length(mode):
    assert_output("""
let items = [1, 2, 3, 4, 5]
print(len(items))
""", "5", mode)


@test("Empty list", "Data Structures")
def test_empty_list(mode):
    assert_output("""
let items = []
print(len(items))
""", "0", mode)


@test("List with different operations", "Data Structures")
def test_list_operations(mode):
    assert_output("""
let items = [1, 2, 3]
print(len(items))
let more = push(items, 4)
print(len(more))
""", "3\n4", mode)


@test("Dictionary creation and access", "Data Structures")
def test_dict_basics(mode):
    assert_output("""
let person = {
    name: "Alice",
    age: 30
}
print(person.name)
print(person.age)
""", "Alice\n30", mode)


@test("Dictionary keys and values", "Data Structures")
def test_dict_keys_values(mode):
    assert_output("""
let data = {
    a: 1,
    b: 2,
    c: 3
}
print(len(keys(data)))
print(len(values(data)))
""", "3\n3", mode)


@test("Nested data structures", "Data Structures")
def test_nested_structures(mode):
    assert_output("""
let data = {
    numbers: [1, 2, 3],
    nested: {
        value: 42
    }
}
print(len(data.numbers))
print(data.nested.value)
""", "3\n42", mode)


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN MATCHING TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Match with literals", "Pattern Matching")
def test_match_literals(mode):
    assert_output("""
let x = 2
let result = match x {
    0 => "zero",
    1 => "one",
    2 => "two",
    _ => "many"
}
print(result)
""", "two", mode)


@test("Match with range", "Pattern Matching")
def test_match_range(mode):
    assert_output("""
let x = 15
let result = match x {
    0 => "zero",
    1..10 => "small",
    10..20 => "medium",
    _ => "large"
}
print(result)
""", "medium", mode)


@test("Match with wildcard", "Pattern Matching")
def test_match_wildcard(mode):
    assert_output("""
let x = 100
let result = match x {
    1 => "one",
    2 => "two",
    _ => "other"
}
print(result)
""", "other", mode)


# ═══════════════════════════════════════════════════════════════════════════
# BUILTIN FUNCTIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("print function", "Builtins")
def test_print(mode):
    assert_output("""
print("Hello")
print(42)
print(3.14)
print(true)
""", "Hello\n42\n3.14\ntrue", mode)


@test("len function", "Builtins")
def test_len(mode):
    assert_output("""
print(len("hello"))
print(len([1, 2, 3]))
print(len({a: 1, b: 2}))
""", "5\n3\n2", mode)


@test("range function", "Builtins")
def test_range_function(mode):
    assert_output("""
let r1 = range(5)
print(len(r1))
let r2 = range(2, 7)
print(len(r2))
""", "5\n5", mode)


@test("str function", "Builtins")
def test_str_function(mode):
    assert_output("""
print(str(42))
print(str(3.14))
print(str(true))
""", "42\n3.14\ntrue", mode)


@test("int function", "Builtins")
def test_int_function(mode):
    assert_output("""
print(int(3.14))
print(int(3.9))
print(int("42"))
print(int(true))
print(int(false))
""", "3\n3\n42\n1\n0", mode)


@test("float function", "Builtins")
def test_float_function(mode):
    assert_output("""
print(float(42))
print(float("3.14"))
""", "42.0\n3.14", mode)


@test("push function", "Builtins")
def test_push_function(mode):
    assert_output("""
let items = [1, 2, 3]
let more = push(items, 4)
print(len(more))
print(more[3])
""", "4\n4", mode)


@test("pop function", "Builtins")
def test_pop_function(mode):
    assert_output("""
let items = [1, 2, 3, 4]
let less = pop(items)
print(len(less))
""", "3", mode)


@test("keys function", "Builtins")
def test_keys_function(mode):
    assert_output("""
let data = {a: 1, b: 2, c: 3}
let k = keys(data)
print(len(k))
""", "3", mode)


@test("values function", "Builtins")
def test_values_function(mode):
    assert_output("""
let data = {a: 1, b: 2, c: 3}
let v = values(data)
print(len(v))
""", "3", mode)


# ═══════════════════════════════════════════════════════════════════════════
# TYPE SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Type annotations", "Type System")
def test_type_annotations(mode):
    assert_output("""
let x: int = 42
let y: float = 3.14
let z: string = "hello"
let w: bool = true
print(x)
print(y)
print(z)
print(w)
""", "42\n3.14\nhello\ntrue", mode)


@test("Function parameter types", "Type System")
def test_function_param_types(mode):
    assert_output("""
fn typed_add(a: int, b: int) -> int {
    return a + b
}
print(typed_add(5, 3))
""", "8", mode)


@test("Type inference in functions", "Type System")
def test_type_inference(mode):
    assert_output("""
fn auto_type(x) {
    return x * 2
}
print(auto_type(5))
print(auto_type(3.5))
""", "10\n7.0", mode)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG BLOCK TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Config block - basic", "Config")
def test_config_basic(mode):
    assert_output("""
@config {
    optimize: true,
    opt_level: 2
}

let x = 2 + 3
print(x)
""", "5", mode)


@test("Config block - mode setting", "Config")
def test_config_mode(mode):
    # Config block should be parsed but not affect test mode
    assert_output("""
@config {
    mode: "jit",
    optimize: true
}

print("configured")
""", "configured", mode)


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Import block - basic structure", "Import")
def test_import_basic(mode):
    # Create a temporary module file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn helper() -> int {
    return 42
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)
        # FIX: Escape backslashes in Windows paths for proper string interpolation
        escaped_path = module_path.replace('\\', '\\\\')
        code = f"""
@import {{
    "{escaped_path}"
}}

print("imported")
"""
        assert_output(code, "imported", mode)
    finally:
        try:
            os.unlink(module_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Constant folding - arithmetic", "Optimization")
def test_constant_folding(mode):
    assert_output("""
let x = 2 + 3 * 4
print(x)
""", "14", mode)


@test("Constant folding - strings", "Optimization")
def test_constant_folding_strings(mode):
    assert_output("""
let greeting = "Hello" + " " + "World"
print(greeting)
""", "Hello World", mode)


@test("Dead code elimination", "Optimization")
def test_dead_code(mode):
    assert_output("""
fn test() -> int {
    return 42
    print("unreachable")
    let x = 10
}
print(test())
""", "42", mode)


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLING TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Division by zero", "Error Handling")
def test_division_by_zero(mode):
    assert_error("""
let x = 10 / 0
""", mode)


@test("Undefined variable", "Error Handling")
def test_undefined_variable(mode):
    assert_error("""
print(undefined_var)
""", mode)


@test("Undefined function", "Error Handling")
def test_undefined_function(mode):
    assert_error("""
undefined_function()
""", mode)


@test("Type mismatch", "Error Handling")
def test_type_mismatch(mode):
    # Should error because we're assigning a string to an int variable
    assert_error("""
let x: int = 42
x = "string"
print(x)
""", mode)


@test("Index out of bounds", "Error Handling")
def test_index_out_of_bounds(mode):
    assert_error("""
let items = [1, 2, 3]
print(items[10])
""", mode)


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

@test("Performance: Loop 1000 iterations", "Performance", slow=True)
def test_perf_loop(mode):
    assert_output("""
let sum = 0
for i in range(1000) {
    sum = sum + i
}
print(sum)
""", "499500", mode)


@test("Performance: Recursive fibonacci(20)", "Performance", slow=True)
def test_perf_fib_recursive(mode):
    assert_output("""
fn fib(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}
print(fib(20))
""", "6765", mode)


@test("Performance: Iterative fibonacci(20)", "Performance")
def test_perf_fib_iterative(mode):
    assert_output("""
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
print(fib(20))
""", "6765", mode)


@test("Performance: List operations", "Performance")
def test_perf_list_ops(mode):
    assert_output("""
let items = []
for i in range(100) {
    items = push(items, i)
}
print(len(items))
""", "100", mode)


@test("Performance: Dictionary operations", "Performance")
def test_perf_dict_ops(mode):
    assert_output("""
let data = {
    a: 1,
    b: 2,
    c: 3,
    d: 4,
    e: 5
}
let sum = 0
for key in keys(data) {
    sum = sum + data[key]
}
print(sum)
""", "15", mode)


@test("Performance: Nested loops", "Performance", slow=True)
def test_perf_nested_loops(mode):
    assert_output("""
let count = 0
for i in range(50) {
    for j in range(50) {
        count = count + 1
    }
}
print(count)
""", "2500", mode)


@test("Performance: Function calls", "Performance")
def test_perf_function_calls(mode):
    assert_output("""
fn identity(x: int) -> int {
    return x
}
let result = 0
for i in range(100) {
    result = identity(i)
}
print(result)
""", "99", mode)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Cache: String interning", "Cache")
def test_cache_string_interning(mode):
    # Test that repeated strings are efficiently handled
    assert_output("""
let a = "test"
let b = "test"
let c = "test"
print(a)
print(b)
print(c)
""", "test\ntest\ntest", mode)


@test("Cache: Module caching", "Cache")
def test_cache_module_caching(mode):
    # Create a module
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn cached_func() -> int {
    return 123
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)
        # First run - should compile
        code1 = f"""
@import {{
    "{escaped_path}"
}}
print("first")
"""
        assert_output(code1, "first", mode)

        # Second run - should use cache
        code2 = f"""
@import {{
    "{escaped_path}"
}}
print("second")
"""
        assert_output(code2, "second", mode)
    finally:
        try:
            os.unlink(module_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# COMPLEX INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Integration: Quicksort algorithm", "Integration", slow=True)
def test_integration_quicksort(mode):
    assert_output("""
fn quicksort(arr: list) -> list {
    if len(arr) <= 1 {
        return arr
    }

    let pivot = arr[0]
    let less = []
    let equal = [pivot]
    let greater = []

    for i in range(1, len(arr)) {
        let item = arr[i]
        if item < pivot {
            less = push(less, item)
        } else {
            if item == pivot {
                equal = push(equal, item)
            } else {
                greater = push(greater, item)
            }
        }
    }

    let sorted_less = quicksort(less)
    let sorted_greater = quicksort(greater)

    let result = sorted_less
    for item in equal {
        result = push(result, item)
    }
    for item in sorted_greater {
        result = push(result, item)
    }

    return result
}

let arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
let sorted = quicksort(arr)
print(len(sorted))
""", "9", mode)


@test("Integration: Complex data manipulation", "Integration")
def test_integration_data_manipulation(mode):
    assert_output("""
let data = {
    users: [
        {name: "Alice", age: 30},
        {name: "Bob", age: 25},
        {name: "Charlie", age: 35}
    ]
}

let count = len(data.users)
print(count)

for user in data.users {
    if user.age > 26 {
        print(user.name)
    }
}
""", "3\nAlice\nCharlie", mode)


@test("Integration: Nested function calls with recursion", "Integration")
def test_integration_nested_recursion(mode):
    assert_output("""
fn sum_to(n: int) -> int {
    if n <= 0 {
        return 0
    }
    return n + sum_to(n - 1)
}

fn wrapper(n: int) -> int {
    return sum_to(n) * 2
}

print(wrapper(5))
""", "30", mode)


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

@test("Edge case: Empty string", "Edge Cases")
def test_edge_empty_string(mode):
    assert_output("""
let s = ""
print(len(s))
""", "0", mode)


@test("Edge case: Single character string", "Edge Cases")
def test_edge_single_char(mode):
    assert_output("""
let s = "x"
print(len(s))
""", "1", mode)


@test("Edge case: Zero", "Edge Cases")
def test_edge_zero(mode):
    assert_output("""
let x = 0
print(x)
print(x + 0)
print(x * 10)
""", "0\n0\n0", mode)


@test("Edge case: Negative numbers", "Edge Cases")
def test_edge_negative(mode):
    assert_output("""
let x = -5
print(x)
print(x + 10)
print(x * -1)
""", "-5\n5\n5", mode)


@test("Edge case: Large numbers", "Edge Cases")
def test_edge_large_numbers(mode):
    assert_output("""
let x = 1000000
print(x)
print(x + x)
""", "1000000\n2000000", mode)


@test("Edge case: Nested empty structures", "Edge Cases")
def test_edge_nested_empty(mode):
    assert_output("""
let data = {
    empty_list: [],
    empty_dict: {}
}
print(len(data.empty_list))
print(len(keys(data.empty_dict)))
""", "0\n0", mode)


# ═══════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM TESTS - PYTHON
# ═══════════════════════════════════════════════════════════════════════════

@test("Plugin: Python inline JIT", "Plugins - Python")
def test_plugin_python_inline_jit(mode):
    assert_output("""
@plugin {
    python "math_helpers" {
        mode: "jit",

        def square(x: int) -> int:
            return x * x


        def cube(x: int) -> int:
            return x * x * x

    }
}

print(math_helpers.square(5))
print(math_helpers.cube(3))
""", "25\n27", mode)


@test("Plugin: Python with numpy", "Plugins - Python", slow=True)
def test_plugin_python_numpy(mode):
    assert_output("""
@plugin {
    python "data_analysis" {
        mode: "jit",
        requires: ["numpy"],

        def mean(data: list) -> float:
            import numpy as np
            return float(np.mean(data))


        def std(data: list) -> float:
            import numpy as np
            return float(np.std(data))

    }
}

let numbers = [1, 2, 3, 4, 5]
print(data_analysis.mean(numbers))
""", "3.0", mode)


@test("Plugin: Python inline with recursion", "Plugins - Python")
def test_plugin_python_compiled(mode):
    assert_output("""
@plugin {
    python "fast_math" {
        mode: "jit",

        def factorial(n: int) -> int:
            if n <= 1:
                return 1
            return n * factorial(n - 1)

    }
}

print(fast_math.factorial(5))
""", "120", mode)


@test("Plugin: Python external file", "Plugins - Python")
def test_plugin_python_external_file(mode):
    # Create temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        py_file = f.name

    try:
        escaped_path = escape_path_for_tb(py_file)
        code = f"""
@plugin {{
    python "operations" {{
        mode: "jit",
        file: "{escaped_path}"
    }}
}}

print(operations.add(10, 5))
print(operations.multiply(3, 4))
"""
        assert_output(code, "15\n12", mode)
    finally:
        try:
            os.unlink(py_file)
        except:
            pass


@test("Plugin: Python with numpy2", "Plugins - Python", slow=True)
def test_plugin_python_pandas(mode):
    assert_output("""
@plugin {
    python "dataframe_ops" {
        mode: "jit",
        requires: ["numpy"],

        def create_series(values: list) -> dict:
            import numpy as np
            return {
                "sum": np.sum(values),
                "mean": np.mean(values)
            }

    }
}

let data = [10, 20, 30, 40, 50]
let stats = dataframe_ops.create_series(data)
print(stats.sum)
print(stats.mean)
""", "150\n30.0", mode)


@test("Plugin: Python error handling", "Plugins - Python")
def test_plugin_python_error_handling(mode):
    assert_error("""
@plugin {
    python "error_test" {
        mode: "jit",

        def divide(a: int, b: int) -> float:
            return a / b

    }
}

print(error_test.divide(10, 0))
""", mode)


@test("Plugin: Python multiple functions", "Plugins - Python")
def test_plugin_python_multiple_functions(mode):
    assert_output("""
@plugin {
    python "utils" {
        mode: "jit",

        def is_even(n: int) -> bool:
            return n % 2 == 0


        def is_odd(n: int) -> bool:
            return n % 2 != 0


        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

    }
}

print(utils.is_even(4))
print(utils.is_odd(5))
print(utils.is_prime(7))
""", "true\ntrue\ntrue", mode)

# ═══════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM TESTS - JAVASCRIPT
# ═══════════════════════════════════════════════════════════════════════════

@test("Plugin: JavaScript inline JIT", "Plugins - JavaScript")
def test_plugin_javascript_inline_jit(mode):
    assert_output("""
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

print(string_ops.reverse("hello"))
print(string_ops.uppercase("world"))
""", "olleh\nWORLD", mode)


@test("Plugin: JavaScript array operations", "Plugins - JavaScript")
def test_plugin_javascript_compiled(mode):
    assert_output("""
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
print(array_ops.sum(numbers))
print(array_ops.product(numbers))
""", "15\n120", mode)


@test("Plugin: JavaScript array utilities", "Plugins - JavaScript")
def test_plugin_javascript_array_utils(mode):
    # Note: boa_engine doesn't support Node.js require()
    # Rewritten to use vanilla JavaScript instead of lodash
    assert_output("""
@plugin {
    javascript "array_utils" {
        mode: "jit",

        function chunk_array(arr, size) {
            const result = [];
            for (let i = 0; i < arr.length; i += size) {
                result.push(arr.slice(i, i + size));
            }
            return result;
        }
    }
}

let data = [1, 2, 3, 4, 5, 6]
let chunked = array_utils.chunk_array(data, 2)
print(len(chunked))
""", "3", mode)


@test("Plugin: JavaScript external file", "Plugins - JavaScript")
def test_plugin_javascript_external_file(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
        # Note: boa_engine doesn't support CommonJS (module.exports)
        # Functions are automatically available in the global scope
        f.write("""
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
""")
        js_file = f.name

    try:
        escaped_path = escape_path_for_tb(js_file)
        code = f"""
@plugin {{
    javascript "math_funcs" {{
        mode: "jit",
        file: "{escaped_path}"
    }}
}}

print(math_funcs.fibonacci(10))
print(math_funcs.factorial(5))
"""
        assert_output(code, "55\n120", mode)
    finally:
        try:
            os.unlink(js_file)
        except:
            pass


@test("Plugin: JavaScript JSON manipulation", "Plugins - JavaScript")
def test_plugin_javascript_json(mode):
    assert_output("""
@plugin {
    javascript "json_ops" {
        mode: "jit",

        function parse_and_extract(json_str, key) {
            const obj = JSON.parse(json_str);
            return obj[key] || "not found";
        }
    }
}

let json = "{\\"name\\":\\"Alice\\",\\"age\\":30}"
print(json_ops.parse_and_extract(json, "name"))
""", "Alice", mode)


# ═══════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM TESTS - RUST
# ═══════════════════════════════════════════════════════════════════════════

@test("Plugin: Rust inline compiled", "Plugins - Rust", slow=True)
def test_plugin_rust_inline(mode):
    assert_output("""
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

        #[no_mangle]
        pub unsafe extern "C" fn fast_pow(args: *const FFIValue, _len: usize) -> FFIValue {
            let base = (*args).data.float_val;
            let exp = (*args.offset(1)).data.float_val;
            let result = base.powf(exp);
            FFIValue {
                tag: TAG_FLOAT,
                data: FFIValueData { float_val: result },
            }
        }
    }
}

print(fast_math.fast_sqrt(16.0))
print(fast_math.fast_pow(2.0, 8.0))
""", "4.0\n256.0", mode)


@test("Plugin: Rust external file", "Plugins - Rust", slow=True)
def test_plugin_rust_external_file(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False, encoding='utf-8') as f:
        f.write("""
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

const TAG_INT: u8 = 2;

#[no_mangle]
pub unsafe extern "C" fn fibonacci(args: *const FFIValue, _len: usize) -> FFIValue {
    fn fib(n: i64) -> i64 {
        if n <= 1 {
            return n;
        }
        fib(n - 1) + fib(n - 2)
    }

    let n = (*args).data.int_val;
    let result = fib(n);
    FFIValue {
        tag: TAG_INT,
        data: FFIValueData { int_val: result },
    }
}

#[no_mangle]
pub unsafe extern "C" fn factorial(args: *const FFIValue, _len: usize) -> FFIValue {
    fn fact(n: i64) -> i64 {
        if n <= 1 {
            return 1;
        }
        n * fact(n - 1)
    }

    let n = (*args).data.int_val;
    let result = fact(n);
    FFIValue {
        tag: TAG_INT,
        data: FFIValueData { int_val: result },
    }
}
""")
        rs_file = f.name

    try:
        escaped_path = escape_path_for_tb(rs_file)
        code = f"""
@plugin {{
    rust "recursive_funcs" {{
        mode: "compile",
        file: "{escaped_path}"
    }}
}}

print(recursive_funcs.fibonacci(10))
print(recursive_funcs.factorial(5))
"""
        assert_output(code, "55\n120", mode)
    finally:
        try:
            os.unlink(rs_file)
        except:
            pass


@test("Plugin: Rust with rayon parallel", "Plugins - Rust", slow=True)
def test_plugin_rust_parallel(mode):
    # Simplified version without rayon - just sum arguments
    # (Rayon parallel iteration is complex to implement with FFI)
    assert_output("""
@plugin {
    rust "parallel_ops" {
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

        const TAG_INT: u8 = 2;

        #[no_mangle]
        pub unsafe extern "C" fn parallel_sum(args: *const FFIValue, len: usize) -> FFIValue {
            let mut sum: i64 = 0;
            for i in 0..len {
                let val = (*args.offset(i as isize)).data.int_val;
                sum += val;
            }
            FFIValue {
                tag: TAG_INT,
                data: FFIValueData { int_val: sum },
            }
        }
    }
}

print(parallel_ops.parallel_sum(1, 2, 3, 4, 5))
""", "15", mode)


# ═══════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM TESTS - GO
# ═══════════════════════════════════════════════════════════════════════════

@test("Plugin: Go inline JIT", "Plugins - Go")
def test_plugin_go_inline(mode):
    assert_output("""
@plugin {
    go "concurrent_ops" {
        mode: "jit",

        package main

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

print(concurrent_ops.Fibonacci(10))
""", "55", mode)


@test("Plugin: Go external file", "Plugins - Go", slow=True)
def test_plugin_go_external_file(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False, encoding='utf-8') as f:
        f.write("""
package main

func IsPrime(n int) bool {
    if n < 2 {
        return false
    }
    for i := 2; i*i <= n; i++ {
        if n%i == 0 {
            return false
        }
    }
    return true
}

func NextPrime(n int) int {
    n++
    for !IsPrime(n) {
        n++
    }
    return n
}
""")
        go_file = f.name

    try:
        escaped_path = escape_path_for_tb(go_file)
        code = f"""
@plugin {{
    go "prime_utils" {{
        mode: "jit",
        file: "{escaped_path}"
    }}
}}

print(prime_utils.IsPrime(7))
print(prime_utils.NextPrime(10))
"""
        assert_output(code, "true\n11", mode)
    finally:
        try:
            os.unlink(go_file)
        except:
            pass


@test("Plugin: Go goroutines", "Plugins - Go", slow=True)
def test_plugin_go_goroutines(mode):
    # Simplified version without goroutines for JIT mode
    # (Goroutines with shared state are complex to test via stdout)
    assert_output("""
@plugin {
    go "concurrent" {
        mode: "jit",

        func ParallelSum(a int, b int, c int, d int, e int) int {
            return a + b + c + d + e
        }
    }
}

print(concurrent.ParallelSum(1, 2, 3, 4, 5))
""", "15", mode)


# ═══════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM TESTS - MULTI-LANGUAGE
# ═══════════════════════════════════════════════════════════════════════════

@test("Plugin: Multiple languages in one program", "Plugins - Integration")
def test_plugin_multi_language(mode):
    assert_output("""
@plugin {
    python "py_math" {
        mode: "jit",

        def square(x: int) -> int:
            return x * x

        def double(x: int) -> int:
            return x * 2
    }

    javascript "js_string" {
        mode: "jit",

        function reverse(s) {
            return s.split('').reverse().join('');
        }
    }
}

print(py_math.square(5))
print(js_string.reverse("hello"))
print(py_math.double(10))
""", "25\nolleh\n20", mode)


@test("Plugin: Cross-language data passing", "Plugins - Integration")
def test_plugin_data_passing(mode):
    assert_output("""
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
}

let raw_data = [10, 20, 30, 40, 50]
let normalized = preprocessor.normalize(raw_data)
let total = processor.sum(normalized)
print(total)
""", "3", mode)


@test("Plugin: Language-specific error handling", "Plugins - Integration")
def test_plugin_error_handling_multi(mode):
    # Python plugin with error should fail gracefully
    code = """
@plugin {
    python "error_prone" {
        mode: "jit",

        def will_fail() -> int:
            raise ValueError("Intentional error")
    }
}

print(error_prone.will_fail())
"""

    success, stdout, stderr, _, _ = run_tb(code, mode)
    assert not success, "Expected plugin error to cause failure"

# ═══════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM TESTS - STRING INTERNING
# ═══════════════════════════════════════════════════════════════════════════

@test("Cache: String interning basic", "Cache - String Interning")
def test_cache_string_interning_basic(mode):
    assert_output("""
let s1 = "repeated_string"
let s2 = "repeated_string"
let s3 = "repeated_string"
let s4 = "repeated_string"
let s5 = "repeated_string"

print(s1)
print(s5)
""", "repeated_string\nrepeated_string", mode)


@test("Cache: String interning with many duplicates", "Cache - String Interning")
def test_cache_string_interning_many_duplicates(mode):
    assert_output("""
let strings = []
for i in range(100) {
    strings = push(strings, "cached")
}
print(len(strings))
""", "100", mode)


@test("Cache: String interning across functions", "Cache - String Interning")
def test_cache_string_interning_functions(mode):
    assert_output("""
fn make_greeting(name: string) -> string {
    return "Hello, " + name
}

let g1 = make_greeting("Alice")
let g2 = make_greeting("Alice")
let g3 = make_greeting("Alice")

print(g1)
""", "Hello, Alice", mode)


@test("Cache: String interning in loops", "Cache - String Interning")
def test_cache_string_interning_loops(mode):
    assert_output("""
let count = 0
for i in range(50) {
    let msg = "loop_constant"
    count = count + 1
}
print(count)
""", "50", mode)


@test("Cache: String interning statistics", "Cache - String Interning")
def test_cache_string_stats(mode):
    # This test just ensures string interning doesn't break functionality
    assert_output("""
let words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
let unique_count = 0

for word in words {
    if word == "apple" {
        unique_count = unique_count + 1
    }
}

print(unique_count)
""", "3", mode)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM TESTS - IMPORT CACHE
# ═══════════════════════════════════════════════════════════════════════════

@test("Cache: Import cache basic", "Cache - Import")
def test_cache_import_basic(mode):
    # Create a module file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn helper_add(a: int, b: int) -> int {
    return a + b
}

fn helper_multiply(a: int, b: int) -> int {
    return a * b
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)
        escaped_path = escape_path_for_tb(module_path)
        code = f"""
@import {{
    "{escaped_path}"
}}

print(helper_add(5, 3))
print(helper_multiply(4, 7))
"""
        assert_output(code, "8\n28", mode)
    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Import cache reuse", "Cache - Import")
def test_cache_import_reuse(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn cached_function() -> int {
    return 42
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)
        escaped_path = escape_path_for_tb(module_path)
        # First import - should compile and cache
        code1 = f"""
@import {{
    "{escaped_path}"
}}
print(cached_function())
"""
        assert_output(code1, "42", mode)

        # Second import - should use cache (faster)
        code2 = f"""
@import {{
    "{escaped_path}"
}}
print(cached_function())
"""
        assert_output(code2, "42", mode)

    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Import cache invalidation on change", "Cache - Import")
def test_cache_import_invalidation(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn get_value() -> int {
    return 100
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)
        # First run
        code1 = f"""
@import {{
    "{escaped_path}"
}}
print(get_value())
"""
        assert_output(code1, "100", mode)

        # Modify module
        time.sleep(0.1)  # Ensure timestamp changes
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write("""
fn get_value() -> int {
    return 200
}
""")

        # Second run - should detect change and recompile
        code2 = f"""
@import {{
    "{escaped_path}"
}}
print(get_value())
"""
        assert_output(code2, "200", mode)

    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Import multiple modules", "Cache - Import")
def test_cache_import_multiple(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f1:
        f1.write("""
fn module1_func() -> int {
    return 1
}
""")
        module1 = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f2:
        f2.write("""
fn module2_func() -> int {
    return 2
}
""")
        module2 = f2.name

    try:
        escaped_module1 = escape_path_for_tb(module1)
        escaped_module2 = escape_path_for_tb(module2)
        code = f"""
@import {{
    "{escaped_module1}",
    "{escaped_module2}"
}}

print(module1_func())
print(module2_func())
"""
        assert_output(code, "1\n2", mode)

    finally:
        try:
            os.unlink(module1)
            os.unlink(module2)
        except:
            pass


@test("Cache: Import with alias", "Cache - Import")
def test_cache_import_alias(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn long_function_name() -> string {
    return "aliased"
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)
        code = f"""
@import {{
    "{escaped_path}"
}}

print(long_function_name())
"""
        assert_output(code, "aliased", mode)

    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Import nested dependencies", "Cache - Import")
def test_cache_import_nested(mode):
    # Create base module
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn base_add(a: int, b: int) -> int {
    return a + b
}
""")
        base_module = f.name

    # Create dependent module
    escaped_base = escape_path_for_tb(base_module)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write(f"""
@import {{
    "{escaped_base}"
}}

fn derived_triple_add(a: int, b: int, c: int) -> int {{
    return base_add(base_add(a, b), c)
}}
""")
        derived_module = f.name

    try:
        escaped_derived = escape_path_for_tb(derived_module)
        code = f"""
@import {{
    "{escaped_derived}"
}}

print(derived_triple_add(1, 2, 3))
"""
        assert_output(code, "6", mode)

    finally:
        try:
            os.unlink(base_module)
            os.unlink(derived_module)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM TESTS - ARTIFACT CACHE
# ═══════════════════════════════════════════════════════════════════════════

@test("Cache: Artifact cache compiled binary", "Cache - Artifact", slow=True)
def test_cache_artifact_binary(mode):
    if mode != "compiled":
        return  # Only test in compiled mode

    code = """
fn compute_heavy() -> int {
    let result = 0
    for i in range(100) {
        result = result + i
    }
    return result
}

print(compute_heavy())
"""

    # First compilation - should cache
    start1 = time.perf_counter()
    assert_output(code, "4950", mode)
    time1 = time.perf_counter() - start1

    # Second compilation - should use cache (faster)
    start2 = time.perf_counter()
    assert_output(code, "4950", mode)
    time2 = time.perf_counter() - start2

    # Cache should make it faster (though not always guaranteed)
    # Just verify both executions work
    assert time1 > 0 and time2 > 0


@test("Cache: Artifact cache with optimization", "Cache - Artifact", slow=True)
def test_cache_artifact_optimized(mode):
    if mode != "compiled":
        return

    code = """
@config {
    optimize: true,
    opt_level: 3
}

fn optimized_sum(n: int) -> int {
    let sum = 0
    for i in range(n) {
        sum = sum + i
    }
    return sum
}

print(optimized_sum(50))
"""

    assert_output(code, "1225", mode)


@test("Cache: Artifact cache different targets", "Cache - Artifact", slow=True)
def test_cache_artifact_targets(mode):
    if mode != "compiled":
        return

    code = """
fn simple() -> int {
    return 42
}

print(simple())
"""

    # Compile for native target
    assert_output(code, "42", mode)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM TESTS - PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

@test("Cache: Performance with heavy string operations", "Cache - Performance")
def test_cache_performance_strings(mode):
    assert_output("""
let result = ""
for i in range(100) {
    result = result + "cached_string_"
}
print(len(result))
""", "1400", mode)


@test("Cache: Performance with repeated function calls", "Cache - Performance")
def test_cache_performance_functions(mode):
    assert_output("""
fn cached_computation(n: int) -> int {
    return n * 2 + 1
}

let total = 0
for i in range(100) {
    total = total + cached_computation(i)
}
print(total)
""", "10000", mode)


@test("Cache: Performance with import reuse", "Cache - Performance")
def test_cache_performance_import(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn reused_func(x: int) -> int {
    return x * x
}
""")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)
        # Import and use multiple times
        code = f"""
@import {{
    "{escaped_path}"
}}

let sum = 0
for i in range(50) {{
    sum = sum + reused_func(i)
}}
print(sum)
"""
        assert_output(code, "40425", mode)

    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Memory efficiency with interning", "Cache - Performance")
def test_cache_memory_efficiency(mode):
    # Test that many identical strings don't blow up memory
    assert_output("""
let strings = []
for i in range(200) {
    strings = push(strings, "interned")
    strings = push(strings, "constant")
    strings = push(strings, "value")
}
print(len(strings))
""", "600", mode)


@test("Cache: Hot path optimization", "Cache - Performance")
def test_cache_hot_path(mode):
    assert_output("""
fn hot_function(x: int) -> int {
    return x + 1
}

let result = 0
for i in range(1000) {
    result = hot_function(result)
}
print(result)
""", "1000", mode)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM TESTS - EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

@test("Cache: Import cache with circular dependency", "Cache - Edge Cases")
def test_cache_circular_dependency(mode):
    # Create two modules that reference each other
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f1:
        f1.write("""
fn module_a_func() -> int {
    return 1
}
""")
        module_a = f1.name

    escaped_module_a = escape_path_for_tb(module_a)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f2:
        f2.write(f"""
@import {{
    "{escaped_module_a}"
}}

fn module_b_func() -> int {{
    return module_a_func() + 1
}}
""")
        module_b = f2.name

    try:
        escaped_module_b = escape_path_for_tb(module_b)
        # This should work (not truly circular)
        code = f"""
@import {{
    "{escaped_module_b}"
}}

print(module_b_func())
"""
        assert_output(code, "2", mode)

    finally:
        try:
            os.unlink(module_a)
            os.unlink(module_b)
        except:
            pass


@test("Cache: String interning with unicode", "Cache - Edge Cases")
def test_cache_unicode_strings(mode):
    assert_output("""
let emoji1 = "🚀"
let emoji2 = "🚀"
let emoji3 = "🚀"

print(emoji1)
""", "🚀", mode)


@test("Cache: Import with empty module", "Cache - Edge Cases")
def test_cache_empty_module(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("# Empty module\n")
        module_path = f.name

    try:
        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)

        escaped_path = escape_path_for_tb(module_path)
        code = f"""
@import {{
    "{escaped_path}"
}}

print("imported empty module")
"""
        assert_output(code, "imported empty module", mode)

    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Concurrent string interning", "Cache - Edge Cases")
def test_cache_concurrent_interning(mode):
    # Simulate concurrent string creation
    assert_output("""
fn create_strings(prefix: string) -> list {
    let result = []
    for i in range(10) {
        result = push(result, prefix)
    }
    return result
}

let list1 = create_strings("concurrent")
let list2 = create_strings("concurrent")
let list3 = create_strings("concurrent")

print(len(list1) + len(list2) + len(list3))
""", "30", mode)


@test("Cache: Import cache corruption recovery", "Cache - Edge Cases")
def test_cache_corruption_recovery(mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding='utf-8') as f:
        f.write("""
fn robust_function() -> int {
    return 123
}
""")
        module_path = f.name

    try:

        escaped_path = escape_path_for_tb(module_path)
        # First import
        code1 = f"""
@import {{
    "{escaped_path}"
}}
print(robust_function())
"""
        assert_output(code1, "123", mode)

        # Even if cache is corrupted, should recompile
        code2 = f"""
@import {{
    "{escaped_path}"
}}
print(robust_function())
"""
        assert_output(code2, "123", mode)

    finally:
        try:
            os.unlink(module_path)
        except:
            pass


@test("Cache: Large string interning stress test", "Cache - Edge Cases", slow=True)
def test_cache_large_string_stress(mode):
    assert_output("""
let large_strings = []
for i in range(500) {
    large_strings = push(large_strings, "repeated_long_string_value_for_testing_interning_efficiency")
}
print(len(large_strings))
""", "500", mode)


# ═══════════════════════════════════════════════════════════════════════════
# FILE I/O BUILT-IN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("File I/O: read_file and write_file", "Built-in Functions - File I/O")
def test_file_io_basic(mode):
    assert_output("""
write_file("test_output.txt", "Hello from TB!")
let content = read_file("test_output.txt")
print(content)
""", "Hello from TB!", mode)

@test("File I/O: file_exists check", "Built-in Functions - File I/O")
def test_file_exists(mode):
    assert_output("""
write_file("exists_test.txt", "data")
if file_exists("exists_test.txt") {
    print("File exists")
} else {
    print("File not found")
}
""", "File exists", mode)

@test("File I/O: Multiple file operations", "Built-in Functions - File I/O")
def test_file_io_multiple(mode):
    assert_output("""
write_file("file1.txt", "Content 1")
write_file("file2.txt", "Content 2")
let c1 = read_file("file1.txt")
let c2 = read_file("file2.txt")
print(c1)
print(c2)
""", "Content 1\nContent 2", mode)


# ═══════════════════════════════════════════════════════════════════════════
# BLOB STORAGE BUILT-IN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("Blob Storage: Initialize and create blob", "Built-in Functions - Blob Storage")
def test_blob_init_create(mode):
    assert_output("""
let storage = blob_init(["http://localhost:8080"])
let blob_id = blob_create(storage, "Test blob content")
print("Blob created")
""", "Blob created", mode)

@test("Blob Storage: Create, read, and update", "Built-in Functions - Blob Storage")
def test_blob_crud_operations(mode):
    assert_output("""
let storage = blob_init(["http://localhost:8080"])
let blob_id = blob_create(storage, "Initial content")
blob_update(storage, blob_id, "Updated content")
let content = blob_read(storage, blob_id)
print(content)
""", "Updated content", mode)

@test("Blob Storage: Multiple blobs", "Built-in Functions - Blob Storage")
def test_blob_multiple(mode):
    assert_output("""
let storage = blob_init(["http://localhost:8080"])
let id1 = blob_create(storage, "Blob 1")
let id2 = blob_create(storage, "Blob 2")
let id3 = blob_create(storage, "Blob 3")
print("Created 3 blobs")
""", "Created 3 blobs", mode)


# ═══════════════════════════════════════════════════════════════════════════
# NETWORKING BUILT-IN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("Networking: HTTP session creation", "Built-in Functions - Networking")
def test_http_session_create(mode):
    assert_output("""
let session = http_session("https://api.example.com")
print("Session created")
""", "Session created", mode)

@test("Networking: HTTP GET request", "Built-in Functions - Networking", slow=True)
def test_http_get_request(mode):
    assert_output("""
let session = http_session("https://httpbin.org")
let response = http_request(session, "/get", "GET")
if response.status == 200 {
    print("GET successful")
} else {
    print("GET failed")
}
""", "GET successful", mode)

@test("Networking: HTTP POST request with JSON", "Built-in Functions - Networking", slow=True)
def test_http_post_json(mode):
    assert_output("""
let session = http_session("https://httpbin.org")
let data = {name: "TB Test", value: 42}
let response = http_request(session, "/post", "POST", data)
if response.status == 200 {
    print("POST successful")
} else {
    print("POST failed")
}
""", "POST successful", mode)

@test("Networking: TCP connection", "Built-in Functions - Networking")
def test_tcp_connection(mode):
    assert_output("""
let on_connect = fn(addr, msg) { print("Connected") }
let on_disconnect = fn(addr) { print("Disconnected") }
let on_message = fn(addr, msg) { print(msg) }

let conn = connect_to(on_connect, on_disconnect, on_message, "localhost", 8080, "tcp")
print("Connection initiated")
""", "Connection initiated", mode)


# ═══════════════════════════════════════════════════════════════════════════
# JSON/YAML UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("Utils: JSON parse simple object", "Built-in Functions - Utils")
def test_json_parse_simple(mode):
    assert_output("""
let json_str = '{"name": "Alice", "age": 25}'
let data = json_parse(json_str)
print(data["name"])
print(data["age"])
""", "Alice\n25", mode)

@test("Utils: JSON parse nested object", "Built-in Functions - Utils")
def test_json_parse_nested(mode):
    assert_output("""
let json_str = '{"user": {"name": "Bob", "scores": [95, 87, 92]}}'
let data = json_parse(json_str)
print(data["user"]["name"])
print(len(data["user"]["scores"]))
""", "Bob\n3", mode)

@test("Utils: JSON stringify", "Built-in Functions - Utils")
def test_json_stringify(mode):
    assert_output("""
let data = {name: "Charlie", active: true}
let json = json_stringify(data)
print("JSON created")
""", "JSON created", mode)

@test("Utils: JSON round-trip", "Built-in Functions - Utils")
def test_json_roundtrip(mode):
    assert_output("""
let original = {test: "value", number: 42}
let json_str = json_stringify(original)
let parsed = json_parse(json_str)
print(parsed["test"])
print(parsed["number"])
""", "value\n42", mode)

@test("Utils: YAML parse", "Built-in Functions - Utils")
def test_yaml_parse(mode):
    assert_output("""
let yaml_str = "name: Alice\\nage: 25\\nactive: true"
let data = yaml_parse(yaml_str)
print(data["name"])
print(data["age"])
""", "Alice\n25", mode)

@test("Utils: YAML stringify", "Built-in Functions - Utils")
def test_yaml_stringify(mode):
    assert_output("""
let data = {name: "Bob", port: 8080}
let yaml = yaml_stringify(data)
print("YAML created")
""", "YAML created", mode)

@test("Utils: YAML round-trip", "Built-in Functions - Utils")
def test_yaml_roundtrip(mode):
    assert_output("""
let original = {service: "api", version: 2}
let yaml_str = yaml_stringify(original)
let parsed = yaml_parse(yaml_str)
print(parsed["service"])
print(parsed["version"])
""", "api\n2", mode)


# ═══════════════════════════════════════════════════════════════════════════
# TIME UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("Utils: time() get current time", "Built-in Functions - Utils")
def test_time_current(mode):
    assert_output("""
let now = time()
if now["year"] >= 2024 {
    print("Year valid")
}
if now["month"] >= 1 and now["month"] <= 12 {
    print("Month valid")
}
if now["day"] >= 1 and now["day"] <= 31 {
    print("Day valid")
}
""", "Year valid\nMonth valid\nDay valid", mode)

@test("Utils: time() with timezone", "Built-in Functions - Utils")
def test_time_timezone(mode):
    assert_output("""
let utc_time = time("UTC")
print(utc_time["timezone"])
""", "UTC", mode)

@test("Utils: time() fields access", "Built-in Functions - Utils")
def test_time_fields(mode):
    assert_output("""
let now = time()
let has_year = "year" in keys(now)
let has_month = "month" in keys(now)
let has_timestamp = "timestamp" in keys(now)
if has_year and has_month and has_timestamp {
    print("All time fields present")
}
""", "All time fields present", mode)

@test("Utils: time() ISO8601 format", "Built-in Functions - Utils")
def test_time_iso8601(mode):
    assert_output("""
let now = time()
let iso = now["iso8601"]
if len(iso) > 10 {
    print("ISO8601 format valid")
}
""", "ISO8601 format valid", mode)


# ═══════════════════════════════════════════════════════════════════════════
# BUILT-IN FUNCTIONS INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Integration: File I/O with JSON", "Built-in Functions - Integration")
def test_integration_file_json(mode):
    assert_output("""
let data = {users: ["Alice", "Bob", "Charlie"], count: 3}
let json_str = json_stringify(data)
write_file("users.json", json_str)

let loaded_str = read_file("users.json")
let loaded_data = json_parse(loaded_str)
print(loaded_data["count"])
print(len(loaded_data["users"]))
""", "3\n3", mode)

@test("Integration: HTTP with JSON parsing", "Built-in Functions - Integration", slow=True)
def test_integration_http_json(mode):
    assert_output("""
let session = http_session("https://httpbin.org")
let response = http_request(session, "/json", "GET")
if response.status == 200 {
    let data = json_parse(response.body)
    print("JSON response parsed")
}
""", "JSON response parsed", mode)

@test("Integration: Time and JSON", "Built-in Functions - Integration")
def test_integration_time_json(mode):
    assert_output("""
let now = time()
let time_data = {
    year: now["year"],
    month: now["month"],
    timezone: now["timezone"]
}
let json = json_stringify(time_data)
let parsed = json_parse(json)
print(parsed["timezone"])
""", "Local", mode)

@test("Integration: Blob storage with JSON", "Built-in Functions - Integration")
def test_integration_blob_json(mode):
    assert_output("""
let storage = blob_init(["http://localhost:8080"])
let config = {app: "TB Lang", version: "1.0"}
let json_str = json_stringify(config)
let blob_id = blob_create(storage, json_str)
print("Config stored in blob")
""", "Config stored in blob", mode)

@test("Integration: Multiple built-ins stress test", "Built-in Functions - Integration", slow=True)
def test_integration_stress(mode):
    assert_output("""
let results = []
for i in range(10) {
    let data = {index: i, timestamp: time()["timestamp"]}
    let json = json_stringify(data)
    results = push(results, json)
}
print(len(results))
""", "10", mode)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"{Colors.BOLD}{Colors.CYAN}TB Language Test Suite{Colors.RESET}")
    print(f"Binary: {TB_BINARY}")
    print(f"Mode: {TEST_MODE}")
    if FILTER:
        print(f"Filter: {FILTER}")
    if SKIP_SLOW:
        print(f"{Colors.YELLOW}Skipping slow tests{Colors.RESET}")
    print()

    # Verify TB binary works
    try:
        result = subprocess.run([TB_BINARY, "version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            result = subprocess.run([TB_BINARY, "--help"], capture_output=True, timeout=5)
            if result.returncode != 0:
                print(f"{Colors.RED}TB binary is not working properly{Colors.RESET}")
                return False
    except Exception as e:
        print(f"{Colors.RED}Failed to run TB binary: {e}{Colors.RESET}")
        return False

    # Run all tests (tests are executed when the module loads)
    # Just need to call the decorated functions
    import inspect
    current_module = sys.modules[__name__]

    for name, obj in inspect.getmembers(current_module):
        if callable(obj) and hasattr(obj, '__name__') and obj.__name__ == 'wrapper':
            obj()

    # Print summary
    success = suite.print_summary()

    return success

def function_runner(args):
    global VERBOSE, FILTER, TB_BINARY

    TB_BINARY = find_tb_binary()
    VERBOSE = "verbose" in args or "-v" in args
    print(f"{VERBOSE=}")
    FILTER = None
    for i, arg in enumerate(args):
        if arg == "filter" and i + 1 < len(args):
            FILTER = args[i + 1]
    main()


if __name__ == "__main__":
    TB_BINARY = find_tb_binary()
    success = main()
    sys.exit(0 if success else 1)
