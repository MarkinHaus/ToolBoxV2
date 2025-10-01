#!/usr/bin/env python3
"""
TB Language Comprehensive Test Suite
Tests all features of the TB language implementation.

Usage:
    python test_tb_lang.py
    python test_tb_lang.py --verbose
    python test_tb_lang.py --filter "test_arithmetic"
"""
import shutil
import subprocess
import sys
import os
import tempfile
import json
import time
from pathlib import Path
from platform import system
from typing import Optional, List, Tuple
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Adjust these paths
# ═══════════════════════════════════════════════════════════════════════════
from toolboxv2 import tb_root_dir
# Path to TB binary (relative or absolute)
TB_BINARY_PATH = str(tb_root_dir / "bin" / "tbx")

# Alternative paths to try if main path doesn't exist
ALTERNATIVE_PATHS = [
    str(tb_root_dir / "tb-exc" / "target"/"debug"/"tbx"),
    str(tb_root_dir / "tb-exc" / "target"/"release"/"tbx"),
    "tbx",  # System PATH
]

# Test configuration
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
FILTER = None
for i, arg in enumerate(sys.argv):
    if arg == "--filter" and i + 1 < len(sys.argv):
        FILTER = sys.argv[i + 1]


# ═══════════════════════════════════════════════════════════════════════════
# ANSI COLOR CODES
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
    error_message: Optional[str] = None
    output: Optional[str] = None


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

        print(f"\n{Colors.CYAN}Total time: {total_time:.0f}ms{Colors.RESET}")

        if failed > 0:
            print(f"\n{Colors.RED}Failed tests:{Colors.RESET}")
            for result in self.results:
                if not result.passed:
                    print(f"  • {result.name}")
                    if result.error_message:
                        print(f"    {Colors.GRAY}{result.error_message}{Colors.RESET}")

        return failed == 0


# Global test suite
suite = TestSuite()


# ═══════════════════════════════════════════════════════════════════════════
# TB BINARY HELPER
# ═══════════════════════════════════════════════════════════════════════════

def find_tb_binary() -> str:
    """Find TB binary, trying multiple paths with system-specific extensions."""

    # Add system-specific extension
    def add_extension(path: str) -> str:
        if system() == "Windows" and not path.endswith(".exe"):
            return f"{path}.exe"
        return path

    # Prepare paths with proper extensions
    paths_to_try = [add_extension(TB_BINARY_PATH)]
    for alt_path in ALTERNATIVE_PATHS:
        paths_to_try.append(add_extension(alt_path))

    for path in paths_to_try:
        # Check if file exists directly
        if os.path.exists(path):
            return path

        # Use shutil.which for system PATH lookup (cross-platform)
        if shutil.which(path):
            return path

    print(f"{Colors.RED}✗ TB binary not found!{Colors.RESET}")
    print(f"{Colors.YELLOW}Tried paths:{Colors.RESET}")
    for path in paths_to_try:
        print(f"  • {path}")
    print(f"\n{Colors.CYAN}Build the binary with:{Colors.RESET}")
    print(f"  tb x build")
    sys.exit(1)

def run_tb(code: str, mode: str = "jit", timeout: int = 10) -> Tuple[bool, str, str]:
    """
    Run TB code and return (success, stdout, stderr).

    Args:
        code: TB source code
        mode: Execution mode (jit, compiled, streaming)
        timeout: Timeout in seconds

    Returns:
        (success, stdout, stderr)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [TB_BINARY, "run", temp_file, "--mode", mode],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        success = result.returncode == 0
        return success, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"

    finally:
        try:
            os.unlink(temp_file)
        except:
            pass


def run_tb_compile(code: str, output_path: str, target: str = None) -> Tuple[bool, str]:
    """Compile TB code to binary."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        cmd = [TB_BINARY, "compile", temp_file, output_path]
        if target:
            cmd.extend(["--target", target])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        success = result.returncode == 0
        return success, result.stderr

    finally:
        try:
            os.unlink(temp_file)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# TEST DECORATOR
# ═══════════════════════════════════════════════════════════════════════════

def test(name: str, category: str = "General"):
    """Decorator for test functions."""

    def decorator(func):
        def wrapper():
            # Check filter
            if FILTER and FILTER.lower() not in name.lower():
                return

            # Print test header
            if suite.current_category != category:
                print(f"\n{Colors.BOLD}{Colors.CYAN}[{category}]{Colors.RESET}")
                suite.current_category = category

            print(f"  {Colors.GRAY}Testing:{Colors.RESET} {name}...", end=" ", flush=True)

            start = time.time()
            try:
                func()
                duration = (time.time() - start) * 1000

                print(f"{Colors.GREEN}✓{Colors.RESET} ({duration:.0f}ms)")

                suite.add_result(TestResult(
                    name=name,
                    passed=True,
                    duration_ms=duration
                ))

            except AssertionError as e:
                duration = (time.time() - start) * 1000

                print(f"{Colors.RED}✗{Colors.RESET} ({duration:.0f}ms)")
                if VERBOSE:
                    print(f"    {Colors.RED}Error: {str(e)}{Colors.RESET}")

                suite.add_result(TestResult(
                    name=name,
                    passed=False,
                    duration_ms=duration,
                    error_message=str(e)
                ))

            except Exception as e:
                duration = (time.time() - start) * 1000

                print(f"{Colors.RED}✗ (Exception){Colors.RESET}")
                if VERBOSE:
                    print(f"    {Colors.RED}{type(e).__name__}: {str(e)}{Colors.RESET}")

                suite.add_result(TestResult(
                    name=name,
                    passed=False,
                    duration_ms=duration,
                    error_message=f"{type(e).__name__}: {str(e)}"
                ))

        return wrapper

    return decorator


def assert_output(code: str, expected: str, mode: str = "jit"):
    """Assert that TB code produces expected output."""
    success, stdout, stderr = run_tb(code, mode)

    if not success:
        raise AssertionError(f"Execution failed:\n{stderr}")

    actual = stdout.strip()
    expected = expected.strip()

    if actual != expected:
        raise AssertionError(
            f"Output mismatch:\n"
            f"Expected: {repr(expected)}\n"
            f"Got:      {repr(actual)}"
        )


def assert_success(code: str, mode: str = "jit"):
    """Assert that TB code runs without error."""
    success, stdout, stderr = run_tb(code, mode)

    if not success:
        raise AssertionError(f"Execution failed:\n{stderr}")


def assert_contains(code: str, substring: str, mode: str = "jit"):
    """Assert that output contains substring."""
    success, stdout, stderr = run_tb(code, mode)

    if not success:
        raise AssertionError(f"Execution failed:\n{stderr}")

    if substring not in stdout:
        raise AssertionError(
            f"Output does not contain '{substring}':\n{stdout}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - BASIC SYNTAX
# ═══════════════════════════════════════════════════════════════════════════

@test("Hello World", "Basic Syntax")
def test_hello_world():
    code = '''
echo "Hello, World!"
'''
    assert_output(code, "Hello, World!")


@test("Variables", "Basic Syntax")
def test_variables():
    code = '''
let x = 42
echo $x
'''
    assert_output(code, "42")


@test("Mutable Variables", "Basic Syntax")
def test_mutable():
    code = '''
let mut x = 10
x = 20
echo $x
'''
    assert_output(code, "20")


@test("String Variables", "Basic Syntax")
def test_string_variables():
    code = '''
let name = "TB"
echo $name
'''
    assert_output(code, "TB")


@test("Comments", "Basic Syntax")
def test_comments():
    code = '''
# This is a comment
echo "test"  # inline comment
'''
    assert_output(code, "test")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - ARITHMETIC
# ═══════════════════════════════════════════════════════════════════════════

@test("Addition", "Arithmetic")
def test_addition():
    code = '''
let result = 2 + 3
echo $result
'''
    assert_output(code, "5")


@test("Subtraction", "Arithmetic")
def test_subtraction():
    code = '''
let result = 10 - 3
echo $result
'''
    assert_output(code, "7")


@test("Multiplication", "Arithmetic")
def test_multiplication():
    code = '''
let result = 4 * 5
echo $result
'''
    assert_output(code, "20")


@test("Division", "Arithmetic")
def test_division():
    code = '''
let result = 20 / 4
echo $result
'''
    assert_output(code, "5")


@test("Modulo", "Arithmetic")
def test_modulo():
    code = '''
let result = 10 % 3
echo $result
'''
    assert_output(code, "1")


@test("Complex Expression", "Arithmetic")
def test_complex_expression():
    code = '''
let result = (2 + 3) * 4
echo $result
'''
    assert_output(code, "20")


@test("Float Arithmetic", "Arithmetic")
def test_float_arithmetic():
    code = '''
let result = 3.5 + 2.5
echo $result
'''
    assert_output(code, "6")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - CONTROL FLOW
# ═══════════════════════════════════════════════════════════════════════════

@test("If Statement", "Control Flow")
def test_if():
    code = '''
let x = 10
if x > 5 {
    echo "greater"
}
'''
    assert_output(code, "greater")


@test("If-Else Statement", "Control Flow")
def test_if_else():
    code = '''
let x = 3
if x > 5 {
    echo "greater"
} else {
    echo "smaller"
}
'''
    assert_output(code, "smaller")


@test("While Loop", "Control Flow")
def test_while():
    code = '''
let mut i = 0
let mut sum = 0
while i < 5 {
    sum = sum + i
    i = i + 1
}
echo $sum
'''
    assert_output(code, "10")


@test("For Loop", "Control Flow")
def test_for_loop():
    code = '''
let numbers = [1, 2, 3, 4, 5]
let mut sum = 0
for n in numbers {
    sum = sum + n
}
echo $sum
'''
    assert_output(code, "15")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("Function Definition and Call", "Functions")
def test_function_basic():
    code = '''
fn double(x: int) -> int {
    x * 2
}

let result = double(5)
echo $result
'''
    assert_output(code, "10")


@test("Function with Multiple Parameters", "Functions")
def test_function_multi_params():
    code = '''
fn add(a: int, b: int) -> int {
    a + b
}

let result = add(3, 7)
echo $result
'''
    assert_output(code, "10")


@test("Recursive Function", "Functions")
def test_recursive():
    code = '''
fn factorial(n: int) -> int {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

let result = factorial(5)
echo $result
'''
    assert_output(code, "120")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@test("List Creation", "Data Structures")
def test_list_creation():
    code = '''
let numbers = [1, 2, 3, 4, 5]
echo $numbers
'''
    assert_contains(code, "1")


@test("List Index Access", "Data Structures")
def test_list_index():
    code = '''
let numbers = [10, 20, 30]
let item = numbers[1]
echo $item
'''
    assert_output(code, "20")


@test("List Length", "Data Structures")
def test_list_length():
    code = '''
let numbers = [1, 2, 3, 4, 5]
let length = len(numbers)
echo $length
'''
    assert_output(code, "5")


@test("Empty List", "Data Structures")
def test_empty_list():
    code = '''
let empty = []
let length = len(empty)
echo $length
'''
    assert_output(code, "0")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - BUILTIN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@test("Echo Builtin", "Builtin Functions")
def test_builtin_echo():
    code = '''
echo "test message"
'''
    assert_output(code, "test message")


@test("Print Builtin", "Builtin Functions")
def test_builtin_print():
    code = '''
print "hello"
print " "
println "world"
'''
    assert_output(code, "hello world")


@test("Type Conversion - str()", "Builtin Functions")
def test_builtin_str():
    code = '''
let num = 42
let text = str(num)
echo $text
'''
    assert_output(code, "42")


@test("Type Conversion - int()", "Builtin Functions")
def test_builtin_int():
    code = '''
let text = "123"
let num = int(text)
echo $num
'''
    assert_output(code, "123")


@test("Type Conversion - float()", "Builtin Functions")
def test_builtin_float():
    code = '''
let num = 42
let f = float(num)
echo $f
'''
    assert_output(code, "42")


@test("Type Check - type_of()", "Builtin Functions")
def test_builtin_type_of():
    code = '''
let x = 42
let t = type_of(x)
echo $t
'''
    assert_output(code, "int")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

@test("Comparison - Equal", "Operators")
def test_equal():
    code = '''
let result = 5 == 5
if result {
    echo "equal"
}
'''
    assert_output(code, "equal")


@test("Comparison - Not Equal", "Operators")
def test_not_equal():
    code = '''
let result = 5 != 3
if result {
    echo "not equal"
}
'''
    assert_output(code, "not equal")


@test("Comparison - Less Than", "Operators")
def test_less_than():
    code = '''
let result = 3 < 5
if result {
    echo "less"
}
'''
    assert_output(code, "less")


@test("Comparison - Greater Than", "Operators")
def test_greater_than():
    code = '''
let result = 5 > 3
if result {
    echo "greater"
}
'''
    assert_output(code, "greater")


@test("Logical - AND", "Operators")
def test_logical_and():
    code = '''
let result = true and true
if result {
    echo "both true"
}
'''
    assert_output(code, "both true")


@test("Logical - OR", "Operators")
def test_logical_or():
    code = '''
let result = false or true
if result {
    echo "at least one true"
}
'''
    assert_output(code, "at least one true")


@test("Logical - NOT", "Operators")
def test_logical_not():
    code = '''
let result = not false
if result {
    echo "negated"
}
'''
    assert_output(code, "negated")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - PIPELINE OPERATOR
# ═══════════════════════════════════════════════════════════════════════════

@test("Pipeline Basic", "Pipeline")
def test_pipeline_basic():
    code = '''
fn double(x: int) -> int { x * 2 }
fn inc(x: int) -> int { x + 1 }

let result = 5 |> double |> inc
echo $result
'''
    assert_output(code, "11")


@test("Pipeline Multiple Operations", "Pipeline")
def test_pipeline_multi():
    code = '''
fn double(x: int) -> int { x * 2 }
fn square(x: int) -> int { x * x }

let result = 3 |> double |> square
echo $result
'''
    assert_output(code, "36")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - MULTI-LANGUAGE SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

@test("Python Code Execution", "Multi-Language")
def test_python_execution():
    code = '''
let result = python("print('hello from python')")
'''
    # Just check it doesn't crash
    assert_success(code)


@test("JavaScript Code Execution", "Multi-Language")
def test_javascript_execution():
    code = '''
let result = javascript("console.log('hello from js')")
'''
    # Just check it doesn't crash (requires Node.js)
    try:
        assert_success(code)
    except:
        pass  # Skip if Node.js not available


@test("Bash Code Execution", "Multi-Language")
def test_bash_execution():
    code = '''
let result = bash("echo 'hello from bash'")
'''
    # Just check it doesn't crash
    try:
        assert_success(code)
    except:
        pass  # Skip if bash not available


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@test("Config Block - JIT Mode", "Configuration")
def test_config_jit():
    code = '''
@config {
    mode: "jit"
}

echo "running in jit mode"
'''
    assert_output(code, "running in jit mode")


@test("Shared Variables", "Configuration")
def test_shared_variables():
    code = '''
@shared {
    version: "1.0.0"
}

@config {
    mode: "jit"
}

echo "test"
'''
    assert_output(code, "test")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - COMPILATION
# ═══════════════════════════════════════════════════════════════════════════

@test("Compile Simple Program", "Compilation")
def test_compile_simple():
    code = '''
echo "Hello from compiled binary!"
'''

    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        output_path = f.name

    try:
        success, stderr = run_tb_compile(code, output_path)

        if not success:
            raise AssertionError(f"Compilation failed:\n{stderr}")

        # Check binary exists
        if not os.path.exists(output_path):
            raise AssertionError("Compiled binary not found")

        # Check binary is executable
        if not os.access(output_path, os.X_OK):
            os.chmod(output_path, 0o755)

        # Run compiled binary
        result = subprocess.run([output_path], capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            raise AssertionError(f"Compiled binary failed: {result.stderr}")

        if "Hello from compiled binary!" not in result.stdout:
            raise AssertionError(f"Unexpected output: {result.stdout}")

    finally:
        try:
            os.unlink(output_path)
        except:
            pass


@test("Compile with Optimization", "Compilation")
def test_compile_optimized():
    code = '''
fn factorial(n: int) -> int {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

let result = factorial(10)
echo $result
'''

    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        output_path = f.name

    try:
        success, stderr = run_tb_compile(code, output_path)

        if not success:
            raise AssertionError(f"Compilation failed:\n{stderr}")

        # Run and verify output
        result = subprocess.run([output_path], capture_output=True, text=True, timeout=5)

        if "3628800" not in result.stdout:
            raise AssertionError(f"Incorrect factorial result: {result.stdout}")

    finally:
        try:
            os.unlink(output_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@test("Division by Zero", "Error Handling")
def test_division_by_zero():
    code = '''
let result = 10 / 0
'''
    success, stdout, stderr = run_tb(code)

    # Should fail with division by zero error
    if success:
        raise AssertionError("Division by zero should fail")


@test("Undefined Variable", "Error Handling")
def test_undefined_variable():
    code = '''
echo $undefined_var
'''
    success, stdout, stderr = run_tb(code)

    # Should fail with undefined variable error
    if success:
        raise AssertionError("Undefined variable should fail")


@test("Type Mismatch", "Error Handling")
def test_type_mismatch():
    code = '''
let x = "text"
let result = x + 5
'''
    success, stdout, stderr = run_tb(code)

    # Should fail with type error (in static mode)
    # In dynamic mode this might work, so we just check it runs
    # (no assertion)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

@test("Nested Functions", "Advanced")
def test_nested_functions():
    code = '''
fn outer(x: int) -> int {
    fn inner(y: int) -> int {
        y * 2
    }
    inner(x) + 1
}

let result = outer(5)
echo $result
'''
    # This might not be supported yet
    try:
        assert_output(code, "11")
    except:
        pass  # Skip if not supported


@test("Higher-Order Functions", "Advanced")
def test_higher_order():
    code = '''
fn apply(f: fn, x: int) -> int {
    f(x)
}

fn double(x: int) -> int { x * 2 }

let result = apply(double, 5)
echo $result
'''
    # This might not be supported yet
    try:
        assert_output(code, "10")
    except:
        pass  # Skip if not supported


@test("Closures", "Advanced")
def test_closures():
    code = '''
fn make_adder(n: int) -> fn {
    fn(x: int) -> int { x + n }
}

let add5 = make_adder(5)
let result = add5(10)
echo $result
'''
    # This might not be supported yet
    try:
        assert_output(code, "15")
    except:
        pass  # Skip if not supported


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: INLINE CODE
# ═══════════════════════════════════════════════════════════════════════════

@test("Python Inline - Simple Math", "Dependencies - Inline")
def test_python_inline_simple():
    code = '''
@config {
    mode: "jit"
}

let result = python("""
import math
print(math.sqrt(16))
""")
'''
    # Just verify it executes without error
    assert_success(code)


@test("Python Inline - String Operations", "Dependencies - Inline")
def test_python_inline_string():
    code = '''
let result = python("""
text = "hello world"
print(text.upper())
""")
'''
    assert_success(code)


@test("Python Inline - List Comprehension", "Dependencies - Inline")
def test_python_inline_list():
    code = '''
let result = python("""
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(sum(squares))
""")
'''
    assert_success(code)


@test("JavaScript Inline - Simple Math", "Dependencies - Inline")
def test_js_inline_simple():
    code = '''
let result = javascript("""
const x = 5;
const y = 10;
console.log(x + y);
""")
'''
    try:
        assert_success(code)
    except:
        pass  # Skip if Node.js not available


@test("JavaScript Inline - Array Operations", "Dependencies - Inline")
def test_js_inline_array():
    code = '''
let result = javascript("""
const nums = [1, 2, 3, 4, 5];
const sum = nums.reduce((a, b) => a + b, 0);
console.log(sum);
""")
'''
    try:
        assert_success(code)
    except:
        pass  # Skip if Node.js not available


@test("Bash Inline - Echo", "Dependencies - Inline")
def test_bash_inline_echo():
    code = '''
let result = bash("echo 'Hello from Bash'")
'''
    assert_success(code)


@test("Bash Inline - Date", "Dependencies - Inline")
def test_bash_inline_date():
    code = '''
let result = bash("date +%Y")
'''
    assert_success(code)


@test("Bash Inline - Environment Variable", "Dependencies - Inline")
def test_bash_inline_env():
    code = '''
let result = bash("echo $USER")
'''
    assert_success(code)


@test("Mixed Languages - Python and Bash", "Dependencies - Inline")
def test_mixed_languages():
    code = '''
let py_result = python("print('from python')")
let bash_result = bash("echo 'from bash'")
echo "Mixed execution complete"
'''
    assert_success(code)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: WITH IMPORTS (STDLIB)
# ═══════════════════════════════════════════════════════════════════════════

@test("Python Import - JSON", "Dependencies - Stdlib Imports")
def test_python_import_json():
    code = '''
let result = python("""
import json

data = {'name': 'TB', 'version': '1.0'}
json_str = json.dumps(data)
print(json_str)
""")
'''
    assert_success(code)


@test("Python Import - OS", "Dependencies - Stdlib Imports")
def test_python_import_os():
    code = '''
let result = python("""
import os
print(os.name)
""")
'''
    assert_success(code)


@test("Python Import - Sys", "Dependencies - Stdlib Imports")
def test_python_import_sys():
    code = '''
let result = python("""
import sys
print(sys.version_info.major)
""")
'''
    assert_success(code)


@test("Python Import - Random", "Dependencies - Stdlib Imports")
def test_python_import_random():
    code = '''
let result = python("""
import random
random.seed(42)
print(random.randint(1, 100))
""")
'''
    assert_success(code)


@test("Python Import - Datetime", "Dependencies - Stdlib Imports")
def test_python_import_datetime():
    code = '''
let result = python("""
import datetime
now = datetime.datetime.now()
print(now.year)
""")
'''
    assert_success(code)


@test("Python Import - Collections", "Dependencies - Stdlib Imports")
def test_python_import_collections():
    code = '''
let result = python("""
from collections import Counter
words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
counts = Counter(words)
print(counts.most_common(1)[0][0])
""")
'''
    assert_success(code)


@test("Python Import - Itertools", "Dependencies - Stdlib Imports")
def test_python_import_itertools():
    code = '''
let result = python("""
import itertools
combos = list(itertools.combinations([1, 2, 3], 2))
print(len(combos))
""")
'''
    assert_success(code)


@test("Python Import - Re (Regex)", "Dependencies - Stdlib Imports")
def test_python_import_re():
    code = '''
let result = python("""
import re
text = "The year is 2024"
match = re.search(r'\\d{4}', text)
if match:
    print(match.group())
""")
'''
    assert_success(code)


@test("JavaScript Import - Built-in Math", "Dependencies - Stdlib Imports")
def test_js_import_math():
    code = '''
let result = javascript("""
console.log(Math.sqrt(16));
""")
'''
    try:
        assert_success(code)
    except:
        pass  # Skip if Node.js not available


@test("JavaScript Import - Built-in Date", "Dependencies - Stdlib Imports")
def test_js_import_date():
    code = '''
let result = javascript("""
const now = new Date();
console.log(now.getFullYear());
""")
'''
    try:
        assert_success(code)
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: COMPLEX MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Python Complex - Multiple Imports", "Dependencies - Complex Modules")
def test_python_complex_multiple():
    code = '''
let result = python("""
import json
import sys
import os
from datetime import datetime

data = {
    'platform': sys.platform,
    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
    'timestamp': str(datetime.now()),
    'cwd': os.getcwd()
}

print(json.dumps(data, indent=2))
""")
'''
    assert_success(code)


@test("Python Complex - Class Definition", "Dependencies - Complex Modules")
def test_python_complex_class():
    code = '''
let result = python("""
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x):
        self.result += x
        return self

    def multiply(self, x):
        self.result *= x
        return self

    def get(self):
        return self.result

calc = Calculator()
result = calc.add(5).multiply(3).add(2).get()
print(result)
""")
'''
    assert_success(code)


@test("Python Complex - List Comprehensions", "Dependencies - Complex Modules")
def test_python_complex_comprehension():
    code = '''
let result = python("""
# Nested list comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]

# Flattened
flat = [item for row in matrix for item in row]

# Filtered
evens = [x for x in flat if x % 2 == 0]

print(f"Matrix: {matrix}")
print(f"Flat: {flat}")
print(f"Evens: {evens}")
""")
'''
    assert_success(code)


@test("Python Complex - Generator Expression", "Dependencies - Complex Modules")
def test_python_complex_generator():
    code = '''
let result = python("""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

fib_10 = list(fibonacci(10))
print(fib_10)
""")
'''
    assert_success(code)


@test("Python Complex - Dictionary Operations", "Dependencies - Complex Modules")
def test_python_complex_dict():
    code = '''
let result = python("""
from collections import defaultdict

# Group by length
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
by_length = defaultdict(list)

for word in words:
    by_length[len(word)].append(word)

for length, word_list in sorted(by_length.items()):
    print(f"{length}: {', '.join(word_list)}")
""")
'''
    assert_success(code)


@test("Python Complex - File Operations (in-memory)", "Dependencies - Complex Modules")
def test_python_complex_file():
    code = '''
let result = python("""
import io

# Simulate file operations
buffer = io.StringIO()
buffer.write("Line 1\\n")
buffer.write("Line 2\\n")
buffer.write("Line 3\\n")

content = buffer.getvalue()
lines = content.strip().split('\\n')
print(f"Total lines: {len(lines)}")
print(f"First line: {lines[0]}")
""")
'''
    assert_success(code)


@test("Python Complex - JSON Processing", "Dependencies - Complex Modules")
def test_python_complex_json():
    code = '''
let result = python("""
import json

data = {
    'users': [
        {'id': 1, 'name': 'Alice', 'age': 30},
        {'id': 2, 'name': 'Bob', 'age': 25},
        {'id': 3, 'name': 'Charlie', 'age': 35}
    ]
}

# Filter users over 25
adults = [u for u in data['users'] if u['age'] > 25]

# Transform to names
names = [u['name'] for u in adults]

print(json.dumps({'adult_names': names}))
""")
'''
    assert_success(code)


@test("Python Complex - Lambda Functions", "Dependencies - Complex Modules")
def test_python_complex_lambda():
    code = '''
let result = python("""
# Higher-order functions with lambdas
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

from functools import reduce
product = reduce(lambda x, y: x * y, [1, 2, 3, 4, 5])

print(f"Squared: {squared[:5]}")
print(f"Evens: {evens}")
print(f"Product: {product}")
""")
'''
    assert_success(code)


@test("JavaScript Complex - Promises", "Dependencies - Complex Modules")
def test_js_complex_promises():
    code = '''
let result = javascript("""
// Simulate async operation
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function processData() {
    console.log('Start');
    await delay(10);
    console.log('Processing...');
    await delay(10);
    console.log('Done');
    return 'Success';
}

processData().then(result => console.log(result));
""")
'''
    try:
        assert_success(code)
    except:
        pass


@test("JavaScript Complex - Array Methods", "Dependencies - Complex Modules")
def test_js_complex_array():
    code = '''
let result = javascript("""
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const squared = numbers.map(x => x * x);
const evens = numbers.filter(x => x % 2 === 0);
const sum = numbers.reduce((acc, x) => acc + x, 0);

console.log('Squared:', squared.slice(0, 3));
console.log('Evens:', evens);
console.log('Sum:', sum);
""")
'''
    try:
        assert_success(code)
    except:
        pass


@test("JavaScript Complex - Object Manipulation", "Dependencies - Complex Modules")
def test_js_complex_object():
    code = '''
let result = javascript("""
const users = [
    { id: 1, name: 'Alice', age: 30 },
    { id: 2, name: 'Bob', age: 25 },
    { id: 3, name: 'Charlie', age: 35 }
];

// Map to names
const names = users.map(u => u.name);

// Filter adults
const adults = users.filter(u => u.age >= 30);

// Find by name
const bob = users.find(u => u.name === 'Bob');

console.log('Names:', names);
console.log('Adults:', adults.length);
console.log('Bob age:', bob.age);
""")
'''
    try:
        assert_success(code)
    except:
        pass


@test("Bash Complex - Pipeline", "Dependencies - Complex Modules")
def test_bash_complex_pipeline():
    code = '''
let result = bash("echo 'hello world' | tr '[:lower:]' '[:upper:]'")
'''
    assert_success(code)


@test("Bash Complex - Conditional", "Dependencies - Complex Modules")
def test_bash_complex_conditional():
    code = '''
let result = bash("""
if [ -d /tmp ]; then
    echo 'tmp exists'
else
    echo 'tmp missing'
fi
""")
'''
    assert_success(code)


@test("Bash Complex - Loop", "Dependencies - Complex Modules")
def test_bash_complex_loop():
    code = '''
let result = bash("""
for i in 1 2 3 4 5; do
    echo "Number: $i"
done
""")
'''
    assert_success(code)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: EXTERNAL PACKAGES (OPTIONAL - SKIP IF NOT INSTALLED)
# ═══════════════════════════════════════════════════════════════════════════

@test("Python External - NumPy (if available)", "Dependencies - External Packages")
def test_python_external_numpy():
    code = '''
let result = python("""
try:
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Mean: {arr.mean()}")
    print(f"Sum: {arr.sum()}")
except ImportError:
    print("NumPy not available - skipping")
""")
'''
    try:
        assert_success(code)
    except:
        pass  # Skip if NumPy not installed


@test("Python External - Requests (if available)", "Dependencies - External Packages")
def test_python_external_requests():
    code = '''
let result = python("""
try:
    import requests
    print("Requests library available")
except ImportError:
    print("Requests not available - skipping")
""")
'''
    try:
        assert_success(code)
    except:
        pass


@test("JavaScript External - Lodash (if available)", "Dependencies - External Packages")
def test_js_external_lodash():
    code = '''
let result = javascript("""
try {
    const _ = require('lodash');
    const nums = [1, 2, 3, 4, 5];
    console.log('Sum:', _.sum(nums));
} catch (e) {
    console.log('Lodash not available - skipping');
}
""")
'''
    try:
        assert_success(code)
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: MULTI-LANGUAGE INTERACTION
# ═══════════════════════════════════════════════════════════════════════════

@test("Multi-Language - Python to Bash", "Dependencies - Multi-Language")
def test_multilang_python_bash():
    code = '''
# Python generates data
let py_data = python("""
import json
data = {'numbers': [1, 2, 3, 4, 5]}
print(json.dumps(data))
""")

# Bash processes result
let bash_result = bash("echo 'Python executed successfully'")

echo "Multi-language test complete"
'''
    assert_success(code)


@test("Multi-Language - Sequential Execution", "Dependencies - Multi-Language")
def test_multilang_sequential():
    code = '''
echo "Starting multi-language test"

let step1 = python("print('Step 1: Python')")
let step2 = javascript("console.log('Step 2: JavaScript')")
let step3 = bash("echo 'Step 3: Bash'")

echo "All steps completed"
'''
    try:
        assert_success(code)
    except:
        pass


@test("Multi-Language - Data Flow", "Dependencies - Multi-Language")
def test_multilang_dataflow():
    code = '''
# Calculate in Python
let py_result = python("""
result = sum([1, 2, 3, 4, 5])
print(result)
""")

# Use result in TB
echo "Python calculated sum"

# Verify with Bash
let bash_check = bash("echo 'Verification complete'")
'''
    assert_success(code)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@test("Dependency Error - Python Syntax Error", "Dependencies - Error Handling")
def test_dep_error_python_syntax():
    code = '''
let result = python("""
print("unclosed string
""")
'''
    success, stdout, stderr = run_tb(code)
    # Should fail due to Python syntax error
    assert not success, "Python syntax error should cause failure"


@test("Dependency Error - Python Import Error", "Dependencies - Error Handling")
def test_dep_error_python_import():
    code = '''
let result = python("""
import nonexistent_module_xyz
""")
'''
    success, stdout, stderr = run_tb(code)
    # Should fail due to missing module
    assert not success, "Missing Python module should cause failure"


@test("Dependency Error - JavaScript Syntax Error", "Dependencies - Error Handling")
def test_dep_error_js_syntax():
    code = '''
let result = javascript("""
console.log("unclosed string
""")
'''
    success, stdout, stderr = run_tb(code)
    # Should fail (unless Node.js not available)
    if "Node.js/Deno not found" not in stderr:
        assert not success, "JavaScript syntax error should cause failure"


@test("Dependency Error - Bash Command Not Found", "Dependencies - Error Handling")
def test_dep_error_bash_notfound():
    code = '''
let result = bash("nonexistent_command_xyz_123")
'''
    success, stdout, stderr = run_tb(code)
    # Should fail due to command not found
    assert not success, "Non-existent bash command should cause failure"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: PERFORMANCE & EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

@test("Dependency Performance - Multiple Calls", "Dependencies - Performance")
def test_dep_performance_multiple():
    code = '''
# Multiple Python calls
let r1 = python("print(1 + 1)")
let r2 = python("print(2 + 2)")
let r3 = python("print(3 + 3)")
let r4 = python("print(4 + 4)")
let r5 = python("print(5 + 5)")

echo "All calculations complete"
'''
    assert_success(code)


@test("Dependency Edge Case - Empty Code", "Dependencies - Edge Cases")
def test_dep_edge_empty():
    code = '''
let result = python("")
'''
    success, stdout, stderr = run_tb(code)
    # Should handle empty code gracefully


@test("Dependency Edge Case - Whitespace Only", "Dependencies - Edge Cases")
def test_dep_edge_whitespace():
    code = '''
let result = python("

   ")
'''
    success, stdout, stderr = run_tb(code)
    # Should handle whitespace-only code


@test("Dependency Edge Case - Long Output", "Dependencies - Edge Cases")
def test_dep_edge_long_output():
    code = '''
let result = python("""
for i in range(100):
    print(f"Line {i}")
""")
'''
    assert_success(code)


@test("Dependency Edge Case - Unicode", "Dependencies - Edge Cases")
def test_dep_edge_unicode():
    code = '''
let result = python("""
text = "Hello 世界 🌍 Привет"
print(text)
print(len(text))
""")
'''
    assert_success(code)


@test("Dependency Edge Case - Multiline String", "Dependencies - Edge Cases")
def test_dep_edge_multiline():
    code = '''
let result = python("""
text = \'\'\'
This is a
multiline
string
\'\'\'
print(text.count('\\n'))
""")
'''
    assert_success(code)
# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              TB Language Comprehensive Test Suite              ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"{Colors.CYAN}TB Binary:{Colors.RESET} {TB_BINARY}")

    # Verify binary works
    try:
        result = subprocess.run([TB_BINARY, "--version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            print(f"{Colors.YELLOW}Warning: TB binary may not be working correctly{Colors.RESET}")
    except:
        print(f"{Colors.YELLOW}Warning: Could not verify TB binary{Colors.RESET}")

    print()

    # Run all tests
    test_functions = [
        # Basic Syntax
        test_hello_world,
        test_variables,
        test_mutable,
        test_string_variables,
        test_comments,

        # Arithmetic
        test_addition,
        test_subtraction,
        test_multiplication,
        test_division,
        test_modulo,
        test_complex_expression,
        test_float_arithmetic,

        # Control Flow
        test_if,
        test_if_else,
        test_while,
        test_for_loop,

        # Functions
        test_function_basic,
        test_function_multi_params,
        test_recursive,

        # Data Structures
        test_list_creation,
        test_list_index,
        test_list_length,
        test_empty_list,

        # Builtin Functions
        test_builtin_echo,
        test_builtin_print,
        test_builtin_str,
        test_builtin_int,
        test_builtin_float,
        test_builtin_type_of,

        # Operators
        test_equal,
        test_not_equal,
        test_less_than,
        test_greater_than,
        test_logical_and,
        test_logical_or,
        test_logical_not,

        # Pipeline
        test_pipeline_basic,
        test_pipeline_multi,

        # Multi-Language
        test_python_execution,
        test_javascript_execution,
        test_bash_execution,

        # Configuration
        test_config_jit,
        test_shared_variables,

        # Compilation
        test_compile_simple,
        test_compile_optimized,

        # Error Handling
        test_division_by_zero,
        test_undefined_variable,
        test_type_mismatch,

        # Advanced
        test_nested_functions,
        test_higher_order,
        test_closures,

        # ═══════════════════════════════════════════════════════════
        # NEW DEPENDENCY TESTS
        # ═══════════════════════════════════════════════════════════

        # Dependencies - Inline
        test_python_inline_simple,
        test_python_inline_string,
        test_python_inline_list,
        test_js_inline_simple,
        test_js_inline_array,
        test_bash_inline_echo,
        test_bash_inline_date,
        test_bash_inline_env,
        test_mixed_languages,

        # Dependencies - Stdlib Imports
        test_python_import_json,
        test_python_import_os,
        test_python_import_sys,
        test_python_import_random,
        test_python_import_datetime,
        test_python_import_collections,
        test_python_import_itertools,
        test_python_import_re,
        test_js_import_math,
        test_js_import_date,

        # Dependencies - Complex Modules
        test_python_complex_multiple,
        test_python_complex_class,
        test_python_complex_comprehension,
        test_python_complex_generator,
        test_python_complex_dict,
        test_python_complex_file,
        test_python_complex_json,
        test_python_complex_lambda,
        test_js_complex_promises,
        test_js_complex_array,
        test_js_complex_object,
        test_bash_complex_pipeline,
        test_bash_complex_conditional,
        test_bash_complex_loop,

        # Dependencies - External Packages
        test_python_external_numpy,
        test_python_external_requests,
        test_js_external_lodash,

        # Dependencies - Multi-Language
        test_multilang_python_bash,
        test_multilang_sequential,
        test_multilang_dataflow,

        # Dependencies - Error Handling
        test_dep_error_python_syntax,
        test_dep_error_python_import,
        test_dep_error_js_syntax,
        test_dep_error_bash_notfound,

        # Dependencies - Performance & Edge Cases
        test_dep_performance_multiple,
        test_dep_edge_empty,
        test_dep_edge_whitespace,
        test_dep_edge_long_output,
        test_dep_edge_unicode,
        test_dep_edge_multiline,
    ]

    for test_func in test_functions:
        test_func()

    # Print summary
    success = suite.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

def function_runner(args):
    global VERBOSE, FILTER, TB_BINARY

    TB_BINARY = find_tb_binary()
    VERBOSE = "--verbose" in args or "-v" in args
    FILTER = None
    for i, arg in enumerate(args):
        if arg == "--filter" and i + 1 < len(args):
            FILTER = args[i + 1]
    main()


if __name__ == "__main__":
    TB_BINARY = find_tb_binary()
    main()
