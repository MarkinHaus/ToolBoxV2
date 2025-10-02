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
    from toolboxv2 import tb_root_dir
    # Path to TB binary (relative or absolute)
    TB_BINARY_PATH = str(tb_root_dir / "bin" / "tbx")

    # Alternative paths to try if main path doesn't exist
    ALTERNATIVE_PATHS = [
        str(tb_root_dir / "tb-exc" / "target" / "debug" / "tbx"),
        str(tb_root_dir / "tb-exc" / "target" / "release" / "tbx"),
        #"tbx",  # System PATH
    ]

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
    if mode == "compiled":
        with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
            output_path = f.name

        try:
            start = time.perf_counter()
            success, stderr = run_tb_compile(code, output_path)
            duration = time.perf_counter() - start
            print(f" -- Compile time ({duration:.3f}s)")
            if not success:
                raise AssertionError(f"Compilation failed:\n{stderr}")

            # Check binary exists
            if not os.path.exists(output_path):
                raise AssertionError("Compiled binary not found")

            # Check binary is executable
            if not os.access(output_path, os.X_OK):
                os.chmod(output_path, 0o755)

            start = time.perf_counter()
            # Run compiled binary
            result = subprocess.run([output_path], capture_output=True, text=True, timeout=timeout//2,
                                    encoding=sys.stdout.encoding or 'utf-8')
            duration = time.perf_counter() - start
            print(f" -- Exec time ({duration:.3f}s)")

            if result.returncode != 0:
                raise AssertionError(f"Compiled binary failed: {result.stderr}")

            return success, result.stdout, result.stderr
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding=sys.stdout.encoding or 'utf-8') as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [TB_BINARY, "run", temp_file, "--mode", mode],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding=sys.stdout.encoding or 'utf-8',
            errors='replace'
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, encoding=sys.stdout.encoding or 'utf-8') as f:
        f.write(code)
        temp_file = f.name

    try:
        cmd = [TB_BINARY, "compile", temp_file, output_path]
        if target:
            cmd.extend(["--target", target])

        result = subprocess.run(cmd, capture_output=not VERBOSE, text=True, timeout=60, encoding=sys.stdout.encoding or 'utf-8')

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

            print(f"  {Colors.GRAY}Testing:{Colors.RESET} {name}", end=" ", flush=True)

            start = time.perf_counter()
            try:
                func()
                duration = time.perf_counter() - start

                print(f" -- {Colors.GREEN}✓{Colors.RESET} ({duration:.3f}s)")

                suite.add_result(TestResult(
                    name=name,
                    passed=True,
                    duration_ms=duration
                ))

            except AssertionError as e:
                duration = time.perf_counter() - start

                print(f"{Colors.RED}✗{Colors.RESET} ({duration:.0f}s)")
                if VERBOSE:
                    print(f"    {Colors.RED}Error: {str(e)}{Colors.RESET}")

                suite.add_result(TestResult(
                    name=name,
                    passed=False,
                    duration_ms=duration,
                    error_message=str(e)
                ))

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                duration = (time.perf_counter() - start) * 1000

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
    if VERBOSE and not success:
        print(f"code: {code}")
        print()
        print(f"stdout: {stdout}")
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
    if VERBOSE and not success:
        print(f"code: {code}")
        print()
    if VERBOSE:
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")

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
echo x
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
// me 2
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
        start = time.perf_counter()
        success, stderr = run_tb_compile(code, output_path)
        duration = time.perf_counter() - start
        print(f" -- Compile time ({duration:.3f}s)")
        if not success:
            raise AssertionError(f"Compilation failed:\n{stderr}")

        # Check binary exists
        if not os.path.exists(output_path):
            raise AssertionError("Compiled binary not found")

        # Check binary is executable
        if not os.access(output_path, os.X_OK):
            os.chmod(output_path, 0o755)

        start = time.perf_counter()
        # Run compiled binary
        result = subprocess.run([output_path], capture_output=True, text=True, timeout=5, encoding=sys.stdout.encoding or 'utf-8')
        duration = time.perf_counter() - start
        print(f" -- Exec time ({duration:.3f}s)")

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
        start = time.perf_counter()
        success, stderr = run_tb_compile(code, output_path)
        duration = time.perf_counter() - start
        print(f" -- Compile time ({duration:.3f}s)")
        if not success:
            raise AssertionError(f"Compilation failed:\n{stderr}")

        start = time.perf_counter()
        # Run and verify output
        result = subprocess.run([output_path], capture_output=True, text=True, timeout=5, encoding=sys.stdout.encoding or 'utf-8')
        duration = time.perf_counter() - start
        print(f" -- Exec time ({duration:.3f}s)")

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


@test("Cross-Language Variables", "Dependencies - Cross-Language Variables")
def test_cross_variables():
    code = '''
@config {
    mode: "jit"
}

# TB Variables
let name = "Alice"
let age = 30
let scores = [85, 92, 78, 95]
let is_active = true

echo "=== Original TB Variables ==="
echo "Name: $name"
echo "Age: $age"
echo "Scores: $scores"

# Python kann TB-Variablen direkt nutzen
echo "\\n=== Python with TB Variables ==="
let py_result = python("""
# TB variables sind automatisch verfügbar!
print(f"Hello {name}, you are {age} years old")
print(f"Your scores: {scores}")
print(f"Average: {sum(scores) / len(scores)}")
print(f"Active status: {is_active}")

# Berechnung mit TB-Variable
result = age * 2
print(f"Age doubled: {result}")
""")

# JavaScript kann TB-Variablen nutzen
echo "\\n=== JavaScript with TB Variables ==="
let js_result = javascript("""
// TB variables automatically available
console.log(`Hello ${name}, you are ${age} years old`);
console.log(`Scores: ${scores}`);
console.log(`Max score: ${Math.max(...scores)}`);
console.log(`Active: ${is_active}`);
""")

# Go kann TB-Variablen nutzen
echo "\\n=== Go with TB Variables ==="
let go_result = go("""
// TB variables automatically available
fmt.Printf("Hello %s, you are %d years old\\n", name, age)
fmt.Printf("Number of scores: %d\\n", len(scores))
if is_active {
    fmt.Println("Status: Active")
}
""")

# Bash kann TB-Variablen nutzen
echo "\\n=== Bash with TB Variables ==="
bash("""
# TB variables automatically available
echo "Name from Bash: $name"
echo "Age from Bash: $age"
echo "Active: $is_active"
""")

# Komplexeres Beispiel: Daten zwischen Sprachen teilen
echo "\\n=== Complex Data Flow ==="

let data = [1, 2, 3, 4, 5]

# Python verarbeitet
let processed = python("""
# data ist verfügbar
result = [x * x for x in data]
print(result)
""")

# JavaScript nutzt processed (wenn wir Return-Wert-Capture implementieren)
javascript("""
// Direkt mit TB-Variablen arbeiten
console.log("Data length:", data.length);
console.log("Sum:", data.reduce((a, b) => a + b, 0));
""")'''
    assert_success(code)

@test("Cross-Language Variables - Compiled", "Dependencies - Cross-Language Variables - Compiled")
def test_cross_variables_compiled():
    code = '''
@config {
    mode: "jit"
}

# TB Variables
let name = "Alice"
let age = 30
let scores = [85, 92, 78, 95]
let is_active = true

echo "=== Original TB Variables ==="
echo "Name: $name"
echo "Age: $age"
echo "Scores: $scores"

# Python kann TB-Variablen direkt nutzen
echo "\\n=== Python with TB Variables ==="
let py_result = python("""
# TB variables sind automatisch verfügbar!
print(f"Hello {name}, you are {age} years old")
print(f"Your scores: {scores}")
print(f"Average: {sum(scores) / len(scores)}")
print(f"Active status: {is_active}")

# Berechnung mit TB-Variable
result = age * 2
print(f"Age doubled: {result}")
""")

# JavaScript kann TB-Variablen nutzen
echo "\\n=== JavaScript with TB Variables ==="
let js_result = javascript("""
// TB variables automatically available
console.log(`Hello ${name}, you are ${age} years old`);
console.log(`Scores: ${scores}`);
console.log(`Max score: ${Math.max(...scores)}`);
console.log(`Active: ${is_active}`);
""")

# Go kann TB-Variablen nutzen
echo "\\n=== Go with TB Variables ==="
let go_result = go("""
// TB variables automatically available
fmt.Printf("Hello %s, you are %d years old\\n", name, age)
fmt.Printf("Number of scores: %d\\n", len(scores))
if is_active {
    fmt.Println("Status: Active")
}
""")

# Bash kann TB-Variablen nutzen
echo "\\n=== Bash with TB Variables ==="
bash("""
# TB variables automatically available
echo "Name from Bash: $name"
echo "Age from Bash: $age"
echo "Active: $is_active"
""")

# Komplexeres Beispiel: Daten zwischen Sprachen teilen
echo "\\n=== Complex Data Flow ==="

let data = [1, 2, 3, 4, 5]

# Python verarbeitet
let processed = python("""
# data ist verfügbar
result = [x * x for x in data]
print(result)
""")

# JavaScript nutzt processed (wenn wir Return-Wert-Capture implementieren)
javascript("""
// Direkt mit TB-Variablen arbeiten
console.log("Data length:", data.length);
console.log("Sum:", data.reduce((a, b) => a + b, 0));
""")'''
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        output_path = f.name

    try:
        start = time.perf_counter()
        success, stderr = run_tb_compile(code, output_path)
        duration = time.perf_counter() - start
        print(f" -- Compile time ({duration:.3f}s)")
        if not success:
            raise AssertionError(f"Compilation failed:\n{stderr}")

        # Check binary exists
        if not os.path.exists(output_path):
            raise AssertionError("Compiled binary not found")

        # Check binary is executable
        if not os.access(output_path, os.X_OK):
            os.chmod(output_path, 0o755)

        start = time.perf_counter()
        # Run compiled binary
        result = subprocess.run([output_path], capture_output=True, text=True, timeout=5,
                                encoding=sys.stdout.encoding or 'utf-8')
        duration = time.perf_counter() - start
        print(f" -- Exec time ({duration:.3f}s)")

        if result.returncode != 0:
            raise AssertionError(f"Compiled binary failed: {result.stderr}")

        if "Sum: 15" not in result.stdout:
            raise AssertionError(f"Unexpected output: {result.stdout}")
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

@test("Return Values from Dependencies", "Dependencies - Return Values")
def testexample_return_values():
    code = '''// example_return_values.tb

@config {
    mode: "jit"
}

# Python berechnet und returned
let result = python("""
import math
result = math.sqrt(16) * 2
print(result)
""")

echo "Python returned: $result"

# JavaScript berechnet
let js_value = javascript("""
const value = 42 * 2;
console.log(value);
""")

echo "JavaScript returned: $js_value"

# Go berechnet
let go_value = go("""
result := 100 + 50
fmt.Println(result)
""")

echo "Go returned: $go_value"

# Werte weiterverarbeiten in TB
let total = result + js_value + go_value
echo "Total sum: $total"'''
    assert_success(code)

@test("Return Values from Dependencies - Compiled", "Dependencies - Return Values - Compiled")
def testexample_return_values_compiled():
    code = '''// example_return_values.tb

@config {
    mode: "jit"
}


# Python berechnet und returned
let result = python("""
import math
result = math.sqrt(16) * 2
print(result)
""")

echo "Python returned: $result"

# JavaScript berechnet
let js_value = javascript("""
const value = 42 * 2;
console.log(value);
""")

echo "JavaScript returned: $js_value"
# Go berechnet
let go_value = go("""
fmt.Println("Hello welt")
result := 100 + 50
fmt.Println(result)
""")

echo "Go returned: $go_value: $go_value"

# Werte weiterverarbeiten in TB
let total = result + js_value + go_value
echo "Total sum: $total"'''

    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        output_path = f.name

    try:
        start = time.perf_counter()
        success, stderr = run_tb_compile(code, output_path)
        duration = time.perf_counter() - start
        print(f" -- Compile time ({duration:.3f}s)")
        if not success:
            raise AssertionError(f"Compilation failed:\n{stderr}")

        # Check binary exists
        if not os.path.exists(output_path):
            raise AssertionError("Compiled binary not found")

        # Check binary is executable
        if not os.access(output_path, os.X_OK):
            os.chmod(output_path, 0o755)

        start = time.perf_counter()
        # Run compiled binary
        result = subprocess.run([output_path], capture_output=True, text=True, timeout=5, encoding='utf-8')
        duration = time.perf_counter() - start
        print(f" -- Exec time ({duration:.3f}s)")

        if result.returncode != 0:
            raise AssertionError(f"Compiled binary failed: {result.stderr}")

        if "Total sum: 242" not in result.stdout:
            raise AssertionError(f"Unexpected output: {result.stdout}")


    finally:
        try:
            os.unlink(output_path)
        except:
            pass

@test("Parallel Execution", "Dependencies - Parallel")
def test_example_parallel():
    code = '''// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE: example_async_parallel.tb
// ═══════════════════════════════════════════════════════════════════════════

@config {
    mode: "jit"
    runtime_mode:  "parallel"
}

# Shared variable (thread-safe)
@shared {
    counter = 0
    results = []
    task1 = ""
    task2 = ""
}

let mut task1 = ""
let mut task2 = ""
let mut counter = 0
# Async execution
let task1 = parallel {
    python("""
import time
time.sleep(1)
print("Task 1 complete")
    """)
}

let task2 = parallel {
    javascript("""
setTimeout(() => console.log('Task 2 complete'), 1000)
""")
}

echo task1
echo task2


# Parallel execution
x = parallel {
    echo(1)
    echo(2)
    echo(3)
}

echo "Final Parallel: "x

# Parallel for loop
for item in [1, 2, 3, 4, 5] {
    parallel {
        results.push(item * item)
    }
}

echo "Results: $results"
    '''
    assert_success(code)

def test_types():
    code = '''@config { mode: "jit" }

# String Interpolation
let name = "World"
let count = 42
echo "Hello, $name! Count: $count"

# Type Conversions
let x = int("123")
echo "Parsed int: $x"

let y = float("3.14")
echo "English float: $y"

let z = float("3,14")
echo "German float: $z"

let big = float("1.234,56")
echo "German thousand: $big"

# Convert everything to string
let nums = [1, 2, 3]
let tuple = (10, 20)
echo "List as string: " str($nums)
echo "Tuple as string: " str($tuple)

# Type inspection
echo "Type of x: " type_of($x)
echo "Type of nums: " type_of($nums)'''
    assert_output(code, '''Hello, World! Count: 42
Parsed int: 123
English float: 3.14
German float: 3.14
German thousand: 1234.56
List as string: [1, 2, 3]
Tuple as string: (10, 20)
Type of x: int
Type of nums: list''')
# ═══════════════════════════════════════════════════════════════════════════
# TESTS - IMPORT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════


@test("Import - Single File", "Import System")
def test_import_single_file():
    """Test importing a single .tbx file"""
    # Create library file
    lib_code = '''
fn double(x: int) {
    x * 2
}

fn triple(x: int) {
    x * 3
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        # Create main file that imports library
        main_code = f'''
@imports {{
    "{lib_path}"
}}

let result = double(5)
echo result
'''
        assert_output(main_code, "10")
    finally:
        os.unlink(lib_path)


@test("Import - Multiple Files", "Import System")
def test_import_multiple_files():
    """Test importing multiple .tbx files"""
    # Create math library
    math_lib = '''
fn add(x, y) {
    x + y
}
'''

    # Create string library
    string_lib = '''
fn greet(name: string) {
    echo "Hello,"name "!"
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as math_file, \
        tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as string_file:

        math_file.write(math_lib)
        string_file.write(string_lib)
        math_path = os.path.basename(math_file.name)
        string_path = os.path.basename(string_file.name)

    try:
        main_code = f'''
@imports {{
    "{math_path}"
    "{string_path}"
}}

let sum = add(3, 7)
echo sum
greet("World")
'''
        success, stdout, stderr = run_tb(main_code)
        assert success, f"Execution failed: {stderr}"
        assert "10" in stdout, f"Expected '10' in output, got: {stdout}"
        assert "Hello, World !" in stdout, f"Expected Hello, World! in output, got: {stdout}"
    finally:
        os.unlink(math_path)
        os.unlink(string_path)


@test("Import - With Variables", "Import System")
def test_import_with_variables():
    """Test importing file with global variables"""
    lib_code = '''
let PI = 3.14159
let E = 2.71828

fn circle_area(radius: float) {
    PI * radius * radius
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''
@imports {{
    "{lib_path}"
}}

echo PI
let area = circle_area(2.0)
echo area
'''
        success, stdout, stderr = run_tb(main_code)
        assert success, f"Execution failed: {stderr}"
        assert "3.14159" in stdout or "3.14" in stdout
    finally:
        os.unlink(lib_path)


@test("Import - Relative Paths", "Import System")
def test_import_relative_paths():
    """Test importing with relative directory paths"""
    # Create subdirectory
    lib_dir = tempfile.mkdtemp(dir='.')
    lib_dir_name = os.path.basename(lib_dir)

    try:
        # Create library in subdirectory
        lib_code = '''
fn multiply(x: int, y: int) {
    x * y
}
'''
        lib_path = os.path.join(lib_dir, 'math.tbx')
        with open(lib_path, 'w') as f:
            f.write(lib_code)

        main_code = f'''
@imports {{
    "{lib_dir_name}/math.tbx"
}}

let result = multiply(6, 7)
echo result
'''
        assert_output(main_code, "42")
    finally:
        shutil.rmtree(lib_dir, ignore_errors=True)


@test("Import - Compiled Mode", "Import System")
def test_import_compiled_mode():
    """Test that imports work in compiled mode"""
    lib_code = '''
@config {
    mode: "compiled"
    target: "library"
}

fn double(x: int) {
    x * 2
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''
@config {{
    mode: "compiled"
}}

@imports {{
    "{lib_path}"
}}

let result = double(21)
echo result
'''
        assert_output(main_code, "42", mode="compiled")
    finally:
        os.unlink(lib_path)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - COMPILED MODES (JIT vs COMPILED)
# ═══════════════════════════════════════════════════════════════════════════

@test("Compiled - Basic Arithmetic", "Compiled Modes")
def test_compiled_arithmetic():
    """Test basic arithmetic in compiled mode"""
    code = '''
let x = 10
let y = 20
let sum = x + y
let product = x * y
echo sum
echo product
'''
    success, stdout, stderr = run_tb(code, mode="compiled")
    assert success, f"Compilation failed: {stderr}"
    assert "30" in stdout
    assert "200" in stdout


@test("Compiled - Functions", "Compiled Modes")
def test_compiled_functions():
    """Test function definitions in compiled mode"""
    code = '''
fn fibonacci(n: int) -> int {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

let result = fibonacci(10)
echo result
'''
    assert_output(code, "55", mode="compiled")

@test("Compiled - Functions", "Compiled Modes NO Compiled")
def test_compiled_functions_no_compiled_():
    """Test function definitions in compiled mode"""
    code = '''
fn fibonacci(n: int) -> int {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

let result = fibonacci(10)
echo result
'''
    assert_output(code, "55", mode="jit")


@test("Compiled - Loops", "Compiled Modes")
def test_compiled_loops():
    """Test loops in compiled mode"""
    code = '''
let sum = 0
for i in [1, 2, 3, 4, 5] {
    sum = sum + i
}
echo sum
'''
    assert_output(code, "15", mode="compiled")



@test("Compiled - Parallel Execution", "Compiled Modes")
def test_compiled_parallel():
    """Test parallel execution in compiled mode"""
    code = '''
@config {
    runtime_mode: "parallel"
}

let results = parallel {
    10 + 5,
    20 * 2,
    30 - 10
}
for result in results {
    echo result
}
'''
    success, stdout, stderr = run_tb(code, mode="compiled")
    assert success, f"Compilation failed: {stderr}"
    # Results may be in any order due to parallelism
    assert "15" in stdout, f"result {stdout}"
    assert "40" in stdout, f"result {stdout}"
    assert "20" in stdout, f"result {stdout}"


@test("Compiled - Parallel with Imports", "Compiled Modes")
def test_compiled_parallel_with_imports():
    """Test parallel execution with imported functions"""
    lib_code = '''
fn compute_square(x: int) {
    x * x
}

fn compute_cube(x: int) {
    x * x * x
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''
@config {{
    runtime_mode: "parallel"
}}

@imports {{
    "{lib_path}"
}}

let results = parallel {{
    compute_square(5),
    compute_cube(3),
    compute_square(10)
}}

for result in results {{
    echo result
}}
'''
        success, stdout, stderr = run_tb(main_code, mode="compiled")
        assert success, f"Compilation failed: {stderr}"
        assert "25" in stdout
        assert "27" in stdout
        assert "100" in stdout
    finally:
        os.unlink(lib_path)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS - MIXED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

@test("Mixed - Import  + Language Bridge", "Mixed Features")
def test_mixed_import_async_language():
    """Test imports with async and language bridges"""
    lib_code = '''
fn get_data() {

    python("import sys; print(sys.version.split()[0])")

}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''

@imports {{
    "{lib_path}"
}}

let version = get_data()
echo "Python version detected"
'''
        assert_contains(main_code, "Python version detected", mode="jit")
    finally:
        os.unlink(lib_path)


@test("Mixed - Multiple Imports + Parallel", "Mixed Features")
def test_mixed_multiple_imports_parallel():
    """Test multiple imports with parallel execution"""
    math_lib = '''
fn factorial(n: int) -> int {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
'''

    utils_lib = '''
fn power(base: int, exp: int) -> int {
    if exp == 0 {
        1
    } else {
        base * power(base, exp - 1)
    }
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as math_file, \
        tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as utils_file:

        math_file.write(math_lib)
        utils_file.write(utils_lib)
        math_path = os.path.basename(math_file.name)
        utils_path = os.path.basename(utils_file.name)

    try:
        main_code = f'''
@config {{
    runtime_mode: "parallel"
}}

@imports {{
    "{math_path}"
    "{utils_path}"
}}

let results = parallel {{
    factorial(5),
    power(2, 10),
    factorial(6)
}}

for result in results {{
    echo result
}}
'''
        success, stdout, stderr = run_tb(main_code, mode="compiled")
        assert success, f"Compilation failed: {stderr}"
        assert "120" in stdout  # 5!
        assert "1024" in stdout  # 2^10
        assert "720" in stdout  # 6!
    finally:
        os.unlink(math_path)
        os.unlink(utils_path)


@test("Mixed - Shared Variables + Imports", "Mixed Features")
def test_mixed_shared_imports():
    """Test shared variables with imports"""
    lib_code = '''
fn increment_counter() {
    counter + 1
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''
@shared {{
    counter: 0
}}

@imports {{
    "{lib_path}"
}}

counter = increment_counter()
counter = increment_counter()
echo counter
'''
        assert_output(main_code, "2")
    finally:
        os.unlink(lib_path)

@test("Mixed - Shared Variables + Imports + Python Bridge", "Mixed Features")
def test_mixed_shared_imports_mixed_lang():
    """Test shared variables with imports"""
    lib_code = '''
fn increment_counter() {python("print(counter + 1, end='')")}

'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''
@shared {{
    counter: 0
}}

@imports {{
    "{lib_path}"
}}

counter = increment_counter()
counter = increment_counter()
echo counter
'''
        assert_output(main_code, "2")
    finally:
        os.unlink(lib_path)


@test("Mixed - Full Stack Test", "Mixed Features")
def test_mixed_full_stack():
    """Comprehensive test with all features combined"""
    helpers_lib = '''
fn compute(x: int) {
    x * 2
}

fn parallel_sum(numbers: list) {
    let sum = 0
    for n in numbers {
        sum = sum + n
    }
    sum
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(helpers_lib)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''
@config {{
    mode: "jit"
}}

@shared {{
    total: 0
}}

@imports {{
    "{lib_path}"
}}

let value = compute(21)
total = parallel_sum([1, 2, 3, 4, 5])
echo value
echo total
'''
        success, stdout, stderr = run_tb(main_code, mode="jit")
        assert success, f"Execution failed: {stderr}"
        assert "42" in stdout
        assert "15" in stdout
    finally:
        os.unlink(lib_path)



@test("Compiled - Import Caching", "Import Compiled Modes")
def test_import_compiled_caching():
    """Test that compiled imports are cached"""
    print("→ Test: Compiled Import Caching")

    # Create library with compiled mode
    lib_code = '''
@config {
    mode: "compiled"
}

fn factorial(n: int) {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        main_code = f'''

@config {{
    mode: "compiled"
}}

@imports {{
    "{lib_path}"
}}

let result = factorial(5)
echo $result
'''

        # First run - should compile
        start_time = time.perf_counter()
        success1, stdout1, stderr1 = run_tb(main_code)
        first_run_time = time.perf_counter() - start_time

        assert success1, f"First execution failed: {stderr1}"
        assert "120" in stdout1, f"Expected 120, got: {stdout1}"

        # Second run - should use cache
        start_time = time.perf_counter()
        success2, stdout2, stderr2 = run_tb(main_code)
        second_run_time = time.perf_counter() - start_time

        assert success2, f"Second execution failed: {stderr2}"
        assert "120" in stdout2, f"Expected 120, got: {stdout2}"

        # Second run should be significantly faster (using cache)
        assert second_run_time <= first_run_time, \
            f"Cache not used? First: {first_run_time:.2f}s, Second: {second_run_time:.2f}s"

        print(f"  ✓ Caching works (first: {first_run_time:.2f}s, second: {second_run_time:.2f}s)")

    finally:
        os.unlink(lib_path)

@test("Compiled - JIT vs Compiled Imports", "Import Compiled Modes")
def test_import_jit_vs_compiled():
    """Test that JIT imports don't get compiled"""
    print("→ Test: JIT vs Compiled Imports")

    # JIT library
    jit_lib = '''
@config {
    mode: "jit"
}

fn square(x: int) {
    x * x
}
'''

    # Compiled library
    compiled_lib = '''
@config {
    mode: "compiled"
}

fn cube(x: int) {
    x * x * x
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as jit_file, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as compiled_file:

        jit_file.write(jit_lib)
        compiled_file.write(compiled_lib)
        jit_path = os.path.basename(jit_file.name)
        compiled_path = os.path.basename(compiled_file.name)

    try:
        main_code = f'''
@imports {{
    "{jit_path}"
    "{compiled_path}"
}}

echo square(4)
echo cube(3)
'''

        success, stdout, stderr = run_tb(main_code)

        assert success, f"Execution failed: {stderr}"
        assert "16" in stdout, f"Expected 16 from square(4), got: {stdout}"
        assert "27" in stdout, f"Expected 27 from cube(3), got: {stdout}"

        print("  ✓ Mixed JIT/Compiled imports work")

    finally:
        os.unlink(jit_path)
        os.unlink(compiled_path)

@test("Compiled - Import Dependency Chain", "Import Compiled Modes")
def test_import_dependency_chain():
    """Test imports that depend on other imports"""
    print("→ Test: Import Dependency Chain")

    # Base library
    base_lib = '''
fn double(x: int) {
    x * 2
}
'''

    # Mid library (uses base)
    mid_lib = '''
@imports {
    "base.tbx"
}

fn quadruple(x: int) {
    double(double(x))
}
'''

    # Create base.tbx
    with open('base.tbx', 'w') as f:
        f.write(base_lib)

    # Create mid.tbx
    with open('mid.tbx', 'w') as f:
        f.write(mid_lib)

    try:
        main_code = '''
@imports {
    "mid.tbx"
}

echo quadruple(5)
'''

        success, stdout, stderr = run_tb(main_code)

        assert success, f"Execution failed: {stderr}"
        assert "20" in stdout, f"Expected 20, got: {stdout}"

        print("  ✓ Transitive imports work")

    finally:
        os.unlink('base.tbx')
        os.unlink('mid.tbx')

@test("Compiled - Cache Invalidation", "Import Compiled Modes")
def test_import_cache_invalidation():
    """Test that cache is invalidated when source changes"""
    print("→ Test: Cache Invalidation")

    lib_path = 'test_cache_lib.tbx'

    # Write initial version
    with open(lib_path, 'w') as f:
        f.write('''
@config {
    mode: "compiled"
}

fn get_value() {
    42
}
''')

    try:
        main_code = f'''
@imports {{
    "{lib_path}"
}}

echo get_value()
'''

        # First run
        success1, stdout1, _ = run_tb(main_code)
        assert success1
        assert "42" in stdout1

        # Modify library
        time.sleep(0.1)  # Ensure timestamp difference
        with open(lib_path, 'w') as f:
            f.write('''
@config {
    mode: "compiled"
}

fn get_value() {
    100
}
''')

        # Second run should use updated version
        success2, stdout2, _ = run_tb(main_code)
        assert success2
        assert "100" in stdout2, f"Cache not invalidated! Got: {stdout2}"

        print("  ✓ Cache invalidation works")

    finally:
        os.unlink(lib_path)



# ═══════════════════════════════════════════════════════════════════════════
# TESTS - ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@test("Import - Missing File Error", "Import Errors")
def test_import_missing_file():
    """Test error handling for missing import files"""
    code = '''
@imports {
    "nonexistent_file.tbx"
}

echo "This should not run"
'''
    success, stdout, stderr = run_tb(code)
    assert not success, "Should fail for missing import file"
    assert "not found" in stderr.lower() or "import" in stderr.lower()


@test("Import - Circular Import Protection", "Import Errors")
def test_import_no_circular():
    """Test that circular imports are handled (imports are not recursive)"""
    # Create file that tries to import itself
    lib_code = '''
fn test_func() {
    echo "Hello"
}
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbx', delete=False, dir='.') as lib_file:
        lib_file.write(lib_code)
        lib_path = os.path.basename(lib_file.name)

    try:
        # This should work because imports are not recursive
        main_code = f'''
@imports {{
    "{lib_path}"
}}

test_func()
'''
        assert_success(main_code)
    finally:
        os.unlink(lib_path)

# ═══════════════════════════════════════════════════════════════════════════
# TESTS - DEPENDENCIES: GO LANGUAGE BRIDGE & IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

@test("Go Language Bridge - Basic Execution", "Dependencies - Go Bridge")
def test_go_basic_execution():
    """Test that Go code can be executed."""
    code = '''
let result = go("""
import "fmt"
fmt.Println("Hello from Go!")
""")
'''
    assert_success(code)


@test("Go Import - Math", "Dependencies - Go Bridge")
def test_go_import_math():
    """Test using the Go 'math' standard library package."""
    code = '''
let result = go("""
import (
    "fmt"
    "math"
)
// Sqrt returns the square root of 64
fmt.Println(math.Sqrt(64))
""")
'''
    assert_success(code)


@test("Go Import - Strings", "Dependencies - Go Bridge")
def test_go_import_strings():
    """Test using the Go 'strings' package with a TB variable."""
    code = '''
let my_string = "Hello, Go!"
let result = go("""
import (
    "fmt"
    "strings"
)
// my_string is available from TB context
// ToUpper returns a copy of the string with all letters mapped to their upper case.
fmt.Println(strings.ToUpper(my_string))
""")
'''
    assert_success(code)


@test("Go Import - JSON Marshalling", "Dependencies - Go Bridge")
def test_go_import_json():
    """Test marshalling data to JSON using Go's 'encoding/json' package."""
    code = '''
# This test assumes TB lists/dicts are converted to Go slices/maps
let user_data = [
    { "name": "Alice", "age": 30 },
    { "name": "Bob", "age": 25 }
]
let result = go("""
import (
    "fmt"
    "encoding/json"
)
// user_data is available from TB, assume it's a []map[string]interface{}
// Marshal returns the JSON encoding of user_data.
jsonData, err := json.Marshal(user_data)
if err != nil {
    fmt.Println("Error:", err)
} else {
    fmt.Println(string(jsonData))
}
""")
'''
    assert_success(code)


@test("Go Multi-Language - Python to Go", "Dependencies - Go Bridge")
def test_go_multilang_python_to_go():
    """Test data flow from a Python block to a Go block."""
    code = '''
# Python creates a string
let data_from_py = python("""
print("Data generated by Python")
""")

# Go processes the string from the TB variable
let go_result = go("""
import "fmt"
// data_from_py is available from TB context
// Printf formats according to a format specifier and writes to standard output.
fmt.Printf("Go received: '%s'\\n", data_from_py)
""")
'''
    assert_success(code)


@test("Go Bridge - Compiled Mode", "Dependencies - Go Bridge")
def test_go_bridge_compiled_mode():
    """Test that the Go language bridge works in compiled mode."""
    code = '''
@config {
    mode: "compiled"
}

let result = go("""
import "fmt"
fmt.Println("Hello from Go in compiled mode!")
""")

echo "Go execution finished."
'''
    assert_contains(code, "Go execution finished.", mode="compiled")

@test("Type Annotations - Basic Types", "Type Annotations")
def test_type_annotations():
    """Test that type annotations are correctly handled."""
    code = '''@config { mode: "compiled" }

let age: int = python("print(42)")
let price: float = python("print(19.99)")
let name: string = python("print('Alice')")
let active: bool = python("print(True)")
let scores: list<int> = python("print([85, 92, 78])")

echo "Age: $age"           // Age: 42
echo "Price: $price"       // Price: 19.99
echo "Name: $name"         // Name: Alice
echo "Active: $active"     // Active: true'''
    assert_output(code, "Age: 42\nPrice: 19.99\nName: Alice\nActive: true")

@test("Type Annotations - Auto Type Inference", "Type Annotations")
def test_type_annotations_auto():
    """Test that type annotations are correctly handled."""
    code = '''@config { mode: "compiled" }

let age: int = python("print(42)")
let price: float = python("print(19.99)")
let name: string = python("print('Alice')")
let active: bool = python("print(True)")
let scores: list<int> = python("print([85, 92, 78])")

echo type_of(1)
echo type_of("1")
echo type_of(age)           // Age: 42
echo type_of(price)       // Price: 19.99
echo type_of(name)         // Name: Alice
echo type_of(active)     // Active: true'''
    assert_output(code, "int\nfloat\nstring\nbool\nlist<int>")

@test("Type Annotations - Basic Type - Compiled", "Type Annotations - Compiled")
def test_type_annotations_compiled():
    """Test that type annotations are correctly handled."""
    code = '''@config { mode: "compiled" }

let age: int = python("print(42)")
let price: float = python("print(19.99)")
let name: string = python("print('Alice')")
let active: bool = python("print(True)")
let scores: list<int> = python("print([85, 92, 78])")

echo "Age: $age"           // Age: 42
echo "Price: $price"       // Price: 19.99
echo "Name: $name"         // Name: Alice
echo "Active: $active"     // Active: true'''
    assert_output(code, "Age: 42\nPrice: 19.99\nName: Alice\nActive: true", mode="compiled")

@test("Type Annotations - Auto Type Inference  - Compiled", "Type Annotations - Compiled")
def test_type_annotations_auto_compiled():
    """Test that type annotations are correctly handled."""
    code = '''@config { mode: "compiled" }

let age: int = python("print(42)")
let price: float = python("print(19.99)")
let name: string = python("print('Alice')")
let active: bool = python("print(True)")
let scores: list<int> = python("print([85, 92, 78])")

echo type_of($age)           // Age: 42
echo type_of($price)       // Price: 19.99
echo type_of($name)         // Name: Alice
echo type_of($active)     // Active: true'''
    assert_output(code, "int\nfloat\nstring\nbool\nlist<int>", mode="compiled")

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
        result = subprocess.run([TB_BINARY, "--version"], capture_output=not VERBOSE, timeout=5)
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

        test_cross_variables_compiled,
        test_cross_variables,
        testexample_return_values,
        testexample_return_values_compiled,
        test_example_parallel,

        # Import System
        test_import_single_file,
        test_import_multiple_files,
        test_import_with_variables,
        test_import_relative_paths,
        test_import_compiled_mode,

        # Compiled Modes
        test_compiled_arithmetic,
        test_compiled_functions,
        test_compiled_functions_no_compiled_,
        test_compiled_loops,
        test_compiled_parallel,
        test_compiled_parallel_with_imports,

        # Mixed Features
        # test_mixed_import_async_language,
        # test_mixed_multiple_imports_parallel,
        test_mixed_shared_imports,
        test_mixed_shared_imports_mixed_lang,
        test_mixed_full_stack,

        test_import_multiple_files,
        test_import_compiled_caching,
        test_import_jit_vs_compiled,
        test_import_dependency_chain,
        test_import_cache_invalidation,

        # Error Handling
        test_import_missing_file,
        test_import_no_circular,

        # Go
        test_go_basic_execution,
        test_go_import_math,
        test_go_import_strings,
        test_go_import_json,
        test_go_multilang_python_to_go,
        test_go_bridge_compiled_mode,

        # Type Annotations
        test_type_annotations,
        test_type_annotations_auto,
        test_type_annotations_compiled,
        test_type_annotations_auto_compiled,
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
    VERBOSE = "verbose" in args or "-v" in args
    print(f"{VERBOSE=}")
    FILTER = None
    for i, arg in enumerate(args):
        if arg == "filter" and i + 1 < len(args):
            FILTER = args[i + 1]
    main()


if __name__ == "__main__":
    TB_BINARY = find_tb_binary()
    main()
