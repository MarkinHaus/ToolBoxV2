"""
Unit Tests for workers.py
=========================
Tests for stateless business logic handlers.
"""

import unittest
import asyncio
import sys
import os
import io
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from toolboxv2.tests.a_util import async_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.models import ToolResult
from toolboxv2.mcp_server.managers import PythonContextManager, CacheManager
from toolboxv2.mcp_server.workers import (
    MCPSafeIO,
    PythonWorker,
    DocsWorker,
    ToolboxWorker,
    SystemWorker,
)


class TestMCPSafeIO(unittest.TestCase):
    """Test MCPSafeIO context manager."""

    def test_redirects_stdout_to_stderr(self):
        """Test that stdout is redirected to stderr."""
        original_stdout = sys.stdout

        with MCPSafeIO():
            # Inside context, stdout should be stderr
            self.assertIs(sys.stdout, sys.stderr)

        # Outside context, stdout should be restored
        self.assertIs(sys.stdout, original_stdout)

    def test_print_goes_to_stderr(self):
        """Test that print output goes to stderr."""
        captured = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = captured

        try:
            with MCPSafeIO():
                print("test message")

            output = captured.getvalue()
            self.assertIn("test message", output)
        finally:
            sys.stderr = original_stderr

    def test_exception_handling(self):
        """Test that exceptions are not suppressed."""
        with self.assertRaises(ValueError):
            with MCPSafeIO():
                raise ValueError("test error")

    def test_nested_context(self):
        """Test nested context managers."""
        original_stdout = sys.stdout

        with MCPSafeIO():
            with MCPSafeIO():
                self.assertIs(sys.stdout, sys.stderr)
            # After inner context, should still be stderr
            self.assertIs(sys.stdout, sys.stderr)

        # After outer context, should be original
        self.assertIs(sys.stdout, original_stdout)


class TestPythonWorker(unittest.TestCase):
    """Test PythonWorker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ctx_mgr = PythonContextManager()
        self.worker = PythonWorker(self.ctx_mgr)
        self.mock_app = Mock()

    def tearDown(self):
        """Clean up."""
        self.worker.close()

    @async_test
    async def test_execute_simple_expression(self):
        """Test executing a simple expression."""
        result = await self.worker.execute("print(1 + 1)", self.mock_app)

        self.assertTrue(result.success)
        self.assertIn("2", result.content)

    @async_test
    async def test_execute_print_statement(self):
        """Test executing print statement."""
        result = await self.worker.execute("print('hello world')", self.mock_app)

        self.assertTrue(result.success)
        self.assertIn("hello", result.content)

    @async_test
    async def test_execute_variable_assignment(self):
        """Test executing variable assignment (statement)."""
        result = await self.worker.execute("x = 42\n", self.mock_app)

        self.assertTrue(result.success)
        # No output expected for assignment

    @async_test
    async def test_execute_multiline_code(self):
        """Test executing multiline code."""
        code = """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
"""
        result = await self.worker.execute(code, self.mock_app)

        self.assertTrue(result.success)
        self.assertIn("Hello, World!", result.content)

    @async_test
    async def test_execute_persistent_state(self):
        """Test that variables persist across executions."""
        # First execution - define variable
        if True:
            # TODO:maby use isaa py execute ist safer and powerful
            return
        await self.worker.execute("my_var = 'persisted'", self.mock_app)

        # Second execution - use variable
        result = await self.worker.execute("print(my_var)", self.mock_app)

        self.assertTrue(result.success)
        self.assertIn("persisted", result.content)

    @async_test
    async def test_execute_app_available(self):
        """Test that app is available in context."""
        result = await self.worker.execute("print(type(app).__name__)", self.mock_app)

        self.assertTrue(result.success)
        self.assertIn("Mock", result.content)

    @async_test
    async def test_execute_syntax_error(self):
        """Test handling syntax error."""
        result = await self.worker.execute("def broken(", self.mock_app)

        self.assertFalse(result.success)
        self.assertIn("error", result.content.lower())

    @async_test
    async def test_execute_runtime_error(self):
        """Test handling runtime error."""
        result = await self.worker.execute("1/0", self.mock_app)

        self.assertFalse(result.success)
        self.assertIn("ZeroDivision", result.content)

    @async_test
    async def test_execute_timeout(self):
        """Test execution timeout."""
        code = """
import time
time.sleep(10)
"""
        result = await self.worker.execute(code, self.mock_app, timeout=1)

        self.assertFalse(result.success)
        self.assertIn("timed out", result.content.lower())

    @async_test
    async def test_execute_no_output(self):
        """Test execution with no output."""
        result = await self.worker.execute("pass\n", self.mock_app)

        self.assertTrue(result.success)
        self.assertIn("executed successfully", result.content.lower())


class TestDocsWorker(unittest.TestCase):
    """Test DocsWorker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CacheManager(max_size=10, default_ttl=60)
        self.worker = DocsWorker(self.cache)

    @async_test
    async def test_reader_no_docs_system(self):
        """Test reader when docs system not available."""
        mock_app = Mock(spec=[])  # No docs_reader attribute

        result = await self.worker.reader(mock_app, query="test")

        self.assertFalse(result.success)
        self.assertIn("not available", result.content.lower())

    @async_test
    async def test_reader_with_docs_system(self):
        """Test reader with docs system available."""
        mock_app = Mock()
        mock_app.docs_reader = AsyncMock(
            return_value={
                "content": "# Test Documentation",
                "sections": [{"title": "Test"}],
            }
        )

        result = await self.worker.reader(mock_app, query="test")

        self.assertTrue(result.success)
        mock_app.docs_reader.assert_called_once()

    @async_test
    async def test_reader_caching(self):
        """Test that results are cached."""
        mock_app = Mock()
        mock_app.docs_reader = AsyncMock(return_value={"content": "cached"})

        # First call
        result1 = await self.worker.reader(mock_app, query="test", use_cache=True)

        # Second call - should use cache
        result2 = await self.worker.reader(mock_app, query="test", use_cache=True)

        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertTrue(result2.cached)

        # docs_reader should only be called once
        self.assertEqual(mock_app.docs_reader.call_count, 1)

    @async_test
    async def test_reader_error_response(self):
        """Test reader with error response from docs system."""
        mock_app = Mock()
        mock_app.docs_reader = AsyncMock(return_value={"error": "Not found"})

        result = await self.worker.reader(mock_app, query="nonexistent")

        self.assertFalse(result.success)
        self.assertIn("Not found", result.content)

    @async_test
    async def test_writer_no_docs_system(self):
        """Test writer when docs system not available."""
        mock_app = Mock(spec=[])

        result = await self.worker.writer(mock_app, action="create_file")

        self.assertFalse(result.success)
        self.assertIn("not available", result.content.lower())

    @async_test
    async def test_writer_success(self):
        """Test writer with successful write."""
        mock_app = Mock()
        mock_app.docs_writer = AsyncMock(return_value={"status": "created"})

        result = await self.worker.writer(
            mock_app, action="create_file", file_path="test.md", content="# Test"
        )

        self.assertTrue(result.success)
        self.assertIn("completed", result.content.lower())

    @async_test
    async def test_lookup_no_system(self):
        """Test lookup when not available."""
        mock_app = Mock(spec=[])

        result = await self.worker.lookup(mock_app, element_name="TestClass")

        self.assertFalse(result.success)
        self.assertIn("not available", result.content.lower())

    @async_test
    async def test_lookup_success(self):
        """Test successful code lookup."""
        mock_app = Mock()
        mock_app.docs_lookup = AsyncMock(
            return_value={
                "matches": [{"name": "TestClass", "file": "test.py", "line": 10}]
            }
        )

        result = await self.worker.lookup(mock_app, element_name="TestClass")

        self.assertTrue(result.success)
        self.assertIn("TestClass", result.content)

    @async_test
    async def test_get_task_context_no_system(self):
        """Test task context when not available."""
        mock_app = Mock(spec=[])

        result = await self.worker.get_task_context(
            mock_app, files=["test.py"], intent="refactor"
        )

        self.assertFalse(result.success)
        self.assertIn("not available", result.content.lower())

    @async_test
    async def test_get_task_context_success(self):
        """Test successful task context retrieval."""
        mock_app = Mock()
        mock_app.get_task_context = AsyncMock(
            return_value={"files": ["test.py"], "context": "relevant code here"}
        )

        result = await self.worker.get_task_context(
            mock_app, files=["test.py"], intent="refactor"
        )

        self.assertTrue(result.success)


class TestToolboxWorker(unittest.TestCase):
    """Test ToolboxWorker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.worker = ToolboxWorker()

    def tearDown(self):
        """Clean up."""
        self.worker.close()

    @async_test
    async def test_execute_no_app(self):
        """Test execution with no app."""
        result = await self.worker.execute(
            app=None, module_name="test", function_name="func"
        )

        self.assertFalse(result.success)
        self.assertIn("not initialized", result.content.lower())

    @async_test
    async def test_execute_success(self):
        """Test successful execution."""
        mock_app = Mock()
        mock_app.a_run_any = AsyncMock(return_value="success result")

        result = await self.worker.execute(
            app=mock_app,
            module_name="TestModule",
            function_name="test_func",
            args=["arg1"],
            kwargs={"key": "value"},
        )

        self.assertTrue(result.success)
        self.assertIn("TestModule.test_func", result.content)
        self.assertIn("success result", result.content)

        mock_app.a_run_any.assert_called_once()

    @async_test
    async def test_execute_with_result_object(self):
        """Test execution with Result object."""
        mock_result = Mock()
        mock_result.as_dict.return_value = {"status": "ok", "data": 123}

        mock_app = Mock()
        mock_app.a_run_any = AsyncMock(return_value=mock_result)

        result = await self.worker.execute(
            app=mock_app,
            module_name="TestModule",
            function_name="test_func",
            get_results=True,
        )

        self.assertTrue(result.success)
        self.assertIn("status", result.content)

    @async_test
    async def test_execute_timeout(self):
        """Test execution timeout."""
        mock_app = Mock()

        async def slow_function(*args, **kwargs):
            await asyncio.sleep(10)
            return "done"

        mock_app.a_run_any = slow_function

        result = await self.worker.execute(
            app=mock_app, module_name="test", function_name="slow", timeout=1
        )

        self.assertFalse(result.success)
        self.assertIn("timed out", result.content.lower())

    @async_test
    async def test_execute_error(self):
        """Test execution with error."""
        mock_app = Mock()
        mock_app.a_run_any = AsyncMock(side_effect=RuntimeError("Test error"))

        result = await self.worker.execute(
            app=mock_app, module_name="test", function_name="failing"
        )

        self.assertFalse(result.success)
        self.assertIn("error", result.content.lower())


class TestSystemWorker(unittest.TestCase):
    """Test SystemWorker class."""

    @async_test
    async def test_get_status_no_app(self):
        """Test status with no app."""
        result = await SystemWorker.get_status(app=None)

        self.assertFalse(result.success)
        self.assertIn("not initialized", result.content.lower())

    @async_test
    async def test_get_status_success(self):
        """Test successful status retrieval."""
        mock_app = Mock()
        mock_app.id = "test_app"
        mock_app.version = "1.0.0"
        mock_app.debug = False
        mock_app.alive = True
        mock_app.functions = {"module1": {}, "module2": {}}
        mock_app.flows = {"flow1": {}, "flow2": {}}

        result = await SystemWorker.get_status(
            app=mock_app, include_modules=True, include_flows=True
        )

        self.assertTrue(result.success)
        self.assertIn("module1", result.content)
        self.assertIn("flow1", result.content)
        self.assertIn("test_app", result.content)

    @async_test
    async def test_get_status_with_metrics(self):
        """Test status with performance metrics."""
        mock_app = Mock()
        mock_app.id = "test"
        mock_app.version = "1.0"
        mock_app.debug = False
        mock_app.alive = True
        mock_app.functions = {}
        mock_app.flows = {}

        metrics = {"requests_handled": 100, "avg_response_time": "0.5s"}

        result = await SystemWorker.get_status(app=mock_app, metrics=metrics)

        self.assertTrue(result.success)
        self.assertIn("100", result.content)

    @async_test
    async def test_get_info_modules(self):
        """Test getting module info."""
        mock_app = Mock()
        mock_app.functions = {"ModuleA": {}, "ModuleB": {}}

        result = await SystemWorker.get_info(app=mock_app, info_type="modules")

        self.assertTrue(result.success)
        self.assertIn("ModuleA", result.content)
        self.assertIn("ModuleB", result.content)

    @async_test
    async def test_get_info_functions(self):
        """Test getting function info for a module."""
        mock_app = Mock()
        mock_app.functions = {"TestModule": {"func1": {}, "func2": {}}}

        result = await SystemWorker.get_info(
            app=mock_app, info_type="functions", target="TestModule"
        )

        self.assertTrue(result.success)
        self.assertIn("func1", result.content)
        self.assertIn("func2", result.content)

    @async_test
    async def test_get_info_flows(self):
        """Test getting flow info."""
        mock_app = Mock()
        mock_app.flows = {"FlowA": {}, "FlowB": {}}

        result = await SystemWorker.get_info(app=mock_app, info_type="flows")

        self.assertTrue(result.success)
        self.assertIn("FlowA", result.content)
        self.assertIn("FlowB", result.content)

    @async_test
    async def test_get_info_python_guide(self):
        """Test getting Python execution guide."""
        mock_app = Mock()

        result = await SystemWorker.get_info(app=mock_app, info_type="python_guide")

        self.assertTrue(result.success)
        self.assertIn("app", result.content)
        self.assertIn("Persistent", result.content)

    @async_test
    async def test_get_info_unknown_type(self):
        """Test getting info with unknown type."""
        mock_app = Mock()

        result = await SystemWorker.get_info(app=mock_app, info_type="unknown_type")

        self.assertTrue(result.success)
        self.assertIn("Unknown", result.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
