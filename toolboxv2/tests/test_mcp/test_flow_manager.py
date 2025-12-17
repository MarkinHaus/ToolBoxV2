"""
Unit Tests for FlowSessionManager
=================================

Tests the flow execution with I/O capture.
"""

import unittest
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.flow_manager import (
    FlowSessionManager,
    FlowSession,
    FlowState,
    FlowExecutor,
    IOType,
    IOEntry,
)


def async_test(coro):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


class TestFlowSession(unittest.TestCase):
    """Test FlowSession dataclass."""

    def test_creation(self):
        """Test basic session creation."""
        session = FlowSession(
            session_id="test_123",
            flow_name="test_flow"
        )

        self.assertEqual(session.session_id, "test_123")
        self.assertEqual(session.flow_name, "test_flow")
        self.assertEqual(session.state, FlowState.CREATED)
        self.assertEqual(session.io_buffer, [])
        self.assertEqual(session.history, [])

    def test_update_activity(self):
        """Test activity update."""
        session = FlowSession("id", "flow")
        old_time = session.last_activity

        time.sleep(0.01)
        session.update_activity()

        self.assertGreater(session.last_activity, old_time)

    def test_is_expired(self):
        """Test expiration check."""
        session = FlowSession("id", "flow")

        # Not expired
        self.assertFalse(session.is_expired(60))

        # Manually expire
        session.last_activity = time.time() - 120
        self.assertTrue(session.is_expired(60))

    def test_add_output(self):
        """Test output buffering."""
        session = FlowSession("id", "flow")

        session.add_output("Hello", IOType.STDOUT)
        session.add_output("World", IOType.STDOUT)

        self.assertEqual(len(session.io_buffer), 2)
        self.assertEqual(session.io_buffer[0].content, "Hello")
        self.assertEqual(session.io_buffer[0].type, IOType.STDOUT)

    def test_get_pending_output(self):
        """Test getting and clearing output."""
        session = FlowSession("id", "flow")
        session.add_output("Test", IOType.STDOUT)

        output = session.get_pending_output()

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["content"], "Test")

        # Buffer should be cleared
        self.assertEqual(len(session.io_buffer), 0)

    def test_to_dict(self):
        """Test serialization."""
        session = FlowSession("test_id", "test_flow")
        session.state = FlowState.RUNNING

        d = session.to_dict()

        self.assertEqual(d["session_id"], "test_id")
        self.assertEqual(d["flow_name"], "test_flow")
        self.assertEqual(d["state"], "running")


class TestIOEntry(unittest.TestCase):
    """Test IOEntry dataclass."""

    def test_creation(self):
        """Test entry creation."""
        entry = IOEntry(
            type=IOType.STDOUT,
            content="Test output",
            timestamp=time.time()
        )

        self.assertEqual(entry.type, IOType.STDOUT)
        self.assertEqual(entry.content, "Test output")

    def test_to_dict(self):
        """Test serialization."""
        entry = IOEntry(
            type=IOType.PROMPT,
            content="Enter name:",
            timestamp=123456.789
        )

        d = entry.to_dict()

        self.assertEqual(d["type"], "prompt")
        self.assertEqual(d["content"], "Enter name:")
        self.assertEqual(d["timestamp"], 123456.789)


class TestFlowSessionManager(unittest.TestCase):
    """Test FlowSessionManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = FlowSessionManager(max_sessions=5, timeout=60)

    def tearDown(self):
        """Clean up."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.manager.stop_cleanup())
        finally:
            loop.close()
        self.manager.close()

    @async_test
    async def test_create_session_without_app(self):
        """Test that creation fails without app."""
        with self.assertRaises(RuntimeError):
            await self.manager.create("test_flow")

    @async_test
    async def test_create_session_with_mock_app(self):
        """Test session creation with mock app."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"test_flow": lambda app, args: None}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        session = await self.manager.create("test_flow", auto_start=False)

        self.assertIsNotNone(session)
        self.assertTrue(session.session_id.startswith("flow_"))
        self.assertEqual(session.flow_name, "test_flow")
        self.assertEqual(session.state, FlowState.CREATED)

    @async_test
    async def test_create_session_custom_id(self):
        """Test session creation with custom ID."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"test_flow": lambda app, args: None}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        session = await self.manager.create(
            "test_flow",
            session_id="custom_123",
            auto_start=False
        )

        self.assertEqual(session.session_id, "custom_123")

    @async_test
    async def test_get_session(self):
        """Test getting a session."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"test_flow": lambda app, args: None}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        created = await self.manager.create("test_flow", auto_start=False)
        retrieved = await self.manager.get(created.session_id)

        self.assertEqual(retrieved.session_id, created.session_id)

    @async_test
    async def test_get_nonexistent_session(self):
        """Test getting nonexistent session."""
        session = await self.manager.get("nonexistent")
        self.assertIsNone(session)

    @async_test
    async def test_delete_session(self):
        """Test deleting a session."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"test_flow": lambda app, args: None}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        session = await self.manager.create("test_flow", auto_start=False)

        success = await self.manager.delete(session.session_id)
        self.assertTrue(success)

        # Should not exist anymore
        retrieved = await self.manager.get(session.session_id)
        self.assertIsNone(retrieved)

    @async_test
    async def test_list_sessions(self):
        """Test listing sessions."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"flow_a": lambda: None, "flow_b": lambda: None}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        await self.manager.create("flow_a", auto_start=False)
        await self.manager.create("flow_b", auto_start=False)

        sessions = await self.manager.list_sessions()

        self.assertEqual(len(sessions), 2)
        flow_names = [s["flow_name"] for s in sessions]
        self.assertIn("flow_a", flow_names)
        self.assertIn("flow_b", flow_names)

    @async_test
    async def test_max_sessions_enforcement(self):
        """Test that max sessions is enforced."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {f"flow_{i}": lambda: None for i in range(10)}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        # Create max sessions
        for i in range(5):
            await self.manager.create(f"flow_{i}", auto_start=False)

        self.assertEqual(self.manager.count, 5)

        # Create one more
        await self.manager.create("flow_5", auto_start=False)

        # Should still be max
        self.assertEqual(self.manager.count, 5)

    @async_test
    async def test_cleanup_expired(self):
        """Test expired session cleanup."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"test_flow": lambda: None}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        session = await self.manager.create("test_flow", auto_start=False)

        # Manually expire
        session.last_activity = time.time() - 120

        count = await self.manager.cleanup_expired()

        self.assertEqual(count, 1)
        self.assertEqual(self.manager.count, 0)


class TestFlowExecution(unittest.TestCase):
    """Test actual flow execution with I/O capture."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = FlowSessionManager(max_sessions=5, timeout=60)

    def tearDown(self):
        """Clean up."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.manager.stop_cleanup())
        finally:
            loop.close()
        self.manager.close()

    @async_test
    async def test_simple_flow_execution(self):
        """Test executing a simple flow without input."""
        from unittest.mock import Mock

        # Simple flow that just prints
        async def simple_flow(app, args_sto, **kwargs):
            print("Hello from flow!")
            print("Processing...")
            return {"status": "completed"}

        mock_app = Mock()
        mock_app.flows = {"simple_flow": simple_flow}
        mock_app.args_sto = {}

        self.manager.set_app(mock_app)

        # Create and start
        session = await self.manager.create("simple_flow", auto_start=False)
        result = await self.manager.start_execution(session.session_id)

        # Wait for completion
        await asyncio.sleep(0.2)

        status = await self.manager.get_status(session.session_id)

        self.assertEqual(status["state"], "completed")
        self.assertIsNotNone(status["result"])
        self.assertEqual(status["result"]["status"], "completed")


class TestFlowHandlers(unittest.TestCase):
    """Test FlowHandlers integration."""

    def setUp(self):
        """Set up test fixtures."""
        from toolboxv2.mcp_server.flow_handlers import FlowHandlers

        self.manager = FlowSessionManager(max_sessions=5, timeout=60)
        self.handlers = FlowHandlers(self.manager)

    def tearDown(self):
        """Clean up."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.manager.stop_cleanup())
        finally:
            loop.close()
        self.manager.close()

    def test_tool_definitions(self):
        """Test that tool definitions are valid."""
        from toolboxv2.mcp_server.flow_handlers import FlowHandlers

        tools = FlowHandlers.get_tool_definitions()

        self.assertGreater(len(tools), 0)

        tool_names = [t["name"] for t in tools]
        self.assertIn("flow_start", tool_names)
        self.assertIn("flow_input", tool_names)
        self.assertIn("flow_status", tool_names)
        self.assertIn("flow_list_available", tool_names)

    @async_test
    async def test_list_available_flows(self):
        """Test listing available flows."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"flow_a": None, "flow_b": None, "flow_c": None}

        result = await self.handlers.handle_list_available(mock_app)

        self.assertTrue(result.success)
        self.assertIn("flow_a", result.content)
        self.assertEqual(len(result.data["flows"]), 3)

    @async_test
    async def test_list_available_with_filter(self):
        """Test listing flows with filter."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"prompt_gen": None, "data_proc": None, "prompt_opt": None}

        result = await self.handlers.handle_list_available(mock_app, filter_str="prompt")

        self.assertTrue(result.success)
        self.assertEqual(len(result.data["flows"]), 2)

    @async_test
    async def test_start_nonexistent_flow(self):
        """Test starting nonexistent flow."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.flows = {"existing_flow": None}
        mock_app.args_sto = {}

        # Mock the flows_dict import
        import sys
        mock_flows_module = Mock()
        mock_flows_module.flows_dict = Mock(return_value={})
        sys.modules['toolboxv2.flows'] = mock_flows_module

        self.manager.set_app(mock_app)

        result = await self.handlers.handle_start(
            mock_app,
            flow_name="nonexistent"
        )

        self.assertFalse(result.success)
        self.assertIn("not found", result.content.lower())

    @async_test
    async def test_list_empty_sessions(self):
        """Test listing when no sessions."""
        result = await self.handlers.handle_list_sessions()

        self.assertTrue(result.success)
        self.assertIn("No active", result.content)


if __name__ == '__main__':
    unittest.main(verbosity=2)
