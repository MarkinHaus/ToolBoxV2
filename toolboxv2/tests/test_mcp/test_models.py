"""
Unit Tests for models.py
========================
Tests for data types, enums, and configuration classes.
"""

import unittest
import time
from dataclasses import FrozenInstanceError

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.models import (
    ResponseFormat,
    ServerMode,
    PermissionLevel,
    FlowState,
    APIKeyInfo,
    FlowSession,
    ToolResult,
    CacheEntry,
    PerformanceMetrics,
    MCPConfig,
    FLOWAGENTS_DISCOVERY_TEMPLATE,
    PYTHON_EXECUTION_TEMPLATE,
    PERFORMANCE_GUIDE_TEMPLATE,
)


class TestEnums(unittest.TestCase):
    """Test enum definitions."""

    def test_response_format_values(self):
        """Test ResponseFormat enum values."""
        self.assertEqual(ResponseFormat.MARKDOWN.value, "markdown")
        self.assertEqual(ResponseFormat.JSON.value, "json")
        self.assertEqual(ResponseFormat.STRUCTURED.value, "structured")

    def test_server_mode_values(self):
        """Test ServerMode enum values."""
        self.assertEqual(ServerMode.STDIO.value, "stdio")
        self.assertEqual(ServerMode.HTTP.value, "http")

    def test_permission_level_values(self):
        """Test PermissionLevel enum values."""
        self.assertEqual(PermissionLevel.READ.value, "read")
        self.assertEqual(PermissionLevel.WRITE.value, "write")
        self.assertEqual(PermissionLevel.EXECUTE.value, "execute")
        self.assertEqual(PermissionLevel.ADMIN.value, "admin")

    def test_flow_state_values(self):
        """Test FlowState enum values."""
        self.assertEqual(FlowState.CREATED.value, "created")
        self.assertEqual(FlowState.RUNNING.value, "running")
        self.assertEqual(FlowState.WAITING.value, "waiting")
        self.assertEqual(FlowState.COMPLETED.value, "completed")
        self.assertEqual(FlowState.ERROR.value, "error")

    def test_enum_string_comparison(self):
        """Test that enums can be compared to strings."""
        self.assertEqual(ResponseFormat.MARKDOWN, "markdown")
        self.assertEqual(ServerMode.STDIO, "stdio")


class TestAPIKeyInfo(unittest.TestCase):
    """Test APIKeyInfo dataclass."""

    def setUp(self):
        self.key_info = APIKeyInfo(
            name="test_key", permissions=["read", "write"], created=time.time()
        )

    def test_creation(self):
        """Test basic creation."""
        self.assertEqual(self.key_info.name, "test_key")
        self.assertEqual(self.key_info.permissions, ["read", "write"])
        self.assertIsNone(self.key_info.last_used)
        self.assertEqual(self.key_info.usage_count, 0)

    def test_has_permission(self):
        """Test permission checking."""
        self.assertTrue(self.key_info.has_permission("read"))
        self.assertTrue(self.key_info.has_permission("write"))
        self.assertFalse(self.key_info.has_permission("admin"))
        self.assertFalse(self.key_info.has_permission("execute"))

    def test_to_dict(self):
        """Test dictionary conversion."""
        d = self.key_info.to_dict()
        self.assertEqual(d["name"], "test_key")
        self.assertEqual(d["permissions"], ["read", "write"])
        self.assertIn("created", d)
        self.assertIsNone(d["last_used"])
        self.assertEqual(d["usage_count"], 0)

    def test_slots_memory_efficiency(self):
        """Test that __slots__ is used (no __dict__)."""
        self.assertFalse(hasattr(self.key_info, "__dict__"))


class TestFlowSession(unittest.TestCase):
    """Test FlowSession dataclass."""

    def setUp(self):
        self.session = FlowSession(
            session_id="test_session_123",
            flow_name="test_flow",
            created=time.time(),
            last_activity=time.time(),
            state=FlowState.CREATED,
            context={"key": "value"},
            history=["step1"],
        )

    def test_creation(self):
        """Test basic creation."""
        self.assertEqual(self.session.session_id, "test_session_123")
        self.assertEqual(self.session.flow_name, "test_flow")
        self.assertEqual(self.session.state, FlowState.CREATED)

    def test_update_activity(self):
        """Test activity timestamp update."""
        old_activity = self.session.last_activity
        time.sleep(0.01)  # Small delay
        self.session.update_activity()
        self.assertGreater(self.session.last_activity, old_activity)

    def test_is_expired_not_expired(self):
        """Test expiration check when not expired."""
        self.assertFalse(self.session.is_expired(timeout=3600))

    def test_is_expired_expired(self):
        """Test expiration check when expired."""
        self.session.last_activity = time.time() - 7200  # 2 hours ago
        self.assertTrue(self.session.is_expired(timeout=3600))

    def test_slots_memory_efficiency(self):
        """Test that __slots__ is used."""
        self.assertFalse(hasattr(self.session, "__dict__"))


class TestToolResult(unittest.TestCase):
    """Test ToolResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(success=True, content="Test output", execution_time=0.5)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Test output")
        self.assertEqual(result.execution_time, 0.5)
        self.assertFalse(result.cached)
        self.assertIsNone(result.error)

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            success=False, content="Error occurred", execution_time=0.1, error="TestError"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "TestError")

    def test_cached_result(self):
        """Test cached result."""
        result = ToolResult(
            success=True, content="Cached", execution_time=0.001, cached=True
        )
        self.assertTrue(result.cached)

    def test_slots_memory_efficiency(self):
        """Test that __slots__ is used."""
        result = ToolResult(success=True, content="", execution_time=0)
        self.assertFalse(hasattr(result, "__dict__"))


class TestCacheEntry(unittest.TestCase):
    """Test CacheEntry dataclass."""

    def test_creation(self):
        """Test basic creation."""
        entry = CacheEntry(key="test_key", value={"data": "test"}, timestamp=time.time())
        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.value, {"data": "test"})
        self.assertEqual(entry.ttl, 300)  # Default

    def test_is_expired_not_expired(self):
        """Test expiration check when not expired."""
        entry = CacheEntry(key="test", value="data", timestamp=time.time(), ttl=300)
        self.assertFalse(entry.is_expired())

    def test_is_expired_expired(self):
        """Test expiration check when expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            timestamp=time.time() - 600,  # 10 minutes ago
            ttl=300,  # 5 minutes TTL
        )
        self.assertTrue(entry.is_expired())

    def test_custom_ttl(self):
        """Test custom TTL."""
        entry = CacheEntry(key="test", value="data", timestamp=time.time(), ttl=60)
        self.assertEqual(entry.ttl, 60)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass."""

    def setUp(self):
        self.metrics = PerformanceMetrics()

    def test_initial_values(self):
        """Test initial values."""
        self.assertEqual(self.metrics.requests_handled, 0)
        self.assertEqual(self.metrics.total_response_time, 0.0)
        self.assertEqual(self.metrics.cache_hits, 0)
        self.assertEqual(self.metrics.cache_misses, 0)
        self.assertEqual(self.metrics.errors, 0)

    def test_avg_response_time_zero_requests(self):
        """Test avg response time with zero requests."""
        self.assertEqual(self.metrics.avg_response_time, 0.0)

    def test_avg_response_time_with_requests(self):
        """Test avg response time calculation."""
        self.metrics.record_request(1.0, cached=False, error=False)
        self.metrics.record_request(2.0, cached=False, error=False)
        self.assertEqual(self.metrics.avg_response_time, 1.5)

    def test_cache_hit_rate_zero(self):
        """Test cache hit rate with no requests."""
        self.assertEqual(self.metrics.cache_hit_rate, 0.0)

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        self.metrics.record_request(0.1, cached=True)
        self.metrics.record_request(0.5, cached=False)
        self.metrics.record_request(0.1, cached=True)
        self.metrics.record_request(0.5, cached=False)
        self.assertEqual(self.metrics.cache_hit_rate, 0.5)

    def test_record_request(self):
        """Test recording a request."""
        self.metrics.record_request(1.5, cached=True, error=False)
        self.assertEqual(self.metrics.requests_handled, 1)
        self.assertEqual(self.metrics.total_response_time, 1.5)
        self.assertEqual(self.metrics.cache_hits, 1)
        self.assertEqual(self.metrics.cache_misses, 0)
        self.assertEqual(self.metrics.errors, 0)

    def test_record_error(self):
        """Test recording an error."""
        self.metrics.record_request(0.1, cached=False, error=True)
        self.assertEqual(self.metrics.errors, 1)


class TestMCPConfig(unittest.TestCase):
    """Test MCPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPConfig()
        self.assertEqual(config.server_name, "toolboxv2_mcp")
        self.assertEqual(config.server_version, "3.0.0")
        self.assertEqual(config.server_mode, ServerMode.STDIO)
        self.assertEqual(config.http_host, "127.0.0.1")
        self.assertEqual(config.http_port, 8765)
        self.assertTrue(config.require_auth)
        self.assertTrue(config.enable_python)
        self.assertTrue(config.enable_docs)
        self.assertTrue(config.enable_flows)
        self.assertTrue(config.enable_system)
        self.assertTrue(config.lazy_load)
        self.assertTrue(config.use_cache)
        self.assertEqual(config.cache_ttl, 300)
        self.assertEqual(config.max_cache_size, 100)
        self.assertEqual(config.session_timeout, 3600)
        self.assertEqual(config.max_sessions, 100)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MCPConfig(
            server_name="custom_mcp",
            server_mode=ServerMode.HTTP,
            http_port=9000,
            require_auth=False,
            enable_python=False,
        )
        self.assertEqual(config.server_name, "custom_mcp")
        self.assertEqual(config.server_mode, ServerMode.HTTP)
        self.assertEqual(config.http_port, 9000)
        self.assertFalse(config.require_auth)
        self.assertFalse(config.enable_python)

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = MCPConfig()
        d = config.to_dict()

        self.assertEqual(d["server_name"], "toolboxv2_mcp")
        self.assertEqual(d["server_version"], "3.0.0")
        self.assertEqual(d["mode"], "stdio")
        self.assertIn("features", d)
        self.assertTrue(d["features"]["python"])
        self.assertIn("performance", d)
        self.assertTrue(d["performance"]["lazy_load"])


class TestTemplates(unittest.TestCase):
    """Test resource templates."""

    def test_discovery_template_exists(self):
        """Test discovery template is defined."""
        self.assertIsInstance(FLOWAGENTS_DISCOVERY_TEMPLATE, str)
        self.assertIn("Server Capabilities", FLOWAGENTS_DISCOVERY_TEMPLATE)
        self.assertIn("toolbox_execute", FLOWAGENTS_DISCOVERY_TEMPLATE)

    def test_python_template_exists(self):
        """Test Python template is defined."""
        self.assertIsInstance(PYTHON_EXECUTION_TEMPLATE, str)
        self.assertIn("app", PYTHON_EXECUTION_TEMPLATE)
        self.assertIn("Persistent State", PYTHON_EXECUTION_TEMPLATE)

    def test_performance_template_format(self):
        """Test performance template can be formatted."""
        formatted = PERFORMANCE_GUIDE_TEMPLATE.format(
            cache_ttl=300, max_cache_size=100, requests=50, avg_time=0.5, hit_rate=0.75
        )
        self.assertIn("300", formatted)
        self.assertIn("100", formatted)
        self.assertIn("50", formatted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
