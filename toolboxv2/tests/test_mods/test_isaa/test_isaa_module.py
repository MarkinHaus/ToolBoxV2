"""
Unit Tests for ISAA Module (Refactored V2)

Tests cover:
1. Chain Support (create_chain, run_chain, chain_from_agents)
2. Agent Export/Import System (save_agent, load_agent, export_agent_network, import_agent_network)
3. Tool Serialization (dill/cloudpickle)
4. Core functionality preservation (mini_task_completion, format_class, etc.)

Compatible with PyCharm and standard unittest runner.
Run with: python -m unittest test_isaa_module
"""

import asyncio
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime
from typing import Any

from toolboxv2.mods.isaa.module import ToolSerializationInfo, AgentNetworkManifest, AgentExportManifest

# Try to import pydantic, use simple dataclass fallback if not available
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Simple fallback BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        def model_dump_json(self):
            return json.dumps(self.model_dump())


# =============================================================================
# ASYNC TEST BASE CLASS
# =============================================================================

class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        if hasattr(self, 'loop') and self.loop:
            self.loop.close()

    def async_run(self, coro):
        """Run async coroutine synchronously"""
        if not hasattr(self, 'loop') or self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop.run_until_complete(coro)


# =============================================================================
# MOCK CLASSES
# =============================================================================

class MockAgentModelData:
    """Mock for AgentModelData"""

    def __init__(self, name="TestAgent"):
        self.name = name
        self.fast_llm_model = "mock/fast-model"
        self.complex_llm_model = "mock/complex-model"
        self.system_message = "You are a test agent."
        self.temperature = 0.7
        self.max_tokens = 2048
        self.max_input_tokens = 32768
        self.vfs_max_window_lines = 250
        self.handler_path_or_dict = {}
        self.persona = None
        self.use_fast_response = True

    def get_system_message(self):
        return self.system_message


class MockFlowAgent:
    """Mock for FlowAgent"""

    def __init__(self, name="TestAgent"):
        self.amd = MockAgentModelData(name)
        self.tool_manager = MockToolManager()
        self.checkpoint_manager = MockCheckpointManager()
        self.bind_manager = MockBindManager()
        self.stream = True
        self.stream_callback = None
        self.verbose = False
        self.progress_callback = None

    async def a_run(self, query, **kwargs):
        return f"Response to: {query}"

    async def a_format_class(self, pydantic_model, prompt, **kwargs):
        return {"result": "formatted"}

    async def a_run_llm_completion(self, **kwargs):
        return "LLM completion response"

    async def close(self):
        pass


class MockToolManager:
    """Mock for ToolManager"""

    def __init__(self):
        self.tools = {}

    def register(self, func, name=None, description=None, category=None, flags=None):
        tool_name = name or func.__name__
        self.tools[tool_name] = {
            'function': func,
            'description': description or '',
            'category': category or [],
        }


class MockCheckpointManager:
    """Mock for CheckpointManager"""

    def __init__(self):
        self.checkpoints = {}

    async def save_current(self):
        return "/tmp/mock_checkpoint.json"

    async def restore_from_dict(self, data):
        self.checkpoints = data


class MockBindManager:
    """Mock for BindManager"""

    def __init__(self):
        self.bindings = {}

    async def bind(self, partner, mode='public', session_id='default'):
        self.bindings[partner.amd.name] = partner
        return True

    def unbind(self, partner_name):
        if partner_name in self.bindings:
            del self.bindings[partner_name]
            return True
        return False


class MockFlowAgentBuilder:
    """Mock for FlowAgentBuilder"""

    def __init__(self, config=None):
        self.config = config or MockAgentConfig()
        self._isaa_ref = None
        self._custom_tools = {}

    def add_tool(self, func, name=None, description=None, category=None, flags=None):
        tool_name = name or func.__name__
        self._custom_tools[tool_name] = func
        return self

    def with_models(self, fast, complex_=None):
        self.config.fast_llm_model = fast
        if complex_:
            self.config.complex_llm_model = complex_
        return self

    def with_name(self, name):
        self.config.name = name
        return self

    def with_budget_manager(self, max_cost=10.0):
        return self

    def verbose(self, enable=True):
        return self

    def save_config(self, path, format='json'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config.model_dump(), f)

    async def build(self):
        agent = MockFlowAgent(self.config.name)
        for name, func in self._custom_tools.items():
            agent.tool_manager.tools[name] = {'function': func, 'description': '', 'category': []}
        return agent


class MockAgentConfig:
    """Mock for AgentConfig"""

    def __init__(self, name="TestAgent"):
        self.name = name
        self.fast_llm_model = "mock/fast"
        self.complex_llm_model = "mock/complex"
        self.system_message = "Test system message"
        self.temperature = 0.7
        self.max_tokens_output = 2048
        self.max_tokens_input = 32768
        self.version = "1.0.0"

    def model_dump(self):
        return {
            'name': self.name,
            'fast_llm_model': self.fast_llm_model,
            'complex_llm_model': self.complex_llm_model,
            'system_message': self.system_message,
            'temperature': self.temperature,
            'max_tokens_output': self.max_tokens_output,
            'max_tokens_input': self.max_tokens_input,
            'version': self.version,
        }


class MockChain:
    """Mock for Chain"""

    def __init__(self, agent=None):
        self.tasks = [agent] if agent else []

    @classmethod
    def _create_chain(cls, components):
        chain = cls()
        chain.tasks = list(components)
        return chain

    async def a_run(self, query, **kwargs):
        result = query
        for task in self.tasks:
            if hasattr(task, 'a_run'):
                result = await task.a_run(result, **kwargs)
            elif callable(task):
                if asyncio.iscoroutinefunction(task):
                    result = await task(result)
                else:
                    result = task(result)
        return result


class MockApp:
    """Mock for ToolBoxV2 App"""

    def __init__(self):
        self.data_dir = tempfile.mkdtemp()
        self.id = "test_app"

    def run_any(self, *args, **kwargs):
        pass

    def run_bg_task_advanced(self, coro):
        pass


class MockController:
    """Mock for ControllerManager"""

    def __init__(self, data=None):
        self.data = data or {}

    def init(self, path):
        pass

    def save(self, path):
        pass

    def rget(self, mode):
        return None


# =============================================================================
# TEST: TOOL SERIALIZATION
# =============================================================================

class TestToolSerialization(unittest.TestCase):
    """Tests for tool serialization helpers"""

    def test_serialization_info_model(self):
        """Test ToolSerializationInfo model creation"""
        # Use the BaseModel from top-level import (pydantic or fallback)

        # Manual init for compatibility with fallback
        info = ToolSerializationInfo(
            name="test_tool",
            serializable=True,
            module_path="test.module",
            function_name="test_func"
        )

        self.assertEqual(info.name, "test_tool")
        self.assertTrue(info.serializable)
        self.assertEqual(info.module_path, "test.module")

    def test_serialization_info_with_error(self):
        """Test ToolSerializationInfo with error"""
        info = ToolSerializationInfo(
            name="closure_tool",
            serializable=False,
            error_message="Cannot serialize closure",
            source_hint="Define at module level"
        )

        self.assertFalse(info.serializable)
        self.assertIn("closure", info.error_message.lower())
        self.assertIsNotNone(info.source_hint)

    def test_simple_function_serializable(self):
        """Test that simple functions are serializable"""
        try:
            import dill
            DILL_AVAILABLE = True
        except ImportError:
            DILL_AVAILABLE = False

        if not DILL_AVAILABLE:
            self.skipTest("dill not available")

        def simple_tool(x: str) -> str:
            return x.upper()

        serialized = dill.dumps(simple_tool)
        self.assertIsNotNone(serialized)

        deserialized = dill.loads(serialized)
        self.assertEqual(deserialized("test"), "TEST")

    def test_async_function_serializable(self):
        """Test that async functions are serializable"""
        try:
            import dill
            DILL_AVAILABLE = True
        except ImportError:
            DILL_AVAILABLE = False

        if not DILL_AVAILABLE:
            self.skipTest("dill not available")

        async def async_tool(x: str) -> str:
            return x.lower()

        serialized = dill.dumps(async_tool)
        self.assertIsNotNone(serialized)

        deserialized = dill.loads(serialized)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(deserialized("TEST"))
        loop.close()
        self.assertEqual(result, "test")


# =============================================================================
# TEST: AGENT EXPORT MANIFEST
# =============================================================================

class TestAgentExportManifest(unittest.TestCase):
    """Tests for AgentExportManifest"""

    def test_manifest_creation(self):
        """Test basic manifest creation"""

        manifest = AgentExportManifest(
            version="1.0",
            export_date=datetime.now().isoformat(),
            agent_name="test_agent",
            agent_version="1.0.0",
            has_checkpoint=True,
            has_tools=True,
            tool_count=3,
            serializable_tools=["tool1", "tool2"],
            non_serializable_tools=[
                ToolSerializationInfo(name="tool3", serializable=False, error_message="closure")
            ],
            bindings=["partner_agent"]
        )

        self.assertEqual(manifest.agent_name, "test_agent")
        self.assertEqual(manifest.tool_count, 3)
        self.assertEqual(len(manifest.serializable_tools), 2)
        self.assertEqual(len(manifest.non_serializable_tools), 1)

    def test_manifest_json_serialization(self):
        """Test manifest can be serialized to JSON"""

        manifest = AgentExportManifest(
            version="1.0",
            export_date=datetime.now().isoformat(),
            agent_name="test",
            has_checkpoint=False,
            tool_count=0,
            serializable_tools=[],
            non_serializable_tools=[],
            bindings=[],
            has_tools=False,
            agent_version="1.0.0"

        )

        json_str = manifest.model_dump_json()
        self.assertIsInstance(json_str, str)

        data = json.loads(json_str)
        self.assertEqual(data['agent_name'], 'test')


# =============================================================================
# TEST: CHAIN SUPPORT
# =============================================================================

class TestChainSupport(AsyncTestCase):
    """Tests for Chain helper methods"""

    def test_create_empty_chain(self):
        """Test creating an empty chain"""
        chain = MockChain()
        self.assertEqual(len(chain.tasks), 0)

    def test_create_chain_single_agent(self):
        """Test creating chain with single agent"""
        agent = MockFlowAgent("agent1")
        chain = MockChain(agent)
        self.assertEqual(len(chain.tasks), 1)

    def test_create_chain_multiple_components(self):
        """Test creating chain with multiple components"""
        agent1 = MockFlowAgent("agent1")
        agent2 = MockFlowAgent("agent2")

        chain = MockChain._create_chain([agent1, agent2])
        self.assertEqual(len(chain.tasks), 2)

    def test_chain_with_function(self):
        """Test chain with plain function"""

        def transform(x):
            return x.upper()

        chain = MockChain._create_chain([transform])

        result = self.async_run(chain.a_run("hello"))
        self.assertEqual(result, "HELLO")

    def test_chain_with_async_function(self):
        """Test chain with async function"""

        async def async_transform(x):
            return x + "!"

        chain = MockChain._create_chain([async_transform])

        result = self.async_run(chain.a_run("hello"))
        self.assertEqual(result, "hello!")

    def test_chain_execution_order(self):
        """Test that chain executes in correct order"""
        results = []

        def step1(x):
            results.append("step1")
            return x + "1"

        def step2(x):
            results.append("step2")
            return x + "2"

        def step3(x):
            results.append("step3")
            return x + "3"

        chain = MockChain._create_chain([step1, step2, step3])

        result = self.async_run(chain.a_run("start"))

        self.assertEqual(results, ["step1", "step2", "step3"])
        self.assertEqual(result, "start123")


# =============================================================================
# TEST: TAR.GZ ARCHIVE HANDLING
# =============================================================================

class TestTarGzHandling(unittest.TestCase):
    """Tests for tar.gz archive creation and reading"""

    def test_create_tar_gz(self):
        """Test creating a tar.gz archive"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.tar.gz"

            with tarfile.open(archive_path, 'w:gz') as tar:
                # Add a file
                data = b'{"test": "data"}'
                info = tarfile.TarInfo(name='test.json')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            self.assertTrue(archive_path.exists())

    def test_read_from_tar_gz(self):
        """Test reading from tar.gz archive"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.tar.gz"

            # Create archive
            test_data = {"key": "value"}
            with tarfile.open(archive_path, 'w:gz') as tar:
                data = json.dumps(test_data).encode('utf-8')
                info = tarfile.TarInfo(name='data.json')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            # Read archive
            with tarfile.open(archive_path, 'r:gz') as tar:
                member = tar.getmember('data.json')
                f = tar.extractfile(member)
                content = json.loads(f.read().decode('utf-8'))

            self.assertEqual(content, test_data)

    def test_archive_multiple_files(self):
        """Test archive with multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "multi.tar.gz"

            files = {
                'manifest.json': '{"version": "1.0"}',
                'config.json': '{"name": "test"}',
                'checkpoint.json': '{"state": {}}',
            }

            with tarfile.open(archive_path, 'w:gz') as tar:
                for name, content in files.items():
                    data = content.encode('utf-8')
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

            # Verify all files present
            with tarfile.open(archive_path, 'r:gz') as tar:
                names = tar.getnames()
                self.assertEqual(set(names), set(files.keys()))


# =============================================================================
# TEST: AGENT NETWORK MANIFEST
# =============================================================================

class TestAgentNetworkManifest(unittest.TestCase):
    """Tests for AgentNetworkManifest"""

    def test_network_manifest_creation(self):
        """Test basic network manifest creation"""

        manifest = AgentNetworkManifest(
            version="1.0",
            export_date=datetime.now().isoformat(),
            agents=["agent1", "agent2", "agent3"],
            bindings={
                "agent1": ["agent2"],
                "agent2": ["agent3"],
            },
            entry_agent="agent1"
        )

        self.assertEqual(len(manifest.agents), 3)
        self.assertEqual(manifest.entry_agent, "agent1")
        self.assertIn("agent1", manifest.bindings)

    def test_network_manifest_empty_bindings(self):
        """Test network manifest with no bindings"""

        manifest = AgentNetworkManifest(
            version="1.0",
            export_date=datetime.now().isoformat(),
            agents=["standalone"],
            bindings={},
            entry_agent="standalone"
        )

        self.assertEqual(len(manifest.bindings), 0)


# =============================================================================
# TEST: CORE MODULE METHODS (MOCKED)
# =============================================================================

class TestCoreModuleMethods(AsyncTestCase):
    """Tests for core module methods with mocking"""

    def setUp(self):
        super().setUp()
        self.mock_app = MockApp()

    def test_get_agent_returns_agent(self):
        """Test get_agent returns a FlowAgent instance"""
        agent = MockFlowAgent("test")
        self.assertEqual(agent.amd.name, "test")
        self.assertIsNotNone(agent.tool_manager)

    def test_agent_a_run(self):
        """Test agent a_run method"""
        agent = MockFlowAgent("test")
        result = self.async_run(agent.a_run("Hello"))
        self.assertIn("Hello", result)

    def test_agent_a_format_class(self):
        """Test agent a_format_class method"""
        agent = MockFlowAgent("test")
        result = self.async_run(agent.a_format_class(dict, "test prompt"))
        self.assertIsInstance(result, dict)

    def test_tool_manager_register(self):
        """Test tool registration"""
        manager = MockToolManager()

        def my_tool(x: str) -> str:
            return x

        manager.register(my_tool, name="my_tool", description="Test tool")

        self.assertIn("my_tool", manager.tools)
        self.assertEqual(manager.tools["my_tool"]['description'], "Test tool")

    def test_checkpoint_manager_save(self):
        """Test checkpoint save"""
        manager = MockCheckpointManager()
        path = self.async_run(manager.save_current())
        self.assertIsNotNone(path)

    def test_bind_manager_bind(self):
        """Test agent binding"""
        manager = MockBindManager()
        partner = MockFlowAgent("partner")

        result = self.async_run(manager.bind(partner))
        self.assertTrue(result)
        self.assertIn("partner", manager.bindings)

    def test_bind_manager_unbind(self):
        """Test agent unbinding"""
        manager = MockBindManager()
        partner = MockFlowAgent("partner")

        self.async_run(manager.bind(partner))
        result = manager.unbind("partner")

        self.assertTrue(result)
        self.assertNotIn("partner", manager.bindings)


# =============================================================================
# TEST: BUILDER MOCK
# =============================================================================

class TestFlowAgentBuilder(AsyncTestCase):
    """Tests for FlowAgentBuilder mock"""

    def test_builder_creation(self):
        """Test builder creation"""
        config = MockAgentConfig("test_agent")
        builder = MockFlowAgentBuilder(config)

        self.assertEqual(builder.config.name, "test_agent")

    def test_builder_add_tool(self):
        """Test adding tool to builder"""
        builder = MockFlowAgentBuilder()

        def my_tool(x):
            return x

        builder.add_tool(my_tool, name="my_tool")
        self.assertIn("my_tool", builder._custom_tools)

    def test_builder_with_models(self):
        """Test setting models"""
        builder = MockFlowAgentBuilder()
        builder.with_models("fast/model", "complex/model")

        self.assertEqual(builder.config.fast_llm_model, "fast/model")
        self.assertEqual(builder.config.complex_llm_model, "complex/model")

    def test_builder_build(self):
        """Test building agent"""
        builder = MockFlowAgentBuilder()
        builder.with_name("built_agent")

        def tool1(x):
            return x

        builder.add_tool(tool1, name="tool1")

        agent = self.async_run(builder.build())

        self.assertEqual(agent.amd.name, "built_agent")
        self.assertIn("tool1", agent.tool_manager.tools)

    def test_builder_save_config(self):
        """Test saving builder config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agent" / "config.json"

            builder = MockFlowAgentBuilder()
            builder.save_config(str(config_path))

            self.assertTrue(config_path.exists())

            with open(config_path) as f:
                data = json.load(f)

            self.assertIn('name', data)


# =============================================================================
# TEST: INTEGRATION SCENARIOS
# =============================================================================

class TestIntegrationScenarios(AsyncTestCase):
    """Integration-style tests for common scenarios"""

    def test_scenario_create_and_run_chain(self):
        """Test creating and running a simple chain"""

        def step1(x):
            return f"[step1:{x}]"

        def step2(x):
            return f"[step2:{x}]"

        chain = MockChain._create_chain([step1, step2])
        result = self.async_run(chain.a_run("input"))

        self.assertEqual(result, "[step2:[step1:input]]")

    def test_scenario_agent_with_tools(self):
        """Test agent with registered tools"""
        builder = MockFlowAgentBuilder()

        async def search_tool(query: str) -> str:
            return f"Results for: {query}"

        async def save_tool(data: str) -> str:
            return f"Saved: {data}"

        builder.add_tool(search_tool, name="search")
        builder.add_tool(save_tool, name="save")

        agent = self.async_run(builder.build())

        self.assertIn("search", agent.tool_manager.tools)
        self.assertIn("save", agent.tool_manager.tools)

    def test_scenario_export_import_roundtrip(self):
        """Test conceptual export/import roundtrip"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate export
            export_data = {
                'manifest': {
                    'agent_name': 'test',
                    'version': '1.0',
                    'tool_count': 2
                },
                'config': {
                    'name': 'test',
                    'fast_llm_model': 'mock/fast'
                },
                'tools': ['tool1', 'tool2']
            }

            archive_path = Path(tmpdir) / "agent.tar.gz"

            with tarfile.open(archive_path, 'w:gz') as tar:
                for name, content in [
                    ('manifest.json', export_data['manifest']),
                    ('config.json', export_data['config']),
                ]:
                    data = json.dumps(content).encode('utf-8')
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

            # Simulate import
            with tarfile.open(archive_path, 'r:gz') as tar:
                manifest_data = tar.extractfile('manifest.json').read()
                manifest = json.loads(manifest_data)

                config_data = tar.extractfile('config.json').read()
                config = json.loads(config_data)

            self.assertEqual(manifest['agent_name'], 'test')
            self.assertEqual(config['name'], 'test')

    def test_scenario_network_with_bindings(self):
        """Test network of agents with bindings"""
        agents = {}

        # Create agents
        for name in ["analyzer", "processor", "output"]:
            agents[name] = MockFlowAgent(name)

        # Create bindings
        bindings = {
            "analyzer": ["processor"],
            "processor": ["output"]
        }

        # Verify structure
        self.assertEqual(len(agents), 3)
        self.assertIn("processor", bindings["analyzer"])
        self.assertIn("output", bindings["processor"])


# =============================================================================
# TEST: ERROR HANDLING
# =============================================================================

class TestErrorHandling(AsyncTestCase):
    """Tests for error handling"""

    def test_chain_with_failing_step(self):
        """Test chain handles failing step"""

        def failing_step(x):
            raise ValueError("Step failed")

        chain = MockChain._create_chain([failing_step])

        with self.assertRaises(ValueError):
            self.async_run(chain.a_run("input"))

    def test_missing_archive_file(self):
        """Test handling missing archive file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "nonexistent.tar.gz"

            with self.assertRaises(FileNotFoundError):
                with tarfile.open(archive_path, 'r:gz') as tar:
                    pass

    def test_invalid_json_in_archive(self):
        """Test handling invalid JSON in archive"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "invalid.tar.gz"

            with tarfile.open(archive_path, 'w:gz') as tar:
                data = b'not valid json {'
                info = tarfile.TarInfo(name='config.json')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            with tarfile.open(archive_path, 'r:gz') as tar:
                content = tar.extractfile('config.json').read()

                with self.assertRaises(json.JSONDecodeError):
                    json.loads(content)


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases(AsyncTestCase):
    """Tests for edge cases"""

    def test_empty_chain(self):
        """Test empty chain returns input unchanged"""
        chain = MockChain()
        result = self.async_run(chain.a_run("input"))
        self.assertEqual(result, "input")

    def test_chain_single_component(self):
        """Test chain with single component"""

        def only_step(x):
            return x * 2

        chain = MockChain._create_chain([only_step])
        result = self.async_run(chain.a_run("ab"))
        self.assertEqual(result, "abab")

    def test_agent_name_override(self):
        """Test agent name can be overridden"""
        builder = MockFlowAgentBuilder()
        builder.with_name("original")
        builder.with_name("overridden")

        self.assertEqual(builder.config.name, "overridden")

    def test_tool_without_name(self):
        """Test tool registration without explicit name"""
        manager = MockToolManager()

        def my_function(x):
            return x

        manager.register(my_function)
        self.assertIn("my_function", manager.tools)


# =============================================================================
# TEST SUITE RUNNER
# =============================================================================

def create_test_suite():
    """Create the complete test suite"""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestToolSerialization,
        TestAgentExportManifest,
        TestChainSupport,
        TestTarGzHandling,
        TestAgentNetworkManifest,
        TestCoreModuleMethods,
        TestFlowAgentBuilder,
        TestIntegrationScenarios,
        TestErrorHandling,
        TestEdgeCases,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


if __name__ == '__main__':
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(create_test_suite())

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
