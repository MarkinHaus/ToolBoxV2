import unittest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

# --- MOCKING EXTERNAL IMPORTS ---
# Da toolboxv2 evtl. nicht installiert ist, mocken wir die Importe vorab
sys.modules['toolboxv2'] = MagicMock()
sys.modules['toolboxv2.mods.isaa.base.Agent.types'] = MagicMock()
sys.modules['toolboxv2.mods.isaa.base.Agent.rule_set'] = MagicMock()
sys.modules['litellm'] = MagicMock()
sys.modules['python_a2a'] = MagicMock()
sys.modules['mcp.server.fastmcp'] = MagicMock()

# --- ASYNC TEST BASE ---
class AsyncTestCase(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        if hasattr(self, 'loop'):
            self.loop.close()

    def async_run(self, coro):
        if not hasattr(self, 'loop'):
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop.run_until_complete(coro)

# --- MOCK CLASSES ---
class MockAgentModelData:
    def __init__(self):
        self.name = "TestAgent"
        self.fast_llm_model = "mock/fast"
        self.complex_llm_model = "mock/complex"
        self.system_message = "You are a test agent."
        self.temperature = 0.5
        self.max_tokens = 1000
        self.max_input_tokens = 4000
        self.vfs_max_window_lines = 100
        self.handler_path_or_dict = {}
        self.persona = None

    def get_system_message(self):
        return self.system_message

class MockMemory:
    """Mock for AISemanticMemory"""
    pass
