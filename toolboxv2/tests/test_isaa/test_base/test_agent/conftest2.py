import unittest
import asyncio
import sys

# Frueher: sys.modules['litellm'] usw. auf Modulebene durch MagicMock ersetzt. Das vergiftete jeden
# spaeteren Test im selben Prozess (testing_guide Regel 6.1), z. B. echte Agentenlaeufe in
# test_icli2_widget_betrieb.py. toolboxv2 ist installiert; die Tests hier brauchen die Ersetzung nicht.

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
        self.enable_lsp = False
        self.enable_docker = False
        self.docker_config = None

    def get_system_message(self):
        return self.system_message

class MockMemory:
    """Mock for AISemanticMemory"""
    pass
