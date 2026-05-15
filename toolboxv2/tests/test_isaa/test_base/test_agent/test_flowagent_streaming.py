import unittest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from toolboxv2.mods.isaa.base.Agent.types import AgentModelData


class TestFlowAgentStreaming(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.amd = AgentModelData(
            name="StreamTestAgent",
            fast_llm_model="test/model",
            complex_llm_model="test/model",
            system_message="System test msg",
            vfs_max_window_lines=100
        )
        self.agent = FlowAgent(self.amd, auto_load_checkpoint=False)
        self.agent.checkpoint_manager.save = AsyncMock()

    async def asyncTearDown(self):
        await self.agent.close()

    @patch("toolboxv2.mods.isaa.base.Agent.execution_engine.ExecutionEngine.execute_stream")
    async def test_a_stream_yields_chunks_only_when_async_iterated(self, mock_execute_stream):
        # Arrange
        async def mock_stream_func(ctx):
            yield {"type": "content", "chunk": "Hello"}
            yield {"type": "done", "success": True, "final_answer": "Hello"}

        mock_execute_stream.return_value = (mock_stream_func, MagicMock())

        chunks = []

        # Act - Wenn hier nicht iteriert würde, gäbe es keine Chunks und keinen Fehler
        async for chunk in self.agent.a_stream(query="test"):
            chunks.append(chunk)

        # Assert
        self.assertGreater(len(chunks), 0, "Generator wurde nicht konsumiert!")
        # Der erste yielded Chunk aus a_stream ist hardcoded das Status-Signal
        self.assertEqual(chunks[0]["type"], "status")
        self.assertEqual(chunks[1]["type"], "status")
        self.assertEqual(chunks[2]["chunk"], "Hello")

    @patch("toolboxv2.mods.isaa.base.Agent.execution_engine.ExecutionEngine.execute_stream")
    async def test_a_stream_verbose_formats_text_correctly(self, mock_execute_stream):
        # Arrange
        async def mock_stream_func(ctx):
            yield {"type": "tool_start", "name": "vfs_shell"}
            yield {"type": "final_answer", "answer": "Done"}

        mock_execute_stream.return_value = (mock_stream_func, MagicMock())

        outputs = []

        # Act
        async for text_output in self.agent.a_stream_verbose(query="test"):
            outputs.append(text_output)

        # Assert
        self.assertTrue(any("vfs_shell" in out for out in outputs))
        self.assertTrue(any("Done" in out for out in outputs))
