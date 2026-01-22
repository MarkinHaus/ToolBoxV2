import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.conftest2 import MockAgentModelData, AsyncTestCase

from toolboxv2.mods.isaa.base.Agent import (
            FlowAgent
        )

class TestIntegration(AsyncTestCase):
    def setUp(self):
        self.amd = MockAgentModelData()
        # Mock dependencies heavily for FlowAgent init
        with patch('toolboxv2.mods.isaa.base.Agent.session_manager.SessionManager'), \
            patch('toolboxv2.mods.isaa.base.Agent.tool_manager.ToolManager'), \
            patch('toolboxv2.mods.isaa.base.Agent.checkpoint_manager.CheckpointManager'), \
            patch('toolboxv2.mods.isaa.base.Agent.bind_manager.BindManager'), \
            patch('toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter.LiteLLMRateLimitHandler'):
            self.agent = FlowAgent(self.amd, auto_load_checkpoint=False)

            # Re-mock execution engine factory to return a mock or partial
            # Use AsyncMock for the execution engine since it has async methods
            self.agent._execution_engine = AsyncMock()

    def test_01_initialization(self):
        self.assertIsNotNone(self.agent.session_manager)
        self.assertIsNotNone(self.agent.tool_manager)

    def test_02_add_tool(self):
        self.agent.add_tool(lambda x: x, "test")
        self.agent.tool_manager.register.assert_called()

    def test_03_run_simple(self):
        # Mock engine execution
        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(return_value="Answer")

        with patch.object(self.agent, '_get_execution_engine', return_value=mock_engine):
            res = self.async_run(self.agent.a_run("Hello"))
            self.assertEqual(res, "Answer")

    def test_04_run_paused(self):
        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(return_value="__PAUSED__")

        with patch.object(self.agent, '_get_execution_engine', return_value=mock_engine):
            res = self.async_run(self.agent.a_run("Wait"))
            self.assertEqual(res, "__PAUSED__")

    def test_06_continue_execution(self):
        with patch.object(self.agent, 'a_run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Done"
            res = self.async_run(self.agent.resume_execution("id", 1))
            self.assertEqual(res, "Error: Execution id not found")

    def test_07_save_restore(self):
        self.agent.checkpoint_manager.save_current = AsyncMock(return_value="path")
        self.agent.checkpoint_manager.auto_restore = AsyncMock(return_value={'success': True})

        self.async_run(self.agent.save())
        self.agent.checkpoint_manager.save_current.assert_called()

        self.async_run(self.agent.restore())
        self.agent.checkpoint_manager.auto_restore.assert_called()

    def test_08_bind_unbind(self):
        partner = MagicMock()
        self.agent.bind_manager.bind = AsyncMock(return_value=True)
        self.agent.bind_manager.unbind = MagicMock(return_value=True)

        self.async_run(self.agent.bind(partner))
        self.agent.bind_manager.bind.assert_called()

        self.agent.unbind("P")
        self.agent.bind_manager.unbind.assert_called()


    def test_12_stats(self):
        stats = self.agent.get_stats()
        self.assertEqual(stats['agent_name'], "TestAgent")
        self.assertIn('sessions', stats)
