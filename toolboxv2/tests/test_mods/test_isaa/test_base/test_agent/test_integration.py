import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.conftest2 import MockAgentModelData, AsyncTestCase

from toolboxv2.mods.isaa.base.Agent import (
            ExecutionResult, FlowAgent
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
            self.agent._execution_engine = MagicMock()

    def test_01_initialization(self):
        self.assertIsNotNone(self.agent.session_manager)
        self.assertIsNotNone(self.agent.tool_manager)

    def test_02_add_tool(self):
        self.async_run(self.agent.add_tool(lambda x: x, "test"))
        self.agent.tool_manager.register.assert_called()

    def test_03_run_simple(self):
        # Mock engine execution
        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(return_value=ExecutionResult(
            success=True, response="Answer", execution_id="1", path_taken="immediate"
        ))

        with patch.object(self.agent, '_get_execution_engine', return_value=mock_engine):
            res = self.async_run(self.agent.a_run("Hello"))
            self.assertEqual(res, "Answer")

    def test_04_run_paused(self):
        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(return_value=ExecutionResult(
            success=False, response="", execution_id="123", path_taken="paused", paused=True
        ))

        with patch.object(self.agent, '_get_execution_engine', return_value=mock_engine):
            res = self.async_run(self.agent.a_run("Wait"))
            self.assertEqual(res, "__PAUSED__:123")

    def test_05_run_needs_human(self):
        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(return_value=ExecutionResult(
            success=False, response="", execution_id="123", path_taken="paused",
            paused=True, needs_human=True, human_query="Why?"
        ))

        with patch.object(self.agent, '_get_execution_engine', return_value=mock_engine):
            res = self.async_run(self.agent.a_run("Help"))
            self.assertEqual(res, "__NEEDS_HUMAN__:123:Why?")

    def test_06_continue_execution(self):
        with patch.object(self.agent, 'a_run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Done"
            res = self.async_run(self.agent.continue_execution("id", "Answer"))
            self.assertEqual(res, "Done")
            mock_run.assert_called_with(query="", execution_id="id", human_response="Answer")

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


    def test_10_format_class_yaml_parse(self):
        from pydantic import BaseModel
        class T(BaseModel): a: int

        # Mock LLM return valid YAML
        self.agent.a_run_llm_completion = AsyncMock(return_value="```yaml\na: 10\n```")

        res = self.async_run(self.agent.a_format_class(T, "prompt"))
        self.assertEqual(res['a'], 10)

    def test_11_format_class_retry(self):
        from pydantic import BaseModel
        class T(BaseModel): a: int

        # Fail once then succeed
        self.agent.a_run_llm_completion = AsyncMock(side_effect=["Bad", "a: 5"])

        res = self.async_run(self.agent.a_format_class(T, "prompt", max_retries=1))
        self.assertEqual(res['a'], 5)

    def test_12_stats(self):
        stats = self.agent.get_stats()
        self.assertEqual(stats['agent_name'], "TestAgent")
        self.assertIn('sessions', stats)
