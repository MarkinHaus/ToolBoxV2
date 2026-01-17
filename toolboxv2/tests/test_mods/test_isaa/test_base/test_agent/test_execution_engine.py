"""
Unit tests for ExecutionEngine.

Compatible with unittest and pytest.
"""


from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.conftest2 import AsyncTestCase

from toolboxv2.mods.isaa.base.Agent.execution_engine import (
            ExecutionState, ExecutionPhase, ExecutionEngine
        )
from unittest.mock import MagicMock, AsyncMock, patch


class TestExecutionEngine(AsyncTestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.session_manager.get_or_create = AsyncMock()
        self.agent.session_manager.get_or_create.return_value = MagicMock()

        # Mock Session
        self.session = self.agent.session_manager.get_or_create.return_value
        self.session.vfs = MagicMock()
        self.session.vfs.to_checkpoint.return_value = {}

        self.agent.a_format_class = AsyncMock()
        self.engine = ExecutionEngine(self.agent)

    def test_01_init(self):
        self.assertIsInstance(self.engine, ExecutionEngine)

    def test_03_immediate_path(self):
        self.agent.a_run_llm_completion = AsyncMock(return_value="Hi there")
        state = ExecutionState("id", "Hello", "sess")

        res = self.async_run(self.engine._immediate_response(state, self.session))
        self.assertEqual(res, "Hi there")
        self.assertEqual(state.phase, ExecutionPhase.COMPLETED)

    def test_04_category_selection(self):
        self.agent.tool_manager.list_categories.return_value = ["cat1"]
        self.agent.a_format_class.return_value = {'categories': ['cat1'], 'reasoning': 'r'}

        state = ExecutionState("id", "Use cat1", "sess")
        self.async_run(self.engine._select_categories(state, self.session))
        self.assertEqual(state.selected_categories, ['cat1'])

    def test_05_tool_selection(self):
        tool = MagicMock();
        tool.name = "t1"
        self.agent.tool_manager.get_by_category.return_value = [tool]

        state = ExecutionState("id", "Use t1", "sess")
        state.selected_categories = ["cat1"]

        self.async_run(self.engine._select_tools(state, self.session))
        self.assertIn("t1", state.selected_tools)

    def test_06_react_loop_thinking(self):
        # Mock native response without tool calls
        resp = MagicMock()
        resp.tool_calls = None
        resp.content = "Thinking..."
        self.agent.a_run_llm_completion.return_value = resp

        state = ExecutionState("id", "Q", "sess", max_iterations=1)
        state.selected_tools = []

        # Mock format fallback
        self.agent.a_format_class.return_value = {'action': 'final_answer', 'answer': '42'}

        res = self.async_run(self.engine._react_loop(state, self.session))
        self.assertEqual(res, "42")

    def test_07_execute_action_vfs(self):
        state = ExecutionState("id", "Q", "sess")
        action = {'type': 'vfs_read', 'filename': 'f1'}
        self.session.vfs.read.return_value = {"success": True, "content": "content"}

        res = self.async_run(self.engine._execute_action(state, self.session, action))
        self.assertEqual(res, "content")

    def test_08_execute_action_tool(self):
        state = ExecutionState("id", "Q", "sess")
        state.selected_tools = ["t1"]
        action = {'type': 'tool_call', 'tool': 't1', 'args': {}}

        self.agent.arun_function = AsyncMock(return_value="result")

        res = self.async_run(self.engine._execute_action(state, self.session, action))
        self.assertEqual(res, "result")

    def test_09_pause(self):
        state = ExecutionState("id", "Q", "sess")
        self.engine._executions["id"] = state

        paused = self.async_run(self.engine.pause("id"))
        self.assertEqual(paused.phase, ExecutionPhase.PAUSED)

    def test_10_cancel(self):
        state = ExecutionState("id", "Q", "sess")
        self.engine._executions["id"] = state

        res = self.async_run(self.engine.cancel("id"))
        self.assertTrue(res)
        self.assertNotIn("id", self.engine._executions)

    def test_12_validation(self):
        self.agent.a_format_class.return_value = {'is_valid': True, 'confidence': 1.0, 'issues': []}
        state = ExecutionState("id", "Q", "sess")

        res = self.async_run(self.engine._validate_result(state, "res"))
        self.assertTrue(res.is_valid)

    def test_13_decomposition(self):
        self.agent.a_format_class.return_value = {
            'subtasks': ['t1'], 'can_parallel': [False]
        }
        state = ExecutionState("id", "Complex", "sess")

        self.async_run(self.engine._decompose_task(state, self.session))
        self.assertEqual(len(state.subtasks), 1)

    def test_14_microagent_run(self):
        # Mock tool selection for subtask
        self.agent.a_format_class.return_value = {'tools': ['t1']}

        # Mock react loop
        with patch.object(self.engine, '_react_loop', new_callable=AsyncMock) as mock_react:
            mock_react.return_value = "Done"
            subtask = {'id': 't1', 'description': 'd'}

            res = self.async_run(self.engine._run_microagent(
                ExecutionState("id", "Q", "sess"), self.session, subtask
            ))
            self.assertEqual(res.result, "Done")

    def test_15_continue_execution(self):
        state = ExecutionState("id", "Q", "sess", phase=ExecutionPhase.PAUSED)

        # Mock _run_execution to avoid real logic
        with patch.object(self.engine, '_run_execution', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Result"
            res = self.async_run(self.engine._continue_execution(state))
            self.assertEqual(res, "Result")
            self.assertIsNotNone(state.resumed_at)

    def test_16_full_execute_flow_simple(self):
        # Setup intent to be immediate
        self.agent.a_format_class.return_value = {
            'can_answer_directly': True, 'needs_tools': False,
            'is_complex_task': False, 'confidence': 1.0
        }
        r = lambda x: None
        r.tool_calls = None
        r.content = "Ans"
        self.agent.a_run_llm_completion = AsyncMock(return_value=r)

        self.session.add_message = AsyncMock()
        self.session.get_reference = AsyncMock()
        res = self.async_run(self.engine.execute("Query"))
        self.assertEqual(res.response, "Ans")

    def test_17_error_handling_retry(self):
        state = ExecutionState("id", "Q", "sess")
        handled = self.async_run(self.engine._handle_error(state, self.session, "err"))
        self.assertTrue(handled)  # Should escalate
        self.assertTrue(state.escalated)

    def test_18_error_handling_human(self):
        state = ExecutionState("id", "Q", "sess")
        state.escalated = True
        state.retry_count = 2
        self.engine.human_online = True

        handled = self.async_run(self.engine._handle_error(state, self.session, "err"))
        self.assertTrue(handled)
        self.assertTrue(state.waiting_for_human)
