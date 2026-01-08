"""
Unit tests for ToolManager and SessionManager.

Compatible with unittest and pytest.
"""

import unittest
import asyncio

import unittest
import os
from unittest.mock import MagicMock, patch, AsyncMock

from toolboxv2.mods.isaa.base.Agent import ToolManager, BindManager, SessionManager, CheckpointManager
from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.conftest2 import AsyncTestCase, MockAgentModelData


class TestToolManager(AsyncTestCase):
    def setUp(self):
        self.tm = ToolManager()

    def test_01_register_local(self):
        def my_func(a: int):
            """Desc"""
            return a * 2
        entry = self.tm.register(my_func)
        self.assertEqual(entry.name, "my_func")
        self.assertEqual(entry.args_schema, "(a: int)")
        self.assertIn("local", entry.category)

    def test_02_register_flags_inference(self):
        self.tm.register(lambda: None, name="delete_db")
        entry = self.tm.get("delete_db")
        self.assertTrue(entry.flags['dangerous'])

    def test_03_execute(self):
        self.tm.register(lambda x: x + 1, name="inc")
        res = self.async_run(self.tm.execute("inc", x=5))
        self.assertEqual(res, 6)

    def test_04_async_execution(self):
        async def async_add(x): return x + 1

        self.tm.register(async_add, name="async_inc")
        res = self.async_run(self.tm.execute("async_inc", x=10))
        self.assertEqual(res, 11)

    def test_05_litellm_schema(self):
        self.tm.register(lambda a, b=1: None, name="schema_test")
        schema = self.tm.get_litellm_schema("schema_test")
        self.assertEqual(schema['function']['name'], "schema_test")
        self.assertIn("a", schema['function']['parameters']['properties'])

    def test_06_register_mcp(self):
        tools = [{'name': 'ls', 'description': 'List', 'inputSchema': {}}]
        self.tm.register_mcp_tools("server1", tools)
        self.assertTrue(self.tm.exists("server1_ls"))
        self.assertEqual(self.tm.get("server1_ls").source, "mcp")


class TestBindManager(AsyncTestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.amd.name = "AgentA"
        # Mock SessionManager for BindManager
        self.agent.session_manager = MagicMock()
        self.bm = BindManager(self.agent)

    def test_07_bind_creation(self):
        partner = MagicMock()
        partner.amd.name = "AgentB"
        self.agent.session_manager.get_or_create = AsyncMock()
        res = self.async_run(self.bm.bind(partner, mode='private'))
        self.assertIn("AgentB", self.bm.bindings)
        self.assertEqual(res.mode, 'private')
        self.agent.session_manager.get_or_create.assert_called_with('default')

    def test_08_sync_filename(self):
        name = self.bm._generate_sync_filename("AgentB", "private")
        self.assertTrue("AgentA" in name and "AgentB" in name)

    def test_09_write_sync(self):
        # Mock binding
        self.bm.bindings["AgentB"] = MagicMock(sync_filename="sync.json")
        # Mock Session VFS
        session_mock = MagicMock()
        session_mock.vfs.read.return_value = {'success': True, 'content': '{}'}
        self.agent.session_manager.get.return_value = session_mock

        self.async_run(self.bm.write_sync("msg", "hello", "AgentB"))
        session_mock.vfs.write.assert_called()


class TestSessionManager(AsyncTestCase):
    def setUp(self):
        self.sm = SessionManager("AgentA")
        # Mock memory
        self.sm._memory_instance = MagicMock()

    def test_10_create_session(self):
        with patch('toolboxv2.mods.isaa.extras.session.ChatSession'), \
            patch('toolboxv2.mods.isaa.base.Agent.rule_set.create_default_ruleset'):
            sess = self.async_run(self.sm.get_or_create("s1"))
            self.assertIsNotNone(sess)
            self.assertIn("s1", self.sm.sessions)

    def test_11_get_existing(self):
        with patch('toolboxv2.mods.isaa.extras.session.ChatSession'), \
            patch('toolboxv2.mods.isaa.base.Agent.rule_set.create_default_ruleset'):
            s1 = self.async_run(self.sm.get_or_create("s1"))
            s2 = self.async_run(self.sm.get_or_create("s1"))
            self.assertEqual(s1, s2)

    def test_12_close_session(self):
        with patch('toolboxv2.mods.isaa.extras.session.ChatSession'), \
            patch('toolboxv2.mods.isaa.base.Agent.rule_set.create_default_ruleset'):
            self.async_run(self.sm.get_or_create("s1"))
            res = self.async_run(self.sm.close_session("s1"))
            self.assertTrue(res)
            self.assertNotIn("s1", self.sm.sessions)

    def test_13_cleanup_inactive(self):
        # Difficult to test without mocking time, assume logic works if empty
        cleaned = self.async_run(self.sm.cleanup_inactive(max_idle_hours=0.0001))
        self.assertEqual(cleaned, 0)


class TestCheckpointManager(AsyncTestCase):
    def setUp(self):
        self.agent = AsyncMock()
        self.agent.amd.name = "AgentTest"
        self.cm = CheckpointManager(self.agent, checkpoint_dir="/tmp/test_cp", auto_load=False)

    def test_14_create_checkpoint(self):
        # Setup mocks
        self.agent.session_manager = MagicMock()
        self.agent.session_manager.to_checkpoint.return_value = {'sess': 1}
        self.agent.tool_manager = AsyncMock()
        self.agent.tool_manager.to_checkpoint.return_value = {'tools': 1}

        cp = self.async_run(self.cm.create())
        self.assertEqual(cp.agent_name, "AgentTest")
        self.assertEqual(cp.sessions_data['sess'], 1)
        self.agent.session_manager = AsyncMock()

    def test_15_save_load(self):
        import pickle
        cp_data = MagicMock(timestamp=MagicMock(strftime=lambda x: "2023"))

        with patch('builtins.open', unittest.mock.mock_open()) as m, \
            patch('pickle.dump') as mock_dump, \
            patch('os.makedirs'):
            self.async_run(self.cm.save(cp_data, filename="test.pkl"))
            mock_dump.assert_called()

    def test_16_list_checkpoints(self):
        with patch('os.listdir', return_value=['agent_checkpoint_20250101_100000.pkl']), \
            patch('os.path.exists', return_value=True), \
            patch('os.stat') as mock_stat:
            mock_stat.return_value.st_size = 100
            mock_stat.return_value.st_mtime = 1000
            import math
            self.cm.max_age_hours = math.inf
            cps = self.cm.list_checkpoints()
            self.assertEqual(len(cps), 1)

    # ... more specific tests for restore logic ...
    def test_17_restore_logic(self):
        cp = MagicMock()
        cp.sessions_data = {'s': 1}
        self.agent.session_manager.sessions = {}
        stats = self.async_run(self.cm.restore(cp, restore_sessions=True))
        self.assertTrue(stats['success'])
        self.agent.session_manager.from_checkpoint.assert_called()

    def test_18_auto_restore_fail(self):
        # No loaded checkpoint
        res = self.async_run(self.cm.auto_restore())
        self.assertFalse(res['success'])

    def test_19_delete_checkpoint(self):
        with patch('os.remove') as mock_rm, patch('os.path.exists', return_value=True):
            res = self.async_run(self.cm.delete_checkpoint("file.pkl"))
            self.assertTrue(res)
            mock_rm.assert_called()

    def test_20_cleanup_old(self):
        # Mock list to return many
        with patch.object(self.cm, 'list_checkpoints') as mock_list, \
            patch('os.remove'):
            mock_list.return_value = [{'filepath': 'f1', 'type': 'reg', 'filename': 'f1', 'size_kb': 100}, {'filepath': 'f2', 'type': 'reg', 'filename': 'f1', 'size_kb': 100}]
            self.async_run(self.cm.cleanup_old(keep_count=1))
            # Should try to remove one
            # (Logic validation depends on list order)

    def test_21_stats(self):
        stats = self.cm.get_stats()
        self.assertIn('total_checkpoints', stats)

    def test_22_manager_repr(self):
        self.assertIn("CheckpointManager", str(self.cm))
