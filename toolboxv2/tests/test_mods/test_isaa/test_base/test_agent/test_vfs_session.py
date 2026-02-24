import unittest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.conftest2 import  AsyncTestCase, MockAgentModelData, MockMemory
from toolboxv2.mods.isaa.base.Agent import VirtualFileSystem, AgentSession, VFSFile
class TestVirtualFileSystem(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.vfs = VirtualFileSystem("sess_1", "AgentX", max_window_lines=10)

    def test_02_read_file(self):
        self.vfs.create("data.json", "{}")
        res = self.vfs.read("data.json")
        self.assertEqual(res['content'], "{}")

    def test_05_delete_file(self):
        self.vfs.create("temp.txt", "")
        self.vfs.delete("temp.txt")
        self.assertNotIn("temp.txt", self.vfs.files)

    def test_06_system_file_readonly(self):
        res = self.vfs.write("system_context.md", "hacked")
        self.assertFalse(res['success'])
        self.assertIn("Read-only", res.get('error', ''))

    def test_11_open_close_state(self):
        self.vfs.create("doc.md", "Content")
        self.vfs.open("doc.md")
        self.assertEqual(self.vfs.files["/doc.md"].state, "open")
        self.async_run(self.vfs.close("/doc.md"))
        self.assertEqual(self.vfs.files["/doc.md"].state, "closed")

    def test_12_view_windowing(self):
        content = "\n".join([f"L{i}" for i in range(20)])
        self.vfs.create("/long.txt", content)
        self.vfs.open("/long.txt", line_start=5, line_end=6)
        f = self.vfs.files["/long.txt"]
        self.assertEqual(f.view_start, 4) # 0-indexed
        self.assertEqual(f.view_end, 6)

    def test_13_context_building(self):
        self.vfs.create("active.txt", "Important Data")
        self.vfs.open("active.txt")
        ctx = self.vfs.build_context_string()
        self.assertIn("Important Data", ctx)
        self.assertIn("system_context.md", ctx)

    def test_14_list_files(self):
        self.vfs.create("a.txt", "A")
        self.vfs.create("b.txt", "B")
        lst = self.vfs.list_files()
        self.assertEqual(len(lst['files']), 3) # a, b, system_context.md

    def test_16_load_local_security(self):
        # Should fail if not allowed dir (simulated)
        res = self.vfs.load_from_local("/etc/passwd", allowed_dirs=["/tmp"])
        self.assertFalse(res['success'])

    def test_17_save_local_new_dir(self):
        # Mock file ops
        with patch('os.makedirs') as mock_md, patch('builtins.open') as mock_open:
            self.vfs.create("save.txt", "content")
            res = self.vfs.save_to_local("save.txt", "/tmp/new/save.txt", create_dirs=True)
            self.assertTrue(res['success'])
            mock_md.assert_called()


class TestAgentSession(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.mem = MockMemory()
        # Mocking RuleSet import inside AgentSession
        with patch('toolboxv2.mods.isaa.base.Agent.rule_set.create_default_ruleset') as mock_rs:
             mock_rs.return_value = MagicMock()
             self.session = AgentSession("sess_id", "TestBot", self.mem)

    def test_18_initialization(self):
        # Mock ChatSession
        with patch('toolboxv2.mods.isaa.extras.session.ChatSession') as MockChat:
            self.async_run(self.session.initialize())
            self.assertTrue(self.session._initialized)

    def test_19_add_message(self):
        self.session._chat_session = AsyncMock()
        self.session._initialized = True
        self.async_run(self.session.add_message({"role": "user", "content": "Hi"}))
        self.session._chat_session.add_message.assert_called_once()

    def test_20_tool_restriction_default(self):
        self.assertTrue(self.session.is_tool_allowed("any_tool"))

    def test_21_tool_restriction_set(self):
        self.session.set_tool_restriction("dangerous_tool", False)
        self.assertFalse(self.session.is_tool_allowed("dangerous_tool"))

    def test_22_vfs_pass_through(self):
        self.session.vfs_create("test.file", "data")
        res = self.session.vfs_read("test.file")
        self.assertEqual(res['content'], "data")

    def test_23_stats(self):
        self.session._initialized = True
        stats = self.session.get_stats()
        self.assertEqual(stats['session_id'], "sess_id")

    def test_24_checkpointing(self):
        self.session.vfs.create("persist.txt", "data")
        data = self.session.to_checkpoint()
        self.assertEqual(data['session_id'], "sess_id")
        self.assertIn("/persist.txt", data['vfs']['files'])

    def test_25_restore(self):
        # Mock necessary components for restore
        data = {
            'session_id': 'restored', 'agent_name': 'A',
            'created_at': '2023-01-01T10:00:00',
            'last_activity': '2023-01-01T10:00:00',
            'vfs': {'files': {'f1': {'filename': 'f1', 'content': 'c'}}},
            'rule_set': {},
            'tool_restrictions': {'t1': False}
        }
        with patch('toolboxv2.mods.isaa.extras.session.ChatSession'):
            restored = self.async_run(AgentSession.from_checkpoint(data, self.mem))
            self.assertEqual(restored.session_id, "restored")
            self.assertFalse(restored.is_tool_allowed("t1"))

    def test_26_close_session(self):
        self.session._chat_session = MagicMock()
        self.async_run(self.session.close())
        self.assertTrue(self.session._closed)
