"""Tests for chat_store.ChatStore — disk roundtrip, seq, replay, rollback."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolboxv2.mods.isaa.ui.chat_store import ChatStore, ChatMeta


class ChatStoreCreateListDelete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_meta_and_jsonl(self):
        meta = self.store.create(agent="test_agent", title="t")
        self.assertTrue(self.store.exists(meta.chat_id))
        self.assertTrue((Path(self.tmp.name) / f"{meta.chat_id}.jsonl").exists())
        self.assertTrue((Path(self.tmp.name) / f"{meta.chat_id}.meta.json").exists())
        self.assertEqual(meta.agent, "test_agent")
        self.assertEqual(meta.session_id, meta.chat_id)

    def test_list_sorted_by_last_update(self):
        a = self.store.create(agent="a", title="a")
        # sleep would slow tests; force write order
        import time
        time.sleep(0.01)
        b = self.store.create(agent="b", title="b")
        items = self.store.list()
        ids = [i["chat_id"] for i in items]
        self.assertIn(a.chat_id, ids)
        self.assertIn(b.chat_id, ids)
        # b created later → first
        self.assertEqual(ids[0], b.chat_id)

    def test_delete_removes_both_files(self):
        m = self.store.create(agent="x")
        self.assertTrue(self.store.delete(m.chat_id))
        self.assertFalse(self.store.exists(m.chat_id))
        self.assertFalse(self.store.delete(m.chat_id))  # already gone

    def test_update_meta_merges_ui(self):
        m = self.store.create(agent="x")
        self.store.update_meta(m.chat_id, ui={"pinned": [1, 2]})
        self.store.update_meta(m.chat_id, ui={"vars": {"a": 1}})
        m2 = self.store.get_meta(m.chat_id)
        self.assertEqual(m2.ui.get("pinned"), [1, 2])
        self.assertEqual(m2.ui.get("vars"), {"a": 1})


class ChatStoreAppendAndSeq(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_assigns_increasing_seq(self):
        s1 = self.store.append(self.meta.chat_id, {"type": "user_msg", "text": "hi"})
        s2 = self.store.append(self.meta.chat_id, {"type": "content", "chunk": "hello"})
        s3 = self.store.append(self.meta.chat_id, {"type": "done", "success": True})
        self.assertEqual([s1, s2, s3], [1, 2, 3])
        self.assertEqual(self.store.last_seq(self.meta.chat_id), 3)

    def test_last_seq_persists_across_instances(self):
        self.store.append(self.meta.chat_id, {"type": "user_msg", "text": "hi"})
        self.store.append(self.meta.chat_id, {"type": "content", "chunk": "x"})
        store2 = ChatStore(self.tmp.name)
        self.assertEqual(store2.last_seq(self.meta.chat_id), 2)
        seq = store2.append(self.meta.chat_id, {"type": "done"})
        self.assertEqual(seq, 3)

    def test_message_count_increments_on_terminal(self):
        self.store.append(self.meta.chat_id, {"type": "user_msg", "text": "q"})
        self.store.append(self.meta.chat_id, {"type": "content", "chunk": "..."})
        self.store.append(self.meta.chat_id, {"type": "done", "success": True})
        m = self.store.get_meta(self.meta.chat_id)
        # user_msg + done counted
        self.assertEqual(m.message_count, 2)


class ChatStoreReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="x")
        for i, t in enumerate(["user_msg", "content", "content", "done"]):
            self.store.append(self.meta.chat_id, {"type": t, "i": i})

    def tearDown(self):
        self.tmp.cleanup()

    def test_replay_from_zero(self):
        frames = list(self.store.replay(self.meta.chat_id, after_seq=0))
        self.assertEqual(len(frames), 4)
        self.assertEqual([f["seq"] for f in frames], [1, 2, 3, 4])

    def test_replay_after_seq(self):
        frames = list(self.store.replay(self.meta.chat_id, after_seq=2))
        self.assertEqual([f["seq"] for f in frames], [3, 4])

    def test_replay_unknown_chat_returns_empty(self):
        self.assertEqual(list(self.store.replay("nonexistent")), [])


class ChatStoreRollback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="x")
        self.store.append(self.meta.chat_id, {"type": "user_msg", "text": "q1"})
        self.store.append(self.meta.chat_id, {"type": "iteration_start", "step_id": "r:1:0"})
        self.store.append(self.meta.chat_id, {"type": "content", "chunk": "...", "step_id": "r:1:0"})
        self.store.append(self.meta.chat_id, {"type": "iteration_start", "step_id": "r:2:0"})
        self.store.append(self.meta.chat_id, {"type": "content", "chunk": "...", "step_id": "r:2:0"})
        self.store.append(self.meta.chat_id, {"type": "done", "success": True, "step_id": "r:2:0"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_truncate_after_drops_step_and_later(self):
        new_last = self.store.truncate_after(self.meta.chat_id, "r:2:0")
        self.assertGreater(new_last, 0)
        frames = list(self.store.replay(self.meta.chat_id))
        # Everything with step_id r:2:0 dropped → 3 frames left (user_msg, r:1:0 start, r:1:0 content)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(f.get("step_id") != "r:2:0" for f in frames))

    def test_truncate_unknown_step_returns_minus1(self):
        self.assertEqual(self.store.truncate_after(self.meta.chat_id, "no-such-step"), -1)

    def test_truncate_updates_seq_cache(self):
        before = self.store.last_seq(self.meta.chat_id)
        new_last = self.store.truncate_after(self.meta.chat_id, "r:2:0")
        self.assertLess(new_last, before)
        # Next append uses new_last + 1
        seq = self.store.append(self.meta.chat_id, {"type": "user_msg", "text": "q2"})
        self.assertEqual(seq, new_last + 1)


class ChatStoreConcurrencySmoke(unittest.TestCase):
    """Not strict concurrency but ensures per-chat locks don't deadlock."""

    def test_many_appends_one_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(tmp)
            meta = store.create(agent="x")
            seqs = []
            for i in range(50):
                seqs.append(store.append(meta.chat_id, {"type": "content", "chunk": f"c{i}"}))
            self.assertEqual(seqs, list(range(1, 51)))


class ChatStoreColdBoot(unittest.TestCase):
    """Simulate server restart — drop the ChatStore instance and recreate from disk."""

    def test_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            # First "process"
            s1 = ChatStore(tmp)
            m = s1.create(agent="alice", title="my chat")
            s1.append(m.chat_id, {"type": "user_msg", "text": "hello"})
            s1.append(m.chat_id, {"type": "iteration_start", "step_id": "r1:1:0", "iter": 1})
            s1.append(m.chat_id, {"type": "content", "chunk": "world", "step_id": "r1:1:0"})
            s1.update_meta(m.chat_id, run_id="r1", ui={"expanded_steps": ["r1:1:0"]})
            del s1

            # Second "process" — fresh ChatStore, same dir
            s2 = ChatStore(tmp)
            chats = s2.list()
            self.assertEqual(len(chats), 1)
            meta = s2.get_meta(m.chat_id)
            self.assertEqual(meta.title, "my chat")
            self.assertEqual(meta.agent, "alice")
            self.assertEqual(meta.run_id, "r1")
            self.assertEqual(meta.ui.get("expanded_steps"), ["r1:1:0"])

            frames = list(s2.replay(m.chat_id))
            self.assertEqual(len(frames), 3)
            self.assertEqual([f["type"] for f in frames], ["user_msg", "iteration_start", "content"])

            # Next append continues seq correctly
            seq = s2.append(m.chat_id, {"type": "done", "success": True})
            self.assertEqual(seq, 4)

    def test_truncate_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = ChatStore(tmp)
            m = s1.create(agent="x")
            s1.append(m.chat_id, {"type": "user_msg", "text": "q"})
            s1.append(m.chat_id, {"type": "iteration_start", "step_id": "r1:1:0"})
            s1.append(m.chat_id, {"type": "content", "chunk": "c", "step_id": "r1:1:0"})
            s1.append(m.chat_id, {"type": "iteration_start", "step_id": "r1:2:0"})
            s1.truncate_after(m.chat_id, "r1:2:0")
            del s1

            s2 = ChatStore(tmp)
            frames = list(s2.replay(m.chat_id))
            self.assertEqual(len(frames), 3)
            self.assertEqual(s2.last_seq(m.chat_id), 3)


if __name__ == "__main__":
    unittest.main()
