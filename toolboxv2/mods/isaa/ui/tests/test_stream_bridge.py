"""Tests for stream_bridge.StreamBridge with a fake ISAA + FlowAgent."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.ui.chat_store import ChatStore
from toolboxv2.mods.isaa.ui.stream_bridge import StreamBridge, BridgeBroadcaster, make_step_id


class FakeEngine:
    def __init__(self):
        self._session_last_run = {}


class FakeAgent:
    def __init__(self, name, chunks):
        self.name = name
        self.chunks = chunks
        self._engine = FakeEngine()
        self.paused = []
        self.cancelled = []

    def _get_execution_engine(self):
        return self._engine

    async def a_stream(self, query, session_id, **kw):
        # Set run_id mapping like the real engine would.
        self._engine._session_last_run[session_id] = "run_xyz"
        for c in self.chunks:
            yield c

    async def pause_execution(self, run_id):
        self.paused.append(run_id)
        return {"run_id": run_id}

    async def cancel_execution(self, run_id):
        self.cancelled.append(run_id)
        return True


class FakeIsaa:
    def __init__(self, agent):
        self._agent = agent

    async def get_agent(self, name):
        return self._agent


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class StreamBridgeStart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="alice")
        self.broadcasts = []

        async def bcast(chat_id, frame):
            self.broadcasts.append((chat_id, frame))

        self.bridge = StreamBridge(
            isaa_mod=FakeIsaa(FakeAgent("alice", [])),
            chat_store=self.store,
            broadcaster=BridgeBroadcaster(send=bcast),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_persists_user_msg(self):
        async def go():
            ok = await self.bridge.start(self.meta.chat_id, "alice", "hello", [])
            # wait briefly for the task to finish (empty chunks list → immediately done)
            rs = self.bridge._running.get(self.meta.chat_id)
            if rs and rs.task:
                try:
                    await asyncio.wait_for(rs.task, timeout=1.0)
                except asyncio.TimeoutError:
                    rs.task.cancel()
            return ok
        _run(go())
        frames = list(self.store.replay(self.meta.chat_id))
        self.assertTrue(any(f["type"] == "user_msg" and f["text"] == "hello" for f in frames))


class StreamBridgeStreaming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="alice")
        self.received = []

        async def bcast(chat_id, frame):
            self.received.append(frame)

        chunks = [
            {"type": "status", "status_msg": "Initializing", "agent": "alice"},
            {"type": "iteration_start", "iter": 1, "agent": "alice"},
            {"type": "content", "chunk": "Hello ", "iter": 1},
            {"type": "content", "chunk": "world", "iter": 1},
            {"type": "tool_start", "name": "vfs_read", "args": {"path": "/x"}, "id": "tc_1", "iter": 1},
            {"type": "tool_result", "name": "vfs_read", "id": "tc_1", "result": "data", "is_final": True, "iter": 1},
            {"type": "final_answer", "answer": "done", "iter": 1},
            {"type": "done", "success": True, "final_answer": "done"},
        ]
        self.agent = FakeAgent("alice", chunks)
        self.bridge = StreamBridge(
            isaa_mod=FakeIsaa(self.agent),
            chat_store=self.store,
            broadcaster=BridgeBroadcaster(send=bcast),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_persists_and_assigns_step_ids(self):
        async def go():
            await self.bridge.start(self.meta.chat_id, "alice", "test", [])
            rs = self.bridge._running.get(self.meta.chat_id)
            if rs and rs.task:
                await asyncio.wait_for(rs.task, timeout=2.0)
        _run(go())

        frames = list(self.store.replay(self.meta.chat_id))
        types = [f["type"] for f in frames]
        self.assertIn("user_msg", types)
        self.assertIn("content", types)
        self.assertIn("tool_start", types)
        self.assertIn("tool_result", types)
        self.assertIn("done", types)

        # Step ids assigned only to relevant chunk types
        content_frames = [f for f in frames if f["type"] == "content"]
        for f in content_frames:
            self.assertIn("step_id", f, f"content frame missing step_id: {f}")

        # Seqs strictly monotonic
        seqs = [f["seq"] for f in frames]
        self.assertEqual(seqs, sorted(seqs))


class StreamBridgeNoConcurrent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="alice")

        async def bcast(chat_id, frame): pass

        # Slow chunks via sleep — keep stream alive
        class SlowAgent(FakeAgent):
            async def a_stream(self, query, session_id, **kw):
                self._engine._session_last_run[session_id] = "run_slow"
                yield {"type": "status", "status_msg": "go"}
                await asyncio.sleep(0.5)
                yield {"type": "done", "success": True}

        self.bridge = StreamBridge(
            isaa_mod=FakeIsaa(SlowAgent("alice", [])),
            chat_store=self.store,
            broadcaster=BridgeBroadcaster(send=bcast),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_start_returns_false_while_running(self):
        async def go():
            ok1 = await self.bridge.start(self.meta.chat_id, "alice", "first", [])
            ok2 = await self.bridge.start(self.meta.chat_id, "alice", "second", [])
            return ok1, ok2

        ok1, ok2 = _run(go())
        self.assertTrue(ok1)
        self.assertFalse(ok2)


class StepIdHelper(unittest.TestCase):
    def test_make_step_id_format(self):
        self.assertEqual(make_step_id("r1", 5, 2), "r1:5:2")
        self.assertEqual(make_step_id("r1", 0), "r1:0:0")


if __name__ == "__main__":
    unittest.main()
