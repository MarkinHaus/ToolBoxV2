"""Tests for register_ui_tools (tasks 20, 21, 22).

Verifies that when a stream starts the bridge:
  - registers create_widget, update_widget, close_widget, set_var, get_var,
    define_template on the agent
  - calling each tool broadcasts the correct frame
  - var_set persists to chat_store meta
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest

from toolboxv2.mods.isaa.ui.chat_store import ChatStore
from toolboxv2.mods.isaa.ui.stream_bridge import (
    StreamBridge, BridgeBroadcaster, _register_ui_tools,
)


class FakeAmd:
    name = "fake"


class FakeAgent:
    """Minimal agent with add_tool + active_session, like FlowAgent surface."""
    def __init__(self):
        self.amd = FakeAmd()
        self.active_session = None
        self.tools = {}  # name -> {fn, desc, category}

    def add_tool(self, fn, name=None, description=None, category=None, **kw):
        self.tools[name or fn.__name__] = {
            "fn": fn, "description": description, "category": category,
        }


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class RegisterUiToolsRegistration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.broadcasts = []

        async def bcast(chat_id, frame):
            self.broadcasts.append((chat_id, frame))

        self.bridge = StreamBridge(
            isaa_mod=None, chat_store=self.store,
            broadcaster=BridgeBroadcaster(send=bcast),
        )
        self.agent = FakeAgent()

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_six_tools_registered(self):
        _register_ui_tools(self.agent, self.bridge, self.store)
        for name in ("create_widget", "update_widget", "close_widget",
                     "set_var", "get_var", "define_template"):
            self.assertIn(name, self.agent.tools, f"missing tool: {name}")

    def test_idempotent(self):
        _register_ui_tools(self.agent, self.bridge, self.store)
        first_create = self.agent.tools["create_widget"]["fn"]
        _register_ui_tools(self.agent, self.bridge, self.store)
        # Same function object — flag-guarded
        self.assertIs(self.agent.tools["create_widget"]["fn"], first_create)

    def test_skips_agent_without_add_tool(self):
        class NoTools:
            amd = FakeAmd()
        # Should not raise
        _register_ui_tools(NoTools(), self.bridge, self.store)

    def test_category_is_ui(self):
        _register_ui_tools(self.agent, self.bridge, self.store)
        for name in ("create_widget", "set_var"):
            self.assertEqual(self.agent.tools[name]["category"], ["ui"])


class RegisterUiToolsExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.tmp.name)
        self.meta = self.store.create(agent="fake", title="t")
        self.broadcasts = []

        async def bcast(chat_id, frame):
            self.broadcasts.append((chat_id, frame))

        self.bridge = StreamBridge(
            isaa_mod=None, chat_store=self.store,
            broadcaster=BridgeBroadcaster(send=bcast),
        )
        self.agent = FakeAgent()
        self.agent.active_session = self.meta.chat_id
        _register_ui_tools(self.agent, self.bridge, self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_widget_emits_frame_and_returns_id(self):
        async def go():
            return await self.agent.tools["create_widget"]["fn"](
                "vega", {"spec": {"data": {"values": []}}}, None
            )
        wid = _run(go())
        self.assertEqual(len(wid), 12)
        # Frame was persisted + broadcast
        frames = list(self.store.replay(self.meta.chat_id))
        widget_frames = [f for f in frames if f["type"] == "widget_create"]
        self.assertEqual(len(widget_frames), 1)
        self.assertEqual(widget_frames[0]["template"], "vega")
        self.assertEqual(widget_frames[0]["widget_id"], wid)

    def test_create_widget_no_active_session_returns_error(self):
        self.agent.active_session = None
        async def go():
            return await self.agent.tools["create_widget"]["fn"]("vega", {}, None)
        result = _run(go())
        self.assertTrue(str(result).startswith("error"))
        self.assertEqual(len(self.broadcasts), 0)

    def test_update_widget_emits_patch(self):
        async def go():
            return await self.agent.tools["update_widget"]["fn"]("wid1", {"title": "X"})
        ok = _run(go())
        self.assertTrue(ok)
        frames = list(self.store.replay(self.meta.chat_id))
        upd = [f for f in frames if f["type"] == "widget_update"]
        self.assertEqual(upd[0]["props_patch"], {"title": "X"})

    def test_close_widget_emits_close(self):
        _run(self.agent.tools["close_widget"]["fn"]("wid1"))
        frames = list(self.store.replay(self.meta.chat_id))
        close = [f for f in frames if f["type"] == "widget_close"]
        self.assertEqual(close[0]["widget_id"], "wid1")

    def test_set_var_persists_to_meta_and_broadcasts(self):
        async def go():
            return await self.agent.tools["set_var"]["fn"]("agent", "color", "blue")
        ok = _run(go())
        self.assertTrue(ok)
        # Persisted in meta
        m = self.store.get_meta(self.meta.chat_id)
        self.assertEqual(m.ui.get("vars_agent"), {"color": "blue"})
        # Broadcast frame
        frames = list(self.store.replay(self.meta.chat_id))
        var_frames = [f for f in frames if f["type"] == "var_set"]
        self.assertEqual(var_frames[0]["key"], "color")
        self.assertEqual(var_frames[0]["value"], "blue")

    def test_get_var_reads_from_meta(self):
        _run(self.agent.tools["set_var"]["fn"]("global", "lang", "de"))
        async def go():
            return await self.agent.tools["get_var"]["fn"]("global", "lang")
        v = _run(go())
        self.assertEqual(v, "de")

    def test_get_var_missing_returns_none(self):
        async def go():
            return await self.agent.tools["get_var"]["fn"]("agent", "nope")
        self.assertIsNone(_run(go()))

    def test_define_template_emits_register_frame(self):
        async def go():
            return await self.agent.tools["define_template"]["fn"](
                "Status Card", "html",
                {"type": "object", "properties": {"label": {"type": "string"}}},
                "(root, props, api) => { root.innerHTML = props.label; }",
            )
        tid = _run(go())
        self.assertEqual(tid, "status_card")
        frames = list(self.store.replay(self.meta.chat_id))
        reg = [f for f in frames if f["type"] == "template_register"]
        self.assertEqual(reg[0]["template_id"], "status_card")
        self.assertEqual(reg[0]["adapter"], "html")
        self.assertIn("root.innerHTML", reg[0]["render_js"])


if __name__ == "__main__":
    unittest.main()
