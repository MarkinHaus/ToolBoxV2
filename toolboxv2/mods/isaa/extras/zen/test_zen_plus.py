"""
Unit tests for zen_plus.py — Phase 3 stability suite.

Run:  python -m unittest test_zen_plus -v
"""

import asyncio
import json
import time
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.extras.zen.zen_plus import (
    # Utilities
    _short, _bar, _detect_lang, syntax_highlight,
    _hl_json, _hl_markdown, _hl_generic,
    # Enums / constants
    ViewMode, SYM, C, SUB_COLORS, AGENT_COLORS, _LABEL_LEN,
    GRAPH_NODE_SYM, GRAPH_NODE_COLOR,
    REL_PARENT, REL_RESOURCE, REL_TASK, REL_SIMILAR, REL_COLOR, REL_CHAR,
    # Graph
    GNode, GEdge, MiniGraph3D,
    GlobalGraph, GlobalNode, GlobalEdge,
    # Data models
    ToolEvent, IterationInfo, JobInfo,
    # Core
    AgentPane, ZenPlus,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _ft_text(ft: list[tuple[str, str]]) -> str:
    """Join FormattedText tuples into plain string."""
    return "".join(t for _, t in ft)


def _ft_has(ft: list[tuple[str, str]], needle: str) -> bool:
    return needle in _ft_text(ft)


def _make_pane(name="test", iterations=3, with_sub=False) -> AgentPane:
    """Create a populated AgentPane for testing."""
    p = AgentPane(name)
    for i in range(1, iterations + 1):
        base = {"agent": name, "iter": i, "max_iter": iterations,
                "tokens_used": i * 2000, "tokens_max": 20000}
        if i == 1:
            base["persona"] = "architect"
            base["skills"] = ["code", "test", "docker"]

        p.ingest({**base, "type": "reasoning",
                  "chunk": f"Deep thought about iteration {i}: analyzing the full "
                           f"system architecture and identifying bottlenecks in the "
                           f"data pipeline that affect performance under load."})

        p.ingest({**base, "type": "tool_start", "name": "vfs_read",
                  "args": json.dumps({"path": f"/src/mod_{i}.py", "encoding": "utf-8"})})
        p.ingest({**base, "type": "tool_result", "name": "vfs_read",
                  "result": json.dumps({"success": True, "content": f"# Module {i}\ndef run():\n    pass",
                                        "lines": 3, "file_type": "py"})})

        p.ingest({**base, "type": "tool_start", "name": "vfs_write",
                  "args": json.dumps({"path": f"/src/mod_{i}.py", "content": "updated"})})
        p.ingest({**base, "type": "tool_result", "name": "vfs_write",
                  "result": json.dumps({"success": True, "bytes_written": 42})})

        if with_sub and i >= 2:
            sub = {**base, "is_sub": True, "_sub_agent_id": "tester-sub"}
            p.ingest({**sub, "type": "tool_start", "name": "docker_run",
                      "args": json.dumps({"command": "pytest -v"})})
            p.ingest({**sub, "type": "tool_result", "name": "docker_run",
                      "result": json.dumps({"success": i % 2 == 0, "stdout": "tests done"})})
            p.ingest({**sub, "type": "reasoning",
                      "chunk": f"Sub-agent analysis for iteration {i}"})

        p.ingest({**base, "type": "content", "chunk": f"Completed step {i}.\n"})

    p.ingest({"agent": name, "type": "done", "success": True,
              "iter": iterations, "max_iter": iterations})
    return p


# ═══════════════════════════════════════════════════════════════
# 1. Utilities
# ═══════════════════════════════════════════════════════════════

class TestUtilities(unittest.TestCase):

    def test_short_within_limit(self):
        self.assertEqual(_short("hello", 10), "hello")

    def test_short_truncates(self):
        result = _short("a" * 50, 20)
        self.assertEqual(len(result), 22)  # 20 + ".."
        self.assertTrue(result.endswith(".."))

    def test_short_exact_boundary(self):
        self.assertEqual(_short("abc", 3), "abc")

    def test_bar_empty(self):
        result = _bar(0, 0, 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(result, SYM["bar_empty"] * 10)

    def test_bar_half(self):
        result = _bar(5, 10, 10)
        self.assertIn(SYM["bar_fill"], result)
        self.assertIn(SYM["bar_empty"], result)
        self.assertEqual(len(result), 10)

    def test_bar_full(self):
        result = _bar(10, 10, 10)
        self.assertEqual(result, SYM["bar_fill"] * 10)

    def test_bar_overflow(self):
        result = _bar(20, 10, 10)
        self.assertEqual(result, SYM["bar_fill"] * 10)


# ═══════════════════════════════════════════════════════════════
# 2. Syntax Highlighting
# ═══════════════════════════════════════════════════════════════

class TestSyntaxHighlighting(unittest.TestCase):

    def test_detect_json(self):
        self.assertEqual(_detect_lang('{"key": "val"}'), "json")
        self.assertEqual(_detect_lang('[1, 2, 3]'), "json")

    def test_detect_markdown(self):
        self.assertEqual(_detect_lang("# Title"), "markdown")
        self.assertEqual(_detect_lang("some **bold** text"), "markdown")

    def test_detect_text(self):
        self.assertEqual(_detect_lang("hello world"), "text")

    def test_detect_invalid_json_fallback(self):
        self.assertNotEqual(_detect_lang("{not json"), "json")

    def test_json_highlight_structure(self):
        ft = syntax_highlight('{"name": "test", "count": 42}')
        text = _ft_text(ft)
        self.assertIn("name", text)
        self.assertIn("test", text)
        # Keys should be cyan
        self.assertTrue(any("#67e8f9" in s for s, _ in ft))
        # String values green
        self.assertTrue(any("#4ade80" in s for s, _ in ft))

    def test_json_highlight_types(self):
        ft = syntax_highlight('{"b": true, "n": null, "i": 42}')
        text = _ft_text(ft)
        self.assertIn("true", text)
        self.assertIn("null", text)
        self.assertIn("42", text)

    def test_markdown_highlight(self):
        ft = syntax_highlight("# Header\n- item\n**bold**\n```code```")
        self.assertTrue(any("bold" in s for s, _ in ft))

    def test_generic_highlight(self):
        ft = syntax_highlight("key = value\nhost: localhost")
        text = _ft_text(ft)
        self.assertIn("key", text)
        self.assertIn("value", text)

    def test_empty_input(self):
        ft = syntax_highlight("")
        self.assertTrue(len(ft) > 0)
        self.assertIn("empty", _ft_text(ft))

    def test_explicit_lang(self):
        ft = syntax_highlight("not json", lang="json")
        self.assertTrue(len(ft) > 0)


# ═══════════════════════════════════════════════════════════════
# 3. MiniGraph3D
# ═══════════════════════════════════════════════════════════════

class TestMiniGraph3D(unittest.TestCase):

    def setUp(self):
        self.g = MiniGraph3D()
        self.g.add_node("i1", "iter 1", "iteration")
        self.g.add_node("t1", "vfs_read", "tool")
        self.g.add_node("th1", "Analyzing the project", "thought")
        self.g.add_node("s1", "sub-tester", "sub_agent", is_sub=True, sub_color_idx=0)
        self.g.add_edge("i1", "t1")
        self.g.add_edge("i1", "th1")
        self.g.add_edge("t1", "s1")

    def test_add_node(self):
        self.assertEqual(len(self.g.nodes), 4)
        self.assertIn("i1", self.g.nodes)
        self.assertEqual(self.g.nodes["i1"].kind, "iteration")

    def test_add_node_idempotent(self):
        self.g.add_node("i1", "different label", "agent")
        self.assertEqual(len(self.g.nodes), 4)
        self.assertEqual(self.g.nodes["i1"].label, "iter 1")  # unchanged

    def test_add_edge(self):
        self.assertEqual(len(self.g.edges), 3)

    def test_add_edge_idempotent(self):
        self.g.add_edge("i1", "t1")
        self.assertEqual(len(self.g.edges), 3)

    def test_add_edge_missing_node(self):
        self.g.add_edge("i1", "nonexistent")
        self.assertEqual(len(self.g.edges), 3)

    def test_current_tracks_latest(self):
        self.assertEqual(self.g._current, "s1")
        self.g.add_node("new", "new node", "tool")
        self.assertEqual(self.g._current, "new")

    def test_select_by_index(self):
        self.g.select(0)
        self.assertEqual(self.g._selected, "i1")
        self.g.select(1)
        self.assertEqual(self.g._selected, "t1")

    def test_select_by_id(self):
        self.g.select_by_id("th1")
        self.assertEqual(self.g._selected, "th1")
        self.assertEqual(self.g.selected_node.label, "Analyzing the project")

    def test_select_invalid(self):
        self.g.select(99)
        # Should not crash, selection unchanged

    def test_selected_edges(self):
        self.g.select_by_id("i1")
        connected = self.g._selected_edges()
        self.assertIn("t1", connected)
        self.assertIn("th1", connected)
        self.assertNotIn("s1", connected)  # not directly connected

    def test_navigate_edge_child(self):
        self.g.select_by_id("i1")
        self.g.navigate_edge(+1)
        self.assertIn(self.g._selected, ("t1", "th1"))

    def test_navigate_edge_parent(self):
        self.g.select_by_id("t1")
        self.g.navigate_edge(-1)
        self.assertEqual(self.g._selected, "i1")

    def test_navigate_edge_no_target(self):
        self.g.select_by_id("th1")
        old = self.g._selected
        self.g.navigate_edge(+1)  # th1 has no children, fallback to any connected
        self.assertIsNotNone(self.g._selected)

    def test_toggle_detail(self):
        self.g.select_by_id("t1")
        self.assertIsNone(self.g._detail_node)
        self.g.toggle_detail()
        self.assertEqual(self.g._detail_node, "t1")
        self.g.toggle_detail()
        self.assertIsNone(self.g._detail_node)

    def test_update_status(self):
        self.g.update_status("t1", "done")
        self.assertEqual(self.g.nodes["t1"].status, "done")

    def test_update_status_missing(self):
        self.g.update_status("nonexistent", "done")  # no crash

    def test_tick_physics(self):
        old_az = self.g.azimuth
        self.g.tick()
        self.assertGreater(self.g.azimuth, old_az)
        self.assertEqual(self.g._frame, 1)

    def test_render_output_format(self):
        ft = self.g.render(60, 15)
        self.assertIsInstance(ft, list)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in ft))

    def test_render_has_stats(self):
        ft = self.g.render(60, 15)
        self.assertTrue(_ft_has(ft, "nodes"))
        self.assertTrue(_ft_has(ft, "edges"))

    def test_render_has_nav_hints(self):
        ft = self.g.render(60, 15)
        self.assertTrue(_ft_has(ft, "↑↓"))
        self.assertTrue(_ft_has(ft, "Enter"))

    def test_render_shows_current(self):
        ft = self.g.render(60, 15)
        self.assertTrue(_ft_has(ft, "▶"))

    def test_render_with_detail_open(self):
        self.g.select_by_id("i1")
        self.g.toggle_detail()
        ft = self.g.render(80, 20)
        self.assertTrue(_ft_has(ft, "id: i1"))

    def test_render_edge_highlight(self):
        self.g.select_by_id("i1")
        ft = self.g.render(80, 20)
        self.assertTrue(_ft_has(ft, "sel:"))

    def test_render_empty_graph(self):
        g = MiniGraph3D()
        ft = g.render(60, 15)
        self.assertTrue(_ft_has(ft, "no nodes"))

    def test_auto_rotation_continuous(self):
        for _ in range(10):
            self.g.render(40, 10)
        self.assertGreater(self.g.azimuth, 0.1)

    def test_sub_agent_node(self):
        self.assertTrue(self.g.nodes["s1"].is_sub)
        self.assertEqual(self.g.nodes["s1"].sub_color_idx, 0)


# ═══════════════════════════════════════════════════════════════
# 4. GlobalGraph
# ═══════════════════════════════════════════════════════════════

class TestGlobalGraph(unittest.TestCase):

    def _make_multi_panes(self):
        panes = {}
        for name in ["agent_a", "agent_b", "agent_c"]:
            p = AgentPane(name)
            for i in range(1, 4):
                base = {"agent": name, "iter": i, "max_iter": 3}
                # Shared resource
                p.ingest({**base, "type": "tool_start", "name": "vfs_read",
                          "args": json.dumps({"path": "/shared/config.py"})})
                p.ingest({**base, "type": "tool_result", "name": "vfs_read",
                          "result": json.dumps({"success": True})})
                # Unique tool
                p.ingest({**base, "type": "tool_start", "name": f"tool_{name}",
                          "args": json.dumps({"path": f"/src/{name}.py"})})
                p.ingest({**base, "type": "tool_result", "name": f"tool_{name}",
                          "result": json.dumps({"success": True})})
            panes[name] = p
        # Sub-agent on agent_b
        panes["agent_b"].ingest({"agent": "agent_b", "type": "tool_start",
                                 "name": "docker_run", "args": "{}",
                                 "iter": 2, "max_iter": 3,
                                 "is_sub": True, "_sub_agent_id": "sub-x"})
        panes["agent_b"].ingest({"agent": "agent_b", "type": "tool_result",
                                 "name": "docker_run",
                                 "result": json.dumps({"success": True}),
                                 "iter": 2, "max_iter": 3,
                                 "is_sub": True, "_sub_agent_id": "sub-x"})
        return panes

    def test_rebuild_creates_hubs(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        hubs = [n for n in gg.nodes.values() if n.kind == "agent_hub"]
        self.assertEqual(len(hubs), 3)

    def test_rebuild_creates_sub_hubs(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        sub_hubs = [n for n in gg.nodes.values() if n.kind == "sub_hub"]
        self.assertGreaterEqual(len(sub_hubs), 1)

    def test_rebuild_creates_actions(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        actions = [n for n in gg.nodes.values() if n.kind == "action"]
        # 3 agents × 3 last actions = 9
        self.assertGreaterEqual(len(actions), 9)

    def test_rebuild_marks_newest(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        newest = [n for n in gg.nodes.values() if n.newest]
        self.assertEqual(len(newest), 3)  # 1 per agent

    def test_resource_edges(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        resource_edges = [e for e in gg.edges if e.rel == REL_RESOURCE]
        self.assertGreater(len(resource_edges), 0)

    def test_similar_edges(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        similar_edges = [e for e in gg.edges if e.rel == REL_SIMILAR]
        self.assertGreater(len(similar_edges), 0)

    def test_parent_edges(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        parent_edges = [e for e in gg.edges if e.rel == REL_PARENT]
        self.assertGreater(len(parent_edges), 0)

    def test_original_nid_for_jump(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        actions = [n for n in gg.nodes.values() if n.kind == "action"]
        with_jump = [n for n in actions if n.original_nid]
        self.assertEqual(len(with_jump), len(actions))

    def test_navigation(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        gg.select(0)
        self.assertIsNotNone(gg._selected)
        gg.navigate_edge(+1)
        self.assertIsNotNone(gg._selected)
        gg.navigate_edge(-1)
        self.assertIsNotNone(gg._selected)

    def test_toggle_detail(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        gg.select(0)
        gg.toggle_detail()
        self.assertIsNotNone(gg._detail_node)
        gg.toggle_detail()
        self.assertIsNone(gg._detail_node)

    def test_render(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        ft = gg.render(100, 25)
        self.assertTrue(len(ft) > 0)
        self.assertTrue(_ft_has(ft, "resource"))
        self.assertTrue(_ft_has(ft, "↑↓"))

    def test_render_with_detail(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        gg.select(3)
        gg.toggle_detail()
        ft = gg.render(100, 25)
        self.assertTrue(_ft_has(ft, "agent:"))

    def test_render_empty(self):
        gg = GlobalGraph()
        ft = gg.render(60, 15)
        self.assertTrue(_ft_has(ft, "no agents"))

    def test_rebuild_preserves_selection(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        gg.select(2)
        old_sel = gg._selected
        gg.rebuild(panes)
        if old_sel in gg.nodes:
            self.assertEqual(gg._selected, old_sel)

    def test_physics_tick(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        old_az = gg.azimuth
        gg.tick()
        self.assertGreater(gg.azimuth, old_az)

    def test_connected(self):
        gg = GlobalGraph()
        panes = self._make_multi_panes()
        gg.rebuild(panes)
        gg.select(0)
        connected = gg._connected()
        self.assertIsInstance(connected, set)


# ═══════════════════════════════════════════════════════════════
# 5. AgentPane
# ═══════════════════════════════════════════════════════════════

class TestAgentPane(unittest.TestCase):

    def test_ingest_reasoning_no_truncation(self):
        p = AgentPane("test")
        long = "A" * 500
        p.ingest({"agent": "test", "type": "reasoning", "chunk": long,
                  "iter": 1, "max_iter": 3})
        self.assertEqual(len(p.thoughts[0]), 500)

    def test_ingest_tool_lifecycle(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "tool_start", "name": "vfs_read",
                  "args": '{"path": "/x"}', "iter": 1, "max_iter": 3})
        p.ingest({"agent": "test", "type": "tool_result", "name": "vfs_read",
                  "result": '{"success": true}', "iter": 1, "max_iter": 3})
        self.assertEqual(len(p.tool_history), 1)
        self.assertTrue(p.tool_history[0].success)
        self.assertEqual(p.tool_history[0].args_parsed, {"path": "/x"})

    def test_ingest_tool_failure(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "tool_start", "name": "docker",
                  "args": "{}", "iter": 1, "max_iter": 3})
        p.ingest({"agent": "test", "type": "tool_result", "name": "docker",
                  "result": '{"success": false, "error": "timeout"}',
                  "iter": 1, "max_iter": 3})
        self.assertFalse(p.tool_history[0].success)

    def test_ingest_content_multiline(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "content", "chunk": "line1\nline2\n",
                  "iter": 1, "max_iter": 1})
        self.assertIn("line1", p.content_lines)
        self.assertIn("line2", p.content_lines)

    def test_ingest_done(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "done", "success": True,
                  "iter": 1, "max_iter": 1})
        self.assertEqual(p.phase, "done")

    def test_ingest_error(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "error", "iter": 1, "max_iter": 1})
        self.assertEqual(p.phase, "error")

    def test_ingest_sub_agent_tracking(self):
        p = AgentPane("main")
        p.ingest({"agent": "main", "type": "tool_start", "name": "x",
                  "args": "{}", "iter": 1, "max_iter": 3,
                  "is_sub": True, "_sub_agent_id": "helper"})
        self.assertIn("helper", p.sub_agents)
        self.assertTrue(p.tool_history[0].is_sub)
        self.assertEqual(p.tool_history[0].sub_agent, "helper")

    def test_iteration_tracking(self):
        p = _make_pane("test", 3)
        self.assertIn(1, p.iterations)
        self.assertIn(2, p.iterations)
        self.assertIn(3, p.iterations)
        it1 = p.iterations[1]
        self.assertGreater(len(it1.tools), 0)
        self.assertGreater(len(it1.thoughts), 0)

    def test_graph_nodes_created(self):
        p = _make_pane("test", 2)
        self.assertGreater(len(p.graph.nodes), 0)
        self.assertIn("iter_1", p.graph.nodes)
        self.assertIn("iter_2", p.graph.nodes)

    def test_graph_edges_created(self):
        p = _make_pane("test", 2)
        self.assertGreater(len(p.graph.edges), 0)

    def test_graph_current_tracks_latest(self):
        p = _make_pane("test", 3)
        self.assertIsNotNone(p.graph._current)

    def test_persona_and_skills(self):
        p = _make_pane("test", 1)
        self.assertEqual(p.persona, "architect")
        self.assertEqual(p.skills, ["code", "test", "docker"])

    def test_tokens_tracking(self):
        p = _make_pane("test", 3)
        self.assertGreater(p.tokens_used, 0)
        self.assertGreater(p.tokens_max, 0)

    # ── Render: compact ──

    def test_render_compact_returns_tuples(self):
        p = _make_pane("test", 2)
        ft = p.render_compact(80, 12)
        self.assertIsInstance(ft, list)
        self.assertTrue(all(isinstance(t, tuple) for t in ft))

    def test_render_compact_shows_name(self):
        p = _make_pane("myagent", 1)
        ft = p.render_compact(80, 12)
        self.assertTrue(_ft_has(ft, "myagent"))

    def test_render_compact_shows_progress(self):
        p = _make_pane("test", 3)
        ft = p.render_compact(80, 12)
        self.assertTrue(_ft_has(ft, "3/3"))

    def test_render_compact_sub_indicator(self):
        p = _make_pane("test", 3, with_sub=True)
        ft = p.render_compact(80, 12)
        self.assertTrue(_ft_has(ft, "✦"))

    # ── Render: focus ──

    def test_render_focus_shows_shortcuts(self):
        p = _make_pane("test", 2)
        ft = p.render_focus(80, 24)
        text = _ft_text(ft)
        self.assertIn("[g]raph", text)
        self.assertIn("[t]ools", text)
        self.assertIn("[i]terations", text)
        self.assertIn("[h]thoughts", text)

    def test_render_focus_shows_persona(self):
        p = _make_pane("test", 1)
        ft = p.render_focus(80, 24)
        self.assertTrue(_ft_has(ft, "architect"))

    def test_render_focus_shows_done(self):
        p = _make_pane("test", 1)
        ft = p.render_focus(80, 24)
        self.assertTrue(_ft_has(ft, "done"))

    # ── Render: detail views ──

    def test_render_detail_tools_full_expansion(self):
        p = _make_pane("test", 2)
        p.selected_item = 0
        ft = p.render_detail(80, 30, "tools")
        text = _ft_text(ft)
        self.assertIn("args", text)
        self.assertIn("result", text)

    def test_render_detail_tools_syntax_highlight(self):
        p = _make_pane("test", 1)
        p.selected_item = 0
        ft = p.render_detail(80, 30, "tools")
        # JSON keys should have cyan coloring
        self.assertTrue(any("#67e8f9" in s for s, t in ft if "path" in t or "success" in t
                            or "#67e8f9" in s))

    def test_render_detail_iterations(self):
        p = _make_pane("test", 3)
        p.selected_item = 0
        ft = p.render_detail(80, 30, "iterations")
        text = _ft_text(ft)
        self.assertIn("iter 1", text)

    def test_render_detail_iterations_shows_tools(self):
        p = _make_pane("test", 2)
        p.selected_item = 0
        ft = p.render_detail(80, 30, "iterations")
        self.assertTrue(_ft_has(ft, "vfs_read"))

    def test_render_detail_thoughts_full(self):
        p = _make_pane("test", 1)
        p.selected_item = 0
        ft = p.render_detail(80, 30, "thoughts")
        text = _ft_text(ft)
        # Should contain substantial text from the thought, not truncated
        self.assertIn("analyzing", text)

    def test_render_detail_graph(self):
        p = _make_pane("test", 2)
        p.selected_item = 0
        ft = p.render_detail(80, 20, "graph")
        self.assertTrue(_ft_has(ft, "nodes"))

    def test_detail_item_count(self):
        p = _make_pane("test", 3)
        self.assertEqual(p._detail_item_count("tools"), len(p.tool_history))
        self.assertEqual(p._detail_item_count("thoughts"), len(p.thoughts))
        self.assertEqual(p._detail_item_count("iterations"), len(p.iterations))
        self.assertEqual(p._detail_item_count("graph"), len(p.graph.nodes))


# ═══════════════════════════════════════════════════════════════
# 6. ZenPlus Singleton + Integration
# ═══════════════════════════════════════════════════════════════

class TestZenPlus(unittest.TestCase):

    def setUp(self):
        ZenPlus.reset()

    def tearDown(self):
        ZenPlus.reset()

    def test_singleton(self):
        a = ZenPlus.get()
        b = ZenPlus.get()
        self.assertIs(a, b)

    def test_reset(self):
        a = ZenPlus.get()
        ZenPlus.reset()
        b = ZenPlus.get()
        self.assertIsNot(a, b)

    def test_initial_state(self):
        zp = ZenPlus.get()
        self.assertFalse(zp.active)
        self.assertEqual(zp._view, ViewMode.GRID)
        self.assertEqual(len(zp._panes), 0)
        self.assertEqual(len(zp._jobs), 0)

    def test_feed_chunk(self):
        zp = ZenPlus.get()
        zp.feed_chunk({"agent": "a", "type": "content", "chunk": "hi"})
        self.assertFalse(zp._queue.empty())

    def test_inject_job(self):
        zp = ZenPlus.get()
        zp.inject_job("j1", "coder", "fix bug", "running", kind="job")
        self.assertIn("j1", zp._jobs)
        self.assertEqual(zp._jobs["j1"].status, "running")
        self.assertEqual(zp._jobs["j1"].kind, "job")

    def test_update_job(self):
        zp = ZenPlus.get()
        zp.inject_job("j1", "coder", "fix", "running")
        zp.update_job("j1", "completed")
        self.assertEqual(zp._jobs["j1"].status, "completed")

    def test_remove_job(self):
        zp = ZenPlus.get()
        zp.inject_job("j1", "coder", "fix", "running")
        zp.remove_job("j1")
        self.assertNotIn("j1", zp._jobs)

    def test_clear_panes(self):
        zp = ZenPlus.get()
        zp._panes["a"] = AgentPane("a")
        zp.inject_job("j1", "x", "y", "running")
        zp.clear_panes()
        self.assertEqual(len(zp._panes), 0)
        self.assertEqual(len(zp._jobs), 0)
        self.assertEqual(zp._view, ViewMode.GRID)

    def test_get_pane_creates_on_demand(self):
        zp = ZenPlus.get()
        pane = zp._get_pane("new_agent")
        self.assertIn("new_agent", zp._panes)
        self.assertEqual(zp._focus, "new_agent")

    def test_ordered_names(self):
        zp = ZenPlus.get()
        zp._get_pane("a")
        zp._get_pane("b")
        zp._get_pane("c")
        self.assertEqual(zp._ordered_names(), ["a", "b", "c"])

    def test_signal_stream_done(self):
        zp = ZenPlus.get()
        zp.signal_stream_done()
        self.assertTrue(zp._stream_done)

    def test_consume_processes_chunks(self):
        async def _test():
            zp = ZenPlus.get()
            zp._running = True
            zp.feed_chunk({"agent": "a", "type": "content", "chunk": "hello\n",
                           "iter": 1, "max_iter": 3})
            # Manually consume one
            chunk = await zp._queue.get()
            agent = chunk.get("agent", "default")
            zp._get_pane(agent).ingest(chunk)
            self.assertIn("a", zp._panes)
            self.assertIn("hello", zp._panes["a"].content_lines[0])
            zp._running = False
        asyncio.run(_test())

    def test_global_graph_rebuild_on_consume(self):
        async def _test():
            zp = ZenPlus.get()
            # Populate panes
            for name in ["x", "y"]:
                p = _make_pane(name, 2)
                zp._panes[name] = p
            zp._global_graph.rebuild(zp._panes)
            self.assertGreater(len(zp._global_graph.nodes), 0)
        asyncio.run(_test())

    # ── Render methods ──

    def test_render_title(self):
        zp = ZenPlus.get()
        ft = zp._render_title()
        self.assertTrue(_ft_has(ft, "ZEN+"))

    def test_render_title_grid_hints(self):
        zp = ZenPlus.get()
        zp._view = ViewMode.GRID
        ft = zp._render_title()
        self.assertTrue(_ft_has(ft, "G=global"))

    def test_render_title_focus_hints(self):
        zp = ZenPlus.get()
        zp._view = ViewMode.FOCUS
        ft = zp._render_title()
        self.assertTrue(_ft_has(ft, "g=graph"))

    def test_render_title_detail_graph_hints(self):
        zp = ZenPlus.get()
        zp._view = ViewMode.DETAIL
        zp._detail_type = "graph"
        ft = zp._render_title()
        self.assertTrue(_ft_has(ft, "edges"))

    def test_render_title_detail_global_hints(self):
        zp = ZenPlus.get()
        zp._view = ViewMode.DETAIL
        zp._detail_type = "global"
        ft = zp._render_title()
        self.assertTrue(_ft_has(ft, "edges"))

    def test_render_jobs_empty(self):
        zp = ZenPlus.get()
        ft = zp._render_jobs()
        self.assertTrue(len(ft) > 0)

    def test_render_jobs_with_jobs(self):
        zp = ZenPlus.get()
        zp.inject_job("j1", "coder", "fix bug", "running", kind="job")
        zp.inject_job("bg1", "deploy", "ship", "running", kind="bg")
        ft = zp._render_jobs()
        self.assertTrue(_ft_has(ft, "Jobs"))
        self.assertTrue(_ft_has(ft, "coder"))

    def test_render_status(self):
        zp = ZenPlus.get()
        zp._panes["a"] = _make_pane("a", 1)
        zp._panes["b"] = _make_pane("b", 1)
        ft = zp._render_status()
        text = _ft_text(ft)
        self.assertIn("a", text)
        self.assertIn("b", text)

    def test_render_global_detail(self):
        zp = ZenPlus.get()
        for name in ["a", "b"]:
            zp._panes[name] = _make_pane(name, 2)
        zp._global_graph.rebuild(zp._panes)
        ft = zp._render_global_detail(100, 30)
        self.assertTrue(_ft_has(ft, "GLOBAL"))

    def test_render_grid_empty(self):
        zp = ZenPlus.get()
        ft = zp._render_grid(80, 20)
        self.assertTrue(_ft_has(ft, "waiting"))

    def test_render_grid_with_panes(self):
        zp = ZenPlus.get()
        zp._panes["a"] = _make_pane("a", 1)
        zp._panes["b"] = _make_pane("b", 1)
        ft = zp._render_grid(80, 20)
        text = _ft_text(ft)
        self.assertIn("a", text)
        self.assertIn("b", text)

    # ── Job system ──

    def test_job_kinds(self):
        zp = ZenPlus.get()
        for kind in ("job", "bg", "delegate"):
            zp.inject_job(f"t_{kind}", "agent", "query", "running", kind=kind)
        self.assertEqual(len(zp._jobs), 3)
        self.assertEqual(zp._jobs["t_job"].kind, "job")
        self.assertEqual(zp._jobs["t_bg"].kind, "bg")
        self.assertEqual(zp._jobs["t_delegate"].kind, "delegate")

    def test_jobs_height(self):
        zp = ZenPlus.get()
        self.assertEqual(zp._jobs_height, 1)  # empty
        zp.inject_job("j1", "a", "q", "running")
        self.assertEqual(zp._jobs_height, 2)  # 1 + header


# ═══════════════════════════════════════════════════════════════
# 7. Data Models
# ═══════════════════════════════════════════════════════════════

class TestDataModels(unittest.TestCase):

    def test_tool_event_defaults(self):
        ev = ToolEvent(name="test")
        self.assertEqual(ev.name, "test")
        self.assertTrue(ev.success)
        self.assertFalse(ev.is_sub)
        self.assertEqual(ev.elapsed, 0.0)

    def test_iteration_info(self):
        it = IterationInfo(number=1)
        self.assertEqual(it.number, 1)
        self.assertEqual(len(it.tools), 0)
        self.assertEqual(len(it.thoughts), 0)

    def test_job_info(self):
        j = JobInfo(task_id="t1", agent_name="a", query="fix")
        self.assertEqual(j.status, "running")
        self.assertEqual(j.kind, "job")

    def test_view_mode_enum(self):
        self.assertEqual(ViewMode.GRID.value, "grid")
        self.assertEqual(ViewMode.FOCUS.value, "focus")
        self.assertEqual(ViewMode.DETAIL.value, "detail")


# ═══════════════════════════════════════════════════════════════
# 8. Constants integrity
# ═══════════════════════════════════════════════════════════════

class TestConstants(unittest.TestCase):

    def test_sym_keys(self):
        for key in ("agent", "tool", "think", "ok", "fail", "done", "star", "iter"):
            self.assertIn(key, SYM)

    def test_colors_valid_hex(self):
        import re
        for name, val in C.items():
            self.assertTrue(re.match(r'^#[0-9a-fA-F]{6}$', val),
                            f"Invalid color {name}: {val}")

    def test_sub_colors_non_empty(self):
        self.assertGreaterEqual(len(SUB_COLORS), 4)

    def test_agent_colors_non_empty(self):
        self.assertGreaterEqual(len(AGENT_COLORS), 4)

    def test_label_lengths(self):
        self.assertGreater(_LABEL_LEN["thought"], _LABEL_LEN["tool"])
        self.assertIn("iteration", _LABEL_LEN)

    def test_rel_types(self):
        self.assertEqual(REL_PARENT, "parent")
        self.assertEqual(REL_RESOURCE, "resource")
        self.assertIn(REL_PARENT, REL_COLOR)
        self.assertIn(REL_RESOURCE, REL_CHAR)

    def test_graph_node_syms(self):
        for kind in ("agent", "tool", "thought", "sub_agent", "iteration"):
            self.assertIn(kind, GRAPH_NODE_SYM)
            self.assertIn(kind, GRAPH_NODE_COLOR)


# ═══════════════════════════════════════════════════════════════
# 9. Edge cases & robustness
# ═══════════════════════════════════════════════════════════════

class TestRobustness(unittest.TestCase):

    def test_ingest_empty_chunk(self):
        p = AgentPane("test")
        p.ingest({})  # no crash

    def test_ingest_unknown_type(self):
        p = AgentPane("test")
        p.ingest({"type": "unknown_xyz", "agent": "test"})  # no crash

    def test_ingest_malformed_json_args(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "tool_start",
                  "name": "x", "args": "not{json",
                  "iter": 1, "max_iter": 1})
        self.assertEqual(len(p.tool_history), 1)
        self.assertIsNone(p.tool_history[0].args_parsed)

    def test_ingest_malformed_json_result(self):
        p = AgentPane("test")
        p.ingest({"agent": "test", "type": "tool_start", "name": "x",
                  "args": "{}", "iter": 1, "max_iter": 1})
        p.ingest({"agent": "test", "type": "tool_result", "name": "x",
                  "result": "plain text error", "iter": 1, "max_iter": 1})
        self.assertEqual(p.tool_history[0].result_raw, "plain text error")

    def test_render_compact_tiny(self):
        p = _make_pane("test", 1)
        ft = p.render_compact(20, 4)
        self.assertTrue(len(ft) > 0)

    def test_render_focus_tiny(self):
        p = _make_pane("test", 1)
        ft = p.render_focus(30, 8)
        self.assertTrue(len(ft) > 0)

    def test_render_detail_empty_pane(self):
        p = AgentPane("empty")
        ft = p.render_detail(80, 20, "tools")
        self.assertTrue(_ft_has(ft, "no tools"))

    def test_render_detail_empty_thoughts(self):
        p = AgentPane("empty")
        ft = p.render_detail(80, 20, "thoughts")
        self.assertTrue(_ft_has(ft, "no thoughts"))

    def test_render_detail_empty_iterations(self):
        p = AgentPane("empty")
        ft = p.render_detail(80, 20, "iterations")
        self.assertTrue(_ft_has(ft, "no iterations"))

    def test_render_graph_empty(self):
        p = AgentPane("empty")
        ft = p.render_detail(80, 20, "graph")
        self.assertTrue(_ft_has(ft, "no nodes"))

    def test_syntax_highlight_huge(self):
        big = json.dumps({"k" + str(i): "v" * 100 for i in range(50)})
        ft = syntax_highlight(big)
        self.assertTrue(len(ft) > 0)

    def test_global_graph_single_agent(self):
        gg = GlobalGraph()
        gg.rebuild({"solo": _make_pane("solo", 2)})
        self.assertGreater(len(gg.nodes), 0)
        # No cross-agent edges with single agent
        cross = [e for e in gg.edges if e.rel != REL_PARENT]
        self.assertEqual(len(cross), 0)

    def test_mini_graph_many_nodes(self):
        g = MiniGraph3D()
        for i in range(50):
            g.add_node(f"n{i}", f"node {i}", "tool")
            if i > 0:
                g.add_edge(f"n{i-1}", f"n{i}")
        ft = g.render(100, 30)
        self.assertTrue(len(ft) > 0)

    def test_concurrent_feed(self):
        """Simulate rapid chunk feeding."""
        zp = ZenPlus.get()
        for i in range(100):
            zp.feed_chunk({"agent": f"a{i % 3}", "type": "content",
                           "chunk": f"msg {i}\n", "iter": 1, "max_iter": 5})
        self.assertEqual(zp._queue.qsize(), 100)
        ZenPlus.reset()


if __name__ == "__main__":
    unittest.main()
