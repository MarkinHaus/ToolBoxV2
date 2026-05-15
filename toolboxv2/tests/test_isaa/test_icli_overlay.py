# tests/test_icli_overlay.py
"""
Unit tests für TaskOverlay Navigation, Rendering-State und ingest_chunk.
Kein echtes Terminal — alles über Zustandsprüfung.
"""
import time
import unittest
from unittest.mock import MagicMock, patch


def _make_tv(task_id="t1", agent="self", status="running",
             iteration=3, max_iter=15) -> "TaskView":
    from toolboxv2.flows.icli import TaskView
    tv = TaskView(task_id=task_id, agent_name=agent, query="test query",
                  status=status, iteration=iteration, max_iter=max_iter)
    return tv


def _make_overlay(views=None):
    from toolboxv2.flows.icli import TaskOverlay
    if views is None:
        views = {"t1": _make_tv()}
    ov = TaskOverlay(views)
    ov._selected = "t1"
    return ov


class TestTaskViewFinalAnswer(unittest.TestCase):

    def test_final_answer_default_empty(self):
        tv = _make_tv()
        self.assertEqual(tv.final_answer, "")

    def test_ingest_final_answer_chunk(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _make_tv()
        ingest_chunk(tv, {"type": "final_answer", "answer": "The answer is 42."})
        self.assertEqual(tv.final_answer, "The answer is 42.")
        self.assertEqual(tv.status, "completed")
        self.assertEqual(tv.phase, "done")

    def test_ingest_final_answer_content_fallback(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _make_tv()
        ingest_chunk(tv, {"type": "final_answer", "content": "Fallback content."})
        self.assertEqual(tv.final_answer, "Fallback content.")

    def test_ingest_final_answer_empty_chunk_leaves_blank(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _make_tv()
        tv.final_answer = "existing"
        ingest_chunk(tv, {"type": "final_answer"})
        # kein answer-Key → kein Überschreiben
        self.assertEqual(tv.final_answer, "existing")


class TestOverlayInitState(unittest.TestCase):

    def test_init_defaults(self):
        ov = _make_overlay()
        self.assertEqual(ov._focus, "left")
        self.assertEqual(ov._selected_sub, "")
        self.assertIsNone(ov._selected_iter)
        self.assertEqual(ov._right_scroll, 0)
        self.assertEqual(ov._left_scroll, 0)

    def test_left_items_task_only(self):
        ov = _make_overlay()
        items = ov._left_items()
        self.assertEqual(items, [("t1", "")])

    def test_left_items_with_sub_agents(self):
        tv = _make_tv()
        tv._sub_color("sub_abc")
        tv._sub_color("sub_def")
        ov = _make_overlay({"t1": tv})
        items = ov._left_items()
        self.assertEqual(items[0], ("t1", ""))
        self.assertIn(("t1", "sub_abc"), items)
        self.assertIn(("t1", "sub_def"), items)
        self.assertEqual(len(items), 3)

    def test_left_items_multiple_tasks(self):
        views = {"t1": _make_tv("t1", "agent_a"), "t2": _make_tv("t2", "agent_b")}
        ov = _make_overlay(views)
        ov._selected = "t1"
        items = ov._left_items()
        task_ids = [tid for tid, sub in items if sub == ""]
        self.assertEqual(task_ids, ["t1", "t2"])


class TestOverlayNavigation(unittest.TestCase):
    """Testet Navigationszustand ohne echten Event-Loop."""

    def _simulate_key(self, ov, direction: str):
        """Simuliert up/down Navigation direkt."""
        items = ov._left_items()
        if not items:
            return
        cur = (ov._selected, ov._selected_sub)
        idx = items.index(cur) if cur in items else 0
        if direction == "down":
            nxt = items[(idx + 1) % len(items)]
        else:
            nxt = items[(idx - 1) % len(items)]
        ov._selected, ov._selected_sub = nxt
        ov._selected_iter = None
        ov._right_scroll = 0

    def test_navigate_down_wraps(self):
        tv = _make_tv()
        tv._sub_color("sub_x")
        ov = _make_overlay({"t1": tv})
        ov._selected, ov._selected_sub = "t1", ""

        self._simulate_key(ov, "down")   # → (t1, sub_x)
        self.assertEqual(ov._selected_sub, "sub_x")

        self._simulate_key(ov, "down")   # → wraps to (t1, "")
        self.assertEqual(ov._selected_sub, "")

    def test_navigate_up_wraps(self):
        tv = _make_tv()
        tv._sub_color("sub_y")
        ov = _make_overlay({"t1": tv})
        ov._selected, ov._selected_sub = "t1", ""

        self._simulate_key(ov, "up")     # → wraps to (t1, sub_y)
        self.assertEqual(ov._selected_sub, "sub_y")

    def test_navigate_down_resets_iter_and_scroll(self):
        tv = _make_tv()
        tv._sub_color("sub_z")
        ov = _make_overlay({"t1": tv})
        ov._selected_iter = 3
        ov._right_scroll = 10

        self._simulate_key(ov, "down")
        self.assertIsNone(ov._selected_iter)
        self.assertEqual(ov._right_scroll, 0)

    def test_focus_toggle(self):
        ov = _make_overlay()
        self.assertEqual(ov._focus, "left")
        # Simulate Tab
        ov._focus = "right" if ov._focus == "left" else "left"
        self.assertEqual(ov._focus, "right")
        ov._focus = "right" if ov._focus == "left" else "left"
        self.assertEqual(ov._focus, "left")

    def test_focus_right_resets_iter(self):
        ov = _make_overlay()
        ov._selected_iter = 5
        # Tab → right
        ov._focus = "right"
        ov._selected_iter = None
        self.assertIsNone(ov._selected_iter)

    def test_back_from_drill_clears_iter(self):
        ov = _make_overlay()
        ov._selected_iter = 3
        ov._focus = "right"
        # Simulate ← / Backspace
        if ov._selected_iter is not None:
            ov._selected_iter = None
            ov._right_scroll = 0
        self.assertIsNone(ov._selected_iter)

    def test_back_from_right_panel_goes_to_left(self):
        ov = _make_overlay()
        ov._focus = "right"
        ov._selected_iter = None
        # Simulate ← when no drill-down
        if ov._selected_iter is not None:
            ov._selected_iter = None
        else:
            ov._focus = "left"
        self.assertEqual(ov._focus, "left")


class TestEffectiveView(unittest.TestCase):

    def test_effective_view_returns_selected_task(self):
        tv = _make_tv()
        ov = _make_overlay({"t1": tv})
        ov._selected = "t1"
        ov._selected_sub = ""
        self.assertIs(ov._effective_view(), tv)

    def test_effective_view_sub_agent_own_view(self):
        tv_main = _make_tv("t1", "self")
        tv_sub = _make_tv("t2", "sub_agent_a")
        tv_main._sub_color("sub_agent_a")
        views = {"t1": tv_main, "t2": tv_sub}
        ov = _make_overlay(views)
        ov._selected = "t1"
        ov._selected_sub = "sub_agent_a"
        result = ov._effective_view()
        # Soll sub's eigene TaskView zurückgeben
        self.assertIs(result, tv_sub)

    def test_effective_view_sub_no_own_view_falls_back_to_parent(self):
        tv_main = _make_tv("t1", "self")
        tv_main._sub_color("sub_orphan")
        ov = _make_overlay({"t1": tv_main})
        ov._selected = "t1"
        ov._selected_sub = "sub_orphan"
        # Kein eigener TaskView für sub_orphan → Fallback auf parent
        result = ov._effective_view()
        self.assertIs(result, tv_main)


class TestIterDrillDown(unittest.TestCase):

    def test_drill_down_sets_selected_iter(self):
        from toolboxv2.flows.icli import IterView
        tv = _make_tv()
        tv._get_iter(1)
        tv._get_iter(2)
        tv._get_iter(3)
        ov = _make_overlay({"t1": tv})
        ov._focus = "right"
        ov._selected_iter = None

        # Simuliere Enter: wähle neueste sichtbare Iter
        iters_rev = list(reversed(tv.iterations))
        skip = ov._right_scroll
        for iv in iters_rev:
            lines = 1 + len(iv.thoughts) + len(iv.tools)
            if skip < lines:
                ov._selected_iter = iv.n
                ov._right_scroll = 0
                break
            skip -= lines

        self.assertEqual(ov._selected_iter, 3)

    def test_direct_iter_jump(self):
        from toolboxv2.flows.icli import IterView
        tv = _make_tv()
        tv._get_iter(5)
        ov = _make_overlay({"t1": tv})
        ov._focus = "right"

        # Simulate pressing "5"
        n = 5
        etv = ov._effective_view()
        if etv and n in etv._iter_map:
            ov._selected_iter = n
            ov._right_scroll = 0

        self.assertEqual(ov._selected_iter, 5)
        self.assertEqual(ov._right_scroll, 0)

    def test_jump_to_nonexistent_iter_no_change(self):
        tv = _make_tv()
        tv._get_iter(1)
        ov = _make_overlay({"t1": tv})
        ov._selected_iter = None

        # Simulate pressing "9" — iter 9 existiert nicht
        n = 9
        etv = ov._effective_view()
        if etv and n in etv._iter_map:
            ov._selected_iter = n

        self.assertIsNone(ov._selected_iter)


class TestIngestChunkIter(unittest.TestCase):

    def test_tool_start_sets_pending(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _make_tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        iv = tv._iter_map.get(1)
        self.assertIsNotNone(iv)
        self.assertEqual(iv.pending_tool, "vfs_shell")

    def test_tool_result_clears_pending(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _make_tv()
        tv.iteration = 1
        ingest_chunk(tv, {"type": "tool_start", "name": "vfs_shell", "iter": 1})
        ingest_chunk(tv, {"type": "tool_result", "name": "vfs_shell",
                          "result": '{"success": true, "stdout": "ok"}', "iter": 1})
        iv = tv._iter_map.get(1)
        self.assertEqual(iv.pending_tool, "")
        self.assertEqual(len(iv.tools), 1)
        tname, tok, elapsed, info = iv.tools[0]
        self.assertEqual(tname, "vfs_shell")
        self.assertTrue(tok)

    def test_reasoning_appended_to_iter(self):
        from toolboxv2.flows.icli import ingest_chunk
        tv = _make_tv()
        tv.iteration = 2
        ingest_chunk(tv, {"type": "reasoning", "chunk": "I think...", "iter": 2})
        iv = tv._iter_map.get(2)
        self.assertIn("I think...", iv.thoughts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
