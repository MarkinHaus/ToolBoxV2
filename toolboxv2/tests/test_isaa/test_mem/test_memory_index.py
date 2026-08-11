"""Tests for memory_index module — snapshot-based API (SQL-driven, zero LLM).

Covers the actual exported API:
  SpaceSnapshot, MemoryIndex,
  load_index, save_index,
  build_snapshot, build_index_from_memory,
  build_initial_index, update_index_after_save,
  render_index, filter_spaces_by_query,
  _top_concepts, _entry_count

Hypothesis tests at the end target the save_index type-confusion bug.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from toolboxv2.mods.isaa.memory_index import (
    MemoryIndex,
    SpaceSnapshot,
    _entry_count,
    _top_concepts,
    build_index_from_memory,
    build_initial_index,
    build_snapshot,
    filter_spaces_by_query,
    load_index,
    render_index,
    save_index,
    update_index_after_save,
)


# ── Mock infrastructure ──────────────────────────────────────────

class MockCursor:
    """Fake DB cursor — iterable + fetchone/fetchall."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else (0,)


class MockConn:
    """Fake SQLite connection. Routes by SQL keyword."""

    def __init__(self, count=0, concepts=None):
        self._count = count
        self._concepts = concepts or []

    def execute(self, q, params=()):
        ql = q.lower()
        if "concept_index" in ql:
            return MockCursor(self._concepts)
        if "count" in ql:
            return MockCursor([(self._count,)])
        return MockCursor([])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class MockStore:
    """Fake HybridMemoryStore — has .space + ._tx()."""

    def __init__(self, space="test", count=0, concepts=None):
        self.space = space
        self._count = count
        self._concepts = concepts or []

    def _tx(self):
        return MockConn(count=self._count, concepts=self._concepts)


class MockMemory:
    """Fake AISemanticMemory — has .memories dict."""

    def __init__(self, stores):
        self.memories = stores


# ── Unit: Schema ─────────────────────────────────────────────────

class TestSpaceSnapshot(unittest.TestCase):
    def test_defaults(self):
        s = SpaceSnapshot()
        self.assertEqual(s.nodes, [])
        self.assertEqual(s.edges, [])
        self.assertEqual(s.concepts, {})
        self.assertEqual(s.entry_count, 0)

    def test_roundtrip_json(self):
        s = SpaceSnapshot(
            nodes=[{"id": "n1"}],
            edges=[{"source": "n1", "target": "n2"}],
            concepts={"auth": 3},
            entry_count=2,
        )
        data = json.loads(s.model_dump_json())
        restored = SpaceSnapshot(**data)
        self.assertEqual(restored.entry_count, 2)
        self.assertEqual(restored.concepts, {"auth": 3})
        self.assertEqual(restored.nodes, [{"id": "n1"}])


class TestMemoryIndexSchema(unittest.TestCase):
    def test_empty_default(self):
        idx = MemoryIndex()
        self.assertEqual(idx.spaces, {})

    def test_entries_backcompat(self):
        """entries property must alias spaces (module.py checks len(idx.entries))."""
        idx = MemoryIndex(spaces={"sp": SpaceSnapshot(entry_count=1)})
        self.assertEqual(len(idx.entries), 1)
        self.assertIs(idx.entries, idx.spaces)

    def test_multiple_spaces(self):
        idx = MemoryIndex(spaces={
            "a": SpaceSnapshot(entry_count=1),
            "b": SpaceSnapshot(entry_count=2),
        })
        self.assertEqual(len(idx.spaces), 2)
        self.assertEqual(idx.spaces["a"].entry_count, 1)
        self.assertEqual(idx.spaces["b"].entry_count, 2)


# ── Unit: _top_concepts / _entry_count ───────────────────────────

class TestTopConcepts(unittest.TestCase):
    def test_returns_dict(self):
        store = MockStore(space="sp", concepts=[("auth", 3), ("jwt", 1)])
        result = _top_concepts(store)
        self.assertEqual(result, {"auth": 3, "jwt": 1})

    def test_empty_store(self):
        store = MockStore(space="sp", concepts=[])
        self.assertEqual(_top_concepts(store), {})

    def test_exception_returns_empty(self):
        store = MagicMock()
        store._tx.side_effect = RuntimeError("db locked")
        store.space = "sp"
        self.assertEqual(_top_concepts(store), {})


class TestEntryCount(unittest.TestCase):
    def test_returns_count(self):
        store = MockStore(space="sp", count=7)
        self.assertEqual(_entry_count(store), 7)

    def test_zero(self):
        store = MockStore(space="sp", count=0)
        self.assertEqual(_entry_count(store), 0)

    def test_exception_returns_zero(self):
        store = MagicMock()
        store._tx.side_effect = RuntimeError("db locked")
        self.assertEqual(_entry_count(store), 0)


# ── Unit: build_snapshot (patched MemoryGraphVisualizer) ─────────

class TestBuildSnapshot(unittest.TestCase):
    @patch("toolboxv2.mods.isaa.memory_index.MemoryGraphVisualizer")
    def test_empty_store(self, mock_vis_cls):
        mock_vis = MagicMock()
        mock_vis.to_json.return_value = {"nodes": [], "edges": []}
        mock_vis_cls.return_value = mock_vis

        store = MockStore(space="sp", count=0, concepts=[])
        snap = build_snapshot(store)
        self.assertEqual(snap.entry_count, 0)
        self.assertEqual(snap.nodes, [])
        self.assertEqual(snap.edges, [])
        self.assertEqual(snap.concepts, {})

    @patch("toolboxv2.mods.isaa.memory_index.MemoryGraphVisualizer")
    def test_with_data(self, mock_vis_cls):
        mock_vis = MagicMock()
        mock_vis.to_json.return_value = {
            "nodes": [{"id": "doc:1", "label": "auth.py", "type": "code"}],
            "edges": [{"source": "doc:1", "target": "concept:auth", "type": "has_concept"}],
        }
        mock_vis_cls.return_value = mock_vis

        store = MockStore(space="sp", count=3, concepts=[("auth", 2), ("jwt", 1)])
        snap = build_snapshot(store)
        self.assertEqual(snap.entry_count, 3)
        self.assertEqual(len(snap.nodes), 1)
        self.assertEqual(snap.nodes[0]["label"], "auth.py")
        self.assertEqual(snap.concepts, {"auth": 2, "jwt": 1})
        self.assertEqual(len(snap.edges), 1)


# ── Unit: build_index_from_memory ────────────────────────────────

class TestBuildIndexFromMemory(unittest.TestCase):
    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    def test_skip_empty_spaces(self, mock_build):
        def side_effect(store):
            if store.space == "empty":
                return SpaceSnapshot(entry_count=0, nodes=[])
            return SpaceSnapshot(entry_count=5, nodes=[{"id": "n1"}], concepts={"x": 1})
        mock_build.side_effect = side_effect

        mem = MockMemory({
            "empty": MockStore("empty", count=0),
            "full": MockStore("full", count=5),
        })
        idx = build_index_from_memory(mem)
        self.assertNotIn("empty", idx.spaces)
        self.assertIn("full", idx.spaces)
        self.assertEqual(idx.spaces["full"].entry_count, 5)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    def test_all_empty_returns_empty(self, mock_build):
        mock_build.return_value = SpaceSnapshot(entry_count=0, nodes=[])
        mem = MockMemory({"a": MockStore("a"), "b": MockStore("b")})
        idx = build_index_from_memory(mem)
        self.assertEqual(idx.spaces, {})

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    def test_snapshot_exception_skipped(self, mock_build):
        mock_build.side_effect = [
            RuntimeError("boom"),
            SpaceSnapshot(entry_count=1, concepts={"ok": 1}),
        ]
        mem = MockMemory({"bad": MockStore("bad"), "good": MockStore("good")})
        idx = build_index_from_memory(mem)
        self.assertNotIn("bad", idx.spaces)
        self.assertIn("good", idx.spaces)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    def test_nodes_only_no_entries(self, mock_build):
        """Space with nodes but 0 entries should still be included."""
        mock_build.return_value = SpaceSnapshot(
            entry_count=0, nodes=[{"id": "n1", "label": "entity"}]
        )
        mem = MockMemory({"sp": MockStore("sp")})
        idx = build_index_from_memory(mem)
        self.assertIn("sp", idx.spaces)


# ── Unit: render_index ───────────────────────────────────────────

class TestRenderIndex(unittest.TestCase):
    def test_empty_index(self):
        result = render_index(MemoryIndex())
        self.assertIn("# Memory Index", result)
        self.assertIn("Empty", result)

    def test_renders_spaces_sorted(self):
        idx = MemoryIndex(spaces={
            "zebra": SpaceSnapshot(entry_count=1, concepts={"z": 1}),
            "alpha": SpaceSnapshot(entry_count=1, concepts={"a": 1}),
        })
        result = render_index(idx)
        self.assertLess(result.index("## alpha"), result.index("## zebra"))

    def test_renders_concepts(self):
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(entry_count=1, concepts={"auth": 3, "jwt": 1}),
        })
        result = render_index(idx)
        self.assertIn("`auth`", result)
        self.assertIn("(3)", result)

    def test_renders_entities_and_relations(self):
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(
                nodes=[
                    {"id": "n1", "label": "auth.py", "type": "code"},
                    {"id": "n2", "label": "login.py", "type": "code"},
                ],
                edges=[{"source": "n1", "target": "n2", "type": "imports"}],
                entry_count=2,
            ),
        })
        result = render_index(idx)
        self.assertIn("**auth.py**", result)
        self.assertIn("**login.py**", result)
        self.assertIn("imports", result)

    def test_entity_without_relations(self):
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "standalone", "type": "doc"}],
                entry_count=1,
            ),
        })
        result = render_index(idx)
        self.assertIn("**standalone**", result)

    def test_skips_empty_spaces(self):
        idx = MemoryIndex(spaces={
            "empty": SpaceSnapshot(entry_count=0, nodes=[]),
            "full": SpaceSnapshot(entry_count=1, concepts={"x": 1}),
        })
        result = render_index(idx)
        self.assertNotIn("## empty", result)
        self.assertIn("## full", result)


# ── Unit: filter_spaces_by_query ─────────────────────────────────

class TestFilterSpacesByQuery(unittest.TestCase):
    def test_empty_index(self):
        self.assertEqual(filter_spaces_by_query(MemoryIndex(), "anything"), [])

    def test_node_label_match(self):
        idx = MemoryIndex(spaces={
            "sp_auth": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "authentication", "type": "code"}],
                entry_count=1,
            ),
            "sp_db": SpaceSnapshot(
                nodes=[{"id": "n2", "label": "database", "type": "code"}],
                entry_count=1,
            ),
        })
        result = filter_spaces_by_query(idx, "authentication")
        self.assertEqual(result[0], "sp_auth")

    def test_concept_substring_match(self):
        """'auth' should match concept 'authentication' via substring."""
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(entry_count=1, concepts={"authentication": 2}),
        })
        result = filter_spaces_by_query(idx, "auth")
        self.assertIn("sp", result)

    def test_edge_type_match(self):
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "x", "type": "doc"}],
                edges=[{"source": "n1", "target": "n2", "type": "auth_dependency"}],
                entry_count=1,
            ),
        })
        result = filter_spaces_by_query(idx, "auth")
        self.assertIn("sp", result)

    def test_ranking_by_hits(self):
        idx = MemoryIndex(spaces={
            "low": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "auth", "type": "code"}],
                entry_count=1,
            ),
            "high": SpaceSnapshot(
                nodes=[{"id": "n2", "label": "auth module", "type": "code"}],
                edges=[{"source": "n2", "target": "n3", "type": "auth"}],
                concepts={"auth": 3, "jwt": 2},
                entry_count=3,
            ),
        })
        result = filter_spaces_by_query(idx, "auth")
        self.assertEqual(result[0], "high")

    def test_no_match(self):
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "auth", "type": "code"}],
                entry_count=1,
            ),
        })
        self.assertEqual(filter_spaces_by_query(idx, "quantum"), [])

    def test_case_insensitive(self):
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "Auth Module", "type": "Code"}],
                entry_count=1,
            ),
        })
        result = filter_spaces_by_query(idx, "auth code")
        self.assertIn("sp", result)

    def test_short_tokens_ignored(self):
        """Tokens with len <= 2 are filtered out."""
        idx = MemoryIndex(spaces={
            "sp": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "ab", "type": "code"}],
                entry_count=1,
            ),
        })
        result = filter_spaces_by_query(idx, "ab")
        self.assertEqual(result, [])

    def test_multiple_spaces_ranked(self):
        idx = MemoryIndex(spaces={
            "second": SpaceSnapshot(
                nodes=[{"id": "n1", "label": "auth", "type": "x"}],
                entry_count=1,
            ),
            "first": SpaceSnapshot(
                nodes=[{"id": "n2", "label": "auth", "type": "x"}],
                concepts={"auth": 5},
                entry_count=3,
            ),
        })
        result = filter_spaces_by_query(idx, "auth")
        self.assertEqual(result[0], "first")
        self.assertEqual(len(result), 2)


# ── Integration: Persistence ─────────────────────────────────────

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_nonexistent_returns_empty(self):
        loaded = load_index(self.tmpdir, "ghost")
        self.assertEqual(loaded.spaces, {})

    def test_load_corrupt_returns_empty(self):
        agent_dir = os.path.join(self.tmpdir, "Agents", "broken")
        os.makedirs(agent_dir)
        with open(os.path.join(agent_dir, "memory_index.json"), "w") as f:
            f.write("{{{invalid json")
        loaded = load_index(self.tmpdir, "broken")
        self.assertEqual(loaded.spaces, {})

    def test_save_creates_directories(self):
        idx = MemoryIndex(spaces={"s": SpaceSnapshot(entry_count=1)})
        save_index(self.tmpdir, "new_agent", idx)
        expected = os.path.join(self.tmpdir, "Agents", "new_agent", "memory_index.json")
        self.assertTrue(os.path.exists(expected))

    def test_save_overwrites_existing(self):
        save_index(self.tmpdir, "ag", MemoryIndex(spaces={"old": SpaceSnapshot(entry_count=1)}))
        save_index(self.tmpdir, "ag", MemoryIndex(spaces={"new": SpaceSnapshot(entry_count=2)}))
        loaded = load_index(self.tmpdir, "ag")
        self.assertNotIn("old", loaded.spaces)
        self.assertIn("new", loaded.spaces)

    def test_save_load_roundtrip(self):
        snap = SpaceSnapshot(
            nodes=[{"id": "doc:1", "label": "auth.py", "type": "code"}],
            edges=[{"source": "doc:1", "target": "concept:auth", "type": "has_concept", "weight": 0.8}],
            concepts={"auth": 3, "jwt": 1},
            entry_count=2,
        )
        idx = MemoryIndex(spaces={"core": snap})
        save_index(self.tmpdir, "agent1", idx)
        loaded = load_index(self.tmpdir, "agent1")
        self.assertEqual(len(loaded.spaces), 1)
        self.assertIn("core", loaded.spaces)
        self.assertEqual(loaded.spaces["core"].entry_count, 2)
        self.assertEqual(loaded.spaces["core"].concepts["auth"], 3)
        self.assertEqual(len(loaded.spaces["core"].nodes), 1)


# ── Integration: build_initial_index ─────────────────────────────

class TestBuildInitialIndex(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_empty_memory_returns_empty(self, mock_build):
        mock_build.return_value = SpaceSnapshot(entry_count=0, nodes=[])
        mem = MockMemory({})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem
        result = await build_initial_index(isaa, "agent", self.tmpdir)
        self.assertEqual(result.spaces, {})

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_successful_build(self, mock_build):
        def side_effect(store):
            if store.space == "empty":
                return SpaceSnapshot(entry_count=0, nodes=[])
            return SpaceSnapshot(entry_count=5, concepts={"auth": 2}, nodes=[{"id": "n1"}])
        mock_build.side_effect = side_effect

        mem = MockMemory({"empty": MockStore("empty"), "core": MockStore("core")})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        result = await build_initial_index(isaa, "agent", self.tmpdir)
        self.assertNotIn("empty", result.spaces)
        self.assertIn("core", result.spaces)
        self.assertEqual(result.spaces["core"].entry_count, 5)

        # Verify persisted to disk
        loaded = load_index(self.tmpdir, "agent")
        self.assertIn("core", loaded.spaces)


# ── Integration: update_index_after_save ─────────────────────────

class TestUpdateIndexAfterSave(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_refresh_existing_space(self, mock_build):
        mock_build.return_value = SpaceSnapshot(
            entry_count=10, concepts={"auth": 5},
            nodes=[{"id": "n1", "label": "updated"}],
        )
        idx = MemoryIndex(spaces={"sp": SpaceSnapshot(entry_count=3)})
        mem = MockMemory({"sp": MockStore("sp", count=10)})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        result = await update_index_after_save(isaa, "agent", self.tmpdir, idx, "sp")
        self.assertEqual(result.spaces["sp"].entry_count, 10)
        self.assertEqual(result.spaces["sp"].concepts["auth"], 5)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_new_space_added(self, mock_build):
        mock_build.return_value = SpaceSnapshot(entry_count=2, concepts={"db": 1})
        idx = MemoryIndex()
        mem = MockMemory({"new_space": MockStore("new_space", count=2)})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        result = await update_index_after_save(isaa, "agent", self.tmpdir, idx, "new_space")
        self.assertIn("new_space", result.spaces)
        self.assertEqual(result.spaces["new_space"].entry_count, 2)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_empty_space_removed(self, mock_build):
        mock_build.return_value = SpaceSnapshot(entry_count=0, nodes=[])
        idx = MemoryIndex(spaces={"sp": SpaceSnapshot(entry_count=5)})
        mem = MockMemory({"sp": MockStore("sp", count=0)})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        result = await update_index_after_save(isaa, "agent", self.tmpdir, idx, "sp")
        self.assertNotIn("sp", result.spaces)

    async def test_missing_space_returns_unchanged(self):
        idx = MemoryIndex(spaces={"existing": SpaceSnapshot(entry_count=1)})
        mem = MockMemory({})  # no "nonexistent" space
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        result = await update_index_after_save(isaa, "agent", self.tmpdir, idx, "nonexistent")
        self.assertEqual(result.spaces, idx.spaces)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_persisted_to_disk(self, mock_build):
        mock_build.return_value = SpaceSnapshot(entry_count=7, concepts={"x": 1})
        idx = MemoryIndex()
        mem = MockMemory({"sp": MockStore("sp", count=7)})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        await update_index_after_save(isaa, "agent", self.tmpdir, idx, "sp")
        loaded = load_index(self.tmpdir, "agent")
        self.assertIn("sp", loaded.spaces)
        self.assertEqual(loaded.spaces["sp"].entry_count, 7)

    @patch("toolboxv2.mods.isaa.memory_index.build_snapshot")
    async def test_ignored_content_and_concepts_params(self, mock_build):
        """content + concepts params are ignored — graph is source of truth."""
        mock_build.return_value = SpaceSnapshot(entry_count=1, concepts={"real": 1})
        idx = MemoryIndex()
        mem = MockMemory({"sp": MockStore("sp", count=1)})
        isaa = MagicMock()
        isaa.get_memory.return_value = mem

        result = await update_index_after_save(
            isaa, "agent", self.tmpdir, idx, "sp",
            content="ignored content", concepts=["ignored"],
        )
        # Concepts come from build_snapshot, not from params
        self.assertEqual(result.spaces["sp"].concepts, {"real": 1})


# ── Hypothesis Tests: save_index type confusion ──────────────────
#
# HYPOTHESIS H1: save_index serializes SpaceSnapshot instead of MemoryIndex
#   when agent_name matches a space key.
#   Root cause line 52:
#     p.write_text(index.spaces.get(agent_name, index).model_dump_json(indent=2))
#   If agent_name IS a key in spaces → returns SpaceSnapshot → JSON has
#   {nodes, edges, concepts, entry_count} but NOT {spaces: {...}}.
#   load_index tries MemoryIndex.model_validate_json → fails → empty index.
#
# HYPOTHESIS H2 (control): roundtrip works when agent_name does NOT match
#   any space key — should always PASS.
#
# HYPOTHESIS H3: Production scenario — agent "self", space key "self".
#   Multi-space index loses all data on save/load cycle.

class TestHypothesisSaveIndexBug(unittest.TestCase):
    """Tests that DISCRINATE the save_index bug.
    Before fix: H1+H3 FAIL, H2 PASSES.
    After fix:  all PASS.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_h1_agent_name_equals_space_key(self):
        """H1: save_index corrupts when agent_name == space key.

        Before fix: saves SpaceSnapshot JSON, load_index can't parse → empty.
        After fix: saves full MemoryIndex, roundtrip works.
        """
        snap = SpaceSnapshot(entry_count=1, concepts={"auth": 1})
        idx = MemoryIndex(spaces={"self": snap})

        save_index(self.tmpdir, "self", idx)
        loaded = load_index(self.tmpdir, "self")

        self.assertIn("self", loaded.spaces,
                      msg="H1 FAIL: save_index saved SpaceSnapshot instead of MemoryIndex")
        self.assertEqual(loaded.spaces["self"].entry_count, 1)

    def test_h2_control_agent_name_differs(self):
        """H2 (control): roundtrip works when agent_name != space key."""
        snap = SpaceSnapshot(
            nodes=[{"id": "n1", "label": "x", "type": "doc"}],
            edges=[{"source": "n1", "target": "n2", "type": "rel"}],
            concepts={"a": 1, "b": 2},
            entry_count=3,
        )
        idx = MemoryIndex(spaces={"workspace": snap})

        save_index(self.tmpdir, "agent_x", idx)
        loaded = load_index(self.tmpdir, "agent_x")

        self.assertEqual(len(loaded.spaces), 1)
        self.assertIn("workspace", loaded.spaces)
        self.assertEqual(loaded.spaces["workspace"].entry_count, 3)
        self.assertEqual(loaded.spaces["workspace"].concepts["a"], 1)
        self.assertEqual(len(loaded.spaces["workspace"].nodes), 1)

    def test_h3_production_self_self_multi_space(self):
        """H3: agent_name='self', multiple spaces including key 'self'.

        Before fix: save_index does spaces.get('self') → returns SpaceSnapshot
        for 'self' space → all other spaces lost on reload.
        After fix: full MemoryIndex saved → all spaces survive.
        """
        idx = MemoryIndex(spaces={
            "self": SpaceSnapshot(entry_count=5, concepts={"python": 2}),
            "work": SpaceSnapshot(entry_count=3, concepts={"api": 1}),
            "projects": SpaceSnapshot(entry_count=10, concepts={"isaa": 4}),
        })

        save_index(self.tmpdir, "self", idx)
        loaded = load_index(self.tmpdir, "self")

        self.assertEqual(len(loaded.spaces), 3,
                         msg="H3 FAIL: multi-space index lost spaces on save/load roundtrip")
        self.assertIn("self", loaded.spaces)
        self.assertIn("work", loaded.spaces)
        self.assertIn("projects", loaded.spaces)
        self.assertEqual(loaded.spaces["work"].entry_count, 3)


if __name__ == "__main__":
    unittest.main()
