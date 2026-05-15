"""Tests for memory_index module — unit + integration layers."""
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import types

# Fake the toolboxv2 import chain so build_initial_index's lazy import resolves
_mka_module = types.ModuleType("toolboxv2.mods.isaa.base.MemoryKnowledgeActor")
_mka_module.MemoryKnowledgeActor = MagicMock  # placeholder, overridden per-test
for _mod_path in [
    "toolboxv2",
    "toolboxv2.mods",
    "toolboxv2.mods.isaa",
    "toolboxv2.mods.isaa.base",
    "toolboxv2.mods.isaa.base.MemoryKnowledgeActor",
]:
    sys.modules.setdefault(_mod_path, types.ModuleType(_mod_path))
sys.modules["toolboxv2.mods.isaa.base.MemoryKnowledgeActor"] = _mka_module

from memory_index import (
    MemoryIndex,
    MemoryIndexEdit,
    MemoryIndexEntry,
    apply_edit,
    build_initial_index,
    filter_spaces_by_query,
    load_index,
    render_index,
    save_index,
    update_index_after_save,
)


# ── Factories ───────────────────────────────────────────────────────────

def make_entry(concepts=None, summary="test summary"):
    return MemoryIndexEntry(
        key_concepts=concepts or ["default"],
        summary=summary,
    )


def make_edit(space="testspace", cluster="auth", info="new auth info"):
    return MemoryIndexEdit(
        space=space,
        concept_cluster=cluster,
        new_information=info,
    )


def make_index(entries=None):
    return MemoryIndex(entries=entries or {})


# ── Unit: Schema ────────────────────────────────────────────────────────

class TestMemoryIndexEntry(unittest.TestCase):
    def test_defaults_empty(self):
        e = MemoryIndexEntry()
        self.assertEqual(e.key_concepts, [])
        self.assertEqual(e.summary, "")

    def test_roundtrip_json(self):
        e = make_entry(["auth", "jwt"], "handles token validation")
        data = json.loads(e.model_dump_json())
        restored = MemoryIndexEntry(**data)
        self.assertEqual(restored.key_concepts, ["auth", "jwt"])
        self.assertEqual(restored.summary, "handles token validation")


class TestMemoryIndexEdit(unittest.TestCase):
    def test_required_fields(self):
        with self.assertRaises(Exception):
            MemoryIndexEdit()  # all fields required

    def test_construction(self):
        e = make_edit()
        self.assertEqual(e.space, "testspace")
        self.assertEqual(e.concept_cluster, "auth")


class TestMemoryIndex(unittest.TestCase):
    def test_empty_default(self):
        idx = MemoryIndex()
        self.assertEqual(idx.entries, {})

    def test_multiple_spaces(self):
        idx = make_index({
            "space_a": [make_entry(["x"])],
            "space_b": [make_entry(["y"]), make_entry(["z"])],
        })
        self.assertEqual(len(idx.entries["space_a"]), 1)
        self.assertEqual(len(idx.entries["space_b"]), 2)


# ── Unit: apply_edit ────────────────────────────────────────────────────

class TestApplyEdit(unittest.TestCase):
    def test_insert_new_space_and_cluster(self):
        idx = make_index()
        edit = make_edit(space="new_space", cluster="db", info="database layer")
        result = apply_edit(idx, edit)
        self.assertIn("new_space", result.entries)
        self.assertEqual(len(result.entries["new_space"]), 1)
        self.assertEqual(result.entries["new_space"][0].summary, "database layer")
        self.assertEqual(result.entries["new_space"][0].key_concepts, ["db"])

    def test_upsert_existing_cluster(self):
        idx = make_index({"sp": [make_entry(["auth"], "old info")]})
        edit = make_edit(space="sp", cluster="auth", info="updated info")
        result = apply_edit(idx, edit)
        self.assertEqual(len(result.entries["sp"]), 1, msg="should not add duplicate")
        self.assertEqual(result.entries["sp"][0].summary, "updated info")

    def test_upsert_case_insensitive(self):
        idx = make_index({"sp": [make_entry(["Auth"], "old")]})
        edit = make_edit(space="sp", cluster="auth", info="new")
        result = apply_edit(idx, edit)
        self.assertEqual(len(result.entries["sp"]), 1)
        self.assertEqual(result.entries["sp"][0].summary, "new")

    def test_add_second_cluster_to_existing_space(self):
        idx = make_index({"sp": [make_entry(["auth"], "auth stuff")]})
        edit = make_edit(space="sp", cluster="db", info="db stuff")
        result = apply_edit(idx, edit)
        self.assertEqual(len(result.entries["sp"]), 2)

    def test_multiple_sequential_edits(self):
        idx = make_index()
        idx = apply_edit(idx, make_edit(space="s", cluster="a", info="1"))
        idx = apply_edit(idx, make_edit(space="s", cluster="b", info="2"))
        idx = apply_edit(idx, make_edit(space="s", cluster="a", info="1-updated"))
        self.assertEqual(len(idx.entries["s"]), 2)
        self.assertEqual(idx.entries["s"][0].summary, "1-updated")


# ── Unit: render_index ──────────────────────────────────────────────────

class TestRenderIndex(unittest.TestCase):
    def test_empty_index(self):
        result = render_index(make_index())
        self.assertIn("Empty", result)
        self.assertIn("# Memory Index", result)

    def test_renders_spaces_sorted(self):
        idx = make_index({
            "zebra": [make_entry(["z"], "z stuff")],
            "alpha": [make_entry(["a"], "a stuff")],
        })
        result = render_index(idx)
        alpha_pos = result.index("## alpha")
        zebra_pos = result.index("## zebra")
        self.assertLess(alpha_pos, zebra_pos, msg="spaces should be sorted alphabetically")

    def test_renders_concepts_and_summary(self):
        idx = make_index({"sp": [make_entry(["auth", "jwt"], "token handling")]})
        result = render_index(idx)
        self.assertIn("auth, jwt", result)
        self.assertIn("token handling", result)

    def test_skips_empty_entry_lists(self):
        idx = make_index({"empty_sp": [], "full_sp": [make_entry(["x"], "y")]})
        result = render_index(idx)
        self.assertNotIn("## empty_sp", result)
        self.assertIn("## full_sp", result)


# ── Unit: filter_spaces_by_query ────────────────────────────────────────

class TestFilterSpacesByQuery(unittest.TestCase):
    def test_empty_index_returns_empty(self):
        self.assertEqual(filter_spaces_by_query(make_index(), "anything"), [])

    def test_exact_concept_match(self):
        idx = make_index({
            "sp_auth": [make_entry(["auth"], "auth stuff")],
            "sp_db": [make_entry(["database"], "db stuff")],
        })
        result = filter_spaces_by_query(idx, "auth system")
        self.assertEqual(result[0], "sp_auth")

    def test_partial_concept_match(self):
        idx = make_index({"sp": [make_entry(["authentication"], "login flow")]})
        result = filter_spaces_by_query(idx, "auth")
        self.assertIn("sp", result, msg="'auth' should match 'authentication' via substring")

    def test_summary_word_match(self):
        idx = make_index({"sp": [make_entry(["x"], "handles JWT token validation")]})
        result = filter_spaces_by_query(idx, "JWT")
        self.assertIn("sp", result)

    def test_ranking_by_hit_count(self):
        idx = make_index({
            "low": [make_entry(["auth"], "one hit")],
            "high": [make_entry(["auth", "jwt"], "auth and jwt tokens")],
        })
        result = filter_spaces_by_query(idx, "auth jwt")
        self.assertEqual(result[0], "high", msg="more hits should rank higher")

    def test_no_match_returns_empty(self):
        idx = make_index({"sp": [make_entry(["auth"], "login")]})
        result = filter_spaces_by_query(idx, "quantum physics")
        self.assertEqual(result, [])

    def test_case_insensitive(self):
        idx = make_index({"sp": [make_entry(["Auth"], "Login Flow")]})
        result = filter_spaces_by_query(idx, "auth login")
        self.assertIn("sp", result)


# ── Integration: Persistence (real filesystem) ─────────────────────────

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        idx = make_index({"sp": [make_entry(["auth", "jwt"], "token stuff")]})
        save_index(self.tmpdir, "test_agent", idx)
        loaded = load_index(self.tmpdir, "test_agent")
        self.assertEqual(len(loaded.entries["sp"]), 1)
        self.assertEqual(loaded.entries["sp"][0].key_concepts, ["auth", "jwt"])
        self.assertEqual(loaded.entries["sp"][0].summary, "token stuff")

    def test_load_nonexistent_returns_empty(self):
        loaded = load_index(self.tmpdir, "nonexistent_agent")
        self.assertEqual(loaded.entries, {})

    def test_load_corrupt_file_returns_empty(self):
        agent_dir = os.path.join(self.tmpdir, "Agents", "broken")
        os.makedirs(agent_dir)
        with open(os.path.join(agent_dir, "memory_index.json"), "w") as f:
            f.write("{{{invalid json")
        loaded = load_index(self.tmpdir, "broken")
        self.assertEqual(loaded.entries, {})

    def test_save_creates_directories(self):
        save_index(self.tmpdir, "new_agent", make_index({"s": [make_entry()]}))
        expected_path = os.path.join(self.tmpdir, "Agents", "new_agent", "memory_index.json")
        self.assertTrue(os.path.exists(expected_path))

    def test_save_overwrites_existing(self):
        save_index(self.tmpdir, "ag", make_index({"old": [make_entry(["old"], "old")]}))
        save_index(self.tmpdir, "ag", make_index({"new": [make_entry(["new"], "new")]}))
        loaded = load_index(self.tmpdir, "ag")
        self.assertNotIn("old", loaded.entries)
        self.assertIn("new", loaded.entries)


# ── Integration: build_initial_index (mocked ISAA) ─────────────────────

class TestBuildInitialIndex(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_isaa_ref(self, spaces, format_class_return):
        """Build a fake isaa_ref with controllable memory and format_class."""
        mem = MagicMock()
        mem.memories = {s: True for s in spaces}

        isaa = MagicMock()
        isaa.get_memory.return_value = mem
        isaa.format_class = AsyncMock(return_value=format_class_return)
        return isaa

    async def test_empty_spaces_returns_empty_index(self):
        isaa = self._make_isaa_ref(spaces=[], format_class_return=[])
        result = await build_initial_index(isaa, "agent", self.tmpdir)
        self.assertEqual(result.entries, {})

    async def test_all_empty_spaces_skipped(self):
        # MKA with no stats and no concepts → space should be skipped
        isaa = self._make_isaa_ref(spaces=["empty1", "empty2"], format_class_return=[])

        with patch("toolboxv2.mods.isaa.base.MemoryKnowledgeActor.MemoryKnowledgeActor") as MockMKA:
            mock_mka = MagicMock()
            mock_mka.get_stats = AsyncMock(return_value={"total_entries": 0})
            mock_mka.list_concepts = AsyncMock(return_value=[])
            MockMKA.return_value = mock_mka

            result = await build_initial_index(isaa, "agent", self.tmpdir)
            self.assertEqual(result.entries, {})
            isaa.format_class.assert_not_called()

    async def test_successful_build_applies_edits(self):
        edits = [
            {"space": "core", "concept_cluster": "auth", "new_information": "auth system overview"},
            {"space": "core", "concept_cluster": "db", "new_information": "database layer"},
        ]
        isaa = self._make_isaa_ref(spaces=["core"], format_class_return=edits)

        with patch("toolboxv2.mods.isaa.base.MemoryKnowledgeActor.MemoryKnowledgeActor") as MockMKA:
            mock_mka = MagicMock()
            mock_mka.get_stats = AsyncMock(return_value={"total_entries": 5})
            mock_mka.list_concepts = AsyncMock(return_value=["auth", "jwt", "db"])
            MockMKA.return_value = mock_mka

            result = await build_initial_index(isaa, "agent", self.tmpdir)
            self.assertEqual(len(result.entries["core"]), 2)
            # verify persisted to disk
            loaded = load_index(self.tmpdir, "agent")
            self.assertEqual(len(loaded.entries["core"]), 2)

    async def test_format_class_failure_returns_empty_index(self):
        isaa = self._make_isaa_ref(spaces=["sp"], format_class_return=None)
        isaa.format_class = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch("toolboxv2.mods.isaa.base.MemoryKnowledgeActor.MemoryKnowledgeActor") as MockMKA:
            mock_mka = MagicMock()
            mock_mka.get_stats = AsyncMock(return_value={"total_entries": 3})
            mock_mka.list_concepts = AsyncMock(return_value=["x"])
            MockMKA.return_value = mock_mka

            result = await build_initial_index(isaa, "agent", self.tmpdir)
            self.assertEqual(result.entries, {})

    async def test_malformed_edits_skipped_gracefully(self):
        edits = [
            {"space": "sp", "concept_cluster": "ok", "new_information": "valid"},
            {"garbage": True},  # malformed
            42,  # not a dict
        ]
        isaa = self._make_isaa_ref(spaces=["sp"], format_class_return=edits)

        with patch("toolboxv2.mods.isaa.base.MemoryKnowledgeActor.MemoryKnowledgeActor") as MockMKA:
            mock_mka = MagicMock()
            mock_mka.get_stats = AsyncMock(return_value={"total_entries": 1})
            mock_mka.list_concepts = AsyncMock(return_value=["ok"])
            MockMKA.return_value = mock_mka

            result = await build_initial_index(isaa, "agent", self.tmpdir)
            self.assertEqual(len(result.entries["sp"]), 1, msg="only valid edit should apply")


# ── Integration: update_index_after_save (mocked ISAA) ──────────────────

class TestUpdateIndexAfterSave(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_adds_new_cluster(self):
        idx = make_index({"sp": [make_entry(["auth"], "auth stuff")]})
        isaa = MagicMock()
        isaa.format_class = AsyncMock(return_value={
            "space": "sp", "concept_cluster": "db", "new_information": "added db layer"
        })

        result = await update_index_after_save(
            isaa, "agent", self.tmpdir, idx, "sp", "new db content", ["db"]
        )
        self.assertEqual(len(result.entries["sp"]), 2)
        # verify persisted
        loaded = load_index(self.tmpdir, "agent")
        self.assertEqual(len(loaded.entries["sp"]), 2)

    async def test_updates_existing_cluster(self):
        idx = make_index({"sp": [make_entry(["auth"], "old auth")]})
        isaa = MagicMock()
        isaa.format_class = AsyncMock(return_value={
            "space": "sp", "concept_cluster": "auth", "new_information": "updated auth"
        })

        result = await update_index_after_save(
            isaa, "agent", self.tmpdir, idx, "sp", "new auth fact", ["auth"]
        )
        self.assertEqual(len(result.entries["sp"]), 1)
        self.assertEqual(result.entries["sp"][0].summary, "updated auth")

    async def test_format_class_returns_none_preserves_index(self):
        idx = make_index({"sp": [make_entry(["auth"], "original")]})
        isaa = MagicMock()
        isaa.format_class = AsyncMock(return_value=None)

        result = await update_index_after_save(
            isaa, "agent", self.tmpdir, idx, "sp", "content", ["x"]
        )
        self.assertEqual(result.entries["sp"][0].summary, "original")

    async def test_format_class_exception_preserves_index(self):
        idx = make_index({"sp": [make_entry(["auth"], "original")]})
        isaa = MagicMock()
        isaa.format_class = AsyncMock(side_effect=RuntimeError("boom"))

        result = await update_index_after_save(
            isaa, "agent", self.tmpdir, idx, "sp", "content", None
        )
        self.assertEqual(result.entries["sp"][0].summary, "original")

    async def test_content_truncated_to_500_chars(self):
        idx = make_index()
        isaa = MagicMock()
        isaa.format_class = AsyncMock(return_value={
            "space": "sp", "concept_cluster": "big", "new_information": "summary"
        })

        long_content = "x" * 1000
        await update_index_after_save(
            isaa, "agent", self.tmpdir, idx, "sp", long_content, None
        )
        # verify the prompt sent to format_class had truncated content
        call_args = isaa.format_class.call_args
        task_prompt = call_args.kwargs.get("task", "")
        self.assertNotIn("x" * 501, task_prompt, msg="content should be capped at 500 chars")


if __name__ == "__main__":
    unittest.main()
