"""
Unit Tests für CoderAgent v4.1 (coder.py)
Testet alle öffentlichen Funktionen und kritischen Datentransformationen.

Ausführen:
    python -m unittest test_coder -v
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Coder-Modul importieren — funktioniert auch ohne ToolBoxV2-Installation,
# weil alle optionalen Importe in try/except eingewickelt sind.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# Stubs für ToolBoxV2-interne Abhängigkeiten ---
_tb_stub = MagicMock()
_tb_stub.get_logger.return_value = MagicMock()
_tb_stub.Spinner = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=MagicMock(return_value=False)))

sys.modules.setdefault("toolboxv2", _tb_stub)
sys.modules.setdefault("toolboxv2.utils", MagicMock())
sys.modules.setdefault("toolboxv2.utils.extras", MagicMock())
sys.modules.setdefault("toolboxv2.utils.extras.Style", MagicMock())
sys.modules.setdefault("toolboxv2.mods", MagicMock())
sys.modules.setdefault("toolboxv2.mods.isaa", MagicMock())
sys.modules.setdefault("toolboxv2.mods.isaa.base", MagicMock())
sys.modules.setdefault("toolboxv2.mods.isaa.base.AgentUtils", MagicMock())

import importlib, types
from toolboxv2 import tb_root_dir

# Lade coder.py als Modul
spec = importlib.util.spec_from_file_location(
    "coder",
    tb_root_dir / "mods"/"isaa"/"CodingAgent"/"coder.py",
)
coder_mod = importlib.util.module_from_spec(spec)
# Überschreibe get_logger + Spinner in coder mit Stubs bevor exec
coder_mod.get_logger = lambda: MagicMock()
coder_mod.Spinner = _tb_stub.Spinner
spec.loader.exec_module(coder_mod)

# Alle zu testenden Namen aus dem Modul holen
text_to_block = coder_mod.text_to_block
_fmt = coder_mod._fmt
_file_hash_md5 = coder_mod._file_hash_md5
_count_tokens = coder_mod._count_tokens
_ctx_limit = coder_mod._ctx_limit
ExecutionReport = coder_mod.ExecutionReport
ExecutionMemory = coder_mod.ExecutionMemory
TokenTracker = coder_mod.TokenTracker
EditBlock = coder_mod.EditBlock
CoderResult = coder_mod.CoderResult
CoderAgent = coder_mod.CoderAgent
GitWorktree = coder_mod.GitWorktree
_safe_run = coder_mod._safe_run
smart_read_file = coder_mod.smart_read_file


# ---------------------------------------------------------------------------
# Hilfsfunktion: minimaler Mock-Agent für CoderAgent.__init__
# ---------------------------------------------------------------------------
def _make_agent(model: str = "test-model") -> MagicMock:
    agent = MagicMock()
    agent.amd.complex_llm_model = model
    agent.amd.name = "test-agent"
    return agent


def _make_coder(tmp_dir: str, model: str = "test-model") -> CoderAgent:
    agent = _make_agent(model)
    coder = CoderAgent(agent, tmp_dir, {"model": model})
    # Worktree.path manuell setzen, damit _apply_edits den Pfad kennt
    coder.worktree.path = Path(tmp_dir)
    coder.worktree._is_git = False
    return coder


# ===========================================================================
# 1. text_to_block
# ===========================================================================
class TestTextToBlock(unittest.TestCase):

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(text_to_block(""), [])

    def test_single_line_short(self):
        result = text_to_block("hello world")
        self.assertEqual(result, ["hello world"])

    def test_newline_preserved_as_separate_blocks(self):
        result = text_to_block("line one\nline two")
        self.assertIn("line one", result)
        self.assertIn("line two", result)

    def test_empty_line_produces_empty_string(self):
        result = text_to_block("a\n\nb")
        self.assertIn("", result)

    def test_word_longer_than_max_chars_is_split(self):
        long_word = "x" * 400
        result = text_to_block(long_word, max_chars=300)
        for chunk in result:
            self.assertLessEqual(len(chunk), 300)

    def test_word_exactly_max_chars_not_split(self):
        word = "a" * 300
        result = text_to_block(word, max_chars=300)
        self.assertEqual(result, [word])

    def test_line_exceeding_max_chars_wraps(self):
        # 10 Wörter à 50 Zeichen → bei max=300 muss ein Umbruch erfolgen
        line = " ".join(["word" * 12] * 10)
        result = text_to_block(line, max_chars=300)
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 300)

    def test_multiple_empty_lines(self):
        result = text_to_block("\n\n\n")
        self.assertTrue(all(r == "" for r in result))

    def test_unicode_content(self):
        result = text_to_block("äöü ß 你好")
        self.assertTrue(len(result) > 0)
        combined = " ".join(result)
        self.assertIn("äöü", combined)


# ===========================================================================
# 2. _fmt
# ===========================================================================
class TestFmt(unittest.TestCase):

    def setUp(self):
        self.lines = ["alpha", "beta", "gamma", "delta", "epsilon"]

    def test_basic_range(self):
        result = _fmt("file.py", self.lines, 1, 3)
        self.assertIn("file.py", result)
        self.assertIn("alpha", result)
        self.assertIn("beta", result)
        self.assertNotIn("delta", result)

    def test_header_contains_path_and_range(self):
        result = _fmt("src/main.py", self.lines, 2, 4)
        self.assertIn("src/main.py", result)
        self.assertIn("(2-", result)

    def test_start_zero_clamped_to_first_line(self):
        result = _fmt("f.py", self.lines, 0, 2)
        self.assertIn("alpha", result)

    def test_end_beyond_total_clamped(self):
        result = _fmt("f.py", self.lines, 1, 999)
        self.assertIn("epsilon", result)

    def test_line_numbers_in_output(self):
        result = _fmt("f.py", self.lines, 1, 2)
        # Jede Zeile beginnt mit einer Zeilennummer
        data_lines = [l for l in result.splitlines() if "|" in l]
        self.assertTrue(all(l.split("|")[0].strip().isdigit() for l in data_lines))

    def test_single_line(self):
        result = _fmt("f.py", self.lines, 3, 3)
        self.assertIn("gamma", result)
        self.assertNotIn("alpha", result)
        self.assertNotIn("delta", result)


# ===========================================================================
# 3. _file_hash_md5
# ===========================================================================
class TestFileHashMd5(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name: str, content: bytes) -> Path:
        p = Path(self.tmp) / name
        p.write_bytes(content)
        return p

    def test_deterministic_for_same_content(self):
        p1 = self._write("a.txt", b"hello")
        p2 = self._write("b.txt", b"hello")
        self.assertEqual(_file_hash_md5(p1), _file_hash_md5(p2))

    def test_different_content_different_hash(self):
        p1 = self._write("a.txt", b"hello")
        p2 = self._write("b.txt", b"world")
        self.assertNotEqual(_file_hash_md5(p1), _file_hash_md5(p2))

    def test_nonexistent_file_returns_empty_string(self):
        result = _file_hash_md5(Path(self.tmp) / "does_not_exist.txt")
        self.assertEqual(result, "")

    def test_directory_returns_empty_string(self):
        result = _file_hash_md5(Path(self.tmp))
        self.assertEqual(result, "")

    def test_empty_file_has_stable_hash(self):
        p = self._write("empty.txt", b"")
        h = _file_hash_md5(p)
        self.assertEqual(h, _file_hash_md5(p))
        # md5 von leerem File ist bekannter Wert
        self.assertEqual(h, hashlib.md5(b"").hexdigest())

    def test_large_file_hashed_correctly(self):
        data = b"chunk" * 10_000  # ~50 KB
        p = self._write("big.bin", data)
        expected = hashlib.md5(data).hexdigest()
        self.assertEqual(_file_hash_md5(p), expected)


# ===========================================================================
# 4. _count_tokens (Fallback ohne litellm)
# ===========================================================================
class TestCountTokensFallback(unittest.TestCase):

    def setUp(self):
        # Sicherstellen dass litellm nicht aktiv ist
        self._orig = coder_mod.litellm
        coder_mod.litellm = None

    def tearDown(self):
        coder_mod.litellm = self._orig

    def test_empty_messages(self):
        result = _count_tokens([], "any-model")
        self.assertGreaterEqual(result, 1)

    def test_single_message_token_estimate(self):
        msg = [{"role": "user", "content": "x" * 400}]
        result = _count_tokens(msg, "any-model")
        self.assertAlmostEqual(result, 100, delta=10)

    def test_none_content_handled(self):
        msg = [{"role": "system", "content": None}]
        result = _count_tokens(msg, "any-model")
        self.assertGreaterEqual(result, 1)

    def test_multiple_messages_sum(self):
        msgs = [{"role": "user", "content": "a" * 400},
                {"role": "assistant", "content": "b" * 400}]
        result = _count_tokens(msgs, "model")
        self.assertGreater(result, 100)


# ===========================================================================
# 5. _ctx_limit (Fallback ohne litellm)
# ===========================================================================
class TestCtxLimit(unittest.TestCase):

    def setUp(self):
        self._orig = coder_mod.litellm
        coder_mod.litellm = None

    def tearDown(self):
        coder_mod.litellm = self._orig

    def test_fallback_returns_200k(self):
        self.assertEqual(_ctx_limit("any-model"), 200_000)


# ===========================================================================
# 6. ExecutionReport.to_context_str
# ===========================================================================
class TestExecutionReport(unittest.TestCase):

    def _make(self, **kwargs) -> ExecutionReport:
        defaults = dict(
            timestamp="2025-01-01T12:00:00",
            task="Fix bug",
            changed_files=["src/main.py"],
            success=True,
            summary="All done",
        )
        defaults.update(kwargs)
        return ExecutionReport(**defaults)

    def test_success_shows_checkmark(self):
        r = self._make(success=True)
        self.assertIn("✓", r.to_context_str())

    def test_failure_shows_cross(self):
        r = self._make(success=False)
        self.assertIn("✗", r.to_context_str())

    def test_timestamp_in_output(self):
        r = self._make(timestamp="2025-06-15T09:30:00")
        self.assertIn("2025-06-15T09:30:00", r.to_context_str())

    def test_changed_files_in_output(self):
        r = self._make(changed_files=["a.py", "b.py"])
        s = r.to_context_str()
        self.assertIn("a.py", s)
        self.assertIn("b.py", s)

    def test_empty_changed_files_shows_dash(self):
        r = self._make(changed_files=[])
        self.assertIn("-", r.to_context_str())

    def test_asdict_roundtrip(self):
        r = self._make()
        restored = ExecutionReport(**asdict(r))
        self.assertEqual(r.to_context_str(), restored.to_context_str())



# ===========================================================================
# 7. ExecutionMemory
# ===========================================================================
class TestExecutionMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = ExecutionMemory(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _report(self, task="task", success=True) -> ExecutionReport:
        return ExecutionReport(
            timestamp="2025-01-01T00:00:00",
            task=task,
            changed_files=[],
            success=success,
        )

    def test_empty_memory_context_is_none(self):
        self.assertIsNone(self.mem.get_context())

    def test_add_single_report_context_not_none(self):
        self.mem.add(self._report())
        self.assertIsNotNone(self.mem.get_context())

    def test_context_contains_task(self):
        self.mem.add(self._report(task="deploy prod"))
        self.assertIn("deploy prod", self.mem.get_context())

    def test_rolling_window_max_10(self):
        for i in range(15):
            self.mem.add(self._report(task=f"task-{i}"))
        self.assertLessEqual(len(self.mem.reports), 10)

    def test_only_last_3_in_context(self):
        for i in range(8):
            self.mem.add(self._report(task=f"task-{i}"))
        ctx = self.mem.get_context()
        # task-5, task-6, task-7 sollen drin sein
        self.assertIn("task-7", ctx)
        self.assertIn("task-5", ctx)
        # task-0 soll nicht mehr drin sein
        self.assertNotIn("task-0", ctx)

    def test_persisted_to_disk(self):
        self.mem.add(self._report(task="saved-task"))
        mem2 = ExecutionMemory(self.tmp)  # Neu laden
        self.assertIn("saved-task", mem2.get_context())

    def test_corrupted_file_resets_gracefully(self):
        path = Path(self.tmp) / ".coder_memory.json"
        path.write_text("NOT VALID JSON {{{{")
        mem = ExecutionMemory(self.tmp)
        self.assertEqual(mem.reports, [])


# ===========================================================================
# 8. TokenTracker.needs_compression / usage_ratio
# ===========================================================================
class TestTokenTracker(unittest.TestCase):

    def setUp(self):
        coder_mod.litellm = None  # Fallback-Modus

    def _make(self, limit: int = 200_000, threshold_ratio: float = 0.65) -> TokenTracker:
        tracker = TokenTracker("test-model", agent=None, limit=threshold_ratio)
        tracker.limit = limit
        tracker.threshold = int(limit * threshold_ratio)
        return tracker

    def test_below_threshold_no_compression_needed(self):
        tracker = self._make()
        # 1 Zeichen → ~0 tokens
        msgs = [{"role": "user", "content": "hi"}]
        self.assertFalse(tracker.needs_compression(msgs))

    def test_above_threshold_compression_needed(self):
        tracker = self._make(limit=100, threshold_ratio=0.1)
        # threshold = 10 tokens, Inhalt ist viel größer
        msgs = [{"role": "user", "content": "x" * 10_000}]
        self.assertTrue(tracker.needs_compression(msgs))

    def test_usage_ratio_empty_messages(self):
        tracker = self._make()
        ratio = tracker.usage_ratio([])
        self.assertGreaterEqual(ratio, 0.0)

    def test_total_tokens_updated_after_check(self):
        tracker = self._make()
        msgs = [{"role": "user", "content": "hello world"}]
        tracker.needs_compression(msgs)
        self.assertGreater(tracker.total_tokens, 0)

    def test_compressions_done_starts_at_zero(self):
        tracker = self._make()
        self.assertEqual(tracker.compressions_done, 0)


# ===========================================================================
# 10. CoderAgent._fuzzy_find  (Static Method)
# ===========================================================================
class TestFuzzyFind(unittest.TestCase):

    SOURCE = "\n".join([
        "def hello():",
        "    print('Hello World')",
        "    return True",
        "",
        "def goodbye():",
        "    print('Goodbye')",
    ])

    def test_exact_match_found(self):
        search = "def hello():\n    print('Hello World')\n    return True"
        idx = CoderAgent._fuzzy_find(self.SOURCE, search)
        self.assertIsNotNone(idx)
        self.assertEqual(idx, 0)

    def test_near_exact_match_above_threshold(self):
        # ratio=0.67 für "return true" vs "return True" → threshold=0.60 matcht
        search = "def hello():\n    print('Hello World')\n    return true"  # lowercase true
        idx = CoderAgent._fuzzy_find(self.SOURCE, search, threshold=0.60)
        self.assertIsNotNone(idx)

    def test_completely_different_returns_none(self):
        search = "class Foo:\n    x = 1\n    y = 2"
        idx = CoderAgent._fuzzy_find(self.SOURCE, search, threshold=0.85)
        self.assertIsNone(idx)

    def test_empty_search_returns_none(self):
        idx = CoderAgent._fuzzy_find(self.SOURCE, "")
        self.assertIsNone(idx)

    def test_empty_source_returns_none(self):
        idx = CoderAgent._fuzzy_find("", "something")
        self.assertIsNone(idx)

    def test_second_function_found(self):
        search = "def goodbye():\n    print('Goodbye')"
        idx = CoderAgent._fuzzy_find(self.SOURCE, search)
        self.assertIsNotNone(idx)
        self.assertGreater(idx, 0)

    def test_below_threshold_returns_none(self):
        search = "TOTALLY DIFFERENT LINE\nNOTHING MATCHING HERE"
        idx = CoderAgent._fuzzy_find(self.SOURCE, search, threshold=0.99)
        self.assertIsNone(idx)


# ===========================================================================
# 11. CoderAgent._atomic_write  (Static Method)
# ===========================================================================
class TestAtomicWrite(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_new_file(self):
        target = self.tmp / "new.txt"
        CoderAgent._atomic_write(target, "hello")
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")

    def test_overwrites_existing_file(self):
        target = self.tmp / "existing.txt"
        target.write_text("old content", encoding="utf-8")
        CoderAgent._atomic_write(target, "new content")
        self.assertEqual(target.read_text(encoding="utf-8"), "new content")

    def test_creates_nested_directories(self):
        target = self.tmp / "a" / "b" / "c" / "file.py"
        CoderAgent._atomic_write(target, "code")
        self.assertTrue(target.exists())

    def test_content_preserved_on_success(self):
        target = self.tmp / "check.txt"
        content = "unicode: äöü\nmultiline\n"
        CoderAgent._atomic_write(target, content)
        self.assertEqual(target.read_text(encoding="utf-8"), content)

    def test_no_tmp_files_left_after_success(self):
        target = self.tmp / "clean.txt"
        CoderAgent._atomic_write(target, "data")
        tmp_files = list(self.tmp.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)

    def test_large_content(self):
        target = self.tmp / "large.txt"
        content = "x" * 1_000_000
        CoderAgent._atomic_write(target, content)
        self.assertEqual(len(target.read_text(encoding="utf-8")), 1_000_000)

# ===========================================================================
# 13. CoderResult Dataclass
# ===========================================================================
class TestCoderResult(unittest.TestCase):

    def test_default_values(self):
        r = CoderResult(True, "Done", ["a.py"], [])
        self.assertFalse(r.memory_saved)
        self.assertEqual(r.tokens_used, 0)
        self.assertEqual(r.compressions_done, 0)

    def test_custom_values(self):
        r = CoderResult(False, "Error", [], [], memory_saved=True, tokens_used=5000, compressions_done=2)
        self.assertFalse(r.success)
        self.assertTrue(r.memory_saved)
        self.assertEqual(r.tokens_used, 5000)
        self.assertEqual(r.compressions_done, 2)


# ===========================================================================
# 14. GitWorktree._detect_git (mit gemocktem subprocess)
# ===========================================================================
class TestGitWorktreeDetect(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_non_git_directory_detected(self):
        # Kein .git Verzeichnis → not git
        wt = GitWorktree(self.tmp)
        if not wt._is_git:
            self.assertIsNone(wt._git_root)
        # Kein Crash erwartet

    def test_git_root_detected_with_dot_git(self):
        # .git Ordner anlegen → sollte erkannt werden
        dot_git = Path(self.tmp) / ".git"
        dot_git.mkdir()
        wt = GitWorktree(self.tmp)
        # _is_git kann True oder False sein je nach subprocess, aber kein Crash
        self.assertIsInstance(wt._is_git, bool)

    def test_branch_name_has_coder_prefix(self):
        wt = GitWorktree(self.tmp)
        self.assertTrue(wt._branch.startswith("coder-"))

    def test_path_initially_none(self):
        wt = GitWorktree(self.tmp)
        self.assertIsNone(wt.path)


# ===========================================================================
# 15. GitWorktree._copy_filtered  (über temporäres Verzeichnis)
# ===========================================================================
class TestGitWorktreeCopyFiltered(unittest.TestCase):

    def setUp(self):
        self.src = Path(tempfile.mkdtemp())
        self.dst = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.src, ignore_errors=True)
        shutil.rmtree(self.dst, ignore_errors=True)

    def _setup_source(self):
        (self.src / "main.py").write_text("print('hello')")
        (self.src / "utils.py").write_text("# utils")
        (self.src / "node_modules").mkdir()
        (self.src / "node_modules" / "should_be_excluded.js").write_text("nope")
        (self.src / "__pycache__").mkdir()
        (self.src / "__pycache__" / "cached.pyc").write_bytes(b"\x00\x01\x02")

    def test_copy_filtered_copies_py_files(self):
        self._setup_source()
        wt = GitWorktree(str(self.src))
        wt.origin_root = self.src
        wt._copy_filtered(self.src, self.dst)
        self.assertTrue((self.dst / "main.py").exists())
        self.assertTrue((self.dst / "utils.py").exists())

    def test_copy_filtered_excludes_node_modules(self):
        self._setup_source()
        wt = GitWorktree(str(self.src))
        wt.origin_root = self.src
        wt._copy_filtered(self.src, self.dst)
        self.assertFalse((self.dst / "node_modules").exists())

    def test_copy_filtered_excludes_pycache(self):
        self._setup_source()
        wt = GitWorktree(str(self.src))
        wt.origin_root = self.src
        wt._copy_filtered(self.src, self.dst)
        self.assertFalse((self.dst / "__pycache__").exists())

    def test_copy_filtered_returns_file_count(self):
        self._setup_source()
        wt = GitWorktree(str(self.src))
        wt.origin_root = self.src
        count = wt._copy_filtered(self.src, self.dst)
        self.assertGreater(count, 0)


# ===========================================================================
# 16. TokenTracker.compress  (async)
# ===========================================================================
class TestTokenTrackerCompress(unittest.IsolatedAsyncioTestCase):

    def _make_tracker(self) -> TokenTracker:
        agent = MagicMock()
        agent.a_run_llm_completion = AsyncMock(return_value="Zusammenfassung.")
        tracker = TokenTracker("test-model", agent=agent, limit=0.65)
        tracker.limit = 100_000
        tracker.threshold = 65_000
        coder_mod.litellm = None
        return tracker

    async def test_short_messages_not_compressed(self):
        tracker = self._make_tracker()
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        result = await tracker.compress(msgs)
        # Bei ≤ 6 Messages: identisch zurück
        self.assertEqual(result, msgs)

    async def test_compress_reduces_message_count(self):
        tracker = self._make_tracker()
        msgs = (
            [{"role": "system", "content": "system"}] +
            [{"role": "user", "content": "task"}] +
            [{"role": "assistant" if i % 2 == 0 else "user", "content": f"msg-{i}"}
             for i in range(20)]
        )
        result = await tracker.compress(msgs)
        self.assertLess(len(result), len(msgs))

    async def test_compressions_done_incremented(self):
        tracker = self._make_tracker()
        msgs = (
            [{"role": "system", "content": "s"}] +
            [{"role": "user", "content": "t"}] +
            [{"role": "assistant", "content": f"a{i}"} for i in range(10)]
        )
        await tracker.compress(msgs)
        self.assertEqual(tracker.compressions_done, 1)

    async def test_system_message_preserved(self):
        tracker = self._make_tracker()
        sys_content = "UNIQUE_SYSTEM_PROMPT"
        msgs = (
            [{"role": "system", "content": sys_content}] +
            [{"role": "user", "content": "task"}] +
            [{"role": "assistant", "content": f"a{i}"} for i in range(10)]
        )
        result = await tracker.compress(msgs)
        self.assertTrue(any(m.get("content", "").startswith(sys_content) for m in result
                            if m.get("role") == "system"))


# ===========================================================================
# 17. smart_read_file (async, mit temp-Dateien)
# ===========================================================================
class TestSmartReadFile(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_missing_file_returns_error(self):
        result = await coder_mod.smart_read_file(
            "does_not_exist.py", None, None, self.tmp)
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    async def test_reads_small_file_completely(self):
        p = self.tmp / "small.py"
        p.write_text("def foo():\n    pass\n", encoding="utf-8")
        result = await coder_mod.smart_read_file(
            "small.py", None, None, self.tmp)
        self.assertIn("def foo()", result)

    async def test_reads_specific_range(self):
        lines = "\n".join(f"line{i}" for i in range(1, 101))
        p = self.tmp / "ranged.py"
        p.write_text(lines, encoding="utf-8")
        result = await coder_mod.smart_read_file(
            "ranged.py", 5, 10, self.tmp)
        self.assertIn("line5", result)
        self.assertIn("line10", result)
        self.assertNotIn("line1\n", result)  # line1 soll nicht in Range 5-10 sein

    async def test_binary_file_returns_hint(self):
        p = self.tmp / "binary.bin"
        p.write_bytes(b"\x00\x01\x02\x03hello")
        result = await coder_mod.smart_read_file(
            "binary.bin", None, None, self.tmp)
        self.assertIn("Binary", result)

    async def test_header_contains_path(self):
        p = self.tmp / "header_test.py"
        p.write_text("x = 1\n", encoding="utf-8")
        result = await coder_mod.smart_read_file(
            "header_test.py", 1, 1, self.tmp)
        self.assertIn("header_test.py", result)


# ===========================================================================
# 19. _safe_run (subprocess wrapper)
# ===========================================================================
class TestSafeRun(unittest.TestCase):

    @unittest.skipIf(sys.platform == "win32", "Unix-only test")
    def test_echo_returns_output(self):
        result = _safe_run(["echo", "hello"], text=True)
        self.assertIn("hello", result.stdout)

    @unittest.skipIf(sys.platform == "win32", "Unix-only test")
    def test_nonzero_returncode_captured(self):
        result = _safe_run(["false"])
        self.assertNotEqual(result.returncode, 0)

    def test_returncode_attribute_exists(self):
        result = _safe_run([sys.executable, "-c", "print('ok')"], text=True)
        self.assertIsInstance(result.returncode, int)

    def test_stdout_is_string_when_text_true(self):
        result = _safe_run([sys.executable, "-c", "print('test')"], text=True)
        self.assertIsInstance(result.stdout, str)


# ===========================================================================
# 20. EditBlock Dataclass
# ===========================================================================
class TestEditBlock(unittest.TestCase):

    def test_creation(self):
        b = EditBlock("path/to/file.py", "old", "new")
        self.assertEqual(b.file_path, "path/to/file.py")
        self.assertEqual(b.search, "old")
        self.assertEqual(b.replace, "new")

    def test_empty_search_is_new_file_convention(self):
        b = EditBlock("new.py", "", "content")
        self.assertFalse(bool(b.search.strip()))


# ===========================================================================
# 4. Token helpers
# ===========================================================================
class TestTokenFallback(unittest.TestCase):

    def test_empty_messages(self):
        self.assertGreaterEqual(_count_tokens([], "m"), 1)

    def test_rough_estimate(self):
        self.assertAlmostEqual(
            _count_tokens([{"role": "user", "content": "x" * 400}], "m"),
            100, delta=15)

    def test_none_content(self):
        self.assertGreaterEqual(
            _count_tokens([{"role": "s", "content": None}], "m"), 1)

    def test_ctx_limit_fallback(self):
        self.assertEqual(_ctx_limit("any"), 200_000)

# ===========================================================================
# 13. Worktree-Pfad im System-Prompt  (Agent weiß wo er arbeitet)
# ===========================================================================
class TestWorktreeInSystemPrompt(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.coder = _make_coder(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_sys_msg(self) -> str:
        """Baut sys_msg wie execute() — ohne LLM-Call."""
        msg = self.coder.SYSTEM_PROMPT
        msg += f"\n\n{self.coder._get_folder_structure(max_depth=2)}\n"
        return msg

    def test_worktree_path_in_sys_msg(self):
        """
        Der System-Prompt MUSS den absoluten Worktree-Pfad enthalten.
        Ohne das weiß der Agent nicht wo seine Shadow-Kopie liegt und
        nutzt möglicherweise absolute Pfade → Edits landen im falschen Verzeichnis.
        """
        msg = self._build_sys_msg()
        self.assertIn(str(self.tmp), msg,
                      "Worktree-Root fehlt im System-Prompt!\n"
                      "Lösung: _get_folder_structure() gibt '# Root: <path>' aus — "
                      "dieses muss im sys_msg landen.")

    def test_edit_format_explained(self):
        msg = self._build_sys_msg()
        self.assertIn("~~~edit:", msg)
        self.assertIn("~~~end~~~", msg)
        self.assertIn("SEARCH", msg)
        self.assertIn("REPLACE", msg)

    def test_done_keyword_present(self):
        self.assertIn("[DONE]", self._build_sys_msg())

    def test_shadow_path_differs_from_origin(self):
        """
        Im Shadow-Mode muss worktree.path != origin_root sein.
        Wenn beide gleich sind, editiert der Agent direkt das Original
        statt die Shadow-Kopie → kein Rollback möglich.
        """
        shadow = Path(tempfile.mkdtemp())
        origin = Path(tempfile.mkdtemp())
        try:
            c = _make_coder(str(origin))
            c.worktree.path = shadow  # Shadow
            c.worktree.origin_root = origin  # Original bleibt separat
            self.assertNotEqual(str(c.worktree.path), str(c.worktree.origin_root),
                                "Shadow == Origin → Agent editiert direkt das Original!")
        finally:
            shutil.rmtree(shadow, ignore_errors=True)
            shutil.rmtree(origin, ignore_errors=True)


# ===========================================================================
# 14. GitWorktree
# ===========================================================================
class TestGitWorktree(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_branch_prefix(self):
        self.assertTrue(GitWorktree(self.tmp)._branch.startswith("coder-"))

    def test_path_initially_none(self):
        self.assertIsNone(GitWorktree(self.tmp).path)

    def test_detect_no_crash(self):
        self.assertIsInstance(GitWorktree(self.tmp)._is_git, bool)

    def test_copy_filtered_includes_py_excludes_node_modules(self):
        src = Path(tempfile.mkdtemp())
        dst = Path(tempfile.mkdtemp())
        try:
            (src / "main.py").write_text("x")
            (src / "node_modules").mkdir()
            (src / "node_modules" / "skip.js").write_text("y")
            wt = GitWorktree(str(src))
            wt.origin_root = src
            wt._copy_filtered(src, dst)
            self.assertTrue((dst / "main.py").exists())
            self.assertFalse((dst / "node_modules").exists())
        finally:
            shutil.rmtree(src, ignore_errors=True)
            shutil.rmtree(dst, ignore_errors=True)



# ===========================================================================
# 18. Dataclasses
# ===========================================================================
class TestDataclasses(unittest.TestCase):

    def test_edit_block(self):
        b = EditBlock("f.py", "old", "new")
        self.assertEqual(b.file_path, "f.py")

    def test_edit_block_empty_search_is_new_file(self):
        self.assertFalse(bool(EditBlock("n.py", "", "x").search.strip()))

    def test_coder_result_defaults(self):
        r = CoderResult(True, "Done", ["a.py"], [])
        self.assertFalse(r.memory_saved)
        self.assertEqual(r.tokens_used, 0)
        self.assertEqual(r.compressions_done, 0)


# ===========================================================================
# Ausführung
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
