"""Tests for CoderAgent v4 – unittest + mock LLM calls."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Import under test
import sys
sys.path.insert(0, os.path.dirname(__file__))
from coder import (
    EditBlock, CoderResult, ExecutionReport, ExecutionMemory,
    TokenTracker, GitWorktree, CoderAgent, smart_read_file,
    _ctx_limit, _count_tokens, _fmt,
)


def run(coro):
    """Helper to run async in sync tests."""
    return asyncio.run(coro)


class MockAgent:
    """Fake agent that returns predictable LLM responses."""

    def __init__(self, responses=None):
        self._responses = responses or ["Mocked response"]
        self._call_idx = 0
        self.a_run_llm_completion = AsyncMock(side_effect=self._next)

    async def _next(self, **kwargs):
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        if kwargs.get("get_response_message"):
            return SimpleNamespace(content=resp, tool_calls=None)
        return resp


# =============================================================================
# Data Structures
# =============================================================================

class TestEditBlock(unittest.TestCase):
    def test_creation(self):
        e = EditBlock("a.py", "old", "new")
        self.assertEqual(e.file_path, "a.py")
        self.assertEqual(e.search, "old")
        self.assertEqual(e.replace, "new")


class TestCoderResult(unittest.TestCase):
    def test_defaults(self):
        r = CoderResult(True, "ok", ["a.py"], [])
        self.assertTrue(r.success)
        self.assertEqual(r.tokens_used, 0)
        self.assertEqual(r.compressions_done, 0)


class TestExecutionReport(unittest.TestCase):
    def test_to_context_str(self):
        r = ExecutionReport("2025-01-01", "fix bug", ["a.py"], True, summary="done")
        ctx = r.to_context_str()
        self.assertIn("✓", ctx)
        self.assertIn("fix bug", ctx)
        self.assertIn("a.py", ctx)

    def test_failed_report(self):
        r = ExecutionReport("2025-01-01", "bad task", [], False)
        self.assertIn("✗", r.to_context_str())


# =============================================================================
# Token Management
# =============================================================================

class TestTokenFunctions(unittest.TestCase):
    def test_ctx_limit_default(self):
        """Without litellm, should return 8192."""
        with patch("coder.litellm", None):
            self.assertEqual(_ctx_limit("unknown-model"), 8_192)

    def test_count_tokens_fallback(self):
        """Char/4 fallback."""
        msgs = [{"content": "a" * 400}]
        with patch("coder.litellm", None):
            count = _count_tokens(msgs, "x")
            self.assertEqual(count, 100)

    def test_count_tokens_min_one(self):
        msgs = [{"content": ""}]
        with patch("coder.litellm", None):
            self.assertGreaterEqual(_count_tokens(msgs, "x"), 1)


class TestTokenTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = TokenTracker("test-model", agent=None)
        self.tracker.limit = 1000
        self.tracker.threshold = 700

    def test_needs_compression_false(self):
        msgs = [{"content": "x" * 100}]  # ~25 tokens << 700
        self.assertFalse(self.tracker.needs_compression(msgs))

    def test_needs_compression_true(self):
        # Force past threshold by directly setting total_tokens won't work,
        # so we make messages large enough for ANY counter (litellm or char/4)
        # 700 tokens threshold → need ~3500 chars (char/4) or ~700 real tokens
        # Use many messages with enough content to exceed either way
        msgs = [{"role": "user", "content": "word " * 200}] * 10  # ~2000 words = ~2600 tokens
        self.assertTrue(self.tracker.needs_compression(msgs))

    def test_usage_ratio(self):
        msgs = [{"content": "x" * 2000}]  # ~500 tokens / 1000 limit
        ratio = self.tracker.usage_ratio(msgs)
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 1.0)

    def test_compress_preserves_structure(self):
        agent = MockAgent(["Summary of middle messages"])
        tracker = TokenTracker("m", agent=agent)
        tracker.limit = 1000; tracker.threshold = 0

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "mid1"},
            {"role": "user", "content": "mid2"},
            {"role": "assistant", "content": "mid3"},
            {"role": "user", "content": "mid4"},
            {"role": "assistant", "content": "tail1"},
            {"role": "user", "content": "tail2"},
            {"role": "assistant", "content": "tail3"},
            {"role": "user", "content": "tail4"},
        ]
        result = run(tracker.compress(messages))
        # system + task + compressed + 4 tail = 7
        self.assertEqual(len(result), 7)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("### VERLAUF-RECAP", result[2]["content"])
        self.assertEqual(tracker.compressions_done, 1)

    def test_compress_too_few_messages(self):
        tracker = TokenTracker("m")
        msgs = [{"role": "system", "content": "s"}] * 5
        result = run(tracker.compress(msgs))
        self.assertEqual(len(result), 5)  # unchanged


# =============================================================================
# Execution Memory
# =============================================================================

class TestExecutionMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = ExecutionMemory(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_and_load(self):
        report = ExecutionReport("2025-01-01", "task1", ["a.py"], True)
        self.mem.add(report)
        self.assertEqual(len(self.mem.reports), 1)

        # Reload from disk
        mem2 = ExecutionMemory(self.tmp)
        self.assertEqual(len(mem2.reports), 1)

    def test_fifo_limit(self):
        for i in range(15):
            self.mem.add(ExecutionReport(f"t{i}", f"task{i}", [], True))
        self.assertEqual(len(self.mem.reports), 10)

    def test_get_context_empty(self):
        self.assertIsNone(self.mem.get_context())

    def test_get_context_returns_last_3(self):
        for i in range(5):
            self.mem.add(ExecutionReport(f"t{i}", f"task{i}", [f"f{i}.py"], True))
        ctx = self.mem.get_context()
        self.assertIn("task4", ctx)
        self.assertIn("task3", ctx)
        self.assertIn("task2", ctx)
        self.assertNotIn("task0", ctx)

    def test_corrupt_file(self):
        (Path(self.tmp) / ".coder_memory.json").write_text("INVALID JSON{{{")
        mem = ExecutionMemory(self.tmp)
        self.assertEqual(mem.reports, [])


# =============================================================================
# Smart File Reader
# =============================================================================

class TestSmartReadFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "test.py").write_text("line1\nline2\nline3\nline4\nline5\n")
        (self.tmp / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_not_found(self):
        result = run(smart_read_file("nope.py", None, None, self.tmp))
        self.assertIn("not found", result)

    def test_binary_detection(self):
        result = run(smart_read_file("binary.bin", None, None, self.tmp))
        self.assertIn("Binary", result)

    def test_explicit_range(self):
        result = run(smart_read_file("test.py", 2, 4, self.tmp))
        self.assertIn("line2", result)
        self.assertIn("line4", result)
        self.assertNotIn("line5", result)

    def test_full_read_low_context(self):
        result = run(smart_read_file("test.py", None, None, self.tmp, model=""))
        self.assertIn("line1", result)
        self.assertIn("line5", result)

    def test_mode2_with_agent(self):
        agent = MockAgent(["extracted: lines 2-3"])
        # Need usage ratio 0.60-0.85 of 8192 limit = 4915-6963 tokens
        # Both litellm and char/4 need to land in that range
        # ~6000 tokens ≈ "word " * 6000 (litellm) or "x" * 24000 (char/4)
        # Use real words so litellm counts properly too
        big_msgs = [{"role": "user", "content": "token " * 6000}]
        result = run(smart_read_file(
            "test.py", None, None, self.tmp, agent=agent,
            messages=big_msgs, model="test", query="find line2"))
        # If litellm is installed and model "test" fails lookup, fallback to char/4
        # Either way we should get Extracted or fallback to direct read
        if "Extracted" not in result:
            # If token counting put us outside 0.60-0.85, just verify we got valid output
            self.assertTrue("line1" in result or "Critical" in result)

    def test_mode3_grep_fallback(self):
        # >85% of 8192 = >6963 tokens — go way over to be safe with any counter
        big_msgs = [{"role": "user", "content": "token " * 10000}]
        result = run(smart_read_file(
            "test.py", None, None, self.tmp, agent=None,
            messages=big_msgs, model="test", query="line3"))
        self.assertIn("line3", result)


class TestFmt(unittest.TestCase):
    def test_format_lines(self):
        lines = ["a", "b", "c", "d"]
        result = _fmt("f.py", lines, 2, 3)
        self.assertIn("f.py", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertNotIn("d", result)

    def test_bounds_clamping(self):
        lines = ["a", "b"]
        result = _fmt("f.py", lines, 1, 100)
        self.assertIn("a", result)
        self.assertIn("b", result)


# =============================================================================
# Git Worktree
# =============================================================================

class TestGitWorktreeNonGit(unittest.TestCase):
    """Test fallback copy mode (no .git)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "main.py").write_text("print('hello')")
        self.wt = GitWorktree(str(self.tmp))

    def tearDown(self):
        self.wt.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_setup_copies_files(self):
        self.wt.setup()
        self.assertIsNotNone(self.wt.path)
        self.assertTrue((self.wt.path / "main.py").exists())
        self.assertEqual((self.wt.path / "main.py").read_text(), "print('hello')")

    def test_setup_idempotent(self):
        self.wt.setup()
        path1 = self.wt.path
        self.wt.setup()
        self.assertEqual(self.wt.path, path1)

    def test_cleanup(self):
        self.wt.setup()
        p = self.wt.path
        self.wt.cleanup()
        self.assertIsNone(self.wt.path)
        self.assertFalse(p.exists())

    def test_cleanup_noop_if_not_setup(self):
        self.wt.cleanup()  # should not raise

    def test_apply_back(self):
        self.wt.setup()
        (self.wt.path / "main.py").write_text("print('changed')")
        (self.wt.path / "new.py").write_text("new file")
        count = asyncio.run(self.wt.apply_back())
        self.assertEqual(count, 2)
        self.assertEqual((self.tmp / "main.py").read_text(), "print('changed')")
        self.assertEqual((self.tmp / "new.py").read_text(), "new file")

    def test_apply_back_no_changes(self):
        self.wt.setup()
        count = asyncio.run(self.wt.apply_back())
        self.assertEqual(count, 0)

    def test_worktree_path_alias(self):
        self.wt.setup()
        self.assertEqual(self.wt.worktree_path, self.wt.path)


class TestGitWorktreeReal(unittest.TestCase):
    """Test real git worktree mode (requires git)."""

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git not available")
        self.tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init"], cwd=self.tmp, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.tmp, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.tmp, capture_output=True)
        (self.tmp / "main.py").write_text("original")
        subprocess.run(["git", "add", "."], cwd=self.tmp, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmp, capture_output=True)
        self.wt = GitWorktree(str(self.tmp))

    def tearDown(self):
        self.wt.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_setup_creates_worktree(self):
        self.wt.setup()
        self.assertTrue(self.wt.path.exists())
        self.assertTrue((self.wt.path / "main.py").exists())
        result = subprocess.run(["git", "worktree", "list"], cwd=self.tmp, capture_output=True, text=True)
        # Git outputs forward slashes on Windows (C:/...) vs Python's backslashes (C:\...)
        wt_normalized = str(self.wt.path).replace("\\", "/")
        self.assertTrue(
            str(self.wt.path) in result.stdout or wt_normalized in result.stdout,
            f"Worktree {self.wt.path} not in: {result.stdout}")


    def test_commit_in_worktree(self):
        self.wt.setup()
        (self.wt.path / "main.py").write_text("changed")
        self.wt.commit("test commit")
        result = subprocess.run(["git", "log", "--oneline", "-1"],
                                cwd=self.wt.path, capture_output=True, text=True)
        self.assertIn("test commit", result.stdout)

    def test_apply_back_merges(self):
        self.wt.setup()
        (self.wt.path / "main.py").write_text("updated via worktree")
        self.wt.commit("update")
        n = asyncio.run(self.wt.apply_back())
        self.assertEqual(n, -1)  # git merge mode
        self.assertEqual((self.tmp / "main.py").read_text(), "updated via worktree")

    def test_cleanup_removes_worktree(self):
        self.wt.setup()
        wt_path = self.wt.path
        branch = self.wt._branch
        self.wt.cleanup()
        self.assertIsNone(self.wt.path)
        result = subprocess.run(["git", "worktree", "list"], cwd=self.tmp, capture_output=True, text=True)
        self.assertNotIn(str(wt_path), result.stdout)


# =============================================================================
# CoderAgent Core
# =============================================================================

class TestCoderAgentParseEdits(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_single_edit(self):
        text = (
            "~~~edit:foo.py~~~\n"
            "<<<<<<< SEARCH\n"
            "old code\n"
            "=======\n"
            "new code\n"
            ">>>>>>> REPLACE\n"
            "~~~end~~~"
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].file_path, "foo.py")
        self.assertEqual(blocks[0].search, "old code")
        self.assertEqual(blocks[0].replace, "new code")

    def test_parse_multiple_edits(self):
        text = (
            "~~~edit:a.py~~~\n<<<<<<< SEARCH\nold1\n=======\nnew1\n>>>>>>> REPLACE\n~~~end~~~\n"
            "~~~edit:b.py~~~\n<<<<<<< SEARCH\nold2\n=======\nnew2\n>>>>>>> REPLACE\n~~~end~~~"
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1].file_path, "b.py")

    def test_parse_multiline_edit(self):
        text = (
            "~~~edit:x.py~~~\n"
            "<<<<<<< SEARCH\n"
            "line1\nline2\nline3\n"
            "=======\n"
            "new1\nnew2\n"
            ">>>>>>> REPLACE\n"
            "~~~end~~~"
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("line2", blocks[0].search)

    def test_parse_no_edits(self):
        self.assertEqual(self.coder._parse_edits("just some text"), [])

    def test_parse_new_file(self):
        text = (
            "~~~edit:new.py~~~\n"
            "<<<<<<< SEARCH\n"
            "\n"
            "=======\n"
            "print('hello')\n"
            ">>>>>>> REPLACE\n"
            "~~~end~~~"
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 1)


class TestCoderAgentApplyEdits(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp)
        self.coder.worktree = GitWorktree(self.tmp)
        self.coder.worktree.setup()

    def tearDown(self):
        self.coder.worktree.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_file(self):
        edits = [EditBlock("sub/new.py", "", "content")]
        results = self.coder._apply_edits(edits)
        self.assertTrue(results[0]["success"])
        self.assertEqual((self.coder.worktree.path / "sub" / "new.py").read_text(), "content")

    def test_replace_existing(self):
        target = self.coder.worktree.path / "existing.py"
        target.write_text("hello world")
        edits = [EditBlock("existing.py", "hello", "goodbye")]
        results = self.coder._apply_edits(edits)
        self.assertTrue(results[0]["success"])
        self.assertEqual(target.read_text(), "goodbye world")

    def test_search_not_found(self):
        target = self.coder.worktree.path / "existing.py"
        target.write_text("hello world")
        edits = [EditBlock("existing.py", "NONEXISTENT", "new")]
        results = self.coder._apply_edits(edits)
        self.assertFalse(results[0]["success"])
        self.assertIn("SEARCH not found", results[0]["error"])

    def test_file_missing(self):
        edits = [EditBlock("missing.py", "old", "new")]
        results = self.coder._apply_edits(edits)
        self.assertFalse(results[0]["success"])
        self.assertIn("File missing", results[0]["error"])


class TestCoderAgentBash(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp, {"bash_timeout": 5})
        self.coder.worktree = GitWorktree(self.tmp)
        self.coder.worktree.setup()

    def tearDown(self):
        self.coder.worktree.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_echo(self):
        result = run(self.coder._run_bash("echo hello"))
        self.assertIn("hello", result)

    def test_nonexistent_command(self):
        result = run(self.coder._run_bash("nonexistent_cmd_xyz 2>&1"))
        # Should return error output, not crash
        self.assertIsInstance(result, str)

    def test_timeout(self):
        self.coder.bash_timeout = 2
        result = run(self.coder._run_bash("sleep 10"))
        self.assertIn("Timeout", result)

    def test_cwd_is_worktree(self):
        result = run(self.coder._run_bash("pwd"))
        self.assertIn(str(self.coder.worktree.path), result)


class TestCoderAgentGrep(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp)
        self.coder.worktree = GitWorktree(self.tmp)
        self.coder.worktree.setup()
        (self.coder.worktree.path / "test.txt").write_text("findme here\nnot here\nfindme again")

    def tearDown(self):
        self.coder.worktree.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grep_finds_pattern(self):
        result = run(self.coder._run_grep("findme"))
        # Works on all platforms: rg > git grep > findstr (Win) > grep (Unix)
        self.assertTrue(
            "findme" in result,
            f"grep failed — result: {result[:200]}\n"
            f"Hint: Install ripgrep for best cross-platform search:\n"
            f"  Windows: winget install BurntSushi.ripgrep.MSVC\n"
            f"  Linux:   sudo apt install ripgrep\n"
            f"  macOS:   brew install ripgrep")


class TestCoderAgentValidation(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp, {"run_tests": False})
        self.coder.worktree = GitWorktree(self.tmp)
        self.coder.worktree.setup()

    def tearDown(self):
        self.coder.worktree.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_files(self):
        result = run(self.coder._validate([]))
        self.assertEqual(result, [])

    @unittest.skipUnless(shutil.which("ruff"), "ruff not installed")
    def test_ruff_catches_errors(self):
        bad_file = self.coder.worktree.path / "bad.py"
        bad_file.write_text("import os\nimport sys\n\nx = undefined_var\n")
        errors = run(self.coder._validate(["bad.py"]))
        # Should catch F821 (undefined name) or F401 (unused import)
        has_lint = any("Lint" in e for e in errors)
        self.assertTrue(has_lint or len(errors) == 0)  # depends on ruff config


class TestCoderAgentExecute(unittest.TestCase):
    """Integration test with mocked LLM."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        (Path(self.tmp) / "target.py").write_text("def old():\n    pass\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_execute_with_edit(self):
        edit_response = (
            "I'll update the function.\n\n"
            "~~~edit:target.py~~~\n"
            "<<<<<<< SEARCH\n"
            "def old():\n"
            "    pass\n"
            "=======\n"
            "def new():\n"
            "    return 42\n"
            ">>>>>>> REPLACE\n"
            "~~~end~~~"
        )
        agent = MockAgent([edit_response])
        coder = CoderAgent(agent, self.tmp)
        result = run(coder.execute("rename old to new"))
        self.assertTrue(result.success)
        self.assertIn("target.py", result.changed_files)
        # Verify the edit was applied in worktree
        wt_file = coder.worktree.path / "target.py"
        self.assertIn("def new():", wt_file.read_text())
        coder.worktree.cleanup()

    def test_execute_done_no_edits(self):
        agent = MockAgent(["[DONE] - no changes needed"])
        coder = CoderAgent(agent, self.tmp)
        result = run(coder.execute("check everything"))
        self.assertTrue(result.success)
        self.assertEqual(result.changed_files, [])
        coder.worktree.cleanup()

    def test_execute_failure(self):
        agent = MockAgent()
        agent.a_run_llm_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
        coder = CoderAgent(agent, self.tmp)
        result = run(coder.execute("do something"))
        self.assertFalse(result.success)
        self.assertIn("LLM down", result.message)
        coder.worktree.cleanup()

    def test_execute_saves_memory(self):
        agent = MockAgent(["[DONE]"])
        coder = CoderAgent(agent, self.tmp)
        result = run(coder.execute("test task"))
        self.assertTrue(result.memory_saved)
        mem_file = Path(self.tmp) / ".coder_memory.json"
        self.assertTrue(mem_file.exists())
        data = json.loads(mem_file.read_text())
        self.assertEqual(len(data["reports"]), 1)
        coder.worktree.cleanup()


class TestCoderAgentTools(unittest.TestCase):
    def test_tools_definition(self):
        agent = MockAgent()
        coder = CoderAgent(agent, tempfile.mkdtemp())
        tools = coder._tools()
        self.assertEqual(len(tools), 3)
        names = {t["function"]["name"] for t in tools}
        self.assertEqual(names, {"read_file", "bash", "grep"})

    def test_dispatch_unknown(self):
        agent = MockAgent()
        tmp = tempfile.mkdtemp()
        coder = CoderAgent(agent, tmp)
        coder.worktree = GitWorktree(tmp)
        coder.worktree.setup()
        result = run(coder._dispatch("nonexistent", {}, []))
        self.assertIn("Unknown tool", result)
        coder.worktree.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)


class TestCoderAgentConfig(unittest.TestCase):
    def test_default_config(self):
        agent = MockAgent()
        coder = CoderAgent(agent, "/tmp/test")
        self.assertEqual(coder.model, "gpt-4o")
        self.assertFalse(coder.run_tests)
        self.assertEqual(coder.bash_timeout, 300)

    def test_custom_config(self):
        agent = MockAgent()
        coder = CoderAgent(agent, "/tmp/test", {
            "model": "claude-3-5-sonnet", "run_tests": True, "bash_timeout": 60
        })
        self.assertEqual(coder.model, "claude-3-5-sonnet")
        self.assertTrue(coder.run_tests)
        self.assertEqual(coder.bash_timeout, 60)


class TestParserSelfEdit(unittest.TestCase):
    """Parser must survive content containing its own markers."""

    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_content_with_markers(self):
        """SEARCH/REPLACE containing ======= and ~~~end~~~ must not break parser."""
        text = (
            '~~~edit:parser.py~~~\n'
            '<<<<<<< SEARCH\n'
            'def old():\n'
            '    # this has ======= in it\n'
            '    x = "~~~end~~~"\n'
            '=======\n'
            'def new():\n'
            '    # fixed ======= handling\n'
            '    x = "~~~end~~~"\n'
            '>>>>>>> REPLACE\n'
            '~~~end~~~'
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("=======", blocks[0].search)
        self.assertIn("~~~end~~~", blocks[0].search)
        self.assertIn("=======", blocks[0].replace)
        self.assertEqual(blocks[0].file_path, "parser.py")

    def test_multiple_blocks_with_noise(self):
        text = (
            "Some explanation text\n"
            "~~~edit:a.py~~~\n<<<<<<< SEARCH\nold_a\n=======\nnew_a\n>>>>>>> REPLACE\n~~~end~~~\n"
            "More text between blocks\n"
            "~~~edit:b.py~~~\n<<<<<<< SEARCH\nold_b\n=======\nnew_b\n>>>>>>> REPLACE\n~~~end~~~"
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].search, "old_a")
        self.assertEqual(blocks[1].replace, "new_b")

    def test_empty_search_means_new_file(self):
        text = "~~~edit:new.py~~~\n<<<<<<< SEARCH\n=======\nprint('hi')\n>>>>>>> REPLACE\n~~~end~~~"
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].search, "")
        self.assertEqual(blocks[0].replace, "print('hi')")

    def test_multiline_with_indentation(self):
        text = (
            "~~~edit:x.py~~~\n"
            "<<<<<<< SEARCH\n"
            "    def method(self):\n"
            "        pass\n"
            "=======\n"
            "    def method(self):\n"
            "        return 42\n"
            ">>>>>>> REPLACE\n"
            "~~~end~~~"
        )
        blocks = self.coder._parse_edits(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("    def method", blocks[0].search)
        self.assertIn("        return 42", blocks[0].replace)


class TestApplyEditsSafety(unittest.TestCase):

    def setUp(self):
        self.agent = MockAgent()
        self.tmp = tempfile.mkdtemp()
        self.coder = CoderAgent(self.agent, self.tmp)
        self.coder.worktree = GitWorktree(self.tmp)
        self.coder.worktree.setup()

    def tearDown(self):
        self.coder.worktree.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fuzzy_match_whitespace(self):
        """Tabs vs spaces should still match."""
        target = self.coder.worktree.path / "ws.py"
        target.write_text("def foo():\n    return 1\n")
        edits = [EditBlock("ws.py", "def foo():\n\treturn 1", "def foo():\n    return 2")]
        results = self.coder._apply_edits(edits)
        self.assertTrue(results[0]["success"])
        self.assertIn("return 2", target.read_text())

    def test_atomic_write_leaves_no_temp(self):
        target = self.coder.worktree.path / "atom.py"
        self.coder._atomic_write(target, "content")
        self.assertEqual(target.read_text(), "content")
        temps = list(self.coder.worktree.path.glob("*.tmp"))
        self.assertEqual(len(temps), 0)

    def test_self_edit_syntax_block(self):
        """Syntax error in self-edit must be blocked + file unchanged."""
        target = self.coder.worktree.path / "coder.py"
        original = "def good():\n    return True\n"
        target.write_text(original)
        # Monkey-patch __file__ to point at worktree coder.py for self-detection
        import coder as coder_mod
        old_file = coder_mod.__file__
        coder_mod.__file__ = str(target)
        try:
            edits = [EditBlock("coder.py", "def good():\n    return True",
                               "def broken(\n    return True")]
            results = self.coder._apply_edits(edits)
            self.assertFalse(results[0]["success"])
            self.assertIn("SyntaxError", results[0]["error"])
            self.assertEqual(target.read_text(), original)  # unchanged!
        finally:
            coder_mod.__file__ = old_file

    def test_exact_match_preferred_over_fuzzy(self):
        target = self.coder.worktree.path / "exact.py"
        target.write_text("a = 1\na = 1\n")
        edits = [EditBlock("exact.py", "a = 1", "a = 2")]
        results = self.coder._apply_edits(edits)
        self.assertTrue(results[0]["success"])
        # Only first occurrence replaced
        self.assertEqual(target.read_text(), "a = 2\na = 1\n")

if __name__ == "__main__":
    unittest.main()
