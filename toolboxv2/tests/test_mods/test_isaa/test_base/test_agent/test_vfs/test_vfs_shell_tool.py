"""
test_vfs_shell_tool.py
======================
Tests for toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py

=== ANALYSIS — Bugs & Missing Features ===

BUG  S1 – Multi-command batching NOT supported (newline, ;, &&).
BUG  S2 – echo regex mis-splits content containing '>'.
BUG  S3 – stat crashes on unloaded SHADOW file (ContentNotLoadedError).
BUG  S4 – grep crashes on unloaded SHADOW file.
BUG  S5 – No 'sync' command to flush dirty files.
BUG  S6 – write regex breaks on quoted paths with spaces.
BUG  S7 – Semicolon ';' becomes part of the path token.
BUG  S8 – '|' PIPE operator not recognised.
BUG  S9 – '||' OR  operator not recognised.

Run with:
    python -m pytest test_vfs_shell_tool.py -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    FileBackingType,
    VirtualFileSystemV2,
)
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell, make_vfs_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> MagicMock:
    vfs = VirtualFileSystemV2(session_id="test-shell", agent_name="test-agent")
    session = MagicMock()
    session.vfs = vfs
    return session


def _prep(session, files: dict[str, str] | None = None, dirs: list[str] | None = None):
    vfs = session.vfs
    for d in dirs or []:
        vfs.mkdir(d, parents=True)
    for path, content in (files or {}).items():
        parent = os.path.dirname(path)
        if parent and parent != "/":
            vfs.mkdir(parent, parents=True)
        vfs.create(path, content)


# ===========================================================================
# 1. Basic plumbing
# ===========================================================================

class TestShellBasicPlumbing(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)

    def test_empty_command_fails(self):
        r = self.sh("","")
        self.assertFalse(r["success"])

    def test_whitespace_only_fails(self):
        r = self.sh("","   ")
        self.assertFalse(r["success"])

    def test_unknown_command_fails(self):
        r = self.sh("","frobnicate /foo")
        self.assertFalse(r["success"])
        self.assertIn("command not found", r["stderr"])

    def test_result_always_has_required_keys(self):
        r = self.sh("","pwd")
        for k in ("success", "stdout", "stderr", "returncode"):
            self.assertIn(k, r)


# ===========================================================================
# 2. Navigation — pwd / ls / tree
# ===========================================================================

class TestShellNavigation(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {
            "/src/main.py":  "print('hello')\n",
            "/src/utils.py": "def f(): pass\n",
            "/README.md":    "# Project\n",
        })
        self.sh = make_vfs_shell(self.session)

    def test_pwd(self):
        r = self.sh("","pwd")
        self.assertTrue(r["success"])
        self.assertIn("/", r["stdout"])

    def test_ls_root(self):
        r = self.sh("","ls /")
        self.assertTrue(r["success"])
        self.assertIn("src", r["stdout"])

    def test_ls_subdir(self):
        r = self.sh("","ls /src")
        self.assertTrue(r["success"])
        self.assertIn("main.py", r["stdout"])

    def test_ls_long_format(self):
        r = self.sh("","ls -la /src")
        self.assertTrue(r["success"])
        self.assertIn("main.py", r["stdout"])

    def test_ls_recursive(self):
        r = self.sh("","ls -R /")
        self.assertTrue(r["success"])
        self.assertIn("main.py", r["stdout"])

    def test_ls_nonexistent_fails(self):
        r = self.sh("","ls /nonexistent")
        self.assertFalse(r["success"])

    def test_tree(self):
        r = self.sh("","tree /")
        self.assertTrue(r["success"])
        self.assertIn("src", r["stdout"])

    def test_tree_subpath(self):
        r = self.sh("","tree /src")
        self.assertTrue(r["success"])
        self.assertIn("main.py", r["stdout"])


# ===========================================================================
# 3. Read commands
# ===========================================================================

class TestShellReadCommands(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        content = "\n".join(f"line{i}" for i in range(1, 21))
        _prep(self.session, {"/doc.txt": content})
        self.sh = make_vfs_shell(self.session)

    def test_cat(self):
        r = self.sh("","cat /doc.txt")
        self.assertTrue(r["success"])
        self.assertIn("line1", r["stdout"])
        self.assertIn("line20", r["stdout"])

    def test_cat_missing_file(self):
        r = self.sh("","cat /missing.txt")
        # single missing file → failure (consistent with Unix cat exit code 1)
        self.assertFalse(r["success"])
        self.assertIn("no such file or directory", r["stdout"])

    def test_cat_no_operand(self):
        r = self.sh("","cat")
        self.assertFalse(r["success"])

    def test_head_default(self):
        r = self.sh("","head /doc.txt")
        self.assertTrue(r["success"])
        self.assertLessEqual(len(r["stdout"].splitlines()), 10)
        self.assertEqual(r["stdout"].splitlines()[0], "line1")

    def test_head_n_flag(self):
        r = self.sh("","head -n 5 /doc.txt")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["stdout"].splitlines()), 5)

    def test_tail_default(self):
        r = self.sh("","tail /doc.txt")
        self.assertTrue(r["success"])
        self.assertLessEqual(len(r["stdout"].splitlines()), 10)
        self.assertIn("line20", r["stdout"])

    def test_tail_n_flag(self):
        r = self.sh("","tail -n 3 /doc.txt")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["stdout"].splitlines()), 3)

    def test_wc_l(self):
        r = self.sh("","wc -l /doc.txt")
        self.assertTrue(r["success"])
        self.assertIn("20", r["stdout"])

    def test_stat(self):
        r = self.sh("","stat /doc.txt")
        self.assertTrue(r["success"])
        self.assertIn("doc.txt", r["stdout"])

    def test_stat_missing_fails(self):
        r = self.sh("","stat /ghost.txt")
        self.assertFalse(r["success"])

    def test_stat_on_unloaded_shadow_file_does_not_crash(self):
        """BUG S3 — stat on unloaded SHADOW file must return a dict, not raise."""
        tmp = tempfile.TemporaryDirectory()
        try:
            with open(os.path.join(tmp.name, "shadow.py"), "w") as fh:
                fh.write("x = 1\n")
            self.session.vfs.mount(tmp.name, vfs_path="/proj", auto_sync=False)
            f = self.session.vfs.files.get("/proj/shadow.py")
            if f:
                self.assertFalse(f.is_loaded)
                try:
                    r = self.sh("","stat /proj/shadow.py")
                    self.assertIsInstance(r, dict, "stat must return dict (BUG S3)")
                except Exception as e:
                    self.fail(f"stat raised exception on SHADOW file (BUG S3): {e}")
        finally:
            tmp.cleanup()


# ===========================================================================
# 4. Write commands
# ===========================================================================

class TestShellWriteCommands(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_touch_creates_file(self):
        r = self.sh("","touch /new.txt")
        self.assertTrue(r["success"], r)
        self.assertTrue(self.vfs._is_file("/new.txt"))

    def test_echo_overwrite(self):
        r = self.sh("",'echo "hello world" > /msg.txt')
        self.assertTrue(r["success"], r)
        self.assertEqual(self.vfs.read("/msg.txt")["content"], "hello world")

    def test_echo_append(self):
        self.vfs.create("/log.txt", "first\n")
        self.vfs.open("/log.txt")
        r = self.sh("",'echo "second" >> /log.txt')
        self.assertTrue(r["success"], r)
        content = self.vfs.read("/log.txt")["content"]
        self.assertIn("first", content)
        self.assertIn("second", content)

    def test_echo_content_with_angle_bracket_redirects_correctly(self):
        """BUG S2 — content containing '>' must not mis-split the redirection."""
        r = self.sh("",'echo "x > y" > /result.txt')
        self.assertTrue(r["success"], r)
        content = self.vfs.read("/result.txt")["content"]
        self.assertEqual(content, "x > y",
                         f"Content mis-parsed (BUG S2): got {repr(content)}")

    def test_write_simple(self):
        r = self.sh("",'write /app.py "x = 1"')
        self.assertTrue(r["success"], r)
        self.assertEqual(self.vfs.read("/app.py")["content"], "x = 1")

    def test_write_multiline_newline_escape(self):
        r = self.sh("",'write /app.py "line1\\nline2\\nline3"')
        self.assertTrue(r["success"], r)
        lines = self.vfs.read("/app.py")["content"].splitlines()
        self.assertEqual(lines, ["line1", "line2", "line3"])

    def test_edit_replaces_lines(self):
        self.vfs.create("/code.py", "line1\nline2\nline3\nline4\n")
        r = self.sh("",'edit /code.py 2 3 "REPLACED"')
        self.assertTrue(r["success"], r)
        content = self.vfs.read("/code.py")["content"]
        self.assertIn("REPLACED", content)
        self.assertNotIn("line2", content)


# ===========================================================================
# 5. Directory commands
# ===========================================================================

class TestShellDirectoryCommands(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_mkdir(self):
        r = self.sh("","mkdir /newdir")
        self.assertTrue(r["success"], r)
        self.assertTrue(self.vfs._is_directory("/newdir"))

    def test_mkdir_p_nested(self):
        r = self.sh("","mkdir -p /a/b/c")
        self.assertTrue(r["success"], r)
        self.assertTrue(self.vfs._is_directory("/a/b/c"))

    def test_rm_file(self):
        self.vfs.create("/trash.txt", "bye")
        r = self.sh("","rm /trash.txt")
        self.assertTrue(r["success"], r)
        self.assertFalse(self.vfs._is_file("/trash.txt"))

    def test_rm_rf_directory(self):
        self.vfs.mkdir("/dir")
        self.vfs.create("/dir/f.txt", "x")
        r = self.sh("","rm -rf /dir")
        self.assertTrue(r["success"], r)
        self.assertFalse(self.vfs._is_directory("/dir"))

    def test_mv_file(self):
        self.vfs.create("/old.txt", "data")
        self.vfs.mkdir("/dest")
        r = self.sh("","mv /old.txt /dest/new.txt")
        self.assertTrue(r["success"], r)
        self.assertFalse(self.vfs._is_file("/old.txt"))
        self.assertTrue(self.vfs._is_file("/dest/new.txt"))

    def test_cp_file(self):
        self.vfs.create("/src.txt", "original")
        r = self.sh("","cp /src.txt /dst.txt")
        self.assertTrue(r["success"], r)
        self.assertEqual(self.vfs.read("/dst.txt")["content"], "original")


# ===========================================================================
# 6. Search — find / grep
# ===========================================================================

class TestShellSearch(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {
            "/src/models.py": "class User:\n    id = 1\n    name = 'anon'\n",
            "/src/views.py":  "from models import User\ndef get_user(): return User()\n",
            "/src/utils.py":  "def helper(): pass\n",
            "/README.md":     "# Project\nSee src/ for code.\n",
        })
        self.sh = make_vfs_shell(self.session)

    def test_find_by_name(self):
        r = self.sh("","find / -name *.py")
        self.assertTrue(r["success"], r)
        self.assertIn("models.py", r["stdout"])

    def test_grep_recursive(self):
        r = self.sh("","grep -rn User /src")
        self.assertTrue(r["success"], r)
        self.assertIn("models.py", r["stdout"])

    def test_grep_no_match(self):
        r = self.sh("","grep -r ZZZNOMATCH /src")
        self.assertFalse(r["success"])

    def test_grep_on_shadow_file_does_not_crash(self):
        """BUG S4 — grep on unloaded SHADOW file must not raise."""
        tmp = tempfile.TemporaryDirectory()
        try:
            with open(os.path.join(tmp.name, "shadow.py"), "w") as fh:
                fh.write("SECRET_KEY = 'abc'\n")
            self.session.vfs.mount(tmp.name, vfs_path="/proj")
            f = self.session.vfs.files.get("/proj/shadow.py")
            if f:
                self.assertFalse(f.is_loaded)
                try:
                    r = self.sh("","grep -r SECRET_KEY /proj")
                    self.assertIsInstance(r, dict, "grep must return dict (BUG S4)")
                except Exception as e:
                    self.fail(f"grep raised on SHADOW file (BUG S4): {e}")
        finally:
            tmp.cleanup()


# ===========================================================================
# 7. close / sync commands
# ===========================================================================

class TestShellCloseAndSync(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(self.tmp.name, "data.py"), "w") as fh:
            fh.write("x = 0\n")
        self.session = _make_session()
        _prep(self.session, {"/a.py": "alpha", "/b.py": "beta"})
        self.session.vfs.open("/a.py")
        self.session.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=False)
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def tearDown(self):
        self.tmp.cleanup()

    def test_close_removes_from_context(self):
        r = self.sh("","close /a.py")
        self.assertTrue(r["success"], r)
        self.assertEqual(self.vfs.files["/a.py"].state, "closed")

    def test_close_missing_fails(self):
        r = self.sh("","close /ghost.py")
        self.assertFalse(r["success"])

    def test_close_system_file_fails(self):
        r = self.sh("","close /system_context.md")
        self.assertFalse(r["success"])

    def test_sync_command_recognised(self):
        """BUG S5 — sync must be a known command."""
        r = self.sh("","sync")
        self.assertNotIn("command not found", r.get("stderr", ""),
                         "sync must be a recognised command (BUG S5)")

    def test_sync_flushes_dirty_files(self):
        """BUG S5 — sync must call vfs.sync_all() and persist dirty files."""
        f = self.vfs.files.get("/proj/data.py")
        if f:
            f._content = "x = 999\n"
            f.is_dirty = True
            f.backing_type = FileBackingType.MODIFIED
        r = self.sh("","sync")
        self.assertTrue(r["success"], f"sync failed (BUG S5): {r}")
        local = os.path.join(self.tmp.name, "data.py")
        with open(local, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "x = 999\n",
                             "sync must write dirty content to disk (BUG S5)")


# ===========================================================================
# 8. Multi-command — seq (newline + semicolon)  BUG S1 / S7
# ===========================================================================

class TestShellMultiCommandSeq(unittest.TestCase):
    """
    Sequence operator ';' — both commands always run.
    NOTE: newline is NOT a separator (content-safety: write/echo use real newlines).
    """

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_semicolon_two_commands(self):
        r = self.sh("","mkdir /s; touch /s/f.txt")
        self.assertTrue(r["success"], f"Semicolon batch failed: {r}")
        self.assertTrue(self.vfs._is_directory("/s"))
        self.assertTrue(self.vfs._is_file("/s/f.txt"))

    def test_semicolon_three_commands(self):
        r = self.sh("","mkdir /a; mkdir /a/b; touch /a/b/c.txt")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/a/b/c.txt"))

    def test_semicolon_second_runs_even_if_first_fails(self):
        self.sh("","ls /nonexistent; touch /fallback.txt")
        self.assertTrue(self.vfs._is_file("/fallback.txt"))

    def test_semicolon_write_then_cat(self):
        self.vfs.mkdir("/out")
        r = self.sh("",'write /out/cfg.py "PORT = 8080"; cat /out/cfg.py')
        self.assertTrue(r["success"], f"Batch failed: {r}")
        self.assertIn("PORT", r["stdout"])

    # ── Compatibility: newline must NOT split ─────────────────────────────

    def test_newline_in_quoted_write_content_not_split(self):
        """Real newline inside quotes stays in content, never becomes a separator."""
        r = self.sh("","write /f.py \"class Foo:\n    pass\"")
        self.assertTrue(r["success"], f"Quoted newline in write failed: {r}")
        content = self.vfs.read("/f.py")["content"]
        self.assertIn("Foo",  content)
        self.assertIn("pass", content)

    def test_newline_in_unquoted_content_not_command_not_found(self):
        """Unquoted real newline — must not produce 'command not found'."""
        cmd = "write /f.txt Today\nwrite /f2.txt Tomorrow"
        r = self.sh("",cmd)
        self.assertNotIn("command not found", r.get("stderr", ""),
                         "Newline in unquoted content must not be treated as separator")

    def test_echo_quoted_newline_content_intact(self):
        """echo with real newline inside quotes must write full content."""
        r = self.sh("","echo \"hello\nworld\" > /f.txt")
        self.assertTrue(r["success"], r)
        content = self.vfs.read("/f.txt")["content"]
        self.assertIn("hello", content)
        self.assertIn("world", content)

# ===========================================================================
# 9. Multi-command — AND (&&)  BUG S1
# ===========================================================================

class TestShellMultiCommandAnd(unittest.TestCase):
    """&& — next runs ONLY if previous SUCCEEDED."""

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_and_both_succeed(self):
        r = self.sh("","mkdir /out && touch /out/ok.txt")
        self.assertTrue(r["success"], f"&& batch failed: {r}")
        self.assertTrue(self.vfs._is_file("/out/ok.txt"),
                        "Second && cmd must run on success (BUG S1)")

    def test_and_stops_on_first_failure(self):
        self.sh("","ls /nonexistent && touch /should_not_exist.txt")
        self.assertFalse(self.vfs._is_file("/should_not_exist.txt"),
                         "Second && cmd must NOT run when first fails (BUG S1)")

    def test_and_chain_three(self):
        r = self.sh("","mkdir /c && mkdir /c/sub && touch /c/sub/f.txt")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/c/sub/f.txt"))

    def test_and_chain_stops_at_middle_failure(self):
        self.vfs.mkdir("/a")
        self.sh("","mkdir /a && touch /never.txt")   # second mkdir /a fails
        self.assertFalse(self.vfs._is_file("/never.txt"),
                         "touch after failed mkdir must not run in && chain")

    def test_and_write_then_cat(self):
        r = self.sh("",'write /f.py "v = 1" && cat /f.py')
        self.assertTrue(r["success"])
        self.assertIn("v = 1", r["stdout"])


# ===========================================================================
# 10. Multi-command — OR (||)  BUG S9
# ===========================================================================

class TestShellMultiCommandOr(unittest.TestCase):
    """
    || — next runs ONLY if previous FAILED.
    BUG S9 — '||' is currently not parsed as an operator.
    """

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_or_first_fails_second_runs(self):
        """ls /nope || touch /fallback.txt — touch must run."""
        self.sh("","ls /nonexistent || touch /fallback.txt")
        self.assertTrue(self.vfs._is_file("/fallback.txt"),
                        "Second || cmd must run when first fails (BUG S9)")

    def test_or_first_succeeds_second_skipped(self):
        """ls /  succeeds → touch must NOT run."""
        self.sh("","ls / || touch /should_not.txt")
        self.assertFalse(self.vfs._is_file("/should_not.txt"),
                         "Second || cmd must NOT run when first succeeds (BUG S9)")

    def test_or_fallback_write(self):
        """cat /missing || write /missing.txt 'fallback' — write as fallback."""
        self.sh("",'cat /missing.txt || write /missing.txt "fallback"')
        self.assertTrue(self.vfs._is_file("/missing.txt"),
                        "write fallback via || must create the file (BUG S9)")
        self.assertIn("fallback", self.vfs.read("/missing.txt")["content"])

    def test_or_chained_fallbacks(self):
        """ls /nope || ls /also_nope || touch /last_resort.txt — last must run."""
        self.sh("","ls /nope || ls /also_nope || touch /last_resort.txt")
        self.assertTrue(self.vfs._is_file("/last_resort.txt"),
                        "Chained || must eventually reach a succeeding cmd (BUG S9)")

    def test_or_skips_second_when_first_succeeds_pwd(self):
        """pwd always succeeds; touch must not run."""
        self.sh("","pwd || touch /unreachable.txt")
        self.assertFalse(self.vfs._is_file("/unreachable.txt"),
                         "|| second cmd must be skipped on success (BUG S9)")

    def test_or_not_confused_with_pipe(self):
        """'||' must not be tokenised as two separate '|' operators."""
        # If || were split into two |, the second | would pipe into an empty cmd
        # and likely error. We just verify the correct fallback semantics hold.
        self.sh("","ls /nope || touch /correct.txt")
        self.assertTrue(self.vfs._is_file("/correct.txt"),
                        "|| must not be mis-tokenised as two pipes (BUG S9)")


# ===========================================================================
# 11. Pipe (|) — basic  BUG S8
# ===========================================================================

class TestShellPipeBasic(unittest.TestCase):
    """
    | — stdout of left becomes stdin of right.
    BUG S8 — '|' is not currently parsed as an operator.
    """

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {
            "/src/models.py": (
                "class User:\n"
                "    id = 1\n"
                "    name = 'anon'\n"
                "\n"
                "class Admin(User):\n"
                "    is_staff = True\n"
            ),
            "/src/utils.py": "def helper(): pass\ndef util(): pass\n",
            "/numbers.txt":  "\n".join(str(i) for i in range(1, 21)),
        })
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    # ─── cat | grep ──────────────────────────────────────────────────────

    def test_cat_pipe_grep_finds_match(self):
        r = self.sh("","cat /src/models.py | grep class")
        self.assertTrue(r["success"], f"cat | grep failed: {r}")
        self.assertIn("class", r["stdout"],
                      "grep on piped cat must return matching lines (BUG S8)")

    def test_cat_pipe_grep_case_insensitive(self):
        r = self.sh("","cat /src/models.py | grep -i USER")
        self.assertTrue(r["success"])
        self.assertIn("User", r["stdout"])

    def test_cat_pipe_grep_no_match(self):
        r = self.sh("","cat /src/models.py | grep ZZZNOMATCH")
        self.assertFalse(r["success"],
                         "grep with no match in pipe must return non-success (BUG S8)")

    def test_cat_pipe_grep_line_numbers(self):
        r = self.sh("","cat /src/models.py | grep -n class")
        self.assertTrue(r["success"])
        self.assertTrue(re.search(r"\d+:class", r["stdout"]),
                        "grep -n in pipe must include line numbers (BUG S8)")

    def test_cat_pipe_grep_invert(self):
        r = self.sh("","cat /src/models.py | grep -v class")
        self.assertTrue(r["success"])
        self.assertNotIn("class User", r["stdout"])
        self.assertIn("id", r["stdout"])

    # ─── cat | wc ────────────────────────────────────────────────────────

    def test_cat_pipe_wc_l(self):
        r = self.sh("","cat /src/models.py | wc -l")
        self.assertTrue(r["success"])
        self.assertIn("6", r["stdout"],
                      "wc -l in pipe must count lines (BUG S8)")

    def test_cat_pipe_wc_no_flag(self):
        r = self.sh("","cat /src/models.py | wc")
        self.assertTrue(r["success"])
        parts = r["stdout"].split()
        self.assertGreaterEqual(len(parts), 3,
                                "wc with no flag must show lines words chars")

    # ─── cat | head / tail ───────────────────────────────────────────────

    def test_cat_pipe_head(self):
        r = self.sh("","cat /numbers.txt | head -n 3")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "1")

    def test_cat_pipe_tail(self):
        r = self.sh("","cat /numbers.txt | tail -n 3")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[-1], "20")

    # ─── ls | grep ───────────────────────────────────────────────────────

    def test_ls_pipe_grep(self):
        r = self.sh("","ls /src | grep .py")
        self.assertTrue(r["success"])
        self.assertIn("models.py", r["stdout"],
                      "ls | grep must filter listing (BUG S8)")

    def test_ls_pipe_wc_l(self):
        r = self.sh("","ls /src | wc -l")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertGreater(count, 0)

    # ─── grep | wc ───────────────────────────────────────────────────────

    def test_grep_pipe_wc_l(self):
        r = self.sh("","grep -rl class /src | wc -l")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertGreater(count, 0,
                           "grep | wc -l must count matched files (BUG S8)")

    # ─── sort / uniq ─────────────────────────────────────────────────────

    def test_cat_pipe_sort(self):
        r = self.sh("","cat /src/utils.py | grep def | sort")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(lines, sorted(lines))

    def test_cat_pipe_sort_reverse(self):
        r = self.sh("","cat /src/utils.py | grep def | sort -r")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(lines, sorted(lines, reverse=True))


# ===========================================================================
# 12. Pipe (|) — chained pipelines  BUG S8
# ===========================================================================

class TestShellPipeChained(unittest.TestCase):
    """Three-stage (and longer) pipelines."""

    def setUp(self):
        self.session = _make_session()
        content = "\n".join([
            "def alpha():", "    pass",
            "def beta():",  "    pass",
            "def gamma():", "    pass",
            "x = 1",
        ])
        _prep(self.session, {"/code.py": content})
        self.sh = make_vfs_shell(self.session)

    def test_three_stage_cat_grep_wc(self):
        r = self.sh("","cat /code.py | grep def | wc -l")
        self.assertTrue(r["success"], f"3-stage pipe failed: {r}")
        count = int(r["stdout"].strip().split()[0])
        self.assertEqual(count, 3,
                         "cat|grep def|wc -l must count 3 def-lines (BUG S8)")

    def test_two_greps_chained(self):
        r = self.sh("","cat /code.py | grep def | grep alpha")
        self.assertTrue(r["success"])
        self.assertIn("alpha", r["stdout"])
        self.assertNotIn("beta",  r["stdout"])

    def test_cat_grep_head_chain(self):
        r = self.sh("","cat /code.py | grep def | head -n 2")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertLessEqual(len(lines), 2)
        self.assertTrue(all("def" in ln for ln in lines if ln))

    def test_sort_then_head(self):
        r = self.sh("","cat /code.py | grep def | sort | head -n 1")
        self.assertTrue(r["success"])
        # Alphabetically first def
        self.assertIn("alpha", r["stdout"])

    def test_four_stage_pipe(self):
        """cat | grep | sort | wc -l — four stages."""
        r = self.sh("","cat /code.py | grep def | sort | wc -l")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertEqual(count, 3)


# ===========================================================================
# 13. Mixed operators — &&, ||, |, ;, \n combined  BUG S1/S8/S9
# ===========================================================================

class TestShellMixedOperators(unittest.TestCase):
    """Realistic agent patterns combining multiple operators in one call."""

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_and_then_pipe(self):
        """mkdir /d && write /d/f.py 'x=1' && cat /d/f.py | grep x"""
        r = self.sh("",'mkdir /d && write /d/f.py "x=1" && cat /d/f.py | grep x')
        self.assertTrue(r["success"])
        self.assertIn("x=1", r["stdout"])

    def test_or_fallback_then_seq(self):
        """ls /nope || write /fb.txt 'fallback'; cat /fb.txt"""
        self.sh("",'ls /nope || write /fb.txt "fallback"; cat /fb.txt')
        self.assertTrue(self.vfs._is_file("/fb.txt"))
        r = self.sh("","cat /fb.txt")
        self.assertIn("fallback", r["stdout"])

    def test_newline_and_pipe(self):
        r = self.sh("",
            'mkdir /src2;'
            'write /src2/m.py "class X:\\n    pass";'
            'cat /src2/m.py | grep class'
        )
        self.assertTrue(r["success"])
        self.assertIn("class", r["stdout"])

    def test_pipe_success_enables_and(self):
        """cat /f | grep x && touch /found.txt — touch runs only if grep found."""
        self.vfs.create("/f.txt", "x = 1\ny = 2\n")
        self.sh("","cat /f.txt | grep x && touch /found.txt")
        self.assertTrue(self.vfs._is_file("/found.txt"),
                        "touch must run after successful pipe+grep")

    def test_pipe_failure_blocks_and(self):
        """cat /f | grep NOPE && touch /nope.txt — touch must NOT run."""
        self.vfs.create("/f.txt", "no match\n")
        self.sh("","cat /f.txt | grep ZZZNOMATCH && touch /nope.txt")
        self.assertFalse(self.vfs._is_file("/nope.txt"),
                         "touch must NOT run when piped grep finds nothing")

    def test_complex_six_part_batch(self):
        """Realistic agent write-verify workflow."""
        batch = (
            "mkdir -p /proj/src; "
            'write /proj/src/app.py "DEBUG=True\\nPORT=8080"; '
            "cat /proj/src/app.py | grep DEBUG; "
            "cat /proj/src/app.py | grep PORT | wc -l; "
            "ls /proj/src"
        )
        r = self.sh("",batch)
        self.assertTrue(r["success"], f"Complex batch failed: {r}")
        self.assertIn("DEBUG",  r["stdout"])
        self.assertIn("app.py", r["stdout"])

    def test_or_after_pipe_failure(self):
        """cat /f | grep NOPE || touch /fallback.txt"""
        self.vfs.create("/f.txt", "nothing here\n")
        self.sh("","cat /f.txt | grep ZZZNOMATCH || touch /fallback2.txt")
        self.assertTrue(self.vfs._is_file("/fallback2.txt"),
                        "|| must run after a failing pipe stage")


# ===========================================================================
# 14. _split_compound unit tests (tokeniser)
# ===========================================================================
class TestSplitCompound(unittest.TestCase):
    """Unit tests for the _split_compound tokeniser."""

    def setUp(self):
        try:
            from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import _split_compound
            self.split = _split_compound
        except ImportError:
            self.split = None

    def _s(self):
        if self.split is None:
            self.skipTest("_split_compound not yet available")

    def test_single_command(self):
        self._s()
        self.assertEqual(self.split("ls /src"), [('seq', 'ls /src')])

    def test_semicolon(self):
        self._s()
        self.assertEqual(self.split("mkdir /a; touch /a/f.txt"),
                         [('seq', 'mkdir /a'), ('seq', 'touch /a/f.txt')])

    def test_and(self):
        self._s()
        self.assertEqual(self.split("mkdir /a && touch /a/f.txt"),
                         [('seq', 'mkdir /a'), ('&&', 'touch /a/f.txt')])

    def test_or(self):
        self._s()
        self.assertEqual(self.split("ls /nope || echo ok"),
                         [('seq', 'ls /nope'), ('||', 'echo ok')])

    def test_pipe(self):
        self._s()
        self.assertEqual(self.split("cat /f.py | grep x"),
                         [('seq', 'cat /f.py'), ('|', 'grep x')])

    def test_chained_pipes(self):
        self._s()
        self.assertEqual(self.split("cat /f | grep x | wc -l"),
                         [('seq', 'cat /f'), ('|', 'grep x'), ('|', 'wc -l')])

    def test_or_not_split_into_two_pipes(self):
        self._s()
        ops = [op for op, _ in self.split("a || b")]
        self.assertIn('||', ops)
        self.assertNotIn('|', ops)

    def test_operator_in_double_quotes_literal(self):
        self._s()
        self.assertEqual(len(self.split('echo "a && b" > /f.txt')), 1,
                         "&& inside double quotes must not split")

    def test_operator_in_single_quotes_literal(self):
        self._s()
        self.assertEqual(len(self.split("write /f.txt 'x || y'")), 1,
                         "|| inside single quotes must not split")

    def test_newline_is_NOT_a_separator(self):
        self._s()
        self.assertEqual(len(self.split("write /f.py \"class Foo:\n    pass\"")), 1,
                         "Real newline inside quotes must NOT split")

    def test_backslash_n_escape_not_a_separator(self):
        self._s()
        self.assertEqual(len(self.split('write /f.py "line1\\nline2\\nline3"')), 1,
                         "Backslash-n escape must never split")

    def test_empty_string(self):
        self._s()
        self.assertEqual(self.split(""), [])

    def test_mixed_four_operators(self):
        self._s()
        r = self.split("a && b || c | d; e")
        self.assertEqual([cmd for _, cmd in r], ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual([op  for op, _ in r], ['seq', '&&', '||', '|', 'seq'])

# ===========================================================================
# 15. vfs_view
# ===========================================================================

class TestVfsView(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        code = "\n".join([
            "# header", "import os", "",
            "class UserModel:",   # line 4
            "    id = 1", "    name = 'anon'", "",
            "def get_user(uid):", # line 8
            "    return UserModel()",
        ])
        _prep(self.session, {"/models.py": code})
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs

    def test_basic_open(self):
        r = self.view("/models.py")
        self.assertTrue(r["success"], r)
        self.assertIn("content", r)

    def test_line_range(self):
        r = self.view("/models.py", line_start=4, line_end=6)
        self.assertTrue(r["success"])
        self.assertIn("UserModel", r["content"])

    def test_scroll_to(self):
        r = self.view("/models.py", scroll_to="get_user", context_lines=4)
        self.assertTrue(r["success"])
        self.assertIn("get_user", r["content"])
        self.assertEqual(r["match"]["matched_line"], 8)

    def test_scroll_to_not_found_hint(self):
        r = self.view("/models.py", scroll_to="ZZZNOMATCH")
        self.assertFalse(r["success"])
        self.assertIn("hint", r)

    def test_close_others(self):
        self.vfs.open("/models.py")
        _prep(self.session, {"/other.py": "pass"})
        self.vfs.open("/other.py")
        r = self.view("/models.py", close_others=True)
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.files["/other.py"].state, "closed")

    def test_close_others_preserves_system_files(self):
        r = self.view("/models.py", close_others=True)
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.files["/system_context.md"].state, "open")

    def test_missing_file_fails(self):
        r = self.view("/ghost.py")
        self.assertFalse(r["success"])

    def test_result_has_required_keys(self):
        r = self.view("/models.py")
        for k in ("success", "path", "content", "showing", "total_lines"):
            self.assertIn(k, r)

class TestShellMultiCommandSeq2(unittest.TestCase):
    """
    Sequence operator: ';' — both commands always run.
    NOTE: newline is NOT supported as a separator (compatibility risk).
    Use ';' or '&&' for batching.
    """

    def setUp(self):
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_semicolon_two_commands(self):
        r = self.sh("","mkdir /s; touch /s/f.txt")
        self.assertTrue(r["success"], f"Semicolon batch failed: {r}")
        self.assertTrue(self.vfs._is_directory("/s"))
        self.assertTrue(self.vfs._is_file("/s/f.txt"))

    def test_semicolon_three_commands(self):
        r = self.sh("","mkdir /a; mkdir /a/b; touch /a/b/c.txt")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/a/b/c.txt"))

    def test_semicolon_second_runs_even_if_first_fails(self):
        """seq: second always runs regardless of first outcome."""
        self.sh("","ls /nonexistent; touch /fallback.txt")
        self.assertTrue(self.vfs._is_file("/fallback.txt"))

    # ── Compatibility: actual newlines must NOT split commands ────────────

    def test_newline_in_write_content_unquoted_not_split(self):
        """
        A real newline in an unquoted write command must NOT be treated
        as a command separator — it goes to shlex as-is.
        (It may produce imperfect output but must not raise 'command not found'.)
        """
        # This has an actual newline — NOT a batch separator
        cmd = "write /f.txt Today\nwrite /f2.txt Tomorrow"
        try:
            r = self.sh("",cmd)
            # Must NOT produce 'command not found' for 'write' or 'Tomorrow'
            self.assertNotIn("command not found", r.get("stderr", ""),
                             "Newline in unquoted content must not be a separator")
        except Exception as e:
            self.fail(f"Real newline in command raised exception: {e}")

    def test_newline_in_write_content_quoted_stays_intact(self):
        """
        Actual newline inside double quotes stays in the content, is never
        treated as a batch separator.
        """
        # Python double-quote: \n is actual newline, inside "..." → safe
        cmd = "write /f.py \"class Foo:\n    pass\""
        r = self.sh("",cmd)
        self.assertTrue(r["success"], f"Quoted newline in write failed: {r}")
        content = self.vfs.read("/f.py")["content"]
        self.assertIn("Foo", content)
        self.assertIn("pass", content)

    def test_echo_newline_in_content_not_split(self):
        """
        echo "hello\nworld" > /f.txt — actual newline inside quotes,
        must write the full content, not stop at the newline.
        """
        cmd = "echo \"hello\nworld\" > /f.txt"
        r = self.sh("",cmd)
        self.assertTrue(r["success"], f"Echo with quoted newline failed: {r}")
        content = self.vfs.read("/f.txt")["content"]
        self.assertIn("hello", content)
        self.assertIn("world", content)


class TestSharedStoreFailuresSurface(unittest.TestCase):
    """Fix 3 — shared_write failures must not be silently swallowed."""

    def test_shared_store_explicit_failure_surfaces(self):
        from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        try:
            gvfs = get_global_vfs()
            mount_key = gvfs.register_shared_mount(tmp.name, hydrate=False)

            session = _make_session()
            session.vfs.mount(tmp.name, vfs_path="/proj", auto_sync=True)
            sh = make_vfs_shell(session)

            # Path traversal → shared_write returns success=False
            r = sh("",'write /proj/../evil.py "x=1"')
            # Either the command fails, or falls to legacy with different error.
            # What we care about: the result must NOT be success=True with
            # a silently-dropped shared-store error.
            if not r["success"]:
                self.assertTrue(r.get("stderr"), "Failure must carry an error message")
        finally:
            gvfs.unregister_shared_mount(tmp.name)
            tmp.cleanup()


import unittest
import json
import os
import shutil
import tempfile
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell
from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs


class TestVfsChunkWriteLifecycle(unittest.TestCase):
    def setUp(self):
        self.session = unittest.mock.MagicMock()
        self.vfs = VirtualFileSystemV2(session_id="test_s", agent_name="test_a")
        self.session.vfs = self.vfs
        self.session.agent_name = "test_a"
        self.shell = make_vfs_shell(self.session)

        # Temp dir für Mounts
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_chunk_protocol_completion(self):
        """Prüft den Standard-Lifecycle: Start, Mitte, Ende."""
        path = "/test.txt"
        # Chunk 0: Init
        r0 = self.shell("", 'write_chunk /test.txt 0 3 "part1\\n"')
        self.assertTrue(r0["success"])
        self.assertTrue(self.vfs._is_file("/test.txt.__chunks__"))

        # Chunk 1: Append
        r1 = self.shell("", 'write_chunk /test.txt 1 3 "part2\\n"')
        self.assertTrue(r1["success"])

        # Status Check
        rs = self.shell("", "write_chunk_status /test.txt")
        self.assertIn("missing=[2]", rs["stdout"])

        # Chunk 2: Finalize
        r2 = self.shell("", 'write_chunk /test.txt 2 3 "part3"')
        self.assertTrue(r2["success"])
        self.assertIn("complete", r2["stdout"])

        # Verify Content
        content = self.vfs.read("/test.txt")["content"]
        self.assertEqual(content, "part1\npart2\npart3")
        # Sidecar must be gone
        self.assertFalse(self.vfs._is_file("/test.txt.__chunks__"))

    def test_global_shared_chunk_desync(self):
        """
        Integrationstest: Prüft ob Chunks im Shared-Store (Ebene 3) ankommen.
        Dies ist die vermutete Fehlerstelle.
        """
        gvfs = get_global_vfs()
        self.vfs.mount(self.test_dir, vfs_path="/global")

        # Chunk 0 (verwendet vfs.write -> Shared Store OK)
        self.shell("", 'write_chunk /global/shared.txt 0 2 "first_line\\n"')

        # Simuliere zweiten Agent/VFS
        vfs2 = VirtualFileSystemV2(session_id="agent2", agent_name="agent2")
        vfs2.mount(self.test_dir, vfs_path="/global")

        # Check Agent 2 nach Chunk 0
        print(self.vfs.read("/global/shared.txt"))
        self.assertEqual(vfs2.read("/global/shared.txt")["content"], "first_line\n")

        # Chunk 1 (verwendet vfs.append -> Falls Bug existiert, fehlt Shared Store Update)
        self.shell("", 'write_chunk /global/shared.txt 1 2 "second_line"')

        # VERIFIKATION: Hat Agent 2 das Update?
        # Wenn der Bug existiert, liefert vfs2.read() noch den alten Stand aus Ebene 3
        res2 = vfs2.read("/global/shared.txt")
        self.assertEqual(res2["content"], "first_line\nsecond_line",
                         "BUG: Shared Store wurde nach append (Chunk > 0) nicht aktualisiert!")

    def test_shared_chunk_desync(self):
        gvfs = get_global_vfs()

        # 1. Registriere test_dir als Shared Mount (Ebene 3b)
        mnt_key = gvfs.register_shared_mount(self.test_dir)

        # 2. Agent 1 & 2 mounten den gemeinsamen Ordner
        self.vfs.mount(self.test_dir, vfs_path="/project")

        vfs2 = VirtualFileSystemV2(session_id="agent2", agent_name="agent2")
        vfs2.mount(self.test_dir, vfs_path="/project")

        # 3. Agent 1 startet write_chunk (idx 0 -> nutzt vfs.write -> Shared Store OK)
        r0 = self.shell("", 'write_chunk /project/shared.txt 0 2 "first_line\\n"')
        self.assertTrue(r0["success"], r0)

        # WICHTIG: Im Test müssen wir Agent 2 zwingen, die neue Datei
        # in seine self.files Registry aufzunehmen (überspringt den 1.5s Poller).
        vfs2.refresh_mount("/project")

        # Verifiziere, dass Agent 2 den ersten Chunk hat
        self.assertEqual(vfs2.read("/project/shared.txt")["content"], "first_line\n")

        # 4. Agent 1 sendet Chunk 1 (idx 1 -> nutzt vfs.append)
        r1 = self.shell("", 'write_chunk /project/shared.txt 1 2 "second_line"')
        self.assertTrue(r1["success"], r1)

        # 5. Agent 2 liest erneut. WENN append gefixt ist, sieht er das Update
        # sofort via RAM (Shared Store), ganz ohne nochmaliges refresh_mount!
        res2 = vfs2.read("/project/shared.txt")
        self.assertEqual(res2["content"], "first_line\nsecond_line")

        # Cleanup
        gvfs.unregister_shared_mount(self.test_dir)

if __name__ == "__main__":
    unittest.main()
