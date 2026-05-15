"""
test_vfs_shell_bullet_proof.py
==============================
Real-world, agent-workflow-driven tests for vfs_shell_tool.

Focus areas:
  1. Content encoding — _decode_content must convert \n → real newlines
  2. Large file editing — multi-step edits, boundary conditions
  3. grep in pipes — real project structures, edge cases
  4. Agent roundtrips — grep → view → edit → verify
  5. write_chunk error recovery — abort, resume, duplicate, out-of-order
  6. Edge cases — Unicode, empty files, whitespace-only, trailing newlines

These tests are designed to FAIL on the current codebase (where _decode_content
is disabled) and pass only when the system works correctly end-to-end.

Run with:
    python -m unittest test_vfs_shell_bullet_proof -v
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import (
    _decode_content,
    _split_compound,
    _strip_quotes,
    make_vfs_shell,
    make_vfs_view,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session():
    vfs = VirtualFileSystemV2(session_id="test-bp", agent_name="test-bp")
    s = MagicMock()
    s.vfs = vfs
    return s


def _prep(session, files=None, dirs=None):
    vfs = session.vfs
    for d in dirs or []:
        vfs.mkdir(d, parents=True)
    for path, content in (files or {}).items():
        parent = os.path.dirname(path)
        if parent and parent != "/":
            vfs.mkdir(parent, parents=True)
        vfs.create(path, content)


class TestStripQuotes(unittest.TestCase):

    def test_double_quotes(self):
        self.assertEqual(_strip_quotes('"hello"'), "hello")

    def test_single_quotes(self):
        self.assertEqual(_strip_quotes("'hello'"), "hello")

    def test_triple_double_quotes(self):
        self.assertEqual(_strip_quotes('"""multi"""'), "multi")

    def test_raw_prefix(self):
        self.assertEqual(_strip_quotes('r"raw"'), "raw")

    def test_no_quotes(self):
        self.assertEqual(_strip_quotes("bare"), "bare")

    def test_empty_quoted(self):
        self.assertEqual(_strip_quotes('""'), "")


# ===========================================================================
# B. Write + Read roundtrip — the agent's bread and butter
# ===========================================================================

class TestWriteReadRoundtrip(unittest.TestCase):
    """Agent writes content with escape sequences, then reads it back.
    This is the #1 real-world operation and must be bulletproof."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_write_multiline_and_cat_back(self):
        """write with \n, then cat — content must have real newlines."""
        self.sh("", 'write /app.py "import os\nimport sys\n\ndef main():\n    pass"')
        r = self.sh("", "cat /app.py")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(lines[0], "import os")
        self.assertEqual(lines[1], "import sys")
        self.assertEqual(lines[2], "")
        self.assertEqual(lines[3], "def main():")
        self.assertEqual(lines[4], "    pass")

    def test_write_with_tabs(self):
        self.sh("", 'write /tsv.txt "col1\\tcol2\\tcol3\nval1\\tval2\\tval3"')
        r = self.sh("", "cat /tsv.txt")
        self.assertTrue(r["success"])
        self.assertIn("\\t", r["stdout"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 2)

    def test_write_empty_content(self):
        self.sh("", 'write /empty.txt ""')
        r = self.sh("", "cat /empty.txt")
        self.assertTrue(r["success"])
        self.assertEqual(r["stdout"].strip(), "")

    def test_write_single_line_no_escape(self):
        self.sh("", 'write /one.txt "just one line"')
        content = self.vfs.read("/one.txt")["content"]
        self.assertEqual(content, "just one line")

    def test_write_preserves_indentation(self):
        """Python indentation via \n + spaces must survive."""
        self.sh("", 'write /indent.py "class Foo:\n    def bar(self):\n        return 42"')
        r = self.sh("", "cat /indent.py")
        lines = r["stdout"].splitlines()
        self.assertEqual(lines[0], "class Foo:")
        self.assertTrue(lines[1].startswith("    def"))
        self.assertTrue(lines[2].startswith("        return"))

    def test_write_with_embedded_quotes(self):
        self.sh("", 'write /q.py "name = \"world\"\nprint(f\"hello {name}\")"')
        content = self.vfs.read("/q.py")["content"]
        self.assertIn('"world"', content)

    def test_echo_redirect_multiline(self):
        self.sh("", 'echo "line1\nline2\nline3" > /echo.txt')
        content = self.vfs.read("/echo.txt")["content"]
        self.assertEqual(len(content.splitlines()), 3)

    def test_echo_append_multiline(self):
        self.vfs.create("/log.txt", "existing\n")
        self.vfs.open("/log.txt")
        self.sh("", 'echo "appended1\nappended2" >> /log.txt')
        content = self.vfs.read("/log.txt")["content"]
        self.assertIn("existing", content)
        self.assertIn("appended1", content)
        self.assertIn("appended2", content)


# ===========================================================================
# C. Large file editing — the agent's main pain point
# ===========================================================================

class TestLargeFileEditing(unittest.TestCase):
    """Agent works on files with 100+ lines: grep to find, edit to change,
    verify nothing else broke."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs
        # Create a realistic Python file with 150 lines
        lines = []
        lines.append("import os")
        lines.append("import sys")
        lines.append("from typing import Optional")
        lines.append("")
        lines.append("# Configuration")
        lines.append("DEBUG = False")
        lines.append("PORT = 8080")
        lines.append("HOST = 'localhost'")
        lines.append("")
        for i in range(10, 60):
            lines.append(f"def function_{i}(x):")
            lines.append(f"    return x * {i}")
            lines.append("")
        lines.append("class MainService:")
        lines.append("    def __init__(self):")
        lines.append("        self.running = False")
        lines.append("")
        lines.append("    def start(self):")
        lines.append("        self.running = True")
        lines.append("        print('started')")
        lines.append("")
        lines.append("    def stop(self):")
        lines.append("        self.running = False")
        lines.append("        print('stopped')")
        lines.append("")
        for i in range(80, 100):
            lines.append(f"# TODO: implement feature_{i}")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    svc = MainService()")
        lines.append("    svc.start()")
        self.big_content = "\n".join(lines)
        _prep(self.session, {"/proj/service.py": self.big_content})
        self.total_lines = len(self.big_content.splitlines())

    def test_grep_finds_class_in_large_file(self):
        r = self.sh("", "grep -n 'class MainService' /proj/service.py")
        self.assertTrue(r["success"])
        # Must report a line number
        self.assertRegex(r["stdout"], r"\d+:.*class MainService")

    def test_grep_finds_all_functions(self):
        r = self.sh("", "grep -n 'def function_' /proj/service.py")
        self.assertTrue(r["success"])
        matches = [l for l in r["stdout"].splitlines() if "def function_" in l]
        self.assertEqual(len(matches), 50, "Must find all 50 function definitions")

    def test_edit_single_line_preserves_rest(self):
        """Edit line 6 (DEBUG = False → DEBUG = True), verify rest unchanged."""
        before = self.vfs.read("/proj/service.py")["content"].splitlines()
        debug_line = None
        for i, l in enumerate(before, 1):
            if "DEBUG = False" in l:
                debug_line = i
                break
        self.assertIsNotNone(debug_line)

        r = self.sh("", f'edit /proj/service.py {debug_line} {debug_line} "DEBUG = True"')
        self.assertTrue(r["success"], r)

        after = self.vfs.read("/proj/service.py")["content"].splitlines()
        self.assertIn("DEBUG = True", after[debug_line - 1])
        # Lines before and after must be unchanged
        self.assertEqual(before[debug_line - 2], after[debug_line - 2])
        if debug_line < len(before):
            self.assertEqual(before[debug_line], after[debug_line])

    def test_edit_multi_line_range(self):
        """Replace the start method (multiple lines)."""
        r = self.sh("", "grep -n 'def start' /proj/service.py")
        self.assertTrue(r["success"])
        match = re.search(r"(\d+):.*def start", r["stdout"])
        self.assertIsNotNone(match)
        start_line = int(match.group(1))

        new_method = "    def start(self, port=None):\n        self.running = True\n        self.port = port or 8080\n        print(f'started on {self.port}')"
        r = self.sh("", f'edit /proj/service.py {start_line} {start_line + 2} "{new_method}"')
        self.assertTrue(r["success"], r)

        content = self.vfs.read("/proj/service.py")["content"]
        self.assertIn("port=None", content)
        self.assertIn("self.port = port", content)

    def test_edit_last_line(self):
        """Edit the very last line of the file."""
        r = self.sh("", f'edit /proj/service.py {self.total_lines} {self.total_lines} "    svc.start()\n    svc.stop()"')
        self.assertTrue(r["success"], r)
        content = self.vfs.read("/proj/service.py")["content"]
        self.assertTrue(content.rstrip().endswith("svc.stop()"))

    def test_sequential_edits_dont_corrupt(self):
        """Two consecutive edits on different lines must both apply cleanly."""
        # Edit 1: change PORT
        self.sh("", 'edit /proj/service.py 7 7 "PORT = 9090"')
        # Edit 2: change HOST
        self.sh("", 'edit /proj/service.py 8 8 "HOST = \'0.0.0.0\'"')

        content = self.vfs.read("/proj/service.py")["content"]
        self.assertIn("PORT = 9090", content)
        self.assertIn("HOST = '0.0.0.0'", content)

    def test_sed_reads_exact_range(self):
        """sed -n 'X,Yp' must return exactly the requested lines."""
        r = self.sh("", "sed -n '1,5p' /proj/service.py")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "import os")

    def test_head_tail_on_large_file(self):
        r = self.sh("", "head -n 3 /proj/service.py")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["stdout"].splitlines()), 3)

        r = self.sh("", "tail -n 3 /proj/service.py")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["stdout"].splitlines()), 3)
        self.assertIn("svc", r["stdout"])

    def test_wc_counts_large_file(self):
        r = self.sh("", "wc -l /proj/service.py")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertEqual(count, self.total_lines)


# ===========================================================================
# D. grep in pipes — agent's primary info-finding workflow
# ===========================================================================

class TestGrepPipeRealWorld(unittest.TestCase):
    """Real project structure: agent greps, pipes, filters — must be reliable."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs
        _prep(self.session, {
            "/proj/src/models/user.py": textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass
                class User:
                    id: int
                    name: str
                    email: str
                    is_active: bool = True

                @dataclass
                class AdminUser(User):
                    permissions: list = None
                """),
            "/proj/src/models/__init__.py": "from .user import User, AdminUser\n",
            "/proj/src/services/auth.py": textwrap.dedent("""\
                from models.user import User, AdminUser
                import hashlib

                class AuthService:
                    def __init__(self, db):
                        self.db = db

                    def login(self, email: str, password: str) -> User:
                        hashed = hashlib.sha256(password.encode()).hexdigest()
                        user = self.db.get_user_by_email(email)
                        if user and user.password_hash == hashed:
                            return user
                        raise ValueError("Invalid credentials")

                    def register(self, name: str, email: str, password: str) -> User:
                        if self.db.get_user_by_email(email):
                            raise ValueError("Email already registered")
                        return self.db.create_user(name=name, email=email, password=password)
                """),
            "/proj/src/services/notification.py": textwrap.dedent("""\
                class NotificationService:
                    def send_email(self, to: str, subject: str, body: str):
                        print(f"Sending email to {to}")

                    def send_sms(self, to: str, message: str):
                        print(f"Sending SMS to {to}")
                """),
            "/proj/src/utils/helpers.py": textwrap.dedent("""\
                import re

                def validate_email(email: str) -> bool:
                    return bool(re.match(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', email))

                def sanitize_input(text: str) -> str:
                    return text.strip().replace('<', '&lt;').replace('>', '&gt;')
                """),
            "/proj/tests/test_auth.py": textwrap.dedent("""\
                import unittest
                from services.auth import AuthService

                class TestAuth(unittest.TestCase):
                    def test_login_success(self):
                        pass

                    def test_login_invalid_password(self):
                        pass

                    def test_register_duplicate_email(self):
                        pass
                """),
            "/proj/README.md": "# My Project\n\nA sample project.\n",
            "/proj/config.yaml": "debug: false\nport: 8080\nhost: localhost\n",
            # Edge case: empty file
            "/proj/src/__init__.py": "",
            # Edge case: binary-like content
            "/proj/data/binary.dat": "\x00\x01\x02data\xff\xfe",
        })

    def test_recursive_grep_across_project(self):
        """grep -rn 'class' must find all class definitions."""
        r = self.sh("", "grep -rn class /proj/src")
        self.assertTrue(r["success"])
        self.assertIn("User", r["stdout"])
        self.assertIn("AuthService", r["stdout"])
        self.assertIn("NotificationService", r["stdout"])

    def test_grep_pipe_wc_counts_matches(self):
        """grep -rl 'import' /proj/src | wc -l — count files with imports."""
        r = self.sh("", "grep -rl import /proj/src | wc -l")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertGreaterEqual(count, 3)

    def test_grep_pipe_grep_narrows_results(self):
        """grep -rn 'def' /proj/src | grep 'login' — find login method."""
        r = self.sh("", "grep -rn def /proj/src | grep login")
        self.assertTrue(r["success"])
        self.assertIn("login", r["stdout"])
        self.assertNotIn("send_email", r["stdout"])

    def test_grep_case_insensitive_in_pipe(self):
        r = self.sh("", "cat /proj/src/services/auth.py | grep -i user")
        self.assertTrue(r["success"])
        stdout_lower = r["stdout"].lower()
        self.assertIn("user", stdout_lower)

    def test_grep_on_empty_file_no_crash(self):
        """grep on empty __init__.py must not crash, just return no matches."""
        r = self.sh("", "grep -n class /proj/src/__init__.py")
        # No match is fine, but no crash
        self.assertIsInstance(r, dict)
        self.assertIn("success", r)

    def test_find_then_grep_workflow(self):
        """Agent pattern: find *.py files, then grep specific in them."""
        r = self.sh("", "find /proj -name *.py -type f")
        self.assertTrue(r["success"])
        py_files = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
        self.assertGreater(len(py_files), 0)

        # Now grep in a specific found file
        r = self.sh("", "grep -n 'def validate' /proj/src/utils/helpers.py")
        self.assertTrue(r["success"])
        self.assertIn("validate_email", r["stdout"])

    def test_grep_files_only_mode(self):
        r = self.sh("", "grep -rl User /proj/src")
        self.assertTrue(r["success"])
        # Must list file paths, not matched lines
        for line in r["stdout"].splitlines():
            self.assertTrue(line.strip().endswith(".py") or line.strip() == "",
                            f"grep -l must return file paths, got: {line}")

    def test_ls_pipe_grep_filters_listing(self):
        r = self.sh("", "ls /proj/src/services | grep auth")
        self.assertTrue(r["success"])
        self.assertIn("auth", r["stdout"])
        self.assertNotIn("notification", r["stdout"])

    def test_cat_pipe_head_shows_first_n(self):
        r = self.sh("", "cat /proj/src/services/auth.py | head -n 5")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertLessEqual(len(lines), 5)
        self.assertIn("import", lines[0])

    def test_three_stage_pipe_grep_grep_wc(self):
        """grep | grep | wc — real narrowing pattern."""
        r = self.sh("", "grep -rn def /proj/src | grep -i auth | wc -l")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertGreater(count, 0)


# ===========================================================================
# E. Agent roundtrip: grep → view → edit → verify
# ===========================================================================

class TestAgentRoundtrip(unittest.TestCase):
    """Full agent workflow: find something, view it, edit it, confirm the edit."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs
        _prep(self.session, {
            "/proj/config.py": textwrap.dedent("""\
                # Application Configuration
                DEBUG = False
                PORT = 8080
                HOST = 'localhost'
                DB_URL = 'sqlite:///app.db'
                SECRET_KEY = 'change-me'
                MAX_RETRIES = 3
                TIMEOUT = 30
                LOG_LEVEL = 'INFO'
                """),
        })

    def test_grep_find_edit_verify_single_line(self):
        """Agent wants to change DEBUG to True."""
        # Step 1: Find
        r = self.sh("", "grep -n DEBUG /proj/config.py")
        self.assertTrue(r["success"])
        match = re.search(r"(\d+):.*DEBUG", r["stdout"])
        self.assertIsNotNone(match)
        line_no = int(match.group(1))

        # Step 2: View context
        rv = self.view("/proj/config.py", scroll_to="DEBUG", context_lines=6)
        self.assertTrue(rv["success"])
        self.assertIn("DEBUG", rv["content"])

        # Step 3: Edit
        r = self.sh("", f'edit /proj/config.py {line_no} {line_no} "DEBUG = True"')
        self.assertTrue(r["success"])

        # Step 4: Verify
        r = self.sh("", "cat /proj/config.py")
        content = r["stdout"]
        self.assertIn("DEBUG = True", content)
        self.assertNotIn("DEBUG = False", content)
        # Other config lines must be untouched
        self.assertIn("PORT = 8080", content)
        self.assertIn("SECRET_KEY", content)

    def test_grep_find_edit_verify_multiline_replace(self):
        """Replace SECRET_KEY and MAX_RETRIES (two consecutive lines)."""
        r = self.sh("", "grep -n SECRET_KEY /proj/config.py")
        match = re.search(r"(\d+):.*SECRET_KEY", r["stdout"])
        sk_line = int(match.group(1))

        r = self.sh("", "grep -n MAX_RETRIES /proj/config.py")
        match = re.search(r"(\d+):.*MAX_RETRIES", r["stdout"])
        mr_line = int(match.group(1))

        new_content = "SECRET_KEY = 'production-key-xyz'\nMAX_RETRIES = 5"
        r = self.sh("", f'edit /proj/config.py {sk_line} {mr_line} "{new_content}"')
        self.assertTrue(r["success"])

        content = self.vfs.read("/proj/config.py")["content"]
        self.assertIn("production-key-xyz", content)
        self.assertIn("MAX_RETRIES = 5", content)
        self.assertIn("TIMEOUT = 30", content)  # untouched

    def test_write_then_grep_finds_new_content(self):
        """Agent writes new file, then greps it — must find the content."""
        self.sh("", 'write /proj/new_module.py "class NewHandler:\n    def handle(self, request):\n        return \'ok\'"')
        r = self.sh("", "grep -n NewHandler /proj/new_module.py")
        self.assertTrue(r["success"], f"grep must find newly written content: {r}")
        self.assertIn("NewHandler", r["stdout"])

    def test_batch_write_and_verify(self):
        """Agent writes a file and immediately verifies via pipe — one batch."""
        r = self.sh("",
            'write /proj/routes.py "from flask import Flask\napp = Flask(__name__)\n\n@app.route(\"/\")\ndef index():\n    return \"hello\""; '
            'cat /proj/routes.py | grep Flask'
        )
        self.assertTrue(r["success"])
        self.assertIn("Flask", r["stdout"])


# ===========================================================================
# F. write_chunk — error recovery and edge cases
# ===========================================================================

class TestWriteChunkRobust(unittest.TestCase):
    """write_chunk must handle real-world failure modes."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_three_chunk_correct_order(self):
        self.sh("", 'write_chunk /big.py 0 3 "# Part 1\nimport os\n"')
        self.sh("", 'write_chunk /big.py 1 3 "# Part 2\ndef main():\n    pass\n"')
        r = self.sh("", 'write_chunk /big.py 2 3 "# Part 3\nif __name__ == \"__main__\":\n    main()"')
        self.assertTrue(r["success"])
        self.assertIn("complete", r["stdout"])

        content = self.vfs.read("/big.py")["content"]
        self.assertIn("import os", content)
        self.assertIn("def main():", content)
        self.assertIn('__name__', content)

    def test_chunk_status_shows_missing(self):
        self.sh("", 'write_chunk /partial.py 0 4 "chunk0\n"')
        self.sh("", 'write_chunk /partial.py 2 4 "chunk2\n"')
        r = self.sh("", "write_chunk_status /partial.py")
        self.assertTrue(r["success"])
        self.assertIn("1", r["stdout"])  # chunk 1 missing
        self.assertIn("3", r["stdout"])  # chunk 3 missing

    def test_duplicate_chunk_idempotent(self):
        """Sending the same chunk twice must not corrupt the file."""
        self.sh("", 'write_chunk /dup.py 0 2 "line1\n"')
        self.sh("", 'write_chunk /dup.py 0 2 "line1\n"')  # duplicate
        r = self.sh("", 'write_chunk /dup.py 1 2 "line2"')
        self.assertTrue(r["success"])

    def test_chunk_without_session_fails(self):
        """Sending chunk idx=1 without prior idx=0 must fail gracefully."""
        r = self.sh("", 'write_chunk /no_session.py 1 3 "data"')
        self.assertFalse(r["success"])
        self.assertIn("no active chunk session", r["stderr"])

    def test_chunk_total_mismatch_fails(self):
        self.sh("", 'write_chunk /mm.py 0 3 "a"')
        r = self.sh("", 'write_chunk /mm.py 1 5 "b"')  # total mismatch
        self.assertFalse(r["success"])
        self.assertIn("mismatch", r["stderr"])

    def test_chunk_sidecar_removed_after_completion(self):
        self.sh("", 'write_chunk /clean.py 0 2 "a\n"')
        self.sh("", 'write_chunk /clean.py 1 2 "b"')
        self.assertFalse(self.vfs._is_file("/clean.py.__chunks__"),
                         "Sidecar must be removed after all chunks received")

    def test_chunk_large_content_per_block(self):
        """Each chunk can hold ~40 lines of real code."""
        block0 = "\n".join([f"# line {i}" for i in range(40)])
        block1 = "\n".join([f"def func_{i}(): pass" for i in range(40)])
        self.sh("", f'write_chunk /huge.py 0 2 "{block0}"')
        r = self.sh("", f'write_chunk /huge.py 1 2 "{block1}"')
        self.assertTrue(r["success"])
        self.assertIn("complete", r["stdout"])
        content = self.vfs.read("/huge.py")["content"]
        self.assertIn("func_39", content)

    def test_status_on_nonexistent_file(self):
        r = self.sh("", "write_chunk_status /ghost.py")
        self.assertTrue(r["success"])
        self.assertIn("No active chunk session", r["stdout"])


# ===========================================================================
# G. Edge cases — Unicode, whitespace, special characters
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Content that trips up naive parsers."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_unicode_content(self):
        self.sh("", 'write /uni.txt "Ünïcödé: äöüß — 日本語 — 中文"')
        content = self.vfs.read("/uni.txt")["content"]
        self.assertIn("äöüß", content)
        self.assertIn("日本語", content)

    def test_whitespace_only_content(self):
        self.sh("", 'write /ws.txt "   \n  \n   "')
        content = self.vfs.read("/ws.txt")["content"]
        self.assertTrue(len(content.strip()) == 0)

    def test_path_with_dots(self):
        self.vfs.mkdir("/a.b.c", parents=True)
        self.sh("", 'write /a.b.c/file.test.py "x = 1"')
        r = self.sh("", "cat /a.b.c/file.test.py")
        self.assertTrue(r["success"])
        self.assertIn("x = 1", r["stdout"])

    def test_very_long_single_line(self):
        long_line = "x" * 5000
        self.sh("", f'write /long.txt "{long_line}"')
        content = self.vfs.read("/long.txt")["content"]
        self.assertEqual(len(content), 5000)

    def test_content_with_hash_not_treated_as_comment(self):
        self.sh("", 'write /hash.py "# This is a comment\ncode = 1  # inline comment"')
        content = self.vfs.read("/hash.py")["content"]
        self.assertIn("# This is a comment", content)
        self.assertIn("# inline comment", content)

    def test_content_with_semicolons_in_quotes(self):
        """Semicolons inside quoted content must NOT split commands."""
        self.sh("", 'write /semi.py "a = 1; b = 2; c = 3"')
        content = self.vfs.read("/semi.py")["content"]
        self.assertEqual(content, "a = 1; b = 2; c = 3")

    def test_content_with_pipe_in_quotes(self):
        """Pipe chars inside quoted content must NOT trigger pipe logic."""
        self.sh("", 'write /pipe.py "result = a | b | c"')
        content = self.vfs.read("/pipe.py")["content"]
        self.assertEqual(content, "result = a | b | c")

    def test_content_with_double_ampersand_in_quotes(self):
        """&& inside quoted content must NOT split commands."""
        self.sh("", 'write /amp.py "if a && b: pass"')
        content = self.vfs.read("/amp.py")["content"]
        self.assertIn("a && b", content)

    def test_trailing_newline_handling(self):
        """Write content without trailing newline, verify exact content."""
        self.sh("", 'write /exact.txt "no trailing"')
        content = self.vfs.read("/exact.txt")["content"]
        self.assertEqual(content, "no trailing")

    def test_grep_unicode_pattern(self):
        _prep(self.session, {"/i18n.txt": "Straße\nCafé\nNaïve\n"})
        r = self.sh("", "grep -n Café /i18n.txt")
        self.assertTrue(r["success"])
        self.assertIn("Café", r["stdout"])


# ===========================================================================
# H. Complex batch operations — realistic agent sessions
# ===========================================================================

class TestComplexBatchOperations(unittest.TestCase):
    """Multi-step operations that mirror what the agent actually does."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_scaffold_project_structure(self):
        """Agent creates a full project skeleton in one batch."""
        batch = (
            "mkdir -p /new/src/models; "
            "mkdir -p /new/src/services; "
            "mkdir -p /new/tests; "
            'write /new/src/__init__.py ""; '
            'write /new/src/models/__init__.py "from .user import User"; '
            'write /new/src/models/user.py "class User:\n    pass"; '
            'write /new/src/services/__init__.py ""; '
            'write /new/tests/__init__.py ""'
        )
        r = self.sh("", batch)
        self.assertTrue(r["success"], f"Scaffold batch failed: {r}")
        self.assertTrue(self.vfs._is_directory("/new/src/models"))
        self.assertTrue(self.vfs._is_directory("/new/tests"))
        content = self.vfs.read("/new/src/models/user.py")["content"]
        self.assertIn("class User:", content)

    def test_find_and_replace_across_files(self):
        """Agent updates imports after renaming a module."""
        _prep(self.session, {
            "/refactor/a.py": "from old_module import Foo\nprint(Foo)\n",
            "/refactor/b.py": "import old_module\nold_module.bar()\n",
            "/refactor/c.py": "# no import here\npass\n",
        })
        # Step 1: Find all files containing old_module
        r = self.sh("", "grep -rl old_module /refactor")
        self.assertTrue(r["success"])
        files = [f.strip() for f in r["stdout"].splitlines() if f.strip()]
        self.assertEqual(len(files), 2)

        # Step 2: For each file, read and verify the line
        for f in files:
            r = self.sh("", f"grep -n old_module {f}")
            self.assertTrue(r["success"])
            self.assertIn("old_module", r["stdout"])

    def test_conditional_write_or_fallback(self):
        """cat /existing || write /existing — only write if missing."""
        # File doesn't exist → write via ||
        self.sh("", 'cat /maybe.txt || write /maybe.txt "created by fallback"')
        content = self.vfs.read("/maybe.txt")["content"]
        self.assertIn("created by fallback", content)

        # File exists now → cat succeeds, write NOT executed
        self.sh("", 'cat /maybe.txt || write /maybe.txt "should not overwrite"')
        content = self.vfs.read("/maybe.txt")["content"]
        self.assertIn("created by fallback", content,
                       "|| must not execute write when cat succeeds")

    def test_write_verify_pipe_chain(self):
        """Write config, then verify specific values via grep+pipe."""
        self.sh("", 'write /settings.ini "host=0.0.0.0\nport=443\ndebug=false\nworkers=4"')
        r = self.sh("", "cat /settings.ini | grep port")
        self.assertTrue(r["success"])
        self.assertIn("443", r["stdout"])

        r = self.sh("", "cat /settings.ini | grep -v debug | wc -l")
        self.assertTrue(r["success"])
        count = int(r["stdout"].strip().split()[0])
        self.assertEqual(count, 3, "Must exclude the debug line")

    def test_incremental_file_building(self):
        """Agent builds a file incrementally via echo append."""
        self.sh("", 'echo "#!/usr/bin/env python3" > /build.py')
        self.sh("", 'echo "import os" >> /build.py')
        self.sh("", 'echo "" >> /build.py')
        self.sh("", 'echo "def main():" >> /build.py')
        self.sh("", 'echo "    print(42)" >> /build.py')

        content = self.vfs.read("/build.py")["content"]
        lines = content.splitlines()
        self.assertEqual(lines[0], "#!/usr/bin/env python3")
        self.assertIn("def main():", content)
        self.assertIn("print(42)", content)


# ===========================================================================
# I. vfs_view — scroll + focus for large files
# ===========================================================================

class TestVfsViewRealWorld(unittest.TestCase):
    """vfs_view as the agent actually uses it: locate, focus, work."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs
        # 100-line file
        lines = [f"# line {i}" for i in range(1, 51)]
        lines.append("")
        lines.append("class TargetClass:")
        lines.append("    def important_method(self):")
        lines.append("        return 'result'")
        lines.append("")
        lines.extend([f"# filler {i}" for i in range(56, 101)])
        _prep(self.session, {"/big.py": "\n".join(lines)})

    def test_scroll_to_class_shows_context(self):
        r = self.view("/big.py", scroll_to="TargetClass", context_lines=10)
        self.assertTrue(r["success"])
        self.assertIn("TargetClass", r["content"])
        self.assertIn("important_method", r["content"])
        self.assertEqual(r["match"]["matched_line"], 52)

    def test_scroll_to_method(self):
        r = self.view("/big.py", scroll_to="important_method", context_lines=6)
        self.assertTrue(r["success"])
        self.assertIn("return 'result'", r["content"])

    def test_line_range_precise(self):
        r = self.view("/big.py", line_start=52, line_end=54)
        self.assertTrue(r["success"])
        self.assertIn("TargetClass", r["content"])
        lines = r["content"].splitlines()
        self.assertLessEqual(len(lines), 3)

    def test_scroll_to_nonexistent_returns_hint(self):
        r = self.view("/big.py", scroll_to="NONEXISTENT_SYMBOL")
        self.assertFalse(r["success"])
        self.assertIn("hint", r)

    def test_close_others_workflow(self):
        """Open multiple files, then focus on one — others must close."""
        _prep(self.session, {"/other1.py": "a", "/other2.py": "b"})
        self.view("/other1.py")
        self.view("/other2.py")
        self.assertEqual(self.vfs.files["/other1.py"].state, "open")

        self.view("/big.py", scroll_to="TargetClass", close_others=True)
        self.assertEqual(self.vfs.files["/other1.py"].state, "closed")
        self.assertEqual(self.vfs.files["/other2.py"].state, "closed")
        self.assertEqual(self.vfs.files["/big.py"].state, "open")

    def test_view_returns_total_lines(self):
        r = self.view("/big.py")
        self.assertTrue(r["success"])
        self.assertEqual(r["total_lines"], 100)


# ===========================================================================
# J. Sed — line extraction for targeted reading
# ===========================================================================

class TestSedLineExtraction(unittest.TestCase):

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        lines = [f"line_{i}" for i in range(1, 51)]
        _prep(self.session, {"/numbered.txt": "\n".join(lines)})

    def test_sed_first_five(self):
        r = self.sh("", "sed -n '1,5p' /numbered.txt")
        self.assertTrue(r["success"])
        out_lines = r["stdout"].splitlines()
        self.assertEqual(len(out_lines), 5)
        self.assertEqual(out_lines[0], "line_1")
        self.assertEqual(out_lines[4], "line_5")

    def test_sed_middle_range(self):
        r = self.sh("", "sed -n '20,25p' /numbered.txt")
        self.assertTrue(r["success"])
        out_lines = r["stdout"].splitlines()
        self.assertEqual(len(out_lines), 6)
        self.assertEqual(out_lines[0], "line_20")

    def test_sed_last_lines(self):
        r = self.sh("", "sed -n '48,50p' /numbered.txt")
        self.assertTrue(r["success"])
        out_lines = r["stdout"].splitlines()
        self.assertEqual(out_lines[-1], "line_50")

    def test_sed_single_line(self):
        r = self.sh("", "sed -n '10p' /numbered.txt")
        self.assertTrue(r["success"])
        self.assertEqual(r["stdout"].strip(), "line_10")

    def test_sed_missing_file(self):
        r = self.sh("", "sed -n '1,5p' /ghost.txt")
        self.assertFalse(r["success"])


# ===========================================================================
# K. Exec command
# ===========================================================================

class TestExec(unittest.TestCase):

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)

    def test_exec_no_operand(self):
        r = self.sh("", "exec")
        self.assertFalse(r["success"])

    def test_exec_missing_file(self):
        r = self.sh("", "exec /nonexistent.py")
        self.assertFalse(r["success"])


# ===========================================================================
# L. Operator precedence and quoting stress tests
# ===========================================================================

class TestOperatorQuotingStress(unittest.TestCase):
    """Ensure operators inside quotes never split, operators outside always do."""

    def setUp(self):
        self.session = _session()
        self.sh = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_all_operators_in_double_quotes(self):
        """Content: 'a && b || c | d; e' must be written literally."""
        self.sh("", 'write /ops.txt "a && b || c | d; e"')
        content = self.vfs.read("/ops.txt")["content"]
        self.assertEqual(content, "a && b || c | d; e")

    def test_all_operators_in_single_quotes(self):
        self.sh("", "write /ops2.txt 'x && y || z | w; v'")
        content = self.vfs.read("/ops2.txt")["content"]
        self.assertEqual(content, "x && y || z | w; v")

    def test_semicolon_outside_quotes_splits(self):
        r = self.sh("", 'write /a.txt "1"; write /b.txt "2"')
        self.assertTrue(self.vfs._is_file("/a.txt"))
        self.assertTrue(self.vfs._is_file("/b.txt"))

    def test_pipe_outside_quotes_pipes(self):
        _prep(self.session, {"/data.txt": "alpha\nbeta\ngamma\n"})
        r = self.sh("", "cat /data.txt | grep beta")
        self.assertTrue(r["success"])
        self.assertIn("beta", r["stdout"])
        self.assertNotIn("alpha", r["stdout"])

    def test_nested_quotes_in_write(self):
        """Write Python code containing string literals."""
        self.sh("", 'write /nested.py "msg = \"hello world\"\nprint(msg)"')
        content = self.vfs.read("/nested.py")["content"]
        self.assertIn('"hello world"', content)
        self.assertIn("print(msg)", content)


if __name__ == "__main__":
    unittest.main()
