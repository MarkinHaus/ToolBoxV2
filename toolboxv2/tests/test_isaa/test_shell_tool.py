"""
Tests for the ISAA host shell tool (toolboxv2.mods.isaa.module.shell_tool_function).

Regression coverage for:
1. `'NoneType' object has no attribute 'startswith'` when the tool is called
   without a command (poll / resume / user_input only)
2. swallowed / mis-attributed output (silent commands, late output)
3. resume: polling a running command returns its output AND its completion
4. honest status + exit code instead of a permanent "running"

Run with: python -m unittest toolboxv2.tests.test_isaa.test_shell_tool
"""

import json
import platform
import time
import unittest
from unittest.mock import patch

from toolboxv2.mods.isaa import module as isaa_module
from toolboxv2.mods.isaa.module import (
    ShellSession,
    cleanup_sessions,
    shell_tool_function,
)

IS_WINDOWS = platform.system() == "Windows"
posix_only = unittest.skipIf(IS_WINDOWS, "POSIX shell syntax")


def sh(**kwargs) -> dict:
    return json.loads(shell_tool_function(**kwargs))


class ShellToolTestCase(unittest.TestCase):
    def setUp(self):
        cleanup_sessions()

    def tearDown(self):
        cleanup_sessions()


class TestNoneCommand(ShellToolTestCase):
    """command=None must never be dereferenced (the reported AttributeError)."""

    def test_poll_without_command_on_fresh_session(self):
        r = sh(session_id="A")
        self.assertTrue(r["success"])
        self.assertEqual(r["status"], "idle")

    def test_poll_with_explicit_none(self):
        r = sh(command=None, session_id="A")
        self.assertEqual(r["status"], "idle")

    def test_empty_and_whitespace_command_is_a_poll(self):
        self.assertEqual(sh(command="")["status"], "idle")
        self.assertEqual(sh(command="   \n")["status"], "idle")

    def test_user_input_without_command(self):
        r = sh(user_input="echo via-input")
        self.assertIn("via-input", r["stdout"])

    def test_none_for_optional_args(self):
        r = sh(command="echo ok", session_id=None, timeout=None, idle_timeout=None)
        self.assertEqual(r["session_id"], "A")
        self.assertEqual(r["stdout"], "ok")

    def test_non_string_command_does_not_crash(self):
        self.assertEqual(sh(command=123)["status"], "idle")


class TestCompletion(ShellToolTestCase):
    def test_finished_status_and_exit_code(self):
        r = sh(command="echo hi")
        self.assertEqual(r["stdout"], "hi")
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 0)
        self.assertTrue(r["success"])
        self.assertNotIn("note", r)

    def test_returns_without_waiting_for_idle_timeout(self):
        sh(command="echo warmup")
        t = time.time()
        r = sh(command="echo fast", idle_timeout=5.0)
        self.assertEqual(r["status"], "finished")
        self.assertLess(time.time() - t, 2.0)

    @posix_only
    def test_failing_command_reports_exit_code(self):
        r = sh(command="ls /definitely/not/here")
        self.assertEqual(r["status"], "finished")
        self.assertNotEqual(r["exit_code"], 0)
        self.assertFalse(r["success"])
        self.assertIn("not/here", r["stderr"])

        r = sh(command="(exit 7)")
        self.assertEqual(r["exit_code"], 7)

    @posix_only
    def test_output_without_trailing_newline(self):
        r = sh(command="printf abc")
        self.assertEqual(r["stdout"], "abc")
        self.assertEqual(r["status"], "finished")

    @posix_only
    def test_marker_never_leaks_into_output(self):
        r = sh(command="echo one; echo two 1>&2")
        self.assertEqual(r["stdout"], "one")
        self.assertEqual(r["stderr"], "two")
        self.assertNotIn("__TB_DONE", json.dumps(r))

    @posix_only
    def test_multiline_heredoc_and_state_persistence(self):
        r = sh(command="cat <<'EOT'\nline 1\n  line 2\nEOT\nexport TB_T=42\ncd /tmp")
        self.assertEqual(r["stdout"], "line 1\n  line 2")
        r = sh(command='echo "$TB_T $(pwd)"')
        self.assertEqual(r["stdout"], "42 /tmp")
        self.assertEqual(r["info"], "Resumed session")

    @posix_only
    def test_large_output_is_complete(self):
        r = sh(command="seq 1 100000")
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 100000)
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[-1], "100000")
        self.assertEqual(r["status"], "finished")

    @posix_only
    def test_return_in_sourced_script_still_completes(self):
        r = sh(command="echo before\nreturn 3\necho never")
        self.assertEqual(r["stdout"], "before")
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 3)


@posix_only
class TestResume(ShellToolTestCase):
    def test_silent_command_then_resume_delivers_output(self):
        r = sh(command="sleep 1.5; echo late-result", idle_timeout=0.3)
        self.assertEqual(r["status"], "running")
        self.assertEqual(r["stdout"], "")
        self.assertIn("note", r)

        # resume: blocks until the command completes, no command needed
        r = sh(session_id="A", idle_timeout=5.0)
        self.assertEqual(r["stdout"], "late-result")
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 0)
        self.assertEqual(r["info"], "Resumed session")

        self.assertEqual(sh(session_id="A")["status"], "idle")

    def test_resume_collects_output_incrementally(self):
        r = sh(command="echo first; sleep 1.2; echo second; (exit 4)", idle_timeout=0.3)
        self.assertEqual(r["stdout"], "first")
        self.assertEqual(r["status"], "running")
        r = sh(idle_timeout=5.0)
        self.assertEqual(r["stdout"], "second")
        self.assertEqual(r["exit_code"], 4)
        self.assertFalse(r["success"])

    def test_command_while_busy_goes_to_the_running_program(self):
        """Multi-turn CLI: `command` reaches the running program's stdin."""
        r = sh(command='echo started; read a; echo "got:$a"; read b; echo "got:$b"',
               idle_timeout=0.3)
        self.assertEqual(r["status"], "running")
        r = sh(command="first turn", idle_timeout=0.4)
        self.assertEqual(r["stdout"], "got:first turn")
        self.assertIn("stdin", r["info"])
        self.assertEqual(r["status"], "running")
        # other labels stay usable for independent commands meanwhile
        self.assertEqual(sh(command="echo other", session_id="B")["stdout"], "other")
        r = sh(command="second turn")
        self.assertEqual(r["stdout"], "got:second turn")
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 0)
        # program is gone → the next command is a normal shell command again
        r = sh(command="echo back-in-shell")
        self.assertEqual(r["stdout"], "back-in-shell")
        self.assertEqual(r["info"], "Resumed session")

    def test_multi_turn_repl_session(self):
        """A real REPL driven over several calls, via command and user_input."""
        import sys
        sh(command=f"'{sys.executable}' -i -q -u", idle_timeout=0.8)
        r = sh(command="x = 20 + 1", idle_timeout=0.5)
        self.assertEqual(r["status"], "running")
        r = sh(user_input="print('val', x * 2)", idle_timeout=0.5)
        self.assertIn("val 42", r["stdout"])
        # multiline block in one turn (blank line terminates the block)
        r = sh(command="for i in range(2):\n    print('i', i)\n", idle_timeout=0.5)
        self.assertIn("i 0\ni 1", r["stdout"])
        r = sh(command="raise SystemExit(3)", idle_timeout=3.0)
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 3)
        self.assertEqual(sh(command="echo shell-alive")["stdout"], "shell-alive")

    def test_pending_output_is_not_lost_when_sending_input(self):
        sh(command="echo early; sleep 0.5; echo mid; read a; echo end:$a", idle_timeout=0.2)
        time.sleep(0.8)
        r = sh(command="x")
        self.assertEqual(r["stdout"], "mid\nend:x")
        self.assertEqual(r["status"], "finished")

    def test_output_of_finished_previous_command_is_not_misattributed(self):
        sh(command="sleep 0.8; echo old-output", idle_timeout=0.2)
        time.sleep(1.2)  # finishes un-polled
        r = sh(command="echo new-output")
        self.assertEqual(r["stdout"], "new-output")
        self.assertEqual(r["late_output_of_previous_command"]["stdout"], "old-output")
        self.assertEqual(r["late_output_of_previous_command"]["exit_code"], 0)

    def test_background_job_output_is_captured_later(self):
        r = sh(command="(sleep 0.6; echo bg-out) &\necho fg")
        self.assertEqual(r["status"], "finished")
        self.assertIn("fg", r["stdout"])
        time.sleep(1.0)
        self.assertIn("bg-out", sh(session_id="A")["stdout"])

    def test_interactive_prompt_roundtrip(self):
        r = sh(command='printf "name? "; read n; echo "hello $n"', idle_timeout=0.4)
        self.assertEqual(r["status"], "waiting_for_input")
        self.assertEqual(r["stdout"], "name?")
        r = sh(user_input="tb")
        self.assertEqual(r["stdout"], "hello tb")
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 0)

    def test_hard_timeout_returns_running(self):
        t = time.time()
        r = sh(command="while true; do echo tick; sleep 0.2; done", timeout=1.0, idle_timeout=1.0)
        self.assertLess(time.time() - t, 4.0)
        self.assertEqual(r["status"], "running")
        self.assertIn("tick", r["stdout"])
        self.assertEqual(sh(command="reset-session")["status"], "ready")
        self.assertEqual(sh(command="echo clean")["stdout"], "clean")


class TestSessions(ShellToolTestCase):
    def test_labels_are_case_insensitive_and_validated(self):
        self.assertEqual(sh(command="echo x", session_id="b")["session_id"], "B")
        r = sh(command="echo x", session_id="Z")
        self.assertFalse(r["success"])
        self.assertIn("unknown shell session", r["stderr"])
        self.assertFalse(sh(command="reset-session Z")["success"])

    def test_new_session_resets_state(self):
        sh(command="echo a")
        r = sh(command="echo b", new_session=True)
        self.assertEqual(r["info"], "Session reset")
        self.assertEqual(r["stdout"], "b")

    @posix_only
    def test_exit_inside_script_is_reported_and_session_recovers(self):
        r = sh(command="echo bye; exit 5")
        self.assertEqual(r["stdout"], "bye")
        self.assertEqual(r["status"], "finished")
        self.assertEqual(r["exit_code"], 5)
        self.assertIn("exited", r["note"])
        r = sh(command="echo again")
        self.assertEqual(r["stdout"], "again")
        self.assertEqual(r["info"], "New session started")

    def test_meta_commands(self):
        sh(command="echo x")
        sessions = sh(command="list-sessions")["sessions"]
        self.assertEqual(sessions[0]["session_id"], "A")
        self.assertFalse(sessions[0]["busy"])
        self.assertEqual(sessions[0]["last_exit_code"], 0)
        self.assertTrue(sh(command="list-processes")["success"])
        self.assertIn("Usage", sh(command="kill-process")["error"])
        self.assertEqual(sh(command="cleanup-sessions")["cleaned"], 1)

    def test_raw_kill_is_blocked_but_input_is_not(self):
        self.assertEqual(sh(command="pkill -f foo")["error"], "RAW_KILL_FORBIDDEN")
        # user_input is data for a prompt, not a script — no guardrail, no crash
        self.assertTrue(sh(user_input="")["success"])

    @posix_only
    def test_kill_process_stops_running_command(self):
        sh(command="sleep 30", idle_timeout=0.3)
        procs = sh(command="list-processes")["processes"]
        sleeper = [p for p in procs if p["name"] == "sleep"]
        self.assertEqual(len(sleeper), 1)
        self.assertTrue(sh(command=f"kill-process {sleeper[0]['pid']}")["success"])
        r = sh(idle_timeout=3.0)
        self.assertEqual(r["status"], "finished")
        self.assertNotEqual(r["exit_code"], 0)

    def test_script_file_is_removed_on_terminate(self):
        import os
        s = ShellSession(label="C")
        s.exec_script("echo x")
        s.read_output(timeout=2.0)
        path = s._script_path
        self.assertTrue(os.path.exists(path))
        s.terminate()
        self.assertFalse(os.path.exists(path))


@posix_only
class TestKillProcess(ShellToolTestCase):
    """kill-process must never be able to take the agent harness down."""

    def _start_sleepers(self, n, label="A"):
        sh(command="\n".join(["sleep 60 &"] * n), session_id=label)
        pids = [p["pid"] for p in sh(command="list-processes")["processes"]
                if p["name"] == "sleep" and p["session"] == label]
        self.assertEqual(len(pids), n)
        return pids

    def test_multi_pid_exactly_as_the_agent_called_it(self):
        import psutil
        a, b, c = self._start_sleepers(3)
        r = sh(command=f"kill-process {a}; kill-process {b}; kill-process {c}", session_id="C")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["killed"], [a, b, c])
        time.sleep(0.3)
        for pid in (a, b, c):
            self.assertFalse(psutil.pid_exists(pid) and
                             psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)

    def test_multi_pid_other_separators(self):
        a, b, c, d = self._start_sleepers(4)
        self.assertEqual(sh(command=f"kill-process {a} {b}")["killed"], [a, b])
        self.assertEqual(sh(command=f"kill-process {c}, {d}")["killed"], [c, d])

    def test_mixed_results_are_reported_per_pid(self):
        import os
        (a,) = self._start_sleepers(1)
        r = sh(command=f"kill-process {a} {os.getpid()} 4999999")
        self.assertFalse(r["success"])
        self.assertEqual(r["killed"], [a])
        self.assertEqual([x["error"] for x in r["results"][1:]],
                         ["PROTECTED_PROCESS", "PROCESS_NOT_FOUND"])

    def test_cannot_be_combined_with_shell_commands(self):
        (a,) = self._start_sleepers(1)
        r = sh(command=f"kill-process {a}; echo done")
        self.assertFalse(r["success"])
        self.assertIn("Usage", r["error"])
        self.assertIn("Nothing was killed", r["stderr"])
        self.assertEqual(len(self._pids_alive([a])), 1)

    def _pids_alive(self, pids):
        import psutil
        return [p for p in pids if psutil.pid_exists(p)]

    def test_harness_and_all_its_ancestors_are_protected(self):
        import os
        import psutil
        lineage = [os.getpid()] + [p.pid for p in psutil.Process().parents()]
        self.assertGreaterEqual(len(lineage), 2)
        for pid in lineage:
            if pid == 0:
                continue
            r = sh(command=f"kill-process {pid}")
            self.assertEqual(r["error"], "PROTECTED_PROCESS", (pid, r))

    def test_session_shell_is_protected(self):
        sh(command="echo x")
        pid = sh(command="list-sessions")["sessions"][0]["pid"]
        self.assertEqual(sh(command=f"kill-process {pid}")["error"], "PROTECTED_PROCESS")

    def test_foreign_process_is_refused(self):
        import subprocess
        import sys
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            r = sh(command=f"kill-process {foreign.pid}")
            self.assertEqual(r["error"], "NOT_A_SESSION_PROCESS")
            self.assertIsNone(foreign.poll())
        finally:
            foreign.kill()
            foreign.wait()

    def test_detached_session_process_is_still_killable(self):
        """Orphan of a reset session: recognised via the session env marker."""
        import subprocess
        import sys
        env = dict(isaa_module.os.environ, __TB_SESSION_ID__="some-old-session",
                   __TB_SESSION_LABEL__="B")
        orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=env)
        try:
            r = sh(command=f"kill-process {orphan.pid}")
            self.assertTrue(r["success"], r)
            self.assertEqual(r["session"], "B")
            orphan.wait(timeout=3)
        finally:
            if orphan.poll() is None:
                orphan.kill()

    def test_killing_the_harness_grandparent_does_not_kill_the_harness(self):
        """
        Reproduces the crash: launcher -> wrapper -> harness (the usual
        `tb.exe -> venv python -> python` chain). The old code only protected
        harness + direct parent, accepted the launcher PID and then killed
        'all its children' — i.e. the harness itself.
        """
        import os
        import subprocess
        import sys
        import tempfile
        d = tempfile.mkdtemp()
        out = os.path.join(d, "result.json")
        harness = os.path.join(d, "harness.py")
        with open(harness, "w") as f:
            f.write(
                "import sys\n"
                "from toolboxv2.mods.isaa.module import shell_tool_function\n"
                "r = shell_tool_function(command='kill-process ' + sys.argv[1])\n"
                "open(sys.argv[2], 'w').write(r)\n"
            )
        wrapper = ("import subprocess, sys; "
                   "sys.exit(subprocess.call([sys.executable] + sys.argv[1:]))")
        launcher = ("import os, subprocess, sys; "
                    "sys.exit(subprocess.call([sys.executable, '-c', sys.argv[1], "
                    "sys.argv[2], str(os.getpid()), sys.argv[3]]))")
        rc = subprocess.call([sys.executable, "-c", launcher, wrapper, harness, out],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        self.assertEqual(rc, 0, "launcher chain was killed")
        self.assertTrue(os.path.exists(out), "harness died before returning a result")
        with open(out) as f:
            self.assertEqual(json.load(f)["error"], "PROTECTED_PROCESS")

    def test_kill_guardrail_aliases(self):
        for cmd in ("spps -Id 1", "(Get-Process -Id 5).Kill()",
                    "wmic process where processid=5 delete"):
            self.assertEqual(sh(command=cmd)["error"], "RAW_KILL_FORBIDDEN", cmd)


class TestMarkerPlumbing(ShellToolTestCase):
    """Shell-specific invoke lines (PowerShell / cmd cannot run on this host,
    so at least pin their shape and the parsing of what they emit)."""

    def _line(self, shell_exe: str, token="__TB_DONE_x_1__") -> str:
        s = ShellSession.__new__(ShellSession)
        with patch.object(isaa_module, "detect_shell", return_value=(shell_exe, "-c")):
            return s._invoke_line("/tmp/s", token), s._invoke_line("/tmp/s", None)

    def test_invoke_lines_are_single_line_and_never_match_the_marker(self):
        import re
        for exe in ("/bin/bash", r"C:\Program Files\PowerShell\7\pwsh.exe",
                    "powershell.EXE", "cmd.exe"):
            with_marker, plain = self._line(exe)
            self.assertNotIn("\n", with_marker)
            self.assertIn("__TB_DONE_x_1__:", with_marker)
            self.assertNotIn("__TB_DONE", plain)
            # an echoed invoke line must not be mistaken for a completed command
            self.assertIsNone(re.search(r"__TB_DONE_x_1__:-?\d+", with_marker))

    def test_shell_specific_invocation(self):
        self.assertTrue(self._line("pwsh")[1].startswith(". '"))
        self.assertTrue(self._line("cmd.exe")[1].startswith('call "'))
        self.assertIn("%^errorlevel%", self._line("cmd.exe")[0])
        self.assertIn("$LASTEXITCODE", self._line("powershell.exe")[0])

    def test_echoed_invoke_line_is_stripped_and_crlf_marker_is_parsed(self):
        s = ShellSession(label="C")
        try:
            token = s._next_token()
            s._pending = token
            # what cmd.exe produces: prompt + echoed line, output, CRLF marker
            s.out_queue.put(f"C:\\>call \"x.bat\" & call echo {token}:%^errorlevel%\r\n".encode())
            s.out_queue.put(b"real output\r\n")
            s.out_queue.put(f"{token}:".encode())      # marker split across chunks
            s.out_queue.put(b"3\r\n")
            out = s.read_output(timeout=1.0)
            self.assertTrue(out["done"])
            self.assertEqual(out["exit_code"], 3)
            self.assertEqual(out["stdout"].strip(), "real output")
            self.assertFalse(s.busy)
        finally:
            s.terminate()

    def test_fallback_when_marker_is_unsupported(self):
        """If the probe fails the tool degrades to idle detection, never blocks."""
        with patch.object(ShellSession, "_probe_marker", lambda self: None):
            r = sh(command="echo legacy", idle_timeout=0.4)
            self.assertEqual(r["stdout"], "legacy")
            self.assertEqual(r["status"], "unknown")
            self.assertNotIn("exit_code", r)
            # no completion tracking → never reported busy
            self.assertEqual(sh(command="echo again", idle_timeout=0.4)["stdout"], "again")


if __name__ == "__main__":
    unittest.main()
