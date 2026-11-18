"""shell(background=True): sofort zurueck, Beobachter liest mit, Listener bekommt das Ende;
Shell-Sessions je scope getrennt (Jobs kollidieren nicht mit dem Chat).

Run with: python -m unittest toolboxv2.tests.test_isaa.test_shell_background
"""
import json
import platform
import threading
import time
import unittest

from toolboxv2.mods.isaa import module as M

posix_only = unittest.skipIf(platform.system() == "Windows", "POSIX shell syntax")


def sh(**kw) -> dict:
    return json.loads(M.shell_tool_function(**kw))


@posix_only
class BackgroundTest(unittest.TestCase):
    def setUp(self):
        M.cleanup_sessions()
        self.events, self.got = [], threading.Event()
        self.fn = lambda e: (self.events.append(e), self.got.set())
        M.add_background_listener(self.fn)

    def tearDown(self):
        M.remove_background_listener(self.fn)
        M.cleanup_sessions()

    def test_returns_at_once_and_notifies(self):
        token = M.SHELL_CONTEXT.set({"agent": "self", "session": "default"})
        try:
            t0 = time.time()
            r = sh(command="echo start; sleep 3; echo ende", session_id="B",
                   background=True)
        finally:
            M.SHELL_CONTEXT.reset(token)
        self.assertLess(time.time() - t0, 2.0)            # wartet NICHT auf das Ende
        self.assertEqual((r["status"], r["background"]), ("running", True))
        mid = sh(session_id="B")                           # Zwischenstand aus dem Puffer
        self.assertEqual(mid["status"], "running")
        self.assertTrue(self.got.wait(15), "kein Ende gemeldet")
        e = self.events[0]
        self.assertEqual((e["agent"], e["session"], e["session_id"]), ("self", "default", "B"))
        self.assertIn("ende", e["tail"])
        self.assertGreaterEqual(e["seconds"], 2.5)
        done = sh(session_id="B")                          # Ergebnis abholbar, danach Session frei
        self.assertEqual(done["status"], "finished")
        self.assertIn("ende", done["stdout"])
        self.assertEqual(sh(command="echo wieder-frei", session_id="B")["stdout"], "wieder-frei")

    def test_exit_code_reported(self):
        sh(command="sleep 1; false", session_id="A", background=True)
        self.assertTrue(self.got.wait(15))
        self.assertEqual(self.events[0]["exit_code"], 1)
        self.assertFalse(self.events[0]["success"])

    def test_new_command_while_running_goes_to_stdin(self):
        sh(command="read x; echo got:$x", session_id="C", background=True)
        r = sh(command="hallo", session_id="C")
        self.assertTrue(r["background"])
        self.assertTrue(self.got.wait(15))
        self.assertIn("got:hallo", self.events[0]["tail"])

    def test_scope_separates_sessions(self):
        sh(command="cd /tmp", session_id="A")
        token = M.SHELL_CONTEXT.set({"scope": "job-nightly"})
        try:
            self.assertNotEqual(sh(command="pwd", session_id="A")["stdout"], "/tmp")
            self.assertIn("job-nightly/A", M._session_store)
        finally:
            M.SHELL_CONTEXT.reset(token)
        self.assertEqual(sh(command="pwd", session_id="A")["stdout"], "/tmp")


if __name__ == "__main__":
    unittest.main()
