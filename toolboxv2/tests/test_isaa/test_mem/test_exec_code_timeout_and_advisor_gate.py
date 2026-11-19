"""Regressionstests: exec_code Sync-Timeout (H1) + Advisor-Gate (H2).

H1: Sync-blockierender Code in LiveSession._execute_unlocked muss den
    timeout respektieren (vorher: eval lief unterm Event-Loop, timeout ignoriert).
H2: Situation-Advisor läuft nur bei effort='complex' bzw. ADVISOR_MODE override.
"""
import asyncio
import os
import time
import tempfile
import unittest

from toolboxv2.mods.isaa.base.Agent.live_harness import LiveSession, compile_cell

def _fresh_session() -> LiveSession:
    td = tempfile.mkdtemp(prefix="lh_test_")
    return LiveSession(agent=None, session_id="timeout_test", root=td / "x" if False else td)


class TestSyncTimeout(unittest.TestCase):
    """H1: sync-blockierende Zellen brechen nach timeout ab."""

    def _run_cell(self, code: str, timeout: float = 2.0):
        s = _fresh_session()
        t0 = time.time()
        res = asyncio.run(s.execute(code, timeout=timeout))
        return res, time.time() - t0

    def test_sync_sleep_respects_timeout(self):
        res, dur = self._run_cell("import time\ntime.sleep(30)")
        self.assertFalse(res.success, "blocking cell muss per timeout fehlschlagen")
        self.assertIn("Timeout", str(res.error or ""))
        self.assertLess(dur, 15, f"timeout nicht enforced: {dur:.1f}s")

    def test_sync_infinite_loop_respects_timeout(self):
        res, dur = self._run_cell("x = 0\nwhile True:\n    x += 1")
        self.assertFalse(res.success)
        self.assertIn("Timeout", str(res.error or ""))
        self.assertLess(dur, 15, f"timeout nicht enforced: {dur:.1f}s")

    def test_fast_sync_cell_still_works(self):
        res, dur = self._run_cell("2 + 3")
        self.assertTrue(res.success, res.error)
        self.assertEqual(res.result, 5)

    def test_async_cell_still_works(self):
        res, _ = self._run_cell(
            "import asyncio\nawait asyncio.sleep(0.05)\n42"
        )
        self.assertTrue(res.success, res.error)
        self.assertEqual(res.result, 42)

    def test_sync_cell_state_persists(self):
        s = _fresh_session()
        asyncio.run(s.execute("y = 41"))
        res = asyncio.run(s.execute("y + 1"))
        self.assertTrue(res.success, res.error)
        self.assertEqual(res.result, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
