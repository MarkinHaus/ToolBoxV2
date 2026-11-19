"""Test: LiveSession path sanitization prevents WinError 123.

Reproduces: discord session_id 'discord_dm:268830485889810432' contains
colon → illegal on Windows → OSError [WinError 123] at mkdir().
"""
import tempfile
import unittest
from pathlib import Path


class TestSafePathComponent(unittest.TestCase):
    """Verify _safe_path_component strips all Windows-illegal chars."""

    def test_colon_replaced(self):
        from toolboxv2.mods.isaa.base.Agent.live_harness import _safe_path_component
        raw = "discord_dm:268830485889810432"
        safe = _safe_path_component(raw)
        self.assertNotIn(":", safe, "Colon must be replaced")
        self.assertIn("discord_dm", safe)

    def test_all_illegal_chars_replaced(self):
        from toolboxv2.mods.isaa.base.Agent.live_harness import _safe_path_component
        for ch in '<>:"/\\|?*\x00':
            safe = _safe_path_component(f"bad{ch}name")
            self.assertNotIn(ch, safe, f"Char {ch!r} must be replaced")

    def test_empty_falls_back(self):
        from toolboxv2.mods.isaa.base.Agent.live_harness import _safe_path_component
        self.assertEqual(_safe_path_component(""), "default")
        self.assertEqual(_safe_path_component(None), "default")


class TestLiveSessionColonCrash(unittest.TestCase):
    """H1: LiveSession with colon-containing session_id must not crash on mkdir."""

    def test_discord_dm_session_id_no_crash(self):
        from toolboxv2.mods.isaa.base.Agent.live_harness import LiveSession
        with tempfile.TemporaryDirectory() as td:
            # This is the exact ID that crashed: discord_dm:268830485889810432
            sid = "discord_dm:268830485889810432"
            ls = LiveSession(
                agent=None,
                session_id=sid,
                root=Path(td),
                persist=False,
            )
            # Must survive mkdir
            self.assertTrue(ls.harness_dir.exists())
            self.assertTrue(ls.work_dir.exists())
            # session_id must be sanitized
            self.assertNotIn(":", ls.session_id)
            self.assertIn("discord_dm", ls.session_id)

    def test_safe_session_id_still_works(self):
        from toolboxv2.mods.isaa.base.Agent.live_harness import LiveSession
        with tempfile.TemporaryDirectory() as td:
            ls = LiveSession(
                agent=None,
                session_id="clean_session_id",
                root=Path(td),
                persist=False,
            )
            self.assertEqual(ls.session_id, "clean_session_id")
            self.assertTrue(ls.harness_dir.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
