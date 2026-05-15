# tests/test_persistence.py
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import dotenv_values

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import (
    _persist_key, _next_slot,
)


class TestPersistKey(unittest.TestCase):
    def test_write_roundtrip(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "GROQ_API_KEY", "gsk_abc123")
            self.assertEqual(dotenv_values(p)["GROQ_API_KEY"], "gsk_abc123")

    def test_preserves_hash_in_value(self):
        """U1 regression: # must not be treated as comment."""
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "K", "sk-abc#def#ghi")
            self.assertEqual(dotenv_values(p)["K"], "sk-abc#def#ghi")

    def test_preserves_spaces_and_quotes(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "K", 'has "quotes" and spaces')
            self.assertEqual(dotenv_values(p)["K"], 'has "quotes" and spaces')

    def test_refuses_overwrite(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "K", "v1")
            with self.assertRaises(ValueError):
                _persist_key(p, "K", "v2")
            self.assertEqual(dotenv_values(p)["K"], "v1")  # untouched

    def test_rejects_newline_in_value(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            with self.assertRaises(ValueError):
                _persist_key(p, "K", "a\nb")

    def test_rejects_invalid_key(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            for bad in ("K=X", "K\nX", "K X", ""):
                with self.assertRaises(ValueError):
                    _persist_key(p, bad, "v")

    def test_creates_parent_dir(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "deep" / "nested" / ".env"
            _persist_key(p, "K", "v")
            self.assertTrue(p.exists())

    @unittest.skipIf(os.name == "nt", "POSIX only")
    def test_sets_0600_permissions(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "K", "v")
            mode = stat.S_IMODE(p.stat().st_mode)
            self.assertEqual(mode, 0o600)

    def test_append_preserves_existing(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "A", "1")
            _persist_key(p, "B", "2")
            _persist_key(p, "C", "3")
            vals = dotenv_values(p)
            self.assertEqual(vals, {"A": "1", "B": "2", "C": "3"})


class TestNextSlotFromFile(unittest.TestCase):
    def test_reads_from_file_not_environ(self):
        """U2 regression: slot must be derived from .env, not os.environ."""
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "GROQ_API_KEY", "k1")
            _persist_key(p, "GROQ_API_KEY_2", "k2")
            # os.environ is clean — function must still see both
            self.assertEqual(_next_slot("GROQ_API_KEY", p), "GROQ_API_KEY_3")

    def test_missing_file_returns_primary(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "does-not-exist.env"
            self.assertEqual(_next_slot("FOO_API_KEY", p), "FOO_API_KEY")

    def test_fills_gaps(self):
        with TemporaryDirectory() as td:
            p = Path(td) / ".env"
            _persist_key(p, "K", "v")
            _persist_key(p, "K_3", "v3")
            # 2 is free
            self.assertEqual(_next_slot("K", p), "K_2")


class TestAbsolutePath(unittest.TestCase):
    def test_relative_resolves_before_write(self):
        with TemporaryDirectory() as td:
            prev = Path.cwd()
            try:
                os.chdir(td)
                _persist_key(Path(".env"), "K", "v")
                env_file = Path(td) / ".env"
                self.assertTrue(env_file.exists())
                self.assertEqual(dotenv_values(env_file)["K"], "v")
            finally:
                os.chdir(prev)

if __name__ == "__main__":
    unittest.main()
