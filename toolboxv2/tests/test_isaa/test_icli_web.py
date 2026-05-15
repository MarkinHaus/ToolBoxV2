"""unittest for icli_web."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestApiKey(unittest.TestCase):

    def setUp(self):
        self._orig = os.environ.pop("ICLI_WEB_API_KEY", None)

    def tearDown(self):
        if self._orig: os.environ["ICLI_WEB_API_KEY"] = self._orig
        else: os.environ.pop("ICLI_WEB_API_KEY", None)

    def test_env_wins(self):
        os.environ["ICLI_WEB_API_KEY"] = "env-123"
        from toolboxv2.mods.icli_web.server import load_key
        self.assertEqual(load_key(), "env-123")

    def test_file_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            kf = Path(d) / "key"
            kf.write_text("file-abc")
            with patch("toolboxv2.mods.icli_web.server._KEY_FILE", kf):
                from toolboxv2.mods.icli_web.server import load_key
                self.assertEqual(load_key(), "file-abc")

    def test_auto_generate(self):
        with tempfile.TemporaryDirectory() as d:
            kf = Path(d) / "key"
            with patch("toolboxv2.mods.icli_web.server._KEY_FILE", kf):
                from toolboxv2.mods.icli_web.server import load_key
                k = load_key()
                self.assertGreater(len(k), 20)
                self.assertTrue(kf.exists())


class TestSentenceSplit(unittest.TestCase):

    def test_no_split_short(self):
        from toolboxv2.mods.icli_web.client import _first_sentence_end
        self.assertIsNone(_first_sentence_end("Hi."))

    def test_split_long(self):
        from toolboxv2.mods.icli_web.client import _first_sentence_end
        s = "This is a longer first sentence that crosses threshold. Second."
        idx = _first_sentence_end(s)
        self.assertEqual(s[idx], ".")
        self.assertGreater(idx, 30)

    def test_split_german(self):
        from toolboxv2.mods.icli_web.client import _first_sentence_end
        s = "Das ist ein ausreichend langer Satz! Noch einer."
        idx = _first_sentence_end(s)
        self.assertEqual(s[idx], "!")

    def test_question_mark(self):
        from toolboxv2.mods.icli_web.client import _first_sentence_end
        s = "Do you know what time it is now in Berlin? Yes I do."
        idx = _first_sentence_end(s)
        self.assertEqual(s[idx], "?")


class TestCoercion(unittest.TestCase):

    def test_tts_enum_conversion(self):
        try:
            from toolboxv2.mods.isaa.base.audio_io.Tts import (
                TTSBackend, TTSEmotion
            )
        except ImportError:
            self.skipTest("audio_io not available")
        from toolboxv2.mods.icli_web.client import _coerce_tts_kwargs
        out = _coerce_tts_kwargs({
            "backend": "piper", "emotion": "friendly",
            "voice": "en_US-amy-medium", "speed": "1.25",
        })
        self.assertEqual(out["backend"], TTSBackend.PIPER)
        self.assertEqual(out["emotion"], TTSEmotion.FRIENDLY)
        self.assertEqual(out["speed"], 1.25)

    def test_tts_bad_enum_dropped(self):
        from toolboxv2.mods.icli_web.client import _coerce_tts_kwargs
        out = _coerce_tts_kwargs({"backend": "nonexistent"})
        self.assertNotIn("backend", out)

    def test_style_prompt_passes_through(self):
        from toolboxv2.mods.icli_web.client import _coerce_tts_kwargs
        out = _coerce_tts_kwargs({"style_prompt": "warm narrator"})
        self.assertEqual(out["style_prompt"], "warm narrator")

    def test_stt_coercion(self):
        try:
            from toolboxv2.mods.isaa.base.audio_io.Stt import STTBackend
        except ImportError:
            self.skipTest("audio_io not available")
        from toolboxv2.mods.icli_web.client import _coerce_stt_kwargs
        out = _coerce_stt_kwargs({
            "backend": "faster_whisper", "model": "small",
            "language": "de", "device": "cpu",
        })
        self.assertEqual(out["backend"], STTBackend.FASTER_WHISPER)
        self.assertEqual(out["model"], "small")


class TestFastAPIRoutes(unittest.TestCase):

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")
        from toolboxv2.mods.icli_web.server import build_app, R
        R.icli_ws = None
        R.orbs.clear()
        R.orb_by_cid.clear()
        R.task_cache.clear()
        R.monitor_subs.clear()
        self.key = "test-key"
        self.app = build_app(self.key)
        from fastapi.testclient import TestClient
        self.client = TestClient(self.app)

    def test_health_open(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertFalse(body["icli_connected"])
        self.assertEqual(body["orb_sessions"], 0)

    def test_orb_page_open(self):
        r = self.client.get("/orb")
        self.assertEqual(r.status_code, 200)
        self.assertIn("audioVisualizer", r.text)
        self.assertIn("settings", r.text)

    def test_monitor_page_open(self):
        r = self.client.get("/monitor")
        self.assertEqual(r.status_code, 200)

    def test_capabilities_needs_auth(self):
        r = self.client.get("/capabilities")
        self.assertEqual(r.status_code, 401)

    def test_capabilities_with_header(self):
        r = self.client.get("/capabilities",
                            headers={"X-API-Key": self.key})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("tts", data)
        self.assertIn("stt", data)

    def test_capabilities_with_query(self):
        r = self.client.get(f"/capabilities?key={self.key}")
        self.assertEqual(r.status_code, 200)

    def test_wrong_key(self):
        r = self.client.get("/capabilities",
                            headers={"X-API-Key": "wrong"})
        self.assertEqual(r.status_code, 401)


class TestRouterState(unittest.TestCase):

    def test_router_single_icli(self):
        from toolboxv2.mods.icli_web.server import Router
        r = Router()
        self.assertIsNone(r.icli_ws)
        r.icli_ws = MagicMock()
        self.assertIsNotNone(r.icli_ws)

    def test_cid_mapping(self):
        from toolboxv2.mods.icli_web.server import Router
        r = Router()
        orb1 = MagicMock()
        orb2 = MagicMock()
        r.orb_by_cid["c1"] = orb1
        r.orb_by_cid["c2"] = orb2
        self.assertIs(r.orb_by_cid["c1"], orb1)
        self.assertIs(r.orb_by_cid["c2"], orb2)
        self.assertEqual(len(r.orb_by_cid), 2)


if __name__ == "__main__":
    unittest.main()
