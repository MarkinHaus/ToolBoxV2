"""
Runnable checks for the P2 fixes (T7 waveform sync, T6 real speaker embedder).

icli.py can't be imported standalone (prompt_toolkit + deep toolboxv2 chain), so
this extracts the REAL `_omni_wave` function and `_PyannoteOmniEmbedder` class
definitions out of the shipped icli.py source via ast and execs just those into a
clean namespace. It tests the actual code, not a copy.

unittest only. Run:
    python toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p2_fixes.py
"""
from __future__ import annotations

import ast
import asyncio
import sys
import types
import unittest
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    root = here
    for _ in range(20):
        if (root / "toolboxv2" / "flows" / "isaa" / "icli.py").exists():
            return root
        root = root.parent
    raise FileNotFoundError("repo root not found")


ICLI = _repo_root() / "toolboxv2" / "flows" / "isaa" / "icli.py"


def _extract(names):
    """Pull top-level def/class nodes named in `names` from icli.py and exec them
    in an isolated namespace with the few module-level deps they touch stubbed."""
    tree = ast.parse(ICLI.read_text())
    wanted = {n: None for n in names}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            wanted[node.name] = node
    missing = [n for n, v in wanted.items() if v is None]
    if missing:
        raise AssertionError(f"definitions not found in icli.py: {missing}")

    ns = {
        "_OMNI_BLOCKS": " ▁▂▃▄▅▆▇█",
        "os": __import__("os"),
        "print_status": lambda *a, **k: None,
        "logger": types.SimpleNamespace(debug=lambda *a, **k: None),
    }
    mod = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    code = compile(mod, filename="icli_extract", mode="exec")
    exec(code, ns)
    return ns


NS = _extract(["_omni_wave", "_PyannoteOmniEmbedder", "_pyannote_speaker_model"])
_omni_wave = NS["_omni_wave"]
_PyannoteOmniEmbedder = NS["_PyannoteOmniEmbedder"]
_pyannote_speaker_model = NS["_pyannote_speaker_model"]


def _load_omni():
    p = _repo_root() / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io" / "omni.py"
    spec = importlib.util.spec_from_file_location("omni_iso_p2bug", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["omni_iso_p2bug"] = m
    spec.loader.exec_module(m)
    return m


import importlib.util  # noqa: E402
omni = _load_omni()


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --- T7: waveform scrolls while playing, stays stable otherwise ---------------
class TestT7WaveformScroll(unittest.TestCase):
    def test_no_anim_is_stable(self):
        levels = [0.1, 0.5, 0.9, 0.3]
        a = _omni_wave(levels, 4, "#fff")
        b = _omni_wave(levels, 4, "#fff")
        self.assertEqual(a, b)  # deterministic without anim

    def test_anim_scrolls_frame_to_frame(self):
        levels = [0.1, 0.5, 0.9, 0.3, 0.7]
        f0 = _omni_wave(levels, 5, "#fff", anim=0)
        f1 = _omni_wave(levels, 5, "#fff", anim=1)
        # rotating the window must change the rendered bar (it "runs")
        self.assertNotEqual(f0[1], f1[1])
        # and it's a pure rotation -> same multiset of glyphs
        self.assertEqual(sorted(f0[1].strip()), sorted(f1[1].strip()))

    def test_empty_levels_safe(self):
        out = _omni_wave([], 6, "#fff", anim=3)
        self.assertIn("▁", out[1])


# --- T6: real embedder shape + graceful None fallback -------------------------
class TestT6Embedder(unittest.TestCase):
    def test_broken_model_returns_none(self):
        emb = _PyannoteOmniEmbedder()
        emb._broken = True  # simulate load failure -> never crashes the loop
        self.assertIsNone(emb.embed(b"\x00\x01" * 100))

    def test_embed_uses_model_and_returns_list(self):
        try:
            import numpy as np  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("numpy/torch not installed")

        import numpy as np

        class _FakeEmb:
            def squeeze(self):
                return self
            def tolist(self):
                return [0.1, 0.2, 0.3]

        class _FakeModel:
            def __call__(self, payload):
                # contract check: pyannote expects waveform + sample_rate
                assert "waveform" in payload and "sample_rate" in payload
                return _FakeEmb()

        emb = _PyannoteOmniEmbedder(sample_rate=16000)
        emb._model = _FakeModel()  # inject -> skip real pyannote load
        pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
        out = emb.embed(pcm)
        self.assertEqual(out, [0.1, 0.2, 0.3])


# --- BUG 1: pyannote token kwarg is version-proof (token vs use_auth_token) ----
class TestBug1PyannoteKwarg(unittest.TestCase):
    def _call_with_fake(self, sig_params):
        """Inject a fake PretrainedSpeakerEmbedding with the given signature and
        capture how _pyannote_speaker_model calls it."""
        captured = {}

        def _make_fake():
            import inspect
            params = [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        default=None) for n in sig_params]
            sig = inspect.Signature([inspect.Parameter("embedding",
                                     inspect.Parameter.POSITIONAL_OR_KEYWORD)] + params)

            def fake(embedding, **kw):
                captured.update(kw)
                return "MODEL"
            fake.__signature__ = sig
            return fake

        mod = types.ModuleType("pyannote.audio.pipelines.speaker_verification")
        mod.PretrainedSpeakerEmbedding = _make_fake()
        # ensure parent packages resolve
        for parent in ("pyannote", "pyannote.audio", "pyannote.audio.pipelines"):
            sys.modules.setdefault(parent, types.ModuleType(parent))
        sys.modules["pyannote.audio.pipelines.speaker_verification"] = mod
        try:
            out = _pyannote_speaker_model("hf_xyz")
        finally:
            del sys.modules["pyannote.audio.pipelines.speaker_verification"]
        return out, captured

    def test_picks_token_when_available(self):
        _, kw = self._call_with_fake(["device", "token", "cache_dir"])
        self.assertEqual(kw.get("token"), "hf_xyz")
        self.assertNotIn("use_auth_token", kw)

    def test_falls_back_to_use_auth_token(self):
        _, kw = self._call_with_fake(["device", "use_auth_token"])
        self.assertEqual(kw.get("use_auth_token"), "hf_xyz")
        self.assertNotIn("token", kw)

    def test_no_token_kwarg_when_neither(self):
        _, kw = self._call_with_fake(["device"])
        self.assertNotIn("token", kw)
        self.assertNotIn("use_auth_token", kw)


# --- BUG 2: VAD warmed up OFF the event loop before the mic starts ------------
class _FakeVAD:
    def __init__(self):
        self.calls = []
        self.reset_count = 0
        self.thread_ids = []

    def is_speech(self, pcm):
        import threading
        self.thread_ids.append(threading.get_ident())
        self.calls.append(len(pcm))
        return -1.0

    def reset(self):
        self.reset_count += 1


def _bare_session(vad):
    s = omni.OmniSession.__new__(omni.OmniSession)
    s.vad = vad
    s.sample_rate = 16000
    return s


class TestBug2VadWarmup(unittest.TestCase):
    def test_warmup_calls_vad_offthread_and_resets(self):
        import threading
        vad = _FakeVAD()
        s = _bare_session(vad)
        run(s._warmup_vad())
        self.assertEqual(len(vad.calls), 1)          # probed once
        self.assertEqual(vad.reset_count, 1)          # state reset after
        # the slow load must run OFF the event-loop thread (executor)
        self.assertNotIn(threading.get_ident(), vad.thread_ids)

    def test_warmup_noop_without_vad(self):
        s = _bare_session(None)
        run(s._warmup_vad())  # must not raise

    def test_warmup_swallows_vad_errors(self):
        class _Boom:
            def is_speech(self, pcm):
                raise RuntimeError("model load failed")
        s = _bare_session(_Boom())
        run(s._warmup_vad())  # must not raise -> start() never blocked


if __name__ == "__main__":
    unittest.main(verbosity=2)
