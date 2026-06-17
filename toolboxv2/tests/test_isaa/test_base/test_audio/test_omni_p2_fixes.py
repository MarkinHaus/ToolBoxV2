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


NS = _extract(["_omni_wave", "_PyannoteOmniEmbedder"])
_omni_wave = NS["_omni_wave"]
_PyannoteOmniEmbedder = NS["_PyannoteOmniEmbedder"]


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
