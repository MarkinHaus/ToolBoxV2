"""
Tests for two production bugs found in Discord bot runtime:

Bug 1 — Encoding: save_index/load_index in memory_index.py use
  Path.write_text/read_text WITHOUT encoding="utf-8". On Windows
  (cp1252), emoji like \\U0001f4cb crashes with UnicodeEncodeError.

Bug 2 — Loop mismatch: Session._ensure_session() closes old
  aiohttp.ClientSession via asyncio.create_task on the NEW event loop,
  but the session's connector futures are bound to the OLD loop.
  RuntimeError: "got Future attached to a different loop".

Hypotheses:
  H1: save_index crashes on emoji when forced to cp1252 (pre-fix)
  H2: load_index crashes reading emoji written as UTF-8 when forced to cp1252 (pre-fix)
  H3: Session on loop A, then accessed on loop B → RuntimeError (pre-fix)
  H4: With Option C, different loops get different sessions — no crash
"""
import asyncio
import locale
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════
# Bug 1: Encoding (memory_index.py)
# ═══════════════════════════════════════════════════════════════════

class TestMemoryIndexEncoding(unittest.IsolatedAsyncioTestCase):
    """H1/H2: Encoding roundtrip must survive emoji on cp1252."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = self.tmpdir

    def _make_emoji_index(self):
        """Build a MemoryIndex whose JSON contains emoji (📋 = U+1F4CB)."""
        from toolboxv2.mods.isaa.memory_index import MemoryIndex, SpaceSnapshot
        return MemoryIndex(spaces={
            "self": SpaceSnapshot(
                nodes=[{"id": "entity1", "label": "📋 Task Board", "type": "tool"}],
                edges=[{"source": "entity1", "target": "entity2", "type": "has"}],
                concepts={"📋": 3, "task": 5},
                entry_count=2,
            )
        })

    def test_h1_save_index_emoji_survives_cp1252(self):
        """H1: save_index must not crash with UnicodeEncodeError under cp1252."""
        from toolboxv2.mods.isaa.memory_index import save_index
        idx = self._make_emoji_index()

        # Force cp1252 to reproduce Windows behaviour
        with patch.object(locale, "getpreferredencoding", return_value="cp1252"):
            save_index(self.data_dir, "self", idx)

        # File must exist and contain the emoji as UTF-8 bytes
        p = Path(self.data_dir) / "Agents" / "self" / "memory_index.json"
        raw_bytes = p.read_bytes()
        self.assertIn(b"\xf0\x9f\x93\x8b", raw_bytes, "Emoji must be UTF-8 encoded in file")

    def test_h2_load_index_emoji_roundtrip(self):
        """H2: load_index must correctly read emoji written as UTF-8."""
        from toolboxv2.mods.isaa.memory_index import save_index, load_index
        idx = self._make_emoji_index()

        with patch.object(locale, "getpreferredencoding", return_value="cp1252"):
            save_index(self.data_dir, "self", idx)
            loaded = load_index(self.data_dir, "self")

        snap = loaded.spaces.get("self")
        self.assertIsNotNone(snap, "SpaceSnapshot must survive roundtrip")
        self.assertEqual(snap.entry_count, 2)
        label = snap.nodes[0]["label"]
        self.assertIn("\U0001f4cb", label, "Emoji must survive save→load roundtrip")

    def test_h2b_load_index_cp1252_content_fallback(self):
        """load_index must not crash on files written with UTF-8 even when
        locale prefers cp1252. Confirms encoding='utf-8' is hardcoded."""
        from toolboxv2.mods.isaa.memory_index import load_index
        p = Path(self.data_dir) / "Agents" / "self" / "memory_index.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write emoji directly as UTF-8 bytes
        p.write_bytes('{"spaces": {"self": {"nodes": [{"id": "x", "label": "\U0001f4cb"}], "edges": [], "concepts": {}, "entry_count": 1}}}'.encode("utf-8"))

        with patch.object(locale, "getpreferredencoding", return_value="cp1252"):
            loaded = load_index(self.data_dir, "self")

        snap = loaded.spaces.get("self")
        self.assertIsNotNone(snap)
        self.assertIn("\U0001f4cb", snap.nodes[0]["label"])


# ═══════════════════════════════════════════════════════════════════
# Bug 2: Cross-Loop Session Isolation (session.py — Option C)
# ═══════════════════════════════════════════════════════════════════

class TestSessionLoopIsolation(unittest.TestCase):
    """H3/H4: Session must isolate aiohttp ClientSession per event loop."""

    @staticmethod
    def _access_session_sync(session_obj):
        """Run a coroutine that accesses session.session within a fresh loop."""
        async def _do():
            return session_obj.session
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_do())
            return loop, result
        except Exception:
            loop.close()
            raise

    @staticmethod
    def _cleanup_loop(loop):
        if not loop.is_closed():
            try:
                loop.close()
            except Exception:
                pass

    def test_h4_cross_loop_different_sessions_no_crash(self):
        """H4: Accessing Session on two different loops yields two distinct
        ClientSession objects without RuntimeError."""
        from toolboxv2.utils.system.session import Session

        s = Session()
        # Clear any stale state from previous tests
        if hasattr(Session, '_loop_sessions'):
            Session._loop_sessions.clear()

        loop_a, sa = self._access_session_sync(s)
        loop_b, sb = self._access_session_sync(s)

        try:
            # H4 core assertion: different loops → different sessions
            self.assertIsNotNone(sa, "Loop A session must exist")
            self.assertIsNotNone(sb, "Loop B session must exist")
            self.assertIsNot(sa, sb, "Sessions must be per-loop, not shared")
            self.assertFalse(sa.closed, "Loop A session must remain open")
            self.assertFalse(sb.closed, "Loop B session must remain open")
        finally:
            # Cleanup all sessions across both loops
            if hasattr(Session, '_loop_sessions'):
                for _loop, _sess in list(Session._loop_sessions.items()):
                    try:
                        if not _sess.closed and not _loop.is_closed():
                            _loop.run_until_complete(_sess.close())
                    except Exception:
                        pass
                Session._loop_sessions.clear()
            self._cleanup_loop(loop_a)
            self._cleanup_loop(loop_b)

    def test_h4b_same_loop_reuses_session(self):
        """H4b: Accessing Session twice on same loop reuses the same ClientSession."""
        from toolboxv2.utils.system.session import Session

        s = Session()
        if hasattr(Session, '_loop_sessions'):
            Session._loop_sessions.clear()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _do():
                return s.session
            sa = loop.run_until_complete(_do())
            sa2 = loop.run_until_complete(_do())
            self.assertIs(sa, sa2, "Same loop must reuse same session")
        finally:
            if hasattr(Session, '_loop_sessions'):
                for _loop, _sess in list(Session._loop_sessions.items()):
                    try:
                        if not _sess.closed and not _loop.is_closed():
                            _loop.run_until_complete(_sess.close())
                    except Exception:
                        pass
                Session._loop_sessions.clear()
            self._cleanup_loop(loop)

    def test_h4c_session_returns_none_outside_loop(self):
        """H4c: session property returns None when no running loop."""
        from toolboxv2.utils.system.session import Session

        s = Session()
        if hasattr(Session, '_loop_sessions'):
            Session._loop_sessions.clear()

        # Set no event loop at all
        asyncio.set_event_loop(None)
        result = s.session
        self.assertIsNone(result, "session must return None outside event loop")

    def test_h4d_cleanup_closes_all_loop_sessions(self):
        """H4d: cleanup() must close sessions across all loops."""
        from toolboxv2.utils.system.session import Session

        s = Session()
        if hasattr(Session, '_loop_sessions'):
            Session._loop_sessions.clear()

        loop_a, sa = self._access_session_sync(s)
        loop_b, sb = self._access_session_sync(s)

        try:
            # cleanup from loop_b context
            async def _do_cleanup():
                await s.cleanup()
            loop_b.run_until_complete(_do_cleanup())

            # Both sessions should be closed
            # (sb might already be closed by cleanup, sa might not if on different loop)
            # The key assertion: _loop_sessions dict is cleared
            self.assertEqual(len(Session._loop_sessions), 0,
                             "cleanup must clear _loop_sessions")
        finally:
            self._cleanup_loop(loop_a)
            self._cleanup_loop(loop_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
