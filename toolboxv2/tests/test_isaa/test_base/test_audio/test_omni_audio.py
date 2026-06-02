"""
test_omni_audio.py
==================

Tests for the audio-output fix and the features added this session:

  - Audio buffering: many AUDIO chunks in one turn are concatenated and handed
    to the player as ONE wav on TURN_END (the fix for "cuts off after 3 words").
  - Output sample rate: meta['sample_rate_out'] (e.g. Gemini 24k) is honoured,
    not the 16k input rate (the fix for "sounds grotty").
  - buffer_audio=False forwards each chunk immediately.
  - INTERRUPTED drops buffered audio (barge-in).
  - Enhancer hook runs on flushed wav; failure is swallowed (best-effort).
  - VFS peek resolves the delegated session via JobManager.session_id (robust,
    not label-reconstructed) and never mutates VFS state.

unittest only.
"""
from __future__ import annotations

import asyncio
import io
import json
import unittest
import wave

from toolboxv2.mods.isaa.base.audio_io.omni import (
    OmniEvent,
    OmniEventType,
    OmniSession,
    StubOmniBackend,
    JobManager,
    make_vfs_peek_tools,
)


class RecordingPlayer:
    def __init__(self):
        self.queued = []  # (wav_bytes, meta)
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def queue_audio(self, wav_bytes, metadata):
        self.queued.append((wav_bytes, metadata))

    def last_wav_info(self):
        wav, _ = self.queued[-1]
        with wave.open(io.BytesIO(wav), "rb") as w:
            return w.getframerate(), w.getnframes()


def _pcm(n_samples, fill=b"\x01\x00"):
    return fill * n_samples


# ---------------------------------------------------------------------------
# Audio buffering
# ---------------------------------------------------------------------------

class TestAudioBuffering(unittest.IsolatedAsyncioTestCase):
    async def test_chunks_buffered_and_flushed_as_one_wav_on_turn_end(self):
        player = RecordingPlayer()
        sess = OmniSession(StubOmniBackend(), player=player, output_sample_rate=24000)
        # three chunks, no flush yet
        for _ in range(3):
            await sess.dispatch(OmniEvent.audio_chunk(_pcm(100), sample_rate_out=24000))
        self.assertEqual(len(player.queued), 0)  # nothing played mid-turn
        # turn end -> single concatenated wav
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(len(player.queued), 1)
        sr, frames = player.last_wav_info()
        self.assertEqual(sr, 24000)  # correct output rate, not 16k
        self.assertEqual(frames, 300)  # 3 x 100 samples concatenated
        self.assertEqual(sess.audio_flushes, 1)

    async def test_output_sample_rate_from_meta_overrides_default(self):
        player = RecordingPlayer()
        sess = OmniSession(StubOmniBackend(), player=player, sample_rate=16000)
        await sess.dispatch(OmniEvent.audio_chunk(_pcm(50), sample_rate_out=24000))
        await sess.dispatch(OmniEvent.turn_end())
        sr, _ = player.last_wav_info()
        self.assertEqual(sr, 24000)

    async def test_unbuffered_mode_plays_each_chunk(self):
        player = RecordingPlayer()
        sess = OmniSession(StubOmniBackend(), player=player, buffer_audio=False,
                           output_sample_rate=24000)
        await sess.dispatch(OmniEvent.audio_chunk(_pcm(40), sample_rate_out=24000))
        await sess.dispatch(OmniEvent.audio_chunk(_pcm(40), sample_rate_out=24000))
        self.assertEqual(len(player.queued), 2)  # each chunk played immediately

    async def test_interrupted_drops_buffered_audio(self):
        player = RecordingPlayer()
        sess = OmniSession(StubOmniBackend(), player=player, output_sample_rate=24000)
        await sess.dispatch(OmniEvent.audio_chunk(_pcm(100), sample_rate_out=24000))
        await sess.dispatch(OmniEvent.interrupted())
        await sess.dispatch(OmniEvent.turn_end())
        # buffer was cleared by interrupt; turn_end flushes nothing
        self.assertEqual(len(player.queued), 0)

    async def test_empty_turn_flush_is_noop(self):
        player = RecordingPlayer()
        sess = OmniSession(StubOmniBackend(), player=player)
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(len(player.queued), 0)


# ---------------------------------------------------------------------------
# Enhancement hook
# ---------------------------------------------------------------------------

class _FakeEnhancer:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def enhance(self, wav_bytes):
        self.calls += 1
        if self.fail:
            raise RuntimeError("enhancer down")
        return b"ENHANCED" + wav_bytes


class TestEnhancer(unittest.IsolatedAsyncioTestCase):
    async def test_enhancer_runs_on_flushed_wav(self):
        player = RecordingPlayer()
        enh = _FakeEnhancer()
        sess = OmniSession(StubOmniBackend(), player=player, enhancer=enh,
                           output_sample_rate=24000)
        await sess.dispatch(OmniEvent.audio_chunk(_pcm(20), sample_rate_out=24000))
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(enh.calls, 1)
        self.assertTrue(player.queued[0][0].startswith(b"ENHANCED"))

    async def test_enhancer_failure_falls_back_to_raw(self):
        player = RecordingPlayer()
        enh = _FakeEnhancer(fail=True)
        sess = OmniSession(StubOmniBackend(), player=player, enhancer=enh,
                           output_sample_rate=24000)
        await sess.dispatch(OmniEvent.audio_chunk(_pcm(20), sample_rate_out=24000))
        await sess.dispatch(OmniEvent.turn_end())
        # raw wav still played despite enhancer raising
        self.assertEqual(len(player.queued), 1)
        self.assertFalse(player.queued[0][0].startswith(b"ENHANCED"))


# ---------------------------------------------------------------------------
# VFS peek (session_id-based) — fakes
# ---------------------------------------------------------------------------

class FakeFile:
    def __init__(self, content):
        self._content = content
        self.state = "closed"
        self.view_start = 0
        self.view_end = -1


class FakeVFS:
    def __init__(self, files, tree="tree-x"):
        self.files = {p: FakeFile(c) for p, c in files.items()}
        self._tree = tree

    def _normalize_path(self, p):
        return p if p.startswith("/") else "/" + p

    def _is_file(self, p):
        return p in self.files

    def read(self, path, max_chars=25000):
        f = self.files.get(path)
        return {"success": True, "content": f._content} if f else {"success": False, "error": "nf"}

    def _build_tree_string(self, path="/", max_depth=3):
        return self._tree


class FakeSession:
    def __init__(self, vfs):
        self.vfs = vfs


class FakeSessionManager:
    def __init__(self, sessions):
        self._sessions = sessions

    def get(self, sid):
        return self._sessions.get(sid)


class FakeAgent:
    def __init__(self, sm):
        self.session_manager = sm


class TestVfsPeekSessionId(unittest.IsolatedAsyncioTestCase):
    async def _setup(self, content):
        jobs = JobManager()

        async def _noop():
            return "done"

        # spawn an agent job WITH an explicit session_id (as the delegate tool does)
        sid = "deleg-scan repo"
        job_id = jobs.spawn("agent", "scan repo files", _noop, session_id=sid)
        await jobs.join(job_id, timeout=1.0)

        vfs = FakeVFS({"/src/models.py": content})
        agent = FakeAgent(FakeSessionManager({sid: FakeSession(vfs)}))
        specs = make_vfs_peek_tools(agent, jobs)
        tools = {s["name"]: s["tool_func"] for s in specs}
        return tools, vfs, job_id

    async def test_peek_resolves_via_session_id_and_slices(self):
        content = "\n".join(f"line{i}" for i in range(1, 21))
        tools, vfs, job_id = await self._setup(content)
        out = json.loads(await tools["vfs_peek"](job_id, "/src/models.py", line_start=2, line_end=4))
        self.assertTrue(out["success"])
        self.assertEqual(out["content"], "line2\nline3\nline4")

    async def test_peek_read_only_invariant(self):
        tools, vfs, job_id = await self._setup("a\nb\nc")
        f = vfs.files["/src/models.py"]
        await tools["vfs_peek"](job_id, "/src/models.py", scroll_to="b")
        self.assertEqual(f.state, "closed")
        self.assertEqual(f.view_start, 0)
        self.assertEqual(f.view_end, -1)

    async def test_tree_peek(self):
        tools, vfs, job_id = await self._setup("x")
        out = json.loads(await tools["vfs_tree_peek"](job_id, "/"))
        self.assertTrue(out["success"])
        self.assertEqual(out["tree"], "tree-x")

    async def test_unknown_job(self):
        tools, vfs, job_id = await self._setup("x")
        out = json.loads(await tools["vfs_peek"]("deadbeef", "/src/models.py"))
        self.assertFalse(out["success"])
        self.assertIn("unknown job", out["error"])


class _ScriptedVAD:
    """Returns scripted speech probabilities per frame for deterministic tests."""

    def __init__(self, probs):
        self._probs = list(probs)
        self.i = 0

    def is_speech(self, pcm):
        p = self._probs[self.i] if self.i < len(self._probs) else 0.0
        self.i += 1
        return p


class TestVadGate(unittest.IsolatedAsyncioTestCase):
    def _sess(self, vad, **kw):
        return OmniSession(StubOmniBackend(), vad=vad, vad_threshold=0.5,
                           vad_hangover_frames=2, **kw)

    async def test_no_vad_sends_everything(self):
        sess = OmniSession(StubOmniBackend())
        self.assertTrue(sess._should_send_frame(b"\x00\x00"))

    async def test_silence_is_gated(self):
        sess = self._sess(_ScriptedVAD([0.0, 0.1, 0.2]))
        self.assertFalse(sess._should_send_frame(b"x"))
        self.assertFalse(sess._should_send_frame(b"x"))

    async def test_speech_passes(self):
        sess = self._sess(_ScriptedVAD([0.9]))
        self.assertTrue(sess._should_send_frame(b"x"))

    async def test_hangover_keeps_sending_briefly_after_speech(self):
        # speech (0.9) then silence: hangover=2 -> next 2 silent frames still sent
        sess = self._sess(_ScriptedVAD([0.9, 0.0, 0.0, 0.0]))
        self.assertTrue(sess._should_send_frame(b"x"))  # speech
        self.assertTrue(sess._should_send_frame(b"x"))  # hangover 1
        self.assertTrue(sess._should_send_frame(b"x"))  # hangover 2
        self.assertFalse(sess._should_send_frame(b"x"))  # gated

    async def test_vad_error_falls_open(self):
        class Boom:
            def is_speech(self, pcm):
                raise RuntimeError("nope")

        sess = self._sess(Boom())
        self.assertTrue(sess._should_send_frame(b"x"))

    async def test_pump_gates_silence_frames(self):
        from test_omni import FakeRecorder  # reuse
        backend = StubOmniBackend(close_on_turn_end=True)
        backend.queue(OmniEvent.turn_end())
        rec = FakeRecorder([b"f1", b"f2", b"f3", b"f4"])
        # frame1 speech, rest silence, hangover 1 -> f1 + f2 sent, f3/f4 gated
        vad = _ScriptedVAD([0.9, 0.0, 0.0, 0.0])
        sess = OmniSession(backend, recorder=rec, vad=vad, vad_threshold=0.5,
                           vad_hangover_frames=1)
        await sess.start()
        await sess.wait(timeout=1.0)
        await asyncio.sleep(0.02)
        await sess.stop()
        self.assertEqual(backend.received_audio, [b"f1", b"f2"])
        self.assertEqual(sess.frames_sent, 2)
        self.assertEqual(sess.frames_gated, 2)


class TestReadableJobIds(unittest.IsolatedAsyncioTestCase):
    async def test_id_is_slug_not_uuid(self):
        jm = JobManager()

        async def w():
            return "ok"

        jid = jm.spawn("agent", "Führe eine Beispielanalyse durch", w)
        self.assertNotRegex(jid, r"^[0-9a-f]{8}$")  # not a raw uuid
        self.assertTrue(jid.startswith("fuhre-eine-") or jid.startswith("f-hre")
                        or "analyse" in jid or jid.endswith("-1"))
        await jm.join(jid, timeout=1.0)

    async def test_unique_suffix_on_collision(self):
        jm = JobManager()

        async def w():
            return "ok"

        a = jm.spawn("agent", "scan repo", w)
        b = jm.spawn("agent", "scan repo", w)
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith("-1"))
        self.assertTrue(b.endswith("-2"))
        await jm.join(a, timeout=1.0)
        await jm.join(b, timeout=1.0)

    async def test_empty_label_uses_kind(self):
        jm = JobManager()

        async def w():
            return "ok"

        jid = jm.spawn("tool", "", w)
        self.assertTrue(jid.startswith("tool-"))
        await jm.join(jid, timeout=1.0)


class TestJobDoneAnnounce(unittest.IsolatedAsyncioTestCase):
    async def test_send_text_used_when_no_handler(self):
        backend = StubOmniBackend()
        sess = OmniSession(backend)
        await sess._announce_job("job-1", {"label": "analysis", "status": "completed",
                                           "job_id": "job-1"})
        self.assertEqual(len(backend.sent_texts), 1)
        self.assertIn("analysis", backend.sent_texts[0])

    async def test_custom_handler_wins(self):
        backend = StubOmniBackend()
        seen = []
        sess = OmniSession(backend, on_job_done=lambda st: seen.append(st))
        await sess._announce_job("job-1", {"label": "x", "status": "completed",
                                           "job_id": "job-1"})
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["job_id"], "job-1")
        self.assertEqual(backend.sent_texts, [])  # handler took over

    async def test_notify_loop_announces_finished_agent_once(self):
        backend = StubOmniBackend()
        jm = JobManager()
        sess = OmniSession(backend, jobs=jm)

        async def quick():
            return "the answer"

        jid = jm.spawn("agent", "do a thing", quick, session_id="deleg-do a thing")
        await jm.join(jid, timeout=1.0)
        # run one notify pass manually (fast)
        sess._running = True
        for st in jm.live_state():
            if st["kind"] == "agent" and st["status"] != "running" \
                and st["job_id"] not in sess._announced_jobs:
                sess._announced_jobs.add(st["job_id"])
                await sess._announce_job(st["job_id"], st)
        self.assertEqual(len(backend.sent_texts), 1)
        self.assertIn("the answer", backend.sent_texts[0])
        # second pass: already announced -> no duplicate
        for st in jm.live_state():
            if st["kind"] == "agent" and st["status"] != "running" \
                and st["job_id"] not in sess._announced_jobs:
                sess._announced_jobs.add(st["job_id"])
                await sess._announce_job(st["job_id"], st)
        self.assertEqual(len(backend.sent_texts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
