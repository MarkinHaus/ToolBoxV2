"""
test_omni.py
============

Hard tests for the Omni core. Drives the *logic and control flow*, not syntax:
  - JobManager: spawn/running->completed/failed, result retrieval, live-state
    across multiple jobs, eviction cap, join, cancel.
  - OmniSession loop: audio pump -> backend, AUDIO events -> player,
    TURN_END counting, ERROR -> fallback switch, finite-script integration.
  - Tool bridge: blocking roundtrip, nonblocking (background) ack+job,
    unknown tool, no-provider, Omni->ToolManager full roundtrip with
    tool-result re-injection driving follow-up audio.
  - Agent tools: delegate returns job_id fast, agent_result before/after,
    agent_status reflects multiple concurrent delegations.
  - pcm16_to_wav: valid container, correct frame/rate/channel math.
  - OmniBackend ABC: cannot instantiate incomplete subclass.

No network. The only mocked thing is the model/API (StubOmniBackend + fakes).
unittest only (no pytest).
"""
from __future__ import annotations

import asyncio
import json
import unittest
import wave
import io

from toolboxv2.mods.isaa.base.audio_io.omni import (
    JobManager,
    OmniBackend,
    OmniEvent,
    OmniEventType,
    OmniSession,
    StubOmniBackend,
    VoiceModeConfig,
    make_agent_tools,
    pcm16_to_wav,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePlayer:
    """AudioPlayer-like sink that records queued WAV bytes."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.queued: list[tuple[bytes, dict]] = []

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def queue_audio(self, wav_bytes: bytes, metadata: dict):
        self.queued.append((wav_bytes, metadata))


class FakeRecorder:
    """AudioRecorder-like source yielding a fixed list of PCM frames."""

    def __init__(self, frames: list[bytes]):
        self._frames = frames
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def frames(self):
        for f in self._frames:
            yield f


class FakeToolManager:
    """ToolManager-like: name->callable registry, async execute, get_function."""

    def __init__(self):
        self._funcs: dict = {}
        self.calls: list[tuple[str, dict]] = []

    def register(self, name, func):
        self._funcs[name] = func

    def get_function(self, name):
        return self._funcs.get(name)

    async def execute(self, function_name, **kwargs):
        self.calls.append((function_name, kwargs))
        func = self._funcs.get(function_name)
        if func is None:
            raise KeyError(function_name)
        res = func(**kwargs)
        if asyncio.iscoroutine(res):
            res = await res
        return res


class FakeAgent:
    """FlowAgent-like with an async a_run that we can make slow/fail."""

    def __init__(self, reply="done", delay=0.0, fail=False):
        self.reply = reply
        self.delay = delay
        self.fail = fail
        self.runs: list[str] = []

    async def a_run(self, query, session_id="default", **kwargs):
        self.runs.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return f"{self.reply}:{query}"


class FakeFallback:
    def __init__(self):
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        pass


# ---------------------------------------------------------------------------
# pcm16_to_wav
# ---------------------------------------------------------------------------

class TestPcmToWav(unittest.TestCase):
    def test_wav_is_parseable_and_correct(self):
        # 1600 samples = 100ms @ 16k mono int16 -> 3200 bytes PCM
        pcm = b"\x01\x00" * 1600
        wav = pcm16_to_wav(pcm, sample_rate=16000, channels=1)
        with wave.open(io.BytesIO(wav), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 16000)
            self.assertEqual(w.getnframes(), 1600)
            self.assertEqual(w.readframes(1600), pcm)

    def test_empty_pcm(self):
        wav = pcm16_to_wav(b"")
        with wave.open(io.BytesIO(wav), "rb") as w:
            self.assertEqual(w.getnframes(), 0)


# ---------------------------------------------------------------------------
# OmniBackend ABC
# ---------------------------------------------------------------------------

class TestBackendABC(unittest.TestCase):
    def test_incomplete_subclass_cannot_instantiate(self):
        class Bad(OmniBackend):
            async def start(self, tools=None):
                pass
            # missing send_audio/send_tool_result/events/stop

        with self.assertRaises(TypeError):
            Bad()

    def test_stub_satisfies_contract(self):
        b = StubOmniBackend()
        self.assertIsInstance(b, OmniBackend)
        self.assertEqual(b.backend_name, "StubOmniBackend")


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

class TestJobManager(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_runs_to_completion(self):
        jm = JobManager()

        async def work():
            return "ok"

        jid = jm.spawn("tool", "work", work)
        self.assertEqual(jm.status(jid), "running")
        self.assertIsNone(jm.result(jid))  # not done yet
        res = await jm.join(jid, timeout=1.0)
        self.assertEqual(res, "ok")
        self.assertEqual(jm.status(jid), "completed")
        self.assertEqual(jm.result(jid), "ok")

    async def test_failed_job_is_captured_not_raised(self):
        jm = JobManager()

        async def boom():
            raise ValueError("nope")

        jid = jm.spawn("tool", "boom", boom)
        await jm.join(jid, timeout=1.0)
        self.assertEqual(jm.status(jid), "failed")
        self.assertIn("Error:", jm.result(jid))
        self.assertIn("nope", jm.result(jid))

    async def test_non_string_result_is_stringified(self):
        jm = JobManager()

        async def num():
            return 42

        jid = jm.spawn("tool", "num", num)
        self.assertEqual(await jm.join(jid, timeout=1.0), "42")

    async def test_live_state_across_multiple_jobs(self):
        jm = JobManager()

        async def fast():
            return "f"

        async def slow():
            await asyncio.sleep(0.05)
            return "s"

        j1 = jm.spawn("agent", "q-fast", fast)
        j2 = jm.spawn("agent", "q-slow", slow)
        await jm.join(j1, timeout=1.0)

        state = jm.live_state()
        self.assertEqual(len(state), 2)
        by_id = {s["job_id"]: s for s in state}
        self.assertEqual(by_id[j1]["status"], "completed")
        self.assertTrue(by_id[j1]["has_result"])
        self.assertEqual(by_id[j2]["status"], "running")
        self.assertFalse(by_id[j2]["has_result"])
        self.assertEqual(by_id[j1]["kind"], "agent")
        self.assertEqual(by_id[j1]["label"], "q-fast")

        await jm.join(j2, timeout=1.0)
        self.assertEqual(jm.status(j2), "completed")

    async def test_result_unknown_job(self):
        jm = JobManager()
        self.assertIsNone(jm.result("deadbeef"))
        self.assertIsNone(jm.status("deadbeef"))

    async def test_eviction_caps_finished_jobs(self):
        jm = JobManager(max_finished=3)

        async def work():
            return "x"

        ids = [jm.spawn("tool", f"w{i}", work) for i in range(6)]
        for jid in ids:
            await jm.join(jid, timeout=1.0)
        # only the cap survives among finished jobs
        finished = [s for s in jm.live_state() if s["status"] != "running"]
        self.assertLessEqual(len(finished), 3)

    async def test_cancel_running_job(self):
        jm = JobManager()

        async def forever():
            await asyncio.sleep(10)

        jid = jm.spawn("tool", "forever", forever)
        ok = await jm.cancel(jid)
        self.assertTrue(ok)
        self.assertEqual(jm.status(jid), "failed")


# ---------------------------------------------------------------------------
# Agent delegation tools
# ---------------------------------------------------------------------------

class TestAgentTools(unittest.IsolatedAsyncioTestCase):
    def _tools(self, jm, agents):
        specs = make_agent_tools(jm, lambda name: agents.get(name))
        return {s["name"]: s["tool_func"] for s in specs}

    async def test_delegate_returns_job_id_immediately_then_result(self):
        jm = JobManager()
        agent = FakeAgent(reply="R", delay=0.02)
        tools = self._tools(jm, {"default": agent})

        job_id = await tools["delegate"]("hello world")
        self.assertIsInstance(job_id, str)
        self.assertEqual(jm.status(job_id), "running")  # nonblocking

        # before completion
        self.assertEqual(await tools["agent_result"](job_id), "<running>")
        # after completion
        await jm.join(job_id, timeout=1.0)
        self.assertEqual(await tools["agent_result"](job_id), "R:hello world")
        self.assertEqual(agent.runs, ["hello world"])

    async def test_delegate_unknown_agent(self):
        jm = JobManager()
        tools = self._tools(jm, {})
        out = await tools["delegate"]("x", agent="ghost")
        self.assertIn("Error", out)

    async def test_agent_status_reflects_multiple(self):
        jm = JobManager()
        a1 = FakeAgent(reply="A", delay=0.0)
        a2 = FakeAgent(reply="B", delay=0.05)
        tools = self._tools(jm, {"default": a1, "slow": a2})

        j1 = await tools["delegate"]("q1")
        j2 = await tools["delegate"]("q2", agent="slow")
        await jm.join(j1, timeout=1.0)

        status = json.loads(await tools["agent_status"]())
        self.assertEqual(len(status), 2)
        statuses = {s["job_id"]: s["status"] for s in status}
        self.assertEqual(statuses[j1], "completed")
        self.assertEqual(statuses[j2], "running")
        await jm.join(j2, timeout=1.0)

    async def test_delegate_failed_agent_surfaces_error(self):
        jm = JobManager()
        agent = FakeAgent(fail=True)
        tools = self._tools(jm, {"default": agent})
        jid = await tools["delegate"]("boom")
        await jm.join(jid, timeout=1.0)
        self.assertIn("Error:", await tools["agent_result"](jid))

    async def test_agent_result_unknown_job(self):
        jm = JobManager()
        tools = self._tools(jm, {})
        self.assertIn("unknown job", await tools["agent_result"]("nope"))


# ---------------------------------------------------------------------------
# OmniSession — tool bridge (single-event dispatch)
# ---------------------------------------------------------------------------

class TestToolBridge(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_tool_roundtrip(self):
        tm = FakeToolManager()
        tm.register("add", lambda a, b: a + b)
        backend = StubOmniBackend()
        sess = OmniSession(backend, tools=tm)

        result = await sess.handle_tool_call({"id": "c1", "name": "add", "arguments": {"a": 2, "b": 3}})
        self.assertEqual(result, "5")
        # tool was actually executed with the right args
        self.assertEqual(tm.calls, [("add", {"a": 2, "b": 3})])
        # result delivered back to the backend
        self.assertEqual(backend.tool_results, [("c1", "5")])

    async def test_blocking_tool_dict_result_is_json(self):
        tm = FakeToolManager()
        tm.register("info", lambda: {"k": "v"})
        backend = StubOmniBackend()
        sess = OmniSession(backend, tools=tm)
        out = await sess.handle_tool_call({"id": "c", "name": "info", "arguments": {}})
        self.assertEqual(json.loads(out), {"k": "v"})

    async def test_tool_exception_becomes_error_string_not_crash(self):
        tm = FakeToolManager()
        def bad():
            raise RuntimeError("kaboom")
        tm.register("bad", bad)
        backend = StubOmniBackend()
        sess = OmniSession(backend, tools=tm)
        out = await sess.handle_tool_call({"id": "c", "name": "bad", "arguments": {}})
        self.assertIn("Error:", out)
        self.assertIn("kaboom", out)
        self.assertEqual(backend.tool_results[0][0], "c")

    async def test_unknown_tool(self):
        tm = FakeToolManager()
        backend = StubOmniBackend()
        sess = OmniSession(backend, tools=tm)
        out = await sess.handle_tool_call({"id": "c", "name": "ghost", "arguments": {}})
        self.assertIn("unknown tool", out)
        self.assertEqual(backend.tool_results[0], ("c", out))

    async def test_no_tool_provider(self):
        backend = StubOmniBackend()
        sess = OmniSession(backend, tools=None)
        out = await sess.handle_tool_call({"id": "c", "name": "x", "arguments": {}})
        self.assertIn("no tool provider", out)

    async def test_background_tool_spawns_and_acks(self):
        tm = FakeToolManager()
        done = asyncio.Event()

        async def long_task(n):
            await asyncio.sleep(0.02)
            done.set()
            return f"processed-{n}"

        tm.register("crunch", long_task)
        backend = StubOmniBackend()
        jm = JobManager()
        sess = OmniSession(backend, tools=tm, jobs=jm, background_tools={"crunch"})

        out = await sess.handle_tool_call({"id": "c", "name": "crunch", "arguments": {"n": 7}})
        ack = json.loads(out)
        self.assertEqual(ack["status"], "started")
        job_id = ack["job_id"]
        # immediate ack: tool not finished yet
        self.assertEqual(jm.status(job_id), "running")
        # backend got the ack, not the final result
        self.assertEqual(backend.tool_results[0][0], "c")
        # later: result retrievable
        await jm.join(job_id, timeout=1.0)
        self.assertTrue(done.is_set())
        self.assertEqual(jm.result(job_id), "processed-7")

    async def test_missing_arguments_key_defaults_empty(self):
        tm = FakeToolManager()
        tm.register("ping", lambda: "pong")
        backend = StubOmniBackend()
        sess = OmniSession(backend, tools=tm)
        out = await sess.handle_tool_call({"id": "c", "name": "ping"})
        self.assertEqual(out, "pong")


# ---------------------------------------------------------------------------
# OmniSession — event dispatch
# ---------------------------------------------------------------------------

class TestEventDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_audio_event_wraps_pcm_and_queues_to_player(self):
        player = FakePlayer()
        backend = StubOmniBackend()
        sess = OmniSession(backend, player=player)
        pcm = b"\x10\x00" * 800
        await sess.dispatch(OmniEvent.audio_chunk(pcm, voice="x"))
        self.assertEqual(len(player.queued), 1)
        wav, meta = player.queued[0]
        self.assertEqual(meta["voice"], "x")
        with wave.open(io.BytesIO(wav), "rb") as w:
            self.assertEqual(w.getnframes(), 800)
        self.assertEqual(sess.audio_chunks_out, 1)

    async def test_text_event_invokes_callback(self):
        seen = []
        backend = StubOmniBackend()
        sess = OmniSession(backend, on_text=lambda t: seen.append(t))
        await sess.dispatch(OmniEvent.text_chunk("hi"))
        self.assertEqual(seen, ["hi"])

    async def test_text_event_async_callback(self):
        seen = []
        async def cb(t):
            seen.append(t)
        backend = StubOmniBackend()
        sess = OmniSession(backend, on_text=cb)
        await sess.dispatch(OmniEvent.text_chunk("yo"))
        self.assertEqual(seen, ["yo"])

    async def test_turn_end_counts(self):
        sess = OmniSession(StubOmniBackend())
        await sess.dispatch(OmniEvent.turn_end())
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(sess.turns, 2)

    async def test_error_event_triggers_fallback(self):
        fb = FakeFallback()
        sess = OmniSession(StubOmniBackend(), fallback=fb)
        await sess.dispatch(OmniEvent.error("model gone"))
        self.assertTrue(fb.started)
        self.assertTrue(sess.fell_back)

    async def test_fallback_only_once(self):
        fb = FakeFallback()
        sess = OmniSession(StubOmniBackend(), fallback=fb)
        await sess.dispatch(OmniEvent.error("e1"))
        fb.started = False  # reset to detect a second start
        await sess.dispatch(OmniEvent.error("e2"))
        self.assertFalse(fb.started)


# ---------------------------------------------------------------------------
# OmniSession — full loop integration (finite scripted backend)
# ---------------------------------------------------------------------------

class TestSessionIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_audio_pump_forwards_frames_to_backend(self):
        backend = StubOmniBackend(close_on_turn_end=True)
        backend.queue(OmniEvent.turn_end())  # makes events() return promptly
        rec = FakeRecorder([b"f1", b"f2", b"f3"])
        sess = OmniSession(backend, recorder=rec, player=FakePlayer())
        await sess.start(tool_specs=[{"name": "noop"}])
        await sess.wait(timeout=1.0)
        await asyncio.sleep(0.01)  # let pump drain
        await sess.stop()
        self.assertEqual(backend.received_audio, [b"f1", b"f2", b"f3"])
        self.assertEqual(backend.advertised_tools, [{"name": "noop"}])
        self.assertTrue(rec.started)

    async def test_full_roundtrip_tool_call_then_followup_audio(self):
        """audio in -> model TOOL_CALL -> ToolManager.execute -> result fed back
        -> model emits follow-up audio -> player. Drives the whole loop with no
        sleeps; ordering is guaranteed by the await-before-next-event contract.
        """
        tm = FakeToolManager()
        tm.register("weather", lambda city: f"sunny in {city}")
        player = FakePlayer()
        backend = StubOmniBackend(close_on_turn_end=True)

        # When the tool result comes back, the model "responds" with audio + ends turn.
        def on_result(call_id, result):
            backend.queue(
                OmniEvent.text_chunk(f"got: {result}"),
                OmniEvent.audio_chunk(b"\x00\x01" * 400),
                OmniEvent.turn_end(),
            )

        backend.on_tool_result = on_result
        backend.queue(
            OmniEvent.audio_chunk(b"\x02\x02" * 100),               # greeting audio
            OmniEvent.call("weather", {"city": "Berlin"}, call_id="t1"),
        )

        texts = []
        sess = OmniSession(backend, player=player, tools=tm, on_text=lambda t: texts.append(t))
        await sess.start()
        await sess.wait(timeout=1.0)
        await sess.stop()

        # tool executed with correct args
        self.assertEqual(tm.calls, [("weather", {"city": "Berlin"})])
        # result was delivered back into the model turn
        self.assertEqual(backend.tool_results, [("t1", "sunny in Berlin")])
        # two audio chunks reached the player (greeting + follow-up)
        self.assertEqual(len(player.queued), 2)
        # text callback saw the follow-up
        self.assertEqual(texts, ["got: sunny in Berlin"])
        # one full turn completed
        self.assertEqual(sess.turns, 1)
        self.assertEqual(sess.tool_calls_handled, 1)

    async def test_start_failure_falls_back(self):
        class FailingBackend(StubOmniBackend):
            async def start(self, tools=None):
                raise RuntimeError("no gpu")

        fb = FakeFallback()
        sess = OmniSession(FailingBackend(), fallback=fb)
        await sess.start()
        self.assertTrue(sess.fell_back)
        self.assertTrue(fb.started)

    async def test_background_tool_via_full_loop(self):
        tm = FakeToolManager()
        finished = asyncio.Event()

        async def index(path):
            finished.set()
            return f"indexed {path}"

        tm.register("index", index)
        jm = JobManager()
        backend = StubOmniBackend(close_on_turn_end=True)

        captured = {}
        def on_result(call_id, result):
            captured["ack"] = json.loads(result)
            backend.queue(OmniEvent.turn_end())

        backend.on_tool_result = on_result
        backend.queue(OmniEvent.call("index", {"path": "/x"}, call_id="bg1"))

        sess = OmniSession(backend, tools=tm, jobs=jm, background_tools={"index"})
        await sess.start()
        await sess.wait(timeout=1.0)
        await sess.stop()

        self.assertEqual(captured["ack"]["status"], "started")
        job_id = captured["ack"]["job_id"]
        await jm.join(job_id, timeout=1.0)
        self.assertTrue(finished.is_set())
        self.assertEqual(jm.result(job_id), "indexed /x")


# ---------------------------------------------------------------------------
# VoiceModeConfig
# ---------------------------------------------------------------------------

class TestVoiceModeConfig(unittest.TestCase):
    def test_stub_mode(self):
        self.assertIsInstance(VoiceModeConfig(mode="stub").build_backend(), StubOmniBackend)

    def test_pipeline_mode_returns_none(self):
        self.assertIsNone(VoiceModeConfig(mode="pipeline").build_backend())

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            VoiceModeConfig(mode="warp").build_backend()

    def test_omni_local_builds_without_loading_model(self):
        # constructing the backend must NOT import torch / load anything
        b = VoiceModeConfig(mode="omni_local").build_backend()
        self.assertEqual(b.backend_name, "LocalOmniBackend")

    def test_omni_cloud_builds_without_connecting(self):
        b = VoiceModeConfig(mode="omni_cloud").build_backend()
        self.assertEqual(b.backend_name, "CloudOmniBackend")


if __name__ == "__main__":
    unittest.main(verbosity=2)
