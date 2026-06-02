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
    BlobStateStore,
    JobManager,
    OmniBackend,
    OmniEvent,
    OmniEventType,
    OmniSession,
    OmniState,
    StubOmniBackend,
    VoiceModeConfig,
    WorldModel,
    _summarize_omni,
    make_agent_tools,
    make_session_tools,
    make_world_model_tools,
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

# ===========================================================================
# Fakes for the persistent-state surface
# ===========================================================================

class FakeLLMAgent:
    """FlowAgent-like with a_run_llm_completion (used for summaries). Records
    every call; on_call lets a test mutate state DURING the await."""

    def __init__(self, reply="SUMMARY", on_call=None):
        self.reply = reply
        self.on_call = on_call
        self.calls: list[dict] = []

    async def a_run_llm_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.on_call is not None:
            self.on_call()
        return self.reply


def make_fake_blob():
    """Return (BlobFileCls, backing_dict, opens). The class mimics the real
    BlobFile semantics this code relies on: context manager, 'r' loads / 'w'
    saves, read_json on empty/missing -> {}. backing simulates the blob store
    keyed by path so save->load round-trips across instances."""
    backing: dict[str, bytes] = {}
    opens: list[dict] = []

    class FakeBlobFile:
        def __init__(self, filename, mode="r", key=None, **kw):
            opens.append({"path": filename, "mode": mode, "key": key})
            self.path = filename
            self.mode = mode
            self.key = key
            self.buf = b""

        def __enter__(self):
            if "r" in self.mode:
                data = backing.get(self.path)
                if data:
                    self.buf = data
            return self

        def __exit__(self, *a):
            if "w" in self.mode:
                backing[self.path] = self.buf
            return False

        def read_json(self):
            if not self.buf:
                return {}
            return json.loads(self.buf.decode())

        def write_json(self, data):
            self.buf += json.dumps(data).encode()

    return FakeBlobFile, backing, opens


# ===========================================================================
# WorldModel / OmniState
# ===========================================================================

class TestWorldModelState(unittest.TestCase):
    def test_world_model_roundtrip_and_render(self):
        wm = WorldModel(user="u", agent_role="r", routines=["a", "b"])
        self.assertEqual(WorldModel.from_dict(wm.to_dict()), wm)
        rendered = wm.render()
        self.assertIn("User: u", rendered)
        self.assertIn("Your role: r", rendered)
        self.assertIn("a, b", rendered)

    def test_omni_state_roundtrip(self):
        st = OmniState(
            world_model=WorldModel(user="u"),
            full_summary="s",
            session_summaries=["x"],
            active_history=[{"role": "user", "text": "h"}],
        )
        self.assertEqual(OmniState.from_dict(st.to_dict()), st)

    def test_from_none_and_empty_are_defaults(self):
        self.assertEqual(OmniState.from_dict(None), OmniState())
        self.assertEqual(WorldModel.from_dict({}), WorldModel())

    def test_empty_world_model_renders_empty(self):
        self.assertEqual(WorldModel().render(), "")


# ===========================================================================
# BlobStateStore
# ===========================================================================

class TestBlobStateStore(unittest.TestCase):
    def test_missing_blob_yields_default_state(self):
        Blob, _backing, _opens = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        self.assertIsInstance(store.state, OmniState)
        self.assertEqual(store.state, OmniState())

    def test_save_then_reload_roundtrips(self):
        Blob, _backing, _opens = make_fake_blob()
        path = "isaa/omni/omni_state.json"
        store = BlobStateStore(path, Blob)
        store.state.full_summary = "hello"
        store.state.world_model.user = "Markin"
        store.save()

        store2 = BlobStateStore(path, Blob)
        self.assertEqual(store2.state.full_summary, "hello")
        self.assertEqual(store2.state.world_model.user, "Markin")

    def test_key_is_passed_through_to_blobfile(self):
        Blob, _backing, opens = make_fake_blob()
        BlobStateStore("isaa/omni/omni_state.json", Blob, key=b"DEVKEY")
        self.assertTrue(any(o["key"] == b"DEVKEY" for o in opens))


# ===========================================================================
# world_model_edit tool
# ===========================================================================

class TestWorldModelEditTool(unittest.IsolatedAsyncioTestCase):
    def _edit(self):
        Blob, _backing, _opens = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        edit = make_world_model_tools(store)[0]["tool_func"]
        return store, edit

    async def test_set_user_and_persists(self):
        store, edit = self._edit()
        out = await edit("user", "set", "Markin, builds ISAA")
        self.assertEqual(json.loads(out)["ok"], True)
        self.assertEqual(store.state.world_model.user, "Markin, builds ISAA")
        # persisted: a fresh store sees it
        store2 = BlobStateStore("isaa/omni/omni_state.json", store._BlobFile)
        self.assertEqual(store2.state.world_model.user, "Markin, builds ISAA")

    async def test_routines_append_dedup_and_remove(self):
        store, edit = self._edit()
        await edit("routines", "append", "standup")
        await edit("routines", "append", "standup")  # dedup
        await edit("routines", "append", "deploy")
        self.assertEqual(store.state.world_model.routines, ["standup", "deploy"])
        await edit("routines", "remove", "standup")
        self.assertEqual(store.state.world_model.routines, ["deploy"])

    async def test_invalid_field_and_op(self):
        _store, edit = self._edit()
        self.assertIn("unknown field", await edit("nope", "set", "x"))
        self.assertIn("Error", await edit("user", "append", "x"))
        self.assertIn("Error", await edit("routines", "set", "x"))


# ===========================================================================
# Transcript aggregation (TEXT -> active_history)
# ===========================================================================

class TestTranscriptAggregation(unittest.IsolatedAsyncioTestCase):
    def _sess(self):
        Blob, _backing, _opens = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        return store, OmniSession(StubOmniBackend(), state_store=store)

    async def test_consecutive_same_source_merge_and_role_mapping(self):
        store, sess = self._sess()
        await sess.dispatch(OmniEvent.text_chunk("hi ", source="user"))
        await sess.dispatch(OmniEvent.text_chunk("there", source="user"))
        await sess.dispatch(OmniEvent.text_chunk("hello", source="model"))
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(
            store.state.active_history,
            [{"role": "user", "text": "hi there"},
             {"role": "assistant", "text": "hello"}],
        )

    async def test_default_source_is_assistant(self):
        store, sess = self._sess()
        await sess.dispatch(OmniEvent.text_chunk("untagged"))
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(store.state.active_history,
                         [{"role": "assistant", "text": "untagged"}])

    async def test_no_store_does_not_buffer(self):
        sess = OmniSession(StubOmniBackend())  # no state_store
        await sess.dispatch(OmniEvent.text_chunk("x", source="user"))
        await sess.dispatch(OmniEvent.turn_end())
        self.assertEqual(sess._turn_buf, [])


# ===========================================================================
# _summarize_omni
# ===========================================================================

class TestSummarize(unittest.IsolatedAsyncioTestCase):
    async def test_uses_fast_completion_without_context_or_stream(self):
        agent = FakeLLMAgent(reply="OUT")
        out = await _summarize_omni(agent, "transcript text", "PRIOR")
        self.assertEqual(out, "OUT")
        kw = agent.calls[-1]["kwargs"]
        self.assertIs(kw["with_context"], False)
        self.assertIs(kw["stream"], False)
        self.assertEqual(kw["model_preference"], "fast")
        body = agent.calls[-1]["messages"][1]["content"]
        self.assertIn("PRIOR", body)
        self.assertIn("transcript text", body)

    async def test_one_line_uses_compact_instruction(self):
        agent = FakeLLMAgent(reply="x")
        await _summarize_omni(agent, "long summary", one_line=True)
        self.assertIn("ONE short line", agent.calls[-1]["messages"][0]["content"])


# ===========================================================================
# _compress + compress_session tool
# ===========================================================================

class TestCompress(unittest.IsolatedAsyncioTestCase):
    def _store(self, history=None, full_summary=""):
        Blob, _backing, _opens = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        store.state.active_history = history or []
        store.state.full_summary = full_summary
        return store

    async def test_compress_folds_history_into_summary(self):
        store = self._store(history=[{"role": "user", "text": "a"}], full_summary="OLD")
        agent = FakeLLMAgent(reply="NEW")
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        out = await sess._compress(merge_old=True)
        self.assertEqual(out, "NEW")
        self.assertEqual(store.state.full_summary, "NEW")
        self.assertEqual(store.state.active_history, [])
        self.assertIn("OLD", agent.calls[-1]["messages"][1]["content"])

    async def test_compress_no_merge_drops_prior(self):
        store = self._store(history=[{"role": "user", "text": "a"}], full_summary="OLD")
        agent = FakeLLMAgent(reply="FRESH")
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        await sess._compress(merge_old=False)
        self.assertNotIn("Previous summary", agent.calls[-1]["messages"][1]["content"])

    async def test_compress_keeps_turns_arriving_during_call(self):
        store = self._store(history=[{"role": "user", "text": "a"}])

        def late():
            store.state.active_history.append({"role": "user", "text": "late"})

        agent = FakeLLMAgent(reply="S", on_call=late)
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        await sess._compress(merge_old=True)
        # the snapshot folded only the first turn; the late one survives
        self.assertEqual(store.state.active_history, [{"role": "user", "text": "late"}])

    async def test_compress_nothing_to_do(self):
        store = self._store()
        agent = FakeLLMAgent()
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        self.assertEqual(await sess._compress(), "nothing to compress")
        self.assertEqual(agent.calls, [])

    async def test_compress_tool_schedules_and_runs(self):
        store = self._store(history=[{"role": "user", "text": "hi"}])
        agent = FakeLLMAgent(reply="DONE")
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        tool = make_session_tools(sess)[0]["tool_func"]
        self.assertEqual(await tool(merge_old=True), "compression scheduled")
        for _ in range(100):
            if store.state.full_summary:
                break
            await asyncio.sleep(0)
        self.assertEqual(store.state.full_summary, "DONE")


# ===========================================================================
# Seed text + session archive
# ===========================================================================

class TestSeedAndArchive(unittest.IsolatedAsyncioTestCase):
    def _store(self):
        Blob, _backing, _opens = make_fake_blob()
        return BlobStateStore("isaa/omni/omni_state.json", Blob)

    def test_build_seed_text_assembles_all_parts(self):
        store = self._store()
        store.state.world_model = WorldModel(user="Markin", agent_role="voice", routines=["x"])
        store.state.full_summary = "Discussed X."
        store.state.active_history = [{"role": "user", "text": "hey"},
                                      {"role": "assistant", "text": "hi"}]
        sess = OmniSession(StubOmniBackend(), state_store=store)
        seed = sess._build_seed_text()
        self.assertIn("resume", seed)
        self.assertIn("User: Markin", seed)
        self.assertIn("Discussed X.", seed)
        self.assertIn("Recent turns:", seed)

    async def test_archive_appends_one_liner(self):
        store = self._store()
        store.state.full_summary = "A long rolling summary of the call."
        agent = FakeLLMAgent(reply="one-liner")
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        await sess._archive_session()
        self.assertEqual(store.state.session_summaries, ["one-liner"])
        self.assertIn("ONE short line", agent.calls[-1]["messages"][0]["content"])

    async def test_archive_noop_without_summary(self):
        store = self._store()
        agent = FakeLLMAgent()
        sess = OmniSession(StubOmniBackend(), state_store=store, fallback=agent)
        await sess._archive_session()
        self.assertEqual(store.state.session_summaries, [])
        self.assertEqual(agent.calls, [])


# ===========================================================================
# Session restart + export
# ===========================================================================

class TestRestart(unittest.IsolatedAsyncioTestCase):
    def _setup(self, **session_kw):
        Blob, _backing, _opens = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        created: list = []

        def factory():
            b = StubOmniBackend()
            created.append(b)
            return b

        agent = FakeLLMAgent(reply="S")
        sess = OmniSession(
            StubOmniBackend(), state_store=store, backend_factory=factory,
            fallback=agent, **session_kw,
        )
        sess._tool_specs = [{"name": "noop"}]
        return store, created, sess

    def test_should_restart_threshold(self):
        store, _created, sess = self._setup(restart_at_turns=3)
        sess.turns = 2
        self.assertFalse(sess._should_restart())
        sess.turns = 3
        self.assertTrue(sess._should_restart())

    def test_should_restart_needs_factory(self):
        Blob, _b, _o = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        sess = OmniSession(StubOmniBackend(), state_store=store, restart_at_turns=2)
        sess.turns = 5
        self.assertFalse(sess._should_restart())  # no backend_factory

    async def test_do_restart_swaps_backend_and_reseeds(self):
        _store, created, sess = self._setup()
        old = sess.backend
        await sess._do_restart(compress=False)
        self.assertIs(sess.backend, created[0])
        self.assertIsNot(sess.backend, old)
        self.assertTrue(created[0].started)
        self.assertEqual(created[0].advertised_tools, [{"name": "noop"}])
        self.assertEqual(sess.turns, 0)
        self.assertEqual(len(created[0].sent_texts), 1)
        self.assertIn("resume", created[0].sent_texts[0])
        self.assertFalse(sess._restarting)

    async def test_turn_end_at_threshold_schedules_restart(self):
        _store, created, sess = self._setup(restart_at_turns=2, restart_compress=False)
        await sess.dispatch(OmniEvent.turn_end())   # turns=1
        self.assertFalse(sess._restarting)
        await sess.dispatch(OmniEvent.turn_end())   # turns=2 -> schedules
        self.assertTrue(sess._restarting)
        for _ in range(100):
            if not sess._restarting:
                break
            await asyncio.sleep(0)
        self.assertFalse(sess._restarting)
        self.assertIs(sess.backend, created[-1])
        self.assertEqual(sess.turns, 0)

    async def test_export_state_flushes_and_persists(self):
        Blob, _backing, _opens = make_fake_blob()
        store = BlobStateStore("isaa/omni/omni_state.json", Blob)
        sess = OmniSession(StubOmniBackend(), state_store=store)
        await sess.dispatch(OmniEvent.text_chunk("note", source="user"))  # only buffered
        out = sess.export_state()
        self.assertEqual(out["active_history"], [{"role": "user", "text": "note"}])
        # persisted to the blob
        store2 = BlobStateStore("isaa/omni/omni_state.json", Blob)
        self.assertEqual(store2.state.active_history, [{"role": "user", "text": "note"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
