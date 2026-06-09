#!/usr/bin/env python3
"""
isaa_voice.py - Mini live voice CLI flow (Omni mode)

Thin app layer over omni.OmniSession. Builds an ISAA agent, registers the
delegation + VFS-peek tools, then runs a hands-free voice loop using the
existing audio_io recorder/player.

Backend is config-driven (VoiceModeConfig.mode):
    omni_local | omni_cloud | stub | pipeline

Flags (via --kwargs):
    mode=omni_cloud         backend selection
    input_device=N          mic index (skip picker)
    output_device=N         speaker index (skip picker)
    enhance=true            run audio_enhancer (LavaSR) on output
    fallback=true           wire the classic STT->LLM->TTS pipeline as fallback
    verbose=true            INFO-level logging to the console

Run:
    python -m toolboxv2 -m isaa -f isaa_voice --kwargs mode=omni_cloud verbose=true
"""
import asyncio
import json
import logging
import signal

from toolboxv2 import App, get_app
from toolboxv2.utils.extras.Style import Style, cls

from toolboxv2.mods.isaa.base.audio_io.omni import (
    OmniSession,
    JobManager,
    VoiceModeConfig,
    make_agent_tools,
    make_vfs_peek_tools,
    StubOmniBackend, BlobStateStore, make_world_model_tools, OMNI_SYSTEM_INSTRUCTION,
)
from toolboxv2.mods.isaa.base.audio_io.audio_recorder import LocalMicRecorder
from toolboxv2.mods.isaa.base.audio_io.audioIo import LocalPlayer, NullPlayer, StreamingLocalPlayer
from toolboxv2.utils.extras.blobs import BlobFile

NAME = "isaa_voice"
ICON = "mic"
AUTH = False


def _truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


async def _announce(session, state: dict) -> None:
    """on_job_done handler: log it for the user and ask the model to speak the
    result. Uses the backend's text channel so the model voices it."""
    label = state.get("label", state.get("job_id"))
    result = state.get("result", "")
    print(Style.GREEN(f"\n[done] {label} ({state['job_id']}) → {state['status']}"))
    note = (
        f"[system] The task '{label}' (job {state['job_id']}) just finished. "
        f"Tell the user briefly and summarize:\n{result[:1500]}"
    )
    try:
        await session.backend.send_text(note)
    except Exception as e:
        print(Style.YELLOW(f"[announce] could not notify model: {e}"))


def _tool_specs_for_backend(agent) -> list:
    """Advertise the agent's tool schemas to the Omni model.

    The Omni model only knows it HAS tools if we hand it their schemas at
    session setup. toolboxv2's ToolManager exposes ready-made OpenAI/LiteLLM
    function schemas via get_all_litellm(); the Gemini backend unpacks that
    format. Falls back to a flat name/description list if that method moves.
    """
    tm = getattr(agent, "tool_manager", None)
    if tm is None:
        return []
    try:
        schemas = tm.get_all_litellm()
        if schemas:
            return schemas
    except Exception:
        pass
    # Fallback: build minimal flat specs from the registry.
    try:
        return [
            {"name": e.name, "description": getattr(e, "description", "")}
            for e in tm.get_all()
        ]
    except Exception:
        return []


def _list_devices(kind: str) -> list:
    """Return [(index, name, default_sr)] for input or output devices."""
    import sounddevice as sd
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev[key] > 0:
            out.append((i, dev["name"], int(dev["default_samplerate"])))
    return out


def _default_index(kind: str) -> int | None:
    try:
        import sounddevice as sd
        d = sd.default.device
        pair = d if isinstance(d, (list, tuple)) else (d, d)
        return pair[0] if kind == "input" else pair[1]
    except Exception:
        return None


async def _choose_device(prompt_session, kind: str) -> int | None:
    """Interactive device picker for 'input' or 'output'. Returns index or None."""
    try:
        devices = _list_devices(kind)
    except Exception as e:
        print(Style.YELLOW(f"Could not enumerate {kind} devices ({e}); using default."))
        return None
    if not devices:
        print(Style.YELLOW(f"No {kind} devices found; using default."))
        return None

    default_idx = _default_index(kind)
    print(Style.CYAN(f"Available {kind} devices:"))
    for idx, name, sr in devices:
        mark = " (default)" if idx == default_idx else ""
        print(f"  [{idx}] {name}  {sr}Hz{mark}")
    print(Style.GREY("Enter device number, or blank for default."))

    raw = (await prompt_session.prompt_async(f"{kind}> ")).strip()
    if not raw:
        return None
    try:
        chosen = int(raw)
    except ValueError:
        print(Style.YELLOW(f"'{raw}' is not a number; using default."))
        return None
    if chosen not in {idx for idx, _, _ in devices}:
        print(Style.YELLOW(f"Device {chosen} not a valid {kind}; using default."))
        return None
    return chosen


async def run(app: App, args=None, mode: str = "stub", device: str = "cuda",
              input_device: int | None = None, output_device: int | None = None,
              enhance=False, fallback=False, verbose=True, vad=True, **kwargs):
    cls()

    if _truthy(verbose):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("omni").setLevel(logging.DEBUG)

    print(Style.CYAN("╔══════════════════════════════════╗"))
    print(Style.CYAN("║    ISAA Voice · Omni Live Mode   ║"))
    print(Style.CYAN("╚══════════════════════════════════╝"))
    print(Style.GREY(f"mode={mode} device={device} enhance={_truthy(enhance)} "
                     f"fallback={_truthy(fallback)} verbose={_truthy(verbose)}"))
    print()

    isaa = app.get_mod("isaa")
    if isaa is None:
        print(Style.RED("ERROR: ISAA module not loaded."))
        return

    await isaa.init_isaa(name="isaa_voice")
    builder: 'FlowAgentBuilder' = isaa.get_agent_builder("isaa_voice", add_base_tools=True)
    await isaa.register_agent(builder
                              .with_stream(True)
                              .with_models(fast_model="9rou/fast", complex_model="9rou/complex")
                              )
    agent = await isaa.get_agent("speed")
    print(Style.GREEN(f"Agent ready: {agent.amd.name}"))

    # --- Tool bridge: shared JobManager for delegation + nonblocking tools ---
    jobs = JobManager()
    agent.add_tools(make_agent_tools(jobs, lambda name: agent))
    agent.add_tools(make_vfs_peek_tools(agent, jobs))
    store = BlobStateStore("isaa/omni/omni_state.json", BlobFile)  # key=... optional
    agent.add_tools(make_world_model_tools(store))
    # OmniSession(..., fallback=agent, state_store=store, backend_factory=cfg.build_backend)
    print(Style.GREEN("Tools: delegate, agent_result, agent_status, vfs_peek, vfs_tree_peek"))
    print(store.load())

    # ---

    # --- Backend per config ---
    cfg = VoiceModeConfig(mode=mode, device=device, agent=agent, kwargs=(
        {"voice": "Algenib", "input_transcription": True, "output_transcription": True, "thinking_level": None,}
        if mode == "omni_cloud" else {})
    ) # Algieba

    def _make_backend():
        b = cfg.build_backend()
        if b is not None and hasattr(b, "system_instruction"):
            b.system_instruction = OMNI_SYSTEM_INSTRUCTION
        return b

    backend = _make_backend()
    if backend is None:
        print(Style.YELLOW("mode=pipeline → classic a_audio path, not OmniSession."))
        return
    print(Style.GREEN(f"Backend: {backend.backend_name}"))

    # --- Device selection (live modes only) ---
    from prompt_toolkit import PromptSession
    prompt_session = PromptSession()
    recorder = None
    player = NullPlayer()
    output_sr = None
    stream_audio = False
    if mode != "stub":
        if input_device is None:
            input_device = await _choose_device(prompt_session, "input")
        print(Style.GREEN(f"Mic: {'default' if input_device is None else f'#{input_device}'}"))
        recorder = LocalMicRecorder(device=input_device)

        if output_device is None:
            output_device = await _choose_device(prompt_session, "output")
        print(Style.GREEN(f"Speaker: {'default' if output_device is None else f'#{output_device}'}"))
        # Gemini streams 24k PCM in many small chunks -> use the gapless streaming
        # player + per-chunk delivery (buffer_audio=False) for minimal latency.
        stream_audio = backend.backend_name == "GeminiLiveBackend"
        if stream_audio:
            output_sr = 24000
            player = StreamingLocalPlayer(device=output_device, sample_rate=24000)
            print(Style.GREEN("[audio] gapless streaming player @24k (low latency)"))
        else:
            player = LocalPlayer(device=output_device or 0)

    # --- Optional audio enhancement (flag) ---
    enhancer = None
    if _truthy(enhance):
        try:
            from toolboxv2.mods.isaa.base.audio_io.audio_enhancer import (
                AudioEnhancer, EnhancerConfig,
            )
            enhancer = AudioEnhancer(EnhancerConfig(device=device))
            if not enhancer.is_available:
                print(Style.YELLOW("[enhance] LavaSR checkpoint missing — passthrough or torch"))
            else:
                print(Style.GREEN("[enhance] LavaSR enabled"))
        except Exception as e:
            print(Style.YELLOW(f"[enhance] unavailable: {e}"))

    # --- VAD gate (default on for live modes): only stream while speaking ---
    vad_obj = None
    if _truthy(vad) and mode != "stub":
        try:
            from toolboxv2.mods.isaa.base.audio_io.audio_live import SileroVAD
            vad_obj = SileroVAD(threshold=0.5)
            print(Style.GREEN("[vad] Silero VAD on — only streaming speech (saves cost)"))
        except Exception as e:
            print(Style.YELLOW(f"[vad] unavailable, streaming continuously: {e}"))
    # --- Optional speaker recognition (flag: speakers=true) ---
    speakers_obj = None
    if _truthy(kwargs.get("speakers")) and mode != "stub":
        from toolboxv2.mods.isaa.base.audio_io.omni import (
            SpeakerRegistry, StubSpeakerEmbedder,
        )
        speakers_obj = SpeakerRegistry(
            "isaa/omni/speakers.json", BlobFile,
            embedder=StubSpeakerEmbedder(),  # swap for a real .embed(pcm)
            label_hook=lambda emb, score: None,  # app names unknown voices here
        )
        print(Style.GREEN("[speakers] registry on (stub embedder — plug real .embed)"))

    session = OmniSession(
        backend,
        backend_factory=_make_backend,
        recorder=recorder,
        player=player,
        tools=agent.tool_manager,
        jobs=jobs,
        background_tools=set(),
        output_sample_rate=output_sr,
        buffer_audio=not stream_audio,  # stream per-chunk into the OutputStream
        enhancer=enhancer,
        vad=vad_obj,
        state_store=store,
        summarizer_agent=agent,
        speakers=speakers_obj,
        #on_phase=lambda ph, meta: print(Style.GREY(
        #    f"  [phase] {ph.value}" + (f" {meta}" if meta else ""))),
        on_text=lambda t: print(Style.BLUE(f"  ‹{t}›"), end="\r"),
        on_job_done=lambda st: _announce(session_ref, st),
    )
    session_ref = session  # captured by the on_job_done lambda above

    agent.add_tools(session.compress_tool)

    # stub: script a deterministic demo turn (no model / mic needed)
    if isinstance(backend, StubOmniBackend):
        from toolboxv2.mods.isaa.base.audio_io.omni import OmniEvent
        backend.close_on_turn_end = True
        backend.on_tool_result = lambda cid, res: backend.queue(
            OmniEvent.text_chunk(f"[stub] tool said: {res}"),
            OmniEvent.turn_end(),
        )
        backend.queue(
            OmniEvent.text_chunk("[stub] hello — checking agent status"),
            OmniEvent.call("agent_status", {}, call_id="s1"),
        )

    # --- Cooperative shutdown via signal -> asyncio.Event ---
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(*_):
        if not stop_event.is_set():
            print(Style.YELLOW("\nStopping… (Ctrl-C again to force-quit)"))
            stop_event.set()

    installed = []
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):
            try:
                signal.signal(sig, lambda *_: _request_stop())
            except (ValueError, OSError):
                pass

    print(Style.GREY("Starting session…  (Ctrl-C to stop)"))
    await session.start(tool_specs=_tool_specs_for_backend(agent))

    try:
        if isinstance(backend, StubOmniBackend):
            wait_task = asyncio.ensure_future(session.wait(timeout=5.0))
            stop_task = asyncio.ensure_future(stop_event.wait())
            await asyncio.wait({wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in (wait_task, stop_task):
                if not t.done():
                    t.cancel()
        else:
            # The OmniSession notify loop announces finished jobs (the model
            # speaks). We just wait for shutdown — no repeated status spam.
            await stop_event.wait()
    finally:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        await session.stop()
        print(Style.GREEN("\n" + session.status_line()))
        print(Style.GREEN("Session stopped."))


if __name__ == "__main__":
    asyncio.run(run(get_app(), mode="stub"))
