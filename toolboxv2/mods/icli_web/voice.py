"""
Voice WebSocket handler for /icli/voice.

Registers with the existing WSWorker on the configured WS port.
In debug_runner, the debug_runner can optionally proxy /icli/voice to the
same port (see patch below). In production, nginx already routes WS there.

Protocol (client → server):
  {"type": "query", "agent": "self", "text": "..."}

Protocol (server → client):
  {"type": "text_chunk", "text": "..."}    # streaming agent text
  {"type": "audio_start"}                  # PCM/OGG chunks follow
  <binary>  <binary>  <binary> ...         # raw audio bytes
  {"type": "audio_end"}
  {"type": "done"}
  {"type": "error", "error": "..."}
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any, Optional

from .client import IcliWebClient
log = logging.getLogger("icli_web.voice")


class VoiceSession:
    """One WS connection per user. Owns an audio sink that forwards to the
    remote WebSocket."""

    def __init__(self, ws):
        self.ws = ws
        self.bridge = IcliWebClient.get()

    async def handle(self) -> None:
        try:
            async for message in self.ws:
                if isinstance(message, (bytes, bytearray)):
                    continue  # ignore inbound binary for now
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "query":
                    await self._handle_query(msg.get("agent", "self"),
                                             msg.get("text", ""))
        except Exception as e:
            log.warning("voice ws closed: %s", e)

    async def _handle_query(self, agent_name: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        icli = self.bridge._icli
        if icli is None:
            await self._send_json({"type": "error", "error": "icli not attached"})
            return

        # Build an audio sink that streams TTS chunks back to the browser.
        sink = WebAudioSink(self.ws)

        # Swap in the web audio sink just for this query. icli's
        # AudioStreamPlayer has a `speak_text(text, on_chunk=...)` or similar
        # API in your code. We try a few variants.
        player = getattr(icli, "audio_player", None)
        if player is None:
            await self._send_json({"type": "error", "error": "no audio_player"})
            return

        # Tell orb we're thinking
        await self._send_json({"type": "text_chunk", "text": ""})

        # Run the agent; stream text back as it arrives.
        # icli has _drain_agent_stream / _run_agent_query — we pick the
        # public-ish one that returns the stream.
        try:
            run = (getattr(icli, "run_agent_for_web", None)
                   or getattr(icli, "_run_agent_query", None))
            if run is None:
                await self._send_json({"type": "error",
                                       "error": "no agent entry point"})
                return

            async for chunk in run(agent_name, text):
                if chunk.get("type") == "content" and chunk.get("chunk"):
                    await self._send_json({"type": "text_chunk",
                                           "text": chunk["chunk"]})
                elif chunk.get("type") == "final_answer":
                    answer = chunk.get("answer") or chunk.get("content", "")
                    if answer:
                        await self._send_json({"type": "text_chunk",
                                               "text": answer})
                    # now speak it
                    await self._send_json({"type": "audio_start"})
                    await self._speak(player, answer, sink)
                    await self._send_json({"type": "audio_end"})

            await self._send_json({"type": "done"})
        except Exception as e:
            log.exception("agent run failed")
            await self._send_json({"type": "error", "error": str(e)})

    async def _speak(self, player, text: str, sink) -> None:
        """Call the existing speak pipeline, routing audio through sink."""
        # Your AudioStreamPlayer exposes .speak(text, on_chunk=...) per the
        # create_speak_tool wiring. If the API differs in your branch, adjust
        # here. We avoid duplicating TTS config — icli already set it up.
        try:
            # Preferred: async streaming API
            speak = getattr(player, "speak_async", None) or getattr(player, "speak", None)
            if speak is None:
                return
            result = speak(text, on_chunk=sink.write)
            if asyncio.iscoroutine(result):
                await result
        except TypeError:
            # Fallback: no on_chunk kwarg — just run and let sink capture via
            # player's own broadcast hook if present
            speak(text)

    async def _send_json(self, obj: dict) -> None:
        try:
            await self.ws.send(json.dumps(obj))
        except Exception:
            pass


class WebAudioSink:
    """on_chunk callback that ships raw audio bytes to the WS client."""
    def __init__(self, ws):
        self.ws = ws
        self._loop = asyncio.get_event_loop()

    def write(self, chunk: bytes) -> None:
        """Called from the TTS thread — schedule send on the WS loop."""
        if not chunk:
            return
        asyncio.run_coroutine_threadsafe(self._send(chunk), self._loop)

    async def _send(self, chunk: bytes) -> None:
        try:
            await self.ws.send(chunk)
        except Exception:
            pass


# ─── Registration hook ────────────────────────────────────────────────────────
# Called by WSWorker on startup to register /icli/voice as a path.

def register(ws_worker) -> None:
    """Wire /icli/voice to VoiceSession."""
    async def _handler(ws, path):
        if path != "/icli/voice":
            await ws.close()
            return
        await VoiceSession(ws).handle()

    # WSWorker exposes add_path or register_handler; shape TBD
    if hasattr(ws_worker, "add_path"):
        ws_worker.add_path("/icli/voice", _handler)
    elif hasattr(ws_worker, "register_handler"):
        ws_worker.register_handler("/icli/voice", _handler)
    else:
        log.warning("WSWorker has no known registration method — "
                    "add manually in ws_worker.py")
