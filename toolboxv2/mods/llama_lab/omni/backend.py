# file: toolboxv2/mods/llama_lab/omni/backend.py
"""OmniBackend contract + a concrete backend for llama.cpp-served omni models.

LlamaOmniBackend turns the served (OpenAI-compatible) llama-server into a
live, turn-based omni endpoint: audio-in (libmtmd) -> streamed text out, with
tool-calls and optional TTS audio out. Latency is kept low by streaming text
deltas as TEXT events the moment they arrive.
"""

import asyncio
import base64
import io
import json
import wave
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import httpx

from .types import OmniEvent, OmniEventType


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ============================================================ contract =======

class OmniBackend(ABC):
    """Contract every Omni backend implements.

    Lifecycle:
        await backend.start(tools=[...])      # open model/session, advertise tools
        await backend.send_audio(pcm)         # push input frames (PCM int16 16k mono)
        await backend.commit_input()          # end-of-user-turn (turn-based backends)
        await backend.send_tool_result(id, s) # answer a TOOL_CALL event
        async for ev in backend.events(): ... # consume AUDIO/TEXT/TOOL_CALL/TURN_END/ERROR
        await backend.stop()                  # teardown
    """
    needs_silence = False    # True -> OmniSession streams ALL frames (no VAD gate)
    supports_restart = True  # False -> OmniSession never restart/reseeds this backend

    @abstractmethod
    async def start(self, tools: Optional[list[dict]] = None) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...

    @abstractmethod
    async def send_tool_result(self, call_id: str, result: str) -> None: ...

    async def commit_input(self) -> None:
        """Signal end of the user's input turn. Optional — realtime backends
        (needs_silence=True) detect turns themselves and leave this a no-op."""
        return None

    async def send_text(self, text: str) -> None:
        """Inject a text turn into the live session. Optional."""
        return None

    async def send_media(self, data: bytes, mime_type: str, *, as_turn: bool = False) -> None:
        """Send an image/video frame|blob. as_turn=True -> discrete content turn;
        else realtime video frame. Optional — no-op by default."""
        return None

    async def cancel(self) -> None:
        """Interrupt the current model turn (barge-in). Optional."""
        return None

    @abstractmethod
    def events(self) -> AsyncIterator[OmniEvent]: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @property
    def backend_name(self) -> str:
        return type(self).__name__


# ============================================================ llama.cpp ======

class LlamaOmniBackend(OmniBackend):
    needs_silence = False        # turn-based: needs the full utterance
    supports_restart = True

    def __init__(self, base_url: str, model: str, *, sample_rate: int = 16000,
                 system: str = "", tts_base_url: str = "", tts_model: str = "",
                 tts_voice: str = "alloy", temperature: float = 0.7):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sr = sample_rate
        self.tts_base_url = tts_base_url.rstrip("/") if tts_base_url else ""
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.temperature = temperature
        self._messages = [{"role": "system", "content": system}] if system else []
        self._tools: list = []
        self._in_buf = bytearray()
        self._pending_media: list = []
        self._q: asyncio.Queue = asyncio.Queue()
        self._http: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task] = None

    # -- lifecycle ----------------------------------------------------------

    async def start(self, tools=None):
        self._tools = tools or []
        self._http = httpx.AsyncClient(timeout=300.0)

    async def stop(self):
        await self.cancel()
        if self._http:
            await self._http.aclose()
        await self._q.put(None)              # close events()

    async def cancel(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    # -- input --------------------------------------------------------------

    async def send_audio(self, pcm: bytes):
        self._in_buf += pcm

    async def send_media(self, data, mime_type, *, as_turn=False):
        b64 = base64.b64encode(data).decode()
        self._pending_media.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
        if as_turn:
            await self._run([{"role": "user", "content": self._drain_media()}])

    async def send_text(self, text: str):
        content = [{"type": "text", "text": text}] + self._drain_media()
        await self._run([{"role": "user", "content": content}])

    async def commit_input(self):
        audio = bytes(self._in_buf)
        self._in_buf.clear()
        if not audio:
            return
        wav = pcm16_to_wav(audio, self.sr)
        part = {"type": "input_audio",
                "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}
        content = [part] + self._drain_media()
        await self._run([{"role": "user", "content": content}])

    def _drain_media(self):
        m, self._pending_media = self._pending_media, []
        return m

    # -- tool answer --------------------------------------------------------

    async def send_tool_result(self, call_id: str, result: str):
        await self._run([{"role": "tool", "tool_call_id": call_id, "content": result}])

    # -- generation ---------------------------------------------------------

    async def _run(self, new_messages: list):
        await self.cancel()
        self._messages += new_messages
        self._task = asyncio.create_task(self._stream())

    async def _stream(self):
        payload = {"model": self.model, "messages": self._messages, "stream": True,
                   "temperature": self.temperature}
        if self._tools:
            payload["tools"] = self._tools
        text_acc = ""
        tool_calls: dict = {}
        try:
            async with self._http.stream("POST", f"{self.base_url}/chat/completions",
                                         json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace")[:300]
                    await self._q.put(OmniEvent(OmniEventType.ERROR, f"{r.status_code}: {body}"))
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta.get("content"):
                        text_acc += delta["content"]
                        await self._q.put(OmniEvent(OmniEventType.TEXT, delta["content"]))
                    for tc in delta.get("tool_calls", []):
                        i = tc.get("index", 0)
                        slot = tool_calls.setdefault(i, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._q.put(OmniEvent(OmniEventType.ERROR, str(e)))
            return

        if tool_calls:
            self._messages.append({"role": "assistant", "content": text_acc or None,
                                   "tool_calls": [{"id": s["id"], "type": "function",
                                                   "function": {"name": s["name"], "arguments": s["args"]}}
                                                  for s in tool_calls.values()]})
            for s in tool_calls.values():
                try:
                    args = json.loads(s["args"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": s["args"]}
                await self._q.put(OmniEvent(OmniEventType.TOOL_CALL, call_id=s["id"],
                                            name=s["name"], arguments=args))
            return                       # wait for send_tool_result -> continues

        self._messages.append({"role": "assistant", "content": text_acc})
        await self._q.put(OmniEvent(OmniEventType.TEXT, "", final=True))
        if self.tts_base_url and text_acc.strip():
            await self._tts(text_acc)
        await self._q.put(OmniEvent(OmniEventType.TURN_END))

    async def _tts(self, text: str):
        try:
            r = await self._http.post(
                f"{self.tts_base_url}/audio/speech",
                json={"model": self.tts_model, "input": text, "voice": self.tts_voice,
                      "response_format": "wav"})
            if r.status_code < 400 and r.content:
                await self._q.put(OmniEvent(OmniEventType.AUDIO, r.content,
                                            encoding="wav"))
        except Exception:
            pass                          # TTS is best-effort

    async def events(self) -> AsyncIterator[OmniEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev
