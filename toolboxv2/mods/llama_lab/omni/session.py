# file: toolboxv2/mods/llama_lab/omni/session.py
"""OmniSession — wires a client (WebSocket) to an OmniBackend full-duplex.

Input  : client PCM frames -> VAD gate (unless backend.needs_silence) -> backend
Output : backend events (TEXT/AUDIO/TOOL_CALL/TURN_END/ERROR) -> client via `send`
Extras : barge-in (user speaks over the model -> cancel current turn) and local
         tool dispatch (TOOL_CALL -> registered async fn -> send_tool_result).
"""

import asyncio
import inspect
from typing import Awaitable, Callable, Optional

from .backend import OmniBackend
from .types import OmniEvent, OmniEventType
from .vad import VAD


class OmniSession:
    def __init__(self, backend: OmniBackend,
                 send: Callable[[dict], Awaitable | None],
                 tools: Optional[dict] = None, tool_specs: Optional[list] = None,
                 sample_rate: int = 16000):
        self.backend = backend
        self._send = send
        self.tools = tools or {}                 # name -> async/sync fn(args)->str
        self.tool_specs = tool_specs or []       # OpenAI tool schemas to advertise
        self.vad = VAD(sample_rate=sample_rate)
        self.user_speaking = False
        self.model_speaking = False
        self._consumer: Optional[asyncio.Task] = None
        self._running = False

    async def _emit(self, payload: dict):
        res = self._send(payload)
        if inspect.isawaitable(res):
            await res

    async def start(self):
        await self.backend.start(tools=self.tool_specs or None)
        self._running = True
        self._consumer = asyncio.create_task(self._consume())
        await self._emit({"type": "ready", "backend": self.backend.backend_name})

    # -- client -> backend --------------------------------------------------

    async def feed_audio(self, pcm: bytes):
        if not pcm:
            return
        if self.backend.needs_silence:           # realtime backend: stream everything
            await self.backend.send_audio(pcm)
            return
        was = self.user_speaking
        starts = False
        for t in self.vad.feed(pcm):
            if t == "speech_start":
                starts = True
                self.user_speaking = True
                if self.model_speaking:
                    await self._barge_in()
            elif t == "speech_end":
                self.user_speaking = False
                await self.backend.commit_input()
        if was or self.user_speaking or starts:
            await self.backend.send_audio(pcm)

    async def feed_text(self, text: str):
        if self.model_speaking:
            await self._barge_in()
        await self.backend.send_text(text)

    async def feed_media(self, data: bytes, mime_type: str, as_turn: bool = False):
        await self.backend.send_media(data, mime_type, as_turn=as_turn)

    async def _barge_in(self):
        if not self.backend.supports_restart:
            return
        await self.backend.cancel()
        self.model_speaking = False
        await self._emit({"type": "interrupt"})

    # -- backend -> client --------------------------------------------------

    async def _consume(self):
        try:
            async for ev in self.backend.events():
                if ev.type in (OmniEventType.TEXT, OmniEventType.AUDIO):
                    self.model_speaking = True
                elif ev.type is OmniEventType.TURN_END:
                    self.model_speaking = False

                if ev.type is OmniEventType.TOOL_CALL:
                    await self._emit(ev.wire())          # show tool call in UI
                    asyncio.create_task(self._run_tool(ev))
                    continue
                await self._emit(ev.wire())
        except asyncio.CancelledError:
            pass

    async def _run_tool(self, ev: OmniEvent):
        fn = self.tools.get(ev.name)
        try:
            if fn is None:
                result = f"error: no tool named '{ev.name}'"
            else:
                out = fn(ev.arguments)
                result = await out if inspect.isawaitable(out) else out
            await self.backend.send_tool_result(ev.call_id, str(result))
        except Exception as e:
            await self.backend.send_tool_result(ev.call_id, f"error: {e}")

    async def stop(self):
        self._running = False
        try:
            await self.backend.stop()
        except Exception:
            pass
        if self._consumer:
            self._consumer.cancel()
            try:
                await self._consumer
            except (asyncio.CancelledError, Exception):
                pass
