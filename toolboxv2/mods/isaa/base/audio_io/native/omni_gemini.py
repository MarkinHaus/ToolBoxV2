"""
omni_gemini.py
==============

Real CloudOmniBackend over the Gemini Live API (BidiGenerateContent websocket).
Consumes only the public seam from omni.py (OmniBackend / OmniEvent). Works out
of the box on the free tier:
    * native-audio model, AUDIO out, PCM16/16k in (Gemini's exact contract)
    * function calling mapped onto OmniSession's tool bridge
    * token efficiency: contextWindowCompression (sliding window) keeps long
      voice sessions inside the free-tier budget; transcription is opt-in
    * "prompt caching": a stable systemInstruction + sliding-window compression
      so the rolling prefix is reused instead of resent each turn

The websocket I/O (start/_recv_loop) is the only untested surface; the setup
payload builder, the per-message parser, and the send envelopes are unit-tested
via an injected fake socket.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Callable, Optional

from toolboxv2.mods.isaa.base.audio_io.omni import OmniBackend, OmniEvent, _SENTINEL

try:
    from toolboxv2 import get_logger
    logger = get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-3.1-flash-live-preview"#"gemini-2.5-flash-native-audio-preview-12-2025"
_WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


class GeminiLiveBackend(OmniBackend):
    """Gemini Live API backend.

    Audio contract (fixed by Gemini):
        input  : raw PCM int16, 16 kHz, little-endian, mime 'audio/pcm;rate=16000'
        output : raw PCM int16, 24 kHz  (emitted as OmniEvent.AUDIO tagged
                 meta['sample_rate_out']=24000 so the session plays it correctly)
    """

    INPUT_SR = 16000
    OUTPUT_SR = 24000

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key_env: str = "GEMINI_API_KEY",
        *,
        system_instruction: str = "You are a helpful real-time voice assistant.",
        voice: Optional[str] = None,
        thinking_level: Optional[str] = None,
        output_transcription: bool = False,
        input_transcription: bool = False,
        compression_trigger_tokens: int = 16000,
        connect_factory: Optional[Callable[[str], Any]] = None,
    ):
        self.model_id = model_id or os.getenv("GEMINI_LIVE_MODEL", _DEFAULT_MODEL)
        self.api_key_env = api_key_env
        self.system_instruction = system_instruction
        self.voice = voice
        self.thinking_level = thinking_level
        self.output_transcription = output_transcription
        self.input_transcription = input_transcription
        self.compression_trigger_tokens = compression_trigger_tokens
        self._connect_factory = connect_factory

        self._ws: Any = None
        self._closed = False
        self._q: "asyncio.Queue[Any]" = asyncio.Queue()
        self._recv_task: Optional[asyncio.Task] = None
        self._tools: list[dict] = []
        self._pending_call_ids: dict[str, str] = {}

    # -- setup payload --------------------------------------------------------
    def _build_setup(self, tools: Optional[list[dict]]) -> dict:
        gen_config: dict = {
            "responseModalities": ["AUDIO"],
            "thinkingConfig": { "thinkingLevel": self.thinking_level },
                            }
        if self.voice:
            gen_config["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}
            }
        setup: dict = {
            "model": f"models/{self.model_id}",
            "generationConfig": gen_config,
            "systemInstruction": {"parts": [{"text": self.system_instruction}]},
            "contextWindowCompression": {
                "slidingWindow": {},
                "triggerTokens": self.compression_trigger_tokens,
            },
            # "historyConfig": {"initialHistoryInClientContent": True},
        }
        if self.output_transcription:
            setup["outputAudioTranscription"] = {}
        if self.input_transcription:
            setup["inputAudioTranscription"] = {}
        fn_decls = self._tools_to_function_declarations(tools or [])
        if fn_decls:
            setup["tools"] = [{"functionDeclarations": fn_decls}]
        return {"setup": setup}

    # Gemini accepts only a subset of OpenAPI 3.0 Schema keys. Anything else
    # (default, additionalProperties, $ref, anyOf/oneOf, title, ...) makes the
    # WHOLE setup fail with code 1007. We sanitize every schema recursively.
    _ALLOWED_SCHEMA_KEYS = {
        "type", "format", "description", "nullable", "enum",
        "properties", "items", "required",
        "minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength",
    }
    _VALID_TYPES = {"string", "number", "integer", "boolean", "array", "object"}

    @classmethod
    def _sanitize_schema(cls, schema: Any) -> dict:
        """Reduce a JSON-schema fragment to Gemini's OpenAPI-3.0 subset.

        - drops unknown keys (default, additionalProperties, $ref, anyOf, title…)
        - normalizes/repairs the 'type' (lowercases; Optional/union -> first
          concrete type; missing -> 'string')
        - recurses into properties/items
        - guarantees object schemas have a 'properties' dict
        """
        if not isinstance(schema, dict):
            return {"type": "string"}

        out: dict = {}
        # ---- type normalization ----
        t = schema.get("type")
        if isinstance(t, list):  # JSON-schema union e.g. ["string","null"]
            concrete = [x for x in t if x != "null"]
            t = concrete[0] if concrete else "string"
            if "null" in schema.get("type", []):
                out["nullable"] = True
        if isinstance(t, str):
            t = t.lower()
        if t not in cls._VALID_TYPES:
            # anyOf/oneOf or unknown -> best-effort fallback
            t = "object" if schema.get("properties") else "string"
        out["type"] = t

        for k in ("description", "format", "enum", "nullable",
                  "minItems", "maxItems", "minimum", "maximum",
                  "minLength", "maxLength"):
            if k in schema and k not in out:
                out[k] = schema[k]

        if t == "object":
            props = schema.get("properties")
            out["properties"] = {
                name: cls._sanitize_schema(sub)
                for name, sub in (props or {}).items()
            } if isinstance(props, dict) else {}
            req = schema.get("required")
            if isinstance(req, list) and req:
                out["required"] = [r for r in req if isinstance(r, str)]

        if t == "array":
            out["items"] = cls._sanitize_schema(schema.get("items") or {"type": "string"})

        return out

    @classmethod
    def _tools_to_function_declarations(cls, tools: list[dict]) -> list[dict]:
        """Map tool specs to Gemini functionDeclarations.

        Accepts BOTH:
          - LiteLLM/OpenAI format: {"type":"function","function":{name,description,parameters}}
            (toolboxv2 ToolManager.get_all_litellm())
          - flat format: {name, description, parameters}
        Every parameters schema is passed through _sanitize_schema so unsupported
        keys can't reject the whole session.
        """
        decls = []
        for t in tools:
            spec = t.get("function") if isinstance(t.get("function"), dict) else t
            name = spec.get("name")
            if not name:
                continue
            raw_params = spec.get("parameters") or {"type": "object", "properties": {}}
            params = cls._sanitize_schema(raw_params)
            if params.get("type") != "object":
                params = {"type": "object", "properties": {}}
            decls.append({
                "name": name,
                "description": (spec.get("description") or "")[:1024],
                "parameters": params,
            })
        return decls

    # -- backend API ----------------------------------------------------------
    async def start(self, tools: Optional[list[dict]] = None) -> None:
        self._tools = list(tools or [])
        key = os.getenv(self.api_key_env)
        if not key:
            await self._q.put(OmniEvent.error(f"{self.api_key_env} not set"))
            raise RuntimeError(f"{self.api_key_env} not set")

        url = f"{_WS_BASE}?key={key}"
        if self._connect_factory is not None:
            self._ws = await self._connect_factory(url)
        else:
            try:
                import websockets
            except ImportError as e:  # pragma: no cover
                await self._q.put(OmniEvent.error("websockets not installed"))
                raise ImportError("pip install websockets") from e
            self._ws = await websockets.connect(url, max_size=None)

        await self._ws.send(json.dumps(self._build_setup(tools)))
        logger.info("GeminiLiveBackend: connected model=%s tools=%d", self.model_id, len(self._tools))
        self._recv_task = asyncio.ensure_future(self._recv_loop())

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is None or self._closed:
            return
        msg = {"realtimeInput": {"audio": {
            "data": base64.b64encode(pcm).decode("ascii"),
            "mimeType": f"audio/pcm;rate={self.INPUT_SR}",
        }}}
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as e:  # noqa: BLE001 - WS may have died; stop quietly
            self._closed = True
            logger.debug("GeminiLiveBackend.send_audio after close: %s", e)

    async def send_tool_result(self, call_id: str, result: str) -> None:
        if self._ws is None:
            return
        name = self._pending_call_ids.pop(call_id, "")
        try:
            payload = json.loads(result)
            if not isinstance(payload, dict):
                payload = {"result": payload}
        except (json.JSONDecodeError, TypeError):
            payload = {"result": result}
        msg = {"toolResponse": {"functionResponses": [
            {"id": call_id, "name": name, "response": payload}
        ]}}
        await self._ws.send(json.dumps(msg))
        logger.debug("GeminiLiveBackend: tool result sent for %s", name)

    async def send_text(self, text: str) -> None:
        if self._ws is None or self._closed:
            return
        try:
            await self._ws.send(json.dumps({"realtimeInput": {"text": text}}))
        except Exception as e:  # noqa: BLE001
            self._closed = True
            logger.debug("GeminiLiveBackend.send_text after close: %s", e)

    async def send_media(self, data: bytes, mime_type: str, *, as_turn: bool = False) -> None:
        """Image / video frame (update 3). as_turn=True -> discrete user turn via
        clientContent (one-shot image/file); else realtime video frame."""
        if self._ws is None or self._closed:
            return
        b64 = base64.b64encode(data).decode("ascii")
        if as_turn:
            msg = {"clientContent": {
                "turns": [{"role": "user",
                           "parts": [{"inlineData": {"mimeType": mime_type, "data": b64}}]}],
                "turnComplete": False,
            }}
        else:
            msg = {"realtimeInput": {"video": {"data": b64, "mimeType": mime_type}}}
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as e:  # noqa: BLE001
            self._closed = True
            logger.debug("GeminiLiveBackend.send_media after close: %s", e)

    async def events(self) -> AsyncIterator[OmniEvent]:
        while True:
            ev = await self._q.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def stop(self) -> None:
        self._closed = True
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
        await self._q.put(_SENTINEL)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass

    # -- receive loop ---------------------------------------------------------
    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                self._handle_server_message(raw)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            self._closed = True
            await self._q.put(OmniEvent.error(f"gemini recv: {e}"))

    def _handle_server_message(self, raw: Any) -> None:
        """Translate one Gemini server message into OmniEvents. Pure parsing —
        unit-testable without a socket."""
        try:
            msg = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        except (json.JSONDecodeError, TypeError):
            return

        sc = msg.get("serverContent")
        if sc:
            # barge-in: Gemini signals the user interrupted the model
            if sc.get("interrupted"):
                self._q.put_nowait(OmniEvent.interrupted())

            model_turn = sc.get("modelTurn") or {}
            for part in model_turn.get("parts", []):
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    try:
                        pcm = base64.b64decode(inline["data"])
                    except Exception:  # noqa: BLE001
                        continue
                    self._q.put_nowait(OmniEvent.audio_chunk(pcm, sample_rate_out=self.OUTPUT_SR))
                elif part.get("text"):
                    self._q.put_nowait(OmniEvent.text_chunk(part["text"]))

            ot = sc.get("outputTranscription")
            if ot and ot.get("text"):
                self._q.put_nowait(OmniEvent.text_chunk(ot["text"], source="output"))
            it = sc.get("inputTranscription")
            if it and it.get("text"):
                self._q.put_nowait(OmniEvent.text_chunk(it["text"], source="input"))

            if sc.get("turnComplete"):
                self._q.put_nowait(OmniEvent.turn_end())

        tc = msg.get("toolCall")
        if tc:
            for fc in tc.get("functionCalls", []):
                call_id = fc.get("id") or fc.get("name", "")
                name = fc.get("name", "")
                self._pending_call_ids[call_id] = name
                self._q.put_nowait(OmniEvent.call(name, fc.get("args", {}), call_id=call_id))
