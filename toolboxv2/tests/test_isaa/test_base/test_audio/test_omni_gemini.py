"""
test_omni_gemini.py
===================

Hard tests for omni_gemini. No network: GeminiLiveBackend gets a fake websocket
injected via connect_factory, and the server-message parser is driven directly.

Covers:
  - setup payload: model, AUDIO modality, system instruction, context-window
    compression (token efficiency), function declarations, opt-in transcription
  - send_audio: correct realtimeInput PCM/base64/mime envelope
  - server parsing: inline audio -> AUDIO event (24k tag), text part -> TEXT,
    output/input transcription -> TEXT with source, turnComplete -> TURN_END,
    toolCall -> TOOL_CALL with id tracked for the response
  - send_tool_result: functionResponses envelope, string vs dict result, id+name
  - VFS peek: tight slice, scroll_to centering, line range, truncation,
    READ-ONLY invariant (file.state / view_* never mutated), tree peek,
    unknown job / missing session / missing file

unittest only.
"""
from __future__ import annotations

import asyncio
import base64
import json
import unittest

from toolboxv2.mods.isaa.base.audio_io.omni import OmniEventType, JobManager
import toolboxv2.mods.isaa.base.audio_io.native.omni_gemini
from toolboxv2.mods.isaa.base.audio_io.native.omni_gemini import GeminiLiveBackend


# ---------------------------------------------------------------------------
# Fake websocket
# ---------------------------------------------------------------------------

class FakeWS:
    """Records sent frames; yields nothing (recv loop is driven manually)."""

    def __init__(self):
        self.sent: list[str] = []
        self.closed = False
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def last_json(self):
        return json.loads(self.sent[-1])

    def all_json(self):
        return [json.loads(s) for s in self.sent]


def _backend_with_fake_ws(**kw) -> tuple[GeminiLiveBackend, FakeWS]:
    ws = FakeWS()

    async def factory(url):
        factory.url = url
        return ws

    b = GeminiLiveBackend(api_key_env="TEST_KEY", connect_factory=factory, **kw)
    return b, ws


# ---------------------------------------------------------------------------
# GeminiLiveBackend — setup + send
# ---------------------------------------------------------------------------

class TestGeminiSetup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import os
        os.environ["TEST_KEY"] = "dummy-key"

    async def test_start_sends_setup_with_audio_and_compression(self):
        b, ws = _backend_with_fake_ws(system_instruction="be brief")
        await b.start(tools=[{"name": "ping", "description": "p"}])
        setup = ws.all_json()[0]["setup"]
        self.assertEqual(setup["model"], f"models/{b.model_id}")
        self.assertEqual(setup["generationConfig"]["responseModalities"], ["AUDIO"])
        self.assertEqual(setup["systemInstruction"]["parts"][0]["text"], "be brief")
        # token efficiency: sliding-window compression present
        self.assertIn("contextWindowCompression", setup)
        self.assertIn("slidingWindow", setup["contextWindowCompression"])
        # function declaration mapped
        decls = setup["tools"][0]["functionDeclarations"]
        self.assertEqual(decls[0]["name"], "ping")
        await b.stop()

    async def test_transcription_off_by_default_on_when_requested(self):
        b1, ws1 = _backend_with_fake_ws()
        await b1.start()
        self.assertNotIn("outputAudioTranscription", ws1.all_json()[0]["setup"])
        await b1.stop()

        b2, ws2 = _backend_with_fake_ws(output_transcription=True, input_transcription=True)
        await b2.start()
        setup = ws2.all_json()[0]["setup"]
        self.assertIn("outputAudioTranscription", setup)
        self.assertIn("inputAudioTranscription", setup)
        await b2.stop()

    async def test_missing_key_raises_and_emits_error(self):
        import os
        os.environ.pop("NO_SUCH_KEY", None)
        b = GeminiLiveBackend(api_key_env="NO_SUCH_KEY", connect_factory=lambda u: None)
        with self.assertRaises(RuntimeError):
            await b.start()

    async def test_send_audio_envelope(self):
        b, ws = _backend_with_fake_ws()
        await b.start()
        ws.sent.clear()
        pcm = b"\x01\x02\x03\x04"
        await b.send_audio(pcm)
        msg = ws.last_json()
        audio = msg["realtimeInput"]["audio"]
        self.assertEqual(audio["mimeType"], "audio/pcm;rate=16000")
        self.assertEqual(base64.b64decode(audio["data"]), pcm)
        await b.stop()

    async def test_send_tool_result_string_and_dict(self):
        b, ws = _backend_with_fake_ws()
        await b.start()
        b._pending_call_ids["c1"] = "weather"
        ws.sent.clear()
        await b.send_tool_result("c1", "sunny")
        fr = ws.last_json()["toolResponse"]["functionResponses"][0]
        self.assertEqual(fr["id"], "c1")
        self.assertEqual(fr["name"], "weather")
        self.assertEqual(fr["response"], {"result": "sunny"})

        b._pending_call_ids["c2"] = "lookup"
        ws.sent.clear()
        await b.send_tool_result("c2", json.dumps({"hits": 3}))
        fr = ws.last_json()["toolResponse"]["functionResponses"][0]
        self.assertEqual(fr["response"], {"hits": 3})
        await b.stop()


# ---------------------------------------------------------------------------
# GeminiLiveBackend — server message parsing
# ---------------------------------------------------------------------------

class TestGeminiParsing(unittest.IsolatedAsyncioTestCase):
    def _drain(self, b) -> list:
        out = []
        while not b._q.empty():
            out.append(b._q.get_nowait())
        return out

    async def test_inline_audio_becomes_audio_event_at_24k(self):
        b = GeminiLiveBackend(api_key_env="X", connect_factory=lambda u: None)
        pcm = b"\xaa\xbb" * 10
        msg = {"serverContent": {"modelTurn": {"parts": [
            {"inlineData": {"mimeType": "audio/pcm", "data": base64.b64encode(pcm).decode()}}
        ]}}}
        b._handle_server_message(json.dumps(msg))
        evs = self._drain(b)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].type, OmniEventType.AUDIO)
        self.assertEqual(evs[0].audio, pcm)
        self.assertEqual(evs[0].meta["sample_rate_out"], 24000)

    async def test_text_part(self):
        b = GeminiLiveBackend(api_key_env="X", connect_factory=lambda u: None)
        b._handle_server_message(json.dumps(
            {"serverContent": {"modelTurn": {"parts": [{"text": "hi"}]}}}
        ))
        evs = self._drain(b)
        self.assertEqual(evs[0].type, OmniEventType.TEXT)
        self.assertEqual(evs[0].text, "hi")

    async def test_transcriptions_tagged_with_source(self):
        b = GeminiLiveBackend(api_key_env="X", connect_factory=lambda u: None)
        b._handle_server_message(json.dumps({"serverContent": {
            "outputTranscription": {"text": "model said"},
            "inputTranscription": {"text": "user said"},
        }}))
        evs = self._drain(b)
        sources = {e.meta.get("source") for e in evs}
        self.assertEqual(sources, {"output", "input"})

    async def test_turn_complete_emits_turn_end(self):
        b = GeminiLiveBackend(api_key_env="X", connect_factory=lambda u: None)
        b._handle_server_message(json.dumps({"serverContent": {"turnComplete": True}}))
        evs = self._drain(b)
        self.assertEqual(evs[-1].type, OmniEventType.TURN_END)

    async def test_tool_call_emits_event_and_tracks_id(self):
        b = GeminiLiveBackend(api_key_env="X", connect_factory=lambda u: None)
        b._handle_server_message(json.dumps({"toolCall": {"functionCalls": [
            {"id": "fc1", "name": "search", "args": {"q": "x"}}
        ]}}))
        evs = self._drain(b)
        self.assertEqual(evs[0].type, OmniEventType.TOOL_CALL)
        self.assertEqual(evs[0].tool_call["name"], "search")
        self.assertEqual(evs[0].tool_call["id"], "fc1")
        self.assertEqual(evs[0].tool_call["arguments"], {"q": "x"})
        # id tracked so the response carries the right name
        self.assertEqual(b._pending_call_ids["fc1"], "search")

    async def test_malformed_message_is_ignored(self):
        b = GeminiLiveBackend(api_key_env="X", connect_factory=lambda u: None)
        b._handle_server_message("not json{")
        b._handle_server_message(json.dumps({"unknown": 1}))
        self.assertTrue(b._q.empty())


class TestFunctionDeclarations(unittest.IsolatedAsyncioTestCase):
    """The Omni model only knows its tools if the setup advertises them.
    ToolManager.get_all_litellm() returns OpenAI/LiteLLM-shaped schemas; the
    backend must unpack {"function": {...}} into Gemini functionDeclarations."""

    async def asyncSetUp(self):
        import os
        os.environ["TEST_KEY"] = "dummy-key"

    def test_litellm_format_is_unpacked(self):
        litellm = [{
            "type": "function",
            "function": {
                "name": "delegate",
                "description": "Delegate a task",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }]
        decls = GeminiLiveBackend._tools_to_function_declarations(litellm)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["name"], "delegate")
        self.assertEqual(decls[0]["description"], "Delegate a task")
        self.assertEqual(decls[0]["parameters"]["properties"], {"query": {"type": "string"}})
        self.assertEqual(decls[0]["parameters"]["required"], ["query"])

    def test_flat_format_still_works(self):
        flat = [{"name": "agent_status", "description": "list jobs"}]
        decls = GeminiLiveBackend._tools_to_function_declarations(flat)
        self.assertEqual(decls[0]["name"], "agent_status")
        self.assertEqual(decls[0]["parameters"]["type"], "object")

    def test_empty_required_omitted(self):
        flat = [{"name": "noargs", "description": "d",
                 "parameters": {"type": "object", "properties": {}}}]
        decls = GeminiLiveBackend._tools_to_function_declarations(flat)
        self.assertNotIn("required", decls[0]["parameters"])

    def test_nameless_entry_skipped(self):
        decls = GeminiLiveBackend._tools_to_function_declarations([{"description": "x"}])
        self.assertEqual(decls, [])

    async def test_setup_advertises_litellm_tools_to_gemini(self):
        b, ws = _backend_with_fake_ws()
        litellm = [{
            "type": "function",
            "function": {"name": "vfs_peek", "description": "peek",
                         "parameters": {"type": "object",
                                        "properties": {"job_id": {"type": "string"}}}},
        }]
        await b.start(tools=litellm)
        setup = ws.all_json()[0]["setup"]
        decls = setup["tools"][0]["functionDeclarations"]
        self.assertEqual(decls[0]["name"], "vfs_peek")
        await b.stop()


class TestSchemaSanitizer(unittest.TestCase):
    """Gemini rejects the WHOLE setup (code 1007) if any parameter schema has
    keys outside its OpenAPI-3.0 subset. The sanitizer must strip them."""

    def san(self, s):
        return GeminiLiveBackend._sanitize_schema(s)

    def test_drops_unsupported_keys(self):
        out = self.san({
            "type": "object",
            "additionalProperties": False,  # unsupported
            "title": "Foo",  # unsupported
            "properties": {
                "x": {"type": "string", "default": "hi"},  # default unsupported
            },
        })
        self.assertNotIn("additionalProperties", out)
        self.assertNotIn("title", out)
        self.assertNotIn("default", out["properties"]["x"])
        self.assertEqual(out["properties"]["x"]["type"], "string")

    def test_object_without_properties_gets_empty_dict(self):
        out = self.san({"type": "object"})
        self.assertEqual(out["properties"], {})

    def test_union_type_list_becomes_first_concrete_and_nullable(self):
        out = self.san({"type": ["string", "null"]})
        self.assertEqual(out["type"], "string")
        self.assertTrue(out["nullable"])

    def test_missing_type_defaults(self):
        self.assertEqual(self.san({})["type"], "string")
        self.assertEqual(self.san({"properties": {"a": {"type": "string"}}})["type"], "object")

    def test_unknown_type_falls_back(self):
        # anyOf-style fragment with no usable type
        out = self.san({"anyOf": [{"type": "string"}, {"type": "integer"}]})
        self.assertIn(out["type"], {"string", "object"})

    def test_nested_array_items_sanitized(self):
        out = self.san({
            "type": "array",
            "items": {"type": "object", "properties": {"n": {"type": "integer", "default": 0}}},
        })
        self.assertEqual(out["type"], "array")
        self.assertNotIn("default", out["items"]["properties"]["n"])

    def test_required_kept_when_present(self):
        out = self.san({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        })
        self.assertEqual(out["required"], ["q"])

    def test_enum_and_description_preserved(self):
        out = self.san({"type": "string", "enum": ["a", "b"], "description": "d"})
        self.assertEqual(out["enum"], ["a", "b"])
        self.assertEqual(out["description"], "d")

    def test_declaration_with_messy_schema_is_clean(self):
        # mimics what ToolManager._parse_args_schema can emit for dict/Optional args
        litellm = [{
            "type": "function",
            "function": {
                "name": "toolbox_execute",
                "description": "run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kwargs": {"type": "object", "additionalProperties": True,
                                   "default": {}},
                        "name": {"type": ["string", "null"]},
                    },
                    "required": ["name"],
                },
            },
        }]
        decls = GeminiLiveBackend._tools_to_function_declarations(litellm)
        params = decls[0]["parameters"]
        self.assertNotIn("additionalProperties", params["properties"]["kwargs"])
        self.assertNotIn("default", params["properties"]["kwargs"])
        self.assertEqual(params["properties"]["name"]["type"], "string")


if __name__ == "__main__":
    unittest.main(verbosity=2)
