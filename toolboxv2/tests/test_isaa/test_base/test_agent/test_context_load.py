"""
Abnahmetests: Kontextlast der ExecutionEngine (``_calculate_context_load``).

Anforderungen
  1. Diff-basiert: Die Kosten einer Lastabfrage haengen nicht von der Groesse der
     gesamten History ab. Bisher hat jeder Stream-Chunk (``enrich`` ->
     ``tokens_used``) die komplette History per ``str(messages)`` neu serialisiert.
  2. Echte Werte: Meldet der Provider ``usage``, ist die Last
     ``echte Prompt-Tokens des letzten Aufrufs + Schaetzung nur fuer das, was danach
     dazugekommen oder geaendert ist``. Bei Anthropic zaehlen Cache-Tokens mit,
     weil ``input_tokens`` dort den Cache nicht enthaelt.

Aufbau (Testleitfaden, Regel 1-3)
  Echter FlowAgent, echte ExecutionEngine, echter CompletionRouter und echte
  Provider-Adapter (OpenAICompatAdapter, AnthropicAdapter). Ersetzt wird nur die
  HTTP-Session des Routers (Prozessgrenze: Netz zum LLM-Anbieter). Sie liefert die
  echten Antwortformen der Anbieter (OpenAI JSON/SSE, Anthropic SSE).

Run:
    python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_context_load.py -v
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

TEST_TIMEOUT_S = 120


# =============================================================================
# Netzgrenze: HTTP-Session mit echten Anbieter-Antwortformen
# =============================================================================


class _Content:
    def __init__(self, lines):
        self._lines = lines

    async def iter_any(self):
        for ln in self._lines:
            yield (ln + "\n").encode()


class _Resp:
    def __init__(self, body=None, sse=None):
        self.status = 200
        self._body = body
        self.content = _Content(sse or [])

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeProviderHttp:
    """Ersetzt nur ``aiohttp.ClientSession.post`` des Routers.

    Aufrufe der Agent-Schleife (erkennbar an ``tools`` im Payload) bekommen der Reihe
    nach die vorbereiteten Antworten; Nebenaufrufe (z. B. Zusammenfassungen nach dem
    Lauf) eine kurze Textantwort ohne usage. ``sent`` haelt jeden Payload fest.
    """

    def __init__(self, scripted, side=None):
        self.scripted = list(scripted)
        self.side = side
        self.sent = []

    def post(self, url, json=None, headers=None):
        self.sent.append(json)
        streaming = bool(json.get("stream"))
        if json.get("tools") and self.scripted:
            kind, data = self.scripted.pop(0)(streaming)
        else:
            kind, data = (self.side or _oai_side)(streaming)
        return _Resp(body=data) if kind == "json" else _Resp(sse=data)

    @property
    def loop_payloads(self):
        return [p for p in self.sent if p.get("tools")]


class _SessionHolder:
    """Form von ``toolboxv2.utils.system.session.Session``: ``.session`` ist die HTTP-Session."""

    def __init__(self, http):
        self.session = http


def _oai_side(streaming):
    if streaming:
        return oai_sse(content="ok", usage=None)
    return oai_json(content="ok", usage=None)


def oai_sse(tool_calls=None, content=None, usage=None, n_chunks=1):
    """OpenAI Chat-Completions-Stream (SSE). ``usage=None``: Anbieter liefert keine usage."""
    lines = []
    if content:
        step = max(1, len(content) // n_chunks)
        for i in range(0, len(content), step):
            lines.append("data: " + json.dumps(
                {"choices": [{"index": 0, "delta": {"content": content[i:i + step]}}]}))
    for i, tc in enumerate(tool_calls or []):
        lines.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}]}}]}))
    lines.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {},
                                                     "finish_reason": "tool_calls" if tool_calls else "stop"}]}))
    if usage is not None:
        prompt, completion = usage
        lines.append("data: " + json.dumps({"choices": [], "usage": {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}}))
    lines.append("data: [DONE]")
    return "sse", lines


def oai_json(tool_calls=None, content=None, usage=None):
    """OpenAI Chat-Completions-Antwort (JSON)."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                              "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                             for tc in tool_calls]
    body = {"id": "chatcmpl-1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}]}
    if usage is not None:
        prompt, completion = usage
        body["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion,
                         "total_tokens": prompt + completion}
    return "json", body


def oai(tool_calls=None, content=None, usage=None, n_chunks=1):
    """Antwort im passenden Format (Stream oder nicht), wie der Aufrufer sie anfordert."""
    def _make(streaming):
        if streaming:
            return oai_sse(tool_calls, content, usage, n_chunks)
        return oai_json(tool_calls, content, usage)
    return _make


def anthropic_sse(tool_calls, input_tokens, cache_read, cache_creation, output=7):
    """Anthropic Messages-Stream (SSE), Feldnamen wie in der Anthropic-API."""
    ev = [{"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": [],
        "usage": {"input_tokens": input_tokens, "output_tokens": 1,
                  "cache_read_input_tokens": cache_read,
                  "cache_creation_input_tokens": cache_creation}}}]
    for i, tc in enumerate(tool_calls):
        ev.append({"type": "content_block_start", "index": i,
                   "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}}})
        ev.append({"type": "content_block_delta", "index": i,
                   "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc["args"])}})
        ev.append({"type": "content_block_stop", "index": i})
    ev.append({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
               "usage": {"output_tokens": output}})
    ev.append({"type": "message_stop"})
    lines = []
    for e in ev:
        lines.append(f"event: {e['type']}")
        lines.append("data: " + json.dumps(e))
        lines.append("")
    return "sse", lines


def _anthropic_side(streaming):
    ev = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
                                                "model": "m", "content": [], "usage": {}}},
          {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
          {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
          {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
          {"type": "message_stop"}]
    if streaming:
        return "sse", ["data: " + json.dumps(e) for e in ev]
    return "json", {"id": "m", "type": "message", "role": "assistant", "model": "m",
                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}


LIST_TOOLS = {"id": "call_1", "name": "list_tools", "args": {}}
FINAL = {"id": "call_2", "name": "final_answer", "args": {"answer": "fertig", "success": True}}


class CountingStr(str):
    """Testdaten, die zaehlen, wie oft sie serialisiert (repr) werden.
    Kein Eingriff in Produktcode: nur der Inhalt einer History-Nachricht."""

    calls = 0

    def __repr__(self):
        CountingStr.calls += 1
        return str.__repr__(self)


# =============================================================================
# Basis: isolierte Umgebung, echter Agent
# =============================================================================


class _ContextLoadBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        tmp = tempfile.mkdtemp(prefix="tb_ctxload_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        env = {
            "HOME": os.path.join(tmp, "home"),
            "USERPROFILE": os.path.join(tmp, "home"),
            "APPDATA": os.path.join(tmp, "appdata"),
            "LOCALAPPDATA": os.path.join(tmp, "local"),
            "MOCK_EXTERNAL_SERVICES": "true",
            "IS_OFFLINE_DB": "true",
        }
        if not os.environ.get("TB_TEST_DATA_LOCK"):
            env["TB_DATA_DIR"] = os.path.join(tmp, "data")
        for d in env.values():
            if d.startswith(tmp):
                os.makedirs(d, exist_ok=True)
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._agents = []

    async def asyncTearDown(self):
        for agent in self._agents:
            await agent.close()

    def make_agent(self, prefix, adapter, http):
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
        from toolboxv2.mods.isaa.base.Agent.types import AgentModelData

        amd = AgentModelData(
            name=f"ctxload_{prefix}",
            fast_llm_model=f"{prefix}/m",
            complex_llm_model=f"{prefix}/m",
            system_message="Du bist ein Testagent.",
            vfs_max_window_lines=100,
        )
        agent = FlowAgent(amd, auto_load_checkpoint=False)
        self._agents.append(agent)
        # Prozessgrenze Netz: echter Router + echter Adapter, nur die HTTP-Session ist ersetzt.
        agent.router._session = _SessionHolder(http)
        agent.router.register(prefix, adapter)
        return agent

    @staticmethod
    def resumed_ctx(history_msgs, max_iterations=4):
        """ExecutionContext wie nach einem pausierten Lauf (Eingang: Resume mit ctx)."""
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionContext

        ctx = ExecutionContext()
        ctx.working_history = [{"role": "system", "content": "Du bist ein Testagent."}]
        ctx.working_history.extend(history_msgs)
        ctx.working_history.append({"role": "user", "content": "weiter"})
        ctx.max_iterations = max_iterations
        return ctx

    async def run_stream(self, agent, ctx):
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        engine = ExecutionEngine(agent)
        gen, ctx = await engine.execute_stream("weiter", session_id="ctxload", ctx=ctx)

        async def _collect():
            return [c async for c in gen(ctx)]

        chunks = await asyncio.wait_for(_collect(), TEST_TIMEOUT_S)
        return engine, ctx, chunks


# =============================================================================
# 2. Echte Werte vom Provider
# =============================================================================


class TestContextLoadUsesProviderUsage(_ContextLoadBase):

    async def test_stream_live_bar_starts_from_provider_prompt_tokens(self):
        """a_stream (Eingang der Live-Bar): Nach Aufruf 1 meldet der Anbieter
        prompt_tokens=500000. tokens_used muss darauf aufbauen und nur die danach
        angehaengten Nachrichten schaetzen."""
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        http = FakeProviderHttp([
            oai(tool_calls=[LIST_TOOLS], usage=(500_000, 12)),
            oai(tool_calls=[FINAL], usage=(500_900, 9)),
        ])
        agent = self.make_agent("tstoai", OpenAICompatAdapter("http://provider.invalid/v1"), http)

        async def _collect():
            return [c async for c in agent.a_stream("Bitte Tools auflisten", session_id="ctxload_a")]

        chunks = await asyncio.wait_for(_collect(), TEST_TIMEOUT_S)

        after_call_1 = [c for c in chunks
                        if c.get("iter") == 1 and c.get("type") in ("tool_start", "tool_result")]
        self.assertTrue(after_call_1, f"keine Tool-Chunks in Iteration 1: {[c.get('type') for c in chunks]}")
        for c in after_call_1:
            self.assertGreaterEqual(c["tokens_used"], 500_000, c)
            self.assertLess(c["tokens_used"], 500_000 + 20_000, c)

    async def test_anthropic_cache_tokens_count_into_load(self):
        """Anthropic meldet input_tokens OHNE Cache. Die Kontextlast ist
        input + cache_read + cache_creation."""
        from toolboxv2.mods.isaa.base.llm_router.adapters.anthropic import AnthropicAdapter

        http = FakeProviderHttp([
            lambda streaming: anthropic_sse([LIST_TOOLS], input_tokens=2_000,
                                            cache_read=300_000, cache_creation=1_000),
            lambda streaming: anthropic_sse([FINAL], input_tokens=500,
                                            cache_read=303_000, cache_creation=0),
        ], side=_anthropic_side)
        agent = self.make_agent("tstanth", AnthropicAdapter(base_url="http://provider.invalid"), http)
        ctx = self.resumed_ctx([{"role": "user", "content": "Hallo"}, {"role": "assistant", "content": "Hi"}])

        _, _, chunks = await self.run_stream(agent, ctx)

        after_call_1 = [c for c in chunks
                        if c.get("iter") == 1 and c.get("type") in ("tool_start", "tool_result")]
        self.assertTrue(after_call_1, [c.get("type") for c in chunks])
        for c in after_call_1:
            self.assertGreaterEqual(c["tokens_used"], 303_000, c)
            self.assertLess(c["tokens_used"], 303_000 + 20_000, c)

    async def test_non_stream_run_context_overview_uses_usage(self):
        """execute() (ohne Stream) und /context (context_overview -> t_effective):
        nach dem Lauf baut die Last auf den echten prompt_tokens des letzten Aufrufs auf."""
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        http = FakeProviderHttp([
            oai(tool_calls=[LIST_TOOLS], usage=(250_000, 12)),
            oai(tool_calls=[FINAL], usage=(251_000, 9)),
        ])
        agent = self.make_agent("tstoaij", OpenAICompatAdapter("http://provider.invalid/v1"), http)
        ctx = self.resumed_ctx([{"role": "user", "content": "Hallo"}, {"role": "assistant", "content": "Hi"}])
        engine = agent._get_execution_engine()

        _, ctx = await asyncio.wait_for(
            engine.execute("weiter", session_id="ctxload_j", ctx=ctx, get_ctx=True), TEST_TIMEOUT_S)

        overview = await asyncio.wait_for(
            agent.context_overview(session_id="ctxload_j", execution_id=ctx.run_id, print_visual=False),
            TEST_TIMEOUT_S)
        self.assertGreaterEqual(overview["t_effective"], 251_000, overview)
        self.assertLess(overview["t_effective"], 251_000 + 20_000, overview)

    async def test_without_usage_load_is_estimate_of_sent_payload(self):
        """Anbieter ohne usage: die Last ist die Schaetzung genau dessen, was gesendet
        wurde (Nachrichten + Tools, chars/3.7) - wie bisher, nur ohne Vollserialisierung."""
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        http = FakeProviderHttp([
            oai(content="Zwischenstand " * 40, tool_calls=[FINAL], usage=None, n_chunks=8),
        ])
        agent = self.make_agent("tstoain", OpenAICompatAdapter("http://provider.invalid/v1"), http)
        ctx = self.resumed_ctx([
            {"role": "user", "content": "Frage " * 400},
            {"role": "assistant", "content": "Antwort " * 400},
        ])

        _, _, chunks = await self.run_stream(agent, ctx)

        payload = http.loop_payloads[0]
        expected = (len(str(payload["messages"])) + len(str(payload["tools"]))) / 3.7
        streamed = [c for c in chunks if c.get("iter") == 1 and c.get("type") == "content"]
        self.assertTrue(streamed, [c.get("type") for c in chunks])
        for c in streamed:
            self.assertAlmostEqual(c["tokens_used"], expected, delta=expected * 0.03)

    async def test_rewritten_history_after_usage_lowers_load(self):
        """Nach einer Antwort mit usage: Wird History umgeschrieben (Offload/Compaction
        wie _retroactive_offload / _maybe_compact_history), sinkt die Last. Die echten
        Tokens des entfernten Teils werden anteilig abgezogen, nicht stehen gelassen."""
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        big = [{"role": "user", "content": f"Dokument {i}: " + ("lorem ipsum " * 2_000)} for i in range(10)]
        http = FakeProviderHttp([oai(tool_calls=[FINAL], usage=(80_000, 9))])
        agent = self.make_agent("tstoair", OpenAICompatAdapter("http://provider.invalid/v1"), http)
        ctx = self.resumed_ctx(big)

        engine, ctx, _ = await self.run_stream(agent, ctx)

        load_full = engine._calculate_context_load(ctx)
        self.assertGreaterEqual(load_full, 80_000)
        self.assertLess(load_full, 80_000 + 5_000)

        # In-place-Offload einer grossen Nachricht (so schreibt _retroactive_offload).
        for msg in ctx.working_history:
            if str(msg.get("content", "")).startswith("Dokument 0:"):
                msg["content"] = "[OFFLOADED to /.overflow/x.txt]"
        load_offloaded = engine._calculate_context_load(ctx)
        self.assertLess(load_offloaded, load_full - 4_000)
        self.assertGreater(load_offloaded, load_full - 12_000)

        # Compaction: alle grossen Nachrichten weg (atomarer Tausch der Liste).
        ctx.working_history = [m for m in ctx.working_history
                               if not str(m.get("content", "")).startswith("Dokument")
                               and not str(m.get("content", "")).startswith("[OFFLOADED")]
        load_compacted = engine._calculate_context_load(ctx)
        self.assertLess(load_compacted, 40_000)


# =============================================================================
# 1. Diff-basiert: Kosten unabhaengig von der Zahl der Abfragen
# =============================================================================


class TestContextLoadIsIncremental(_ContextLoadBase):

    async def _reprs_per_history_msg(self, n_chunks):
        from toolboxv2.mods.isaa.base.llm_router.adapters.openai_compat import OpenAICompatAdapter

        http = FakeProviderHttp([oai(content="x" * 600, tool_calls=[FINAL], usage=(1_000, 5),
                                     n_chunks=n_chunks)])
        agent = self.make_agent(f"tstperf{n_chunks}", OpenAICompatAdapter("http://provider.invalid/v1"), http)
        n_big = 40
        history = []
        for _ in range(n_big):
            history.append({"role": "user", "content": CountingStr("u" * 20_000)})
            history.append({"role": "assistant", "content": "ok"})
        ctx = self.resumed_ctx(history)
        CountingStr.calls = 0
        _, _, chunks = await self.run_stream(agent, ctx)
        content_chunks = [c for c in chunks if c.get("type") == "content"]
        return CountingStr.calls / n_big, len(content_chunks)

    async def test_stream_chunks_do_not_reserialize_history(self):
        """Je Stream-Chunk wird tokens_used gemeldet. Die Zahl der Serialisierungen
        einer alten History-Nachricht darf dabei nicht mit der Chunk-Zahl wachsen."""
        few, n_few = await self._reprs_per_history_msg(n_chunks=3)
        many, n_many = await self._reprs_per_history_msg(n_chunks=60)
        self.assertGreaterEqual(n_many - n_few, 50, (n_few, n_many))
        # Je LLM-Aufruf serialisiert auch der Router (Token-Vorschaetzung) die Nachrichten,
        # und ein Narrator-Nebenaufruf kann bei laengerer Ausgabe dazukommen. Beides ist
        # pro Aufruf, nicht pro Chunk. Pro Chunk (alt: +1 je Chunk und Nachricht) waeren
        # es hier >= 50 zusaetzliche Serialisierungen je Nachricht.
        self.assertLess(many - few, 1.0,
                        f"repr je Nachricht: {few} bei {n_few} Chunks, {many} bei {n_many} Chunks")


if __name__ == "__main__":
    unittest.main()
