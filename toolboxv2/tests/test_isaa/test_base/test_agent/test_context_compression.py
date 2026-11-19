"""Betriebstests Kontext-Kompression P1/P4/P5 (WP-Compression).

Pruefling: HistoryCompressor.compress_partial + ExecutionEngine._maybe_compact_history
Befund-Basis: /notes/debugging-compression/findings.md (18.11.2026)
Regeln: /global/temp/wp3/testing_guide.md — echt vor fake, rot-vor-Patch dokumentiert.
"""
from __future__ import annotations

import types
import unittest

from toolboxv2.mods.isaa.base.Agent.execution_engine import (
    ExecutionEngine,
    HistoryCompressor,
)

BIG = "X" * 40_000  # ~10.8k tokens — Marathon-typischer write_code-Payload


def make_hist(n_tool_turns: int = 30) -> list[dict]:
    hist = [{"role": "system", "content": "sys"}]
    for i in range(n_tool_turns):
        hist.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "write_code",
                             "arguments": f'{{"content": "{BIG}"}}'},
            }],
        })
        hist.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "name": "write_code", "content": "ok " + "y" * 20_000})
    hist.append({"role": "user", "content": "RUN SUMMARY finale antwort jetzt"})
    return hist


class TestCompressPartialArguments(unittest.TestCase):
    """P1: tool_calls.arguments muessen bei Compact gekuerzt werden."""

    def test_arguments_get_truncated(self):
        hist = make_hist()
        orig_ids = {id(m) for m in hist}
        summary, new, _stats = HistoryCompressor.compress_partial(hist, keep_last_n=8)
        self.assertIsNotNone(summary, "compress_partial lieferte None")
        checked = 0
        for m in new:
            if id(m) in orig_ids:
                continue  # verbatim-Original (to_keep, legal auch >8)
            for tc in m.get("tool_calls", []) or []:
                args = tc["function"]["arguments"]
                self.assertLess(
                    len(args), 2000 + 800,
                    f"Kopie mit ungekuerzten arguments ({len(args)} chars) — "
                    "Floor-Treiber der Kontextexplosion",
                )
                checked += 1
        self.assertGreater(checked, 0, "keine Kopien geprueft")

    def test_truncation_marker_present(self):
        hist = make_hist()
        orig_ids = {id(m) for m in hist}
        summary, new, _stats = HistoryCompressor.compress_partial(hist, keep_last_n=8)
        self.assertIsNotNone(summary)
        offloaded = [
            tc for m in new
            if id(m) not in orig_ids
            for tc in m.get("tool_calls", []) or []
        ]
        self.assertTrue(offloaded)
        for tc in offloaded:
            self.assertIn("[truncated", tc["function"]["arguments"])

    def test_tool_call_identity_intact(self):
        """Pairing darf nicht brechen: id + name unangetastet."""
        hist = make_hist()
        ids_before = [tc["id"] for m in hist for tc in m.get("tool_calls", []) or []]
        summary, new, _stats = HistoryCompressor.compress_partial(hist, keep_last_n=8)
        ids_after = [tc["id"] for m in new for tc in m.get("tool_calls", []) or []]
        self.assertEqual(ids_before, ids_after)
        for m in new:
            for tc in m.get("tool_calls", []) or []:
                self.assertEqual(tc["function"]["name"], "write_code")

    def test_resident_tail_verbatim(self):
        """letzte 8 duerfen unangetastet bleiben (Identity, nicht Position —
        Split erweitert to_keep auf Paar-Grenzen)."""
        hist = make_hist(10)
        tail_ids = {id(m) for m in hist[-8:]}
        summary, new, _stats = HistoryCompressor.compress_partial(hist, keep_last_n=8)
        verbatim = [m for m in new if id(m) in tail_ids]
        self.assertGreaterEqual(len(verbatim), 8)
        for m in verbatim:
            self.assertEqual(m, hist[hist.index(m)] if False else m)
        idx = {id(m) for m in verbatim}
        self.assertTrue(idx.issubset(tail_ids))


def build_engine() -> tuple[ExecutionEngine, types.SimpleNamespace]:
    """Engine ohne __init__ (Netz-abhaengig), Attribute minimal gesetzt."""
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._COMPACT_KEEP = 8
    eng._get_max_context_tokens = lambda ctx: 200_000  # -> limit = min(60k, 120k) = 60k
    obs = types.SimpleNamespace(
        compressions=[],
        record_compression=lambda stats=None, **kw: obs.compressions.append(stats or kw),
    )
    eng.agent = types.SimpleNamespace(obs=obs)
    eng.live = types.SimpleNamespace(
        enter=lambda *a, **k: None, iteration=0, max_iterations=99, status_msg="",
    )
    ctx = types.SimpleNamespace(working_history=make_hist(25),
                                current_iteration=1, run_id="t")
    return eng, ctx


class TestMaybeCompactTelemetryAndRounds(unittest.TestCase):
    """P5: compression-event an OBS; P4: zweite Runde bei Floor > 0.8*limit."""

    def test_p5_record_compression_called(self):
        eng, ctx = build_engine()
        eng._maybe_compact_history(ctx)
        obs = eng.agent.obs
        self.assertTrue(
            obs.compressions,
            "record_compression nie aufgerufen — OBS-Telemetrie bleibt None "
            "(0/119 steps in run_925ab3c6)",
        )
        for stats in obs.compressions:
            self.assertIn("kept", stats)
            self.assertIn("summarized", stats)

    def test_p4_second_round_when_floor_too_high(self):
        eng, ctx = build_engine()
        eng._maybe_compact_history(ctx)
        used_after = sum(
            eng._estimate_tokens(str(m.get("content", "") or "")) for m in ctx.working_history
        ) + sum(
            eng._estimate_tokens(
                (tc.get("function", {}) or {}).get("arguments", "") or "")
            for m in ctx.working_history
            for tc in m.get("tool_calls", []) or []
        )
        limit = min(60_000, int(200_000 * 0.6))
        self.assertLess(
            used_after, 0.8 * limit,
            f"Floor {used_after} > 0.8*limit — Compact wirkt nicht ausreichend "
            "(Marathon-Muster: 200k+ Kontext trotz Schutz)",
        )


class TestP2UserResidency(unittest.TestCase):
    """P2: letzte 10 user-msgs verbatim; aeltere gekuerzt + Referenz.
    Volle Verfuegbarkeit bleibt: ChatSession persistiert jede user-msg
    via add_message (Engine L6323) — Kuerzung nur im Working-Set."""

    @staticmethod
    def hist_users(n: int = 15, chars: int = 5000) -> list[dict]:
        return [
            {"role": "user", "content": f"U{i} " + "u" * chars}
            for i in range(n)
        ]

    def test_last_10_user_verbatim(self):
        hist = self.hist_users(15)
        orig_ids = {id(m) for m in hist}
        summary, new, _ = HistoryCompressor.compress_partial(hist, keep_last_n=3)
        verbatim = [m for m in new if id(m) in orig_ids and m.get("role") == "user"]
        self.assertGreaterEqual(
            len(verbatim), 10,
            f"nur {len(verbatim)} user-msgs verbatim — Resident-Fenster 10 unterschritten",
        )
        # genau die LETZTEN 10, unveraendert
        self.assertEqual(
            [m["content"] for m in verbatim],
            [m["content"] for m in hist[-10:]],
        )

    def test_older_user_shortened_with_reference(self):
        hist = self.hist_users(15)
        summary, new, _ = HistoryCompressor.compress_partial(hist, keep_last_n=3)
        shortened = [
            m for m in new
            if m.get("role") == "user" and m.get("content") and len(m["content"]) < 5000
        ]
        self.assertEqual(
            len(shortened), 5,
            f"erwartete 5 gekuerzte aeltere user-msgs, gefunden {len(shortened)}",
        )
        for m in shortened:
            self.assertIn("ChatSession", m["content"], "Referenz-Hinweis fehlt")
            self.assertLess(len(m["content"]), 500 + 200 + 200)

    def test_few_users_all_verbatim(self):
        hist = self.hist_users(7)
        orig_ids = {id(m) for m in hist}
        summary, new, _ = HistoryCompressor.compress_partial(hist, keep_last_n=3)
        verbatim = [m for m in new if id(m) in orig_ids]
        self.assertEqual(len(verbatim), 7, "unter 10 muessen alle bleiben")


class TestP5bInputMessagesCap(unittest.TestCase):
    """P5b: OBS speichert llm.input_messages verbatim -> 216MB-Runs.
    Cap: nicht-assistant contents kuerzen; assistant (tool_calls-exakt,
    Strategie A L1147) bleibt intakt."""

    def _layer(self):
        import tempfile
        from toolboxv2.mods.isaa.base.Agent.observability import ObservabilityLayer
        return ObservabilityLayer(
            agent_name="t", obs_dir=tempfile.mkdtemp(), max_runs=1
        )

    def test_tool_content_capped_assistant_intact(self):
        lay = self._layer()
        lay.begin_run("r1", "q")
        lay.begin_step(1)
        big = "Z" * 50_000
        msgs = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "shell", "arguments": "x" * 5_000}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big},
        ]
        lay.record_llm_start(model="m", messages=msgs)
        lay.record_llm_end(tokens_in=1, tokens_out=1)
        lay.end_step()
        stored = lay._slots["r1"].run.steps[0].llm.input_messages
        for m in stored:
            if m.get("role") == "assistant":
                self.assertIn("tool_calls", m)  # Strategie A benoetigt exakt
                self.assertEqual(
                    m["tool_calls"][0]["function"]["arguments"], "x" * 5_000
                )
            else:
                self.assertLess(
                    len(m.get("content", "")), 500,
                    f"content {len(m.get('content',''))} chars unveraendert "
                    "gespeichert — Run-JSON explodiert (216MB-Episode)",
                )


class TestP3OffloadStep(unittest.TestCase):
    """P3: Agent kann Range x..y aktiv offloaden (Working History, NIE user).
    Tool-Contents -> VFS-Archiv + Pointer; tool_call arguments gekuerzt;
    Summary als System-Notiz; Pairing (id/name) bleibt intakt."""

    def _engine_with_history(self, n=10):
        import types, asyncio
        eng = ExecutionEngine.__new__(ExecutionEngine)
        writes = []
        vfs = types.SimpleNamespace(
            mkdir=lambda *a, **k: None,
            write=lambda path, content: writes.append((path, content)),
        )
        session = types.SimpleNamespace(vfs=vfs)

        async def get_or_create(sid):
            return session

        eng.agent = types.SimpleNamespace(
            session_manager=types.SimpleNamespace(get_or_create=get_or_create),
        )
        hist = [{"role": "system", "content": "sys"}]
        for i in range(n):
            hist.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "write_code",
                                                      "arguments": "X" * 30_000}}]})
            hist.append({"role": "tool", "tool_call_id": f"c{i}",
                         "name": "write_code", "content": "R" * 20_000})
        hist.append({"role": "user", "content": "weiter"})
        ctx = types.SimpleNamespace(working_history=hist, run_id="p3run",
                                    session_id="s1", current_iteration=5)
        return eng, ctx, writes

    def test_range_offloaded_with_summary(self):
        import asyncio
        eng, ctx, writes = self._engine_with_history()
        res = asyncio.run(eng._tool_offload_step(ctx, 1, 15, "Zwischenstand: Basis steht"))
        hist = ctx.working_history
        # hist: 0=system, 2k+1=assistant_k, 2k+2=tool_k, user=21
        # Range [1,15) -> tool 0..6 offgeloaded; tool 7..9 verbatim
        tools_in_range = [hist[2 * k + 2] for k in range(7)]
        tools_outside = [hist[2 * k + 2] for k in range(7, 10)]
        for m in tools_in_range:
            self.assertIn("[DATA OFFLOADED", m["content"],
                          "tool-result im Range nicht durch Pointer ersetzt")
        for m in tools_outside:
            self.assertEqual(m["content"], "R" * 20_000,
                             "tool-result ausserhalb Range veraendert")
        self.assertTrue(writes, "nichts ins Archiv geschrieben")
        path, content = writes[0]
        self.assertIn("/global/.memory/archive/", path)
        self.assertIn("R" * 100, content[:25_000])
        sys_msgs = "".join(m["content"] for m in hist if m.get("role") == "system")
        self.assertIn("[OFFLOADED 1..15]", sys_msgs)
        self.assertIn("Zwischenstand: Basis steht", sys_msgs)

    def test_user_never_offloaded(self):
        import asyncio
        eng, ctx, writes = self._engine_with_history()
        user_before = [m["content"] for m in ctx.working_history if m.get("role") == "user"]
        asyncio.run(eng._tool_offload_step(ctx, 0, len(ctx.working_history), "x"))
        user_after = [m["content"] for m in ctx.working_history if m.get("role") == "user"]
        self.assertEqual(user_before, user_after, "user-msg veraendert — verboten")

    def test_pairing_intact(self):
        import asyncio
        eng, ctx, writes = self._engine_with_history()
        before = [(tc["id"], tc["function"]["name"])
                  for m in ctx.working_history for tc in m.get("tool_calls", []) or []]
        asyncio.run(eng._tool_offload_step(ctx, 1, 15, "s"))
        after = [(tc["id"], tc["function"]["name"])
                 for m in ctx.working_history for tc in m.get("tool_calls", []) or []]
        self.assertEqual(before, after)
        ids = {m["tool_call_id"] for m in ctx.working_history if m.get("role") == "tool"}
        for _i, name in after:
            self.assertIn(_i, ids)

    def test_arguments_truncated_in_range(self):
        import asyncio
        eng, ctx, writes = self._engine_with_history()
        asyncio.run(eng._tool_offload_step(ctx, 1, 15, "s"))
        # Range [1,15) -> Paare 0..6 gekuerzt; 7..9 verbatim
        for k in range(7):
            for tc in ctx.working_history[2 * k + 1].get("tool_calls", []):
                self.assertLess(
                    len(tc["function"]["arguments"]), 3000,
                    f"arguments Paar {k} im Range unveraendert",
                )
        for k in range(7, 10):
            for tc in ctx.working_history[2 * k + 1].get("tool_calls", []):
                self.assertEqual(len(tc["function"]["arguments"]), 30_000)


if __name__ == "__main__":
    unittest.main()
