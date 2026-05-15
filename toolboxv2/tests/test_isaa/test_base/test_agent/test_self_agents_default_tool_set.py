"""
Live Unit Test: Self-Agent Tool Inventory & Per-Tool Health Check

Lädt den echten self-Agent und testet jedes Tool einzeln via subTest.
Gibt klaren Report: welche Tools existieren, welche haben keine test_inputs,
wie schneiden sie ab.

Kein LLM-Call in dieser Datei.

Run:
    python -m unittest test_self_agent_live_tools -v
"""

import asyncio
import time
import unittest
from dataclasses import dataclass, field


# =============================================================================
# Report-Datenstruktur
# =============================================================================

@dataclass
class ToolReport:
    name: str
    source: str                        # local | mcp | a2a
    has_test_inputs: bool
    guaranteed: bool
    has_function: bool
    args_schema: str
    description_preview: str
    status: str = "PENDING"            # HEALTHY | DEGRADED | FAILED | SKIPPED | GUARANTEED | NO_FUNCTION
    error: str | None = None
    execution_time_ms: float = 0.0
    result_preview: str | None = None
    contract_violations: list[str] = field(default_factory=list)


# =============================================================================
# Loader (einmalig, geteilt zwischen Tests)
# =============================================================================

_agent_cache = None
_load_error = None

async def _load_agent():
    global _agent_cache, _load_error
    if _agent_cache is not None:
        return _agent_cache, None
    if _load_error is not None:
        return None, _load_error
    try:
        from toolboxv2 import get_app
        app = get_app()
        isaa = app.get_mod("isaa")
        _agent_cache = await isaa.get_agent("self")
        return _agent_cache, None
    except Exception as e:
        _load_error = str(e)
        return None, _load_error


# =============================================================================
# Helper: einzelnes Tool testen (kein LLM)
# =============================================================================

async def _run_single_tool_health(entry) -> ToolReport:
    from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolHealthResult

    report = ToolReport(
        name=entry.name,
        source=entry.source,
        has_test_inputs=bool(entry.live_test_inputs),
        guaranteed=entry.flags.get("guaranteed_healthy", False),
        has_function=entry.function is not None,
        args_schema=entry.args_schema,
        description_preview=entry.description[:80],
    )

    # GUARANTEED → kein execution nötig
    if report.guaranteed:
        report.status = "GUARANTEED"
        return report

    # MCP/A2A oder local ohne function
    if not report.has_function:
        report.status = "NO_FUNCTION"
        report.error = f"{entry.source} tool hat keine lokale function"
        return report

    # Keine Test-Inputs → SKIPPED
    if not report.has_test_inputs:
        report.status = "SKIPPED"
        report.error = "Keine live_test_inputs — füge test_inputs hinzu oder setze guaranteed_healthy=True"
        return report

    # Live execution
    test_input = entry.live_test_inputs[0]
    start = time.time()

    try:
        if asyncio.iscoroutinefunction(entry.function):
            result = await entry.function(**test_input)
        else:
            result = entry.function(**test_input)

        if asyncio.iscoroutine(result):
            result = await result

        report.execution_time_ms = (time.time() - start) * 1000

        # Contract validation
        contract = entry.result_contract or {}
        violations = []

        if not contract.get("allow_none", True) and result is None:
            violations.append("result=None aber allow_none=False")

        if not contract.get("allow_empty_string", True) and result == "":
            violations.append("result='' aber allow_empty_string=False")

        expected_type = contract.get("expected_type", "any")
        if expected_type != "any" and result is not None:
            _type_map = {
                "str": str, "dict": dict, "list": list,
                "int": int, "bool": bool, "float": float
            }
            if expected_type in _type_map and not isinstance(result, _type_map[expected_type]):
                violations.append(
                    f"expected_type={expected_type}, got={type(result).__name__}"
                )

        report.result_preview = repr(result)[:150] if result is not None else "None"
        report.contract_violations = violations

        if violations:
            report.status = "DEGRADED"
            report.error = "; ".join(violations)
        else:
            report.status = "HEALTHY"

    except Exception as e:
        report.execution_time_ms = (time.time() - start) * 1000
        report.status = "FAILED"
        report.error = str(e)[:300]

    return report


# =============================================================================
# Test Suite
# =============================================================================

class TestSelfAgentLiveTools(unittest.IsolatedAsyncioTestCase):
    """
    Lädt den self-Agent und testet jedes Tool einzeln.

    Struktur:
      test_01  — Agent ladbar, Tools registriert
      test_02  — Inventory: gibt klaren Report aller Tools nach Kategorie
      test_03  — subTest je Tool: execution oder klarer Skip-Grund
      test_04  — Zusammenfassung: kein Tool darf FAILED sein
      test_05  — Zusammenfassung: SKIPPED-Tools brauchen Attention-Marker
    """

    @classmethod
    def setUpClass(cls):
        cls._reports: list[ToolReport] = []
        cls._agent = None

    async def asyncSetUp(self):
        agent, err = await _load_agent()
        if err:
            self.skipTest(f"self-Agent nicht ladbar: {err}")
        self.agent = agent

        session = await self.agent.session_manager.get_or_create("default")
        self.agent.init_session_tools(session)
        self.tm = agent.tool_manager

    # -------------------------------------------------------------------------
    # test_01 — Agent ladbar
    # -------------------------------------------------------------------------

    async def test_01_agent_loads_and_has_tools(self):
        count = self.tm.count()
        self.assertGreater(count, 0, "self-Agent hat 0 Tools registriert")
        print(f"\n  ✓ self-Agent geladen: {count} Tools registriert")

    # -------------------------------------------------------------------------
    # test_02 — Inventory Report (kein assert, nur Ausgabe)
    # -------------------------------------------------------------------------

    async def test_02_inventory_report(self):
        entries = self.tm.get_all()

        by_category = {
            "GUARANTEED":    [],
            "HAS_INPUTS":    [],
            "NO_INPUTS":     [],
            "NO_FUNCTION":   [],
        }

        for e in entries:
            if e.flags.get("guaranteed_healthy", False):
                by_category["GUARANTEED"].append(e)
            elif e.function is None:
                by_category["NO_FUNCTION"].append(e)
            elif e.live_test_inputs:
                by_category["HAS_INPUTS"].append(e)
            else:
                by_category["NO_INPUTS"].append(e)

        print("\n" + "=" * 70)
        print("  SELF-AGENT TOOL INVENTORY")
        print("=" * 70)

        print(f"\n  ✅ GUARANTEED ({len(by_category['GUARANTEED'])} Tools — manuell verifiziert):")
        for e in sorted(by_category["GUARANTEED"], key=lambda x: x.name):
            print(f"     • {e.name:<40} [{e.source}]")

        print(f"\n  🧪 HAS_INPUTS ({len(by_category['HAS_INPUTS'])} Tools — werden live getestet):")
        for e in sorted(by_category["HAS_INPUTS"], key=lambda x: x.name):
            n_inputs = len(e.live_test_inputs)
            has_contract = "contract" if e.result_contract else "no-contract"
            print(f"     • {e.name:<40} [{e.source}] {n_inputs} input(s), {has_contract}")

        print(f"\n  ⚠️  NO_INPUTS ({len(by_category['NO_INPUTS'])} Tools — brauchen Attention):")
        for e in sorted(by_category["NO_INPUTS"], key=lambda x: x.name):
            print(f"     • {e.name:<40} [{e.source}]  schema: {e.args_schema[:150]}")

        print(f"\n  🔌 NO_FUNCTION ({len(by_category['NO_FUNCTION'])} Tools — MCP/A2A):")
        for e in sorted(by_category["NO_FUNCTION"], key=lambda x: x.name):
            srv = e.server_name or "?"
            print(f"     • {e.name:<40} [{e.source}] server={srv}")

        print("=" * 70)

        # Speichere Kategorien für spätere Tests
        self.__class__._inventory = by_category

        # Soft-assert: mind. Anzahl bekannt
        total = sum(len(v) for v in by_category.values())
        self.assertEqual(total, self.tm.count())

    # -------------------------------------------------------------------------
    # test_03 — Per-Tool subTest (execution)
    # -------------------------------------------------------------------------

    async def test_03_per_tool_subtests(self):
        entries = self.tm.get_all()
        reports: list[ToolReport] = []

        print(f"\n  Running per-tool health checks ({len(entries)} tools)...\n")

        for entry in sorted(entries, key=lambda e: e.name):
            with self.subTest(tool=entry.name, source=entry.source):
                report = await _run_single_tool_health(entry)
                reports.append(report)

                # Status-Icon
                icon = {
                    "HEALTHY":     "✅",
                    "GUARANTEED":  "🔒",
                    "DEGRADED":    "⚠️ ",
                    "FAILED":      "❌",
                    "SKIPPED":     "⏭️ ",
                    "NO_FUNCTION": "🔌",
                }.get(report.status, "?")

                line = f"  {icon} {entry.name:<45} {report.status}"
                if report.execution_time_ms > 0:
                    line += f"  ({report.execution_time_ms:.1f}ms)"
                if report.error and report.status not in ("SKIPPED", "NO_FUNCTION"):
                    line += f"\n       └─ {report.error[:120]}"
                if report.contract_violations:
                    line += f"\n       └─ violations: {report.contract_violations}"
                print(line)

                # FAILED ist ein echter Fehler — Subtest schlägt fehl
                self.assertNotEqual(
                    report.status, "FAILED",
                    f"Tool '{entry.name}' FAILED: {report.error}"
                )

        # Speichere Reports für test_04 / test_05
        self.__class__._reports = reports

    # -------------------------------------------------------------------------
    # test_04 — Keine FAILED Tools insgesamt
    # -------------------------------------------------------------------------

    async def test_04_no_failed_tools(self):
        # Falls test_03 noch nicht lief (z.B. einzeln ausgeführt)
        if not self.__class__._reports:
            entries = self.tm.get_all()
            for e in entries:
                r = await _run_single_tool_health(e)
                self.__class__._reports.append(r)

        failed = [r for r in self.__class__._reports if r.status == "FAILED"]

        if failed:
            details = "\n".join(
                f"  • {r.name}: {r.error}" for r in failed
            )
            self.fail(
                f"{len(failed)} Tool(s) FAILED:\n{details}"
            )

    # -------------------------------------------------------------------------
    # test_05 — SKIPPED Tools Report + Attention-Marker prüfen
    # -------------------------------------------------------------------------

    async def test_05_skipped_tools_need_attention(self):
        if not self.__class__._reports:
            entries = self.tm.get_all()
            for e in entries:
                r = await _run_single_tool_health(e)
                self.__class__._reports.append(r)

        skipped = [r for r in self.__class__._reports if r.status == "SKIPPED"]
        total = len(self.__class__._reports)
        testable = [r for r in self.__class__._reports
                    if r.status in ("HEALTHY", "GUARANTEED", "DEGRADED", "FAILED")]

        print(f"\n  COVERAGE SUMMARY")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total tools:       {total}")
        print(f"  Testable/verified: {len(testable)}  ({100*len(testable)//total if total else 0}%)")
        print(f"  SKIPPED (local):   {len(skipped)}")
        healthy    = sum(1 for r in self.__class__._reports if r.status == "HEALTHY")
        guaranteed = sum(1 for r in self.__class__._reports if r.status == "GUARANTEED")
        degraded   = sum(1 for r in self.__class__._reports if r.status == "DEGRADED")
        failed     = sum(1 for r in self.__class__._reports if r.status == "FAILED")
        no_func    = sum(1 for r in self.__class__._reports if r.status == "NO_FUNCTION")
        print(f"  HEALTHY:           {healthy}")
        print(f"  GUARANTEED:        {guaranteed}")
        print(f"  DEGRADED:          {degraded}")
        print(f"  FAILED:            {failed}")
        print(f"  NO_FUNCTION (mcp): {no_func}")

        if skipped:
            print(f"\n  ⚠️  ATTENTION — folgende Tools brauchen live_test_inputs oder guaranteed_healthy=True:")
            for r in sorted(skipped, key=lambda x: x.name):
                print(f"     • {r.name:<45} schema: {r.args_schema[:60]}")

        # Kein hard-fail für SKIPPED — aber Coverage unter 50% ist ein Warning
        if total > 0:
            coverage = len(testable) / total
            if coverage < 0.5:
                print(f"\n  ⚠️  WARNING: Tool-Coverage unter 50% ({coverage:.0%})")
                print(f"     Empfehlung: live_test_inputs für SKIPPED-Tools ergänzen")

        # Einzige harte Regel: kein FAILED
        self.assertEqual(failed, 0)


# =============================================================================
# Standalone ausführbar
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
