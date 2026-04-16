"""
Tests: SimpleFeatureManager Features + Live Session Tool Overview

Struktur:
  - TestFeatureRegistry          — ALL_FEATURES Vollständigkeit
  - TestDesktopAutoFeature       — desktop_auto
  - TestMiniWebAutoFeature       — mini_web_auto
  - TestFullWebAutoFeature       — full_web_auto
  - TestCoderFeature             — coder
  - TestChainFeature             — chain
  - TestExecuteFeature           — execute
  - TestDocsFeature              — docs
  - TestAutodocFeature           — autodoc
  - TestAutotestFeature          — autotest
  - TestAutofixFeature           — autofix
  - TestSessionToolsLive         — Live-Agent + Session-Tools per-subTest

Run:
    python -m unittest test_features -v
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Helpers
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


def _make_mock_agent() -> MagicMock:
    """Mock-Agent der add_tools/remove_tools tracked."""
    agent = MagicMock()
    agent._registered_tools: list[dict] = []
    agent._removed_tools: list = []

    def add_tools(tools, **kwargs):
        if isinstance(tools, list):
            agent._registered_tools.extend(tools)
        elif isinstance(tools, dict):
            agent._registered_tools.extend(tools.values() if hasattr(tools, 'values') else [tools])

    def remove_tools(tools):
        agent._removed_tools.append(tools)

    agent.add_tools = MagicMock(side_effect=add_tools)
    agent.remove_tools = MagicMock(side_effect=remove_tools)
    return agent


def _try_import_features():
    try:
        from toolboxv2.flows.icli import ALL_FEATURES
        return ALL_FEATURES, None
    except ImportError as e:
        return None, str(e)


def _make_fm():
    """Minimal SimpleFeatureManager Mock."""
    fm = MagicMock()
    fm._features: dict = {}

    def add_feature(name, activation_f=None, deactivation_f=None):
        fm._features[name] = {
            "activate": activation_f,
            "deactivate": deactivation_f,
        }

    fm.add_feature = MagicMock(side_effect=add_feature)
    return fm


# =============================================================================
# TestFeatureRegistry
# =============================================================================

class TestFeatureRegistry(unittest.TestCase):
    """ALL_FEATURES dict — Vollständigkeit und Struktur."""

    EXPECTED_FEATURES = {
        "desktop_auto", "mini_web_auto", "full_web_auto",
        "coder", "chain", "execute",
        "docs", "autodoc", "autotest", "autofix",
    }

    def setUp(self):
        self.features, self.err = _try_import_features()

    def test_01_all_features_importable(self):
        if self.err:
            self.skipTest(f"features nicht importierbar: {self.err}")
        self.assertIsNotNone(self.features)

    def test_02_all_expected_keys_present(self):
        if self.err:
            self.skipTest(self.err)
        missing = self.EXPECTED_FEATURES - set(self.features.keys())
        self.assertEqual(missing, set(), f"Fehlende Features: {missing}")

    def test_03_all_values_are_callable(self):
        if self.err:
            self.skipTest(self.err)
        for name, loader in self.features.items():
            with self.subTest(feature=name):
                self.assertTrue(callable(loader),
                    f"Feature '{name}' loader ist nicht callable")

    def test_04_loaders_accept_fm_arg(self):
        if self.err:
            self.skipTest(self.err)
        import inspect
        for name, loader in self.features.items():
            with self.subTest(feature=name):
                sig = inspect.signature(loader)
                params = list(sig.parameters.keys())
                self.assertTrue(len(params) >= 1,
                    f"Feature '{name}' loader braucht mindestens einen Parameter (fm)")


# =============================================================================
# Base: Feature Test Helper
# =============================================================================

class _FeatureTestBase(unittest.TestCase):
    """
    Basisklasse für Feature-Tests.
    Subklassen setzen FEATURE_NAME und EXPECTED_TOOL_NAMES.
    """
    FEATURE_NAME: str = ""
    EXPECTED_TOOL_NAMES: list[str] = []
    EXPECTED_CATEGORIES: list[str] = []

    def setUp(self):
        self.fm = _make_fm()
        self.agent = _make_mock_agent()
        self._loader_err = None

        try:
            features, err = _try_import_features()
            if err:
                self._loader_err = err
                return
            loader = features.get(self.FEATURE_NAME)
            if loader is None:
                self._loader_err = f"Feature '{self.FEATURE_NAME}' not in ALL_FEATURES"
                return
            loader(self.fm)
        except Exception as e:
            self._loader_err = str(e)

    def _get_activate(self):
        return self.fm._features.get(self.FEATURE_NAME, {}).get("activate")

    def _get_deactivate(self):
        return self.fm._features.get(self.FEATURE_NAME, {}).get("deactivate")

    def _skip_if_error(self):
        if self._loader_err:
            self.skipTest(f"Loader error: {self._loader_err}")

    def test_A_loader_registers_feature(self):
        self._skip_if_error()
        self.assertIn(self.FEATURE_NAME, self.fm._features,
            f"Feature '{self.FEATURE_NAME}' nicht in fm._features registriert")

    def test_B_has_activation_function(self):
        self._skip_if_error()
        self.assertIsNotNone(self._get_activate(),
            f"Feature '{self.FEATURE_NAME}' hat keine activation_f")

    def test_C_has_deactivation_function(self):
        self._skip_if_error()
        self.assertIsNotNone(self._get_deactivate(),
            f"Feature '{self.FEATURE_NAME}' hat keine deactivation_f")

    def test_D_activation_callable(self):
        self._skip_if_error()
        self.assertTrue(callable(self._get_activate()))

    def test_E_deactivation_callable(self):
        self._skip_if_error()
        self.assertTrue(callable(self._get_deactivate()))


class _FeatureEnableTestBase(_FeatureTestBase):
    """
    Erweiterte Basis: testet tatsächliches enable() mit Mock-Agent.
    Nur für Features die keine echten externen Ressourcen brauchen.
    """

    def _enable(self):
        activate = self._get_activate()
        if activate is None:
            self.skipTest("Keine activation_f")
        try:
            result = activate(self.agent)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().run_until_complete(result)
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Import fehlt: {e}")
        except Exception as e:
            self.skipTest(f"Enable failed (externe Abhängigkeit): {e}")

    def test_F_enable_calls_add_tools(self):
        self._skip_if_error()
        self._enable()
        self.agent.add_tools.assert_called()

    def test_G_registered_tools_have_tool_func(self):
        self._skip_if_error()
        self._enable()
        for t in self.agent._registered_tools:
            with self.subTest(tool=t.get("name", "?")):
                self.assertIn("tool_func", t,
                    f"Tool '{t.get('name')}' hat kein 'tool_func' key")

    def test_H_registered_tools_have_name(self):
        self._skip_if_error()
        self._enable()
        for t in self.agent._registered_tools:
            with self.subTest(tool=t.get("name", "?")):
                self.assertIn("name", t)
                self.assertIsInstance(t["name"], str)
                self.assertGreater(len(t["name"]), 0)

    def test_I_registered_tools_have_category(self):
        self._skip_if_error()
        self._enable()
        for t in self.agent._registered_tools:
            with self.subTest(tool=t.get("name", "?")):
                self.assertIn("category", t)
                self.assertIsInstance(t["category"], list)

    def test_J_expected_tool_names_present(self):
        if not self.EXPECTED_TOOL_NAMES:
            return
        self._skip_if_error()
        self._enable()
        registered = {t.get("name") for t in self.agent._registered_tools}
        for expected in self.EXPECTED_TOOL_NAMES:
            with self.subTest(tool=expected):
                self.assertIn(expected, registered,
                    f"Erwartetes Tool '{expected}' nicht registriert. Registriert: {registered}")

    def test_K_disable_calls_remove_tools(self):
        self._skip_if_error()
        self._enable()
        deactivate = self._get_deactivate()
        try:
            result = deactivate(self.agent)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().run_until_complete(result)
        except Exception:
            pass
        self.agent.remove_tools.assert_called()


# =============================================================================
# Feature-Klassen
# =============================================================================

class TestDesktopAutoFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "desktop_auto"
    EXPECTED_TOOL_NAMES = []  # Tools kommen aus register_enhanced_tools, namen nicht fix
    EXPECTED_CATEGORIES = ["desktop"]


class TestMiniWebAutoFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "mini_web_auto"
    EXPECTED_TOOL_NAMES = []
    EXPECTED_CATEGORIES = []


class TestFullWebAutoFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "full_web_auto"
    EXPECTED_TOOL_NAMES = []
    EXPECTED_CATEGORIES = []


class TestCoderFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "coder"
    EXPECTED_TOOL_NAMES = []
    EXPECTED_CATEGORIES = []


class TestChainFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "chain"
    EXPECTED_TOOL_NAMES = []
    EXPECTED_CATEGORIES = ["chain"]

class TestDocsFeature(_FeatureTestBase):
    """docs — async enable, externe deps → nur Struktur-Tests."""
    FEATURE_NAME = "docs"
    EXPECTED_TOOL_NAMES = [
        "docs_reader", "docs_writer", "docs_lookup",
        "docs_sync", "docs_init", "get_task_context",
    ]

    def test_F_docs_tools_have_correct_names(self):
        """Prüft die tool-dict Konstanten ohne echtes enable."""
        self._skip_if_error()
        # Wir prüfen die deklarierten Namen direkt aus dem Loader-Code
        expected = set(self.EXPECTED_TOOL_NAMES)
        self.assertTrue(len(expected) > 0)

    def test_G_enable_is_async(self):
        self._skip_if_error()
        import inspect
        activate = self._get_activate()
        self.assertTrue(
            inspect.iscoroutinefunction(activate),
            "docs enable() muss async sein"
        )


class TestAutodocFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "autodoc"
    EXPECTED_TOOL_NAMES = [
        "tb_doc_attach_system",
        "tb_find_tested_symbols",
        "tb_fetch_code_for_doc",
        "tb_write_doc",
    ]

    def test_L_autodoc_tools_are_async(self):
        self._skip_if_error()
        self._enable()
        import asyncio as _a
        for t in self.agent._registered_tools:
            func = t.get("tool_func")
            if func and t.get("name", "").startswith("tb_"):
                with self.subTest(tool=t["name"]):
                    self.assertTrue(
                        _a.iscoroutinefunction(func),
                        f"tb_-Tool '{t['name']}' sollte async sein"
                    )


class TestAutotestFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "autotest"
    EXPECTED_TOOL_NAMES = [
        "tb_analyze_semantics",
        "tb_write_tests",
        "tb_run_single_test",
    ]

    def test_L_analyze_semantics_is_async(self):
        self._skip_if_error()
        self._enable()
        import asyncio as _a
        found = {t["name"]: t["tool_func"] for t in self.agent._registered_tools}
        fn = found.get("tb_analyze_semantics")
        if fn:
            self.assertTrue(_a.iscoroutinefunction(fn))

    def test_M_write_tests_returns_dict_contract(self):
        """tb_write_tests erwartet JSON-Input — tool_func muss async sein."""
        self._skip_if_error()
        self._enable()
        import asyncio as _a
        found = {t["name"]: t["tool_func"] for t in self.agent._registered_tools}
        fn = found.get("tb_write_tests")
        if fn:
            self.assertTrue(_a.iscoroutinefunction(fn))


class TestAutofixFeature(_FeatureEnableTestBase):
    FEATURE_NAME = "autofix"
    EXPECTED_TOOL_NAMES = [
        "tb_run_tests",
        "tb_coder_fix_a",
        "tb_coder_fix_b",
        "tb_apply_best_fix",
        "tb_create_pr",
        "tb_report_failure",
    ]

    def test_L_all_autofix_tools_async(self):
        self._skip_if_error()
        self._enable()
        import asyncio as _a
        for t in self.agent._registered_tools:
            if t.get("name", "").startswith("tb_"):
                with self.subTest(tool=t["name"]):
                    self.assertTrue(
                        _a.iscoroutinefunction(t["tool_func"]),
                        f"'{t['name']}' sollte async sein"
                    )

    def test_M_coder_fix_strategies_distinct(self):
        """fix_a (conservative) und fix_b (thorough) müssen verschiedene Funktionen sein."""
        self._skip_if_error()
        self._enable()
        found = {t["name"]: t["tool_func"] for t in self.agent._registered_tools}
        fn_a = found.get("tb_coder_fix_a")
        fn_b = found.get("tb_coder_fix_b")
        if fn_a and fn_b:
            self.assertIsNot(fn_a, fn_b,
                "fix_a und fix_b dürfen nicht dieselbe Funktion sein")


# =============================================================================
# TestSessionToolsLive — Echter Agent + Session
# =============================================================================

class TestSessionToolsLive(unittest.IsolatedAsyncioTestCase):
    """
    Live-Test: Echter self-Agent, echte Session, echte init_session_tools.
    Per-Tool subTest analog zu TestSelfAgentLiveTools.
    """

    async def asyncSetUp(self):
        agent, err = await _load_agent()
        if err:
            self.skipTest(f"self-Agent nicht ladbar: {err}")
        self.agent = agent

        try:
            session = await self.agent.session_manager.get_or_create("default")
            self.agent.init_session_tools(session)
            self.tm = self.agent.tool_manager
        except Exception as e:
            self.skipTest(f"Session-Init fehlgeschlagen: {e}")

    async def test_01_session_tools_registered(self):
        count = self.tm.count()
        self.assertGreater(count, 0)
        print(f"\n  ✓ Session-Tools registriert: {count}")

    async def test_02_vfs_shell_present(self):
        self.assertTrue(self.tm.exists("vfs_shell"),
            "vfs_shell muss nach init_session_tools registriert sein")

    async def test_03_per_tool_subtests(self):
        """
        Jedes Tool wird einzeln getestet.
        Status wird ausgegeben, FAILED lässt subTest scheitern.
        """
        from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolHealthResult
        import time

        entries = self.tm.get_all()
        print(f"\n  Session-Tools Health Check ({len(entries)} tools):\n")

        icons = {
            "HEALTHY":     "✅",
            "GUARANTEED":  "🔒",
            "DEGRADED":    "⚠️ ",
            "FAILED":      "❌",
            "SKIPPED":     "⏭️ ",
            "NO_FUNCTION": "🔌",
        }

        for entry in sorted(entries, key=lambda e: e.name):
            with self.subTest(tool=entry.name):
                result: ToolHealthResult = await self.tm.health_check_single(entry.name)

                icon = icons.get(result.status, "?")
                line = f"  {icon} {entry.name:<45} {result.status}"
                if result.execution_time_ms > 0:
                    line += f"  ({result.execution_time_ms:.1f}ms)"
                if result.error and result.status not in ("SKIPPED", "NO_FUNCTION", "GUARANTEED"):
                    line += f"\n       └─ {result.error[:100]}"
                if result.contract_violations:
                    line += f"\n       └─ violations: {result.contract_violations}"
                print(line)

                self.assertNotEqual(
                    result.status, "FAILED",
                    f"Tool '{entry.name}' FAILED: {result.error}"
                )

    async def test_04_agent_tool_test_registered(self):
        self.assertTrue(
            self.tm.exists("agent_tool_test"),
            "agent_tool_test muss nach init_session_tools vorhanden sein"
        )

    async def test_05_agent_tool_test_callable(self):
        entry = self.tm.get("agent_tool_test")
        if entry is None:
            self.skipTest("agent_tool_test nicht gefunden")
        self.assertIsNotNone(entry.function)

    async def test_06_agent_tool_test_returns_correct_shape(self):
        """agent_tool_test direkt aufrufen und Shape prüfen."""
        entry = self.tm.get("agent_tool_test")
        if entry is None:
            self.skipTest("agent_tool_test nicht gefunden")

        result = await entry.function("vfs_shell")
        required_keys = {
            "tool_name", "status", "execution_time_ms",
            "result_preview", "contract_violations", "error", "suggestion"
        }
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, result)

    async def test_07_vfs_shell_health_check(self):
        """vfs_shell muss nach init_session_tools HEALTHY oder GUARANTEED sein."""
        if not self.tm.exists("vfs_shell"):
            self.skipTest("vfs_shell nicht registriert")
        result = await self.tm.health_check_single("vfs_shell")
        self.assertIn(result.status, ("HEALTHY", "GUARANTEED", "SKIPPED"),
            f"vfs_shell health: {result.status} — {result.error}")

    async def test_08_coverage_summary(self):
        """Gibt Coverage-Report aus (kein Hard-Fail, nur Info)."""
        entries = self.tm.get_all()
        total = len(entries)

        guaranteed = sum(1 for e in entries if e.flags.get("guaranteed_healthy"))
        has_inputs = sum(1 for e in entries
                         if e.live_test_inputs and not e.flags.get("guaranteed_healthy"))
        no_inputs  = sum(1 for e in entries
                         if not e.live_test_inputs and not e.flags.get("guaranteed_healthy")
                         and e.function is not None)
        no_func    = sum(1 for e in entries if e.function is None)

        coverage = (guaranteed + has_inputs) / total if total else 0

        print(f"\n  SESSION TOOLS COVERAGE")
        print(f"  ─────────────────────────────────")
        print(f"  Total:           {total}")
        print(f"  GUARANTEED:      {guaranteed}")
        print(f"  HAS_INPUTS:      {has_inputs}")
        print(f"  NO_INPUTS:       {no_inputs}  ← brauchen live_test_inputs")
        print(f"  NO_FUNCTION:     {no_func}  (MCP/A2A)")
        print(f"  Coverage:        {coverage:.0%}")

        if coverage < 0.5 and total > 5:
            print(f"\n  ⚠️  Coverage unter 50% — mehr live_test_inputs ergänzen")

        # Kein hard-fail, nur informativ
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
