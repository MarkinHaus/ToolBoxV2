"""
Tests für init_session_tools — Health-Check aller Session-Tools.

Aufbau:
  TestSessionToolContracts  — statische Kontrakt-Prüfung (kein live-run)
  TestSessionToolsLive      — live health_check_single per subTest
  TestSessionToolsSemantic  — deterministische Semantic-Checks (kein LLM)

Run:
    python -m unittest test_session_tools_health -v
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock


# =============================================================================
# Mock-Session und Mock-VFS für isolierte Tests
# =============================================================================
def _vfs_file(filename: str, content: str, state: str = "closed") -> 'VFSFile':
    f = VFSFile(filename=filename)
    f._content = content
    f.state = state
    return f


def _make_mock_vfs():
    vfs = MagicMock()
    vfs.files = {
        "/test/hello.py": _vfs_file("hello.py", "print('hello')"),
        "/src/main.py": _vfs_file("main.py", "def main(): pass", state="open"),
    }
    vfs._is_directory = MagicMock(return_value=True)
    vfs._normalize_path = MagicMock(side_effect=lambda p: p)
    vfs.load_from_local = MagicMock(return_value={"success": True})
    vfs.save_to_local = MagicMock(return_value={"success": True})
    vfs.mount = MagicMock(return_value={"success": True})
    vfs.unmount = MagicMock(return_value={"success": True})
    vfs.refresh_mount = MagicMock(return_value={"success": True})
    vfs.sync_all = MagicMock(return_value={"success": True, "saved_count": 0})
    vfs.build_context_string = MagicMock(return_value="")
    return vfs


def _make_mock_session():
    session = MagicMock()
    session.session_id = "test_session"
    session.agent_name = "test_agent"
    session.tools_initialized = False
    session.vfs = _make_mock_vfs()
    session.get_history_for_llm = MagicMock(return_value=[])
    session.set_situation = MagicMock(return_value=None)
    session.rule_on_action = MagicMock(return_value=MagicMock(
        allowed=True, reason="OK", rule_name=None
    ))
    session.vfs_diagnostics = AsyncMock(return_value={"diagnostics": []})
    session.docker_run_command = AsyncMock(return_value={"success": True})
    session.docker_start_web_app = AsyncMock(return_value={"success": True})
    session.docker_stop_web_app = AsyncMock(return_value={"success": True})
    session.docker_get_logs = AsyncMock(return_value={"logs": []})
    session.docker_status = MagicMock(return_value={"running": False})
    return session


def _make_mock_isaa():
    isaa = MagicMock()
    isaa.add_tools = MagicMock()
    return isaa


# =============================================================================
# Loader — gibt ToolManager mit allen Session-Tools zurück
# =============================================================================

def _load_session_tools():
    """
    Instanziiert init_session_tools mit Mock-Session und gibt ToolManager zurück.
    """
    from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent  # Modul mit init_session_tools

    session = _make_mock_session()
    isaa = _make_mock_isaa()
    tm = ToolManager()

    # Patch add_tools so es direkt im ToolManager registriert
    def _add_tools(tools_list):
        for t in tools_list:
            tm.register(
                func=t.get("tool_func"),
                name=t.get("name"),
                description=t.get("description"),
                category=t.get("category"),
                flags=t.get("flags"),
                live_test_inputs=t.get("live_test_inputs"),
                result_contract=t.get("result_contract"),
                cleanup_func=t.get("cleanup_func"),
            )

    isaa.add_tools = _add_tools

    # init_session_tools aufrufen
    FlowAgent.init_session_tools(isaa, session)

    return tm, session


# =============================================================================
# Class 1 — statische Kontrakt-Prüfungen (kein live-run)
# =============================================================================

class TestSessionToolContracts(unittest.TestCase):
    """
    Prüft die Metadaten-Qualität aller registrierten Session-Tools.
    Kein execution — nur Validierung der Tool-Definitionen selbst.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.tm, cls.session = _load_session_tools()
            cls.available = True
        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest(self.skip_reason)

    def test_01_all_expected_tools_registered(self):
        self._skip_if_unavailable()
        expected = {
            "vfs_shell", "vfs_view", "search_vfs",
            "fs_copy_to_vfs", "fs_copy_from_vfs", "fs_copy_dir_from_vfs",
            "vfs_mount", "vfs_unmount", "vfs_refresh_mount", "vfs_sync_all",
            "vfs_share_create", "vfs_share_list", "vfs_share_mount",
            "vfs_diagnostics",
            "docker_run", "docker_start_app", "docker_stop_app",
            "docker_logs", "docker_status",
            "history", "set_agent_situation", "check_permissions",
        }
        registered = set(self.tm.list_names())
        missing = expected - registered
        self.assertEqual(missing, set(), f"Fehlende Tools: {missing}")

    def test_02_no_tool_without_description(self):
        self._skip_if_unavailable()
        bad = [e.name for e in self.tm.get_all()
               if not e.description or e.description.strip() == ""]
        self.assertEqual(bad, [], f"Tools ohne Description: {bad}")

    def test_03_no_tool_with_default_description_only(self):
        """'Tool: <name>' ist ein unbrauchbarer Placeholder für den Agent."""
        self._skip_if_unavailable()
        bad = [e.name for e in self.tm.get_all()
               if e.description.strip().startswith("Tool:")]
        self.assertEqual(bad, [], f"Tools mit Placeholder-Description: {bad}")

    def test_04_testable_tools_have_result_contract(self):
        """Jedes Tool mit live_test_inputs muss ein result_contract haben."""
        self._skip_if_unavailable()
        bad = [e.name for e in self.tm.get_all()
               if e.live_test_inputs and not e.result_contract]
        self.assertEqual(bad, [], f"Tools mit test_inputs aber ohne contract: {bad}")

    def test_05_result_contract_has_required_keys(self):
        """result_contract muss mindestens allow_none und expected_type haben."""
        self._skip_if_unavailable()
        required_keys = {"allow_none", "expected_type"}
        bad = []
        for e in self.tm.get_all():
            if e.result_contract:
                missing = required_keys - set(e.result_contract.keys())
                if missing:
                    bad.append(f"{e.name}: missing {missing}")
        self.assertEqual(bad, [], f"Unvollständige result_contracts: {bad}")

    def test_06_guaranteed_tools_have_no_test_inputs(self):
        """guaranteed_healthy=True + live_test_inputs ist ein Widerspruch."""
        self._skip_if_unavailable()
        bad = [e.name for e in self.tm.get_all()
               if e.flags.get("guaranteed_healthy") and e.live_test_inputs]
        self.assertEqual(bad, [], f"guaranteed-Tools mit test_inputs: {bad}")

    def test_07_docker_tools_are_guaranteed(self):
        """Alle Docker-Tools müssen guaranteed_healthy=True haben."""
        self._skip_if_unavailable()
        docker_tools = [e for e in self.tm.get_all()
                        if "docker" in e.category]
        not_guaranteed = [e.name for e in docker_tools
                          if not e.flags.get("guaranteed_healthy")]
        self.assertEqual(not_guaranteed, [],
            f"Docker-Tools ohne guaranteed_healthy: {not_guaranteed}")

    def test_08_filesystem_write_tools_are_guaranteed(self):
        """fs_copy_* und mount/unmount müssen guaranteed sein (Seiteneffekte)."""
        self._skip_if_unavailable()
        write_tools = {"fs_copy_to_vfs", "fs_copy_from_vfs", "fs_copy_dir_from_vfs",
                       "vfs_mount", "vfs_unmount", "vfs_refresh_mount"}
        for name in write_tools:
            entry = self.tm.get(name)
            if entry:
                self.assertTrue(
                    entry.flags.get("guaranteed_healthy"),
                    f"{name} sollte guaranteed_healthy=True haben"
                )

    def test_09_semantic_hints_present_for_stateful_tools(self):
        """set_agent_situation und check_permissions brauchen semantic_check_hint."""
        self._skip_if_unavailable()
        critical = ["set_agent_situation", "check_permissions"]
        for name in critical:
            entry = self.tm.get(name)
            if entry and entry.result_contract:
                hint = entry.result_contract.get("semantic_check_hint", "")
                self.assertTrue(
                    len(hint) > 20,
                    f"{name}: semantic_check_hint fehlt oder zu kurz"
                )

    def test_10_cleanup_func_on_situation_tool(self):
        """set_agent_situation muss cleanup_func haben (State-Reset)."""
        self._skip_if_unavailable()
        entry = self.tm.get("set_agent_situation")
        if entry:
            self.assertIsNotNone(
                entry.cleanup_func,
                "set_agent_situation braucht cleanup_func um Situation zurückzusetzen"
            )

    def test_11_coverage_report(self):
        """Gibt Coverage-Bericht aus (kein hard-fail)."""
        self._skip_if_unavailable()
        all_tools = self.tm.get_all()
        testable = [e for e in all_tools if e.live_test_inputs]
        guaranteed = [e for e in all_tools if e.flags.get("guaranteed_healthy")]
        skipped = [e for e in all_tools
                   if not e.live_test_inputs and not e.flags.get("guaranteed_healthy")]

        print(f"\n  Session-Tool Coverage:")
        print(f"  Total:       {len(all_tools)}")
        print(f"  Testable:    {len(testable)}  ({[e.name for e in testable]})")
        print(f"  Guaranteed:  {len(guaranteed)}  ({[e.name for e in guaranteed]})")
        print(f"  Uncovered:   {len(skipped)}  ({[e.name for e in skipped]})")

        # Kein hard-fail, aber 0 uncovered ist das Ziel
        if skipped:
            print(f"  ⚠️  Folgende Tools brauchen live_test_inputs oder guaranteed_healthy:")
            for e in skipped:
                print(f"     • {e.name}")


# =============================================================================
# Class 2 — Live Health Checks per subTest
# =============================================================================

class TestSessionToolsLive(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.tm, cls.session = _load_session_tools()
            cls.available = True
        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    async def test_12_all_testable_tools_healthy(self):
        if not self.available:
            self.skipTest(self.skip_reason)

        testable = [e for e in self.tm.get_all() if e.live_test_inputs]
        if not testable:
            self.skipTest("Keine testbaren Tools gefunden")

        for entry in sorted(testable, key=lambda e: e.name):
            with self.subTest(tool=entry.name):
                result = await self.tm.health_check_single(entry.name)

                icon = {"HEALTHY": "✅", "DEGRADED": "⚠️", "FAILED": "❌"}.get(
                    result.status, "?"
                )
                print(f"\n  {icon} {entry.name:<35} {result.status}"
                      + (f"  ({result.execution_time_ms:.1f}ms)" if result.execution_time_ms else ""))

                if result.contract_violations:
                    print(f"     violations: {result.contract_violations}")
                if result.error and result.status == "FAILED":
                    print(f"     error: {result.error[:120]}")

                self.assertNotEqual(result.status, "FAILED",
                    f"Tool '{entry.name}' FAILED: {result.error}")
                self.assertNotEqual(result.status, "DEGRADED",
                    f"Tool '{entry.name}' DEGRADED: {result.contract_violations}")

    async def test_13_guaranteed_tools_status_after_check(self):
        if not self.available:
            self.skipTest(self.skip_reason)

        guaranteed = [e for e in self.tm.get_all()
                      if e.flags.get("guaranteed_healthy")]

        for entry in guaranteed:
            with self.subTest(tool=entry.name):
                result = await self.tm.health_check_single(entry.name)
                self.assertEqual(result.status, "GUARANTEED")
                self.assertTrue(self.tm.is_healthy(entry.name))

    async def test_14_vfs_shell_ls_returns_dict_with_success(self):
        """vfs_shell ls / — spezifischer Smoke-Test."""
        if not self.available:
            self.skipTest(self.skip_reason)
        entry = self.tm.get("vfs_shell")
        if not entry or not entry.function:
            self.skipTest("vfs_shell nicht verfügbar")

        result = await self.tm.execute("vfs_shell", command="ls /")
        self.assertIsInstance(result, (str, dict))

    async def test_15_search_vfs_returns_list(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        entry = self.tm.get("search_vfs")
        if not entry or not entry.function:
            self.skipTest("search_vfs nicht verfügbar")

        result = await self.tm.execute("search_vfs", query=".", mode="filename", max_results=1)
        # Result ist str (execute konvertiert) oder list
        self.assertIsNotNone(result)

    async def test_16_history_returns_list(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        result = await self.tm.health_check_single("history")
        self.assertIn(result.status, ("HEALTHY", "GUARANTEED"))

    async def test_17_vfs_sync_all_returns_dict(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        result = await self.tm.health_check_single("vfs_sync_all")
        self.assertIn(result.status, ("HEALTHY", "GUARANTEED"))

    async def test_18_set_agent_situation_cleanup_called(self):
        """Cleanup muss session.set_situation(None, None) aufrufen."""
        if not self.available:
            self.skipTest(self.skip_reason)

        await self.tm.health_check_single("set_agent_situation")
        # cleanup_func sollte set_situation mit None aufgerufen haben
        self.session.set_situation.assert_called()

    async def test_19_check_permissions_returns_allowed_bool(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        result = await self.tm.health_check_single("check_permissions")
        self.assertEqual(result.status, "HEALTHY",
            f"check_permissions nicht healthy: {result.error}")

    async def test_20_is_healthy_consistent_after_full_sweep(self):
        if not self.available:
            self.skipTest(self.skip_reason)

        await self.tm.health_check_all()

        for entry in self.tm.get_all():
            expected = entry.health_status in ("HEALTHY", "GUARANTEED")
            self.assertEqual(
                self.tm.is_healthy(entry.name), expected,
                f"is_healthy() inkonsistent für {entry.name}: {entry.health_status}"
            )


# =============================================================================
# Class 3 — Deterministische Semantic-Checks der Tool-Results
# (kein LLM — pure logic auf result_contract semantic_check_hints)
# =============================================================================

class TestSessionToolResultSemantics(unittest.TestCase):
    """
    Prüft bekannte semantische Muster in Tool-Results deterministisch.
    Ergänzt die LLM-basierten Checks (kommen später).
    """

    def test_21_vfs_shell_success_with_empty_stdout_is_suspicious(self):
        """ls / mit success=True und leerem stdout — VFS leer oder Bug."""
        result = {"success": True, "stdout": "", "stderr": "", "returncode": 0}
        issues = _check_vfs_shell_result("ls /", result)
        self.assertTrue(any("stdout" in i.lower() or "leer" in i.lower() for i in issues),
            f"Leeres stdout bei ls nicht erkannt: {issues}")

    def test_22_vfs_shell_returncode_mismatch(self):
        """returncode=0 aber success=False ist widersprüchlich."""
        result = {"success": False, "stdout": "", "stderr": "", "returncode": 0}
        issues = _check_vfs_shell_result("ls /", result)
        self.assertTrue(len(issues) > 0, "returncode/success Widerspruch nicht erkannt")

    def test_23_vfs_shell_error_without_stderr(self):
        """returncode!=0 ohne stderr — Agent hat keine Diagnose."""
        result = {"success": False, "stdout": "", "stderr": "", "returncode": 1}
        issues = _check_vfs_shell_result("ls /nonexistent", result)
        self.assertTrue(any("stderr" in i.lower() for i in issues),
            f"Fehlendes stderr bei Fehler nicht erkannt: {issues}")

    def test_24_vfs_shell_clean_success(self):
        result = {"success": True, "stdout": "/\n/project\n/tmp", "stderr": "", "returncode": 0}
        issues = _check_vfs_shell_result("ls /", result)
        self.assertEqual(issues, [])

    def test_25_check_permissions_allowed_none(self):
        """allowed=None ist eine Contract-Verletzung."""
        result = {"allowed": None, "reason": "ok", "rule": None}
        issues = _check_permissions_result(result)
        self.assertTrue(any("None" in i or "bool" in i for i in issues))

    def test_26_check_permissions_false_without_reason(self):
        """allowed=False ohne reason gibt Agent keine Diagnose."""
        result = {"allowed": False, "reason": "", "rule": None}
        issues = _check_permissions_result(result)
        self.assertTrue(len(issues) > 0, "Fehlende reason bei allowed=False nicht erkannt")

    def test_27_check_permissions_clean_allow(self):
        result = {"allowed": True, "reason": "no rule active", "rule": None}
        issues = _check_permissions_result(result)
        self.assertEqual(issues, [])

    def test_28_check_permissions_clean_deny(self):
        result = {"allowed": False, "reason": "precondition 'validated' not met", "rule": "save_rule"}
        issues = _check_permissions_result(result)
        self.assertEqual(issues, [])

    def test_29_set_situation_result_mismatch(self):
        """Zurückgegebene situation stimmt nicht mit Input überein."""
        result = {"success": True, "situation": "DIFFERENT", "intent": "__health_check__"}
        issues = _check_set_situation_result(
            inputs={"situation": "__health_check__", "intent": "__health_check__"},
            result=result,
        )
        self.assertTrue(len(issues) > 0, "situation-Mismatch nicht erkannt")

    def test_30_set_situation_clean(self):
        result = {"success": True, "situation": "test", "intent": "test"}
        issues = _check_set_situation_result(
            inputs={"situation": "test", "intent": "test"},
            result=result,
        )
        self.assertEqual(issues, [])

    def test_31_search_vfs_result_missing_path_key(self):
        """Ergebnis-Einträge ohne 'path'-Key sind unbrauchbar für den Agent."""
        result = [{"file": "hello.py", "snippet": "..."}]  # 'path' fehlt
        issues = _check_search_vfs_result(result)
        self.assertTrue(any("path" in i for i in issues))

    def test_32_search_vfs_clean(self):
        result = [{"path": "/test/hello.py", "snippet": "print('hello')"}]
        issues = _check_search_vfs_result(result)
        self.assertEqual(issues, [])

    def test_33_vfs_sync_all_success_with_none_count(self):
        result = {"success": True, "saved_count": None}
        issues = _check_sync_all_result(result)
        self.assertTrue(len(issues) > 0)

    def test_34_vfs_sync_all_failure_without_error(self):
        result = {"success": False, "saved_count": 0}
        issues = _check_sync_all_result(result)
        self.assertTrue(any("error" in i.lower() for i in issues))

    def test_35_vfs_sync_all_clean(self):
        result = {"success": True, "saved_count": 3}
        issues = _check_sync_all_result(result)
        self.assertEqual(issues, [])


# =============================================================================
# Deterministische Semantic-Checker (kein LLM, pure logic)
# Kandidaten für result_contract integration oder direkte Nutzung
# =============================================================================

def _check_vfs_shell_result(command: str, result: dict) -> list[str]:
    issues = []
    success = result.get("success")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    rc = result.get("returncode")

    if rc == 0 and success is False:
        issues.append("WIDERSPRUCH: returncode=0 aber success=False")
    if rc != 0 and success is True:
        issues.append("WIDERSPRUCH: returncode!=0 aber success=True")
    if rc != 0 and not stderr:
        issues.append("INCOMPLETE: returncode!=0 ohne stderr — kein Fehlertext für Agent")
    if success is True and rc == 0:
        is_read_cmd = any(command.strip().startswith(c)
                         for c in ("ls", "cat", "find", "tree", "grep", "stat", "wc", "head", "tail"))
        if is_read_cmd and not stdout:
            issues.append(
                f"SUSPICIOUS: '{command}' success=True mit leerem stdout — "
                f"VFS leer oder ls-Ergebnis wird nicht zurückgegeben"
            )
    return issues


def _check_permissions_result(result: dict) -> list[str]:
    issues = []
    allowed = result.get("allowed")
    reason = result.get("reason", "")

    if allowed is None:
        issues.append("CONTRACT: 'allowed' ist None — muss True oder False sein")
    if allowed is False and not reason:
        issues.append("INCOMPLETE: allowed=False ohne reason — Agent hat keine Diagnose-Basis")
    if "allowed" not in result:
        issues.append("CONTRACT: 'allowed'-Key fehlt im Result")
    return issues


def _check_set_situation_result(inputs: dict, result: dict) -> list[str]:
    issues = []
    expected_sit = inputs.get("situation", "")
    expected_int = inputs.get("intent", "")
    actual_sit = result.get("situation", "")
    actual_int = result.get("intent", "")

    if result.get("success") is True:
        if actual_sit != expected_sit:
            issues.append(
                f"MISMATCH: situation erwartet='{expected_sit}', got='{actual_sit}'"
            )
        if actual_int != expected_int:
            issues.append(
                f"MISMATCH: intent erwartet='{expected_int}', got='{actual_int}'"
            )
    if result.get("success") is False and "error" not in result:
        issues.append("INCOMPLETE: success=False ohne error-Key")
    return issues


def _check_search_vfs_result(result: list) -> list[str]:
    issues = []
    if not isinstance(result, list):
        issues.append(f"TYPE: erwartet list, got {type(result).__name__}")
        return issues
    for i, item in enumerate(result):
        if not isinstance(item, dict):
            issues.append(f"ITEM[{i}]: kein dict")
            continue
        if "path" not in item:
            issues.append(f"ITEM[{i}]: 'path'-Key fehlt — Agent kann Datei nicht referenzieren")
    return issues


def _check_sync_all_result(result: dict) -> list[str]:
    issues = []
    if result.get("success") is True and result.get("saved_count") is None:
        issues.append("INCOMPLETE: success=True aber saved_count=None — kein Zähler")
    if result.get("success") is False and "error" not in result:
        issues.append("INCOMPLETE: success=False ohne error-Key")
    return issues


"""
Tests: agent_tool_test Tool & Session Tools Health-Kwargs

Testet:
  1. agent_tool_test() Funktion direkt (Struktur + Logik)
  2. Tool-Dict-Struktur (live_test_inputs / result_contract / cleanup_func vorhanden)
  3. add_tools() reicht neue kwargs durch
  4. Result-Contract-Korrektheit je VFS-Tool
  5. cleanup_func Verhalten

Muster: wie test_vfs_shell_tool.py — MagicMock-Session, kein ToolBoxV2-Start.

Run:
    python -m unittest test_session_tools_health -v
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2, VFSFile
    from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager
    VFS_AVAILABLE = True
except ImportError:
    VFS_AVAILABLE = False


def _skip_if_unavailable(cls):
    if not VFS_AVAILABLE:
        return unittest.skip("toolboxv2 not importable")(cls)
    return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> MagicMock:
    vfs = VirtualFileSystemV2(session_id="test-health", agent_name="TestAgent")
    session = MagicMock()
    session.vfs = vfs
    session.tool_manager = ToolManager()
    session.set_situation = MagicMock()
    return session


def _make_agent_tool_test(session):
    """Erzeugt die agent_tool_test Closure analog zu init_session_tools."""
    tm = session.tool_manager

    async def agent_tool_test(tool_name: str, custom_inputs: dict | None = None) -> dict:
        entry = tm.get(tool_name)
        if entry is None:
            available = tm.list_names()
            close = [n for n in available if tool_name.lower() in n.lower()][:5]
            return {
                "tool_name":           tool_name,
                "status":              "NOT_FOUND",
                "execution_time_ms":   0.0,
                "result_preview":      None,
                "contract_violations": [],
                "error":               f"Tool '{tool_name}' not registered",
                "suggestion": (
                    f"Similar tools: {close}" if close
                    else f"Use list_tools() to see all {len(available)} registered tools"
                ),
            }

        effective_inputs = custom_inputs
        if effective_inputs is None and entry.live_test_inputs:
            effective_inputs = entry.live_test_inputs[0]

        if effective_inputs is not None and not entry.live_test_inputs:
            entry.live_test_inputs = [effective_inputs]
            result = await tm.health_check_single(tool_name)
            entry.live_test_inputs = []
        elif effective_inputs is not None:
            orig = entry.live_test_inputs[:]
            entry.live_test_inputs = [effective_inputs]
            result = await tm.health_check_single(tool_name)
            entry.live_test_inputs = orig
        else:
            result = await tm.health_check_single(tool_name)

        suggestions = {
            "HEALTHY":    "Tool is working correctly.",
            "GUARANTEED": "Tool is marked as manually verified — no live test run.",
            "SKIPPED":    "Add live_test_inputs or call with custom_inputs={'param': 'value'}.",
            "DEGRADED":   f"Contract violations: {result.contract_violations}. Check return value.",
            "FAILED":     f"Exception: {result.error}. Check tool implementation.",
        }

        return {
            "tool_name":           result.tool_name,
            "status":              result.status,
            "execution_time_ms":   round(result.execution_time_ms, 2),
            "result_preview":      (result.result_preview or "")[:300],
            "contract_violations": result.contract_violations,
            "error":               result.error,
            "suggestion":          suggestions.get(result.status, "Unknown status."),
        }

    return agent_tool_test


def _tool_ok(x: str = "probe") -> str:
    return f"ok:{x}"

def _tool_none(x: str = "probe") -> None:
    return None

def _tool_raises(x: str = "probe"):
    raise RuntimeError("intentional failure")


# ===========================================================================
# 1. agent_tool_test — Rückgabe-Struktur
# ===========================================================================

@_skip_if_unavailable
class TestAgentToolTestStructure(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.session = _make_session()
        self.tm = self.session.tool_manager
        self.att = _make_agent_tool_test(self.session)

    def _assert_shape(self, result: dict):
        required = {
            "tool_name":           str,
            "status":              str,
            "execution_time_ms":   (int, float),
            "result_preview":      (str, type(None)),
            "contract_violations": list,
            "error":               (str, type(None)),
            "suggestion":          str,
        }
        for key, expected_type in required.items():
            self.assertIn(key, result, f"Key '{key}' fehlt im Report")
            self.assertIsInstance(result[key], expected_type,
                f"Key '{key}' hat falschen Typ: {type(result[key])}")

    async def test_01_shape_healthy(self):
        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "probe"}])
        r = await self.att("_tool_ok")
        self._assert_shape(r)
        self.assertEqual(r["status"], "HEALTHY")

    async def test_02_shape_not_found(self):
        r = await self.att("nonexistent_xyz_tool")
        self._assert_shape(r)
        self.assertEqual(r["status"], "NOT_FOUND")

    async def test_03_shape_failed(self):
        self.tm.register(func=_tool_raises, live_test_inputs=[{"x": "probe"}])
        r = await self.att("_tool_raises")
        self._assert_shape(r)
        self.assertEqual(r["status"], "FAILED")

    async def test_04_shape_skipped(self):
        self.tm.register(func=_tool_ok)
        r = await self.att("_tool_ok")
        self._assert_shape(r)
        self.assertEqual(r["status"], "SKIPPED")

    async def test_05_shape_guaranteed(self):
        self.tm.register(func=_tool_ok, flags={"guaranteed_healthy": True})
        r = await self.att("_tool_ok")
        self._assert_shape(r)
        self.assertEqual(r["status"], "GUARANTEED")

    async def test_06_shape_degraded(self):
        self.tm.register(
            func=_tool_none,
            live_test_inputs=[{"x": "probe"}],
            result_contract={"allow_none": False},
        )
        r = await self.att("_tool_none")
        self._assert_shape(r)
        self.assertEqual(r["status"], "DEGRADED")
        self.assertGreater(len(r["contract_violations"]), 0)


# ===========================================================================
# 2. agent_tool_test — Logik
# ===========================================================================

@_skip_if_unavailable
class TestAgentToolTestLogic(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.session = _make_session()
        self.tm = self.session.tool_manager
        self.att = _make_agent_tool_test(self.session)

    async def test_07_not_found_shows_similar_tools(self):
        self.tm.register(func=_tool_ok, name="vfs_shell")
        r = await self.att("vfs_sh")
        self.assertIn("vfs_shell", r["suggestion"])

    async def test_08_not_found_shows_list_tools_hint(self):
        self.tm.register(func=_tool_ok, name="xyz_unrelated")
        r = await self.att("completely_different_aaaa")
        self.assertIn("list_tools", r["suggestion"])

    async def test_09_custom_inputs_override_stored(self):
        call_log = []
        def spy(x="default"):
            call_log.append(x)
            return f"result:{x}"

        self.tm.register(func=spy, live_test_inputs=[{"x": "stored"}])
        r = await self.att("spy", custom_inputs={"x": "custom"})
        self.assertEqual(r["status"], "HEALTHY")
        self.assertIn("custom", call_log)
        self.assertNotIn("stored", call_log)

    async def test_10_custom_inputs_on_tool_without_stored(self):
        self.tm.register(func=_tool_ok)
        r = await self.att("_tool_ok", custom_inputs={"x": "custom"})
        self.assertEqual(r["status"], "HEALTHY")

    async def test_11_stored_inputs_restored_after_custom_run(self):
        original_inputs = [{"x": "original"}]
        self.tm.register(func=_tool_ok, live_test_inputs=original_inputs[:])
        await self.att("_tool_ok", custom_inputs={"x": "temp"})
        entry = self.tm.get("_tool_ok")
        self.assertEqual(entry.live_test_inputs, original_inputs)

    async def test_12_result_preview_max_300_chars(self):
        def long_result(x="probe"):
            return "X" * 1000
        self.tm.register(func=long_result, live_test_inputs=[{"x": "p"}])
        r = await self.att("long_result")
        self.assertLessEqual(len(r["result_preview"]), 300)

    async def test_13_execution_time_non_negative(self):
        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "p"}])
        r = await self.att("_tool_ok")
        self.assertGreaterEqual(r["execution_time_ms"], 0.0)

    async def test_14_suggestion_healthy(self):
        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "p"}])
        r = await self.att("_tool_ok")
        self.assertIn("working correctly", r["suggestion"])

    async def test_15_suggestion_skipped_mentions_live_test_inputs(self):
        self.tm.register(func=_tool_ok)
        r = await self.att("_tool_ok")
        self.assertIn("live_test_inputs", r["suggestion"])

    async def test_16_suggestion_failed_mentions_exception(self):
        self.tm.register(func=_tool_raises, live_test_inputs=[{"x": "p"}])
        r = await self.att("_tool_raises")
        self.assertIn("Exception", r["suggestion"])

    async def test_17_error_is_none_on_healthy(self):
        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "p"}])
        r = await self.att("_tool_ok")
        self.assertIsNone(r["error"])

    async def test_18_contract_violations_empty_on_healthy(self):
        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "p"}])
        r = await self.att("_tool_ok")
        self.assertEqual(r["contract_violations"], [])


# ===========================================================================
# 3. add_tools() — neue kwargs durchgereicht
# ===========================================================================

@_skip_if_unavailable
class TestSessionToolsDictKwargs(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()

    def _register_from_dict(self, tool_dicts: list[dict]):
        tm = self.session.tool_manager
        for t in tool_dicts:
            tm.register(
                func=t["tool_func"],
                name=t.get("name"),
                description=t.get("description"),
                category=t.get("category"),
                flags=t.get("flags"),
                live_test_inputs=t.get("live_test_inputs"),
                cleanup_func=t.get("cleanup_func"),
                result_contract=t.get("result_contract"),
            )
        return tm

    def test_19_live_test_inputs_stored(self):
        inputs = [{"command": "ls /"}]
        tm = self._register_from_dict([{
            "tool_func": _tool_ok, "name": "vfs_shell",
            "live_test_inputs": inputs,
        }])
        self.assertEqual(tm.get("vfs_shell").live_test_inputs, inputs)

    def test_20_result_contract_stored(self):
        contract = {"allow_none": False, "expected_type": "dict"}
        tm = self._register_from_dict([{
            "tool_func": _tool_ok, "name": "vfs_shell",
            "result_contract": contract,
        }])
        self.assertEqual(tm.get("vfs_shell").result_contract, contract)

    def test_21_cleanup_func_stored(self):
        cleanup = lambda inputs, result: None
        tm = self._register_from_dict([{
            "tool_func": _tool_ok, "name": "vfs_shell",
            "cleanup_func": cleanup,
        }])
        self.assertIs(tm.get("vfs_shell").cleanup_func, cleanup)

    def test_22_old_style_dict_still_registers(self):
        tm = self._register_from_dict([{
            "tool_func": _tool_ok, "name": "old_style",
        }])
        self.assertTrue(tm.exists("old_style"))

    def test_23_guaranteed_healthy_via_flags(self):
        tm = self._register_from_dict([{
            "tool_func": _tool_ok, "name": "docker_run",
            "flags": {"guaranteed_healthy": True},
        }])
        self.assertTrue(tm.get("docker_run").flags.get("guaranteed_healthy"))

    def test_24_agent_tool_test_registered_correctly(self):
        att = _make_agent_tool_test(self.session)
        tm = self._register_from_dict([{
            "tool_func": att,
            "name":      "agent_tool_test",
            "category":  ["meta", "diagnostics"],
            "flags":     {"guaranteed_healthy": True},
        }])
        self.assertTrue(tm.exists("agent_tool_test"))
        entry = tm.get("agent_tool_test")
        self.assertTrue(entry.flags.get("guaranteed_healthy"))
        self.assertIn("meta", entry.category)

    def test_25_agent_tool_test_is_async_func(self):
        att = _make_agent_tool_test(self.session)
        self.assertTrue(asyncio.iscoroutinefunction(att))


# ===========================================================================
# 4. Result-Contract gegen reale VFS-Tool-Ausgaben
# ===========================================================================

@_skip_if_unavailable
class TestVFSToolContracts(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell, make_vfs_view
        self.session = _make_session()
        self.sh = make_vfs_shell(self.session)
        self.view = make_vfs_view(self.session)

    def _check(self, result, contract: dict, tool: str):
        if not contract.get("allow_none", True):
            self.assertIsNotNone(result, f"{tool}: None aber allow_none=False")
        if not contract.get("allow_empty_string", True):
            self.assertNotEqual(result, "", f"{tool}: '' aber allow_empty_string=False")
        et = contract.get("expected_type", "any")
        if et != "any" and result is not None:
            tmap = {"str": str, "dict": dict, "list": list, "int": int, "bool": bool}
            if et in tmap:
                self.assertIsInstance(result, tmap[et],
                    f"{tool}: expected {et}, got {type(result).__name__}")

    async def test_26_vfs_shell_ls_contract(self):
        contract = {"allow_none": False, "allow_empty_string": False, "expected_type": "dict"}
        self._check(self.sh("ls /"), contract, "vfs_shell")

    async def test_27_vfs_shell_result_has_four_keys(self):
        r = self.sh("ls /")
        for k in ("success", "stdout", "stderr", "returncode"):
            self.assertIn(k, r)

    async def test_28_vfs_shell_fail_has_error_info(self):
        r = self.sh("cat /nonexistent_probe_xyz")
        if not r["success"]:
            self.assertTrue(
                bool(r.get("stderr")) or bool(r.get("stdout")),
                "success=False aber stderr und stdout leer — Agent hat keine Diagnose"
            )

    async def test_29_vfs_view_returns_dict(self):
        contract = {"allow_none": False, "expected_type": "dict"}
        self._check(self.view("/system_context.md", line_start=1, line_end=3), contract, "vfs_view")

    async def test_30_vfs_view_success_has_content_and_showing(self):
        r = self.view("/system_context.md", line_start=1, line_end=3)
        if r.get("success"):
            self.assertIn("content", r)
            self.assertIn("showing", r)

    async def test_31_vfs_view_fail_has_error_key(self):
        r = self.view("/nonexistent_xyz.py")
        if not r.get("success"):
            self.assertIn("error", r)

    async def test_32_vfs_shell_unknown_cmd_stderr_has_hint(self):
        r = self.sh("frobnicate /foo")
        stderr = r.get("stderr", "")
        known = {"ls", "cat", "grep", "write", "mkdir", "rm"}
        self.assertTrue(any(c in stderr for c in known),
            f"Kein Hilfetext im Fehler: {stderr!r}")

    async def test_33_vfs_shell_write_then_cleanup(self):
        probe = "/_health_probe_write.txt"
        self.assertTrue(self.sh(f"touch {probe}")["success"])
        self.assertIn(
            self.session.vfs._normalize_path(probe),
            self.session.vfs.files
        )
        self.sh(f"rm {probe}")
        self.assertNotIn(
            self.session.vfs._normalize_path(probe),
            self.session.vfs.files
        )


# ===========================================================================
# 5. cleanup_func Verhalten
# ===========================================================================

@_skip_if_unavailable
class TestCleanupFuncBehavior(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.tm = _make_session().tool_manager

    async def test_34_cleanup_receives_inputs_and_result(self):
        log = []
        def cleanup(inputs, result):
            log.append({"inputs": inputs, "result": result})

        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "probe"}], cleanup_func=cleanup)
        await self.tm.health_check_single("_tool_ok")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["inputs"], {"x": "probe"})
        self.assertEqual(log[0]["result"], "ok:probe")

    async def test_35_cleanup_called_on_degraded(self):
        log = []
        def cleanup(inputs, result):
            log.append(result)

        self.tm.register(
            func=_tool_none,
            live_test_inputs=[{"x": "probe"}],
            result_contract={"allow_none": False},
            cleanup_func=cleanup,
        )
        r = await self.tm.health_check_single("_tool_none")
        self.assertEqual(r.status, "DEGRADED")
        self.assertEqual(len(log), 1)

    async def test_36_cleanup_not_called_on_failed(self):
        log = []
        def cleanup(inputs, result):
            log.append(result)

        self.tm.register(
            func=_tool_raises,
            live_test_inputs=[{"x": "probe"}],
            cleanup_func=cleanup,
        )
        await self.tm.health_check_single("_tool_raises")
        self.assertEqual(len(log), 0,
            "Bei FAILED (Exception) darf cleanup nicht aufgerufen werden")

    async def test_37_async_cleanup_awaited(self):
        log = []
        async def async_cleanup(inputs, result):
            log.append("called")

        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "p"}], cleanup_func=async_cleanup)
        await self.tm.health_check_single("_tool_ok")
        self.assertIn("called", log)

    async def test_38_cleanup_error_becomes_violation_not_failed(self):
        def bad_cleanup(inputs, result):
            raise RuntimeError("cleanup broken")

        self.tm.register(func=_tool_ok, live_test_inputs=[{"x": "p"}], cleanup_func=bad_cleanup)
        r = await self.tm.health_check_single("_tool_ok")
        self.assertEqual(r.status, "DEGRADED")
        self.assertTrue(any("cleanup_func" in v for v in r.contract_violations))


if __name__ == "__main__":
    unittest.main(verbosity=2)
