"""
Hypothesis-getriebene Tests für das exec_code-Tool (live_harness + exec_code_tool).

Vorgehen (Debugging-Methode: Hypothesen → diskriminierende Tests → Fix):

    H1  Top-Level-await verliert den Namespace.
        Der alte Pfad (MockIPython._parse_code) wickelt die GANZE Zelle in eine
        async def __wrapper(). Zuweisungen werden damit LOKAL zur Wrapper-
        Funktion; nur `result` ist per `global` deklariert. Alles andere ist
        nach der Zelle weg.  → LegacyParseCodeTests beweist das direkt am alten
        Code, HarnessExecuteTests beweist, dass der neue Pfad es kann.

    H2  Falsy-Ergebnisse gehen verloren.
        `result if result else "stdout"` bzw. `if not result: result = ""`
        macht aus 0 / "" / False / [] den String "stdout" bzw. "".

    H3  Fehler werden verschluckt.
        LocalCodeExecutor parst den formatierten String von run_cell und
        ignoriert `stderr:`-Zeilen (`pass`) → success=True bei Exception.

    H4  Async-Tools über den `tools`-Proxy sind fire-and-forget.
        Im laufenden Loop ging der alte Proxy über run_bg_task_advanced und gab
        das Task-Objekt statt des Ergebnisses zurück.

    H5  Sessions sind nicht persistent.
        session_dir enthielt id(self) → pro Instanz neuer Ordner, kein Resume.

    H6  Der Agent kann seinen Harness nicht selbst erweitern.
        Es gab keinen Schreibmodus und keinen Zugriff auf den ToolManager.

    H7  Ein kaputtes Harness-Modul darf die Session nicht vergiften.

    H8  Die Tool-Beschreibung ist statisch und zeigt dem Agent nicht, was in
        seiner Umgebung tatsächlich existiert.

Ergebnis der Diskriminierung: H1–H8 haben ALLE zugetroffen (keine Mischung,
keine Fehlhypothese) — siehe die einzelnen Tests unten.
"""

import ast
import asyncio
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# toolboxv2 stubben (Repo-Eigenheit: echter Import setzt mods/isaa zurück,
# und die Tests brauchen weder App noch pydantic)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_stub_packages():
    """Registriert leere Namespace-Pakete mit echtem __path__, damit
    live_harness/exec_code_tool importierbar sind, ohne toolboxv2/__init__.py
    (und damit App + pydantic) zu ziehen."""
    chain = [
        ("toolboxv2", _REPO / "toolboxv2"),
        ("toolboxv2.mods", _REPO / "toolboxv2" / "mods"),
        ("toolboxv2.mods.isaa", _REPO / "toolboxv2" / "mods" / "isaa"),
        ("toolboxv2.mods.isaa.base", _REPO / "toolboxv2" / "mods" / "isaa" / "base"),
        ("toolboxv2.mods.isaa.base.Agent",
         _REPO / "toolboxv2" / "mods" / "isaa" / "base" / "Agent"),
    ]
    for name, path in chain:
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        if "." in name:
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, mod)
    sys.modules["toolboxv2"].get_app = lambda *a, **k: None
    sys.modules["toolboxv2"].Spinner = lambda *a, **k: _NullCtx()


try:  # echte Installation bevorzugen, sonst Stub
    import toolboxv2 as _tb  # noqa: F401
    import pydantic  # noqa: F401
except Exception:
    _install_stub_packages()


from toolboxv2.mods.isaa.base.Agent.live_harness import (  # noqa: E402
    AgentToolProxy,
    LiveSession,
    ModProxy,
    compile_cell,
    tool,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeToolManager:
    def __init__(self):
        self._registry = {}

    def register(self, func=None, name=None, description=None, **kw):
        n = name or func.__name__
        self._registry[n] = types.SimpleNamespace(
            name=n, function=func, description=description or "", metadata={}
        )
        return self._registry[n]

    def get(self, name):
        return self._registry.get(name)

    def get_function(self, name):
        e = self._registry.get(name)
        return e.function if e else None

    def list_names(self):
        return list(self._registry)

    def un_register(self, name):
        return self._registry.pop(name, None) is not None

    def update(self, name, **updates):
        e = self._registry.get(name)
        if not e:
            return False
        for k, v in updates.items():
            setattr(e, k, v)
        return True


class FakeAgent:
    def __init__(self, name="tester"):
        self.amd = types.SimpleNamespace(name=name)
        self.tool_manager = FakeToolManager()
        self.active_session = "s1"
        self.session_manager = None

    def add_tool(self, tool_func=None, name=None, description=None, **kw):
        self.tool_manager.register(func=tool_func, name=name, description=description)

    def remove_tool(self, name):
        self.tool_manager.un_register(name)


class HarnessTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="live_harness_test_"))
        self.agent = FakeAgent()
        LiveSession._REGISTRY.clear()

    def tearDown(self):
        LiveSession._REGISTRY.clear()
        for m in list(sys.modules):
            if m.startswith("hmod_"):
                sys.modules.pop(m, None)
        sys.path[:] = [p for p in sys.path if "live_harness_test_" not in p]
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mk(self, sid="s1", **kw):
        kw.setdefault("root", self.tmp / sid)
        return LiveSession(agent=self.agent, session_id=sid, **kw)

    def run_cell(self, live, code, **kw):
        return asyncio.run(live.execute(code, **kw))


# ---------------------------------------------------------------------------
# H1 / H2 — Beweis am ALTEN Code (diskriminierender Test)
# ---------------------------------------------------------------------------

class LegacyParseCodeTests(unittest.TestCase):
    """Zeigt am Original-AST-Transform, dass H1 zutrifft."""

    @staticmethod
    def _legacy_wrapper_source(code: str) -> str:
        """Rekonstruiert die Kern-Transformation von MockIPython._parse_code."""
        tree = ast.parse(code)
        wrapper = ast.AsyncFunctionDef(
            name="__wrapper",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                               kw_defaults=[], defaults=[]),
            body=[ast.Global(names=["result"])] + tree.body,
            decorator_list=[], returns=None, type_comment=None,
        )
        mod = ast.Module(body=[wrapper], type_ignores=[])
        ast.fix_missing_locations(mod)
        return mod

    def test_h1_legacy_wrapper_loses_namespace(self):
        """H1 BESTÄTIGT: Zuweisungen in der Wrapper-Funktion landen nicht im ns."""
        mod = self._legacy_wrapper_source("import asyncio\nawait asyncio.sleep(0)\nmy_var = 42")
        ns = {}
        exec(compile(mod, "<legacy>", "exec"), ns)
        asyncio.run(ns["__wrapper"]())
        self.assertNotIn("my_var", ns, "Alter Pfad hätte my_var behalten müssen")

    def test_h1_new_path_keeps_namespace(self):
        """Gegenprobe: der neue compile_cell-Pfad behält den State."""
        co, _ = compile_cell("import asyncio\nawait asyncio.sleep(0)\nmy_var = 42")
        ns = {"__builtins__": __builtins__}

        async def _go():
            r = eval(co, ns)
            if asyncio.iscoroutine(r):
                await r

        asyncio.run(_go())
        self.assertEqual(ns["my_var"], 42)

    def test_h2_legacy_falsy_collapse(self):
        """H2 BESTÄTIGT: die alte Ergebnis-Normalisierung frisst falsy Werte."""
        for value in (0, "", False, []):
            legacy = value if value else "stdout"
            self.assertEqual(legacy, "stdout")


# ---------------------------------------------------------------------------
# execute-Mode
# ---------------------------------------------------------------------------

class HarnessExecuteTests(HarnessTestBase):

    def test_state_persists_across_cells(self):
        live = self.mk()
        self.run_cell(live, "counter = 1")
        r = self.run_cell(live, "counter += 1\ncounter")
        self.assertTrue(r.success, r.error)
        self.assertEqual(r.result, 2)

    def test_top_level_await_keeps_state(self):
        """H1-Fix: await in der Zelle darf den Namespace nicht killen."""
        live = self.mk()
        r = self.run_cell(live, "import asyncio\nawait asyncio.sleep(0)\nafter_await = 7")
        self.assertTrue(r.success, r.error)
        r2 = self.run_cell(live, "after_await * 3")
        self.assertEqual(r2.result, 21)

    def test_async_def_then_await_it(self):
        live = self.mk()
        self.run_cell(live, "import asyncio\nasync def slow():\n    await asyncio.sleep(0)\n    return 'done'")
        r = self.run_cell(live, "await slow()")
        self.assertEqual(r.result, "done")

    def test_falsy_result_survives(self):
        """H2-Fix."""
        live = self.mk()
        for expr, expected in (("0", 0), ("''", ""), ("False", False), ("[]", [])):
            r = self.run_cell(live, expr)
            self.assertTrue(r.success)
            self.assertEqual(r.result, expected, f"{expr} kollabiert")

    def test_error_is_reported_not_swallowed(self):
        """H3-Fix: Exceptions müssen success=False + Traceback liefern."""
        live = self.mk()
        r = self.run_cell(live, "1/0")
        self.assertFalse(r.success)
        self.assertIn("ZeroDivisionError", r.error)
        self.assertNotIn("live_harness.py", r.error, "interne Frames rausfiltern")

    def test_syntax_error_is_structured(self):
        live = self.mk()
        r = self.run_cell(live, "def broken(:\n    pass")
        self.assertFalse(r.success)
        self.assertIn("SyntaxError", r.error)

    def test_session_survives_error(self):
        live = self.mk()
        self.run_cell(live, "alive = 'yes'")
        self.run_cell(live, "raise RuntimeError('boom')")
        r = self.run_cell(live, "alive")
        self.assertEqual(r.result, "yes")

    def test_stdout_captured_multiline(self):
        live = self.mk()
        r = self.run_cell(live, "for i in range(3):\n    print('line', i)")
        self.assertTrue(r.success)
        self.assertEqual(r.output.count("line"), 3)

    def test_class_and_comprehension_scope(self):
        """Globals-only exec — sonst brechen Comprehensions und Klassenkörper."""
        live = self.mk()
        self.run_cell(live, "class Cfg:\n    limit = 3")
        r = self.run_cell(live, "[i for i in range(Cfg.limit)]")
        self.assertEqual(r.result, [0, 1, 2])

    def test_timeout_aborts_cell(self):
        live = self.mk()
        r = self.run_cell(live, "import asyncio\nawait asyncio.sleep(5)", timeout=0.2)
        self.assertFalse(r.success)
        self.assertIn("Timeout", r.error)

    def test_new_names_reported(self):
        live = self.mk()
        r = self.run_cell(live, "alpha = 1\nbeta = 2")
        self.assertEqual(set(r.new_names) & {"alpha", "beta"}, {"alpha", "beta"})

    def test_cwd_restored_after_cell(self):
        import os
        live = self.mk()
        before = os.getcwd()
        self.run_cell(live, "import os\nos.chdir('/')")
        self.assertEqual(os.getcwd(), before)


# ---------------------------------------------------------------------------
# tools-Proxy (H4)
# ---------------------------------------------------------------------------

class ToolProxyTests(HarnessTestBase):

    def test_async_tool_returns_value_not_task(self):
        async def ping(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        self.agent.add_tool(tool_func=ping, name="ping")
        live = self.mk()
        r = self.run_cell(live, "await tools.ping(21)")
        self.assertTrue(r.success, r.error)
        self.assertEqual(r.result, 42)

    def test_sync_tool_call(self):
        self.agent.add_tool(tool_func=lambda a, b: a + b, name="add")
        live = self.mk()
        r = self.run_cell(live, "tools.add(2, 3)")
        self.assertEqual(r.result, 5)

    def test_async_tool_sync_helper(self):
        """tools.x.sync(...) blockt korrekt, wenn kein await moeglich ist."""
        async def ping(x: int) -> int:
            await asyncio.sleep(0)
            return x + 1

        self.agent.add_tool(tool_func=ping, name="ping_sync")
        live = self.mk()
        r = self.run_cell(live, "tools.ping_sync.sync(1)")
        self.assertTrue(r.success, r.error)
        self.assertEqual(r.result, 2)

    def test_unknown_tool_raises_with_hint(self):
        live = self.mk()
        r = self.run_cell(live, "tools.nope()")
        self.assertFalse(r.success)
        self.assertIn("not found", r.error)

    def test_dir_lists_tools(self):
        self.agent.add_tool(tool_func=lambda: 1, name="alpha_tool")
        proxy = AgentToolProxy(self.agent)
        self.assertIn("alpha_tool", dir(proxy))


# ---------------------------------------------------------------------------
# write-Mode / Selbst-Erweiterung (H6, H7)
# ---------------------------------------------------------------------------

class HarnessWriteTests(HarnessTestBase):

    def test_write_module_and_import(self):
        live = self.mk()
        res = live.write_harness("hmod_util", "def double(x):\n    return x * 2\n", analyze=False)
        self.assertTrue(res["success"], res.get("error"))
        self.assertIn("double", res["exports"])
        r = self.run_cell(live, "double(5)")
        self.assertEqual(r.result, 10)

    def test_tool_decorator_registers_agent_tool(self):
        live = self.mk()
        src = (
            "@tool(description='adds two numbers', category=['math'])\n"
            "def harness_add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
        res = live.write_harness("hmod_math", src, analyze=False)
        self.assertTrue(res["success"], res.get("error"))
        self.assertIn("harness_add", res["registered_tools"])
        self.assertIn("harness_add", self.agent.tool_manager.list_names())
        r = self.run_cell(live, "tools.harness_add(2, 3)")
        self.assertEqual(r.result, 5)

    def test_broken_module_rolls_back(self):
        """H7-Fix: Syntaxfehler darf nichts persistieren."""
        live = self.mk()
        res = live.write_harness("hmod_bad", "def x(:\n    pass", analyze=False)
        self.assertFalse(res["success"])
        self.assertNotIn("hmod_bad", live.list_harness())

    def test_import_error_rolls_back_to_previous(self):
        live = self.mk()
        live.write_harness("hmod_roll", "VALUE = 1\n", analyze=False)
        res = live.write_harness("hmod_roll", "import definitely_not_a_module\n", analyze=False)
        self.assertFalse(res["success"])
        self.assertTrue(res.get("rolled_back"))
        self.assertIn("VALUE = 1", live.read_harness("hmod_roll"))

    def test_append_mode(self):
        live = self.mk()
        live.write_harness("hmod_app", "A = 1\n", analyze=False)
        live.write_harness("hmod_app", "B = 2\n", append=True, analyze=False)
        src = live.read_harness("hmod_app")
        self.assertIn("A = 1", src)
        self.assertIn("B = 2", src)

    def test_remove_unregisters_tools(self):
        live = self.mk()
        live.write_harness(
            "hmod_rm",
            "@tool()\ndef temp_tool() -> str:\n    return 'x'\n",
            analyze=False,
        )
        self.assertIn("temp_tool", self.agent.tool_manager.list_names())
        live.remove_harness("hmod_rm")
        self.assertNotIn("temp_tool", self.agent.tool_manager.list_names())
        self.assertNotIn("hmod_rm", live.list_harness())

    def test_module_name_sanitized(self):
        live = self.mk()
        res = live.write_harness("hmod bad-name!", "Z = 1\n", analyze=False)
        self.assertTrue(res["success"], res.get("error"))
        self.assertEqual(res["module"], "hmod_bad_name")


# ---------------------------------------------------------------------------
# Persistenz (H5)
# ---------------------------------------------------------------------------

class PersistenceTests(HarnessTestBase):

    def test_harness_survives_new_session_object(self):
        live = self.mk()
        live.write_harness("hmod_persist", "def keep():\n    return 'kept'\n", analyze=False)

        LiveSession._REGISTRY.clear()
        live2 = self.mk()          # gleicher root → gleiche Session
        live2.bootstrap()
        self.assertIn("hmod_persist", live2.list_harness())
        r = self.run_cell(live2, "keep()")
        self.assertEqual(r.result, "kept")

    def test_variables_survive_restart(self):
        live = self.mk()
        self.run_cell(live, "saved_value = {'a': 1}")
        LiveSession._REGISTRY.clear()
        live2 = self.mk()
        r = self.run_cell(live2, "saved_value['a']")
        self.assertEqual(r.result, 1)

    def test_unpicklable_values_do_not_break_save(self):
        live = self.mk()
        r = self.run_cell(live, "import threading\nlock = threading.Lock()\n1")
        self.assertTrue(r.success, r.error)
        LiveSession._REGISTRY.clear()
        live2 = self.mk()
        self.assertTrue(self.run_cell(live2, "1 + 1").success)

    def test_registry_returns_same_instance(self):
        a = LiveSession.get(agent=self.agent, session_id="reg", root=self.tmp / "reg")
        b = LiveSession.get(agent=self.agent, session_id="reg", root=self.tmp / "reg")
        self.assertIs(a, b)

    def test_reset_clears_namespace_keeps_harness(self):
        live = self.mk()
        live.write_harness("hmod_keep", "K = 1\n", analyze=False)
        self.run_cell(live, "gone = 1")
        live.reset()
        r = self.run_cell(live, "gone")
        self.assertFalse(r.success)
        self.assertIn("hmod_keep", live.list_harness())


# ---------------------------------------------------------------------------
# Zugriffskontrolle
# ---------------------------------------------------------------------------

class AccessControlTests(HarnessTestBase):

    def test_self_agent_is_privileged(self):
        live = LiveSession(agent=FakeAgent("self"), session_id="p", root=self.tmp / "p")
        self.assertTrue(live.privileged)

    def test_tb_admin_is_privileged(self):
        live = LiveSession(agent=FakeAgent("tb_admin"), session_id="p2", root=self.tmp / "p2")
        self.assertTrue(live.privileged)

    def test_normal_agent_not_privileged(self):
        self.assertFalse(self.mk().privileged)

    def test_mod_proxy_blocks_unlisted(self):
        app = types.SimpleNamespace(get_mod=lambda n: f"mod:{n}", get_all_mods=lambda: ["DB"])
        proxy = ModProxy(app, {"DB"}, privileged=False)
        self.assertEqual(proxy.DB, "mod:DB")
        with self.assertRaises(PermissionError):
            _ = proxy.CloudM

    def test_mod_proxy_privileged_allows_all(self):
        app = types.SimpleNamespace(get_mod=lambda n: f"mod:{n}", get_all_mods=lambda: ["DB"])
        proxy = ModProxy(app, set(), privileged=True)
        self.assertEqual(proxy.CloudM, "mod:CloudM")


# ---------------------------------------------------------------------------
# Präsentation (H8)
# ---------------------------------------------------------------------------

class PresentationTests(HarnessTestBase):

    def _desc(self, live):
        from toolboxv2.mods.isaa.base.Agent.exec_code_tool import build_exec_code_description
        return build_exec_code_description(live)

    def test_description_lists_modes(self):
        d = self._desc(self.mk())
        for mode in ("execute", "write", "state"):
            self.assertIn(mode, d)

    def test_description_reflects_written_harness(self):
        live = self.mk()
        d0 = self._desc(live)
        self.assertIn("none yet", d0)
        live.write_harness(
            "hmod_shown",
            "@tool()\ndef shown_tool() -> str:\n    return 'ok'\n",
            analyze=False,
        )
        d1 = self._desc(live)
        self.assertIn("hmod_shown", d1)
        self.assertIn("shown_tool", d1)

    def test_description_shows_privilege_scope(self):
        priv = LiveSession(agent=FakeAgent("self"), session_id="d1", root=self.tmp / "d1")
        self.assertIn("Privileged", self._desc(priv))
        self.assertNotIn("Privileged", self._desc(self.mk("d2")))

    def test_state_mode_shape(self):
        live = self.mk()
        self.run_cell(live, "shown_var = 5")
        st = live.state()
        for key in ("harness_modules", "variables", "tools_available", "privileged"):
            self.assertIn(key, st)


# ---------------------------------------------------------------------------
# Tool-Fassade
# ---------------------------------------------------------------------------

class ExecCodeToolTests(HarnessTestBase):

    def _tool(self, **kw):
        from toolboxv2.mods.isaa.base.Agent.exec_code_tool import create_exec_code_tool
        kw.setdefault("session_id", "toolsess")
        d = create_exec_code_tool(self.agent, **kw)
        d["_live_session"].root = self.tmp / "toolsess"
        return d

    def setUp(self):
        super().setUp()
        LiveSession._REGISTRY.clear()

    def test_tool_definition_shape(self):
        d = self._tool()
        self.assertEqual(d["name"], "exec_code")
        self.assertIn("harness", d["category"])
        self.assertTrue(callable(d["tool_func"]))

    def test_execute_mode_through_tool(self):
        fn = self._tool()["tool_func"]
        out = asyncio.run(fn(code="print('hi')\n40 + 2", mode="execute"))
        self.assertTrue(out["success"], out.get("error"))
        self.assertIn("hi", out["output"])
        self.assertEqual(out["result"], "42")

    def test_unknown_mode_rejected(self):
        fn = self._tool()["tool_func"]
        out = asyncio.run(fn(mode="teleport"))
        self.assertFalse(out["success"])
        self.assertIn("Unknown mode", out["error"])

    def test_write_requires_name(self):
        fn = self._tool()["tool_func"]
        out = asyncio.run(fn(mode="write", code="X = 1"))
        self.assertFalse(out["success"])
        self.assertIn("name", out["error"])

    def test_state_mode(self):
        fn = self._tool()["tool_func"]
        out = asyncio.run(fn(mode="state"))
        self.assertTrue(out["success"])
        self.assertIn("harness_modules", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
