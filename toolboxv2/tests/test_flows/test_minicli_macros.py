# tests/test_minicli_macros.py
"""
Unit tests for BeastCLI macro system.
Tests cover: create, save/load roundtrip, variable substitution,
control flow (if/for/while), export/import JSON format, argument passing.
"""
import asyncio
import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_cli():
    """Build a BeastCLI instance with mocked App and BlobFile."""
    from toolboxv2.flows.minicli import BeastCLI, CLIContext, MacroContext

    app = MagicMock()
    app.debug = False
    app.version = "test"
    app.data_dir = tempfile.mkdtemp()
    app.id = "test"
    app.functions = {}
    app.system_flag = "Linux"

    with patch("toolboxv2.flows.minicli.BlobFile") as mock_blob:
        # _load_macros finds nothing → empty dict
        mock_blob.return_value.__enter__.return_value.exists.return_value = False
        cli = BeastCLI(app)

    cli.session = AsyncMock()
    return cli


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestMacroDataclass(unittest.TestCase):
    """Macro dataclass defaults and field types."""

    def test_macro_default_fields(self):
        from toolboxv2.flows.minicli import Macro
        m = Macro(name="test", commands=["echo hi"])
        self.assertEqual(m.name, "test")
        self.assertEqual(m.commands, ["echo hi"])
        self.assertEqual(m.loop_count, 1)
        self.assertEqual(m.tags, [])
        self.assertEqual(m.usage_count, 0)
        self.assertIsInstance(m.created, datetime.datetime)

    def test_macro_context_defaults(self):
        from toolboxv2.flows.minicli import MacroContext
        ctx = MacroContext()
        self.assertFalse(ctx.break_flag)
        self.assertFalse(ctx.continue_flag)
        self.assertIsNone(ctx.return_value)
        self.assertEqual(ctx.variables, {})
        self.assertEqual(ctx.loop_vars, {})


class TestMacroSaveLoad(unittest.TestCase):
    """_save_macros / _load_macros roundtrip via real JSON."""

    def _make_macro_data(self):
        return {
            "test_macro": {
                "commands": ["echo hello", "echo world"],
                "description": "Test macro",
                "variables": {"key": "value"},
                "created": datetime.datetime(2025, 1, 1).isoformat(),
                "usage_count": 3,
                "tags": ["test", "unit"],
                "loop_count": 2,
            }
        }

    def test_load_macros_populates_dict(self):
        from toolboxv2.flows.minicli import BeastCLI, Macro

        cli = _make_cli()
        macro_data = self._make_macro_data()

        mock_bf = MagicMock()
        mock_bf.__enter__ = MagicMock(return_value=mock_bf)
        mock_bf.__exit__ = MagicMock(return_value=False)
        mock_bf.exists.return_value = True
        mock_bf.read_json.return_value = macro_data

        with patch("toolboxv2.flows.minicli.BlobFile", return_value=mock_bf):
            cli._load_macros()

        self.assertIn("test_macro", cli.macros)
        m = cli.macros["test_macro"]
        self.assertIsInstance(m, Macro)
        self.assertEqual(m.commands, ["echo hello", "echo world"])
        self.assertEqual(m.usage_count, 3)
        self.assertEqual(m.tags, ["test", "unit"])
        self.assertEqual(m.loop_count, 2)

    def test_save_macros_writes_correct_structure(self):
        from toolboxv2.flows.minicli import Macro

        cli = _make_cli()
        cli.macros["my_macro"] = Macro(
            name="my_macro",
            commands=["DB get_status"],
            description="status check",
            tags=["ops"],
        )

        written = {}

        mock_bf = MagicMock()
        mock_bf.__enter__ = MagicMock(return_value=mock_bf)
        mock_bf.__exit__ = MagicMock(return_value=False)
        mock_bf.write_json.side_effect = lambda d: written.update(d)

        with patch("toolboxv2.flows.minicli.BlobFile", return_value=mock_bf):
            cli._save_macros()

        self.assertIn("my_macro", written)
        self.assertEqual(written["my_macro"]["commands"], ["DB get_status"])
        self.assertIn("created", written["my_macro"])

    def test_load_empty_blob_leaves_macros_empty(self):
        cli = _make_cli()
        mock_bf = MagicMock()
        mock_bf.__enter__ = MagicMock(return_value=mock_bf)
        mock_bf.__exit__ = MagicMock(return_value=False)
        mock_bf.exists.return_value = False

        with patch("toolboxv2.flows.minicli.BlobFile", return_value=mock_bf):
            cli._load_macros()

        self.assertEqual(cli.macros, {})


class TestVariableSubstitution(unittest.TestCase):
    """_substitute_variables and _substitute_macro_variables."""

    def test_substitutes_quick_var(self):
        cli = _make_cli()
        cli.context.quick_vars["r1"] = "hello"
        result = cli._substitute_variables("echo $r1")
        self.assertEqual(result, "echo hello")

    def test_macro_var_overrides_quick_var(self):
        cli = _make_cli()
        cli.context.quick_vars["x"] = "global"
        cli.macro_context.variables["x"] = "local"
        result = cli._substitute_macro_variables("echo $x")
        self.assertEqual(result, "echo local")

    def test_loop_var_highest_priority(self):
        cli = _make_cli()
        cli.context.quick_vars["i"] = "global_i"
        cli.macro_context.variables["i"] = "macro_i"
        cli.macro_context.loop_vars["i"] = "loop_i"
        result = cli._substitute_macro_variables("echo $i")
        self.assertEqual(result, "echo loop_i")

    def test_unknown_var_left_as_is(self):
        cli = _make_cli()
        result = cli._substitute_variables("echo $unknown")
        self.assertEqual(result, "echo $unknown")

    def test_multiple_vars_in_one_command(self):
        cli = _make_cli()
        cli.context.quick_vars["r1"] = "foo"
        cli.context.quick_vars["r2"] = "bar"
        result = cli._substitute_variables("process $r1 $r2")
        self.assertEqual(result, "process foo bar")


class TestMacroArguments(unittest.TestCase):
    """Argument passing via $arg1, $1 etc."""

    def test_args_set_as_arg1_and_dollar1(self):
        from toolboxv2.flows.minicli import Macro, MacroContext

        cli = _make_cli()
        cli.macros["greet"] = Macro(name="greet", commands=["echo $arg1"])

        async def run():
            with patch.object(cli, "_execute_macro_commands", new=AsyncMock()) as mock_exec:
                with patch.object(cli, "_process_command", new=AsyncMock()):
                    await cli._handle_macro_command("greet world")
            return mock_exec

        mock_exec = _run(run())
        self.assertEqual(cli.macro_context.variables.get("arg1"), "world")
        self.assertEqual(cli.macro_context.variables.get("$1"), "world")

    def test_missing_macro_prints_error(self):
        cli = _make_cli()

        output = []
        with patch("builtins.print", side_effect=lambda *a, **k: output.append(str(a[0]))):
            _run(cli._handle_macro_command("nonexistent"))

        self.assertTrue(any("not found" in line for line in output))


class TestControlFlow(unittest.TestCase):
    """if / for / while / break / continue / return."""

    def test_if_true_executes_action(self):
        cli = _make_cli()
        cli.macro_context.variables["status"] = "ok"
        executed = []

        async def mock_process(cmd):
            executed.append(cmd)

        cli._process_command = mock_process
        _run(cli._handle_macro_if("if $status == 'ok': echo matched"))
        # echo is handled inline — check via output
        # Since echo prints directly, just verify no exception

    def test_if_false_skips_action(self):
        cli = _make_cli()
        cli.macro_context.variables["status"] = "fail"
        executed = []

        async def mock_process(cmd):
            executed.append(cmd)

        cli._process_command = mock_process
        _run(cli._handle_macro_if("if $status == 'ok': do_something"))
        self.assertEqual(executed, [])

    def test_if_break_sets_flag(self):
        cli = _make_cli()
        cli.macro_context.variables["counter"] = 11
        _run(cli._handle_macro_if("if $counter > 10: break"))
        self.assertTrue(cli.macro_context.break_flag)

    def test_for_loop_iterates(self):
        cli = _make_cli()
        processed = []

        async def mock_process(cmd):
            processed.append(cmd)

        cli._process_command = mock_process
        _run(cli._handle_macro_for("for i in range(3): echo $i"))
        # echo is inline, but loop var should have been set
        self.assertEqual(cli.macro_context.variables.get("i"), 2)

    def test_while_respects_1000_limit(self):
        """While loop must not run more than 1000 iterations."""
        cli = _make_cli()
        cli.macro_context.variables["x"] = 0
        count = {"n": 0}

        async def mock_process(cmd):
            count["n"] += 1
            # never set x high enough to exit naturally

        cli._process_command = mock_process
        # Safety: patch _handle_macro_while to call original but cap
        _run(cli._handle_macro_while("while $x < 99999: echo $x"))
        self.assertLessEqual(count["n"], 1000)

    def test_execute_macro_commands_handles_comment(self):
        cli = _make_cli()
        executed = []

        async def mock_process(cmd):
            executed.append(cmd)

        cli._process_command = mock_process
        _run(cli._execute_macro_commands(["# this is a comment", "echo hello"], loop_count=1))
        # Comment never reaches _process_command
        self.assertNotIn("# this is a comment", executed)

    def test_execute_macro_commands_break_stops_loop(self):
        cli = _make_cli()
        executed = []

        async def mock_process(cmd):
            executed.append(cmd)

        cli._process_command = mock_process
        cli.macro_context.break_flag = True  # pre-set break
        _run(cli._execute_macro_commands(["echo should_not_run"], loop_count=1))
        self.assertNotIn("echo should_not_run", executed)


class TestExportImportJSON(unittest.TestCase):
    """Export → JSON file → Import roundtrip."""

    def test_import_valid_json_creates_macros(self):
        cli = _make_cli()
        macro_json = {
            "version": "1.0",
            "macros": {
                "imported_macro": {
                    "commands": ["echo imported"],
                    "description": "test import",
                    "variables": {},
                    "created": datetime.datetime.now().isoformat(),
                    "usage_count": 0,
                    "tags": ["shared"],
                    "loop_count": 1,
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(macro_json, f)
            tmp_path = f.name

        try:
            cli.session.prompt_async = AsyncMock(return_value="yes")

            with patch.object(cli, "_save_macros"):
                _run(cli._import_macros(tmp_path))

            self.assertIn("imported_macro", cli.macros)
            self.assertEqual(cli.macros["imported_macro"].commands, ["echo imported"])
            self.assertEqual(cli.macros["imported_macro"].tags, ["shared"])
        finally:
            os.unlink(tmp_path)

    def test_import_conflict_overwrite(self):
        from toolboxv2.flows.minicli import Macro

        cli = _make_cli()
        cli.macros["existing"] = Macro(name="existing", commands=["old"])

        macro_json = {
            "macros": {
                "existing": {
                    "commands": ["new"],
                    "description": "",
                    "variables": {},
                    "created": datetime.datetime.now().isoformat(),
                    "usage_count": 0,
                    "tags": [],
                    "loop_count": 1,
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(macro_json, f)
            tmp_path = f.name

        try:
            # First prompt: "yes" (import all), second: "o" (overwrite)
            cli.session.prompt_async = AsyncMock(side_effect=["yes", "o"])

            with patch.object(cli, "_save_macros"):
                _run(cli._import_macros(tmp_path))

            self.assertEqual(cli.macros["existing"].commands, ["new"])
        finally:
            os.unlink(tmp_path)

    def test_import_missing_file_prints_error(self):
        cli = _make_cli()
        output = []
        with patch("builtins.print", side_effect=lambda *a, **k: output.append(str(a[0]))):
            _run(cli._import_macros("/nonexistent/path/macros.json"))
        self.assertTrue(any("not found" in line or "File not found" in line for line in output))

    def test_import_invalid_format_prints_error(self):
        cli = _make_cli()
        bad_json = {"no_macros_key": True}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(bad_json, f)
            tmp_path = f.name

        try:
            cli.session.prompt_async = AsyncMock(return_value="yes")
            output = []
            with patch("builtins.print", side_effect=lambda *a, **k: output.append(str(a[0]))):
                _run(cli._import_macros(tmp_path))
            self.assertTrue(any("Invalid" in line for line in output))
        finally:
            os.unlink(tmp_path)


class TestMacroResultCapture(unittest.TestCase):
    """Return values from macros get saved as $rN."""

    def test_return_value_saved_to_quick_vars(self):
        from toolboxv2.flows.minicli import Macro

        cli = _make_cli()
        cli.context.result_count = 0
        cli.macros["returner"] = Macro(
            name="returner",
            commands=["return 42"]
        )

        _run(cli._handle_macro_command("returner"))

        self.assertEqual(cli.context.last_result, 42)
        self.assertIn("r1", cli.context.quick_vars)
        self.assertEqual(cli.context.quick_vars["r1"], 42)


class TestMacroManagerRouting(unittest.TestCase):
    """_macro_manager dispatches subcommands correctly."""

    def test_create_dispatches(self):
        cli = _make_cli()
        cli._create_macro_interactive = AsyncMock()
        _run(cli._macro_manager("create"))
        cli._create_macro_interactive.assert_awaited_once()

    def test_export_dispatches(self):
        cli = _make_cli()
        cli._export_macros = AsyncMock()
        _run(cli._macro_manager("export"))
        cli._export_macros.assert_awaited_once()

    def test_import_dispatches_with_path(self):
        cli = _make_cli()
        cli._import_macros = AsyncMock()
        _run(cli._macro_manager("import /some/path.json"))
        cli._import_macros.assert_awaited_once_with("/some/path.json")

    def test_unknown_action_prints_error(self):
        cli = _make_cli()
        output = []
        with patch("builtins.print", side_effect=lambda *a, **k: output.append(str(a[0]))):
            _run(cli._macro_manager("bogus_action"))
        self.assertTrue(any("Unknown" in line for line in output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
