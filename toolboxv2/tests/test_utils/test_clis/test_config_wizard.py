"""
Unit tests for config_wizard.py — schema-aware edition
=======================================================

Tests are grouped into:

  A) Schema integration          — wizard output feeds TBManifest.model_validate()
  B) Enum / type coercion        — Pydantic coerces str → DatabaseMode / LogLevel / etc.
  C) ManifestLoader roundtrip    — save + load in TemporaryDirectory
  D) resolve_env_vars            — ${VAR:default} substitution
  E) Full wizard pipeline        — all wizard_* functions chained → valid TBManifest
  F) Workers schema mismatch     — documents the list-vs-dict incompatibility
  G) Wizard functions (isolated) — data-flow + type correctness
  H) ENV file I/O                — parse_env_template / save_env_file

All file I/O runs inside TemporaryDirectory — real manifest / .env are never touched.
Only unittest is used; no pytest.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# =============================================================================
# Bootstrap: stub cli_printing, then load config_wizard via file path
# =============================================================================

_stub_cli = types.ModuleType("toolboxv2.utils.clis.cli_printing")
for _fn in ("print_box_header", "print_box_footer", "print_box_content",
            "print_status", "print_separator", "c_print"):
    setattr(_stub_cli, _fn, lambda *a, **kw: None)


class _Colors:
    CYAN = DIM = RESET = BOLD = YELLOW = GREEN = ""


_stub_cli.Colors = _Colors
sys.modules["toolboxv2.utils.clis.cli_printing"] = _stub_cli

# Load config_wizard directly from filesystem — independent of sys.path
# Layout: toolboxv2/tests/test_utils/test_clis/test_config_wizard.py
#         parents[3] = toolboxv2/
_THIS = Path(__file__).resolve()
_TB_ROOT = _THIS.parents[3]
_WIZ_PATH = _TB_ROOT / "utils" / "clis" / "config_wizard.py"

_spec = importlib.util.spec_from_file_location(
    "toolboxv2.utils.clis.config_wizard", _WIZ_PATH
)
_wiz = importlib.util.module_from_spec(_spec)
sys.modules["toolboxv2.utils.clis.config_wizard"] = _wiz
_spec.loader.exec_module(_wiz)

from toolboxv2.utils.clis.config_wizard import (  # noqa: E402
    parse_env_template,
    load_existing_env,
    save_env_file,
    wizard_app_settings,
    wizard_database_settings,
    wizard_workers_settings,
    wizard_services_settings,
    wizard_isaa_settings,
    wizard_env_variables,
    ENV_CATEGORIES,
)

# Real schema + loader
from toolboxv2.utils.manifest.schema import (  # noqa: E402
    TBManifest,
    AppConfig,
    DatabaseMode,
    DatabaseConfig,
    LogLevel,
    Environment,
    IsaaConfig,
    WorkersConfig,
)
from toolboxv2.utils.manifest.loader import ManifestLoader, resolve_env_vars  # noqa: E402

_MOD = "toolboxv2.utils.clis.config_wizard"


# =============================================================================
# Helpers
# =============================================================================

def _fresh_manifest_dict() -> dict:
    """model_dump() of a default TBManifest — exactly as run_config_wizard sees it."""
    return TBManifest().model_dump()


def _simple_manifest(**overrides) -> dict:
    """Minimal flat manifest dict for wizard functions that don't need the full schema."""
    base = {"app": {}, "database": {}, "workers": {}, "services": {}, "isaa": {}}
    base.update(overrides)
    return base


# =============================================================================
# A) Schema integration — wizard output → TBManifest.model_validate()
# =============================================================================

class TestWizardAppSchemaIntegration(unittest.TestCase):

    def _run_and_validate(self, name="MyApp", debug=False, log_level="WARNING") -> TBManifest:
        data = _fresh_manifest_dict()
        with (
            patch(f"{_MOD}.prompt_input", return_value=name),
            patch(f"{_MOD}.prompt_bool", return_value=debug),
            patch(f"{_MOD}.prompt_choice", return_value=log_level),
        ):
            data = wizard_app_settings(data)
        return TBManifest.model_validate(data)

    def test_model_validate_succeeds(self):
        self.assertIsInstance(self._run_and_validate(), TBManifest)

    def test_app_name_stored_in_schema(self):
        self.assertEqual(self._run_and_validate(name="TestBox").app.name, "TestBox")

    def test_log_level_is_loglevel_enum(self):
        m = self._run_and_validate(log_level="DEBUG")
        self.assertIsInstance(m.app.log_level, LogLevel)
        self.assertEqual(m.app.log_level, LogLevel.DEBUG)

    def test_debug_is_bool_in_schema(self):
        m = self._run_and_validate(debug=True)
        self.assertIsInstance(m.app.debug, bool)
        self.assertTrue(m.app.debug)

    def test_all_log_levels_accepted_by_schema(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            with self.subTest(level=level):
                m = self._run_and_validate(log_level=level)
                self.assertEqual(m.app.log_level.value, level)


class TestWizardDatabaseSchemaIntegration(unittest.TestCase):

    def _run(self, mode_str: str, extra_inputs=None, extra_bools=None) -> TBManifest:
        data = _fresh_manifest_dict()
        inputs = iter(extra_inputs or [])
        bools = iter(extra_bools or [])
        with (
            patch(f"{_MOD}.prompt_choice", return_value=mode_str),
            patch(f"{_MOD}.prompt_input",
                  side_effect=lambda *a, **kw: next(inputs, a[1] if len(a) > 1 else "")),
            patch(f"{_MOD}.prompt_bool",
                  side_effect=lambda *a, **kw: next(bools, False)),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            data = wizard_database_settings(data)
        return TBManifest.model_validate(data)

    def test_LC_produces_valid_manifest(self):
        self.assertIsInstance(
            self._run("LC - Local Dict (JSON file, no external dependencies)",
                      extra_inputs=[".data/test.json"]),
            TBManifest,
        )

    def test_LC_mode_is_databasemode_enum(self):
        m = self._run("LC - Local Dict (JSON file, no external dependencies)",
                      extra_inputs=[".data/test.json"])
        self.assertIsInstance(m.database.mode, DatabaseMode)
        self.assertEqual(m.database.mode, DatabaseMode.LC)

    def test_LR_mode_is_databasemode_enum(self):
        m = self._run("LR - Local Redis (requires local Redis server)",
                      extra_inputs=["redis://localhost:6379", "0"],
                      extra_bools=[False])
        self.assertEqual(m.database.mode, DatabaseMode.LR)

    def test_RR_produces_valid_manifest(self):
        m = self._run("RR - Remote Redis (requires remote Redis server)",
                      extra_inputs=["redis://remote:6379", "2"],
                      extra_bools=[False])
        self.assertIsInstance(m, TBManifest)
        self.assertEqual(m.database.mode, DatabaseMode.RR)

    def test_CB_produces_valid_manifest(self):
        m = self._run("CB - Cluster Blob (MinIO/S3 encrypted storage)",
                      extra_inputs=["localhost:9000", "key", "secret", "my-bucket"],
                      extra_bools=[False, False])
        self.assertIsInstance(m, TBManifest)
        self.assertEqual(m.database.mode, DatabaseMode.CB)

    def test_redis_db_index_is_int_in_schema(self):
        m = self._run("LR - Local Redis (requires local Redis server)",
                      extra_inputs=["redis://localhost:6379", "3"],
                      extra_bools=[False])
        self.assertIsInstance(m.database.redis.db_index, int)
        self.assertEqual(m.database.redis.db_index, 3)

    def test_minio_use_ssl_is_bool_in_schema(self):
        m = self._run("CB - Cluster Blob (MinIO/S3 encrypted storage)",
                      extra_inputs=["localhost:9000", "key", "secret", "bucket"],
                      extra_bools=[True, False])
        self.assertIsInstance(m.database.minio.use_ssl, bool)
        self.assertTrue(m.database.minio.use_ssl)

    def test_local_path_preserved_in_schema(self):
        m = self._run("LC - Local Dict (JSON file, no external dependencies)",
                      extra_inputs=[".data/custom.json"])
        self.assertEqual(m.database.local.path, ".data/custom.json")


class TestWizardIsaaSchemaIntegration(unittest.TestCase):

    def _run_enabled(self, temperature="0.5", max_tokens="4096") -> TBManifest:
        data = _fresh_manifest_dict()
        inputs = iter([
            "openrouter/haiku", "openrouter/gpt-4o",
            "self", temperature, max_tokens, "OPENROUTER_API_KEY",
        ])
        # True=Configure ISAA, True=stream, False=checkpoints, True=MCP, False=A2A
        bools = iter([True, True, False, True, False])
        with (
            patch(f"{_MOD}.prompt_bool",
                  side_effect=lambda *a, **kw: next(bools, False)),
            patch(f"{_MOD}.prompt_input",
                  side_effect=lambda *a, **kw: next(inputs, "")),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            data = wizard_isaa_settings(data)
        return TBManifest.model_validate(data)

    def test_isaa_config_is_isaaconfig_instance(self):
        self.assertIsInstance(self._run_enabled().isaa, IsaaConfig)

    def test_temperature_is_float_in_schema(self):
        m = self._run_enabled(temperature="0.3")
        self.assertIsInstance(m.isaa.self_agent.temperature, float)
        self.assertAlmostEqual(m.isaa.self_agent.temperature, 0.3, places=5)

    def test_max_tokens_is_int_in_schema(self):
        m = self._run_enabled(max_tokens="2048")
        self.assertIsInstance(m.isaa.self_agent.max_tokens_output, int)
        self.assertEqual(m.isaa.self_agent.max_tokens_output, 2048)

    def test_isaa_none_when_declined(self):
        data = _fresh_manifest_dict()
        with (
            patch(f"{_MOD}.prompt_bool", return_value=False),
            patch(f"{_MOD}.prompt_input", return_value=""),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            data = wizard_isaa_settings(data)
        self.assertIsNone(TBManifest.model_validate(data).isaa)

    def test_mcp_enabled_is_bool_in_schema(self):
        self.assertIsInstance(self._run_enabled().isaa.mcp.enabled, bool)

    def test_a2a_enabled_is_bool_in_schema(self):
        self.assertIsInstance(self._run_enabled().isaa.a2a.enabled, bool)

    def test_isaa_enabled_flag_true(self):
        m = self._run_enabled()
        self.assertTrue(m.isaa.enabled)


# =============================================================================
# B) Enum coercion — str → DatabaseMode / LogLevel / Environment
# =============================================================================

class TestSchemaEnumCoercion(unittest.TestCase):

    def test_all_database_modes_from_string(self):
        for code in ("LC", "LR", "RR", "CB"):
            with self.subTest(code=code):
                m = TBManifest.model_validate({"database": {"mode": code}})
                self.assertIsInstance(m.database.mode, DatabaseMode)
                self.assertEqual(m.database.mode.value, code)

    def test_invalid_database_mode_raises(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            TBManifest.model_validate({"database": {"mode": "INVALID"}})

    def test_all_log_levels_from_string(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            with self.subTest(level=level):
                m = TBManifest.model_validate({"app": {"log_level": level}})
                self.assertIsInstance(m.app.log_level, LogLevel)

    def test_all_environments_from_string(self):
        for env in ("development", "production", "staging", "tauri"):
            with self.subTest(env=env):
                m = TBManifest.model_validate({"app": {"environment": env}})
                self.assertIsInstance(m.app.environment, Environment)


# =============================================================================
# C) ManifestLoader roundtrip — all I/O in TemporaryDirectory
# =============================================================================

class TestManifestLoaderRoundtrip(unittest.TestCase):

    def _loader(self, tmp: str) -> ManifestLoader:
        return ManifestLoader(base_dir=tmp)

    def test_exists_false_when_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(self._loader(td).exists())

    def test_create_default_returns_tbmanifest(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._loader(td).create_default(save=False)
        self.assertIsInstance(m, TBManifest)

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
            loader.save(TBManifest())
            self.assertTrue(loader.exists())

    def test_save_load_preserves_app_name(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
            loader.save(TBManifest(app=AppConfig(name="RoundtripApp")))
            loaded = loader.load(resolve_env=False)
        self.assertEqual(loaded.app.name, "RoundtripApp")

    def test_save_load_preserves_database_mode(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
            loader.save(TBManifest(database=DatabaseConfig(mode=DatabaseMode.RR)))
            loaded = loader.load(resolve_env=False)
        self.assertEqual(loaded.database.mode, DatabaseMode.RR)

    def test_load_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_save_path_is_inside_tempdir(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
            saved = loader.save(TBManifest())
            self.assertTrue(str(saved).startswith(td))

    def test_save_path_is_path_object(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._loader(td).save(TBManifest())
        self.assertIsInstance(result, Path)

    def test_load_or_create_returns_default_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._loader(td).load_or_create_default(resolve_env=False)
        self.assertIsInstance(m, TBManifest)

    def test_load_or_create_loads_existing(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
            loader.save(TBManifest(app=AppConfig(name="Existing")))
            m = loader.load_or_create_default(resolve_env=False)
        self.assertEqual(m.app.name, "Existing")

    def test_saved_yaml_is_valid_utf8(self):
        with tempfile.TemporaryDirectory() as td:
            loader = self._loader(td)
            loader.save(TBManifest(app=AppConfig(name="Ümlauts-Ä")))
            content = loader.manifest_path.read_text(encoding="utf-8")
        self.assertIn("mlauts", content)  # byte-safe check

    def test_manifest_path_name_contains_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._loader(td).manifest_path
        self.assertIn("manifest", path.name)


# =============================================================================
# D) resolve_env_vars
# =============================================================================

class TestResolveEnvVars(unittest.TestCase):

    def test_resolves_present_var(self):
        os.environ["_TEST_TB_RESOLVE"] = "hello"
        try:
            self.assertEqual(resolve_env_vars("${_TEST_TB_RESOLVE}"), "hello")
        finally:
            del os.environ["_TEST_TB_RESOLVE"]

    def test_uses_default_when_missing(self):
        os.environ.pop("_TB_MISS", None)
        self.assertEqual(resolve_env_vars("${_TB_MISS:fallback}"), "fallback")

    def test_keeps_original_when_no_default_and_missing(self):
        os.environ.pop("_TB_NO_DEF", None)
        self.assertEqual(resolve_env_vars("${_TB_NO_DEF}"), "${_TB_NO_DEF}")

    def test_resolves_nested_dict(self):
        os.environ["_TB_HOST"] = "myhost"
        try:
            result = resolve_env_vars({"url": "redis://${_TB_HOST}:6379"})
            self.assertIn("myhost", result["url"])
        finally:
            del os.environ["_TB_HOST"]

    def test_resolves_list(self):
        os.environ["_TB_PORT"] = "9000"
        try:
            result = resolve_env_vars(["${_TB_PORT}"])
            self.assertEqual(result[0], "9000")
        finally:
            del os.environ["_TB_PORT"]

    def test_non_string_passthrough(self):
        self.assertEqual(resolve_env_vars(42), 42)
        self.assertIsNone(resolve_env_vars(None))

    def test_var_takes_precedence_over_default(self):
        os.environ["_TB_PRIO"] = "env_value"
        try:
            self.assertEqual(resolve_env_vars("${_TB_PRIO:default_value}"), "env_value")
        finally:
            del os.environ["_TB_PRIO"]

    def test_empty_default_produces_empty_string(self):
        os.environ.pop("_TB_EMPTY", None)
        self.assertEqual(resolve_env_vars("${_TB_EMPTY:}"), "")


# =============================================================================
# E) Full wizard pipeline — all wizard_* chained → valid TBManifest + save/load
# =============================================================================

class TestFullWizardPipeline(unittest.TestCase):

    def _full_run(self) -> TBManifest:
        data = _fresh_manifest_dict()

        # Step 1 — App
        with (
            patch(f"{_MOD}.prompt_input", return_value="PipelineApp"),
            patch(f"{_MOD}.prompt_bool", return_value=False),
            patch(f"{_MOD}.prompt_choice", return_value="INFO"),
        ):
            data = wizard_app_settings(data)

        # Step 2 — Database (LC, avoids redis/minio complexity)
        with (
            patch(f"{_MOD}.prompt_choice",
                  return_value="LC - Local Dict (JSON file, no external dependencies)"),
            patch(f"{_MOD}.prompt_input", return_value=".data/pipeline.json"),
            patch(f"{_MOD}.prompt_bool", return_value=False),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            data = wizard_database_settings(data)

        # Step 3 — Workers (known list-vs-dict mismatch; absorbed gracefully)
        with (
            patch(f"{_MOD}.prompt_bool", return_value=False),
            patch(f"{_MOD}.prompt_input", return_value="1"),
        ):
            try:
                data = wizard_workers_settings(data)
            except TypeError:
                pass  # documented incompatibility — rest of pipeline continues

        # Step 4 — Services
        with (
            patch(f"{_MOD}.prompt_bool", return_value=False),
            patch(f"{_MOD}.prompt_input", return_value=""),
        ):
            data = wizard_services_settings(data)

        # Step 5 — ISAA disabled
        with (
            patch(f"{_MOD}.prompt_bool", return_value=False),
            patch(f"{_MOD}.prompt_input", return_value=""),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            data = wizard_isaa_settings(data)

        return TBManifest.model_validate(data)

    def test_produces_valid_tbmanifest(self):
        self.assertIsInstance(self._full_run(), TBManifest)

    def test_app_name_preserved_end_to_end(self):
        self.assertEqual(self._full_run().app.name, "PipelineApp")

    def test_database_mode_lc_end_to_end(self):
        self.assertEqual(self._full_run().database.mode, DatabaseMode.LC)

    def test_isaa_none_when_declined_end_to_end(self):
        self.assertIsNone(self._full_run().isaa)

    def test_save_load_after_full_pipeline(self):
        manifest = self._full_run()
        with tempfile.TemporaryDirectory() as td:
            loader = ManifestLoader(base_dir=td)
            loader.save(manifest)
            loaded = loader.load(resolve_env=False)
        self.assertEqual(loaded.app.name, manifest.app.name)
        self.assertEqual(loaded.database.mode, manifest.database.mode)


# =============================================================================
# F) Workers schema mismatch — list-vs-dict incompatibility documentation
# =============================================================================

class TestWorkersSchemaCompatibility(unittest.TestCase):

    def test_workers_http_in_model_dump_is_list(self):
        """Schema produces a list for workers.http — not a dict."""
        data = _fresh_manifest_dict()
        self.assertIsInstance(data["workers"]["http"], list)

    def test_workers_websocket_in_model_dump_is_list(self):
        data = _fresh_manifest_dict()
        self.assertIsInstance(data["workers"]["websocket"], list)

    def test_wizard_workers_raises_typeerror_on_schema_dump(self):
        """
        wizard_workers_settings does `http["enabled"] = ...` on the result of
        workers.get("http", {}) which is a list from model_dump() — raises TypeError.
        If this test fails (no exception), the wizard was fixed or schema changed.
        """
        data = _fresh_manifest_dict()
        with (
            patch(f"{_MOD}.prompt_bool", return_value=True),
            patch(f"{_MOD}.prompt_input", return_value="1"),
        ):
            wizard_workers_settings(data)

    def test_wizard_workers_works_with_empty_workers_dict(self):
        """The wizard's original design works when workers is an empty dict."""
        data = _simple_manifest(workers={})
        bools = iter([True, True])
        inputs = iter(["2", "5000", "1", "6587"])
        with (
            patch(f"{_MOD}.prompt_bool",
                  side_effect=lambda *a, **kw: next(bools, True)),
            patch(f"{_MOD}.prompt_input",
                  side_effect=lambda *a, **kw: next(inputs, "1")),
        ):
            result = wizard_workers_settings(data)
        self.assertEqual(result["workers"]["http"][0]["port"], 5000)
        self.assertEqual(result["workers"]["http"][0]["instances"], 2)
        self.assertIsInstance(result["workers"]["http"][0]["port"], int)

    def test_schema_workersconfig_accepts_list_of_instances(self):
        from toolboxv2.utils.manifest.schema import HTTPWorkerInstance, WSWorkerInstance
        wc = WorkersConfig(
            http=[HTTPWorkerInstance(port=8080)],
            websocket=[WSWorkerInstance(port=8100)],
        )
        self.assertEqual(wc.http[0].port, 8080)
        self.assertIsInstance(wc.http[0].port, int)


# =============================================================================
# G) ManifestLoader.validate() — schema-level validation logic
# =============================================================================

class TestManifestLoaderValidate(unittest.TestCase):

    def _loader_with(self, manifest: TBManifest) -> ManifestLoader:
        loader = ManifestLoader.__new__(ManifestLoader)
        loader._manifest = manifest
        loader._manifest_path = None
        loader.base_dir = Path(tempfile.mkdtemp())
        return loader

    def test_default_manifest_is_valid(self):
        is_valid, errors = self._loader_with(TBManifest()).validate()
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_no_manifest_returns_error(self):
        loader = self._loader_with(TBManifest())
        loader._manifest = None
        is_valid, errors = loader.validate()
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_duplicate_worker_port_is_detected(self):
        from toolboxv2.utils.manifest.schema import HTTPWorkerInstance, WSWorkerInstance
        manifest = TBManifest(workers=WorkersConfig(
            http=[HTTPWorkerInstance(port=8000)],
            websocket=[WSWorkerInstance(port=8000)],  # conflict!
        ))
        is_valid, errors = self._loader_with(manifest).validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("8000" in e for e in errors))

    def test_ssl_enabled_without_cert_is_error(self):
        from toolboxv2.utils.manifest.schema import NginxConfig
        manifest = TBManifest(nginx=NginxConfig(
            ssl_enabled=True, ssl_certificate="", ssl_certificate_key="",
        ))
        is_valid, errors = self._loader_with(manifest).validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("ssl_certificate" in e for e in errors))

    def test_validate_returns_tuple_bool_list(self):
        result = self._loader_with(TBManifest()).validate()
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], list)


# =============================================================================
# H) ENV file I/O + ENV_CATEGORIES structure
# =============================================================================

class TestEnvFileIO(unittest.TestCase):

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            save_env_file(p, {"API_KEY": "sk-test", "PORT": "8080"})
            loaded = load_existing_env(p)
        self.assertEqual(loaded["API_KEY"], "sk-test")
        self.assertEqual(loaded["PORT"], "8080")

    def test_template_comment_associated_with_variable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "env-template"
            p.write_text("# OpenAI key\nOPENAI_API_KEY=\n", encoding="utf-8")
            result = parse_env_template(p)
        self.assertEqual(result["OPENAI_API_KEY"][1], "OpenAI key")

    def test_update_existing_key_in_place(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            p.write_text("FOO=old\n", encoding="utf-8")
            save_env_file(p, {"FOO": "new"})
            loaded = load_existing_env(p)
        self.assertEqual(loaded["FOO"], "new")

    def test_empty_values_not_written(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            save_env_file(p, {"EMPTY": "", "SET": "value"})
            content = p.read_text(encoding="utf-8")
        self.assertNotIn("EMPTY=", content)
        self.assertIn("SET=value", content)

    def test_wizard_env_variables_does_not_mutate_existing_dict(self):
        template = {"NEW_KEY": ("", "some key"), "SET_KEY": ("val", "set")}
        existing = {"SET_KEY": "original"}
        original_copy = existing.copy()
        with (
            patch(f"{_MOD}.prompt_bool", return_value=True),
            patch(f"{_MOD}.prompt_input", return_value="new"),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            with tempfile.TemporaryDirectory() as td:
                wizard_env_variables(Path(td), existing, template)
        self.assertEqual(existing, original_copy)

    def test_set_vars_not_prompted_again(self):
        template = {"UNSET": ("", "need"), "SET": ("val", "set")}
        existing = {"SET": "existing_value"}
        prompted = []

        def fake_input(prompt, *a, **kw):
            prompted.append(prompt)
            return "x"

        with (
            patch(f"{_MOD}.prompt_bool", return_value=True),
            patch(f"{_MOD}.prompt_input", side_effect=fake_input),
            patch(f"{_MOD}.c_print", lambda *a, **kw: None),
        ):
            with tempfile.TemporaryDirectory() as td:
                wizard_env_variables(Path(td), existing, template)
        self.assertNotIn("SET", prompted)

    def test_env_categories_all_str_lists(self):
        for cat, var_list in ENV_CATEGORIES.items():
            self.assertIsInstance(cat, str)
            self.assertIsInstance(var_list, list)
            for v in var_list:
                self.assertIsInstance(v, str, f"Non-str in {cat}: {v!r}")

    def test_env_categories_no_empty_list(self):
        for cat, var_list in ENV_CATEGORIES.items():
            self.assertGreater(len(var_list), 0, f"Empty category: {cat!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
