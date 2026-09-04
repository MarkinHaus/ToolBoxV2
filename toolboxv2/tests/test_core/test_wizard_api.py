"""Self-checks fuer wizard_api (pure Funktionen, keine Netz-/Server-Abhaengigkeiten).

Run: python -m unittest toolboxv2.utils.workers.fast.test_wizard_api -v
"""
import unittest


class TestProfileBridge(unittest.TestCase):
    def test_bridge_covers_all_first_run_profiles(self):
        from toolboxv2.utils.workers.fast.wizard_api import PROFILE_BRIDGE
        from toolboxv2.utils.clis.first_run import PROFILES
        self.assertEqual(set(PROFILE_BRIDGE.keys()), set(PROFILES.keys()))

    def test_bridge_values_are_valid_init_presets(self):
        from toolboxv2.utils.workers.fast.wizard_api import PROFILE_BRIDGE
        from toolboxv2.init_onboarding import _PROFILE_PRESETS
        for p, preset in PROFILE_BRIDGE.items():
            self.assertIn(preset, _PROFILE_PRESETS, f"bridge {p} -> {preset} unknown preset")


class TestSetGetNested(unittest.TestCase):
    def test_roundtrip(self):
        from toolboxv2.utils.workers.fast.wizard_api import _set_nested, _get_nested
        d = {}
        _set_nested(d, "database.redis.url", "redis://x")
        _set_nested(d, "database.redis.db_index", 0)
        self.assertEqual(_get_nested(d, "database.redis.url"), "redis://x")
        self.assertEqual(_get_nested(d, "database.redis.db_index"), 0)
        self.assertIsNone(_get_nested(d, "database.redis.missing"))
        self.assertEqual(_get_nested(d, "database.redis.missing", "def"), "def")


class TestWizardSteps(unittest.TestCase):
    def test_non_server_has_no_workers_step(self):
        from toolboxv2.utils.workers.fast.wizard_api import wizard_steps
        steps = wizard_steps({}, "consumer")
        ids = [s["id"] for s in steps]
        self.assertNotIn("workers", ids)
        self.assertNotIn("services", ids)
        self.assertEqual(ids, ["app", "database", "llm", "autostart", "features"])

    def test_server_includes_workers_and_services(self):
        from toolboxv2.utils.workers.fast.wizard_api import wizard_steps
        ids = [s["id"] for s in wizard_steps({}, "server")]
        self.assertIn("workers", ids)
        self.assertIn("services", ids)

    def test_db_fields_follow_mode(self):
        from toolboxv2.utils.workers.fast.wizard_api import wizard_steps
        steps = wizard_steps({"database": {"mode": "LC"}}, "consumer")
        db = next(s for s in steps if s["id"] == "database")
        names = {f["name"] for f in db["fields"]}
        self.assertIn("database.local.path", names)
        self.assertNotIn("database.redis.url", names)

        steps = wizard_steps({"database": {"mode": "CB"}}, "consumer")
        db = next(s for s in steps if s["id"] == "database")
        names = {f["name"] for f in db["fields"]}
        self.assertIn("database.minio.endpoint", names)


class TestValidateValues(unittest.TestCase):
    FIELDS = [
        {"name": "app.name", "type": "str"},
        {"name": "app.debug", "type": "bool"},
        {"name": "app.log_level", "type": "choice", "choices": ["DEBUG", "INFO"]},
        {"name": "autostart.services", "type": "multi", "choices": ["daemon", "workers"]},
        {"name": "workers.http_port", "type": "int"},
    ]

    def test_valid(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        self.assertIsNone(validate_values(self.FIELDS, {
            "app.name": "TB", "app.debug": True, "app.log_level": "INFO",
            "autostart.services": ["daemon"], "workers.http_port": "5000"}))

    def test_unknown_field_rejected(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        err = validate_values(self.FIELDS, {"app.evil": 1})
        self.assertIn("Unknown", err)

    def test_bool_rejects_string(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        self.assertIn("expected bool", validate_values(self.FIELDS, {"app.debug": "true"}))

    def test_int_rejects_garbage(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        self.assertIn("expected int", validate_values(self.FIELDS, {"workers.http_port": "80a"}))

    def test_int_accepts_numeric_string_and_int(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        self.assertIsNone(validate_values(self.FIELDS, {"workers.http_port": 5000}))
        self.assertIsNone(validate_values(self.FIELDS, {"workers.http_port": "5000"}))

    def test_choice_rejects_unknown(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        self.assertIn("not in", validate_values(self.FIELDS, {"app.log_level": "LOUD"}))

    def test_multi_rejects_bad_member(self):
        from toolboxv2.utils.workers.fast.wizard_api import validate_values
        self.assertIn("invalid", validate_values(self.FIELDS, {"autostart.services": ["daemon", "nginx"]}))


class TestApplyValues(unittest.TestCase):
    def test_nested_merge(self):
        from toolboxv2.utils.workers.fast.wizard_api import apply_values
        draft = apply_values({"database": {"mode": "LC"}},
                             [{"name": "database.mode", "type": "choice"}],
                             {"database.mode": "CB"})
        self.assertEqual(draft["database"]["mode"], "CB")

    def test_worker_http_special_case(self):
        from toolboxv2.utils.workers.fast.wizard_api import apply_values
        draft = apply_values({}, [
            {"name": "workers.http_enabled", "type": "bool"},
            {"name": "workers.http_port", "type": "int"}],
            {"workers.http_enabled": True, "workers.http_port": "8080"})
        http = draft["workers"]["http"][0]
        self.assertTrue(http["enabled"])
        self.assertEqual(http["port"], 8080)


    def test_current_profile_unwraps_enum(self):
        from enum import Enum
        from toolboxv2.utils.workers.fast.wizard_api import _current_profile

        class P(Enum):
            SERVER = "server"

        self.assertEqual(_current_profile({"app": {"profile": P.SERVER}}, {}), "server")
        self.assertEqual(_current_profile({"app": {"profile": "developer"}}, {}), "developer")
        self.assertEqual(_current_profile({}, {"profile": "homelab"}), "homelab")
        self.assertIsNone(_current_profile({}, {}))


if __name__ == "__main__":
    unittest.main()
