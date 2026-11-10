"""Self-checks for wizard_api helpers (pure functions).

Run: python -m unittest toolboxv2.utils.workers.fast.test_wizard_api -v
"""
import unittest


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


class TestCurrentProfile(unittest.TestCase):
    def test_current_profile_unwraps_enum(self):
        from enum import Enum
        from toolboxv2.utils.workers.fast.wizard_api import _current_profile

        class P(Enum):
            SERVER = "server"

        self.assertEqual(_current_profile({"app": {"profile": P.SERVER}}, {}), "server")
        self.assertEqual(_current_profile({"app": {"profile": "developer"}}, {}), "developer")
        self.assertEqual(_current_profile({}, {"profile": "local"}), "local")
        self.assertIsNone(_current_profile({}, {}))


if __name__ == "__main__":
    unittest.main()
