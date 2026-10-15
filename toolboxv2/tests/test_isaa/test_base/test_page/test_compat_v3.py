"""Bridge: run the Page-Engine *V3-drop-in* acceptance tests from the repo tree.

Mirrors ``test_core.py`` in this folder, but for the ``test_compat_v3`` module
(the ``PageExecutionEngine`` V3 compatibility adapter).
"""
from __future__ import annotations

import unittest

try:
    from toolboxv2.mods.isaa.base.Agent.page.tests import test_compat_v3 as _mod

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    _mod = None
    _IMPORT_ERROR = exc


if _mod is not None:
    for _name in dir(_mod):
        _obj = getattr(_mod, _name)
        if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase):
            globals()[_name] = _obj
    del _name, _obj
else:  # pragma: no cover - environment dependent

    class TestPageCompatV3Import(unittest.TestCase):
        def test_compat_v3_module_importable(self):
            self.skipTest(f"page.tests.test_compat_v3 not importable: {_IMPORT_ERROR!r}")


if __name__ == "__main__":
    unittest.main()
