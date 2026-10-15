"""Bridge: Page-Engine *pretty-printer* (reusable card formatter) tests into the
repo test tree.  See ``test_core.py`` in this folder for the pattern + rationale.
"""
from __future__ import annotations

import unittest

try:
    from toolboxv2.mods.isaa.base.Agent.page.tests import test_pretty as _mod

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

    class TestPagePrettyImport(unittest.TestCase):
        def test_module_importable(self):
            self.skipTest(f"page.tests.test_pretty not importable: {_IMPORT_ERROR!r}")


if __name__ == "__main__":
    unittest.main()
