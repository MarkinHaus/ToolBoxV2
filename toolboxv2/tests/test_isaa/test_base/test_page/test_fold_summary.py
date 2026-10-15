"""Bridge: Page-Engine *fold summary-notes* (C) tests into the repo tree.
See ``test_core.py`` in this folder for the pattern + rationale.
"""
from __future__ import annotations

import unittest

try:
    from toolboxv2.mods.isaa.base.Agent.page.tests import test_fold_summary as _mod

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

    class TestPageFoldSummaryImport(unittest.TestCase):
        def test_module_importable(self):
            self.skipTest(f"page.tests.test_fold_summary not importable: {_IMPORT_ERROR!r}")


if __name__ == "__main__":
    unittest.main()
