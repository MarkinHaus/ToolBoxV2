"""Bridge: run the Page-Engine *core* acceptance tests from the repo test tree.

The real test code lives next to the engine implementation
(``toolboxv2/mods/isaa/base/Agent/page/tests/test_core.py``) so it can run
isolated via the iso_pkg method described in START_PROMPT.md.  This module
re-exports the ``unittest.TestCase`` subclasses so pytest / unittest discovery
under ``toolboxv2/tests/test_isaa/test_base/test_page`` picks them up too.

If the page package (or its optional ISAA deps) cannot be imported, the tests
are skipped rather than erroring the whole suite.
"""
from __future__ import annotations

import unittest

try:
    from toolboxv2.mods.isaa.base.Agent.page.tests import test_core as _mod

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    _mod = None
    _IMPORT_ERROR = exc


if _mod is not None:
    # Re-export every TestCase subclass defined in the source module so that
    # unittest / pytest discovery finds them under this module's namespace.
    for _name in dir(_mod):
        _obj = getattr(_mod, _name)
        if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase):
            globals()[_name] = _obj
    del _name, _obj
else:  # pragma: no cover - environment dependent

    class TestPageCoreImport(unittest.TestCase):
        def test_page_package_importable(self):
            self.skipTest(f"page package not importable: {_IMPORT_ERROR!r}")


if __name__ == "__main__":
    unittest.main()
