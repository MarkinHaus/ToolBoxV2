"""Schutz gegen Modul-Stubs in der Testsuite.

Frueher ersetzte diese Datei beim Import `toolboxv2.utils.system.types`,
`toolboxv2.utils.workers.event_manager`, `requests` und `multipart` dauerhaft in
`sys.modules`. Alle Tests, die danach liefen, bekamen die Attrappen (im Gesamtlauf
schlugen test_account_bridge und test_types fehl, einzeln waren sie gruen).
test_fast fuer fast_tb laeuft mit den echten Modulen, die Stubs wurden von keiner
Testdatei benutzt.
"""

import importlib
import unittest


class TestNoModuleStubs(unittest.TestCase):
    """Eine Testdatei darf echte Module nicht in sys.modules ersetzen (Testleitfaden, Regel 6.1).

    Ein Stub unter dem Namen eines echten Moduls wuerde jede spaeter laufende
    Testdatei treffen (test_account_bridge, test_types).
    """

    def test_system_types_is_the_real_module(self):
        mod = importlib.import_module("toolboxv2.utils.system.types")
        self.assertIsNotNone(getattr(mod, "__file__", None),
                             "toolboxv2.utils.system.types ist ein Stub ohne Datei")
        from toolboxv2.utils.system.types import Result, ToolBoxError  # noqa: F401

    def test_event_manager_is_the_real_module(self):
        mod = importlib.import_module("toolboxv2.utils.workers.event_manager")
        self.assertIsNotNone(getattr(mod, "__file__", None),
                             "toolboxv2.utils.workers.event_manager ist ein Stub ohne Datei")
