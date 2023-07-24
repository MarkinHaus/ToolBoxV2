import time
import unittest

from toolboxv2 import App


class TestIsaaIDE(unittest.TestCase):
    t0 = 0
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test")
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.inplace_load("isaa", "toolboxv2.mods.")
        cls.app.new_ac_mod("isaa")
        cls.isaa_tool_class = cls.app.AC_MOD

    @classmethod
    def tearDownClass(cls):
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")
