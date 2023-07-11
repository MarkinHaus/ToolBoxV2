#!/usr/bin/env python
import time

from rich.traceback import install
import os
import threading

install(show_locals=True)

"""Tests for `cloudM` package."""
from coverage.annotate import os

from toolboxv2 import App, Style

import unittest


class TestCloudM(unittest.TestCase):
    t0 = None
    api = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test")
        api = threading.Thread(target=cls.app.run_any, args=('api_manager', 'start-api', ['start-api', 'test-api']))
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.load_mod("cloudM")
        cls.app.new_ac_mod("cloudM")
        cls.app.run_function("first-web-connection", command=['first-web-connection', 'http://127.0.0.1:5000/api'])
        api.start()
        time.sleep(3)
        cls.app.new_ac_mod("cloudM")

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing API')
        cls.app.AC_MOD.delete_file()
        cls.app.run_any('api_manager', 'stop-api', ['start-api', 'test-api'])
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()

        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def test_show_version(self):
        comd = []
        res = self.app.run_function("Version", comd)
        self.assertEqual(res, "0.0.1")

    def test_new_module(self):
        comd = ["", "test_module"]
        res = self.app.run_function("NEW", command=comd)
        self.assertTrue(res)
        res = self.app.run_function("NEW", command=comd)
        self.assertFalse(res)
        comd = ["", "cloudM"]
        res = self.app.run_function("NEW", command=comd)
        self.assertFalse(res)
        self.assertTrue(os.path.exists("./mods_dev/test_module.py"))
        os.remove("./mods_dev/test_module.py")
        self.assertFalse(os.path.exists("./mods_dev/test_module.py"))

# def test_upload(self):
#   comd = ["", "test_module"]
#   res = self.app.run_function("NEW", command=comd)
#   self.assertTrue(res)
#   res = self.app.run_function("upload", command=comd)
#   self.assertIsNone(res)
#   self.assertTrue(os.path.exists("./mods_dev/test_module.py"))
#   os.remove("./mods_dev/test_module.py")
#   self.assertFalse(os.path.exists("./mods_dev/test_module.py"))
#
# def test_download(self):
#    comd = []
#    res = self.app.run_function("download", command=comd)
#    self.assertFalse(res)
#

#
# def test_create_account(self):
#    pass
#    #comd = []
#    #res = self.app.run_function("create-account", command=comd)
#
#
# def test_log_in(self):
#    comd = []
#    res = self.app.run_function("login", command=comd)
#    assert res == ""
#
# def test_create_user(self):
#    comd = []
#    res = self.app.run_function("create_user", command=comd)
#    assert res == ""
#
# def test_log_in_user(self):
#    comd = []
#    res = self.app.run_function("log_in_user", command=comd)
#    assert res == ""
#
# def test_validate_jwt(self):
#    comd = []
#    res = self.app.run_function("validate_jwt", command=comd)
#    assert res == ""
#
# def test_download_api_files(self):
#    comd = []
#    res = self.app.run_function("download_api_files", command=comd)
#    assert res == ""
#
# def test_update_core(self):
#    comd = []
#    res = self.app.run_function("#update-core", command=comd)
#    assert res == ""
