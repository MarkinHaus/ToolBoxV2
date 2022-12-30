#!/usr/bin/env python
from rich.traceback import install
import os


install(show_locals=True)

"""Tests for `cloudM` package."""
from coverage.annotate import os

from toolboxv2 import App

import unittest


class TestCloudM(unittest.TestCase):

    def setUp(self):
        self.name = "cloudM"
        self.app = App("test")

        self.app.mlm = "I"
        self.app.debug = True
        self.app.load_mod("cloudM")
        self.app.new_ac_mod("cloudM")
        self.app.run_function("first-web-connection", command=['first-web-connection', 'http://127.0.0.1:5000'])
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.app.remove_all_modules()
        self.app.save_exit()
        self.app.exit()

    def test_show_version(self):
        comd = []
        res = self.app.run_function("Version", command=comd)
        api_ver = self.app.AC_MOD.api_version
        if api_ver:
            self.assertEqual(api_ver, "0.0.1")
        else:
            print("API Not Found")
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
        self.assertTrue(os.path.exists(f".\\mods_dev\\test_module.py"))
        os.remove(".\\mods_dev\\test_module.py")
        self.assertFalse(os.path.exists(f".\\mods_dev\\test_module.py"))

    # def test_upload(self):
    #    comd = []
    #    res = self.app.run_function("upload", command=comd)
    #    self.assertIsNone(res)
#
# def test_download(self):
#    comd = []
#    res = self.app.run_function("download", command=comd)
#    self.assertFalse(res)
#
# def test_add_url_con(self):
#    pass
#    # comd = []
#    # res = self.app.run_function("first-web-connection", command=comd)
#    # self.assertIsNone(res)
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
