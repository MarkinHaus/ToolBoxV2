#!/usr/bin/env python
import requests
from rich.traceback import install
import os


install(show_locals=True)

"""Tests for `cloudM` package."""
from coverage.annotate import os

from toolboxv2 import App

import unittest


class TestCloudM(unittest.TestCase):

    def setUp(self):
        self.app = App("test")

        self.app.mlm = "I"
        self.app.debug = True
        self.app.dev_modi = False
        self.token = ""

        self.app.load_mod("quickNote")
        self.app.new_ac_mod("quickNote")

        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.app.remove_all_modules()
        self.app.save_exit()
        self.app.exit()

    def test_show_version(self):
        comd = []
        res = self.app.run_function("Version", command=comd)
        self.assertEqual(res, "0.0.1")

    def test_get_inbox(self):
        #url = " http://127.0.0.1:5000/post/quickNote/run/get_inbox_api?command="
        #print(f"testing {url}")
        #j_data = {"token": self.token,
        #          "data": {}
        #          }
        #r = requests.post(url, json=j_data)
        #print(f"res : {r.json()}")
        pass
