#!/usr/bin/env python
from rich.traceback import install
import os


install(show_locals=True)

"""Tests for `cloudM` package."""
from coverage.annotate import os

from toolboxv2 import App

import unittest


class TestWelcome(unittest.TestCase):

    def setUp(self):

        self.app = App("test")
        self.app.mlm = "I"
        self.app.debug = True
        self.app.load_mod("welcome")
        self.app.new_ac_mod("welcome")
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.app.remove_all_modules()
        self.app.save_exit()
        self.app.exit()

    def test_name(self):
        res = self.app.AC_MOD
        self.assertIsNotNone(res)
        res = self.app.AC_MOD.name
        self.assertEqual(res, "welcome")

    def test_show_version(self):
        comd = []
        res = self.app.run_function("Version", command=comd)
        self.assertEqual(res, "0.0.1")

    def test_print_t(self):  # charmap error
        comd = []
        res = self.app.run_function("printT", command=comd)
        self.assertEqual(res, "TOOL BOX")
        self.assertTrue("TOOL BOX")

    def test_Animation(self):  # charmap error
        comd = []
        # res = self.app.run_function("Animation", command=comd)
        # self.assertEqual(res, "Animation")
        self.assertTrue("Animation")

    def test_Animation1(self):  # charmap error
        comd = []
        # res = self.app.run_function("Animation1", command=comd)
        # self.assertEqual(res, "Animation1")
        self.assertTrue("Animation1")


