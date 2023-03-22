#!/usr/bin/env python
from rich.traceback import install
import os

install(show_locals=True)

"""Tests for `DB` module."""

from toolboxv2 import App
import unittest


class TestDB(unittest.TestCase):

    def setUp(self):
        self.app = App("test")
        self.app.mlm = "I"
        self.app.debug = True
        self.app.load_mod("DB")
        self.app.new_ac_mod("DB")

        # self.redis_url = "redis://localhost:6379"
        self.test_key = "test_key"
        self.test_value = "test_value"
        self.test_dict = {"key1": "value1", "key2": "value2"}

    def tearDown(self):
        # self.rcon.flushall()
        self.app.remove_all_modules()
        self.app.save_exit()
        self.app.exit()

    def test_set_key_string(self):
        command = ["set", self.test_key, self.test_value]
        res = self.app.run_function("set", command=command)
        self.assertTrue(res)
        redis_value = self.app.run_function("get", command=["get", self.test_key])
        self.assertEqual(redis_value, self.test_value)

    def test_set_key_invalid_type(self):
        with self.assertRaises(TypeError):
            command = ["set", self.test_key, 123]
            self.app.run_function("set", command=command)
