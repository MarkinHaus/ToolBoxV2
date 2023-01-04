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
        self.token = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9" \
                     ".eyJ2IjoiMC4yLjMiLCJleHAiOjE2ODY4OTI" \
                     "yNTQsInVzZXJuYW1lIjoidCIsInBhc3N3b3J" \
                     "kIjoiZGQ3OGYwMWMxYWJlNjE0OTlmNWQ5Y2Q" \
                     "0NjA4NzliNGQyMjA0M2NiOWQ4ZjRkMmNlMTA" \
                     "0OThiNWNkMTI5ZTg4NWY0MGM0OTgxNzkzMWN" \
                     "mMGJhOGI5OGIzY2ZhM2Q3ODg5ZGExM2JjYmZ" \
                     "hOGZiMjM5YzVkNmI0NzExMTFiODExMWYwY2N" \
                     "iNWUyNmYzMGY2YmZmZGRjNDYxMGQwNmFlOWF" \
                     "kMGZkYWI4NjdlOThiNjkwZDU1NjJiOTk1YjI" \
                     "yYTE0MmIyIiwiZW1haWwiOiJ0IiwidWlkIjo" \
                     "iYjI2N2NmMDItZTFhMi00M2YzLWFlZTgtMjB" \
                     "jMzU1NzVlYTE3IiwiYXVkIjoiYXBpLURFU0t" \
                     "UT1AtQ0k1N1YxTCJ9.d5O_ArM2lSN9pgPlsK" \
                     "thq6_nPLDjfmnyBwYv1csUPif8UqUYglGU0L" \
                     "X9NJgg30Elw8ePd0ocdlq0-mG7UN4fJg"

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
