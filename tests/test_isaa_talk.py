#!/usr/bin/env python

"""Tests for `isaa_talk` cli."""
import unittest

from toolboxv2 import App, MainTool, FileHandler, Style
from rich.traceback import install

from toolboxv2.cryp import Code
import time

from toolboxv2.isaa_talk import set_up_app_for_isaa_talk, combine_sentences
from toolboxv2.toolbox import ApiOb

install(show_locals=True)


class TestToolboxv2(unittest.TestCase):
    """Tests for `isaa_talk` cli."""

    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test")
        cls.app.mlm = "I"
        cls.app.debug = True
        set_up_app_for_isaa_talk(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def setUp(self):
        self.app.logger.info(Style.BEIGEBG(f"Next Test"))
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.app.remove_all_modules()
        self.app.reset()
        self.app.logger.info(Style.BEIGEBG(f"tearDown"))

    def test_combine_sentences(self):
        sentence1 = "Ich gehe gerne"
        sentence2 = ", wenn ich draußen bin."
        combined_sentence = combine_sentences(sentence1, sentence2, self.app)
        self.app.logger.info(f"New Sentence : {combined_sentence}")
        self.assertNotEqual(combined_sentence, sentence1 + ' ' + sentence2)
