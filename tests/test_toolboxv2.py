#!/usr/bin/env python

"""Tests for `toolboxv2` package."""
import unittest

from toolboxv2 import App, MainTool, FileHandler, Style
from rich.traceback import install

from toolboxv2.util.cryp import Code
import time

from toolboxv2.util.toolbox import ApiOb

install(show_locals=True)


class TestToolboxv2(unittest.TestCase):
    """Tests for `toolboxv2` package."""

    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test")
        cls.app.mlm = "I"
        cls.app.debug = True

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

    def test_crypt(self):
        t0 = time.time()
        self.app.logger.info(Style.GREYBG("Testing crypt"))
        test_string = "1234567890"
        code = Code()
        self.app.logger.info(Style.WHITE("encode test string"))
        encode_string = code.encode_code(test_string)
        self.app.logger.info(Style.WHITE("test for differences between encode_string and test_string"))
        if encode_string == test_string:
            self.app.logger.warning(Style.YELLOW(f"No crypt active please init or crate owen "))

        self.app.logger.info(Style.WHITE("decode test string"))
        out_string = code.decode_code(encode_string)
        self.app.logger.info(f" {test_string=} {encode_string=} {out_string=} don in {time.time() - t0}")

        self.app.logger.info(Style.WHITE("Test if test_string and out_string are equal"))
        self.assertEqual(test_string, out_string)

    def test_file_handler(self):
        t0 = time.time()
        self.app.logger.info(Style.GREYBG("Testing file handler"))
        self.fh_test("")
        self.fh_test(0)
        self.fh_test([])
        self.fh_test({})
        self.fh_test(())

        self.fh_test("test")
        self.fh_test(124354)
        self.fh_test([1233, "3232"])
        self.fh_test({"test": "test", "value": -1})
        self.fh_test((0, 0, 0, 0))

        self.app.logger.info(Style.WHITE(f"finish testing in {time.time()-t0}"))

    def fh_test(self, test_value):
        t0 = time.time()
        self.app.logger.info(Style.GREYBG(f"Testing value : {test_value} of type : {type(test_value)}"))
        self.app.logger.info(Style.WHITE("initialized file handler"))
        fh = FileHandler("test.config", keys={"TestKey": "test~~~~~:"}, defaults={"TestKey": "Default"})

        self.app.logger.info(Style.WHITE("Verify that the object was initialized correctly"))
        self.assertEqual(fh.file_handler_filename, "test.config")
        self.assertEqual(fh.file_handler_file_prefix, ".config/mainTool/")

        # Open the storage file in write mode and verify that it was opened correctly
        self.app.logger.info(Style.WHITE("testStorage "))
        self.assertIsNone(fh.file_handler_storage)
        self.app.logger.info(Style.WHITE("load data from storage"))
        fh.load_file_handler()

        self.assertIsNone(fh.file_handler_storage)
        self.app.logger.info(Style.WHITE("getting default value for file handler storage"))
        value = fh.get_file_handler("TestKey")
        value2 = fh.get_file_handler("test~~~~~:")

        self.assertEqual(value, "Default")
        self.assertEqual(value, value2)
        self.app.logger.info(Style.WHITE("update value and testing update function"))
        t = fh.add_to_save_file_handler("test~~~~~:", str(test_value))
        f = fh.add_to_save_file_handler("test~~~~:", str(test_value))

        value = fh.get_file_handler("TestKey")

        self.assertTrue(t)
        self.assertFalse(f)
        self.assertEqual(value, test_value)
        self.app.logger.info(Style.WHITE("value updated successfully"))

        fh.save_file_handler()

        del fh

        self.app.logger.info(Style.WHITE("test if updated value saved in file"))
        fh2 = FileHandler("test.config", keys={"TestKey": "test~~~~~:"}, defaults={"TestKey": "Default"})
        # Verify that the object was initialized correctly
        self.assertEqual(fh2.file_handler_filename, "test.config")
        self.assertEqual(fh2.file_handler_file_prefix, ".config/mainTool/")

        # Open the storage file in write mode and verify that it was opened correctly
        self.assertIsNone(fh2.file_handler_storage)

        fh2.load_file_handler()

        self.assertIsNone(fh2.file_handler_storage)

        value = fh2.get_file_handler("TestKey")

        self.assertEqual(value, test_value)
        self.app.logger.info(Style.WHITE("success"))
        self.app.logger.info(f"don testing FileHandler in {time.time() - t0}")
        self.app.logger.info(Style.WHITE("cleaning up"))
        fh2.delete_file()

    def test_main_tool(self):
        main_tool = MainTool(v="1.0.0", tool={}, name="TestTool", logs=[], color="RED", on_exit=None, load=None)
        main_tool.print("Hello, world!")
        ob = ApiOb()
        ob.token = ""
        ob.data = {"": ""}
        uid, err = main_tool.get_uid([ob, ], self.app)
        self.assertTrue(err)
        print(uid)

    def test_styels(self):
        st = Style()
        st.color_demo()

    def test_utils(self):
        pass  # TODO Logging server first
