#!/usr/bin/env python

"""Tests for `toolboxv2` package."""
import asyncio
import time
import unittest

from rich.traceback import install

from toolboxv2 import MainTool, FileHandler, Style, get_app
from toolboxv2.utils.security.cryp import Code

install(show_locals=True)


class TestToolboxv2(unittest.TestCase):
    """Tests for `toolboxv2` package."""

    t0 = None
    app = None

    @classmethod
    async def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.perf_counter()
        cls.app = get_app(from_="test.toolbox", name="test-debug")
        cls.app.mlm = "I"
        cls.app.debug = True
        await cls.app.load_all_mods_in_file()

    @classmethod
    def tearDownClass(cls):
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.perf_counter() - cls.t0}")

    def setUp(self):
        self.app.logger.info(Style.BEIGEBG(f"Next Test"))
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # self.app.remove_all_modules()
        self.app.logger.info(Style.BEIGEBG(f"tearDown"))

    def test_crypt(self):
        t0 = time.perf_counter()
        self.app.logger.info(Style.GREYBG("Testing crypt"))
        test_string = "1234567890abcdefghijklmnop"
        code = Code()
        self.app.logger.info(Style.WHITE("encode test string"))
        encode_string = code.encode_code(test_string)
        self.app.logger.info(Style.WHITE("test for differences between encode_string and test_string"))
        if encode_string == test_string:
            self.app.logger.warning(Style.YELLOW(f"No crypt active please init or crate owen "))

        self.app.logger.info(Style.WHITE("decode test string"))
        out_string = code.decode_code(encode_string)
        self.app.logger.info(f" {test_string=} {encode_string=} {out_string=} don in {time.perf_counter() - t0}")

        self.app.logger.info(Style.WHITE("Test if test_string and out_string are equal"))
        self.assertEqual(test_string, out_string)

        seed0 = 0
        seed = code.generate_seed()
        seed2 = code.generate_seed()

        print(f"Generating seeds {seed} ,, {seed2}")
        self.assertNotEqual(seed, seed2)

        hash0 = code.one_way_hash(test_string)
        hash1 = code.one_way_hash(test_string, 'something-')
        hash = code.one_way_hash(test_string)

        print(f"Generating hashs {hash} ,, {hash0}")
        self.assertEqual(hash, hash0)
        self.assertNotEqual(hash, hash1)

        key0 = code.generate_symmetric_key()
        key = code.generate_symmetric_key()

        print(f"Generating keys {key} ,, {key0}")
        self.assertNotEqual(key, key0)

        t0 = code.encrypt_symmetric(test_string, key0)
        print(f"encrypt_symmetric {t0}")
        self.assertNotEqual(t0, test_string)

        t1 = code.decrypt_symmetric(t0, key0)
        print(f"og:data {t1}")
        self.assertEqual(test_string, t1)
        t1 = code.decrypt_symmetric(t0, key)
        self.assertNotEqual(test_string, t1)

        pem_public_key1, pem_private_key1 = code.generate_asymmetric_keys()
        pem_public_key2, pem_private_key2 = code.generate_asymmetric_keys()

        self.assertNotEqual(pem_public_key1, pem_public_key2)
        self.assertNotEqual(pem_public_key2, pem_private_key2)
        self.assertNotEqual(pem_private_key1, pem_public_key1)
        self.assertNotEqual(pem_private_key2, pem_private_key1)

        t0 = code.encrypt_asymmetric(test_string, pem_public_key2)
        print(f"encrypt_asymmetric {t0}")
        self.assertNotEqual(t0, test_string)

        t1 = code.decrypt_asymmetric(t0, pem_private_key2)
        print(f"og:data {t1}")
        self.assertEqual(test_string, t1)
        t1 = code.decrypt_asymmetric(t0, pem_private_key1)
        self.assertNotEqual(test_string, t1)

        t0 = code.encrypt_asymmetric(test_string, pem_private_key2)
        t1 = code.decrypt_asymmetric(t0, pem_public_key2)
        self.assertNotEqual(test_string, t1)

    def test_file_handler(self):
        t0 = time.perf_counter()
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

        self.app.logger.info(Style.WHITE(f"finish testing in {time.perf_counter() - t0}"))

    def fh_test(self, test_value):
        t0 = time.perf_counter()
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
        self.app.logger.info(f"don testing FileHandler in {time.perf_counter() - t0}")
        self.app.logger.info(Style.WHITE("cleaning up"))
        fh2.delete_file()

    async def test_main_tool(self):
        main_tool = await MainTool(v="1.0.0", tool={}, name="TestTool", logs=[], color="RED", on_exit=None, load=None)
        main_tool.print("Hello, world!")
        # uid, err = main_tool.get_uid([ob, ], self.app)
        # self.assertTrue(err)
        # print(uid)

    def test_styles(self):
        st = Style()
        st.color_demo()

    def test_save_instance(self):
        # Testen der save_instance-Funktion
        res = self.app.save_instance(None, 'welcome', 'app2', 'file/application')
        # Überprüfen Sie, ob die Instanz korrekt gespeichert wurde
        self.assertIn('welcome', self.app.functions)
        self.assertIsNone(res)
        # Weitere Überprüfungen je nach Funktionslogik

    def test_mod_online(self):
        # Testen der mod_online-Funktion
        self.app.remove_all_modules(True)
        online = self.app.mod_online('welcome')
        # Überprüfen Sie, ob das Modul als online markiert ist
        self.assertFalse(online)
        self.app.get_mod("welcome")
        online = self.app.mod_online('welcome')
        # Überprüfen Sie, ob das Modul als online markiert ist
        self.assertTrue(online)

    def test_get_function(self):
        # Testen der _get_function-Funktion
        self.app.get_mod("welcome")
        result, e = self.app._get_function(None, as_str=("welcome", "Version"))
        # Überprüfen Sie das Ergebnis
        self.assertEqual(e, 0)
        self.assertIsNotNone(result)
        """
    def _get_function(self,
                      name: Enum or None,
                      state: bool = True,
                      specification: str = "app",
                      metadata=False, as_str: tuple or None = None):
        pass
        # if function is None:
        #     self.logger.warning(f"No function found")
        #     return "404", 300
        # if metadata and not state:
        #     self.logger.info(f"returning metadata stateless")
        #     return (function_data, None), 0
        # if not state:  # mens a stateless function
        #     self.logger.info(f"returning stateless function")
        #     return function, 0
        # # instance = self.functions[modular_id].get(f"{specification}_instance")
        # if instance is None:
        #     return "404", 400
        #     if metadata:
        #         self.logger.info(f"returning metadata stateless")
        #         return (function_data, function), 0
        #     return function, 0
        # if metadata:
        #     self.logger.info(f"returning metadata stateful")
        #     return (function_data, higher_order_function), 0
        # self.logger.info(f"returning stateful function")
        # return higher_order_function, 0"""

    def test_load_mod(self):
        # Testen der load_mod-Funktion
        result = self.app.load_mod('welcome')
        self.assertIsNotNone(result)
        # self.app.mlm = 'C'
        result = self.app.load_mod('welcome', spec="welcome")
        self.assertEqual(result.spec, "welcome")
        # self.app.mlm = 'I'
        # # Überprüfen Sie das Ergebnis
        # self.assertIsNotNone(result)
        # result = self.app.load_mod('some_mod_name')
        # # Überprüfen Sie das Ergebnis
        # self.assertIsNotNone(result)

    async def test_load_all_mods_in_file(self):
        # Testen der load_all_mods_in_file-Funktion
        result = await self.app.load_all_mods_in_file()
        # Überprüfen Sie das Ergebnis
        self.assertTrue(result)

    def test_get_all_mods(self):
        # Testen der get_all_mods-Funktion
        mods = self.app.get_all_mods()
        # Überprüfen Sie das Ergebnis
        self.assertIsInstance(mods, list)
        self.assertIsInstance(mods[0], str)

    def test_remove_all_modules(self):
        # Testen der remove_all_modules-Funktion
        self.app.remove_all_modules(delete=True)
        # Überprüfen Sie, ob alle Module entfernt wurden
        for i in self.app.functions.keys():
            print(i)
        self.assertEqual(self.app.functions, {})

    def test_remove_mod(self):
        # Testen der remove_mod-Funktion
        self.app.remove_mod('welcome')
        # Überprüfen Sie, ob das Modul entfernt wurde
        self.assertNotIn('welcome', self.app.functions)

        self.app.remove_mod('some_mod_name')
        # Überprüfen Sie, ob das Modul entfernt wurde
        self.assertNotIn('some_mod_name', self.app.functions)

    def test_get_mod(self):
        # Testen der get_mod-Funktion
        mod = self.app.get_mod('welcome')
        # Überprüfen Sie, ob das Modul korrekt geladen wurde
        self.assertIsNotNone(mod)
        mod.printc("test123")
        self.app.remove_mod('welcome')
        mod = self.app.get_mod('welcome')
        mod.printc("test123")
        # Überprüfen Sie, ob das Modul korrekt geladen wurde
        self.assertIsNotNone(mod)
        # Weitere Überprüfungen je nach Funktionslogik

    def test_run(self):
        self.app.remove_mod("welcome")
        self.assertEqual(self.app.run_any(("welcome", "Version")), self.app.run_function(("welcome", "Version")).get())

    async def test_all_functions(self):
        print("STARTING test")
        res = await self.app.execute_all_functions()
        print("RES: ", res.result.data_info)
        # data = res.get()
        # print(res.result.data_info)
        # time.sleep(15)
        self.assertEqual(res.get('modular_run', 0), res.get('modular_sug', -1))
        print("DONE RUNNING ALL FUNCTIONS")


# This allows running the async tests with `unittest`
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))

    return wrapper


# Apply async_test decorator to each async test method
TestToolboxv2.test_all_functions = async_test(TestToolboxv2.test_all_functions)
TestToolboxv2.setUpClass = async_test(TestToolboxv2.setUpClass)
TestToolboxv2.test_main_tool = async_test(TestToolboxv2.test_main_tool)
TestToolboxv2.test_load_all_mods_in_file = async_test(TestToolboxv2.test_load_all_mods_in_file)
