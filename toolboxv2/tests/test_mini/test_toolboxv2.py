#!/usr/bin/env python
"""Tests for `toolboxv2` package.

Design:
- One App instance, one event loop, shared across all tests.
- Modules loaded ONCE in setUpClass via async load_all_mods_in_file.
- Teardown uses a_exit() (async) to properly await on_exit hooks.
- Test order is explicit via numbered prefixes to match the module lifecycle:
    01-02: crypto & file_handler (no module dependency)
    03:    styles (no module dependency)
    04:    save_instance / load_mod / get_mod / mod_online (module CRUD)
    05:    run / all_functions (execution against loaded modules)
    06:    remove_mod / remove_all_modules (destructive — last)
- xdist: mark the whole class serial — these tests share state by design.
"""
import asyncio
import os
import time
import unittest

from cryptography.fernet import InvalidToken

from toolboxv2 import Style
from toolboxv2.utils.system.file_handler import FileHandler
from toolboxv2.utils.system.main_tool import MainTool
from toolboxv2.utils.toolbox import App
from toolboxv2.tests.a_util import PersistentAppTestCase
from toolboxv2.utils.security.cryp import Code

try:
    from rich.traceback import install
    install(show_locals=True)
except ImportError:
    pass

try:
    import pytest
    # Prevent xdist from distributing these tests across workers.
    # They MUST run in one process sequentially.
    _serial = pytest.mark.serial
except ImportError:
    def _serial(cls):
        return cls


def _run_async(coro):
    """Run a coroutine on the class-shared event loop (must exist)."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Fallback: create a new loop (should not happen in normal flow)
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    return loop.run_until_complete(coro)


@_serial
class TestToolboxv2Mods(PersistentAppTestCase):
    """Integration tests for toolboxv2 modules.

    Lifecycle:
        setUpClass  → create App, load all mods (once)
        test_01_*   → pure-utility tests (crypto, file_handler, styles)
        test_02_*   → module CRUD (save_instance, load_mod, get_mod, mod_online)
        test_03_*   → execution (run, all_functions)
        test_04_*   → destructive (remove_mod, remove_all_modules)
        tearDownClass → async exit (awaits all on_exit hooks)
    """

    app: App = None
    _loop: asyncio.AbstractEventLoop = None
    t0: float = None

    # ------------------------------------------------------------------
    # Setup / Teardown
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print("Setting up TestToolboxv2Mods")
        cls.t0 = time.perf_counter()

        # One event loop for the entire class
        cls._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls._loop)

        cls.app = App()
        cls.app.mlm = "I"
        cls.app.debug = True

        # Load all mods ONCE
        result = cls._loop.run_until_complete(cls.app.load_all_mods_in_file())
        print(f"setUpClass: {result}")

    @classmethod
    def tearDownClass(cls):
        if cls.app and cls.app.alive:
            cls.app.logger.info(f"Accomplished in {time.perf_counter() - cls.t0}")
            # Use async exit to properly await on_exit hooks (SocketManager etc.)
            try:
                cls._loop.run_until_complete(cls.app.a_exit())
            except Exception as e:
                print(f"tearDownClass a_exit error (ignored): {e}")
                # Fallback: sync exit
                try:
                    cls.app.exit()
                except Exception:
                    pass

        # Close the event loop
        if cls._loop and not cls._loop.is_closed():
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(cls._loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    cls._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            finally:
                cls._loop.close()
                asyncio.set_event_loop(None)

        cls.app = None
        cls._loop = None
        super().tearDownClass()

    def setUp(self):
        self.app.logger.info(Style.BEIGEBG(f">> {self._testMethodName}"))

    def tearDown(self):
        self.app.logger.info(Style.BEIGEBG(f"<< {self._testMethodName}"))

    # ------------------------------------------------------------------
    # 01 — Pure utility tests (no module dependency)
    # ------------------------------------------------------------------

    def test_01a_crypt(self):
        """Crypto round-trips: encode/decode, hash, symmetric, asymmetric."""
        test_string = "1234567890abcdefghijklmnop"
        code = Code()

        # Encode / decode
        encoded = code.encode_code(test_string)
        decoded = code.decode_code(encoded)
        self.assertEqual(test_string, decoded)

        # Seeds uniqueness
        self.assertNotEqual(code.generate_seed(), code.generate_seed())

        # Hash determinism + salt sensitivity
        h1 = code.one_way_hash(test_string)
        h2 = code.one_way_hash(test_string)
        h_salted = code.one_way_hash(test_string, 'something-')
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h_salted)

        # Symmetric encrypt/decrypt
        key_a = code.generate_symmetric_key()
        key_b = code.generate_symmetric_key()
        self.assertNotEqual(key_a, key_b)

        cipher = code.encrypt_symmetric(test_string, key_a)
        self.assertNotEqual(cipher, test_string)
        self.assertEqual(test_string, code.decrypt_symmetric(cipher, key_a))
        with self.assertRaises(InvalidToken):
            code.decrypt_symmetric(cipher, key_b)

        # Asymmetric encrypt/decrypt
        pub1, priv1 = code.generate_asymmetric_keys()
        pub2, priv2 = code.generate_asymmetric_keys()
        self.assertNotEqual(pub1, pub2)

        cipher = code.encrypt_asymmetric(test_string, pub2)
        self.assertNotEqual(cipher, test_string)
        self.assertEqual(test_string, code.decrypt_asymmetric(cipher, priv2))
        # Wrong key → should not decrypt correctly
        self.assertNotEqual(test_string, code.decrypt_asymmetric(cipher, priv1))

    def test_01b_file_handler(self):
        """FileHandler: CRUD round-trip for various value types."""
        test_values = [
            "", 0, [], {},
            "test", 124354, [1233, "3232"],
            {"test": "test", "value": -1}, [0, 0, 0, 0],
        ]
        for val in test_values:
            with self.subTest(value=val):
                self._fh_roundtrip(val)

    def _fh_roundtrip(self, test_value):
        config_path = os.path.join("config", "mainTool", "test.config")
        if os.path.exists(config_path):
            os.remove(config_path)

        fh = FileHandler(
            "test.config",
            keys={"TestKey": "test~~~~~:"},
            defaults={"TestKey": "Default"},
        )
        self.assertEqual(fh.file_handler_filename, "test.config")
        self.assertIsNone(fh.file_handler_storage)

        fh.load_file_handler()
        self.assertIsNone(fh.file_handler_storage)

        # Default value accessible via both key and raw key
        v1 = fh.get_file_handler("TestKey")
        v2 = fh.get_file_handler("test~~~~~:")
        self.assertEqual(v1, v2)

        # Write, read back
        self.assertTrue(fh.add_to_save_file_handler("test~~~~~:", test_value))
        self.assertEqual(fh.get_file_handler("TestKey"), test_value)

        # Persist and reload
        fh.save_file_handler()
        del fh

        fh2 = FileHandler(
            "test.config",
            keys={"TestKey": "test~~~~~:"},
            defaults={"TestKey": "Default"},
        )
        fh2.load_file_handler()
        self.assertEqual(fh2.get_file_handler("TestKey"), test_value)
        fh2.delete_file()

    def test_01c_main_tool(self):
        """MainTool instantiation."""
        async def _test():
            mt = await MainTool(
                v="1.0.0", tool={}, name="TestTool",
                logs=[], color="RED", on_exit=None, load=None,
            )
            mt.print("Hello, world!")
        _run_async(_test())

    def test_01d_styles(self):
        """Style color demo runs without error."""
        Style().color_demo()

    # ------------------------------------------------------------------
    # 02 — Module CRUD (non-destructive)
    # ------------------------------------------------------------------

    def test_02a_save_instance(self):
        res = self.app.save_instance(None, 'welcome', 'app2', 'file/application')
        self.assertIn('welcome', self.app.functions)
        self.assertIsNone(res)

    def test_02b_load_mod(self):
        result = self.app.load_mod('welcome')
        self.assertIsNotNone(result)
        result = self.app.load_mod('welcome', spec="welcome")
        self.assertEqual(result.spec, "welcome")

    def test_02c_get_mod(self):
        mod = self.app.get_mod('welcome')
        self.assertIsNotNone(mod)
        mod.printc("test123")

        # Remove and re-get (lazy load)
        self.app.remove_mod('welcome')
        mod = self.app.get_mod('welcome')
        self.assertIsNotNone(mod)
        mod.printc("test123")

    def test_02d_mod_online(self):
        self.app.remove_all_modules(True)
        self.assertFalse(self.app.mod_online('welcome'))
        self.app.get_mod('welcome')
        self.assertTrue(self.app.mod_online('welcome'))

    def test_02e_get_all_mods(self):
        mods = self.app.get_all_mods()
        self.assertIsInstance(mods, list)
        self.assertTrue(len(mods) > 0)
        self.assertIsInstance(mods[0], str)

    # ------------------------------------------------------------------
    # 03 — Execution
    # ------------------------------------------------------------------

    def test_03a_run(self):
        """run_any vs run_function equivalence."""
        # Ensure welcome is loaded
        if not self.app.mod_online('welcome'):
            self.app.get_mod('welcome')
        self.assertEqual(
            self.app.run_any(("welcome", "Version")),
            self.app.run_function(("welcome", "Version")).get(),
        )

    def test_03b_all_functions(self):
        """Execute all registered module test functions."""
        async def _test():
            if "test" not in self.app.id:
                self.app.id += "test"
            # Do NOT reload modules — they are already loaded from setUpClass.
            # Just ensure we have a data_dir for test artifacts.
            self.app.data_dir = "../test_mods/.test_data"

            # Reload modules only if they were removed by earlier tests
            await self.app.load_all_mods_in_file()

            res = await self.app.execute_all_functions_(test_class=self)
            print("RES:", res.result.data_info)
            self.assertEqual(
                res.get('modular_run', 0),
                res.get('modular_sug', -1),
            )
        _run_async(_test())

    # ------------------------------------------------------------------
    # 04 — Destructive tests (run last)
    # ------------------------------------------------------------------

    def test_04a_remove_mod(self):
        if not self.app.mod_online('welcome'):
            self.app.get_mod('welcome')
        self.app.remove_mod('welcome')
        self.assertNotIn('welcome', self.app.functions)
        # Removing non-existent mod should not raise
        self.app.remove_mod('some_mod_name')
        self.assertNotIn('some_mod_name', self.app.functions)

    def test_04b_remove_all_modules(self):
        """Must be the very last test — removes everything."""
        self.app.remove_all_modules(delete=True)
        self.assertEqual(self.app.functions, {})


if __name__ == '__main__':
    unittest.main()
