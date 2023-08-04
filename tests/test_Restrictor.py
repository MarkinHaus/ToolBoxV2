import unittest
import time

from toolboxv2 import App
from toolboxv2.utils.toolbox import ApiOb


class TestRestrictor(unittest.TestCase):
    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App('test-Restrictor')
        cls.app.mlm = 'I'
        cls.app.debug = True
        cls.app.load_mod('Restrictor')
        cls.tool = cls.app.get_mod('Restrictor')
        cls.app.new_ac_mod('Restrictor')

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.time() - cls.t0}')

    def test_show_version(self):
        command = ApiOb(
            data={'username': 'test', 'password': 'test', 'email': 'test@test.com', 'invitation': 'test'},
            token='')
        res = self.app.run_function('Version', [command, ])
        self.assertEqual(res, '0.0.2')

    def test_restrict(self):
        mod = self.app.get_mod('welcome')
        function_name = 'api_show_version'
        by = 'trest'
        resid = 'resid'
        real_name = 'show_version'
        print("Running un restrict")
        mod.show_version()
        self.app.run_any('welcome', 'version')
        self.tool.restrict(mod, function_name, by, resid, real_name)
        self.assertIn(f'{by}-{function_name}', self.tool.seves.keys())
        print("Running restrict")
        mod.show_version()
        self.app.run_any('welcome', 'version')
        self.tool.un_lock(by, resid, function_name)
        print("Running un restrict")
        mod.show_version()
        self.app.run_any('welcome', 'version')
        self.assertNotIn(f'{by}-{function_name}', self.tool.seves.keys())
