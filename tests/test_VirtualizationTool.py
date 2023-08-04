import time
import unittest

from toolboxv2 import App


class TestVirtualizationTool(unittest.TestCase):
    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App('test-VirtualizationTool')
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.load_mod('VirtualizationTool')
        cls.tool = cls.app.get_mod('VirtualizationTool')
        cls.app.new_ac_mod('VirtualizationTool')

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.time() - cls.t0}')

    def test_show_version(self):
        expected_version = "0.0.2"
        actual_version = self.tool.show_version()
        self.assertEqual(expected_version, actual_version,
                         f'Expected version {expected_version}, but got {actual_version}')

    def test_create_instance(self):
        name = 'test_instance'
        mod_name = 'welcome'
        instance = self.tool.create_instance(name, mod_name)
        self.assertIsNotNone(instance, f'Failed to create instance {name} with mod {mod_name}')

    def test_set_ac_instances(self):
        name = 'test_instance'
        self.tool.create_instance(name, 'welcome')
        result = self.tool.set_ac_instances(name)
        self.assertTrue(result, f'Failed to set ac instance {name}')

    def test_list_instances(self):
        self.tool.create_instance('test_instance', 'welcome')
        self.tool.list_instances()  # This method doesn't return anything, it's just for printing the instances.

    def test_shear_function(self):
        name = 'test_instance'
        mod_name = 'welcome'
        function_name = 'show_version'
        self.tool.create_instance(name, mod_name)
        self.tool.shear_function(mod_name, name, function_name)
        self.assertTrue(hasattr(self.tool.instances[name], function_name),
                        f'Failed to shear function {function_name} to instance {name}')

    def test_flow(self):
        name = 'test_instance'
        mod_name = 'welcome'
        instance = self.tool.create_instance(name, mod_name)
        self.assertIsNotNone(instance, f'Failed to create instance {name} with mod {mod_name}')

        self.app.new_ac_mod("VirtualizationTool")
        print(f"Testing run_function ")
        if self.app.run_function('set-ac', name):
            self.app.run_function('api_' + "version", [])

        print(f"Testing run_any")

        self.app.run_any(name, 'api_' + "version", [])

        print("Testing run_any normal")
        self.app.run_any("welcome", 'api_' + "version", [])

        print("D")
        function_name = 'show_version'
        self.tool.shear_function(mod_name, name, function_name)
        self.assertTrue(hasattr(self.tool.instances[name], function_name),
                        f'Failed to shear function {function_name} to instance {name}')

        print(f"Testing run_any v with shear_function")
        self.app.run_any(name, 'api_' + "version", [])

        print("Testing run_any n with shear_function")
        self.app.run_any(mod_name, 'api_' + "version", [])


