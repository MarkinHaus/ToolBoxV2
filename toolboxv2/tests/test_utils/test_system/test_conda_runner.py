import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from toolboxv2 import get_app

# Import the functions to be tested
from toolboxv2.utils.system.conda_runner import (
    add_dependency,
    create_conda_env,
    create_env_registry,
    delete_conda_env,
    run_command,
    run_script_in_conda_env,
)


class TestCondaRunner(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a unique test environment name to avoid conflicts
        self.test_env_name = f"test_conda_runner_{os.getpid()}"

        # Create a temporary directory for file-based tests
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the test conda environment if it exists
        with contextlib.suppress(Exception):
            delete_conda_env(self.test_env_name)

        # Remove the temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_command_success(self):
        """Test run_command with a successful command."""
        if not get_app(name="test").local_test:
            return
        success, output = run_command("echo Hello World", live=False)
        self.assertTrue(success)
        self.assertIn("Hello World", output)

    def test_run_command_failure(self):
        """Test run_command with a failing command."""
        success, output = run_command("invalid_command_that_doesnt_exist", live=False)
        self.assertFalse(success)

    def test_create_conda_env(self):
        """Test creating a new conda environment."""
        # Create the environment
        if not get_app(name="test").local_test:
            return
        result = create_conda_env(self.test_env_name)
        self.assertTrue(result)

        # Check if the environment exists
        try:
            output = subprocess.check_output("conda env list", shell=True, text=True)
            self.assertIn(self.test_env_name, output)
        except subprocess.CalledProcessError:
            self.fail(f"Environment {self.test_env_name} was not created")

    def test_delete_conda_env(self):
        """Test deleting a conda environment."""
        # First create the environment
        if not get_app(name="test").local_test:
            return
        create_conda_env(self.test_env_name)

        # Then delete it
        result = delete_conda_env(self.test_env_name)
        self.assertTrue(result)

        # Check that the environment no longer exists
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_output(f"conda env list | grep {self.test_env_name}", shell=True, text=True)

    def test_add_dependency(self):
        """Test adding a dependency to a conda environment."""
        # Create a test environment
        if not get_app(name="test").local_test:
            return
        create_conda_env(self.test_env_name)

        # Add a simple dependency
        result = add_dependency(self.test_env_name, "numpy")
        self.assertTrue(result)

        # Verify the dependency was added by checking conda list
        try:
            output = subprocess.check_output(f"conda list -n {self.test_env_name} numpy", shell=True, text=True)
            self.assertIn("numpy", output)
        except subprocess.CalledProcessError:
            self.fail("NumPy was not added to the environment")

    def test_create_env_registry(self):
        """Test creating an environment registry."""
        # Create a test environment with some packages
        if not get_app(name="test").local_test:
            return
        create_conda_env(self.test_env_name)
        add_dependency(self.test_env_name, "numpy")
        add_dependency(self.test_env_name, "pandas")

        # Create registry
        create_env_registry(self.test_env_name)

        # Check if registry file was created
        registry_file = f"{self.test_env_name}_registry.json"
        self.assertTrue(os.path.exists(registry_file))

        # Verify registry contents
        with open(registry_file) as f:
            registry = json.load(f)

        if isinstance(registry[0], dict):
            registry = [r.get('name') for r in registry]
        if isinstance(registry[0], str):
            self.assertIn('numpy', registry)
            self.assertIn('pandas', registry)

    def test_run_script_in_conda_env(self):
        """Test running a script in a conda environment."""
        # Create a test environment
        if not get_app(name="test").local_test:
            return
        create_conda_env(self.test_env_name)

        # Create a test Python script
        test_script_path = os.path.join(self.test_dir, 'test_script.py')
        with open(test_script_path, 'w') as f:
            f.write("print('Script executed successfully')")

        # Run the script in the test environment
        success, output = run_script_in_conda_env(test_script_path, self.test_env_name, [])
        self.assertTrue(success)

    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Try to add a non-existent dependency
        if not get_app(name="test").local_test:
            return
        with self.assertRaises(Exception):
            res = add_dependency(self.test_env_name, "non_existent_package_xyz")
            if res is False:
                raise Exception


if __name__ == '__main__':
    unittest.main()
