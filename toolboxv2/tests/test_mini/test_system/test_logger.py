import logging
import os
import tempfile
import unittest

# Import the functions to be tested
from toolboxv2.utils.system.tb_logger import get_logger, setup_logging, unstyle_log_files


class TestLoggingImplementation(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for logs
        self.test_logs_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        for file in os.listdir(self.test_logs_dir):
            try:
                os.remove(os.path.join(self.test_logs_dir, file))
            except PermissionError:
                return
        os.rmdir(self.test_logs_dir)

    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        logger, filename = setup_logging(
            level=logging.INFO,
            logs_directory=self.test_logs_dir,
            app_name="test_app"
        )

        # Check logger is created
        self.assertIsNotNone(logger)

        # Check filename is generated correctly
        self.assertTrue(filename.startswith("Logs-toolboxV2-"))
        self.assertTrue(filename.endswith("-INFO"))

        # Verify log file is created
        log_files = os.listdir(self.test_logs_dir)
        self.assertTrue(any(filename in log_file for log_file in log_files))

    def test_setup_logging_invalid_levels(self):
        """Test setup_logging with invalid log levels"""
        with self.assertRaises(ValueError):
            setup_logging(level=999, logs_directory=self.test_logs_dir)

        with self.assertRaises(ValueError):
            setup_logging(level=1, online_level=999, logs_directory=self.test_logs_dir)

        with self.assertRaises(ValueError):
            setup_logging(level=1, file_level=999, logs_directory=self.test_logs_dir)

    def test_log_file_exists(self):
        """Test that log file exist"""
        logger, filename = setup_logging(
            level=logging.INFO,
            logs_directory=self.test_logs_dir,
            app_name="test_app"
        )

        # Find the log file and check its content
        log_files = os.listdir(self.test_logs_dir)
        log_file_path = [os.path.join(self.test_logs_dir, f) for f in log_files if filename in f][0]
        print(filename, log_file_path)
        self.assertTrue(os.path.exists(log_file_path))

    def test_get_logger(self):
        """Test get_logger() returns a logger"""
        setup_logging(
            level=logging.INFO,
            logs_directory=self.test_logs_dir,
            app_name="test_app"
        )

        logger = get_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_unstyle_log_files(self):
        """Test unstyle_log_files functionality"""
        # Create a sample log file with styles
        test_log_file = os.path.join(self.test_logs_dir, "test_styled_log.log")
        with open(test_log_file, 'w') as f:
            f.write("\033[31mRed styled log\033[0m")

        # Remove styles
        unstyle_log_files(test_log_file)

        # Check file content
        with open(test_log_file) as f:
            content = f.read()
            self.assertEqual(content.strip(), "Red styled log\n no-styles")


if __name__ == '__main__':
    unittest.main()
