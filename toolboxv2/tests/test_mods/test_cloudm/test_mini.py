# file: toolboxv2/tests/test_mods/test_cloudm/test_mini.py
"""
Tests for CloudM mini module.

Tests service status monitoring, PID management, and process checking:
- Service PID extraction from files
- Multi-process status checking
- Platform-specific process monitoring
- Service status display formatting
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
import subprocess

from toolboxv2.mods.CloudM.mini import (
    get_service_pids,
    check_multiple_processes,
    get_service_status,
    GREEN_CIRCLE,
    YELLOW_CIRCLE,
    RED_CIRCLE
)


class TestGetServicePids(unittest.TestCase):
    """Tests for get_service_pids function"""

    def setUp(self):
        """Create a temporary directory for test PID files"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_service_pids_empty_directory(self):
        """Test with empty directory"""
        services = get_service_pids(self.test_dir)
        self.assertEqual(services, {})

    def test_get_service_pids_single_service(self):
        """Test extracting PID from single service file"""
        # Create a test PID file
        pid_file = os.path.join(self.test_dir, "web-myservice.pid")
        with open(pid_file, 'w') as f:
            f.write("12345")

        services = get_service_pids(self.test_dir)

        self.assertEqual(len(services), 1)
        self.assertIn("myservice - web", services)
        self.assertEqual(services["myservice - web"], 12345)

    def test_get_service_pids_multiple_services(self):
        """Test extracting PIDs from multiple service files"""
        # Create multiple PID files
        pid_files = {
            "web-service1.pid": "1001",
            "api-service2.pid": "1002",
            "worker-service3.pid": "1003"
        }

        for filename, pid in pid_files.items():
            with open(os.path.join(self.test_dir, filename), 'w') as f:
                f.write(pid)

        services = get_service_pids(self.test_dir)

        self.assertEqual(len(services), 3)
        self.assertEqual(services["service1 - web"], 1001)
        self.assertEqual(services["service2 - api"], 1002)
        self.assertEqual(services["service3 - worker"], 1003)

    def test_get_service_pids_ignores_invalid_files(self):
        """Test that invalid filenames are ignored"""
        # Create valid and invalid files
        with open(os.path.join(self.test_dir, "web-valid.pid"), 'w') as f:
            f.write("1234")
        with open(os.path.join(self.test_dir, "invalid.txt"), 'w') as f:
            f.write("5678")
        with open(os.path.join(self.test_dir, "nopid"), 'w') as f:
            f.write("9999")

        services = get_service_pids(self.test_dir)

        self.assertEqual(len(services), 1)
        self.assertIn("valid - web", services)

    def test_get_service_pids_with_whitespace(self):
        """Test handling PIDs with whitespace"""
        pid_file = os.path.join(self.test_dir, "web-service.pid")
        with open(pid_file, 'w') as f:
            f.write("  12345  \n")

        services = get_service_pids(self.test_dir)

        self.assertEqual(services["service - web"], 12345)


class TestCheckMultipleProcesses(unittest.TestCase):
    """Tests for check_multiple_processes function"""

    def test_check_multiple_processes_empty_list(self):
        """Test with empty PID list"""
        result = check_multiple_processes([])
        self.assertEqual(result, {})

    @patch('os.name', 'posix')
    @patch('subprocess.run')
    def test_check_multiple_processes_unix_all_running(self, mock_run):
        """Test Unix process checking with all processes running"""
        mock_run.return_value = MagicMock(
            stdout="1001\n1002\n1003\n",
            returncode=0
        )

        pids = [1001, 1002, 1003]
        result = check_multiple_processes(pids)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[1001], GREEN_CIRCLE)
        self.assertEqual(result[1002], GREEN_CIRCLE)
        self.assertEqual(result[1003], GREEN_CIRCLE)

    @patch('os.name', 'posix')
    @patch('subprocess.run')
    def test_check_multiple_processes_unix_some_stopped(self, mock_run):
        """Test Unix process checking with some processes stopped"""
        mock_run.return_value = MagicMock(
            stdout="1001\n1003\n",
            returncode=0
        )

        pids = [1001, 1002, 1003]
        result = check_multiple_processes(pids)

        self.assertEqual(result[1001], GREEN_CIRCLE)
        self.assertEqual(result[1002], RED_CIRCLE)
        self.assertEqual(result[1003], GREEN_CIRCLE)

    @patch('os.name', 'posix')
    @patch('subprocess.run')
    def test_check_multiple_processes_unix_error(self, mock_run):
        """Test Unix process checking with subprocess error"""
        mock_run.side_effect = subprocess.SubprocessError("Test error")

        pids = [1001, 1002]
        result = check_multiple_processes(pids)

        # All should be marked as YELLOW on error
        self.assertEqual(result[1001], YELLOW_CIRCLE)
        self.assertEqual(result[1002], YELLOW_CIRCLE)

    @patch('os.name', 'nt')
    @patch('subprocess.run')
    def test_check_multiple_processes_windows_running(self, mock_run):
        """Test Windows process checking with running processes"""
        mock_run.return_value = MagicMock(
            stdout="python.exe    1001 Console\npython.exe    1002 Console\n",
            returncode=0
        )

        pids = [1001, 1002]
        result = check_multiple_processes(pids)

        self.assertEqual(result[1001], GREEN_CIRCLE)
        self.assertEqual(result[1002], GREEN_CIRCLE)

    @patch('os.name', 'nt')
    @patch('subprocess.run')
    def test_check_multiple_processes_windows_not_running(self, mock_run):
        """Test Windows process checking with stopped processes"""
        mock_run.return_value = MagicMock(
            stdout="python.exe    9999 Console\n",
            returncode=0
        )

        pids = [1001, 1002]
        result = check_multiple_processes(pids)

        self.assertEqual(result[1001], RED_CIRCLE)
        self.assertEqual(result[1002], RED_CIRCLE)

    @patch('os.name', 'nt')
    @patch('subprocess.run')
    def test_check_multiple_processes_windows_encoding_error(self, mock_run):
        """Test Windows process checking with encoding fallback"""
        # First call fails with UnicodeDecodeError
        mock_run.side_effect = [
            UnicodeDecodeError('cp850', b'', 0, 1, 'test'),
            MagicMock(stdout="python.exe    1001 Console\n", returncode=0)
        ]

        pids = [1001]
        result = check_multiple_processes(pids)

        # Should retry with utf-8 and succeed
        self.assertEqual(result[1001], GREEN_CIRCLE)


class TestGetServiceStatus(unittest.TestCase):
    """Tests for get_service_status function"""

    def setUp(self):
        """Create a temporary directory for test"""
        self.test_dir = tempfile.mkdtemp()
        # Clear cache
        import toolboxv2.mods.CloudM.mini as mini_module
        mini_module.services_data_sto[0] = {}
        mini_module.services_data_sto_last_update_time[0] = 0

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_service_status_no_services(self):
        """Test status display with no services"""
        status = get_service_status(self.test_dir)
        self.assertEqual(status, "No services found")

    @patch('toolboxv2.mods.CloudM.mini.check_multiple_processes')
    def test_get_service_status_single_service(self, mock_check):
        """Test status display with single service"""
        # Create a PID file
        with open(os.path.join(self.test_dir, "web-myservice.pid"), 'w') as f:
            f.write("1234")

        # Mock process check
        mock_check.return_value = {1234: GREEN_CIRCLE}

        status = get_service_status(self.test_dir)

        self.assertIn("Service(s):", status)
        self.assertIn("myservice - web", status)
        self.assertIn("1234", status)
        self.assertIn(GREEN_CIRCLE, status)

    @patch('toolboxv2.mods.CloudM.mini.check_multiple_processes')
    def test_get_service_status_multiple_services(self, mock_check):
        """Test status display with multiple services"""
        # Create multiple PID files
        with open(os.path.join(self.test_dir, "web-service1.pid"), 'w') as f:
            f.write("1001")
        with open(os.path.join(self.test_dir, "api-service2.pid"), 'w') as f:
            f.write("1002")

        # Mock process check
        mock_check.return_value = {
            1001: GREEN_CIRCLE,
            1002: RED_CIRCLE
        }

        status = get_service_status(self.test_dir)

        self.assertIn("service1 - web", status)
        self.assertIn("service2 - api", status)
        self.assertIn("1001", status)
        self.assertIn("1002", status)
        self.assertIn(GREEN_CIRCLE, status)
        self.assertIn(RED_CIRCLE, status)

    @patch('toolboxv2.mods.CloudM.mini.check_multiple_processes')
    @patch('time.time')
    def test_get_service_status_caching(self, mock_time, mock_check):
        """Test that service status is cached for 30 seconds"""
        # Create a PID file
        with open(os.path.join(self.test_dir, "web-service.pid"), 'w') as f:
            f.write("1234")

        mock_check.return_value = {1234: GREEN_CIRCLE}

        # First call at time 0
        mock_time.return_value = 0
        status1 = get_service_status(self.test_dir)
        call_count_1 = mock_check.call_count

        # Second call at time 10 (within 30 seconds)
        mock_time.return_value = 10
        status2 = get_service_status(self.test_dir)
        call_count_2 = mock_check.call_count

        # Should use cache, no new call to check_multiple_processes
        self.assertEqual(call_count_1, call_count_2)

        # Third call at time 35 (after 30 seconds)
        mock_time.return_value = 35
        status3 = get_service_status(self.test_dir)
        call_count_3 = mock_check.call_count

        # Should refresh, new call to check_multiple_processes
        self.assertGreater(call_count_3, call_count_2)


class TestServiceStatusIntegration(unittest.TestCase):
    """Integration tests for service status monitoring"""

    def setUp(self):
        """Create a temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        # Clear cache
        import toolboxv2.mods.CloudM.mini as mini_module
        mini_module.services_data_sto[0] = {}
        mini_module.services_data_sto_last_update_time[0] = 0

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('toolboxv2.mods.CloudM.mini.check_multiple_processes')
    def test_full_service_monitoring_workflow(self, mock_check):
        """Test complete workflow of service monitoring"""
        # Create service PID files
        services = {
            "web-frontend.pid": "2001",
            "api-backend.pid": "2002",
            "worker-processor.pid": "2003"
        }

        for filename, pid in services.items():
            with open(os.path.join(self.test_dir, filename), 'w') as f:
                f.write(pid)

        # Mock some running, some stopped
        mock_check.return_value = {
            2001: GREEN_CIRCLE,
            2002: GREEN_CIRCLE,
            2003: RED_CIRCLE
        }

        status = get_service_status(self.test_dir)

        # Verify all services are in output
        self.assertIn("frontend - web", status)
        self.assertIn("backend - api", status)
        self.assertIn("processor - worker", status)

        # Verify status indicators
        lines = status.split('\n')
        self.assertGreater(len(lines), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

