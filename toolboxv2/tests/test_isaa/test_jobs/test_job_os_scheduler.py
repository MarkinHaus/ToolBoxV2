"""
Tests for OS Scheduler
======================

Tests platform-specific auto-wake registration (schtasks / crontab / LaunchAgent)
via mocked subprocess calls, plus real file-system integration tests.

Run:
    pytest toolboxv2/tests/test_jobs/test_job_os_scheduler.py -v
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from toolboxv2.mods.isaa.extras.jobs.os_scheduler import (
    _TASK_NAME,
    _CHECK_INTERVAL_MINUTES,
    autowake_status,
    install_autowake,
    remove_autowake,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _jobs_file() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    f.write("[]")
    f.close()
    return Path(f.name)


def _mock_run_ok():
    return MagicMock(returncode=0, stdout="", stderr="")


# =============================================================================
# Windows
# =============================================================================

class TestWindowsScheduler(unittest.TestCase):
    """schtasks install / remove / status via mocked subprocess."""

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_creates_two_tasks(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = _mock_run_ok()
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        result = install_autowake(_jobs_file())

        self.assertIn("installed", result.lower())
        self.assertEqual(mock_sub.run.call_count, 2)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_periodic_task_args(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = _mock_run_ok()
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        install_autowake(_jobs_file())

        periodic_args = mock_sub.run.call_args_list[0][0][0]
        self.assertIn("schtasks", periodic_args)
        self.assertIn("/Create", periodic_args)
        self.assertIn("MINUTE", periodic_args)
        self.assertIn(str(_CHECK_INTERVAL_MINUTES), periodic_args)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_boot_task_args(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = _mock_run_ok()
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        install_autowake(_jobs_file())

        boot_args = mock_sub.run.call_args_list[1][0][0]
        self.assertIn("ONSTART", boot_args)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_periodic_failure_returns_error_message(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.run.side_effect = subprocess.CalledProcessError(1, "schtasks", stderr="Access denied")

        result = install_autowake(_jobs_file())
        self.assertIn("Failed", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_boot_failure_reports_partial(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.run.side_effect = [
            _mock_run_ok(),
            subprocess.CalledProcessError(1, "schtasks", stderr="boot denied"),
        ]
        result = install_autowake(_jobs_file())
        self.assertIn("boot task failed", result.lower())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_remove_deletes_both_tasks(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = _mock_run_ok()
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        result = remove_autowake()

        self.assertIn("removed", result.lower())
        self.assertEqual(mock_sub.run.call_count, 2)
        for c in mock_sub.run.call_args_list:
            self.assertIn("/Delete", c[0][0])

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_remove_partial_failure_reports_which(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.run.side_effect = [
            _mock_run_ok(),
            subprocess.CalledProcessError(1, "schtasks"),
        ]
        result = remove_autowake()
        self.assertIn("Partially", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_status_parses_ready(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = MagicMock(returncode=0, stdout="Status: Ready\n")
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        result = autowake_status()
        self.assertIn("status", result.lower())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_status_not_found(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = MagicMock(returncode=1, stdout="")
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        result = autowake_status()
        self.assertIn("Not found", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_task_name_contains_marker(self, mock_sub, mock_plat):
        """Both created tasks must include the ISAA task name for idempotent removal."""
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = _mock_run_ok()
        mock_sub.CalledProcessError = subprocess.CalledProcessError

        install_autowake(_jobs_file())

        for c in mock_sub.run.call_args_list:
            args = c[0][0]
            # /TN value should contain the base task name
            tn_idx = args.index("/TN")
            self.assertIn(_TASK_NAME, args[tn_idx + 1])


# =============================================================================
# Linux
# =============================================================================

class TestLinuxScheduler(unittest.TestCase):
    """crontab install / remove / status via mocked subprocess."""

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_adds_entries_to_crontab(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        existing = "# existing cron\n0 * * * * echo test\n"
        mock_sub.run.side_effect = [
            MagicMock(returncode=0, stdout=existing),
            MagicMock(returncode=0),
        ]

        result = install_autowake(_jobs_file())

        self.assertIn("installed", result.lower())
        self.assertIn("crontab", result.lower())

        written = mock_sub.run.call_args_list[1][1]["input"]
        self.assertIn(_TASK_NAME, written)
        self.assertIn("@reboot", written)
        self.assertIn(f"*/{_CHECK_INTERVAL_MINUTES}", written)
        self.assertIn("echo test", written)  # existing entries preserved

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_replaces_old_entries(self, mock_sub, mock_plat):
        """Running install twice must not duplicate the ISAA lines."""
        mock_plat.system.return_value = "Linux"
        old_entry = (
            f"# {_TASK_NAME} - ISAA Job Runner (auto-generated)\n"
            f"*/15 * * * * python -m runner # {_TASK_NAME}\n"
            f"@reboot python -m runner # {_TASK_NAME}\n"
        )
        mock_sub.run.side_effect = [
            MagicMock(returncode=0, stdout=old_entry),
            MagicMock(returncode=0),
        ]

        install_autowake(_jobs_file())

        written = mock_sub.run.call_args_list[1][1]["input"]
        # Should appear only once (old removed, new added)
        count = written.count("@reboot")
        self.assertEqual(count, 1)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_empty_crontab(self, mock_sub, mock_plat):
        """No existing crontab should be handled gracefully."""
        mock_plat.system.return_value = "Linux"
        mock_sub.run.side_effect = [
            MagicMock(returncode=1, stdout=""),  # crontab -l → no crontab
            MagicMock(returncode=0),
        ]

        result = install_autowake(_jobs_file())
        self.assertIn("installed", result.lower())

        written = mock_sub.run.call_args_list[1][1]["input"]
        self.assertIn(_TASK_NAME, written)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_write_failure(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=1, stderr="permission denied"),
        ]

        result = install_autowake(_jobs_file())
        self.assertIn("Failed", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_remove_strips_isaa_lines_preserves_others(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        existing = (
            "# existing\n"
            "0 * * * * echo test\n"
            f"*/15 * * * * python -m runner  # {_TASK_NAME}\n"
            f"# {_TASK_NAME} - auto-generated\n"
            f"@reboot python -m runner  # {_TASK_NAME}\n"
        )
        mock_sub.run.side_effect = [
            MagicMock(returncode=0, stdout=existing),
            MagicMock(returncode=0),
        ]

        result = remove_autowake()

        self.assertIn("removed", result.lower())
        written = mock_sub.run.call_args_list[1][1]["input"]
        self.assertNotIn(_TASK_NAME, written)
        self.assertIn("echo test", written)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_remove_no_crontab_safe(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.return_value = MagicMock(returncode=1, stdout="")

        result = remove_autowake()
        self.assertIn("No crontab", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_status_installed(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.return_value = MagicMock(
            returncode=0,
            stdout=f"*/15 * * * * python -m runner  # {_TASK_NAME}\n"
        )

        result = autowake_status()
        self.assertIn("Installed", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_status_not_installed(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.return_value = MagicMock(returncode=0, stdout="0 * * * * something_else\n")

        result = autowake_status()
        self.assertIn("Not installed", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_status_no_crontab(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.return_value = MagicMock(returncode=1, stdout="")

        result = autowake_status()
        self.assertIn("Not installed", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_includes_jobs_file_path(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0),
        ]
        jf = _jobs_file()
        install_autowake(jf)

        written = mock_sub.run.call_args_list[1][1]["input"]
        self.assertIn(str(jf), written)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_install_exception_returns_error(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.side_effect = Exception("unexpected")

        result = install_autowake(_jobs_file())
        self.assertIn("Failed", result)


# =============================================================================
# macOS
# =============================================================================

class TestMacOSScheduler(unittest.TestCase):
    """LaunchAgent plist install / remove / status via mocked subprocess + real tmp file."""

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_install_writes_valid_plist(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = _mock_run_ok()

        result = install_autowake(_jobs_file())

        self.assertIn("installed", result.lower())
        self.assertIn("LaunchAgent", result)
        self.assertTrue(tmp.exists())

        content = tmp.read_text()
        self.assertIn("com.isaa.jobrunner", content)
        self.assertIn("StartInterval", content)
        self.assertIn("RunAtLoad", content)
        tmp.unlink(missing_ok=True)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_install_plist_contains_jobs_file(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = _mock_run_ok()

        jf = _jobs_file()
        install_autowake(jf)

        content = tmp.read_text()
        self.assertIn(str(jf), content)
        tmp.unlink(missing_ok=True)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_install_interval_matches_constant(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = _mock_run_ok()

        install_autowake(_jobs_file())

        content = tmp.read_text()
        expected_secs = str(_CHECK_INTERVAL_MINUTES * 60)
        self.assertIn(expected_secs, content)
        tmp.unlink(missing_ok=True)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_install_launchctl_load_failure(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.side_effect = [
            _mock_run_ok(),  # unload (no-op)
            MagicMock(returncode=1, stderr="plist error"),  # load fails
        ]

        result = install_autowake(_jobs_file())
        self.assertIn("launchctl load failed", result.lower())
        tmp.unlink(missing_ok=True)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_remove_unlinks_plist(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        tmp.write_text("dummy plist content")

        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = _mock_run_ok()

        result = remove_autowake()

        self.assertIn("removed", result.lower())
        self.assertFalse(tmp.exists())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_remove_no_plist_is_safe(self, mock_path, mock_sub, mock_plat):
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = Path("/tmp/does_not_exist_abc.plist")
        mock_sub.run.return_value = _mock_run_ok()

        result = remove_autowake()
        self.assertIn("removed", result.lower())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_status_not_installed(self, mock_path, mock_sub, mock_plat):
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = Path("/tmp/does_not_exist.plist")

        result = autowake_status()
        self.assertIn("Not installed", result)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_status_installed_and_loaded(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        tmp.write_text("dummy")

        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = MagicMock(returncode=0, stdout="")

        result = autowake_status()
        self.assertIn("Installed and loaded", result)
        tmp.unlink(missing_ok=True)

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_status_plist_exists_not_loaded(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        tmp.write_text("dummy")

        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = MagicMock(returncode=1, stdout="")

        result = autowake_status()
        self.assertIn("not loaded", result.lower())
        tmp.unlink(missing_ok=True)


# =============================================================================
# Platform dispatch
# =============================================================================

class TestPlatformDispatch(unittest.TestCase):
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    def test_unsupported_platform_install(self, mock_plat):
        mock_plat.system.return_value = "FreeBSD"
        self.assertIn("Unsupported", install_autowake(_jobs_file()))

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    def test_unsupported_platform_remove(self, mock_plat):
        mock_plat.system.return_value = "Haiku"
        self.assertIn("Unsupported", remove_autowake())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    def test_unsupported_platform_status(self, mock_plat):
        mock_plat.system.return_value = "Plan9"
        self.assertIn("Unsupported", autowake_status())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_linux_routes_to_crontab(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_sub.run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0),
        ]
        result = install_autowake(_jobs_file())
        self.assertIn("crontab", result.lower())

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    def test_windows_routes_to_schtasks(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = _mock_run_ok()
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        result = install_autowake(_jobs_file())
        # schtasks result talks about "Windows schtasks" or similar
        self.assertIn("installed", result.lower())
        # Verify schtasks was actually called
        all_args = [str(c[0][0]) for c in mock_sub.run.call_args_list]
        self.assertTrue(any("schtasks" in a for a in all_args))

    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.platform")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler.subprocess")
    @patch("toolboxv2.mods.isaa.extras.jobs.os_scheduler._get_plist_path")
    def test_darwin_routes_to_launchagent(self, mock_path, mock_sub, mock_plat):
        tmp = Path(tempfile.mktemp(suffix=".plist"))
        mock_plat.system.return_value = "Darwin"
        mock_path.return_value = tmp
        mock_sub.run.return_value = _mock_run_ok()
        result = install_autowake(_jobs_file())
        self.assertIn("LaunchAgent", result)
        tmp.unlink(missing_ok=True)


# =============================================================================
# Real file-system integration — runs only on the current platform
# =============================================================================

@unittest.skipIf(
    sys.platform not in ("linux", "darwin"),
    "Integration tests only on Linux / macOS (no schtasks available in CI)"
)
class TestRealFilesystemIntegration(unittest.TestCase):
    """
    Actual crontab / LaunchAgent manipulation on the running system.
    Safe: uses a temp jobs.json; cleans up via remove_autowake() in tearDown.
    """

    def setUp(self):
        self.jf = _jobs_file()

    def tearDown(self):
        # Always clean up, ignore errors
        try:
            remove_autowake()
        except Exception:
            pass
        self.jf.unlink(missing_ok=True)

    def test_install_and_status_cycle(self):
        install_result = install_autowake(self.jf)
        self.assertIn("installed", install_result.lower(),
                      f"Install failed: {install_result}")

        status_result = autowake_status()
        self.assertIn("installed", status_result.lower(),
                      f"Status wrong after install: {status_result}")

    def test_remove_after_install(self):
        install_autowake(self.jf)
        remove_result = remove_autowake()
        self.assertIn("removed", remove_result.lower(),
                      f"Remove failed: {remove_result}")

        status_result = autowake_status()
        self.assertIn("not installed", status_result.lower(),
                      f"Status wrong after remove: {status_result}")

    def test_idempotent_install(self):
        """Installing twice must not duplicate entries."""
        install_autowake(self.jf)
        install_autowake(self.jf)

        if sys.platform == "linux":
            import subprocess as sp
            result = sp.run(["crontab", "-l"], capture_output=True, text=True)
            entries = [l for l in result.stdout.split("\n")
                       if _TASK_NAME in l and not l.startswith("#")]
            # Expect exactly 2 entries: periodic + @reboot
            self.assertEqual(len(entries), 2,
                             f"Expected 2 ISAA entries, got {len(entries)}:\n{result.stdout}")

    def test_remove_is_idempotent(self):
        """Removing when not installed should not raise."""
        remove_autowake()  # nothing installed
        remove_autowake()  # again — must not raise

    @unittest.skipIf(sys.platform != "linux", "Linux-only crontab check")
    def test_crontab_entry_has_correct_interval(self):
        import subprocess as sp
        install_autowake(self.jf)
        result = sp.run(["crontab", "-l"], capture_output=True, text=True)
        interval_line = next(
            (l for l in result.stdout.split("\n") if _TASK_NAME in l and "reboot" not in l),
            None
        )
        self.assertIsNotNone(interval_line, "No interval crontab entry found")
        self.assertIn(f"*/{_CHECK_INTERVAL_MINUTES}", interval_line)

    @unittest.skipIf(sys.platform != "linux", "Linux-only crontab check")
    def test_crontab_entry_has_jobs_file(self):
        import subprocess as sp
        install_autowake(self.jf)
        result = sp.run(["crontab", "-l"], capture_output=True, text=True)
        self.assertIn(str(self.jf), result.stdout)


# =============================================================================
# Constants sanity
# =============================================================================

class TestConstants(unittest.TestCase):
    def test_task_name_nonempty(self):
        self.assertTrue(_TASK_NAME)

    def test_check_interval_positive(self):
        self.assertGreater(_CHECK_INTERVAL_MINUTES, 0)

    def test_check_interval_reasonable(self):
        # Should be between 1 min and 60 min
        self.assertGreaterEqual(_CHECK_INTERVAL_MINUTES, 1)
        self.assertLessEqual(_CHECK_INTERVAL_MINUTES, 60)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
