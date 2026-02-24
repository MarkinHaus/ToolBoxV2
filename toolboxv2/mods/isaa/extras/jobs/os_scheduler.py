"""
OS Scheduler - Platform-agnostic auto-wake registration
========================================================

Registers/removes OS-level scheduled tasks so the CLI can auto-wake
when jobs are due even if it's not running.

Supports:
- Windows: schtasks
- Linux: crontab
- macOS: LaunchAgent plist

Author: ISAA Team
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

_TASK_NAME = "ISAA_JobRunner"
_CHECK_INTERVAL_MINUTES = 15


def _get_python_executable() -> str:
    """Get the current Python executable path."""
    return sys.executable


def _get_runner_command(jobs_file: Path) -> str:
    """Build the headless runner command."""
    runner_module = "toolboxv2.mods.isaa.extras.jobs.headless_runner"
    return f'"{_get_python_executable()}" -m {runner_module} --jobs-file "{jobs_file}"'


# =============================================================================
# WINDOWS
# =============================================================================

def _install_windows(jobs_file: Path) -> str:
    """Install Windows scheduled tasks via schtasks."""
    python = _get_python_executable()
    runner = "toolboxv2.mods.isaa.extras.jobs.headless_runner"
    args = f'--jobs-file "{jobs_file}"'

    # Periodic check every 15 minutes
    try:
        subprocess.run([
            "schtasks", "/Create", "/F",
            "/TN", f"{_TASK_NAME}_Periodic",
            "/SC", "MINUTE", "/MO", str(_CHECK_INTERVAL_MINUTES),
            "/TR", f'"{python}" -m {runner} {args}',
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return f"Failed to create periodic task: {e.stderr}"

    # On system startup
    try:
        subprocess.run([
            "schtasks", "/Create", "/F",
            "/TN", f"{_TASK_NAME}_Boot",
            "/SC", "ONSTART",
            "/TR", f'"{python}" -m {runner} {args}',
            "/DELAY", "0001:00",  # 1 minute delay after boot
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return f"Periodic task installed, but boot task failed: {e.stderr}"

    return f"Auto-wake installed (Windows schtasks, every {_CHECK_INTERVAL_MINUTES}min + on boot)"


def _remove_windows() -> str:
    """Remove Windows scheduled tasks."""
    errors = []
    for suffix in ("_Periodic", "_Boot"):
        try:
            subprocess.run([
                "schtasks", "/Delete", "/F", "/TN", f"{_TASK_NAME}{suffix}"
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            errors.append(suffix)
    if errors:
        return f"Partially removed (failed: {', '.join(errors)})"
    return "Auto-wake removed (Windows schtasks)"


def _status_windows() -> str:
    """Check Windows scheduled task status."""
    results = []
    for suffix in ("_Periodic", "_Boot"):
        try:
            r = subprocess.run([
                "schtasks", "/Query", "/TN", f"{_TASK_NAME}{suffix}", "/FO", "LIST"
            ], capture_output=True, text=True)
            if r.returncode == 0:
                # Extract status line
                for line in r.stdout.split("\n"):
                    if "Status" in line:
                        results.append(f"{suffix}: {line.strip()}")
                        break
                else:
                    results.append(f"{suffix}: Registered")
            else:
                results.append(f"{suffix}: Not found")
        except Exception:
            results.append(f"{suffix}: Check failed")
    return "Auto-wake status:\n  " + "\n  ".join(results)


# =============================================================================
# LINUX
# =============================================================================

def _install_linux(jobs_file: Path) -> str:
    """Install Linux crontab entries."""
    cmd = _get_runner_command(jobs_file)

    try:
        # Read existing crontab
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        existing = result.stdout if result.returncode == 0 else ""

        # Remove old entries
        lines = [l for l in existing.split("\n") if _TASK_NAME not in l and l.strip()]

        # Add new entries
        lines.append(f"# {_TASK_NAME} - ISAA Job Runner (auto-generated)")
        lines.append(f"*/{_CHECK_INTERVAL_MINUTES} * * * * {cmd}")
        lines.append(f"@reboot {cmd}")
        lines.append("")  # trailing newline

        # Write crontab
        new_crontab = "\n".join(lines)
        proc = subprocess.run(
            ["crontab", "-"], input=new_crontab, capture_output=True, text=True
        )
        if proc.returncode != 0:
            return f"Failed to install crontab: {proc.stderr}"

        return f"Auto-wake installed (crontab, every {_CHECK_INTERVAL_MINUTES}min + @reboot)"
    except Exception as e:
        return f"Failed to install crontab: {e}"


def _remove_linux() -> str:
    """Remove Linux crontab entries."""
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if result.returncode != 0:
            return "No crontab to clean"

        lines = [l for l in result.stdout.split("\n") if _TASK_NAME not in l]
        new_crontab = "\n".join(lines)

        subprocess.run(["crontab", "-"], input=new_crontab, capture_output=True, text=True)
        return "Auto-wake removed (crontab)"
    except Exception as e:
        return f"Failed to remove crontab entries: {e}"


def _status_linux() -> str:
    """Check Linux crontab status."""
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if result.returncode != 0:
            return "Auto-wake: Not installed (no crontab)"

        entries = [l for l in result.stdout.split("\n") if _TASK_NAME in l and not l.startswith("#")]
        if entries:
            return f"Auto-wake: Installed ({len(entries)} crontab entries)"
        return "Auto-wake: Not installed"
    except Exception as e:
        return f"Auto-wake: Check failed ({e})"


# =============================================================================
# macOS
# =============================================================================

_PLIST_LABEL = "com.isaa.jobrunner"


def _get_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_LABEL}.plist"


def _install_macos(jobs_file: Path) -> str:
    """Install macOS LaunchAgent."""
    python = _get_python_executable()
    runner = "toolboxv2.mods.isaa.extras.jobs.headless_runner"
    plist_path = _get_plist_path()

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>{runner}</string>
        <string>--jobs-file</string>
        <string>{jobs_file}</string>
    </array>
    <key>StartInterval</key>
    <integer>{_CHECK_INTERVAL_MINUTES * 60}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/isaa_jobrunner.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/isaa_jobrunner.err</string>
</dict>
</plist>
"""

    try:
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(plist_content)

        # Unload if exists, then load
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
        if result.returncode != 0:
            return f"Plist written but launchctl load failed: {result.stderr}"

        return f"Auto-wake installed (LaunchAgent, every {_CHECK_INTERVAL_MINUTES}min + RunAtLoad)"
    except Exception as e:
        return f"Failed to install LaunchAgent: {e}"


def _remove_macos() -> str:
    """Remove macOS LaunchAgent."""
    plist_path = _get_plist_path()
    try:
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
        return "Auto-wake removed (LaunchAgent)"
    except Exception as e:
        return f"Failed to remove LaunchAgent: {e}"


def _status_macos() -> str:
    """Check macOS LaunchAgent status."""
    plist_path = _get_plist_path()
    if not plist_path.exists():
        return "Auto-wake: Not installed"

    try:
        result = subprocess.run(
            ["launchctl", "list", _PLIST_LABEL],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Auto-wake: Installed and loaded ({plist_path})"
        return f"Auto-wake: Plist exists but not loaded ({plist_path})"
    except Exception:
        return f"Auto-wake: Plist exists at {plist_path}"


# =============================================================================
# PUBLIC API (platform dispatch)
# =============================================================================

def install_autowake(jobs_file: Path) -> str:
    """Install OS-level auto-wake. Platform auto-detected."""
    system = platform.system()
    if system == "Windows":
        return _install_windows(jobs_file)
    elif system == "Linux":
        return _install_linux(jobs_file)
    elif system == "Darwin":
        return _install_macos(jobs_file)
    return f"Unsupported platform: {system}"


def remove_autowake() -> str:
    """Remove OS-level auto-wake."""
    system = platform.system()
    if system == "Windows":
        return _remove_windows()
    elif system == "Linux":
        return _remove_linux()
    elif system == "Darwin":
        return _remove_macos()
    return f"Unsupported platform: {system}"


def autowake_status() -> str:
    """Check auto-wake status."""
    system = platform.system()
    if system == "Windows":
        return _status_windows()
    elif system == "Linux":
        return _status_linux()
    elif system == "Darwin":
        return _status_macos()
    return f"Unsupported platform: {system}"
