"""S6: _start_local_ui-Vertrag — cwd, Detach-Flags, Log-Redirect."""
import os
import sys
from unittest import mock

import toolboxv2.utils.extras.fallback_tray as ft

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__import__("toolboxv2").__file__)))


def test_start_local_ui_cwd_and_logfile():
    """Popen bekommt Install-Root als cwd + Logfile statt geerbter Handles."""
    with mock.patch.object(ft.subprocess, "Popen") as p, \
         mock.patch.object(ft, "_workers_start_cmd", return_value=["tb", "workers", "start"]):
        ft._start_local_ui()
    assert p.call_count == 1
    kw = p.call_args.kwargs
    assert os.path.normcase(kw["cwd"]) == os.path.normcase(PKG_ROOT)
    assert kw["stdout"] is not None and kw["stderr"] is not None
    assert kw["stdin"] == ft.subprocess.DEVNULL
    if sys.platform == "win32":
        assert kw["creationflags"] & ft.subprocess.DETACHED_PROCESS
    else:
        assert kw.get("start_new_session") is True


def test_workers_start_cmd_prefers_tb_exe():
    """shutil.which('tb')-Hit -> [tb, workers, start] statt Modul-Spawn."""
    with mock.patch("shutil.which", return_value=r"C:\fake\bin\tb.exe"):
        assert ft._workers_start_cmd() == [r"C:\fake\bin\tb.exe", "workers", "start"]


def test_workers_start_cmd_fallback_module_spawn():
    """Kein tb auf PATH -> sys.executable -m toolboxv2 (Dev-Setup)."""
    with mock.patch("shutil.which", return_value=None):
        cmd = ft._workers_start_cmd()
    assert cmd == [sys.executable, "-m", "toolboxv2", "workers", "start"]


def test_popen_failure_is_logged_not_raised():
    """Startfehler darf den Tray-Thread nicht killen."""
    with mock.patch.object(ft.subprocess, "Popen",
                           side_effect=OSError("boom")), \
         mock.patch.object(ft, "_workers_start_cmd", return_value=["tb"]):
        ft._start_local_ui()  # darf nicht raisen
