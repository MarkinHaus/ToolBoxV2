import os
import sys
import subprocess


def run_executable_in_background(executable_path, args=None, show_output=False):
    args = args or []

    # Wenn show_output True ist, erbt der Prozess das Terminal (None).
    # Anderenfalls wird die Ausgabe komplett verschluckt (DEVNULL).
    stdout_behavior = None if show_output else subprocess.DEVNULL
    stderr_behavior = None if show_output else subprocess.DEVNULL
    stdin_behavior = None if show_output else subprocess.DEVNULL

    cmd = executable_path if isinstance(executable_path, list) else [executable_path]
    if args:
        cmd += args if isinstance(args, list) else [args]
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        return subprocess.Popen(
            cmd,
            creationflags=CREATE_NO_WINDOW,
            stdout=stdout_behavior,
            stderr=stderr_behavior,
            stdin=stdin_behavior
        )
    else:
        return subprocess.Popen(
            cmd,
            stdout=stdout_behavior,
            stderr=stderr_behavior,
            stdin=stdin_behavior,
            preexec_fn=os.setsid  # Fully detached on Linux/macOS
        )
