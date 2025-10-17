import os
import subprocess
import sys


def run_executable_in_background(executable_path, args=None):
    args = args or []
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(
            [executable_path] + args,
            creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
    else:
        subprocess.Popen(
            [executable_path] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # Fully detached on Linux/macOS
        )
