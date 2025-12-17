# Build a toolbox version -> push to git
# run the live server
import os
import signal
import subprocess
import time

from ..utils.system.getting_and_closing_app import get_app

Name = 'ProcessManager'
export = get_app("cli_functions.Export").tb
version = '0.0.1'
default_export = export(mod_name=Name, version=version, test=False)


class ProcessManagerClass:
    def __init__(self, *commands):
        self.processes = []
        for command in commands:
            process = subprocess.Popen(command, shell=True)
            self.processes.append(process)

    def stop_processes(self):
        for process in self.processes:
            os.kill(process.pid, signal.SIGTERM)

    def monitor_processes(self):
        while True:
            for process in self.processes:
                status = process.poll()
                if status is not None:
                    self.stop_processes()
                    return
            time.sleep(1)


@default_export
def startDbug(p1_command="toolboxv2 -fg -c FastApi startDUG main-debug -m cli", p2_command="npm run dev"):
    manager = ProcessManagerClass(p1_command, p2_command)
    manager.monitor_processes()


@default_export
def startDev(p1_command="toolboxv2 -fg -c FastApi startDev main-debug -m cli", p2_command="npm run dev"):
    manager = ProcessManagerClass(p1_command, p2_command)
    manager.monitor_processes()


@default_export
def start(p1_command="toolboxv2 -fg -c FastApi start main -m cli", p2_command="npm run live"):
    manager = ProcessManagerClass(p1_command, p2_command)
    manager.monitor_processes()


@default_export
def start_client(p0_command="toolboxv2 -fg -c FastApi start main -m cli", p1_command="npm run previewClient", p2_command="npm run tauri"):
    manager = ProcessManagerClass(p0_command, p1_command, p2_command)
    manager.monitor_processes()


@default_export
def custom(*commands, monitor=True):
    manager = ProcessManagerClass(commands)
    if not monitor:
        return manager
    manager.monitor_processes()


if __name__ == "__main__":
    start_client()
