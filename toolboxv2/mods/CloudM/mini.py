import os
import re
import subprocess
import time

# Define the directory where PID files are stored
PID_DIRECTORY = "./.info"

GREEN_CIRCLE = "🟢"
YELLOW_CIRCLE = "🟡"
RED_CIRCLE = "🔴"


def get_service_pids(info_dir):
    """Extracts service names and PIDs from pid files."""
    services = {}
    pid_files = [f for f in os.listdir(info_dir) if re.match(r'(.+)-(.+)\.pid', f)]
    for pid_file in pid_files:
        match = re.match(r'(.+)-(.+)\.pid', pid_file)
        if match:
            services_type, service_name = match.groups()
            # Read the PID from the file
            with open(os.path.join(info_dir, pid_file)) as file:
                pid = file.read().strip()
                # Store the PID using a formatted key
                services[f"{service_name} - {services_type}"] = int(pid)
    return services


def check_multiple_processes(pids: list[int]) -> dict[int, str]:
    """
    Checks the status of multiple processes in a single system call.
    Returns a dictionary mapping PIDs to their status (GREEN_CIRCLE, RED_CIRCLE, or YELLOW_CIRCLE).
    """
    if not pids:
        return {}

    pid_status = {}

    if os.name == 'nt':  # Windows
        try:
            # Windows tasklist requires separate /FI for each filter
            command = 'tasklist'

            # Add encoding handling for Windows
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                encoding='cp850'  # Use cp850 for Windows console output
            )
            # Create a set of running PIDs from the output
            running_pids = set()
            for line in result.stdout.lower().split('\n'):
                for pid in pids:
                    if str(pid) in line:
                        running_pids.add(pid)
            # Assign status based on whether PID was found in output
            for pid in pids:
                if pid in running_pids:
                    pid_status[pid] = GREEN_CIRCLE
                else:
                    pid_status[pid] = RED_CIRCLE

        except subprocess.SubprocessError as e:
            print(f"SubprocessError: {e}")  # For debugging
            # Mark all as YELLOW_CIRCLE if there's an error running the command
            for pid in pids:
                pid_status[pid] = YELLOW_CIRCLE
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")  # For debugging
            # Try alternate encoding if cp850 fails
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    shell=True,
                    encoding='utf-8'
                )
                running_pids = set()
                for line in result.stdout.lower().split('\n'):
                    for pid in pids:
                        if str(pid) in line:
                            running_pids.add(pid)

                for pid in pids:
                    pid_status[pid] = GREEN_CIRCLE if pid in running_pids else RED_CIRCLE
            except Exception as e:
                print(f"Failed with alternate encoding: {e}")  # For debugging
                for pid in pids:
                    pid_status[pid] = YELLOW_CIRCLE

    else:  # Unix/Linux/Mac
        try:
            pids_str = ','.join(str(pid) for pid in pids)
            command = f'ps -p {pids_str} -o pid='

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                encoding='utf-8'
            )
            running_pids = set(int(pid) for pid in result.stdout.strip().split())

            for pid in pids:
                pid_status[pid] = GREEN_CIRCLE if pid in running_pids else RED_CIRCLE

        except subprocess.SubprocessError as e:
            print(f"SubprocessError: {e}")  # For debugging
            for pid in pids:
                pid_status[pid] = YELLOW_CIRCLE

    return pid_status


services_data_sto = [{}]
services_data_sto_last_update_time = [0]
services_data_display = [""]

def get_service_status(dir: str) -> str:
    """Displays the status of all services."""
    if time.time()-services_data_sto_last_update_time[0] > 30:
        services = get_service_pids(dir)
        services_data_sto[0] = services
        services_data_sto_last_update_time[0] = time.time()
    else:
        services = services_data_sto[0]
    if not services:
        return "No services found"

    # Get status for all PIDs in a single call
    pid_statuses = check_multiple_processes(list(services.values()))

    # Build the status string
    res_s = "Service(s):" + ("\n" if len(services) > 1 else ' ')
    for service_name, pid in services.items():
        status = pid_statuses.get(pid, YELLOW_CIRCLE)
        res_s += f"{status} {service_name} (PID: {pid})\n"
    services_data_display[0] = res_s.strip()
    return res_s.rstrip()
