import os
import re
import subprocess


# Define the directory where PID files are stored
PID_DIRECTORY = "./pids"

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
            with open(os.path.join(info_dir, pid_file), 'r') as file:
                pid = file.read().strip()
                # Store the PID using a formatted key
                services[f"{service_name} - {services_type}"] = pid
    return services


def check_process_status(pid):
        """Checks the status of a process using the PID."""
    #try:
        # Cross-platform command to check if a process is running
        if os.name == 'nt':  # Windows
            command = f'tasklist /FI "PID eq {pid}"'
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            output = result.stdout
            if output is None:
                return YELLOW_CIRCLE
            elif pid in output:
                return GREEN_CIRCLE
            else:
                return RED_CIRCLE
        else:  # Unix/Linux/Mac
            command = f'ps -p {pid} -o comm='
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            output = result.stdout.strip()
            print(output)
            return GREEN_CIRCLE if output else RED_CIRCLE
    #except Exception as e:
    #    return f"error: {e}"


def get_service_status(dir):
    """Displays the status of all services."""
    services = get_service_pids(dir)
    res_s = "Service(s) :" + ("\n" if len(services.keys()) > 1 else '')
    for service_name, pid in services.items():
        status = check_process_status(pid)
        res_s += f" - {service_name}: PID {pid} - Status: {status}\n"
        if status == YELLOW_CIRCLE and os.path.exists(dir+'/'+'-'.join(service_name.replace(' ','').split('-')[::-1])+'.pid'):
            os.remove(dir+'/'+'-'.join(service_name.replace(' ','').split('-')[::-1])+'.pid')
    return res_s
