import json
import logging
import multiprocessing
import os
import signal
import time
from datetime import datetime
from platform import system
from typing import Any

from toolboxv2 import FileHandler, MainTool
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.extras.qr import print_qrcode_to_console
from toolboxv2.utils.system.session import get_local_ip, get_public_ip

# Global metadata
NAME = "FastApi"
VERSION = "0.2.2"


class Tools(MainTool, FileHandler):
    """
    A production-ready API Manager for running, monitoring, and managing FastAPI instances.

    This class allows you to:
      - Start API instances (live, development, debug)
      - Stop and restart running APIs
      - Update configuration for APIs
      - Get live diagnostic info about running APIs
    """

    def __init__(self, app: Any | None = None) -> None:
        # Running APIs will be stored as a mapping from api_name to subprocess.Popen
        self.running_apis: dict[str, multiprocessing.Process] = {}
        self.api_config: dict[str, dict[str, str | int]] = {}
        self.version: str = VERSION
        self.name: str = NAME
        self.logger: logging.Logger = app.logger if app else logging.getLogger(__name__)
        self.color: str = "WHITE"
        self.keys: dict[str, str] = {"Apis": "api~config"}
        # In case app is not passed in, ensure that we have a dummy object with required properties

        # Define available tool commands
        self.tools: dict[str, Any] = {
            "all": [
                ["Version", "Shows current Version"],
                ["edit-api", "Set default API for name, host and port"],
                ["start-api", "Start an API instance"],
                ["stop-api", "Stop a running API instance"],
                ["restart-api", "Restart an API instance"],
                ["info", "Show API configurations and running APIs"],
            ],
            "name": "api_manager",
            "Version": self.show_version,
            "edit-api": self.conf_api,
            "stop-api": self.stop_api,
            "start": self.start_live,
            "startE": self._start_api,
            "startDev": self.start_dev,
            "startDUG": self.start_debug,
            "info": self.show_running,
            "restart-api": self.restart_api,
        }

        # Initialize FileHandler with default configuration data
        default_config = {
            "Apis": {
                'main': {
                    "Name": 'main',
                    "version": self.version,
                    "port": 5000,
                    "host": '127.0.0.1'
                }
            }
        }
        FileHandler.__init__(self, "apis.config", self.app.id, self.keys, default_config)
        MainTool.__init__(
            self,
            load=self.on_start,
            v=self.version,
            tool=self.tools,
            name=self.name,
            logs=self.logger,
            color=self.color,
            on_exit=self.on_exit,
        )
        os.makedirs("./.data", exist_ok=True)

    @staticmethod
    def _get_pid_file_path(api_name: str) -> str:
        """Get the path to the PID file for an API."""
        return os.path.join("./.data", f"api_pid_{api_name}")


    def show_version(self) -> str:
        """Display and return the current version."""
        self.logger.info("Version: %s", self.version)
        return self.version

    def info(self) -> dict[str, Any]:
        """
        Return diagnostic information about API configurations and currently running APIs.
        """
        config_info = {name: cfg for name, cfg in self.api_config.items()}
        running_info = {name: proc.pid for name, proc in self.running_apis.items() if proc.is_alive()}
        self.logger.info("API Configurations: %s", config_info)
        self.logger.info("Running APIs: %s", running_info)
        # Optionally, print to console as well
        for api_name, cfg in config_info.items():
            print(f"Configured API - Name: {api_name}, Config: {cfg}")
        print("Running APIs:")
        for api_name, pid in running_info.items():
            print(f"API: {api_name}, Process ID: {pid}")
        return {"configurations": config_info, "running": running_info}

    def conf_api(self, api_name: str, host: str = "localhost", port: int = 5000) -> None:
        """
        Update or create an API configuration.

        Args:
            api_name (str): The name of the API.
            host (str): The host address (default "localhost"). Use "lh" for "127.0.0.1" or "0" for "0.0.0.0".
            port (int): The port number (default 5000; use "0" for port 8000).
        """
        if host.lower() == "lh":
            host = "127.0.0.1"
        if host == "0":
            host = "0.0.0.0"
        if str(port) == "0":
            port = 8000

        self.api_config[api_name] = {
            "Name": api_name,
            "version": self.version,
            "port": int(port),
            "host": host,
        }
        self.logger.info("Updated API configuration for '%s': %s", api_name, self.api_config[api_name])
        print(f"API configuration updated: {self.api_config[api_name]}")

    def start_dev(self, api_name: str, *modules: str, **kwargs: Any) -> str | None:
        """
        Start an API in development mode.

        If additional modules are provided, they are stored in a BlobFile for later use.

        Args:
            api_name (str): The API name.
            *modules (str): Additional modules for the API.

        Returns:
            Optional[str]: Status message.
        """
        if modules:
            api_name_dev = f"{api_name}_D"
            with BlobFile(f"FastApi/{api_name_dev}/dev", mode='w') as f:
                f.write_json({'modules': modules})
            api_name = api_name_dev

        return self._start_api(api_name, live=False, reload=False, test_override=False, host="localhost")

    def start_live(self, api_name: str) -> str | None:
        """
        Start an API in live mode.
        """
        return self._start_api(api_name, live=True, reload=False, test_override=False)

    def start_debug(self, api_name: str) -> str | None:
        """
        Start an API in debug mode.
        """
        return self._start_api(api_name, live=False, reload=True, test_override=True, host="localhost")

    def _start_api(
        self,
        api_name: str,
        live: bool = False,
        reload: bool = False,
        test_override: bool = False,
        host: str = "localhost"
    ) -> str | None:
        """
        Start an API process with the given configuration.

        Args:
            api_name (str): The API name.
            live (bool): Whether to run in live mode.
            reload (bool): Whether to enable auto-reload.
            test_override (bool): If True, allow start even if running in a test environment.
            host (str): Host to bind the API on.

        Returns:
            Optional[str]: A status message or error message.
        """
        # Prevent starting an API if in test mode unless explicitly overridden.
        if 'test' in self.app.id and not test_override:
            msg = "No API allowed in test mode"
            self.logger.warning(msg)
            return msg

        if not api_name:
            self.logger.error("No API name provided.")
            return None

        # Check if API is already running.
        if api_name in self.running_apis and self.running_apis[api_name].is_alive():
            msg = f"API '{api_name}' is already running."
            self.logger.info(msg)
            return msg

        # Ensure that live and reload are not both enabled.
        if live and reload:
            raise ValueError("Live mode and reload mode cannot be enabled simultaneously.")

        # If configuration does not exist, add it automatically.
        if api_name not in self.api_config:
            self.api_config[api_name] = {
                "Name": api_name,
                "version": self.version,
                "port": self.app.args_sto.port,
                "host": host if host and isinstance(host, str) else "localhost",
            }
            if live:
                self.api_config[api_name]['host'] = "0.0.0.0"
            self.logger.info("Auto-added API configuration for '%s': %s", api_name, self.api_config[api_name])

        # For live mode, always bind to all interfaces.
        if live:
            self.api_config[api_name]['host'] = "0.0.0.0"

        api_data = self.api_config[api_name]

        # Check for required frontend dependencies.
        node_modules_path = os.path.join(self.app.start_dir, "web", "node_modules")
        if not os.path.exists(node_modules_path):
            self.logger.info("Node modules folder not found. Installing dependencies in '%s'", node_modules_path)
            os.system("npm install --prefix ./web ./web")

        # Build the uvicorn command.
        cmd_parts: list[str] = [
            # sys.executable,
            # "-m",
            "uvicorn",
            "toolboxv2.mods.FastApi.fast_api_main:app",
            f"--host {api_data['host']}",
            f"--port {api_data['port']}",
            f"--header data:{self.app.debug}:{api_name}"
        ]
        if reload:
            # Reload directories can be adjusted as needed.
            cmd_parts.append("--reload")
            cmd_parts.append("--reload-dir ./utils")
            cmd_parts.append("--reload-dir ./mods/FastApi")
        command: str = " ".join(cmd_parts)
        self.logger.info("Starting API '%s' with command: %s", api_name, command)

        print(command)

        # Print QR codes for local and public IPs for convenience.
        protocol = "http"  # Adjust if SSL is configured
        local_url = f"{protocol}://{get_local_ip()}:{api_data['port']}"
        public_url = f"{protocol}://{get_public_ip()}:{api_data['port']}"
        print_qrcode_to_console(local_url)
        print_qrcode_to_console(public_url)

        try:

            process = multiprocessing.Process(
                target=os.system,
                args=(command,),
                # daemon=True
            )
            process.start()

            # Store the process
            self.running_apis[api_name] = process

            # Save PID to file
            with open(self._get_pid_file_path(api_name), "w") as f:
                f.write(str(process.pid))

            # Store process info in file handler
            self.add_to_save_file_handler(
                key=f"pr{api_name}",
                value=json.dumps({
                    "pid": process.pid,
                    "start_time": datetime.now().isoformat(),
                    "host": api_data['host'],
                    "port": api_data['port']
                })
            )

            msg = f"Starting API '{api_name}' at {api_data['host']}:{api_data['port']} (PID: {process.pid})"
            self.logger.info(msg)
            return msg
        except Exception as e:
            self.logger.exception("Failed to start API '%s': %s", api_name, e)
            return f"Failed to start API '{api_name}': {e}"

    async def stop_api(self, api_name: str, delete: bool = True) -> str:
        """
        Stop a running API and clean up resources.
        """
        if api_name not in self.api_config:
            msg = f"API with the name '{api_name}' is not configured."
            self.logger.warning(msg)
            return msg

        pid_file = self._get_pid_file_path(api_name)
        if not os.path.exists(pid_file):
            self.logger.warning("No pid file found for API '%s'", api_name)
            return f"No pid file found for API '{api_name}'."

        try:
            # Read PID from file
            with open(pid_file) as f:
                api_pid = int(f.read().strip())

            # Try graceful shutdown first
            if 'core' in self.app.id:
                if not await self.app.session.login():
                    self.logger.warning("Could not login with username '%s'", self.app.get_username())
                try:
                    response = await self.app.session.fetch(f"/api/exit/{api_pid}", method="POST")
                    self.logger.info("Exit response for API '%s': %s", api_name, response)
                except Exception as e:
                    self.logger.warning("Failed to stop API gracefully: %s", e)

            # Force kill if process still exists
            process = self.running_apis.get(api_name)
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

            # Fallback to system commands if needed
            try:
                if system() == "Windows":
                    os.system(f"taskkill /pid {api_pid} /F")
                else:
                    os.kill(api_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already terminated

            # Cleanup
            if os.path.exists(pid_file):
                os.remove(pid_file)
            if delete and api_name in self.running_apis:
                del self.running_apis[api_name]

            # Update file handler
            self.add_to_save_file_handler(
                key=f"pr{api_name}",
                value=json.dumps({
                    "stop_time": datetime.now().isoformat(),
                    "status": "stopped"
                })
            )
            self.save_file_handler()

            msg = f"Stopped API '{api_name}'."
            self.logger.info(msg)
            return msg

        except Exception as e:
            self.logger.exception("Error stopping API '%s': %s", api_name, e)
            return f"Error stopping API '{api_name}': {e}"

    def nf(self, name):
        if len(name) > 10:
            return name[:10]
        elif len(name) < 10:
            return name + '~' * (len(name)-10)
        else:
            return name

    def show_running(self) -> list[str]:
        """
        Display and return the list of currently running APIs with their status.
        """
        self.on_start()
        running_list = []
        print(self.api_config)
        for api_name in self.api_config:

            # Get stored process info
            process_info = self.get_file_handler(f"pr{api_name}")
            print('#',api_name, '#',process_info)
            if process_info is None:
                process_info = {}
            status = {
                "name": api_name,
                "online": api_name in self.running_apis,
                "start_time": process_info.get("start_time", "offline"),
                "pid": process_info.get("pid", ''),
                "host": process_info.get("host", ''),
                "port": process_info.get("port", '')
            }
            running_list.append(status)

        # Log and print current status
        self.logger.info("APIs: %s", running_list)
        print("\nAPIs:")
        for api in running_list:
            print(f"- {api['name']}: at {api['host']}:{api['port']}")
            print(f"  Started: {api['start_time']}")

        return [api["name"] for api in running_list]

    async def restart_api(self, api_name: str) -> str:
        """
        Restart the given API by stopping it and starting it again.

        Args:
            api_name (str): The name of the API to restart.

        Returns:
            str: A status message.
        """
        stop_message = await self.stop_api(api_name)
        self.logger.info("Restart: %s", stop_message)
        # Allow some time for the process to fully terminate.
        time.sleep(4)
        start_message = self._start_api(api_name)
        return f"Restarting API '{api_name}': {start_message}"

    def on_start(self) -> None:
        """
        Load API configuration from file when the tool starts.
        """
        self.load_file_handler()
        data = self.get_file_handler(self.keys["Apis"])
        try:
            if isinstance(data, str):
                self.api_config = json.loads(data)
            else:
                self.api_config = data
            self.logger.info("Loaded API configuration: %s", self.api_config)
        except Exception as e:
            self.logger.exception("Error loading API configuration: %s", e)
            self.api_config = {}

    async def on_exit(self) -> None:
        """
        Gracefully stop all running APIs and save configuration upon exit.
        """
        # Save configuration data.
        if len(self.api_config) != 0:
            self.add_to_save_file_handler(self.keys["Apis"], json.dumps(self.api_config))
        # Attempt to stop all running APIs.
        # for api_name in list(self.running_apis.keys()):
        #     await self.stop_api(api_name, delete=False)
        self.running_apis = {}
        self.save_file_handler()
        self.logger.info("Exiting API Manager. All running APIs stopped and configuration saved.")
