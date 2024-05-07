import json
import logging
import multiprocessing
import os
import time
from platform import system

import requests

from toolboxv2 import MainTool, FileHandler, get_app
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.extras.qr import print_qrcode_to_console
from toolboxv2.utils.system.session import Session

export = get_app("api_manager.Export").tb
Name = "FastApi"


class Tools(MainTool, FileHandler):  # FileHandler

    def __init__(self, app=None):
        self.running_apis = {}
        self.version = "0.2.0"
        self.name = "FastApi"
        self.logger: logging.Logger = app.logger if app else None
        self.color = "WHITE"
        self.keys = {"Apis": "api~config"}
        self.api_pid = None
        self.api_config = {}
        self.tools = {
            "all": [
                ["Version", "Shows current Version"],
                ["edit-api", "Set default API for name host port "],
                ["start-api", ""],
                ["stop-api", ""],
                ["restart-api", ""],
                ["info", ""],
            ],
            "name":
                "api_manager",
            "Version":
                self.show_version,
            "edit-api":
                self.conf_api,
            "stop-api":
                self.stop_api,
            "start":
                self.start_live,
            "startE":
                self._start_api,
            "startDev":
                self.start_dev,
            "startDUG":
                self.start_debug,
            "info":
                self.info,
            "restart-api":
                self.restart_api,
        }
        FileHandler.__init__(
            self, "api-m.data", app.id if app else __name__, self.keys, {
                "Apis": {
                    'main': {
                        "Name": 'main',
                        "version": self.version,
                        "port": 5000,
                        "host": '127.0.0.1'
                    }
                }
            })
        MainTool.__init__(self,
                          load=self.on_start,
                          v=self.version,
                          tool=self.tools,
                          name=self.name,
                          logs=self.logger,
                          color=self.color,
                          on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def info(self):
        for api in list(self.api_config.keys()):
            self.print(f"Name: {api}")
            self.print(self.api_config[api])
        return self.api_config

    def conf_api(self, api_name: str, host: str = "localhost", port: int = 5000):
        """Update api configuration
            Args:
                *name* - api_name of the api configuration same api_name to use for (start, stop)
                *host* - host of the api default = "localhost"
                *port* - port of the api default = "5000"
        """
        if host == "lh":
            host = "127.0.0.1"
        if host == "0":
            host = "0.0.0.0"
        if port == "0":
            port = "8000"
        self.api_config[api_name] = {
            "Name": api_name,
            "version": self.version,
            "port": port,
            "host": host
        }

        self.print(self.api_config[api_name])

    def start_dev(self, api_name: str, *modules, **kwargs):
        if modules:
            api_name += '_D'
            with BlobFile(f"FastApiManager/{api_name}.specs", 'w') as f:
                f.write_json({
                    'modules': modules,
                })
            print("Started FastApiManagerDev", api_name)
        return self._start_api(api_name, live=False, reload=True, test_override=False, host="0.0.0.0")

    def start_live(self, api_name):
        return self._start_api(api_name, live=True, reload=False, test_override=False)

    def start_debug(self, api_name):
        return self._start_api(api_name, live=False, reload=True, test_override=True, host="127.0.0.1")

    def _start_api(self, api_name: str, live=False, reload=False, test_override=False, host="localhost"):
        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"

        if isinstance(live, str):
            live = bool(live)
        if isinstance(reload, str):
            reload = bool(reload)

        self.print(f"{api_name=}: str, {live=}=False, {reload=}=False")
        api_thread = self.running_apis.get(api_name)

        if api_thread is not None:
            return "api is already running"

        if live is True and reload is True:
            raise ValueError("Live and reload should not be used together")

        if api_name not in self.api_config.keys():

            self.api_config[api_name] = {
                "Name": api_name,
                "version": self.version,
                "port": self.app.args_sto.port,
                "host": host
            }

            if live:
                self.api_config[api_name]['host'] = "0.0.0.0"

            self.print(f"Auto addet {api_name} to config : {self.api_config[api_name]}")

        if live:
            self.api_config[api_name]['host'] = "0.0.0.0"

        api_data = self.api_config[api_name]

        self.print(api_data)
        g = f"uvicorn toolboxv2.mods.FastApi.fast_api_main:app --host {api_data['host']}" \
            f" --port {api_data['port']} --header data:{self.app.debug}:{api_name}"
        # if test_override:
        #     g += ' --reload --reload-dir ./utils' if reload else ''

        print("Running command : " + g)

        # def runner():
        #     uvicorn.run("toolboxv2.mods.FastApi.fast_api_main:app", host=api_data['host'],
        #                 port=api_data['port'], headers=[('data', str(self.app.debug)+':'+ api_name)], reload=reload)

        if api_thread is None:
            self.running_apis[api_name] = True
            _ = "http" + '' if (api_data.get('host', '0').startswith('0') or api_data.get('host', '0').startswith('1') or
                                api_data.get('host', '0').startswith('localhost')) else 's'
            _ += '://'
            print_qrcode_to_console(_ + api_data['host'] + ':' + str(api_data['port']))
            self.running_apis[api_name] = multiprocessing.Process(target=os.system, args=(g,), daemon=True)
            self.running_apis[api_name].start()
            return "starting api at " + api_data['host'] + ':' + str(api_data['port'])

        self.print("API is already running")

    async def stop_api(self, api_name: str, delete=True):
        if api_name not in list(self.api_config.keys()):
            return f"Api with the name {api_name} is not listed"

        api_thread = self.running_apis.get(api_name)
        api_data = self.api_config[api_name]
        host = api_data.get("host")
        port = api_data.get("port")
        if api_thread is None:
            self.print("API is not running")

        if not os.path.exists(f"./.data/api_pid_{api_name}"):
            self.logger.warning("no api_pid file found ")
            return "No such api_pid file found on the filesystem"
        with open(f"./.data/api_pid_{api_name}", "r") as f:
            api_pid = f.read()
            try:
                requests.get(f"http://{host}:{port}/api/exit/{api_pid}")
                session = Session(self.app.get_username())
                if not await session.login():
                    print(f"Couldn't login with username : {self.app.get_username()}'")
                response = await session.fetch(f"http://{host}:{port}/api/exit/{api_pid}", method="POST")
                print(response)
            except Exception as e:
                self.print("API Not Responding")
            if system() == "Windows":
                os.system(f"taskkill /pid {api_pid} /F")
            else:
                os.system(f"kill -9 {api_pid}")
        api_thread = self.running_apis.get(api_name)
        # if api_thread:
        #     api_thread.join()
        if delete:
            del self.running_apis[api_name]
        os.remove(f"./.data/api_pid_{api_name}")

    @export(mod_name="api_manager", test=False)
    def show_running(self):
        self.print(f"Status : {list(self.running_apis.keys())}")
        return list(self.running_apis.keys())

    @export(mod_name="api_manager", test=False)
    async def restart_api(self, api_name: str):
        await self.stop_api(api_name)
        time.sleep(4)
        self._start_api(api_name)

    def on_start(self):
        self.load_file_handler()
        data = self.get_file_handler(self.keys["Apis"])
        if isinstance(data, str):
            self.api_config = json.loads(data)
        else:
            self.api_config = data

    async def on_exit(self):
        self.add_to_save_file_handler(self.keys["Apis"], json.dumps(self.api_config))
        for key in self.running_apis:
            await self.stop_api(key, delete=False)
        self.running_apis = {}
        self.save_file_handler()

