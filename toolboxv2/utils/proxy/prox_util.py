import time
from enum import Enum
from typing import Any, Optional

from ..extras.Style import Spinner
from ..system.types import ApiResult, AppType
from ..toolbox import App
from ..system.all_functions_enums import SOCKETMANAGER


class ProxyUtil:
    def __init__(self, class_instance: Any, host='0.0.0.0', port=6587, timeout=15, app: Optional[App or AppType] = None,
                 remote_functions=None, peer=False, name='daemonApp-client', do_connect=True):
        self.class_instance = class_instance
        self.client = None
        self.port = port
        self.host = host
        self.timeout = timeout
        self.app = app
        self._name = name
        if remote_functions is None:
            remote_functions = ["run_any", "remove_mod", "save_load", "exit_main", "show_console", "hide_console",
                                "rrun_runnable",
                                "get_autocompletion_dict",
                                "exit_main"]
        self.remote_functions = remote_functions

        from toolboxv2.mods.SocketManager import SocketType
        self.connection_type = SocketType.client
        if peer:
            self.connection_type = SocketType.peer
        if do_connect:
            self.connect()

    def connect(self):
        client_result = self.app.run_local(SOCKETMANAGER.CREATE_SOCKET,
                                           get_results=True,
                                           name=self._name,
                                           host=self.host,
                                           port=self.port,
                                           type_id=self.connection_type,
                                           max_connections=-1,
                                           return_full_object=True)

        if client_result.is_error():
            raise Exception(f"Client {self._name} error: {client_result.print(False)}")
        if not client_result.is_data():
            raise Exception(f"Client {self._name} error: {client_result.print(False)}")
        if client_result.get('connection_error') != 0:
            raise Exception(f"Client {self._name} error: {client_result.print(False)}")
        # 'socket': socket,
        # 'receiver_socket': r_socket,
        # 'host': host,
        # 'port': port,
        # 'p2p-port': endpoint_port,
        # 'sender': send,
        # 'receiver_queue': receiver_queue,
        # 'connection_error': connection_error,
        # 'receiver_thread': s_thread,
        # 'keepalive_thread': keep_alive_thread,
        # 'running_dict': running_dict,
        # 'client_to_receiver_thread': to_receive,
        # 'client_receiver_threads': threeds,
        self.client = client_result

    def disconnect(self):
        time.sleep(1)
        running_dict = self.client.get("running_dict")
        sender = self.client.get("sender")
        sender({'exit': True})
        running_dict["server_receiver"] = False
        running_dict["receive"]['main'] = False
        running_dict["keep_alive_var"] = False
        self.client = None

    def reconnect(self):
        if self.client is not None:
            self.disconnect()
        self.connect()

    def verify(self):
        time.sleep(1)
        # self.client.get('sender')({'keepalive': 0})
        self.client.get('sender')(b"verify")

    def __getattr__(self, name):

        if self.client is None:
            self.reconnect()
        # print(f"ProxyApp: {name}, {self.client is None}")
        if name == "on_exit":
            self.disconnect()
        if name == "rc":
            self.reconnect()
            return
        if name == "r":
            try:
                return self.client.get('receiver_queue').get(timeout=self.timeout)
            except:
                return "No data"

        app_attr = getattr(self.class_instance, name)

        def method(*args, **kwargs):
            # if name == 'run_any':
            #     print("method", name, kwargs.get('get_results', False), args[0])
            if kwargs.get('spec', '-') == 'app':
                return app_attr(*args, **kwargs)
            try:
                if name in self.remote_functions:
                    if name == 'run_any' and not kwargs.get('get_results', False):
                        return app_attr(*args, **kwargs)
                    if name == 'run_any' and kwargs.get('get_results', False):
                        if isinstance(args[0], Enum):
                            args = (args[0].__class__.NAME.value, args[0].value), args[1:]
                    self.app.sprint(f"Calling method {name}, {args=}, {kwargs}=")
                    self.client.get('sender')({'name': name, 'args': args, 'kwargs': kwargs})
                    while Spinner("Waiting for result"):
                        try:
                            data = self.client.get('receiver_queue').get(timeout=self.timeout)
                            if isinstance(data, dict) and 'identifier' in data:
                                del data["identifier"]
                            if 'error' in data and 'origin' in data and 'result' in data and 'info' in data:
                                data = ApiResult(**data).as_result()
                            return data
                        except:
                            print("No data look later with app.r")
                            return "No data"
            except:
                if self.client.get('socket') is None:
                    self.client = None
            return app_attr(*args, **kwargs)

        if callable(app_attr) and name in self.remote_functions and self.client is not None:
            return method
        return app_attr
