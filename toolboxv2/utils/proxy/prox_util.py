import asyncio
import time
from enum import Enum
from typing import Any, Optional

from ..extras.Style import Spinner
from ..system.all_functions_enums import SOCKETMANAGER
from ..system.types import ApiResult, AppType, Result
from ..toolbox import App


class ProxyUtil:
    def __init__(self, *args, **kwargs):
        """
        Standard constructor used for arguments pass
        Do not override. Use __ainit__ instead
        """
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __initobj(self):
        """Crutch used for __await__ after spawning"""
        # assert not self.async_initialized
        self.async_initialized = True
        # pass the parameters to __ainit__ that passed to __init__
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self):
        return self.__initobj().__await__()

    async def __ainit__(self, class_instance: Any, host='0.0.0.0', port=6587, timeout=15,
                        app: Optional[App or AppType] = None,
                        remote_functions=None, peer=False, name='daemonApp-client', do_connect=True, unix_socket=False):
        self.class_instance = class_instance
        self.client = None
        self.port = port
        self.host = host
        self.timeout = timeout
        self.app = app
        self._name = name
        self.unix_socket = unix_socket
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
            await self.connect()

    async def connect(self):
        client_result = self.app.run_local(SOCKETMANAGER.CREATE_SOCKET,
                                           get_results=True,
                                           name=self._name,
                                           host=self.host,
                                           port=self.port,
                                           type_id=self.connection_type,
                                           max_connections=-1,
                                           return_full_object=True,
                                           unix_file=self.unix_socket)

        if client_result.is_error():
            raise Exception(f"Client {self._name} error: {client_result.print(False)}")
        if not client_result.is_data():
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
        result = await client_result.aget()
        if result is None or result.get('connection_error') != 0:
            raise Exception(f"Client {self._name} error: {client_result.print(False)}")
        self.client = Result.ok(result)

    async def disconnect(self):
        time.sleep(1)
        close = self.client.get("close")
        await close()
        self.client = None

    async def reconnect(self):
        if self.client is not None:
            await self.disconnect()
        await self.connect()

    async def verify(self):
        await asyncio.sleep(1)
        # self.client.get('sender')({'keepalive': 0})
        await self.client.get('sender')(b"verify")

    def __getattr__(self, name):

        # print(f"ProxyApp: {name}, {self.client is None}")
        if name == "on_exit":
            return self.disconnect
        if name == "rc":
            return self.reconnect

        if name == "r":
            try:
                return self.client.get('receiver_queue').get(timeout=self.timeout)
            except:
                return "No data"

        app_attr = getattr(self.class_instance, name)

        async def method(*args, **kwargs):
            # if name == 'run_any':
            #     print("method", name, kwargs.get('get_results', False), args[0])
            if self.client is None:
                await self.reconnect()
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
                    await self.client.get('sender')({'name': name, 'args': args, 'kwargs': kwargs})
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
