import asyncio
import json
import queue
import threading
from typing import Any, Optional, Tuple

from ..toolbox import App
from ..system.types import Result, AppType
from ..system.all_functions_enums import *

from ..system.getting_and_closing_app import get_app
from ..system.tb_logger import get_logger
from ..extras.Style import Style
from ..extras.show_and_hide_console import show_console


class DaemonUtil:

    def __init__(self, *args, **kwargs):
        """
        Standard constructor used for arguments pass
        Do not override. Use __ainit__ instead
        """
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __initobj(self):
        """Crutch used for __await__ after spawning"""
        assert not self.async_initialized
        self.async_initialized = True
        # pass the parameters to __ainit__ that passed to __init__
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self):
        return self.__initobj().__await__()

    async def __ainit__(self, class_instance: Any, host='0.0.0.0', port=6587, t=False,
                        app: Optional[App or AppType] = None,
                        peer=False, name='daemonApp-server', on_register=None, on_client_exit=None, on_server_exit=None,
                        unix_socket=False):
        self.class_instance = class_instance
        self.server = None
        self.port = port
        self.host = host
        self.alive = False
        self._name = name
        if on_register is None:
            on_register = lambda *args: None
        self._on_register = on_register
        if on_client_exit is None:
            on_client_exit = lambda *args: None
        self.on_client_exit = on_client_exit
        if on_server_exit is None:
            on_server_exit = lambda: None
        self.on_server_exit = on_server_exit
        self.unix_socket = unix_socket
        self.online = None
        from toolboxv2.mods.SocketManager import SocketType
        connection_type = SocketType.server
        if peer:
            connection_type = SocketType.peer

        await self.start_server(connection_type)
        app = app if app is not None else get_app(from_=f"DaemonUtil.{self._name}")
        self.online = await asyncio.to_thread(self.connect, app)
        if t:
            await self.online

    async def start_server(self, connection_type):
        """Start the server using app and the socket manager"""
        server_result = get_app(from_="Starting.Daemon").run_any(SOCKETMANAGER.CREATE_SOCKET,
                                                                       get_results=True,
                                                                       name=self._name,
                                                                       host=self.host,
                                                                       port=self.port,
                                                                       type_id=connection_type,
                                                                       max_connections=-1,
                                                                       return_full_object=True,
                                                                       unix_file=self.unix_socket)
        if server_result.is_error():
            raise Exception(f"Server error: {server_result.print(False)}")
        if not server_result.is_data():
            raise Exception(f"Server error: {server_result.print(False)}")
        self.alive = True
        self.server = server_result
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

    async def send(self, data: dict or bytes or str, identifier: Tuple[str, int] or str = "main"):
        result = await self.server.aget()
        sender = result.get('sender')
        await sender(data, identifier)
        return "Data Transmitted"

    async def connect(self, app):
        result = await self.server.aget()
        if not isinstance(result, dict) or result.get('connection_error') != 0:
            raise Exception(f"Server error: {result}")
        self.server = Result.ok(result)
        receiver_queue: queue.Queue = self.server.get('receiver_queue')
        client_to_receiver_thread = self.server.get('client_to_receiver_thread')
        running_dict = self.server.get('running_dict')
        sender = self.server.get('sender')
        known_clients = {}
        valid_clients = {}
        while self.alive:

            if receiver_queue.not_empty:
                data = receiver_queue.get()
                if not data:
                    continue
                if 'identifier' not in data:
                    continue

                identifier = data.get('identifier', 'unknown')
                try:

                    if identifier == "new_con":
                        client, address = data.get('data')
                        get_logger().info(f"New connection: {address}")
                        known_clients[str(address)] = client
                        await client_to_receiver_thread(client, str(address))
                        self._on_register(identifier, address)

                    print("Receiver queue", identifier, identifier in known_clients, identifier in valid_clients)
                    # validation
                    if identifier in known_clients:
                        get_logger().info(identifier)
                        if identifier.startswith("('127.0.0.1'"):
                            valid_clients[identifier] = known_clients[identifier]
                            self._on_register(identifier, data)
                        elif data.get("claim", False):
                            do = app.run_any(("CloudM.UserInstances", "validate_ws_id"),
                                             ws_id=data.get("claim"))[0]
                            get_logger().info(do)
                            if do:
                                valid_clients[identifier] = known_clients[identifier]
                                self._on_register(identifier, data)
                        else:
                            valid_clients[identifier] = known_clients[identifier]
                            self._on_register(identifier, data)
                            # TODO: add support for verification
                            # get_logger().warning(f"Validating Failed: {identifier}")
                            # sender({'Validating Failed': -1}, eval(identifier))
                        get_logger().info(f"Validating New: {identifier}")
                        del known_clients[identifier]

                    elif identifier in valid_clients:
                        get_logger().info(f"New valid Request: {identifier}")
                        name = data.get('name')
                        args = data.get('args')
                        kwargs = data.get('kwargs')

                        get_logger().info(f"Request data: {name=}{args=}{kwargs=}{identifier=}")

                        if name == 'exit_main':
                            self.alive = False
                            break

                        if name == 'show_console':
                            show_console(True)
                            await sender({'ok': 0}, identifier)
                            continue

                        if name == 'hide_console':
                            show_console(False)
                            await sender({'ok': 0}, identifier)
                            continue

                        if name == 'rrun_runnable':
                            show_console(True)
                            runnner = getattr(self.class_instance, "run_runnable")
                            threading.Thread(target=runnner, args=args, kwargs=kwargs, daemon=True).start()
                            await sender({'ok': 0}, identifier)
                            show_console(False)
                            continue

                        async def _helper_runner():
                            try:
                                attr_f = getattr(self.class_instance, name)

                                if asyncio.iscoroutinefunction(attr_f):
                                    res = await attr_f(*args, **kwargs)
                                else:
                                    res = attr_f(*args, **kwargs)

                                if res is None:
                                    res = {'data': res}
                                elif isinstance(res, Result):
                                    if asyncio.iscoroutine(res.get()) or isinstance(res.get(), asyncio.Task):
                                        res_ = await res.aget()
                                        res.result.data = res_
                                    res = json.loads(res.to_api_result().json())
                                elif isinstance(res, bytes):
                                    pass
                                elif isinstance(res, dict):
                                    pass
                                else:
                                    res = {'data': 'unsupported type', 'type': str(type(res))}

                                get_logger().info(f"sending response {res} {type(res)}")

                                await sender(res, identifier)
                            except Exception as e:
                                await sender({"data": str(e)}, identifier)

                        await _helper_runner()

                except Exception as e:
                    get_logger().warning(Style.RED(f"An error occurred on {identifier} {str(e)}"))
                    if identifier != "unknown":
                        running_dict["receive"][str(identifier)] = False
                        self.on_client_exit(identifier)
        running_dict["server_receiver"] = False
        for x in running_dict["receive"].keys():
            running_dict["receive"][x] = False
        running_dict["keep_alive_var"] = False
        self.on_server_exit()
        return Result.ok()

    async def a_exit(self):
        result = await self.server.aget()
        await result.get("close")()
        self.alive = False
        if self.result is not None:
            r = await self.result
            print("Connection result :", r)
