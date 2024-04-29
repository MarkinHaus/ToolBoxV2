"""
The SocketManager Supports 2 types of connections
1. Client Server
2. Peer to Peer

"""
import gzip
import inspect
import io
import json
import os
import random
import time
from dataclasses import dataclass
import logging
from enum import Enum
import zipfile
from io import BytesIO
import shutil
from typing import Optional
from zipfile import ZipInfo, ZIP_LZMA
import uuid

from tqdm import tqdm

from toolboxv2 import MainTool, FileHandler, App, Style, get_app, Result, Spinner
import requests

import socket
import threading
import queue
import asyncio

version = "0.1.9"
Name = "SocketManager"

export = get_app("SocketManager.Export").tb


def zip_folder_to_bytes(folder_path):
    bytes_buffer = BytesIO()
    with zipfile.ZipFile(bytes_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                # Get the modification time of the file
                mtime = os.path.getmtime(file_path)
                # If modification time is before 1980, set it to 1980
                if mtime < 315532800:  # 315532800 seconds represent the beginning of 1980
                    mtime = 315532800
                    # Add the file to the ZIP archive with the modified modification time
                # Set the modification time of the added file in the ZIP archive
                try:
                    zipf.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)
                except ValueError:
                    print(f"skipping arcname {arcname}")
    return bytes_buffer.getvalue()


def zip_folder_to_file(folder_path):
    output_path = f"{folder_path.replace('_', '_').replace('-', '_')}_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return output_path


def unzip_bytes_to_folder(zip_bytes, extract_path):
    bytes_buffer = BytesIO(zip_bytes)
    with zipfile.ZipFile(bytes_buffer, 'r') as zipf:
        zipf.extractall(extract_path)


def unzip_file_to_folder(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(extract_path)


@dataclass
class SocketType(Enum):
    server = "server"
    client = "client"
    peer = "peer"


create_socket_samples = [{'name': 'test', 'host': '0.0.0.0', 'port': 62435,
                          'type_id': SocketType.client,
                          'max_connections': -1, 'endpoint_port': None,
                          'return_full_object': False,
                          'keepalive_interval': 1000},
                         {'name': 'sever', 'host': '0.0.0.0', 'port': 62435,
                          'type_id': SocketType.server,
                          'max_connections': -1, 'endpoint_port': None,
                          'return_full_object': False,
                          'keepalive_interval': 1000},
                         {'name': 'peer', 'host': '0.0.0.0', 'port': 62435,
                          'type_id': SocketType.server,
                          'max_connections': -1, 'endpoint_port': 62434,
                          'return_full_object': False,
                          'keepalive_interval': 1000}, ]


def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        ip_address = response.json()['ip']
        return ip_address
    except Exception as e:
        print(f"Fehler beim Ermitteln der öffentlichen IP-Adresse: {e}")
        return None


def get_local_ip():
    try:
        # Erstellt einen Socket, um eine Verbindung mit einem öffentlichen DNS-Server zu simulieren
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Verwendet Google's öffentlichen DNS-Server als Ziel, ohne tatsächlich eine Verbindung herzustellen
            s.connect(("8.8.8.8", 80))
            # Ermittelt die lokale IP-Adresse, die für die Verbindung verwendet würde
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception as e:
        print(f"Fehler beim Ermitteln der lokalen IP-Adresse: {e}")
        return None


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.tasks = {}
        self.running = False
        self.version = version
        self.name = "SocketManager"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"], ["create_socket", "crate a socket", -1],
                    ["tbSocketController", "run daemon", -1]],
            "name": "SocketManager",
            "tbSocketController": self.run_as_single_communication_server,
            "Version": self.show_version,
        }
        self.local_ip = None
        self.public_ip = None
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)
        self.sockets = {}
        self.loop = asyncio.new_event_loop()

    async def on_start(self):
        self.logger.info(f"Starting SocketManager")
        t0 = asyncio.create_task(asyncio.to_thread(self.set_print_public_ip))
        t1 = asyncio.create_task(asyncio.to_thread(self.set_print_local_ip))
        await asyncio.sleep(3)
        await t0
        await t1
        # ~ self.load_file_handler()

    async def on_exit(self):
        self.logger.info(f"Closing SocketManager")
        for socket_name, socket_data in self.sockets.items():
            if not socket_data.get("alive"):
                continue
            self.print(f"consing Socket : {socket_name}")
            try:
                await socket_data.get("close")()
            except:
                self.print(f"consing Socket : {socket_name}")
            self.sockets[socket_name] = None
        self.sockets = {}
        return "OK"

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    async def set_print_local_ip(self):

        if self.local_ip is None:
            self.local_ip = get_local_ip()
            self.print(f"Device IP : {self.local_ip}")

    async def set_print_public_ip(self):

        if self.local_ip is None:
            self.public_ip = get_public_ip()
            self.print(f"Network IP : {self.public_ip}")

    def create_server(self, name: str, port: int, host: str, socket_type, max_connections=-1, unix_file=False,
                      handler=None) -> Result:
        self.logger.debug(f"Starting:{name} server on port {port} with host {host}")

        sock = socket.socket(socket_type, socket.SOCK_STREAM)

        try:
            if unix_file:
                sock.bind(host)
            else:
                sock.bind((host, port))
            sock.listen(max_connections)
            self.print(f"Server:{name} online at {host}:{port}")
        except Exception as e:
            self.print(Style.RED(f"Server:{name} error at {host}:{port} {e}"))
            return Result.default_internal_error(exec_code=-1, info=str(e), data="Server creation failed")

        def start_server():
            if handler is None:
                return sock
            s_thread = threading.Thread(target=handler, args=(name, sock,), daemon=True)
            s_thread.start()
            return s_thread

        return Result.ok(start_server())

    async def server_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        pass

    async def a_create_server(self, name: str, port: int, host: str, unix_file=False, handler=None) -> Result:
        self.logger.debug(f"Starting:{name} server on port {port} with host {host}")

        if handler is None:
            handler = self.server_handler

        try:
            if unix_file:
                server = await asyncio.start_unix_server(handler, host)
            else:
                server = await asyncio.start_server(handler, host, port)
            self.print(f"Server:{name} online at {host}:{port}")
        except Exception as e:
            self.print(Style.RED(f"Server:{name} error at {host}:{port} {e}"))
            return Result.default_internal_error(exec_code=-1, info=str(e), data="Server creation failed")

        async def running_server_instance():
            async with server:
                await server.start_serving()
                while server.is_serving():
                    if not self.app.alive or self.sockets[name]["running_dict"]["server_receiver"]:
                        break
                    await asyncio.sleep(1)

        return Result.future(asyncio.create_task(running_server_instance()))

    def create_client(self, name: str, port: int, host: str, socket_type, unix_file: bool, handler=None) -> Result:
        self.logger.debug(f"Starting:{name} client on port {port} with host {host}")
        sock = socket.socket(socket_type, socket.SOCK_STREAM)
        time.sleep(random.choice(range(1, 100)) // 100)
        if unix_file:
            connection_error = sock.connect_ex(host)
        else:
            connection_error = sock.connect_ex((host, port))
        if connection_error != 0:
            sock.close()
            self.print(f"Client:{name}-{host}-{port} connection_error:{connection_error}")
            return Result.default_internal_error(exec_code=connection_error,
                                                 info="Client creation failed, check connection and Server")
        else:
            self.print(f"Client:{name} online at {host}:{port}")

        # sock.sendall(bytes(self.app.id, 'utf-8'))

        def start_client():
            if handler is None:
                return None, sock
            c_thread = threading.Thread(target=handler, args=(sock,), daemon=True)
            c_thread.start()
            return c_thread, sock

        return Result.ok(start_client())

    async def a_create_client(self, name: str, port: int, host: str, unix_file: bool) -> Result:
        self.logger.debug(f"Starting:{name} client on port {port} with host {host}")

        time.sleep(random.choice(range(1, 100)) // 100)
        try:
            if unix_file:
                reader, writer = await asyncio.open_unix_connection(host)
            else:
                reader, writer = await asyncio.open_connection(host, port)
        except Exception as e:
            self.print(f"Client:{name}-{host}-{port} connection_error:{str(e)}")
            return Result.default_internal_error(exec_code=-1,
                                                 info="Client creation failed, check connection and Server",
                                                 data=str(e))
        self.print(f"Client:{name} online at {host}:{port}")
        # sock.sendall(bytes(self.app.id, 'utf-8'))
        return Result.ok((reader, writer))

    def create_peer(self, name: str, port: int, endpoint_port: int, host: str) -> Result:

        if host == "localhost" or host == "127.0.0.1":
            self.print("LocalHost Peer2Peer is not supported use server client architecture")
            return Result.default_internal_error(exec_code=-1,
                                                 info="LocalHost Peer2Peer is not supported use server client architecture")

        if host == '0.0.0.0':
            public_ip = self.local_ip
        else:
            if self.public_ip is None:
                self.print("Getting IP address")
                self.public_ip = get_public_ip()
            public_ip = self.public_ip

        self.logger.debug(f"Starting:{name} peer on port {port} with host {host}")

        try:

            r_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            r_socket.bind(('0.0.0.0', endpoint_port))
            self.print(f"Peer:{name} receiving on {public_ip}:{endpoint_port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('0.0.0.0', port))
            sock.sendto(b'k', (host, endpoint_port))
            self.print(f"Peer:{name} sending to on {host}:{port}")
            return Result.ok((sock, r_socket))
        except Exception as e:
            return Result.default_internal_error(exec_code=-1,
                                                 info="Client creation failed, check connection and Server",
                                                 data=str(e))

    async def a_create_peer(self, name, port, endpoint_port, host):
        if host == "localhost" or host == "127.0.0.1":
            self.print("LocalHost Peer2Peer is not supported use server client architecture")
            return Result.default_internal_error(exec_code=-1,
                                                 info="LocalHost Peer2Peer is not supported use server client architecture")

        if host == '0.0.0.0':
            public_ip = self.local_ip
        else:
            if self.public_ip is None:
                self.print("Getting IP address")
                await self.set_print_public_ip()
            public_ip = self.public_ip

        self.logger.debug(f"Starting:{name} peer on port {port} with host {host}")

        try:

            r_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            r_socket.bind(('0.0.0.0', endpoint_port))

            self.print(f"Peer:{name} receiving on {public_ip}:{endpoint_port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('0.0.0.0', port))
            sock.sendto(b'k', (host, endpoint_port))

            server = await self.a_create_server(name, endpoint_port, '0.0.0.0', False)

            clinet = await self.a_create_client(name, port, host, False)

            self.print(f"Peer:{name} sending to on {host}:{port}")
            return Result.ok((sock, r_socket))
        except Exception as e:
            return Result.default_internal_error(exec_code=-1,
                                                 info="Client creation failed, check connection and Server",
                                                 data=str(e))

    ############ Helper fuction ###################

    def exit_socket(self, name):
        self.sockets[name]["alive"] = False
        self.sockets[name]["running_dict"]["server_receiver"].set()
        self.sockets[name]["running_dict"]["keep_alive_var"].set()
        list(map(lambda client_receiver_threads_event: client_receiver_threads_event.set(),
                 self.sockets[name]["running_dict"]["receive"].values()))

        if self.sockets[name]["running_dict"]["thread_receiver"] is not None:
            try:
                self.sockets[name]["running_dict"]["thread_receiver"].join(timeout=0.251 if not self.app.debug else 0.1)
                if not self.sockets[name]["do_async"] and self.sockets[name]["running_dict"][
                    "server_receiver_"] is not None:
                    self.sockets[name]["running_dict"]["server_receiver_"].join(
                        timeout=0.251 if not self.app.debug else 0.1)
            except TimeoutError as e:
                pass
            self.sockets[name]["running_dict"]["thread_receiver"] = None

    async def a_exit_socket(self, name):
        if name not in self.sockets:
            return
        if self.sockets[name]["type_id"] == SocketType.client.name:
            await self.sockets[name]["sender"]({"exit": True})
        self.exit_socket(name)

    def register_identifier(self, name, connection, identifier="main"):
        connection_key = None
        if isinstance(connection, tuple) and isinstance(connection[1], socket.socket) and connection[0] is None:
            # Therefor the connection is synchrony and no receiver running
            if self.sockets[name]["type_id"] == SocketType.server.name:
                addr = connection[1].getpeername()
            else:
                addr = connection[1].getsockname()
            connection_key = addr[0] + str(addr[1])
            if connection_key in self.sockets[name]["client_sockets_dict"].keys():
                pass
            elif addr[0] == "127.0.0.1":
                connection_key = "localhost]" + str(addr[1])
                if connection_key in self.sockets[name]["client_sockets_dict"].keys():
                    pass
                else:
                    connection_key = None
            else:
                connection_key = None
        elif isinstance(connection, socket.socket):
            addr = connection.getpeername()
            connection_key = addr[0] + str(addr[1])
            if connection_key in self.sockets[name]["client_sockets_dict"].keys():
                pass
            elif addr[0] == "127.0.0.1":
                connection_key = "localhost" + str(addr[1])
                if connection_key in self.sockets[name]["client_sockets_dict"].keys():
                    pass
                else:
                    connection_key = None
            else:
                connection_key = None
        else:
            for _k, _v in self.sockets[name]["client_sockets_dict"].items():
                # print(f"\nregister_identifier {name} {_k=}, {_v=} {connection=}\n")
                if _v == connection:
                    connection_key = _k
                    break
        if connection_key is None:
            raise ValueError(f"No Unknown Connection : {connection} known {self.sockets[name]['client_sockets_dict']}")
        self.print(f"{name} registered with {connection_key} as {identifier}")
        self.sockets[name]["client_sockets_identifier"][identifier] = connection_key

    def register_new_connection_helper(self, name, client_socket, endpoint):
        self.sockets[name]["connections"] += 1
        self.print(
            f"New connection: on {name} acc:{self.sockets[name]['connections']}"
            f" max:{self.sockets[name]['max_connections']} connect on :{endpoint}")
        self.sockets[name]["client_sockets_dict"][endpoint[0] + str(endpoint[1])] = client_socket
        if self.sockets[name]['max_connections'] != -1:
            if self.sockets[name]["connections"] >= self.sockets[name]['max_connections']:
                self.sockets[name]["running_dict"]["server_receiver"].set()

    async def a_register_new_connection(self, name, client_socket, endpoint):
        await self.sockets[name]["a_receiver_queue"].put({'data': (client_socket, endpoint), 'identifier': "new_con"})
        self.register_new_connection_helper(name, client_socket, endpoint)

    def register_new_connection(self, name, client_socket, endpoint):
        self.sockets[name]["receiver_queue"].put({'data': (client_socket, endpoint), 'identifier': "new_con"})
        self.register_new_connection_helper(name, client_socket, endpoint)

    def server_receiver(self, name, sock_):
        self.print(f"Server {name} receiver started")
        while not self.sockets[name]["running_dict"]["server_receiver"].is_set() and self.sockets[name]["alive"]:
            try:
                client_socket, endpoint = sock_.accept()
            except OSError as e:
                print("Error", e)
                self.sockets[name]["running_dict"]["server_receiver"].set()
                break

            self.register_new_connection(name, client_socket, endpoint)
        self.print(f"Server {name} receiver Closed")

    async def send(self, name, msg: bytes or dict, identifier="main"):
        receiver = None

        self.print(f"Sending data to {identifier} as {name}")

        async def send_(chunk, drain=False):
            async def a_send_():
                try:
                    writer.write(chunk)
                    if drain:
                        await writer.drain()
                except Exception as e:
                    self.logger.error(f"Error sending data: {e}")

            def s_send_():
                try:
                    if self.sockets[name]["type_id"] == SocketType.client.name:
                        # self.print(f"Start sending data to client {_socket.getpeername()}")
                        _socket.sendall(chunk)
                    elif self.sockets[name]["type_id"] == SocketType.server.name:
                        # self.print(f"Start sending data to {address}")
                        _socket.sendto(chunk, (self.sockets[name]["host"], self.sockets[name]["port"]))
                    elif self.sockets[name]["type_id"] == SocketType.peer.name:
                        # self.print(
                        #     f"Start sending data to peer at {(self.sockets[name]['host'], self.sockets[name]['p2p-port'])}")
                        _socket.sendto(chunk, (self.sockets[name]["host"], self.sockets[name]["p2p-port"]))
                    else:
                        self.print(f"Start sending data with {_socket}")
                        _socket.sendall(chunk)
                except Exception as e:
                    self.logger.error(f"Error sending data: {e}")

            if self.sockets[name]["do_async"]:
                _, writer = receiver
                await a_send_()
            else:
                _socket = receiver
                s_send_()

        t0 = time.perf_counter()
        if identifier in self.sockets[name]["client_sockets_identifier"]:
            receiver = self.sockets[name]["client_sockets_dict"][
                self.sockets[name]["client_sockets_identifier"][identifier]]
        else:
            self.logger.warn(
                Style.YELLOW(f"Invalid {identifier=} valid ar : {self.sockets[name]['client_sockets_identifier']}"))
            return f"Invalid {identifier=} valid ar : {self.sockets[name]['client_sockets_identifier']}"

        # Prüfen, ob die Nachricht ein Dictionary ist und Bytes direkt unterstützen
        if isinstance(msg, bytes):
            sender_bytes = b'b' + msg  # Präfix für Bytes
            msg_json = 'sending bytes'
        elif isinstance(msg, dict):
            if 'exit' in msg:
                sender_bytes = b'e'  # Präfix für "exit"
                msg_json = 'exit'
                self.sockets[name]["running_dict"]["receive"][identifier].set()
            elif 'keepalive' in msg:
                sender_bytes = b'k'  # Präfix für "keepalive"
                msg_json = 'keepalive'
            else:
                msg_json = json.dumps(msg)
                sender_bytes = b'j' + msg_json.encode('utf-8')  # Präfix für JSON
        else:
            self.print(Style.YELLOW(f"Unsupported message type: {type(msg)}"))
            return

        if sender_bytes != b'k' and self.app.debug:
            self.print(Style.GREY(f"Sending Data: {msg_json} {self.sockets[name]['host']}"))

        if sender_bytes == b'k':
            await send_(sender_bytes)
            return
        if sender_bytes == b'e':
            await send_(sender_bytes)
            self.exit_socket(name)
            return

        total_steps = len(sender_bytes) // self.sockets[name]["package_size"]
        if len(sender_bytes) % self.sockets[name]["package_size"] != 0:
            total_steps += 1  # Einen zusätzlichen Schritt hinzufügen, falls ein Rest existiert
        self.logger.info("Start sending data")
        # tqdm Fortschrittsanzeige initialisieren
        with tqdm(total=total_steps, unit='chunk', desc='Sending data') as pbar:
            for i in range(0, len(sender_bytes), self.sockets[name]["package_size"]):
                chunk_ = sender_bytes[i:i + self.sockets[name]["package_size"]]
                await send_(chunk_, receiver)
                pbar.update(1)
                time.sleep(1 / 10 ** 18)
        # self.print(f"\n\n{len(sender_bytes)=}, {i + package_size}")
        if len(sender_bytes) != i + self.sockets[name]["package_size"]:
            await send_(sender_bytes[i + self.sockets[name]["package_size"]:], receiver)

        if len(sender_bytes) < self.sockets[name]["package_size"]:
            await send_(b' ' * (len(sender_bytes) - self.sockets[name]["package_size"]), receiver)
        if len(sender_bytes) % self.sockets[name]["package_size"] != 0:
            pass
        if self.sockets[name]["type_id"] == SocketType.peer.name:
            await send_(b'E' * 6, receiver)
        else:
            await send_(b'E' * (self.sockets[name]["package_size"] // 10), receiver)

        if self.sockets[name]["do_async"]:
            _, writer = receiver
            await writer.drain()

        self.logger.info(f"{name} :S Parsed Time ; {time.perf_counter() - t0:.2f}")

    async def chunk_receive(self, name, r_socket_, identifier="main"):
        try:
            if self.sockets[name]["do_async"]:
                chunk = await r_socket_.read(self.sockets[name]["package_size"])
            elif self.sockets[name]["type_id"] == SocketType.client.name:
                chunk, _ = await asyncio.to_thread(r_socket_.recvfrom, self.sockets[name]["package_size"])
            else:
                try:
                    chunk = await asyncio.to_thread(r_socket_.recv, self.sockets[name]["package_size"])
                except Exception as e:
                    return Result.custom_error(data=str(e), data_info="Connection down and closed")
        except ConnectionResetError and ConnectionAbortedError as e:
            self.print(f"Closing Receiver {name}:{identifier} {str(e)}")
            self.sockets[name]["running_dict"]["receive"][identifier].set()
            if self.sockets[name]["type_id"] == SocketType.client.name:
                if self.sockets[name]["do_async"]:
                    await self.a_exit_socket(name)
                else:
                    self.exit_socket(name)
            return Result.custom_error(data=str(e), data_info="Connection down and closed")
        if not chunk:
            return Result.default_internal_error("No data available pleas exit")

        return Result.ok(data=chunk).set_origin((name, identifier))

    async def compute_bytes(self, name, row_data, identifier="main"):
        self.logger.info(f"{name} -- received bytes --")
        return {'bytes': row_data, 'identifier': identifier}

    async def compute_json(self, name, row_data, identifier="main"):
        try:
            msg = json.loads(row_data)
            msg['identifier'] = identifier
            self.logger.info(f"{name} -- received JSON -- {msg['identifier']}")
            return msg
        except json.JSONDecodeError and UnicodeDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")

        return -1

    async def compute_data(self, name, row_data, data_type, identifier="main"):
        if data_type == b'b':
            # Behandlung von Byte-Daten
            return await self.compute_bytes(name, row_data, identifier)
        elif data_type == b'j':
            # Behandlung von JSON-Daten
            return await self.compute_json(name, row_data, identifier)
        else:
            self.logger.error("Unbekannter Datentyp")
            self.print(f"Received unknown data type: {data_type}")
        return None

    def helper_1_receive(self, name, identifier="main"):
        task = self.loop.create_task(self.receive(name, identifier))
        self.tasks[name] = task

    async def helper_0_receive(self):
        await asyncio.gather(*self.tasks.values())

    async def receive_helper(self, name, identifier="mian"):

        self.sockets[name]["running_dict"]["receive"][identifier] = asyncio.Event()

        self.helper_1_receive(name, identifier)

        if self.sockets[name]["running_dict"]["thread_receiver_"] and self.loop.is_running():
            return

        def helper_():
            self.loop.run_until_complete(self.helper_0_receive())

        t = threading.Thread(
            target=helper_, daemon=True
        )

        await asyncio.sleep(1)

        t.start()

        self.sockets[name]["running_dict"]["thread_receiver"] = t
        self.sockets[name]["running_dict"]["thread_receiver_"] = True

    async def receive(self, name, identifier="main"):
        print(f"Received Started for {name} {identifier}")
        data_type = None
        data_buffer = io.BytesIO()
        max_size = -1
        ac_size = 0
        extra = None

        if identifier in self.sockets[name]["client_sockets_identifier"]:
            receiver = self.sockets[name]["client_sockets_dict"][
                self.sockets[name]["client_sockets_identifier"][identifier]]
        else:
            return Result.default_internal_error(f"Unknown identifier {identifier} ",
                                                 data=self.sockets[name]["client_sockets_identifier"].keys())

        if self.sockets[name]["do_async"]:
            r_socket_, extra = receiver
        else:
            r_socket_ = receiver
        self.print(f"Receiver running for {name} to {identifier}")
        self.sockets[name]["running_dict"]["receive"][identifier] = asyncio.Event()
        while not self.sockets[name]["running_dict"]["receive"][identifier].is_set() and self.sockets[name]["alive"]:
            # t0 = time.perf_counter()
            chunk_result = await self.chunk_receive(name, r_socket_, identifier=identifier)

            # chunk_result.print()

            if chunk_result.is_error():
                return chunk_result

            chunk = chunk_result.get()

            if chunk == b'k':
                # Behandlung von Byte-Daten
                self.logger.info(f"{name} -- received keepalive signal--")
                continue

            if data_type is None:
                data_type = chunk[:1]  # Erstes Byte ist der Datentyp
                chunk = chunk[1:]  # Rest der Daten
                self.print(f"Register date type : {data_type} :{name}-{identifier}")
            if data_type == b'e':
                if isinstance(self.sockets[name]["running_dict"]["receive"][identifier], asyncio.Event):
                    self.sockets[name]["running_dict"]["receive"][identifier].set()
                if isinstance(self.sockets[name]["running_dict"]["receive"][identifier], bool):
                    self.sockets[name]["running_dict"]["receive"][identifier] = False
                if isinstance(self.sockets[name]['running_dict']["keep_alive_var"], asyncio.Event):
                    self.sockets[name]['running_dict']["keep_alive_var"].set()
                if isinstance(self.sockets[name]['running_dict']["keep_alive_var"], bool):
                    self.sockets[name]['running_dict']["keep_alive_var"] = False
                self.logger.info(f"{name} -- received exit signal --")
                print(f"{name} -- received exit signal --")
                break
            if not (chunk[0] == b'E'[0] and chunk[-1] == b'E'[0]):
                # Check if adding this chunk would exceed the max size limit
                if max_size != -1 and ac_size + len(chunk) > max_size:
                    # If the chunk would exceed the max size, add only the necessary part of it
                    chunk = chunk[:(max_size - ac_size) + 1]
                    data_buffer.write(chunk)
                else:
                    data_buffer.write(chunk)
                ac_size += len(chunk)

            if max_size > -1 and ac_size > 0 and data_type == b'b':
                print(
                    f"don {chunk[0] == b'E'[0] and chunk[-1] == b'E'[0]} {(ac_size / max_size) * 100:.2f}% total byts: {ac_size} von {max_size}",
                    end='\r')

            elif chunk[0] == b'E'[0] and chunk[-1] == b'E'[0] and ac_size > 0 and (
                ac_size == max_size or max_size == -1):
                max_size = -1
                ac_size = 0
                # Letzter Teil des Datensatzes
                data = await self.compute_data(name, data_buffer.getvalue(), data_type, identifier=identifier)

                if data:
                    self.print(f"Daten wurden empfangen {name} {identifier}")
                    if max_size == -1 and isinstance(data, dict) and "max_size" in data:
                        max_size = data.get("max_size")
                    if self.sockets[name]["type_id"] == SocketType.client.name and 'identifier' in data and data.get(
                        "identifier") == 'main':
                        del data['identifier']
                    if self.sockets[name]["do_async"]:
                        await self.sockets[name]["a_receiver_queue"].put(data)
                    else:
                        self.sockets[name]["receiver_queue"].put(data)
                # Zurücksetzen für den nächsten Datensatz
                del data_buffer
                data_buffer = io.BytesIO()
                data_type = None

        data_buffer.close()
        self.print(f"{name} :closing connection to {self.sockets[name]['host']}")
        if name in self.sockets and self.sockets[name]['type_id'] == SocketType.client.name:
            self.sockets[name]['alive'] = False
        if self.sockets[name]["do_async"] and extra is not None:
            if self.sockets[name]['type_id'] == SocketType.peer.name or self.sockets[name][
                'type_id'] == SocketType.client.name:
                extra.write(b'e')
                await extra.drain(b'e')
            extra.close()
            await extra.wait_close()
        else:
            print("CLOSING SOKET")
            r_socket_.close()
            # if type_id == SocketType.peer.name and extra is not None:
            #    extra.close()

    ############### END ###############

    @export(mod_name="SocketManager", version=version, samples=create_socket_samples, test=False)
    async def create_socket(self, name: str = 'local-host', host: str = '0.0.0.0', port: int or None = None,
                            type_id: SocketType = SocketType.client,
                            max_connections=-1, endpoint_port=None,
                            return_full_object=False, keepalive_interval=6, test_override=False, package_size=1024,
                            start_keep_alive=True, unix_file=False, do_async=False) -> Result:

        # start queues sender, receiver, acceptor
        a_receiver_queue = asyncio.Queue()
        receiver_queue = queue.Queue()

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"

        if not isinstance(type_id, SocketType):
            return

        # setup sockets
        type_id = type_id.name
        server_result = Result.default()
        connection_error = 0

        if self.local_ip is None:
            self.local_ip = await self.set_print_local_ip()

        if unix_file:
            socket_type = socket.AF_UNIX
        else:
            socket_type = socket.AF_INET

        def close_helper():
            return lambda: self.a_exit_socket(name)

        async def to_receive(client, identifier='main'):
            if isinstance(client, str):
                print("Client $$", client, identifier)
                return
            self.register_identifier(name, client, identifier)
            await self.receive_helper(name, identifier)

        self.sockets[name] = {
            'alive': True,
            'close': close_helper(),
            'max_connections': max_connections,
            'type_id': type_id,
            'do_async': do_async,
            'package_size': package_size,
            'host': host,
            'port': port,
            'p2p-port': endpoint_port,
            'sender': lambda msg, identifier="main": self.send(name, msg=msg, identifier=identifier),
            'connections': 0,
            'receiver_queue': receiver_queue,
            'a_receiver_queue': a_receiver_queue,
            'connection_error': connection_error,
            'running_dict': {
                "server_receiver": asyncio.Event(),
                "server_receiver_": None,
                "thread_receiver": None,
                "thread_receiver_": False,
                "receive": {

                },
                "keep_alive_var": asyncio.Event()
            },
            'client_sockets_dict': {},
            'client_sockets_identifier': {},
            'client_to_receiver_thread': to_receive,
        }

        # server receiver

        if type_id == SocketType.server.name:

            async def a_server_receiver(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
                endpoint = writer.get_extra_info("peername")
                await self.a_register_new_connection(name, (reader, writer), endpoint)

            # create sever
            server_result = self.create_server(name=name,
                                               port=port,
                                               host=host,
                                               socket_type=socket_type,
                                               max_connections=max_connections,
                                               unix_file=unix_file,
                                               handler=self.server_receiver) \
                if not do_async else await self.a_create_server(name=name,
                                                                port=port,
                                                                host=host,
                                                                unix_file=unix_file,
                                                                handler=a_server_receiver)

            if server_result.is_error():
                return server_result
            if not do_async:
                self.sockets[name]["running_dict"]["server_receiver_"] = server_result.get()
            # sock = await server_result.aget()

        elif type_id == SocketType.client.name:
            # create client
            client_result = self.create_client(name, port, host, socket_type, unix_file) \
                if not do_async else await self.a_create_client(name, port, host, unix_file)

            if client_result.is_error():
                return client_result

            if do_async:
                r_socket = await client_result.aget()
            else:
                c_thread, c_socket = client_result.get()
                r_socket = c_socket
            self.sockets[name]["client_sockets_dict"][host + str(port)] = r_socket
            await to_receive(r_socket, "main")

        elif type_id == SocketType.peer.name:
            # create peer

            if endpoint_port is None and port is None:
                port = 62435

            if port is None:
                port = endpoint_port - 1

            if endpoint_port is None:
                endpoint_port = port + 1

            if endpoint_port == port:
                endpoint_port += 1

            if do_async:
                raise NotImplementedError("peer is not supported yet in async")
            peer_result = self.create_peer(name, port, endpoint_port, host)
            if peer_result.is_error():
                return peer_result

            sock, r_socket = peer_result.get()
        else:
            self.print(f"Invalid SocketType {type_id}:{name}")
            raise ValueError(f"Invalid SocketType {type_id}:{name}")

        # manually starting server/(client:peer)receive if needed

        if type_id == SocketType.peer.name:

            def keep_alive():
                i = 0
                while not self.sockets[name]["running_dict"]["keep_alive_var"].is_set() and self.sockets[name]["alive"]:
                    time.sleep(keepalive_interval)
                    try:
                        self.send(name, {'keepalive': True})
                    except Exception as e:
                        self.print(f"Exiting keep alive {e}")
                        break
                    i += 1
                self.print("Closing KeepAlive")
                self.send(name, {"exit": True})

            keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
            if start_keep_alive:
                keep_alive_thread.start()

        elif type_id == SocketType.server.name:

            self.sockets[name]["client_to_receiver_thread"] = to_receive
            await asyncio.sleep(1)
        elif type_id == SocketType.client.name:
            await asyncio.sleep(1)

        if return_full_object:
            return Result.ok(self.sockets[name])

        return Result.ok(self.sockets[name]["sender"], receiver_queue)

        # sender queue

    ####### P2P server ##############
    @export(mod_name=Name, name="run_as_ip_echo_server_a", test=False)
    async def run_as_ip_echo_server_a(self, name: str = 'local-host', host: str = '0.0.0.0', port: int = 62435,
                                      max_connections: int = -1, test_override=False):

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"
        socket_data = await self.create_socket(name, host, port, SocketType.server, max_connections=max_connections)
        if not socket_data.is_error():
            return socket_data
        send, receiver_queue = socket_data.get()
        clients = {}

        self.running = True

        def send_to_all(sender_ip, sender_port, sender_socket):
            c_clients = {}
            offline_clients = []
            for client_name_, client_ob_ in clients.items():
                client_port_, client_ip_, client_socket_ = client_ob_.get('port', None), client_ob_.get('ip',
                                                                                                        None), client_ob_.get(
                    'client_socket', None)

                if client_port_ is None:
                    continue
                if client_ip_ is None:
                    continue
                if client_socket_ is None:
                    continue

                if (sender_ip, sender_port) != (client_ip_, client_port_):
                    try:
                        client_socket_.sendall(
                            json.dumps({'data': 'Connected client', 'ip': sender_ip, 'port': sender_port}).encode(
                                'utf-8'))
                        c_clients[str(client_ip_)] = client_port_
                    except Exception as e:
                        offline_clients.append(client_name_)

            sender_socket.sendall(json.dumps({'data': 'Connected clients', 'clients': c_clients}).encode('utf-8'))
            for offline_client in offline_clients:
                del clients[offline_client]

        max_connections_ = 0
        while self.running:

            if receiver_queue.not_empty:
                client_socket, connection = receiver_queue.get()
                max_connections_ += 1
                ip, port = connection

                client_dict = clients.get(str(port))
                if client_dict is None:
                    clients[str(port)] = {'ip': ip, 'port': port, 'client_socket': client_socket}

                send_to_all(ip, port, client_socket)

            if max_connections_ >= max_connections:
                self.running = False
                break

        self.print("Stopping server closing open clients")

        for client_name, client_ob in clients.items():
            client_port, client_ip, client_socket = client_ob.get('port', None), client_ob.get('ip',
                                                                                               None), client_ob.get(
                'client_socket', None)

            if client_port is None:
                continue
            if client_ip is None:
                continue
            if client_socket is None:
                continue

            client_socket.sendall("exit".encode('utf-8'))

    @export(mod_name=Name, name="run_as_single_communication_server", test=False)
    async def run_as_single_communication_server(self, name: str = 'local-host', host: str = '0.0.0.0',
                                                 port: int = 62435,
                                                 test_override=False):

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"

        socket_data = await self.create_socket(name, host, port, SocketType.server, max_connections=1)
        if not socket_data.is_error():
            return socket_data
        send, receiver_queue = socket_data.get()
        status_queue = queue.Queue()
        running = [True]  # Verwenden einer Liste, um den Wert referenzierbar zu machen

        def server_thread(client, address):
            self.print(f"Receiver connected to address {address}")
            status_queue.put(f"Server received client connection {address}")
            while running[0]:
                t0 = time.perf_counter()
                try:
                    msg_json = client.recv(1024).decode()
                except socket.error:
                    break

                self.print(f"run_as_single_communication_server -- received -- {msg_json}")
                status_queue.put(f"Server received data {msg_json}")
                if msg_json == "exit":
                    running[0] = False
                    break
                if msg_json == "keepAlive":
                    status_queue.put("KEEPALIVE")
                else:
                    msg = json.loads(msg_json)
                    data = self.app.run_any(**msg, get_results=True)
                    status_queue.put(f"Server returned data {data.print(show=False, show_data=False)}")
                    data = data.get()

                    if not isinstance(data, dict):
                        data = {'data': data}

                    client.send(json.dumps(data).encode('utf-8'))

                self.print(f"R Parsed Time ; {time.perf_counter() - t0}")

            client.close()
            status_queue.put("Server closed")

        def helper():
            client, address = receiver_queue.get(block=True)
            thread = threading.Thread(target=server_thread, args=(client, address), daemon=True)
            thread.start()

        threading.Thread(target=helper, daemon=True).start()

        def stop_server():
            running[0] = False
            status_queue.put("Server stopping")

        def get_status():
            while status_queue.not_empty:
                yield status_queue.get()

        return {"stop_server": stop_server, "get_status": get_status}

    @export(mod_name=Name, name="send_file_to_sever", test=False)
    async def send_file_to_sever(self, filepath, host, port):
        if isinstance(port, str):
            try:
                port = int(port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{port} is not an int or not cast to int")
        # Überprüfen, ob die Datei existiert
        if not os.path.exists(filepath):
            self.logger.error(f"Datei {filepath} nicht gefunden.")
            return f"Datei {filepath} nicht gefunden."

        if '.' in filepath.split('/')[-1]:
            with open(filepath, 'rb') as f:
                to_send_data = gzip.compress(f.read())
        else:
            to_send_data = zip_folder_to_bytes(filepath)
        # Datei komprimieren
        compressed_data = gzip.compress(to_send_data)

        # Peer-to-Peer Socket erstellen und verbinden
        socket_data = await self.create_socket(name="sender", host=host, port=port, type_id=SocketType.client,
                                               return_full_object=True)

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
        if not socket_data.is_error():
            return socket_data

        socket_data = socket_data.get()
        send = socket_data['sender']

        # Komprimierte Daten senden
        try:
            # Größe der komprimierten Daten senden
            send({'data_size': len(compressed_data)})
            # Komprimierte Daten senden
            time.sleep(2)
            send(compressed_data + b'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
            self.logger.info(f"Datei {filepath} erfolgreich gesendet.")
            self.print(f"Datei {filepath} erfolgreich gesendet.")
            send({'exit': True})
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Datei: {e}")
            self.print(f"Fehler beim Senden der Datei: {e}")
            return False
        finally:
            socket_data['running_dict']["keep_alive_var"].set()

    @export(mod_name=Name, name="receive_and_decompress_file_as_server", test=False)
    async def receive_and_decompress_file_from_client(self, save_path, listening_port):
        # Empfangs-Socket erstellen
        if isinstance(listening_port, str):
            try:
                listening_port = int(listening_port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{listening_port} is not an int or not cast to int")

        socket_data = await self.create_socket(name="receiver", host='0.0.0.0', port=listening_port,
                                               type_id=SocketType.server,
                                               return_full_object=True, max_connections=1)
        if not socket_data.is_error():
            return socket_data
        socket_data = socket_data.get()
        receiver_queue = socket_data['receiver_queue']
        to_receiver = socket_data['client_to_receiver_thread']
        data = receiver_queue.get(block=True)
        client, address = data.get('data')
        to_receiver(client, 'client-' + str(address))

        file_data = b''
        file_size = -1
        while True:
            # Auf Daten warten
            data = receiver_queue.get()
            if 'data_size' in data:
                file_size = data['data_size']
                self.logger.info(f"Erwartete Dateigröße: {file_size} Bytes")
                self.print(f"Erwartete Dateigröße: {file_size} Bytes")
            elif 'bytes' in data:
                file_data += data['bytes']
                self.print(f"Erhaltende Bytes: {len(file_data)} Bytes")
                # Daten dekomprimieren
                if len(file_data) > 0:
                    print(f"{len(file_data) / file_size * 100:.2f}%")

                if len(file_data) > file_size:
                    file_data = file_data[:file_size]
                else:
                    continue

                decompressed_data = gzip.decompress(file_data)
                # Datei speichern
                if '.' in save_path.split('/')[-1]:
                    with open(save_path, 'wb') as f:
                        f.write(decompressed_data)
                else:
                    unzip_bytes_to_folder(decompressed_data, save_path)
                self.logger.info(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                self.print(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                break
            elif 'exit' in data:
                print(f"{len(file_data) / file_size * 100:.2f}%")
                break
            else:
                self.print(f"Unexpected data : {data}")

        socket_data['running_dict']["keep_alive_var"].set()

    @export(mod_name=Name, name="send_file_to_peer", test=False)
    async def send_file_to_peer(self, filepath, host, port):
        if isinstance(port, str):
            try:
                port = int(port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{port} is not an int or not cast to int")
        # Überprüfen, ob die Datei existiert
        if not os.path.exists(filepath):
            self.logger.error(f"Datei {filepath} nicht gefunden.")
            return False

        if '.' in filepath.split('/')[-1]:
            with open(filepath, 'rb') as f:
                to_send_data = gzip.compress(f.read())
        else:
            to_send_data = zip_folder_to_bytes(filepath)
        # Datei komprimieren
        compressed_data = gzip.compress(to_send_data)

        # Peer-to-Peer Socket erstellen und verbinden
        socket_data = await self.create_socket(name="sender", host=host, endpoint_port=port, type_id=SocketType.peer,
                                               return_full_object=True, keepalive_interval=1, start_keep_alive=False)

        if not socket_data.is_error():
            return socket_data
        socket_data = socket_data.get()

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

        send = socket_data['sender']

        # Komprimierte Daten senden
        try:
            # Größe der komprimierten Daten senden
            send({'data_size': len(compressed_data)})
            # Komprimierte Daten senden
            time.sleep(1.2)
            send(compressed_data)
            self.logger.info(f"Datei {filepath} erfolgreich gesendet.")
            self.print(f"Datei {filepath} erfolgreich gesendet.")
            # peer_result = receiver_queue.get(timeout=60*10)
            # print(f"{peer_result}")
            send({'exit': True})
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Datei: {e}")
            self.print(f"Fehler beim Senden der Datei: {e}")
            return False
        finally:
            socket_data['running_dict']["keep_alive_var"].set()

    @export(mod_name=Name, name="receive_and_decompress_file", test=False)
    async def receive_and_decompress_file_peer(self, save_path, listening_port, sender_ip='0.0.0.0'):
        # Empfangs-Socket erstellen
        if isinstance(listening_port, str):
            try:
                listening_port = int(listening_port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{listening_port} is not an int or not cast to int")

        socket_data = await self.create_socket(name="receiver", host=sender_ip, port=listening_port,
                                               type_id=SocketType.peer,
                                               return_full_object=True, max_connections=1)

        if not socket_data.is_error():
            return socket_data
        socket_data = socket_data.get()

        receiver_queue: queue.Queue = socket_data['receiver_queue']

        file_data = b''
        while True:
            # Auf Daten warten
            data = receiver_queue.get()
            if 'data_size' in data:
                file_size = data['data_size']
                self.logger.info(f"Erwartete Dateigröße: {file_size} Bytes")
                self.print(f"Erwartete Dateigröße: {file_size} Bytes")
            elif 'bytes' in data:

                file_data += data['bytes']
                # Daten dekomprimieren
                decompressed_data = gzip.decompress(file_data)
                # Datei speichern
                if '.' in save_path.split('/')[-1]:
                    with open(save_path, 'wb') as f:
                        f.write(decompressed_data)
                else:
                    unzip_bytes_to_folder(decompressed_data, save_path)
                self.logger.info(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                self.print(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                break
            elif 'exit' in data:
                break
            else:
                self.print(f"Unexpected data : {data}")

        socket_data['keepalive_var'][0] = False
