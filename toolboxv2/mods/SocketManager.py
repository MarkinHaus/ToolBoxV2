"""
The SocketManager Supports 2 types of connections
1. Client Server
2. Peer to Peer

"""
import json
import random
import time
from dataclasses import dataclass
import logging
from enum import Enum

from toolboxv2 import MainTool, FileHandler, App, Style, get_app

import socket
import threading
import queue
import asyncio

version = "0.0.2"
Name = "SocketManager"

export = get_app("SocketManager.Export").tb


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
                          'keepalive_interval': 1000},]


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.running = False
        self.version = "0.0.2"
        self.name = "SocketManager"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"], ["create_socket", "crate a socket", -1],
                    ["tbSocketController", "run demon", -1]],
            "name": "SocketManager",
            "create_socket": self.create_socket,
            "tbSocketController": self.run_as_single_communication_server,
            "Version": self.show_version,
        }

        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)
        self.sockets = {}

    def on_start(self):
        self.logger.info(f"Starting SocketManager")
        # ~ self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing SocketManager")
        # ~ self.save_file_handler()

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    @export(mod_name="SocketManager", version=version, samples=create_socket_samples, test=False)
    def create_socket(self, name: str = 'local-host', host: str = '0.0.0.0', port: int = 62435,
                      type_id: SocketType = SocketType.client,
                      max_connections=-1, endpoint_port=None,
                      return_full_object=False, keepalive_interval=1000, test_override=False):

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"

        if endpoint_port is None:
            endpoint_port = port

        if not isinstance(type_id, SocketType):
            return

        # setup sockets
        type_id = type_id.name

        r_socket = None
        connection_error = 0

        if type_id == SocketType.server.name:
            # create sever
            self.logger.debug(f"Starting:{name} server on port {port} with host {host}")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:
                sock.bind((host, port))
                sock.listen(max_connections)
            except Exception:
                connection_error = -1

            self.print(f"Server:{name} online at {host}:{port}")

        elif type_id == SocketType.client.name:
            # create client
            self.logger.debug(f"Starting:{name} client on port {port} with host {host}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            time.sleep(random.choice(range(1, 100)) // 100)
            connection_error = sock.connect_ex((host, port))
            if connection_error != 0:
                sock.close()
                self.print(f"Client:{name} connection_error:{connection_error}")
            else:
                self.print(f"Client:{name} online at {host}:{port}")
            # sock.sendall(bytes(self.app.id, 'utf-8'))
            r_socket = sock

        elif type_id == SocketType.peer.name:
            # create peer

            if host == "localhost" or host == "127.0.0.1":
                self.print("LocalHost Peer2Peer is not supported use server client architecture")
                return

            self.logger.debug(f"Starting:{name} peer on port {port} with host {host}")
            r_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            try:
                r_socket.bind(('0.0.0.0', endpoint_port))
                self.print(f"Peer:{name} listening on {endpoint_port}")

                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(('0.0.0.0', port))
                sock.sendto(b'0', (host, endpoint_port))
            except Exception:
                connection_error = -1
            self.print(f"Peer:{name} sending default to at {host}:{endpoint_port}")

        else:
            self.print(f"Invalid SocketType {type_id}:{name}")
            raise ValueError(f"Invalid SocketType {type_id}:{name}")

        # start queues sender, receiver, acceptor
        receiver_queue = queue.Queue()

        # server receiver

        def server_receiver(sock_):
            running = True
            connctions = 0
            while running:
                client_socket, endpoint = sock_.accept()
                connctions += 1
                self.print(f"Server Receiver:{name} new connection:{connctions}:{max_connections} {endpoint=}")
                receiver_queue.put((client_socket, endpoint))
                if connctions >= max_connections:
                    running = False

        def send(msg, address=None):
            t0 = time.perf_counter()

            if not isinstance(msg, dict):
                self.print(Style.YELLOW("Only dicts supported with keys msg and address"))
                self.logger.warning("Try send none dic object thrue socket")
                return

            if not len(msg.keys()):
                self.logger.warning("no msg to send")
                return

            if msg.get('exit'):
                msg_json = 'exit'
            else:
                msg_json = json.dumps(msg)

            self.print(Style.GREY(f"Sending Data {msg_json}"))

            if not msg_json:
                self.logger.warning("no msg_json to send")
                return

            try:
                if type_id == SocketType.client.name:
                    sock.sendall(msg_json.encode('utf-8'))
                elif address is not None:
                    sock.sendto(msg_json.encode('utf-8'), address)
                else:
                    sock.sendto(msg_json.encode('utf-8'), (host, endpoint_port))

                self.print(Style.GREY("-- Sendet --"))
            except Exception:
                pass

            self.print(f"{name} :S Parsed Time ; {time.perf_counter() - t0:.2f}")

        def receive(r_socket_):
            t0 = time.perf_counter()
            running = True
            while running:
                msg_json = r_socket_.recv(1024).decode()
                if not msg_json: break
                self.print(Style.GREY(f"{name} -- received -- '{msg_json}'"))
                if msg_json == "exit":
                    running = False
                else:
                    if msg_json:
                        msg = json.loads(msg_json)
                        receiver_queue.put(msg)

                self.print(f"{name} :R Parsed Time ; {time.perf_counter() - t0:.2f}")
                t0 = time.perf_counter()

            self.print(f"{name} :closing connection to {host}")
            r_socket_.close()
            if type_id == SocketType.peer.name:
                sock.close()

        s_thread = None

        if connection_error == 0:
            if type_id == SocketType.server.name:
                s_thread = threading.Thread(target=server_receiver, args=(sock,))
                s_thread.start()
            elif connection_error == 0:
                s_thread = threading.Thread(target=receive, args=(r_socket,))
                s_thread.start()
            else:
                self.print(f"No receiver connected {name}:{type_id}")

        keep_alive_thread = None

        if type_id == SocketType.peer.name:

            def keep_alive():
                i = 0
                while True:
                    time.sleep(keepalive_interval / 1000)
                    try:
                        send({'keep_alive': i}, (host, endpoint_port))
                    except Exception as e:
                        self.print("Exiting keep alive")
                        break
                    i += 1

            keep_alive_thread = threading.Thread(target=keep_alive)
            keep_alive_thread.start()

        self.sockets[name] = {
            'socket': socket,
            'receiver_socket': r_socket,
            'host': host,
            'port': port,
            'p2p-port': endpoint_port,
            'sender': send,
            'receiver_queue': receiver_queue,
            'connection_error': connection_error,
            'receiver_thread': s_thread,
            'keepalive_thread': keep_alive_thread,
        }

        if return_full_object:
            return self.sockets[name]

        return send, receiver_queue

        # sender queue

    @export(mod_name=Name, name="run_as_ip_echo_server_a", test=False)
    def run_as_ip_echo_server_a(self, name: str = 'local-host', host: str = '0.0.0.0', port: int = 62435,
                                max_connections: int = -1, test_override=False):

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"
        send, receiver_queue = self.create_socket(name, host, port, SocketType.server, max_connections=max_connections)

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
    def run_as_single_communication_server(self, name: str = 'local-host', host: str = '0.0.0.0', port: int = 62435, test_override=False):

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"

        send, receiver_queue = self.create_socket(name, host, port, SocketType.server, max_connections=1)
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
            thread = threading.Thread(target=server_thread, args=(client, address))
            thread.start()

        threading.Thread(target=helper).start()

        def stop_server():
            running[0] = False
            status_queue.put("Server stopping")

        def get_status():
            while status_queue.not_empty:
                yield status_queue.get()

        return {"stop_server": stop_server, "get_status": get_status}
