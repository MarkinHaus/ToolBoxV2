"""
The SocketManager Supports 2 types of connections
1. Client Server
2. Peer to Peer

"""
import gzip
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
from zipfile import ZipInfo, ZIP_LZMA
import uuid

from tqdm import tqdm

from toolboxv2 import MainTool, FileHandler, App, Style, get_app
import requests

import socket
import threading
import queue
import asyncio

version = "0.1.8"
Name = "SocketManager"

export = get_app("SocketManager.Export").tb


def zip_folder_to_bytes(folder_path):
    bytes_buffer = BytesIO()
    with zipfile.ZipFile(bytes_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
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
        self.running = False
        self.version = version
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
        self.local_ip = None
        self.public_ip = None
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)
        self.sockets = {}

    def on_start(self):
        self.logger.info(f"Starting SocketManager")
        # ~ self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing SocketManager")
        for socket_name, socket_data in self.sockets.items():
            self.print(f"consing Socket : {socket_name}")
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
            # '['running_dict']["keep_alive_var"] ': keep_alive_var,
            socket_data['running_dict']["keep_alive_var"] = False
            try:
                socket_data['sender']({'exit': True})
            except:
                pass
        # ~ self.save_file_handler()

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    @export(mod_name="SocketManager", version=version, samples=create_socket_samples, test=False)
    def create_socket(self, name: str = 'local-host', host: str = '0.0.0.0', port: int or None = None,
                      type_id: SocketType = SocketType.client,
                      max_connections=-1, endpoint_port=None,
                      return_full_object=False, keepalive_interval=6, test_override=False, package_size=1024,
                      start_keep_alive=True):

        if 'test' in self.app.id and not test_override:
            return "No api in test mode allowed"

        if endpoint_port is None and port is None:
            port = 62435

        if port is None:
            port = endpoint_port - 1

        if endpoint_port is None:
            endpoint_port = port + 1

        if endpoint_port == port:
            endpoint_port += 1

        if not isinstance(type_id, SocketType):
            return

        # setup sockets
        type_id = type_id.name

        r_socket = None
        connection_error = 0
        if self.local_ip is None:
            self.local_ip = get_local_ip()

        self.print(f"Device IP : {self.local_ip}")
        if type_id == SocketType.server.name:
            # create sever
            self.logger.debug(f"Starting:{name} server on port {port} with host {host}")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:
                sock.bind((host, port))
                sock.listen(max_connections)
                self.print(f"Server:{name} online at {host}:{port}")
            except Exception as e:
                connection_error = -1
                self.print(Style.RED(f"Server:{name} error at {host}:{port} {e}"))

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

            except Exception:
                connection_error = -1

        else:
            self.print(f"Invalid SocketType {type_id}:{name}")
            raise ValueError(f"Invalid SocketType {type_id}:{name}")

        # start queues sender, receiver, acceptor
        receiver_queue = queue.Queue()

        running_dict = {
            "server_receiver": True,
            "receive": {
                "main": True,
            },
            "keep_alive_var": True
        }

        client_sockets = {}

        # server receiver

        def server_receiver(sock_):
            connctions = 0
            while running_dict["server_receiver"]:
                try:
                    client_socket, endpoint = sock_.accept()
                except OSError:
                    running_dict["server_receiver"] = False
                    break
                connctions += 1
                self.print(f"Server Receiver:{name} new connection:{connctions}:{max_connections} {endpoint=}")
                receiver_queue.put({'data': (client_socket, endpoint), 'identifier': "new_con"})
                client_sockets[endpoint[0]+str(endpoint[1])] = client_socket
                if max_connections != -1:
                    if connctions >= max_connections:
                        running_dict["server_receiver"] = False

        def send(msg: bytes or dict, address=None):
            t0 = time.perf_counter()

            if type_id == SocketType.client.name:
                to = (host, port)
            elif address is not None:
                to = address
            else:
                to = (host, endpoint_port)

            # Prüfen, ob die Nachricht ein Dictionary ist und Bytes direkt unterstützen
            if isinstance(msg, bytes):
                sender_bytes = b'b' + msg  # Präfix für Bytes
                msg_json = 'sending bytes'
            elif isinstance(msg, dict):
                if 'exit' in msg:
                    sender_bytes = b'e'  # Präfix für "exit"
                    msg_json = 'exit'
                    running_dict["receive"]["main"] = False
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
                self.print(Style.GREY(f"Sending Data: {msg_json} {to}"))

            def send_(chunk):
                try:
                    if type_id == SocketType.client.name:
                        sock.sendall(chunk)
                    elif address is not None and type_id == SocketType.server.name:
                        _sock = client_sockets.get(address[0] + str(address[1]), sock)
                        _sock.sendto(chunk, address)
                    elif address is not None:
                        sock.sendto(chunk, address)
                    else:
                        sock.sendto(chunk, (host, endpoint_port))
                except Exception as e:
                    self.logger.error(f"Error sending data: {e}")

            if sender_bytes == b'k':
                send_(sender_bytes)
                return
            if sender_bytes == b'e':
                send_(sender_bytes)
                sock.close()
                return

            total_steps = len(sender_bytes) // package_size
            if len(sender_bytes) % package_size != 0:
                total_steps += 1  # Einen zusätzlichen Schritt hinzufügen, falls ein Rest existiert

            # tqdm Fortschrittsanzeige initialisieren
            with tqdm(total=total_steps, unit='chunk', desc='Sending data') as pbar:
                for i in range(0, len(sender_bytes), package_size):
                    chunk_ = sender_bytes[i:i + package_size]
                    send_(chunk_)
                    pbar.update(1)
                    time.sleep(1 / 10 ** 18)

            if len(sender_bytes) < package_size:
                send_(b' ' * (len(sender_bytes) - package_size))
            if len(sender_bytes) % package_size != 0:
                pass
            if type_id == SocketType.peer.name:
                send_(b'E' * 6)
            else:
                send_(b'E' * 1024)
            print(" ", end='\r')
            self.logger.info(f"{name} :S Parsed Time ; {time.perf_counter() - t0:.2f}")

        def receive(r_socket_, identifier="main"):
            data_type = None
            data_buffer = b''
            max_size = -1
            running_dict["receive"][identifier] = True
            while running_dict["receive"][identifier]:
                # t0 = time.perf_counter()
                try:
                    if type_id == SocketType.client.name:
                        chunk, add = r_socket_.recvfrom(1024)

                        if not chunk:
                            break

                    else:
                        chunk = r_socket_.recv(1024)

                        if not chunk:
                            continue
                except ConnectionResetError and ConnectionAbortedError and Exception:
                    print(f"Cloning Receiver {identifier}")
                    running_dict["receive"][identifier] = False
                    break

                if chunk == b'k':
                    # Behandlung von Byte-Daten
                    self.logger.info(f"{name} -- received keepalive signal--")
                    continue

                if not data_type:
                    data_type = chunk[:1]  # Erstes Byte ist der Datentyp
                    chunk = chunk[1:]  # Rest der Daten
                    self.print(f"Register date type : {data_type}")

                if max_size > -1 and len(data_buffer) > 0 and data_type == b'b':
                    print(
                        f"don {chunk[0] == b'E'[0] and chunk[-1] == b'E'[0]} {(len(data_buffer) / max_size) * 100:.2f}% total byts: {len(data_buffer)} von {max_size}",
                        end='\r')
                if data_type == b'e':
                    running_dict["receive"][identifier] = False
                    self.logger.info(f"{name} -- received exit signal --")
                    self.sockets[name]['running_dict']["keep_alive_var"]  = False
                elif chunk[0] == b'E'[0] and chunk[-1] == b'E'[0] and len(data_buffer) > 0:
                    max_size = -1
                    # Letzter Teil des Datensatzes
                    if data_type == b'b':
                        # Behandlung von Byte-Daten
                        receiver_queue.put({'bytes': data_buffer, 'identifier': identifier})
                        self.logger.info(f"{name} -- received bytes --")
                    elif data_type == b'j':
                        # Behandlung von JSON-Daten
                        try:
                            msg = json.loads(data_buffer)
                            msg['identifier'] = identifier
                            receiver_queue.put(msg)
                            self.logger.info(f"{name} -- received JSON -- {msg['identifier']} {len(msg.keys())}")
                            if 'data_size' in msg.keys():
                                max_size = msg['data_size']

                                self.logger.info(f"Erwartete Bytes: {max_size}")
                                self.print(f"Erwartete Bytes: {max_size}")
                        except json.JSONDecodeError and UnicodeDecodeError as e:
                            self.logger.error(f"JSON decode error: {e}")
                    else:
                        self.logger.error("Unbekannter Datentyp")
                        self.print(f"Received unknown data type: {data_type}")
                    # Zurücksetzen für den nächsten Datensatz
                    data_buffer = b''
                    data_type = None
                else:
                    # print(b' ' in chunk, b' '[0], chunk)
                    if b' ' in chunk and chunk[-1] == b' '[0] and type_id == SocketType.client.name:
                        chunk = chunk.replace(b' ', b'')
                    data_buffer += chunk

                # self.print(
                #     f"{name} :R Parsed Time ; {time.perf_counter() - t0:.2f} port :{endpoint_port if type_id == SocketType.peer.name else port}")

            self.print(f"{name} :closing connection to {host}")
            r_socket_.close()
            if type_id == SocketType.peer.name:
                sock.close()

        s_thread = None

        if connection_error == 0:
            if type_id == SocketType.server.name:
                s_thread = threading.Thread(target=server_receiver, args=(sock,), daemon=True)
                s_thread.start()
            elif connection_error == 0:
                s_thread = threading.Thread(target=receive, args=(r_socket,), daemon=True)
                s_thread.start()
            else:
                self.print(f"No receiver connected {name}:{type_id}")

        keep_alive_thread = None
        to_receive = None
        threeds = []

        if type_id == SocketType.peer.name:

            def keep_alive():
                i = 0
                while running_dict["keep_alive_var"]:
                    time.sleep(keepalive_interval)
                    try:
                        send({'keepalive': True}, (host, endpoint_port))
                    except Exception as e:
                        self.print(f"Exiting keep alive {e}")
                        break
                    i += 1
                self.print("Closing KeepAlive")
                send({"exit": True})

            keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
            if start_keep_alive:
                keep_alive_thread.start()

        elif type_id == SocketType.server.name:

            def to_receive(client, identifier='main'):
                t = threading.Thread(target=receive, args=(client, identifier,), daemon=True)
                t.start()
                threeds.append(t)

        elif type_id == SocketType.client.name:
            time.sleep(2)

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
            'running_dict': running_dict,
            'client_sockets_dict': client_sockets,
            'client_to_receiver_thread': to_receive,
            'client_receiver_threads': threeds,
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
    def run_as_single_communication_server(self, name: str = 'local-host', host: str = '0.0.0.0', port: int = 62435,
                                           test_override=False):

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
    def send_file_to_sever(self, filepath, host, port):
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
        socket_data = self.create_socket(name="sender", host=host, port=port, type_id=SocketType.client,
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

        send = socket_data['sender']

        # Komprimierte Daten senden
        try:
            # Größe der komprimierten Daten senden
            send({'data_size': len(compressed_data)})
            # Komprimierte Daten senden
            time.sleep(2)
            send(compressed_data)
            self.logger.info(f"Datei {filepath} erfolgreich gesendet.")
            self.print(f"Datei {filepath} erfolgreich gesendet.")
            send({'exit': True})
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Datei: {e}")
            self.print(f"Fehler beim Senden der Datei: {e}")
            return False
        finally:
            socket_data['running_dict']["keep_alive_var"] = False

    @export(mod_name=Name, name="receive_and_decompress_file_as_server", test=False)
    def receive_and_decompress_file_from_client(self, save_path, listening_port):
        # Empfangs-Socket erstellen
        if isinstance(listening_port, str):
            try:
                listening_port = int(listening_port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{listening_port} is not an int or not cast to int")

        socket_data = self.create_socket(name="receiver", host='0.0.0.0', port=listening_port,
                                         type_id=SocketType.server,
                                         return_full_object=True, max_connections=1)
        receiver_queue = socket_data['receiver_queue']
        to_receiver = socket_data['client_to_receiver_thread']
        client, address = receiver_queue.get(block=True)
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
                if len(file_data) < file_size:
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
                break
            else:
                self.print(f"Unexpected data : {data}")

        socket_data['running_dict']["keep_alive_var"]  = False

    @export(mod_name=Name, name="send_file_to_peer", test=False)
    def send_file_to_peer(self, filepath, host, port):
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
        socket_data = self.create_socket(name="sender", host=host, endpoint_port=port, type_id=SocketType.peer,
                                         return_full_object=True, keepalive_interval=1, start_keep_alive=False)

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
            socket_data['running_dict']["keep_alive_var"]  = False

    @export(mod_name=Name, name="receive_and_decompress_file", test=False)
    def receive_and_decompress_file_peer(self, save_path, listening_port, sender_ip='0.0.0.0'):
        # Empfangs-Socket erstellen
        if isinstance(listening_port, str):
            try:
                listening_port = int(listening_port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{listening_port} is not an int or not cast to int")

        socket_data = self.create_socket(name="receiver", host=sender_ip, port=listening_port,
                                         type_id=SocketType.peer,
                                         return_full_object=True, max_connections=1)
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
