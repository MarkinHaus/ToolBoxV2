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

from tqdm import tqdm

from toolboxv2 import MainTool, FileHandler, App, Style, get_app

import socket
import threading
import queue
import asyncio

version = "0.1.6"
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
                          'keepalive_interval': 1000}, ]


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
        self.local_ip = get_local_ip()
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
            # 'keepalive_var': keep_alive_var,
            socket_data['keepalive_var'][0] = False
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
                      return_full_object=False, keepalive_interval=6, test_override=False):

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
        self.print(f"Device IP : {self.local_ip}")
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
                sock.sendto(b'k', (host, endpoint_port))

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
            # Prüfen, ob die Nachricht ein Dictionary ist und Bytes direkt unterstützen
            if isinstance(msg, bytes):
                sender_bytes = b'b' + msg  # Präfix für Bytes
                msg_json = 'sending bytes'
            elif isinstance(msg, dict):
                if 'exit' in msg:
                    sender_bytes = b'e'  # Präfix für "exit"
                    msg_json = 'exit'
                elif 'keepalive' in msg:
                    sender_bytes = b'k'  # Präfix für "exit"
                    msg_json = 'keepalive'
                else:
                    msg_json = json.dumps(msg)
                    sender_bytes = b'j' + msg_json.encode('utf-8')  # Präfix für JSON
            else:
                self.print(Style.YELLOW(f"Unsupported message type: {type(msg)}"))
                return

            self.print(Style.GREY(f"Sending Data: {msg_json}"))

            def send_(chunk):
                print("Sending chunk", chunk)
                try:
                    if type_id == SocketType.client.name:
                        sock.sendall(chunk)
                    elif address is not None:
                        sock.sendto(chunk, address)
                    else:
                        sock.sendto(chunk, (host, endpoint_port))
                except Exception as e:
                    self.logger.error(f"Error sending data: {e}")

            print("all data - sender_bytes -", sender_bytes)
            if type_id == SocketType.client.name:
                to = (host, port)
            elif address is not None:
                to = address
            else:
                to = (host, endpoint_port)
            self.print(Style.GREY(f"-- Sent to : {to} --"))
            total_steps = len(sender_bytes) // 1024
            if len(sender_bytes) % 1024 != 0:
                total_steps += 1  # Einen zusätzlichen Schritt hinzufügen, falls ein Rest existiert

            # tqdm Fortschrittsanzeige initialisieren
            with tqdm(total=total_steps, unit='chunk', desc='Sending data') as pbar:
                for i in range(0, len(sender_bytes), 1024):
                    chunk_ = sender_bytes[i:i + 1024]
                    send_(chunk_)
                    pbar.update(1)
            if len(sender_bytes) % 1024 != 0:
                pass
            send_(b'E' * 1024)
            self.print(f"{name} :S Parsed Time ; {time.perf_counter() - t0:.2f}")

        def receive(r_socket_, identifier="main"):
            running = True
            data_type = None
            data_buffer = b''
            while running:
                t0 = time.perf_counter()

                if type_id == SocketType.client.name:
                    chunk, add = r_socket_.recvfrom(1024)
                else:
                    chunk = r_socket_.recv(1024)

                if not chunk:
                    break  # Verbindung wurde geschlossen

                if not data_type:
                    data_type = chunk[:1]  # Erstes Byte ist der Datentyp
                    chunk = chunk[1:]  # Rest der Daten
                    self.print(f"Register date type : {data_type}")

                print(data_type, len(chunk), chunk)

                if data_type == b'k':
                    data_buffer = b''
                    data_type = None
                elif chunk[0] == b'E' and chunk[-1] == b'E' and len(data_buffer) > 0:
                    print("all data restructured", data_buffer)
                    # Letzter Teil des Datensatzes
                    if data_type == b'e':
                        running = False
                        self.logger.info(f"{name} -- received exit signal --")
                        self.sockets[name]['keepalive_var'][0] = False
                    elif data_type == b'b':
                        # Behandlung von Byte-Daten
                        receiver_queue.put({'bytes': data_buffer, 'identifier': identifier})
                        self.logger.info(f"{name} -- received bytes --")
                    elif data_type == b'j':
                        # Behandlung von JSON-Daten
                        try:
                            msg = json.loads(data_buffer)
                            msg['identifier'] = identifier
                            receiver_queue.put(msg)
                            self.logger.info(f"{name} -- received JSON -- {msg}")
                        except json.JSONDecodeError and UnicodeDecodeError as e:
                            self.logger.error(f"JSON decode error: {e}")
                    else:
                        self.logger.error("Unbekannter Datentyp")
                        self.print(f"Received unknown data type: {data_type}")
                    # Zurücksetzen für den nächsten Datensatz
                    data_buffer = b''
                    data_type = None
                else:
                    data_buffer += chunk

                self.print(
                    f"{name} :R Parsed Time ; {time.perf_counter() - t0:.2f} port :{endpoint_port if type_id == SocketType.peer.name else port}")

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
        to_receive = None
        threeds = []
        keep_alive_var = [True]

        if type_id == SocketType.peer.name:

            def keep_alive():
                i = 0
                while keep_alive_var[0]:
                    time.sleep(keepalive_interval)
                    try:
                        send({'keepalive': True}, (host, endpoint_port))
                    except Exception as e:
                        self.print(f"Exiting keep alive {e}")
                        break
                    i += 1
                self.print("Closing KeepAlive")

            keep_alive_thread = threading.Thread(target=keep_alive)
            keep_alive_thread.start()

        elif type_id == SocketType.server.name:

            threeds = []

            def to_receive(client, identifier='main'):
                t = threading.Thread(target=receive, args=(client, identifier,))
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
            'keepalive_var': keep_alive_var,
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
            return False

        # Datei komprimieren
        with open(filepath, 'rb') as f:
            compressed_data = gzip.compress(f.read())

        # Peer-to-Peer Socket erstellen und verbinden
        socket_data = self.create_socket(name="sender", host=host, port=port, type_id=SocketType.server,
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
            socket_data['keepalive_var'][0] = False

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

        # Datei komprimieren
        with open(filepath, 'rb') as f:
            compressed_data = gzip.compress(f.read())

        # Peer-to-Peer Socket erstellen und verbinden
        socket_data = self.create_socket(name="sender", host=host, endpoint_port=port, type_id=SocketType.peer,
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
        receiver_queue: queue.Queue = socket_data['receiver_queue']

        # Komprimierte Daten senden
        try:
            # Größe der komprimierten Daten senden
            send({'data_size': len(compressed_data)})
            # Komprimierte Daten senden
            time.sleep(2)
            send(compressed_data)
            self.logger.info(f"Datei {filepath} erfolgreich gesendet.")
            self.print(f"Datei {filepath} erfolgreich gesendet.")
            peer_result = receiver_queue.get(timeout=60*10)
            print(f"{peer_result}")
            send({'exit': True})
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Datei: {e}")
            self.print(f"Fehler beim Senden der Datei: {e}")
            return False
        finally:
            socket_data['keepalive_var'][0] = False

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
        file_size = 0
        while True:
            # Auf Daten warten
            data = receiver_queue.get()
            if 'data_size' in data:
                file_size = data['data_size']
                self.logger.info(f"Erwartete Dateigröße: {file_size} Bytes")
                self.print(f"Erwartete Dateigröße: {file_size} Bytes")
            elif 'bytes' in data:
                print("dasdadad", data)
                file_data += data['bytes']
                # Daten dekomprimieren
                if len(file_data) > 0:
                    print(f"{file_size / len(file_data) * 100:.2f}% of 100% | {file_size}, {len(file_data)}")
                if len(file_data) != file_size:
                    continue
                decompressed_data = gzip.decompress(file_data)
                # Datei speichern
                with open(save_path, 'wb') as f:
                    f.write(decompressed_data)
                self.logger.info(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                self.print(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                break
            elif 'exit' in data:
                break
            else:
                self.print(f"Unexpected data : {data}")

        socket_data['keepalive_var'][0] = False

    @export(mod_name=Name, name="receive_and_decompress_file", test=False)
    def receive_and_decompress_file_peer(self, save_path, listening_port):
        # Empfangs-Socket erstellen
        if isinstance(listening_port, str):
            try:
                listening_port = int(listening_port)
            except:
                return self.return_result(exec_code=-1, data_info=f"{listening_port} is not an int or not cast to int")

        socket_data = self.create_socket(name="receiver", host='0.0.0.0', port=listening_port,
                                         type_id=SocketType.peer,
                                         return_full_object=True, max_connections=1)
        receiver_queue: queue.Queue = socket_data['receiver_queue']

        file_data = b''
        file_size = 0
        while True:
            # Auf Daten warten
            data = receiver_queue.get()
            print("receiver_queue received: ", data)
            if 'data_size' in data:
                file_size = data['data_size']
                self.logger.info(f"Erwartete Dateigröße: {file_size} Bytes")
                self.print(f"Erwartete Dateigröße: {file_size} Bytes")
            elif 'bytes' in data:

                file_data += data['bytes']
                # Daten dekomprimieren
                if len(file_data) > 0:
                    print(f"{file_size / len(file_data) * 100:.2f}% of 100% | {file_size}, {len(file_data)}")
                if len(file_data) != file_size:
                    continue
                decompressed_data = gzip.decompress(file_data)
                # Datei speichern
                with open(save_path, 'wb') as f:
                    f.write(decompressed_data)
                self.logger.info(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                self.print(f"Datei erfolgreich empfangen und gespeichert in {save_path}")
                break
            elif 'exit' in data:
                break
            else:
                self.print(f"Unexpected data : {data}")

        socket_data['keepalive_var'][0] = False
