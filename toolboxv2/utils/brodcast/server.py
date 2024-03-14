import json
import socket
import time


def make_known(name: str, get_flag: bytes = b"R", port=44667):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Enable port reusage so we will be able to run multiple clients and servers on single (host, port).
    # Do not use socket.SO_REUSEADDR except you using linux(kernel<3.9): goto https://stackoverflow.com/questions/14388706/how-do-so-reuseaddr-and-so-reuseport-differ for more information.
    # For linux hosts all sockets that want to share the same address and port combination must belong to processes that share the same effective user ID!
    # So, on linux(kernel>=3.9) you have to run multiple servers and clients under one user to share the same (host, port).
    # Thanks to @stevenreddie
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Enable broadcasting mode
    server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    message = get_flag + b":" + name.encode('utf-8')
    # Set a timeout so the socket does not block
    # indefinitely when trying to receive data.
    server.settimeout(0.2)
    # server.bind(("", port))
    # server.send(message)
    server.sendto(message, ("255.255.255.255", port))
    print("message sent!", flush=True)
    time.sleep(4)
    data = server.recv(1024)
    print(f"data received! {data.decode()}", flush=True)
    return json.loads(data.decode())
