import json
import os
import socket


def start_client(host_ip, port=44667):
    known = {}
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP

    # Enable port reusage, so we will be able to run multiple clients and servers on single (host, port).
    # Do not use socket.SO_REUSEADDR except you using linux(kernel<3.9): goto https://stackoverflow.com/questions/14388706/how-do-so-reuseaddr-and-so-reuseport-differ for more information.
    # For linux hosts all sockets that want to share the same address and port combination must belong to processes that share the same effective user ID!
    # So, on linux(kernel>=3.9) you have to run multiple servers and clients under one user to share the same (host, port).
    # Thanks to @stevenreddie
    client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Enable broadcasting mode
    client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    client.bind(("", port))
    alive = True
    while alive:
        # Thanks @seym45 for a fix
        data, addr = client.recvfrom(1024)
        print("received message: %s" % data)
        data = data.decode('utf')
        info_t, name = data.split(':')[0], data.split(':')[1]
        known[name] = addr
        if info_t == "A":
            client.sendto(json.dumps(known).encode('utf-8'), addr)
        if info_t == "R":
            client.sendto(json.dumps(
                {'host': host_ip, 'port': int(os.getenv("TOOLBOXV2_BASE_PORT"))
                 }).encode('utf-8'), addr)
        alive = yield name, addr
