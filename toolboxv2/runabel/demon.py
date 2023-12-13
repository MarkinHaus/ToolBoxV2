import json
import time
from threading import Thread

from toolboxv2 import App, AppArgs, tbef
from toolboxv2.mods.SocketManager import SocketType

NAME = 'demon'


def run(app: App, args: AppArgs, programmabel_interface=False, as_server=True):
    """
    The Demon runner is responsible for running a lightweight toolbox instance in the background
    The name of the demon instance is also the communication bridge.

    workflow :

    run the demon as a .py script the demon will then kill the Terminal interface and runs in the background
    the demon can then be used to start other toolbox runnabel processes like the cli thru a nothe Terminal by simply
    naming the new instance as the demon. This new generated instance the shadow demon is then used to control the demon.

    crate the Demon

        $ ToolBoxV2 -m demon -n main # use the same name default is main

    creating the shadow demon

        same device

            $ ToolBoxV2 -m AnyMode[(default), cli, api]

            # to stop the demon
            $ ToolBoxV2 -m demon --kill

        remote

            $ ToolBoxV2 -m AnyMode[(default), cli, api] -n (full-name) --remote
                                                        optional --remote-direct-key [key] --host [host] --port [port]

    """

    # Start a New Demon

    status = 'unknown'

    client = app.run_any('SocketManager', 'create_socket',
                         name="demon", host="localhost" if args.host == '0.0.0.0' else args.host, port=62436,
                         type_id=SocketType.client,
                         max_connections=-1,
                         endpoint_port=None,
                         return_full_object=True)
    sender = None
    receiver_queue = None

    as_client = True

    if client is None:
        as_client = False

    if as_client:
        as_client = client.get('connection_error') == 0

    if as_client:
        status = 'client'
        sender = client.get('sender')
        receiver_queue = client.get('receiver_queue')

    if not as_client and as_server:
        status = 'server'
        server_controler = app.run_any('SocketManager', 'tbSocketController',
                                       name="demon", host=args.host, port=62436)
        if programmabel_interface:
            return 0, server_controler["get_status"], server_controler["stop_server"]

        def helper():
            t0 = time.perf_counter()
            while time.perf_counter() < t0 + 9999:
                time.sleep(2)
                for status_info in server_controler["get_status"]():
                    if status_info == "KEEPALIVE":
                        t0 = time.perf_counter()
                    print(f"Server status :", status_info)
                    if status_info == "Server closed":
                        break
        t_1 = Thread(target=helper)
        t_1.start()
        gc = app.run_any(tbef.CLI_FUNCTIONS.GET_CHARACTER)
        for data in gc:
            if data.word == "EXIT":
                server_controler["stop_server"]()
            if data.char == "x":
                server_controler["stop_server"]()
            print(data.char, data.word)
        t_1.join()

    if status != 'client':
        app.logger.info(f"closing demon {app.id}'{status}'")
        return -1, status, status

    if programmabel_interface:
        return 1, sender, receiver_queue

    alive = True

    while alive:
        user_input = input("input dict from :")
        if user_input == "exit":
            user_input = '{"exit": True}'
            alive = False
        sender(eval(user_input))

        if receiver_queue.not_empty:
            print(receiver_queue.get())

    # {
    #     'socket': socket,
    #     'receiver_socket': r_socket,
    #     'host': host,
    #     'port': port,
    #     'p2p-port': endpoint_port,
    #     'sender': send,
    #     'receiver_queue': receiver_queue,
    #     'connection_error': connection_error
    # }
