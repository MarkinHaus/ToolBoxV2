from toolboxv2 import App, AppArgs
from toolboxv2.mods.SocketManager import SocketType

NAME = 'demon'


def run(app: App, args: AppArgs):
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
                         name="demon", host="localhost" if args.host == '0.0.0.0' else args.host, port=62436, type_id=SocketType.client,
                         max_connections=-1,
                         endpoint_port=None,
                         return_full_object=True)
    sender = None
    receiver_queue = None
    if client.get('connection_error') == 0:
        status = 'client'
        sender = client.get('sender')
        receiver_queue = client.get('receiver_queue')
    else:
        status = 'server'
        app.run_any('SocketManager', 'tbSocketController',
                    name="demon", host=args.host, port=62436)

    if status != 'client':
        app.logger.info(f"closing demon {app.id}'{status}'")
        return

    alive = True

    while alive:
        print("module_name: str, function_name: str, command=None")
        user_input = input("input dict from :")
        if user_input == "exit":
            user_input = {'exit': True}
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
