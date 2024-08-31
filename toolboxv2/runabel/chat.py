import asyncio
import os
import queue
import random
import threading

from tqdm import tqdm

from toolboxv2 import App, AppArgs, Style, TBEF, get_app, Result
from toolboxv2.mods import SocketManager

try:
    from toolboxv2.mods.SocketManager import SocketType
    from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope, EventID

    Online = True
except ImportError as e:
    SocketType = None
    EventManagerClass, SourceTypes, Scope = None, None, None
    Online = False
    print(f"Chat runner is missing modules Pleas install {e}")
NAME = 'chat'


async def init(app: App, pbar):
    """Initial Modules"""
    if not app.mod_online("CloudM"):
        await app.init_module("CloudM")
    pbar.update()
    if not app.mod_online("SocketManager"):
        await app.init_module("SocketManager")
    pbar.update()
    if not app.mod_online("EventManager"):
        await app.init_module("EventManager")
    pbar.update()
    # try:
    #     if not app.mod_online("audio"):
    #         app.load_mod("audio")
    #     pbar.write("audio Online")
    # except ImportError and RuntimeError as e:
    #     app.print(Style.YELLOW(f"Error importing audio module no voice active"))
    #     pbar.write("audio Offline")
    pbar.update()


async def get_connection_point(payload):
    app = get_app("Event get_connection_point")
    payload_key = payload.payload['key']
    ev: EventManagerClass = app.get_mod("EventManager").get_manager()
    try:
        rute_data = list(ev.routes.keys())[list(ev.routes.values()).index(payload_key)]
        return rute_data
    except ValueError:
        return "Invalid payload"


def debug_receiver(q, prefix):
    while True:
        try:
            # Nachricht aus der Queue abrufen
            message = q.get(timeout=1)  # timeout in Sekunden, um nicht endlos zu blockieren
            if message is None:
                break
            if message == "[exit]":
                break
            # Nachricht mit Präfix in der Konsole ausgeben
            print(f"{prefix}: {message}")
        except queue.Empty:
            continue


async def run(app: App, _: AppArgs):
    if not Online:
        app.print(Style.RED("Chat runner is offline install modules SocketManager/EventManager"))
        return
    app.print("Starting P2P E2E messaging service")
    with tqdm(total=4, unit='modules', desc='Opening') as pbar:
        await init(app, pbar)

    ev: EventManagerClass = app.get_mod("EventManager").get_manager()
    sm: SocketManager = app.get_mod("SocketManager")

    if "core0" in app.id:

        # if ev.identification != "P0|S0":
        #     ev.identification = "P0|S0"
        #     await ev.identity_post_setter()
        service_event = ev.make_event_from_fuction(get_connection_point,
                                                   "get-connection-point",
                                                   source_types=SourceTypes.AP,
                                                   scope=Scope.global_network,
                                                   threaded=True)
        await ev.register_event(service_event)
        app.print("Service P2P Online")
        await app.run_runnable("cli")
        return
    else:
        source_id = "app.core0-simplecore.app"
        if _ := input(f"source ID: default {source_id} :"):
            source_id = _

    user_name = app.get_username(True)

    ev.identification += app.config_fh.generate_symmetric_key()[:6]

    await ev.identity_post_setter()

    await ev.connect_to_remote(host="localhost")
    await asyncio.sleep(1.57)

    res = await ev.trigger_event(EventID.crate(f"{source_id}:S0", "get-connection-point",
                                               payload={'key': ev.identification}))
    print(res, type(res.get()))
    if not res.is_error():
        if isinstance(res.get(), str):
            self_host, self_port = eval(res.get())
        else:
            self_host, self_port = res.get()
    else:
        return res

    app.print(f"ur {user_name} ontime_key is : {ev.identification}")
    input("Wait till both ar online")

    p_user_name = input("Username of Partner: ")
    res = await ev.trigger_event(EventID.crate(f"{source_id}:S0", "get-connection-point",
                                               payload={'key': input("onetime_key of Partner: ")}))
    print(res)
    if not res.is_error():
        if isinstance(res.get(), str):
            peer_host, peer_port = eval(res.get())
        else:
            peer_host, peer_port = res.get()
    else:
        return res

    sm.public_ip = self_host

    # connection_key = ""
    # one_time_user_name = app.config_fh.one_way_hash(user, connection_key, "P2P-E2E")
    # app.print(f"1 time user name is {one_time_user_name[:6]}")

    p_client = await app.a_run_any(TBEF.SOCKETMANAGER.CREATE_SOCKET,
                                   name='messaging-service',
                                   host=peer_host,
                                   port=peer_port,
                                   type_id=SocketType.peer,
                                   max_connections=-1,
                                   endpoint_port=self_port,
                                   return_full_object=False,
                                   keepalive_interval=6,
                                   test_override=False,
                                   package_size=1024,
                                   start_keep_alive=True,
                                   unix_file=False,
                                   do_async=False)

    await asyncio.sleep(10)

    if p_client is None:
        return p_client

    if not isinstance(p_client, Result):
        return p_client

    if p_client.is_error():
        return p_client

    custom_prefix = f"[{p_user_name}] "

    # Starten des Debug-Receivers in einem eigenen Thread
    receiver_thread = threading.Thread(target=debug_receiver, args=(p_client.get('receiver_queue'), custom_prefix))
    receiver_thread.start()

    sender = p_client.get('sender')
    while True:

        if u_imp := input(f"[{user_name}]:"):
            await sender(u_imp)
        elif u_imp == "[exit]":
            await sender(u_imp)
            break
        else:
            print("Invalid type '[exit]' to quit.")
    receiver_thread.join()

    await p_client.get('close')()

    # 'close': close_helper(),
    # 'max_connections': max_connections,
    # 'type_id': type_id,
    # 'do_async': do_async,
    # 'package_size': package_size,
    # 'host': host,
    # 'port': port,
    # 'p2p-port': endpoint_port,
    # 'sender': lambda msg, identifier="main": self.send(name, msg=msg, identifier=identifier),
    # 'connections': 0,
    # 'receiver_queue': receiver_queue,
    # 'a_receiver_queue': a_receiver_queue,
    # 'connection_error': connection_error,
    # 'running_dict': {
    #     "server_receiver": asyncio.Event(),
    #     "server_receiver_": None,
    #     "thread_receiver": None,
    #     "thread_receiver_": False,
    #     "receive": {
    #     },
    #     "keep_alive_var": asyncio.Event()
    # },
    # 'client_sockets_dict': {},
    # 'client_sockets_identifier': {},
    # 'client_to_receiver_thread': to_receive,
