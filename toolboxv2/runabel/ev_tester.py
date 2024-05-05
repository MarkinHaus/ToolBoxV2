import socket

from toolboxv2.mods.EventManager.module import EventManagerClass, Event, EventID, Scope

NAME = "evT"


#  for windows pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl
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
        return "localhost"


def run_(app, _):  # Event von Pn zu P0
    from toolboxv2 import tbef

    # app = get_app()
    # get_local_ip()
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.GETEVENTMANAGERC)

    def event_fuction(num=-1):
        print(f"event_fuction triggered num {num}")
        return f"EV{num}"

    if e_id := input('T:'):
        ev.identification = "Pn"
        ret = ev.trigger_event(EventID.crate("app.main-DESKTOP-CI57V1L", e_id))
        print("ret :", str(ret))
    else:

        ev.identification = "P0"
        event = ev.make_event_from_fuction(event_fuction, "event_fuction", num=0)
        ev_id = event.event_id
        event.scope = Scope.local
        print("event ID :", str(ev_id))
        ev.register_event(event)

    while True:
        pass


def run__(app, _):  # Event von Pn to Pn
    from toolboxv2 import tbef

    # app = get_app()
    # get_local_ip()
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.GETEVENTMANAGERC)

    def event_fuction(num=-1):
        print(f"event_fuction triggered num {num}")
        return f"EV{num}"

    if e_id := input('T:'):
        ev.identification = "Pn1"
        ev.add_client_route("Pn2", ('localhost', 6677))
        ret = ev.trigger_event(EventID.crate("app.main-DESKTOP-CI57V1L:Pn2", e_id))
        print("ret :", str(ret))
    else:
        ev.identification = "Pn2"
        ev.open_connection_server(6677)
        # ev.add_server_route("Pn2", ('0.0.0.0', 6677))
        event = ev.make_event_from_fuction(event_fuction, "event_fuction", num=0)
        ev_id = event.event_id
        event.scope = Scope.local
        print("event ID :", str(ev_id))
        ev.register_event(event)

    while True:
        pass


def run___(app, _):  # Event von Pn to Pn over P0
    from toolboxv2 import tbef

    # app = get_app()
    # get_local_ip()
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.GETEVENTMANAGERC)

    def event_fuction(num=-1):
        print(f"event_fuction triggered num {num}")
        return f"EV{num}"

    if 'y' in input('as p0:'):
        ev.identification = "P0"
    elif e_id := input('T:'):
        ev.identification = "Pn1"
        ev.add_client_route("P0", ("127.0.0.1", 6568))
        ret = ev.trigger_event(EventID.crate("app.main-DESKTOP-CI57V1L:Pn2:P0", e_id))
        print("ret :", str(ret))
    else:
        ev.identification = "Pn2"
        ev.add_client_route("P0", ("127.0.0.1", 6568))
        event = ev.make_event_from_fuction(event_fuction, "event_fuction", num=0)
        ev_id = event.event_id
        event.scope = Scope.local
        print("event ID :", str(ev_id))
        ev.register_event(event)

    while True:
        pass


def run(app, _):  # Event von Pn -> Sn2 über P0 wobei P0 und Pn same device und Sn2 nicht + ohne ip
    from toolboxv2 import tbef

    # app = get_app()
    # get_local_ip()
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.GETEVENTMANAGERC)

    def event_fuction(num=-1):
        print(f"event_fuction triggered num {num}")
        return f"EV{num}"

    if 'y' in input('as p0:'):
        ev.identification = "P0"
        ev.register_event(Event("broadcast_event", source=event_fuction, scope=Scope.local_network))
    elif e_id := input('T:'):
        ev.identification = "Pn"
        # Pn und P0 same diveice
        ev.add_client_route("P0", ("127.0.0.1", 6568))
        # Pn und P0 on different devices in same network
        # ev.register_event(Event("broadcast_event", source=event_fuction, scope=Scope.local_network))
        surce = "app.main-vanillam:Sn2:P0"
        surce = "app.main-vanillam:*:*"
        ret = ev.trigger_event(EventID.crate("app.main-vanillam:*:*", e_id))
        print("ret :", str(ret))
    else:
        ev.identification = "Sn2"
        ev.register_event(Event("broadcast_event", source=event_fuction, scope=Scope.local_network))
        event = ev.make_event_from_fuction(event_fuction, "event_fuction", num=0)
        ev_id = event.event_id
        event.scope = Scope.local
        print("event ID :", str(ev_id))
        ev.register_event(event)

    while True:
        pass
