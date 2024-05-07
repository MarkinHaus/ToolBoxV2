import asyncio
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Tuple, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor

from toolboxv2 import get_app, Result, Spinner, MainTool, get_logger
from toolboxv2.mods.SocketManager import get_local_ip
from toolboxv2.utils import Singleton
from toolboxv2.utils.brodcast.client import start_client
from toolboxv2.utils.brodcast.server import make_known
from toolboxv2.utils.daemon import DaemonUtil
from toolboxv2.utils.proxy import ProxyUtil

Name = "EventManager"
version = "0.0.1"
export = get_app(from_=f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
test_only = export(mod_name=Name, test_only=True, version=version, state=True)
to_api = export(mod_name=Name, api=True, version=version)


class Scope(Enum):
    instance: str = "instance"
    local: str = "local"
    local_network: str = "local_network"
    global_network: str = "global_network"


class ExecIn(Enum):
    remote: str = "remote"
    local: str = "local"


class SourceTypes(Enum):
    F: str = "Fuction"
    R: str = "RunAny"
    S: str = "String"
    P: str = "Fuction withe payload"
    # D: str = "File"
    # E: str = "Event"


@dataclass
class EventID:
    source: str
    path: str
    ID: str
    payload: Any
    rpayload: Any

    def __str__(self):
        return f"ID:{self.ID:-^20} [{self.source:}] - ({self.path:})"

    def __init__(self, source, path, ID, payload=None, rpayload=None):
        self.source = source
        self.path = path
        self.ID = ID
        self.payload = payload
        self.rpayload = rpayload

    def set_payload(self, payload):
        if not isinstance(payload, str):
            try:
                payload = str(payload)
                self.payloadr = payload
            except ValueError:
                # raise ValueError("Payload must be a string or converted")
                self.payloadr = 'ValueError("Payload must be a string or converted")'
        else:
            self.rpayload = payload

    @classmethod
    def crate_empty(cls):
        return cls(source="", path="", ID=str(uuid.uuid4()))

    @classmethod
    def crate_name_as_id(cls, name: str):
        return cls(source="", path="", ID=name)

    @classmethod
    def crate_with_source(cls, source: str):
        return cls(source=source, path="E", ID=str(uuid.uuid4()))

    @classmethod
    def crate(cls, source: str, ID, path="E", payload=None):
        return cls(source=source, path=path, ID=ID, payload=payload)

    def add_source(self, source):
        self.source += ':' + source
        return self

    def add_path(self, path):
        self.path += ':' + path
        return self

    def get_path(self):
        return self.path.split(':')

    def get_source(self):
        return self.source.split(':')


@dataclass
class Event:
    name: str
    source: Union[Callable, str, Tuple[str, str]]
    source_types: SourceTypes = SourceTypes.F
    scope: Scope = Scope.local
    exec_in: ExecIn = ExecIn.local
    event_id: EventID = field(default_factory=EventID.crate_empty())
    threaded: bool = False
    args: Optional[Tuple] = None
    kwargs_: Optional[Dict] = None

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self.event_id.ID == other.event_id.ID

    def __hash__(self):
        return hash(self.event_id.ID)

    def to_dict(self):
        if self.source_types.name == SourceTypes.F.name:
            raise ValueError("Source types is not supported")

        return asdict(self)


@dataclass
class Rout:
    _from: str
    _to: str

    _from_port: int
    _from_host: str

    _to_port: int
    _to_host: str

    routing_function: Callable

    @property
    def to_host(self):
        return self._to_host

    @property
    def to_port(self):
        return self._to_port

    async def put_data(self, event_id_data: Dict[str, str]):
        event_id: EventID = EventID(**event_id_data)
        return await self.routing_function(event_id)

    def close(self):
        """ Close """

    # def send(self, e: EventID):
    #     return self.put_data(asdict(e))


class DaemonRout(DaemonUtil):
    def __init__(self, rout: Rout, t=True, name="daemonRoute", on_r=None, on_c=None, unix_socket=False):
        host, port = rout.to_host, rout.to_port
        super().__init__(class_instance=rout, host=host, port=port, t=t, app=get_app(from_=f"{Name}.Rout.Daemon"),
                         peer=False, name=name, on_register=on_r, on_client_exit=on_c, unix_socket=unix_socket)


class ProxyRout(ProxyUtil):

    @classmethod
    def toProxy(cls, rout, timeout=10, name="proxyRout", unix_socket=False):
        host, port = rout.to_host, rout.to_port
        return cls(
            class_instance=rout,
            host=host,
            port=port, timeout=timeout,
            app=get_app(from_=f"{Name}.Rout.Proxy"),
            remote_functions=["put_data"], peer=False, name=name, unix_socket=unix_socket
        )

    # def put_data(self, *args, **kwargs):
    #     return self.class_instance.put_data(*args, **kwargs)


class EventManagerClass:
    events: set[Event] = set()
    source_id: str
    _name: str
    _identification: str

    routes_client: Dict[str, ProxyRout] = {}
    routers_servers: Dict[str, DaemonRout] = {}

    receiver_que: queue.Queue
    response_que: queue.Queue

    def add_c_route(self, name, route: ProxyRout):
        self.routes_client[name] = route

    async def receive_all_client_data(self):

        close_connections = []
        add_ev = []
        for name, client in self.routes_client.items():
            if client.client is None or not client.client.get('alive', False):
                close_connections.append(name)
                continue
            data = client.r

            if isinstance(data, str) and data == "No data":
                continue
            elif isinstance(data, EventID) and len(data.get_source()) != 0:
                await self.trigger_event(data)
            elif isinstance(data, EventID) and len(data.get_source()) == 0:
                print(f"Event returned {data.payload}")
                self.response_que.put(data)
            elif isinstance(data,
                            dict) and 'source' in data and 'path' in data and 'ID' in data and 'identifier' in data:
                del data['identifier']
                ev_id = EventID(**data)
                await self.trigger_event(ev_id)
            elif isinstance(data, Event):
                print("Event:", str(data.event_id), data.name)
                add_ev.append(data)
            elif isinstance(data, Result):
                data.print()
            else:
                print(f"Unknown Data {data}")

        for ev in add_ev:
            await self.register_event(ev)

        for client_name in close_connections:
            print(f"Client {client_name} closing connection")
            self.remove_c_route(client_name)

    def remove_c_route(self, name):
        self.routes_client[name].close()
        del self.routes_client[name]

    def crate_rout(self, source, addr=None):
        if addr is None:
            addr = ('0.0.0.0', 6588)
        host, port = addr
        if isinstance(port, str):
            port = int(port)
        return Rout(
            _from=self.source_id,
            _to=source,
            _from_port=int(os.getenv("TOOLBOXV2_BASE_PORT", 6588)),
            _from_host=os.getenv("TOOLBOXV2_BASE_HOST"),
            _to_port=port,
            _to_host=host,
            routing_function=self.routing_function_router,
        )

    def __init__(self, source_id, _identification="Pn"):
        self.bo = False
        self.running = False
        self.source_id = source_id
        self.receiver_que = queue.Queue()
        self.response_que = queue.Queue()
        self._identification = _identification
        self._name = self._identification + '-' + str(uuid.uuid4()).split('-')[1]
        self.routes = {}
        self.logger = get_logger()

    @property
    def identification(self) -> str:
        return self._identification

    @identification.setter
    def identification(self, _identification: str):
        self.stop()
        self._identification = _identification
        self._name = self._identification + '-' + str(uuid.uuid4()).split('-')[1]
        if _identification == "P0":
            self.add_server_route(_identification, ('0.0.0.0', 6568))
        if _identification == "P0|S0":
            self.add_server_route(_identification, ('0.0.0.0', 6567))
        self.start()
        self.reconnect("ALL")

    def open_connection_server(self, port):
        self.add_server_route(self._identification, ('0.0.0.0', port))

    def start(self):
        self.running = True
        asyncio.to_thread(self.receiver)
        # threading.Thread(target=self.receiver, daemon=True).start()

    def make_event_from_fuction(self, fuction, name, *args, source_types=SourceTypes.F,
                                scope=Scope.local,
                                exec_in=ExecIn.local,
                                threaded=False, **kwargs):

        return Event(source=fuction,
                     name=name,
                     event_id=EventID.crate_with_source(self.source_id), args=args,
                     kwargs_=kwargs,
                     source_types=source_types,
                     scope=scope,
                     exec_in=exec_in,
                     threaded=threaded,
                     )

    async def add_client_route(self, source_id, addr):
        if source_id in self.routes_client:
            if self.routes_client[source_id].client is None or not self.routes_client[source_id].client.get('alive'):
                self.routes_client[source_id].reconnect()
                return True
            print("Already connected")
            return False
        try:
            pr = await ProxyRout.toProxy(rout=self.crate_rout(source_id, addr=addr), name=source_id)
            time.sleep(0.25)
            pr.client.get('sender')({"id": self._identification, "continue": False})
            time.sleep(0.25)
            self.add_c_route(source_id, pr)
            return True
        except Exception as e:
            print(f"Check the port {addr} Sever likely not Online : {e}")
            return False

    async def add_mini_client(self, name: str, addr: Tuple[str, int]):

        mini_proxy = await ProxyRout(class_instance=None, timeout=15, app=get_app(),
                                     remote_functions=[""], peer=False, name=name, do_connect=False)

        async def _(x):
            return await self.routers_servers[self._identification].send(x, addr)

        mini_proxy.put_data = _
        mini_proxy.connect = lambda *x, **_: None
        mini_proxy.reconnect = lambda *x, **_: None
        mini_proxy.close = lambda *x, **_: None
        mini_proxy.client = {'alive': True}
        mini_proxy.r = "No data"
        self.routes_client[name] = mini_proxy

    def on_register(self, id_, data):
        try:
            print("on_register", id_, "##", data)
            if "unknown" not in self.routes:
                self.routes["unknown"] = {}

            if id_ != "new_con" and 'id' in data:
                id_data = data.get('id')
                id_ = eval(id_)
                c_host, c_pot = id_
                print(f"Registering: new client {id_data} : {c_host, c_pot} | {data}")
                if id_data not in self.routes_client.keys():
                    self.add_mini_client(id_data, (c_host, c_pot))
                    self.routes[str((c_host, c_pot))] = id_data

            # print("self.routes:", self.routes)
        except Exception as e:
            print("Error in on_register", str(e))

    def on_client_exit(self, id_):

        if isinstance(id_, str):
            id_ = eval(id_)

        c_name = self.routes.get(id_)

        if c_name is None:
            return

        if c_name in self.routes_client:
            self.remove_c_route(c_name)
            print(f"Removed route to {c_name}")

    async def add_server_route(self, source_id, addr=None):
        if addr is None:
            addr = ('0.0.0.0', 6588)
        try:
            self.routers_servers[source_id] = await DaemonRout(rout=self.crate_rout(source_id, addr=addr),
                                                               name=source_id,
                                                               on_r=self.on_register)
        except Exception as e:
            print(f"Sever already Online : {e}")

    async def add_js_route(self, source_id="js:web"):
        await self.add_server_route(source_id, ("./web/scripts/tb_socket.sock", 0))

    async def register_event(self, event: Event):

        if event in self.events:
            return Result.default_user_error("Event registration failed Event already registered")

        print(f"Registration new Event : {event.name}, {str(event.event_id)}")
        self.events.add(event)

        if event.scope.name == Scope.instance.name:
            return

        if event.scope.name == Scope.local.name:
            if not self.bo and "P0" not in self.routes_client and os.getenv("TOOLBOXV2_BASE_HOST",
                                                                            "localhost") != "localhost":
                await self.add_client_route("P0", (os.getenv("TOOLBOXV2_BASE_HOST", "localhost"),
                                                   os.getenv("TOOLBOXV2_BASE_PORT", 6568)))
                self.bo = True
            return

        if event.scope.name == Scope.local_network.name:
            if self.identification == "P0" and not self.bo:
                t0 = threading.Thread(target=self.start_brodcast_router_local_network, daemon=True)
                t0.start()
            elif not self.bo and "P0" not in self.routes_client and os.getenv("TOOLBOXV2_BASE_HOST",
                                                                              "localhost") == "localhost":
                self.bo = True
                # self.add_server_route(self.identification, ("127.0.0.1", 44667))
                with Spinner(message="Sercheing for Rooter instance", count_down=True, time_in_s=6):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        t0 = executor.submit(make_known, self.identification)
                        try:
                            data = t0.result(timeout=6)
                        except TimeoutError:
                            print("No P0 found in network or on device")
                            return
                    print(f"Found P0 on {type(data)} {data.get('host')}")
                    await self.add_client_route("P0", (data.get("host"), os.getenv("TOOLBOXV2_BASE_PORT", 6568)))
            elif not self.bo and "P0" not in self.routes_client and os.getenv("TOOLBOXV2_BASE_HOST",
                                                                              "localhost") != "localhost":
                do = await self.add_client_route("P0", (
                    os.getenv("TOOLBOXV2_BASE_HOST", "localhost"), os.getenv("TOOLBOXV2_BASE_PORT", 6568)))
                self.bo = do
                if not do:
                    print("Connection failed")
                    os.environ["TOOLBOXV2_BASE_HOST"] = "localhost"

        if event.scope.name == Scope.global_network.name:
            await self.add_server_route(self.source_id, ('0.0.0.0', os.getenv("TOOLBOXV2_REMOTE_PORT", 6587)))

    async def connect_to_remote(self, host=os.getenv("TOOLBOXV2_REMOTE_IP"),
                                port=os.getenv("TOOLBOXV2_REMOTE_PORT", 6587)):
        await self.add_client_route("S0", (host, port))

    def start_brodcast_router_local_network(self):
        self.bo = True

        print("Starting brodcast router 0")
        router = start_client(get_local_ip())
        print("Starting brodcast router 1")
        # next(router)
        print("Starting brodcast router")
        while self.running:
            source_id, connection = next(router)
            print(f"Infos :{source_id}, connection :{connection}")
            self.routes[source_id] = connection[0]
            router.send(self.running)

        router.send(False)

    def _get_event_by_id_or_name(self, event_id: str or EventID):
        if isinstance(event_id, str):
            events = [e for e in self.events if e.name == event_id]
            if len(events) < 1:
                return Result.default_user_error("Event not registered")
            event = events[0]

        elif isinstance(event_id, EventID):
            events = [e for e in self.events if e.event_id.ID == event_id.ID]
            if len(events) < 1:
                events = [e for e in self.events if e.name == event_id.ID]
            if len(events) < 1:
                return Result.default_user_error("Event not registered")
            event = events[0]

        elif isinstance(event_id, Event):
            if event_id not in self.events:
                return Result.default_user_error("Event not registered")
            event = event_id

        else:
            event = Result.default_user_error("Event not registered")

        return event

    def remove_event(self, event: Event or EventID or str):

        event = self._get_event_by_id_or_name(event)
        if isinstance(event, Event):
            self.events.remove(event)
        else:
            return event

    async def _trigger_local(self, event_id: EventID):
        """
        Exec source based on

        source_types
            F -> call directly
            R -> use get_app(str(event_id)).run_any(*args, **kwargs)
            S -> evaluate string
        scope
            instance -> _trigger_local
            local -> if you ar proxy app run the event through get_app(str(event_id)).run_any(tbef.EventManager._trigger_local, args=args, kwargs=kwargs, get_result=True)
            local_network -> use proxy0 app to communicate withe Daemon0 then local
            global_network ->
        exec_in
        event_id
        threaded

                       """
        event = self._get_event_by_id_or_name(event_id)

        if isinstance(event, Result):
            event.print()
            if self.identification == "P0":
                return event
            print(f"Routing to P0 {self.events}")
            if self.source_id not in self.routes_client:
                # self.routers[self.source_id] = DaemonRout(rout=self.crate_rout(self.source_id))
                await self.add_client_route("P0", ('127.0.0.1', 6568))
            return await self.route_event_id(event_id)

        # if event.threaded:
        #    threading.Thread(target=self.runner, args=(event, event_id), daemon=True).start()
        #    return "Event running In Thread"
        # else:

        return self.runner(event, event_id)

    def runner(self, event, event_id):

        if event.source_types.name is SourceTypes.P.name:
            return event.source(*event.args, payload=event_id, **event.kwargs_)

        if event.source_types.name is SourceTypes.F.name:
            return event.source(*event.args, **event.kwargs_)

        if event.source_types.name is SourceTypes.R.name:
            return get_app(str(event_id)).run_any(mod_function_name=event.source, get_results=True, args_=event.args,
                                                  kwargs_=event.kwargs_)

        if event.source_types.name is SourceTypes.S.name:
            return eval(event.source, __locals={'app': get_app(str(event_id)), 'event': event, 'eventManagerC': self})

    async def routing_function_router(self, event_id: EventID):

        result = await self.trigger_event(event_id)

        if result is None:
            result = Result.default_user_error("Invalid Event ID")

        if isinstance(result, bytes):
            pass
        elif isinstance(result, dict):
            pass
        elif isinstance(result, Result):
            result.result.data_info = str(event_id)
        elif isinstance(result, EventID):
            result = Result.default_internal_error("Event not found", data=result)
        else:
            result = Result.ok(data=result, data_info="<automatic>", info=str(event_id.path))

        if isinstance(result, str):
            result = result.encode()

        return result

    async def trigger_evnet_by_name(self, name: str):
        await self.trigger_event(EventID.crate_name_as_id(name=name))

    async def trigger_event(self, event_id: EventID):

        print(f"event-id Ptah : {event_id.get_path()}")
        print(f"testing trigger_event for {event_id.get_source()} {event_id.get_source()[-1] == self.source_id} ")
        if event_id.get_source()[-1] == self.source_id:
            payload = await self._trigger_local(event_id)
            event_id.set_payload(payload)
            if len(event_id.path) > 1:
                event_id.source = ':'.join([e.split('-')[0] for e in event_id.get_path() if e != "E"])
                res = await self.route_event_id(event_id)
                if isinstance(res, Result):
                    res.print()
                else:
                    print(res)
            return payload
        return await self.route_event_id(event_id)

    async def route_event_id(self, event_id: EventID):

        print(f"testing route_event_id for {event_id.get_source()[-1]}")
        if event_id.get_source()[-1] == '*':  # self.identification == "P0" and
            responses = []
            event_id.source = ':'.join(event_id.get_source()[:-1])
            event_id.add_path(f"{self._name}({self.source_id})")
            data = asdict(event_id)
            for name, rout_ in self.routes_client.items():
                if name in event_id.path:
                    continue
                ret = await rout_.put_data(data)
                responses.append(ret)
            return responses
        route = self.routes_client.get(event_id.get_source()[-1])
        print("route:", route)
        if route is None:
            route = self.routes_client.get(event_id.get_path()[-1])
        if route is None:
            return event_id.add_path(f"404#{self.identification}")
        # time.sleep(0.25)
        event_id.source = ':'.join(event_id.get_source()[:-1])
        event_id.add_path(f"{self._name}({self.source_id})")
        return await route.put_data(asdict(event_id))

    async def receiver(self):

        t0 = time.time()

        while self.running:
            time.sleep(0.25)  # um z verhindern das, dass netzwerk weiter als 4 notes tief geht
            if not self.receiver_que.empty():
                event_id = self.receiver_que.get()
                print("Receiver Event", str(event_id))
                await self.trigger_event(event_id)

            if time.time() - t0 > 5:
                await self.receive_all_client_data()
                t0 = time.time()

    def info(self):
        return {"source": self.source_id, "known_routs:": self.routers_servers, "_router": self.routes_client,
                "events": self.events}

    def stop(self):
        self.running = False
        list(map(lambda x: x.disconnect(), self.routes_client.values()))
        list(map(lambda x: x.stop(), self.routers_servers.values()))

    def reconnect(self, name):
        if name is None:
            pass
        elif name in self.routes_client:
            self.routes_client[name].reconnect()
            return
        list(map(lambda x: x.reconnect(), self.routes_client.values()))

    async def verify(self, name):
        if name is None:
            pass
        elif name in self.routes_client:
            await self.routes_client[name].verify()
            return
        for x in self.routes_client.values():
            await x.verify()


@export(name=Name, mod_name=Name, version=version)
class Tools(MainTool, EventManagerClass):
    version = version

    def __init__(self, app=None):
        self.name = Name
        self.color = "BLINK"

        self.keys = {"mode": "db~mode~~:"}
        self.encoding = 'utf-8'
        _identification = "Pn"
        EventManagerClass.__init__(self, f"{self.spec}.{self.app.id}", _identification=_identification)
        MainTool.__init__(self,
                          load=self.startEventManager,
                          v=self.version,
                          name=self.name,
                          color=self.color,
                          on_exit=self.closeEventManager)

    @export(
        mod_name=Name,
        name="Version",
        version=version,
    )
    def get_version(self):
        return self.version

    # Exportieren der Scheduler-Instanz für die Nutzung in anderen Modulen
    @export(mod_name=Name, name='startEventManager', version=version)
    def startEventManager(self):
        if self.app.args_sto.background_application_runner:
            self.identification = 'P0'
        else:
            self.identification = self.app.id.split('-')[0]

    @export(mod_name=Name, name='closeEventManager', version=version)
    def closeEventManager(self):
        self.stop()
        return "closedEventManager"

    @export(mod_name=Name, name='start_web_events', version=version, test=False)
    async def startWebEvents(self):
        await self.add_js_route()

    @export(mod_name=Name, name='getEventManagerC', version=version, exit_f=True)
    def get_manager(self) -> EventManagerClass:
        return self
