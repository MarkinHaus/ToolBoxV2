import asyncio
import logging
import math
import queue
import threading
import time

import os.path
from typing import List
import websockets
from websockets.sync.client import connect

import json

from toolboxv2 import MainTool, FileHandler, Style
from toolboxv2.utils.toolbox import get_app, ApiOb
from fastapi import WebSocket, HTTPException


async def valid_id(ws_id, id_v, websocket=None):
    if not ws_id.startswith(id_v):
        if websocket is not None:
            await websocket.close()
        raise HTTPException(status_code=403, detail="Access forbidden invalid id")

    return ws_id


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "WebSocketManager"
        self.logger: logging.Logger or None = app.logger if app else None
        if app is None:
            app = get_app()

        self.app_ = app
        self.color = "BLUE"
        self.active_connections: dict = {}
        self.active_connections_client: dict = {}
        self.app_id = get_app().id
        self.keys = {
            "tools": "v-tools~~~"
        }
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["connect", "connect to a socket async (Server side)"],
                    ["disconnect", "disconnect a socket async (Server side)"],
                    ["send_message", "send_message to a socket group"],
                    ["list", "list all instances"],
                    ["srqw", "Gent an WebSocket with url and ws_id", math.inf, 'srqw_wrapper'],
                    ["construct_render", "construct_render"],
                    ],
            "name": "WebSocketManager",
            "Version": self.show_version,
            "connect": self.connect,
            "disconnect": self.disconnect,
            "send_message": self.send_message,
            "list": self.list_instances,
            "get": self.get_instances,
            "srqw": self.srqw_wrapper,
            "construct_render": self.construct_render,
        }

        self.validated_instances = {

        }
        self.vtID = None
        FileHandler.__init__(self, "WebSocketManager.config", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting WebSocketManager")
        self.load_file_handler()
        pass

    def on_exit(self):
        self.logger.info(f"Closing WebSocketManager")
        self.save_file_handler()
        for key in list(self.active_connections_client.keys()):
            self.close_websocket(key)

    def get_vt(self, uid):
        if self.vtID is not None:
            return self.vtID(uid)

        cloudM = self.app_.get_mod("cloudM")

        self.vtID = cloudM.get_vt_id
        return self.vtID(uid)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def get_instances(self, name):
        if name not in self.active_connections.keys():
            self.print(Style.RED("Pleas Create an instance before calling it!"))
            return None
        return self.active_connections[name]

    def list_instances(self):
        for name, instance in self.active_connections.items():
            self.print(f"{name}: {instance.name}")

    def srqw_wrapper(self, command):

        s, r = self.get_sender_receiver_que_ws(command[0], command[1])

        return s, r

    def get_sender_receiver_que_ws(self, url, websocket_id):

        self.print(Style.WHITE("Starting WebSocket Builder"))

        send_queue = queue.Queue()
        recv_queue = queue.Queue()
        loop = asyncio.new_event_loop()

        async def send(ws):
            t0 = time.time()
            running = True
            while running:
                msg = await loop.run_in_executor(None, send_queue.get)
                msg_json = msg
                if isinstance(msg, dict):
                    msg_json = json.dumps(msg)
                if isinstance(msg, list):
                    msg_json = str(msg)
                self.print(Style.GREY("Sending Data"))
                if msg_json == "exit":
                    running = False
                await ws.send(msg_json)
                self.print(Style.GREY("-- Sendet --"))

                self.print(f"S Parsed Time ; {t0 - time.time()}")
                if t0 - time.time() > (60 * 60) * 1:
                    ws.close()

            print("SENDER received exit stop running")

        async def receive(ws):
            t0 = time.time()
            running = True
            while running:
                msg_json = await ws.recv()
                self.print(Style.GREY("-- received --"))
                print(msg_json)
                if msg_json == "exit":
                    running = False
                msg = json.loads(msg_json)
                recv_queue.put(msg)

                self.print(f"R Parsed Time ; {t0 - time.time()}")
                if t0 - time.time() > (60 * 60) * 1:
                    ws.close()

            print("receiver received exit call")

        async def websocket_handler():

            with self.create_websocket(websocket_id, url) as websocket:
                send_task = asyncio.create_task(send(websocket))
                recv_task = asyncio.create_task(receive(websocket))
                try:
                    await asyncio.gather(send_task, recv_task)
                except Exception as e:
                    self.logger.error(f"Error in Client WS : {e}")
                except websockets.exceptions.ConnectionClosedOK:
                    return True
                finally:
                    self.close_websocket(websocket_id)

            return True

        def websocket_thread():
            asyncio.set_event_loop(loop)
            # websocket_handler()
            # loop.run_forever()
            # loop.run_in_executor(None, websocket_handler)
            loop.run_until_complete(websocket_handler())

        ws_thread = threading.Thread(target=websocket_thread)
        ws_thread.start()

        return send_queue, recv_queue

    def create_websocket(self, websocket_id, url='wss://0.0.0.0:5000/ws'):  # wss:
        uri = f"{url}/{websocket_id}"
        self.logger.info(f"Crating websocket to {url}")
        websocket = connect(uri)
        if websocket:
            self.print(f"Connection to {url} established")
            self.active_connections_client[websocket_id] = websocket
        return websocket

    def close_websocket(self, websocket_id):
        self.print(f"close_websocket called")
        if websocket_id not in self.active_connections_client.keys():
            self.print(f"websocket not found")
        self.active_connections_client[websocket_id].close()
        del self.active_connections_client[websocket_id]

    async def connect(self, websocket: WebSocket, websocket_id):
        websocket_id_sto = await valid_id(websocket_id, self.app_id, websocket)

        data = self.app_.run_any("cloudM", "validate_ws_id", [websocket_id])
        valid, key = False, ''
        if isinstance(data, list) or isinstance(data, tuple):
            if len(data) == 2:
                valid, key = data
            else:
                self.logger.error(f"list error connect {data}")
                return False
        else:
            self.logger.error(f"isinstance error connect {data}, {type(data)}")
            return False

        if valid:
            self.validated_instances[websocket_id_sto] = key

        if websocket_id_sto in self.active_connections.keys():
            print(f"Active connection - added nums {len(self.active_connections[websocket_id_sto])}")
            await self.send_message(json.dumps({"res": f"New connection : {websocket_id}"}), websocket, websocket_id)
            self.active_connections[websocket_id_sto].append(websocket)
        else:
            self.active_connections[websocket_id_sto] = [websocket]
        await websocket.accept()
        return True

    async def disconnect(self, websocket: WebSocket, websocket_id):
        websocket_id_sto = await valid_id(websocket_id, self.app_id)
        await self.send_message(json.dumps({"res": f"Closing connection : {websocket_id}"}), websocket, websocket_id)
        self.active_connections[websocket_id_sto].remove(websocket)
        if len(self.active_connections[websocket_id_sto]) == 0:
            del self.active_connections[websocket_id_sto]
        await websocket.close()

    async def send_message(self, message: str, websocket: WebSocket or None, websocket_id):
        websocket_id_sto = await valid_id(websocket_id, self.app_id)
        for connection in self.active_connections[websocket_id_sto]:
            if connection != websocket:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    self.logger.error(f"{Style.YELLOW('Error')} Connection in {websocket_id} lost to {connection}")
                    self.logger.error(str(e))
                    self.print(f"{Style.YELLOW('Error')} Connection in {websocket_id} lost to {connection}")
                    self.active_connections[websocket_id_sto].remove(connection)

    async def manage_data_flow(self, websocket, websocket_id, data):
        self.logger.info(f"Managing data flow: data {data}")
        websocket_id_sto = await valid_id(websocket_id, self.app_id)

        if websocket_id_sto not in self.active_connections.keys():
            return '{"res": "No websocket connection pleas Log in"}'

        if websocket_id_sto not in self.validated_instances.keys():
            content = self.construct_render(content="""<p id="infoText" color: style="color:var(--error-color);">Pleas Log in
            </p>
            """, element_id="infoText")
            return content

        si_id = self.validated_instances[websocket_id_sto]

        data_type = "Noice"
        try:
            data = json.loads(data)
            data_type = "dict"
        except ValueError as e:
            self.logger.error(Style.YELLOW(f"ValueError json.loads data : {e}"))
            if websocket_id_sto in data:
                data_type = "str"

        self.logger.info(f"type: {data_type}:{type(data)}")

        if data_type == "Noice":
            return

        if data_type == "dict" and isinstance(data, dict):
            keys = list(data.keys())
            if "ServerAction" in keys:
                action = data["ServerAction"]

                if action == "getsMSG":
                    # Sendeng system MSG message
                    systemMSG_content = self.construct_render(content="./app/systemMSG/text.html",
                                                              element_id="extra",
                                                              externals=["/app/systemMSG/speech_balloon.js"],
                                                              from_file=True)

                    await websocket.send_text(systemMSG_content)
                if action == "getTextWidget":  # WigetNav
                    # Sendeng system MSG message
                    widgetText_content = self.construct_render(content="./app/1/textWidet/text.html",
                                                               element_id="widgetText",
                                                               externals=["/app/1/textWidet/testWiget.js"],
                                                               from_file=True)

                    await websocket.send_text(widgetText_content)
                if action == "getPathWidget":
                    widgetPath_content = self.construct_render(content="./app/1/PathWidet/text.html",
                                                               element_id="widgetPath",
                                                               externals=["/app/1/PathWidet/pathWiget.js"],
                                                               from_file=True)

                    await websocket.send_text(widgetPath_content)
                if action == "getWidgetNave":
                    # Sendeng system MSG message
                    widgetText_content = self.construct_render(content="./app/1/WigetNav/navDrow.html",
                                                               element_id="controls",
                                                               externals=["/app/1/WigetNav/navDrow.js"],
                                                               from_file=True)

                    await websocket.send_text(widgetText_content)
                if action == "getDrag":
                    drag_content = self.construct_render(content="./app/Drag/drag.html",
                                                         element_id="DragControls",
                                                         externals=["/app/Drag/drag.js"],
                                                         from_file=True)
                    await websocket.send_text(drag_content)
                if action == "getControls":
                    controller_content = self.construct_render(content="",
                                                               element_id="editorWidget",
                                                               externals=["/app/1/Controler/controller.js"])

                    await websocket.send_text(controller_content)
                if action == "serviceWorker":
                    sw_content = self.construct_render(content="",
                                                       element_id="control1",
                                                       externals=["/app/index.js", "/app/sw.js"])
                    await websocket.send_text(sw_content)
                if action == "logOut":
                    user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])
                    if user_instance is None or not user_instance:
                        return '{"res": "No User Instance Found"}'

                    if data['data']['token'] == "**SelfAuth**":
                        data['data']['token'] = user_instance['token']

                    api_data = ApiOb()
                    api_data.data = data['data']['data']
                    api_data.token = data['data']['token']
                    command = [api_data, data['command'].split('|')]

                    self.app_.run_any("cloudM", "api_log_out_user", command)
                    websocket_id_sto = await valid_id(websocket_id, self.app_id)
                    for websocket_ in self.active_connections[websocket_id_sto]:
                        if websocket == websocket_:
                            continue
                        await self.disconnect(websocket_, websocket_id)

                    if len(self.active_connections[websocket_id_sto]) > 1:
                        await self.send_message(json.dumps({'exit': 'exit'}), websocket, websocket_id)

                    home_content = self.construct_render(content="",
                                                         element_id="main",

                                                         externals=["/app/scripts/go_home.js"])

                    await websocket.send_text(home_content)
                if action == "getModListAll":
                    return json.dumps({'modlistA': self.app_.get_all_mods()})
                if action == "getModListInstalled":
                    user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])
                    if user_instance is None or not user_instance:
                        self.logger.info("No valid user instance")
                        return '{"res": "No Mods Installed"}'

                    return json.dumps({'modlistI': user_instance['save']['mods']})
                if action == "getModData":
                    mod_name = data["mod-name"]
                    try:
                        mod = self.app_.get_mod(mod_name)
                        return {"settings": {'mod-description': mod.description}}
                    except ValueError:
                        content = self.construct_render(
                            content=f"""<p id="infoText" color: style="color:var(--error-color);">Mod {mod_name} not found
                        </p>
                        """, element_id="infoText")
                        return content
                if action == "installMod":
                    user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])
                    if user_instance is None or not user_instance:
                        self.logger.info("No valid user instance")
                        return '{"res": "No User Instance Found Pleas Log in"}'

                    if data["name"] not in user_instance['save']['mods']:
                        self.logger.info(f"Appending mod {data['name']}")
                        user_instance['save']['mods'].append(data["name"])

                    self.app_.new_ac_mod("cloudM")
                    self.app_.AC_MOD.hydrate_instance(user_instance)
                    self.print("Sending webInstaller")
                    installer_content = user_instance['live'][data["name"]].webInstall(user_instance,
                                                                                       self.construct_render)
                    self.app_.new_ac_mod("cloudM")
                    self.app_.AC_MOD.save_user_instances(user_instance)
                    await websocket.send_text(installer_content)
                if action == "addConfig":
                    user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])
                    if data["name"] in user_instance['live'].keys():
                        user_instance['live'][data["name"]].add_str_to_config([data["key"], data["value"]])
                    else:
                        await websocket.send_text('{"res": "Mod nod installed or available"}')
                if action == "runMod":
                    user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])

                    self.print(f"{user_instance}, {data}")
                    if user_instance is None or not user_instance:
                        return '{"res": "No User Instance Found pleas log in"}'

                    if data['data']['token'] == "**SelfAuth**":
                        data['data']['token'] = user_instance['token']

                    api_data = ApiOb()
                    api_data.data = data['data']['data']
                    api_data.token = data['data']['token']
                    command = [api_data, data['command'].split('|')]

                    token_data = self.app_.run_any('cloudM', "validate_jwt", command)

                    if not isinstance(token_data, dict):
                        return json.dumps({'res': 'u ar using an invalid token pleas log in again'})

                    if token_data["uid"] != user_instance['save']['uid']:
                        self.logger.critical(
                            f"{Style.RED(f'''User {user_instance['save']['username']} {Style.CYAN('Accessed')} : {Style.Bold(token_data['username'])} token both log aut.''')}")
                        self.app_.run_any('cloudM', "close_user_instance", token_data["uid"])
                        self.app_.run_any('cloudM', "close_user_instance", user_instance['save']['uid'])
                        return json.dumps({'res': "The server registered: you are"
                                                  " trying to register with an not fitting token "})

                    if data['name'] not in user_instance['save']['mods']:
                        user_instance['save']['mods'].append(data['name'])

                    if data['name'] not in user_instance['live'].keys():
                        self.logger.info(f"'Crating live module:{data['name']}'")
                        self.app_.new_ac_mod("cloudM")
                        self.app_.AC_MOD.hydrate_instance(user_instance)
                        self.app_.new_ac_mod("cloudM")
                        self.app_.AC_MOD.save_user_instances(user_instance)

                    try:
                        self.app_.new_ac_mod("VirtualizationTool")
                        if self.app_.run_function('set-ac', user_instance['live']['v-' + data['name']]):
                            res = self.app_.run_function('api_' + data['function'], command)
                        else:
                            res = "Mod Not Found 404"
                    except Exception as e:
                        res = "Mod Error " + str(e)

                    if type(res) == str:
                        if (res.startswith('{') or res.startswith('[')) or res.startswith('"[') or res.startswith('"{') \
                            or res.startswith('\"[') or res.startswith('\"{') or res.startswith(
                            'b"[') or res.startswith('b"{'): \
                            res = eval(res)
                    if not isinstance(res, dict):
                        res = {"res": res, data['name']: True}
                    await websocket.send_text(json.dumps(res))

            if "ValidateSelf" in keys:
                user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])
                if user_instance is None or not user_instance:
                    self.logger.info("No valid user instance")
                    return json.dumps({"res": "No User Instance Found Pleas Log in", "valid": False})
                return json.dumps({"res": "User Instance is valid", "valid": True})
            if "ChairData" in keys:
                user_instance = self.app_.run_any("cloudM", "wsGetI", [si_id])
                if user_instance is None or not user_instance:
                    self.logger.info("No valid user instance")
                    return json.dumps({"res": "No User Instance Found Pleas Log in", "valid": False})
                if len(self.active_connections[websocket_id_sto]) < 1:
                    return json.dumps({"res": "No other connections found", "valid": True})
                await self.send_message(json.dumps(data['data']), websocket, websocket_id)
                return json.dumps({"res": "Data Send", "valid": True})

        if data_type == "str":
            await self.send_message(data, websocket, websocket_id)

    def construct_render(self, content: str, element_id: str, externals: List[str] or None = None,
                         placeholder_content: str or None = None, from_file=False):

        if externals is None:
            externals = []

        if placeholder_content is None:
            placeholder_content = "<h1>Loading...</h1>"

        if from_file:
            if os.path.exists(content):
                with open(content, 'r') as f:
                    self.logger.info(f"File read from {content}")
                    content = f.read()
            else:
                self.print(f"{Style.RED('Could not find file ')}to create renderer {from_file}")

        render_data = {
            "render": {
                "content": content,
                "place": '#' + element_id,
                "id": element_id,
                "externals": externals,
                "placeholderContent": placeholder_content
            }
        }

        self.logger.info(f"render content :  {render_data}")

        return json.dumps(render_data)
