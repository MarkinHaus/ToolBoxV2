import json
import logging
import os.path
from importlib import import_module

from toolboxv2 import MainTool, FileHandler, App, Style
from toolboxv2.utils.toolbox import get_app
from fastapi import WebSocket, HTTPException


def valid_id(ws_id, id_v):
    print(f"USer ID {ws_id}, {id_v}")
    if not ws_id.startswith(id_v):
        raise HTTPException(status_code=403, detail="Access forbidden invalid id")

    return ws_id


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "WebSocketManager"
        self.logger: logging.Logger or None = app.logger if app else None
        if app is None:
            app = get_app()

        self.app = app
        self.color = "BLUE"
        self.active_connections: dict = {}
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
                    ],
            "name": "WebSocketManager",
            "Version": self.show_version,
            "connect": self.connect,
            "disconnect": self.disconnect,
            "send_message": self.send_message,
            "list": self.list_instances,
            "get": self.get_instances,
        }
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
        pass

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

    async def connect(self, websocket: WebSocket, websocket_id):
        websocket_id_sto = valid_id(websocket_id, self.app_id)
        if websocket_id_sto in self.active_connections.keys():
            print(f"Active connection - added nums {len(self.active_connections[websocket_id_sto])}")
            await self.send_message(f"New connection : {websocket_id}", websocket, websocket_id)
            self.active_connections[websocket_id_sto].append(websocket)
        else:
            self.active_connections[websocket_id_sto] = [websocket]
        await websocket.accept()

    async def disconnect(self, websocket: WebSocket, websocket_id):
        websocket_id_sto = valid_id(websocket_id, self.app_id)
        await self.send_message(f"Closing connection : {websocket_id}", websocket, websocket_id)
        self.active_connections[websocket_id_sto].remove(websocket)
        if len(self.active_connections[websocket_id_sto]) == 0:
            del self.active_connections[websocket_id_sto]
        await websocket.close()

    async def send_message(self, message: str, websocket: WebSocket or None, websocket_id):
        websocket_id_sto = valid_id(websocket_id, self.app_id)
        for connection in self.active_connections[websocket_id_sto]:
            if connection != websocket:
                await connection.send_text(message)

    @staticmethod
    def construct_render(content: str, element_id: str, externals: list or None = None,
                         placeholder_content: str or None = None, from_file=False):

        if externals is None:
            externals = []

        if placeholder_content is None:
            placeholder_content = "<h1>Loading...</h1>"

        if from_file:
            if os.path.exists(content):
                with open(content, 'r') as f:
                    content = f.read()

        render_data = {
            "render": {
                "content": content,
                "place": '#' + element_id,
                "id": element_id,
                "externals": externals,
                "placeholderContent": placeholder_content
            }
        }

        return json.dumps(render_data)

    async def initialize_Simple(self, websocket_id, websocket_, index=0):
        print("Initializing")
        websocket_id_sto = valid_id(websocket_id, self.app_id)
        if websocket_id_sto not in self.active_connections.keys():
            return "No websocket connection"
        print("Getting websocket")
        websocket = self.active_connections[websocket_id_sto][index]
        print('#'*20,websocket_ == websocket)
        print("websocket is %s" % websocket)
        content = """<div class="main-content frosted-glass">
    <h2>Hello and welcome to Simple. Simple supports you in your digital work.The System will adjust itself according to your specific requirements and over time will become more koplex and effective.Let the emergence begin</h2>
</div>"""
        c = self.construct_render(content=content, element_id='chat')
        print("content is %s" % c)
        res = await websocket_.send_text(c)
        print("response:", res)
