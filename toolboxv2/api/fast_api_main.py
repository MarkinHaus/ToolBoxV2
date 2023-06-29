import json
import os

from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from toolboxv2 import App

from fastapi import FastAPI, Request, WebSocket
import sys
import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..utils.toolbox import get_app

app = FastAPI()

origins = [
    "http://194.233.168.22:8000",
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://0.0.0.0",
    "http://localhost",
    "http://194.233.168.22",
    "https://simpelm.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def index():
    # return RedirectResponse(url="/app")
    return "Willkommen bei Simple V0 powered by ToolBoxV2-0.0.3"


@app.get("/exit")
async def exit_code():
    tb_app.exit()
    exit(0)


@app.get("/HotReload")
async def exit_code():
    if "HotReload" in tb_app.id:
        tb_app.reset()
        tb_app.remove_all_modules()
        tb_app.load_all_mods_in_file()
        return "OK"
    return "Not found"

@app.websocket("/ws/{ws_id}")
async def websocket_endpoint(websocket: WebSocket, ws_id: str):
    websocket_id = ws_id
    print(f'websocket: {websocket_id}')
    await manager.connect(websocket, websocket_id)
    #try:

    while True:
        try:
            data = await websocket.receive_text()
        except WebSocketDisconnect as e:
            print(e)
            break
        # try:
        res = await manager.manage_data_flow(websocket, websocket_id, data)
        # print("manage_data_flow")
        # except Exception as e:
        #    print(e)
        #    res = '{"res": "error"}'
        if res is not None:
            print(f"RESPONSE: {res}")
            await websocket.send_text(res)
            print("Sending data to websocket")
        print("manager Don listening ->")
    #except Exception as e:
    #    print("websocket_endpoint - Exception: ", e)
    #finally:
    #    await manager.disconnect(websocket, websocket_id)


print("API: ", __name__)  # https://www.youtube.com/watch?v=_Im4_3Z1NxQ watch NOW
if __name__ == 'toolboxv2.api.fast_api_main':

    config_file = "api.config"
    id_name = ""

    for i in sys.argv[2:]:
        if i.startswith('data'):
            d = i.split(':')
            config_file = d[1]
            id_name = d[2]

    tb_app = get_app(id_name)
    if id_name == tb_app.id:
        print("🟢 START")
    with open(f"./data/api_pid_{id_name}", "w") as f:
        f.write(str(os.getpid()))
    # tb_app.load_all_mods_in_file()
    tb_app.save_load("welcome")
    tb_img = tb_app.MOD_LIST["welcome"].tools["printT"]
    tb_img()

    tb_app.save_load("WebSocketManager")
    tb_app.new_ac_mod("WebSocketManager")
    manager = tb_app.AC_MOD

    from .fast_api_install import router as install_router
    from .fast_app import router as app_router
    from .fast_api import router as api_router

    if "modInstaller" in tb_app.id:
        print("ModInstaller Init")
        api_router.include_router(install_router)

    app.include_router(app_router)
    app.include_router(api_router)
