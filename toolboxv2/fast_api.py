import os

from toolboxv2 import App, AppArgs, ToolBox_ovner

from fastapi import FastAPI, Request, UploadFile
from typing import Union
import sys
import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class PostRequest(BaseModel):
    token: str
    data: dict


owner = ToolBox_ovner
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


@app.get("/api/")
def root():
    result = tb_img()
    return {"res": result}


@app.post("/api/exit/{pid}")
def close(data: PostRequest, pid: int):
    if pid == os.getpid():
        res = tb_app.run_any('cloudm', "validate_jwt", [data])
        if isinstance(res, str):
            return {"res": res}
        if not isinstance(res, dict):
            return {"res": str(res)}
        if "res" in res.keys():
            res = res["res"]
        if "uid" not in res.keys():
            return {"res": str(res)}

        if res["username"] in (owner if not isinstance(owner, str) else [owner]):
            tb_app.save_exit()
            tb_app.exit()
            exit(0)
    return {"res": "0"}


@app.get("/api/id")
def id_api():
    return {"res": str(tb_app.id)}


@app.get("/api/mod-list")
def mod_list():
    return {"res": list(tb_app.MOD_LIST.keys())}


@app.get("/api/SUPER_SET")
def super_set():
    return {"res": tb_app.SUPER_SET}


@app.get("/api/prefix")
def prefix_working_dir():
    return {"res": tb_app.PREFIX}


@app.get("/api/logs")
def logs_app():
    logs = {}
    for log in tb_app.logger:
        logs[log[0].name] = []
        logs[log[0].name].append(log[1:])
    print(logs)
    return {"res": logs}


@app.get("/api/test-exist/{name}")
def test_mod_dow(name: str):
    res = "mod-404"
    if name.lower() in tb_app.MOD_LIST:
        tb_app.new_ac_mod(name.lower())
        res = f"{name}-mod-online"
    return {"res": res}


@app.get("/api/mod/index/{name}")
def get_mod_index(name: str):
    try:
        tb_app.new_ac_mod(name)
        result = tb_app.help('')
    except:
        result = "None"
    return {"res": result}


@app.get("/api/get/{mod}/run/{name}")
def get_mod_run(mod: str, name: str, command: Union[str, None] = None):
    print("get_mod_run")
    res = {}
    if not command:
        command = ''
    if tb_app.AC_MOD.name != mod.lower():
        if mod.lower() in tb_app.MOD_LIST:
            tb_app.new_ac_mod(mod)

    if tb_app.AC_MOD:
        res = tb_app.run_function(name, command.split('|'))

    if type(res) == str:
        if (res.startswith('{') or res.startswith('[')) or res.startswith('"[') or res.startswith('"{') \
            or res.startswith('\"[') or res.startswith('\"{') or res.startswith('b"[') or res.startswith('b"{'):
            res = eval(res)
    return {"res": res}


@app.post("/api/post/{mod}/run/{name}")
async def post_mod_run(data: PostRequest, mod: str, name: str, command: Union[str, None] = None):
    res = {}
    if not command:
        command = ''

    command = [data, command.split('|')]
    res = tb_app.run_any(mod, name, command)

    if type(res) == str:
        if (res.startswith('{') or res.startswith('[')) or res.startswith('"[') or res.startswith('"{') \
            or res.startswith('\"[') or res.startswith('\"{') or res.startswith('b"[') or res.startswith('b"{'):
            res = eval(res)

    return {"res": res}


@app.post("/api/upload-file/")
async def create_upload_file(file: UploadFile):
    if tb_app.debug:
        do = False
        try:
            tb_app.load_mod(file.filename.split(".py")[0])
        except ModuleNotFoundError:
            do = True

        if do:
            try:
                with open("./mods/" + file.filename, 'wb') as f:
                    while contents := file.file.read(1024 * 1024):
                        f.write(contents)
            except Exception:
                return {"res": "There was an error uploading the file"}
            finally:
                file.file.close()

            return {"res": f"Successfully uploaded {file.filename}"}
    return {"res": "not avalable"}


if __name__ == 'fast_api':  # do stuw withe ovner to secure ur self

    print("online")

    config_file = "api.config"
    id_name = ""

    for i in sys.argv[2:]:
        if i.startswith('data'):
            d = i.split(':')
            config_file = d[1]
            id_name = d[2]
    print(os.getcwd())
    tb_app = App("api")
    with open(f"api_pid_{id_name}", "w") as f:
        f.write(str(os.getpid()))
    tb_app.load_all_mods_in_file()
    tb_img = tb_app.MOD_LIST["welcome"].tools["printT"]
    tb_img()
    tb_app.new_ac_mod("welcome")
