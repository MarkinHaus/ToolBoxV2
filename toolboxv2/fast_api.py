import os
from typing import Union

from fastapi import APIRouter, UploadFile

from fast_api_main import tb_app, PostRequest
from toolboxv2 import ToolBox_ovner

router = APIRouter()


@router.get("/api/")
def root():
    result = "ToolBoxV2"
    return {"res": result}


@router.post("/api/exit/{pid}")
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

        if res["username"] in (ToolBox_ovner if not isinstance(ToolBox_ovner, str) else [ToolBox_ovner]):
            tb_app.save_exit()
            tb_app.exit()
            exit(0)
    return {"res": "0"}


@router.get("/api/id")
def id_api():
    return {"res": str(tb_app.id)}


@router.get("/api/mod-list")
def mod_list():
    return {"res": list(tb_app.MOD_LIST.keys())}


@router.get("/api/SUPER_SET")
def super_set():
    return {"res": tb_app.SUPER_SET}


@router.get("/api/prefix")
def prefix_working_dir():
    return {"res": tb_app.PREFIX}


@router.get("/api/logs")
def logs_app():
    logs = {}
    for log in tb_app.logger:
        logs[log[0].name] = []
        logs[log[0].name].append(log[1:])
    print(logs)
    return {"res": logs}


@router.get("/api/test-exist/{name}")
def test_mod_dow(name: str):
    res = "mod-404"
    if name.lower() in tb_app.MOD_LIST:
        tb_app.new_ac_mod(name.lower())
        res = f"{name}-mod-online"
    return {"res": res}


@router.get("/api/mod/index/{name}")
def get_mod_index(name: str):
    try:
        tb_app.new_ac_mod(name)
        result = tb_app.help('')
    except:
        result = "None"
    return {"res": result}


@router.get("/api/get/{mod}/run/{name}")
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


@router.post("/api/post/{mod}/run/{name}")
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


@router.post("/api/upload-file/")
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
