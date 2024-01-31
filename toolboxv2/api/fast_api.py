import os
import sys
from typing import Union
from fastapi import APIRouter, UploadFile

from toolboxv2 import ToolBox_over, App
from .util import PostRequest
from ..utils.toolbox import get_app

router = APIRouter(
    prefix="/api",
    # responses={404: {"description": "Not found"}},
)


@router.get("")
def root():
    result = "ToolBoxV2"
    return {"res": result}


@router.post("/exit/{pid}")
def close(data: PostRequest, pid: int):
    print(f"{pid=}, {os.getpid()}, {pid == os.getpid()}")
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

        if res["username"] in (ToolBox_over if not isinstance(ToolBox_over, str) else [ToolBox_over]):
            tb_app.exit()
            exit(0)
    return {"res": "0"}


@router.get("/id")
def id_api():
    return {"res": str(tb_app.id)}


@router.post("/post/{mod}/run/{name}")
async def post_mod_run(data: PostRequest, mod: str, name: str, command: Union[str, None] = None):
    res = {}
    if not command:
        command = ''

    command = [data, command.split('|')]
    res = tb_app.run_any(mod, 'api_' + name, command)

    if type(res) == str:
        if (res.startswith('{') or res.startswith('[')) or res.startswith('"[') or res.startswith('"{') \
            or res.startswith('\"[') or res.startswith('\"{') or res.startswith('b"[') or res.startswith('b"{'):
            res = eval(res)
    return {"res": res}


if __name__ == 'fast_api':
    tb_app = get_app(from_="api")
    tb_app.load_all_mods_in_file()
