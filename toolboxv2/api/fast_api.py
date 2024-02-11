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


@router.get("/id")
def id_api():
    return {"res": str(tb_app.id)}


if __name__ == 'fast_api':
    tb_app = get_app(from_="api")
    tb_app.load_all_mods_in_file()
