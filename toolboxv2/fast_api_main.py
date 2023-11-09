import os

from toolboxv2 import App, AppArgs

from fastapi import FastAPI, Request, UploadFile
from typing import Union
import sys
import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class PostRequest(BaseModel):
    token: str
    data: dict


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
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


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
