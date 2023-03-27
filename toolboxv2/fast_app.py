import os
import re
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import jwt
import datetime

from toolboxv2.fast_api_main import tb_app

app = FastAPI()

level = 0  # Setzen Sie den Level-Wert, um verschiedene Routen zu aktivieren oder zu deaktivieren
pattern = re.compile('.png|.jpg|.jpeg|.js|.css|.ico|.gif|.svg|.wasm', re.IGNORECASE)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    return tb_app.run_any("cloudM", "validate_jwt", ["token", token, {}])
def check_access_level(required_level: int):
    if level != required_level:
        raise HTTPException(status_code=403, detail="Access forbidden")
    return True


@app.get("/")
async def index(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('index.html')


@app.get("/login")
async def login_page(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('login.html')


@app.get("/signup")
async def signup_page(access_allowed: bool = Depends(lambda: check_access_level(2))):
    return serve_app_func('signup.html')


@app.get("/quicknote")
async def quicknote(current_user: str = Depends(get_current_user),
                    access_allowed: bool = Depends(lambda: check_access_level(0))):
    print("[current_user]", current_user)
    print("[access_allowed]", access_allowed)
    return serve_app_func('quicknote/index.html')


@app.get("/daytree")
async def daytree(current_user: str = Depends(get_current_user),
                  access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('daytree/index.html')


@app.get("/serverInWartung")
async def server_in_wartung(access_allowed: bool = Depends(lambda: check_access_level(-1))):
    return serve_app_func('serverInWartung.html')


@app.get("/{path:path}")
async def serve_files(path: str, request: Request, access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func(path)



def serve_app_func(path: str):
    request_file_path = Path(path)
    ext = request_file_path.suffix
    if not request_file_path.is_file() and not pattern.match(ext):
        path = 'test.html'

    return FileResponse(path)
