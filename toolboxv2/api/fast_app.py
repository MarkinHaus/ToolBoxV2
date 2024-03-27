import os
import re
from pathlib import Path
from fastapi import Request, HTTPException, Depends, APIRouter
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer

import os

from starlette.responses import RedirectResponse

from toolboxv2 import App
from toolboxv2.utils.toolbox import get_app

router = APIRouter(
    prefix="/web",
    tags=["token"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)

level = 2  # Setzen Sie den Level-Wert, um verschiedene Routen zu aktivieren oder zu deaktivieren
pattern = ['.png', '.jpg', '.jpeg', '.js', '.css', '.ico', '.gif', '.svg', '.wasm']


def check_access_level(required_level: int):
    if level < required_level:
        raise HTTPException(status_code=403, detail="Access forbidden")
    return True


@router.get("")
async def index(access_allowed: bool = Depends(lambda: check_access_level(-1))):
    if level == -1:
        return serve_app_func('assets/serverInWartung.html')
    return serve_app_func('index.html')


@router.get("/")
async def index2(access_allowed: bool = Depends(lambda: check_access_level(-1))):
    if level == -1:
        return serve_app_func('assets/serverInWartung.html')
    return RedirectResponse(url="/web")


@router.get("/web/")
async def index2_(access_allowed: bool = Depends(lambda: check_access_level(-1))):
    if level == -1:
        return serve_app_func('assets/serverInWartung.html')
    return RedirectResponse(url="/web")


@router.get("/login")
async def login_page(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('assets/login.html')


@router.get("/logout")
async def login_page(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('assets/logout.html')


@router.get("/signup")
async def signup_page(access_allowed: bool = Depends(lambda: check_access_level(2))):
    return serve_app_func('assets/signup.html')


@router.get("/dashboard")
async def quicknote(access_allowed: bool = Depends(lambda: check_access_level(2))):
    return serve_app_func('dashboards/dashboard.html')  # 'dashboards/dashboard_builder.html')


@router.get("/{path:path}")
async def serve_files(path: str, request: Request, access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func(path)


def serve_app_func(path: str, prefix: str = os.getcwd() + "/web/"):
    if not path:
        path = "index.html"

    print("serving", path, prefix)
    request_file_path = Path(prefix + path)
    ext = request_file_path.suffix
    print(request_file_path, ext)

    # Define a dictionary to map file extensions to MIME types
    mime_types = {
        '.js': 'application/javascript',
        '.html': 'text/html',
        '.css': 'text/css',
        # Add other MIME types if needed
    }

    # Set the default MIME type to 'text/html'
    content_type = 'text/html'

    # Check if the file extension exists in the mime_types dictionary
    if ext in mime_types.keys():
        content_type = mime_types[ext]

    request_file_path.is_file()
    if request_file_path.exists():
        return FileResponse(request_file_path, media_type=content_type)
    return FileResponse("./web/assets/404.html", media_type=content_type)
