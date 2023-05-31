import re
from pathlib import Path
from fastapi import Request, HTTPException, Depends, APIRouter
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer

from toolboxv2 import App
from toolboxv2.utils.toolbox import get_app

router = APIRouter(
    prefix="/app",
    tags=["token"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)

level = 0  # Setzen Sie den Level-Wert, um verschiedene Routen zu aktivieren oder zu deaktivieren
pattern = ['.png', '.jpg', '.jpeg', '.js', '.css', '.ico', '.gif', '.svg', '.wasm']
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    tb_app: App = get_app()
    return tb_app.run_any("cloudM", "validate_jwt", ["token", token, {}])


def check_access_level(required_level: int):
    if level < required_level:
        raise HTTPException(status_code=403, detail="Access forbidden")
    return True


@router.get("")
async def index(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('index.html')


@router.get("/login")
async def login_page(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('login.html')


@router.get("/signup")
async def signup_page(access_allowed: bool = Depends(lambda: check_access_level(2))):
    return serve_app_func('signup.html')


@router.get("/quicknote")
async def quicknote(current_user: str = Depends(get_current_user),
                    access_allowed: bool = Depends(lambda: check_access_level(0))):
    print("[current_user]", current_user)
    print("[access_allowed]", access_allowed)
    return serve_app_func('quicknote/index.html')


@router.get("/daytree")
async def daytree(current_user: str = Depends(get_current_user),
                  access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('daytree/index.html')


@router.get("/serverInWartung")
async def server_in_wartung(access_allowed: bool = Depends(lambda: check_access_level(-1))):
    return serve_app_func('serverInWartung.html')


@router.get("/{path:path}")
async def serve_files(path: str, request: Request, access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func(path)


def serve_app_func(path: str, prefix: str = "../app/"): # test location
    request_file_path = Path(prefix + path)
    ext = request_file_path.suffix

    if ext in pattern:
        path = 'main.html'

    return FileResponse(path)
