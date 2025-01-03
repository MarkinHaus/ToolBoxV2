import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from inspect import signature
from pathlib import Path
from typing import List

import fastapi
from numpy.core.defchararray import endswith
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, HTMLResponse, FileResponse, Response
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import RedirectResponse
from watchfiles import awatch

from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.system.session import RequestSession
from toolboxv2.utils.extras.blobs import BlobFile
from toolboxv2.utils.security.cryp import DEVICE_KEY, Code

from fastapi import FastAPI, Request, WebSocket, APIRouter, HTTPException, Depends
import sys
import time
from fastapi.middleware.cors import CORSMiddleware

from toolboxv2 import TBEF, AppArgs, ApiResult, Spinner, get_app, Result
# from toolboxv2.__main__ import setup_app
from toolboxv2.utils.system.getting_and_closing_app import a_get_proxy_app

from toolboxv2.utils.system.state_system import get_state_from_app
from functools import partial, wraps

dev_hr_index = "v0.0.1"


def create_partial_function(original_function, partial_function):
    @wraps(original_function)
    async def wrapper(*args, **kwargs):
        # Call the partial function with the same arguments
        res = await partial_function(*args, **kwargs)
        if asyncio.iscoroutine(res):
            res = await res
        print("RESULT ::::", res)
        return res

    # Return the wrapper function which mimics the original function's signature
    return wrapper


id_name = ""
debug = False
for i in sys.argv[2:]:
    if i.startswith('data'):
        d = i.split(':')
        debug = True if d[1] == "True" else False
        id_name = d[2]
args = AppArgs().default()
args.name = id_name
args.debug = debug
args.sysPrint = True
tb_app = get_app(from_="init-api-get-tb_app", name=id_name, args=args, sync=True)

manager = tb_app.get_mod("WebSocketManager").get_pools_manager()

pattern = ['.png', '.jpg', '.jpeg', '.js', '.css', '.ico', '.gif', '.svg', '.wasm']


# with Spinner("loding mods", symbols="b"):
#     module_list = tb_app.get_all_mods()
#     open_modules = tb_app.functions.keys()
#     start_len = len(open_modules)
#     for om in open_modules:
#         if om in module_list:
#             module_list.remove(om)
#     _ = {tb_app.save_load(mod, 'app') for mod in module_list}
#
# tb_app.watch_mod(mod_name="WidgetsProvider")


class RateLimitingMiddleware(BaseHTTPMiddleware):
    # Rate limiting configurations
    RATE_LIMIT_DURATION = timedelta(seconds=2)
    RATE_LIMIT_REQUESTS_app = 800
    RATE_LIMIT_REQUESTS_api = 60
    WHITE_LIST_IPS = ["127.0.0.1"]

    def __init__(self, app):
        super().__init__(app)
        # Dictionary to store request counts for each IP
        self.request_counts = {}

    async def dispatch(self, request, call_next):
        # Get the client's IP address
        client_ip = request.client.host

        # Check if IP is already present in request_counts
        request_count_app: int
        request_count_api: int
        last_request: datetime
        request_count_app, request_count_api, last_request = self.request_counts.get(client_ip, (0, 0, datetime.min))

        # Calculate the time elapsed since the last request
        elapsed_time = datetime.now() - last_request
        if request.url.path.split('/')[1] == "web":
            pass
        if elapsed_time > self.RATE_LIMIT_DURATION:
            # If the elapsed time is greater than the rate limit duration, reset the count
            request_count_app: int = 1
            request_count_api: int = 1
        else:
            if request_count_app >= self.RATE_LIMIT_REQUESTS_app:
                # If the request count exceeds the rate limit, return a JSON response with an error message
                return JSONResponse(
                    status_code=429,
                    content={"message": "Rate limit exceeded. Please try again later. app"}
                )
            if request_count_api >= self.RATE_LIMIT_REQUESTS_api:
                # If the request count exceeds the rate limit, return a JSON response with an error message
                return JSONResponse(
                    status_code=429,
                    content={"message": "Rate limit exceeded. Please try again later. api"}
                )
            if 'web' in request.url.path:
                request_count_app += 1
            else:
                request_count_api += 1

        # Update the request count and last request timestamp for the IP
        if client_ip not in self.WHITE_LIST_IPS:
            self.request_counts[client_ip] = (request_count_app, request_count_api, datetime.now())

        # Proceed with the request
        response = await call_next(request)
        return response


class SessionAuthMiddleware(BaseHTTPMiddleware):
    # Rate limiting configurations
    SESSION_DURATION = timedelta(minutes=5)
    GRAY_LIST = []
    BLACK_LIST = []

    def __init__(self, app):
        super().__init__(app)
        # Dictionary to store request counts for each IP
        # 'session-id' : {'jwt-claim', 'validate', 'exit on ep time from jwt-claim', 'SiID'}
        self.sessions = {

        }
        self.cookie_key = tb_app.config_fh.one_way_hash(tb_app.id, 'session')

    async def set_body(self, request: Request):
        receive_ = await request._receive()

        async def receive():
            return receive_

        request._receive = receive

    async def crate_new_session_id(self, request: Request, jwt_claim: str or None, username: str or None,
                                   session_id: str = None):

        if session_id is None:
            session_id = hex(tb_app.config_fh.generate_seed())
            tb_app.logger.debug(f"Crating New Session {session_id}")
            h_session_id = '#0'
        else:
            tb_app.logger.debug(f"Evaluating Session {session_id}")
            h_session_id = session_id
            session_id = hex(tb_app.config_fh.generate_seed())

        request.session['ID'] = session_id

        self.sessions[session_id] = {
            'jwt-claim': jwt_claim,
            'validate': False,
            'live_data': {},
            'exp': datetime.now(),
            'ip': request.client.host,
            'port': request.client.port,
            'c': 0,
            'h-sid': h_session_id
        }
        # print("[jwt_claim]:, ", jwt_claim)
        # print(username)
        # print(request.json())
        if request.client.host in self.GRAY_LIST and not request.url.path.split('/')[-1] in ['login', 'signup']:
            return JSONResponse(
                status_code=403,
                content={"message": "Pleas Login or signup"}
            )
        if request.client.host in self.BLACK_LIST:
            return JSONResponse(
                status_code=401,
                content={"message": "!ACCESS_DENIED!"}
            )
        if jwt_claim is None or username is None:
            tb_app.logger.debug(f"Session Handler New session no jwt no username {username}")
            return '#0'
        return await self.verify_session_id(session_id, username, jwt_claim)

    async def verify_session_id(self, session_id, username, jwt_claim):

        if not await tb_app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.JWT_CHECK_CLAIM_SERVER_SIDE,
                                      username=username,
                                      jwt_claim=jwt_claim):
            # del self.sessions[session_id]
            self.sessions[session_id]['CHECK'] = 'failed'
            self.sessions[session_id]['c'] += 1
            tb_app.logger.debug(f"Session Handler V invalid jwt from : {username}")
            return '#0'

        user_result = await tb_app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME,
                                             username=username,
                                             get_results=True)

        if user_result.is_error():
            # del self.sessions[session_id]
            self.sessions[session_id]['CHECK'] = user_result.print(show=False)
            self.sessions[session_id]['c'] += 1
            tb_app.logger.debug(f"Session Handler V invalid Username : {username}")
            return '#0'

        user = user_result.get()

        user_instance = await tb_app.a_run_any(TBEF.CLOUDM_USERINSTANCES.GET_USER_INSTANCE, uid=user.uid, hydrate=False,
                                               get_results=True)

        if user_instance.is_error():
            user_instance.print()
            tb_app.logger.debug(f"Session Handler V no UsernameInstance : {username}")
            return '#0'

        self.sessions[session_id] = {
            'jwt-claim': jwt_claim,
            'validate': True,
            'exp': datetime.now(),
            'user_name': tb_app.config_fh.encode_code(user.name),
            'c': 0,
            'live_data': {
                'SiID': user_instance.get().get('SiID'),
                'level': user.level if user.level > 1 else 1,
                'spec': user_instance.get().get('VtID'),
                'user_name': tb_app.config_fh.encode_code(user.name)
            },

        }

        return session_id

    async def validate_session(self, session_id):

        tb_app.logger.debug(f"validating id {session_id}")

        if session_id is None:
            return False

        exist = session_id in self.sessions

        if not exist:
            return False

        session = self.sessions[session_id]

        if not session.get('validate', False):
            return False

        c_user_name, jwt = session.get('user_name'), session.get('jwt-claim')
        if c_user_name is None or jwt is None:
            return False

        if datetime.now() - session.get('exp', datetime.min) > self.SESSION_DURATION:
            user_name = tb_app.config_fh.decode_code(c_user_name)
            return await self.verify_session_id(session_id, user_name, jwt) != 0

        return True

    async def dispatch(self, request: Request, call_next):
        # Get the client's IP address
        session = request.cookies.get(self.cookie_key)
        tb_app.logger.debug(f"({request.session} --> {request.url.path})")
        if request.url.path == '/validateSession':
            print("INSSSSO")
            await self.set_body(request)
            body = await request.body()
            print("BODY #####", body)
            if body == b'':
                return JSONResponse(
                    status_code=401,
                    content={"message": "Invalid Auth data.", "valid": False}
                )
            body = json.loads(body)
            jwt_token = body.get('Jwt_claim', None)
            username = body.get('Username', None)
            session_id = await self.crate_new_session_id(request, jwt_token, username,
                                                         session_id=request.session.get('ID'))
            return JSONResponse(
                status_code=200,
                content={"message": "Valid Session", "valid": True}
            ) if await self.validate_session(session_id) else JSONResponse(
                status_code=401,
                content={"message": "Invalid Auth data.", "valid": False}
            )
        elif not session:
            session_id = await self.crate_new_session_id(request, None, "Unknown")
        elif request.session.get('ID', '') not in self.sessions:
            print("Session Not Found")
            session_id = await self.crate_new_session_id(request, None, "Unknown", session_id=request.session.get('ID'))
            request.session['valid'] = False
        else:
            session_id: str = request.session.get('ID', '')
        request.session['live_data'] = {}
        # print("testing session")
        if await self.validate_session(session_id):
            print("valid session")
            request.session['valid'] = True
            request.session['live_data'] = self.sessions[session_id]['live_data']
            if request.url.path == '/web/logoutS':
                uid = tb_app.run_any(TBEF.CLOUDM_USERINSTANCES.GET_INSTANCE_SI_ID,
                                     si_id=self.sessions[session_id]['live_data']['SiID']).get('save', {}).get('uid')
                if uid is not None:
                    print("start closing istance :t", uid)
                    tb_app.run_any(TBEF.CLOUDM_USERINSTANCES.CLOSE_USER_INSTANCE, uid=uid)
                    del self.sessions[session_id]
                    print("Return redirect :t", uid)
                    return RedirectResponse(
                        url="/web/logout")  # .delete_cookie(tb_app.config_fh.one_way_hash(tb_app.id, 'session'))
                else:
                    del request.session['live_data']
                    return JSONResponse(
                        status_code=403,
                        content={"message": "Invalid Auth data."}
                    )
            elif request.url.path == '/SessionAuthMiddlewareLIST':
                return JSONResponse(
                    status_code=200,
                    content={"message": "Valid Session", "GRAY_LIST": self.GRAY_LIST, "BLACK_LIST": self.BLACK_LIST}
                )
            elif request.url.path == '/IsValiSession':
                return JSONResponse(
                    status_code=200,
                    content={"message": "Valid Session", "valid": True}
                )  # .set_cookie(self.cookie_key, value=request.cookies.get('session'))
        elif request.url.path == '/IsValiSession':
            return JSONResponse(
                status_code=401,
                content={"message": "Invalid Auth data.", "valid": False}
            )
        elif session_id == '#0':
            return await call_next(request)
        elif isinstance(session_id, JSONResponse):
            return session_id
        else:
            request.session['valid'] = False
            ip = self.sessions[session_id].get('ip', "unknown")
            tb_app.logger.warning(f"SuS Request : IP : {ip} count : {self.sessions[session_id]['c']}")
            if protect_url_split_helper(request.url.path.split('/')):
                self.sessions[session_id]['c'] += 1
            if ip not in RateLimitingMiddleware.WHITE_LIST_IPS:
                c = self.sessions[session_id]['c']
                if c < 20:
                    tb_app.logger.warning(f"SuS Request : IP : {ip} ")
                elif c == 460:
                    self.GRAY_LIST.append(self.sessions[session_id].get('ip', "unknown"))
                    self.sessions[session_id]['what_user'] = True
                elif c == 6842:
                    self.GRAY_LIST.append(self.sessions[session_id].get('ip', "unknown"))
                    self.sessions[session_id]['ratelimitWarning'] = True
                    return JSONResponse(
                        status_code=401,
                        content={"message": "Login or Signup for further access"}
                    )
                elif c > 540_000_000:
                    self.BLACK_LIST.append(self.sessions[session_id].get('ip', "unknown"))
                    return JSONResponse(
                        status_code=403,
                        content={"message": "u got BLACK_LISTED"}
                    )
        return await call_next(request)

        # if session:
        #     response.set_cookie(
        #         self.cookie_key,
        #     )


app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://0.0.0.0",
    "http://localhost",
    "https://simplecore.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitingMiddleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    # if response.body.get("info", {}).get("exec_code", 0) != 0:
    return response


@app.middleware("http")
async def session_protector(request: Request, call_next):
    response = await call_next(request)
    if 'session' in request.scope.keys() and 'live_data' in request.session.keys():
        del request.session['live_data']
    return response


def protect_level_test(request):

    if 'live_data' not in request.session.keys():
        return None

    user_level = request.session['live_data'].get('level', -1)
    user_spec = request.session['live_data'].get('spec', 'app')

    if len(request.url.path.split('/')) < 4:
        tb_app.logger.info(f'not protected url {request.url.path}')
        return user_level >= -1

    modul_name = request.url.path.split('/')[2]
    fuction_name = request.url.path.split('/')[3]
    print(tb_app.functions.get(modul_name, {}).keys())
    if not (modul_name in tb_app.functions.keys() and fuction_name in tb_app.functions.get(modul_name, {}).keys()):
        request.session['live_data']['RUN'] = False
        tb_app.logger.warning(
            f"Path is not for level testing {request.url.path} Function {modul_name}.{fuction_name} dos not exist")
        return None  # path is not for level testing

    fod, error = tb_app.get_function((modul_name, fuction_name), metadata=True, specification=user_spec)

    if error:
        tb_app.logger.error(f"Error getting function for user {(modul_name, fuction_name)}{request.session}")
        return None

    fuction_data, fuction = fod

    fuction_level = fuction_data.get('level', 0)
    print(f"{user_level=} >= {fuction_level=}")
    request.session['live_data']['GET_R'] = fuction_data.get('request_as_kwarg', False)

    request.session['live_data']['RUN'] = user_level >= fuction_level
    return request.session['live_data']['RUN']



def protect_url_split_helper(url_split):
    if len(url_split) < 3:
        tb_app.logger.info(f'not protected url {url_split}')
        return False

    elif url_split[1] == "web" and len(url_split[2]) == 1 and url_split[2] != "0":
        tb_app.logger.info(f'protected url {url_split}')
        return True

    elif url_split[1] == "web" and url_split[2] in [
        'dashboards',
        'dashboard',
    ]:
        tb_app.logger.info(f'protected url dashboards {url_split}')
        return True

    elif url_split[1] == "web":
        return False

    elif url_split[1] == "api" and url_split[2] in [
        'CloudM.AuthManager',
        'email_waiting_list'
    ] + tb_app.api_allowed_mods_list:
        return False

    return True


@app.middleware("http")
async def protector(request: Request, call_next):
    needs_protection = protect_url_split_helper(request.url.path.split('/'))
    if not needs_protection:
        return await user_runner(request, call_next)

    plt =  protect_level_test(request)
    if plt is None:

        if not request.session.get('valid'):
            # do level test
            return FileResponse("./web/assets/401.html", media_type="text/html", status_code=401)

    elif plt is False:
        return JSONResponse(
            status_code=403,
            content={"message": "Protected resource invalid_level  <a href='/web'>Back To Start</a>"}
        )

    return await user_runner(request, call_next)


async def request_to_request_session(request):
    jk = request.json()
    if asyncio.iscoroutine(jk):
        try:
            jk = await jk
        except Exception:
            pass
    js = lambda :jk
    return RequestSession(
        session=request.session,
        body=request.body,
        json=js,
        row=request,
    )


async def user_runner(request, call_next):
    if not request:
        return HTMLResponse(status_code=501, content="No request")
    run_fuction = request.session.get("live_data", {}).get('RUN', False)
    if not run_fuction:
        response = await call_next(request)
        return response
    print("user_runner", request.session.get('live_data'))
    print(request.url.path.split('/'))

    if len(request.url.path.split('/')) < 4:
        response = await call_next(request)
        return response

    modul_name = request.url.path.split('/')[2]
    fuction_name = request.url.path.split('/')[3]

    path_params = request.path_params
    query_params = dict(request.query_params)

    if request.session['live_data'].get('GET_R', False):
        query_params['request'] = await request_to_request_session(request)

    async def execute_in_threadpool(coroutine, *args):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, lambda: asyncio.run(coroutine(*args)))

    # Wrappe die asynchrone Funktion in einem separaten Thread
    async def task():
        return await tb_app.a_run_function((modul_name, fuction_name),
                                         tb_run_with_specification=request.session['live_data'].get('spec', 'app'),
                                         args_=path_params.values(),
                                         kwargs_=query_params)

    # Starte die Aufgabe in einem separaten Thread
    future = asyncio.create_task(execute_in_threadpool(task))
    result = None
    # Nicht blockierendes Warten
    while tb_app.alive:
        if future.done():
            result = future.result()
            break
        await asyncio.sleep(0.1)  # Ermöglicht anderen FastAPI-Requests, weiter zu laufen

    request.session['live_data']['RUN'] = False
    request.session['live_data']['GET_R'] = False

    print(f"RESULT is ========== type {type(result)}")

    if result is None:
        return HTMLResponse(status_code=200, content=result)

    if isinstance(result, str):
        return HTMLResponse(status_code=200, content=result)

    if not isinstance(result, Result) and not isinstance(result, ApiResult):
        if isinstance(result, Response):
            return result
        return JSONResponse(result)

    if result.info.exec_code == 100:
        response = await call_next(request)
        return response

    if result.info.exec_code == 0:
        result.info.exec_code = 200

    result.print()

    try:
        content = result.to_api_result().json()
    except TypeError:
        result.result.data = await result.result.data
        content = result.to_api_result().json()

    return JSONResponse(status_code=result.info.exec_code if result.info.exec_code > 0 else 500,
                        content=content)


@app.get("/")
async def index():
    return RedirectResponse(url="/web/")


@app.get("/index.js")
async def index0():
    return serve_app_func("main.js")


@app.get("/index.html")
async def indexHtml():
    return serve_app_func("")


@app.get("/tauri")
async def index():
    return serve_app_func("/web/assets/widgetControllerLogin.html")


@app.get("/favicon.ico")
async def index():
    return serve_app_func('/web/favicon.ico')
    # return "Willkommen bei Simple V0 powered by ToolBoxV2-0.0.3"


# @app.get("/exit")
# async def exit_code():
#     tb_app.exit()
#     exit(0)


"""@app.websocket("/ws/{ws_id}")
async def websocket_endpoint(websocket: WebSocket, ws_id: str):
    websocket_id = ws_id
    print(f'websocket: {websocket_id}')
    if not await manager.connect(websocket, websocket_id):
        await websocket.close()
        return
    try:
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect as e:
                print(e)
                break
            try:
                res = await manager.manage_data_flow(websocket, websocket_id, data)
                print("manage_data_flow")
            except Exception as e:
                print(e)
                res = '{"res": "error"}'
            if res is not None:
                print(f"RESPONSE: {res}")
                await websocket.send_text(res)
                print("Sending data to websocket")
            print("manager Don  ->")
    except Exception as e:
        print("websocket_endpoint - Exception: ", e)
    finally:
        await manager.disconnect(websocket, websocket_id)
"""

level = 2  # Setzen Sie den Level-Wert, um verschiedene Routen zu aktivieren oder zu deaktivieren

def check_access_level(required_level: int):
    if level < required_level:
        raise HTTPException(status_code=403, detail="Access forbidden")
    return True

@app.websocket("/ws/{pool_id}/{ws_id}")
async def websocket_endpoint(websocket: WebSocket, pool_id: str, ws_id: str):
    connection_id = ws_id
    tb_app.logger.info(f'New WebSocket connection: pool_id={pool_id}, connection_id={connection_id}')

    await websocket.accept()
    await manager.add_connection(pool_id, connection_id, websocket)

    try:
        while True:
            try:
                data = await websocket.receive_text()
                tb_app.logger.debug(f"Received data from {connection_id} in pool {pool_id}: {data}")

                await manager.handle_message(pool_id, connection_id, data)

            except WebSocketDisconnect:
                tb_app.logger.info(f"WebSocket disconnected: pool_id={pool_id}, connection_id={connection_id}")
                break

            except Exception as e:
                tb_app.logger.error(f"Error in websocket_endpoint: {str(e)}")
                await websocket.send_text(json.dumps({"error": "An unexpected error occurred"}))

    finally:
        await manager.remove_connection(pool_id, connection_id)
        tb_app.logger.info(f"Connection closed and removed: pool_id={pool_id}, connection_id={connection_id}")


@app.get("/web/login")
async def login_page(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('web/assets/login.html')


@app.get("/web/logout")
async def login_page(access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func('web/assets/logout.html')


@app.get("/web/signup")
async def signup_page(access_allowed: bool = Depends(lambda: check_access_level(2))):
    return serve_app_func('web/assets/signup.html')


@app.get("/web/dashboard")
async def quicknote(access_allowed: bool = Depends(lambda: check_access_level(2))):
    return serve_app_func('web/dashboards/dashboard.html')  # 'dashboards/dashboard_builder.html')


def serve_app_func(path: str, prefix: str = os.getcwd() + "/dist/"):
    # Default to 'index.html' if no specific path is given
    if not path or '.' not in path:  # No file extension, assume SPA route
        path = "index.html"

    # Full path to the requested file
    request_file_path = Path(prefix) / path

    # MIME types dictionary
    mime_types = {
        '.js': 'application/javascript',
        '.html': 'text/html',
        '.css': 'text/css',
    }

    # Determine MIME type based on file extension, default to 'text/html'
    content_type = mime_types.get(request_file_path.suffix, 'text/html')

    # Serve the requested file if it exists, otherwise fallback to index.html for SPA
    if request_file_path.exists():
        return FileResponse(request_file_path, media_type=content_type)

    # Fallback to a 404 page if the file does not exist
    return FileResponse(os.path.join(os.getcwd(), "dist", "web/assets/404.html"), media_type="text/html")


@app.on_event("startup")
async def startup_event():
    print('Server started :', __name__, datetime.now())


@app.on_event("shutdown")
async def shutdown_event():
    print('server Shutdown :', datetime.now())


'''from fastapi.testclient import TestClient
client = TestClient(app)

def test_modify_request_response_middleware():
    # Send a GET request to the hello endpoint
    response = client.get("/")
    # Assert the response status code is 200
    assert response.status_code == 200
    # Assert the middleware has been applied
    assert response.headers.get("X-Process-Time") > 0
    # Assert the response content
    print(response)
    # assert response.json() == {"message": "Hello, World!"}


def test_rate_limiting_middleware():
    time.sleep(0.2)
    response = client.get("/")
    # Assert the response status code is 200
    assert response.status_code == 200

    for _ in range(10):
        time.sleep(0.2)
        response = client.get("/")
        # Assert the response status code is 200
        assert response.status_code == 200

    time.sleep(0.2)
    response = client.get("/")
    # Assert the response status code is 200
    assert response.status_code == 429

'''


async def helper(id_name):
    global tb_app
    is_proxy = False
    # tb_app = await a_get_proxy_app(tb_app)

    if "HotReload" in tb_app.id:
        @app.get("/HotReload")
        async def exit_code():
            if tb_app.debug:
                tb_app.remove_all_modules()
                await tb_app.load_all_mods_in_file()
                return "OK"
            return "Not found"

    try:
        with open(f"./.data/api_pid_{id_name}", "w") as f:
            f.write(str(os.getpid()))
            f.close()
    except FileNotFoundError:
        pass
    await tb_app.load_all_mods_in_file()
    if id_name.endswith("_D"):
        with BlobFile(f"FastApi/{id_name}/dev", mode='r') as f:
            modules = f.read_json().get("modules", [])
        for mods in modules:
            tb_app.print(f"ADDING :  {mods}")
            tb_app.watch_mod(mods)

    d = tb_app.get_mod("DB")
    d.initialize_database()
    # c = d.edit_cli("RR")
    # await tb_app.watch_mod("CloudM.AuthManager", path_name="/CloudM/AuthManager.py")
    c = d.initialized()
    tb_app.sprint("DB initialized")
    c.print()
    if not c.get():
        exit()
    tb_app.get_mod("WebSocketManager")

    from .fast_api_install import router as install_router
    tb_app.sprint("loading CloudM")
    tb_app.get_mod("CloudM")
    # all_mods = tb_app.get_all_mods()
    provider = os.environ.get("MOD_PROVIDER", default="http://127.0.0.1:5000/")

    tb_state = [None]

    def get_d(name="CloudM"):
        if tb_state[0] is None:
            tb_state[0] = get_state_from_app(tb_app, simple_core_hub_url=provider)
        return tb_app.get_mod("CloudM").get_mod_snapshot(name)

    install_router.add_api_route('/' + "get", get_d, methods=["GET"], description="get_species_data")
    tb_app.sprint("include Installer")
    app.include_router(install_router)

    async def proxi_helper(*__args, **__kwargs):
        await tb_app.client.get('sender')({'name': "a_run_any", 'args': __args, 'kwargs': __kwargs})
        while Spinner("Waiting for result"):
            try:
                return tb_app.client.get('receiver_queue').get(timeout=tb_app.timeout)
            except Exception as _e:
                tb_app.sprint("Error", _e)
                return HTMLResponse(status_code=408)

    tb_app.sprint("Start Processioning Functions")
    for mod_name, functions in tb_app.functions.items():
        tb_app.print(f"Processing : {mod_name} \t\t", end='\r')
        add = False
        router = APIRouter(
            prefix=f"/api/{mod_name}",
            tags=["token", mod_name],
            # dependencies=[Depends(get_token_header)],
            # responses={404: {"description": "Not found"}},
        )
        for function_name, function_data in functions.items():
            if not isinstance(function_data, dict):
                continue
            api: list = function_data.get('api')
            if api is False:
                continue
            add = True
            params: list = function_data.get('params')
            sig: signature = function_data.get('signature')
            state: bool = function_data.get('state')
            api_methods: List[str] = function_data.get('api_methods', ["AUTO"])

            tb_func, error = tb_app.get_function((mod_name, function_name), state=state, specification="app")
            if not hasattr(tb_func, "__name__"):
                tb_func.__name__ = function_name
            tb_app.logger.debug(f"Loading fuction {function_name} , exec : {error}")

            if error != 0:
                continue
            tb_app.print(f"working on fuction {function_name}", end='\r')
            if 'main' in function_name and 'web' in function_name:
                tb_app.sprint(f"creating Rout {mod_name} -> {function_name}")
                app.add_api_route('/' + mod_name, tb_func, methods=["GET"],
                                     description=function_data.get("helper", ""))
                continue

            if 'websocket' in function_name:
                tb_app.sprint(f"adding websocket Rout {mod_name} -> {function_name}")
                router.add_api_websocket_route('/' + function_name, tb_func)
                continue

            try:
                if tb_func and is_proxy:
                    tb_func = create_partial_function(tb_func, partial(proxi_helper,
                                                                       mod_function_name=(
                                                                           mod_name, function_name),
                                                                       get_results=True))
                if tb_func:
                    if api_methods != "AUTO":
                        router.add_api_route('/' + function_name, tb_func, methods=api_methods,
                                             description=function_data.get("helper", ""))
                    if len(params):
                        router.add_api_route('/' + function_name, tb_func, methods=["POST"],
                                             description=function_data.get("helper", ""))
                    else:
                        router.add_api_route('/' + function_name, tb_func, methods=["GET"],
                                             description=function_data.get("helper", ""))
                    # print("Added live", function_name)
                else:
                    raise ValueError(f"fuction '{function_name}' not found")

            except fastapi.exceptions.FastAPIError as e:
                raise SyntaxError(f"fuction '{function_name}' prove the signature error {e}")
        if add:
            app.include_router(router)

    app.add_api_route("/{path:path}", serve_files)
    if id_name in tb_app.id:
        print("🟢 START")


print("API: ", __name__)
app.add_middleware(SessionAuthMiddleware)

app.add_middleware(SessionMiddleware,
                   session_cookie=Code.one_way_hash(tb_app.id, 'session'),
                   https_only='live' in tb_app.id,
                   secret_key=Code.one_way_hash(DEVICE_KEY(), tb_app.id))
tb_app.run_a_from_sync(helper, id_name)


async def serve_files(path: str, request: Request, access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func(path)



# print("API: ", __name__)
# if __name__ == 'toolboxv2.api.fast_api_main':
#     global tb_app
