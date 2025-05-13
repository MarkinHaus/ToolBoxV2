import asyncio
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# from toolboxv2.__main__ import setup_app
from functools import partial, wraps

import fastapi
import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from toolboxv2 import TBEF, ApiResult, AppArgs, Result, Spinner, get_app
from toolboxv2.mods.FastApi.fast_lit import BidirectionalStreamlitAppManager
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage
from toolboxv2.utils.security.cryp import DEVICE_KEY, Code
from toolboxv2.utils.system.session import RequestSession

from .fast_nice import create_nicegui_manager
from .util import serve_app_func

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
        debug = d[1] == "True"
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
    BLACK_LIST_IPS = []

    def __init__(self, app):
        super().__init__(app)
        # Dictionary to store request counts for each IP
        self.request_counts = {}

    async def dispatch(self, request, call_next):
        # Get the client's IP address
        client_ip = request.client.host

        if client_ip in self.BLACK_LIST_IPS:
            return JSONResponse(
                status_code=200,
                content={"message": "NO ACCESS"}
            )

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
            if 'web' in request.url.path or 'gui' in request.url.path or 'index.html' in request.url.path or 'vendors-' in request.url.path:
                request_count_app += 1
            elif 'api' in request.url.path:
                request_count_api += 1
            else:
                request_count_api += 2
                request_count_app += 10

        # Proceed with the request
        response = await call_next(request)
        if hasattr(response, 'status_code'):
            if not protect_url_split_helper(request.url.path.split('/')):
                if response.status_code == 307:
                    request_count_app += 50
                    request_count_api += 30
                if response.status_code != 200:
                    request_count_app += 50
                    request_count_api += 15
                if response.status_code == 401:
                    request_count_app += 60
                    request_count_api += 10
            else:
                if response.status_code == 307:
                    request_count_app += 25
                    request_count_api += 3
                if response.status_code != 200:
                    request_count_app += 15
                    request_count_api += 5
                if response.status_code == 401:
                    request_count_app += 300
                    request_count_api += 40
                if response.status_code == 404:
                    request_count_app += 15
                    request_count_api += 5
        else:
            if not protect_url_split_helper(request.url.path.split('/')):
                request_count_app += 15
                request_count_api += 5
            else:
                request_count_app += 350
                request_count_api += 20
        # Update the request count and last request timestamp for the IP
        if client_ip not in self.WHITE_LIST_IPS:
            self.request_counts[client_ip] = (request_count_app, request_count_api, datetime.now())
        tb_app.logger.warning(f"SuS Request : IP : {client_ip} count : {request_count_app=} | {request_count_api=}")
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
        self.is_init = False
        self.cookie_key = tb_app.config_fh.one_way_hash(tb_app.id, 'session')
        self.db = None
        self.sessions = {}

    # --- Database Setup ---
    def get_db(self):
        if self.db is not None:
            return self.db
        db = get_app().get_mod("DB", spec="FastApi.sessions")
        if not self.is_init:
            self.is_init = True
            db.edit_cli("LD")
            db.initialize_database()
        self.db = db
        return db

    # --- Session State Management ---
    def get_session(self, session_id: str) -> dict:
        if len(session_id) == 0:
            return {}
        if session_id in self.sessions:
            return self.sessions[session_id]
        db = self.get_db()
        session = db.get(f"FastApi::session:{session_id}")

        try:
            session = json.loads(session.get().decode('utf-8').replace("'", '"'))
        except Exception:
            session = {
                'jwt-claim': '',
                'validate': False,
                'live_data': {},
                'exp': datetime.now(),
                'ip': '',
                'port': '',
                'c': 0,
                'CHECK': '',
                'h-sid': '',
                'new': True
            }

        return session

    def save_session(self, session_id: str, state: dict, remote=False):

        self.sessions[session_id] = state
        if not remote:
            return
        db = self.get_db()
        db.set(f"FastApi::session:{session_id}", state)

    def del_session(self, session_id: str):
        db = self.get_db()
        db.delete(f"FastApi::session:{session_id}")

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
        request.session['IDh'] = h_session_id

        self.save_session(session_id, {
            'jwt-claim': jwt_claim,
            'validate': False,
            'live_data': {},
            'exp': datetime.now(),
            'ip': request.client.host,
            'port': request.client.port,
            'c': 0,
            'CHECK': '',
            'h-sid': h_session_id
        }, remote=True)
        # print("[jwt_claim]:, ", jwt_claim)
        # print(username)
        # print(request.json())
        if request.client.host in self.GRAY_LIST and request.url.path.split('/')[-1] not in ['login', 'signup']:
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
        session = self.get_session(session_id)

        if not await tb_app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.JWT_CHECK_CLAIM_SERVER_SIDE,
                                      username=username,
                                      jwt_claim=jwt_claim):
            session['CHECK'] = 'failed'
            session['c'] += 1
            self.save_session(session_id, session)
            tb_app.logger.debug(f"Session Handler V invalid jwt from : {username}")
            return '#0'

        user_result = await tb_app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME,
                                             username=username,
                                             get_results=True)

        if user_result.is_error():
            # del session
            session['CHECK'] = user_result.print(show=False)
            session['c'] += 1
            self.save_session(session_id, session)
            tb_app.logger.debug(f"Session Handler V invalid Username : {username}")
            return '#0'

        user = user_result.get()

        user_instance = await tb_app.a_run_any(TBEF.CLOUDM_USERINSTANCES.GET_USER_INSTANCE, uid=user.uid, hydrate=False,
                                               get_results=True)

        if user_instance.is_error():
            user_instance.print()
            tb_app.logger.debug(f"Session Handler V no UsernameInstance : {username}")
            return '#0'

        self.save_session(session_id, {
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

        }, remote=True)

        return session_id

    async def validate_session(self, session_id):

        tb_app.logger.debug(f"validating id {session_id}")

        if session_id is None:
            return False

        session = self.get_session(session_id)

        if session.get('new', False):
            return False

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
            await self.set_body(request)
            body = await request.body()
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
        elif request.session.get('ID', True) and self.get_session(request.session.get('ID', '')).get("new", False):
            print("Session Not Found")
            if request.session.get('ID') in self.sessions:
                session_id = request.session.get('ID')
            else:
                session_id = await self.crate_new_session_id(request, None, "Unknown", session_id=request.session.get('ID'))
            request.session['valid'] = False
        else:
            session_id: str = request.session.get('ID', '')
        request.session['live_data'] = {}
        # print("testing session")
        session_data = self.get_session(session_id)
        if await self.validate_session(session_id):
            print("valid session")

            request.session['valid'] = True
            request.session['live_data'] = session_data['live_data']
            if request.url.path == '/web/logoutS':
                uid = tb_app.run_any(TBEF.CLOUDM_USERINSTANCES.GET_INSTANCE_SI_ID,
                                     si_id=session_data['live_data']['SiID']).get('save', {}).get('uid')
                if uid is not None:
                    print("start closing istance :t", uid)
                    tb_app.run_any(TBEF.CLOUDM_USERINSTANCES.CLOSE_USER_INSTANCE, uid=uid)
                    self.del_session(session_id)
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
            if request.session.get('valid', False):
                session_data['valid'] = False
                self.save_session(session_id, session_data, remote=True)
            request.session['valid'] = False
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
    "https://simplecorehub.acom",
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
    if 'session' in request.scope and 'live_data' in request.session:
        del request.session['live_data']
    return response


def protect_level_test(request):

    if 'live_data' not in request.session:
        return None

    user_level = request.session['live_data'].get('level', -1)
    user_spec = request.session['live_data'].get('spec', 'app')

    if len(request.url.path.split('/')) < 4:
        tb_app.logger.info(f'not protected url {request.url.path}')
        return user_level >= -1

    modul_name = request.url.path.split('/')[2]
    fuction_name = request.url.path.split('/')[3]
    print(tb_app.functions.get(modul_name, {}).keys())
    if not (modul_name in tb_app.functions and fuction_name in tb_app.functions.get(modul_name, {})):
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

    elif url_split[1] == "web" or url_split[1] == "static" or url_split[-1] == "favicon.ico" or "_nicegui" in url_split and url_split[-1] == "codehilite.css" or "_nicegui" in url_split and "static" in url_split or "_nicegui" in url_split and "components" in url_split or "_nicegui" in url_split and "libraries" in url_split or url_split[1] == "api" and url_split[2] in [
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
        with contextlib.suppress(Exception):
            jk = await jk
    def js():
        return jk
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

# Example WebSocket message handler registration
@app.on_event("startup")
async def startup_event():
    pass
    #async def handle_data_update(session_id: str, message: dict):
     #   await app.state.ws_manager.broadcast_to_session(
    #        session_id,
     #       {"type": "data_update", "data": message.get("data")}
     #   )

    #app.state.ws_manager.register_handler("data_update", handle_data_update)


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


# Configure a longer timeout and more robust handling
async def forward_request(port: int, request: Request):
    try:
        # Construct the target URL dynamically
        target_url = f"http://127.0.0.1:{port}{request.url.path.replace(f'/whatsappHook/{port}', '')}"

        # Prepare query parameters
        query_params = request.query_params
        if query_params:
            target_url += '?' + '&'.join(f'{k}={v}' for k, v in query_params.items())

        # Extract method, headers, and body
        method = request.method
        headers = dict(request.headers)
        body = await request.body()

        # Create an async client with extended timeout
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=150.0,  # Connection timeout
                read=300.0,  # Extended read timeout for long-running requests (5 minutes)
                write=150.0,  # Write timeout
                pool=None  # No pool timeout
            )
        ) as client:
            # Forward the request with additional error handling
            try:
                response = await client.request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body,
                )

                # Return the response, handling different content types
                try:
                    return response.json()
                except ValueError:
                    # If not JSON, return text content
                    return response.text

            except httpx.RequestError as e:
                # More specific error handling for network-related issues
                raise HTTPException(
                    status_code=500,
                    detail=f"Request failed: {str(e)}"
                )
            except httpx.HTTPStatusError as e:
                # Handle HTTP error status codes
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"HTTP error: {str(e)}"
                )

    except Exception as e:
        # Catch-all error handling with more detailed logging
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error in webhook forwarding: {str(e)}"
        )

@app.api_route("/whatsappHook/{port}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def webhook_handler(port: int, request: Request):
    return await forward_request(port, request)

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
    if tb_app.mod_online("isaa") and not tb_app.get_mod("isaa").async_initialized:
        await tb_app.get_mod("isaa")
    if id_name.endswith("_D"):
        with BlobFile(f"FastApi/{id_name}/dev", mode='r', storage=BlobStorage(storage_directory=get_app(from_="BlobStorage").data_dir.replace('_D', ''))) as f:
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

    from .fast_api_install import register
    from .fast_api_install import router as install_router
    tb_app.sprint("loading CloudM")
    tb_app.get_mod("CloudM")
    # all_mods = tb_app.get_all_mods()
    os.environ.get("MOD_PROVIDER", default="http://127.0.0.1:5000/")

    def get_d(name="CloudM"):
        return tb_app.get_mod("CloudM").get_mod_snapshot(name)

    install_router.add_api_route('/' + "version", get_d, methods=["GET"], description="get_species_data")
    tb_app.sprint("include Installer")
    app.include_router(install_router)
    nicegui_manager.register_gui("install", register(), install_router.prefix, only_valid=True)

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
            function_data.get('signature')
            state: bool = function_data.get('state')
            api_methods: list[str] = function_data.get('api_methods', ["AUTO"])

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
    from toolboxv2.flows.apiFlow import run as run_flow
    await run_flow(tb_app, app)
    if id_name in tb_app.id:
        print("🟢 START")


print("API: ", __name__)

app.add_middleware(
    BidirectionalStreamlitAppManager,
    streamlit_apps_dir="./apps"
)

app.add_middleware(SessionAuthMiddleware)

app.add_middleware(SessionMiddleware,
                   session_cookie=Code.one_way_hash(tb_app.id, 'session'),
                   https_only='live' in tb_app.id,
                   secret_key=Code.one_way_hash(DEVICE_KEY(), tb_app.id))


nicegui_manager = create_nicegui_manager(app)
print("UI Manager online:", nicegui_manager)

# tb_app.run_a_from_sync(helper, id_name)
asyncio.ensure_future(helper(id_name))

async def serve_files(path: str, request: Request, access_allowed: bool = Depends(lambda: check_access_level(0))):
    return serve_app_func(path)



# print("API: ", __name__)
# if __name__ == 'toolboxv2.api.fast_api_main':
#     global tb_app
