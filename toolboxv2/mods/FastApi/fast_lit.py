import asyncio
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Any

import jwt
from fastapi import FastAPI, Request, Response, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from toolboxv2 import Singleton

host = os.getenv("HOSTNAME", "localhost")


class StreamlitAppManager:
    def __init__(self):
        self.active_apps = {}
        self.port_counter = 8501  # Starting port for Streamlit apps
        # self.ws_manager = WebSocketManager()

    async def start_app(self, app_path: str, session_id: str):
        if session_id not in self.active_apps:
            port = self.port_counter
            self.port_counter += 1

            # Start Streamlit process with session information
            process = subprocess.Popen([sys.executable, '-m',
                                        "streamlit", "run", app_path,
                                        "--server.port", str(port),
                                        "--server.address", host,
                                        "--server.headless", "true",
                                        "--server.runOnSave", "false"
                                        ])

            self.active_apps[session_id] = {
                "process": process,
                "port": port,
                "start_time": datetime.now()
            }

            # Wait for Streamlit to start
            await asyncio.sleep(2)
            return port
        return self.active_apps[session_id]["port"]

    def cleanup_inactive_apps(self, max_age: timedelta = timedelta(hours=1)):
        current_time = datetime.now()
        for session_id, app_info in list(self.active_apps.items()):
            if current_time - app_info["start_time"] > max_age:
                app_info["process"].terminate()
                del self.active_apps[session_id]


class WebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, dict[str, WebSocket]] = {}
        self.message_handlers = {}

    async def connect(self, websocket: WebSocket, session_id: str, app_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}
        self.active_connections[session_id][app_id] = websocket

    async def disconnect(self, session_id: str, app_id: str):
        if session_id in self.active_connections:
            if app_id in self.active_connections[session_id]:
                del self.active_connections[session_id][app_id]
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for websocket in self.active_connections[session_id].values():
                await websocket.send_json(message)

    def register_handler(self, message_type: str, handler):
        self.message_handlers[message_type] = handler

    async def handle_message(self, session_id: str, message: dict):
        message_type = message.get('type')
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](session_id, message)


class APIRequestHelper:
    def __init__(self, token_secret: str):
        self.token_secret = token_secret

    async def make_api_request(self, endpoint: str, method: str, data: dict | None = None,
                               headers: dict | None = None, session_token: str | None = None) -> Any:
        """
        Make API requests while maintaining session context
        """
        import httpx

        if headers is None:
            headers = {}

        if session_token:
            try:
                session_data = jwt.decode(session_token, self.token_secret, algorithms=["HS256"])
                headers['X-Session-ID'] = session_data.get('session_id')
                headers['Authorization'] = f'Bearer {session_token}'
            except jwt.InvalidTokenError:
                raise ValueError("Invalid session token")

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=endpoint,
                json=data,
                headers=headers
            )

            return response.json()


class BidirectionalStreamlitAppManager(BaseHTTPMiddleware, metaclass=Singleton):
    def __init__(self, app: FastAPI, streamlit_apps_dir: str = "./apps"):
        super().__init__(app)
        self.streamlit_manager = StreamlitAppManager()
        self.streamlit_apps_dir = streamlit_apps_dir
        self.token_secret = os.getenv("TOKEN_SECRET", "your-secret-key")
        self.api_helper = APIRequestHelper(self.token_secret)

        # Run cleanup task
        asyncio.create_task(self.periodic_cleanup())

    #def add_ws(self, fast_app):
        # Register WebSocket routes
     #   fast_app.add_api_websocket_route("/ws/{session_id}/{app_id}", self.websocket_endpoint, "StWebSocket")

    async def periodic_cleanup(self):
        while True:
            self.streamlit_manager.cleanup_inactive_apps()
            await asyncio.sleep(3600)

    def create_streamlit_token(self, session_data: dict, app_name: str) -> str:
        payload = {
            "app_name": app_name,
            "session_id": session_data.get("ID"),
            "user_data": session_data.get("live_data"),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, self.token_secret, algorithm="HS256")

    #async def websocket_endpoint(self, websocket: WebSocket, session_id: str, app_id: str):
    #    await self.streamlit_manager.ws_manager.connect(websocket, session_id, app_id)
    #    try:
    #        while True:
    #            message = await websocket.receive_json()
    #            await self.streamlit_manager.ws_manager.handle_message(session_id, message)
    #    except WebSocketDisconnect:
    #        await self.streamlit_manager.ws_manager.disconnect(session_id, app_id)

    async def resolve_session_token(self, request: Request) -> str | None:
        """
        Extract and validate session token from request
        """
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            token = request.query_params.get('token')

        if token:
            try:
                jwt.decode(token, self.token_secret, algorithms=["HS256"])
                return token
            except jwt.InvalidTokenError:
                return None
        return None

    async def dispatch(self, request: Request, call_next) -> Response:
        # Handle API routes with session token resolution
        if request.url.path.startswith("/api/"):
            session_token = await self.resolve_session_token(request)
            if session_token:
                # Inject session data into request state
                request.state.session_token = session_token
                request.state.api_helper = self.api_helper

        # Handle Streamlit routes
        elif request.url.path.startswith("/apps/"):
            app_name = request.url.path.split("/")[-1]
            app_path = os.path.join(self.streamlit_apps_dir, f"{app_name}.py")

            # Verify session is valid
            if 'public' not in app_name and not request.session.get("valid", False):
                return JSONResponse(
                    status_code=401,
                    content={"message": "Invalid session"}
                )

            if not os.path.exists(app_path):
                return JSONResponse(
                    status_code=401,
                    content={"message": "no app found"}
                )

            streamlit_token = self.create_streamlit_token(request.session, app_name)
            port = await self.streamlit_manager.start_app(app_path, request.session.get("ID")+app_name)
            streamlit_url = f"http://{host}:{port}?token={streamlit_token}"
            return RedirectResponse(url=streamlit_url)

        resposee = await call_next(request)
        return resposee


async def make_api_request(endpoint: str, method: str = "GET", data: dict | None = None):
    """Helper function for making API requests from Streamlit apps"""
    import streamlit as st

    if not hasattr(st.session_state, 'token'):
        st.error("No valid session token found")
        st.stop()

    headers = {
        'Authorization': f'Bearer {st.session_state.token}',
        'Content-Type': 'application/json'
    }

    try:
        api_helper = APIRequestHelper(os.getenv("TOKEN_SECRET", "your-secret-key"))
        response = await api_helper.make_api_request(
            endpoint=endpoint,
            method=method,
            data=data,
            headers=headers,
            session_token=st.session_state.token
        )
        return response
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return None


# Streamlit authentication helper (to be used in Streamlit apps)
def verify_streamlit_session():
    import jwt
    import streamlit as st

    # Get token from URL
    token = st.query_params.get("token", [None])[0]

    if not token:
        st.error("No valid session found")
        st.stop()

    try:
        # Verify token
        secret_key = os.getenv("TOKEN_SECRET", "your-secret-key")
        session_data = jwt.decode(token, secret_key, algorithms=["HS256"])

        # Store session data in Streamlit session state
        st.session_state.user_data = session_data.get("user_data")
        st.session_state.session_id = session_data.get("session_id")
        return True
    except jwt.ExpiredSignatureError:
        st.error("Session expired")
        st.stop()
    except jwt.InvalidTokenError:
        st.error("Invalid session")
        st.stop()


def inject_custom_css(css_file_path="./web/assets/styles.css"):
    """
    Liest eine CSS-Datei ein und injiziert sie in die Streamlit-App.
    """
    import streamlit as st
    try:
        with open(css_file_path) as f:
            css_content = f.read()

        # CSS in einen <style>-Tag einbetten
        css_injection = f"<style>{css_content}</style>"

        # CSS in Streamlit injizieren
        st.markdown(css_injection, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Fehler beim Laden des CSS: {e}")

    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)


# Example Streamlit app using the new features
"""
# example_streamlit_app.py
import streamlit as st
from bidirectional_manager import verify_streamlit_session, make_api_request
import asyncio

# Verify session
if verify_streamlit_session():
    # Make API request
    async def fetch_data():
        data = await make_api_request("/api/get_user_data")
        if data:
            st.write(data)

    if st.button("Fetch Data"):
        asyncio.run(fetch_data())

    # WebSocket connection (using streamlit-websocket-client)
    from streamlit_websocket_client import websocket_client

    ws = websocket_client(f"ws://localhost:8000/ws/{st.session_state.session_id}/example_app")

    if ws.connected():
        if st.button("Send Update"):
            ws.send_json({
                "type": "data_update",
                "data": {"message": "Update from Streamlit!"}
            })
"""
