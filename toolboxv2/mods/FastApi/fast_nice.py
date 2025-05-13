import asyncio
import contextlib
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from nicegui import ui
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

from toolboxv2 import Singleton, get_app
from toolboxv2.utils.extras.base_widget import get_s_id, get_spec, get_user_from_request
from toolboxv2.utils.system.session import RequestSession


@dataclass
class UIEndpoint:
    path: str
    title: str
    description: str = ""
    show: bool = True
    only_valid: bool = False
    only_root: bool = False


class NiceGUIManager(metaclass=Singleton):
    init = False
    def __init__(self, fastapi_app: FastAPI = None, styles_path: str = "./web/assets/styles.css"):

        if fastapi_app is None:
            return None
        self.admin_password = os.getenv("TB_R_KEY", "root@admin")
        self.app = fastapi_app
        self.styles_path = styles_path
        self.registered_guis: dict[str, dict[str, Any]] = {}
        self.ws_connections: dict[str, dict[str, WebSocket]] = {}
        self.mount_path = "/gui"
        self.endpoints: list[UIEndpoint] = []

        self.helper_contex = open("./dist/helper.html", encoding="utf-8").read()

        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.middleware_dispatch)

        # Add WebSocket endpoint
        self.app.websocket("/ws/{session_id}/{gui_id}")(self.websocket_endpoint)
        self._setup_admin_gui()
        self._setup_endpoints_api()

    def _setup_endpoints_api(self):
        @self.app.get("/api/CloudM/openui")
        def get_ui_endpoints(request: Request) -> list[dict]:
            def _(endpoint):
                add_true = True
                if endpoint.only_valid:
                    add_true = request.session['valid']

                if add_true and endpoint.only_root:
                    add_true = request.session.get('live_data', {}).get('user_name') == 'root'
                return add_true
            return [{"path": endpoint.path,
    "title": endpoint.title,
    "description": endpoint.description} for endpoint in self.endpoints if endpoint.show and _(endpoint)]

    def _setup_admin_gui(self):
        """Setup the admin GUI interface"""

        @ui.page('/admin')
        def admin_gui(user=None):
            print("admin_gui;", user)
            if user is None or user.name != "root":
                return

            with ui.card().style("background-color: var(--background-color) !important").classes('w-full'):
                ui.label('NiceGUI Manager Admin Interface').classes('text-2xl font-bold mb-4')

                # GUI Management Section
                with ui.tabs().style("background-color: var(--background-color) !important") as tabs:
                    ui.tab('Registered GUIs')
                    ui.tab('Add New GUI')
                    ui.tab('System Status')

                with ui.tab_panels(tabs, value='Registered GUIs').style(
                    "background-color: var(--background-color) !important"):
                    with ui.tab_panel('Registered GUIs'):
                        self._show_registered_guis()

                    with ui.tab_panel('Add New GUI'):
                        self._show_add_gui_form()

                    with ui.tab_panel('System Status'):
                        self._show_system_status()

        self.register_gui("admin", admin_gui, "/admin", only_root=True)

    def _show_registered_guis(self):
        """Show list of registered GUIs with management options"""
        with ui.column().classes('w-full gap-4'):
            for gui_id, gui_info in self.registered_guis.items():
                with ui.card().classes('w-full').style("background-color: var(--background-color) !important"):
                    with ui.row().classes('w-full items-center justify-between').style(
                        "background-color: var(--background-color) !important"):
                        ui.label(f'GUI ID: {gui_id}').classes('font-bold')
                        ui.label(f'Path: {gui_info["path"]}')

                        created_at = gui_info['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                        ui.label(f'Created: {created_at}')

                        with ui.row().classes('gap-2').style("background-color: var(--background-color) !important"):
                            ui.button('View', on_click=lambda g=gui_info['path']: ui.navigate.to(g))
                            ui.button('Remove', on_click=lambda g=gui_id: self._handle_gui_removal(g))
                            ui.button('Restart', on_click=lambda g=gui_id: self._handle_gui_restart(g))

                    # Show connection status
                    active_connections = sum(
                        1 for connections in self.ws_connections.values()
                        if gui_id in connections
                    )
                    ui.label(f'Active Connections: {active_connections}')

    def _show_add_gui_form(self):
        """Show form for adding new GUI"""
        with ui.card().classes('w-full').style("background-color: var(--background-color) !important"):
            gui_id = ui.input('GUI ID').classes('w-full')
            mount_path = ui.input('Mount Path (optional)').classes('w-full')

            # Code editor for GUI setup
            code_editor = ui.editor(
                value='def setup_gui():\n    ui.label("New GUI")\n',
            ).classes('w-full h-64')

            def add_new_gui():
                try:
                    # Create setup function from code
                    setup_code = code_editor.value
                    setup_namespace = {}
                    exec(setup_code, {'ui': ui}, setup_namespace)
                    setup_func = setup_namespace['setup_gui']

                    # Register the new GUI
                    self.register_gui(
                        gui_id.value,
                        setup_func,
                        mount_path.value if mount_path.value else None
                    )

                    ui.notify('GUI added successfully')
                    ui.navigate.to('admin')  # Refresh page
                except Exception as e:
                    ui.notify(f'Error adding GUI: {str(e)}', color='negative')

            ui.button('Add GUI', on_click=add_new_gui).classes('w-full mt-4')

    def _show_system_status(self):
        """Show system status information"""
        with ui.card().classes('w-full').style("background-color: var(--background-color) !important"):
            ui.label('System Status').classes('text-xl font-bold mb-4')

            # System stats
            ui.label(f'Total GUIs: {len(self.registered_guis)}')
            ui.label(f'Total WebSocket Connections: {sum(len(conns) for conns in self.ws_connections.values())}')

            # Memory usage
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            ui.label(f'Memory Usage: {memory_usage:.2f} MB')

            # Add refresh button
            ui.button('Refresh Stats', on_click=lambda: ui.navigate.to('/admin'))

    def _handle_gui_removal(self, gui_id: str):
        """Handle GUI removal with confirmation"""

        def confirm_remove():
            if self.remove_gui(gui_id):
                ui.notify(f'GUI {gui_id} removed successfully')
                ui.navigate.to('/admin')  # Refresh page
            else:
                ui.notify('Error removing GUI', color='negative')

        ui.notify('Are you sure?',
                  actions=[{'label': 'Yes', 'on_click': confirm_remove},
                           {'label': 'No'}])

    def _handle_gui_restart(self, gui_id: str):
        """Handle GUI restart"""
        try:
            if gui_id in self.registered_guis:
                gui_info = self.registered_guis[gui_id]
                # Re-register the GUI with the same setup
                self.register_gui(gui_id, gui_info['setup'], gui_info['path'])
                ui.notify(f'GUI {gui_id} restarted successfully')
            else:
                ui.notify('GUI not found', color='negative')
        except Exception as e:
            ui.notify(f'Error restarting GUI: {str(e)}', color='negative')

    def _load_styles(self) -> str:
        """Load custom styles from CSS file"""
        try:
            with open(self.styles_path) as f:
                return f.read()
        except Exception as e:
            print(f"Error loading styles: {e}")
            return ""

    def register_gui(self, gui_id: str, setup_func: Callable, mount_path: str | None = None, additional: str | None = None, title: str | None = None , description: str | None = None, **kwargs) -> None:
        """Register a new NiceGUI application"""
        path = mount_path or f"/{gui_id}"
        self.endpoints.append(UIEndpoint(path=self.mount_path+path, title=title if title is not None else path.replace('/', '') , description=description if description is not None else '', **kwargs))
        if additional is None:
            additional = ""

        def has_parameters(func, *params):
            """
            Überprüft, ob die Funktion bestimmte Parameter hat.

            :param func: Die zu analysierende Funktion.
            :param params: Eine Liste der zu suchenden Parameter.
            :return: Ein Dictionary mit den Parametern und einem booleschen Wert.
            """
            signature = inspect.signature(func)
            func_params = signature.parameters.keys()
            return {param: param in func_params for param in params}

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

        get_app()

        @ui.page(path)
        async def wrapped_gui(request: Request):
            # Inject custom styles
            ui.add_body_html(self.helper_contex + additional)
            # ui.switch('Dark').bind_value(ui, 'dark_mode')
            # ui.add_css("q-card {background-color: var(--background-color)} !important")
            # ui.add_body_html('<script src="../index.js" type="module" defer></script>')

            # Initialize the GUI
            params_ = {}
            params = has_parameters(setup_func, 'request', 'user', 'session', 'id', 'sid')

            if params.get('request'):
                params_['request'] = await request_to_request_session(request)
            if params.get('user'):
                params_['user'] = await get_user_from_request(get_app(), request)
            if params.get('session'):
                params_['session'] = request.session
            if params.get('spec'):
                params_['spec'] = get_spec(request)
            if params.get('sid'):
                params_['sid'] = get_s_id(request)

            async def task():
                if asyncio.iscoroutine(setup_func):

                    # Event Listener für Button hinzufügen
                    await ui.run_javascript('''
                            Quasar.Dark.set("auto");
                            tailwind.config.darkMode = "media";
                        ''')

                    await ui.run_javascript("""
                    document.getElementById('darkModeToggle').addEventListener('click', function () {
                    const labelToggel = document.getElementById('toggleLabel')
                    if (labelToggel.innerHTML == `<span class="material-symbols-outlined">
dark_mode
</span>`){
                            Quasar.Dark.set(true);
                            tailwind.config.darkMode = "class";
                            document.body.classList.add("dark");
                        }else{
                            Quasar.Dark.set(false);
                            tailwind.config.darkMode = "class"
                            document.body.classList.remove("dark");
                        }
                    });
                    """)

                    if not params_:
                        await setup_func()
                    else:
                        await setup_func(**params_)
                else:
                    if not params_:
                        setup_func()
                    else:
                        setup_func(**params_)




            await task()
            # return result

        self.registered_guis[gui_id] = {
            'path': path,
            'setup': setup_func,
            'created_at': datetime.now()
        }

        print("Registered GUI:", self.registered_guis[gui_id])
        return True

    def remove_gui(self, gui_id: str) -> bool:
        """Remove a registered GUI application"""
        if gui_id in self.registered_guis:
            # Remove from registry
            del self.registered_guis[gui_id]

            # Clean up any WebSocket connections
            for session_id in self.ws_connections:
                if gui_id in self.ws_connections[session_id]:
                    del self.ws_connections[session_id][gui_id]

            return True
        return False

    async def websocket_endpoint(self, websocket: WebSocket, session_id: str, gui_id: str):
        """Handle WebSocket connections for real-time updates"""
        await websocket.accept()

        if session_id not in self.ws_connections:
            self.ws_connections[session_id] = {}
        self.ws_connections[session_id][gui_id] = websocket

        try:
            while True:
                data = await websocket.receive_json()
                # Handle incoming WebSocket messages
                await self.handle_ws_message(session_id, gui_id, data)
        except WebSocketDisconnect:
            if session_id in self.ws_connections:
                if gui_id in self.ws_connections[session_id]:
                    del self.ws_connections[session_id][gui_id]

    async def handle_ws_message(self, session_id: str, gui_id: str, message: dict):
        """Handle incoming WebSocket messages"""
        # Implement custom WebSocket message handling
        if message.get('type') == 'update':
            # Broadcast updates to all connected clients for this GUI
            await self.broadcast_to_gui(gui_id, {
                'type': 'update',
                'data': message.get('data')
            })

    async def broadcast_to_gui(self, gui_id: str, message: dict):
        """Broadcast a message to all sessions connected to a specific GUI"""
        for session_connections in self.ws_connections.values():
            if gui_id in session_connections:
                await session_connections[gui_id].send_json(message)

    async def middleware_dispatch(self, request: Request, call_next) -> Response:
        """Custom middleware for session handling and authentication"""
        async def callN():
            response = await call_next(request)
            return response

        if not request.url.path.startswith(self.mount_path):
            return await callN()

        if request.url.path.endswith("/favicon.ico"):
            return await callN()
        if "_nicegui" in request.url.path and "static" in request.url.path:
            return await callN()
        if "_nicegui" in request.url.path and "components" in request.url.path:
            return await callN()
        if "_nicegui" in request.url.path and "codehilite" in request.url.path:
            return await callN()
        if "_nicegui" in request.url.path and "libraries" in request.url.path:
            return await callN()

        if "open" in request.url.path:
            return await callN()

        # Verify session if needed
        if not request.session.get("valid", False):
            return RedirectResponse(f"/web/login?next={request.url.path}")

        response = await call_next(request)
        return response

    def init_app(self) -> None:
        """Initialize the FastAPI application with NiceGUI integration"""
        self.init = True
        ui.run_with(
            self.app,
            mount_path=self.mount_path,
            favicon=os.getenv("FAVI"), # "/root/Toolboxv2/toolboxv2/favicon.ico"
            show_welcome_message=False,
            # prod_js=False,
        )


manager_online = [False]


# Usage example:
def create_nicegui_manager(app: FastAPI, token_secret: str | None = None) -> NiceGUIManager:
    """Create and initialize a NiceGUI manager instance"""
    manager = NiceGUIManager(app, token_secret)
    manager.init_app()
    manager_online[0] = True
    return manager


def register_nicegui(gui_id: str, setup_func: Callable, mount_path: str | None = None, additional: str | None = None, **kwargs) -> None:
    if not manager_online[0]:
        return
    print("ADDED GUI:", gui_id)
    return NiceGUIManager().register_gui(gui_id, setup_func, mount_path, additional=additional, **kwargs)
