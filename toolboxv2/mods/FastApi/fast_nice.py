from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from nicegui import app as nicegui_app, ui
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from typing import Dict, Optional, Callable, Any
from datetime import datetime
import os

from toolboxv2 import Singleton


class NiceGUIManager(metaclass=Singleton):
    def __init__(self, fastapi_app: FastAPI, styles_path: str = "./web/assets/styles.css"):
        self.admin_password = os.getenv("TB_R_KEY", "root@admin")
        self.app = fastapi_app
        self.styles_path = styles_path
        self.registered_guis: Dict[str, Dict[str, Any]] = {}
        self.ws_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.mount_path = "/gui"
        self.init = False

        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.middleware_dispatch)

        # Add WebSocket endpoint
        self.app.websocket("/ws/{session_id}/{gui_id}")(self.websocket_endpoint)
        self._setup_admin_gui()

    def _setup_admin_gui(self):
        """Setup the admin GUI interface"""

        @ui.page(f'{self.mount_path}/admin')
        def admin_gui():
            if not nicegui_app.storage.user.get('is_admin', False):
                self._show_admin_login()
                return

            with ui.card().classes('w-full'):
                ui.label('NiceGUI Manager Admin Interface').classes('text-2xl font-bold mb-4')

                # Dark mode toggle
                with ui.row().classes('items-center mb-4'):
                    ui.icon('dark_mode')
                    ui.switch('Dark Mode').bind_value(nicegui_app.storage.user, 'dark_mode')

                # GUI Management Section
                with ui.tabs() as tabs:
                    ui.tab('Registered GUIs')
                    ui.tab('Add New GUI')
                    ui.tab('System Status')

                with ui.tab_panels(tabs, value='Registered GUIs'):
                    with ui.tab_panel('Registered GUIs'):
                        self._show_registered_guis()

                    with ui.tab_panel('Add New GUI'):
                        self._show_add_gui_form()

                    with ui.tab_panel('System Status'):
                        self._show_system_status()

    def _show_admin_login(self):
        """Show admin login form"""
        with ui.card().classes('w-full max-w-md mx-auto mt-8'):
            ui.label('Admin Login').classes('text-xl font-bold mb-4')

            password = ui.input('Password', password=True).classes('w-full')

            def try_login():
                if password.value == self.admin_password:
                    nicegui_app.storage.user['is_admin'] = True
                    ui.notify('Login successful')
                    ui.navigate(f'{self.mount_path}/admin')  # Refresh page
                else:
                    ui.notify('Invalid password', color='negative')

            ui.button('Login', on_click=try_login).classes('w-full mt-4')

    def _show_registered_guis(self):
        """Show list of registered GUIs with management options"""
        with ui.column().classes('w-full gap-4'):
            for gui_id, gui_info in self.registered_guis.items():
                with ui.card().classes('w-full'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label(f'GUI ID: {gui_id}').classes('font-bold')
                        ui.label(f'Path: {gui_info["path"]}')

                        created_at = gui_info['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                        ui.label(f'Created: {created_at}')

                        with ui.row().classes('gap-2'):
                            ui.button('View', on_click=lambda g=gui_info['path']: ui.navigate(g))
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
        with ui.card().classes('w-full'):
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
                    ui.navigate(f'{self.mount_path}/admin')  # Refresh page
                except Exception as e:
                    ui.notify(f'Error adding GUI: {str(e)}', color='negative')

            ui.button('Add GUI', on_click=add_new_gui).classes('w-full mt-4')

    def _show_system_status(self):
        """Show system status information"""
        with ui.card().classes('w-full'):
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
            ui.button('Refresh Stats', on_click=lambda: ui.navigate(f'{self.mount_path}/admin'))

    def _handle_gui_removal(self, gui_id: str):
        """Handle GUI removal with confirmation"""

        def confirm_remove():
            if self.remove_gui(gui_id):
                ui.notify(f'GUI {gui_id} removed successfully')
                ui.navigate(f'{self.mount_path}/admin')  # Refresh page
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
            with open(self.styles_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading styles: {e}")
            return ""

    def register_gui(self, gui_id: str, setup_func: Callable, mount_path: Optional[str] = None) -> None:
        """Register a new NiceGUI application"""
        path = mount_path or f"{self.mount_path}/{gui_id}"

        @ui.page(path)
        def wrapped_gui():
            # Inject custom styles
            ui.html(f'<style>{self._load_styles()}</style>')

            # Add dark mode toggle
            with ui.row().classes('items-center'):
                ui.icon('dark_mode')
                ui.switch('Dark Mode')

            # Initialize the GUI
            setup_func()

        self.registered_guis[gui_id] = {
            'path': path,
            'setup': setup_func,
            'created_at': datetime.now()
        }

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
        if request.url.path.startswith(self.mount_path):
            # Verify session if needed
            if not request.session.get("valid", False):
                return JSONResponse(
                    status_code=401,
                    content={"message": "Invalid session"}
                )

        response = await call_next(request)
        return response

    def init_app(self) -> None:
        """Initialize the FastAPI application with NiceGUI integration"""
        self.init = True
        ui.run_with(
            self.app,
            mount_path=self.mount_path
        )


is_gui_online = [False]


# Usage example:
def create_nicegui_manager(app: FastAPI, token_secret: Optional[str] = None) -> NiceGUIManager:
    """Create and initialize a NiceGUI manager instance"""
    is_gui_online[0] = True
    manager = NiceGUIManager(app, token_secret)
    manager.init_app()
    return manager


def get_nicegui_manager() -> NiceGUIManager:
    if is_gui_online[0]:
        return NiceGUIManager(..., ...)
    raise ValueError("Gui not online")
