"""
app_singleton.py - Global App Singleton für Nuitka-kompilierte Module

Dieses Modul wird selbst mit Nuitka kompiliert und stellt das globale
App-Singleton für alle anderen Module bereit.

WICHTIG: Wird mit Nuitka kompiliert zu app_singleton.pyd/.so
"""

import sys
import os

# =================== DLL Directory Setup (Windows) ===================
# WICHTIG: Muss VOR allen anderen Imports stehen, damit Python die DLLs findet!
# Dies ist notwendig für native Extensions wie cryptography._rust.pyd

if sys.platform == 'win32':
    # Windows-spezifische DLL Logik
    if hasattr(os, 'add_dll_directory'):
        # Helper um Pfade sicher hinzuzufügen
        def safe_add_dll(path):
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                    print(f"✅ Added DLL directory: {path}")
                except Exception as e:
                    print(f"⚠️ Failed to add DLL directory {path}: {e}")

        # Add venv site-packages cryptography bindings
        safe_add_dll(os.path.join(sys.prefix, 'Lib', 'site-packages', 'cryptography', 'hazmat', 'bindings'))
        # Add venv Scripts
        safe_add_dll(os.path.join(sys.prefix, 'Scripts'))
        # Add base Python DLLs
        safe_add_dll(os.path.join(sys.base_prefix, 'DLLs'))

# =================== Standard Imports ===================

import traceback
import json
from typing import Optional, Dict, Any, List
import asyncio
import uuid

# Importiere toolboxv2
try:
    from toolboxv2 import App, get_app as _get_app
except ImportError as e:
    print(f"FATAL: Failed to import toolboxv2: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# =================== Global State ===================

_GLOBAL_APP: Optional[App] = None
_INSTANCE_ID: str = "nuitka_global"
_GLOBAL_LOOPs: Dict[str, Any] = {}

# =================== LoopHelpers ===================
# Handel multiple loops

def get_loop():
    global _GLOBAL_LOOPs
    for key, (loop, is_running) in _GLOBAL_LOOPs.items():
        if not is_running:
            _GLOBAL_LOOPs[key][1] = True
            return key, loop
    loop = asyncio.new_event_loop()
    key = str(uuid.uuid4())
    _GLOBAL_LOOPs[key] = [loop, False]
    return key, loop

def free_loop(key: str):
    global _GLOBAL_LOOPs
    if key in _GLOBAL_LOOPs:
        _GLOBAL_LOOPs[key][1] = False

# =================== Initialization ===================

def init_app(instance_id: str = "nuitka_global", **kwargs) -> Dict[str, Any]:
    """
    Initialisiert das globale App-Singleton (Cross-Platform).
    """
    global _GLOBAL_APP, _INSTANCE_ID

    try:
        print(f"[app_singleton] init_app() called with instance_id={instance_id}")

        if _GLOBAL_APP is not None:
            return {
                "status": "already_initialized",
                "instance_id": _INSTANCE_ID
            }

        _INSTANCE_ID = instance_id

        # --- Platform Specific Path Setup ---

        # Windows: PyWin32 handling
        if sys.platform == 'win32':
            python_executable = os.environ.get("PYTHON_EXECUTABLE", sys.executable)
            python_home = os.path.dirname(python_executable)

            # Pfad generisch mit os.path.join bauen
            pywin32_dll_path = os.path.join(python_home, "Lib", "site-packages", "pywin32_system32")

            if os.path.exists(pywin32_dll_path):
                if pywin32_dll_path not in sys.path:
                    sys.path.insert(0, pywin32_dll_path)

                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(pywin32_dll_path)
                    except Exception:
                        pass

                # Versuch pywintypes zu importieren (nur auf Windows nötig)
                try:
                    import pywintypes
                except ImportError:
                    pass

        # Erstelle App via toolboxv2
        print(f"[app_singleton] Importing server_helper...")
        from toolboxv2.__main__ import server_helper

        print(f"[app_singleton] Initializing App...")
        _GLOBAL_APP = server_helper(instance_id=instance_id, **kwargs)

        return {
            "status": "success",
            "instance_id": instance_id,
            "platform": sys.platform
        }

    except Exception as e:
        error_msg = f"[app_singleton] FATAL ERROR in init_app(): {e}"
        print(error_msg)
        traceback.print_exc()

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

def get_app() -> App:
    """
    Holt das globale App-Singleton.

    Returns:
        App-Instanz

    Raises:
        RuntimeError: Wenn App nicht initialisiert
    """
    global _GLOBAL_APP

    if _GLOBAL_APP is None:
        raise RuntimeError(
            "App not initialized! Call init_app() first from Rust."
        )

    return _GLOBAL_APP

def reset_app() -> Dict[str, Any]:
    """
    Setzt das Singleton zurück (nur für Tests).

    Returns:
        dict mit Status
    """
    global _GLOBAL_APP, _INSTANCE_ID

    _GLOBAL_APP = None
    _INSTANCE_ID = "nuitka_global"

    return {
        "status": "reset",
        "message": "App singleton has been reset"
    }

# =================== Module Operations ===================
import cProfile
import pstats
import io
import asyncio
import traceback
from typing import Optional, List, Dict, Any

def _profile_execution(func, context_info=""):
    """
    Führt eine Funktion mit cProfile aus und printet die Top-Stats.
    """
    pr = cProfile.Profile()
    pr.enable()
    try:
        result = func()
        return result
    finally:
        pr.disable()
        s = io.StringIO()
        # sort_stats('cumulative') ist hier am wichtigsten:
        # Es zeigt die Zeit inkl. aller Unteraufrufe an -> gut um Bottlenecks zu finden.
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")

        print(f"\n{'=' * 20} PROFILER START: {context_info} {'=' * 20}")
        # Zeige die Top 30 langsamsten Aufrufe
        ps.print_stats(30)
        print(f"{'=' * 20} PROFILER END {'=' * 20}\n")
        print(s.getvalue())

def call_module_function(
    module_name: str,
    function_name: str,
    args: Optional[List] = None,
    kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ruft eine Modul-Funktion über das App-Singleton auf.

    Args:
        module_name: Name des Moduls
        function_name: Name der Funktion
        args: Positionelle Argumente (optional)
        kwargs: Keyword-Argumente (optional)

    Returns:
        dict im ApiResult-Format (kompatibel mit Rust ApiResult struct)
    """
    print(f"[app_singleton] call_module_function CALLED: module={module_name}, function={function_name}")
    try:
        # Auto-initialize app if not already initialized (redundant safety check)
        global _GLOBAL_APP
        if _GLOBAL_APP is None:
            print(f"[app_singleton] call_module_function: App not initialized, calling init_app()...")
            init_result = init_app()
            print(f"[app_singleton] call_module_function: init_app() returned: {init_result}")

            if init_result.get("status") not in ["success", "already_initialized"]:
                error_msg = f"Failed to auto-initialize app: {init_result.get('error', 'Unknown error')}"
                print(f"[app_singleton] ERROR: {error_msg}")
                return {
                    "error": "InternalError",
                    "origin": [module_name, function_name],
                    "result": {
                        "data_to": "REMOTE",
                        "data_info": "App initialization failed",
                        "data": None,
                        "data_type": "NoneType"
                    },
                    "info": {
                        "exec_code": 500,
                        "help_text": error_msg
                    }
                }

        print(f"[app_singleton] call_module_function: Getting app...")
        app = get_app()
        print(f"[app_singleton] call_module_function: Got app successfully")

        # Konvertiere args/kwargs
        args = args or ()
        kwargs = kwargs or {}

        # Rufe Funktion auf
        # WICHTIG: Wir werden von Rust aus aufgerufen, NICHT aus einem async Context
        # Daher können wir NICHT run_coroutine_threadsafe verwenden (Deadlock!)
        # Stattdessen rufen wir die Funktion SYNCHRON auf

        if "args" in kwargs:
            args_ = kwargs.pop("args")

            if args_ and args:
                args += args_
            if args_ and not args:
                args = args_

            if 'spec' in kwargs:
                kwargs['tb_run_with_specification'] = kwargs.pop('spec')
        # result = app.run(*args, mod_function_name=(module_name, function_name), request=kwargs.pop('request') if 'request' in kwargs else None, **kwargs)
        print(f"[CALL] {module_name}.{function_name} with args={args} and kwargs={kwargs}")
        def _perform_execution():
            # Async call - wir müssen einen neuen Event Loop erstellen
            # NICHT den existierenden Loop verwenden (Deadlock!)
            print(f"[app_singleton] call_module_function: app.a_run_any is async, creating new event loop")
            key, loop = get_loop()
            asyncio.set_event_loop(loop)
            try:
                # WICHTIG: get_results=True damit wir das vollständige Result-Objekt bekommen
                # Ohne get_results=True gibt a_run_any nur res.get() zurück (extrahierte Daten)
                return loop.run_until_complete(
                    app.a_run_any((module_name, function_name), *args, get_results=True, **kwargs)
                )
            finally:
                free_loop(key)
                asyncio.set_event_loop(None)

        is_debug = getattr(app, "debug", False) or os.getenv("PROFILING", "false") == "true"

        if is_debug:
            result = _profile_execution(
                _perform_execution, context_info=f"{module_name}.{function_name}"
            )
        else:
            result = _perform_execution()

        # Konvertiere Result zu ApiResult-kompatiblem Format
        print(f"Result: {str(result)}")
        res = _convert_result_to_api_result(result, module_name, function_name)
        print(f"ApiResult: {str(res)}")
        return res

    except Exception as e:
        error_msg = f"Error calling {module_name}.{function_name}: {e}"
        print(error_msg)
        traceback.print_exc()
        # Return error in ApiResult format with proper result structure
        return {
            "error": "InternalError",
            "origin": [module_name, function_name],
            "result": {
                "data_to": "REMOTE",
                "data_info": "Error occurred",
                "data": None,
                "data_type": "NoneType"
            },
            "info": {
                "exec_code": 500,
                "help_text": error_msg
            }
        }


def _convert_result_to_api_result(result: Any, module_name: str, function_name: str) -> Dict[str, Any]:
    """
    Konvertiert ein Python Result-Objekt zu einem ApiResult-kompatiblen Dict.

    Args:
        result: Das Result-Objekt von app.run_any
        module_name: Name des Moduls (für origin)
        function_name: Name der Funktion (für origin)

    Returns:
        dict im ApiResult-Format
    """
    # Prüfe, ob result ein Result-Objekt ist
    if hasattr(result, 'to_api_result'):
        # Result hat to_api_result Methode - nutze sie
        api_result = result.to_api_result()
        if hasattr(api_result, 'model_dump'):
            # Pydantic BaseModel
            return api_result.model_dump()
        elif hasattr(api_result, 'dict'):
            # Pydantic v1
            return api_result.dict()
        elif hasattr(api_result, '__dict__'):
            return api_result.__dict__
        else:
            return {
                "error": None,
                "origin": [module_name, function_name],
                "result": {
                    "data_to": "REMOTE",
                    "data_info": "Converted result",
                    "data": str(api_result),
                    "data_type": "str"
                },
                "info": {
                    "exec_code": 0,
                    "help_text": "OK"
                }
            }
    elif hasattr(result, 'as_dict'):
        # Result hat as_dict Methode
        result_dict = result.as_dict()
        # Stelle sicher, dass origin gesetzt ist
        if result_dict.get('origin') is None:
            result_dict['origin'] = [module_name, function_name]
        return result_dict
    elif hasattr(result, '__dict__'):
        # Generisches Objekt mit __dict__
        return {
            "error": None,
            "origin": [module_name, function_name],
            "result": {
                "data_to": "REMOTE",
                "data_info": "Converted result",
                "data": result.__dict__,
                "data_type": "dict"
            },
            "info": {
                "exec_code": 0,
                "help_text": "OK"
            }
        }
    else:
        # Primitiver Wert oder unbekannter Typ
        return {
            "error": None,
            "origin": [module_name, function_name],
            "result": {
                "data_to": "REMOTE",
                "data_info": "Direct result",
                "data": result,
                "data_type": type(result).__name__
            },
            "info": {
                "exec_code": 0,
                "help_text": "OK"
            }
        }


def get_loaded_modules() -> Dict[str, Any]:
    """
    Holt Liste aller geladenen Module.

    Returns:
        dict mit Modul-Informationen
    """
    try:
        app = get_app()

        if hasattr(app, 'get_all_mods'):
            modules = app.get_all_mods()
            return {
                "status": "success",
                "modules": [str(m) for m in modules],
                "count": len(modules)
            }
        else:
            return {
                "status": "success",
                "modules": [],
                "count": 0,
                "message": "App has no get_all_mods method"
            }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# =================== Health Check ===================

def health_check() -> Dict[str, Any]:
    """
    Einfacher Health-Check.

    Returns:
        dict mit Health-Status
    """
    try:
        app = get_app()

        # Sammle Informationen
        modules_count = 0
        if hasattr(app, 'get_all_mods'):
            modules_count = len(app.get_all_mods())

        return {
            "status": "healthy",
            "app_initialized": True,
            "instance_id": _INSTANCE_ID,
            "modules_loaded": modules_count,
            "python_version": sys.version,
        }

    except RuntimeError as e:
        return {
            "status": "unhealthy",
            "app_initialized": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# =================== Debugging ===================

def get_app_info() -> Dict[str, Any]:
    """
    Holt detaillierte App-Informationen für Debugging.

    Returns:
        dict mit App-Details
    """
    try:
        app = get_app()

        info = {
            "status": "success",
            "instance_id": _INSTANCE_ID,
            "app_type": str(type(app)),
            "app_attributes": dir(app),
            "python_version": sys.version,
            "python_path": sys.path,
        }

        # Versuche zusätzliche Infos zu sammeln
        if hasattr(app, 'id'):
            info["app_id"] = app.id

        if hasattr(app, 'get_all_mods'):
            info["modules"] = [str(m) for m in app.get_all_mods()]

        return info

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# =================== JSON Helpers ===================

def json_call(json_str: str) -> str:
    """
    Ruft eine Funktion via JSON-String auf.
    Nützlich für einfache FFI-Integration.

    Args:
        json_str: JSON string mit {"module": "...", "function": "...", "kwargs": {...}}

    Returns:
        JSON string mit Ergebnis
    """
    try:
        data = json.loads(json_str)

        module_name = data.get("module")
        function_name = data.get("function")
        kwargs = data.get("kwargs", {})

        if not module_name or not function_name:
            return json.dumps({
                "status": "error",
                "error": "Missing module or function name"
            })

        result = call_module_function(module_name, function_name, kwargs=kwargs)
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# =================== app_singleton.py ADDITIONS ===================
# Add these functions and classes to app_singleton.py

# Add to imports at the top:
# import aiohttp  # for async HTTP calls (optional, for non-blocking WS)

# =================== Global WS Bridge State ===================
# Add after the existing global state variables around line 60

_RUST_WS_BRIDGE_ENABLED: bool = False
_WS_SERVER_URL: str = os.getenv("APP_BASE_URL", "http://localhost:8080") # Configurable server URL

# =================== WebSocket Context Class ===================
# Add this class for WebSocket handler context

# =================== WebSocket Context Class ===================


class WebSocketContext:
    """
    Context object passed to WebSocket handlers.
    Contains connection information and authenticated session data.
    """

    def __init__(
        self,
        conn_id: str,
        channel_id: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        cookies: Optional[Dict[str, Any]] = None,
    ):
        self.conn_id = conn_id
        self.channel_id = channel_id
        # 'user' enthält die validierten User-Daten, die von on_connect zurückkamen
        self.user = user or {}
        # Die Session-ID (aus Cookie oder Header)
        self.session_id = session_id
        # Raw Headers und Cookies (hauptsächlich für on_connect relevant)
        self.headers = headers or {}
        self.cookies = cookies or {}

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> "WebSocketContext":
        """
        Creates a WebSocketContext robustly from arguments passed by Rust.
        Rust passes 'session_data' (stored context) and request info.
        """
        # 1. Versuche, persistierte Session-Daten zu finden (von on_message)
        session_data = kwargs.get("session_data", {})
        if not session_data and "session" in kwargs:
            session_data = kwargs.get("session", {})

        # 2. Extrahiere spezifische Felder
        conn_id = kwargs.get("conn_id", "")
        channel_id = kwargs.get("channel_id")

        # User-Daten kommen entweder direkt oder aus dem session_data blob
        user = (
            session_data.get("user") if isinstance(session_data, dict) else session_data
        )

        # 3. Request-Daten (Headers/Cookies) - meist nur bei on_connect verfügbar
        headers = kwargs.get("headers", {})
        cookies = kwargs.get("cookies", {})

        # Fallback: Session ID aus Cookies holen, wenn nicht explizit übergeben
        s_id = session_data.get("session_id")
        if not s_id and isinstance(cookies, dict):
            s_id = cookies.get("session_id") or cookies.get("id")

        return cls(
            conn_id=conn_id,
            channel_id=channel_id,
            user=user if isinstance(user, dict) else {},
            session_id=s_id,
            headers=headers if isinstance(headers, dict) else {},
            cookies=cookies if isinstance(cookies, dict) else {},
        )

    @property
    def is_authenticated(self) -> bool:
        """Returns True if the connection has a valid user ID."""
        return bool(self.user and (self.user.get("id") or self.user.get("user_id")))

    @property
    def user_id(self) -> Optional[str]:
        """Helper to get the user ID agnostic of key naming."""
        return self.user.get("id") or self.user.get("user_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conn_id": self.conn_id,
            "user": self.user,
            "session_id": self.session_id,
            "authenticated": self.is_authenticated,
        }


# =================== Enhanced WS Bridge Functions ===================
# Replace the existing ws_send_message and ws_broadcast_message functions


def set_ws_server_url(url: str) -> Dict[str, Any]:
    """
    Sets the WebSocket server URL.

    Args:
        url: The base URL of the Rust server (e.g., "http://localhost:8080")

    Returns:
        dict with status
    """
    global _WS_SERVER_URL
    _WS_SERVER_URL = url.rstrip("/")
    print(f"[app_singleton] WebSocket server URL set to: {_WS_SERVER_URL}")
    return {"status": "success", "url": _WS_SERVER_URL}


def set_rust_ws_bridge() -> Dict[str, Any]:
    """
    Initializes and sets up the Rust WebSocket bridge.
    Called by Rust after app initialization.

    Returns:
        dict with status
    """
    global _RUST_WS_BRIDGE_ENABLED

    print("[app_singleton] set_rust_ws_bridge() called")

    _RUST_WS_BRIDGE_ENABLED = True

    # Inject the bridge into the App
    try:
        app = get_app()
        if hasattr(app, "_set_rust_ws_bridge"):
            # Create a bridge object that the Python App can use
            class RustWsBridgeWrapper:
                """Wrapper that provides async WebSocket methods for Python code."""

                async def send_message(self, conn_id: str, payload: str):
                    """Sends a message to a single WebSocket connection."""
                    return await ws_send_message(conn_id, payload)

                async def broadcast_message(
                    self,
                    channel_id: str,
                    payload: str,
                    source_conn_id: str = "python_broadcast",
                ):
                    """Broadcasts a message to all clients in a channel."""
                    return await ws_broadcast_message(channel_id, payload, source_conn_id)

                async def broadcast_all(
                    self, payload: str, source_conn_id: Optional[str] = None
                ):
                    """Broadcasts a message to ALL connected clients."""
                    return await ws_broadcast_all(payload, source_conn_id)

                async def is_connected(self, conn_id: str) -> bool:
                    """Checks if a connection is active."""
                    return await ws_is_connected(conn_id)

                async def get_status(self) -> Dict[str, Any]:
                    """Gets the WebSocket status (connection count, etc.)."""
                    return await ws_get_status()

            bridge_wrapper = RustWsBridgeWrapper()
            app._set_rust_ws_bridge(bridge_wrapper)
            print(f"[app_singleton] Rust WebSocket bridge injected into App successfully")

            return {
                "status": "success",
                "message": "Rust WebSocket bridge enabled and injected into App",
            }
        else:
            print(
                f"[app_singleton] WARNING: App does not have _set_rust_ws_bridge method"
            )
            # Still mark as enabled - direct functions can be used
            return {
                "status": "partial",
                "message": "Rust WebSocket bridge enabled but App._set_rust_ws_bridge not available",
            }
    except Exception as e:
        error_msg = f"Error setting up Rust WebSocket bridge: {e}"
        print(f"[app_singleton] ERROR: {error_msg}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc(),
        }


    """import aiohttp

    async def ws_send_message_async(conn_id: str, payload: str) -> Dict[str, Any]:
        '''Async version of ws_send_message using aiohttp.'''
        print(f"[app_singleton] ws_send_message_async() called: conn_id={conn_id}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{_WS_SERVER_URL}/internal/ws/send",
                    json={"conn_id": conn_id, "payload": payload},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return {"status": "success", "conn_id": conn_id}
                    else:
                        text = await response.text()
                        return {"status": "error", "message": text}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    async def ws_broadcast_message_async(
        channel_id: str,
        payload: str,
        source_conn_id: str = "python_broadcast"
    ) -> Dict[str, Any]:
        '''Async version of ws_broadcast_message using aiohttp.'''
        print(f"[app_singleton] ws_broadcast_message() called: channel_id={channel_id}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{_WS_SERVER_URL}/internal/ws/broadcast",
                    json={
                        "channel_id": channel_id,
                        "payload": payload,
                        "source_conn_id": source_conn_id
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return {"status": "success", "channel_id": channel_id}
                    else:
                        text = await response.text()
                        return {"status": "error", "message": text}
        except Exception as e:
            return {"status": "error", "message": str(e)}"""
# =================== Enhanced WS Bridge Functions ===================


async def ws_send_message(conn_id: str, payload: str) -> Dict[str, Any]:
    """
    Sends a WebSocket message via the Rust internal HTTP server.
    """
    try:
        # Payload muss ein String (JSON) sein
        if not isinstance(payload, str):
            payload = json.dumps(payload)

        url = f"/internal/ws/send"
        if get_app().session.base !=_WS_SERVER_URL:
            get_app().session.base = _WS_SERVER_URL
        response = await get_app().session.fetch(url, method="POST", json={"conn_id": conn_id, "payload": payload})

        status_code = response.status_code if hasattr(response, 'status_code') else response.status
        if status_code == 200:
            return {"status": "success", "conn_id": conn_id}
        return {"status": "error", "code": status_code, "msg": response.text}

    except Exception as e:
        print(f"[app_singleton] ERROR sending WS message: {e}")
        return {"status": "error", "message": str(e)}


async def ws_broadcast_message(
    channel_id: str, payload: str, source_conn_id: str = "python_broadcast"
) -> Dict[str, Any]:
    """
    Broadcasts to a specific channel via Rust.
    """
    try:
        if not isinstance(payload, str):
            payload = json.dumps(payload)

        url = f"/internal/ws/broadcast"
        if get_app().session.base !=_WS_SERVER_URL:
            get_app().session.base = _WS_SERVER_URL
        response = await get_app().session.fetch(
            url,
            method="POST",
            json={
                "channel_id": channel_id,
                "payload": payload,
                "source_conn_id": source_conn_id,
            },
            timeout=2,
        )

        status_code = response.status_code if hasattr(response, 'status_code') else response.status
        if status_code == 200:
            return {"status": "success", "conn_id": channel_id}
        return {"status": "error", "code": status_code, "msg": response.text}

    except Exception as e:
        print(f"[app_singleton] ERROR broadcasting: {e}")
        return {"status": "error", "message": str(e)}


async def ws_broadcast_all(
    payload: str, source_conn_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Global broadcast via Rust.
    """
    try:
        if not isinstance(payload, str):
            payload = json.dumps(payload)

        url = f"/internal/ws/broadcast_all"
        if get_app().session.base !=_WS_SERVER_URL:
            get_app().session.base = _WS_SERVER_URL
        response = await get_app().session.fetch(
            url,
            method="POST",
            json={"payload": payload, "source_conn_id": source_conn_id or "global"},
            timeout=2,
        )

        status_code = response.status_code if hasattr(response, 'status_code') else response.status
        if status_code == 200:
            return {"status": "success"}
        return {"status": "error", "code": status_code, "msg": response.text}

    except Exception as e:
        print(f"[app_singleton] ERROR global broadcast: {e}")
        return {"status": "error", "message": str(e)}


async def ws_is_connected(conn_id: str) -> bool:
    """
    Checks if a WebSocket connection is active.

    Args:
        conn_id: The connection ID to check

    Returns:
        True if the connection is active
    """

    try:
        if get_app().session.base !=_WS_SERVER_URL:
            get_app().session.base = _WS_SERVER_URL
        response = await get_app().session.fetch(
            f"/internal/ws/check/{conn_id}", timeout=5
        )
        if  response.status_code if hasattr(response, 'status_code') else response.status == 200:
            data = response.json()
            return data.get("connected", False)
        return False

    except Exception as e:
        print(
            f"[app_singleton] ws_is_connected: Error checking connection {conn_id}: {e}"
        )
        return False


async def ws_get_status() -> Dict[str, Any]:
    """
    Gets the current WebSocket status from the Rust server.

    Returns:
        dict with active_connections count and connection_ids
    """

    try:
        if get_app().session.base !=_WS_SERVER_URL:
            get_app().session.base = _WS_SERVER_URL
        response = await get_app().session.fetch(f"/internal/ws/status", timeout=5)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "message": f"Failed to get status: {response.text}",
            }

    except Exception as e:
        return {"status": "error", "message": f"Error getting WebSocket status: {e}"}


def ws_is_bridge_enabled() -> bool:
    """
    Returns True if the Rust WebSocket bridge is enabled.
    """
    return _RUST_WS_BRIDGE_ENABLED



# =================== Main (für Testing) ===================

if __name__ == "__main__":
    print("=== App Singleton Test ===")

    # Test 1: Init
    print("\n1. Initializing App...")
    result = init_app("test_instance")
    print(f"Result: {json.dumps(result, indent=2)}")

    # Test 2: Health Check
    print("\n2. Health Check...")
    health = health_check()
    print(f"Health: {json.dumps(health, indent=2)}")

    # Test 3: Get Info
    print("\n3. App Info...")
    info = get_app_info()
    print(f"Info: {json.dumps(info, indent=2)}")

    # Test 4: Reset
    print("\n4. Reset...")
    reset_result = reset_app()
    print(f"Reset: {json.dumps(reset_result, indent=2)}")

    print("\n=== Tests Complete ===")

