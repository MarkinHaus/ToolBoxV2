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

if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    # Add venv site-packages cryptography bindings
    venv_crypto = os.path.join(sys.prefix, 'Lib', 'site-packages', 'cryptography', 'hazmat', 'bindings')
    if os.path.exists(venv_crypto):
        try:
            os.add_dll_directory(venv_crypto)
            print(f"✅ Added DLL directory: {venv_crypto}")
        except Exception as e:
            print(f"⚠️ Failed to add DLL directory {venv_crypto}: {e}")

    # Add venv Scripts
    venv_scripts = os.path.join(sys.prefix, 'Scripts')
    if os.path.exists(venv_scripts):
        try:
            os.add_dll_directory(venv_scripts)
            print(f"✅ Added DLL directory: {venv_scripts}")
        except Exception as e:
            print(f"⚠️ Failed to add DLL directory {venv_scripts}: {e}")

    # Add base Python DLLs
    base_dlls = os.path.join(sys.base_prefix, 'DLLs')
    if os.path.exists(base_dlls):
        try:
            os.add_dll_directory(base_dlls)
            print(f"✅ Added DLL directory: {base_dlls}")
        except Exception as e:
            print(f"⚠️ Failed to add DLL directory {base_dlls}: {e}")

# =================== Standard Imports ===================

import traceback
import json
from typing import Optional, Dict, Any, List
import asyncio

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

# =================== Initialization ===================

def init_app(instance_id: str = "nuitka_global", **kwargs) -> Dict[str, Any]:
    """
    Initialisiert das globale App-Singleton.

    Args:
        instance_id: Eindeutige ID für diese App-Instanz
        **kwargs: Zusätzliche Argumente für App-Initialisierung

    Returns:
        dict mit Status-Informationen
    """
    global _GLOBAL_APP, _INSTANCE_ID

    try:
        print(f"[app_singleton] init_app() called with instance_id={instance_id}, kwargs={kwargs}")

        if _GLOBAL_APP is not None:
            print(f"[app_singleton] App already initialized with instance_id={_INSTANCE_ID}")
            return {
                "status": "already_initialized",
                "instance_id": _INSTANCE_ID,
                "message": "App singleton already exists"
            }

        # Setze Instance ID
        _INSTANCE_ID = instance_id
        print(f"[app_singleton] Instance ID set to: {_INSTANCE_ID}")

        # Add pywin32 DLL path to sys.path for pywintypes
        import sys

        # Use PYTHON_EXECUTABLE environment variable to get the correct Python home
        python_executable = os.environ.get("PYTHON_EXECUTABLE")
        if python_executable:
            python_home = os.path.dirname(python_executable)
        else:
            # Fallback to sys.executable (which might be the Rust executable)
            python_home = os.path.dirname(sys.executable)

        pywin32_dll_path = os.path.join(python_home, "Lib", "site-packages", "pywin32_system32")
        print(f"[app_singleton] DEBUG: python_executable = {python_executable}")
        print(f"[app_singleton] DEBUG: python_home = {python_home}")
        print(f"[app_singleton] DEBUG: pywin32_dll_path = {pywin32_dll_path}")
        print(f"[app_singleton] DEBUG: os.path.exists(pywin32_dll_path) = {os.path.exists(pywin32_dll_path)}")
        print(f"[app_singleton] DEBUG: pywin32_dll_path in sys.path = {pywin32_dll_path in sys.path}")

        # ALWAYS add the path, regardless of whether it exists or is already in sys.path
        if pywin32_dll_path not in sys.path:
            sys.path.insert(0, pywin32_dll_path)
            print(f"[app_singleton] Added pywin32 DLL path to sys.path: {pywin32_dll_path}")
        else:
            print(f"[app_singleton] pywin32 DLL path already in sys.path: {pywin32_dll_path}")

        # Add DLL directory for Windows (Python 3.8+)
        if hasattr(os, 'add_dll_directory') and os.path.exists(pywin32_dll_path):
            try:
                print(f"[app_singleton] Adding DLL directory: {pywin32_dll_path}")
                os.add_dll_directory(pywin32_dll_path)
                print(f"[app_singleton] DLL directory added successfully!")
            except Exception as e:
                print(f"[app_singleton] WARNING: Failed to add DLL directory: {e}")

        # Try to import pywintypes directly to ensure it's loaded
        try:
            print(f"[app_singleton] Attempting to import pywintypes directly...")
            import pywintypes
            print(f"[app_singleton] pywintypes imported successfully!")
        except ImportError as e:
            print(f"[app_singleton] WARNING: Failed to import pywintypes: {e}")
            print(f"[app_singleton] Continuing anyway - mcp_server import might fail...")

        # Erstelle App via toolboxv2
        # NOTE: Do NOT add -l flag here! Modules will be loaded AFTER init_app() returns
        # to avoid race condition where modules call get_app() before _GLOBAL_APP is set

        print(f"[app_singleton] Importing server_helper from toolboxv2.__main__...")
        from toolboxv2.__main__ import server_helper
        print(f"[app_singleton] server_helper imported successfully")

        print(f"[app_singleton] Calling server_helper(instance_id={instance_id}, kwargs={kwargs})...")
        print(f"[app_singleton] NOTE: Modules will be loaded AFTER this returns to avoid race condition")

        _GLOBAL_APP = server_helper(instance_id=instance_id, **kwargs)

        print(f"[app_singleton] server_helper() returned successfully!")
        print(f"[app_singleton] App type: {type(_GLOBAL_APP)}")

        return {
            "status": "success",
            "instance_id": instance_id,
            "python_version": sys.version,
            "app_type": str(type(_GLOBAL_APP)),
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
        if asyncio.iscoroutinefunction(app.a_run_any):
            # Async call - wir müssen einen neuen Event Loop erstellen
            # NICHT den existierenden Loop verwenden (Deadlock!)
            print(f"[app_singleton] call_module_function: app.a_run_any is async, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # WICHTIG: get_results=True damit wir das vollständige Result-Objekt bekommen
                # Ohne get_results=True gibt a_run_any nur res.get() zurück (extrahierte Daten)
                result = loop.run_until_complete(
                    app.a_run_any((module_name, function_name), *args, get_results=True, **kwargs)
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        else:
            # Sync call
            print(f"[app_singleton] call_module_function: app.a_run_any is sync")
            # WICHTIG: get_results=True damit wir das vollständige Result-Objekt bekommen
            result = app.a_run_any(
                (module_name, function_name),
                *args,
                get_results=True,
                **kwargs
            )

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

# =================== WebSocket Bridge ===================

# Globale Referenz zur Rust WebSocket Bridge
_RUST_WS_BRIDGE_ENABLED = False

def enable_rust_ws_bridge() -> Dict[str, Any]:
    """
    Aktiviert die Rust WebSocket Bridge.
    Diese Funktion wird von Rust aufgerufen, um die Bridge zu initialisieren.

    Die Bridge verwendet externe Rust-Funktionen, die über ctypes aufgerufen werden.

    Returns:
        dict mit Status
    """
    global _RUST_WS_BRIDGE_ENABLED

    print("[app_singleton] enable_rust_ws_bridge() called")

    _RUST_WS_BRIDGE_ENABLED = True

    # Injiziere die Bridge in die App
    try:
        app = get_app()
        if hasattr(app, '_set_rust_ws_bridge'):
            # Erstelle ein Bridge-Objekt, das die Python App verwenden kann
            # Die tatsächlichen Rust-Funktionen werden über ws_send_message() und
            # ws_broadcast_message() aufgerufen, die von Rust bereitgestellt werden
            class RustWsBridgeWrapper:
                async def send_message(self, conn_id: str, payload: str):
                    """Sendet eine Nachricht an eine einzelne WebSocket-Verbindung."""
                    # Rufe die Rust-Funktion auf
                    ws_send_message(conn_id, payload)

                async def broadcast_message(self, channel_id: str, payload: str, source_conn_id: str = "python_broadcast"):
                    """Sendet eine Nachricht an alle Clients in einem Kanal."""
                    # Rufe die Rust-Funktion auf
                    ws_broadcast_message(channel_id, payload, source_conn_id)

            bridge_wrapper = RustWsBridgeWrapper()
            app._set_rust_ws_bridge(bridge_wrapper)
            print(f"[app_singleton] Rust WebSocket bridge injected into App successfully")

            return {
                "status": "success",
                "message": "Rust WebSocket bridge enabled successfully"
            }
        else:
            print(f"[app_singleton] WARNING: App does not have _set_rust_ws_bridge method")
            return {
                "status": "warning",
                "message": "App does not have _set_rust_ws_bridge method"
            }
    except Exception as e:
        error_msg = f"Error enabling Rust WebSocket bridge: {e}"
        print(f"[app_singleton] ERROR: {error_msg}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def ws_send_message(conn_id: str, payload: str):
    """
    Sendet eine WebSocket-Nachricht an eine einzelne Verbindung über den Rust-Server.
    """
    import requests
    print(f"[app_singleton] ws_send_message() called: conn_id={conn_id}, payload_len={len(payload)}")
    try:
        response = requests.post(
            "http://localhost:8080/internal/ws/send",
            json={"conn_id": conn_id, "payload": payload},
            timeout=5
        )
        if response.status_code == 200:
            print(f"[app_singleton] ws_send_message: Message sent to {conn_id}")
        else:
            print(f"[app_singleton] ws_send_message: Failed to send message to {conn_id}: {response.text}")
    except Exception as e:
        print(f"[app_singleton] ws_send_message: Error sending message to {conn_id}: {e}")


def ws_broadcast_message(channel_id: str, payload: str, source_conn_id: str = "python_broadcast"):
    """
    Sendet eine WebSocket-Broadcast-Nachricht an einen Kanal über den Rust-Server.
    """
    import requests
    print(f"[app_singleton] ws_broadcast_message() called: channel_id={channel_id}, source_conn_id={source_conn_id}, payload_len={len(payload)}")
    try:
        response = requests.post(
            "http://localhost:8080/internal/ws/broadcast",
            json={"channel_id": channel_id, "payload": payload, "source_conn_id": source_conn_id},
            timeout=5
        )
        if response.status_code == 200:
            print(f"[app_singleton] ws_broadcast_message: Broadcast sent to channel {channel_id}")
        else:
            print(f"[app_singleton] ws_broadcast_message: Failed to broadcast to channel {channel_id}: {response.text}")
    except Exception as e:
        print(f"[app_singleton] ws_broadcast_message: Error broadcasting to channel {channel_id}: {e}")


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

