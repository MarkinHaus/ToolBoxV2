#!/usr/bin/env python3
"""
toolbox_native_host.py — Chrome Native Messaging Host für ToolBoxV2

Ermöglicht der Browser Extension, toolboxv2-Module (ISAA, PasswordManager, etc.)
DIREKT aufzurufen — ohne lokalen HTTP-Server, ohne Remote-Verbindung.

PROTOKOLL (Chrome Native Messaging):
    - Eingehend (stdin):  4 Bytes Little-Endian uint32 Länge + JSON-Payload
    - Ausgehend (stdout): 4 Bytes Little-Endian uint32 Länge + JSON-Payload

INSTALLATION:
    python install.py native    # registriert Host in OS
    oder manuell: python toolbox_native_host.py --register

UNTERSTÜTZTE ACTIONS:
    ping                → Verbindungstest
    validate_session    → BlobFile CLI-Session validieren (kein Server!)
    get_session_jwt     → JWT aus CLI-Session für Tauri/HTTP-Backend
    tauri_check         → Prüft ob Tauri Worker auf Port 5000 läuft
    isaa_chat           → ISAA mini_task_completion
    isaa_stream_start   → ISAA Streaming (via connectNative)
    password_get        → PasswordManager get_password_for_autofill
    password_add        → PasswordManager add_password
    password_list       → PasswordManager get_all_passwords
    version_check       → openVersion-äquivalent
"""

import sys
import os
import threading
import time

from toolboxv2.mods.isaa.base.IntelligentRateLimiter import setup_inception_provider

# ─── Windows: Binary-Modus auf stdin/stdout setzen ───────────────────────────
if sys.platform == "win32":
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

# ─── Protokoll-Pipe sichern BEVOR toolboxv2 print()s sie korrumpieren ────────
_proto_fd = os.dup(sys.stdout.fileno())          # fd 1 duplizieren → Protokoll-Kanal
_PROTOCOL_STDOUT = os.fdopen(_proto_fd, 'wb', buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno()) # fd 1 selbst → stderr (OS-Level!)
sys.stdout = sys.stderr                           # Python-Objekt auch

import json
import struct
import asyncio
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional

# ─── Logging (nur stderr — stdout ist reserviert für Chrome) ─────────────────
_log_file = Path(__file__).parent / "native_host_debug.log"
logging.basicConfig(
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
    level=logging.DEBUG,  # DEBUG für maximale Info
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("toolbox_native_host")


# ─── Native Messaging Protokoll ──────────────────────────────────────────────

def read_message() -> Optional[Dict]:
    """Liest eine Native-Messaging-Nachricht von stdin."""
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) == 0:
        return None
    msg_len = struct.unpack("@I", raw_len)[0]
    raw_msg = sys.stdin.buffer.read(msg_len)
    return json.loads(raw_msg.decode("utf-8"))


def send_message(data: Dict) -> None:
    """Sendet eine Native-Messaging-Antwort über den gesicherten Protokoll-fd."""
    encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _PROTOCOL_STDOUT.write(struct.pack("@I", len(encoded)))
    _PROTOCOL_STDOUT.write(encoded)
    _PROTOCOL_STDOUT.flush()


def error_response(msg: str, code: int = 500) -> Dict:
    return {"success": False, "error": msg, "code": code}


# ─── ToolBoxV2 App-Initialisierung ───────────────────────────────────────────

_app = None

def _get_logger():
    return logging.getLogger("toolbox_native_host")

def get_toolbox_app():
    """Singleton: ToolBoxV2 App-Instanz (einmalig initialisiert)."""
    global _app
    if _app is None:
        try:
            from toolboxv2 import get_app, get_logger
            import toolboxv2

            _original_get_logger = toolboxv2.get_logger

            def patch_logger():
                # 1. Modul patchen
                toolboxv2.get_logger = _get_logger

                # 2. Bereits importierte Module fixen
                for module in list(sys.modules.values()):
                    if not module:
                        continue

                    if hasattr(module, "get_logger"):
                        if getattr(module, "get_logger") is _original_get_logger:
                            setattr(module, "get_logger", _get_logger)

            patch_logger()
            _app = get_app("native_host")
            logger.info("ToolBoxV2 app initialized")

            setup_inception_provider()
        except Exception as e:
            logger.error(f"Failed to initialize toolboxv2: {e}")
            raise
    return _app


# ─── Action Handler ──────────────────────────────────────────────────────────

async def handle_ping(app, payload: Dict) -> Dict:
    return {"success": True, "pong": True, "message": "ToolBoxV2 Native Host aktiv"}


async def handle_version_check(app, payload: Dict) -> Dict:
    try:
        result = await app.a_run_any(("CloudM", "openVersion"), get_results=True)
        if not result.is_error():
            return {"success": True, "data": result.get()}
        import toolboxv2
        return {"success": True, "data": {"version": getattr(toolboxv2, "__version__", "unknown")}}
    except Exception as e:
        return {"success": True, "data": {"version": "native", "note": str(e)}}


async def handle_validate_session(app, payload: Dict) -> Dict:
    """
    Validiert die CLI-Session aus BlobFile — KEIN SERVER NÖTIG.
    Funktioniert unabhängig ob Tauri, local HTTP oder Native-Modus.
    """
    try:
        from toolboxv2.mods.CloudM.LogInSystem import (
            _check_existing_session,
            _apply_session_to_app,
        )
        username = payload.get("username")
        session = await _check_existing_session(app, username=username)

        if session:
            _apply_session_to_app(app, session)
            logger.info(f"Session valid for: {session.get('username')}")
            return {
                "success": True,
                "authenticated": True,
                "username": session.get("username", ""),
                "level": session.get("level", 1),
                "user_id": session.get("user_id", ""),
                "provider": session.get("provider", ""),
            }

        return {
            "success": True,
            "authenticated": False,
            "message": "Keine gültige Session. Bitte 'tb login' ausführen.",
        }
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return error_response(f"Session-Validierung fehlgeschlagen: {e}")


async def handle_get_session_jwt(app, payload: Dict) -> Dict:
    """
    Liest das JWT aus der lokalen CLI-Session und gibt es zurück.
    Wird vom background.js verwendet um sich beim Tauri-Worker zu authentifizieren
    (Tauri Worker = HTTP-Worker ohne /dist, kein Web-Login-Endpunkt).

    Flow: Native Host liest BlobFile → extrahiert access_token →
          background.js schickt diesen als Bearer an http://localhost:5000
    """
    try:
        from toolboxv2.mods.CloudM.LogInSystem import (
            _check_existing_session,
            _apply_session_to_app,
        )
        username = payload.get("username")
        session = await _check_existing_session(app, username=username)

        if not session:
            return {"success": False, "error": "Keine CLI-Session gefunden. Bitte 'tb login' ausführen."}

        _apply_session_to_app(app, session)

        # JWT aus Session extrahieren
        access_token = session.get("access_token") or session.get("jwt") or session.get("token")
        if not access_token:
            # Fallback: frischen Token generieren
            try:
                from toolboxv2.mods.CloudM.LogInSystem import _generate_access_token
                access_token = await _generate_access_token(app, session)
            except Exception as e:
                logger.warning(f"Token generation failed: {e}")

        if not access_token:
            return {"success": False, "error": "Kein JWT in CLI-Session gefunden"}

        return {
            "success": True,
            "jwt": access_token,
            "username": session.get("username", ""),
            "level": session.get("level", 1),
        }

    except Exception as e:
        logger.error(f"get_session_jwt error: {e}")
        return error_response(f"JWT-Extraktion fehlgeschlagen: {e}")


async def handle_tauri_check(app, payload: Dict) -> Dict:
    """
    Prüft ob ein Tauri Worker (HTTP-Worker) auf Port 5000 erreichbar ist.
    Wird von background.js DETECT_TAURI Nachricht aufgerufen.
    """
    import urllib.request
    port = payload.get("port", 5000)
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/health",
            method="GET"
        )
        req.add_header("User-Agent", "ToolBoxNativeHost/1.0")
        with urllib.request.urlopen(req, timeout=1) as resp:
            running = resp.status == 200
    except Exception:
        running = False

    return {
        "success": True,
        "tauri_running": running,
        "port": port,
        "url": f"http://localhost:{port}" if running else None,
    }

async def handle_list_agents(app, payload: Dict) -> Dict:
    try:
        result = await app.a_run_any(("isaa", "listAllAgents"), get_results=True)
        if not result.is_error():
            data = result.get()
            return {"success": True, "result": {"data": data}, "data": data}
        agents = list(getattr(app, 'agents', {}).keys()) or ["speed"]
        return {"success": True, "result": {"data": agents}, "data": agents}
    except Exception as e:
        logger.warning(f"listAllAgents fallback: {e}")
        return {"success": True, "result": {"data": ["speed"]}, "data": ["speed"]}

async def handle_format_class(app, payload: Dict) -> Dict:
    from pydantic import BaseModel, create_model
    from typing import Any
    format_schema = payload.get("format_schema")
    task = payload.get("task", "")
    if not format_schema or not task:
        return error_response("'format_schema' und 'task' sind erforderlich", 400)

    try:
        # JSON-Schema → dynamisches Pydantic-Modell
        properties = format_schema.get("properties", {})
        required   = set(format_schema.get("required", []))
        field_defs = {}
        for name, info in properties.items():
            type_map = {"string": str, "integer": int, "boolean": bool,
                        "number": float, "array": list, "object": dict}
            py_type = type_map.get(info.get("type", "string"), Any)
            if name not in required:
                from typing import Optional
                py_type = Optional[py_type]
            field_defs[name] = (py_type, ...)

        DynamicModel = create_model(
            format_schema.get("title", "DynamicSchema"),
            **field_defs
        )
        # log task and full payload
        logger.info(f"format_class {task=}  {payload}")
        result = await app.a_run_any(
            ("isaa", "format_class"),
            get_results=True,
            data={
                "format_schema": DynamicModel,
                "task": task,
                "agent_name": payload.get("agent_name", "speed"),
                "auto_context": payload.get("auto_context", False),
            },
        )

        if result.is_error():
            return error_response(f"format_class Fehler: {getattr(result.info, 'help_text', '')}")

        return {"success": True, "result": {"data": result.get()}, "data": result.get()}

    except Exception as e:
        logger.error(f"handle_format_class error: {e}")
        return error_response(f"format_class nicht verfügbar: {e}")

async def handle_isaa_chat(app, payload: Dict) -> Dict:
    # Alle Feldnamen die popup.js / background.js senden können
    mini_task = payload.get("mini_task") or payload.get("task", "")
    user_task = payload.get("user_task")
    if not user_task or not mini_task:
        return error_response(f"'task' ist erforderlich {not mini_task} {not user_task}", 400)

    try:
        data_dict = {
            "mini_task": mini_task,
            "agent_name": payload.get("agent_name", "speed"),
            "task_from": payload.get("task_from", "system"),
            "message_history": payload.get("message_history"),
            "use_complex": payload.get("use_complex", False),
        }
        if user_task:
            data_dict["user_task"] = user_task

        logger.info(f"handle_isaa_chat: {data_dict}")
        result = await app.a_run_any(
            ("isaa", "mini_task_completion"),
            get_results=True,
            data=data_dict,
        )

        if result.is_error():
            error_info = getattr(getattr(result, "info", None), "help_text", "")
            return error_response(f"ISAA-Fehler: {error_info}")

        return {"success": True, "data": result.get()}

    except Exception as e:
        logger.error(f"ISAA chat error: {e}")
        return error_response(f"ISAA nicht verfügbar: {e}")


async def handle_password_get(app, payload: Dict) -> Dict:
    url = payload.get("url", "")
    if not url:
        return error_response("'url' ist erforderlich", 400)
    try:
        result = await app.a_run_any(
            ("PasswordManager", "get_password_for_autofill"),
            get_results=True, url=url,
        )
        if result.is_error():
            return {"success": False, "data": None, "message": "Kein Passwort gefunden"}
        return {"success": True, "data": result.get()}
    except Exception as e:
        return error_response(f"PasswordManager nicht verfügbar: {e}")


async def handle_password_add(app, payload: Dict) -> Dict:
    try:
        result = await app.a_run_any(
            ("PasswordManager", "add_password"),
            get_results=True, **payload,
        )
        if result.is_error():
            return error_response("Passwort konnte nicht gespeichert werden")
        return {"success": True, "message": "Passwort gespeichert"}
    except Exception as e:
        return error_response(f"PasswordManager nicht verfügbar: {e}")


async def handle_password_list(app, payload: Dict) -> Dict:
    try:
        result = await app.a_run_any(
            ("PasswordManager", "get_all_passwords"),
            get_results=True,
        )
        if result.is_error():
            return {"success": False, "data": []}
        return {"success": True, "data": result.get()}
    except Exception as e:
        return error_response(f"PasswordManager nicht verfügbar: {e}")


async def handle_tts_speak(app, payload: Dict) -> Dict:
    text = payload.get("text", "")
    lang = payload.get("lang", "de").lower()
    if not text:
        return error_response("'text' ist erforderlich", 400)
    try:
        result = await app.a_run_any(("TTS", "speak"), get_results=True, text=text, lang=lang)
        return {"success": not result.is_error(), "data": result.get()}
    except Exception as e:
        return error_response(f"TTS nicht verfügbar: {e}")


# ─── Action Router ───────────────────────────────────────────────────────────

HANDLERS = {
    "ping": handle_ping,
    "version_check": handle_version_check,
    "validate_session": handle_validate_session,
    "get_session_jwt": handle_get_session_jwt,   # NEU: für Tauri-Backend Auth
    "tauri_check": handle_tauri_check,           # NEU: Tauri Worker Detection
    "isaa_chat": handle_isaa_chat,
    "password_get": handle_password_get,
    "password_add": handle_password_add,
    "password_list": handle_password_list,
    "tts_speak": handle_tts_speak,
    "isaa_list_agents": handle_list_agents,
    "isaa_listAllAgents": handle_list_agents,
    "isaa_format_class":         handle_format_class,
    "isaa_format_class_format":  handle_format_class,
    # Aliase
    "CloudM_openVersion": handle_version_check,
    "isaa_mini_task_completion": handle_isaa_chat,
    "PasswordManager_get_password_for_autofill": handle_password_get,
    "PasswordManager_add_password": handle_password_add,
    "TTS_speak": handle_tts_speak,
}


async def dispatch(app, message: Dict) -> Dict:
    action  = message.get("action", "")
    payload = message.get("payload", {})
    call_id = message.get("_callId")

    handler = HANDLERS.get(action)
    if not handler:
        result = error_response(f"Unbekannte Action: '{action}'", 404)
    else:
        try:
            result = await handler(app, payload)
        except Exception as e:
            logger.error(f"Handler error for '{action}': {e}", exc_info=True)
            result = error_response(f"Interner Fehler: {e}")

    if call_id:
        result["_callId"] = call_id
    return result

# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    logger.info("ToolBoxV2 Native Host starting...")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    app = None
    INACTIVITY_TIMEOUT = 5 * 60  # 5 Minuten ohne Nachricht → Exit
    last_message_time = [time.time()]  # mutable für closure

    def inactivity_watchdog():
        while True:
            time.sleep(30)
            if time.time() - last_message_time[0] > INACTIVITY_TIMEOUT:
                logger.info("Inactivity timeout — exiting")
                os._exit(0)

    threading.Thread(target=inactivity_watchdog, daemon=True).start()
    logger.info("Native Host ready — waiting for messages")

    while True:
        try:
            message = read_message()
            if message is None:
                logger.info("stdin closed, exiting")
                break

            last_message_time[0] = time.time()

            if app is None:
                try:
                    app = get_toolbox_app()
                except Exception as e:
                    logger.error(f"Failed to initialize toolboxv2: {e}")
                    send_message(error_response(f"toolboxv2 konnte nicht geladen werden: {e}"))
                    continue

            logger.debug(f"Received: action={message.get('action')}")

            # ISAA (und andere TB-Module) rufen intern run_until_complete() auf.
            # Lösung: Dispatch in eigenem Thread mit eigenem Event Loop —
            # kein nested-loop Konflikt möglich.
            import concurrent.futures
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(dispatch(app, message))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(run_in_thread)
                result = future.result(timeout=120)

            send_message(result)

        except concurrent.futures.TimeoutError:
            logger.error("Dispatch timeout after 120s")
            send_message(error_response("Timeout nach 120s"))
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Message loop error: {e}", exc_info=True)
            try:
                send_message(error_response(f"Nachrichten-Fehler: {e}"))
            except Exception:
                break


# ─── Native Host Registrierung ───────────────────────────────────────────────

def get_host_manifest(extension_id: str = None) -> dict:
    script_path = Path(__file__).absolute()

    if platform.system() == "Windows":
        wrapper = script_path.parent / "toolbox_native_host_wrapper.bat"
        wrapper.write_text(
            f'@echo off\n"{sys.executable}" -u "{script_path}" %*\n',
            encoding="utf-8"
        )
        host_path = str(wrapper)
    else:
        host_path = str(script_path)

    manifest = {
        "name": "com.toolbox.native",
        "description": "ToolBoxV2 Native Messaging Bridge — ISAA & mehr direkt",
        "path": host_path,
        "type": "stdio",
        "allowed_origins": [],
    }

    if extension_id:
        manifest["allowed_origins"].append(f"chrome-extension://{extension_id}/")
    else:
        manifest["allowed_origins"].append("chrome-extension://EXTENSION_ID_HIER/")

    return manifest


def register_host(extension_id: str = None, browser: str = "chrome") -> bool:
    manifest = get_host_manifest(extension_id)
    manifest_json = json.dumps(manifest, indent=2)
    system = platform.system()

    if system == "Windows":
        import winreg
        manifest_path = Path(__file__).parent / "com.toolbox.native.json"
        manifest_path.write_text(manifest_json, encoding="utf-8")

        browsers_registry = {
            "chrome":   r"Software\Google\Chrome\NativeMessagingHosts\com.toolbox.native",
            "chromium": r"Software\Chromium\NativeMessagingHosts\com.toolbox.native",
            "edge":     r"Software\Microsoft\Edge\NativeMessagingHosts\com.toolbox.native",
            "brave":    r"Software\BraveSoftware\Brave-Browser\NativeMessagingHosts\com.toolbox.native",
        }
        key_path = browsers_registry.get(browser, browsers_registry["chrome"])
        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(manifest_path))
            print(f"✅ Registry-Eintrag erstellt: HKCU\\{key_path}")
            return True
        except Exception as e:
            print(f"❌ Registry-Fehler: {e}")
            return False

    elif system == "Darwin":
        browser_dirs = {
            "chrome":   Path.home() / "Library/Application Support/Google/Chrome/NativeMessagingHosts",
            "chromium": Path.home() / "Library/Application Support/Chromium/NativeMessagingHosts",
            "edge":     Path.home() / "Library/Application Support/Microsoft Edge/NativeMessagingHosts",
            "brave":    Path.home() / "Library/Application Support/BraveSoftware/Brave-Browser/NativeMessagingHosts",
        }
        dest_dir = browser_dirs.get(browser, browser_dirs["chrome"])
    else:
        browser_dirs = {
            "chrome":   Path.home() / ".config/google-chrome/NativeMessagingHosts",
            "chromium": Path.home() / ".config/chromium/NativeMessagingHosts",
            "edge":     Path.home() / ".config/microsoft-edge/NativeMessagingHosts",
            "brave":    Path.home() / ".config/BraveSoftware/Brave-Browser/NativeMessagingHosts",
        }
        dest_dir = browser_dirs.get(browser, browser_dirs["chrome"])

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dest_dir / "com.toolbox.native.json"
        manifest_path.write_text(manifest_json, encoding="utf-8")
        Path(__file__).chmod(0o755)
        print(f"✅ Manifest gespeichert: {manifest_path}")
        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--register":
        ext_id = sys.argv[2] if len(sys.argv) > 2 else None
        browser = sys.argv[3] if len(sys.argv) > 3 else "chrome"

        print(f"Registriere Native Messaging Host...")
        print(f"  Extension ID: {ext_id or 'NICHT GESETZT'}")
        print(f"  Browser: {browser}")

        if register_host(ext_id, browser):
            print("\n✅ Native Host erfolgreich registriert!")
            if not ext_id:
                print("\n⚠️  Extension ID noch nicht gesetzt!")
                print("   1. Extension in Chrome installieren")
                print("   2. chrome://extensions/ öffnen")
                print("   3. Extension-ID kopieren")
                print("   4. Erneut ausführen:")
                print("      python toolbox_native_host.py --register <EXTENSION_ID>")
        else:
            print("❌ Registrierung fehlgeschlagen")
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        import time
        import concurrent.futures


        def run_dispatch(app, message):
            """Dispatch in eigenem Thread+Loop — verhindert nested-loop Konflikt mit ISAA."""

            def _in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(dispatch(app, message))
                finally:
                    loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_in_thread).result(timeout=120)


        def test(label, result, check_fn, warn_only=False):
            ok = False
            try:
                ok = check_fn(result)
            except Exception as e:
                print(f"❌ {label}: check threw {e} — result={result}")
                return False
            icon = "✅" if ok else ("⚠️ " if warn_only else f"❌ {ok}")
            print(f"{icon} {label}: {result}")
            return ok or warn_only


        print("🧪 ToolBoxV2 Native Host Selbsttest...")
        print("─" * 60)

        # ── 1. App laden ────────────────────────────────────────────
        try:
            app = get_toolbox_app()
            print("✅ toolboxv2 geladen")
        except Exception as e:
            print(f"❌ toolboxv2 nicht verfügbar: {e}")
            sys.exit(1)

        # ── 2. Basis-Tests (wie bisher) ─────────────────────────────
        print("\n── Basis ───────────────────────────────────────────────")

        r = run_dispatch(app, {"action": "ping", "payload": {}})
        test("ping", r, lambda r: r.get("success") and r.get("pong"))

        r = run_dispatch(app, {"action": "validate_session", "payload": {}})
        authenticated = r.get("authenticated", False)
        username = r.get("username", "N/A")
        test("validate_session", r, lambda r: r.get("success") is not None, warn_only=not authenticated)
        print(f"   → authenticated={authenticated}, user={username}")

        r = run_dispatch(app, {"action": "tauri_check", "payload": {}}),
        test("tauri_check", r[0], lambda r: "tauri_running" in r, warn_only=True)

        r = run_dispatch(app, {"action": "get_session_jwt", "payload": {}})
        test("get_session_jwt", r, lambda r: r.get("success") and r.get("jwt"), warn_only=True)
        print(f"   → jwt={'<vorhanden>' if r.get('jwt') else 'nicht gefunden'}")

        # ── 3. ISAA Chat Test ────────────────────────────────────────
        print("\n── ISAA Chat ───────────────────────────────────────────")

        # 3a. mini_task via korrektem Alias
        print("   Sende einfache Chat-Nachricht (kann 5-30s dauern)...")
        t0 = time.time()
        r = run_dispatch(app, {
            "action": "isaa_mini_task_completion",
            "payload": {
                "mini_task": "Antworte NUR mit dem Wort: PONG",
                "agent_name": "speed",
            }
        })
        elapsed = time.time() - t0
        ok = test(
            f"isaa_mini_task_completion ({elapsed:.1f}s)",
            r,
            lambda r: r.get("success") and r.get("data") and len(str(r.get("data", ""))) > 0
        )
        if ok:
            print(f"   → Antwort: {str(r.get('data', ''))[:120]}")
        else:
            print(f"   → Fehler: {r.get('error', 'keine Antwort')} | success={r.get('success')} | data={r.get('data')}")

        # 3b. Field-Name Varianten (popup.js sendet 'mini_task' + 'user_task')
        print("   Teste field-name Variante 'task'...")
        r2 = run_dispatch(app, {
            "action": "isaa_mini_task_completion",
            "payload": {
                "user_task": "Antworte NUR mit dem Wort: OK",
                "mini_task": "Antworte kurtz",
                "agent_name": "speed",
            }
        })
        ok2 = test(
            "isaa_mini_task_completion via 'task'-field",
            r2,
            lambda r: r.get("success") and r.get("data") and len(str(r.get("data", ""))) > 0
        )
        if ok2:
            print(f"   → Antwort: {str(r2.get('data', ''))[:120]}")

        # 3c. Antwort-Struktur: data darf nicht None sein wenn success=True
        r3 = run_dispatch(app, {
            "action": "isaa_mini_task_completion",
            "payload": {"mini_task": "Sag 'ja'", "user_task":".", "agent_name": "speed"}
        })
        test(
            "isaa: success=True → data nicht None",
            r3,
            lambda r: not (r.get("success") and r.get("data") is None)
        )

        # ── 4. ISAA Agent List ───────────────────────────────────────
        print("\n── ISAA Agents ─────────────────────────────────────────")

        r = run_dispatch(app, {"action": "isaa_listAllAgents", "payload": {}})
        ok = test(
            "isaa_listAllAgents",
            r,
            lambda r: r.get("success") and (
                (r.get("result", {}) or {}).get("data") or r.get("data")
            )
        )
        if ok:
            agents = (r.get("result") or {}).get("data") or r.get("data") or []
            print(f"   → Agents ({len(agents)}): {agents}")
            test(
                "speed-Agent vorhanden",
                agents,
                lambda a: "speed" in a
            )

        # ── 5. ISAA Chat mit History ─────────────────────────────────
        print("\n── ISAA Chat mit History ───────────────────────────────")

        history = [
            {"role": "user", "content": "Mein Name ist TestUser."},
            {"role": "assistant", "content": "Verstanden, hallo TestUser!"},
        ]
        r = run_dispatch(app, {
            "action": "isaa_mini_task_completion",
            "payload": {
                "mini_task": "Antworte in einem Satz.",
                "user_task": "Wie lautet mein Name?",
                "agent_name": "speed",
                "message_history": history,
            }
        })
        ok = test(
            "isaa chat mit message_history",
            r,
            lambda r: r.get("success") and r.get("data") and len(str(r.get("data", ""))) > 0,
            warn_only=True
        )
        if ok:
            print(f"   → Antwort: {str(r.get('data', ''))[:120]}")

        # ── 6. Nested Event Loop Test (kritisch) ─────────────────────
        print("\n── Nested Loop Robustheit ──────────────────────────────")

        results = []
        for i in range(3):
            r = run_dispatch(app, {"action": "ping", "payload": {}})
            results.append(r.get("success", False))
        test(
            "3× sequentielle Dispatches ohne Loop-Fehler",
            results,
            lambda r: all(r)
        )

        # ── 7. TTS Test ──────────────────────────────────────────────
        print("\n── TTS (Text-to-Speech) ────────────────────────────────")

        r = run_dispatch(app, {
            "action": "tts_speak",
            "payload": {"text": "Selbsttest erfolgreich"}
        })
        tts_ok = test(
            "tts_speak → Antwort-Struktur",
            r,
            lambda r: r.get("success") is not None
        )
        if not tts_ok:
            print(f"   → TTS Fehler: {r.get('error', 'unbekannt')}")
            print("   ℹ️  TTS erfordert: pip install pyttsx3 oder espeak (Linux)")

        # Leerer Text → Fehler erwartet
        r_empty = run_dispatch(app, {"action": "tts_speak", "payload": {"text": ""}})
        test(
            "tts_speak mit leerem Text → Fehler",
            r_empty,
            lambda r: not r.get("success"),
            warn_only=True
        )

        # ── Zusammenfassung ──────────────────────────────────────────
        print("\n" + "─" * 60)
        print("✅ Selbsttest abgeschlossen")
        print("   Wichtig: ⚠️  = warnung (kein harter Fehler), ❌ = Bug")

    else:
        main()
