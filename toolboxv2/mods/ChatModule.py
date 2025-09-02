# toolboxv2/mods/chat_module.py

from toolboxv2 import App, get_app
from toolboxv2.utils.system.types import Result

# -----------------
# Setup & Metadaten
# -----------------
# Diese Einrichtung ist notwendig, damit das Modul von der App-Instanz erkannt wird.
app = get_app("ChatModule")
export = app.tb
Name = "ChatModule"
version = "1.0.0"


# --------------------
# WebSocket-Handler
# --------------------
# Dies sind die asynchronen Kernfunktionen, die WebSocket-Ereignisse verarbeiten.

async def on_user_connect(app: App, conn_id: str, session: dict):
    """
    Wird vom Rust WebSocket Actor aufgerufen, wenn ein neuer Client eine Verbindung herstellt.
    """
    username = session.get("user_name", "Anonymous")
    app.print(f"WS CONNECT: User '{username}' connected with conn_id: {conn_id}")

    # Sende eine Willkommensnachricht direkt an den neuen Benutzer (1-zu-1)
    await app.ws_send(conn_id, {"event": "welcome", "data": f"Welcome to the public chat, {username}!"})

    # Kündige den neuen Benutzer allen anderen im Raum an (1-zu-n)
    await app.ws_broadcast(
        channel_id="ChatModule/public_room",
        payload={"event": "user_joined", "data": f"👋 {username} has joined the chat."},
        source_conn_id=conn_id  # Schließt den Absender von diesem Broadcast aus
    )


async def on_chat_message(app: App, conn_id: str, session: dict, payload: dict):
    """
    Wird aufgerufen, wenn eine Nachricht von einem Client empfangen wird.
    """
    username = session.get("user_name", "Anonymous")
    print(f"WS MESSAGE from {username} ({conn_id}): {session}")
    message_text = payload.get("data", {}).get("message", "").strip()

    if not message_text:
        return  # Ignoriere leere Nachrichten

    app.print(f"WS MESSAGE from {username} ({conn_id}): {message_text}")

    # Sende die Nachricht an alle im Raum (einschließlich des Absenders)
    await app.ws_broadcast(
        channel_id="ChatModule/public_room",
        payload={"event": "new_message", "data": {"user": username, "text": message_text}}
    )


async def on_user_disconnect(app: App, conn_id: str, session: dict=None):
    """
    Wird aufgerufen, wenn die Verbindung eines Clients geschlossen wird.
    """
    if session is None:
        session = {}
    username = session.get("user_name", "Anonymous")
    app.print(f"WS DISCONNECT: User '{username}' disconnected (conn_id: {conn_id})")

    # Kündige den Weggang des Benutzers allen verbleibenden Benutzern im Raum an
    await app.ws_broadcast(
        channel_id="ChatModule/public_room",
        payload={"event": "user_left", "data": f"😥 {username} has left the chat."}
    )


# ----------------------------------------
# WebSocket-Handler-Registrierung
# ----------------------------------------
# Diese spezielle Funktion verwendet den neuen `websocket_handler`-Parameter des Decorators.
# Sie teilt toolboxv2 mit, dass dies keine normale API-Funktion ist, sondern eine
# Einrichtungsroutine für einen WebSocket-Endpunkt. Der String "public_room" wird Teil der URL
# und der Kanal-ID.

# init
@export(mod_name=Name, version=version, initial=True)
def init_chat_module(app: App) -> Result:
    app.run_any(("CloudM", "add_ui"), name=Name, title="Public Chat",
                path=f"/api/{Name}/ui", description="A public chat room for all users.")
    return Result.ok(info="ChatModule initialized and UI registered.")

@export(mod_name=Name, version=version, websocket_handler="public_room")
def register_chat_handlers(app: App) -> dict:
    """
    Registriert die asynchronen Funktionen als Handler für spezifische WebSocket-Ereignisse.
    Der Funktionsname (`register_chat_handlers`) ist beliebig. Der Decorator ist entscheidend.

    Returns:
        Ein Dictionary, das Ereignisnamen auf ihre Handler-Funktionen abbildet.
    """
    return {
        "on_connect": on_user_connect,
        "on_message": on_chat_message,
        "on_disconnect": on_user_disconnect,
    }


# -----------------
# HTML-UI-Endpunkt
# -----------------
# Diese Standard-API-Funktion liefert die HTML-Seite für die Chat-Anwendung.

@export(mod_name=Name, version=version, api=True, name="ui", row=True)
def get_chat_ui(app: App) -> Result:
    """
    Liefert das Haupt-HTML-UI für das Chat-Widget.
    Es verwendet `app.web_context()`, um das notwendige tbjs CSS und JS einzubinden.
    """

    html_content = f"""
        {app.web_context()}
        <style>
            body {{
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                padding: 1rem;
                background-color: var(--theme-bg);
            }}
        </style>
        <main id="chat-container" style="width: 100%; height: 80vh;">
            <!-- Das Chat-Widget wird hier initialisiert -->
        </main>

        <script unsave="true">
            // Verwende TB.once, um sicherzustellen, dass das Framework vollständig initialisiert ist,
            // bevor unser Code ausgeführt wird.
            TB.once(() => {{
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer && TB.ui.ChatWidget) {{
                    // Initialisiere das Chat-Widget in unserem Container
                    TB.ui.ChatWidget.init(chatContainer);

                    // Verbinde mit dem in diesem Modul definierten WebSocket-Endpunkt
                    TB.ui.ChatWidget.connect();
                }} else {{
                    console.error("Chat UI initialization failed: container or ChatWidget not found.");
                }}
            }});
        </script>
    """

    return Result.html(data=html_content)
