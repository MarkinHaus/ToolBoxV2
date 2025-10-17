# test_registry.py

import asyncio
import json
from typing import Any

import aiohttp

# Wichtiger Hinweis: Stellen Sie sicher, dass Ihr PYTHONPATH so konfiguriert ist,
# dass er Ihr toolboxv2-Verzeichnis findet.
from toolboxv2 import get_app
from toolboxv2.mods.isaa.module import Tools as ISAA_Tools

# --- Globale Objekte zur Kommunikation zwischen den Tasks ---
# Dient zur Synchronisation, damit der Endbenutzer-Test erst startet, wenn der Agent publiziert ist.
published_event = asyncio.Event()
# Speichert die öffentlichen Agenten-Infos, die der Client vom Server erhält.
published_info: dict[str, Any] = {}


# --------------------------------------------------------------------------
# 1. SIMULATION: DER ÖFFENTLICHE REGISTRY SERVER
# --------------------------------------------------------------------------
async def run_registry_server():
    """Startet die erste toolboxv2-Instanz als unseren öffentlichen Server."""
    print("--- [SERVER] Initialisiere Registry Server Instanz ---")

    # Holt sich eine App-Instanz. Das Laden des 'registry'-Moduls geschieht
    # automatisch durch die __init__.py-Struktur von toolboxv2.
    server_app = get_app("RegistryServerInstance")

    # Startet den actix-web Server auf Port 8080.
    # `blocking=False` ist entscheidend, damit asyncio weiterlaufen kann.
    server_app.start_server()

    print("--- [SERVER] Registry Server läuft auf http://127.0.0.1:8080 ---")
    print("--- [SERVER] Wartet auf eingehende Client-Verbindungen... ---")

    # Hält diesen Task am Leben, um den Server laufen zu lassen.
    await asyncio.Future()


# --------------------------------------------------------------------------
# 2. SIMULATION: DER LOKALE CLIENT MIT DEM ISAA AGENTEN
# --------------------------------------------------------------------------
async def run_local_client():
    """Startet die zweite toolboxv2-Instanz als lokalen Client, der einen Agenten hostet."""
    print("--- [CLIENT] Initialisiere lokale Client Instanz ---")
    client_app = get_app("LocalClientInstance")

    # ISAA-Modul für diese Instanz holen und initialisieren
    isaa: ISAA_Tools = client_app.get_mod("isaa")
    await isaa.init_isaa()
    print("--- [CLIENT] ISAA initialisiert. ---")

    # --- Agenten erstellen ---
    print("--- [CLIENT] Erstelle einen einfachen 'EchoAgent'... ---")
    builder = isaa.get_agent_builder("EchoAgent")
    builder.with_system_message("You are an echo agent. Repeat the user's query exactly, but prefix it with 'Echo: '.")
    await isaa.register_agent(builder)

    # Agenten-Instanz holen (dieser Schritt ist nicht zwingend für das Publizieren per Name, aber gut zur Demo)
    echo_agent = await isaa.get_agent("EchoAgent")
    print(f"--- [CLIENT] 'EchoAgent' ({type(echo_agent).__name__}) erstellt. ---")

    # --- Agenten publizieren ---
    # Warten, bis der Server sicher läuft
    await asyncio.sleep(2)

    server_ws_url = "ws://127.0.0.1:8080/ws/registry/connect"
    print(f"--- [CLIENT] Publiziert 'EchoAgent' am Server: {server_ws_url} ---")

    # Die neue `publish_agent` Methode aufrufen
    reg_info = await isaa.host_agent_ui(
        agent=echo_agent,
        public_name="Public Echo Service",
        server_url=server_ws_url,
        description="A simple agent that echoes your input."
    )

    if reg_info:
        print("--- [CLIENT] Agent erfolgreich publiziert! Details erhalten: ---")
        print(f"  > Public URL: {reg_info.public_url}")
        print(f"  > API Key: {reg_info.public_api_key}")

        # Speichere die Info und signalisiere dem Endbenutzer-Task, dass er starten kann
        published_info.update(reg_info.model_dump())
        published_event.set()
    else:
        print("--- [CLIENT] FEHLER: Agenten-Publizierung fehlgeschlagen. ---", file=sys.stderr)

    # Hält diesen Task am Leben, um auf Weiterleitungsanfragen zu lauschen.
    await asyncio.Future()


# --------------------------------------------------------------------------
# 3. SIMULATION: DER EXTERNE ENDBENUTZER
# --------------------------------------------------------------------------
async def run_end_user_test():
    """Simuliert einen externen Aufruf an die öffentliche API des Registry Servers."""
    print("--- [USER] Warte darauf, dass der Agent publiziert wird... ---")
    await published_event.wait()
    print("--- [USER] Agent ist jetzt öffentlich. Starte Testaufruf in 3 Sekunden... ---")
    await asyncio.sleep(3)

    public_url = published_info.get("public_url")
    api_key = published_info.get("public_api_key")

    if not public_url or not api_key:
        print("--- [USER] FEHLER: Keine öffentlichen Agenten-Infos gefunden!", file=sys.stderr)
        return

    print(f"--- [USER] Sende POST-Anfrage an: {public_url} ---")

    request_payload = {
        "query": "Hallo, weitergeleitete Welt!",
        "session_id": "ext-user-session-001"
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(public_url, json=request_payload, headers=headers) as response:
                print(f"--- [USER] Antwort-Status: {response.status} ---")

                if response.status == 200:
                    print("--- [USER] Beginne mit dem Streamen der Antwort-Events: ---")
                    # Die Antwort ist application/json-seq, also lesen wir zeilenweise
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                event_type = data.get('event_type', 'unknown')
                                status = data.get('status', '...')
                                print(f"  [STREAM] Event: {event_type:<20} | Status: {status} {data}")

                                # Der finale Event enthält das Ergebnis
                                if event_type == "final_result":
                                    final_result = data.get('details', {}).get('result')
                                    print("\n--- [USER] Endgültiges Ergebnis erhalten: ---")
                                    print(f"  >>> {final_result}")

                            except json.JSONDecodeError:
                                print(f"  [STREAM] Konnte Zeile nicht als JSON parsen: {line.decode()}")
                else:
                    error_text = await response.text()
                    print(f"--- [USER] FEHLER vom Server: {error_text}", file=sys.stderr)
        except aiohttp.ClientConnectorError as e:
            print(f"--- [USER] VERBINDUNGSFEHLER: Konnte den Server nicht erreichen. Läuft er? Fehler: {e}",
                  file=sys.stderr)


# --------------------------------------------------------------------------
# 4. ORCHESTRIERUNG
# --------------------------------------------------------------------------
async def main():
    print("==========================================================")
    print("=  ToolboxV2 Registry & Forwarding - Lokales Beispiel  =")
    print("==========================================================")

    # Starte Server und Client parallel.
    # `asyncio.gather` führt sie gleichzeitig aus.
    main_tasks = asyncio.gather(
        #run_registry_server(),
        run_local_client()
    )

    # Starte den Test-Task, der auf das Event wartet
    test_task = asyncio.create_task(run_end_user_test())

    try:
        # Lass die Haupttasks laufen. Das Skript endet hier nur durch manuellen Abbruch (Strg+C).
        await main_tasks
    except asyncio.CancelledError:
        print("\nProgramm wird beendet.")
    finally:
        # Aufräumen
        if not main_tasks.done():
            main_tasks.cancel()
        if not test_task.done():
            test_task.cancel()

        # Gib den App-Instanzen einen Moment zum sauberen Beenden.
        server_app = get_app("RegistryServerInstance")
        await server_app.a_exit()
        client_app = get_app("LocalClientInstance")
        await client_app.a_exit()

        print("Beendet.")


if __name__ == "__main__":
    import sys

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nManually interrupted.")
