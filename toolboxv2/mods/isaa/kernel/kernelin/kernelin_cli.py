import asyncio
import sys
from typing import Optional

# Deine Imports
from toolboxv2 import get_app
from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import (
    Signal as KernelSignal,
    SignalType,
    KernelConfig,
    IOutputRouter,
    KernelState
)

# -----------------------------------------------------------------------------
# 1. Output Router Implementation
# -----------------------------------------------------------------------------

class CLIOutputRouter(IOutputRouter):
    """
    Konkrete Implementierung des Routers für das Terminal.
    Leitet Antworten und Benachrichtigungen direkt an stdout weiter.
    """
    def __init__(self):
        self.printer = ProgressiveTreePrinter()

    async def send_response(
        self,
        user_id: str,
        content: str,
        role: str = "assistant",
        metadata: dict = None
    ):
        """Gibt die Antwort des Agenten farbig formatiert aus."""
        # Einfache Formatierung zur Unterscheidung vom User-Input
        print(f"\n\033[96m[AI - {role}]:\033[0m {content}\n")

    async def send_notification(
        self,
        user_id: str,
        content: str,
        priority: int = 5,
        metadata: dict = None
    ):
        """Gibt proaktive Benachrichtigungen aus."""
        color = "\033[93m" if priority > 7 else "\033[94m" # Gelb für hoch, Blau für normal
        print(f"\n{color}[NOTIFICATION p={priority}]:\033[0m {content}\n")


# -----------------------------------------------------------------------------
# 2. Async Input Helper
# -----------------------------------------------------------------------------

async def ainput(prompt: str = "") -> str:
    """
    Nicht-blockierender Input, damit der Kernel-Loop (Heartbeat) nicht stoppt,
    während auf User-Eingabe gewartet wird.
    """
    return await asyncio.get_event_loop().run_in_executor(
        None, sys.stdin.readline
    )

# -----------------------------------------------------------------------------
# 3. Main CLI Loop
# -----------------------------------------------------------------------------

async def main():
    # 1. App & Agent laden
    app = get_app()

    # Hier nehmen wir an, dass ein Agent namens 'default' oder ähnlich existiert.
    # Falls du einen spezifischen Agentennamen hast, hier ändern.
    agent_name = "self"
    print(f"--- Lade Agent '{agent_name}' ---")

    try:
        app = get_app()
        isaa = app.get_mod("isaa")
        agent = await isaa.get_agent(agent_name)
        if not agent:
            print(f"Fehler: Agent '{agent_name}' konnte nicht geladen werden.")
            return
    except Exception as e:
        print(f"Kritischer Fehler beim Laden des Agenten: {e}")
        return

    # 2. Kernel Konfiguration & Setup
    config = KernelConfig(
        max_signal_queue_size=100,
        heartbeat_interval=1.0,
        signal_timeout=0.5
    )

    router = CLIOutputRouter()

    # Kernel Instanziierung
    kernel = Kernel(
        agent=agent,
        config=config,
        output_router=router
    )

    # 3. Kernel Starten
    print("--- Starte ProA Kernel ---")
    await kernel.start()

    user_id = "cli_user"
    print(f"\nKernel Status: {kernel.state.value}")
    print("Tippe 'exit' oder 'quit' zum Beenden. oder /status um Kernel Status zu prüfen.\n")

    # 4. Interaction Loop
    try:
        while kernel.state == KernelState.RUNNING:
            # Prompt anzeigen (ohne Newline, damit Input dahinter steht)
            print("\033[92mYou:\033[0m ", end="", flush=True)

            # Warten auf Input (non-blocking)
            line = await ainput()
            user_input = line.strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Beende Session...")
                break

            if user_input.lower() == "/status":
                # Debug Kommando um Kernel Status zu prüfen
                status = kernel.get_status()
                print(f"\n[SYSTEM]: {status}\n")
                continue

            # Input an den Kernel senden
            await kernel.handle_user_input(
                user_id=user_id,
                content=user_input,
                metadata={"source": "cli"}
            )

            # Kurzes Yield, damit der Kernel reagieren kann
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
    finally:
        # 5. Sauberer Shutdown
        print("--- Stoppe Kernel ---")
        await kernel.stop()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
