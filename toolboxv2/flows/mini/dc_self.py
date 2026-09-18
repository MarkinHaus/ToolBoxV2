# toolboxv2/flows/mini/dc_self.py
# DC-Self v2 - 1:1 der icli-Host (ISAA_Host) mit dem BESTEHENDEN Self-Agent "self".
#
# Design (ToolBox-Konventionen: kein neuer Agent, keine Wrapper, keine Duplikate):
#   - Agent   : ISAA_Host._init_self_agent() baut "self" (root, volle Tools) exakt
#               wie die icli (toolboxv2/flows/isaa/icli.py:5190).
#   - Jobs    : host.job_scheduler = isaa_tools.job_scheduler; createJob / listJobs /
#               deleteJob / createDreamJob kommen aus _register_self_agent_tools
#               (icli.py:5721-5745). Kein eigener Job-Code.
#   - Discord : patch_cli_for_discord(host) -> DiscordCLIExtension._connect baut
#               exakt die Verbindung, die die icli nutzt (create_discord_interface
#               mit self_agent + discord_moderator, host, runner_loop).
#   - Modus   : host.run() startet JobScheduler, on_cli_start, missed-jobs und die
#               icli-TUI (Command-Handling inkl. /discord) 1:1 wie die icli.
#
# Start : tb -m dc_self
# Screen: (als markin, Vorbild: builder-Screen)
#   screen -dmS dc_self bash -c 'cd /home/markin/ToolBoxV2 && set -a; \
#     source /home/markin/builder/.env; set +a; \
#     exec /home/markin/ToolBoxV2/.venv/bin/tb -m dc_self'
#
# Env:
#   DISCORD_BOT_TOKEN    Bot-Token (Pflicht fuer Auto-Connect, sonst /discord connect)
#   DC_SELF_DISCORD      1 (default) = auto-connect nach Self-Agent-Init
#   DC_SELF_PING         1 (default) = Boot-DM an DC_SELF_PING_TARGET
#   DC_SELF_PING_TARGET  default discord://dm:268830485889810432 (Markin)

import asyncio
import os
from datetime import datetime

NAME = "dc_self"
ICON = "robot"
AUTH = False

_PING_TARGET = os.environ.get(
    "DC_SELF_PING_TARGET", "discord://dm:268830485889810432"
)


async def _auto_connect_discord(host) -> None:
    """Discord wie in der icli verbinden (gleiche Extension, gleiche Verbindung)."""
    if os.environ.get("DC_SELF_DISCORD", "1") != "1":
        return
    if not os.environ.get("DISCORD_BOT_TOKEN"):
        print(
            "[dc_self] DISCORD_BOT_TOKEN fehlt - Discord spaeter via "
            "'/discord connect <token>' in der dc_self-Screen verbinden."
        )
        return
    try:
        await host.discord_ext.handle_command(["connect"])
    except Exception as exc:  # noqa: BLE001 - gleiche Fehlertoleranz wie icli
        print(f"[dc_self] Discord auto-connect fehlgeschlagen: {exc}")


async def _boot_ping(host) -> None:
    """Boot-Nachricht ueber das BESTEHENDE Interface (kein eigener Transport)."""
    if os.environ.get("DC_SELF_PING", "1") != "1":
        return
    iface = getattr(host.discord_ext, "interface", None)
    if iface is None:
        return

    bot = getattr(iface, "bot", None)

    async def _wait_ready() -> None:
        for _ in range(25):
            if bot is not None and bot.is_ready():
                return
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(_wait_ready(), timeout=30)
    except asyncio.TimeoutError:
        print("[dc_self] Bot nicht rechtzeitig ready - Boot-Ping uebersprungen.")
        return

    ts = datetime.now().strftime("%d.%m. %H:%M")
    route_coro = iface.router.route_response(
        content=(
            f"dc_self online ({ts}) - Self-Agent 'self' + "
            "Jobs + Discord aktiv (icli-Host 1:1)."
        ),
        target_address=_PING_TARGET,
    )
    try:
        # Bot lebt in eigenem Thread/Loop (run_bg_task_advanced) - Bridge-Pattern
        # wie DiscordInterface._collect: Coro auf den Bot-Loop schedulen.
        bot_loop = getattr(bot, "loop", None)
        if bot_loop is not None and bot_loop is not asyncio.get_running_loop():
            res = await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(route_coro, bot_loop)
                ),
                timeout=30,
            )
        else:
            res = await asyncio.wait_for(route_coro, timeout=30)
        print(f"[dc_self] Boot-Ping: {res}")
    except Exception as exc:  # noqa: BLE001 - Ping darf Start nicht blockieren
        print(f"[dc_self] Boot-Ping fehlgeschlagen (Discord laeuft weiter): {exc}")


async def run(app=None, *args):
    """dc_self = icli-Host 1:1 (Self-Agent + Job-Tools + Discord-Verbindung)."""
    from toolboxv2 import get_app

    app = app or get_app("dc_self")

    from toolboxv2.flows.isaa.icli import ISAA_Host
    host = ISAA_Host(app)

    # Gleiche Discord-Integration wie der icli-Entry (icli.py:14731-14738)
    try:
        from toolboxv2.mods.isaa.extras.discord_interface.integration_example import (
            patch_cli_for_discord,
        )
        patch_cli_for_discord(host)
        print("[dc_self] Discord integration enabled (icli-Pfad).")
    except ImportError as exc:
        print(f"[dc_self] Discord integration not available: {exc}")

    # Self-Agent vorab initialisieren (idempotent, Pattern wie icli main_cli:14844)
    await host._init_self_agent()

    await _auto_connect_discord(host)
    await _boot_ping(host)

    # Uebernimmt 1:1 aus der icli: JobScheduler-Start, on_cli_start,
    # missed-jobs, autowake-Check und die interaktive TUI-Loop.
    await host.run()
