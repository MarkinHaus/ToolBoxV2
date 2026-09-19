# toolboxv2/flows/mini/agent_builder.py
# BUILDER v6 (from scratch) — tb_admin auf Discord, out-of-the-box.
#
#   CODER     = "tb_admin_coder": 1:1 toolbox_admin (SYSTEM_PROMPT + alle 4
#               Admin-Toolgruppen + dangerous shell). Belegt den DEFAULT-Route
#               des Discord-Interfaces (kein !agent noetig).
#   VALIDATOR = "builder_validator": eigene Instanz, gleiche Admin-Toolgruppen,
#               aber END-TO-END-Validierungs-Prompt. Erreichbar nur via "!v":
#               Markin tippt "!v <aufgabe>" -> Interface-Extension (validator_enabled)
#               routed an den Validator. Der Coder hat zusaetzlich das Tool
#               validator_dispatch (gleicher Weg, icli-Delegationsmechanik).
#   DISCORD   = create_discord_interface(agent=CODER, self_agent=None,
#               admin_ids=[Markin]) — reagiert nur auf Markin.
#   Token:    BUILDER_DISCORD_TOKEN (aktiver BuilderTandem-Bot).
#
# Start (als markin, eigenem Screen):
#   screen -dmS builder bash -c 'set -a; source /home/markin/builder/.env; set +a;
#     exec /home/markin/ToolBoxV2/.venv/bin/tb -m agent_builder \
#     >> /home/markin/builder/builder_screen.log 2>&1'

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

NAME = "agent_builder"
ICON = "construction"
AUTH = False

_ADMIN_ID = int(os.environ.get("BUILDER_ADMIN_ID", "268830485889810432"))
_DC_SELF_BOT_ID = 1142855321849700523  # Bot-Peer (Auftraege vom dc_self-Bot)

_VALIDATOR_PROMPT_SUFFIX = """

## Deine Rolle im Builder-Tandem: VALIDATOR (Spezialisierung)
Du bist die separate Validator-Instanz. Deine einzige Aufgabe: END-TO-END
validieren, was der Coder gebaut hat — nicht selbst umbauen.
Vorgehen bei jedem Auftrag:
1. Kriterium klaeren (was heisst "funktioniert"?).
2. Real pruefen: shell/run_tests/analyze_code/docs_lookup nutzen — Belege sammeln.
3. Urteil: ERSTE Zeile "PASS" oder "FAIL", dann max. 5 Belegzeilen (Befehl + Ergebnis).
4. Nur auf ausdruecklichen Auftrag hin aenderst du Dateien.
"""

_CODER_PROMPT_SUFFIX = """

## Deine Rolle im Builder-Tandem: CODER
Du bist der Coder dieses Tandems (volle tb_admin-Faehigkeiten, siehe oben).
Zusatz-Tool: validator_dispatch(task, wait=True, session_id="default") —
delegiert eine Validierung an die separate Validator-Instanz und liefert
deren PASS/FAIL-Befund zurueck. Nutze es nach jedem Build.
Wenn Markin (Admin) dir einen Auftrag gibt: baue es, validiere via
validator_dispatch und berichte das Ergebnis kurz und praezise.
"""

# Modelle aus der Builder-Env (User-Entscheidung 19.09. abends: 9router-Aliase
# primaer — 9rou/fast + 9rou/complex — mit glm-5.3-flash/5.3 als Fallback).
_FAST_MODEL = os.environ.get("BUILDER_FAST_MODEL", "9rou/fast")
_COMPLEX_MODEL = os.environ.get("BUILDER_COMPLEX_MODEL", "9rou/complex")
_FALLBACK_CHAIN = [
    m.strip() for m in os.environ.get(
        "BUILDER_FALLBACK_CHAIN", "glm/glm-5.3-flash,glm/glm-5.3,glm/glm-4.7"
    ).split(",") if m.strip()
]


def _admin_tool_names(isaa_tools, app) -> list[str]:
    """Sammelt die Namen aller tb_admin-Tools aus den 4 Original-Gruppen
    (fuer die Discord moderator_safelist, damit der Coder als Default-Route
    nicht gedrosselt wird)."""
    from toolboxv2.flows.mini.toolbox_admin import (
        _build_dev_tools,
        _build_docs_tools,
        _build_manifest_tools,
        _build_toolbox_tools,
    )

    names: list[str] = []
    for group in (
        _build_toolbox_tools(isaa_tools, app),
        _build_docs_tools(app),
        _build_manifest_tools(app),
        _build_dev_tools(app),
    ):
        names.extend(name for _func, name, _desc, _cats in group)
    return names


def _build_and_register(builder, isaa_tools, app) -> list[str]:
    """Registriert 1:1 die 4 Admin-Toolgruppen am Builder. Gibt die Toolnamen
    zurueck (fuer die Safelist)."""
    from toolboxv2.flows.mini.toolbox_admin import (
        _build_dev_tools,
        _build_docs_tools,
        _build_manifest_tools,
        _build_toolbox_tools,
    )

    names: list[str] = []
    groups = (
        _build_toolbox_tools(isaa_tools, app),
        _build_docs_tools(app),
        _build_manifest_tools(app),
        _build_dev_tools(app),
    )
    for group in groups:
        for func, name, desc, cats in group:
            builder.add_tool(
                func, name, desc, category=cats,
                flags={"system_tool_by_name": True},
            )
            names.append(name)
    return names


async def _spawn_coder(host) -> tuple[object, list[str]]:
    """CODER = 1:1 tb_admin: Original-Prompt + alle 4 Toolgruppen."""
    from toolboxv2.flows.mini.toolbox_admin import SYSTEM_PROMPT

    builder = host.isaa_tools.get_agent_builder(
        name="tb_admin_coder", add_base_tools=True, with_dangerous_shell=True
    )
    host._apply_rate_limiter_to_builder(builder)
    builder.config.system_message = SYSTEM_PROMPT + _CODER_PROMPT_SUFFIX
    builder.with_models(_FAST_MODEL, _COMPLEX_MODEL)
    for fallback in _FALLBACK_CHAIN:
        builder.add_fallback_chain(_COMPLEX_MODEL, [fallback])
    builder.with_stream(True)
    names = _build_and_register(builder, host.isaa_tools, host.app)
    await host.isaa_tools.register_agent(builder)
    coder = await host.isaa_tools.get_agent("tb_admin_coder")
    # Runtime-Garantie (unabhaengig von Persistenz-Semantik des Registers):
    coder.amd.fast_llm_model = _FAST_MODEL
    coder.amd.complex_llm_model = _COMPLEX_MODEL
    print(
        f"[builder] coder 'tb_admin_coder' registered (1:1 toolbox_admin) "
        f"| fast={_FAST_MODEL} complex={_COMPLEX_MODEL} fallback={_FALLBACK_CHAIN}"
    )
    return coder, names


async def _spawn_validator(host, admin_tool_names: list[str]) -> object:
    """VALIDATOR = eigene Instanz, gleiche Admin-Tools, Validierungs-Prompt."""
    from toolboxv2.flows.mini.toolbox_admin import SYSTEM_PROMPT

    if "builder_validator" in host.agent_registry:
        return await host.isaa_tools.get_agent("builder_validator")

    builder = host.isaa_tools.get_agent_builder(
        name="builder_validator", add_base_tools=True, with_dangerous_shell=True
    )
    host._apply_rate_limiter_to_builder(builder)
    builder.config.system_message = SYSTEM_PROMPT + _VALIDATOR_PROMPT_SUFFIX
    builder.with_models(_FAST_MODEL, _COMPLEX_MODEL)
    for fallback in _FALLBACK_CHAIN:
        builder.add_fallback_chain(_COMPLEX_MODEL, [fallback])
    _build_and_register(builder, host.isaa_tools, host.app)
    await host.isaa_tools.register_agent(builder)
    validator = await host.isaa_tools.get_agent("builder_validator")
    # Runtime-Garantie: Persistenz (agent.json alter Stände) darf die Modelle
    # nie ueberschreiben — siehe Root-Case 'ollama/llama3.1 statt gpt-oss'.
    validator.amd.fast_llm_model = _FAST_MODEL
    validator.amd.complex_llm_model = _COMPLEX_MODEL

    from toolboxv2.flows.isaa.icli import AgentInfo
    host.agent_registry["builder_validator"] = AgentInfo(
        name="builder_validator",
        persona="End-to-end validator (full admin tools, PASS/FAIL evidence)",
        has_shell_access=True,
    )
    print(
        f"[builder] validator 'builder_validator' registered "
        f"| fast={_FAST_MODEL} complex={_COMPLEX_MODEL} fallback={_FALLBACK_CHAIN}"
    )
    return await host.isaa_tools.get_agent("builder_validator")


async def _wire_coder_validator(host, coder, validator) -> None:
    """Coder bekommt validator_dispatch (icli-Delegationsmechanik)."""

    async def validator_dispatch(
        task: str, wait: bool = True, session_id: str = "default"
    ) -> str:
        """Validierungsauftrag an builder_validator (die separate !v-Instanz)."""
        exc = await host._start_delegation("builder_validator", task, session_id)
        if wait:
            result = await asyncio.shield(exc.async_task)
            return str(result) if result else "(validator: kein Output)"
        return f"validator task started: {exc.task_id} (RunID: {exc.run_id})"

    coder.add_tool(
        validator_dispatch,
        name="validator_dispatch",
        description=(
            "Delegiert eine Validierung an builder_validator (die separate "
            "!v-Instanz). wait=True wartet auf PASS/FAIL + Belege, "
            "wait=False startet nur (returns task id)."
        ),
        category=["validator", "delegate"],
    )
    print("[builder] coder: validator_dispatch tool attached")


async def _connect_discord(host, coder, admin_tool_names, validator) -> object | None:
    """Discord via PROGRAMMATISCHE API (agent_api) — der bessere Weg statt
    icli-Bridge: Agent + Config rein, Handle mit start/stop/configure raus.
    Coder ist DEFAULT-Route, self_agent=None (kein !agent noetig)."""
    from toolboxv2.mods.isaa.extras.discord_interface.agent_api import (
        AgentDiscordConfig,
        DiscordAgentAPI,
    )

    token = os.environ.get("BUILDER_DISCORD_TOKEN")
    if not token:
        print("[builder] BUILDER_DISCORD_TOKEN fehlt - Discord aus.")
        return None

    handle = DiscordAgentAPI.register_agent(
        coder,
        AgentDiscordConfig(
            token=token,
            admin_ids=[_ADMIN_ID],
            respond_to_mentions_only=True,
            language="de",
            # Safelist schuetzt die Admin-Tools vor dem Safe-Mode-Pruning:
            moderator_safelist=set(admin_tool_names) | {"tb", "validator_dispatch"},
            # !v-Extension: Auftraege von Markin an den Validator:
            validator_agent=validator,
            validator_enabled=True,
            # dc_self-Bot darf weiterhin Auftraege senden:
            bot_peers={_DC_SELF_BOT_ID},
        ),
    )
    ok = handle.start(wait_ready=True, timeout=35)
    print(
        f"[builder] discord via agent_api started (coder=default, validator=!v) "
        f"ready={handle.ready} start_ok={ok}"
    )
    # Konvention: /discord-Commands via host — nur wenn die icli-Extension existiert
    # (in der reinen agent_api-Variante ohne icli-Patch gibt es kein discord_ext).
    if hasattr(host, "discord_ext"):
        host.discord_ext.interface = handle.interface
    return handle


async def _boot_dm(interface) -> None:
    """Boot-Quitung als DM an Markin (Cross-Loop-Bridge)."""
    bot = getattr(interface, "bot", None)
    if bot is None:
        return

    async def _wait_ready() -> None:
        for _ in range(30):
            if bot.is_ready():
                return
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(_wait_ready(), timeout=32)
    except TimeoutError:
        print("[builder] bot nicht ready - Boot-DM uebersprungen.")
        return

    ts = datetime.now().strftime("%d.%m. %H:%M")
    coro = interface.router.route_response(
        content=(
            f"builder v6 online ({ts})\n"
            "- coder = tb_admin 1:1 (out-of-the-box, kein !agent noetig)\n"
            "- validator = builder_validator, erreichbar via '!v <auftrag>'\n"
            "- reagiert nur auf dich (admin)"
        ),
        target_address=f"discord://dm:{_ADMIN_ID}",
    )
    bot_loop = getattr(bot, "loop", None)
    try:
        if bot_loop is not None and bot_loop is not asyncio.get_running_loop():
            res = await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(coro, bot_loop)
                ),
                timeout=30,
            )
        else:
            res = await asyncio.wait_for(coro, timeout=30)
        print(f"[builder] Boot-DM: {res}")
    except Exception as exc:  # noqa: BLE001 - Boot-DM blockiert Start nicht
        print(f"[builder] Boot-DM fehlgeschlagen: {exc}")


async def run(app=None, *args):
    """Builder v6: tb_admin-Coder auf Discord (out-of-the-box) + separate
    Validator-Instanz via !v / validator_dispatch."""
    from toolboxv2 import get_app

    app = app or get_app("agent_builder")

    from toolboxv2.flows.isaa.icli import ISAA_Host
    host = ISAA_Host(app)

    coder, admin_tool_names = await _spawn_coder(host)
    validator = await _spawn_validator(host, admin_tool_names)
    await _wire_coder_validator(host, coder, validator)

    interface = None
    try:
        interface = await _connect_discord(host, coder, admin_tool_names, validator)
    except Exception as exc:  # noqa: BLE001
        print(f"[builder] discord connect fehlgeschlagen: {exc}")

    if interface is not None:
        await _boot_dm(interface)

    await host.run()
