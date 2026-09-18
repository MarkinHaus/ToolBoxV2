# toolboxv2/flows/mini/agent_builder.py
# BUILDER-TANDEM v4 - EIN Flow, zwei Rollen, Toolbox-Konventionen 1:1.
#
#   CODER     = der BESTEHENDE Self-Agent "self" (ISAA_Host._init_self_agent,
#               icli.py:5190) -> Toolbox-Admin 1:1: volle Tools, dangerous shell,
#               iCLI-Jobtools (createJob/listJobs/deleteJob/createDreamJob),
#               teachSkill, delegate. KEIN eigener Code, KEINE Wrapper.
#   VALIDATOR = NEUE Agent-Instanz "builder_validator" (Bauplan exakt wie
#               ISAA_Host._tool_spawn_agent, icli.py:5310): base tools + eigene
#               Test-Tools (Multi-Turn-Shell, Playwright-Browser-Aktionen,
#               Beweis-Sammlung). Volle Umfaenge: kann Websites testen, CLIs in
#               Mehrschritt-Sessions fahren, Beweise liefern.
#   DISCORD   = DIE Interface-Verbindung (DiscordCLIExtension._connect ->
#               create_discord_interface) mit dem EIGENEN Builder-Bot-Token
#               (BUILDER_DISCORD_TOKEN). Bot-Peer-Patch (discord_interface.py:
#               bot_peers) erlaubt Auftraege VOM dc_self-Bot (1142855321849700523).
#
#   Routing (discord_interface._resolve_route):
#     - Markin (BUILDER_ADMIN_ID) kann per "!agent self" auf den CODER schalten,
#       Default = VALIDATOR (moderator slot).
#     - dc_self-Auftraege landen als Bot-Peer-Nachrichten im Interface.
#
# Start (als markin, Screen "builder"):
#   cd /home/markin/ToolBoxV2 && set -a; source /home/markin/builder/.env; set +a
#   exec .venv/bin/tb -m agent_builder
# Env: BUILDER_DISCORD_TOKEN, BUILDER_ADMIN_ID, BUILDER_GUILD_ID,
#      BUILDER_DOCS_ROOT, BUILDER_DOCS_FOCUS, BUILDER_KEEP_CHANNELS

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

NAME = "agent_builder"
ICON = "hammer"
AUTH = False

# V5: Beide Rollen = 1:1 toolbox_admin (Prompt + 4 Toolgruppen importiert aus dem
# Bestand, kein Duplikat). Coder erhaelt den Validator zusaetzlich als Tool.
from toolboxv2.flows.mini.toolbox_admin import (  # noqa: E402
    SYSTEM_PROMPT as _ADMIN_PROMPT,
    _build_dev_tools as _admin_build_dev_tools,
    _build_docs_tools as _admin_build_docs_tools,
    _build_manifest_tools as _admin_build_manifest_tools,
    _build_toolbox_tools as _admin_build_toolbox_tools,
)

_DC_SELF_BOT_ID = 1142855321849700523  # Bot-Peer: dc_self darf Auftraege senden
_PING_TARGET = os.environ.get(
    "BUILDER_PING_TARGET", "discord://dm:268830485889810432"
)

_TANDEM_SKILL = (
    "Du arbeitest im Builder-Tandem: DU (self) bist der Coder = 1:1 Toolbox-Admin "
    "(volle Tools, Job-Tools, teachSkill). Der Discord-Moderator-Kanal bedient den "
    "Agenten 'builder_validator' (volle Test-Rechte: Multi-Turn-Shell via "
    "validator_shell, Browser-Tests via validator_browser_action, Beweise via "
    "validator_collect_evidence). Arbeitsweise: (1) Auftraege von Markin oder dem "
    "dc_self-Bot entgegennehmen. (2) Implementierung mit deinen Admin-Tools "
    "vornehmen (write_code/patch_code, Manifest, Tests). (3) Validierung an "
    "'builder_validator' delegieren: konkreter Testauftrag mit erwartetem "
    "Beobachtungs-Ergebnis. (4) Validator-Ergebnis (Pass/Fail + Beweise) "
    "abwarten, bei Fail loop bis Pass. (5) Abschlussbericht: was, wo (Pfad), "
    "Validierungsnachweis. Nutze teachSkill NICHT fuer dieses Protokoll - es ist "
    "bereits dieser Skill."
)


# ---------------------------------------------------------------------------
# Validator-Tools (einmalige,duenne Ausfuehrungshelfer - keine Wrapper um tb)
# ---------------------------------------------------------------------------

_CWD_STATE: dict[str, str] = {}


async def validator_shell(command: str, reset: bool = False) -> str:
    """Multi-Turn-Shell fuer Validierung: bash in persistentem Arbeitsordner.

    Args:
        command: Bash-Befehl (Pipes, env, kompilierte Tests erlaubt).
        reset: True -> Arbeitsordner-Zustand zuruecksetzen (neue Session).
    """
    cwd = _CWD_STATE.get("cwd", "/home/markin/ToolBoxV2")
    if reset:
        cwd = "/home/markin/ToolBoxV2"
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        text = out.decode("utf-8", errors="replace")[-4000:]
        if command.strip().startswith("cd "):
            _CWD_STATE["cwd"] = command.strip()[3:].split(";")[0].strip() or cwd
        return f"[cwd={_CWD_STATE.get('cwd', cwd)} rc={proc.returncode}]\n{text}"
    except TimeoutError:
        if proc is not None:
            proc.kill()
        return f"[cwd={cwd}] TIMEOUT nach 120s"


async def validator_browser_action(action: str, target: str = "", value: str = "") -> str:
    """Playwright-Website-Test: goto|click|fill|extract|screenshot|assert_text.

    Args:
        action: goto, click, fill, extract, screenshot oder assert_text.
        target: URL (goto) bzw. CSS-Selector (click/fill/extract/assert_text).
        value: Text fuer fill bzw. erwarteter Text bei assert_text.
    """
    script = """
import asyncio, json, sys
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        result = {"ok": False, "data": "", "url": ""}
        action, target, value = sys.argv[1], sys.argv[2], sys.argv[3]
        try:
            if action == "goto":
                resp = await page.goto(target, wait_until="domcontentloaded", timeout=30000)
                result["data"] = f"status={resp.status if resp else 'n/a'}"
                result["ok"] = bool(resp and resp.ok)
            elif action == "click":
                await page.click(target, timeout=15000)
                await page.wait_for_load_state("domcontentloaded")
                result["ok"] = True
            elif action == "fill":
                await page.fill(target, value, timeout=15000)
                result["ok"] = True
            elif action == "extract":
                el = await page.query_selector(target)
                result["data"] = (await el.inner_text())[:3000] if el else "selector-not-found"
                result["ok"] = el is not None
            elif action == "screenshot":
                import os
                os.makedirs("/home/markin/builder/validator_shots", exist_ok=True)
                path = "/home/markin/builder/validator_shots/shot.png"
                await page.screenshot(path=path, full_page=True)
                result["data"] = path
                result["ok"] = True
            elif action == "assert_text":
                await page.wait_for_selector(f"text={value}", timeout=15000)
                result["ok"] = True
                result["data"] = f"found: {value}"
            result["url"] = page.url
        except Exception as exc:
            result["data"] = repr(exc)
        await browser.close()
        print(json.dumps(result))

asyncio.run(main())
"""
    proc = await asyncio.create_subprocess_exec(
        "/home/markin/ToolBoxV2/.venv/bin/python", "-c", script,
        action, target, value,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out = b""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
    except TimeoutError:
        if proc is not None:
            proc.kill()
    return out.decode("utf-8", errors="replace")[-2500:] if out else "[no output]"


_VALIDATOR_EXTRA_PROMPT = """

## Deine Rolle im Builder-Tandem: VALIDATOR
- Du bist die eigenstaendige Validator-Instanz (1:1 ToolBox Admin, alle Admin-Tools).
- Aufgabe: Auftraege TESTEN und VALIDIEREN, nicht selbst umbauen.
- Auftraggeber: Markin (BUILDER_ADMIN_ID), dc_self-Bot (Bot-Peer), der Coder
  (via validator_dispatch). Aufgaben kurz halten; bei Unklarheit nachfragen.
- Antworte mit Befund + Belegen (Befehle, Exit-Codes, Ausgaben). PASS/FAIL voranstellen.
- Kein Umbau drumherum: reporten, nur auf ausdruecklichen Auftrag aendern.
"""

_CODER_EXTRA_PROMPT = """

## Deine Rolle im Builder-Tandem: CODER
- Du bist die Coder-Instanz (1:1 ToolBox Admin, alle Admin-Tools).
- Dir ist der Validator ("builder_validator") als Tool angehaengt:
  `validator_dispatch(task, wait=True, session_id="default")` delegiert Test-/
  Validierungsauftraege an die Validator-Instanz (icli-Delegationsmechanik).
- Arbeitsweise: baust/pruefst, delegierst Test+Validierung an den Validator und
  beziehst dessen Befund ein. Bei Widerspruch gilt der Validator-Befund.
"""


async def _apply_admin_tooling(builder, isaa_tools, app, allow_toolbox: bool = True) -> None:
    """1:1 toolbox_admin-Ausstattung (die 4 Original-Toolgruppen aus dem Bestand).
    allow_toolbox=False verhindert Rekursiv-Self-Delegation (Validator ruft
    toolbox_execute nicht - dort lebt die Admin-Delegation)."""
    for func, name, desc, cats in _admin_build_toolbox_tools(isaa_tools, app):
        if name == "toolbox_execute" and not allow_toolbox:
            continue
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _admin_build_docs_tools(app):
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _admin_build_manifest_tools(app):
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _admin_build_dev_tools(app):
        builder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})


def _register_validator_tools(builder) -> None:
    """Eigene Validierungs-Tools an den Validator-Builder haengen (Konvention:
    builder.add_tool wie icli.py:5772)."""
    docs_root = os.environ.get("BUILDER_DOCS_ROOT", "/home/markin/ToolBoxV2")
    builder.add_tool(
        validator_shell,
        "validator_shell",
        "Multi-Turn-Bash-Shell mit persistentem Arbeitsordner: CLIs, Tests, "
        "Builds in mehreren Schritten validieren. reset=True startet neu.",
        category=["validator", "shell", "test"],
    )
    builder.add_tool(
        validator_browser_action,
        "validator_browser_action",
        "Playwright-Browser-Aktion fuer Website-Tests: goto|click|fill|extract|"
        "screenshot|assert_text (target=URL/Selector, value=Text).",
        category=["validator", "browser", "test"],
    )

    async def validator_collect_evidence(topic: str, findings: str) -> str:
        """Validierungs-Beweis ablegen (JSONL) und Quitung zurueckgeben."""
        path = Path(docs_root) / "toolboxv2" / ".data" / "validator_evidence.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(),
            "topic": topic,
            "findings": findings,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return f"evidence stored: {path} ({topic})"

    builder.add_tool(
        validator_collect_evidence,
        "validator_collect_evidence",
        "Validierungsnachweis (topic + findings) dauerhaft ablegen.",
        category=["validator", "evidence"],
    )


async def _spawn_validator(host):
    """Neue Instanz 'builder_validator' - 1:1 toolbox_admin (voller Prompt +
    alle 4 Admin-Toolgruppen) + eigene Validierungs-Tools."""
    if "builder_validator" in host.agent_registry:
        return await host.isaa_tools.get_agent("builder_validator")

    builder = host.isaa_tools.get_agent_builder(
        name="builder_validator", add_base_tools=True, with_dangerous_shell=True
    )
    host._apply_rate_limiter_to_builder(builder)
    # 1:1 toolbox_admin: Original-Prompt + Rollen-Zusatz
    builder.config.system_message = _ADMIN_PROMPT + _VALIDATOR_EXTRA_PROMPT
    # 1:1 toolbox_admin: alle 4 Admin-Toolgruppen (keine Rekursiv-Delegation)
    await _apply_admin_tooling(builder, host.isaa_tools, host.app, allow_toolbox=False)
    # Zusaetzlich die Validator-Spezialtools (Multi-Turn-Shell, Browser, Evidence)
    _register_validator_tools(builder)
    await host.isaa_tools.register_agent(builder)
    from toolboxv2.flows.isaa.icli import AgentInfo

    host.agent_registry["builder_validator"] = AgentInfo(
        name="builder_validator",
        persona="Full-permission validator (shell+browser+evidence)",
        has_shell_access=True,
    )
    print("[builder] validator agent 'builder_validator' registered")
    return await host.isaa_tools.get_agent("builder_validator")


async def _connect_discord(host) -> bool:
    """DIE Interface-Verbindung (wie iCLI) mit dem Builder-Bot-Token.

    _connect liest den Token aus args[0] (oder DISCORD_BOT_TOKEN); wir
    uebergeben ihn explizit. admin_ids werden nach dem Connect gesetzt,
    damit '!agent self' (Routing auf den Coder) fuer Markin funktioniert.
    """
    token = os.environ.get("BUILDER_DISCORD_TOKEN")
    if not token:
        print("[builder] BUILDER_DISCORD_TOKEN fehlt - Discord aus.")
        return False
    await host.discord_ext.handle_command(["connect", token])
    iface = getattr(host.discord_ext, "interface", None)
    if iface is None:
        return False
    admin_raw = os.environ.get("BUILDER_ADMIN_ID", "268830485889810432")
    try:
        iface.admin_ids = [int(x) for x in admin_raw.split(",") if x.strip()]
    except ValueError:
        iface.admin_ids = [268830485889810432]
    # Bot-Peer: Auftraege des dc_self-Bots zulassen (eigene ID bleibt gefiltert)
    if hasattr(iface, "bot_peers"):
        iface.bot_peers.add(_DC_SELF_BOT_ID)
    return True


async def _boot_report(host, discord_ok: bool) -> None:
    """Boot-Quitung per DM (Cross-Thread-Bridge wie dc_self)."""
    if not discord_ok:
        print("[builder] Discord nicht verbunden - Boot-Report uebersprungen.")
        return
    iface = getattr(host.discord_ext, "interface", None)
    bot = getattr(iface, "bot", None)
    if iface is None or bot is None:
        print("[builder] Interface/Bot fehlt - Boot-Report uebersprungen.")
        return

    async def _wait_ready() -> None:
        for _ in range(30):
            if bot.is_ready():
                return
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(_wait_ready(), timeout=32)
    except TimeoutError:
        print("[builder] Bot nicht rechtzeitig ready - Boot-Report uebersprungen.")
        return

    ts = datetime.now().strftime("%d.%m. %H:%M")
    coro = iface.router.route_response(
        content=(
            f"builder tandem online ({ts})\n"
            "- coder = self (Toolbox-Admin 1:1, Job-Tools aktiv)\n"
            "- validator = builder_validator (Multi-Turn-Shell + Playwright)\n"
            f"- discord: {'OK' if discord_ok else 'FEHLER'} | "
            "Auftraege: #builder (Mention) oder DM; '!agent self' = Coder; "
            "dc_self-Bot-Peer aktiv"
        ),
        target_address=_PING_TARGET,
    )
    try:
        bot_loop = getattr(bot, "loop", None)
        if bot_loop is not None and bot_loop is not asyncio.get_running_loop():
            res = await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(coro, bot_loop)
                ),
                timeout=30,
            )
        else:
            res = await asyncio.wait_for(coro, timeout=30)
        print(f"[builder] Boot-Report: {res}")
    except Exception as exc:  # noqa: BLE001 - Report blockiert Start nicht
        print(f"[builder] Boot-Report fehlgeschlagen: {exc}")


async def run(app=None, *args):
    """Builder-Tandem v5: Coder (self) + Validator (builder_validator), BEIDE
    1:1 toolbox_admin (Original-Prompt + alle 4 Admin-Toolgruppen aus dem
    Bestand). Der Coder erhaelt den Validator zusaetzlich als Tool
    (validator_dispatch, icli-Delegationsmechanik)."""
    from toolboxv2 import get_app

    app = app or get_app("agent_builder")

    from toolboxv2.flows.isaa.icli import ISAA_Host
    host = ISAA_Host(app)

    #1 DISCORD_BOT_TOKEN (iCLI-Extensions-Lesepfad) auf den Builder-Bot setzen
    token = os.environ.get("BUILDER_DISCORD_TOKEN")
    if token:
        os.environ["DISCORD_BOT_TOKEN"] = token

    # Gleiche Discord-Integration wie iCLI-Entry (nur connect, kein safe-mode-
    # Patch am Moderator: der Validator behaelt seine vollen Rechte).
    try:
        from toolboxv2.mods.isaa.extras.discord_interface.integration_example import (
            patch_cli_for_discord,
        )
        patch_cli_for_discord(host)
    except ImportError as exc:
        print(f"[builder] Discord integration nicht verfuegbar: {exc}")

    # Coder = BESTEHENDER Self-Agent (1:1 iCLI: root, Job-Tools, teachSkill)
    await host._init_self_agent()

    # Tandem-Protokoll als Skill beibringen (System-Mechanik teachSkill)
    try:
        await host._tool_teach_skill(
            "self", "builder_tandem", _TANDEM_SKILL,
            ["builder", "tandem", "validate", "auftrag", "umsetzen"],
        )
        print("[builder] teachSkill: builder_tandem an self uebergeben")
    except Exception as exc:  # noqa: BLE001 - Skill darf Start nicht blockieren
        print(f"[builder] teachSkill uebersprungen: {exc}")

    # Validator = NEUE Instanz (1:1 toolbox_admin + Validierungs-Tools)
    validator = await _spawn_validator(host)

    # Coder = 1:1 toolbox_admin: Admin-Toolgruppen + Validator als Tool.
    # (self hat die Admin-Gruppen teilweise als System-Tools; doppelte Namen
    #  werden vom ToolManager ignoriert, die Delegation ist neu.)
    try:
        coder = await host.isaa_tools.get_agent("self")
        coder.amd.system_message = _ADMIN_PROMPT + _CODER_EXTRA_PROMPT

        async def validator_dispatch(
            task: str, wait: bool = True, session_id: str = "default"
        ) -> str:
            """Delegiert einen Test-/Validierungsauftrag an builder_validator
            (icli-Delegationsmechanik: _start_delegation)."""
            exc = await host._start_delegation("builder_validator", task, session_id)
            if wait:
                result = await asyncio.shield(exc.async_task)
                return str(result) if result else "(validator: kein Output)"
            return f"✓ Validator-Task gestartet: {exc.task_id} (RunID: {exc.run_id})"

        coder.add_tool(
            validator_dispatch,
            name="validator_dispatch",
            description=(
                "Test-/Validierungsauftrag an builder_validator delegieren "
                "(1:1 toolbox_admin-Validator des Tandems). wait=True wartet auf "
                "den Befund (PASS/FAIL + Beweise), wait=False startet nur "
                "(returns task id)."
            ),
            category=["validator", "delegate"],
        )
        print("[builder] coder: admin-Prompt gesetzt + validator_dispatch-Tool angehaengt")
    except Exception as exc:  # noqa: BLE001 - Tandem-Tool blockiert Start nicht
        print(f"[builder] validator_dispatch-Registrierung fehlgeschlagen: {exc}")

    # Discord (Builder-Bot) + Bot-Peer fuer dc_self-Auftraege
    discord_ok = False
    try:
        discord_ok = await _connect_discord(host)
    except Exception as exc:  # noqa: BLE001
        print(f"[builder] Discord connect fehlgeschlagen: {exc}")

    await _boot_report(host, discord_ok)

    # iCLI-Host-Loop 1:1 (Scheduler, missed jobs, TUI)
    await host.run()
