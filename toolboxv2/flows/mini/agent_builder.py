# toolboxv2/flows/mini/agent_builder.py
# Builder-Flow (CODER + NAVIGATOR Tandem) - aufgebaut wie toolbox_admin:
# ISAA-Agent via AgentBuilder, gleiche Toolgruppen + Builder-Tandem-Logik,
# startbar via: tb -m isaa -f agent_builder  |  tb flow --flow agent_builder
# Regeln (Markin 16.09.): flexible Pakete (3-10, jedes 100%), ASCII-Reports,
# Docs-Feuer Default ToolBoxV2, Browser-Gate, zglm-Modelle (Subscription).

import asyncio
import json
import os
import sys
from pathlib import Path

NAME = "agent_builder"
ICON = "construction"
AUTH = False

FAST_DEFAULT = os.environ.get("BUILDER_FAST_MODEL", "zglm/glm-5.3-flash")
COMPLEX_DEFAULT = os.environ.get("BUILDER_COMPLEX_MODEL", "zglm/glm-5.3")

# toolboxv2/flows/mini/agent_builder.py
# Builder-Flow V3 (CODER + NAVIGATOR + Discord-Bridge) - wie toolbox_admin:
# Start: tb -m agent_builder   (alt: tb flow --flow agent_builder)
# V3-NEU: Discord-Connection (senden+lesen via Bot-Token, Default #builder)
# + normal chatten (kein Bau-Auftrag -> direkte Antwort, kein Tandem-Ritual).
# Regeln (Markin 16.09.): 2 Agenten, CODER mit vollen TB-Admin-Dev-Tools,
# flexible Pakete 1-10 (1 = Mini-Edit erlaubt), jedes Paket 100%,
# mehrschrittige Validierung (Terminal + Browser), ASCII-Reports,
# Docs-Feuer Default ToolBoxV2.

import asyncio
import json
import os
import sys
from pathlib import Path

NAME = "agent_builder"
ICON = "construction"
AUTH = False

FAST_DEFAULT = os.environ.get("BUILDER_FAST_MODEL", "zglm/glm-5.3-flash")
COMPLEX_DEFAULT = os.environ.get("BUILDER_COMPLEX_MODEL", "zglm/glm-5.3")

CODER_PROMPT = """# Builder-CODER v2.0 (Umsetzer)

Du bist CODER. Du setzt Bau-Auftraege um - vom 1-Zeilen Mini-Edit bis zum
10-Pakete Projekt. Du hast die volle TB-Admin-Ausstattung: write_code,
patch_code, shell, tb-CLI, docs, manifest, dev-Tools.

## Kernprinzipien (Markin-Regeln 16.09.)

1. **Quellcode > Docs**: Docs sind Orientierung, Ground Truth ist echter Code.
   Verifiziere via docs_lookup(include_code=True).
2. **Flexible Pakete**: Zerlege in 1-10 Pakete - so wenige wie moeglich, so
   viele wie noetig. 1 Paket = Mini-Edit ist voll ok. JEDES Paket 100%.
3. **Freiraum**: Ein Paket darf mehrschrittig sein (Terminal-Befehle,
   mehrere Dateien) - solange das Ziel klar abgrenzbar ist.
4. **ASCII-Reports**: Reines ASCII, Datei-Pfade + was getan wurde.
5. **Docs-Feuer Default ToolBoxV2**: Vor dem Zerlegen docs_inventory lesen,
   echte Pfade/APIs nutzen, nichts erfinden.
6. **Kein Raten**: Was du nicht weisst, schlaegst du nach.
7. **Normal chatten**: Kein Bau-Auftrag (Frage, Plausch, Feedback)? Dann
   antworte direkt und kurz - KEIN Zerlegen, KEIN Tandem-Ritual.
8. **Discord**: Status/Fragen via builder_discord_send an #builder
   (Default-Kanal), Feedback via builder_discord_read holen.
"""

NAVIGATOR_PROMPT = """# Builder-NAVIGATOR v2.0 (Pruefer, kein Coder)

Du bist NAVIGATOR. Du pruefst jedes CODER-Paket mehrschrittig - du schreibst
selbst KEINEN Bau-Code um (nur Repair-Vorschlaege als Text).

## Pruef-Schritte pro Paket

1. **Terminal**: compile/lint/tests via shell bzw. builder_terminal_check
   (z.B. python -m compileall, pytest, ruff) - je nach Artefakt.
2. **Browser** (nur HTML/Web): builder_browser_check - Seite oeffnen,
   Titel/Text assert, Screenshot pruefen.
3. **Gate-JSON**: Antworte NUR JSON:
   {"pass":true/false,"score":0-10,"browser_expect":[...],"issues":[...]}
4. **Gate-Regel**: pass=true UND score>=9 = bestanden. Sonst konkreter
   Repair-Vorschlag (max 3 Versuche), dann ABBRUCH empfehlen.

## Reports: nur ASCII, kurz, mit Beleg (Befehl + Output-Kurzform).
Beleg ggf. aus Discord-Verlauf (builder_discord_read). Selbst NICHT posten -
Reports gehen ueber CODER.
"""
# ---------------------------------------------------------------------------
# Builder-Tandem Tools
# ---------------------------------------------------------------------------

BUILDER_CHANNEL = "1549821547886288996"
DISCORD_GUILD_DEFAULT = "346680165348409345"

def _discord_headers():
    import urllib.request as _u
    tok = os.environ.get("BUILDER_DISCORD_TOKEN", "")
    return {"Authorization": "Bot " + tok, "Content-Type": "application/json",
            "User-Agent": "BuilderFlow/3.0"}


def _build_discord_tools(app):
    """Discord-Connection (Vorbild: discord_interface._register_agent_tools):
    send/read/channels via Discord-REST mit BUILDER_DISCORD_TOKEN.
    CODER: send+read. NAVIGATOR: read-only (kein Posten in fremdem Namen)."""
    tools = []

    async def builder_discord_send(channel_id: str = BUILDER_CHANNEL,
                                   text: str = "") -> str:
        """Discord-Nachricht senden (Default #builder). Reports/Fragen an Markin."""
        import urllib.request, json as _j
        try:
            req = urllib.request.Request(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                data=_j.dumps({"content": text[:1900]}).encode(),
                headers=_discord_headers(), method="POST")
            with urllib.request.urlopen(req, timeout=20) as r:
                d = _j.loads(r.read().decode("utf-8", "replace"))
            return f"gesendet (msg {d.get('id')})"
        except Exception as e:
            return f"discord-send-error: {e}"

    async def builder_discord_read(channel_id: str = BUILDER_CHANNEL,
                                   limit: int = 10) -> str:
        """Discord-Verlauf lesen (Default #builder). Feedback von Markin holen."""
        import urllib.request, json as _j
        try:
            req = urllib.request.Request(
                f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={min(limit, 50)}",
                headers=_discord_headers())
            with urllib.request.urlopen(req, timeout=20) as r:
                msgs = _j.loads(r.read().decode("utf-8", "replace"))
            out = []
            for m in msgs:
                a = (m.get("author") or {}).get("username", "?")
                out.append(f"{a}: {(m.get('content') or '')[:300]}")
            return "\n".join(out) or "(leer)"
        except Exception as e:
            return f"discord-read-error: {e}"

    async def builder_discord_channels() -> str:
        """Textkanaele des test6-Servers auflisten (IDs fuer send/read)."""
        import urllib.request, json as _j
        try:
            req = urllib.request.Request(
                f"https://discord.com/api/v10/guilds/{DISCORD_GUILD_DEFAULT}/channels",
                headers=_discord_headers())
            with urllib.request.urlopen(req, timeout=20) as r:
                chans = _j.loads(r.read().decode("utf-8", "replace"))
            return "\n".join(f"{c.get('id')} | {c.get('name')}"
                              for c in sorted(chans, key=lambda x: x.get("position", 0))
                              if c.get("type") == 0)
        except Exception as e:
            return f"discord-channels-error: {e}"

    tools.extend([
        (builder_discord_send, "builder_discord_send",
         "Discord-Nachricht senden (channel_id=Default #builder, text)",
         ["builder", "discord"]),
        (builder_discord_read, "builder_discord_read",
         "Discord-Verlauf lesen (channel_id=Default, limit) - Feedback holen",
         ["builder", "discord"]),
        (builder_discord_channels, "builder_discord_channels",
         "Textkanaele auflisten (IDs fuer send/read)",
         ["builder", "discord"]),
    ])
    return tools


def _build_builder_tools(app):
    tools = []

    async def builder_terminal_check(cmd: str, cwd: str = "") -> str:
        """Terminal-Validierung: Befehl ausfuehren (compile/lint/pytest), Output zurueck."""
        import subprocess as sp
        try:
            r = sp.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=120, cwd=cwd or None)
            out = (r.stdout or "") + (r.stderr or "")
            return f"RC={r.returncode}\n" + out[:3000]
        except Exception as e:
            return f"terminal-error: {e}"

    async def builder_decompose(instruction: str, docs_context: str = "") -> str:
        """Auftrag in 1-10 Pakete zerlegen (1 = Mini-Edit ok). JSON-Liste [{nr,title,goal}]."""
        prompt = (
            "Zerlege den Auftrag in 1 bis 10 sinnvolle Pakete (1 = Mini-Edit erlaubt) - so wenige wie "
            "moeglich, so viele wie noetig. Jedes Paket einzeln 100%-umsetzbar. "
            'NUR JSON-Liste: [{"nr":1,"title":"...","goal":"..."}]\nAuftrag: '
            + instruction
            + ("\nProjekt-Kontext:\n" + docs_context[:4000] if docs_context else "")
        )
        return "PROMPT_FOR_AGENT:\n" + prompt

    async def builder_validate_gate(packet_nr: int = 0, title: str = "",
                                    goal: str = "", result: str = "") -> str:
        """Navigator-Gate: pass=true + score>=9/10 erforderlich."""
        prompt = (
            f"Pruefe Paket {packet_nr} '{title}'\nZiel: {goal}\n"
            f"Coder-Ergebnis:\n{result[:3500]}\n"
            'NUR JSON: {"pass":true/false,"score":0-10,'
            '"browser_expect":["sichtbare Texte"],"issues":["..."]}'
        )
        return "GATE-REGEL: pass=true UND score>=9 sonst Nachbau (max 3x).\n" + prompt

    async def builder_browser_check(html_path: str, expects: str = "") -> str:
        """HTML-Artefakt LIVE im Chromium oeffnen, Titel/Text pruefen, Screenshot."""
        import subprocess as sp
        want = [e.strip().lower() for e in expects.split(",") if e.strip()]
        pylines = [
            "from playwright.sync_api import sync_playwright",
            "import os",
            f"path = {html_path!r}",
            f"want = {want!r}",
            "with sync_playwright() as p:",
            "    b = p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])",
            "    page = b.new_page()",
            "    page.goto('file://' + os.path.abspath(path), timeout=15000)",
            "    page.wait_for_load_state('networkidle', timeout=10000)",
            "    body = (page.inner_text('body') + ' ' + page.title()).lower()",
            "    page.screenshot(path='validation_' + os.path.basename(path) + '.png')",
            "    b.close()",
            "missing = [e for e in want if e not in body]",
            "print('browser-OK' if not missing else 'fehlt im Browser: ' + ', '.join(missing))",
        ]
        code = "\n".join(pylines)
        try:
            r = sp.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=90)
            out = (r.stdout or "") + (r.stderr or "")
            ok = "browser-OK" in out
            return ("browser-OK" if ok else "browser-FAIL") + "\n" + out[:1500]
        except Exception as e:
            return f"browser-error: {e}"

    async def builder_send_file(file_path: str) -> str:
        """Datei-Inhalt zur Uebergabe bereitstellen (Pfad + Groesse)."""
        try:
            p = Path(file_path)
            if not p.exists():
                return f"Datei nicht gefunden: {p}"
            return f"Datei bereit: {p} ({p.stat().st_size} Bytes)"
        except Exception as e:
            return f"Fehler: {e}"

    tools.extend([
        (builder_decompose, "builder_decompose",
         "Auftrag in 1-10 Pakete zerlegen (instruction, docs_context)",
         ["builder", "plan"]),
        (builder_terminal_check, "builder_terminal_check",
         "Terminal-Validierung: Befehl ausfuehren (cmd, cwd)",
         ["builder", "terminal"]),
        (builder_validate_gate, "builder_validate_gate",
         "Navigator-Gate: pass+score>=9 (packet_nr, title, goal, result)",
         ["builder", "validate"]),
        (builder_browser_check, "builder_browser_check",
         "HTML live im Browser pruefen + Screenshot (html_path, expects)",
         ["builder", "browser"]),
        (builder_send_file, "builder_send_file",
         "Datei zur Uebergabe bereitstellen (file_path)",
         ["builder", "file"]),
    ])
    return tools

# ---------------------------------------------------------------------------
# Main Entry (Spiegel von toolbox_admin.run)
# ---------------------------------------------------------------------------

async def _run_agent_stream(agent, text: str, session_id: str):
    """Auftrag an einen der beiden Agenten streamen (wie _print_stream)."""
    from toolboxv2.flows.mini.toolbox_admin import _print_stream
    await _print_stream(agent, text, session_id=session_id)


async def run(app, args=None):
    """Main entry point for the agent_builder flow (2 Agenten: CODER + NAVIGATOR)."""
    from toolboxv2.utils.extras.Style import Style, cls
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
    from prompt_toolkit.history import FileHistory
    from toolboxv2.flows.mini.toolbox_admin import (
        _build_toolbox_tools, _build_docs_tools, _build_manifest_tools,
        _build_dev_tools,
    )

    cls()
    print(Style.CYAN("+-------------------------------+"))
    print(Style.CYAN("|  Builder (TB-Flow V2, 2 Agents)|"))
    print(Style.CYAN("|  CODER + NAVIGATOR via ISAA   |"))
    print(Style.CYAN("+-------------------------------+"))
    print()

    isaa = app.get_mod("isaa")
    if isaa is None:
        print(Style.RED("ERROR: ISAA module not loaded!"))
        print("Start with: tb -m agent_builder")
        return
    isaa.stuf = True

    print(Style.YELLOW("Initializing ISAA..."))
    await isaa.init_isaa(name="agent_builder")

    # --- CODER: volle TB-Admin-Ausstattung (wie toolbox_admin) ---
    print(Style.YELLOW("Building CODER agent (volle Dev-Tools)..."))
    coder = isaa.get_agent_builder(
        "builder_coder",
        add_base_tools=True,
        with_dangerous_shell=True,
    )
    coder.with_stream(True)
    coder.with_models(FAST_DEFAULT, COMPLEX_DEFAULT)
    for func, name, desc, cats in _build_toolbox_tools(isaa, app):
        coder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_docs_tools(app):
        coder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_manifest_tools(app):
        coder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_dev_tools(app):
        coder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_builder_tools(app):
        coder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_discord_tools(app):
        coder.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    coder.config.system_message = CODER_PROMPT
    await isaa.register_agent(coder)

    # --- NAVIGATOR: read-only Pruefer (kein Bau-Code, kein dangerous shell) ---
    print(Style.YELLOW("Building NAVIGATOR agent (Pruefer, read-only)..."))
    navigator = isaa.get_agent_builder(
        "builder_navigator",
        add_base_tools=True,
        with_dangerous_shell=False,
    )
    navigator.with_stream(True)
    navigator.with_models(FAST_DEFAULT, COMPLEX_DEFAULT)
    for func, name, desc, cats in _build_docs_tools(app):
        navigator.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_manifest_tools(app):
        navigator.add_tool(func, name, desc, category=cats, flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_builder_tools(app):
        if name in ("builder_validate_gate", "builder_browser_check",
                    "builder_terminal_check", "builder_send_file"):
            navigator.add_tool(func, name, desc, category=cats,
                               flags={"system_tool_by_name": True})
    for func, name, desc, cats in _build_discord_tools(app):
        if name in ("builder_discord_read", "builder_discord_channels"):
            navigator.add_tool(func, name, desc, category=cats,
                               flags={"system_tool_by_name": True})
    navigator.config.system_message = NAVIGATOR_PROMPT
    await isaa.register_agent(navigator)

    coder_agent = await isaa.get_agent("builder_coder")
    navi_agent = await isaa.get_agent("builder_navigator")
    try:
        coder_agent.tool_manager.register_cli_tool("tb", executable="uv", executable_args=["run"],
                                                   flags={"system_tool_by_name": True},
                                                   cli_tool_executable="tb", category="system")
    except Exception:
        pass

    print(Style.GREEN("Agents ready: builder_coder + builder_navigator"))
    print(Style.GREEN(f"  CODER:     fast={coder_agent.amd.fast_llm_model} complex={coder_agent.amd.complex_llm_model}"))
    print(Style.GREEN(f"  NAVIGATOR: fast={navi_agent.amd.fast_llm_model} complex={navi_agent.amd.complex_llm_model}"))
    print()
    print(Style.GREY("Commands: /status  /flows  /help  exit"))
    print(Style.GREY("Bau-Auftrag -> CODER zerlegt (1-10 Pakete, 1=Mini-Edit ok),"))
    print(Style.GREY("baut mehrschrittig, NAVIGATOR prueft (Terminal+Browser, Gate)."))
    print(Style.GREY("/navi <text> = direkt an NAVIGATOR (Nachpruefung)."))
    print()

    history_path = Path(app.data_dir) / ".agent_builder_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=FuzzyCompleter(
            WordCompleter(["/status", "/flows", "/help", "exit", "quit"], ignore_case=True)),
    )

    while True:
        try:
            user_input = await session.prompt_async("builder> ")
        except (EOFError, KeyboardInterrupt):
            print(Style.YELLOW("\nBye!"))
            break
        text = user_input.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "/quit", "/q", "/e"):
            print(Style.YELLOW("Bye!"))
            break
        if text == "/status":
            print(Style.GREEN(f"builder_coder | fast={coder_agent.amd.fast_llm_model} "
                              f"complex={coder_agent.amd.complex_llm_model}"))
            print(Style.GREEN(f"builder_navigator | fast={navi_agent.amd.fast_llm_model} "
                              f"complex={navi_agent.amd.complex_llm_model}"))
            continue
        if text == "/flows":
            for func, name, *_ in _build_toolbox_tools(isaa, app):
                if name == "flow_manage":
                    print(await func(action="list"))
                    break
            continue
        if text == "/help":
            print(Style.CYAN("Builder V2 (2 Agenten)"))
            print("  Bau-Auftrag -> CODER baut (1-10 Pakete, Mini-Edit ok).")
            print("  Danach/parallel: NAVIGATOR prueft (Terminal+Browser, Gate).")
            print("  /navi <text> = direkt an NAVIGATOR.")
            print("  /status /flows /help exit")
            continue
        if text.startswith("/navi "):
            await _run_agent_stream(navi_agent, text[len("/navi "):].strip(),
                                    session_id="builder_navigator")
            continue
        await _run_agent_stream(coder_agent, text, session_id="builder_coder")

    print(Style.YELLOW("Saving agent state..."))
    try:
        await isaa.on_exit()
    except Exception:
        pass
    print(Style.GREEN("Auf Wiedersehen!"))


if __name__ == "__main__":
    from toolboxv2 import get_app
    asyncio.run(run(get_app()))
