# ISAA-Auftrag 07 – Migration auf mcp 2 (Server, ISAA-Client, FlowAgent)

> Komplexität: **nach Owner-Entscheidung – mittel, 5 Dateien** · Fix-Aufträge: FIX-03 · Test-IDs: 45
> Braucht keine Antwort des Owners. Unabhängig von den anderen Aufträgen (01–07); läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** `pyproject.toml` verlangt `mcp>=1.6.0`, installiert wird mcp 2.x. Damit startet der ToolBoxV2-MCP-Server nicht (45 rote Tests), `FlowAgent.setup_mcp_server` meldet „MCP not available“, und der ISAA-Client sieht von einem angebundenen MCP-Server **0 statt 17 Werkzeuge**, ohne Fehlermeldung. Der Owner hat entschieden: **auf mcp 2 migrieren**. Die Migration ist auf `master` @ `ed32daf` mit mcp 2.3.0 getestet (Unit-Tests, echtes Protokoll, ISAA-Client, FlowAgent) und liegt als Patch bei.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu sieben ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/07 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa07 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa07
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e .
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa07 liegen
```

`.venv` und `.venv310` stehen in `.gitignore` bzw. werden nie committet. In jeder neuen Shell: venv aktivieren und die zwei Umgebungsvariablen setzen.

### 0.2 Regeln

1. **Nur die Dateien aus Abschnitt 1 ändern.** Jede andere Datei gehört einer anderen Instanz oder ist gesperrt.
2. **Kein Grün durch Wegsehen:** kein `skip`, `skipIf`, `xfail`, `importorskip`, kein `continue-on-error`, kein `|| true`, keinen Test löschen, keine Assertion abschwächen, keinen Erwartungswert auf den Ist-Wert setzen, ohne dass der Auftrag es sagt.
3. **Ein Commit je Fix-Auftrag.** Dateien einzeln hinzufügen (`git add <pfad> <pfad>`), nie `git add -A`, `git add .` oder `git commit -a`. Vor jedem Commit `git status --short` ansehen: nur deine Dateien. Nie committen: `.venv*`, `docs/.docs_index.json`, Test-Artefakte, Logs.
4. **Kein `--force`, kein `--no-verify`, keine Merge-Commits auf `master`, kein Umschreiben fremder Commits.**
5. **Owner-Umgebung schützen:** Ports 8467 und 8468 nicht belegen, wenn die Owner-Instanz läuft; keine fremden Prozesse beenden; keine Daten unter dem Haupt-Checkout oder im Benutzerprofil löschen. Keine Geheimnisse (API-Schlüssel, Tokens) in Dateien, Commits oder den Bericht.
6. **Stoppen und melden**, statt weiterzumachen, wenn: die Abnahme rot bleibt; eine Datei außerhalb deiner Liste geändert werden müsste; `git rebase` einen Konflikt meldet; der Code an der genannten Stelle anders aussieht als im Auftrag beschrieben. Dann `git rebase --abort` bzw. Änderung zurücksetzen (`git checkout -- <pfad>`), den Auftrag als „gestoppt“ in den Bericht, weiter mit dem nächsten Auftrag.

### 0.3 Commit und Push nach `master`

Commit-Nachricht: erste Zeile `FIX-NN (Karte): kurze Beschreibung`, darunter `WP2-Analyse, Karte <Karte>.` und die letzte Zeile der Abnahme-Ausgabe.

```powershell
git fetch origin
git rebase origin/master          # holt die Commits der anderen Instanzen unter deinen Commit
# Gate-Check (0.4) erneut laufen lassen, wenn der Rebase fremde Commits gebracht hat
git push origin HEAD:master
```

Wird der Push abgelehnt (`non-fast-forward`): die drei Schritte wiederholen, höchstens dreimal. Danach melden.

### 0.4 Gate: keine neue rote Test-ID

Lege dieses Skript **außerhalb** des Repos ab, als `C:\Users\Markin\Workspace\ISAABOX\tb_gate.py` (einmal; alle Instanzen nutzen dieselbe Datei):

```python
# tb_gate.py - Baseline/Check fuer rote Test-IDs (WP2). Liegt AUSSERHALB des Repos.
import json, os, re, subprocess, sys
def run(paths):
    env = dict(os.environ, MOCK_EXTERNAL_SERVICES="true", IS_OFFLINE_DB="true")
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "-rfE", "--tb=no", "-p", "no:cacheprovider"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    red = sorted({m.group(2).replace("\\", "/") for m in re.finditer(r"^(FAILED|ERROR) (\S+)", out, re.M)})
    summ = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    return red, (summ[-1] if summ else out[-400:])
mode, file, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
red, summ = run(paths)
print(summ)
if mode == "baseline":
    json.dump(red, open(file, "w", encoding="utf-8"), indent=0)
    print("Baseline gespeichert:", len(red), "rote IDs ->", file)
    sys.exit(0)
base = set(json.load(open(file, encoding="utf-8")))
new = [t for t in red if t not in base]
gone = [t for t in sorted(base) if t not in set(red)]
print(f"rot: {len(red)} | gruen geworden: {len(gone)} | NEU rot: {len(new)}")
for t in new:
    print("  NEU:", t)
sys.exit(1 if new else 0)
```

Dein Gate-Bereich:

```powershell
$GATE = @('toolboxv2/tests/test_isaa/test_mcp', 'toolboxv2/tests/test_mini/test_system/feature_tests_progressive.py', 'toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py')
```

**Vor der ersten Änderung** einmal die Baseline schreiben, **nach jedem Fix** den Check:

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa07_baseline.json @GATE
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa07_baseline.json @GATE   # Exit 0 = ok, 'NEU rot: 0'
```

`NEU rot` muss 0 sein. Erscheint nach einem `git rebase` eine neue rote ID, prüfe, ob sie von dir kommt:

1. `git switch --detach origin/master`, dann den Check erneut.
2. Auch dort rot → Ursache ist ein fremder Commit: ID im Bericht unter „fremd“ notieren, dort (auf `origin/master`) die Baseline neu schreiben, `git switch isaa/07`, weiterarbeiten.
3. Dort grün → Ursache bist du: `git switch isaa/07`, nicht pushen, Fehler suchen oder Fix zurücknehmen.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa07_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

- `toolboxv2/mods/isaa/base/Agent/flow_agent.py` ändert auch Auftrag 03 (nur Zeile 3613). Du änderst **nur** die Zeilen 72, 429, 4518 und 4529. Git führt beides beim `git rebase` zusammen; bei Konflikt stoppen (Regel 6).
- Deine venv hat vor der Änderung schon mcp 2.x (weil nichts gepinnt ist). Die Baseline zeigt deshalb ~46 rote MCP-Tests – das ist der Ausgangszustand.
- Nach dem Push installiert die CI mcp 2.3+ (`mcp>=2.3,<3`).

## 1. Deine Dateien

- pyproject.toml (nur die mcp-Zeile im Extra `isaa`), toolboxv2/mods/isaa/pyproject.toml (nur die mcp-Zeile), toolboxv2/mcp_server/server.py, toolboxv2/mods/isaa/extras/mcp_session_manager.py, toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeilen 72, 429, 4518, 4529)

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 01 (CI-Signal reparieren (Workflow-Dateien)): .github/workflows/build.yml (nur Job `py-check`); .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)
- Auftrag 02 (Minimale Fixes, alle vorab getestet): toolboxv2/tests/conftest.py; toolboxv2/tests/test_mini/test_system/test_session.py; toolboxv2/tests/test_clis/test_minicli_macros.py; toolboxv2/tests/test_clis/test_config_wizard.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py; toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py; toolboxv2/utils/extras/mkdocs.py; toolboxv2/mods/isaa/base/ai_semantic_memory.py; toolboxv2/tests/test_core/test_port_unification.py; toolboxv2/tests/test_core/test_tray_daemon_integration.py
- Auftrag 03 (VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle): toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py
- Auftrag 04 (Sporadische Tests stabilisieren): toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py
- Auftrag 05 (Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben): installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py; toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942); toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py; toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py; toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py; toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden; toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)
- Auftrag 06 (Eine Sitzungsdatei je Nutzer, mit Übernahme der alten Datei): toolboxv2/utils/system/session.py; toolboxv2/tests/test_mini/test_system/test_session.py (nur Zeilen 49, 59, 80, 104); toolboxv2/tests/test_mini/test_cli_session.py (nur `test_long_username_truncated`)
- Außerdem gesperrt (offene Owner-Fragen oder nicht Teil der Aufträge): `pyproject.toml` und `toolboxv2/mods/isaa/pyproject.toml` außer der mcp-Zeile (Auftrag 07), `toolboxv2/mcp_server/models.py`, `toolboxv2/tests/test_isaa/test_mcp/test_models.py`, `toolboxv2/utils/brodcast/`, `docs/`.

**Mit einer anderen Instanz geteilte Dateien (nur die genannten Zeilen):** `toolboxv2/mods/isaa/base/Agent/flow_agent.py` – du nur die Zeilen 72, 429, 4518, 4529; Auftrag 03 nur Zeile 3613.

## 2. Aufträge in dieser Reihenfolge

### FIX-03 – Karte K-ENV-02

**Ursache:** `mcp` ist ungepinnt (`mcp>=1.6.0`); installiert wird 2.x. Dort gibt es die Dekorator-API (`Server.list_tools()` …) und `mcp.server.fastmcp` nicht mehr, und Modelle liefern Felder nur noch unter snake_case-Namen (`input_schema`, `uri_template`).

**Beleg:** `toolboxv2/mcp_server/server.py:84, 747-790, 804-815`; `toolboxv2/mods/isaa/base/Agent/flow_agent.py:72, 429, 4518, 4529` (`FastMCP`); `toolboxv2/mods/isaa/extras/mcp_session_manager.py:306, 367, 395-397`; `pyproject.toml:189`, `toolboxv2/mods/isaa/pyproject.toml:28`. CI: `AttributeError: 'Server' object has no attribute 'list_tools'` (45). Unter mcp 2.3 zusätzlich: `No module named 'mcp.server.fastmcp'` und – ohne Fehlermeldung – der ISAA-Client meldet 0 statt 17 Werkzeuge eines MCP-Servers (`tool.inputSchema` wirft `AttributeError`, wird verschluckt).

**Warum:** Abhängigkeits-Drift, betrifft Nutzer genauso: der MCP-Server startet nicht, `FlowAgent.setup_mcp_server` meldet „MCP not available“, und ISAA sieht die Werkzeuge angebundener MCP-Server nicht. **Owner-Entscheidung 2026-11-15: auf mcp 2 migrieren** (nicht auf <2 pinnen).

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** pyproject.toml (nur die mcp-Zeile im Extra `isaa`), toolboxv2/mods/isaa/pyproject.toml (nur die mcp-Zeile), toolboxv2/mcp_server/server.py, toolboxv2/mods/isaa/extras/mcp_session_manager.py, toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeilen 72, 429, 4518, 4529)

**Vorher lesen:** `server.py:78-120, 740-820`; `mcp_session_manager.py:255-420`; `flow_agent.py:66-76, 4510-4535`

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_mcp toolboxv2/tests/test_mini/test_system/feature_tests_progressive.py -q   # mit mcp 2.3.0
```

Erwartet (Analyse): `47 failed, 187 passed, 5 skipped` (45× `'Server' object has no attribute 'list_tools'`, 1× Port K-API-15, 1× `test_disk_not_mutated` – lokal schon vorher rot, nicht im Katalog)

**Schritte:**

1. Patch aus diesem Auftrag anwenden (s. „Patch“). Gelingt `git apply --check` nicht, die Änderungen von Hand nach dem Patch machen – er ist die Vorlage.
2. venv neu installieren, damit mcp 2 drin ist: `uv pip install -e toolboxv2/mods/isaa` und `uv pip install -e .`; dann `python -c "import importlib.metadata as m; print(m.version('mcp'))"` → 2.3 oder höher.
3. Das Prüfskript aus „Zusätzliche Abnahme“ ausführen (prüft Protokoll, ISAA-Client und FlowAgent in einem Lauf).

**Nicht tun:** Keine Tests ändern. `test_isaa/test_mcp/test_models.py` (Port 8791) nicht anfassen – offene Frage. Kein Kompatibilitäts-Zweig für mcp 1.x.

**Abnahme:** `python -m pytest toolboxv2/tests/test_isaa/test_mcp toolboxv2/tests/test_mini/test_system/feature_tests_progressive.py -q` → höchstens 2 rot (`test_models.py::TestMCPConfig::test_default_values`, `feature_tests_progressive.py::TestNoSingletonBleeding::test_disk_not_mutated`), und das Prüfskript gibt die Zeilen 1–7 mit den Werten aus „Nachher“ aus (`2 tools: 17`, `6 isaa client tools: 17`, `7 flowagent: MCPServer ['agent_run'] R:hi`). Danach Gate-Check (0.4).

**Patch (getestet):** Speichere den Block unten als `C:\Users\Markin\Workspace\ISAABOX\isaa07.patch` (außerhalb des Repos, UTF-8; nicht mit `Out-File`/`>` aus Windows PowerShell 5.1, das schreibt UTF-16). Dann im Wurzelordner deiner Arbeitskopie:

```powershell
git fetch origin
git apply --3way --check C:\Users\Markin\Workspace\ISAABOX\isaa07.patch
git apply --3way C:\Users\Markin\Workspace\ISAABOX\isaa07.patch
git diff --cached --stat     # --3way legt die Änderung direkt in den Index
```

`--3way` nutzt die Blob-IDs im Patch und funktioniert auch mit CRLF-Zeilenenden in der Arbeitskopie. Meldet es einen Konflikt, hat sich eine Stelle seit `ed32daf` geändert: `git checkout -- .` und die Änderungen von Hand nach dem Patch übertragen; passt eine Stelle inhaltlich nicht mehr, stoppen (Regel 6).

<details><summary>Patch</summary>

```diff
diff --git a/pyproject.toml b/pyproject.toml
index f42f12e..cc6d1f8 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -186,7 +186,7 @@ isaa = [
     "pypdf2>=3.0.1",
     "nest-asyncio>=1.5.0",
     "pyvis>=0.3.0",
-    "mcp>=1.6.0",
+    "mcp>=2.3,<3",
     "tenacity>=9.1.2",
     "playwright",
     "html2text>=2025.4.15",
diff --git a/toolboxv2/mcp_server/server.py b/toolboxv2/mcp_server/server.py
index 3179e01..9805891 100644
--- a/toolboxv2/mcp_server/server.py
+++ b/toolboxv2/mcp_server/server.py
@@ -14,7 +14,6 @@ from typing import Any, Dict, List, Optional, Union
 
 # MCP SDK imports
 from mcp.server import Server, NotificationOptions
-from mcp.server.models import InitializationOptions
 import mcp.server.stdio
 import mcp.types as types
 from pydantic import AnyUrl
@@ -81,7 +80,15 @@ class ToolBoxV2MCPServer:
         self.config = config or MCPConfig()
 
         # MCP Server instance
-        self.server = Server(self.config.server_name)
+        # mcp>=2: handlers are passed to the constructor (no decorator API)
+        self.server = Server(
+            self.config.server_name,
+            version=self.config.server_version,
+            on_list_tools=self._on_list_tools,
+            on_list_resources=self._on_list_resources,
+            on_read_resource=self._on_read_resource,
+            on_call_tool=self._on_call_tool,
+        )
 
         # State
         self._app: Optional[Any] = None  # ToolBoxV2 App (lazy loaded)
@@ -742,52 +749,36 @@ class ToolBoxV2MCPServer:
     # =========================================================================
 
     def _setup_handlers(self):
-        """Setup MCP protocol handlers."""
-
-        @self.server.list_tools()
-        async def handle_list_tools() -> List[types.Tool]:
-            """Return static tool definitions immediately (no app loading)."""
-            tools = self.get_tool_definitions()
-            return [
-                types.Tool(
-                    name=t["name"],
-                    description=t["description"],
-                    inputSchema=t["inputSchema"]
-                )
-                for t in tools
-            ]
-
-        @self.server.list_resources()
-        async def handle_list_resources() -> List[types.Resource]:
-            """Return resource definitions."""
-            resources = self.get_resource_definitions()
-            return [
-                types.Resource(
-                    uri=r["uri"],
-                    name=r["name"],
-                    description=r["description"],
-                    mimeType=r["mimeType"]
-                )
-                for r in resources
-            ]
-
-        @self.server.read_resource()
-        async def handle_read_resource(uri: AnyUrl) -> str:
-            """Read a resource."""
-            return await self.read_resource(str(uri))
-
-        @self.server.call_tool()
-        async def handle_call_tool(
-            name: str,
-            arguments: Dict[str, Any]
-        ) -> List[types.TextContent]:
-            """Execute a tool."""
-            results = await self.call_tool(name, arguments)
-
-            return [
-                types.TextContent(type="text", text=r.content)
-                for r in results
-            ]
+        """Handlers are registered in __init__ via Server(on_...=...) (mcp>=2)."""
+
+    async def _on_list_tools(self, ctx, params) -> types.ListToolsResult:
+        """Return static tool definitions immediately (no app loading)."""
+        return types.ListToolsResult(tools=[
+            types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
+            for t in self.get_tool_definitions()
+        ])
+
+    async def _on_list_resources(self, ctx, params) -> types.ListResourcesResult:
+        """Return resource definitions."""
+        return types.ListResourcesResult(resources=[
+            types.Resource(uri=r["uri"], name=r["name"], description=r["description"], mimeType=r["mimeType"])
+            for r in self.get_resource_definitions()
+        ])
+
+    async def _on_read_resource(self, ctx, params) -> types.ReadResourceResult:
+        """Read a resource."""
+        uri = str(params.uri)
+        text = await self.read_resource(uri)
+        mime = next((r["mimeType"] for r in self.get_resource_definitions() if r["uri"] == uri), "text/plain")
+        return types.ReadResourceResult(contents=[types.TextResourceContents(uri=uri, mimeType=mime, text=text)])
+
+    async def _on_call_tool(self, ctx, params) -> types.CallToolResult:
+        """Execute a tool."""
+        results = await self.call_tool(params.name, params.arguments or {})
+        return types.CallToolResult(
+            content=[types.TextContent(type="text", text=r.content) for r in results],
+            isError=any(not r.success for r in results),
+        )
 
     # =========================================================================
     # SERVER LIFECYCLE
@@ -805,13 +796,8 @@ class ToolBoxV2MCPServer:
                 await self.server.run(
                     read_stream,
                     write_stream,
-                    InitializationOptions(
-                        server_name=self.config.server_name,
-                        server_version=self.config.server_version,
-                        capabilities=self.server.get_capabilities(
-                            notification_options=NotificationOptions(),
-                            experimental_capabilities={}
-                        )
+                    self.server.create_initialization_options(
+                        notification_options=NotificationOptions(),
                     )
                 )
         finally:
diff --git a/toolboxv2/mods/isaa/base/Agent/flow_agent.py b/toolboxv2/mods/isaa/base/Agent/flow_agent.py
index 1fe7e15..c9500ba 100644
--- a/toolboxv2/mods/isaa/base/Agent/flow_agent.py
+++ b/toolboxv2/mods/isaa/base/Agent/flow_agent.py
@@ -69,7 +69,7 @@ except ImportError:
         pass
 
 try:
-    from mcp.server.fastmcp import FastMCP
+    from mcp.server.mcpserver import MCPServer  # mcp>=2: FastMCP was renamed
     MCP_AVAILABLE = True
 except ImportError:
     MCP_AVAILABLE = False
@@ -426,7 +426,7 @@ class FlowAgent:
 
         # Servers
         self.a2a_server: A2AServer | None = None
-        self.mcp_server: FastMCP | None = None
+        self.mcp_server: "MCPServer | None" = None
 
         # Execution engine instance (lazy loaded)
         self._execution_engine_cache = None
@@ -4515,7 +4515,7 @@ class FlowAgent:
 
     def setup_mcp_server(self, name: str | None = None):
         try:
-            from mcp.server.fastmcp import FastMCP
+            from mcp.server.mcpserver import MCPServer  # mcp>=2: FastMCP was renamed
 
             MCP_AVAILABLE = True
         except ImportError:
@@ -4526,7 +4526,7 @@ class FlowAgent:
             return
 
         server_name = name or f"{self.amd.name}_MCP"
-        self.mcp_server = FastMCP(server_name)
+        self.mcp_server = MCPServer(server_name)
 
         @self.mcp_server.tool()
         async def agent_run(query: str, session_id: str = "mcp_session") -> str:
diff --git a/toolboxv2/mods/isaa/extras/mcp_session_manager.py b/toolboxv2/mods/isaa/extras/mcp_session_manager.py
index d52c501..9cd234a 100644
--- a/toolboxv2/mods/isaa/extras/mcp_session_manager.py
+++ b/toolboxv2/mods/isaa/extras/mcp_session_manager.py
@@ -302,7 +302,7 @@ class MCPSessionManager:
     async def _create_http_session(self, server_name: str, server_config: dict[str, Any]) -> ClientSession | None:
         """Create HTTP MCP session with timeout"""
         try:
-            import httpx
+            import httpx2 as httpx  # mcp>=2 transports take an httpx2.AsyncClient
             from mcp.client.streamable_http import streamable_http_client
 
             url = server_config.get("url", f"http://localhost:{server_config.get('port', 8000)}/mcp")
@@ -364,7 +364,7 @@ class MCPSessionManager:
                     capabilities['tools'][tool.name] = {
                         'name': tool.name,
                         'description': tool.description or '',
-                        'input_schema': tool.inputSchema,
+                        'input_schema': tool.input_schema,
                         'output_schema': getattr(tool, 'outputSchema', None),
                         'display_name': getattr(tool, 'title', tool.name)
                     }
@@ -392,9 +392,9 @@ class MCPSessionManager:
             try:
                 templates_response = await asyncio.wait_for(session.list_resource_templates(), timeout=3.0)
                 for template in templates_response.resourceTemplates:
-                    capabilities['resource_templates'][template.uriTemplate] = {
-                        'uri_template': template.uriTemplate,
-                        'name': template.name or template.uriTemplate,
+                    capabilities['resource_templates'][template.uri_template] = {
+                        'uri_template': template.uri_template,
+                        'name': template.name or template.uri_template,
                         'description': template.description or ''
                     }
             except TimeoutError:
diff --git a/toolboxv2/mods/isaa/pyproject.toml b/toolboxv2/mods/isaa/pyproject.toml
index b1be63a..d086da2 100644
--- a/toolboxv2/mods/isaa/pyproject.toml
+++ b/toolboxv2/mods/isaa/pyproject.toml
@@ -25,7 +25,7 @@ dependencies = [
     "pyvis>=0.3.0",
     "redis>=5.0.0",
     # "python-a2a[all]>=0.5.1",
-    "mcp>=1.6.0",
+    "mcp>=2.3,<3",
     "google-cloud-aiplatform>=1.38.0",
     "pyyaml>=6.0",
     "docker>=7.0.0",
```
</details>

**Zusätzliche Abnahme (Prüfskript):** Speichere den Block als `C:\Users\Markin\Workspace\ISAABOX\isaa07_check_mcp.py` und führe ihn im Wurzelordner deiner Arbeitskopie aus (venv aktiv): `python C:\Users\Markin\Workspace\ISAABOX\isaa07_check_mcp.py`. Das Skript schreibt nur in ein Temp-Verzeichnis.

```python
"""ISAA-Auftrag 07, Abnahme: echtes MCP-Protokoll (mcp>=2) gegen ToolBoxV2MCPServer,
ISAA-Client (MCPSessionManager.extract_capabilities) und FlowAgent.setup_mcp_server.
Start im Repo-Wurzelordner:  python <pfad>\\isaa07_check_mcp.py
Schreibt nur in ein Temp-Verzeichnis."""
import asyncio, os, sys, tempfile, types
TMP = tempfile.mkdtemp(prefix="isaa07_")
os.environ["TB_DATA_DIR"] = TMP
os.environ.setdefault("MOCK_EXTERNAL_SERVICES", "true"); os.environ.setdefault("IS_OFFLINE_DB", "true")
sys.path.insert(0, os.getcwd())
from unittest.mock import Mock
import anyio, importlib.metadata as md
from mcp import ClientSession
from mcp.server import NotificationOptions
from mcp.shared.memory import create_client_server_memory_streams
from toolboxv2.mcp_server.models import MCPConfig
from toolboxv2.mcp_server.server import ToolBoxV2MCPServer
import toolboxv2.mods.isaa.extras.mcp_session_manager as M
print("mcp version:", md.version("mcp"))

async def protocol():
    srv = ToolBoxV2MCPServer(MCPConfig(api_keys_file=os.path.join(TMP, "keys.json")))
    srv._initialized = True; srv._app = Mock()          # no app boot, like the unit tests
    async with create_client_server_memory_streams() as (cs, ss):
        async with anyio.create_task_group() as tg:
            opts = srv.server.create_initialization_options(notification_options=NotificationOptions())
            tg.start_soon(lambda: srv.server.run(ss[0], ss[1], opts))
            async with ClientSession(cs[0], cs[1]) as c:
                init = await c.initialize()
                print("1 init:", init.server_info.name, init.server_info.version)
                print("2 tools:", len((await c.list_tools()).tools))
                res = await c.list_resources(); print("3 resources:", len(res.resources))
                rr = await c.read_resource(str(res.resources[0].uri)); print("4 read_resource:", type(rr.contents[0]).__name__)
                ct = await c.call_tool("python_execute", {"code": "2 + 2"}); print("5 call_tool:", ct.is_error, ct.content[0].text.strip()[:20])
                caps = await M.MCPSessionManager().extract_capabilities(c, "tb"); print("6 isaa client tools:", len(caps["tools"]))
            tg.cancel_scope.cancel()
    srv.python_worker.close(); srv.api_keys.close()
anyio.run(protocol)

from toolboxv2.mods.isaa.base.Agent import flow_agent as fa
dummy = types.SimpleNamespace(amd=types.SimpleNamespace(name="Probe"))
async def a_run(q, session_id=None): return "R:" + q
dummy.a_run = a_run
fa.FlowAgent.setup_mcp_server(dummy)
print("7 flowagent:", type(dummy.mcp_server).__name__, [t.name for t in asyncio.run(dummy.mcp_server.list_tools())],
      asyncio.run(dummy.mcp_server.call_tool("agent_run", {"query": "hi"})).content[0].text)
```

Erwartete Ausgabe nach dem Fix (aus der Analyse): `2 failed, 232 passed, 5 skipped` (Port K-API-15 und `test_disk_not_mutated`, beide ohne Patch ebenso rot). Protokoll-Probe mit echtem `ClientSession` über In-Memory-Streams: `init server: toolboxv2_mcp 3.0.0`, `tools: 17`, `resources: 5`, `read_resource: TextResourceContents`, `call_tool: False 4`. ISAA-Client `extract_capabilities`: vorher `tools: 0`, nachher `tools: 17`. `FlowAgent.setup_mcp_server`: `server: MCPServer`, `tools: ['agent_run']`, Aufruf liefert `R:hi`.

**Test-IDs (45):** in der CI unter Windows rot und damit lokal rot erwartet: 45; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EAPIKeyWorkflow::test_complete_api_key_lifecycle
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EAPIKeyWorkflow::test_multiple_keys_different_permissions
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECaching::test_cache_bypass
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECaching::test_cache_hit
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECompleteWorkflow::test_admin_workflow
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECompleteWorkflow::test_developer_workflow
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPerformanceTracking::test_performance_metrics_recorded
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPerformanceTracking::test_performance_to_dict
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_app_available
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_error_handling
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_function_definition_and_use
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_multiline_code
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_persistent_variables
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_simple_expression
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_syntax_error
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EResourceReading::test_read_all_resources
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EServerLifecycle::test_resource_definitions_complete
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EServerLifecycle::test_server_creation_and_configuration
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EServerLifecycle::test_tool_definitions_complete
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_info_flows
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_info_modules
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_info_python_guide
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_status
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestLazyLoading::test_not_initialized_on_creation
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestLazyLoading::test_resource_definitions_without_loading
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestLazyLoading::test_tool_definitions_without_loading
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestPerformanceTracking::test_performance_recorded
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_get_resource_definitions
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_discovery
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_performance
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_python_guide
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_unknown
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_resource_structure
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_custom_config
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_default_config
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_managers_initialized
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_workers_initialized
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_all_enabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_docs_disabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_flows_disabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_python_disabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_tool_schema_structure
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolExecution::test_call_tool_toolbox_info
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolExecution::test_call_tool_toolbox_status
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolExecution::test_call_tool_unknown
```
</details>

**Commit:** `FIX-03 (K-ENV-02): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Bericht (0.5) geschrieben.
3. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa07`.
