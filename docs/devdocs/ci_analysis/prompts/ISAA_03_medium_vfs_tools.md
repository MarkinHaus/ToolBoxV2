# ISAA-Auftrag 03 – VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle

> Komplexität: **3 von 5 – mittel, mehrere Dateien** · Fix-Aufträge: FIX-17, FIX-18 (Teil WS-Port) · Test-IDs: 47
> Braucht keine Antwort des Owners. Unabhängig von den anderen Aufträgen (01–07); läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** `vfs_view` ist seit dem VFS-Umbau eine Koroutine, `vfs_shell` verlangt zusätzlich `reason`. 44 Tests und der eingebaute Health-Check des Agenten rufen beide noch alt auf. Der Health-Check ist ein echter Produktfehler (das Werkzeug meldet im laufenden Agenten immer `FAILED`). Dazu ein Port-Test, der den alten WebSocket-Port 8100 erwartet. Alles wurde auf `master` @ `ed32daf` probehalber angewendet und geprüft.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu sieben ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/03 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa03 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa03
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e .
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa03 liegen
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
$GATE = @('toolboxv2/tests/test_isaa/test_base/test_agent', 'toolboxv2/tests/test_mini/test_manifest')
```

**Vor der ersten Änderung** einmal die Baseline schreiben, **nach jedem Fix** den Check:

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa03_baseline.json @GATE
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa03_baseline.json @GATE   # Exit 0 = ok, 'NEU rot: 0'
```

`NEU rot` muss 0 sein. Erscheint nach einem `git rebase` eine neue rote ID, prüfe, ob sie von dir kommt:

1. `git switch --detach origin/master`, dann den Check erneut.
2. Auch dort rot → Ursache ist ein fremder Commit: ID im Bericht unter „fremd“ notieren, dort (auf `origin/master`) die Baseline neu schreiben, `git switch isaa/03`, weiterarbeiten.
3. Dort grün → Ursache bist du: `git switch isaa/03`, nicht pushen, Fehler suchen oder Fix zurücknehmen.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa03_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

Der Gate-Bereich enthält `test_agent_fetures.py`; die Datei braucht allein mehrere Minuten. Das ist normal.

## 1. Deine Dateien

- toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py
- toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 01 (CI-Signal reparieren (Workflow-Dateien)): .github/workflows/build.yml (nur Job `py-check`); .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)
- Auftrag 02 (Minimale Fixes, alle vorab getestet): toolboxv2/tests/conftest.py; toolboxv2/tests/test_mini/test_system/test_session.py; toolboxv2/tests/test_clis/test_minicli_macros.py; toolboxv2/tests/test_clis/test_config_wizard.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py; toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py; toolboxv2/utils/extras/mkdocs.py; toolboxv2/mods/isaa/base/ai_semantic_memory.py; toolboxv2/tests/test_core/test_port_unification.py; toolboxv2/tests/test_core/test_tray_daemon_integration.py
- Auftrag 04 (Sporadische Tests stabilisieren): toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py
- Auftrag 05 (Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben): installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py; toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942); toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py; toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py; toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py; toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden; toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)
- Auftrag 06 (Eine Sitzungsdatei je Nutzer, mit Übernahme der alten Datei): toolboxv2/utils/system/session.py; toolboxv2/tests/test_mini/test_system/test_session.py (nur Zeilen 49, 59, 80, 104); toolboxv2/tests/test_mini/test_cli_session.py (nur `test_long_username_truncated`)
- Auftrag 07 (Migration auf mcp 2 (Server, ISAA-Client, FlowAgent)): pyproject.toml (nur die mcp-Zeile im Extra `isaa`), toolboxv2/mods/isaa/pyproject.toml (nur die mcp-Zeile), toolboxv2/mcp_server/server.py, toolboxv2/mods/isaa/extras/mcp_session_manager.py, toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeilen 72, 429, 4518, 4529)
- Außerdem gesperrt (offene Owner-Fragen oder nicht Teil der Aufträge): `pyproject.toml` und `toolboxv2/mods/isaa/pyproject.toml` außer der mcp-Zeile (Auftrag 07), `toolboxv2/mcp_server/models.py`, `toolboxv2/tests/test_isaa/test_mcp/test_models.py`, `toolboxv2/utils/brodcast/`, `docs/`.

**Mit einer anderen Instanz geteilte Dateien (nur die genannten Zeilen):** `toolboxv2/mods/isaa/base/Agent/flow_agent.py` – du nur Zeile 3613; Auftrag 07 nur die Zeilen 72, 429, 4518, 4529.

## 2. Aufträge in dieser Reihenfolge

### FIX-17 – Karte K-API-02

**Ursache:** `vfs_view` ist jetzt eine Koroutine und `vfs_shell` hat die Signatur `(reason, command)`. Die Tests rufen beide noch synchron bzw. ohne `reason` auf – und der eingebaute Gesundheitscheck des Produkts ebenfalls.

**Beleg:** `toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py:445` `def vfs_shell(reason: str, command: str)`, `:1196` `async def vfs_view(`. `toolboxv2/mods/isaa/base/Agent/flow_agent.py:3613` `"live_test_inputs": [{"command": "ls /"}]` (ohne `reason`). CI: `TypeError: 'coroutine' object is not subscriptable` (29), `<coroutine object make_vfs_view.<locals>.vfs_view …> is not an instance of <class 'dict'>`, `'closed' != 'open'`, `vfs_shell() missing 1 required positional argument: 'command'` (5), `vfs_shell health: FAILED — … missing 1 required positional argument: 'reason'`

**Warum:** Schnittstelle bewusst geändert (Docstring), Tests nicht nachgezogen. `'closed' != 'open'` ist derselbe Fehler: die Koroutine wurde nie ausgeführt, die Datei nie geöffnet. Der Gesundheitscheck (`tool_manager.py:1116` ruft `entry.function(**live_test_inputs[0])`) ist ein **echter Produktfehler**: `vfs_shell` meldet im laufenden Agenten bei jedem Health-Check `FAILED`.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py

**Vorher lesen:** `vfs_shell_tool.py:437-450` und `:1188-1200`; `tool_manager.py:1080-1130` (`health_check_single`); `flow_agent.py:3608-3640` (`_TOOL_HEALTH_EXTENSIONS`); in jeder Testdatei, wie `self.view`/`self.sh` gebaut werden

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shared_store_edits.py toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py -q
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py -q -k "test_03_per_tool_subtests or test_07_vfs_shell_health_check"
```

Erwartet (Analyse): erste Zeile: 44 der 46 IDs rot; zweite Zeile: `2 failed, 1 passed, 225 deselected`

**Schritte:**

1. `flow_agent.py:3613`: `"live_test_inputs": [{"command": "ls /"}],` → `"live_test_inputs": [{"reason": "health check", "command": "ls /"}],`. Sonst nichts in der Datei ändern (die Zeile `vfs_view` mit `/system_context.md` gehört zu einer anderen Karte).
2. In die vier vfs-Testdateien vor der ersten `class`-Zeile einfügen:

```python
def _sync_tool(fn):
    """vfs_view is a coroutine function since the async VFS refactor; these sync
    tests call it directly, so run it to completion on a private loop."""
    import asyncio, functools, inspect
    if not inspect.iscoroutinefunction(fn):
        return fn
    @functools.wraps(fn)
    def _run(*a, **k):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*a, **k))
        finally:
            loop.close()
    return _run
```
3. Dort jedes `make_vfs_view(<arg>)` – nicht die Import-Zeile – durch `_sync_tool(make_vfs_view(<arg>))` ersetzen (test_vfs_v2.py 5, test_vfs_shell_tool.py 1, test_vfs_shell_bullet_proof.py 2, test_vfs_shared_store_edits.py 1).
4. `test_agent_session_tools.py`, Klasse `TestVFSToolContracts.setUp`: die zwei Zeilen

```python
        self.sh = make_vfs_shell(self.session)
        self.view = make_vfs_view(self.session)
```

ersetzen durch

```python
        _sh = make_vfs_shell(self.session)
        # vfs_shell(reason, command): reason is required since the VFS tool refactor
        self.sh = lambda command: _sh("test", command)
        self.view = make_vfs_view(self.session)  # coroutine function: await its result
```

und in derselben Klasse die drei Aufrufe `self.view(` zu `await self.view(` machen (`self._check(self.view(` → `self._check(await self.view(`, zweimal `r = self.view(` → `r = await self.view(`).

**Nicht tun:** `vfs_shell_tool.py` nicht ändern (die neue Signatur ist gewollt). `test_self_agents_default_tool_set.py` und `test_agent_fetures.py` nicht ändern – sie werden durch Schritt 1 grün.

**Abnahme:** Repro-Befehle der Karte: erste Zeile genau 3 rot (`test_close_others_does_not_close_system_files`, `test_close_others_preserves_system_files`, `test_write_with_tabs`), zweite Zeile `2 passed`. Danach Gate-Check (0.4).

**Test-IDs (46):** in der CI unter Windows rot und damit lokal rot erwartet: 46; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestSessionToolsLive::test_07_vfs_shell_health_check
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_26_vfs_shell_ls_contract
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_27_vfs_shell_result_has_four_keys
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_28_vfs_shell_fail_has_error_info
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_29_vfs_view_returns_dict
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_30_vfs_view_success_has_content_and_showing
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_31_vfs_view_fail_has_error_key
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_32_vfs_shell_unknown_cmd_stderr_has_hint
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_33_vfs_shell_write_then_cleanup
toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py::TestSelfAgentLiveTools::test_04_no_failed_tools
toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py::TestSelfAgentLiveTools::test_05_skipped_tools_need_attention
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shared_store_edits.py::TestAgentRoundtripSharedMount::test_view_after_edit_shows_new_content
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestAgentRoundtrip::test_grep_find_edit_verify_single_line
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_close_others_workflow
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_line_range_precise
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_scroll_to_class_shows_context
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_scroll_to_method
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_scroll_to_nonexistent_returns_hint
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_view_returns_total_lines
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_basic_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_close_others
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_close_others_preserves_system_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_line_range
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_missing_file_fails
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_result_has_required_keys
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_scroll_to
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_scroll_to_not_found_hint
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewCloseOthers::test_close_others_closes_open_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewCloseOthers::test_close_others_does_not_close_system_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewCloseOthers::test_close_others_note_in_result
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_full_file
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range_clamps_end
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range_clamps_start
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range_updates_vfs_state
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewReturnSchema::test_basic_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewReturnSchema::test_nonexistent
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewReturnSchema::test_scroll_to
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_case_insensitive
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_context_lines
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_finds_line
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_missing_pattern
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_overrides_line_range
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_regex
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_sets_state_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewWorkflow::test_full_xy_workflow
```
</details>

**Commit:** `FIX-17 (K-API-02): …`

### FIX-18 (Teil WS-Port) – Karte K-API-15

**Ursache:** `test_manifest_schema.py::TestWorkersConfig::test_default_workers` erwartet den alten WebSocket-Port 8100; seit Commit `d1db5c9` ist der kanonische Port 8468.

**Beleg:** `toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py:106` `assert config.websocket[0].port == 8100` → `AssertionError: assert 8468 == 8100`. Commit `d1db5c9`: „One local port pair: HTTP 8467, WS 8468 (was 5000/8080/8100/5001)“. `test_core/test_port_unification.py` verlangt bereits, dass `'8100'` nirgends mehr vorkommt.

**Warum:** Absichtliche, dokumentierte Port-Vereinheitlichung; nur dieser Test ist nicht nachgezogen.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py

**Vorher lesen:** `test_default_workers` (Zeilen 100–107); `toolboxv2/utils/manifest/schema.py`, Klasse `WorkersConfig`

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py -q
```

Erwartet (Analyse): `1 failed, 12 passed`

**Schritte:**

1. Zeile 106: `assert config.websocket[0].port == 8100` → `assert config.websocket[0].port == 8468`.

**Nicht tun:** Die Zeile `http[0].port == 8000` nicht ändern. `toolboxv2/mcp_server/` und `test_isaa/test_mcp/test_models.py` nicht anfassen (dort ist eine Frage an den Owner offen).

**Abnahme:** `python -m pytest toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py -q` → `13 passed`. Danach Gate-Check (0.4).

**Test-IDs (1):** in der CI unter Windows rot und damit lokal rot erwartet: 1; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py::TestWorkersConfig::test_default_workers
```
</details>

**Commit:** `FIX-18 (K-API-15): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Bericht (0.5) geschrieben.
3. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa03`.
