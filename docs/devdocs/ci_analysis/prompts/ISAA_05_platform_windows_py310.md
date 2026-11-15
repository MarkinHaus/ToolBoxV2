# ISAA-Auftrag 05 – Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben

> Komplexität: **5 von 5 – am schwersten (Hypothesen ungetestet)** · Fix-Aufträge: FIX-35, FIX-36, FIX-37, FIX-38, FIX-41, FIX-43, FIX-42 · Test-IDs: 34
> Braucht keine Antwort des Owners. Unabhängig von den anderen Aufträgen (01–07); läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Fehler, die in der CI nur unter Windows, nur unter Python 3.10 oder nur auf dem Linux-Runner auftreten. Die Analyse lief unter Linux; die Ursachen sind **Hypothesen, nicht geprüft**. Du arbeitest auf dem Windows-Rechner des Owners und kannst sie als Erste prüfen. Vorgehen je Auftrag: reproduzieren → Ursache belegen → beheben → prüfen. `FIX-35` (Installer-Protokoll unter Windows) betrifft den Installationspfad des Produkts und hat Vorrang.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu sieben ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/05 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa05 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa05
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e .
# zusätzlich für die Aufträge mit Python 3.10:
uv venv -p 3.10 .venv310
uv pip install --python .venv310\Scripts\python.exe pytest pytest-asyncio
uv pip install --python .venv310\Scripts\python.exe -e toolboxv2/mods/isaa
uv pip install --python .venv310\Scripts\python.exe -e .
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa05 liegen
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
$GATE = @('toolboxv2/tests/test_core', 'toolboxv2/tests/test_isaa/test_shell_tool.py', 'toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs', 'toolboxv2/tests/test_mini/test_cloudm/test_LiveSync', 'toolboxv2/tests/test_isaa/test_codig/test_base_coder.py')
```

**Vor der ersten Änderung** einmal die Baseline schreiben, **nach jedem Fix** den Check:

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa05_baseline.json @GATE
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa05_baseline.json @GATE   # Exit 0 = ok, 'NEU rot: 0'
```

`NEU rot` muss 0 sein. Erscheint nach einem `git rebase` eine neue rote ID, prüfe, ob sie von dir kommt:

1. `git switch --detach origin/master`, dann den Check erneut.
2. Auch dort rot → Ursache ist ein fremder Commit: ID im Bericht unter „fremd“ notieren, dort (auf `origin/master`) die Baseline neu schreiben, `git switch isaa/05`, weiterarbeiten.
3. Dort grün → Ursache bist du: `git switch isaa/05`, nicht pushen, Fehler suchen oder Fix zurücknehmen.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa05_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

1. **Erst reproduzieren.** Schlägt der Repro-Befehl lokal *nicht* fehl, rate keinen Fix. Notiere „lokal grün, nur auf dem CI-Runner rot“, lege – wenn möglich – Diagnose-Ausgaben fest, die beim nächsten CI-Lauf die Ursache zeigen (nur wenn das ohne Änderung am Testergebnis geht), und gehe zum nächsten Auftrag.
2. **Ursache belegen**, bevor du änderst: Datei:Zeile und die wörtliche Fehlermeldung kommen in den Bericht.
3. **Testdateien der VFS-Tests** (`test_vfs_v2.py`, `test_vfs_shell_tool.py`, `test_vfs_session_tools.py`, `test_vfs_pooling.py`) bearbeiten andere Instanzen. Für `FIX-37` änderst du nur Produktcode.
4. `FIX-42` (`test_state_system`, nur auf dem Ubuntu-Runner rot) machst du nur, wenn eine Linux-Umgebung (z. B. WSL) da ist; sonst „nicht bearbeitet, keine Linux-Umgebung“.

## 1. Deine Dateien

- installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py
- toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942)
- toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py
- toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py
- toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py
- toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden
- toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 01 (CI-Signal reparieren (Workflow-Dateien)): .github/workflows/build.yml (nur Job `py-check`); .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)
- Auftrag 02 (Minimale Fixes, alle vorab getestet): toolboxv2/tests/conftest.py; toolboxv2/tests/test_mini/test_system/test_session.py; toolboxv2/tests/test_clis/test_minicli_macros.py; toolboxv2/tests/test_clis/test_config_wizard.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py; toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py; toolboxv2/utils/extras/mkdocs.py; toolboxv2/mods/isaa/base/ai_semantic_memory.py; toolboxv2/tests/test_core/test_port_unification.py; toolboxv2/tests/test_core/test_tray_daemon_integration.py
- Auftrag 03 (VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle): toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py
- Auftrag 04 (Sporadische Tests stabilisieren): toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py
- Auftrag 06 (Eine Sitzungsdatei je Nutzer, mit Übernahme der alten Datei): toolboxv2/utils/system/session.py; toolboxv2/tests/test_mini/test_system/test_session.py (nur Zeilen 49, 59, 80, 104); toolboxv2/tests/test_mini/test_cli_session.py (nur `test_long_username_truncated`)
- Auftrag 07 (Migration auf mcp 2 (Server, ISAA-Client, FlowAgent)): pyproject.toml (nur die mcp-Zeile im Extra `isaa`), toolboxv2/mods/isaa/pyproject.toml (nur die mcp-Zeile), toolboxv2/mcp_server/server.py, toolboxv2/mods/isaa/extras/mcp_session_manager.py, toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeilen 72, 429, 4518, 4529)
- Außerdem gesperrt (offene Owner-Fragen oder nicht Teil der Aufträge): `pyproject.toml` und `toolboxv2/mods/isaa/pyproject.toml` außer der mcp-Zeile (Auftrag 07), `toolboxv2/mcp_server/models.py`, `toolboxv2/tests/test_isaa/test_mcp/test_models.py`, `toolboxv2/utils/brodcast/`, `docs/`.

## 2. Aufträge in dieser Reihenfolge

### FIX-35 – Karte K-OS-01

**Ursache:** Unter Windows schlagen die Tests des Installer-Protokolls (`installer.ps1`) und von `verify_install_windows.ps1` fehl.

**Beleg:** CI nur Windows (alle 4 Versionen): `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` (7), `AssertionError: 1 != 0` (2), `FileNotFoundError: … \\Temp\\tb_update_…\\uv.log` (2), `'core_build: "new-build"' not found in 'installer_version: "1.0.0" …'`, `'ToolBoxV2[cli,web,isaa] @ file://' not found in ''`, `test_verify_install_windows: 'FAIL' != 'PASS'`

**Warum:** Die Tests starten `installer.ps1` und lesen JSON-Zeilen von stdout. `Expecting value … char 0` heißt: stdout ist leer oder beginnt nicht mit JSON. Kandidaten: Skript bricht früh ab (Exit-Code 1), PowerShell-Version/Ausführungsrichtlinie auf dem Runner, BOM/Kodierung (R9 hat ein Doppel-BOM behoben), oder stdout/stderr vertauscht. **Produktpfad von WP1** (Installation und Kern-Update unter Windows).

> **Hypothese ungetestet.** UNGETESTET (nur unter Windows; Analyseumgebung ist Linux, dort 16 passed bzw. 5 skipped). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py

**Vorher lesen:** `toolboxv2/tests/test_core/test_installer_protocol.py` (wie das Skript gestartet wird), `installer.ps1` (Protokoll auf fd 3/stdout), Handoff R8/R9

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_core/test_installer_protocol.py toolboxv2/tests/test_core/test_verify_install_windows.py -q   # unter Windows
```

Erwartet (Analyse): CI: 14 rot

**Schritte:**

1. Auf dem Windows-Rechner des Owners: Befehl aus `Repro` ausführen. Sind die Tests dort grün, liegt es am Runner (PowerShell-Version, fehlendes `uv`, Ausführungsrichtlinie) → stderr des Skripts im Test mit ausgeben und einen CI-Lauf ansehen.
2. Sind sie dort rot: stdout/stderr des Skripts ansehen, erste Zeile, Exit-Code.
3. Karte mit dem Befund ergänzen (Status GETESTET), dann beheben.

**Nicht tun:** Keine `skipIf(win32)`.

**Abnahme:** 14 IDs grün in allen 4 Windows-Jobs. Danach Gate-Check (0.4).

**Test-IDs (14):** in der CI unter Windows rot und damit lokal rot erwartet: 14; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_exit_ok_and_done_event
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_manifest_records_new_build_and_keeps_the_rest
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_profile_and_env_untouched
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_wheel_reinstalled_with_the_installed_features
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_bundled_uv_sidecar_is_used
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_ends_with_done
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_exit_ok_and_config_path_parsed
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_fail_emits_error_event
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_progress_is_fraction_and_monotonic_enough
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_stdout_is_only_json_with_type
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_tauri_bundle_copy_in_sync
toolboxv2/tests/test_core/test_installer_protocol.py::TestWheelAndFeaturesDryRun::test_dry_run_names_pkg_with_extras_and_file_url
toolboxv2/tests/test_core/test_installer_protocol.py::TestWheelAndFeaturesDryRun::test_exit_ok_and_json_only
toolboxv2/tests/test_core/test_verify_install_windows.py::TestVerifyInstallWindows::test_all_v1_to_v7_pass_and_exit_nonzero_due_v8
```
</details>

**Commit:** `FIX-35 (K-OS-01): …`

### FIX-36 – Karte K-OS-02

**Ursache:** Unter Windows enthält die Ausgabe des Shell-Werkzeugs den PowerShell-Prompt (`hiPS D:\a\ToolBoxV2\ToolBoxV2>`).

**Beleg:** CI nur Windows: `AssertionError: 'hiPS D:\\a\\ToolBoxV2\\ToolBoxV2>' != 'hi'`, `'okPS …>' != 'ok'`, `'bPS …>' != 'b'`, `'' != 'legacy'` (4 IDs in `test_shell_tool.py`)

**Warum:** Wahrscheinlich Produktfehler: die dauerhafte PowerShell-Sitzung schreibt nach dem Befehl den Prompt, und die Ende-Markierung schneidet ihn nicht ab. Ein Agent bekäme unter Windows bei jedem Befehl den Prompt im Ergebnis.

> **Hypothese ungetestet.** UNGETESTET (nur unter Windows). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942)

**Vorher lesen:** `shell_tool_function`, Marker-Erkennung, `TestMarkerPlumbing`

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_shell_tool.py -q   # unter Windows
```

Erwartet (Analyse): CI: 4 rot

**Schritte:**

1. Unter Windows reproduzieren.
2. Prompt unterdrücken (`function prompt {''}` beim Sitzungsstart) oder Ausgabe bis zur Markierung sauber schneiden.
3. `test_fallback_when_marker_is_unsupported` einzeln prüfen (`'' != 'legacy'`).

**Nicht tun:** Test nicht auf `startswith` abschwächen.

**Abnahme:** 4 IDs grün unter Windows; Linux/macOS unverändert grün. Danach Gate-Check (0.4).

**Test-IDs (4):** in der CI unter Windows rot und damit lokal rot erwartet: 4; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_shell_tool.py::TestCompletion::test_finished_status_and_exit_code
toolboxv2/tests/test_isaa/test_shell_tool.py::TestMarkerPlumbing::test_fallback_when_marker_is_unsupported
toolboxv2/tests/test_isaa/test_shell_tool.py::TestNoneCommand::test_none_for_optional_args
toolboxv2/tests/test_isaa/test_shell_tool.py::TestSessions::test_new_session_resets_state
```
</details>

**Commit:** `FIX-36 (K-OS-02): …`

### FIX-37 – Karte K-OS-06

**Ursache:** Unter Windows schlagen VFS-Verzeichnisoperationen (`rmdir`, `rm -rf`, `mv` in ein Verzeichnis) und ein Rechte-Test (`'666' != '600'`) fehl.

**Beleg:** CI nur Windows: `test_vfs_v2.py::TestVFSV2Directories::test_rmdir_force…` `False is not true`, `::TestVfsShellWrite::test_mv_into_directo…`, `::test_rm_rf_directory` `True is not false`; `test_vfs_shell_tool.py::TestShellDirectoryCommands::…`; `test_vfs_session_tools.py::TestDirectoryOps::test_rmdir…`; `test_LiveSync_init.py::TestShareRegistry::test_store_file_i…` `'666' != '600'`

**Warum:** (1) Vermutlich Pfadtrenner: das VFS vergleicht Präfixe mit `/`, unter Windows entstehen an einer Stelle `\`-Pfade (z. B. `os.path.join`/`PurePath` statt `posixpath`). Wäre ein Produktfehler unter Windows. (2) `chmod 600` wirkt unter Windows nicht; `stat` meldet 666.

> **Hypothese ungetestet.** UNGETESTET (nur unter Windows). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py

**Vorher lesen:** `rmdir`, `rm`, `mv` im VFS; `ShareRegistry` Dateirechte

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py -q -k "rmdir or rm_rf or mv_into"   # unter Windows
```

Erwartet (Analyse): CI: 6 rot

**Schritte:**

1. Unter Windows reproduzieren; im VFS alle Pfadoperationen auf `posixpath` prüfen.
2. Rechte-Test: unter Windows prüfen, was `ShareRegistry` zusichern kann; Produkt (ACL) oder Test (plattformbezogene Erwartung) nach Befund.

**Nicht tun:** Kein skip für Windows ohne Owner-Bestätigung.

**Abnahme:** 6 IDs grün unter Windows. Danach Gate-Check (0.4).

**Test-IDs (6):** in der CI unter Windows rot und damit lokal rot erwartet: 6; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestDirectoryOps::test_rmdir_force_removes_contents
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestShellDirectoryCommands::test_rm_rf_directory
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVFSV2Directories::test_rmdir_force
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsShellWrite::test_mv_into_directory
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsShellWrite::test_rm_rf_directory
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_init.py::TestShareRegistry::test_store_file_is_encrypted_and_private
```
</details>

**Commit:** `FIX-37 (K-OS-06): …`

### FIX-38 – Karte K-OS-03

**Ursache:** Tests vergleichen Temp-Pfade als Text; das Betriebssystem liefert eine andere Schreibweise desselben Pfads (macOS `/var/…` ↔ `/private/var/…`, Windows `RUNNER~1` ↔ `runneradmin`).

**Beleg:** macOS und Windows: `test_endpoint_dist_dir.py` `PosixPath('/var/folders/…') != PosixPath('/private/var/folders/…')` (2). Windows: `test_base_coder.py` `'C:\\Users\\RUNNER~1\\AppData\\Local\\Temp\\coder_cp_…' not found in …`, `Worktree C:\Users\RUNNER~1\… not in: C:/Users/runneradmin/…` (2)

**Warum:** Eine Seite ruft `resolve()`/`realpath` auf, die andere nicht. Unter Linux fällt es nicht auf.

> **Hypothese ungetestet.** UNGETESTET (nur macOS/Windows). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py

**Vorher lesen:** die 4 Tests und die Produktfunktion, die den Pfad liefert

**Repro (vor der Änderung, muss rot sein):**

```powershell
nur macOS/Windows
```

Erwartet (Analyse): CI: 4 rot

**Schritte:**

1. Im Test den Temp-Pfad direkt nach `mkdtemp()` mit `os.path.realpath` auflösen und nur noch den aufgelösten verwenden.
2. Bei `test_base_coder` zusätzlich Trenner normalisieren (`Path(x).resolve()` auf beiden Seiten).

**Nicht tun:** Nicht auf Teilstring-Vergleich des letzten Ordnernamens abschwächen.

**Abnahme:** 4 IDs grün (macOS: 2, Windows: 4). Danach Gate-Check (0.4).

**Test-IDs (4):** in der CI unter Windows rot und damit lokal rot erwartet: 4; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_core/test_endpoint_dist_dir.py::TestDistDirManifestBase::test_dist_dir_relative_manifest_value_resolved_against_root
toolboxv2/tests/test_core/test_endpoint_dist_dir.py::TestDistDirManifestBase::test_dist_dir_uses_manifest_base_dir
toolboxv2/tests/test_isaa/test_codig/test_base_coder.py::TestCoderAgentBash::test_cwd_is_worktree
toolboxv2/tests/test_isaa/test_codig/test_base_coder.py::TestGitWorktreeReal::test_setup_creates_worktree
```
</details>

**Commit:** `FIX-38 (K-OS-03): …`

### FIX-41 – Karte K-PY-02

**Ursache:** Zwei Tests laufen nur unter 3.10 nicht: `asyncio.timeout` gibt es erst ab 3.11; `'NoneType' object has no attribute 'get_memory'`.

**Beleg:** CI nur 3.10: `test_performance_and_limiter.py::TestAgentPerformance::test_rate_limiter_blocks_correctly - AttributeError: module 'asyncio' has no attribute 'timeout'`; `test_agent_performance_stability.py::TestAgentPerformance::test_vfs_mount_blocking_analysis - AttributeError: 'NoneType' object has no attribute 'get_memory'`

**Warum:** (1) Der Test selbst nutzt `asyncio.timeout(...)`; im Produktcode kommt der Aufruf nicht vor (`grep` leer). `requires-python = ">=3.10"`. (2) Ungeklärt, vermutlich Folgefehler eines 3.10-Imports.

> **Hypothese ungetestet.** UNGETESTET (Probe-Fix nicht angewendet; zweite ID ungeklärt). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py

**Vorher lesen:** die beiden Testmethoden

**Repro (vor der Änderung, muss rot sein):**

```powershell
.\.venv310\Scripts\python -m pytest toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py -q
```

Erwartet (Analyse): CI: 2 rot unter 3.10

**Schritte:**

1. `async with asyncio.timeout(x): await c` → `await asyncio.wait_for(c, timeout=x)`.
2. Zweite ID unter 3.10 einzeln laufen lassen, Traceback lesen, Ursache in die Karte nachtragen, dann beheben.

**Nicht tun:** 3.10 nicht aus der Matrix nehmen (offene Owner-Entscheidung).

**Abnahme:** beide IDs grün unter 3.10. Danach Gate-Check (0.4).

**Test-IDs (2):** in der CI unter Windows rot und damit lokal rot erwartet: 2; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py::TestAgentPerformance::test_vfs_mount_blocking_analysis
toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py::TestAgentPerformance::test_rate_limiter_blocks_correctly
```
</details>

**Commit:** `FIX-41 (K-PY-02): …`

### FIX-43 – Karte K-STATE-04

**Ursache:** `test_job_execution_state.py`: `ValueError: Could not find isaa in ['isaa'] …` – das Modul `isaa` wird im Test über `get_mod` gesucht und nicht geladen.

**Beleg:** CI: 3 IDs, nur 3.10 (ubuntu, windows) und windows/3.13. Einzeln lokal unter 3.12: `3 passed`.

**Warum:** Abhängig von Reihenfolge oder Ladezeit: `get_mod('isaa')` findet das Modul in der Liste, kann es aber nicht laden (Importfehler in `isaa` unter 3.10, z. B. Syntax/Typing, oder Zeitüberschreitung unter Windows). Die eigentliche Ausnahme wird verschluckt („use --debug for infos“).

> **Hypothese ungetestet.** UNGETESTET (unter 3.10 lokal nicht einzeln geprüft; Ursache der verschluckten Ausnahme unbekannt). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden

**Vorher lesen:** `toolboxv2/utils/toolbox.py` (`get_mod`, Meldung „Could not find“), `toolboxv2/flows/isaa/icli.py`

**Repro (vor der Änderung, muss rot sein):**

```powershell
.\.venv310\Scripts\python -m pytest toolboxv2/tests/test_core/test_job_execution_state.py -q
```

Erwartet (Analyse): CI: 3 rot

**Schritte:**

1. Unter Python 3.10 den Befehl aus `Repro` mit `--debug`/Log-Ausgabe laufen lassen und die verschluckte Ausnahme sichtbar machen.
2. Ist es ein 3.10-Syntax-/Typing-Fehler in ISAA: dort korrigieren. Ist es Reihenfolge: Test abschirmen.

**Nicht tun:** Kein skip für 3.10.

**Abnahme:** 3 IDs grün unter 3.10 (ubuntu, windows) und windows/3.13. Danach Gate-Check (0.4).

**Test-IDs (3):** in der CI unter Windows rot und damit lokal rot erwartet: 3; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h1_create_execution_crash_leaves_ghost_run
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h2_manual_job_marks_completed
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h3_manual_job_failure_marks_failed
```
</details>

**Commit:** `FIX-43 (K-STATE-04): …`

### FIX-42 – Karte K-STATE-03

**Ursache:** `test_state_system.py::TestTbState::test_get_state_from_app` schreibt nach `/root/toolboxv2/features`.

**Beleg:** CI ubuntu (alle 4 Versionen): `PermissionError: [Errno 13] Permission denied: '/root/toolboxv2/features'`. Einzeln lokal (als root): `4 passed`.

**Warum:** Ein Pfad wird aus `$HOME`/Startverzeichnis eines früheren Tests gebildet (auf dem Ubuntu-Runner zeigt etwas auf `/root`), oder ein früherer Test hinterlässt `HOME`/`TB_*`-Variablen. Nur ubuntu betroffen, macOS und Windows nicht.

> **Hypothese ungetestet.** UNGETESTET (nur auf dem Ubuntu-Runner im Gesamtlauf; lokal als root nicht nachstellbar). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)

**Vorher lesen:** `toolboxv2/utils/system/state_system.py` (`get_state_from_app`), woher `…/toolboxv2/features` kommt

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_mini/test_system/test_state_system.py -q   # allein grün
```

Erwartet (Analyse): nur im CI-Gesamtlauf rot

**Schritte:**

1. Im Test vor dem Aufruf ausgeben, welche Variable auf `/root` zeigt (`HOME`, `TB_DATA_DIR`, `app.start_dir`).
2. Den Test, der sie setzt, mit `patch.dict`/`addCleanup` abschirmen (Muster wie FIX zu K-STATE-01).

**Nicht tun:** Nicht `/root` beschreibbar machen, kein skip.

**Abnahme:** CI ubuntu: ID grün. Danach Gate-Check (0.4).

**Test-IDs (1):** in der CI unter Windows rot und damit lokal rot erwartet: 0; nur auf Linux/macOS rot: 1.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_mini/test_system/test_state_system.py::TestTbState::test_get_state_from_app    # nur Linux/macOS
```
</details>

**Commit:** `FIX-42 (K-STATE-03): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Bericht (0.5) geschrieben.
3. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa05`.
