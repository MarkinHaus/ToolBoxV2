# ISAA-Auftrag 04 – Sporadische Tests stabilisieren

> Komplexität: **4 von 5 – Analyse je Test** · Fix-Aufträge: FIX-44 · Test-IDs: 7
> Braucht keine Antwort des Owners. Unabhängig von den Aufträgen 01–05; läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Sieben Tests schlagen mal fehl und mal nicht (zwischen zwei CI-Läufen desselben Codes und zwischen den Jobs). Gemeinsame Ursache: feste Wartezeiten, Uhrzeit, nicht festgelegte Zufallswerte. Du stabilisierst jeden Test einzeln, **nur in den Testdateien**. Erfolg heißt: 20 von 20 Läufen grün.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu fünf ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/04 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa04 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa04
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e .
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa04 liegen
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

Gate je Test: Baseline und Check nur über die Testdatei, die du gerade änderst:

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa04_baseline.json <Testdatei>
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa04_baseline.json <Testdatei>
```

`NEU rot` muss 0 sein. Erscheint nach einem `git rebase` eine neue rote ID, prüfe, ob sie von dir kommt:

1. `git switch --detach origin/master`, dann den Check erneut.
2. Auch dort rot → Ursache ist ein fremder Commit: ID im Bericht unter „fremd“ notieren, dort (auf `origin/master`) die Baseline neu schreiben, `git switch isaa/04`, weiterarbeiten.
3. Dort grün → Ursache bist du: `git switch isaa/04`, nicht pushen, Fehler suchen oder Fix zurücknehmen.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa04_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

- Du änderst **nur Testdateien**. Braucht ein Test eine Änderung im Produktcode (z. B. im VFS-Watcher), dann nicht ändern, sondern im Bericht beschreiben – an diesem Code arbeitet Instanz 05.
- Ein Test, der unter Windows in 20 Läufen nie fehlschlägt, ist damit nicht behoben. Dann suchst du die Stelle, die auf Zeit oder Zufall beruht, durch Lesen, machst sie deterministisch und prüfst erneut 20×.
- Gate je Test: die eigene Testdatei (Baseline vor der Änderung, Check danach).

## 1. Deine Dateien

- toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 01 (CI-Signal reparieren (Workflow-Dateien)): .github/workflows/build.yml (nur Job `py-check`); .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)
- Auftrag 02 (Minimale Fixes, alle vorab getestet): toolboxv2/tests/conftest.py; toolboxv2/tests/test_mini/test_system/test_session.py; toolboxv2/tests/test_clis/test_minicli_macros.py; toolboxv2/tests/test_clis/test_config_wizard.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py; toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py; toolboxv2/utils/extras/mkdocs.py; toolboxv2/mods/isaa/base/ai_semantic_memory.py; toolboxv2/tests/test_core/test_port_unification.py; toolboxv2/tests/test_core/test_tray_daemon_integration.py
- Auftrag 03 (VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle): toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py
- Auftrag 05 (Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben): installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py; toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942); toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py; toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py; toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py; toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden; toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)
- Außerdem gesperrt (offene Owner-Fragen): `pyproject.toml`, `toolboxv2/mods/isaa/pyproject.toml`, `toolboxv2/utils/system/session.py`, `toolboxv2/mcp_server/`, `toolboxv2/utils/brodcast/`, `docs/`.

## 2. Aufträge in dieser Reihenfolge

### FIX-44 – Karte K-FLAKY-01

**Ursache:** Zeit- oder zufallsabhängige Tests: Datei-Watcher mit festen Wartezeiten, Profiler-Zählwerte, Backup-Aufräumen nach Zeitstempel, Vektorsuche, Audio-Reihenfolge.

**Beleg:** `test_vfs_pooling.py`: `Extern erstelltes File muss nach 4.0s sichtbar sein`, `Extern gelöschtes File muss nach 4.0s entfernt sein`, `Agent B muss File sehen das A erstellt hat` – je Job andere Auswahl; `test_profiler.py`: `22 != 30 : compute should have 30 calls` (nur ein Job); `test_rollback_manager.py`: `assert 4 == 7` (zwei Jobs, lokal `5 == 7`); `test_hybrid_memory_realworld.py`: `'SpaceX' not found in ['Tesla, Inc.', …]` (Ergebnisliste je Job anders); `test_audio_core.py::test_multiple_texts_ordered` (ein Job)

**Warum:** Vergleich der beiden Läufe (`91d984a` und `ebed4d6`, gleicher Baum) und der Jobs untereinander: diese IDs wechseln. Gemeinsame Art: feste `sleep`-Zeiten, Zeitstempel mit Sekundenauflösung, zufällige oder nicht festgelegte Einbettungen.

> **Hypothese ungetestet.** UNGETESTET (sporadisch; kein gemeinsamer Probe-Fix). Erst den Repro ausführen und den Befund bestätigen.

**Dateien (nur diese):** toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py

**Vorher lesen:** je Test die Wartelogik und die Stelle, an der Zeit oder Zufall einfließt

**Repro (vor der Änderung, muss rot sein):**

```powershell
1..20 | % { python -m pytest <Testdatei>::<Klasse>::<Test> -q -p no:cacheprovider | Select-Object -Last 1 }
```

Erwartet (Analyse): wechselnd

**Schritte:**

1. Feste `sleep(x)` + einmalige Prüfung durch „warte bis Bedingung, höchstens T Sekunden“ ersetzen.
2. `test_rollback_manager`: Zeitstempel der Test-Backups ausdrücklich setzen statt auf die Uhr zu vertrauen.
3. `test_hybrid_memory_realworld`: Einbettungen im Test festlegen (deterministischer Mock) oder die Erwartung an die Rangfolge begründen.
4. Jede ID 20× hintereinander laufen lassen; erst dann als behoben melden.

**Nicht tun:** Kein `flaky`-Plugin, kein Wiederholen bis grün, keine längeren festen Wartezeiten als Lösung.

**Abnahme:** jede ID 20 von 20 Läufen grün. Danach Gate-Check (0.4).

**Test-IDs (7):** in der CI unter Windows rot und damit lokal rot erwartet: 1; nur auf Linux/macOS rot: 6.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_a_creates_file_b_sees_it    # nur Linux/macOS
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_b_creates_file_a_sees_it    # nur Linux/macOS
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_b_writes_to_a_file_a_reads_new_content    # nur Linux/macOS
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestPollIntegration::test_external_delete_removes_from_vfs    # nur Linux/macOS
toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py::TestAudioStreamPlayer::test_multiple_texts_ordered    # nur Linux/macOS
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py::TestHybridMemoryRealWorld::test_02_vector_search
toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py::test_cleanup_old_backups    # nur Linux/macOS
```
</details>

**Commit:** `FIX-44 (K-FLAKY-01): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Bericht (0.5) geschrieben.
3. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa04`.
