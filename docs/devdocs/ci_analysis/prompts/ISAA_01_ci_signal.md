# ISAA-Auftrag 01 – CI-Signal reparieren (Workflow-Dateien)

> Komplexität: **1 von 5 – sehr klein** · Fix-Aufträge: FIX-01, FIX-02, FIX-06 · Test-IDs: 1
> Braucht keine Antwort des Owners. Unabhängig von den Aufträgen 01–05; läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Die CI zeigt heute grün, obwohl in jedem Python-Job 500+ Tests fehlschlagen, und rot aus dem falschen Grund. Du änderst drei Stellen in zwei Workflow-Dateien. Danach sind alle 12 `py-check`-Jobs **rot – das ist das gewünschte, ehrliche Ergebnis**; die anderen Instanzen machen sie grün.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu fünf ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/01 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa01 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa01
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e .
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa01 liegen
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
# YAML-Prüfung erneut, wenn der Rebase fremde Commits gebracht hat
git push origin HEAD:master
```

Wird der Push abgelehnt (`non-fast-forward`): die drei Schritte wiederholen, höchstens dreimal. Danach melden.

### 0.4 Gate: keine neue rote Test-ID

Entfällt (keine Python-Datei betroffen), s. „Besonderheit“.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa01_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

- Du änderst keinen Python-Code und keine Tests; ein Gate-Lauf ist nicht nötig. Prüfe stattdessen nach jeder Änderung, dass die YAML-Datei gültig ist:

```powershell
python -c "import sys,yaml; [yaml.safe_load(open(f,encoding='utf-8')) for f in sys.argv[1:]]; print('YAML ok')" .github/workflows/build.yml .github/workflows/release.yml
```

- Die Abnahme findet in der CI statt. Wenn `gh` installiert und angemeldet ist: `gh run list --workflow build.yml --branch master -L 3` und `gh run view <id> --log-failed`. Sonst trägst du im Bericht ein, was der Owner im nächsten Lauf prüfen soll.
- `FIX-06` (Nuitka in `release.yml`) wird erst beim nächsten Lauf von `release.yml` geprüft.

## 1. Deine Dateien

- .github/workflows/build.yml (nur Job `py-check`)
- .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 02 (Minimale Fixes, alle vorab getestet): toolboxv2/tests/conftest.py; toolboxv2/tests/test_mini/test_system/test_session.py; toolboxv2/tests/test_clis/test_minicli_macros.py; toolboxv2/tests/test_clis/test_config_wizard.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py; toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py; toolboxv2/utils/extras/mkdocs.py; toolboxv2/mods/isaa/base/ai_semantic_memory.py; toolboxv2/tests/test_core/test_port_unification.py; toolboxv2/tests/test_core/test_tray_daemon_integration.py
- Auftrag 03 (VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle): toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py
- Auftrag 04 (Sporadische Tests stabilisieren): toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py
- Auftrag 05 (Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben): installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py; toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942); toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py; toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py; toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py; toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden; toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)
- Außerdem gesperrt (offene Owner-Fragen): `pyproject.toml`, `toolboxv2/mods/isaa/pyproject.toml`, `toolboxv2/utils/system/session.py`, `toolboxv2/mcp_server/`, `toolboxv2/utils/brodcast/`, `docs/`.

## 2. Aufträge in dieser Reihenfolge

### FIX-01 – Karte K-ENV-01

**Ursache:** Der Fallback `|| python -m unittest discover` entscheidet das Job-Ergebnis; er findet 0 Tests. Bis 3.11 ist das Exit-Code 0 (grün), ab 3.12 Exit-Code 5 (rot).

**Beleg:** `.github/workflows/build.yml:154`. Log ubuntu/3.10 (Lauf 43742117802): `= 556 failed, 4052 passed, 238 skipped, 38 warnings, 16 errors … =`, danach `Ran 0 tests in 0.000s` / `OK` → Job grün. Log ubuntu/3.11: `= 529 failed, 4103 passed … =`, danach `Ran 0 tests in 0.000s` / `OK` → Job grün. Log ubuntu/3.12: `= 529 failed … =`, `Ran 0 tests in 0.000s`, `NO TESTS RAN`, `Process completed with exit code 5.`

**Warum:** pytest schlägt in allen 12 Jobs mit 509–556 Fehlern fehl. `unittest discover -s toolboxv2/tests/` findet nichts, weil die Unterordner keine `__init__.py` haben und die einzige Datei auf oberster Ebene (`test_gitignore.py`) pytest-Funktionen enthält. Python 3.12 hat für „0 Tests gelaufen“ den Exit-Code 5 eingeführt. Die roten 3.12/3.13-Jobs sind also kein 3.12-Problem, und die grünen 3.10/3.11-Jobs sind nicht geprüft.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** .github/workflows/build.yml

**Vorher lesen:** Job `py-check`, Schritt `Run Tests` (Zeilen 148–154)

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m unittest discover -s toolboxv2/tests/; "exit=$LASTEXITCODE"
# mit der 3.12-venv: "Ran 0 tests", "NO TESTS RAN", exit=5. Unter 3.10/3.11 wäre es "OK", exit=0.
```

Erwartet (Analyse): 3.10: `Ran 0 tests in 0.000s` `OK` exit=0
3.12: `Ran 0 tests in 0.000s` `NO TESTS RAN` exit=5

**Schritte:**

1. Zeile 154 ersetzen durch `run: python -m pytest toolboxv2/tests/ -v --tb=short -rfE`.
2. Nichts anderes im Job ändern (FIX-02 ändert die Installation getrennt).

**Nicht tun:** Kein `continue-on-error`, kein `|| true`, keine Matrix-Einträge entfernen.

**Abnahme:** CI-Lauf auf `master`: alle 12 `py-check`-Jobs enden mit dem Exit-Code von pytest. Direkt nach diesem Fix sind alle 12 rot (509–556 Fehler je Job) – das ist richtig. Im Log steht kein `unittest discover` mehr.

**Commit:** `FIX-01 (K-ENV-01): …`

### FIX-02 – Karte K-ENV-04

**Ursache:** `toolboxv2` ist in der CI nicht editierbar installiert; ein Unterprozess, der nicht im Checkout startet, importiert die Kopie aus `site-packages`, in der `mods/isaa` fehlt.

**Beleg:** `build.yml:147` `uv pip install . --system`; `pyproject.toml:89` schließt `toolboxv2.mods.[!CDW]*` aus dem Paket aus. Test: `test_shell_tool.py:405-422` schreibt `harness.py` in ein Temp-Verzeichnis und startet es. CI: `AssertionError: 1 != 0 : launcher chain was killed`

**Warum:** O5 ist kein Plattform- oder Prozessketten-Problem. `harness.py` liegt in `/tmp/…`, also ist `sys.path[0]` das Temp-Verzeichnis und nicht der Checkout. `from toolboxv2.mods.isaa.module import …` trifft `site-packages/toolboxv2` → `ModuleNotFoundError: No module named 'toolboxv2.mods.isaa'` → Exit-Code 1. Lokal beim Owner grün, weil dort editierbar installiert ist. Unter Windows fällt der Test in der CI nicht auf, weil er dort übersprungen bzw. anders verzweigt wird (nicht in den Windows-Rohlisten).

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** .github/workflows/build.yml

**Vorher lesen:** Job `py-check`, Schritt `Install toolboxv2 with ISAA` (Zeilen 143–147)

**Repro (vor der Änderung, muss rot sein):**

```powershell
# Unter Windows nicht rot (der Test verzweigt dort anders). Der Beleg stammt aus dem CI-Log ubuntu/3.12.
# Nachvollziehen, warum: aus einem fremden Ordner importiert eine NICHT editierbare Installation die Kopie in site-packages:
Push-Location $env:TEMP; python -c "import toolboxv2; print(toolboxv2.__file__)"; Pop-Location
```

Erwartet (Analyse): `1 failed` – `AssertionError: 1 != 0 : launcher chain was killed`; direkter Aufruf des Harness: `ModuleNotFoundError: No module named 'toolboxv2.mods.isaa'`

**Schritte:**

1. Zeile 147: `python -m uv pip install -e . --system` (statt `.`).
2. Zeile 146 unverändert lassen.

**Nicht tun:** Nicht `PYTHONPATH` im Workflow setzen. Test nicht ändern.

**Abnahme:** CI ubuntu/3.12: `test_shell_tool.py::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness` grün.

**Hinweis:** Zweite Ursache derselben Art → ARCH-02: das editierbare `isaa`-Paket legt die Top-Level-Namen `base`, `extras`, `CodingAgent` in die Umgebung, nicht `toolboxv2.mods.isaa`.

**Test-IDs (1):** in der CI unter Windows rot und damit lokal rot erwartet: 0; nur auf Linux/macOS rot: 1.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_shell_tool.py::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness    # nur Linux/macOS
```
</details>

**Commit:** `FIX-02 (K-ENV-04): …`

### FIX-06 – Karte K-ENV-06

**Ursache:** `release.yml`: die Jobs `build-windows-x64` und `build-macos-arm64` rufen `python -m nuitka` auf, installieren Nuitka aber nicht (O4).

**Beleg:** `.github/workflows/release.yml:100-110` (Windows) und `:126-137` (macOS): nach `setup-python` folgt direkt `python -m nuitka …`. Der Linux-Job hat `pip install nuitka zstandard` (Zeile 55) und einen Schritt `Install project`.

**Warum:** Kopierfehler beim Anlegen der beiden Jobs.

> UNGETESTET (Workflow-Datei; prüfbar erst im nächsten Lauf von `release.yml`). Die Ursache ist durch das CI-Log belegt; ein lokaler Repro ist nicht möglich.

**Dateien (nur diese):** .github/workflows/release.yml

**Vorher lesen:** alle drei Build-Jobs; der Linux-Job ist die Vorlage

**Repro (vor der Änderung, muss rot sein):**

```powershell
# nur in der CI. Beleg: Lauf 43742117826, Job build-windows-x64, Schritt "Build onefile": No module named nuitka
```

Erwartet (Analyse): `No module named nuitka`

**Schritte:**

1. In `.github/workflows/release.yml`, Job `build-windows-x64`: direkt nach dem Schritt `- uses: actions/setup-python@v5` (mit seinem `with:`-Block) einfügen, gleiche Einrückung wie die anderen Schritte (6 Leerzeichen vor `- name`):

```yaml
      - name: Install build deps
        run: |
          python -m pip install --upgrade pip
          pip install nuitka zstandard

      - name: Install project
        run: |
          pip install .
```
2. Denselben Block im Job `build-macos-arm64` an derselben Stelle einfügen.
3. YAML-Prüfung aus „Besonderheit“ ausführen.

**Nicht tun:** Linux-Job nicht ändern. Release-Schritt (`Update nightly prerelease`) nicht ändern, das ist O3.

**Abnahme:** `release.yml` manuell auslösen → alle drei Build-Jobs grün, Smoke-Test läuft.

**Commit:** `FIX-06 (K-ENV-06): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Bericht (0.5) geschrieben.
3. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa01`.
