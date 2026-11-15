# ISAA-Auftrag 02 – Minimale Fixes, alle vorab getestet

> Komplexität: **2 von 5 – klein, viele** · Fix-Aufträge: FIX-07, FIX-08, FIX-09, FIX-10, FIX-11, FIX-12, FIX-13, FIX-14, FIX-15, FIX-16 · Test-IDs: 275
> Braucht keine Antwort des Owners. Unabhängig von den Aufträgen 01–05; läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Zehn voneinander unabhängige Fehlerursachen, zusammen 275 fehlschlagende Test-IDs. Jede wurde in der Analyse reproduziert, probehalber behoben und wieder zurückgesetzt; die Änderung steht hier wörtlich. Zwei davon sind echte Produktfehler (FIX-13, FIX-14). Arbeite die Aufträge in der angegebenen Reihenfolge ab, ein Commit je Auftrag.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu fünf ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/02 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa02 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa02
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
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa02 liegen
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
$GATE = @('toolboxv2/tests/test_clis', 'toolboxv2/tests/test_core/test_port_unification.py', 'toolboxv2/tests/test_core/test_tray_daemon_integration.py', 'toolboxv2/tests/test_mini/test_system', 'toolboxv2/tests/test_mini/test_toolboxv2.py', 'toolboxv2/tests/test_mini/test_extras/test_mkdocs.py', 'toolboxv2/tests/test_mini/test_cloudm/test_LiveSync', 'toolboxv2/tests/test_isaa/test_isaa_module.py', 'toolboxv2/tests/test_isaa/test_jobs', 'toolboxv2/tests/test_isaa/test_mem', 'toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py', 'toolboxv2/tests/test_isaa/test_base/test_embeddings.py', 'toolboxv2/tests/test_isaa/test_base/test_adapters.py', 'toolboxv2/tests/test_isaa/test_base/test_llm_router.py', 'toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py', 'toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer')
```

**Vor der ersten Änderung** einmal die Baseline schreiben, **nach jedem Fix** den Check:

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa02_baseline.json @GATE
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa02_baseline.json @GATE   # Exit 0 = ok, 'NEU rot: 0'
```

`NEU rot` muss 0 sein. Erscheint nach einem `git rebase` eine neue rote ID, prüfe, ob sie von dir kommt:

1. `git switch --detach origin/master`, dann den Check erneut.
2. Auch dort rot → Ursache ist ein fremder Commit: ID im Bericht unter „fremd“ notieren, dort (auf `origin/master`) die Baseline neu schreiben, `git switch isaa/02`, weiterarbeiten.
3. Dort grün → Ursache bist du: `git switch isaa/02`, nicht pushen, Fehler suchen oder Fix zurücknehmen.

### 0.4.1 Gesamt-Gate (nur dieser Auftrag)

```powershell
# Terminal 2, gleich zu Beginn (vor FIX-07):
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa02_full.json toolboxv2/tests
# nach dem letzten Fix, vor dem letzten Push:
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa02_full.json toolboxv2/tests
```

Läuft die Gesamt-Baseline noch, wenn du mit dem ersten Fix fertig bist: Fixes trotzdem einzeln mit dem Gate-Bereich prüfen und pushen; der Gesamt-Check kommt am Ende.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa02_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

`FIX-07` ändert `toolboxv2/tests/conftest.py` und wirkt damit auf **jeden** Test. Darum ist hier zusätzlich ein Gesamtlauf Pflicht (Abschnitt 0.4.1, „Gesamt-Gate“). Starte die Gesamt-Baseline gleich zu Beginn in einem zweiten Terminal; sie dauert unter Windows 40–50 Minuten.

## 1. Deine Dateien

- toolboxv2/tests/conftest.py
- toolboxv2/tests/test_mini/test_system/test_session.py
- toolboxv2/tests/test_clis/test_minicli_macros.py
- toolboxv2/tests/test_clis/test_config_wizard.py
- toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py
- toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py
- toolboxv2/utils/extras/mkdocs.py
- toolboxv2/mods/isaa/base/ai_semantic_memory.py
- toolboxv2/tests/test_core/test_port_unification.py
- toolboxv2/tests/test_core/test_tray_daemon_integration.py

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 01 (CI-Signal reparieren (Workflow-Dateien)): .github/workflows/build.yml (nur Job `py-check`); .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)
- Auftrag 03 (VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle): toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py
- Auftrag 04 (Sporadische Tests stabilisieren): toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py
- Auftrag 05 (Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben): installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py; toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942); toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py; toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py; toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py; toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden; toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)
- Außerdem gesperrt (offene Owner-Fragen): `pyproject.toml`, `toolboxv2/mods/isaa/pyproject.toml`, `toolboxv2/utils/system/session.py`, `toolboxv2/mcp_server/`, `toolboxv2/utils/brodcast/`, `docs/`.

## 2. Aufträge in dieser Reihenfolge

### FIX-07 – Karte K-ASYNC-01

**Ursache:** Synchrone Tests rufen `asyncio.get_event_loop().run_until_complete(…)` auf; ein früherer Test hat die Schleife geschlossen oder abgemeldet.

**Beleg:** `toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py:49`; dasselbe Muster in 16 Testdateien. Verschmutzer z. B. `toolboxv2/tests/test_isaa/test_isaa_module.py:54-65` (`new_event_loop`/`set_event_loop`/`close`) und `toolboxv2/tests/a_util.py:10-22`. CI: `RuntimeError: Event loop is closed` (92) und `RuntimeError: There is no current event loop in thread 'MainThread'.` (68)

**Warum:** Reihenfolge-Fehler, kein 3.12-Fehler: 159 der 160 IDs schlagen in allen 12 Jobs fehl, auch unter 3.10. Jede der 10 Dateien ist einzeln grün. Sobald ein Test davor die aktuelle Schleife schließt (`loop.close()`) oder mit `asyncio.run()` abmeldet, liefert `get_event_loop()` eine geschlossene oder keine Schleife.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/conftest.py

**Vorher lesen:** `toolboxv2/tests/conftest.py` ganz; `toolboxv2/tests/a_util.py:10-22`

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_isaa_module.py toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py -q
```

Erwartet (Analyse): 40× `RuntimeError: Event loop is closed` (dieselben 40 IDs wie in der CI). Allein: `test_job_manager.py` → `107 passed`

**Schritte:**

1. Nach der Fixture `reset_app_state_before_test` diese Fixture einfügen:

```python
@pytest.fixture(autouse=True)
def ensure_open_event_loop():
    """Sync tests that call asyncio.get_event_loop().run_until_complete(...) need
    an open current loop. Earlier tests may have closed or unset it."""
    import asyncio
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = asyncio.get_event_loop_policy().get_event_loop()
        stale = loop.is_closed()
    except RuntimeError:
        stale = True
    if stale:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield
```

**Nicht tun:** Keine Testdatei ändern. Kein `nest_asyncio`. Die Fixture schließt keine Schleife.

**Abnahme:** `python -m pytest toolboxv2/tests/test_isaa/test_isaa_module.py toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py -q` → `0 failed`. Alle 160 IDs der Karte grün im Gesamtlauf. Danach Gate-Check (0.4).

**Hinweis:** Die Fixture behebt das Symptom für alle 160 IDs. Die Ursache im Aufbau (wer besitzt die Schleife, drei verschiedene async-Hilfen in den Tests) steht in ARCH-01.

**Test-IDs (160):** in der CI unter Windows rot und damit lokal rot erwartet: 160; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_aggregate_never_raises
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_effort_ratio_gated_until_three_entries
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_fuzzy_only_classification_and_persist
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_guid_md_never_touched
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_guide_written_and_extended
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_happypath_only_on_improvement
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_index_stats_accumulate
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_llm_classification_used
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_llm_failure_falls_back_to_fuzzy_or_new
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_llm_slash_in_task_type_splits_correctly
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_unclear_goes_to_new_sink
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_dominant_fuzzy_skips_llm
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_injection_block_contains_guide_happypath_and_swap_notice
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_narrator_call_used_when_ambiguous
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_never_raises
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_new_type_injects_nothing
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_no_data_injects_nothing
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_uncertain_returns_new
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT2TextSerialize::test_flush_noop_when_empty
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT2TextSerialize::test_passes_through_when_idle
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT2TextSerialize::test_queues_while_speaking_then_flushes_once
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT8ChatViewTools::test_chat_executions_returns_running
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT8ChatViewTools::test_chat_history_returns_last_n
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT8ChatViewTools::test_provider_errors_are_caught
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT9StatusReportsOnce::test_each_finished_job_new_once
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH1_ChangedFilesAfterCommit::test_changed_files_empty_after_commit
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH1_ChangedFilesAfterCommit::test_changed_files_nonempty_before_commit
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH3_ApplyBackSubfolder::test_apply_back_creates_new_subfolder_file
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH3_ApplyBackSubfolder::test_apply_back_modifies_subfolder_file
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH4_ApplyBackFallback::test_explicit_no_commit_no_changes_returns_zero
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH4_ApplyBackFallback::test_uncommitted_changes_apply_via_fallback
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH5_CommitStagingFailure::test_ignored_file_NOT_propagated_by_apply_back
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH6_ApplyBackCopySubfolderListing::test_subfolder_apply_back_propagates_new_nested_file
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH7_AcceptFlowCleanupRace::test_apply_back_persists_before_cleanup
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestCustomTrigger::test_always_fire_evaluator
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestCustomTrigger::test_evaluator_exception_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestCustomTrigger::test_unknown_trigger_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_emits_completed_event
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_emits_failed_event
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_exception_sets_failed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_persists_state_to_file
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_removes_from_firing_set_on_failure
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_removes_from_firing_set_on_success
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_timeout_sets_fail
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_updates_run_count_and_result
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJobWithLiveState::test_live_state_finish_written_on_failure
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJobWithLiveState::test_live_state_finish_written_on_success
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireMissedJobs::test_fires_missed_jobs_and_returns_count
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireMissedJobs::test_fires_multiple_missed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireMissedJobs::test_returns_zero_when_nothing_missed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestJobChainIntegration::test_a_completes_b_fires
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestJobChainIntegration::test_job_b_does_not_fire_on_wrong_event
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestJobScheduler::test_paused_job_not_fired_during_tick
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnCliLifecycleEvaluator::test_mark_pending_and_evaluate
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnCliLifecycleEvaluator::test_multiple_jobs_independent
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnCliLifecycleEvaluator::test_teardown_clears_pending
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_does_not_fire_when_recent
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_exactly_at_boundary
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_fires_when_elapsed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_fires_when_never_run
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_no_interval_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_job_chain_fires
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_paused_job_not_notified
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_wrong_event_type_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_wrong_source_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_does_not_fire_future
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_fires_exactly_at_boundary
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_fires_when_past
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_invalid_datetime_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_naive_datetime_treated_as_utc
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_no_datetime_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestSchedulerLifecycle::test_double_start_is_idempotent
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestSchedulerLifecycle::test_fire_lifecycle_on_cli_start
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestSchedulerLifecycle::test_start_and_stop
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestFireLifecycleCliExit::test_cli_exit_job_fires
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestFireLifecycleCliExit::test_cli_start_does_not_fire_exit_jobs
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestLastFiredTsAfterFire::test_interval_does_not_refire_immediately_after_completion
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestLastFiredTsAfterFire::test_last_fired_ts_updated_after_failed_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestLastFiredTsAfterFire::test_last_fired_ts_updated_after_successful_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestMultiJobPersistenceRoundTrip::test_run_count_survives_reload
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_does_not_fire_when_active_recently
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_does_not_fire_without_activity_record
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_fires_when_agent_idle_long_enough
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_mark_activity_resets_idle_state
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_multiple_agents_tracked_independently
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_same_agent_two_jobs_throttled_independently
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_setup_and_teardown_do_not_raise
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_teardown_clears_per_job_check_entry
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_throttle_prevents_rapid_checks
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnCronEvaluatorEdgeCases::test_missing_cron_expression_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_all_four_event_types_work
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_all_jobs_dict_cleaned_on_teardown
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_consumed_after_evaluate
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_fires_after_notify_matching_event
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_multiple_jobs_same_event_all_notified
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_not_fires_without_notify
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_paused_job_not_notified
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_setup_registers_job
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_teardown_clears_all_pending
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_wrong_event_type_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnJobEventMultipleWatchers::test_chain_length_3
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnJobEventMultipleWatchers::test_multiple_watchers_on_same_source_all_notified
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnJobEventMultipleWatchers::test_one_watcher_per_source_does_not_affect_other_source
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_evaluate_before_setup_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_multiple_setups_accumulate_job_ids
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_setup_initializes_job_ids
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_teardown_before_setup_does_not_raise
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_consumed_after_evaluate
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_fires_after_trigger_webhook
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_multiple_webhooks_independent
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_not_fires_without_trigger
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_scheduler_trigger_webhook_delegates
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_teardown_clears_pending
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_wrong_job_id_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestSchedulerWebhookIntegration::test_webhook_job_fires_via_scheduler
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestSchedulerWebhookIntegration::test_webhook_not_fired_for_wrong_job_id
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestTriggerConfigEmptyList::test_interval_zero_round_trip_handled_safely
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_job_file_updated_after_fire_contains_correct_timestamps
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_job_fires_after_simulated_offline_period
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_multiple_restarts_accumulate_run_count
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_on_time_job_fires_after_offline_period
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_failed_job_increments_fail_count
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_fires_due_job
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_fires_multiple_jobs
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_returns_zero_when_nothing_missed
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_updates_run_count
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_compute_diff_with_conflicts
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_compute_local_diff_missing_remote
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_compute_local_diff_new_files
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_download_checksum_verification
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_download_pipeline
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_download_uses_supplied_url_without_asking
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_handle_file_deleted_moves_to_trash
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_upload_pipeline
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_upload_without_url_grant_marks_pending
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestExportIndex::test_export_gzipped
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestExportIndex::test_import_gzipped_roundtrip
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_delete_file
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_get_all_checksums
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_get_pending
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_init_creates_tables
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_set_sync_state
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_upsert_and_get
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_upsert_overwrites
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestSyncLog::test_log_for_share
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestSyncLog::test_log_limit
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestSyncLog::test_log_sync_event
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_claiming_new_when_we_hold_a_version_is_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_divergent_base_is_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_legacy_client_without_base_is_not_flagged
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_new_file_is_not_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_sequential_update_is_not_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestServerModes::test_index_does_not_scan_the_vault
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestServerModes::test_replica_client_connects_to_loopback
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestServerModes::test_replica_without_stored_token_falls_back_to_relay
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_broadcast_handles_dead_connection
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_broadcast_skips_originator
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_conflict_detection
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_handle_file_changed_updates_index
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_init_opens_index_without_scanning
```
</details>

**Commit:** `FIX-07 (K-ASYNC-01): …`

### FIX-08 – Karte K-STATE-01

**Ursache:** `test_session.py` setzt `TB_DATA_DIR=/fake/dir`; `Session()` legt dort wirklich Verzeichnisse an. Auf dem Runner ist das verboten, `asyncSetUp` bricht ab, der Patch bleibt aktiv und trifft die nächste Datei.

**Beleg:** `toolboxv2/tests/test_mini/test_system/test_session.py:26,133` (`patch.dict(os.environ, {"TB_DATA_DIR": "/fake/dir"})`), Stop erst in `asyncTearDown` (Zeile 157). Aufruf: `session.py:76 → :114 _init_storage → blobs.py:525 → os.makedirs`. CI: `PermissionError: [Errno 13] Permission denied: '/fake'` (ubuntu), `OSError: [Errno 30] Read-only file system: '/fake'` (macOS); 13× `ERROR` in `test_toolboxv2.py::…::setUpClass` (`toolbox.py:88`)

**Warum:** Zwei Fehler hintereinander. (1) Der Test ist nicht so abgeschirmt, wie sein Kommentar sagt: Geräteschlüssel und Sitzungsordner werden echt unter `/fake/dir` angelegt. (2) Schlägt `asyncSetUp` fehl, ruft unittest `asyncTearDown` nicht auf; `TB_DATA_DIR=/fake/dir` bleibt für den Rest des Laufs gesetzt. `test_toolboxv2.py` läuft direkt danach und scheitert in `setUpClass`.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

> Windows: Nur auf Linux/macOS-Runnern von selbst rot; s. Repro.

**Dateien (nur diese):** toolboxv2/tests/test_mini/test_system/test_session.py

**Vorher lesen:** Datei ganz, besonders `setUp`/`asyncSetUp`/`tearDown` der drei Klassen

**Repro (vor der Änderung, muss rot sein):**

```powershell
# Unter Windows ist /fake/dir anlegbar, der Fehler zeigt sich nicht von selbst.
# Zum Nachstellen in test_session.py VORÜBERGEHEND beide "/fake/dir" durch "Z:/fake/dir" ersetzen
# (ein Laufwerk Z: darf es nicht geben), dann:
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_toolboxv2.py -q
# erwartet: ERROR in test_toolboxv2.py (setUpClass) und FAILED mit Z:/fake. Danach: git checkout -- toolboxv2/tests/test_mini/test_system/test_session.py
```

Erwartet (Analyse): `PermissionError: [Errno 1] Operation not permitted: '/sys/fake'` – 7 FAILED in `test_session.py` (dieselben 7 Tests wie CI) + 13 ERROR in `test_toolboxv2.py`

**Schritte:**

1. Oben `import tempfile` ergänzen und nach den Imports `_TEST_DATA_DIR = tempfile.mkdtemp(prefix="tb_test_session_")`.
2. Beide Stellen `{"TB_DATA_DIR": "/fake/dir"}` → `{"TB_DATA_DIR": _TEST_DATA_DIR}`.
3. Direkt nach jedem `self.env_patcher.start()`: `self.addCleanup(self.env_patcher.stop)`; die `self.env_patcher.stop()`-Zeilen in `tearDown`/`asyncTearDown` entfernen (sonst doppelter Stop).

**Nicht tun:** Keine Assertions ändern. `test_toolboxv2.py` nicht ändern.

**Abnahme:** `python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_toolboxv2.py -q` → genau 7 rote IDs (K-API-13: 2, K-API-14: 5 inkl. Download), keine ERROR. Danach Gate-Check (0.4).

**Test-IDs (20):** in der CI unter Windows rot und damit lokal rot erwartet: 2; nur auf Linux/macOS rot: 18.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_invalid_token_triggers_successful_refresh    # nur Linux/macOS
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_valid_token_sets_session_valid
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_with_magic_link_sends_correct_payload    # nur Linux/macOS
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_without_token_returns_false    # nur Linux/macOS
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_verify_magic_link_saves_token_on_success    # nur Linux/macOS
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionNetwork::test_download_file_writes_chunks_to_disk
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionNetwork::test_fetch_get_adds_auth_headers_and_uses_session    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01a_crypt    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01b_file_handler    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01c_main_tool    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01d_styles    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02a_save_instance    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02b_load_mod    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02c_get_mod    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02d_mod_online    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02e_get_all_mods    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_03a_run    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_03b_all_functions    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_04a_remove_mod    # nur Linux/macOS
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_04b_remove_all_modules    # nur Linux/macOS
```
</details>

**Commit:** `FIX-08 (K-STATE-01): …`

### FIX-09 – Karte K-MISS-01

**Ursache:** `test_minicli_macros.py` patcht `toolboxv2.flows.minicli.BlobFile`; das Modul heißt `toolboxv2.flows.mini.minicli`.

**Beleg:** `toolboxv2/tests/test_clis/test_minicli_macros.py:30,96` u. a.; Import in derselben Datei schon richtig (`from toolboxv2.flows.mini.minicli import BeastCLI`, Zeile 20). CI: `AttributeError: module 'toolboxv2.flows' has no attribute 'minicli'` (26)

**Warum:** Das Modul wurde nach `flows/mini/` verschoben; die Imports wurden angepasst, die Patch-Ziele (Zeichenketten) nicht.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/test_clis/test_minicli_macros.py

**Vorher lesen:** alle `patch("toolboxv2.flows.minicli.` Stellen

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_clis/test_minicli_macros.py -q
```

Erwartet (Analyse): `26 failed, 2 passed`

**Schritte:**

1. Jede Zeichenkette `"toolboxv2.flows.minicli.` durch `"toolboxv2.flows.mini.minicli.` ersetzen (4 Stellen).

**Nicht tun:** Nichts sonst.

**Abnahme:** `python -m pytest toolboxv2/tests/test_clis/test_minicli_macros.py -q` → `28 passed`. Danach Gate-Check (0.4).

**Test-IDs (26):** in der CI unter Windows rot und damit lokal rot erwartet: 26; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_execute_macro_commands_break_stops_loop
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_execute_macro_commands_handles_comment
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_for_loop_iterates
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_if_break_sets_flag
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_if_false_skips_action
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_if_true_executes_action
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_while_respects_1000_limit
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_conflict_overwrite
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_invalid_format_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_missing_file_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_valid_json_creates_macros
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroArguments::test_args_set_as_arg1_and_dollar1
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroArguments::test_missing_macro_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_create_dispatches
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_export_dispatches
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_import_dispatches_with_path
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_unknown_action_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroResultCapture::test_return_value_saved_to_quick_vars
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroSaveLoad::test_load_empty_blob_leaves_macros_empty
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroSaveLoad::test_load_macros_populates_dict
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroSaveLoad::test_save_macros_writes_correct_structure
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_loop_var_highest_priority
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_macro_var_overrides_quick_var
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_multiple_vars_in_one_command
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_substitutes_quick_var
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_unknown_var_left_as_is
```
</details>

**Commit:** `FIX-09 (K-MISS-01): …`

### FIX-10 – Karte K-PY-01

**Ursache:** `test_config_wizard.py` lädt `config_wizard` per Dateipfad in `sys.modules`; unter Python 3.10 löst `mock.patch("toolboxv2.utils.clis.config_wizard.…")` den Namen über `getattr` am Elternpaket auf und findet ihn nicht.

**Beleg:** `toolboxv2/tests/test_clis/test_config_wizard.py:47-60`. CI (nur 3.10, alle OS): `AttributeError: module 'toolboxv2.utils.clis' has no attribute 'config_wizard'` (28)

**Warum:** Ab 3.11 nutzt `mock` `pkgutil.resolve_name`, das importiert und `sys.modules` beachtet; 3.10 macht `import toolboxv2.utils.clis` und dann `getattr`. Das per `spec_from_file_location` geladene Modul ist aber nie als Attribut am Paket gesetzt worden.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

> Windows: Nur unter Python 3.10 rot. Die venv `.venv310` legst du in Abschnitt 0 an.

**Dateien (nur diese):** toolboxv2/tests/test_clis/test_config_wizard.py

**Vorher lesen:** Zeilen 30–65 (Bootstrap)

**Repro (vor der Änderung, muss rot sein):**

```powershell
.\.venv310\Scripts\python -m pytest toolboxv2/tests/test_clis/test_config_wizard.py -q
```

Erwartet (Analyse): `32 failed, 36 passed` (lokal 3.10.20; CI: 28 der 32 – dieselbe Meldung)

**Schritte:**

1. Direkt nach `_spec.loader.exec_module(_wiz)` einfügen:

```python
import toolboxv2.utils.clis as _clis_pkg
_clis_pkg.config_wizard = _wiz  # py3.10: mock.patch resolves via getattr on the parent package
```

**Nicht tun:** Bootstrap nicht umbauen, keine Tests anfassen.

**Abnahme:** `.\.venv310\Scripts\python -m pytest toolboxv2/tests/test_clis/test_config_wizard.py -q` → 0 failed; unter 3.12 ebenfalls. Danach Gate-Check (0.4).

**Test-IDs (28):** in der CI unter Windows rot und damit lokal rot erwartet: 28; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_clis/test_config_wizard.py::TestEnvFileIO::test_set_vars_not_prompted_again
toolboxv2/tests/test_clis/test_config_wizard.py::TestEnvFileIO::test_wizard_env_variables_does_not_mutate_existing_dict
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_app_name_preserved_end_to_end
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_database_mode_lc_end_to_end
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_isaa_none_when_declined_end_to_end
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_produces_valid_tbmanifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_save_load_after_full_pipeline
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_app_name_stored_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_debug_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_log_level_is_loglevel_enum
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_model_validate_succeeds
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_CB_produces_valid_manifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_LC_mode_is_databasemode_enum
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_LC_produces_valid_manifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_LR_mode_is_databasemode_enum
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_RR_produces_valid_manifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_local_path_preserved_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_minio_use_ssl_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_redis_db_index_is_int_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_a2a_enabled_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_isaa_config_is_isaaconfig_instance
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_isaa_enabled_flag_true
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_isaa_none_when_declined
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_max_tokens_is_int_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_mcp_enabled_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_temperature_is_float_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWorkersSchemaCompatibility::test_wizard_workers_raises_typeerror_on_schema_dump
toolboxv2/tests/test_clis/test_config_wizard.py::TestWorkersSchemaCompatibility::test_wizard_workers_works_with_empty_workers_dict
```
</details>

**Commit:** `FIX-10 (K-PY-01): …`

### FIX-11 – Karte K-API-03

**Ursache:** `ExecutionEngine` liest `self.agent.obs`; die Testattrappe `MockAgent` hat das Attribut nicht.

**Beleg:** `toolboxv2/mods/isaa/base/Agent/execution_engine.py:3895` `if self.agent.obs: …` (auch `:3256, :3723, :3789`); `flow_agent.py:435` `self.obs = None`. `toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py:260-267`. CI: `AttributeError: 'MockAgent' object has no attribute 'obs'` (14)

**Warum:** Die Beobachtungsschicht kam dazu; echte Agenten haben immer `obs` (auch `None`), die Attrappe nicht.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py

**Vorher lesen:** `class MockAgent` (Zeile 260)

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py -q
```

Erwartet (Analyse): `23 failed, 43 passed`

**Schritte:**

1. In `MockAgent.__init__` nach `self.skills_manager = None`: `self.obs = None  # FlowAgent.obs is optional (flow_agent.py:435)`.

**Nicht tun:** Produktcode nicht ändern.

**Abnahme:** 12 IDs grün; `test_think_decrements_iteration` und `test_think_records_in_auto_focus` bleiben rot → gehören dann zu K-API-04. Danach Gate-Check (0.4).

**Test-IDs (14):** in der CI unter Windows rot und damit lokal rot erwartet: 14; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_final_answer_is_final
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_final_answer_not_in_history
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_list_tools_decrements_iteration
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_list_tools_returns_string
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_minimax_style_id_preserved_exactly
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_shift_focus_decrements_5
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_decrements_iteration
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_en
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_live
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_not_final
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_records_in_auto_focus
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_tool_call_sequence_no_orphans_in_history
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_tool_result_has_correct_structure
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_unknown_tool_no_exception
```
</details>

**Commit:** `FIX-11 (K-API-03): …`

### FIX-12 – Karte K-MISS-02

**Ursache:** Tests patchen bzw. importieren `llm_router.…` als Top-Level-Modul; es heißt `toolboxv2.mods.isaa.base.llm_router.…`.

**Beleg:** `toolboxv2/tests/test_isaa/test_base/test_embeddings.py:36,48,60,73,86,100` (`@patch("llm_router.embeddings._get_router")`), `test_llm_router.py:177`, `test_adapters.py:422,425,438,441` (`from llm_router.router import …`). CI: `ModuleNotFoundError: No module named 'llm_router'` (9)

**Warum:** Die Tests entstanden, als `llm_router` ein eigenständiges Paket war. Die übrigen Imports derselben Dateien nutzen schon den vollen Pfad.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py

**Vorher lesen:** alle Vorkommen von `llm_router.` ohne `toolboxv2.`-Präfix

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_base/test_embeddings.py toolboxv2/tests/test_isaa/test_base/test_adapters.py toolboxv2/tests/test_isaa/test_base/test_llm_router.py -q
```

Erwartet (Analyse): `12 failed` (9× `No module named 'llm_router'`, 3× K-API-06)

**Schritte:**

1. `patch("llm_router.` → `patch("toolboxv2.mods.isaa.base.llm_router.` (7 Stellen).
2. `from llm_router.` → `from toolboxv2.mods.isaa.base.llm_router.` (4 Stellen in `test_adapters.py`).

**Nicht tun:** Keine Assertions ändern.

**Abnahme:** 8 IDs grün; `test_registers_only_available_providers` bleibt rot mit `ProviderError` → melden, nicht „reparieren“. Danach Gate-Check (0.4).

**Hinweis:** Zweite Ursache für `test_registers_only_available_providers`: der Test erwartet einen registrierten groq-Adapter. Nach diesem Fix als eigene Karte unter K-API aufnehmen.

**Test-IDs (9):** in der CI unter Windows rot und damit lokal rot erwartet: 9; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_base/test_adapters.py::TestSetupDefaultAdapters::test_ollama_always_registered
toolboxv2/tests/test_isaa/test_base/test_adapters.py::TestSetupDefaultAdapters::test_registers_only_available_providers
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_basic_embed
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_dimension_truncation
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_dimensions_passed_to_router
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_input_type_passed
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_litellm_embed_compat
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_smart_embed_delegates
toolboxv2/tests/test_isaa/test_base/test_llm_router.py::TestRouter::test_resolve_unknown_fallback
```
</details>

**Commit:** `FIX-12 (K-MISS-02): …`

### FIX-13 – Karte K-BUG-01

**Ursache:** `FileScanner._should_include` prüft die Ausschlussliste (`tmp`, `temp`, `build`, `dist`, `target`, …) gegen den **absoluten** Pfad. Liegt das Projekt selbst unter so einem Ordner, wird jede Datei ausgeschlossen.

**Beleg:** `toolboxv2/utils/extras/mkdocs.py:2015-2018` (`parts = path.parts`), Liste `:1940-1964`. CI nur ubuntu (Temp = `/tmp/…`): `AssertionError: 'main.py' not found in []`, `0 not greater than 0`, `'rebuilt' != 'loaded'` (11)

**Warum:** Echter Produktfehler. Unter Linux liegt `tempfile.mkdtemp()` in `/tmp`, also enthält jeder Pfad den Teil `tmp`. macOS (`/var/folders/…`) und Windows (`…\Temp\…`) treffen die Liste nicht (Groß-/Kleinschreibung). Ein Nutzerprojekt unter `~/build/x` oder `/tmp/x` hätte einen leeren Doku-Index.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

> Windows: Nur rot, wenn der Temp-Pfad einen ausgeschlossenen Ordnernamen enthält; s. Repro.

**Dateien (nur diese):** toolboxv2/utils/extras/mkdocs.py

**Vorher lesen:** `FileScanner` (`scan`, `_get_search_roots`, `_should_include`), `iter_files` (Zeile 1916)

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_mini/test_extras/test_mkdocs.py -q
# Unter Windows meist schon grün: der Fehler entsteht nur, wenn der Temp-Pfad einen Ordner
# aus der Ausschlussliste enthält (Linux: /tmp). Nachstellen unter Windows:
$env:TMP="$env:USERPROFILE\tb_tmp\build"; $env:TEMP=$env:TMP; mkdir $env:TMP -Force | Out-Null
python -m pytest toolboxv2/tests/test_mini/test_extras/test_mkdocs.py -q    # jetzt rot
# danach in derselben Shell TMP/TEMP zurücksetzen (neue Shell öffnen genügt)
```

Erwartet (Analyse): `11 failed, 120 passed`

**Schritte:**

1. `_should_include` ersetzen durch:

```python
    def _should_include(self, path: Path) -> bool:
        """Check if file should be included (exclude only check)."""
        # Only directories *inside* the project decide; a project that lives
        # under e.g. /tmp or .../build must not be excluded as a whole.
        parts = path.parts
        for base in (self.root, self.docs_root):
            try:
                parts = path.resolve().relative_to(Path(base).resolve()).parts
                break
            except (ValueError, OSError):
                continue
        return not any(exc in parts for exc in self.exclude_dirs)
```

**Nicht tun:** Ausschlussliste nicht kürzen. `iter_files` nicht ändern.

**Abnahme:** `python -m pytest toolboxv2/tests/test_mini/test_extras/test_mkdocs.py -q` → `131 passed` (Linux). Danach Gate-Check (0.4).

**Test-IDs (11):** in der CI unter Windows rot und damit lokal rot erwartet: 0; nur auf Linux/macOS rot: 11.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_initialize_builds_index    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_initialize_loads_existing    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_lookup_code_python    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_read_all_sections    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_read_by_query    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_read_markdown_format    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemLiveWorkflow::test_cross_reference_docs_and_code    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemLiveWorkflow::test_full_documentation_workflow    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemLiveWorkflow::test_multi_language_support    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestFileScanner::test_scan_markdown_files    # nur Linux/macOS
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestFileScanner::test_scan_python_files    # nur Linux/macOS
```
</details>

**Commit:** `FIX-13 (K-BUG-01): …`

### FIX-14 – Karte K-BUG-02

**Ursache:** In `add_data` wurde die Längenbegrenzung mitten in eine `if/elif`-Kette gesetzt. Jeder String bis 6000 Zeichen fällt in `else: raise ValueError("Unsupported data type")`.

**Beleg:** `toolboxv2/mods/isaa/base/ai_semantic_memory.py:258-268`: `elif isinstance(data, str): texts = […]` / `if len(data) > 6000: data = data[:6000]` / `elif isinstance(data, list): …` / `else: raise ValueError("Unsupported data type")`. Eingeführt mit `2d51dd4` (2026-09-05). CI: `ValueError: Unsupported data type` (5)

**Warum:** Echter Produktfehler: Texte lassen sich nicht mehr ins semantische Gedächtnis schreiben. Das neue `if` beginnt eine eigene Kette; ein kurzer String ist weder > 6000 noch Liste noch dict. Bei langen Strings wird `data` gekürzt, aber `texts` enthält schon den ungekürzten Text.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/mods/isaa/base/ai_semantic_memory.py

**Vorher lesen:** `add_data`, Zeilen 240–275

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py -q
```

Erwartet (Analyse): `5 failed, 71 passed, 1 skipped`

**Schritte:**

1. Den Block so umstellen:

```python
        elif isinstance(data, str):
            # fix4: cap single-string embeds - O(seq^2) attention blows up on long texts
            if len(data) > 6000:  # ponytail: add real chunking when recall quality drops
                data = data[:6000]
            texts = [data.replace("\\t", "").replace("\t", "")]
        elif isinstance(data, list):
```

**Nicht tun:** Grenze 6000 nicht ändern. Die anderen Zweige nicht anfassen.

**Abnahme:** `python -m pytest toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py -q` → 0 failed. Danach Gate-Check (0.4).

**Test-IDs (5):** in der CI unter Windows rot und damit lokal rot erwartet: 5; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_add_data_v1_signature
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_delete_memory
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_query_multi_space
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_query_to_str
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_query_v1_signature
```
</details>

**Commit:** `FIX-14 (K-BUG-02): …`

### FIX-15 – Karte K-MISS-03

**Ursache:** `test_port_unification.py` baut den Pfad zu `utils/workers/fast_tb.py` vom Repo-Wurzelverzeichnis statt vom Paketordner `toolboxv2/`.

**Beleg:** `toolboxv2/tests/test_core/test_port_unification.py:45` `base = Path(__file__).resolve().parents[3]`. CI: `FileNotFoundError: … '/home/runner/work/ToolBoxV2/ToolBoxV2/utils/workers/fast_tb.py'`

**Warum:** Ein Verzeichnis zu weit oben (`parents[3]` = Repo, `parents[2]` = `toolboxv2`).

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/test_core/test_port_unification.py

**Vorher lesen:** `test_embedded_js_fallback_matches_canon`

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_core/test_port_unification.py -q
```

Erwartet (Analyse): `1 failed, 6 passed`

**Schritte:**

1. Zeile 45: `parents[3]` → `parents[2]`.

**Nicht tun:** Die Assertions (`'8100'` nicht enthalten, `'8468'` enthalten) bleiben.

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_port_unification.py -q` → `7 passed`. Danach Gate-Check (0.4).

**Test-IDs (1):** in der CI unter Windows rot und damit lokal rot erwartet: 1; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_core/test_port_unification.py::test_embedded_js_fallback_matches_canon
```
</details>

**Commit:** `FIX-15 (K-MISS-03): …`

### FIX-16 – Karte K-API-17

**Ursache:** Test sucht im Quelltext nach `_fetch_tray_state`; die Funktion heißt `fetch_tray_state`.

**Beleg:** `toolboxv2/utils/extras/fallback_tray.py:76` `def fetch_tray_state() -> dict | None:`; `toolboxv2/tests/test_core/test_tray_daemon_integration.py:461`. CI: `'_fetch_tray_state' not found in …`

**Warum:** Umbenennung (öffentlich gemacht).

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/tests/test_core/test_tray_daemon_integration.py

**Vorher lesen:** `test_fallback_tray_fetches_state`

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_core/test_tray_daemon_integration.py -q
```

Erwartet (Analyse): `1 failed, 41 passed`

**Schritte:**

1. Zeile 461: `"_fetch_tray_state"` → `"fetch_tray_state"`.

**Nicht tun:** Die zweite Prüfung (`"/tray/state"`) bleibt.

**Abnahme:** `42 passed`. Danach Gate-Check (0.4).

**Test-IDs (1):** in der CI unter Windows rot und damit lokal rot erwartet: 1; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_core/test_tray_daemon_integration.py::TestDaemonServiceE2EValidation::test_fallback_tray_fetches_state
```
</details>

**Commit:** `FIX-16 (K-API-17): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Gesamt-Gate (0.4.1): `NEU rot: 0`.
3. Bericht (0.5) geschrieben.
4. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa02`.
