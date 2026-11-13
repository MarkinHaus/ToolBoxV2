# Agent-Bericht R9-Messung (2026-11-12)

Windows-Datenpersistenz-Nachweis: 3 Mess-Laeufe (A/B/C), verify_install_windows.ps1
(V1-V12, Baseline U1-U5), TDD. Basis: Branch claude/wizardly-cannon-o944j4 nach R1-R8.

## Ergebnis: ALLE 3 LAEFE GRUEN

| Lauf | Setup | Ergebnis |
|---|---|---|
| A v7 | Custom-Pfad m.L. ("my install dir"), Installer -Config, workers start, start_app isaa+obs, verify -SaveBaseline, stop/start, -CompareBaseline | 12/12 PASS, U1/U2/U4/U5 PASS, 0 Tracebacks, start_app isaa source:app:isaa merged:true |
| B | Update in-place (-Update -AssumeYes yes, TB_CORE_BUILD=OTHERBUILD), vorher -init developer + Marker | 17 PASS 0 FAIL, manifest-Diff nur installed_at+core_build, Marker ueberlebt, developer-Profil erhalten |
| C v2 | Default-Pfad (ohne install_path, B3-Fallback -> LOCALAPPDATA\toolboxv2, redirected) | 12/12 PASS, 0 Tracebacks, start_app isaa merged:true bei 0s (B4b), base-c.json gesichert |

3 Schutz-Belege (jeder Lauf): USER-PATH-Registry unveraendert, echtes LOCALAPPDATA unberuehrt
(Snapshot before/after identisch), verbotene TB_*/TOOLBOX* env-Vars in Consumer geloescht + Guard.

## M1: scripts/verify_install_windows.ps1 (commit 19019ff7)
Read-only Pruefskript, powershell.exe 5.1, ASCII-only, mutiert nichts (Guard-Test gegen
mutierende Cmdlets). 12 Checks + Baseline: V1 manifest, V2 exactly-one-.env, V3 JWT-Secret
^[0-9a-f]{128}$, V4 kein TB_DATA_DIR/REQUIRED, V5 .data-Layout, V6 site-packages clean,
V7 tb-manifest profile, V8 /health, V9 /api/wizard/state first_run=false+profile-match,
V10 UI /(>10KB)+/setup, V11 local_ui.json install_dir, V12 kein litellm.
Baseline: secret-hash (nicht Wert!), profile, core_build, features, Datei-Liste.
U-Checks: U1 secret-identisch, U2 profile, U3 features gleich+core_build NEU, U4 Dateien
vorhanden+nicht geschrumpft (wal/shm/logs ausgenommen), U5 .env/.data-Kardinalitaet.
Tests: toolboxv2/tests/test_core/test_verify_install_windows.py (5 Faelle, Temp-Ordner).

## Gefundene + gefixte Bugs (TDD je commit)

| Bug | Commit | Fix |
|---|---|---|
| B1 kritisch: NginxManager._ensure_symlink os.symlink OSError auf Windows ohne DevMode -> start_all crasht NACH worker-spawn, http_worker wird zur Waise (ports belegt, workers stop wirkungslos) | 0fd1e0bc | Copy-Fallback + logger.warning |
| B2: PID-Dateien in site-packages\toolboxv2\.info\pids (V6-Verletzung) | 0fd1e0bc | _state_dir.pids_dir() als eine Quelle (__main__.py x3, getting_and_closing_app.py) |
| B2-Folgefehler: mein Patch erzeugte " (backslash-Quote) SyntaxError in __main__.py | 5d2e4d04 | " escaped, compile()-Guard im Test |
| B3: installer.ps1 ohne Default-Pfad bei fehlendem install_path -> GetPathRoot("")-Crash (installer.sh hatte Fallback) | cf3be285 | LOCALAPPDATA\toolboxv2-Fallback, src-tauri-Kopie gesynced |
| Doppel-BOM in installer.ps1 (B3-Sync-Artefakt) -> PS5.1 param-Parser crasht | 6dbc80f7 | BOM-Normalisierung, beide Kopien byte-identisch |
| B4: Auto-Mount registriert "feature:isaa", start_app sucht "app:isaa" -> Konflikt statt already | c86cef6b | source=app:<name> in _auto_mount_features (local_ui.py) |
| B4b: Kaltstart-Race - isaa-Routen beim Owner-Boot gemountet BEVOR _mounted_sources[app:isaa] gesetzt; mount_app prueft Konflikte vor unmount -> start_app bei 0s merged:false | f4fe4225 | mount_app adoptiert Konflikt-Routen, deren Handler identisch/gleiches Modul (Adoption), 3 Tests |

## Race-Beweis (Doppel-POST-Experiment, altes Wheel, Datei-Log)
T0s: {"merged":false,"conflicts":["GET /isaa","GET /api/chats","GET /api/vfs/...","/ws/isaa"...]}
T+5s: {"merged":true,"already":true}
-> Routen kommen beim Boot (Auto-Mount/Feature-Load), Registrierung app:isaa
   passiert spaeter (Lazy). Mit B4b-Adoption: 0s merged:true.

## V9 warm-up (Lauf A v5, vor B4b)
/api/wizard/state first call 6.2s (lazy mount), 2nd <0.3s. Verify-Timeout 5s->15s
(19019ff7+5d2e4d04). Im finalen Wheel (mit B4b) first call <1s.

## Beobachtungen (NICHT gefixt, out of scope)
1. tb workers stop hinterlaesst manchmal http_worker-Prozess (Waise) - Port 8467 bleibt belegt;
   zweiter workers start failed mit Port-Konflikt. Workaround: Waisen-Prozess beenden
   (Task-Manager). Zusammenhang mit B1 (Manager-Crash nach Spawn) - nach B1-Fix nur noch nach
   hartem Abbruch.
2. Startbanner zeigt "Version 0.1.27" trotz Paket 0.1.29 (kosmetisch, install.manifest korrekt).
3. tb.exe direkt ausserhalb des Install-Ordners starten: tb-worker findet State nicht
   (manifest vom install-dir abhaengig) - Dokumentation/UX-Thema.

## Test-Gates (final, alle gruen)
test_mount_adoption (3) + test_app_mount_alias (2) + test_verify_install_windows (5)
+ test_state_dir_pids (3) + test_worker_manager_symlink (3) + test_state_dir (13)
+ test_installer_protocol (7) = 36 Tests OK.

## Dateien
- scripts/verify_install_windows.ps1 (neu)
- toolboxv2/tests/test_core/test_verify_install_windows.py, test_mount_adoption.py,
  test_app_mount_alias.py, test_state_dir_pids.py, test_worker_manager_symlink.py (neu)
- toolboxv2/utils/workers/fast_tb.py (mount_app Adoption + _adoptable_conflicts)
- toolboxv2/utils/workers/fast/local_ui.py (source app:<name>)
- toolboxv2/utils/clis/cli_worker_manager.py (_ensure_symlink Fallback)
- toolboxv2/__main__.py, toolboxv2/utils/system/getting_and_closing_app.py (pids_dir)
- installer.ps1 + toolboxv2/simple-core/src-tauri/installer.ps1 (B3-Fallback, BOM, byte-identisch)
- toolboxv2/simple-core/src-tauri/wheels/: toolboxv2-0.1.29-py3-none-any.whl (a33d6c54fa94) + BUILD_ID
