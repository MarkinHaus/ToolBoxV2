# Handoff - Review: R8/R9-Abnahme Windows-Installation (2026-11-12)

> Branch: `claude/wizardly-cannon-o944j4` (ISAABOX-Checkout). Basis fuer Review:
> Commits `deb079cd` (R8b), `58442dbd` (TB-Patch-Port), `22901d92` (R9-Installer-Fixes).
> Dieses Dokument klaert offene Punkte aus dem Review-Doc (R-Punkte) und liefert
> Reproduktions-Schritte fuer die Abnahme.

---

## 1. Erledigt in diesem Abschnitt

| Punkt | Commit | Inhalt |
|-------|--------|--------|
| R8a | `2c7a9be0` | `.gitignore`: `node_modules/`, `package-lock.json`, `minio_cluster_config` + Regressionstests |
| R8b | `deb079cd` | `endpoint.dist_dir()`: `ManifestLoader(base_dir=str(manifest_base_dir()))` statt `tb_root_dir` + 3 Tests (`tests/test_core/test_endpoint_dist_dir.py`) |
| Port | `58442dbd` | Owner-Patch aus ToolBoxV2-Checkout: `mods/isaa/module.py` + `tests/test_isaa/test_shell_tool.py` (4 Tests failen identisch im TB-Quell-Checkout = patch-inherent, nicht ISAABOX-bedingt) |
| R9-Fix | `22901d92` | 2 Installer-Bugs: (1) `$script:tbFeaturesCfg` Default-Init top-level (StrictMode-Crash ohne Config-Zweig), (2) expliziter `-Path` gewinnt ueber "update existing" bei `assume=yes` + Tests (`tests/test_core/test_installer_r9_bugs.py`) |

## 2. R9-Abnahme: was belegt ist, was offen ist

### Belegt (Repro s.u.)

- Installation aus Wheel funktioniert: `EXIT=0`, `done`-JSON
- Installiertes Paket ist 0.1.29 (`importlib.metadata`), `install.manifest` = `uv/0.1.29`
- `.env` mit 9 Keys wird geschrieben
- `/health` antwortet 200: `{"status":"healthy","worker_id":"http_29420"}` (Modus `http_worker`)
- Login `local_admin` nach Env-Bereinigung automatisch (kein Sign-in-Hang)
- Kein `C:\global`-Muellpfad, keine Tracebacks im Worker-Log

### Offen (Teil der Klaerung)

1. **UI `/` und `/setup` liefern 404 im `http_worker`-Modus.**
   Grund: Die Wizard-UI-Routen (`/`, `/setup`, `/api/wizard/*`) werden von
   `local_ui.py` (`register_local_ui`, inkl. `register_wizard_routes`) gemountet.
   Der `http_worker`-Dispatcher laedt die `local_ui`-App nicht — sie ist nur im
   `first_run`/Tauri-Flow (`init_onboarding._serve_local_ui`) aktiv.
   **Bewertung erbeten:** Ist das Verhalten so gewollt (UI nur via Tauri/first_run)
   oder soll der Dispatcher `local_ui` immer mounten? Letzteres waere ein kleiner
   Patch in `server_worker.py`/`toolbox_integration.py`.
2. **`tb --version` crasht** (`ValueError: app.alive` in local-cli). Identisch im
   unberuehrten TB-Checkout → pre-existing, nicht installationsbedingt. Eigener
   Fix-Punkt, nicht R9-Blocker.
3. **`start_app` (isaa/obs) hat keine HTTP-Route im Dispatcher** — `start_app` ist
   CLI/Tray-Pfad. Fuer die Tauri-App relevant, fuer Installer-Abnahme sekundaer.
4. Default-Kommando (`tb.exe` ohne Argument) startet keinen dauerhaften UI-Server
   (beendet sich nach Login) — Design-Frage, gleiche wie Punkt 1.

### Umgebungsfalle (dokumentiert, kein Code-Bug)

Die Harness-Session des Agenten exportiert `TOOLBOXV2_REMOTE_BASE=https://simplecore.app`.
Kind-Prozesse erben das → Consumer-Start landet im Sign-in-remote-Flow statt
`local_admin`-Autologin. Vor manuellen Start-Tests in Agent-Sessions:
`Remove-Item Env:TOOLBOXV2_REMOTE_BASE` (plus `_PORT`/`_IP`/`TOOLBOXV2_BASE`).

## 3. Reproduktion R9 (PowerShell, ~10 min)

```powershell
# 1) Wheel bauen (Reihenfolge wichtig: feature-zips VOR uv build)
Set-Location <repo>
python -m toolboxv2.utils.system.ci.ci_build build   # packs features_packed/*.zip incl web
uv build --wheel --out-dir dist .                    # -> dist/toolboxv2-0.1.29-py3-none-any.whl

# 2) Frische Installation (expliziter -Path, AssumeYes)
$env:TB_WHEEL = "<repo>\dist\toolboxv2-0.1.29-py3-none-any.whl"
$inst = Join-Path $env:TEMP "tb-r9-install"
pwsh -NoProfile -ExecutionPolicy Bypass -File installer.ps1 -Mode uv -Path $inst -AssumeYes yes -Ui none -TbProfile local
# Erwartung: EXIT=0, JSON "done" ..., $inst enthaelt bin/, uv-tools/, .env, install.manifest

# 3) Version + Health
& "$inst\uv-tools\toolboxv2\Scripts\python.exe" -c "import importlib.metadata as m; print(m.version('toolboxv2'))"   # 0.1.29
(Get-Content "$inst\.env" | Where-Object {$_ -match '^\s*[A-Z_]+='}).Count                                              # 9
$env:TB_INSTALL_DIR = $inst; $env:TB_DATA_DIR = "$inst\.data"
Start-Process "$inst\bin\tb.exe" -ArgumentList 'http_worker','-p','8467' -WorkingDirectory $inst
Invoke-WebRequest http://127.0.0.1:8467/health -UseBasicParsing    # 200 healthy
```

Regressionstests (TDD-Gates dieses Abschnitts):

```powershell
python -m pytest toolboxv2/tests/test_core/test_endpoint_dist_dir.py -q      # 3 passed (R8b)
python -m pytest toolboxv2/tests/test_core/test_installer_r9_bugs.py -q      # 2 passed (R9-Fix)
python -m pytest toolboxv2/tests/test_core/test_gitignore.py -q              # (R8a)
```

## 4. Empfohlene naechste Schritte

1. Entscheidung Punkt 1 (UI im Dispatcher mounten: ja/nein) → kleiner Patch + Test.
2. `tb --version`-Crash als eigener Punkt (pre-existing, auch master betroffen).
3. Push/Review dieses Branches; danach K- und S-Punkte laut Board.
