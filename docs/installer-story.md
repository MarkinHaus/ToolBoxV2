# Installer-Story — Der komplette Weg von „null" zur laufenden App

> Stand: 05.09.2026 · gültig für v0.1.28+ · Quelle: R9–R17 Sessions, S5-Docker-Matrix, Live-Tests

## Die 4 Rollen und ihre Pfade

| Rolle | Wie installieren | Wohin | Startet mit |
|-------|-----------------|-------|-------------|
| **USER (Desktop)** | Tauri-Installer (`SimpleCore_*_x64-setup.exe` / `.msi` / `.dmg` / `.AppImage` / `.deb` / `.rpm`) | `%LOCALAPPDATA%/ToolBoxV2` (Windows) · `/Applications/ToolBoxV2` (macOS) · `/opt/toolboxv2` (Linux) | Doppelklick → `simple-core.exe` / `simple-core` Binary |
| **USER (CLI)** | `curl -fsSL https://get.simplecore.app | bash` (installer.sh) oder installer.ps1 | `$INSTALL_PATH` (default: `~/.local/share/toolboxv2` Linux, `%LOCALAPPDATA%/toolboxv2` Win) | `tb` im PATH (Wrapper `bin/tb.cmd` bzw. Kanon `%LOCALAPPDATA%/toolboxv2/bin/tb.exe`) |
| **SERVER** | installer.sh, Mode `native` oder `uv` | `/opt/toolboxv2` (typisch) | `tb workers start` (Manager :9010, UI :8467, WS :8468) |
| **DEV / HOMELAB** | installer.sh, Mode `uv` (oder git clone + `pip install -e .`) | beliebig (typ. `~/ToolBoxV2`) | `tb -m icli` + `tb --test` (braucht tb-registry Sibling!) |

## Installer-Modi (installer.sh / installer.ps1)

| Mode | Was passiert | Braucht | Ergebnis |
|------|-------------|---------|----------|
| **1 · native** | Lädt Single-Binary von GitHub-Release (GLIBC-geprüft) | kompatible libc (≥2.31; Check beim Start) | `tb` = statisches Binary, kein Python |
| **2 · uv** | Bootstrap `uv` → `uv tool install toolboxv2` (pip in isolierter venv) | Python 3.10+ via uv | `tb` = venv-Wrapper (bin/tb.cmd) + `.venv/` |
| **3 · docker** | `docker pull ghcr.io/markinhaus/toolboxv2:latest` | Docker | Container, `tb` via `docker run` |
| **4 · source** | `git clone` + editable install | git + Python | `tb` aus Source-Baum |

**Non-interaktiv** (CI/Docker): `yes "" | installer.sh` oder `printf "2\n"; yes "" | bash installer.sh`
→ Fragen werden mit Defaults beantwortet (Mode 1, sonst 2 wegen GLIBC). EOF ohne Feed = Exit 1 (seit 94e7ec71 sauber).

## Installer-Entscheidungen (Triff-Mich-Entscheidungskette)

1. **Install-Modus?** → native (schnell, kein Python) / uv (kompatibel) / docker (isolated) / source (dev)
2. **Features** → cli (default: yes), web, desktop, isaa, exotic (je y/N)
3. **Environment** → development (default) / production
4. **Install-Pfad** → default je OS (siehe Tabelle oben), leer lassen = default
5. **Profil** (nur Tauri-Wizard) → consumer / developer → steuert `app.profile` in `tb-manifest.yaml`

## Ports (kanonisch — nicht ändern!)

| Port | Was | Wer |
|------|-----|-----|
| **8467** | local-ui (HTTP, /health) | fast_tb server |
| **8468** | WebSocket | worker ws_worker |
| **9010** | Manager-API (Cluster-Secret-Gate; `/` und `/api/health` offen) | cli_worker_manager |
| 9000/9001 | MinIO (Storage) | nur bei Storage-Mod |
| 8000 | Worker-HTTP (intern) | http_worker |

## Kanon-Pfade (launch.rs tb_candidates — Reihenfolge!)

1. `TB_TB_PATH` Env-Override
2. `%LOCALAPPDATA%/toolboxv2/bin/tb.exe` ← **Kanon (installer.ps1 spiegelt dorthin, R14)**
3. `<exe-dir>/tb.exe`
4. `<install>/.venv/Scripts/tb.exe` (Windows) / `.venv/bin/tb` (Linux)
5. `PATH` (aber: GUI erbt User-PATH aus Registry, NICHT Dev-Shell-PATH!)

## First-Run-Flow (Tauri-App)

```
App startet (simple-core.exe)
  ├─ probe :8467/health
  │    ├─ 200 → target_route() → /api/wizard/state
  │    │            ├─ first_run:true → /welcome (Wizard)
  │    │            └─ sonst → /install (Manager-Hub)
  │    └─ down → discover_tb()
  │              ├─ gefunden → start_background() → wait_until_running (25s)
  │              │              → probe nochmal → UI oder Fehlermeldung
  │              └─ NICHT gefunden → Wizard (web/installer/index.html)
  │                                 → run_install (installer-Script, --json-output)
  │                                 → install-event (phase/progress/done)
  │                                 → postInstallExit(): 8467-ok → /install, sonst Tauri-close
  └─ first_run im Wizard → installation_state.json (Mirror: tb_root_dir/system/)
```

## Bekannte Fallstricke (alle bewiesen, R9–R17)

- `tb -v` startet eine App-Instanz! Status prüfen: `curl :8467/health` oder `pip show toolboxv2`
- `tb workers restart` ist IN-PLACE (lädt keinen neuen Code) → immer `stop && start`
- `cargo test --lib` braucht `toolboxv2/dist/index.html` (frontendDist, proc-macro!) → Stub setzen
- build.yml lud `requirements.txt` (existiert nicht) statt `_requirements.txt`
- python-multipart <0.0.18 ist sdist-only und braucht setuptools (gibt's ab py3.12 nicht mehr in venvs)
- `install_state.json` im BlobStorage-Backend ≠ Tauri-Kontext → Mirror nach `tb_root_dir/system/`
- `tb --test` braucht `tb-registry` als Sibling von `tb_root_dir` (kein eigenes Repo — sparse-clone aus ToolBoxV2@master)
- Alte Native-Binaries brauchen GLIBC ≥2.38; Ubuntu 20.04 hat 2.31 → Release-Build läuft in 20.04-Container
- GitHub `releases[0]` kann ein Demo-Release sein → installer.js und installer.sh filtern jetzt `draft|demo`
- CRLF: Dateien mit `\r\n` brechen bash/YAML → immer LF committen (`.gitattributes`)

## Relevante Commits (Referenz)

- `94e7ec71` installer.sh: EOF-Crash-Fix (3× `read || X=""`) + Menü `echo -e` + NO_COLOR/FORCE_COLOR
- `8e3d1a08` discover_tb-Test: PATH-unabhängig (Contract-Asserts statt `assert None`)
- `b6863ca0` build.yml L457 YAML colon-quote + frontendDist-Stub für cargo test (nightly+build)
- `02625cdd` installer.sh/.ps1 + installer.rs getrackt (build.rs resource-Panic behoben)
- `b3cb64f7` R14/R15: Ports 8467/8468/9010, Kanon-Garantie in installer.ps1, postInstallExit()
- `d721746c` python-multipart>=0.0.18 + _requirements.txt Fix (py3.12/3.13)
