# ToolBoxV2 — Installation Guide

> Single source of truth for all install paths. The installer handles everything — runtime detection, feature selection, manifest generation, and lifecycle management.

---

## Quick Install

### Linux / macOS
```bash
curl -fsSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh | bash
```

### Windows (PowerShell)
```powershell
irm "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.ps1" | tee tbInstaller.ps1 | % { & ([scriptblock]::Create($_)) }
```

---

## Install Modes

The installer supports four modes. Choose based on your use case.

### `native` — Single Binary *(default, recommended)*

No Python required. Downloads a pre-built Nuitka binary from the registry or GitHub Releases.

```bash
bash installer.sh --mode native
```

### `uv` — Python Package via uv

Installs via `uv tool install`. uv manages its own Python — no system Python required. Falls back to `pip + venv` if Python 3.11+ is found but uv is absent.

```bash
bash installer.sh --mode uv
```

### `docker` — Containerized

Pulls the Docker image and writes a `tb` wrapper script. No local runtime needed.

```bash
bash installer.sh --mode docker
```

### `source` — Full Source

Clones from GitHub or downloads a release tarball from the registry. Ideal for contributors and developers.

```bash
# From Git (editable dev tree)
bash installer.sh --mode source

# From registry tarball
# → select "registry" when prompted
```

---

## Custom Config (Unattended Install)

Create a `tb-install.yaml` in the current directory. The installer reads it automatically.

```yaml
install_mode: native         # native | uv | docker | source
source_from: git             # git | registry  (source mode only)
source_branch: main
install_path: ""             # empty = OS default
environment: development     # development | production | staging
instance_id: tbv2_main
features:
  - core
  - cli
  - web
optional:
  nginx: false
  docker_runtime: false
  ollama: false
  minio: false
  registry: false            # also install tb-registry service
registry_url: "https://registry.simplecore.app"
```

Pass it explicitly:

```bash
bash installer.sh --config my-server.yaml
```

---

## Features

`mini` and `core` are always included and cannot be disabled.

| Feature   | Description                     | Default |
|-----------|---------------------------------|---------|
| `mini`    | Minimal app + types             | ✅ always |
| `core`    | Core ToolBox functionality      | ✅ always |
| `cli`     | Command-line interface          | ✅ |
| `web`     | HTTP/WS workers + API           | ❌ |
| `desktop` | Desktop UI (PyQt6)              | ❌ |
| `isaa`    | AI/LLM agent integration        | ❌ |
| `exotic`  | Scientific computing extras     | ❌ |

Install with extras:

```bash
# uv mode — extras selected interactively or via config
bash installer.sh --mode uv

# pip/venv (fallback) — same extras syntax
pip install "ToolBoxV2[web,isaa]"
```

---

## Install Locations (OS Defaults)

| OS      | Default `TOOLBOX_HOME`                              |
|---------|-----------------------------------------------------|
| Linux   | `/opt/toolboxv2`                                    |
| macOS   | `~/Library/Application Support/toolboxv2`           |
| Windows | `%LOCALAPPDATA%\toolboxv2`                          |

Override at runtime:

```bash
bash installer.sh --path /srv/myapp/toolbox
```

---

## Runtime Detection Chain

The installer selects the runtime automatically. You never need to configure this manually.

```
uv installed?
├─ yes → uv (preferred)
└─ no
    ├─ Python 3.11+ found? → pip + venv (full fallback)
    └─ neither found
        ├─ native / docker mode → no runtime needed
        └─ uv / source mode → bootstrap uv automatically
```

---

## Update

```bash
bash installer.sh --update
# or
bash installer.sh --config /path/to/install.manifest --update
```

Update strategy per mode:

| Mode     | Strategy                          |
|----------|-----------------------------------|
| `native` | Registry API check → new binary   |
| `uv`     | `uv tool upgrade ToolBoxV2`       |
| `docker` | `docker pull`                     |
| `source/git`      | `git pull` + `uv sync`   |
| `source/registry` | new tarball + `uv sync`  |

---

## Uninstall

```bash
bash installer.sh --uninstall
```

Uninstall is manifest-driven — removes exactly what was installed, nothing more.

---

## What the Installer Creates

```
$TOOLBOX_HOME/
├── bin/
│   └── tb             ← binary / wrapper script
├── .data/             ← runtime data (DB, cache)
├── .config/           ← config files
├── logs/              ← log output
├── src/               ← source tree (source mode only)
├── .venv/             ← virtualenv (venv / source mode)
├── install.manifest   ← installer source of truth
├── tb-manifest.yaml   ← TB runtime config
└── .env               ← environment variables
```

`install.manifest` records everything: mode, runtime, version, paths, installed features. It is the input for update and uninstall operations.

---

## Environment Variables Written

| Variable          | Description                          |
|-------------------|--------------------------------------|
| `TOOLBOX_HOME`    | Root install directory               |
| `TB_INSTALL_DIR`  | Same as above                        |
| `TB_DATA_DIR`     | Data directory (`$HOME/.data`)       |
| `TB_DIST_DIR`     | Distribution/static files            |
| `TB_ENV`          | Runtime environment                  |
| `TB_JWT_SECRET`   | ⚠ Set before production use         |
| `TB_COOKIE_SECRET`| ⚠ Set before production use         |

---

## Manual Install (no installer)

If you prefer full control:

```bash
# pip
pip install ToolBoxV2

# pip with extras
pip install "ToolBoxV2[web,isaa]"

# uv
uv tool install ToolBoxV2

# from source
git clone https://github.com/MarkinHaus/ToolBoxV2.git
cd ToolBoxV2
uv sync
# or
pip install -e .
```

---

## tb-registry (Self-Hosted)

The TB package registry can be self-hosted. Select `optional.registry: true` in your install config, or enable it when prompted:

```bash
bash installer.sh --config server.yaml
# → "Install tb-registry service? [y/N]"
```

The registry service itself is installed to `/opt/tb-registry` with a systemd unit. See [`tb-registry/deploy/`](https://github.com/MarkinHaus/ToolBoxV2/tree/main/tb-registry/deploy) for the service definition and manual setup.

---

## Troubleshooting

**`tb` not found after install**
```bash
source ~/.bashrc   # or open a new terminal
echo $PATH         # verify $TOOLBOX_HOME/bin is in PATH
```

**Permission denied on `/opt/toolboxv2`**
```bash
# Run with sudo, or use a user-writable path:
bash installer.sh --path ~/.local/share/toolboxv2
```

**Registry unreachable**
The installer automatically falls back to GitHub Releases. No action needed.

**Wrong Python version**
The installer requires Python 3.11+ for `uv`/`source` modes. Run `python3 --version` to verify, or let the installer bootstrap uv (which bundles its own Python).

---

## Next Steps

After installation:

```bash
tb              # first-run profile selection
tb status       # verify services
tb manifest validate   # check config
```

- [Quickstart](quickstart.md)
- [Configuration Guide](../configuration.md)
- [Module Management](../mods.md)
- [ISAA Integration](../isaa.md)
