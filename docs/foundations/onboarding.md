# Onboarding

###### GETTING STARTED · VERIFIED AGAINST `31a117e`

From `pip install` to a running & configured instance.
## 1 · Install

ToolBoxV2 requires **Python ≥ 3.10**. The package ships feature-gated extras
install only what your use case needs:

```bash
pip install toolboxv2              # core only
pip install toolboxv2[cli]         # + interactive CLI tooling
pip install toolboxv2[web]         # + workers, web UI
pip install toolboxv2[isaa]        # + agent framework (ISAA)
pip install toolboxv2[all]         # everything
```

Available extras: `core`, `cli`, `web`, `desktop`, `isaa`, `exotic`, `all`.
The install registers one entry point: `tb`.

<!-- verified: pyproject.toml::project.scripts,optional-dependencies @ 31a117e -->

## Also as an oneliner

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh | bash
```

### Windows

```powershell
irm "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.ps1" | tee tbInstaller.ps1 | % { & ([scriptblock]::Create($_)) }
```


## 2 · First run — choose a profile

The first `tb` start without a configured profile triggers the interactive first-run flow. You pick one of five profiles; the choice is written to the manifest as `app.profile` and decides what `tb` launches by default from then on:

| Profile | Who it's for | Default behavior on `tb` |
|---|---|---|
| `consumer` | I use an app / a mod. Just start. | Opens the local web UI |
| `homelab` | I run several mods, features, flows locally. *(default)* | Opens the local web UI |
| `server` | I manage a distributed system / IT infrastructure. | Headless — inspect via `tb status` |
| `business` | I need a fast health status. | Headless health summary |
| `developer` | I develop mods, features, or the core. | Opens the terminal CLI |

The profile is not a lock-in — it's a default. Change it any time:

```bash
tb manifest        # interactive manifest editor
```

<!-- verified: toolboxv2/utils/clis/first_run.py::PROFILES,run_first_run @ 31a117e -->

## 3 · Config wizard

After the profile choice, the first-run flow offers the config wizard (`Y` by default). It walks **seven steps** and writes the manifest plus a `.env` file derived from `env-template`:

1. Application Settings
2. Database Configuration
3. Workers
4. Services
5. ISAA (agent framework — models, API keys)
6. Environment Variables
7. Review & Save

Skip it and re-run later — the wizard is idempotent and pre-fills existing values:

```bash
tb -init config
```

<!-- verified: toolboxv2/utils/clis/config_wizard.py::run_config_wizard @ 31a117e -->

## 4 · Verify the instance

```bash
tb status          # DB, API, P2P, Workers at a glance
tb -v              # version + loaded modules (-l for all)
tb access          # make tb command global callable
tb --guide         # interactive usage guide
```

## 5 · Where to go next

###### BY PROFILE

- **Consumer / Homelab** — `tb gui` starts the graphical interface; `tb mods` opens the interactive module manager. Install mods with `tb -i <name>`.
- **Server** — [Worker System](../runtime/index.md): `tb workers start`, sessions, the ZMQ broker (`tb broker`), and the [ServiceManager](../runtime/service_manager.md) for auto-start/restart (`tb --init-sm`).
- **Developer** — [Coding Guidelines](../devdocs/guidelines.md), then [Core Types](../devdocs/types.md). See also [First Run Wizard](first_run.md). `tb -c <MOD> <FUNC>` executes any module function directly; `--debug` enables hot-reload.
- **Agents** — [ISAA](../mods/isaa/index.md) is the agent framework. `tb mcp` starts an MCP server so external agents can drive this instance.

## Command map

The full surface, grouped the way `tb -h` groups it:

| Group | Commands |
|---|---|
| Interfaces | `gui` · `mods` · `browser` · `docksh` · `access` |
| Execution | `flow` · `run` · `-c MOD FUNC` · `--kwargs k=v` |
| Accounts | `login` · `logout` · `user` |
| Operations | `status` · `obs` · `workers` · `session` · `broker` · `http_worker` · `ws_worker` |
| Data | `db` · `--delete-config` · `--delete-data` *(destructive — guarded)* |
| Build & Dev | `build` · `venv` · `--test` · `--profiler` · `--debug` · `-sfe` |
| Agents | `mcp` |

Every subcommand has its own help: `tb <command> -h`.

<!-- verified: tb -h @ 31a117e (live output) -->
