# ToolBoxV2 CLI Reference

## Usage

```bash
tb [command] [options]
tb [runner] [runner_args]
```

**Entry Point:** `toolboxv2.__main__::main_runner`
<!-- verified: __main__.py::main_runner -->

---

## Commands

### Core Runners

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `default` | Interactive dashboard | `interactive_user_dashboard` |
| `gui` | Launch GUI interface | `helper_gui` |
| `status` | System status display | `status_helper` |

### Module & Extension Management

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `mods` | Interactive module manager | `mods_manager` |
| `install` | Install module | `-i` flag |
| `update` | Update module | `-u` flag |
| `remove` | Remove module | `-r` flag |

### Services & Workers

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `workers` | Worker management | `cli_worker_manager` |
| `session` | Session management | `cli_session` |
| `broker` | ZMQ event broker | `cli_event` |
| `http_worker` | HTTP worker | `cli_http_worker` |
| `ws_worker` | WebSocket worker | `cli_ws_worker` |
| `services` | Service manager | `cli_services` |

### Database & Storage

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `db` | Database CLI | `cli_db_runner` |

### Authentication

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `login` | CLI web login | `cli_web_login` |
| `logout` | Logout | `logout` |
| `user` | User management | `user_manager_main` |

### Network & API

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `mcp` | MCP server (for agents) | `cli_mcp_server` |
| `p2p` | P2P client | `cli_tcm_runner` |

### Configuration & Registry

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `manifest` | Manifest configuration | `cli_manifest_main` |
| `registry` | Registry server | `registry` |
| `llm-gateway` | LLM Gateway CLI | `cli_llm_gateway` |

### Development

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `venv` | Conda environment | `venv_runner` |
| `browser` | Browser extension | `tb_browser.install` |
| `flow` | Execute flows | `run_flow_from_file_or_load_all_flows_and_mods_from_dir` |
| `run` | Execute .tbx files | `cli_tbx_main` |

### Docker & Container

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `docksh` | Docker CLI | `_run_docksh` |
| `docker-image` | Docker image CLI | `docker_image_cli` |

### Observability

| Command | Description | Entry Point |
|---------|-------------|--------------|
| `obs` | Observability layer | `observability_helper` |

<!-- verified: __main__.py::runner_setup -->

---

## Runner Keys (split_args_by_runner)

The CLI splits arguments by these runner keywords:

```python
RUNNER_KEYS = [
    "venv", "db", "gui", "p2p", "default", "status", "browser",
    "mcp", "login", "logout", "run", "mods", "flow", "user",
    "workers", "session", "event", "broker", "http_worker", "obs",
    "ws_worker", "services", "registry", "manifest", "llm-gateway",
    "docksh", "docker-image", "fl"
]
```

<!-- verified: __main__.py::RUNNER_KEYS -->

---

## Global Flags

### Core Options

| Flag | Description |
|------|-------------|
| `-h, --help` | Show help message |
| `--guide` | Show interactive usage guide |
| `-v, --get-version` | Display version and modules |
| `-init TYPE` | Initialize (main/config/manifest) |
| `-l, --load-all-mod-in-files` | Load all modules |
| `-c MODULE FUNCTION [ARGS...]` | Execute module command |

### Module Management

| Flag | Description |
|------|-------------|
| `-i, --install MODULE` | Install module by name |
| `-u, --update MODULE` | Update module by name |
| `-r, --remove MODULE` | Uninstall module by name |
| `-m, --modi MODE` | Interface mode (default: cli) |

### Runtime Control

| Flag | Description |
|------|-------------|
| `--kill` | Terminate running instance |
| `-bg, --background-application` | Run in background |
| `-fg, --live-application` | Run in foreground |
| `--remote` | Enable remote access mode |
| `--debug` | Enable debug mode |

### Network

| Flag | Description |
|------|-------------|
| `-n, --name ID` | Instance identifier (default: main) |
| `-p, --port PORT` | Interface port (default: 5000) |
| `-w, --host HOST` | Interface host (default: 0.0.0.0) |

### Docker

| Flag | Description |
|------|-------------|
| `--docker` | Run in Docker container |
| `--build` | Build Docker image |

### Service Management

| Flag | Description |
|------|-------------|
| `--sm` | Service Manager (auto-start/restart) |
| `--init-sm` | Initialize Service Manager |
| `--lm` | Log Manager |

### Data Operations (⚠️ Caution)

| Flag | Description |
|------|-------------|
| `--delete-config NAME` | Delete named config |
| `--delete-data NAME` | Delete named data |
| `--delete-config-all` | Delete ALL configs (DANGER) |
| `--delete-data-all` | Delete ALL data (DANGER) |

### Development

| Flag | Description |
|------|-------------|
| `--test` | Run complete test suite |
| `--profiler` | Profile all functions |
| `-sfe, --save-function-enums-in-file` | Generate function enums |
| `--sysPrint` | Enable verbose output |

### Advanced

| Flag | Description |
|------|-------------|
| `--kwargs KEY=VALUE...` | Pass key-value pairs |
| `--print-root` | Print ToolBoxV2 root directory |

<!-- verified: __main__.py::parse_args -->

---

## Examples

```bash
# Basic usage
tb                              # Start CLI
tb gui                           # Launch GUI
tb status                        # Check system status

# Module commands
tb -c CloudM Version            # Get module version
tb -c helper create-user john john@example.com

# With kwargs
tb -c MyMod my_func --kwargs key1:value1 key2:value2

# Docker mode
tb --docker -m dev -p 8000 -w 0.0.0.0

# Workers
tb workers start                # Start workers
tb workers status              # Worker status

# Interactive guides
tb --guide                     # Show full guide
```
