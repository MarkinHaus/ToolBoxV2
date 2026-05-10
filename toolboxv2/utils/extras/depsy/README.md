# depmanager

Cross-platform system package manager abstraction. Single-file, zero dependencies (stdlib only).

Detects available managers on the current OS, picks the best one (or your preferred), and gives you one API for all of them.

## Supported Managers

| Manager | OS | sudo | binary |
|---------|-----|------|--------|
| winget | Windows | no | `winget` |
| choco | Windows | no | `choco` |
| scoop | Windows | no | `scoop` |
| brew | macOS/Linux | no | `brew` |
| apt | Linux (Debian) | yes | `apt` |
| dnf | Linux (Fedora) | yes | `dnf` |
| pacman | Linux (Arch) | yes | `pacman` |
| zypper | Linux (SUSE) | yes | `zypper` |
| nix | any | no | `nix-env` |
| uv | any | no | `uv` |
| pip | any | no | `pip` |

## Manager Resolution

Priority: **function arg** > **`DEPX_MANAGER` env var** > **auto-detect** (first available, ranked by platform).

Default ranking per OS:

- **Windows:** winget → choco → scoop → uv → pip
- **macOS:** brew → nix → uv → pip
- **Linux:** apt → dnf → pacman → zypper → nix → brew → uv → pip

---

## CLI Usage

```bash
# which managers are available?
python depmanager.py detect

# search
python depmanager.py search ollama
python depmanager.py -m brew search ollama     # force brew
python depmanager.py search ollama -i          # interactive (pick from list)

# install
python depmanager.py install cowsay
python depmanager.py install ollama -i         # search → pick → install live
python depmanager.py -m winget install Ollama.Ollama

# uninstall
python depmanager.py uninstall cowsay

# update
python depmanager.py update cowsay

# list installed (optional filter)
python depmanager.py list
python depmanager.py list python               # only entries matching "python"
```

**Note:** `-m`/`--manager` must come **before** the subcommand (argparse limitation).

## Python API

```python
import depmanager as dm

# ── detection ──
dm.detect_available()          # ['apt', 'uv', 'pip']
dm.resolve_manager()           # ManagerDef(name='apt', ...)
dm.resolve_manager("pip")      # force pip

# ── search ──
results = dm.search("ollama")
# [{"id": "ollama", "name": "ollama", "version": "0.3.x"}, ...]

results = dm.search("requests", manager="pip")

# interactive: prints list, user picks, returns selected id or None
choice = dm.search("ollama", interactive=True)

# ── install / uninstall / update ──
r = dm.install("cowsay")
r = dm.install("cowsay", manager="pip")
r = dm.install("Ollama.Ollama", manager="winget")

r = dm.uninstall("cowsay")
r = dm.update("cowsay")

# interactive install: live terminal output (stdin/stdout passthrough)
r = dm.install("cowsay", interactive=True)

# ── list installed ──
entries = dm.list_installed()
entries = dm.list_installed(filter_str="python")
entries = dm.list_installed(manager="pip")

# ── full interactive workflow ──
# search → user picks from list → install with live output
r = dm.interactive_install("ollama")
```

### RunResult

Every `install`, `uninstall`, `update` call returns a `RunResult`:

```python
@dataclass
class RunResult:
    ok: bool          # True if returncode == 0
    returncode: int   # process exit code, -1 = not found, -2 = timeout
    stdout: str       # captured output (empty in interactive mode)
    stderr: str       # captured stderr
    cmd: list[str]    # the actual command that was executed
```

### Threaded Wrapper

Wrap any sync function to run in a background thread:

```python
from concurrent.futures import Future
import depmanager as dm

fut: Future = dm.threaded(dm.search, "ollama")
results = fut.result(timeout=60)

fut = dm.threaded(dm.install, "cowsay", manager="pip")
run_result = fut.result(timeout=120)
```

### Environment Variable

```bash
export DEPX_MANAGER=brew    # all calls default to brew
```

Override per-call with the `manager=` arg.

---

## sudo Handling

Managers that need root (apt, dnf, pacman, zypper) get `sudo` auto-prefixed when:

1. `needs_sudo=True` in the manager definition
2. OS is not Windows
3. Current process is not already root (`euid != 0`)
4. `sudo` binary exists on PATH

**Fallback:** If `sudo` install fails, depmanager retries without sudo and logs a warning. If `sudo` is not found at all, it runs without and warns.

---

## Debugging

### Enable debug logging

```bash
# CLI
DEPX_DEBUG=1 python depmanager.py search ollama

# Python
import logging
logging.basicConfig(level=logging.DEBUG)
import depmanager as dm
dm.search("ollama")
```

Debug output shows every command before execution:

```
DEBUG:depmanager:exec: apt search ollama
```

### Inspect raw command output

```python
import depmanager as dm

# _run is the internal executor — use it to see raw output
r = dm._run(["apt", "search", "ollama"])
print(r.stdout)   # raw stdout
print(r.stderr)   # raw stderr
print(r.cmd)      # ['apt', 'search', 'ollama']
print(r.ok)       # True/False
```

### Check what command would be built

```python
import depmanager as dm

mdef = dm.resolve_manager("apt")
cmd = dm._build_cmd(mdef.install_cmd, "ollama", mdef)
print(cmd)  # ['sudo', 'apt', 'install', '-y', 'ollama']  (if non-root)
```

### Search parsing issues

If search returns unexpected results, check the raw output:

```python
import depmanager as dm

mdef = dm.resolve_manager("apt")
r = dm._run(list(mdef.search_cmd[:-1]) + ["ollama"] )
print(repr(r.stdout[:500]))  # see raw output + newlines

# then parse manually
entries = dm._parse_search(r.stdout, "apt")
print(entries)
```

### Common issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `pip install` fails with "externally-managed-environment" | PEP 668 on modern distros | Already handled: `--break-system-packages` flag is included |
| `uv pip install` fails with "no virtual environment" | uv defaults to venv-only | Already handled: `--system --break-system-packages` flags |
| `apt install` fails with permission denied | Not root, no sudo | Run as root or install sudo |
| Search returns empty list | Package doesn't exist in that manager's repo | Try a different manager: `search("ollama", manager="brew")` |
| `winget` not found on Windows | Old Windows 10 or App Installer not installed | Install via Microsoft Store |
| `-m pip` after subcommand is ignored | argparse positional ordering | Put `-m` before subcommand: `python depmanager.py -m pip search X` |

---

## Tests

```bash
# run all tests (auto-skips managers not present on your system)
python -m unittest test_depmanager -v

# run only detection tests
python -m unittest test_depmanager.TestDetection -v

# run only install/uninstall cycle for a specific manager
python -m unittest test_depmanager.TestInstallUninstall.test_cycle_apt -v
python -m unittest test_depmanager.TestInstallUninstall.test_cycle_pip -v
```

Tests perform **real** install/uninstall cycles with `cowsay` (tiny, no daemon, no side effects). Each test cleans up after itself.

Test classes:

| Class | What it tests |
|-------|---------------|
| `TestDetection` | detect_available, resolve_manager, env/arg override priority |
| `TestSearch` | search output parsing per manager |
| `TestList` | list_installed + filter |
| `TestInstallUninstall` | Full cycle: install → verify → list → uninstall → verify gone |
| `TestThreaded` | threaded() wrapper with Future |
| `TestRunResult` | _run with nonexistent command, success case |

---

## Adding a New Manager

Add a `ManagerDef` in `depmanager.py`:

```python
_reg(ManagerDef(
    "yay", "yay",                                    # name, binary
    search_cmd=("yay", "-Ss", "{pkg}"),              # {pkg} gets replaced
    install_cmd=("yay", "-S", "--noconfirm", "{pkg}"),
    uninstall_cmd=("yay", "-R", "--noconfirm", "{pkg}"),
    update_cmd=("yay", "-Syu", "--noconfirm"),
    list_cmd=("yay", "-Q"),
))
```

Then add it to `_RANK` for its platform, and add a parser case in `_parse_search` if the output format differs from existing ones (generic fallback: one entry per non-empty line).

## Files

```
depmanager.py          # the module (single file, zero deps)
test_depmanager.py     # integration tests
README.md              # this file
```
