# BeastCLI — Macro System for Power Users

The BeastCLI (`tb cli`) is ToolBoxV2's interactive shell. Its macro system
lets you record, build, share, and run multi-step automation — from a single
keystroke or a Docker container.

## Starting BeastCLI
```bash
tb cli
```

Profile `developer` or `homelab` opens it as default via bare `tb`.

---

## Core Concepts

| Concept | What it is |
|---|---|
| **Macro** | Named sequence of CLI commands, stored in `cli/macros.json` via BlobFile |
| **Variable** | `$r1`, `$r2` — auto-captured results; `$arg1`, `$1` — macro arguments |
| **MacroContext** | Isolated execution state per macro run (variables, loop vars, flags) |
| **Workspace** | Named environment with its own modules + startup commands |

Variable scope priority (highest wins):
```
loop vars  >  macro vars  >  CLI quick vars ($r1, $r2, ...)
```

---

## Creating Macros

### Method A — Interactive wizard
```
:macro create
```

Walks you through: name → description → commands (empty line ends) → tags → loop count.

### Method B — Record live
```
:record my_deploy
# ... run commands normally ...
stop_recording
```

Every command you execute gets captured verbatim.

### Method C — Import from file
```
:macro import /path/to/macros.json
# or
:macro import https://... (when registry sharing is set up)
```

File format (from `_export_macros`):
```json
{
  "version": "1.0",
  "exported_at": "2025-01-01T00:00:00",
  "macros": {
    "my_macro": {
      "commands": ["echo hello", "DB get_status"],
      "description": "Example",
      "variables": {},
      "tags": ["example"],
      "loop_count": 1
    }
  }
}
```

---

## Running Macros
```
:play my_macro          # interactive selector if no name
:play my_macro arg1 arg2   # with arguments → $1, $arg1
@my_macro               # shortcut prefix (same as :play my_macro)
```

Result is stored automatically:
```
💾 Macro result saved as $r3
```

---

## Control Flow Reference

All of these work inside macro `commands` lists:
```
# This is a comment

set counter = 0
set items = ['a', 'b', 'c']

for item in $items: echo $item
for i in range(5): mymod process $i

while $counter < 10: set counter = $counter + 1

if $r1 == 'ok': echo success
if $r1 != 'ok': break

sleep 2.5
return $counter
```

Loop safety limit: **1000 iterations** (hardcoded in `_handle_macro_while`).

---

## Export & Share

### Export to file
```
:macro export
```

Select which macros, enter output path → JSON file with schema above.

### Share via Registry
```bash
# Pack macros as feature
tb manifest pack my_macros

# Upload
tb registry upload ./features_sto/tbv2-feature-my_macros-1.0.0.zip

# Others install:
tb fl unpack my_macros
# Then in BeastCLI:
:macro import ~/.local/share/ToolBoxV2/features/my_macros/macros.json
```

---

## Pre-defined Macro Packs

### `server-ops` — Server Admin Pack
```json
{
  "macros": {
    "health_check": {
      "commands": [
        "echo '=== Service Health ==='",
        "DB get_status",
        "set db_ok = $r1",
        "if $db_ok != 'ok': echo 'WARNING: DB not healthy'",
        "echo '=== Worker Status ==='",
        "echo 'Health check done'"
      ],
      "description": "Quick server health check",
      "tags": ["server", "ops"]
    },
    "rotate_logs": {
      "commands": [
        "set date = '$(date +%Y%m%d)'",
        "echo 'Rotating logs for $date'",
        "sleep 1",
        "echo 'Log rotation complete'"
      ],
      "description": "Rotate and archive logs",
      "tags": ["server", "maintenance"]
    }
  }
}
```

### `dev-workflow` — Developer Pack
```json
{
  "macros": {
    "test_and_pack": {
      "commands": [
        "echo 'Running tests...'",
        "set mod = $arg1",
        "if $mod == '': return 'ERROR: provide mod name as arg1'",
        "echo 'Packing $mod...'",
        "echo 'Done: $mod'"
      ],
      "description": "Test then pack a mod. Usage: :play test_and_pack mymod",
      "tags": ["dev", "workflow"]
    }
  }
}
```

---

## Multi-Instance: Each in its own Docker Container

Run isolated BeastCLI sessions — one per project, team member, or environment.

### Setup

**`Dockerfile.minicli`** — place in repo root:
```dockerfile
FROM python:3.12-slim

RUN useradd -m tbuser
WORKDIR /home/tbuser

# Install ToolBoxV2
RUN pip install --no-cache-dir ToolBoxV2

# Pre-load macros from a pack (optional)
COPY macros/ /home/tbuser/.local/share/ToolBoxV2/features/macros/

USER tbuser

# Data dir per container (macros, history, context persist in volume)
ENV TB_DATA_DIR=/home/tbuser/.local/share/ToolBoxV2/.data

CMD ["tb", "cli"]
```

**Start isolated sessions:**
```bash
# Instance 1 — Project A
docker run -it --rm \
  -v tb_proj_a:/home/tbuser/.local/share/ToolBoxV2/.data \
  --name tb_proj_a \
  toolbox-cli

# Instance 2 — Project B (completely separate macros + state)
docker run -it --rm \
  -v tb_proj_b:/home/tbuser/.local/share/ToolBoxV2/.data \
  --name tb_proj_b \
  toolbox-cli
```

Each container gets its own `cli/macros.json` and `cli/context.c` via the
named volume. `BlobFile` writes to `TB_DATA_DIR` — so volumes cleanly isolate
all state.

### Execute a macro non-interactively
```bash
# Run macro and exit — useful for CI or scripting
docker run --rm \
  -v tb_proj_a:/home/tbuser/.local/share/ToolBoxV2/.data \
  toolbox-cli \
  tb cli -- :play health_check
```

### Multiple sessions on the same machine (no Docker)
```bash
# Session 1: default workspace
TB_DATA_DIR=~/.tb/workspace_a tb cli

# Session 2: different data dir → completely separate macro set
TB_DATA_DIR=~/.tb/workspace_b tb cli
```

`BlobFile("cli/macros.json", ...)` resolves relative to `TB_DATA_DIR`, so
this cleanly separates all state without Docker overhead.

---

## Keyboard Shortcuts Reference

| Shortcut | Action |
|---|---|
| `Ctrl+P` | Command palette |
| `Ctrl+R` | History search |
| `Ctrl+S` | Save last result as `$rN` |
| `Ctrl+T` | Toggle compact mode |
| `Alt+H` | Smart help for current input |
| `Alt+V` | Show variables |
| `:macro` | Macro manager |
| `:record <name>` | Start recording |
| `:play <name>` | Execute macro |
| `:v` | View all `$` variables |
