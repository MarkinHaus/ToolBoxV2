# How-to: Mod erstellen & extern verlinken

> **Typ:** How-to Guide (Diátaxis)
> Lerne: Wie erstelle ich ein neues ToolBoxV2-Mod — im Terminal UND im Code.

---

## 1. Mod-Struktur

Jedes Mod lebt unter `toolboxv2/mods/<ModName>/`:

```
toolboxv2/mods/MyMod/
├── __init__.py          # Mod-Entry-Point
├── main.py              # Hauptlogik (optional)
├── helpers.py           # Hilfsfunktionen (optional)
└── static/              # Statische Dateien (optional)
```

### Externe Ordner verlinken

Wenn dein Mod-Code außerhalb von `toolboxv2/mods/` liegt:

```bash
# Symlink erstellen (Linux/Mac)
ln -s /path/to/external/MyMod toolboxv2/mods/MyMod

# Junction (Windows)
mklink /J toolboxv2\mods\MyMod C:\path\to\external\MyMod
```

Oder via `__init__.py` Forwarding:

```python
# toolboxv2/mods/MyMod/__init__.py
import sys
sys.path.insert(0, "/external/path/to/")
from my_external_mod import *
```

---

## 2. Mod schreiben — Minimalbeispiel

### `__init__.py` (Pflicht)

```python
from toolboxv2 import App, Result, ToolBoxInterfaces  # Core Types

# ============ MOD METADATA ============
MOD_NAME = "MyMod"
MOD_VERSION = "1.0.0"

# Wird beim Laden ausgeführt
def on_start(app):
    """Called when mod is loaded."""
    app.logger.info(f"{MOD_NAME} v{MOD_VERSION} loaded")
    return App.on_start_result

# Wird beim Entladen ausgeführt
def on_exit(app):
    """Called when mod is unloaded."""
    app.logger.info(f"{MOD_NAME} unloaded")
    return App.on_exit_result

# ============ EXPORTED FUNCTIONS ============

# Methode 1: @app.tb Decorator (empfohlen)
@app.tb(name="greet", mod_name=MOD_NAME, api=True)
async def greet(name: str = "World"):
    """Greet someone. Available via CLI, API, and ISAA."""
    return Result.ok(
        data=f"Hello {name}!",
        info="greeting generated",
        interface=ToolBoxInterfaces.cli
    )

# Methode 2: Manuelles Registration
@app.tb(name="calculate", mod_name=MOD_NAME, api=True, level=1)
async def calculate(operation: str, a: float, b: float):
    """Do math. Level 1 = requires login."""
    operations = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
    }
    if operation not in operations:
        return Result.error(
            data=f"Unknown operation: {operation}",
            info="valid: add, sub, mul"
        )
    result = operations[operation](a, b)
    return Result.ok(data=result)

# Methode 3: Sync-Funktion (wird auto-wrapped)
@app.tb(name="health_check", mod_name=MOD_NAME, api=True)
def health_check():
    """Sync function — automatically wrapped async."""
    return Result.ok(data={"status": "healthy", "mod": MOD_NAME})
```

---

## 3. Mod verwenden — Terminal

### Installieren & Laden

```bash
# Mod laden (automatisch bei tb Start wenn unter mods/)
tb -l                    # Liste alle geladenen Mods
tb -i MyMod              # Mod explizit installieren/installieren

# Nach Code-Änderungen: Enums regenerieren!
tb -l -sfe              # Scan-for-enums: aktualisiert all_functions_enums.py
```

### CLI Aufrufe

```bash
# Funktion aufrufen
tb -c MyMod greet --name "Alice"
# → Hello Alice!

# Mit Parameter
tb -c MyMod calculate --operation add --a 5 --b 3
# → 8.0

# Health check
tb -c MyMod health_check
# → {"status": "healthy", "mod": "MyMod"}

# Debug-Modus (hot-reload bei Code-Änderung)
tb -c MyMod greet --name "Bob" --debug
```

### API Aufrufe (HTTP)

```bash
# GET health
curl http://localhost:8500/api/MyMod/health_check

# POST mit JSON
curl -X POST http://localhost:8500/api/MyMod/calculate \
  -H "Content-Type: application/json" \
  -d '{"operation": "add", "a": 5, "b": 3}'
```

---

## 4. Mod in Code verwenden (Python)

```python
from toolboxv2 import get_app

app = get_app(name="my-app")

# Mod-Funktion aufrufen
result = await app.a_run_any("MyMod.greet", name="Alice")
print(result.get())  # → "Hello Alice!"

# Oder direkt
result = await app.a_run_any("MyMod.calculate", operation="mul", a=4, b=7)
print(result.get())  # → 28.0

# ISAA Agent mit Mod-Tool ausstatten
builder = app.get_agent_builder()("my_agent")
agent = (
    builder
    .add_tool("MyMod.greet")
    .add_tool("MyMod.calculate")
    .build()
)
```

---

## 5. Fortgeschritten: RequestData & Session

```python
@app.tb(name="user_data", mod_name=MOD_NAME, api=True, request_as_kwarg=True)
async def user_data(request, session=None):
    """Access user session data."""
    if session is None or not session.is_authenticated:
        return Result.error(data="Not authenticated", info="401")

    user_id = session.user_id
    # ... do user-specific work ...
    return Result.ok(data={"user": user_id, "level": session.level})
```

---

## 6. Fortgeschritten: FileHandler für Configs

```python
from toolboxv2.utils.system.file_handler import FileHandlerV2, StorageScope

@app.tb(name="get_config", mod_name=MOD_NAME)
async def get_config():
    """Load mod config (encrypted, auto-local)."""
    fh = FileHandlerV2("mymod.config", name=MOD_NAME)
    await fh.aload()
    return Result.ok(data=fh.to_dict())

@app.tb(name="set_config", mod_name=MOD_NAME)
async def set_config(key: str, value: str):
    """Save mod config."""
    fh = FileHandlerV2("mymod.config", name=MOD_NAME)
    await fh.aload()
    await fh.aset(key, value)
    await fh.asave()
    return Result.ok(data="saved")
```

---

## 7. Cheat Sheet

| Was | Terminal | Code |
|-----|----------|------|
| Mod laden | `tb -l` | `app.a_add_mod("MyMod")` |
| Enums aktualisieren | `tb -l -sfe` | — (auto-generated) |
| Funktion aufrufen | `tb -c MyMod greet` | `await app.a_run_any("MyMod.greet")` |
| Alle Funktionen | `tb -c MyMod --help` | `app.mods_functions["MyMod"]` |
| Debug/Hot-reload | `--debug` flag | — |
| Config speichern | `tb -c MyMod set_config` | `FileHandlerV2("mymod.config")` |
| API exponieren | `@app.tb(api=True)` | auto-routed bei Worker-Start |

## Related

- [Core Types](types.md) — `AppType`, `Result`, `@tb` Decorator
- [FileHandlerV2](file_handler.md) — Config/Data Storage
- [All Functions Enums](all_functions_enums.md) — Dispatch-Tabelle
- [Toolbox Integration](toolbox_integration.md) — AccessController, API Routing
- [Onboarding](../foundations/onboarding.md) — Installation
