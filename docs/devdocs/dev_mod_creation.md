# Mod-Erstellung in ToolBoxV2

## Schnellstart

Ein neues Mod besteht aus mindestens 3 Dateien im Verzeichnis `toolboxv2/mods/{MOD_NAME}/`:

```
toolboxv2/mods/MyMod/
├── __init__.py      # Mod-Export, Name, Tools
├── types.py         # Datentypen, Enums
└── tb_adapter.py    # MainTool-Implementierung
```

---

## 1. __init__.py — Mod-Exports

Die `__init__.py` ist der Einstiegspunkt für das Modul.

```python
<!-- verified: mods/DB/__init__.py -->
from .tb_adapter import Tools
from .tb_adapter import Tools as DB
from .ui import db_manager_ui

Name = "DB"
Tools = Tools

version = Tools.version
# private = True
```

**Pflichtfelder:**

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `Name` | str | Modul-Identifikator |
| `Tools` | class | Haupt-Tool-Klasse |
| `version` | str | Versionsnummer |

**Optional:**

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `private` | bool | Mod nicht öffentlich verfügbar |

---

## 2. types.py — Datenmodelle

Definiert alle Datentypen, Enums und Config-Strukturen.

```python
<!-- verified: types.py::DatabaseModes -->
from dataclasses import dataclass
from enum import Enum

@dataclass
class DatabaseModes(Enum):
    LC = "LOCAL_DICT"
    LR = "LOCAL_REDDIS"
    RR = "REMOTE_REDDIS"
    CB = "CLUSTER_BLOB"

    @classmethod
    def crate(cls, mode: str):
        if mode == "LC":
            return DatabaseModes.LC
        elif mode == "LR":
            return DatabaseModes.LR
        elif mode == "RR":
            return DatabaseModes.RR
        elif mode == "CB":
            return DatabaseModes.CB
        else:
            raise ValueError(f"{mode} != RR,LR,LC,CB")

<!-- verified: types.py::AuthenticationTypes -->
@dataclass
class AuthenticationTypes(Enum):
    UserNamePassword = "password"
    Uri = "url"
    PassKey = "passkey"
    location = "location"
    none = "none"
```

**Best Practices:**
- Enums für feste Optionslisten
- `@dataclass` für zusammengesetzte Typen
- Keine Business-Logik in `types.py`
- Version-Konstanten optional

---

## 3. tb_adapter.py — MainTool-Implementierung

Das Herzstück des Moduls. Erbt von `MainTool`.

```python
<!-- verified: main_tool.py::MainTool -->
from toolboxv2.utils.system.main_tool import MainTool

class Tools(MainTool):
    toolID: str = "my-unique-id"
    spec = "app"
    version = "1.0.0"

    def __init__(self, *args, **kwargs):
        # Standard-Konstruktor - NICHT überschreiben!
        super().__init__(*args, **kwargs)

    async def __ainit__(self, *args, **kwargs):
        # Asynchrone Initialisierung
        self.version = kwargs.get("v", kwargs.get("version", "0.0.0"))
        self.name = kwargs["name"]
        self.tools = kwargs.get("tool", {})
        self.logger = kwargs.get("logs")

        # Config initialisieren falls nicht vorhanden
        if not hasattr(self, 'config'):
            self.config = {}

        # Optional: User auslesen
        self.user = None
        self.description = kwargs.get("description", "My mod description")

    # === TOOL METHODS ===

    def my_tool_method(self, param1: str, param2: int) -> Result:
        """Dokumentation für die Tool-Methode"""
        try:
            # Business Logic hier
            result_data = {"processed": param1, "count": param2}

            return self.return_result(
                error=ToolBoxError.none,
                exec_code=0,
                help_text="Success",
                data=result_data
            )
        except Exception as e:
            return self.return_result(
                error=ToolBoxError.unknown,
                exec_code=1,
                help_text=str(e)
            )

    # === CALLBACKS ===

    async def on_start(self):
        """Wird beim Modul-Start aufgerufen (optional)"""
        self.logger.info(f"{self.name} started")

    async def on_exit(self):
        """Wird beim Modul-Ende aufgerufen (optional)"""
        self.logger.info(f"{self.name} stopped")
```

---

## Modul registrieren

### Im Hauptmodul (toolboxv2/mods/)

In der zentralen Modul-Registry (z.B. `mods/__init__.py`):

```python
from .MyMod import Name, Tools, version

__all__ = ["Name", "Tools", "version", ...]
```

---

## Tool-Methoden schreiben

### Standard-Signatur

```python
def tool_method(self, param1: str, param2: int = 10, **kwargs) -> Result:
    """Beschreibung was das Tool tut.

    Args:
        param1: Erster Parameter
        param2: Zweiter Parameter (default: 10)

    Returns:
        Result mit data und data_info
    """
    # Validierung
    if not param1:
        return self.return_result(
            error=ToolBoxError.invalid_params,
            exec_code=1,
            help_text="param1 is required"
        )

    # Business Logic
    data = {"result": f"Processed: {param1}"}

    # Erfolg
    return self.return_result(
        error=ToolBoxError.none,
        exec_code=0,
        help_text="Operation successful",
        data=data,
        data_info={"count": len(data)}
    )
```

### Asynchrone Tool-Methoden

```python
async def async_tool(self, query: str) -> Result:
    """Async Tool für I/O-Operationen"""
    try:
        result = await self.app.a_run_any(
            "SomeOtherMod",
            "some_method",
            query=query,
            get_results=True
        )

        if result.is_error():
            return self.return_result(
                error=ToolBoxError.execution_failed,
                exec_code=1,
                help_text=result.get()
            )

        return self.return_result(
            error=ToolBoxError.none,
            exec_code=0,
            data=result.get()
        )
    except Exception as e:
        return self.return_result(
            error=ToolBoxError.unknown,
            exec_code=1,
            help_text=str(e)
        )
```

---

## App-Instanz-Zugriff

```python
<!-- verified: main_tool.py::app -->
@property
def app(self):
    return get_app(
        from_=f"{self.spec}.{self.name}|{self.toolID} {self.interface}"
    )

# Verwendung:
self.app.print("Status:", "Alles gut!")  # Output mit Modul-Präfix

# Cross-Modul Aufruf:
result = await self.app.a_run_any(
    "CloudM.AuthManager",
    "authenticate",
    username=username,
    get_results=True
)
```

---

## Logging

```python
<!-- verified: main_tool.py::__ainit__::get_logger -->
self.logger = kwargs.get("logs", get_logger())

# Verwendung:
self.logger.info(f"{self.name}: Operation started")
self.logger.error(f"{self.name}: Error: {e}")
self.logger.warning(f"{self.name}: Warning: {msg}")
```

---

## Debug-Modus

```python
# Im Fehlerfall bei aktivem Debug:
if self.app.debug:
    import traceback
    traceback.print_exc()
```

---

## Versionsverwaltung

```python
<!-- verified: main_tool.py::get_version -->
def get_version(self) -> str:
    """Returns the version"""
    return self.version

# Alternativ: aus pyproject.toml
<!-- verified: main_tool.py::get_version_from_pyproject -->
version = get_version_from_pyproject('pyproject.toml')
```

---

## Komplettes Beispiel: Minimal-Modul

```
toolboxv2/mods/Hello/
├── __init__.py
└── tb_adapter.py
```

**__init__.py:**
```python
from .tb_adapter import Tools

Name = "Hello"
Tools = Tools
version = Tools.version
```

**tb_adapter.py:**
```python
from toolboxv2.utils.system.main_tool import MainTool

class Tools(MainTool):
    toolID = "hello-tool-v1"
    spec = "app"
    version = "1.0.0"

    async def __ainit__(self, *args, **kwargs):
        self.name = kwargs["name"]
        self.version = kwargs.get("v", "1.0.0")
        self.config = {}

    def say_hello(self, name: str = "World") -> Result:
        return self.return_result(
            error=ToolBoxError.none,
            exec_code=0,
            help_text=f"Hello, {name}!",
            data={"message": f"Hello, {name}!"}
        )
```
