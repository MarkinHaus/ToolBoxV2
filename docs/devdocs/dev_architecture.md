# ToolBoxV2 Architektur — 4-Layer Pattern

## Übersicht

Das ToolBoxV2 Framework folgt einem strikten **4-Layer-Architektur-Pattern**, das Separation of Concerns durch klare Abstraktionsebenen gewährleistet.

```
┌─────────────────────────────────────────────────────────┐
│                    FACADE LAYER                          │
│  (MainTool, @export decorator, Mod-Interface)            │
├─────────────────────────────────────────────────────────┤
│                   MANAGER LAYER                          │
│  (Modul-Management, App-Koordination)                    │
├─────────────────────────────────────────────────────────┤
│                   WORKER LAYER                           │
│  (Business Logic, Features, Services)                    │
├─────────────────────────────────────────────────────────┤
│                     DATA LAYER                           │
│  (Types, Enums, Config, Storage)                         │
└─────────────────────────────────────────────────────────┘
```

## Layer im Detail

### 1. DATA Layer — Grundstruktur

Enthält primitive Typen, Enums und Konfigurationsobjekte.

**Verantwortlichkeiten:**
- Typsicherheit durch `dataclass` und `Enum`
- Config-Strukturen
- Datenmodelle ohne Logik

**Beispiel:**
```python
<!-- verified: types.py::DatabaseModes -->
@dataclass
class DatabaseModes(Enum):
    LC = "LOCAL_DICT"
    LR = "LOCAL_REDDIS"
    RR = "REMOTE_REDDIS"
    CB = "CLUSTER_BLOB"
```

<!-- verified: types.py::AuthenticationTypes -->
```python
@dataclass
class AuthenticationTypes(Enum):
    UserNamePassword = "password"
    Uri = "url"
    PassKey = "passkey"
    location = "location"
    none = "none"
```

### 2. WORKER Layer — Business Logic

Implementiert die eigentliche Funktionalität des Modules.

**Verantwortlichkeiten:**
- Geschäftslogik
- Feature-Implementierung
- Datenverarbeitung
- Keine UI/Print-Logik

**Charakteristik:**
- Meist `async` für I/O-Operationen
- Keine direkten `print()` Aufrufe
- Arbeitet mit `Result`-Objekten

### 3. MANAGER Layer — Koordination

Verwaltet Worker-Instanzen und koordiniert deren Interaktion.

**Verantwortlichkeiten:**
- Modul-Lifecycle (start/stop)
- Tool-Registrierung
- Cross-Modul Kommunikation

**Beispiel:**
```python
<!-- verified: cli_registry.py::get_app -->
def get_app(name: str):
    from toolboxv2 import get_app as _get_app
    return _get_app(name)
```

### 4. FACADE Layer — API Surface

Der einzige öffentliche Kontaktpunkt nach außen.

**Verantwortlichkeiten:**
- `MainTool` als Basisklasse für alle Module
- Decorator-Integration (`@export`)
- `return_result()` für standardisierte Rückgaben
- Logging und Output

**Kernklasse:**
```python
<!-- verified: main_tool.py::MainTool -->
class MainTool:
    toolID: str = ""
    interface = None
    spec = "app"
    name = ""
    color = "Bold"
    
    def __init__(self, *args, **kwargs):
        # Standard constructor - NICHT überschreiben!
        # Stattdessen: __ainit__() verwenden
        pass
    
    async def __ainit__(self, *args, **kwargs):
        # Asynchrone Initialisierung
        # Hier: version, name, tools, config setzen
        pass
    
    @staticmethod
    def return_result(error, exec_code, help_text, data_info, data, data_to):
        # Standardisierte Ergebnisrückgabe
        pass
```

## Der @export Decorator Flow

<!-- verified: main_tool.py::__init__ -->
```python
# Im __init__ wird der @export Decorator angewendet:
self.on_exit = self.app.tb(
    mod_name=self.name,
    name=kwargs.get("on_exit").__name__,
    version=self.version
)(kwargs.get("on_exit"))
```

## Result-Pattern

<!-- verified: main_tool.py::return_result -->
```python
@staticmethod
def return_result(
    error: ToolBoxError = ToolBoxError.none,
    exec_code: int = 0,
    help_text: str = "",
    data_info=None,
    data=None,
    data_to=None
) -> Result:
    # Standardisierte Rückgabe für alle Mod-Operationen
    return Result(...)
```

## App-Property (Facade Access)

<!-- verified: main_tool.py::app -->
```python
@property
def app(self):
    return get_app(
        from_=f"{self.spec}.{self.name}|{self.toolID} {self.interface}"
    )

@app.setter
def app(self, v):
    raise PermissionError("You cannot set the App Instance!")
```

## Logging Integration

<!-- verified: main_tool.py::__ainit__::get_logger -->
```python
self.logger = kwargs.get("logs", get_logger())
# Verwendet tb_logger für zentrales Logging
```

## Version Loading

<!-- verified: main_tool.py::get_version_from_pyproject -->
```python
def get_version_from_pyproject(pyproject_path='../pyproject.toml') -> str:
    # Liest Version aus pyproject.toml [project].version
    pass
```

## Fazit

Das 4-Layer-Pattern ermöglicht:
- **Testbarkeit**: Jeder Layer isoliert testbar
- **Wiederverwendbarkeit**: Data/Worker ohne Facade nutzbar
- **Konsistenz**: Einheitliche Modulstruktur über alle Components
- **Erweiterbarkeit**: Neue Module folgen dem bewährten Pattern