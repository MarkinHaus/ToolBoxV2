# ToolBoxV2 Mod-Architektur (Mini-Doc)

> Zweck: Schnellreferenz für ISAA-Agents / Entwickler, um zu verstehen, **wie ein Mod in ToolBoxV2 aufgebaut ist**, wie die `Tool`-Klasse eingebunden wird und welche Rückgabe-Konventionen gelten.

---

## 1. Was ist ein Mod?

Ein **Mod** (Modul) ist eine Python-Datei oder ein Python-Paket im Verzeichnis:

```
toolboxv2/mods/<mod_name>.py        # Single-File Mod
toolboxv2/mods/<mod_name>/__init__.py   # Paket-Mod (mit Unterordnern)
```

Mods stellen dem System **Funktionen** (sogenannte *Tools*) zur Verfügung, die:
- über die HTTP-API aufrufbar sind (sofern `api=True`)
- im CLI verfügbar sind (via `tb -c <mod_name>`)
- im Worker/ISAA-Framework registriert sind

---

## 2. Zwei Schreibstile

ToolBoxV2 unterstützt **beide** Architekturen. Neue Mods sollten den **modernen Funktions-Stil** nutzen.

### 2.1 Moderner Stil (empfohlen) ✅

Datei: `toolboxv2/mods/<name>.py`

```python
from toolboxv2 import get_app

# 1) Den export-Decorator vom App-Singleton holen
export = get_app(from_="<Name>.Export").tb
Name = "<name>"
version = get_app(from_="<Name>.Export").version


# 2) Funktionen mit @export registrieren
@export(name="hello", mod_name=Name, version=version, api=True)
async def hello(app, request, name: str = "World"):
    """Gibt einen Gruss zurück."""
    return f"Hallo {name}!"


@export("init", initial=True)  # wird beim App-Start automatisch aufgerufen
async def init_func(app):
    print("Mod initialisiert")


@export("cleanup", exit_f=True)  # wird beim App-Stop aufgerufen
async def cleanup_func(app):
    print("Mod aufgeräumt")
```

**Wichtige Decorator-Parameter** (Quelle: `toolboxv2/utils/toolbox.py`, `App.tb(...)`):

| Parameter | Typ | Bedeutung                                         |
|---|---|---------------------------------------------------|
| `name` | str | Funktionsname (kann als erstes positional kommen) |
| `mod_name` | str | Mod-Name (für API-Routing)                        |
| `version` | str | Version des Tools                                 |
| `api` | bool | True = über HTTP-API erreichbar                   |
| `api_methods` | list[str] | Erlaubte HTTP-Methoden, z.B. `["GET","POST"]`     |
| `initial` | bool | True = beim App-Start ausführen                   |
| `exit_f` | bool | True = beim App-Stop ausführen                    |
| `test_only` | bool | Nur in Test-Runs                                  |
| `request_as_kwarg` | bool | Request als Keyword-Argument statt positional     |
| `helper` | str | Beschreibung für UI/API-Doku                      |
| `samples` | list | Beispiel-Aufrufe (für API-Doku)                   |
| `level` | int | Berechtigungslevel (default -1)                   |
| `interface` | enum | Standard-Interface (native/remote)                |

### 2.2 Klassischer Stil (Legacy) ⚠️

Datei: `toolboxv2/mods/<name>.py`

```python
from toolboxv2.utils.system.main_tool import MainTool


class MyTool(MainTool):
    def __init__(self, load=True, v="1.0.0"):
        self.version = v
        self.name = "<name>"
        self.color = "blue"
        self.logger = ...
        self.tools = {
            "all": ["run"],
            "name": self.name,
            "run": self.run,    # Funktion -> Methoden-Referenz
        }
        MainTool.__init__(
            self, load=load, v=v, tool=self.tools,
            name=self.name, logs=self.logger,
            color=self.color, on_exit=self.on_exit,
        )

    def on_start(self):
        ...

    def on_exit(self):
        ...

    async def run(self, *args, **kwargs):
        return self.return_result(data="ok", info="OK")
```

---

## 3. Die `Tool`-Klasse (Hauptschnittstelle)

Datei: `toolboxv2/utils/system/main_tool.py`

```python
class MainTool:
    toolID: str
    interface: ToolBoxInterfaces
    spec: Spec
    name: str
    color: str
    stuf: bool          # mute printing

    def __init__(self, load=True, v=None, tool=None,
                 name=None, logs=None, color=None, on_exit=None):
        self.tools = tool or {}
        self.logger = logs
        self.color = color
        self.todo = []    # Pending tasks

    async def __ainit__(self):    # Async-Init (awaitable!)
        self.version = ...
        self.name = ...
        self.tools = ...
        self.logger = ...
        self.todo = ...
        self.config = ...
        self.user = ...
        self.description = ...

    @property
    def app(self) -> "App":
        return get_app(from_=self.spec.name)

    @staticmethod
    def return_result(*, data=None, info="OK", exec_code=0,
                      status_code=200, interface=ToolBoxInterfaces.remote) -> Result:
        return Result.default(data=data, info=info, exec_code=exec_code,
                              status_code=status_code, interface=interface)

    def print(self, *a, **kw):
        ...  # formatierte Konsolen-Ausgabe

    def add_str_to_config(self, key, value):
        ...

    def webInstall(self, user_instance, construct_render) -> str:
        ...
```

### Wichtige Felder

| Feld | Bedeutung |
|---|---|
| `tools` | Dict mit `{"all": [fn_names], "name": mod_name, fn_name: callable}` |
| `todo` | Liste pending async Tasks |
| `spec` | Mod-Spec (Name, Version, Description) |
| `app` | Zugriff auf das App-Singleton (Properties, Config, andere Mods) |
| `logger` | Logger-Instanz für strukturierte Logs |
| `config` | Mod-Konfiguration (aus Manifest) |
| `user` | Aktueller User-Context (Auth) |

### Lifecycle-Methoden (überschreibbar)

| Methode | Wann |
|---|---|
| `__init__` | Sync-Init |
| `__ainit__` | Async-Init (empfohlen für I/O) |
| `on_start` | Nach Registrierung |
| `on_exit` | Beim App-Stop |

---

## 4. Rückgabe-Werte: `Result`

**Mods dürfen rohe Werte zurückgeben** — die Runtime packt sie automatisch in ein `Result`. Für explizite Kontrolle:

Datei: `toolboxv2/utils/system/types.py`

```python
from toolboxv2.utils.system.types import Result

# Erfolg
Result.ok(data={"x": 1}, info="OK")
Result.json(data={"x": 1})                    # JSON-API Response
Result.html(data="<h1>Hi</h1>")               # HTML Render
Result.text(data="plain text")                # Plain Text
Result.binary(data=bytes, download_name="x")  # Binary Download
Result.file(data, filename="x.pdf")           # File Download
Result.redirect(url="/dashboard")            # HTTP Redirect
Result.stream(generator)                      # SSE / Stream
Result.sse(generator)                         # Server-Sent Events
Result.future(data=payload)                   # Async Future
Result.empty()                                # Leere Antwort

# Fehler
Result.error(data=None, info="Boom", exec_code=450)
Result.default_user_error(info="Bad input", exec_code=-3)
Result.default_internal_error(info="Oops", exec_code=-2)
```

**`exec_code` Konventionen** (typisch):

| Code | Bedeutung |
|---|---|
| `0` | OK |
| `450` | Generischer Fehler |
| `-2` | Internal Error |
| `-3` | User Error (Input) |
| `-4` | Not Found |

---

## 5. Ein Mod erstellen — Schritt für Schritt

### 5.1 Datei anlegen

```bash
# Single-File:
toolboxv2/mods/mein_mod.py

# ODER als Paket:
toolboxv2/mods/mein_mod/__init__.py
```

### 5.2 Minimaler Inhalt (modern)

```python
from toolboxv2 import get_app

export = get_app(from_="MeinMod.Export").tb
Name = "MeinMod"
version = "1.0.0"


@export(name="ping", mod_name=Name, version=version, api=True)
async def ping(app, request):
    return {"pong": True}
```

### 5.3 Testen

```bash
# CLI
tb -c MeinMod ping

# HTTP
curl http://localhost:5000/api/MeinMod/ping
```

---

## 6. Best Practices

1. **Moderner Stil** für neue Mods (Funktionen + `@export`).
2. **Async-Funktionen** wenn I/O involviert ist.
3. **`Result` als Rückgabe** bei kontrollierten Status-Codes.
4. **Logs via `self.logger`** (klassisch) oder `app.logger` (modern).
5. **Konfiguration via Manifest** (`manifest_set`/`manifest_get`).
6. **Lifecycle-Hooks** (`initial=True`, `exit_f=True`) für Setup/Cleanup.
7. **Berechtigungen** über `level` setzen.
8. **API-Doku** via `helper` + `samples` annotieren.

---

## 7. Wo schauen bei Problemen

| Problem | Datei |
|---|---|
| Decorator-Optionen | `toolboxv2/utils/toolbox.py` → `App.tb()` |
| Result-Klassen | `toolboxv2/utils/system/types.py` → `Result` |
| MainTool-Basis | `toolboxv2/utils/system/main_tool.py` → `MainTool` |
| Mod-Loader | `toolboxv2/utils/system/loader.py` |
| Beispiele (modern) | `toolboxv2/mods/welcome.py`, `toolboxv2/mods/Minu/__init__.py` |
| Beispiele (klassisch) | `toolboxv2/mods/MinimalHtml.py` |

---

*Stand: Auto-generiert aus Quellcode-Sicht (Ground Truth = `toolboxv2/utils/toolbox.py`, `toolboxv2/utils/system/types.py`, `toolboxv2/utils/system/main_tool.py`)*
