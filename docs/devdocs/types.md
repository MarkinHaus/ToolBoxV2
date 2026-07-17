# Core Types (`utils/system/types.py`)

> **File:** `toolboxv2/utils/system/types.py` (~3235 Zeilen)
> Zentrale Typdefinitionen: `AppType`, `Result`, `Request`, Dispatch-System.

## Why This Matters

`AppType` ist die **Wurzel** des gesamten ToolBoxV2-Systems. Jede Mod-Funktion, jeder Worker, jedes Event läuft durch eine `AppType`-Instanz. Ohne Verständnis von `AppType` und `Result` kann man keine Mods schreiben, keine Workers konfigurieren und keine API-Endpunkte erstellen.

`Result` ist der **universelle Return-Typ** — vergleichbar mit `Promise` in JavaScript oder `Result<T>` in Rust. Jede Mod-Funktion MUSS ein `Result` zurückgeben.

## AppType

Haupt-Anwendungsklasse. Jede ToolBoxV2-Instanz ist ein `AppType`. Verwaltet:
- Mod-Registry (geladene Module + Funktionen)
- DB-Verbindung
- ISAA Agent-Framework
- Event-/WebSocket-Manager
- Manifest-Konfiguration
- Logger

```python
from toolboxv2.utils.system.getting_and_closing_app import get_app
app = get_app(name="my-app")
```

### Wichtige Methoden (Auswahl, 200+ total)

| Kategorie | Methoden |
|-----------|----------|
| **Mod-Management** | `a_add_mod`, `a_remove_mod`, `a_run_any`, `a_run_all` |
| **DB** | `app.db` → DB-Instanz (LC/LR/RR/CB) |
| **Config** | `app.alive`, `app.id`, `app.info`, `app.logger` |
| **Decorator** | `app.tb(name, mod_name, api, ...)` → Funktions-Registrierung |
| **ISAA** | `app.get_agent(name)`, `app.get_agent_builder()` |
| **VFS** | `app.vfs` → Virtuelles Dateisystem |
| **Auth** | `app.validate_session(token)` |
| **Lifecycle** | `app.save_state()`, `app.exit()` |

### `@tb` Decorator

Registriert Funktionen im Dispatch-System:

```python
@app.tb(name="hello", mod_name="MyMod", api=True)
async def hello(name: str) -> str:
    return f"Hello {name}"
```

Wichtige Parameter:
- `api=True` → Exponiert via REST API
- `initial=True` → Wird beim Mod-Load ausgeführt
- `exit_f=True` → Wird beim Mod-Unload ausgeführt
- `memory_cache=True` → In-Memory Cache mit TTL
- `level=0` → Access-Level (0=public, -1=admin)
- `request_as_kwarg=True` → Injiziert `RequestData` als kwarg

## Result

Universeller Return-Typ für alle Mod-Funktionen. Kapselt Daten, Metadaten, Execution-Status.

```python
Result.ok(data="hello", info="greeting", interface=ToolBoxInterfaces.cli)
Result.error(data="not found", info="404")
Result.default_internal_error()
```

### Result-Factory-Methoden

| Methode | Beschreibung |
|---------|-------------|
| `Result.ok(data, info, ...)` | Erfolg (exec_code=1) |
| `Result.error(data, info, ...)` | Fehler (exec_code=-1) |
| `Result.default_internal_error()` | Standard 500-Fehler |
| `Result.default_user_error()` | Standard 400-Fehler |
| `Result.custom_error(code, data, info)` | Custom Error-Code |

### ToolBoxResult (intern)

Internes Datenmodell für `Result`:

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `data_to` | `str` | Ziel-Interface (cli/remote/native) |
| `data_info` | `str \| None` | Info-String |
| `data` | `Any \| None` | Payload |
| `data_type` | `str \| None` | Typ-Annotation |

## ToolBoxInterfaces (Enum)

| Wert | Beschreibung |
|------|-------------|
| `cli` | CLI-Ausgabe |
| `remote` | API/Remote-Antwort |
| `future` | Future/Promise |
| `native` | Native Integration |

## ToolBoxError (Enum)

| Wert | Beschreibung |
|------|-------------|
| `none` | Kein Fehler |
| `input_error` | Fehlende/ungültige Eingabe |
| `internal_error` | Interner Fehler |
| `custom_error` | Benutzerdefinierter Fehler |

## AppArgs

CLI-Argument-Wrapper, geparst von `tb`:

```python
args = AppArgs().default()
args.verbose = True
app = get_app(name="my-app", args=args)
```

## WebSocketContext

WebSocket-Verbindungskontext für Worker:

| Feld | Typ |
|------|-----|
| `conn_id` | `str` |
| `session` | `SessionData` |
| `worker_id` | `str` |

## Used By

Nahezu **alles** — AppType ist die Wurzel des gesamten Dispatch-Systems. Result ist der universelle Return-Typ aller Mod-Funktionen.

## Related

- [FileHandlerV2](file_handler.md) — Storage-Backbone
- [Session Management](../runtime/session.md) — Nutzt `types.py` SessionData
- [All Functions Enums](all_functions_enums.md) — Dispatch-Mechanismus
- [ISAA Overview](../mods/isaa/index.md) — Nutzt AppType für Agent-Building
