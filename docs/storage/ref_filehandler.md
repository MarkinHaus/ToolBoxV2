# ToolBoxV2 FileHandler V2 API Reference

Unified Storage mit UserDataAPI Integration.

<!-- verified: file_handler.py::FileHandlerV2 -->

---

## Überblick

**FileHandlerV2** ist der moderne, einheitliche Storage-Handler mit:
- Automatische Scope-Erkennung aus Dateinamen
- UserDataAPI Backend für Cloud-Storage
- Sync und Async APIs
- Konfigurierbare Verschlüsselung
- Legacy-Kompatibilität

<!-- verified: file_handler.py -->

---

## StorageScope Enum

Definiert die Zugriffsebenen für Daten.

<!-- verified: file_handler.py::StorageScope -->

| Scope | Beschreibung |
|-------|--------------|
| `USER_PRIVATE` | Nur Benutzer, verschlüsselt, Cloud-Sync |
| `USER_PUBLIC` | Benutzer-Datei, andere können lesen |
| `PUBLIC_READ` | Admin schreibt, alle lesen |
| `PUBLIC_RW` | Alle lesen/schreiben |
| `SERVER_SCOPE` | Server-spezifische Daten |
| `MOD_DATA` | Mod-spezifisch (Standard) |
| `CONFIG` | Konfigurationsdateien |

---

## StorageBackend Enum

<!-- verified: file_handler.py::StorageBackend -->

| Backend | Beschreibung |
|---------|--------------|
| `LOCAL` | Lokale JSON-Dateien |
| `USER_DATA_API` | UserDataAPI (MinIO/Redis) |
| `AUTO` | Automatische Erkennung |

---

## Dateinamen-Konventionen

Für `.data` Dateien:

| Präfix | Scope |
|--------|-------|
| `private.data` | USER_PRIVATE |
| `public.data` | USER_PUBLIC |
| `shared.data` | PUBLIC_RW |
| `server.data` | SERVER_SCOPE |
| `mod.data` | MOD_DATA |

<!-- verified: file_handler.py -->

---

## FileHandlerV2 Konstruktor

```python
FileHandlerV2(
    filename: str,           # ".config" oder ".data"
    name: str = "mainTool", # Modulname
    scope: Optional[StorageScope] = None,
    backend: StorageBackend = StorageBackend.AUTO,
    encrypt: Optional[bool] = None,
    request: Optional[RequestData] = None,
    user_context: Optional[UserContext] = None,
    base_path: Optional[Path] = None,
)
```

<!-- verified: file_handler.py::FileHandlerV2.__init__ -->

---

## Legacy API (Sync)

### load_file_handler() -> FileHandlerV2
Lädt alle Daten aus dem Storage.

```python
fh = FileHandlerV2("settings.config", name="MyMod")
fh.load_file_handler()
```

<!-- verified: file_handler.py::FileHandlerV2.load_file_handler -->

### save_file_handler() -> FileHandlerV2
Speichert alle Daten persistenter.

```python
fh.add_to_save_file_handler("key", "value")
fh.save_file_handler()
```

<!-- verified: file_handler.py::FileHandlerV2.save_file_handler -->

### get_file_handler(key: str, default: Any = None) -> Any
Holt einen Wert.

```python
value = fh.get_file_handler("username", "default_user")
```

<!-- verified: file_handler.py::FileHandlerV2.get_file_handler -->

### add_to_save_file_handler(key: str, value: Any) -> bool
Setzt einen Wert.

```python
fh.add_to_save_file_handler("theme", "dark")
```

<!-- verified: file_handler.py::FileHandlerV2.add_to_save_file_handler -->

### remove_key_file_handler(key: str) -> None
Löscht einen Schlüssel.

<!-- verified: file_handler.py::FileHandlerV2.remove_key_file_handler -->

### set_defaults_keys_file_handler(keys: Dict, defaults: Dict) -> None
Setzt Key-Mappings und Standardwerte.

```python
fh.set_defaults_keys_file_handler(
    {"un": "username"},
    {"un": "default_user"}
)
```

<!-- verified: file_handler.py::FileHandlerV2.set_defaults_keys_file_handler -->

### delete_file() -> None
Löscht die gesamte Storage-Datei.

<!-- verified: file_handler.py::FileHandlerV2.delete_file -->

---

## V2 API (Sync)

### get(key: str, default: Any = None) -> Any
Holt einen Wert.

### set(key: str, value: Any) -> bool
Setzt einen Wert.

### delete(key: str) -> None
Löscht einen Schlüssel.

### keys() -> List[str]
Gibt alle Schlüssel zurück.

### items() -> List[tuple]
Gibt alle Key-Value Paare zurück.

### to_dict() -> Dict
Gibt alle Daten als Dictionary.

### update(data: Dict) -> FileHandlerV2
Aktualisiert mehrere Werte.

<!-- verified: file_handler.py::FileHandlerV2.get -->
<!-- verified: file_handler.py::FileHandlerV2.set -->

---

## Async API

### aload_file_handler() -> FileHandlerV2
Async laden.

### asave_file_handler() -> FileHandlerV2
Async speichern.

### aget_file_handler(key: str, default: Any = None) -> Any
Async holen.

### aget(key: str, default: Any = None) -> Any
Async holen (V2).

### aset(key: str, value: Any) -> bool
Async setzen.

<!-- verified: file_handler.py::FileHandlerV2.aload_file_handler -->

---

## Context Manager

```python
# Sync
with FileHandlerV2("settings.config", name="MyMod") as fh:
    value = fh["key"]

# Async
async with FileHandlerV2("settings.config", name="MyMod") as fh:
    value = await fh.aget("key")
```

<!-- verified: file_handler.py::FileHandlerV2.__enter__ -->

---

## Dict-ähnlicher Zugriff

```python
fh = FileHandlerV2("data.data", name="MyMod")
fh.load()

fh["name"] = "Test"     # Set
print(fh["name"])         # Get
del fh["name"]            # Delete
"name" in fh              # Check
len(fh)                    # Length
for k in fh:               # Iteration
    print(k)
```

<!-- verified: file_handler.py::FileHandlerV2.__getitem__ -->

---

## Factory Functions

### create_config_handler(name: str, filename: str = "settings.config", defaults: Dict = None) -> FileHandlerV2

Erstellt einen Handler für Konfigurationsdaten (immer lokal, verschlüsselt).

```python
fh = create_config_handler("MyMod", defaults={"theme": "light"})
```

### create_data_handler(name: str, scope: StorageScope = MOD_DATA, request: RequestData = None, backend: StorageBackend = AUTO) -> FileHandlerV2

Erstellt einen Handler für Benutzer/Mod-Daten.

```python
fh = create_data_handler("MyMod", scope=StorageScope.USER_PRIVATE, request=request)
```

<!-- verified: file_handler.py::create_config_handler -->

---

## UserContext

```python
from toolboxv2.utils.system.file_handler import (
    FileHandlerV2,
    StorageScope,
    StorageBackend,
    UserContext,
    set_current_context,
    get_current_context,
    create_config_handler,
    create_data_handler,
    LocalStorageBackend
)

# Aus Request
ctx = UserContext.from_request(request)

# Aus Session
ctx = UserContext.from_session(session)

# System-Kontext
ctx = UserContext.system()

# Anonym
ctx = UserContext.anonymous()

# Global setzen
set_current_context(ctx)
current = get_current_context()
```

<!-- verified: file_handler.py::UserContext -->
