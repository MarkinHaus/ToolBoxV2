# ToolBoxV2 — Service Manager & Start Profiles

> **Datei:** `toolboxv2/utils/clis/service_manager.py`
> **Entry-Point:** `tb --sm` (Boot-Startup) · `tb services <cmd>` (interaktiv)
> **Decoupled von:** ISAA Job-System (`headless_runner.py`, `os_scheduler.py`) — komplett unabhängig

---

## 1. Architektur-Überblick

```
tb --sm ──────────────────► run_service_manager_startup()
                                  │
                                  ▼
tb services <cmd> ──────► cli_services() → ServiceManager
                                                │
                        ┌───────────────────────┤
                        ▼                       ▼
               ServiceRegistry           ServiceManager
               (Singleton)               (PID-File Tracking)
               ┌──────────────┐          ┌──────────────────┐
               │ Built-in Svcs│          │ .info/pids/*.pid  │
               │ per category │          │ .info/services.json│
               └──────────────┘          └──────────────────┘
```

**Kernprinzip:** Kein Daemon, kein OS-Service-API. Jeder Service ist ein detached Subprocess (`tb <name>`), getrackt via PID-File.

---

## 2. Datenstrukturen

### `ServiceDefinition` (dataclass)

| Feld | Typ | Default | Bedeutung |
|---|---|---|---|
| `name` | `str` | — | Eindeutiger Key |
| `description` | `str` | — | Anzeige-Text |
| `category` | `str` | — | `core` / `infrastructure` / `extension` |
| `module` | `str` | — | Python-Modul-Pfad |
| `entry_point` | `str` | — | Funktion im Modul |
| `is_async` | `bool` | `False` | Asyncio-Entrypoint |
| `runner_key` | `Optional[str]` | `None` | CLI-Argument-Key |

### `ServiceStartResult` (dataclass)

| Feld | Typ | Bedeutung |
|---|---|---|
| `name` | `str` | Service-Name |
| `success` | `bool` | Start erfolgreich? |
| `pid` | `Optional[int]` | Prozess-ID (bei Erfolg) |
| `error` | `Optional[str]` | Fehlermeldung (bei Misserfolg) |

---

## 3. `ServiceRegistry` — Singleton

Registriert alle Built-in Services beim ersten Aufruf, danach identische Instanz.

```python
registry = ServiceRegistry()   # immer dieselbe Instanz
```

### Built-in Services

#### 🔷 Core
| Name | Beschreibung |
|---|---|
| `custom` | Beliebiger TB-Custom-Command |
| `workers` | Worker-Orchestrierung (HTTP, WS, Broker) |
| `db` | MinIO Blob Storage Management (async) |
| `isaaK` | Discord & Telegram Kernel Agent (async) |

#### 🔧 Infrastructure
| Name | Beschreibung |
|---|---|
| `broker` | Event-Management via ZMQ |
| `http_worker` | HTTP Worker Server |
| `ws_worker` | WebSocket Worker Server |

#### 🔌 Extension
| Name | Beschreibung |
|---|---|
| `p2p` | P2P Chat, File Transfer, Voice |
| `mcp` | MCP Server für AI Agents |
| `gui` | Graphical User Interface |
| `llm-gateway` | OpenAI-kompatibler LLM-Proxy (llama.cpp) |

### Registry API

```python
registry.get("workers")               # → ServiceDefinition | None
registry.get_all()                    # → Dict[str, ServiceDefinition]
registry.get_by_category("core")      # → List[ServiceDefinition]
registry.list_names()                 # → List[str]
registry.register(ServiceDefinition(...))  # Custom Service hinzufügen
```

---

## 4. `ServiceManager` — Kern-Klasse

### Dateipfade

```
tb_root_dir/.info/
    pids/
        workers.pid       ← PID-File pro Service
        db.pid
        ...
    services.json         ← Persistente Konfiguration
```

### Konfiguration (`services.json`)

```json
{
  "services": {
    "workers": {
      "auto_start": true,
      "auto_restart": false,
      "args": ["--port", "8080"]
    },
    "mcp": {
      "auto_start": true,
      "auto_restart": false,
      "args": []
    }
  }
}
```

### Methoden-Referenz

#### Config

```python
manager.load_config()                                      # → Dict
manager.save_config(config: Dict)                          # speichert JSON
manager.configure_service(name, auto_start, auto_restart, args)
manager.get_auto_start_services()                          # → List[str]
manager.get_service_args(name)                             # → List[str]
```

#### PID / Laufzeitstatus

```python
running, pid = manager.is_service_running(name)
# Prüft via:
#   Windows → tasklist /FI "PID eq <pid>"
#   Unix    → os.kill(pid, 0)
# Stale PID-Files werden automatisch gelöscht
```

#### Start / Stop / Restart

```python
result = manager.start_service(name, args=None, save_args=True)
# Subprocess: python -m toolboxv2 <name> [args...]
# Windows: CREATE_NO_WINDOW | DETACHED_PROCESS
# Unix:    start_new_session=True, stdout/stderr=DEVNULL
# Wartet 0.5s und verifiziert, ob Prozess noch läuft
# args=None → gespeicherte Args aus Config

manager.stop_service(name, graceful=True)
# Unix:    SIGTERM (graceful) / SIGKILL (force)
# Windows: taskkill [/F]
# Wartet bis 5s auf Shutdown, räumt PID-File auf
```

#### Status / Info

```python
status = manager.get_all_status(include_registry=True)
# → Dict[name → {running, pid, auto_start, auto_restart, category, ...}]
# include_registry=True: ergänzt alle Registry-Services (auch unkonfigurierte)
# include_registry=False: nur explizit konfigurierte Services

info = manager.get_service_info(name)
# → Dict mit: running, pid, auto_start, auto_restart, category,
#             description, module, entry_point, is_async, runner_key
# → None wenn Service komplett unbekannt
```

---

## 5. Boot-Startup (`tb --sm`)

```python
# __main__.py
if "--sm" in sys.argv:
    from toolboxv2.utils.clis.service_manager import run_service_manager_startup
    if 'init' in sys.argv:
        # Linux: setup_service_linux()
        # Windows: asyncio.run(setup_service_windows())
    sys.exit(run_service_manager_startup())
```

`run_service_manager_startup()`:
1. Lädt alle `auto_start=True` Services aus der Config
2. Startet jeden mit `start_service()` (benutzt gespeicherte Args)
3. Gibt Exit-Code `0` (alle OK) oder `1` (mind. ein Fehler) zurück

---

## 6. CLI — `tb services`

```
tb services status [--name <n>]         Laufzeit-Tabelle aller Services
tb services start [name] [args...]      Start (ohne name: alle auto-start)
  --auto-start                          Auto-start direkt beim Start aktivieren
tb services stop [name] [--force]       Stop (ohne name: alle)
tb services restart [name] [args...]    Restart (ohne name: alle laufenden)
  --force
tb services config <name>               Konfiguriere Service
  --auto-start=true|false
  --auto-restart=true|false
  --args [ARG...]                       Standardargumente setzen
  --clear-args                          Gespeicherte Args löschen
tb services list                        Konfigurierte Services (mit Flags)
tb services info <name>                 Detailview: PID, Modul, Entry-Point, Args
tb services registry                    Alle Built-in Services nach Kategorie
```

---

## 7. Decoupling: Service Manager vs. ISAA Job-System

```
Service Manager                    ISAA Job-System (isaa-spezifisch)
───────────────                    ──────────────────────────────────
TB-interne Prozesse                Geplante Agent-Aufgaben
PID-File Tracking                  jobs.json + JobDefinition
tb services start/stop             JobManager, headless_runner.py
auto_start / auto_restart          on_time / on_interval / on_cron / on_boot
via tb --sm                        via os_scheduler.py (schtasks/cron/launchd)
keine ISAA-Dependency              braucht toolboxv2.mods.isaa.*
```

---

---

# Start Profiles — Design-Spec

> **Idee:** Nutzungsprofile (`work`, `development`, `freetime`) die automatisch TB-Services starten UND externe Anwendungen an definierten Bildschirmpositionen öffnen.
> **Integration:** Vollständig in `service_manager.py`, Profile wechselbar via `tb services profile switch <name>`.

---

## 8. Konzept

Ein Profil definiert:
- Welche **TB-Services** gestartet werden (ersetzt individuelle `auto_start` Flags)
- Welche **externen Apps** geöffnet werden (Browser, IDE, Terminal, etc.)
- Für jede App: **Fensterposition** (Monitor, x/y, Breite/Höhe) und **Startup-URL/Argument**

### Vordefinierte Profile

| Profil | TB-Services | Apps |
|---|---|---|
| `work` | workers, db, mcp | Browser (SimpleCore), IDE, Slack |
| `development` | workers, db, broker, llm-gateway | VS Code, Browser (localhost), Terminal |
| `freetime` | gui | — |
| `minimal` | (keine) | — |

---

## 9. Datenstrukturen (Erweiterung)

```python
@dataclass
class AppLaunchConfig:
    """Externe Anwendung mit Fensterposition"""
    name: str
    command: List[str]            # z.B. ["code", "/path/to/project"]
    args: List[str] = field(default_factory=list)

    # Fensterpositionierung (optional)
    window_x: Optional[int] = None
    window_y: Optional[int] = None
    window_width: Optional[int] = None
    window_height: Optional[int] = None
    monitor: int = 0              # 0 = primary

    # Verhalten
    wait_seconds: float = 0.5     # Wartezeit nach Start (für Fenster-Init)
    platform: Optional[str] = None  # None = alle, "windows"/"linux"/"darwin"


@dataclass
class StartProfile:
    """Nutzungsprofil mit Services + Apps"""
    name: str
    description: str

    # TB-Services (überschreibt auto_start)
    services: List[str] = field(default_factory=list)
    service_args: Dict[str, List[str]] = field(default_factory=dict)

    # Externe Apps
    apps: List[AppLaunchConfig] = field(default_factory=list)

    # Verhalten
    stop_other_services: bool = True   # Services nicht im Profil stoppen
    is_default: bool = False
```

### Konfig-Erweiterung `services.json`

```json
{
  "services": { ... },
  "active_profile": "work",
  "profiles": {
    "work": {
      "description": "Arbeitsumgebung",
      "services": ["workers", "db", "mcp"],
      "service_args": {
        "workers": ["--env", "prod"]
      },
      "apps": [
        {
          "name": "Browser",
          "command": ["firefox", "https://simplecore.app"],
          "window_x": 0, "window_y": 0,
          "window_width": 1280, "window_height": 1080,
          "monitor": 0
        },
        {
          "name": "IDE",
          "command": ["code", "/srv/projects/ToolBoxV2"],
          "window_x": 1280, "window_y": 0,
          "window_width": 1280, "window_height": 1080,
          "monitor": 0
        }
      ],
      "stop_other_services": true
    },
    "development": {
      "description": "Entwicklungsumgebung",
      "services": ["workers", "db", "broker", "llm-gateway"],
      "apps": [
        {
          "name": "VS Code",
          "command": ["code", "/srv/projects/ToolBoxV2"],
          "window_x": 0, "window_y": 0,
          "window_width": 1920, "window_height": 1080
        },
        {
          "name": "Dev Browser",
          "command": ["firefox", "http://localhost:8080"],
          "monitor": 1
        }
      ]
    },
    "freetime": {
      "description": "Freizeit",
      "services": ["gui"],
      "apps": [],
      "stop_other_services": true
    }
  }
}
```

---

## 10. `ProfileManager` — Implementierung (Sketch)

```python
class ProfileManager:
    """Verwaltet Start-Profile, integriert in ServiceManager"""

    def __init__(self, manager: ServiceManager):
        self.manager = manager

    # ── Profil-CRUD ───────────────────────────────────────────────────

    def get_profile(self, name: str) -> Optional[StartProfile]:
        config = self.manager.load_config()
        raw = config.get("profiles", {}).get(name)
        return StartProfile(**raw) if raw else None

    def save_profile(self, profile: StartProfile) -> None:
        config = self.manager.load_config()
        config.setdefault("profiles", {})[profile.name] = asdict(profile)
        self.manager.save_config(config)

    def list_profiles(self) -> List[str]:
        config = self.manager.load_config()
        return list(config.get("profiles", {}).keys())

    def get_active(self) -> Optional[str]:
        return self.manager.load_config().get("active_profile")

    # ── Switch ────────────────────────────────────────────────────────

    def switch(self, name: str, launch_apps: bool = True) -> None:
        profile = self.get_profile(name)
        if not profile:
            raise ValueError(f"Profile '{name}' not found")

        # 1. Andere Services stoppen (wenn konfiguriert)
        if profile.stop_other_services:
            running = self.manager.get_all_status(include_registry=False)
            for svc_name, info in running.items():
                if info["running"] and svc_name not in profile.services:
                    self.manager.stop_service(svc_name)

        # 2. Profil-Services starten
        for svc_name in profile.services:
            args = profile.service_args.get(svc_name)
            self.manager.start_service(svc_name, args=args)

        # 3. Apps öffnen + positionieren
        if launch_apps:
            for app in profile.apps:
                self._launch_app(app)

        # 4. Aktives Profil speichern
        config = self.manager.load_config()
        config["active_profile"] = name
        self.manager.save_config(config)

    # ── App-Launcher ──────────────────────────────────────────────────

    def _launch_app(self, app: AppLaunchConfig) -> None:
        """Startet externe App und positioniert Fenster (platform-agnostic)"""
        import subprocess, time

        # Platform-Filter
        if app.platform and app.platform != sys.platform:
            return

        cmd = app.command + app.args
        subprocess.Popen(cmd, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Fensterpositionierung (nach kurzer Wartezeit)
        if any(v is not None for v in [app.window_x, app.window_y,
                                        app.window_width, app.window_height]):
            time.sleep(app.wait_seconds)
            self._position_window(app)

    def _position_window(self, app: AppLaunchConfig) -> None:
        """Plattform-spezifische Fensterpositionierung"""
        if IS_LINUX:
            self._position_linux(app)
        elif IS_WINDOWS:
            self._position_windows(app)
        elif IS_MACOS:
            self._position_macos(app)

    def _position_linux(self, app: AppLaunchConfig) -> None:
        # wmctrl oder xdotool
        # wmctrl -r :ACTIVE: -e 0,<x>,<y>,<w>,<h>
        # Suche Fenster per Name: wmctrl -l | grep <app.name>
        try:
            subprocess.run([
                "wmctrl", "-r", app.name, "-e",
                f"0,{app.window_x},{app.window_y},{app.window_width},{app.window_height}"
            ], capture_output=True)
        except FileNotFoundError:
            pass  # wmctrl nicht installiert, skip

    def _position_windows(self, app: AppLaunchConfig) -> None:
        # pywin32: win32gui.MoveWindow / win32gui.FindWindow
        # Oder PowerShell: Add-Type -A System.Windows.Forms
        pass

    def _position_macos(self, app: AppLaunchConfig) -> None:
        # AppleScript: tell application "X" to set bounds of window 1 to {x,y,w,h}
        script = (
            f'tell application "{app.name}" to '
            f'set bounds of window 1 to '
            f'{{{app.window_x}, {app.window_y}, '
            f'{app.window_x + (app.window_width or 0)}, '
            f'{app.window_y + (app.window_height or 0)}}}'
        )
        subprocess.run(["osascript", "-e", script], capture_output=True)
```

---

## 11. CLI-Erweiterung: `tb services profile`

```
tb services profile list               Alle Profile anzeigen (mit aktivem Marker)
tb services profile switch <name>      Profil wechseln (Stop/Start Services + Apps)
  --no-apps                            Nur Services, keine Apps öffnen
tb services profile show <name>        Profil-Details (Services + Apps)
tb services profile set-default <name> Standard-Profil für tb --sm
tb services profile create <name>      Interaktiv neues Profil erstellen (guided)
tb services profile edit <name>        Profil in $EDITOR öffnen (JSON)
tb services profile delete <name>      Profil löschen
```

### Integration mit `tb --sm`

```bash
# Boot: Standard-Profil aktivieren
tb --sm                          # → aktives/default Profil laden
tb --sm --profile work           # → explizit Profil angeben
```

```python
# __main__.py Erweiterung
if "--sm" in sys.argv:
    profile_idx = sys.argv.index("--profile") if "--profile" in sys.argv else -1
    profile_name = sys.argv[profile_idx + 1] if profile_idx >= 0 else None

    from toolboxv2.utils.clis.service_manager import (
        run_service_manager_startup, ServiceManager, ProfileManager
    )

    if profile_name:
        manager = ServiceManager()
        pm = ProfileManager(manager)
        pm.switch(profile_name, launch_apps=True)
        sys.exit(0)
    else:
        sys.exit(run_service_manager_startup())
```

---

## 12. Beispiel-Workflow

```bash
# Einmalig: Profile konfigurieren
tb services profile create work       # geführter Assistent

# Oder direkt in JSON editieren
tb services profile edit work

# Profil testen
tb services profile switch work

# Als Boot-Standard setzen
tb services profile set-default work

# Dann bei Boot
tb --sm                                # → work-Profil startet automatisch

# Schnell wechseln während der Arbeit
tb services profile switch development
tb services profile switch freetime

# Aktuelles Profil prüfen
tb services profile list
# Output:
#   ● work          (aktiv)  — workers, db, mcp
#     development            — workers, db, broker, llm-gateway
#     freetime               — gui
```

---

## 13. Test-Erweiterungen (unittest)

Neue Test-Klassen für `test_cli_service_manager.py`:

```python
class TestStartProfile(unittest.TestCase):
    """Tests für StartProfile Dataclass"""

    def test_profile_creation(self): ...
    def test_profile_defaults(self): ...


class TestProfileManager(unittest.TestCase):
    """Tests für ProfileManager"""

    def test_save_and_load_profile(self): ...
    def test_list_profiles(self): ...
    def test_get_active_profile(self): ...
    def test_switch_stops_other_services(self): ...  # mock stop_service
    def test_switch_starts_profile_services(self): ...
    def test_switch_sets_active_profile(self): ...
    def test_switch_unknown_profile_raises(self): ...


class TestAppLaunchConfig(unittest.TestCase):
    """Tests für AppLaunchConfig"""

    def test_platform_filter(self): ...   # app.platform != sys.platform → skip
    def test_no_window_position_skips_positioning(self): ...
```

---

## 14. Abhängigkeiten (Linux-Fensterpositionierung)

```bash
# Empfohlen: wmctrl (X11)
sudo apt install wmctrl

# Alternative: xdotool
sudo apt install xdotool

# Wayland (kein wmctrl): ydotool oder KDE-spezifisch
```

Für Windows: `pywin32` (`pip install pywin32`) — optional, graceful skip wenn nicht vorhanden.

---

*Generiert aus: `service_manager.py`, `test_cli_service_manager.py`, `os_scheduler.py`, `headless_runner.py`*
