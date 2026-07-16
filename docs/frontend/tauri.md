# ToolBoxV2 Tauri App - Architektur

## Übersicht

Die ToolBoxV2 Tauri App ist eine Desktop-Anwendung, die als Frontend für das ToolBoxV2 Backend dient. Sie bietet zwei Hauptmodi: **App-Modus** (vollständige Anwendung) und **HUD-Modus** (kompaktes Overlay).

## Verzeichnisstruktur

```
toolboxv2/simple-core/
├── src-tauri/
│   ├── src/
│   │   ├── lib.rs           # Haupteinstiegspunkt, Tauri Commands
│   │   ├── main.rs          # Rust main() Funktion
│   │   ├── hud_settings.rs  # HUD Konfiguration & Persistenz
│   │   ├── mode_manager.rs  # App/HUD Modus-Wechsel mit Animation
│   │   └── worker_manager.rs # Python Worker Prozess-Management
│   ├── capabilities/        # Tauri 2.0 Permissions
│   │   └── default.json     # Erlaubte Commands
│   ├── gen/                 # Auto-generierte Permission-Dateien
│   ├── Cargo.toml           # Rust Dependencies
│   └── tauri.conf.json      # Tauri Konfiguration
├── src/
│   ├── index.html           # App-Modus Hauptseite
│   ├── hud.html             # HUD-Modus Overlay
│   └── mobile.html          # Mobile Version
└── package.json
```

## Rust Module

### lib.rs - Hauptmodul

Definiert alle Tauri Commands und den Application State:

```rust
struct AppState {
    worker_manager: Mutex<WorkerManager>,  // Python Worker
    mode_manager: Mutex<ModeManager>,      // App/HUD Modus
    hud_settings: Mutex<HudSettings>,      // HUD Konfiguration
}
```

**Wichtige Commands:**

| Command | Beschreibung |
|---------|-------------|
| `start_worker` | Startet den Python Worker Prozess |
| `stop_worker` | Stoppt den Python Worker |
| `get_worker_status` | Status des Workers (running, endpoint, etc.) |
| `switch_mode` | Wechselt zwischen App und HUD Modus |
| `save_hud_position` | Speichert HUD Position/Größe |
| `set_hud_opacity` | Setzt HUD Transparenz |
| `get_hud_functions` | Lädt verfügbare HUD Widgets vom Backend |
| `call_hud_function` | Ruft eine HUD Widget-Funktion auf |

### hud_settings.rs - HUD Konfiguration

Verwaltet persistente HUD-Einstellungen:

```rust
pub struct HudSettings {
    pub x: i32,                    // X Position
    pub y: i32,                    // Y Position
    pub width: u32,                // Breite
    pub height: u32,               // Höhe
    pub opacity: f32,              // Transparenz (0.1-1.0)
    pub animation_steps: u32,      // Animationsschritte (5-50)
    pub animation_delay_ms: u32,   // Animationsverzögerung (5-50ms)
    pub selected_miniui_app: Option<String>,  // Ausgewählte MiniUI App
    pub saved_app_state: Option<SavedAppState>, // Gespeicherter App-Zustand
}
```

**Speicherort:** `~/.config/toolboxv2/hud_settings.json` (Linux/Mac) oder `%APPDATA%\toolboxv2\hud_settings.json` (Windows)

### mode_manager.rs - Modus-Wechsel

Verwaltet den animierten Übergang zwischen App und HUD Modus:

```rust
pub enum AppMode {
    App,  // Vollständige Anwendung
    Hud,  // Kompaktes Overlay
}
```

**Animation:**
- Sanfter Übergang mit konfigurierbaren Schritten
- Position und Größe werden interpoliert
- Fenster-Dekorationen werden ein-/ausgeblendet

### worker_manager.rs - Python Worker

Verwaltet den Python Backend-Prozess:

- Startet/Stoppt den Worker als Sidecar
- Health-Check via HTTP
- Unterstützt lokale und Remote-Endpoints

## Frontend (HTML/JS)

### index.html - App Modus

Vollständige Web-Anwendung mit:
- Navigation
- Einstellungen
- Volle Funktionalität

### hud.html - HUD Modus

Kompaktes Overlay mit:
- Dynamischen Widgets
- Live-Suche
- Transparenz-Einstellungen
- WebSocket-Verbindung für Echtzeit-Updates

**HUD JavaScript API:**

```javascript
// Widget-Action senden
HUD.action('widget_id', 'action_name', {payload: 'data'});

// Widget aktualisieren
HUD.refresh('widget_id');

// Notification anzeigen
HUD.notify('Nachricht', 'success', 3000);

// In Clipboard kopieren
HUD.copy('text', 'Kopiert!');
```

## Tauri 2.0 Permissions

Alle Commands müssen in `capabilities/default.json` erlaubt werden:

```json
{
  "permissions": [
    "core:default",
    "commands:default",
    "shell:allow-open"
  ]
}
```

Custom Commands werden in `gen/commands.toml` definiert:

```toml
[commands]
allow = [
    "start_worker",
    "stop_worker",
    "switch_mode",
    ...
]
```

## Build & Development

### Development

```bash
cd toolboxv2/simple-core
npm run tauri dev
```

### Production Build

```bash
cd toolboxv2/simple-core
npm run tauri build
```

### Rust Tests

```bash
cd toolboxv2/simple-core/src-tauri
cargo test
```

## Kommunikation

```
┌─────────────────┐     IPC      ┌─────────────────┐
│   Frontend      │◄────────────►│   Rust Backend  │
│   (WebView)     │              │   (Tauri)       │
└─────────────────┘              └────────┬────────┘
                                          │
                                          │ HTTP/WS
                                          ▼
                                 ┌─────────────────┐
                                 │  Python Worker  │
                                 │  (ToolBoxV2)    │
                                 └─────────────────┘
```

## Konfigurationsdateien

| Datei | Beschreibung |
|-------|-------------|
| `tauri.conf.json` | Tauri Hauptkonfiguration |
| `Cargo.toml` | Rust Dependencies |
| `capabilities/default.json` | Erlaubte Permissions |
| `gen/commands.toml` | Custom Command Permissions |

