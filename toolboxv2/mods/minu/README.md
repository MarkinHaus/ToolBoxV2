# Minu UI Framework für Toolbox V2

Ein leichtgewichtiges, reaktives UI-Framework für das Toolbox-System. Minu ermöglicht die einfache Erstellung von Benutzeroberflächen direkt aus Python-Code mit automatischen Live-Updates über WebSocket.

## 🎯 Design-Philosophie

1. **Einfacher Python-Code** - UI wird als Python-Objekte definiert
2. **Reaktiver State** - Änderungen triggern automatisch UI-Updates
3. **Minimale Payloads** - Nur Diffs werden über WebSocket gesendet
4. **Native Toolbox-Integration** - Volle Kompatibilität mit Result, Export, etc.
5. **TBJS-kompatibel** - Nutzt das vorhandene CSS Design System

## 📁 Projektstruktur

```
minu/
├── __init__.py       # Toolbox-Modul mit @export Endpoints
├── core.py           # Kern-Framework (Components, State, Views)
├── flows.py          # Hilfsfunktionen für Flow-basierte UI
├── examples.py       # Beispiel-Implementierungen
../tbjs/src/ui/components/Minu/MinuRenderer.js # TBJS Frontend-Renderer
```

## 🚀 Schnellstart

### 1. Installation

Kopiere den `minu/` Ordner in dein `toolboxv2/mods/` Verzeichnis:

```bash
cp -r minu/ /path/to/toolboxv2/mods/
```

### 2. Einfaches Beispiel

```python
from minu import (
    State, MinuView, register_view,
    Card, Text, Button, Row
)

class CounterView(MinuView):
    count = State(0)

    def render(self):
        return Card(
            Text(f"Zähler: {self.count.value}"),
            Row(
                Button("-", on_click="decrement"),
                Button("+", on_click="increment"),
            ),
            title="Mein Counter"
        )

    async def increment(self, event):
        self.count.value += 1

    async def decrement(self, event):
        self.count.value -= 1

# View registrieren
register_view("counter", CounterView)
```

### 3. Im Frontend einbinden

```html
<div id="app"></div>

<script type="module">
import { mountMinuView } from '/static/js/minu/MinuRenderer.js';

// View mounten
const renderer = await mountMinuView('#app', 'counter');
</script>
```

## 📦 Komponenten-Übersicht

### Layout-Komponenten

| Komponente | Beschreibung | CSS-Klasse |
|------------|--------------|------------|
| `Card()` | Container mit Schatten | `.card` |
| `Row()` | Horizontale Flex-Box | `.flex` |
| `Column()` | Vertikale Flex-Box | `.flex.flex-col` |
| `Grid()` | CSS Grid | `.grid` |
| `Spacer()` | Abstandshalter | `.h-{size}` |
| `Divider()` | Trennlinie | `<hr>` |

### Content-Komponenten

| Komponente | Beschreibung |
|------------|--------------|
| `Text(content)` | Textanzeige |
| `Heading(text, level)` | Überschrift h1-h6 |
| `Icon(name)` | Material Symbol |
| `Image(src)` | Bild |
| `Badge(text, variant)` | Kleine Markierung |

### Input-Komponenten

| Komponente | Beschreibung |
|------------|--------------|
| `Button(label, on_click)` | Schaltfläche |
| `Input(placeholder, bind)` | Texteingabe |
| `Select(options, bind)` | Dropdown |
| `Checkbox(label, bind)` | Checkbox |
| `Switch(label, bind)` | Toggle-Schalter |

### Feedback-Komponenten

| Komponente | Beschreibung |
|------------|--------------|
| `Alert(message, variant)` | Hinweismeldung |
| `Progress(value)` | Fortschrittsbalken |
| `Spinner()` | Ladeanimation |

### Spezial-Komponenten

| Komponente | Beschreibung |
|------------|--------------|
| `Modal(children, open)` | Dialog |
| `Widget(children, title)` | Schwebendes Fenster |
| `Form(children, on_submit)` | Formular |
| `Tabs(tabs)` | Tab-Navigation |
| `Table(columns, data)` | Datentabelle |

## 🔄 Reaktiver State

```python
from minu import State, MinuView

class MyView(MinuView):
    # State-Definitionen auf Klassen-Ebene
    name = State("")
    count = State(0)
    items = State([])

    def render(self):
        # State-Werte mit .value lesen
        return Text(f"Name: {self.name.value}")

    async def update_name(self, event):
        # State-Werte mit .value setzen -> triggert UI-Update
        self.name.value = event.get("value", "")
```

### Bindings

```python
Input(
    value=self.name.value,  # Initialer Wert
    bind="name"              # Two-Way Binding zum State
)
```

## 🌐 API Endpoints

Das Framework registriert automatisch folgende Endpoints:

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/Minu/render` | GET/POST | View initial rendern |
| `/api/Minu/event` | POST | Event an Handler senden |
| `/api/Minu/state` | POST | State aktualisieren |
| `/api/Minu/list_views` | GET | Registrierte Views auflisten |
| `/ws/Minu/ui` | WebSocket | Live-Updates |
| `/sse/Minu/stream` | GET | Server-Sent Events Alternative |

## 🔧 Flow-Integration

Für einfache, datengetriebene UIs:

```python
from minu.flows import ui_for_data, data_card, data_table, form_for

async def run(app, data):
    # Automatische UI aus Dict
    return ui_for_data({"name": "John", "score": 100})

    # Oder spezifische Komponenten
    return data_card(
        {"name": "John", "email": "john@example.com"},
        title="Benutzer",
        actions=[{"label": "Bearbeiten", "handler": "edit"}]
    )

    # Datentabelle
    return data_table(
        [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        on_row_click="select_item"
    )

    # Formular aus Schema
    return form_for({
        "name": {"type": "text", "label": "Name", "required": True},
        "role": {"type": "select", "options": [...]}
    }, on_submit="save")
```

## 🎨 CSS Integration

Minu nutzt das TBJS Design System. Alle Komponenten verwenden die vordefinierten CSS-Klassen:

```python
# Vordefinierte Varianten
Button("Speichern", variant="primary")   # .btn.btn-primary
Button("Abbrechen", variant="secondary") # .btn.btn-secondary
Alert("Erfolg!", variant="success")      # .alert.alert-success

# Eigene Klassen
Card(
    Text("Inhalt"),
    className="card animate-fade-in max-w-md"
)
```

## 📡 WebSocket-Protokoll

### Client → Server

```json
// View abonnieren
{"type": "subscribe", "viewName": "counter", "props": {}}

// Event auslösen
{"type": "event", "viewId": "...", "handler": "increment", "payload": {}}

// State aktualisieren (Two-Way Binding)
{"type": "state_update", "viewId": "...", "path": "name", "value": "John"}
```

### Server → Client

```json
// Initiales Render
{"type": "render", "sessionId": "...", "view": {...}}

// State-Patches
{
  "type": "patches",
  "patches": [
    {"type": "state_update", "viewId": "...", "path": "count", "value": 5}
  ]
}
```

## 🏗️ Architektur

```
┌──────────────────────────────────────────────────────────────┐
│                        Python Backend                         │
├────────────────┬─────────────────────┬───────────────────────┤
│   MinuView     │   ReactiveState     │    MinuSession        │
│   - render()   │   - value           │    - views            │
│   - handlers   │   - observers       │    - send_callback    │
│                │   - notify()        │    - patches          │
└───────┬────────┴──────────┬──────────┴───────────┬───────────┘
        │                   │                      │
        │     State Change  │                      │
        └─────────┬─────────┘                      │
                  │ Patches (JSON)                 │
                  ▼                                │
┌─────────────────────────────────────────────────▼────────────┐
│                     WebSocket / SSE                           │
└─────────────────────────────────────────────────┬────────────┘
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────────┐
│                      TBJS Frontend                            │
├────────────────┬─────────────────────┬───────────────────────┤
│ MinuRenderer   │   Component         │   Event Handler        │
│ - renderComp() │   Registry          │   - bindEvent()        │
│ - applyPatch() │   - renderers{}     │   - triggerEvent()     │
│                │                     │                        │
└────────────────┴─────────────────────┴───────────────────────┘
                                │
                                ▼
                           DOM Updates
```

## 📝 Vollständiges Beispiel: Dashboard

```python
from minu import *

class DashboardView(MinuView):
    users = State([])
    loading = State(False)
    selected_user = State(None)

    def render(self):
        return Column(
            # Header
            Row(
                Heading("Dashboard", level=1),
                Button("Aktualisieren", on_click="refresh", variant="primary"),
                justify="between"
            ),

            # Stats
            Grid(
                Card(
                    Icon("person"),
                    Heading(str(len(self.users.value)), level=2),
                    Text("Benutzer"),
                    className="text-center"
                ),
                cols=4
            ),

            # Loading State
            Spinner() if self.loading.value else None,

            # User Table
            Table(
                columns=[
                    {"key": "name", "label": "Name"},
                    {"key": "email", "label": "Email"}
                ],
                data=self.users.value,
                on_row_click="select_user"
            ),

            # Selected User Modal
            Modal(
                Text(f"Ausgewählt: {self.selected_user.value.get('name', '')}"),
                Button("Schließen", on_click="close_modal"),
                open=self.selected_user.value is not None,
                on_close="close_modal"
            ),

            gap="6"
        )

    async def refresh(self, event):
        self.loading.value = True
        # API-Aufruf hier
        self.users.value = await fetch_users()
        self.loading.value = False

    async def select_user(self, event):
        self.selected_user.value = event

    async def close_modal(self, event):
        self.selected_user.value = None

register_view("dashboard", DashboardView)
```

## ⚡ Performance-Tipps

1. **Debouncing**: State-Updates werden automatisch für 16ms debounced
2. **Partial Updates**: Nur geänderte State-Pfade werden gesendet
3. **Effizientes Rendering**: Frontend patcht DOM statt komplett neu zu rendern
4. **Lazy Loading**: Views werden erst bei Subscription instanziiert

## 🔒 Sicherheit

- Session-basierte Authentifizierung über Toolbox SessionManager
- WebSocket-Verbindungen erben Session-Context
- Event-Handler werden server-seitig validiert

## 📚 Weitere Ressourcen

- [Toolbox V2 Dokumentation](https://github.com/MarkinHaus/ToolBoxV2)
- [TBJS Design System](./tbjs-design-system.md)
- [API Reference](./api-reference.md)

---

**Minu UI Framework** - Einfache UIs für Toolbox V2 🐍✨
