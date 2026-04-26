# WebAgentToolkit

Playwright-basierter Browser-Agent. Gibt dir vollständige Web-Kontrolle:
navigieren, klicken, tippen, scrapen, suchen, Session speichern.

---

## Architektur

```
WebAgentToolkit
├── ManagedBrowser          ← Browser-Lifecycle (in-process oder subprocess)
│   └── WebAgent            ← Playwright-Wrapper (direkte Page-Ops)
│       └── SearXNGClient   ← Meta-Suche
└── Tools []                ← Alle Operationen als aufrufbare Funktionen
```

**Wichtig für Planung:**
- Der Browser hat **eine aktive Page**. Alle Tools operieren auf dieser Page.
- `goto()` ändert die aktive Page — danach zeigen `extract`, `click` etc. auf die neue URL.
- Session-State (Cookies, LocalStorage) überlebt zwischen `goto()`-Calls.
- `out_of_process=True` → Browser läuft in separatem Prozess (kein Event-Loop-Konflikt mit dem Agent).

---

## Setup

```python
from toolboxv2.mods.isaa.extras.web_helper.tooklit import WebAgentToolkit

toolkit = WebAgentToolkit(
    headless=True,           # False = sichtbarer Browser (Debug)
    out_of_process=False,    # True = Browser in eigenem Prozess
    auto_start=True,         # Browser startet beim ersten Tool-Call
    keep_open=True,          # Browser bleibt zwischen Calls offen
    verbose=False,           # Interne Logs
)

# Tools an Agent übergeben
tools = toolkit.get_tools()
agent.add_tools(tools)

# Oder bestimmte Kategorien:
from toolboxv2.mods.isaa.extras.web_helper.tooklit import ToolCategory
tools = toolkit.get_tools(categories=[ToolCategory.SEARCH, ToolCategory.EXTRACTION])
```

**Convenience-Funktionen:**

```python
from toolboxv2.mods.isaa.extras.web_helper.tooklit import get_full_tools, get_minimal_tools

toolkit, tools = get_full_tools(headless=True, out_of_process=True)
toolkit, tools = get_minimal_tools(headless=True)   # Nur 8 Essential-Tools
```

---

## Tool-Referenz

### BROWSER CONTROL

---

#### `browser_start`
Browser starten.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `headless` | bool | `True` | `False` = sichtbarer Browser für Debugging |

**Returns:** `{"status": "started", "headless": bool, "message": str}`

**Wann nutzen:** Wenn `auto_start=False` gesetzt wurde, oder um zwischen headless/sichtbar zu wechseln.

---

#### `browser_stop`
Browser stoppen und Ressourcen freigeben.

**Returns:** `{"status": "stopped", "message": str}`

---

#### `browser_status`
Aktuellen Browser-Zustand abfragen.

**Returns:** `{"running": bool, "headless": bool, "auto_start": bool, "keep_open": bool}`

---

#### `browser_set_headless`
Zwischen sichtbar/unsichtbar wechseln. Erfordert Browser-Neustart.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `headless` | bool | ✓ | `True` = unsichtbar, `False` = sichtbar |

**Returns:** `{"headless": bool, "message": str}`

---

### SEARCH

---

#### `web_search`
Web-Suche via SearXNG mit Google Dorks Support.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `query` | str | required | Suchbegriff |
| `site` | str | `""` | Nur diese Domain, z.B. `"github.com"` |
| `filetype` | str | `""` | Nur dieser Dateityp, z.B. `"pdf"`, `"md"` |
| `inurl` | str | `""` | URL muss diesen String enthalten |
| `intitle` | str | `""` | Titel muss diesen String enthalten |
| `exclude` | str | `""` | Ausschließen, komma-separiert, z.B. `"beginner,tutorial"` |
| `max_results` | int | `10` | Anzahl Ergebnisse |
| `engines` | str | `""` | Engines komma-separiert, z.B. `"google,bing,duckduckgo"` |

**Returns:**
```json
{
  "query": "Python async site:github.com",
  "total": 10,
  "engines": ["google", "bing"],
  "results": [
    {"title": "...", "url": "https://...", "snippet": "...", "engine": "google"}
  ]
}
```

**Beispiele:**
```
web_search(query="Python async")
web_search(query="REST API docs", site="github.com")
web_search(query="machine learning paper", filetype="pdf")
web_search(query="FastAPI", exclude="tutorial,beginner", max_results=5)
```

---

#### `search_site`
Suche innerhalb einer bestimmten Website.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `site` | str | required | Domain, z.B. `"docs.python.org"` |
| `query` | str | `""` | Suchbegriff |
| `max_results` | int | `10` | Anzahl Ergebnisse |

**Returns:** Identisch mit `web_search`.

---

#### `search_files`
Suche nach bestimmten Dateitypen.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `filetype` | str | ✓ | `pdf`, `doc`, `xls`, `ppt`, `md`, `txt`, ... |
| `query` | str | ✓ | Suchbegriff |
| `max_results` | int | — | Default: 10 |

**Returns:** Identisch mit `web_search`.

---

### NAVIGATION

---

#### `goto`
Zu URL navigieren und warten bis geladen.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `url` | str | required | Ziel-URL |
| `wait_until` | str | `"networkidle"` | `"load"`, `"domcontentloaded"`, `"networkidle"` |

**Returns:** `{"url": "https://...", "title": "Seitentitel"}`

**Hinweis:** `"networkidle"` wartet bis keine Netzwerk-Requests mehr aktiv sind — sicher aber langsamer. Für schnelle Seiten reicht `"domcontentloaded"`.

---

#### `back`
Zur vorherigen Seite zurück (basiert auf internem History-Stack).

**Returns:** `{"url": "https://..."}`

---

#### `refresh`
Aktuelle Seite neu laden.

**Returns:** `{"status": "refreshed", "url": "https://..."}`

---

#### `current_url`
Aktuelle URL und Titel abfragen ohne Navigation.

**Returns:** `{"url": "https://...", "title": "..."}`

---

### INTERACTION

---

#### `click`
Element klicken.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `selector` | str | required | CSS Selector |
| `wait_after` | float | `0.5` | Wartezeit in Sekunden nach dem Klick |

**Returns:** `{"status": "clicked", "selector": "..."}`

**Selector-Beispiele:**
```
"#submit-button"
".btn-primary"
"button[type='submit']"
"a[href*='login']"
"nav > ul > li:first-child > a"
```

---

#### `type`
Text in Eingabefeld eingeben.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `selector` | str | required | CSS Selector des Eingabefelds |
| `text` | str | required | Einzugebender Text |
| `clear` | bool | `True` | Feld vorher leeren |

**Returns:** `{"status": "typed", "selector": "...", "length": int}`

---

#### `select`
Option in Dropdown auswählen. Eines der drei Identifikations-Parameter angeben.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `selector` | str | required | CSS Selector des `<select>` |
| `value` | str | `""` | Option-`value`-Attribut |
| `label` | str | `""` | Sichtbarer Text der Option |
| `index` | int | `-1` | Null-basierter Index |

**Returns:** `{"status": "selected", "selector": "..."}`

---

#### `scroll`
Seite scrollen.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `direction` | str | `"down"` | `"down"` oder `"up"` |
| `amount` | int | `500` | Pixel |

**Returns:** `{"status": "scrolled", "direction": "...", "amount": int}`

---

#### `scroll_to_bottom`
Zum Seitenende scrollen. Löst Lazy-Loading aus (Social Media, Infinite Scroll).

**Returns:** `{"status": "scrolled_to_bottom"}`

---

#### `wait`
Auf Element warten oder bis Seite vollständig geladen.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `selector` | str | `""` | CSS Selector — leer = auf `networkidle` warten |
| `timeout` | int | `30` | Max Wartezeit in Sekunden |

**Returns:** `{"status": "ready", "selector": "..."}`

---

#### `hover`
Maus über Element bewegen. Für Tooltips und Dropdown-Menüs.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `selector` | str | ✓ | CSS Selector |

**Returns:** `{"status": "hovered", "selector": "..."}`

---

### EXTRACTION

---

#### `extract`
Aktuellen Seiteninhalt als strukturiertes Markdown extrahieren. **Haupt-Scraping-Tool.**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `include_links` | bool | `True` | Links mit extrahieren (max 100) |
| `include_headings` | bool | `True` | Überschriften-Struktur mit extrahieren |

**Returns:**
```json
{
  "url": "https://...",
  "title": "Seitentitel",
  "markdown": "# Überschrift\n\nText...",
  "meta": {"description": "...", "lang": "de"},
  "headings": [{"level": 1, "text": "Überschrift"}],
  "links": [{"text": "Linktext", "href": "https://..."}]
}
```

**Hinweis:** Entfernt automatisch Navigation, Sidebar, Ads, Scripts. Extrahiert Hauptinhalt.

---

#### `extract_text`
Reinen Text aus einem Element extrahieren.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `selector` | str | `"body"` | CSS Selector |

**Returns:** `{"text": "...", "length": int}`

---

#### `extract_html`
Rohes HTML aus einem Element extrahieren.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `selector` | str | `"body"` | CSS Selector |

**Returns:** `{"html": "...", "length": int}`

---

#### `extract_links`
Alle Links von der aktuellen Seite extrahieren.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `filter_pattern` | str | `""` | Regex-Pattern für URL oder Linktext |

**Returns:**
```json
{
  "total": 42,
  "links": [{"text": "GitHub", "href": "https://github.com/..."}]
}
```

---

#### `extract_attribute`
Einzelnes HTML-Attribut aus einem Element extrahieren.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `selector` | str | ✓ | CSS Selector |
| `attribute` | str | ✓ | Attributname: `href`, `src`, `data-id`, `value`, ... |

**Returns:** `{"attribute": "href", "value": "https://..."}`

---

#### `scrape_url`
URL öffnen und Inhalt in einem Schritt extrahieren. Kombination aus `goto` + `extract`.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `url` | str | ✓ | Ziel-URL |

**Returns:** Identisch mit `extract`.

---

### SESSION

---

#### `session_save`
Browser-Session speichern (Cookies, LocalStorage, IndexedDB).

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `name` | str | ✓ | Session-Name (ohne Extension) |

**Returns:** `{"status": "saved", "name": "...", "path": "/path/to/session.json"}`

---

#### `session_load`
Gespeicherte Session laden. Stellt Login-Zustand wieder her.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `name` | str | ✓ | Session-Name |

**Returns:** `{"status": "loaded"|"failed", "name": "...", "success": bool}`

---

#### `session_list`
Alle gespeicherten Sessions auflisten.

**Returns:** `{"sessions": ["session_a", "github_session", ...]}`

---

#### `login`
Automatischen Login durchführen und optional Session speichern.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `url` | str | required | Login-URL |
| `username_selector` | str | required | CSS Selector für Username-Feld |
| `password_selector` | str | required | CSS Selector für Passwort-Feld |
| `submit_selector` | str | required | CSS Selector für Submit-Button |
| `username` | str | required | Benutzername |
| `password` | str | required | Passwort |
| `success_indicator` | str | `""` | CSS Selector der nach Login erscheint (Verifikation) |
| `save_as` | str | `""` | Session-Name zum Speichern (leer = nicht speichern) |

**Returns:** `{"success": bool, "url": "...", "session_saved"?: "name"}`

---

### UTILITY

---

#### `screenshot`
Screenshot der aktuellen Seite machen.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `name` | str | `""` | Dateiname ohne `.png` (leer = Timestamp) |
| `full_page` | bool | `False` | `True` = gesamte Seite (auch nicht sichtbarer Teil) |

**Returns:** `{"path": "/path/to/screenshot.png"}`

---

#### `execute_js`
JavaScript auf der aktuellen Seite ausführen.

| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| `script` | str | ✓ | JavaScript-Code |

**Returns:** `{"result": <Rückgabewert des Scripts>}`

**Beispiele:**
```javascript
// DOM abfragen
"return document.querySelector('h1').textContent"

// Alle Links extrahieren
"return Array.from(document.links).map(a => a.href)"

// Seiten-Metadaten
"return {title: document.title, url: location.href}"

// Element-Existenz prüfen
"return !!document.querySelector('#login-form')"
```

---

#### `get_logs`
Agent-Logs und Session-Statistiken abfragen.

**Returns:** Log-Summary-Dict mit Timings, Error-Counts, etc.

---

## Workflows

### Einfaches Scraping
```
scrape_url(url="https://example.com")
```

### Login + Session
```
1. login(url, username_sel, password_sel, submit_sel, user, pass,
         success_indicator=".dashboard", save_as="my_session")
2. goto(url="https://app.example.com/data")
3. extract()
```

**Nächste Session:**
```
1. session_load(name="my_session")
2. goto(url="https://app.example.com/data")
3. extract()
```

### Suchen + Top-Ergebnisse scrapen
```
1. web_search(query="FastAPI tutorial", max_results=3)
2. scrape_url(url=results[0].url)
3. scrape_url(url=results[1].url)
```

### Formular ausfüllen
```
1. goto(url="https://example.com/form")
2. type(selector="#name", text="Max Mustermann")
3. type(selector="#email", text="max@example.com")
4. select(selector="#country", label="Germany")
5. click(selector="button[type='submit']")
6. wait(selector=".success-message")
7. extract_text(selector=".success-message")
```

### Lazy-Load Seite vollständig laden
```
1. goto(url="https://social.example.com/feed")
2. scroll_to_bottom()
3. extract()
```

### Dynamischen Inhalt abfragen
```
1. goto(url="https://app.example.com")
2. wait(selector=".data-loaded", timeout=10)
3. execute_js(script="return window.__APP_STATE__")
```

---

## Tool-Kategorien

| Kategorie | Tools | Flags |
|-----------|-------|-------|
| `browser` | `browser_start`, `browser_stop`, `browser_status`, `browser_set_headless` | `no_thread: True` |
| `search` | `web_search`, `search_site`, `search_files` | `read: True` |
| `navigation` | `goto`, `back`, `refresh`, `current_url` | `read+write` |
| `interaction` | `click`, `type`, `select`, `scroll`, `scroll_to_bottom`, `wait`, `hover` | `write: True` |
| `extraction` | `extract`, `extract_text`, `extract_html`, `extract_links`, `extract_attribute`, `scrape_url` | `read: True` |
| `session` | `session_save`, `session_load`, `session_list`, `login` | `write: True` |
| `utility` | `screenshot`, `execute_js`, `get_logs` | `no_thread: True` |

---

## Bekannte Grenzen

- **Eine Page:** Kein Multi-Tab. Parallele Navigation nicht möglich.
- **execute_js** hat Zugriff auf den vollen DOM, aber kein Netzwerk (kein `fetch` aus dem Script heraus).
- **session_load** erstellt einen neuen Browser-Context — Event-Listener auf der alten Page gehen verloren.
- **SearXNG:** Wenn keine eigene Instanz konfiguriert, wird eine Public-Instanz genutzt (Rate-Limits möglich).
- **Timeouts:** `goto` mit `wait_until="networkidle"` kann auf SPAs hängen — dann `"domcontentloaded"` + explizites `wait(selector=...)` nutzen.
