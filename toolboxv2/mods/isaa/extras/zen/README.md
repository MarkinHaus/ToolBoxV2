

# 🌌 Zen System (Zen+ & ZenRendererV2)

Das Zen System ist das fortschrittliche Terminal-User-Interface (TUI) und Rendering-Framework für die ISAA-Architektur. Es verwandelt komplexe Multi-Agenten-Ausführungen, Hintergrund-Jobs und Tool-Aufrufe in eine interaktive, übersichtliche und visuell ansprechende Terminal-Erfahrung.

## 🏗 Architektur

Das System besteht aus zwei Hauptkomponenten, die nahtlos ineinandergreifen:

1. **`ZenRendererV2` (Der Orchestrator)**
   * Liest den `engine.live`-Status in Echtzeit.
   * Modus 1: Einzeilige, minimierbare CLI-Ausgabe (perfekt, um den Agenten ungestört im Hintergrund laufen zu lassen).
   * Modus 2: Vollständiges Debug-Dashboard (`ENGINE_DEBUG=1`).
   * Leitet bei Bedarf alle Events (Chunks) an *Zen+* weiter.

2. **`ZenPlus` (Das Fullscreen TUI)**
   * Eine `prompt_toolkit`-basierte Vollbild-Anwendung.
   * Empfängt Chunks asynchron über eine Thread-safe Queue.
   * Bietet 3 Navigationsebenen, Syntax-Highlighting und eine Physik-basierte 3D-Graphen-Engine.

---

## ✨ Haupt-Features (Phase 3 Abgeschlossen)

* **3D Force-Directed ASCII Graphen (`MiniGraph3D`)**
  * Auto-rotierende, Z-gepufferte 3D-Visualisierung der Agenten-Gedankengänge und Tool-Aufrufe.
  * *Sub-Agenten* werden als funkelnde Sterne (✦) mit eigenen Farbzyklen dargestellt.
* **Globaler Multi-Agent Graph (`GlobalGraph`)**
  * Zeigt netzwerkartige Verbindungen zwischen *allen* aktiven Agenten.
  * Visualisiert geteilte Ressourcen (z. B. wenn zwei Agenten dieselbe Datei bearbeiten), Sub-Agenten-Hierarchien und parallele Aufgaben.
* **Skill & Dream Visualisierung (`DreamZenAdapter`)**
  * Integriert den `DreamGraphV2` direkt in Zen+.
  * Rendert den **Pipeline Tree** und den **Skill Diamond** inklusive Bloat-Warnungen und Confidence-Sparklines nahtlos im Terminal.
* **Absturzsicheres Detail-Tracking**
  * Gedankengänge (Thoughts) und Tool-Ergebnisse werden **nie abgeschnitten** (kein Truncation in der Detail-Ansicht).
  * Integriertes Syntax-Highlighting für JSON, Markdown und Konfigurationsdateien in Echtzeit.
* **Job- & Hintergrund-Task-Management**
  * Dedizierte Task-Leiste für injizierte Background-Jobs, Delegationen und System-Tasks.

---

## 🗺 Navigation: Die 3 Ebenen (UI Layers)

Die Steuerung in Zen+ ist fließend und erfordert keine Maus. Sie ist in drei Informationsebenen unterteilt:

### 1. GRID View (Übersicht)
Zeigt alle aktiven Agenten als Kacheln an. Ideal, um parallele Prozesse zu überwachen.
* `Tab` / `Shift+Tab` / `↑↓←→`: Agenten-Kachel auswählen
* `Enter`: Zum ausgewählten Agenten in die Focus-Ansicht wechseln
* `G`: Den globalen Cross-Agent Graphen öffnen
* `Esc`: Zen+ beenden (fällt auf ZenRendererV2 zurück)

### 2. FOCUS View (Der Agent im Detail)
Zeigt den aktuellen Status, Token-Verbrauch, Iterationen und laufende Tools eines spezifischen Agenten.
* `↑↓`: Durch den Konsolen-Output scrollen
* `g`: 3D-Graphen (MiniGraph3D) des Agenten öffnen
* `t`: Tool-Historie öffnen
* `i`: Iterations-Übersicht öffnen
* `h`: Gedanken-Historie (Thoughts) öffnen
* `Esc`: Zurück zur GRID View

### 3. DETAIL View (Deep-Dive)
Vollbild-Inspektion der ausgewählten Daten.
* `↑↓`: Durch Listen (Tools, Iterationen, Thoughts) navigieren
* `Enter`: Eintrag erweitern (z. B. komplettes JSON-Ergebnis eines Tools anzeigen) oder im Graph zu Nodes springen.
* `d`: *Spezial-Hotkey* (wenn aktiv) – Öffnet die Dream-Visualisierung (Tree/Diamond Toggle).
* `Esc`: Zurück zur FOCUS View

---

## 🚀 Demos & Quick Start

Um die Leistungsfähigkeit von Zen+ direkt zu testen, ist eine Demo-Suite integriert:

```bash
# Demo 1: Einzelner Agent mit Sub-Agenten und 3D-Graph
python demo_zen_plus.py single

# Demo 2: Drei parallel arbeitende Agenten + Hintergrund-Jobs
python demo_zen_plus.py multi

# Demo 3: Reiner Job-Scheduler (Fokus auf Background-Task-Leiste)
python demo_zen_plus.py jobs
```

### Integration in eigenen Code

**ZenRenderer aktivieren:**
```python
from toolboxv2.mods.isaa.extras.zen.zen_renderer import ZenRendererV2
renderer = ZenRendererV2(engine)
```

**Zen+ anbinden:**
```python
from toolboxv2.mods.isaa.extras.zen.zen_plus import ZenPlus
zp = ZenPlus.get()
renderer.set_zen_plus(zp)

# Startet die UI (blockiert bis Esc gedrückt wird)
await zp.start()
```

**Dream Adapter (Skill Diamond) aktivieren:**
```python
from toolboxv2.mods.isaa.extras.zen.dream_zen_adapter import patch_zen_dream
# Einmalig beim Start aufrufen, schaltet die 'd'-Taste im Focus Mode frei
patch_zen_dream()
```

---

## 🧪 Tests & Stabilität

Zen+ ist für den produktiven Einsatz in asynchronen Umgebungen konzipiert (Thread-safe Queues, IDD / Interface-Driven Design).
Führe die Phase 3 Stabilitäts-Suite aus, um die Integrität der Syntax-Highlighter, Graph-Physik und Status-Manager zu validieren:

```bash
python -m unittest test_zen_plus.py -v
```
