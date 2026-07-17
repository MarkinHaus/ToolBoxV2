# Navbar-Konzepte — 3 Varianten

> Ziel: Übersichtlich, nicht überfordend, alle Docs erreichbar.
> Basierend auf [Diátaxis](https://diataxis.fr) + aktueller `docs/` Struktur.

---

## Konzept A: Themen-Cluster (Empfohlen ✅)

**Prinzip:** Gruppiert nach Systembereichen. 6 Top-Level-Kategorien, max 2 Ebenen tief.

```
ToolBoxV2 Docs
├── 📦 Loslegen
│   ├── Installation
│   ├── First Run Wizard
│   └── Onboarding
├── 🧠 ISAA · Agents
│   ├── Overview
│   ├── AgentBuilder
│   ├── ToolManager
│   ├── Sessions
│   └── Hybrid Memory
├── ☁️ CloudM
│   ├── Overview
│   ├── Auth System
│   ├── User Data API
│   ├── Mod Manager
│   ├── FolderSync (deprecated)
│   ├── LiveSync (neu)
│   └── Sub-Module (8)
├── ⚙️ Runtime
│   ├── HTTP Worker
│   ├── FastTB API
│   ├── FastTBHandler
│   ├── WebSocket Worker
│   ├── Event Manager
│   ├── Session Mgmt
│   ├── Config
│   └── Debug Runner
├── 💾 Storage
│   ├── Overview (DB Modes)
│   ├── BlobDB Reference
│   └── Blob Sharing API
└── 🔧 Core Internals
    ├── DevDocs Index
    ├── Core Types (AppType, Result)
    ├── FileHandlerV2
    ├── Crypto (Code class)
    ├── Enums (auto-generated)
    ├── RegistryClient
    ├── WorkerManager
    ├── WSWorker
    ├── Toolbox Integration
    ├── Style & Terminal
    ├── Notifications
    ├── P2P CLI
    ├── DB CLI Manager
    └── User Manager
```

**Vorteile:**
- Max 2 Klicks zu jedem Doc
- Klar getrennt: User-Facing (oben) vs Internal (unten)
- Skaliert: Neue Mods einfach unter CloudM/ISAA hinzufügbar
- Entspricht mentaler Map der Entwickler

**Nachteile:**
- "Core Internals" wird groß (14 Einträge)
- Abhilfe: DevDocs eigene Index-Seite als Hub

**MkDocs `nav:` Config:**
```yaml
nav:
  - Loslegen: foundations/
  - ISAA: mods/isaa/
  - CloudM: mods/CloudM/
  - Runtime: runtime/
  - Storage: storage/
  - Core Internals: devdocs/
  - Flows: flows/
  - Services: services/
```

---

## Konzept B: User-Journey (Lernpfad)

**Prinzip:** Führt den Nutzer von "Ich kenne nichts" → "Ich baue Mods". 
Diátaxis-konform: Tutorial → How-to → Reference → Explanation.

```
ToolBoxV2 Docs
├── 🚀 Start (Tutorials)
│   ├── Installation
│   ├── First Run
│   └── Dein erstes Mod
├── 🛠️ Bauen (How-to Guides)
│   ├── Mod erstellen
│   ├── Worker konfigurieren
│   ├── Auth einrichten
│   ├── MinIO/Storage setup
│   └── P2P Chat nutzen
├── 📚 Referenz (API Docs)
│   ├── AppType & Result
│   ├── FileHandlerV2
│   ├── Crypto
│   ├── ISAA API
│   ├── CloudM API
│   ├── Worker API
│   └── Storage API
├── 🔬 Deep Dive (Explanation)
│   ├── Architektur
│   ├── Dispatch System
│   ├── Worker/Nginx Setup
│   └── Security Modell
└── 📋 CLI & Tools
    ├── tb Befehle
    ├── DevDocs Index
    └── Utils Analysis
```

**Vorteile:**
- Diátaxis-konform
- Perfekt für Onboarding neuer Entwickler
- Natürliche Lern-Reihenfolge

**Nachteile:**
- "Referenz" wird sehr groß
- Erfahrene Entwickler müssen erst suchen
- Schwerer abzubilden in MkDocs (cross-cutting concerns)

---

## Konzept C: Hub & Spoke (Landing-Page zentriert)

**Prinzip:** `index.md` ist das einzige Zentrum. Große Kacheln mit klaren Fragen. 
Sidebar bleibt flach, jede Kachel führt zu einem Sub-Index.

```
[index.md — Landing Page]
┌──────────────┬──────────────┐
│ "Ich will    │ "Ich will    │
│  starten"    │  Agents      │
│ → foundations│ → mods/isaa  │
├──────────────┼──────────────┤
│ "Ich will    │ "Ich will    │
│  CloudM      │  Storage     │
│  nutzen"     │  verstehen"  │
│ → mods/CloudM│ → storage    │
├──────────────┼──────────────┤
│ "Ich will    │ "Ich will    │
│  deployen"   │  debuggen"   │
│ → runtime    │ → devdocs    │
├──────────────┴──────────────┤
│ "Ich will CLIs nutzen"      │
│ → services + flows          │
└─────────────────────────────┘

Sidebar (flach):
├── Home
├── Foundations
├── ISAA
├── CloudM
├── Runtime
├── Storage
├── Core Internals
├── Flows
└── Services
```

**Vorteile:**
- Minimalste Sidebar (9 Einträge)
- Landing Page ist "Task-Oriented" (Was willst du tun?)
- Jeder Sub-Index ist eigenständiger Hub
- Am besten für "nicht überfordert"

**Nachteile:**
- 3 Klicks zu spezifischen Docs (Hub → Sub-Index → Doc)
- Erfordert gute Sub-Index-Seiten
- Landing Page muss gepflegt werden

---

## Empfehlung

| Kriterium | A (Cluster) | B (Journey) | C (Hub) |
|-----------|:-----------:|:-----------:|:-------:|
| Übersichtlichkeit | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Nicht überfordert | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Alle Docs erreichbar | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Onboarding-freundlich | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| erfahrene Devs | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| MkDocs-Umsetzung | Einfach | Schwer | Mittel |

**Winner: Konzept A (Themen-Cluster)** mit Elementen aus C (gute Landing Page).

Begründung: Beste Balance aus Übersicht und Tiefe. Max 2 Klicks. Klare Trennung User-Facing vs Internal. MkDocs `nav:` ist straightforward.
