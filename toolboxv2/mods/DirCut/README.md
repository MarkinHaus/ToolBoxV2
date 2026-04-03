# 🎬 DirCut - Director's Cut Pipeline

**Version:** 0.1.0  
**Status:** MVP - Grundstruktur implementiert

---

## Übersicht

DirCut ist ein zweistufiges KI-gestütztes System zur Transformation von Story-Input in strukturierte, konsistente Director's Cut Prompts für Bild- und Video-Generatoren.

### Zweistufiger Workflow

1. **Phase 1: Story Consultation** - Interaktive Story-Analyse und Optimierung
2. **Phase 2: Director's Cut Generation** - Sequenzielle Szenen-Generierung mit Konsistenz-Checks

---

## Features (MVP)

### ✅ Implementiert

- **Modul-Grundstruktur**
  - Pydantic Datenmodelle für Story, Szenen, Templates
  - Tools-Klasse mit MainTool Integration
  - Projekt-Management (in-memory)

- **Web UI**
  - 3-Tab Interface (Phase 1, Phase 2, Export)
  - Story Input & Parameter-Formular
  - Style Template Auswahl
  - Responsive Design

- **API Endpoints**
  - `GET /api/DirCut/version` - Modul-Version
  - `GET /api/DirCut/list_templates` - Style Templates auflisten
  - `POST /api/DirCut/create_project` - Neues Projekt erstellen
  - `GET /api/DirCut/get_project` - Projekt abrufen
  - `GET /api/DirCut/ui` - Web UI

- **Style Templates**
  - Viral TikTok Energy
  - YouTube Documentary
  - Cinematic Short Film
  - Instagram Story Flow
  - Paranormal Mystery

### 🚧 In Entwicklung

- ISAA Agent Integration
- Story-Analyse mit KI
- Sequenzielle Szenen-Generierung
- Konsistenz-Tracking
- Export-Funktionen

---

## Verwendung

### 1. Modul starten

Das Modul wird automatisch beim Start von ToolBoxV2 geladen, wenn es in der Konfiguration aktiviert ist.

### 2. Web UI öffnen

Navigiere zu: `/api/DirCut/ui`

### 3. Projekt erstellen

1. **Phase 1 Tab öffnen**
2. Projekt-Titel eingeben
3. Story-Text eingeben
4. Style Template auswählen
5. "Projekt erstellen & Analysieren" klicken

### 4. Director's Cut generieren (Coming Soon)

Nach Projekt-Erstellung zu Phase 2 wechseln und "Director's Cut generieren" klicken.

---

## API Beispiele

### Projekt erstellen

```javascript
const response = await TB.api.request('DirCut', 'create_project', {
    title: "Meine Story",
    story_text: "Es war einmal...",
    template: "cinematic"
});

const projectId = response.get().project_id;
```

### Templates auflisten

```javascript
const response = await TB.api.request('DirCut', 'list_templates');
const templates = response.get().templates;
```

### Projekt abrufen

```javascript
const response = await TB.api.request('DirCut', 'get_project', {
    project_id: "abc123"
});

const project = response.get();
```

---

## Datenmodelle

### StoryParameters

```python
{
    "total_scenes": 8,
    "total_duration": "2-3 Minuten",
    "target_platform": "Universal",
    "primary_emotion": "Joy",
    "secondary_emotions": [],
    "art_style": "Cinematic",
    "pacing": "medium",
    "approved": false
}
```

### SceneStructure

```python
{
    "scene_number": 1,
    "scene_title": "Opening",
    "duration": "15-20 Sekunden",
    "what_happens": "...",
    "camera_type": "Medium Shot",
    "lighting_setup": "Natural",
    "art_style": "Cinematic",
    "setting": "...",
    "characters": [],
    "image_prompt": "...",
    "video_prompt": "...",
    "audio_prompt": "..."
}
```

---

## Architektur

```
DirCut/
├── module.py           # Haupt-Modul mit Tools-Klasse und API
├── specs.md            # Vollständige Spezifikation
├── task.md             # Task-Driven Development Tracking
├── README.md           # Diese Datei
└── __init__.py         # Modul-Initialisierung
```

### Komponenten

- **Tools Class** - Modul-Logik und State Management
- **Pydantic Models** - Typsichere Datenstrukturen
- **API Endpoints** - REST API für Frontend
- **Web UI** - Interaktive Benutzeroberfläche
- **Style Templates** - Vordefinierte Stil-Vorlagen

---

## Nächste Schritte

1. **ISAA Agent Integration**
   - Story Consultation Agent konfigurieren
   - Director's Cut Generator Agent konfigurieren
   - Custom Tools für Agents erstellen

2. **Story-Analyse**
   - Emotional Trigger Detection
   - Viral Potential Scoring
   - Optimization Suggestions

3. **Szenen-Generierung**
   - Sequenzielle Generation (Szene für Szene)
   - Konsistenz-Tracking (Charaktere, Orte, Zeit)
   - Guided Generation für optimalen Output

4. **Export-Funktionen**
   - Markdown Export
   - JSON Export
   - Generator-Ready Prompts
   - Shot List CSV

---

## Entwicklungs-Prinzipien

1. **Einfachheit vor Komplexität** - Minimale Lösungen bevorzugen
2. **Übersichtlicher Code** - Gut wartbar und lesbar
3. **Sequenzielle Generierung** - Szenen Schritt für Schritt erstellen
4. **Guided Generation** - LLM nicht überfordern
5. **Konsistenz-Tracking** - Charaktere, Orte, Zeitlinien tracken

---

## Lizenz

Teil des ToolBoxV2 Frameworks

---

**Letzte Aktualisierung:** 2025-11-15  
**Entwickler:** Task-Driven Development mit AI Assistant

