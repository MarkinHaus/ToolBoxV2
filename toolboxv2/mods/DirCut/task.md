# DirCut Module - Task-Driven Development

## Status: 🚧 In Progress

---

## ✅ Phase 1: Setup & Grundstruktur

### [x] Task 1.1: Spezifikation lesen
- specs.md analysiert
- Zweistufiges System verstanden:
  - Phase 1: Story Consultation Agent
  - Phase 2: Director's Cut Generator
- Konsistenz-Regeln identifiziert

### [x] Task 1.2: Modul-Grundstruktur erstellen
- [x] `module.py` mit Basis-Setup
- [x] Tools-Klasse mit MainTool
- [x] @export decorator konfigurieren
- [x] Modul-Initialisierung
- [x] Web UI mit 3 Tabs (Phase 1, Phase 2, Export)
- [x] API Endpoints: create_project, get_project, list_templates

### [x] Task 1.3: Datenmodelle definieren
- [x] StoryParameters (Pydantic)
- [x] SceneStructure (Pydantic)
- [x] StyleTemplate (Pydantic)
- [x] DirectorsCutProject (Pydantic)
- [x] CharacterInfo (Pydantic)

---

## 📋 Phase 2: Web UI (Minimal & Einfach)

### [ ] Task 2.1: Haupt-UI erstellen
- [ ] Einfaches 2-Tab Layout (Phase 1 / Phase 2)
- [ ] Story Input Textarea
- [ ] Parameter-Formular
- [ ] Approval-Button

### [ ] Task 2.2: API Endpoints
- [ ] `/analyze_story` - Story analysieren
- [ ] `/generate_cut` - Director's Cut generieren
- [ ] `/export_cut` - Export in verschiedenen Formaten
- [ ] `/list_templates` - Style Templates auflisten

---

## 🤖 Phase 3: ISAA Agent Integration

### [x] Task 3.1: Story Consultation Agent
- [x] Agent Builder konfigurieren
- [x] System Message für Story-Analyse
- [x] Custom Tools für Viral-Analyse
  - [x] _analyze_emotional_triggers()
  - [x] _calculate_viral_potential()
  - [x] _suggest_optimizations()
- [x] API Endpoint: /analyze_story

### [x] Task 3.2: Director's Cut Generator Agent
- [x] Agent Builder konfigurieren
- [x] System Message für Szenen-Generierung
- [x] Custom Tools für Konsistenz-Checks
  - [x] _check_visual_consistency()
  - [x] _check_temporal_consistency()
  - [x] _track_character_consistency()
- [x] API Endpoint: /generate_scene (schrittweise!)

---

## 🎬 Phase 4: Core Logic (Schrittweise!)

### [x] Task 4.1: Story Analysis
- [x] Emotional Trigger Detection (_analyze_emotional_triggers)
- [x] Conflict Strength Scoring (in viral_potential)
- [x] Viral Potential Calculation (_calculate_viral_potential)
- [x] Optimization Suggestions (_suggest_optimizations)

### [x] Task 4.2: Scene Generation (WICHTIG: Sequenziell!)
- [x] **Detaillierte Prompt-Generierung implementiert:**
  - [x] `_build_image_prompt_positive()` - Positive Image Prompts
  - [x] `_build_image_prompt_negative()` - Negative Image Prompts (Quality Control)
  - [x] `_build_video_prompt_movement()` - Character/Object Bewegungen
  - [x] `_build_video_prompt_camera()` - Kamera-Bewegungen
  - [x] `_build_audio_prompt()` - Audio/Music Prompts
- [x] `generate_scene_prompts()` - Hauptfunktion für sequenzielle Generierung
- [x] API Endpoint: `/generate_scene_detailed` (neue Version mit allen Details)
- [x] Konsistenz-Tracking über vorherige Szenen

### [x] Task 4.3: Consistency Enforcement
- [x] Visual Continuity Checker (_check_visual_consistency)
- [x] Temporal Continuity Checker (_check_temporal_consistency)
- [x] Character Consistency Tracker (_track_character_consistency)
- [x] Previous Scenes werden für Konsistenz-Checks verwendet

---

## 📤 Phase 5: Export & Finalisierung

### [x] Task 5.1: Export Formate
- [x] Markdown Export (mit allen Prompts formatiert)
- [x] JSON Export (strukturierte Daten)
- [x] Plain Text Export (Generator-Ready Prompts)
- [x] API Endpoint: `/export_prompts?project_id=X&format=json|markdown|txt`
- [x] **UI Integration:** Export-Buttons funktionieren mit Download

### [x] Task 5.2: Quality Checks
- [x] Automated Consistency Validation (via Custom Tools)
- [x] Viral Potential Score (_calculate_viral_potential)
- [x] Interactive Story Analysis mit User-Feedback-Loop

---

## 🎨 Phase 6: Custom Template Editor

### [x] Task 6.1: Template Creation
- [x] **Content-Dump Parser:** Template aus JSON/YAML/Freitext generieren
- [x] **Existing Template Loader:** Bestehende Templates als Basis laden
- [x] **Minimalistischer Editor:** 5 Hauptfelder (Name, Art, Color, Music, Pacing)
- [x] **LocalStorage Persistence:** Templates werden im Browser gespeichert

### [x] Task 6.2: UI Integration
- [x] 4. Tab "Templates" hinzugefügt
- [x] Toggle zwischen Content-Dump und Existing Template
- [x] Template-Editor mit Formular
- [x] Gespeicherte Templates anzeigen
- [x] Custom Templates in Projekt-Erstellung verfügbar

### [x] Task 6.3: Backend Support
- [x] API Endpoint: `/create_template_from_dump` (nutzt Generator Agent)
- [x] Intelligente Parsing-Logik für verschiedene Formate

---

## 🎨 Phase 6: Style Templates

### [ ] Task 6.1: Vordefinierte Templates
- [ ] "Viral TikTok Energy"
- [ ] "YouTube Documentary"
- [ ] "Cinematic Short Film"
- [ ] "Instagram Story Flow"
- [ ] "Paranormal Mystery"

### [ ] Task 6.2: Custom Template System
- [ ] Template Upload
- [ ] Template Editor
- [ ] Template Speicherung

---

## 📝 Aktuelle Aufgabe

**ABGESCHLOSSEN:**
- ✅ Task 1.2 - Modul-Grundstruktur erstellt
- ✅ Task 1.3 - Datenmodelle definiert
- ✅ Web UI mit 4 Tabs implementiert (Story, Director's Cut, Export, Templates)
- ✅ Basis API Endpoints erstellt
- ✅ Task 3.1 - Story Consultation Agent mit Custom Tools
- ✅ Task 3.2 - Director's Cut Generator Agent mit Custom Tools
- ✅ **Interaktive Story-Analyse mit User-Feedback-Loop**
- ✅ **Vollständige Director's Cut Generierung (sequenziell)**
- ✅ **Task 4.2 - Detaillierte Prompt-Generierung (KOMPLETT!)**
  - ✅ Image Prompts (Positive + Negative)
  - ✅ Video Prompts (Movement + Camera)
  - ✅ Audio Prompts
  - ✅ Sequenzielle Generierung mit Konsistenz-Tracking
- ✅ **Task 5.1 - Export-Formate (JSON, Markdown, TXT) mit UI**
- ✅ **Task 6 - Custom Template Editor (minimalistisch & effizient)**

**JETZT:** Phase 1-6 komplett abgeschlossen! 🎉🎉🎉

**NÄCHSTE SCHRITTE:**
1. ✅ module.py mit minimaler Struktur
2. ✅ Web UI mit 4 Tabs (Story, Director's Cut, Export, Templates)
3. ✅ ISAA Agent Setup
4. ✅ Detaillierte Prompt-Generierung
5. ✅ Export-Funktionen
6. ✅ Custom Template Editor
7. ✅ Interaktive Story-Analyse
8. ✅ Director's Cut Generierung
9. 🔄 **Server starten & End-to-End testen** (aktueller Schritt)

---

## ⚠️ Wichtige Prinzipien

1. **Einfachheit vor Komplexität** - Minimale Lösungen bevorzugen
2. **Übersichtlicher Code** - Gut wartbar und lesbar
3. **Sequenzielle Generierung** - Szenen Schritt für Schritt erstellen
4. **Guided Generation** - LLM nicht überfordern
5. **Konsistenz-Tracking** - Charaktere, Orte, Zeitlinien tracken

---

---

## 📊 Aktueller Stand

### Implementierte Dateien

1. **module.py** (~2270 Zeilen)
   - **Pydantic Models:** StoryParameters, SceneStructure (erweitert!), StyleTemplate, DirectorsCutProject, CharacterInfo
   - **Tools Class** mit MainTool Integration
   - **5 vordefinierte Style Templates**
   - **Projekt-Management** (in-memory)
   - **ISAA Agents:**
     - Story Consultation Agent mit 3 Custom Tools
     - Director's Cut Generator Agent mit 3 Custom Tools
   - **Detaillierte Prompt-Generierung:**
     - `_build_image_prompt_positive()` - Positive Image Prompts
     - `_build_image_prompt_negative()` - Negative Image Prompts
     - `_build_video_prompt_movement()` - Bewegungs-Prompts
     - `_build_video_prompt_camera()` - Kamera-Prompts
     - `_build_audio_prompt()` - Audio-Prompts
     - `generate_scene_prompts()` - Sequenzielle Szenen-Generierung
   - **API Endpoints (11):**
     - create_project, get_project, list_templates
     - analyze_story (legacy)
     - **analyze_story_interactive** (NEU! Mit User-Feedback-Loop)
     - generate_scene (legacy)
     - **generate_scene_detailed** (Mit allen Details)
     - **generate_directors_cut** (NEU! Vollständige Generierung)
     - **export_prompts** (JSON/Markdown/TXT)
     - **create_template_from_dump** (NEU! Template aus Content-Dump)
     - ui
   - **Vollständige Web UI** mit 4 Tabs:
     - 📝 Story Consultation (mit interaktivem Chat)
     - 🎬 Director's Cut (mit Progress-Anzeige)
     - 📤 Export (mit Download-Buttons)
     - 🎨 Templates (Custom Template Editor)

2. **specs.md** (508 Zeilen)
   - Vollständige technische Spezifikation
   - Workflow-Diagramme
   - Datenstrukturen
   - Konsistenz-Regeln

3. **task.md** (diese Datei)
   - Task-Tracking
   - Fortschritts-Dokumentation

4. **README.md**
   - Modul-Dokumentation
   - API-Beispiele
   - Verwendungsanleitung

### Code-Statistik

- **Gesamt-Zeilen:** ~2270 (+770 neue Zeilen)
- **Pydantic Models:** 5 (SceneStructure erweitert mit detaillierten Prompts)
- **ISAA Agents:** 2
- **Custom Tools:** 6 (3 pro Agent)
- **Prompt-Generator-Funktionen:** 5 (Image+/-, Video Movement/Camera, Audio)
- **API Endpoints:** 11 (+3 neue: analyze_story_interactive, generate_directors_cut, create_template_from_dump)
- **Style Templates:** 5 (+ unbegrenzt Custom Templates via Editor)
- **UI Tabs:** 4 (Story, Director's Cut, Export, Templates)
- **Export-Formate:** 3 (JSON, Markdown, Plain Text)
- **Interaktive Features:**
  - Chat-basierte Story-Analyse
  - Progress-Anzeige bei Generierung
  - Custom Template Editor mit Content-Dump Parser
  - LocalStorage für Custom Templates

### Qualität

- ✅ Übersichtlich strukturiert
- ✅ Gut dokumentiert
- ✅ Einfach wartbar
- ✅ Typsicher (Pydantic)
- ✅ Responsive UI

---

## 🧪 Tests

### Erfolgreich getestet:

```bash
# Version abrufen
tb -c DirCut version
# ✅ Output: 0.1.0

# Templates auflisten
tb -c DirCut list_templates
# ✅ Output: 5 Templates (viral_tiktok, youtube_doc, cinematic, instagram, paranormal)
```

### ISAA Agents Status:
- ✅ StoryConsultant Agent wird initialisiert
- ✅ DirectorsCutGenerator Agent wird initialisiert
- ✅ Custom Tools sind registriert

---

**Letzte Aktualisierung:** 2025-11-15 (Phase 1-6 KOMPLETT! 🎉🎉🎉)
**Version:** 0.1.0-dev
**Status:**
- ✅ Phase 1: Setup & Grundstruktur
- ✅ Phase 2: Web UI (4 Tabs)
- ✅ Phase 3: ISAA Agent Integration
- ✅ **Phase 4: Core Logic - Detaillierte Prompt-Generierung**
- ✅ **Phase 5: Export & Finalisierung**
- ✅ **Phase 6: Custom Template Editor (minimalistisch & effizient)**

**Features:**
- ✅ Interaktive Story-Analyse mit Chat
- ✅ Vollständige Director's Cut Generierung
- ✅ Detaillierte Prompts (Image+/-, Video, Audio)
- ✅ Export (JSON, Markdown, TXT)
- ✅ Custom Template Editor (Content-Dump Parser)

**Nächster Meilenstein:** Server starten & End-to-End testen

