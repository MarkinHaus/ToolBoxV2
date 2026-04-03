# DirCut - Feature-Übersicht

## 🎉 Vollständig implementierte Features (Phase 1-6)

### ✅ Phase 1: Interaktive Story-Analyse

**Interaktiver Chat mit Consultation Agent:**
- User erstellt Projekt mit Story-Text
- Agent analysiert Story automatisch:
  - Emotional Triggers Detection
  - Viral Potential Calculation (0-40 Punkte)
  - Optimization Suggestions
- Agent stellt gezielte Fragen zur Optimierung
- User antwortet im Chat
- Agent verfeinert Analyse basierend auf Antworten
- Wenn fertig: "READY_FOR_GENERATION" → Phase 2 aktiviert

**API Endpoints:**
- `POST /api/DirCut/create_project` - Projekt erstellen
- `POST /api/DirCut/analyze_story_interactive` - Interaktive Analyse mit Chat

**UI Features:**
- Chat-Interface mit Agent/User Messages
- Automatische Scroll-to-Bottom
- Input-Feld für User-Antworten
- Visual Feedback (Agent = Blau, User = Grün)

---

### ✅ Phase 2: Director's Cut Generierung

**Vollständige sequenzielle Generierung:**
- Auto-Detection der optimalen Szenen-Anzahl (3-10)
- Oder manuelle Eingabe durch User
- Szenen werden einzeln generiert (nicht alle auf einmal!)
- Jede Szene nutzt vorherige Szenen für Konsistenz
- Progress-Anzeige während Generierung
- Detaillierte Prompts für jede Szene:
  - Image Prompt (Positive)
  - Image Prompt (Negative)
  - Video Prompt (Movement)
  - Video Prompt (Camera)
  - Audio Prompt

**API Endpoints:**
- `POST /api/DirCut/generate_directors_cut` - Vollständige Generierung
- `POST /api/DirCut/generate_scene_detailed` - Einzelne Szene

**UI Features:**
- Szenen-Anzahl Input (optional)
- Progress-Anzeige mit Status
- Szenen-Liste mit Preview
- Automatischer Wechsel zu Export-Tab

---

### ✅ Phase 3: Detaillierte Prompt-Generierung

**Image Prompts (Positive):**
```
Format: [Subject] [Action] [Setting] [Lighting] [Camera] [Style] [Quality]

Beispiel:
"Alex (30s, explorer outfit, brown leather jacket), walking slowly, curious expression,
Character enters mysterious room, Abandoned mansion, Foggy atmosphere, Dusk, Dramatic lighting,
Wide Shot, Wide Angle 24mm, Slow Dolly In camera, Cinematic, 35mm film, muted colors,
highly detailed, 8k uhd, sharp focus"
```

**Image Prompts (Negative):**
```
Standard-Negatives:
- Quality: low quality, blurry, pixelated, noisy
- Anatomy: bad anatomy, extra limbs, deformed
- Composition: watermark, text, cropped
- Style: cartoon, anime, unrealistic
```

**Video Prompts (Movement):**
```
Character Actions + Object Movements + Environmental Changes + Emotional Tone

Beispiel:
"Alex: walking slowly, looking around cautiously,
Action: Character enters mysterious room,
Atmosphere: Foggy effect,
Emotional tone: curious"
```

**Video Prompts (Camera):**
```
Camera Movement + Framing + Lens Effects

Beispiel:
"Slow Dolly In camera movement,
establishing perspective, deep focus,
wide field of view, slight distortion"
```

**Audio Prompts:**
```
Music Genre + Emotional Tone + Pacing + Duration

Beispiel:
"Indie folk, acoustic, melancholic music,
curious mood,
Slow, contemplative tempo,
15-20 Sekunden duration"
```

---

### ✅ Phase 4: Export & Download

**3 Export-Formate:**

1. **JSON Export** - Strukturierte Daten
```json
{
    "project_id": "abc123",
    "title": "My Story",
    "total_scenes": 5,
    "scenes": [
        {
            "scene_number": 1,
            "prompts": {
                "image_positive": "...",
                "image_negative": "...",
                "video_movement": "...",
                "video_camera": "...",
                "audio": "..."
            }
        }
    ]
}
```

2. **Markdown Export** - Formatierte Dokumentation
```markdown
# My Story

## Scene 1: Opening Shot

### Image Prompts
**Positive:** ...
**Negative:** ...

### Video Prompts
**Movement:** ...
**Camera:** ...
```

3. **Plain Text Export** - Generator-Ready Prompts
```
SCENE 1: Opening Shot

IMAGE PROMPT (POSITIVE):
...

IMAGE PROMPT (NEGATIVE):
...

VIDEO PROMPT (MOVEMENT):
...
```

**API Endpoints:**
- `GET /api/DirCut/export_prompts?project_id=X&format=json|markdown|txt`

**UI Features:**
- Download-Buttons für alle Formate
- Automatischer Datei-Download
- Dateiname: `dircut_{project_id}.{format}`

---

### ✅ Phase 5: Custom Template Editor

**Minimalistisch & Effizient:**

**2 Erstellungs-Modi:**

1. **Aus Content-Dump:**
   - User fügt JSON, YAML, oder Freitext ein
   - Generator Agent parst den Dump
   - Extrahiert Template-Parameter
   - Füllt Editor automatisch

2. **Aus bestehendem Template:**
   - User wählt Basis-Template (5 vordefiniert)
   - Template wird geladen
   - User passt Parameter an

**Template-Editor (5 Hauptfelder):**
- Name (z.B. "My Custom Style")
- Art Direction (z.B. "Cinematic")
- Color Grading (z.B. "Warm tones, desaturated")
- Music Genre (z.B. "Indie folk, acoustic")
- Pacing Style (z.B. "Slow, contemplative")

**Persistence:**
- Templates werden in LocalStorage gespeichert
- Automatisch in Projekt-Erstellung verfügbar
- Liste aller Custom Templates anzeigen

**API Endpoints:**
- `POST /api/DirCut/create_template_from_dump` - Template aus Dump generieren

**UI Features:**
- Toggle zwischen Dump und Existing
- Textarea für Content-Dump
- Dropdown für Basis-Template
- Formular-Editor mit 5 Feldern
- Speichern-Button
- Liste gespeicherter Templates

---

## 📊 Technische Details

### ISAA Agents

**1. Story Consultation Agent:**
- Custom Tools:
  - `analyze_emotional_triggers()` - Erkennt Emotionen
  - `calculate_viral_potential()` - Berechnet Score (0-40)
  - `suggest_optimizations()` - Gibt Verbesserungsvorschläge
- Temperature: 0.7 (kreativ)
- Interaktiver Chat-Modus

**2. Director's Cut Generator Agent:**
- Custom Tools:
  - `_check_visual_consistency()` - Prüft Art Style
  - `_check_temporal_consistency()` - Prüft zeitliche Logik
  - `_track_character_consistency()` - Prüft Character-Kontinuität
- Temperature: 0.5 (präzise)
- Sequenzielle Szenen-Generierung

### Konsistenz-Tracking

**Automatische Checks:**
- Visual Continuity (Art Style, Color Grading)
- Temporal Continuity (Zeit-Progression, Logik)
- Character Consistency (Appearance, Clothing)
- Location Consistency (Spatial Logic)

**Implementierung:**
- Jede Szene erhält vorherige Szenen als Kontext
- Generator Agent prüft Konsistenz vor Generierung
- Warnings bei Inkonsistenzen

---

## 🎯 Verwendung

### 1. Projekt erstellen
```javascript
POST /api/DirCut/create_project
{
    "title": "My Story",
    "story_text": "...",
    "template": "cinematic"
}
```

### 2. Interaktive Analyse
```javascript
POST /api/DirCut/analyze_story_interactive
{
    "project_id": "abc123"
}

// Agent antwortet mit Fragen
// User antwortet:
POST /api/DirCut/analyze_story_interactive
{
    "project_id": "abc123",
    "user_message": "Die Hauptemotion soll Neugier sein"
}
```

### 3. Director's Cut generieren
```javascript
POST /api/DirCut/generate_directors_cut
{
    "project_id": "abc123",
    "num_scenes": 5  // Optional
}
```

### 4. Exportieren
```javascript
GET /api/DirCut/export_prompts?project_id=abc123&format=markdown
```

---

## 🚀 Nächste Schritte

1. **Server starten** und UI testen
2. **End-to-End Test** durchführen
3. **Feedback sammeln** und iterieren
4. **Optional:** CSV Shot List Export implementieren

---

**Version:** 0.1.0-dev
**Status:** Alle Phasen 1-6 komplett! 🎉
**Letzte Aktualisierung:** 2025-11-15

