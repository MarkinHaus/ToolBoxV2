# DirCut - Detaillierte Prompt-Generierung

## Übersicht

Das DirCut-Modul generiert **hochdetaillierte, optimierte Prompts** für Image-, Video- und Audio-Generatoren.

### Features

✅ **Image Prompts** - Positive + Negative Prompts für maximale Qualität
✅ **Video Prompts** - Separate Movement & Camera Prompts
✅ **Audio Prompts** - Musik & Sound Design
✅ **Sequenzielle Generierung** - Szene für Szene mit Konsistenz-Tracking
✅ **Export-Formate** - JSON, Markdown, Plain Text

---

## API Endpoints

### 1. Projekt erstellen

```bash
POST /api/DirCut/create_project
Content-Type: application/json

{
    "title": "My Story",
    "story_text": "A character discovers something amazing...",
    "template": "cinematic"
}

# Response:
{
    "result": {
        "data": {
            "project_id": "abc123",
            "title": "My Story",
            "template": "cinematic"
        }
    }
}
```

### 2. Szene mit detaillierten Prompts generieren

```bash
POST /api/DirCut/generate_scene_detailed
Content-Type: application/json

{
    "project_id": "abc123",
    "scene_number": 1,
    "scene_data": {
        "scene_title": "Opening Shot",
        "what_happens": "Character enters a mysterious room",
        "emotional_beat": "curious",
        "camera_type": "Wide Shot",
        "camera_movement": "Slow Dolly In",
        "lens": "Wide Angle 24mm",
        "time_of_day": "Dusk",
        "lighting_setup": "Dramatic",
        "mood": "Moody",
        "setting": "Abandoned mansion, dusty interior",
        "atmosphere": "Foggy",
        "color_palette": ["deep blue", "warm orange", "dark shadows"],
        "characters": [
            {
                "name": "Alex",
                "appearance": "30s, explorer outfit, brown leather jacket, messy hair",
                "action": "walking slowly, looking around cautiously",
                "expression": "curious but alert"
            }
        ]
    }
}
```

**Response:**

```json
{
    "result": {
        "data": {
            "scene_number": 1,
            "scene_title": "Opening Shot",
            "duration": "15-20 Sekunden",
            "what_happens": "Character enters a mysterious room",
            "emotional_beat": "curious",
            
            "prompts": {
                "image": {
                    "positive": "Alex (30s, explorer outfit, brown leather jacket, messy hair), walking slowly, looking around cautiously, curious but alert expression, Character enters a mysterious room, Abandoned mansion, dusty interior, Foggy atmosphere, Dusk, Dramatic lighting, Neutral 5600K, Moody mood, Wide Shot, Wide Angle 24mm, Slow Dolly In camera, Cinematic, 35mm film, slight grain, muted colors, vintage film look, Warm tones, desaturated, slightly faded, deep blue, warm orange, dark shadows, highly detailed, professional photography, 8k uhd, sharp focus, cinematic composition",
                    
                    "negative": "low quality, blurry, out of focus, pixelated, jpeg artifacts, low resolution, grainy, noisy, distorted, deformed, disfigured, bad anatomy, extra limbs, missing limbs, extra fingers, fused fingers, mutated hands, poorly drawn hands, poorly drawn face, ugly, duplicate, morbid, cropped, cut off, watermark, signature, text, logo, username, error, malformed, gross proportions, cartoon, anime, 3d render, painting, drawing, sketch, unrealistic, artificial, fake, worst quality, normal quality, amateur"
                },
                
                "video": {
                    "movement": "Alex: walking slowly, looking around cautiously, Action: Character enters a mysterious room, Atmosphere: Foggy effect, Emotional tone: curious",
                    
                    "camera": "Slow Dolly In camera movement, establishing perspective, deep focus, wide field of view, slight distortion",
                    
                    "duration": "15-20 Sekunden"
                },
                
                "audio": "Indie folk, acoustic, melancholic music, curious mood, Slow, contemplative tempo, 15-20 Sekunden duration"
            },
            
            "visuals": {
                "camera_type": "Wide Shot",
                "camera_movement": "Slow Dolly In",
                "lens": "Wide Angle 24mm",
                "lighting": {
                    "time_of_day": "Dusk",
                    "setup": "Dramatic",
                    "mood": "Moody"
                },
                "art_style": "Cinematic",
                "atmosphere": "Foggy"
            },
            
            "characters": [
                {
                    "name": "Alex",
                    "appearance": "30s, explorer outfit, brown leather jacket, messy hair",
                    "action": "walking slowly, looking around cautiously",
                    "expression": "curious but alert"
                }
            ]
        }
    }
}
```

### 3. Prompts exportieren

```bash
GET /api/DirCut/export_prompts?project_id=abc123&format=json
GET /api/DirCut/export_prompts?project_id=abc123&format=markdown
GET /api/DirCut/export_prompts?project_id=abc123&format=txt
```

**Format: JSON**
```json
{
    "project_id": "abc123",
    "title": "My Story",
    "total_scenes": 3,
    "scenes": [
        {
            "scene_number": 1,
            "scene_title": "Opening Shot",
            "duration": "15-20 Sekunden",
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

**Format: Markdown**
```markdown
# My Story

**Total Scenes:** 3
**Style:** cinematic

---

## Scene 1: Opening Shot

**Duration:** 15-20 Sekunden
**What Happens:** Character enters a mysterious room

### Image Prompts

**Positive:**
```
Alex (30s, explorer outfit...), walking slowly...
```

**Negative:**
```
low quality, blurry, out of focus...
```

### Video Prompts

**Movement:**
```
Alex: walking slowly, looking around cautiously...
```

**Camera:**
```
Slow Dolly In camera movement...
```

### Audio Prompt
```
Indie folk, acoustic, melancholic music...
```

---
```

**Format: Plain Text (Generator-Ready)**
```
=== My Story ===

SCENE 1: Opening Shot

IMAGE PROMPT (POSITIVE):
Alex (30s, explorer outfit...), walking slowly...

IMAGE PROMPT (NEGATIVE):
low quality, blurry, out of focus...

VIDEO PROMPT (MOVEMENT):
Alex: walking slowly, looking around cautiously...

VIDEO PROMPT (CAMERA):
Slow Dolly In camera movement...

AUDIO PROMPT:
Indie folk, acoustic, melancholic music...

================================================================================
```

---

## Prompt-Struktur

### Image Prompt (Positive)

**Format:**
```
[Subject] [Action] [Setting] [Lighting] [Camera] [Style] [Quality]
```

**Komponenten:**
1. **Subject & Action** - Charaktere mit Beschreibung, Aktion, Ausdruck
2. **Setting & Atmosphere** - Ort, Atmosphäre
3. **Lighting** - Tageszeit, Setup, Farbtemperatur, Stimmung
4. **Camera** - Shot-Typ, Objektiv, Bewegung
5. **Style** - Art Direction, Color Grading
6. **Quality Tags** - "highly detailed, 8k uhd, sharp focus..."

### Image Prompt (Negative)

**Standard-Negatives:**
- **Qualität:** low quality, blurry, pixelated, noisy
- **Anatomie:** bad anatomy, extra limbs, deformed
- **Komposition:** cropped, watermark, text
- **Stil:** cartoon, anime, unrealistic

### Video Prompt (Movement)

**Beschreibt:**
- Character Actions
- Object Movements
- Environmental Changes
- Emotional Tone

### Video Prompt (Camera)

**Beschreibt:**
- Camera Movement Type
- Framing Characteristics
- Lens Effects
- Focus Changes

---

## Sequenzielle Generierung

**WICHTIG:** Szenen müssen **sequenziell** generiert werden!

```python
# ✅ RICHTIG: Szene für Szene
scene_1 = generate_scene_detailed(project_id, scene_number=1, ...)
scene_2 = generate_scene_detailed(project_id, scene_number=2, ...)
scene_3 = generate_scene_detailed(project_id, scene_number=3, ...)

# ❌ FALSCH: Alle auf einmal
scenes = [generate_scene(...) for i in range(10)]  # Überlastet LLM!
```

**Warum sequenziell?**
1. **Konsistenz** - Jede Szene nutzt vorherige Szenen für Konsistenz-Checks
2. **LLM-Optimierung** - Verhindert Überlastung des Generators
3. **Guided Generation** - Schrittweise Verfeinerung

---

## Style Templates

Verfügbare Templates:
- `viral_tiktok` - Fast cuts, high energy
- `youtube_doc` - Professional, informative
- `cinematic` - Slow, artistic, film-look
- `instagram` - Episodic, 15s segments
- `paranormal` - Surreal, dreamlike, cool tones

---

## Best Practices

### 1. Detaillierte Character Descriptions
```json
{
    "name": "Alex",
    "appearance": "30s, explorer outfit, brown leather jacket, messy hair, weathered face",
    "action": "walking slowly, looking around cautiously, hand on flashlight",
    "expression": "curious but alert, slight frown"
}
```

### 2. Spezifische Settings
```json
{
    "setting": "Abandoned Victorian mansion, dusty interior, cobwebs, broken furniture, dim light through cracked windows"
}
```

### 3. Konsistente Color Palettes
```json
{
    "color_palette": ["deep blue #1a2a3a", "warm orange #ff8c42", "dark shadows #0d0d0d"]
}
```

### 4. Emotionale Beats
```json
{
    "emotional_beat": "curious → surprised → fearful"
}
```

---

## Verwendung mit Generatoren

### Midjourney
```
/imagine [image_prompt_positive] --no [image_prompt_negative] --ar 16:9 --v 6
```

### Runway Gen-3
```
[video_prompt_movement]
Camera: [video_prompt_camera]
Duration: [video_prompt_duration]
```

### Suno / Udio
```
[audio_prompt]
```

---

**Version:** 0.1.0
**Letzte Aktualisierung:** 2025-11-15

