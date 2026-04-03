Absolut. Hier ist ein umfassender und detaillierter Prompt, der eine KI anleitet, den Refactoring-Plan für Ihre Engine zu erstellen. Dieser Prompt ist so konzipiert, dass er die gesamte Komplexität Ihrer Anfrage erfasst und die KI dazu anleitet, eine modulare, mehrstufige und hochgradig konfigurierbare "Timeline"-Generierungslogik zu entwerfen.

Sie können diesen Prompt direkt an eine fortschrittliche KI (wie mich) weitergeben.

---

### **Prompt für die KI: Refactoring der "Story Generator" Engine zu einer "Timeline Generation Engine"**

**Ihre Aufgabe:**
Erstellen Sie einen umfassenden Refactoring-Plan, um die bestehende "Story Generator"-Engine in eine fortschrittliche, mehrstufige **"Timeline Generation Engine"** umzuwandeln. Das primäre Ziel ist es, die gesamte textbasierte Erstellung (Prompts, Skripte, etc.) von der ressourcenintensiven Mediengenerierung zu trennen. Die neue Engine muss modular, hochgradig konfigurierbar und auf Zuverlässigkeit und Geschwindigkeit bei der Textgenerierung ausgelegt sein.

Das Endergebnis dieses Plans ist nicht die Story selbst, sondern die vollständige, datenstrukturierte **"Timeline"**, die als exakter Bauplan für die spätere, vom Benutzer bestätigte Mediengenerierung dient.

**Kernprinzipien des neuen Designs:**

1.  **Upfront Text Generation (Batching):** Alle LLM-Aufrufe zur Erstellung von Prompts für Bilder, Videoclips, Erzählung, Dialoge und Soundeffekte müssen *im Voraus* in einer einzigen Batch-Phase erfolgen. Es dürfen keine "Mini-Tasks" oder LLM-Aufrufe während der eigentlichen Bild- oder Videogenerierung stattfinden.
2.  **Benutzer-Bestätigungs-Tor (User Confirmation Gate):** Nachdem die vollständige textbasierte "Timeline" generiert wurde, muss das System anhalten. Die Mediengenerierung beginnt erst, nachdem der Benutzer (über die UI oder API) die generierten Texte überprüft, optional bearbeitet und die Produktion explizit startet.
3.  **Ablaufende Generierungsaufgabe:** Die Bestätigung durch den Benutzer erstellt eine Aufgabe (einen "Generierungs-Warenkorb"), die 24 Stunden lang gültig ist. Innerhalb dieser Zeit kann der Benutzer die Generierung starten.
4.  **Maximale Konfigurierbarkeit:** Kritische Parameter, insbesondere die zu verwendenden Videomodelle (z.B. "veo3", "fal-ai/minimax"), müssen über den API-Aufruf konfigurierbar sein, sodass die Benutzeroberfläche dem Benutzer eine Auswahl bieten kann.
5.  **Zuverlässigkeit & Geschwindigkeit:** Die Batch-Generierung von Texten muss auf Robustheit ausgelegt sein, indem sie exponentielles Backoff für fehlgeschlagene Einzelaufgaben und API-Key-Rolling für eine schnelle, parallele Verarbeitung unterstützt.
6.  **Flexibilität im Inhalt:** Die Engine muss in der Lage sein, Videos nur aus einer Story (ohne Charaktere) zu erstellen. Die Erzählung muss optional sein.

---

### **Der Refactoring-Plan: Von der Story zur Timeline**

Befolgen Sie diese Schritte, um den neuen Generierungsfluss zu entwerfen. Sie sollen diesen Plan nicht auf einmal generieren, sondern jeden Schritt detailliert ausarbeiten.

#### **Schritt 1: Definition der zentralen Datenstruktur: Die `Timeline` Klasse**

Definieren Sie zuerst die Pydantic-`BaseModel`-Klasse, die die gesamte generierte "Timeline" repräsentiert. Dies ist das zentrale Datenobjekt des gesamten Prozesses.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

# --- Konfigurationsmodelle ---
class StyleConfig(BaseModel):
    image_style: str = Field(description="Visueller Stil für Bilder, z.B. 'Anime', 'Photorealistic'.")
    video_style: str = Field(description="Kinematischer Stil für Videos, z.B. 'Hollywood Blockbuster', 'Documentary'.")
    # ... weitere Stilparameter

class GenerationConfig(BaseModel):
    video_model: str = Field(default="fal-ai/minimax", description="Das zu verwendende Videomodell, konfigurierbar pro API-Aufruf.")
    input_mode: Literal["single_image", "multi_reference", "start_end_frames", "text_only"]
    generate_narration: bool = True
    combined_video_audio_model: bool = Field(default=False, description="True, wenn Modelle wie Veo3 verwendet werden, die Video+Audio zusammen generieren.")
    api_keys: Optional[List[str]] = Field(default=None, description="Liste von API-Schlüsseln für Key-Rolling.")

# --- Inhaltsmodelle ---
class DialogueLine(BaseModel):
    character_name: str
    text: str
    emotion: Optional[str] = "neutral"

class SoundEffect(BaseModel):
    timestamp_in_clip: float = Field(description="Sekunde innerhalb des Clips, an der der Effekt auftreten soll.")
    prompt: str = Field(description="Text-Prompt für den Soundeffekt-Generator (z.B. 'leises Windrauschen').")
    provider: Literal["elevenlabs", "fal_ai_audio"] = "elevenlabs"

class VideoClipDetails(BaseModel):
    duration_seconds: int
    image_prompt: str = Field(description="Vollständiger, optimierter Prompt für die Generierung des Startbildes/der Referenzbilder.")
    video_prompt: str = Field(description="Vollständiger, für Video optimierter Prompt. Enthält Details zu Kamerabewegung, Charakteraktionen, Emotionen und Übergängen.")
    start_image_ref: Optional[str] = Field(default=None, description="Optionaler Verweis auf ein vom Benutzer bereitgestelltes Startbild.")
    end_image_ref: Optional[str] = Field(default=None, description="Optionaler Verweis auf ein vom Benutzer bereitgestelltes Endbild.")
    reference_images_ref: Optional[List[str]] = Field(default=None, description="Optionale Verweise auf Referenzbilder.")

class Scene(BaseModel):
    scene_number: int
    title: str
    setting_description: str
    video_clip: VideoClipDetails
    narration_script: Optional[str] = None
    dialogue_script: List[DialogueLine] = []
    sound_effects: List[SoundEffect] = []

# --- Haupt-Timeline-Modell ---
class Timeline(BaseModel):
    title: str
    genre: str
    characters: List[Dict] = []
    world_summary: str
    global_style: StyleConfig
    generation_config: GenerationConfig
    scenes: List[Scene]
    estimated_total_duration: int
```

---

#### **Schritt 2: Entwurf des neuen, mehrstufigen Generierungsflusses**

Beschreiben Sie den neuen Prozess, der die `Timeline`-Struktur erzeugt.

##### **2.1: Initialer Input & optionale Story-Anreicherung**

Der Prozess beginnt mit einem einfachen Benutzer-Prompt. Ein optionaler erster LLM-Aufruf kann diesen anreichern.

*   **Input:** Ein einfacher Text-Prompt vom Benutzer (z.B. "Ein einsamer Roboter in einer postapokalyptischen Wüstenstadt").
*   **Aktion:** Führen Sie eine optionale "Mini-Task" aus, um den Prompt zu einer kurzen Story mit Anfang, Mitte und Ende zu erweitern, wenn der Benutzer dies wünscht.
*   **Output:** Eine angereicherte Kern-Story.

##### **2.2: Generierung des `Timeline`-Outlines (Erster Hauptaufruf)**

Dies ist der entscheidende erste Schritt. Ein LLM-Aufruf generiert das Grundgerüst der `Timeline`, füllt jedoch die detaillierten Prompts und Skripte nur mit Platzhaltern.

*   **Aktion:** Rufen Sie ein LLM mit der angereicherten Story und den Konfigurationen auf und weisen Sie es an, eine `Timeline`-Instanz zu erstellen.
*   **Anweisung für das LLM:** "Generiere basierend auf der folgenden Story eine `Timeline`-Struktur. Fülle alle Felder wie `title`, `genre`, `characters`, `world_summary`, `global_style` und `generation_config`. Erstelle eine Gliederung für die `scenes`, inklusive `scene_number`, `title` und einer kurzen `setting_description`. Für die Felder `image_prompt`, `video_prompt`, `narration_script`, `dialogue_script` und `sound_effects` setze vorerst nur beschreibende Platzhalter ein, wie z.B. `"[Platzhalter für detaillierten Bild-Prompt für Szene 1]"`.
*   **Output:** Ein teilweise gefülltes `Timeline`-Objekt, das als Arbeitsplan dient.

##### **2.3: Batch-Generierung der Details (Zweite Hauptphase)**

Jetzt werden die Platzhalter im `Timeline`-Outline in einer hochgradig parallelisierten Batch-Operation mit detaillierten Inhalten gefüllt.

*   **Aktion:** Iterieren Sie durch das `Timeline`-Outline und erstellen Sie für jeden Platzhalter eine separate LLM-Aufgabe. Führen Sie diese Aufgaben parallel aus. Implementieren Sie hierfür eine Logik für **exponentielles Backoff** bei Fehlschlägen einzelner Aufgaben und **API-Key-Rolling**, um Ratenbegrenzungen zu umgehen.

*   **2.3.1: Bild- und Videoprompts generieren:**
    *   **Sub-Task:** Für jede Szene, generiere `image_prompt` und `video_prompt`.
    *   **Anweisung für das LLM:** "Erstelle einen detaillierten und für die Bild-/Videogenerierung optimierten Prompt. Der Prompt muss **konsistent** mit den globalen Informationen (`world_summary`, `global_style`, Charakterbeschreibungen) sein. Beschreibe explizit:
        *   **Kamerabewegung:** (z.B. "langsamer Kameraschwenk von links nach rechts", "dramatischer Zoom auf das Gesicht des Charakters").
        *   **Charakteraktionen & Emotionen:** (z.B. "Charakter A hebt zögernd eine Hand, sein Gesicht zeigt eine Mischung aus Angst und Neugier").
        *   **Licht und Atmosphäre:** (z.B. "goldenes Licht der untergehenden Sonne, lange Schatten, staubige Luft").
        *   **Übergänge (Transitions):** (z.B. "Die Szene endet mit einer harten Blende zur nächsten Einstellung")."

*   **2.3.2: Erzähl- und Dialogskripte generieren:**
    *   **Sub-Task:** Für jede Szene, generiere `narration_script` und `dialogue_script`, falls angefordert.
    *   **Anweisung für das LLM:** "Schreibe das Skript für die Erzählung und die Dialoge für diese Szene. Die Sprache sollte zum Genre passen. Formatiere die Dialoge mit Charaktername und Emotion."

*   **2.3.3: Soundeffekt-Prompts generieren:**
    *   **Sub-Task:** Für jede Szene, analysiere den `video_prompt` und das Skript und generiere eine Liste von `SoundEffect`-Prompts.
    *   **Anweisung für das LLM:** "Identifiziere Schlüsselmomente in der Szene und erstelle kurze, prägnante Prompts für Soundeffekte, z.B. 'Schritte auf Metall', 'elektrisches Surren', 'entfernter Donnerschlag'."

*   **Output:** Das vollständig ausgefüllte `Timeline`-Objekt. Alle textuellen Inhalte sind nun finalisiert.

---

#### **Schritt 3: Entwurf des Ausführungsflusses (nach der Textgenerierung)**

Beschreiben Sie die Systemlogik, die nach der Erstellung der `Timeline` abläuft.

##### **3.1: Benutzer-Bestätigung und Aufgaben-Erstellung**

*   **Aktion:** Das Backend speichert die vollständige `Timeline` und sendet sie an das Frontend. Die UI zeigt dem Benutzer alle generierten Texte (Story, Prompts, Skripte) in einem editierbaren Format an.
*   **Benutzerinteraktion:** Der Benutzer kann die Texte prüfen und bei Bedarf anpassen.
*   **Bestätigung:** Ein "Generierung starten"-Button sendet die (ggf. modifizierte) `Timeline` zurück an die API.
*   **Backend-Aktion:** Das Backend erstellt eine neue Aufgabe in einer persistenten Warteschlange (z.B. Redis + Celery). Diese Aufgabe enthält die finale `Timeline` und hat eine **Lebensdauer von 24 Stunden**. Der API-Aufruf gibt eine Job-ID zurück.

##### **3.2: Asynchrone Mediengenerierungs-Pipeline (Worker-Prozess)**

Ein separater Worker-Prozess holt die bestätigte Aufgabe aus der Warteschlange und führt die Mediengenerierung durch.

*   **Aktion:** Der Worker parst das `Timeline`-Objekt und führt die folgenden Schritte parallel und nicht-blockierend aus:
    1.  **Bildgenerierung:** Generiere alle Start-/Referenz-/Endbilder basierend auf den `image_prompt`s in der `Timeline`.
    2.  **Videoclip-Generierung:** Sobald die Bilder für eine Szene fertig sind, starte die Videoclip-Generierung mit dem in `generation_config.video_model` spezifizierten Modell und dem `video_prompt`.
    3.  **Audiogenerierung (parallel):**
        *   Generiere alle Erzähl- und Dialog-Audiospuren.
        *   Generiere alle Soundeffekte basierend auf den `sound_effects`-Prompts.
    4.  **Zusammenfügen:** Montiere die Videoclips in der Reihenfolge der Szenen.
    5.  **Audio-Post-Processing & Mischen:**
        *   Lege die Dialog- und Erzählspuren über die entsprechenden Videoclips.
        *   Füge die Soundeffekte an den spezifizierten Zeitstempeln hinzu.
        *   **Optionaler Audio-Effekt-Schritt:** Falls in der `Timeline` konfiguriert, nutze ein Video-zu-Audio-Modell (`fal.ai/video-to-audio`), um eine Basis-Audiospur aus dem Video zu extrahieren, und mische diese mit den generierten Dialogen und Effekten.
    6.  **Finale Ausgabe:** Erstelle die endgültige Videodatei.

*   **Output:** Das fertige Video.
*
alle ffmpeg fuctionen müssen mit dem contevt von toolboxv2/mods/videoFlow/ffmpeg_hep.md generirt werden !
Alle anderungen müssen in allen relaventen datein umgesetez und integirt werden lese die interne docs unter toolboxv2/mods/videoFlow/docs für informationen. nach einem edeit müssen alle test erneut asugefütr werden und erfogerich sein bevor zum nöcshten schrit weider gegeneg werden darf ! toolboxv2/mods/videoFlow/tests
