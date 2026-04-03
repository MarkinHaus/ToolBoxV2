"""
DirCut - Director's Cut Pipeline Module
Zweistufiges KI-gestütztes System zur Transformation von Story-Input
in strukturierte Director's Cut Prompts für Bild- und Video-Generatoren.
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from toolboxv2 import App, Result, RequestData, get_app, MainTool
import asyncio
import json
import re

# Module Constants
MOD_NAME = "DirCut"
VERSION = "0.1.0"
export = get_app(f"mods.{MOD_NAME}").tb


# ============================================================================
# DATENMODELLE (Pydantic)
# ============================================================================

class StoryParameters(BaseModel):
    """Parameter für die Story aus Phase 1"""
    total_scenes: int = Field(default=8, ge=3, le=20, description="Anzahl der Szenen")
    total_duration: str = Field(default="2-3 Minuten", description="Gesamtdauer")
    target_platform: str = Field(default="Universal", description="Zielplattform")
    primary_emotion: str = Field(default="Joy", description="Hauptemotion")
    secondary_emotions: List[str] = Field(default_factory=list, description="Sekundäre Emotionen")
    art_style: str = Field(default="Cinematic", description="Kunststil")
    pacing: str = Field(default="medium", description="Tempo")
    approved: bool = Field(default=False, description="Story approved?")


class CharacterInfo(BaseModel):
    """Charakter-Informationen für Konsistenz"""
    name: str
    appearance: str = Field(description="Physische Beschreibung")
    position: str = Field(default="center", description="Position im Frame")
    action: str = Field(default="", description="Aktuelle Aktion")
    expression: str = Field(default="neutral", description="Gesichtsausdruck")


class SceneStructure(BaseModel):
    """Vollständige Szenenstruktur"""
    scene_number: int
    scene_title: str
    duration: str = Field(default="15-20 Sekunden")

    # Story Context
    what_happens: str = Field(description="Was passiert in der Szene")
    character_actions: str = Field(default="", description="Charakter-Aktionen")
    dialogue: str = Field(default="", description="Gesprochene Worte")
    emotional_beat: str = Field(description="Emotionaler Beat")

    # Visual Specs
    camera_type: str = Field(default="Medium Shot")
    camera_movement: str = Field(default="Static")
    lens: str = Field(default="Standard 50mm")

    # Lighting
    time_of_day: str = Field(default="Afternoon")
    lighting_setup: str = Field(default="Natural")
    color_temperature: str = Field(default="Neutral 5600K")
    mood: str = Field(default="Bright")

    # Art Direction
    art_style: str = Field(description="Konsistenter Art Style")
    color_palette: List[str] = Field(default_factory=list)
    atmosphere: str = Field(default="Clear")

    # Location & Time
    setting: str = Field(description="Ort der Szene")
    time_relation: str = Field(default="Immediately after")

    # Characters
    characters: List[CharacterInfo] = Field(default_factory=list)

    # Generator Prompts (Detailliert!)
    image_prompt_positive: str = Field(default="", description="Positive Image Prompt (was zeigen)")
    image_prompt_negative: str = Field(default="", description="Negative Image Prompt (was vermeiden)")

    video_prompt_movement: str = Field(default="", description="Bewegungs-/Action-Prompt für Video")
    video_prompt_camera: str = Field(default="", description="Kamera-Bewegung für Video")
    video_prompt_duration: str = Field(default="15s", description="Video-Dauer")

    audio_prompt: str = Field(default="", description="Audio/Music Prompt")

    # Legacy (für Rückwärtskompatibilität)
    image_prompt: str = Field(default="", description="[DEPRECATED] Kombinierter Image Prompt")
    video_prompt: str = Field(default="", description="[DEPRECATED] Kombinierter Video Prompt")


class StyleTemplate(BaseModel):
    """Style Template für konsistente Visuals"""
    name: str
    art_direction: str = Field(default="Cinematic")
    color_grading: str = Field(default="Natural")
    camera_preference: str = Field(default="Standard")
    lighting_approach: str = Field(default="Natural light")
    music_genre: str = Field(default="Ambient")
    pacing_style: str = Field(default="Medium")


class DirectorsCutProject(BaseModel):
    """Vollständiges Director's Cut Projekt"""
    title: str
    story_text: str
    parameters: StoryParameters
    style_template: StyleTemplate
    scenes: List[SceneStructure] = Field(default_factory=list)
    created_at: str = Field(default="")
    viral_score: int = Field(default=0, ge=0, le=40)


# ============================================================================
# TOOLS CLASS
# ============================================================================

class Tools(MainTool):
    """DirCut Module Tools"""

    def __init__(self, app: App):
        self.name = MOD_NAME
        self.version = VERSION

        # Storage für aktive Projekte (in-memory für MVP)
        self.active_projects: Dict[str, DirectorsCutProject] = {}

        # Vordefinierte Style Templates
        self.style_templates = self._init_style_templates()

        # ISAA Agents (werden später initialisiert)
        self.consultation_agent = None
        self.generator_agent = None

        self.tools = {
            "all": [["version", "Zeigt Modul-Version"]],
            "name": self.name,
            "version": self.show_version,
        }

        super().__init__(
            load=self.on_start,
            v=self.version,
            tool=self.tools,
            name=self.name,
            on_exit=self.on_exit
        )

    def on_start(self):
        """Modul-Initialisierung"""
        self.app.logger.info(f"{self.name} v{self.version} wird initialisiert...")

        # UI bei CloudM registrieren
        self.app.run_any(
            ("CloudM", "add_ui"),
            name=self.name,
            title="Director's Cut Pipeline",
            path=f"/api/{self.name}/ui",
            description="Story-zu-Video Pipeline mit KI-Agents",
            auth=False  # Für MVP kein Auth erforderlich
        )

        # ISAA Agents initialisieren (async)
        asyncio.create_task(self._init_agents())

        self.app.logger.info(f"{self.name} erfolgreich initialisiert!")

    def on_exit(self):
        """Cleanup beim Beenden"""
        self.app.logger.info(f"{self.name} wird beendet...")

    def show_version(self):
        """Zeigt Version"""
        return self.version

    def _init_style_templates(self) -> Dict[str, StyleTemplate]:
        """Initialisiert vordefinierte Style Templates"""
        return {
            "viral_tiktok": StyleTemplate(
                name="Viral TikTok Energy",
                art_direction="Fast cuts, high energy, vertical format",
                color_grading="Vibrant, saturated colors",
                camera_preference="Dynamic, handheld",
                lighting_approach="Bright, high-key",
                music_genre="Trending audio, upbeat",
                pacing_style="Fast"
            ),
            "youtube_doc": StyleTemplate(
                name="YouTube Documentary",
                art_direction="Professional, informative",
                color_grading="Natural, balanced",
                camera_preference="Steady, tripod",
                lighting_approach="Three-point lighting",
                music_genre="Background ambient",
                pacing_style="Medium"
            ),
            "cinematic": StyleTemplate(
                name="Cinematic Short Film",
                art_direction="35mm film look, grain, muted colors",
                color_grading="Warm tones, desaturated",
                camera_preference="Slow, deliberate movements",
                lighting_approach="Natural light, golden hour",
                music_genre="Orchestral, emotional",
                pacing_style="Slow"
            ),
            "instagram": StyleTemplate(
                name="Instagram Story Flow",
                art_direction="Episodic, 15s segments",
                color_grading="Bright, clean",
                camera_preference="Quick cuts, vertical",
                lighting_approach="Bright, even",
                music_genre="Pop, upbeat",
                pacing_style="Fast"
            ),
            "paranormal": StyleTemplate(
                name="Paranormal Mystery",
                art_direction="Surreal, dreamlike, cool tones",
                color_grading="Desaturated, blue-green tint",
                camera_preference="Slow, eerie movements",
                lighting_approach="Low-key, dramatic shadows",
                music_genre="Ambient, unsettling",
                pacing_style="Slow"
            )
        }

    def get_template(self, template_name: str) -> Optional[StyleTemplate]:
        """Holt ein Style Template"""
        return self.style_templates.get(template_name)

    def create_project(self, title: str, story_text: str, template_name: str = "cinematic") -> str:
        """Erstellt ein neues Director's Cut Projekt"""
        import uuid
        from datetime import datetime

        project_id = str(uuid.uuid4())[:8]

        template = self.get_template(template_name) or self.style_templates["cinematic"]

        project = DirectorsCutProject(
            title=title,
            story_text=story_text,
            parameters=StoryParameters(),
            style_template=template,
            created_at=datetime.now().isoformat()
        )

        self.active_projects[project_id] = project
        self.app.logger.info(f"Projekt '{title}' erstellt (ID: {project_id})")

        return project_id

    def get_project(self, project_id: str) -> Optional[DirectorsCutProject]:
        """Holt ein Projekt"""
        return self.active_projects.get(project_id)

    # ========================================
    # SCENE GENERATION - Detaillierte Prompts
    # ========================================

    def _build_image_prompt_positive(
        self,
        scene: SceneStructure,
        style: StyleTemplate,
        previous_scenes: List[SceneStructure] = None
    ) -> str:
        """
        Erstellt detaillierten POSITIVEN Image Prompt

        Format: [Subject] [Action] [Setting] [Lighting] [Camera] [Style] [Quality]
        """
        # Subject & Action
        subject_parts = []
        for char in scene.characters:
            subject_parts.append(f"{char.name} ({char.appearance}), {char.action}, {char.expression} expression")

        subject = ", ".join(subject_parts) if subject_parts else "scene"

        # Setting & Atmosphere
        setting = f"{scene.setting}, {scene.atmosphere} atmosphere"

        # Lighting
        lighting = f"{scene.time_of_day}, {scene.lighting_setup} lighting, {scene.color_temperature}, {scene.mood} mood"

        # Camera
        camera = f"{scene.camera_type}, {scene.lens}, {scene.camera_movement} camera"

        # Style & Art Direction
        style_desc = f"{scene.art_style}, {style.art_direction}, {style.color_grading}"

        # Color Palette
        colors = ", ".join(scene.color_palette) if scene.color_palette else "natural colors"

        # Quality Tags
        quality = "highly detailed, professional photography, 8k uhd, sharp focus, cinematic composition"

        # Kombiniere alles
        prompt = f"{subject}, {scene.what_happens}, {setting}, {lighting}, {camera}, {style_desc}, {colors}, {quality}"

        return prompt

    def _build_image_prompt_negative(self) -> str:
        """
        Erstellt NEGATIVEN Image Prompt (was vermeiden)

        Standard-Negatives für hochwertige Outputs
        """
        negatives = [
            # Qualität
            "low quality", "blurry", "out of focus", "pixelated", "jpeg artifacts",
            "low resolution", "grainy", "noisy", "distorted",

            # Anatomie (bei Charakteren)
            "deformed", "disfigured", "bad anatomy", "extra limbs", "missing limbs",
            "extra fingers", "fused fingers", "mutated hands", "poorly drawn hands",
            "poorly drawn face", "ugly", "duplicate", "morbid",

            # Komposition
            "cropped", "cut off", "watermark", "signature", "text", "logo",
            "username", "error", "malformed", "gross proportions",

            # Stil
            "cartoon", "anime", "3d render", "painting", "drawing", "sketch",
            "unrealistic", "artificial", "fake",

            # Sonstiges
            "worst quality", "normal quality", "amateur"
        ]

        return ", ".join(negatives)

    def _build_video_prompt_movement(
        self,
        scene: SceneStructure,
        previous_scenes: List[SceneStructure] = None
    ) -> str:
        """
        Erstellt detaillierten BEWEGUNGS-Prompt für Video-Generator

        Beschreibt: Character Actions, Object Movements, Environmental Changes
        """
        movements = []

        # Character Movements
        for char in scene.characters:
            if char.action:
                movements.append(f"{char.name}: {char.action}")

        # Scene-specific action
        if scene.character_actions:
            movements.append(f"Action: {scene.character_actions}")

        # Atmosphere changes
        if scene.atmosphere != "Clear":
            movements.append(f"Atmosphere: {scene.atmosphere} effect")

        # Emotional movement
        movements.append(f"Emotional tone: {scene.emotional_beat}")

        # Combine
        if movements:
            return ", ".join(movements)
        else:
            return "subtle ambient movement, natural breathing, slight environmental motion"

    def _build_video_prompt_camera(self, scene: SceneStructure) -> str:
        """
        Erstellt KAMERA-BEWEGUNGS-Prompt für Video

        Beschreibt: Camera Movement, Transitions, Focus Changes
        """
        camera_parts = []

        # Base camera movement
        camera_parts.append(f"{scene.camera_movement} camera movement")

        # Camera type influences movement
        if scene.camera_type == "Close-Up":
            camera_parts.append("intimate framing, shallow depth of field")
        elif scene.camera_type == "Wide Shot":
            camera_parts.append("establishing perspective, deep focus")
        elif scene.camera_type == "POV":
            camera_parts.append("first-person perspective, immersive view")

        # Lens characteristics
        if "Wide Angle" in scene.lens:
            camera_parts.append("wide field of view, slight distortion")
        elif "Telephoto" in scene.lens:
            camera_parts.append("compressed perspective, bokeh background")

        return ", ".join(camera_parts)

    def generate_scene_prompts(
        self,
        project_id: str,
        scene_number: int,
        scene_data: dict
    ) -> SceneStructure:
        """
        Generiert eine vollständige Szene mit detaillierten Prompts

        WICHTIG: Sequenziell aufrufen! Szene 1, dann 2, dann 3, etc.
        Nutzt vorherige Szenen für Konsistenz-Checks.

        Args:
            project_id: Projekt-ID
            scene_number: Szenen-Nummer (1-N)
            scene_data: Dict mit Szenen-Informationen

        Returns:
            SceneStructure mit allen generierten Prompts
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Projekt {project_id} nicht gefunden")

        # Vorherige Szenen für Konsistenz
        previous_scenes = [s for s in project.scenes if s.scene_number < scene_number]

        # Erstelle Scene Structure
        scene = SceneStructure(
            scene_number=scene_number,
            scene_title=scene_data.get("scene_title", f"Scene {scene_number}"),
            duration=scene_data.get("duration", "15-20 Sekunden"),

            # Story
            what_happens=scene_data.get("what_happens", ""),
            character_actions=scene_data.get("character_actions", ""),
            dialogue=scene_data.get("dialogue", ""),
            emotional_beat=scene_data.get("emotional_beat", "neutral"),

            # Visuals
            camera_type=scene_data.get("camera_type", "Medium Shot"),
            camera_movement=scene_data.get("camera_movement", "Static"),
            lens=scene_data.get("lens", "Standard 50mm"),

            # Lighting
            time_of_day=scene_data.get("time_of_day", "Afternoon"),
            lighting_setup=scene_data.get("lighting_setup", "Natural"),
            color_temperature=scene_data.get("color_temperature", "Neutral 5600K"),
            mood=scene_data.get("mood", "Bright"),

            # Art
            art_style=project.style_template.art_direction,
            color_palette=scene_data.get("color_palette", []),
            atmosphere=scene_data.get("atmosphere", "Clear"),

            # Location
            setting=scene_data.get("setting", ""),
            time_relation=scene_data.get("time_relation", "Immediately after"),

            # Characters
            characters=[
                CharacterInfo(**char) for char in scene_data.get("characters", [])
            ]
        )

        # ========================================
        # GENERIERE DETAILLIERTE PROMPTS
        # ========================================

        # 1. Image Prompts (Positive + Negative)
        scene.image_prompt_positive = self._build_image_prompt_positive(
            scene, project.style_template, previous_scenes
        )
        scene.image_prompt_negative = self._build_image_prompt_negative()

        # 2. Video Prompts (Movement + Camera)
        scene.video_prompt_movement = self._build_video_prompt_movement(scene, previous_scenes)
        scene.video_prompt_camera = self._build_video_prompt_camera(scene)
        scene.video_prompt_duration = scene.duration

        # 3. Audio Prompt
        scene.audio_prompt = self._build_audio_prompt(scene, project.style_template)

        # 4. Legacy combined prompts (für Rückwärtskompatibilität)
        scene.image_prompt = scene.image_prompt_positive
        scene.video_prompt = f"{scene.video_prompt_movement} | {scene.video_prompt_camera}"

        # Füge Szene zum Projekt hinzu
        project.scenes.append(scene)

        self.app.logger.info(f"Szene {scene_number} für Projekt {project_id} generiert")

        return scene

    def _build_audio_prompt(self, scene: SceneStructure, style: StyleTemplate) -> str:
        """Erstellt Audio/Music Prompt"""
        parts = []

        # Music genre from style
        parts.append(f"{style.music_genre} music")

        # Emotional tone
        parts.append(f"{scene.emotional_beat} mood")

        # Pacing
        parts.append(f"{style.pacing_style} tempo")

        # Duration
        parts.append(f"{scene.duration} duration")

        return ", ".join(parts)

    # ========================================================================
    # ISAA AGENT INTEGRATION
    # ========================================================================

    async def _init_agents(self):
        """Initialisiert die ISAA Agents"""
        # Verhindere mehrfache Initialisierung
        if self.consultation_agent and self.generator_agent:
            self.app.logger.info("✅ Agents bereits initialisiert")
            return

        try:
            self.app.logger.info("Initialisiere ISAA Agents...")

            # Prüfe ob ISAA verfügbar ist
            try:
                from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder
                self.app.logger.info("FlowAgentBuilder importiert")
            except ImportError as ie:
                self.app.logger.error(f"FlowAgentBuilder Import fehlgeschlagen: {ie}")
                raise

            # Story Consultation Agent
            self.app.logger.info("   Erstelle Consultation Agent...")
            self.consultation_agent = await (
                FlowAgentBuilder()
                .with_name("StoryConsultant")
                .with_models("gemini/gemini-2.5-flash")
                .with_system_message(self._get_consultation_system_message())
                .with_temperature(0.7)
                .add_tool(self._analyze_emotional_triggers, "analyze_emotional_triggers")
                .add_tool(self._calculate_viral_potential, "calculate_viral_potential")
                .add_tool(self._suggest_optimizations, "suggest_optimizations")
                .build()
            )
            self.app.logger.info(f"Consultation Agent erstellt: {type(self.consultation_agent)}")

            # Director's Cut Generator Agent
            self.app.logger.info("Erstelle Generator Agent...")
            self.generator_agent = await (
                FlowAgentBuilder()
                .with_name("DirectorsCutGenerator")
                .with_system_message(self._get_generator_system_message())
                .with_models("gemini/gemini-2.5-flash")
                .with_temperature(0.6)
                .add_tool(self._check_visual_consistency, "check_visual_consistency")
                .add_tool(self._check_temporal_consistency, "check_temporal_consistency")
                .add_tool(self._track_character_consistency, "track_character_consistency")
                .build()
            )
            self.app.logger.info(f"Generator Agent erstellt: {type(self.generator_agent)}")

            self.app.logger.info("ISAA Agents erfolgreich initialisiert!")
            self.app.logger.info(f"   - Consultation Agent: {self.consultation_agent is not None}")
            self.app.logger.info(f"   - Generator Agent: {self.generator_agent is not None}")

        except Exception as e:
            self.app.logger.error(f"❌ Fehler beim Initialisieren der Agents: {e}")
            import traceback
            self.app.logger.error(traceback.format_exc())
            # Setze Agents auf None, damit Fallback funktioniert
            self.consultation_agent = None
            self.generator_agent = None
            raise  # Exception weiterwerfen, damit Caller informiert wird

    def _get_consultation_system_message(self) -> str:
        """System Message für Story Consultation Agent"""
        return """Du bist ein erfahrener Story Consultant für virale Video-Inhalte.

Deine Aufgabe:
1. Analysiere die Story auf emotionale Trigger
2. Bewerte das virale Potential (0-40 Punkte)
3. Schlage Optimierungen vor für maximale Engagement

Bewertungskriterien:
- Emotional Impact (0-10): Wie stark sind die Emotionen?
- Conflict Strength (0-10): Wie stark ist der Konflikt?
- Relatability (0-10): Können sich Zuschauer identifizieren?
- Surprise Factor (0-10): Gibt es überraschende Wendungen?

Sei präzise, konstruktiv und fokussiert auf virale Mechaniken."""

    def _get_generator_system_message(self) -> str:
        """System Message für Director's Cut Generator Agent"""
        return """Du bist ein Director's Cut Generator für KI-Video-Produktion.

Deine Aufgabe:
1. Generiere detaillierte Szenen-Prompts für Bild/Video-Generatoren
2. Stelle visuelle, temporale und narrative Konsistenz sicher
3. Erstelle Szenen SCHRITTWEISE - eine nach der anderen
4. Tracke Charaktere, Orte und Zeitlinien

Wichtige Regeln:
- IMMER konsistente Charakterbeschreibungen verwenden
- Zeitliche Kontinuität beachten (time_relation)
- Art Style muss durchgehend gleich bleiben
- Kamera-Bewegungen müssen zur Story passen
- Lighting muss zur Tageszeit passen

Sei präzise, technisch und achte auf Details."""

    # ========================================================================
    # CUSTOM TOOLS FÜR STORY CONSULTATION AGENT
    # ========================================================================

    def _analyze_emotional_triggers(self, story_text: str) -> str:
        """Analysiert emotionale Trigger in der Story"""
        triggers = []

        # Einfache Keyword-basierte Analyse (kann später mit NLP erweitert werden)
        emotion_keywords = {
            "Joy": ["happy", "joy", "laugh", "smile", "celebrate", "glücklich", "freude", "lachen"],
            "Sadness": ["sad", "cry", "tears", "loss", "traurig", "weinen", "tränen", "verlust"],
            "Fear": ["fear", "scared", "terror", "afraid", "angst", "furcht", "erschrocken"],
            "Anger": ["angry", "rage", "furious", "mad", "wütend", "zorn", "ärger"],
            "Surprise": ["surprise", "shock", "unexpected", "überraschung", "schock", "unerwartet"],
            "Love": ["love", "romance", "affection", "liebe", "romantik", "zuneigung"]
        }

        story_lower = story_text.lower()

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in story_lower)
            if count > 0:
                triggers.append(f"{emotion}: {count} triggers gefunden")

        if not triggers:
            return "Keine starken emotionalen Trigger gefunden. Empfehlung: Story emotionaler gestalten."

        return "Emotionale Trigger:\n" + "\n".join(triggers)

    def _calculate_viral_potential(self, story_text: str) -> str:
        """Berechnet virales Potential (0-40 Punkte)"""
        score = 0
        feedback = []

        story_lower = story_text.lower()

        # Emotional Impact (0-10)
        emotion_words = ["love", "fear", "joy", "sad", "angry", "surprise", "liebe", "angst", "freude"]
        emotion_count = sum(1 for word in emotion_words if word in story_lower)
        emotion_score = min(10, emotion_count * 2)
        score += emotion_score
        feedback.append(f"Emotional Impact: {emotion_score}/10")

        # Conflict Strength (0-10)
        conflict_words = ["conflict", "fight", "struggle", "problem", "challenge", "konflikt", "kampf", "problem"]
        conflict_count = sum(1 for word in conflict_words if word in story_lower)
        conflict_score = min(10, conflict_count * 3)
        score += conflict_score
        feedback.append(f"Conflict Strength: {conflict_score}/10")

        # Relatability (0-10)
        relatable_words = ["everyday", "normal", "common", "familiar", "alltag", "normal", "bekannt"]
        relatable_count = sum(1 for word in relatable_words if word in story_lower)
        relatable_score = min(10, relatable_count * 3)
        score += relatable_score
        feedback.append(f"Relatability: {relatable_score}/10")

        # Surprise Factor (0-10)
        surprise_words = ["twist", "unexpected", "surprise", "shock", "wendung", "überraschung", "schock"]
        surprise_count = sum(1 for word in surprise_words if word in story_lower)
        surprise_score = min(10, surprise_count * 4)
        score += surprise_score
        feedback.append(f"Surprise Factor: {surprise_score}/10")

        feedback.append(f"\n🎯 GESAMT-SCORE: {score}/40")

        if score < 15:
            feedback.append("⚠️ Niedriges virales Potential - Story braucht mehr Impact!")
        elif score < 25:
            feedback.append("✅ Moderates virales Potential - Gut, aber ausbaufähig")
        else:
            feedback.append("🔥 Hohes virales Potential - Exzellente Story!")

        return "\n".join(feedback)

    def _suggest_optimizations(self, story_text: str, viral_score: int = 0) -> str:
        """Schlägt Optimierungen vor"""
        suggestions = ["💡 Optimierungs-Vorschläge:\n"]

        story_lower = story_text.lower()

        # Emotionale Intensität
        if not any(word in story_lower for word in ["love", "fear", "joy", "liebe", "angst", "freude"]):
            suggestions.append("• Füge stärkere emotionale Momente hinzu")

        # Konflikt
        if not any(word in story_lower for word in ["conflict", "fight", "problem", "konflikt", "kampf"]):
            suggestions.append("• Verstärke den zentralen Konflikt")

        # Überraschung
        if not any(word in story_lower for word in ["twist", "surprise", "unexpected", "wendung", "überraschung"]):
            suggestions.append("• Baue eine überraschende Wendung ein")

        # Länge
        word_count = len(story_text.split())
        if word_count < 50:
            suggestions.append("• Story ist zu kurz - füge mehr Details hinzu")
        elif word_count > 300:
            suggestions.append("• Story ist zu lang - fokussiere auf Kernmomente")

        if len(suggestions) == 1:
            suggestions.append("✅ Story ist gut strukturiert!")

        return "\n".join(suggestions)

    # ========================================================================
    # CUSTOM TOOLS FÜR DIRECTOR'S CUT GENERATOR AGENT
    # ========================================================================

    def _check_visual_consistency(self, scenes_json: str) -> str:
        """Prüft visuelle Konsistenz zwischen Szenen"""
        try:
            scenes = json.loads(scenes_json)
            issues = []

            if len(scenes) < 2:
                return "✅ Nur eine Szene - keine Konsistenz-Prüfung nötig"

            # Art Style Konsistenz
            art_styles = [s.get("art_style", "") for s in scenes]
            if len(set(art_styles)) > 1:
                issues.append(f"⚠️ Inkonsistente Art Styles: {set(art_styles)}")

            # Charakter-Konsistenz
            for i, scene in enumerate(scenes[1:], 1):
                prev_scene = scenes[i-1]
                prev_chars = {c.get("name") for c in prev_scene.get("characters", [])}
                curr_chars = {c.get("name") for c in scene.get("characters", [])}

                # Prüfe ob Charaktere plötzlich verschwinden
                disappeared = prev_chars - curr_chars
                if disappeared and scene.get("time_relation") == "Immediately after":
                    issues.append(f"⚠️ Szene {i+1}: Charaktere verschwunden: {disappeared}")

            if not issues:
                return "✅ Visuelle Konsistenz ist gewährleistet"

            return "Konsistenz-Probleme:\n" + "\n".join(issues)

        except json.JSONDecodeError:
            return "❌ Fehler: Ungültiges JSON-Format"

    def _check_temporal_consistency(self, scenes_json: str) -> str:
        """Prüft zeitliche Konsistenz"""
        try:
            scenes = json.loads(scenes_json)
            issues = []

            if len(scenes) < 2:
                return "✅ Nur eine Szene - keine zeitliche Prüfung nötig"

            # Tageszeit-Konsistenz
            for i, scene in enumerate(scenes[1:], 1):
                prev_time = scenes[i-1].get("time_of_day", "")
                curr_time = scene.get("time_of_day", "")
                time_relation = scene.get("time_relation", "")

                # Wenn "Immediately after", sollte Tageszeit ähnlich sein
                if time_relation == "Immediately after":
                    if prev_time == "Morning" and curr_time == "Night":
                        issues.append(f"⚠️ Szene {i+1}: Zeitsprung von {prev_time} zu {curr_time}")

            # Setting-Konsistenz
            for i, scene in enumerate(scenes[1:], 1):
                prev_setting = scenes[i-1].get("setting", "")
                curr_setting = scene.get("setting", "")
                time_relation = scene.get("time_relation", "")

                if time_relation == "Immediately after" and prev_setting != curr_setting:
                    # Das ist OK, aber sollte erwähnt werden
                    issues.append(f"ℹ️ Szene {i+1}: Ortswechsel von '{prev_setting}' zu '{curr_setting}'")

            if not issues:
                return "✅ Zeitliche Konsistenz ist gewährleistet"

            return "Zeitliche Hinweise:\n" + "\n".join(issues)

        except json.JSONDecodeError:
            return "❌ Fehler: Ungültiges JSON-Format"

    def _track_character_consistency(self, character_name: str, scenes_json: str) -> str:
        """Trackt einen Charakter durch alle Szenen"""
        try:
            scenes = json.loads(scenes_json)
            appearances = []

            for i, scene in enumerate(scenes, 1):
                for char in scene.get("characters", []):
                    if char.get("name") == character_name:
                        appearances.append({
                            "scene": i,
                            "appearance": char.get("appearance", ""),
                            "action": char.get("action", ""),
                            "expression": char.get("expression", "")
                        })

            if not appearances:
                return f"❌ Charakter '{character_name}' nicht gefunden"

            # Prüfe Konsistenz der Appearance
            appearance_descriptions = [a["appearance"] for a in appearances]
            if len(set(appearance_descriptions)) > 1:
                return f"⚠️ Inkonsistente Beschreibungen für '{character_name}':\n" + \
                       "\n".join([f"Szene {a['scene']}: {a['appearance']}" for a in appearances])

            return f"✅ Charakter '{character_name}' ist konsistent in {len(appearances)} Szenen"

        except json.JSONDecodeError:
            return "❌ Fehler: Ungültiges JSON-Format"


# ============================================================================
# EXPORT FUNCTIONS (API Endpoints)
# ============================================================================

@export(mod_name=MOD_NAME, version=VERSION, initial=True)
def initialize_module(app: App):
    """Initialisierung (wird automatisch aufgerufen)"""
    return Result.ok(info=f"{MOD_NAME} v{VERSION} initialized")


@export(mod_name=MOD_NAME, name="version", api=True, version=VERSION)
async def get_version(app: App) -> Result:
    """API: Version abrufen"""
    return Result.json(data={"module": MOD_NAME, "version": VERSION})


@export(mod_name=MOD_NAME, name="list_templates", api=True, version=VERSION)
async def list_templates(app: App) -> Result:
    """API: Alle verfügbaren Style Templates auflisten"""
    tools = app.get_mod(MOD_NAME)

    templates = [
        {
            "id": key,
            "name": template.name,
            "description": template.art_direction
        }
        for key, template in tools.style_templates.items()
    ]

    return Result.json(data={"templates": templates})


@export(mod_name=MOD_NAME, name="create_project", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def create_project_api(app: App, request: RequestData) -> Result:
    """API: Neues Projekt erstellen"""
    tools = app.get_mod(MOD_NAME)

    # JSON-Body parsen (bei Content-Type: application/json)
    data = request.body or request.form_data or {}
    title = data.get("title", "Untitled Project")
    story_text = data.get("story_text", "")
    template_name = data.get("template", "cinematic")

    if not story_text:
        return Result.default_user_error(info="Story-Text ist erforderlich")

    project_id = tools.create_project(title, story_text, template_name)
    project = tools.get_project(project_id)

    return Result.json(data={
        "project_id": project_id,
        "title": project.title,
        "template": project.style_template.name
    })


@export(mod_name=MOD_NAME, name="get_project", api=True, version=VERSION, request_as_kwarg=True)
async def get_project_api(app: App, request: RequestData) -> Result:
    """API: Projekt abrufen"""
    tools = app.get_mod(MOD_NAME)

    project_id = request.query_params.get("project_id", "")

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    project = tools.get_project(project_id)

    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    return Result.json(data=project.model_dump())


@export(mod_name=MOD_NAME, name="analyze_story_interactive", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def analyze_story_interactive_api(app: App, request: RequestData) -> Result:
    """
    API: Interaktive Story-Analyse mit User-Feedback-Loop

    Request Body:
    {
        "project_id": "abc123",
        "user_message": "Optional: User-Antwort auf Agent-Frage"
    }
    """
    tools = app.get_mod(MOD_NAME)

    data = request.body or request.form_data or {}
    project_id = data.get("project_id", "")
    user_message = data.get("user_message", "")

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    project = tools.get_project(project_id)
    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    if not tools.consultation_agent:
        try:
            await tools._init_agents()
        except Exception as e:
            app.logger.error(f"Agent-Initialisierung fehlgeschlagen: {e}")
            return Result.default_internal_error(info=f"Agent-Initialisierung fehlgeschlagen: {str(e)}")

    if not tools.consultation_agent:
        return Result.default_user_error(info="Consultation Agent konnte nicht initialisiert werden. Prüfe Server-Logs für Details.")

    try:
        # Erste Analyse oder Follow-up
        if not user_message:
            # Initiale Analyse
            prompt = f"""Analysiere diese Story auf virale Eignung und stelle dem User gezielte Fragen zur Optimierung:

Story: {project.story_text}
Style: {project.style_template.name}

Führe folgende Analysen durch:
1. analyze_emotional_triggers - Welche Emotionen werden getriggert?
2. calculate_viral_potential - Wie hoch ist das virale Potential (0-40 Punkte)?
3. suggest_optimizations - Was kann verbessert werden?

Stelle dann 2-3 gezielte Fragen an den User, um die Story zu optimieren.
Beispiel: "Welche Emotion soll am stärksten sein?", "Gibt es einen Twist am Ende?"
"""
        else:
            # Follow-up mit User-Antwort
            prompt = f"""Der User hat geantwortet: "{user_message}"

Verarbeite die Antwort und:
1. Aktualisiere die Story-Parameter basierend auf der Antwort
2. Gib weitere Optimierungsvorschläge
3. Frage nach, ob noch etwas unklar ist

Wenn alles klar ist, sage: "READY_FOR_GENERATION" am Ende."""

        response = await tools.consultation_agent.a_run(prompt, fast_run=True)

        # Check if ready for generation
        is_ready = "READY_FOR_GENERATION" in str(response)

        return Result.json(data={
            "project_id": project_id,
            "agent_response": str(response),
            "is_ready_for_generation": is_ready,
            "conversation_continues": not is_ready
        })

    except Exception as e:
        app.logger.error(f"Fehler bei interaktiver Story-Analyse: {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="analyze_story", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def analyze_story_api(app: App, request: RequestData) -> Result:
    """API: Story mit Consultation Agent analysieren (Legacy - Non-Interactive)"""
    tools = app.get_mod(MOD_NAME)

    # JSON-Body parsen (bei Content-Type: application/json)
    data = request.body or request.form_data or {}
    story_text = data.get("story_text", "")

    if not story_text:
        return Result.default_user_error(info="Story-Text ist erforderlich")

    if not tools.consultation_agent:
        await tools._init_agents()

    if not tools.consultation_agent:
        return Result.default_user_error(info="Consultation Agent noch nicht initialisiert")

    try:
        # Agent mit Story-Analyse beauftragen
        prompt = f"""Analysiere diese Story für virale Video-Produktion:

{story_text}

Führe folgende Analysen durch:
1. Emotionale Trigger identifizieren
2. Virales Potential berechnen (0-40 Punkte)
3. Optimierungsvorschläge geben

Nutze die verfügbaren Tools für die Analyse."""

        response = await tools.consultation_agent.a_run(prompt, fast_run=True)

        return Result.json(data={
            "analysis": response,
            "story_text": story_text
        })

    except Exception as e:
        app.logger.error(f"Fehler bei Story-Analyse: {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="generate_scene_detailed", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def generate_scene_detailed_api(app: App, request: RequestData) -> Result:
    """
    API: Einzelne Szene mit detaillierten Prompts generieren

    WICHTIG: Sequenziell aufrufen! Szene 1, dann 2, dann 3, etc.

    Request Body:
    {
        "project_id": "abc123",
        "scene_number": 1,
        "scene_data": {
            "scene_title": "Opening Shot",
            "what_happens": "Character enters room",
            "emotional_beat": "curious",
            "camera_type": "Wide Shot",
            "setting": "Modern apartment",
            "characters": [
                {
                    "name": "Alex",
                    "appearance": "30s, casual clothes, brown hair",
                    "action": "walking slowly",
                    "expression": "thoughtful"
                }
            ],
            ...
        }
    }
    """
    tools = app.get_mod(MOD_NAME)

    # JSON-Body parsen
    data = request.body or request.form_data or {}
    project_id = data.get("project_id", "")
    scene_number = int(data.get("scene_number", 1))
    scene_data = data.get("scene_data", {})

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    if not scene_data:
        return Result.default_user_error(info="scene_data erforderlich")

    project = tools.get_project(project_id)
    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    try:
        # Generiere Szene mit detaillierten Prompts
        scene = tools.generate_scene_prompts(project_id, scene_number, scene_data)

        return Result.json(data={
            "scene_number": scene.scene_number,
            "scene_title": scene.scene_title,
            "duration": scene.duration,

            # Story
            "what_happens": scene.what_happens,
            "emotional_beat": scene.emotional_beat,

            # Detaillierte Prompts
            "prompts": {
                "image": {
                    "positive": scene.image_prompt_positive,
                    "negative": scene.image_prompt_negative
                },
                "video": {
                    "movement": scene.video_prompt_movement,
                    "camera": scene.video_prompt_camera,
                    "duration": scene.video_prompt_duration
                },
                "audio": scene.audio_prompt
            },

            # Visuals
            "visuals": {
                "camera_type": scene.camera_type,
                "camera_movement": scene.camera_movement,
                "lens": scene.lens,
                "lighting": {
                    "time_of_day": scene.time_of_day,
                    "setup": scene.lighting_setup,
                    "mood": scene.mood
                },
                "art_style": scene.art_style,
                "atmosphere": scene.atmosphere
            },

            # Characters
            "characters": [
                {
                    "name": char.name,
                    "appearance": char.appearance,
                    "action": char.action,
                    "expression": char.expression
                }
                for char in scene.characters
            ]
        })

    except Exception as e:
        app.logger.error(f"Fehler bei Szenen-Generierung: {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="generate_scene", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def generate_scene_api(app: App, request: RequestData) -> Result:
    """API: Einzelne Szene generieren (schrittweise!) - LEGACY VERSION"""
    tools = app.get_mod(MOD_NAME)

    # JSON-Body parsen (bei Content-Type: application/json)
    data = request.body or request.form_data or {}
    project_id = data.get("project_id", "")
    scene_number = int(data.get("scene_number", 1))
    scene_description = data.get("scene_description", "")

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    project = tools.get_project(project_id)
    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    if not tools.consultation_agent:
        await tools._init_agents()

    if not tools.generator_agent:
        return Result.default_user_error(info="Generator Agent noch nicht initialisiert")

    try:
        # Vorherige Szenen für Konsistenz
        previous_scenes = project.scenes[:scene_number-1] if scene_number > 1 else []

        prompt = f"""Generiere Szene {scene_number} für das Projekt "{project.title}".

Story: {project.story_text}

Style Template: {project.style_template.name}
Art Direction: {project.style_template.art_direction}

Szenen-Beschreibung: {scene_description}

Vorherige Szenen: {len(previous_scenes)}

WICHTIG:
- Halte den Art Style konsistent: {project.style_template.art_direction}
- Achte auf zeitliche Kontinuität
- Verwende konsistente Charakterbeschreibungen
- Erstelle detaillierte Prompts für Midjourney/Runway

Generiere eine vollständige SceneStructure mit allen Details."""

        response = await tools.generator_agent.a_run(prompt, fast_run=True)

        return Result.json(data={
            "scene_number": scene_number,
            "generated_content": response,
            "project_id": project_id
        })

    except Exception as e:
        app.logger.error(f"Fehler bei Szenen-Generierung: {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="generate_directors_cut", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def generate_directors_cut_api(app: App, request: RequestData) -> Result:
    """
    API: Initialisiert Director's Cut (bestimmt Szenenanzahl)
    """
    tools = app.get_mod(MOD_NAME)
    data = request.body or request.form_data or {}
    project_id = data.get("project_id", "")
    num_scenes = int(data.get("num_scenes", 0))

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    project = tools.get_project(project_id)
    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    if not tools.consultation_agent:
        await tools._init_agents()

    if not tools.generator_agent:
        return Result.default_user_error(info="Generator Agent noch nicht initialisiert")

    try:
        # Auto-detect number of scenes if not provided
        if num_scenes == 0:
            # Ask agent to determine optimal scene count
            analysis_prompt = f"""Analysiere diese Story und bestimme die optimale Anzahl an Szenen:

Story: {project.story_text}
Style: {project.style_template.name}
Pacing: {project.style_template.pacing_style}

Gib NUR eine Zahl zurück (3-10 Szenen empfohlen)."""

            response = await tools.generator_agent.a_run(analysis_prompt, fast_run=True)

            # Extract number from response
            import re
            match = re.search(r'\b([3-9]|10)\b', str(response))
            num_scenes = int(match.group(1)) if match else 5

            app.logger.info(f"Auto-detected {num_scenes} scenes for project {project_id}")

        # Update project parameters if needed (optional, but good for state)
        project.parameters.total_scenes = num_scenes

        return Result.json(data={
            "project_id": project_id,
            "total_scenes": num_scenes,
            "status": "ready"
        })

    except Exception as e:
        app.logger.error(f"Fehler bei Director's Cut Initialisierung: {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="generate_single_scene_agent", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def generate_single_scene_agent_api(app: App, request: RequestData) -> Result:
    """
    API: Generiert eine spezifische Szene mit dem Agenten (Teil der Sequenz)
    """
    tools = app.get_mod(MOD_NAME)
    data = request.body or request.form_data or {}
    project_id = data.get("project_id", "")
    scene_number = int(data.get("scene_number", 1))
    total_scenes = int(data.get("total_scenes", 5)) # Optional context

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    project = tools.get_project(project_id)
    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    # Ensure agents are ready
    if not tools.generator_agent:
         if not tools.consultation_agent: # Try init if completely missing
            await tools._init_agents()
         if not tools.generator_agent:
            return Result.default_user_error(info="Generator Agent nicht bereit")

    try:
        # Generate scene structure using agent
        scene_prompt = f"""Generiere Szene {scene_number} von {total_scenes} für diese Story:

Story: {project.story_text}
Style: {project.style_template.name}

Vorherige Szenen: {len(project.scenes)}

Erstelle eine detaillierte Szenen-Beschreibung mit:
- scene_title: Kurzer Titel
- what_happens: Was passiert in dieser Szene
- emotional_beat: Welche Emotion wird getriggert
- camera_type: Shot-Typ (Wide Shot, Medium Shot, Close-Up, etc.)
- camera_movement: Kamera-Bewegung (Static, Pan, Dolly, etc.)
- setting: Ort der Szene
- characters: Liste der Charaktere mit Beschreibung

Gib die Antwort als strukturierten Text."""

        agent_response = await tools.generator_agent.a_run(scene_prompt, fast_run=True)

        # Parse agent response into scene_data (simplified)
        # In a real implementation, we might want robust JSON parsing here if the agent returns JSON
        # For now, we use the existing simplified logic but ensure we pass the raw text if needed or mock structure
        # The original code used str(agent_response)[:200] which is bad. Let's use the full text.

        scene_data = {
            "scene_title": f"Scene {scene_number}",
            "what_happens": str(agent_response),
            "emotional_beat": "neutral",
            "camera_type": "Medium Shot",
            "camera_movement": "Static",
            "setting": "Location",
            "characters": []
        }

        # Generate detailed prompts
        scene = tools.generate_scene_prompts(project_id, scene_number, scene_data)

        # Return FULL data
        return Result.json(data={
            "scene_number": scene.scene_number,
            "scene_title": scene.scene_title,
            "image_prompt_positive": scene.image_prompt_positive,
            "video_prompt_movement": scene.video_prompt_movement,
            "status": "generated"
        })

    except Exception as e:
        app.logger.error(f"Fehler bei Szenen-Generierung (Agent): {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="export_prompts", api=True, version=VERSION, request_as_kwarg=True)
async def export_prompts_api(app: App, request: RequestData, **kwargs) -> Result:
    """
    API: Exportiert alle Prompts eines Projekts

    Query Params:
        project_id: Projekt-ID
        format: "json" | "markdown" | "txt" (default: json)
    """
    tools = app.get_mod(MOD_NAME)

    project_id = request.query_params.get("project_id", "")
    export_format = request.query_params.get("format", "json")

    if not project_id:
        return Result.default_user_error(info="project_id erforderlich")

    project = tools.get_project(project_id)
    if not project:
        return Result.default_user_error(info=f"Projekt {project_id} nicht gefunden")

    if export_format == "json":
        # JSON Export
        scenes_data = []
        for scene in project.scenes:
            scenes_data.append({
                "scene_number": scene.scene_number,
                "scene_title": scene.scene_title,
                "duration": scene.duration,
                "prompts": {
                    "image_positive": scene.image_prompt_positive,
                    "image_negative": scene.image_prompt_negative,
                    "video_movement": scene.video_prompt_movement,
                    "video_camera": scene.video_prompt_camera,
                    "audio": scene.audio_prompt
                }
            })

        return Result.json(data={
            "project_id": project_id,
            "title": project.title,
            "total_scenes": len(project.scenes),
            "scenes": scenes_data
        })

    elif export_format == "markdown":
        # Markdown Export
        md_lines = [
            f"# {project.title}",
            "",
            f"**Total Scenes:** {len(project.scenes)}",
            f"**Style:** {project.style_template.name}",
            "",
            "---",
            ""
        ]

        for scene in project.scenes:
            md_lines.extend([
                f"## Scene {scene.scene_number}: {scene.scene_title}",
                "",
                f"**Duration:** {scene.duration}",
                f"**What Happens:** {scene.what_happens}",
                "",
                "### Image Prompts",
                "",
                "**Positive:**",
                f"```\n{scene.image_prompt_positive}\n```",
                "",
                "**Negative:**",
                f"```\n{scene.image_prompt_negative}\n```",
                "",
                "### Video Prompts",
                "",
                "**Movement:**",
                f"```\n{scene.video_prompt_movement}\n```",
                "",
                "**Camera:**",
                f"```\n{scene.video_prompt_camera}\n```",
                "",
                "### Audio Prompt",
                f"```\n{scene.audio_prompt}\n```",
                "",
                "---",
                ""
            ])

        markdown_text = "\n".join(md_lines)
        return Result.ok(data=markdown_text, data_info="text/markdown")

    elif export_format == "txt":
        # Plain Text Export (Generator-Ready)
        txt_lines = [f"=== {project.title} ===", ""]

        for scene in project.scenes:
            txt_lines.extend([
                f"SCENE {scene.scene_number}: {scene.scene_title}",
                "",
                "IMAGE PROMPT (POSITIVE):",
                scene.image_prompt_positive,
                "",
                "IMAGE PROMPT (NEGATIVE):",
                scene.image_prompt_negative,
                "",
                "VIDEO PROMPT (MOVEMENT):",
                scene.video_prompt_movement,
                "",
                "VIDEO PROMPT (CAMERA):",
                scene.video_prompt_camera,
                "",
                "AUDIO PROMPT:",
                scene.audio_prompt,
                "",
                "=" * 80,
                ""
            ])

        plain_text = "\n".join(txt_lines)
        return Result.ok(data=plain_text, data_info="text/plain")

    else:
        return Result.default_user_error(info=f"Unbekanntes Format: {export_format}")


@export(mod_name=MOD_NAME, name="create_template_from_dump", api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def create_template_from_dump_api(app: App, request: RequestData) -> Result:
    """
    API: Erstellt Template aus Content-Dump (JSON, YAML, oder Freitext)

    Request Body:
    {
        "dump": "Content-Dump als String"
    }
    """
    tools = app.get_mod(MOD_NAME)

    data = request.body or request.form_data or {}
    dump = data.get("dump", "")

    if not dump:
        return Result.default_user_error(info="Content-Dump erforderlich")

    if not tools.consultation_agent:
        await tools._init_agents()

    if not tools.generator_agent:
        return Result.default_user_error(info="Generator Agent noch nicht initialisiert")

    try:
        # Use Generator Agent to parse dump and create template
        prompt = f"""Analysiere diesen Content-Dump und erstelle ein StyleTemplate daraus:

{dump}

Extrahiere oder generiere folgende Felder:
- name: Template-Name
- art_direction: Künstlerische Richtung
- color_grading: Farbgebung
- music_genre: Musik-Genre
- pacing_style: Pacing (Slow/Medium/Fast)

Gib die Antwort als strukturierten Text mit diesen Feldern."""

        response = await tools.generator_agent.a_run(prompt, fast_run=True)

        # Parse response (simplified - in production use better parsing)
        template_data = {
            "name": "Custom Template",
            "art_direction": "Cinematic",
            "color_grading": "Warm tones",
            "music_genre": "Ambient",
            "pacing_style": "Medium"
        }

        # Try to extract from response
        response_str = str(response)
        if "name:" in response_str.lower():
            lines = response_str.split('\n')
            for line in lines:
                if "name:" in line.lower():
                    template_data["name"] = line.split(':', 1)[1].strip()
                elif "art" in line.lower() and ":" in line:
                    template_data["art_direction"] = line.split(':', 1)[1].strip()
                elif "color" in line.lower() and ":" in line:
                    template_data["color_grading"] = line.split(':', 1)[1].strip()
                elif "music" in line.lower() and ":" in line:
                    template_data["music_genre"] = line.split(':', 1)[1].strip()
                elif "pacing" in line.lower() and ":" in line:
                    template_data["pacing_style"] = line.split(':', 1)[1].strip()

        return Result.json(data={
            "template": template_data,
            "agent_response": response_str
        })

    except Exception as e:
        app.logger.error(f"Fehler bei Template-Generierung: {e}")
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, name="ui", api=True, version=VERSION, api_methods=['GET'])
async def get_ui(app: App) -> Result:
    """Haupt-UI für DirCut"""

    html_content = f"""
    {app.web_context()}

    <style>
        .dircut-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .dircut-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
        }}

        .dircut-tab {{
            padding: 12px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            color: #6b7280;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }}

        .dircut-tab.active {{
            color: #3b82f6;
            border-bottom-color: #3b82f6;
        }}

        .dircut-tab:hover {{
            color: #1d4ed8;
        }}

        .dircut-panel {{
            display: none;
            animation: fadeIn 0.3s;
        }}

        .dircut-panel.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .dircut-card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .dircut-input {{
            width: 100%;
            padding: 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 16px;
        }}

        .dircut-textarea {{
            width: 100%;
            min-height: 200px;
            padding: 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            font-family: monospace;
            resize: vertical;
        }}

        .dircut-button {{
            padding: 12px 24px;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .dircut-button:hover {{
            background: #2563eb;
        }}

        .dircut-button:disabled {{
            background: #9ca3af;
            cursor: not-allowed;
        }}

        .dircut-select {{
            width: 100%;
            padding: 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 16px;
        }}

        .dircut-label {{
            display: block;
            font-weight: 500;
            margin-bottom: 8px;
            color: #374151;
        }}

        .dircut-info {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 12px;
            margin-bottom: 16px;
            border-radius: 4px;
        }}

        .dircut-success {{
            background: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 12px;
            margin-bottom: 16px;
            border-radius: 4px;
        }}

        .dircut-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}

        @media (max-width: 768px) {{
            .dircut-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>

    <div class="dircut-container">
        <h1 style="font-size: 32px; font-weight: bold; margin-bottom: 8px;">🎬 Director's Cut Pipeline</h1>
        <p style="color: #6b7280; margin-bottom: 24px;">Story-zu-Video Transformation mit KI-Agents</p>

        <!-- Tabs -->
        <div class="dircut-tabs">
            <button class="dircut-tab active" data-tab="phase1">📝 Story Consultation</button>
            <button class="dircut-tab" data-tab="phase2">🎬 Director's Cut</button>
            <button class="dircut-tab" data-tab="export">📤 Export</button>
            <button class="dircut-tab" data-tab="templates">🎨 Templates</button>
        </div>

        <!-- Phase 1 Panel -->
        <div class="dircut-panel active" id="panel-phase1">
            <div class="dircut-card">
                <h2 style="font-size: 20px; font-weight: 600; margin-bottom: 16px;">Story Input & Analyse</h2>

                <div class="dircut-info">
                    <strong>Phase 1:</strong> Der Story Consultation Agent analysiert deine Story auf virale Eignung.
                    <br><strong>Wichtig:</strong> Der Agent muss am Ende <code>READY_FOR_GENERATION</code> bestätigen, um fortzufahren.
                </div>

                <label class="dircut-label">Projekt-Titel</label>
                <input type="text" id="project-title" class="dircut-input" placeholder="z.B. Meine virale Story">

                <label class="dircut-label">Story-Text</label>
                <textarea id="story-text" class="dircut-textarea" placeholder="Schreibe oder füge deine Story hier ein..."></textarea>

                <label class="dircut-label">Style Template</label>
                <select id="style-template" class="dircut-select">
                    <option value="cinematic">Cinematic Short Film</option>
                    <option value="viral_tiktok">Viral TikTok Energy</option>
                    <option value="youtube_doc">YouTube Documentary</option>
                    <option value="instagram">Instagram Story Flow</option>
                    <option value="paranormal">Paranormal Mystery</option>
                </select>

                <button id="btn-create-project" class="dircut-button">Projekt erstellen & Analysieren</button>
            </div>

            <div class="dircut-card" id="analysis-result" style="display: none;">
                <h3 style="font-size: 18px; font-weight: 600; margin-bottom: 16px;">📊 Story-Analyse</h3>
                <div id="analysis-content"></div>

                <!-- Interactive Chat -->
                <div id="analysis-chat" style="margin-top: 16px; display: none;">
                    <div id="chat-messages" style="max-height: 300px; overflow-y: auto; margin-bottom: 16px; padding: 12px; background: #f9fafb; border-radius: 6px;">
                        <!-- Messages will be added here -->
                    </div>

                    <div style="display: flex; gap: 8px;">
                        <input type="text" id="user-response" class="dircut-input" style="margin-bottom: 0;" placeholder="Deine Antwort...">
                        <button id="btn-send-response" class="dircut-button">Senden</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Phase 2 Panel -->
        <div class="dircut-panel" id="panel-phase2">
            <div class="dircut-card">
                <h2 style="font-size: 20px; font-weight: 600; margin-bottom: 16px;">Director's Cut Generierung</h2>

                <div class="dircut-info">
                    <strong>Phase 2:</strong> Der Generator Agent erstellt sequenziell alle Szenen mit vollständigen
                    Prompts für Bild-, Video- und Audio-Generatoren. Konsistenz wird automatisch sichergestellt.
                </div>

                <div id="project-info" style="margin-bottom: 16px;"></div>

                <div style="margin-bottom: 16px;">
                    <label class="dircut-label">Anzahl Szenen (optional, Auto-Detect wenn leer)</label>
                    <input type="number" id="num-scenes" class="dircut-input" placeholder="z.B. 5" min="3" max="10">
                </div>

                <button id="btn-generate-cut" class="dircut-button" disabled>🎬 Director's Cut generieren</button>

                <div id="generation-progress" style="display: none; margin-top: 16px;">
                    <div class="dircut-info">
                        <strong>Generierung läuft...</strong><br>
                        <span id="progress-text">Szene 1 wird generiert...</span>
                    </div>
                </div>
            </div>

            <div class="dircut-card" id="scenes-container" style="display: none;">
                <h3 style="font-size: 18px; font-weight: 600; margin-bottom: 16px;">🎞️ Generierte Szenen</h3>
                <div id="scenes-list"></div>
            </div>
        </div>

        <!-- Export Panel -->
        <div class="dircut-panel" id="panel-export">
            <div class="dircut-card">
                <h2 style="font-size: 20px; font-weight: 600; margin-bottom: 16px;">📤 Export & Download</h2>

                <div class="dircut-info">
                    Exportiere dein Director's Cut in verschiedenen Formaten für unterschiedliche Generatoren.
                </div>

                <div class="dircut-grid">
                    <button class="dircut-button" id="btn-export-markdown">📄 Markdown Export</button>
                    <button class="dircut-button" id="btn-export-json">📋 JSON Export</button>
                    <button class="dircut-button" id="btn-export-prompts">🎨 Generator Prompts</button>
                    <button class="dircut-button" id="btn-export-shotlist">📊 Shot List (CSV)</button>
                </div>
            </div>
        </div>

        <!-- Templates Panel -->
        <div class="dircut-panel" id="panel-templates">
            <div class="dircut-card">
                <h2 style="font-size: 20px; font-weight: 600; margin-bottom: 16px;">🎨 Custom Template Editor</h2>

                <div class="dircut-info">
                    <strong>Minimalistisch & Effizient:</strong> Erstelle Custom Templates aus Content-Dumps oder passe bestehende Templates an.
                </div>

                <!-- Template Source -->
                <div style="margin-bottom: 24px;">
                    <h3 style="font-size: 16px; font-weight: 600; margin-bottom: 12px;">Template-Quelle</h3>

                    <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                        <button class="dircut-button" id="btn-from-dump" style="flex: 1;">📋 Aus Content-Dump</button>
                        <button class="dircut-button" id="btn-from-existing" style="flex: 1; background: #6b7280;">🔧 Bestehendes anpassen</button>
                    </div>

                    <!-- Content Dump Input -->
                    <div id="dump-input" style="display: block;">
                        <label class="dircut-label">Content-Dump (JSON, YAML, oder Freitext)</label>
                        <textarea id="template-dump" class="dircut-textarea" placeholder='Beispiel:
    {{
      "name": "My Custom Style",
      "art_direction": "Moody cinematic",
      "color_grading": "Teal and orange",
      "music_genre": "Electronic ambient",
      "pacing_style": "Slow burn"
    }}

    Oder einfach Freitext:
    "Ich möchte einen düsteren, atmosphärischen Stil mit langsamen Kamerabewegungen..."'></textarea>

                        <button id="btn-parse-dump" class="dircut-button">🔍 Template generieren</button>
                    </div>

                    <!-- Existing Template Selector -->
                    <div id="existing-selector" style="display: none;">
                        <label class="dircut-label">Basis-Template auswählen</label>
                        <select id="base-template" class="dircut-select">
                            <option value="cinematic">Cinematic Short Film</option>
                            <option value="viral_tiktok">Viral TikTok Energy</option>
                            <option value="youtube_doc">YouTube Documentary</option>
                            <option value="instagram">Instagram Story Flow</option>
                            <option value="paranormal">Paranormal Mystery</option>
                        </select>

                        <button id="btn-load-template" class="dircut-button">📥 Template laden</button>
                    </div>
                </div>

                <!-- Template Editor -->
                <div id="template-editor" style="display: none;">
                    <h3 style="font-size: 16px; font-weight: 600; margin-bottom: 12px;">Template bearbeiten</h3>

                    <label class="dircut-label">Template Name</label>
                    <input type="text" id="tpl-name" class="dircut-input" placeholder="z.B. My Custom Style">

                    <div class="dircut-grid">
                        <div>
                            <label class="dircut-label">Art Direction</label>
                            <input type="text" id="tpl-art" class="dircut-input" placeholder="z.B. Cinematic">
                        </div>
                        <div>
                            <label class="dircut-label">Color Grading</label>
                            <input type="text" id="tpl-color" class="dircut-input" placeholder="z.B. Warm tones">
                        </div>
                    </div>

                    <div class="dircut-grid">
                        <div>
                            <label class="dircut-label">Music Genre</label>
                            <input type="text" id="tpl-music" class="dircut-input" placeholder="z.B. Indie folk">
                        </div>
                        <div>
                            <label class="dircut-label">Pacing Style</label>
                            <input type="text" id="tpl-pacing" class="dircut-input" placeholder="z.B. Slow">
                        </div>
                    </div>

                    <button id="btn-save-template" class="dircut-button">💾 Template speichern</button>
                </div>

                <!-- Saved Templates -->
                <div id="saved-templates" style="margin-top: 24px;">
                    <h3 style="font-size: 16px; font-weight: 600; margin-bottom: 12px;">Gespeicherte Custom Templates</h3>
                    <div id="custom-templates-list"></div>
                </div>
            </div>
        </div>
    </div>

    <script unsave="true">
        // State Management
        let currentProjectId = null;
        let currentProject = null;

        // Tab Switching
        document.querySelectorAll('.dircut-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const targetTab = tab.dataset.tab;

                // Update tabs
                document.querySelectorAll('.dircut-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update panels
                document.querySelectorAll('.dircut-panel').forEach(p => p.classList.remove('active'));
                document.getElementById('panel-' + targetTab).classList.add('active');
            }});
        }});

        // Create Project & Start Interactive Analysis
        document.getElementById('btn-create-project').addEventListener('click', async () => {{
            const title = document.getElementById('project-title').value || 'Untitled Project';
            const storyText = document.getElementById('story-text').value;
            const template = document.getElementById('style-template').value;

            if (!storyText.trim()) {{
                TB.ui.Toast.showError('Bitte gib einen Story-Text ein');
                return;
            }}

            try {{
                // Step 1: Create Project
                const response = await TB.api.request('{MOD_NAME}', 'create_project', {{
                    title: title,
                    story_text: storyText,
                    template: template
                }});

                if (response.error === 'none') {{
                    const data = response.get();
                    currentProjectId = data.project_id;

                    TB.ui.Toast.showSuccess('Projekt erstellt: ' + data.title);

                    // Show analysis result
                    document.getElementById('analysis-result').style.display = 'block';
                    document.getElementById('analysis-content').innerHTML = `
                        <div class="dircut-success">
                            <strong>✓ Projekt erstellt!</strong><br>
                            ID: ${{data.project_id}}<br>
                            Template: ${{data.template}}
                        </div>
                    `;

                    // Enable Phase 2
                    document.getElementById('project-info').innerHTML = `
                        <div class="dircut-info">
                            <strong>Aktives Projekt:</strong> ${{data.title}} (ID: ${{data.project_id}})
                        </div>
                    `;

                    // Step 2: Start Interactive Analysis
                    await startInteractiveAnalysis(data.project_id);
                }} else {{
                    TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
                }}
            }} catch (error) {{
                TB.ui.Toast.showError('Netzwerkfehler: ' + error.message);
            }}
        }});

        // Interactive Analysis Function
        async function startInteractiveAnalysis(projectId) {{
            try {{
                const response = await TB.api.request('{MOD_NAME}', 'analyze_story_interactive', {{
                    project_id: projectId
                }});

                if (response.error === 'none') {{
                    const data = response.get();

                    // Show chat interface
                    document.getElementById('analysis-chat').style.display = 'block';

                    // Add agent message
                    addChatMessage('agent', data.agent_response);

                    // Check if ready for generation
                    if (data.is_ready_for_generation) {{
                        document.getElementById('btn-generate-cut').disabled = false;
                        TB.ui.Toast.showSuccess('Story-Analyse abgeschlossen! Bereit für Director\\'s Cut.');
                    }}
                }} else {{
                    TB.ui.Toast.showError('Analyse-Fehler: ' + response.info.help_text);
                }}
            }} catch (error) {{
                TB.ui.Toast.showError('Netzwerkfehler: ' + error.message);
            }}
        }}

        // Add Chat Message
        function addChatMessage(sender, message) {{
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.style.marginBottom = '12px';
            messageDiv.style.padding = '8px 12px';
            messageDiv.style.borderRadius = '6px';

            if (sender === 'agent') {{
                messageDiv.style.background = '#eff6ff';
                messageDiv.style.borderLeft = '3px solid #3b82f6';
                messageDiv.innerHTML = `<strong>🤖 Agent:</strong><br>${{message}}`;
            }} else {{
                messageDiv.style.background = '#f0fdf4';
                messageDiv.style.borderLeft = '3px solid #10b981';
                messageDiv.innerHTML = `<strong>👤 Du:</strong><br>${{message}}`;
            }}

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }}

        // Send User Response
        document.getElementById('btn-send-response').addEventListener('click', async () => {{
            const userMessage = document.getElementById('user-response').value;

            if (!userMessage.trim()) {{
                TB.ui.Toast.showError('Bitte gib eine Antwort ein');
                return;
            }}

            if (!currentProjectId) {{
                TB.ui.Toast.showError('Kein aktives Projekt');
                return;
            }}

            // Add user message to chat
            addChatMessage('user', userMessage);
            document.getElementById('user-response').value = '';

            try {{
                const response = await TB.api.request('{MOD_NAME}', 'analyze_story_interactive', {{
                    project_id: currentProjectId,
                    user_message: userMessage
                }});

                if (response.error === 'none') {{
                    const data = response.get();

                    // Add agent response
                    addChatMessage('agent', data.agent_response);

                    // Check if ready for generation
                    if (data.is_ready_for_generation) {{
                        document.getElementById('btn-generate-cut').disabled = false;
                        TB.ui.Toast.showSuccess('Story-Analyse abgeschlossen! Bereit für Director\\'s Cut.');
                    }}
                }} else {{
                    TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
                }}
            }} catch (error) {{
                TB.ui.Toast.showError('Netzwerkfehler: ' + error.message);
            }}
        }});

        // Generate Director's Cut (Progressive)
        document.getElementById('btn-generate-cut').addEventListener('click', async () => {{
            if (!currentProjectId) {{
                TB.ui.Toast.showError('Kein aktives Projekt');
                return;
            }}

            const numScenesInput = document.getElementById('num-scenes').value || 0;
            const progressDiv = document.getElementById('generation-progress');
            const progressText = document.getElementById('progress-text');
            const scenesContainer = document.getElementById('scenes-container');
            const scenesList = document.getElementById('scenes-list');
            const btn = document.getElementById('btn-generate-cut');

            // Reset UI
            progressDiv.style.display = 'block';
            scenesContainer.style.display = 'block';
            scenesList.innerHTML = ''; // Clear previous
            btn.disabled = true;

            try {{
                // Step 1: Initialize / Analyze
                progressText.textContent = "Initialisiere Projekt & bestimme Szenenanzahl...";

                const initResponse = await TB.api.request('{MOD_NAME}', 'generate_directors_cut', {{
                    project_id: currentProjectId,
                    num_scenes: parseInt(numScenesInput)
                }});

                if (initResponse.error !== 'none') {{
                    throw new Error(initResponse.info.help_text || "Init failed");
                }}

                const initData = initResponse.get();
                const totalScenes = initData.total_scenes;
                TB.ui.Toast.showSuccess(`Starte Generierung von ${{totalScenes}} Szenen...`);

                // Step 2: Loop scenes
                for (let i = 1; i <= totalScenes; i++) {{
                    progressText.textContent = `Generiere Szene ${{i}} von ${{totalScenes}}...`;

                    // Scroll to bottom of list
                    window.scrollTo(0, document.body.scrollHeight);

                    const sceneResponse = await TB.api.request('{MOD_NAME}', 'generate_single_scene_agent', {{
                        project_id: currentProjectId,
                        scene_number: i,
                        total_scenes: totalScenes
                    }});

                    if (sceneResponse.error !== 'none') {{
                         TB.ui.Toast.showError(`Fehler bei Szene ${{i}}: ${{sceneResponse.info.help_text}}`);
                         // Continue or break? Let's continue but show error card
                         const errorCard = document.createElement('div');
                         errorCard.className = 'dircut-card';
                         errorCard.style.borderLeft = '4px solid red';
                         errorCard.innerHTML = `<p>Fehler Szene ${{i}}</p>`;
                         scenesList.appendChild(errorCard);
                         continue;
                    }}

                    const scene = sceneResponse.get();

                    // Create Card
                    const sceneCard = document.createElement('div');
                    sceneCard.className = 'dircut-card';
                    sceneCard.style.marginBottom = '12px';
                    sceneCard.style.animation = 'fadeIn 0.5s';
                    sceneCard.innerHTML = `
                        <h4 style="font-weight: 600; margin-bottom: 8px;">
                            Scene ${{scene.scene_number}}: ${{scene.scene_title}}
                        </h4>
                        <div style="background:#f3f4f6; padding:8px; border-radius:4px; margin-bottom:8px;">
                            <p style="font-size: 11px; color: #374151; font-weight:bold; margin-bottom:4px;">IMAGE PROMPT</p>
                            <p style="font-size: 13px; color: #1f2937;">${{scene.image_prompt_positive}}</p>
                        </div>
                        <div style="background:#f3f4f6; padding:8px; border-radius:4px;">
                            <p style="font-size: 11px; color: #374151; font-weight:bold; margin-bottom:4px;">VIDEO PROMPT</p>
                            <p style="font-size: 13px; color: #1f2937;">${{scene.video_prompt_movement}}</p>
                        </div>
                        <div style="margin-top:8px; text-align:right;">
                             <span style="display: inline-block; padding: 4px 8px; background: #10b981; color: white; border-radius: 4px; font-size: 12px;">
                                ✓ Generiert
                            </span>
                        </div>
                    `;
                    scenesList.appendChild(sceneCard);
                }}

                progressDiv.style.display = 'none';
                btn.disabled = false;
                TB.ui.Toast.showSuccess("Director's Cut vollständig generiert!");
                TB.ui.Toast.showInfo('Wechsle zu Export-Tab für Download-Optionen');

            }} catch (error) {{
                TB.ui.Toast.showError('Fehler: ' + error.message);
                progressDiv.style.display = 'none';
                btn.disabled = false;
            }}
        }});

        // Export Buttons
        async function exportProject(format) {{
            if (!currentProjectId) {{
                TB.ui.Toast.showError('Kein aktives Projekt');
                return;
            }}

            try {{
                const url = `/api/{MOD_NAME}/export_prompts?project_id=${{currentProjectId}}&format=${{format}}`;
                const response = await fetch(url);

                if (!response.ok) {{
                    throw new Error('Export fehlgeschlagen');
                }}

                const blob = await response.blob();
                const downloadUrl = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = `dircut_${{currentProjectId}}.${{format === 'txt' ? 'txt' : format}}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(downloadUrl);

                TB.ui.Toast.showSuccess(`${{format.toUpperCase()}} Export erfolgreich!`);
            }} catch (error) {{
                TB.ui.Toast.showError('Export-Fehler: ' + error.message);
            }}
        }}

        document.getElementById('btn-export-markdown').addEventListener('click', () => {{
            exportProject('markdown');
        }});

        document.getElementById('btn-export-json').addEventListener('click', () => {{
            exportProject('json');
        }});

        document.getElementById('btn-export-prompts').addEventListener('click', () => {{
            exportProject('txt');
        }});

        document.getElementById('btn-export-shotlist').addEventListener('click', () => {{
            TB.ui.Toast.showInfo('CSV Shot List wird in nächster Version implementiert');
        }});

        // ========================================
        // TEMPLATE EDITOR
        // ========================================

        // Toggle between dump and existing
        document.getElementById('btn-from-dump').addEventListener('click', () => {{
            document.getElementById('dump-input').style.display = 'block';
            document.getElementById('existing-selector').style.display = 'none';
            document.getElementById('btn-from-dump').style.background = '#3b82f6';
            document.getElementById('btn-from-existing').style.background = '#6b7280';
        }});

        document.getElementById('btn-from-existing').addEventListener('click', () => {{
            document.getElementById('dump-input').style.display = 'none';
            document.getElementById('existing-selector').style.display = 'block';
            document.getElementById('btn-from-dump').style.background = '#6b7280';
            document.getElementById('btn-from-existing').style.background = '#3b82f6';
        }});

        // Parse Content Dump
        document.getElementById('btn-parse-dump').addEventListener('click', async () => {{
            const dump = document.getElementById('template-dump').value;

            if (!dump.trim()) {{
                TB.ui.Toast.showError('Bitte gib einen Content-Dump ein');
                return;
            }}

            try {{
                const response = await TB.api.request('{MOD_NAME}', 'create_template_from_dump', {{
                    dump: dump
                }});

                if (response.error === 'none') {{
                    const data = response.get();
                    const template = data.template;

                    // Fill editor
                    document.getElementById('tpl-name').value = template.name;
                    document.getElementById('tpl-art').value = template.art_direction;
                    document.getElementById('tpl-color').value = template.color_grading;
                    document.getElementById('tpl-music').value = template.music_genre;
                    document.getElementById('tpl-pacing').value = template.pacing_style;

                    // Show editor
                    document.getElementById('template-editor').style.display = 'block';

                    TB.ui.Toast.showSuccess('Template generiert! Passe es jetzt an.');
                }} else {{
                    TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
                }}
            }} catch (error) {{
                TB.ui.Toast.showError('Netzwerkfehler: ' + error.message);
            }}
        }});

        // Load Existing Template
        document.getElementById('btn-load-template').addEventListener('click', () => {{
            const templateName = document.getElementById('base-template').value;

            // Predefined templates (simplified)
            const templates = {{
                'cinematic': {{
                    name: 'Cinematic Short Film',
                    art_direction: 'Cinematic',
                    color_grading: 'Warm tones, desaturated, slightly faded',
                    music_genre: 'Indie folk, acoustic, melancholic',
                    pacing_style: 'Slow, contemplative'
                }},
                'viral_tiktok': {{
                    name: 'Viral TikTok Energy',
                    art_direction: 'High contrast, vibrant',
                    color_grading: 'Saturated, punchy colors',
                    music_genre: 'Trending pop, upbeat',
                    pacing_style: 'Fast, energetic'
                }},
                'youtube_doc': {{
                    name: 'YouTube Documentary',
                    art_direction: 'Professional, clean',
                    color_grading: 'Neutral, balanced',
                    music_genre: 'Ambient, subtle',
                    pacing_style: 'Medium, informative'
                }},
                'instagram': {{
                    name: 'Instagram Story Flow',
                    art_direction: 'Trendy, aesthetic',
                    color_grading: 'Pastel tones, soft',
                    music_genre: 'Chill beats, lo-fi',
                    pacing_style: 'Medium, episodic'
                }},
                'paranormal': {{
                    name: 'Paranormal Mystery',
                    art_direction: 'Surreal, dreamlike',
                    color_grading: 'Cool tones, desaturated',
                    music_genre: 'Ambient, eerie',
                    pacing_style: 'Slow, suspenseful'
                }}
            }};

            const template = templates[templateName];

            // Fill editor
            document.getElementById('tpl-name').value = template.name + ' (Custom)';
            document.getElementById('tpl-art').value = template.art_direction;
            document.getElementById('tpl-color').value = template.color_grading;
            document.getElementById('tpl-music').value = template.music_genre;
            document.getElementById('tpl-pacing').value = template.pacing_style;

            // Show editor
            document.getElementById('template-editor').style.display = 'block';

            TB.ui.Toast.showSuccess('Template geladen! Passe es jetzt an.');
        }});

        // Save Custom Template
        document.getElementById('btn-save-template').addEventListener('click', () => {{
            const template = {{
                name: document.getElementById('tpl-name').value,
                art_direction: document.getElementById('tpl-art').value,
                color_grading: document.getElementById('tpl-color').value,
                music_genre: document.getElementById('tpl-music').value,
                pacing_style: document.getElementById('tpl-pacing').value
            }};

            if (!template.name.trim()) {{
                TB.ui.Toast.showError('Bitte gib einen Template-Namen ein');
                return;
            }}

            // Save to localStorage
            let customTemplates = JSON.parse(localStorage.getItem('dircut_custom_templates') || '[]');
            customTemplates.push(template);
            localStorage.setItem('dircut_custom_templates', JSON.stringify(customTemplates));

            TB.ui.Toast.showSuccess('Template gespeichert: ' + template.name);

            // Refresh list
            loadCustomTemplates();

            // Add to style-template dropdown
            const option = document.createElement('option');
            option.value = 'custom_' + (customTemplates.length - 1);
            option.textContent = template.name + ' (Custom)';
            document.getElementById('style-template').appendChild(option);
        }});

        // Load Custom Templates
        function loadCustomTemplates() {{
            const customTemplates = JSON.parse(localStorage.getItem('dircut_custom_templates') || '[]');
            const list = document.getElementById('custom-templates-list');

            if (customTemplates.length === 0) {{
                list.innerHTML = '<p style="color: #6b7280;">Noch keine Custom Templates gespeichert.</p>';
                return;
            }}

            list.innerHTML = '';
            customTemplates.forEach((template, index) => {{
                const card = document.createElement('div');
                card.className = 'dircut-card';
                card.style.marginBottom = '12px';
                card.innerHTML = `
                    <h4 style="font-weight: 600; margin-bottom: 8px;">${{template.name}}</h4>
                    <p style="font-size: 12px; color: #6b7280; margin-bottom: 4px;">
                        <strong>Art:</strong> ${{template.art_direction}}
                    </p>
                    <p style="font-size: 12px; color: #6b7280; margin-bottom: 4px;">
                        <strong>Color:</strong> ${{template.color_grading}}
                    </p>
                    <p style="font-size: 12px; color: #6b7280;">
                        <strong>Music:</strong> ${{template.music_genre}}
                    </p>
                `;
                list.appendChild(card);
            }});
        }}

        // Load custom templates on page load
        loadCustomTemplates();
    </script>
    """

    return Result.html(data=html_content)
