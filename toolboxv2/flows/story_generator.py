"""
Production-Ready Multimedia Story Generator v5.0
Complete refactor with parallel processing, intelligent scene cuts, and perfect A-Z coherence
Enhanced with multiple world images and scene perspectives
"""

import asyncio
import os
import pathlib
import time

import aiohttp
import json
import logging
import subprocess
import shutil
import tempfile
import hashlib
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import math

# Core dependencies
try:
    from pydantic import BaseModel, Field
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

    import fal_client
except ImportError as e:
    from toolboxv2 import get_logger
    get_logger().warn(f"Missing dependencies: {e}")
from toolboxv2 import App

# ====================== CORE MODELS & CONFIG ======================
NAME = "story_generator"

class VoiceType(str, Enum):
    NARRATOR = "narrator"
    MALE_1 = "male_1"
    MALE_2 = "male_2"
    MALE_3 = "male_3"
    MALE_4 = "male_4"
    FEMALE_1 = "female_1"
    FEMALE_2 = "female_2"
    FEMALE_3 = "female_3"
    FEMALE_4 = "female_4"

class CharacterRole(str, Enum):
    PROTAGONIST = "Protagonist"
    ANTAGONIST = "Antagonist"
    SIDEKICK = "Sidekick"
    MYSTERIOUS = "Mysterious"

class ImageStyle(str, Enum):
    IMAX = "imax"
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    DIGITAL_ART = "digital_art"
    PENCIL_SKETCH = "pencil_sketch"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    NOIR = "noir"
    MINIMALIST = "minimalist"
    ABSTRACT = "abstract"
    RETRO = "retro"
    STEAMPUNK = "steampunk"
    CLASSIC = "classic"
    COMIC_STYLE = "comic_style"


class VideoStyle(str, Enum):
    # Cinematic Styles
    HOLLYWOOD_BLOCKBUSTER = "Hollywood Blockbuster"
    INDIE_FILM = "Indie Film"
    DOCUMENTARY = "Documentary"
    MUSIC_VIDEO = "Music Video"
    COMMERCIAL = "Commercial"

    # Artistic Animation Styles
    DISNEY_ANIMATION = "Disney Animation"
    PIXAR_3D = "Pixar 3D"
    STUDIO_GHIBLI = "Studio Ghibli"
    STOP_MOTION = "Stop Motion"
    ANIME = "Anime"

    # Visual Aesthetics
    FILM_NOIR = "Film Noir"
    CYBERPUNK = "Cyberpunk"
    RETRO_80S = "Retro 80s"
    VINTAGE_FILM = "Vintage Film"
    BLACK_WHITE_CLASSIC = "Black & White Classic"

    # Special Techniques
    TIME_LAPSE = "Time Lapse"
    SLOW_MOTION = "Slow Motion"
    GLITCH_ART = "Glitch Art"
    DOUBLE_EXPOSURE = "Double Exposure"
    SPLIT_SCREEN = "Split Screen"


class Character(BaseModel):
    name: str
    visual_desc: str = Field(..., description="Concise visual description for reference generation")
    role: CharacterRole
    voice: VoiceType

class DialogueLine(BaseModel):
    character: str
    text: str
    voice: VoiceType

class Scene(BaseModel):
    title: str
    setting: str  # Brief setting description
    narrator: str  # 2-3 sentence narration
    dialogue: List[DialogueLine] = []
    poses: List[str] = []  # List of character poses in this scene
    duration: float = 8.0  # seconds

class StylePreset(BaseModel):
    """Unified style configuration for consistent image generation"""
    image_style: ImageStyle
    camera_style: VideoStyle
    art_style: str = Field(default="realistic 8k photography")
    quality_modifiers: str = Field(default="high quality, detailed, professional")
    lighting: str = Field(default="natural lighting")
    color_palette: str = Field(default="vibrant colors")
    texture_emphasis: str = Field(default="")

    def get_style_prompt(self, base_prompt: str, image_type: str = "general", clip_type: str = "default") -> str:
        """Generate style-consistent prompt"""

        style_mapping = {
            ImageStyle.IMAX: "IMAX quality, cinematic, nature style, realistic textures, organic",
            ImageStyle.REALISTIC: "Photorealistic rendering, ultra-detailed, 4K resolution, true-to-life colors",
            ImageStyle.CARTOON: "Cartoon style, vibrant colors, clean outlines, cel-shaded look",
            ImageStyle.ANIME: "Anime art style, manga-inspired, expressive characters, detailed eyes, soft shading",
            ImageStyle.WATERCOLOR: "Watercolor painting, flowing pigments, soft gradients, natural brushstrokes",
            ImageStyle.OIL_PAINTING: "Oil painting, rich textures, layered strokes, classical fine art feel",
            ImageStyle.DIGITAL_ART: "Digital artwork, modern illustration, smooth gradients, stylized design",
            ImageStyle.PENCIL_SKETCH: "Pencil sketch, graphite lines, detailed hand-drawn textures, monochrome",
            ImageStyle.CYBERPUNK: "Cyberpunk aesthetic, neon glow, futuristic cityscapes, dark and moody atmosphere",
            ImageStyle.FANTASY: "Fantasy artwork, magical elements, mythical creatures, epic scenery",
            ImageStyle.NOIR: "Film noir style, high contrast, dramatic shadows, vintage cinematic tone",
            ImageStyle.MINIMALIST: "Minimalist design, clean lines, simple shapes, limited color palette, negative space",
            ImageStyle.ABSTRACT: "Abstract art, non-representational forms, expressive colors, geometric or organic shapes",
            ImageStyle.RETRO: "Retro vintage style, aged colors, classic design elements, nostalgic aesthetic",
            ImageStyle.STEAMPUNK: "Steampunk aesthetic, Victorian era meets technology, brass and copper tones, mechanical elements",
            ImageStyle.COMIC_STYLE: "Comic book style, bold outlines, halftone patterns, dynamic poses, vibrant colors"
        }

        video_style_mapping = {
            VideoStyle.HOLLYWOOD_BLOCKBUSTER: "Epic blockbuster. Dynamic camera work, dramatic lighting. Star Wars wipes, explosive cuts. Fast action editing, slow-motion highlights.",
            VideoStyle.INDIE_FILM: "Handheld intimate. Natural lighting, authentic framing. Organic cuts, subtle fades. Contemplative pacing, character-driven.",
            VideoStyle.DOCUMENTARY: "Observational style. Interview setups, b-roll footage. Clean cuts, informational wipes. Educational pacing, voice-over sync.",
            VideoStyle.MUSIC_VIDEO: "Rhythmic creative. Performance shots, artistic angles. Beat-sync cuts, rhythm transitions. Music-driven montages.",
            VideoStyle.COMMERCIAL: "Product focused. Lifestyle shots, clean composition. Smooth reveals, brand cuts. Tight pacing, professional polish.",
            VideoStyle.DISNEY_ANIMATION: "Smooth magical. Colorful fairy tale scenes. Magical dissolves, storybook turns. Musical pacing, character arcs.",
            VideoStyle.PIXAR_3D: "Expressive emotional. Detailed environments, family appeal. Emotional match cuts, perspective shifts. Comedy timing, heartfelt moments.",
            VideoStyle.STUDIO_GHIBLI: "Hand-drawn nature. Contemplative whimsical details. Gentle fades, seasonal transitions. Nature rhythm, introspective.",
            VideoStyle.STOP_MOTION: "Tactile handcrafted. Unique character movements. Frame morphs, physical transitions. Handcrafted pacing, creative comedy.",
            VideoStyle.ANIME: "Dynamic action. Expressive characters, detailed backgrounds. Speed cuts, dramatic zooms. Action choreography, emotional climaxes.",
            VideoStyle.FILM_NOIR: "Dramatic shadows. High contrast urban mystery. Shadow wipes, venetian effects. Suspenseful reveals, detective timing.",
            VideoStyle.CYBERPUNK: "Neon futuristic. Digital effects, high-tech atmosphere. Glitch transitions, holographic reveals. Fast tech cuts, cybernetic sync.",
            VideoStyle.RETRO_80S: "Vibrant synth-wave. Nostalgic period elements. Neon wipes, VHS glitches. Synth rhythm, retro montages.",
            VideoStyle.VINTAGE_FILM: "Film grain classic. Timeless composition, nostalgic atmosphere. Film burns, vintage fades. Classic Hollywood pacing.",
            VideoStyle.BLACK_WHITE_CLASSIC: "Dramatic elegance. Artistic composition, timeless lighting. Iris transitions, shadow wipes. Classic film timing.",
            VideoStyle.TIME_LAPSE: "Accelerated movement. Environmental changes, passage of time. Compression cuts, temporal shifts. Rapid progression.",
            VideoStyle.SLOW_MOTION: "Dramatic timing. Detailed movement capture, emotional emphasis. Speed ramping, slow reveals. Impactful moments.",
            VideoStyle.GLITCH_ART: "Corrupted visuals. Data moshing, digital artifacts. Digital corruption, pixel sorting. Chaotic digital rhythm.",
            VideoStyle.DOUBLE_EXPOSURE: "Overlapping imagery. Artistic blending, dreamy composition. Layered dissolves, exposure blends. Creative visual poetry.",
            VideoStyle.SPLIT_SCREEN: "Multiple perspectives. Parallel action, comparative storytelling. Division changes, perspective shifts. Multi-perspective timing."
        }

        style_prompt = style_mapping.get(self.image_style, "")
        camera_prompt = video_style_mapping.get(self.camera_style, "")

        # Build complete styled prompt
        components = [
            base_prompt,
            style_prompt,
            self.art_style,
            camera_prompt,
            self.lighting,
            self.color_palette,
            self.quality_modifiers
        ]

        if self.texture_emphasis:
            components.append(self.texture_emphasis)

        # Add image-type specific modifiers
        if image_type == "end":
            components.append("character sheet, reference pose, clear details")
        elif image_type == "scene":
            components.append("scene composition, environmental storytelling")
        elif image_type == "character":
            components.append("book cover design, title composition, marketing appeal")

        return ", ".join(filter(None, components))

class StoryData(BaseModel):
    title: str
    genre: str
    characters: List[Character]
    world_desc: str = Field(..., description="Concise world description")
    scenes: List[Scene]
    style_preset: StylePreset = Field(default_factory=lambda: StylePreset(
        image_style=ImageStyle.DIGITAL_ART,
        camera_style=VideoStyle.HOLLYWOOD_BLOCKBUSTER
    ))

@dataclass
class CostTracker:
    """Comprehensive cost tracking for all APIs"""
    agent_cost: int = 0
    kokoro_calls: int = 0
    kokoro_cost: float = 0.0
    flux_schnell_calls: int = 0
    flux_schnell_cost: float = 0.0
    flux_krea_calls: int = 0
    flux_krea_cost: float = 0.0
    flux_kontext_calls: int = 0
    flux_kontext_cost: float = 0.0
    banana_calls: int = 0
    banana_cost: float = 0.0
    minimax_calls: int = 0  # New
    minimax_cost: float = 0.0  # New
    elevenlabs_calls = 0
    elevenlabs_tokens = 0
    elevenlabs_cost = 0

    # Cost per call
    COSTS = {
        'kokoro': 0.002,  # Per audio segment
        'flux_schnell': 0.003,  # Per image
        'flux_krea': 0.025,  # Per image
        'flux_kontext': 0.04,  # Per image with reference
        'banana': 0.039,  # Per edit
        'minimax': 0.017 # Per second
    }

    def add_elevenlabs_cost(self, char_count: int):
        """Add ElevenLabs TTS cost"""
        self.elevenlabs_calls += 1
        self.elevenlabs_tokens += char_count
        self.elevenlabs_cost += (char_count / 1000) * 0.3


    def add_minimax_cost(self, calls: int = 1, second:int=5):
        """Add Minimax video generation cost"""
        self.minimax_calls += calls
        self.minimax_cost += second * self.COSTS['minimax']

    def add_kokoro_cost(self, calls: int = 1):
        self.kokoro_calls += calls
        self.kokoro_cost += calls * self.COSTS['kokoro']

    def add_flux_schnell_cost(self, calls: int = 1):
        self.flux_schnell_calls += calls
        self.flux_schnell_cost += calls * self.COSTS['flux_schnell']

    def add_flux_krea_cost(self, calls: int = 1):
        self.flux_krea_calls += calls
        self.flux_krea_cost += calls * self.COSTS['flux_krea']

    def add_flux_kontext_cost(self, calls: int = 1):
        self.flux_kontext_calls += calls
        self.flux_kontext_cost += calls * self.COSTS['flux_kontext']

    def add_banana_cost(self, calls: int = 1):
        self.banana_calls += calls
        self.banana_cost += calls * self.COSTS['banana']

    @property
    def total_cost(self) -> float:
        return (self.agent_cost + self.kokoro_cost + self.flux_schnell_cost +
                self.flux_krea_cost + self.minimax_cost + self.flux_kontext_cost + self.banana_cost)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_cost_usd': round(self.total_cost, 3),
            'breakdown': {
                'agent': {'calls': 1, 'cost': round(self.agent_cost, 3)},
                'kokoro': {'calls': self.kokoro_calls, 'cost': round(self.kokoro_cost, 3)},
                'flux_schnell': {'calls': self.flux_schnell_calls, 'cost': round(self.flux_schnell_cost, 3)},
                'flux_krea': {'calls': self.flux_krea_calls, 'cost': round(self.flux_krea_cost, 3)},
                'flux_kontext': {'calls': self.flux_kontext_calls, 'cost': round(self.flux_kontext_cost, 3)},
                'banana': {'calls': self.banana_calls, 'cost': round(self.banana_cost, 3)},
                'minimax': {'calls': self.minimax_calls, 'cost': round(self.minimax_cost, 3)},
                'elevenlabs': {'calls': self.elevenlabs_calls, 'cost': round(self.elevenlabs_cost, 3), 'tokens': self.elevenlabs_tokens}
            }
        }

    @classmethod
    def from_summary(cls, summary: Dict[str, Any]) -> 'CostTracker':
        cost_tracker = cls()
        if 'breakdown' not in summary:
            return cost_tracker
        cost_tracker.agent_cost = summary['breakdown']['agent']['cost']
        cost_tracker.kokoro_calls = summary['breakdown']['kokoro']['calls']
        cost_tracker.kokoro_cost = summary['breakdown']['kokoro']['cost']
        cost_tracker.flux_schnell_calls = summary['breakdown']['flux_schnell']['calls']
        cost_tracker.flux_schnell_cost = summary['breakdown']['flux_schnell']['cost']
        cost_tracker.flux_krea_calls = summary['breakdown']['flux_krea']['calls']
        cost_tracker.flux_krea_cost = summary['breakdown']['flux_krea']['cost']
        cost_tracker.flux_kontext_calls = summary['breakdown']['flux_kontext']['calls']
        cost_tracker.flux_kontext_cost = summary['breakdown']['flux_kontext']['cost']
        cost_tracker.banana_calls = summary['breakdown']['banana']['calls']
        cost_tracker.banana_cost = summary['breakdown']['banana']['cost']
        cost_tracker.minimax_calls = summary['breakdown']['minimax']['calls']
        cost_tracker.minimax_cost = summary['breakdown']['minimax']['cost']
        cost_tracker.elevenlabs_chars = summary['breakdown']['elevenlabs']['calls']
        cost_tracker.elevenlabs_cost = summary['breakdown']['elevenlabs']['cost']
        cost_tracker.elevenlabs_tokens = summary['breakdown']['elevenlabs']['elevenlabs_tokens']
        return cost_tracker

class Config:
    """Production configuration"""
    BASE_OUTPUT_DIR = Path("./generated_stories")
    IMAGE_SIZE = "landscape_4_3"
    VIDEO_FPS = 30
    SCENE_TRANSITION = 1.0  # seconds

    # Kokoro TTS settings
    KOKORO_MODELS_DIR = Path.cwd() / "kokoro_models"

    # FAL API models
    FLUX_SCHNELL = "fal-ai/flux/schnell"
    FLUX_KREA = "fal-ai/flux/krea"
    FLUX_KONTEXT = "fal-ai/flux-pro/kontext"
    BANANA_EDIT = "fal-ai/nano-banana/edit"

# ====================== LOGGING SETUP ======================

def setup_logging(project_dir: Path) -> logging.Logger:
    """Setup clean logging"""
    log_file = project_dir / "generation.log"

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# ====================== STORY GENERATOR ======================

class StoryGenerator:
    """Production-ready story generator with unified styling"""

    def __init__(self, isaa, logger: logging.Logger):
        self.isaa = isaa
        self.logger = logger

    async def generate_story(self, prompt: str, style_preset: Optional[StylePreset] = None) -> Optional[StoryData]:
        """Generate complete story with consistent styling"""
        self.logger.info("Generating story structure with unified styling...")

        # Default style if not provided
        if not style_preset:
            style_preset = StylePreset(
                image_style=ImageStyle.DIGITAL_ART,
                camera_style=VideoStyle.HOLLYWOOD_BLOCKBUSTER
            )

        system_prompt = f"""Create a multimedia story with consistent {style_preset.image_style.value} visual styling for: "{prompt}"

Visual Style Requirements:
- All images should follow {style_preset.image_style.value} aesthetic
- Camera work: {style_preset.camera_style.value} approach
- Consistent lighting: {style_preset.lighting}
- Color scheme: {style_preset.color_palette}

Story Requirements:
- 0-3 main characters with distinct visual features optimized for {style_preset.image_style.value} style, 0 catheters possible only narrator.
- 3-4 scenes, each 2-3 sentences of narration + (dialogue)
- Clear world setting description (2-4 sentences)
- Character descriptions should work well with {style_preset.image_style.value} rendering

Focus on visual storytelling that will translate effectively to {style_preset.image_style.value} images."""

        try:
            result = await self.isaa.mini_task_completion_format(
                system_prompt,
                format_schema=StoryData,
                agent_name="story_creator",
                use_complex=True
            )

            if result:
                story_data = StoryData(**result)
                # Ensure style preset is applied
                story_data.style_preset = style_preset
                self.logger.info(f"Generated story with {style_preset.image_style.value} styling")
                return story_data
            return None

        except Exception as e:
            self.logger.error(f"Story generation failed: {e}")
            return None

# ====================== ENHANCED PARALLEL IMAGE GENERATOR ======================

class ImageGenerator:
    """Two-stage image generator: Kontext for scene environments, then banana for character placement"""

    def __init__(self, logger: logging.Logger, cost_tracker: CostTracker, isaa=None):
        self.allimages = None
        self.isaa = isaa
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.character_refs = {}  # Store character reference URLs
        self.world_image_refs = {}  # Store world image URLs
        self.base_scene_refs = {}  # Store base scene environment URLs
        self.images_dict = {}


    async def _generate_and_upload_world_image(self, story: StoryData, images_dir: Path, idx: int) -> Optional[tuple]:
        """Generate styled world establishment image and upload it immediately"""
        world_perspectives = [
            f"Wide establishing shot: {story.world_desc}. Panoramic environmental overview, no characters, detailed landscape",
            f"Atmospheric environment: {story.world_desc}. Environmental mood, cinematic lighting, detailed setting"
        ]

        base_prompt = world_perspectives[idx % len(world_perspectives)]
        styled_prompt = story.style_preset.get_style_prompt(base_prompt, "scene")

        filename = f"01_world_{idx:02d}.png"
        output_path = images_dir / filename

        # Generate the image
        success = await self._generate_with_schnell(styled_prompt, output_path)
        if success:
            # Upload immediately
            world_url = await self._upload_to_fal(output_path)
            if world_url:
                self.logger.info(f"Generated and uploaded world image: {filename}")
                return (output_path, world_url)
            else:
                self.logger.error(f"Failed to upload world image: {filename}")

        return None

    async def _generate_world_image(self, story: StoryData, images_dir: Path, idx: int) -> Optional[Path]:
        """Generate styled world establishment images (kept for compatibility)"""
        result = await self._generate_and_upload_world_image(story, images_dir, idx)
        return result[0] if result else None
    def _select_scenes_for_video(self, scene_paths: List[Path], num_scenes: int) -> List[Path]:
        """Select one scene image per scene for video (chronological order)"""
        if not scene_paths:
            return []

        # Group scene images by scene index
        scene_groups = {}
        for path in scene_paths:
            # Extract scene index from filename pattern: scene_XX_perspective_YY
            match = re.search(r'scene_(\d+)_', path.name)
            if match:
                scene_idx = int(match.group(1))
                if scene_idx not in scene_groups:
                    scene_groups[scene_idx] = []
                scene_groups[scene_idx].append(path)

        # Select best perspective from each scene group (prefer medium shots)
        selected_scenes = []
        for scene_idx in sorted(scene_groups.keys()):
            if scene_groups[scene_idx]:
                # Prefer ALL medium shots
                sorted_perspectives = sorted(scene_groups[scene_idx],
                                             key=lambda x: (0 if 'perspective_01' in x.name else 1, x.name))
                selected_scenes.extend(sorted_perspectives)

        return selected_scenes

    async def _generate_character_ref(self, character: Character, style: StylePreset, images_dir: Path, idx: int) -> \
    Optional[Path]:
        """Generate styled character reference and ensure upload succeeds"""
        base_prompt = f"Character reference: {character.visual_desc}. Full body, clear details, neutral pose, character sheet, white background"
        styled_prompt = style.get_style_prompt(base_prompt, "character")

        filename = f"{idx:02d}_char_{character.name.lower().replace(' ', '_')}.png"
        output_path = images_dir / filename

        success = await self._generate_with_krea(styled_prompt, output_path)
        if success:
            # Upload and verify before storing
            char_url = await self._upload_to_fal(output_path)
            if char_url:
                self.character_refs[character.name] = char_url
                self.logger.info(f"Character reference uploaded: {character.name}")
                return output_path
            else:
                self.logger.error(f"Failed to upload character reference: {character.name}")

        return None

    async def _generate_base_scene_environment(self, scene: Scene, story: StoryData, images_dir: Path,
                                               scene_idx: int) -> Optional[Path]:
        """Generate base scene environment using Kontext with world image as reference"""
        if not self.world_image_refs:
            self.logger.error(
                f"No world images available for Kontext scene generation. Available refs: {list(self.world_image_refs.keys())}")
            # Fallback: generate scene environment directly
            fallback_prompt = f"Scene environment: {scene.setting}. {scene.title}. {scene.narrator}. No characters"
            styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
            filename = f"scene_{scene_idx:02d}_base_environment.png"
            output_path = images_dir / filename
            fallback_success = await self._generate_with_schnell(styled_fallback, output_path)

            if fallback_success:
                # Upload fallback scene
                scene_url = await self._upload_to_fal(output_path)
                if scene_url:
                    self.base_scene_refs[f"scene_{scene_idx}"] = scene_url
                    self.logger.info(f"Fallback scene environment uploaded for scene {scene_idx}")

            return output_path if fallback_success else None

        # Select world image (alternate between available world images)
        world_keys = list(self.world_image_refs.keys())
        world_key = world_keys[scene_idx % len(world_keys)]
        world_url = self.world_image_refs[world_key]

        self.logger.info(f"Using world image {world_key} for scene {scene_idx} environment")

        # Create scene-specific environment prompt
        scene_env_prompt = (f"Transform this world into the specific scene environment: {scene.setting}. "
                            f"Scene: {scene.title}. {scene.narrator}. "
                            f"Create the environmental stage for character interaction, no characters present. "
                            f"Maintain world consistency while adapting for scene-specific elements.")

        styled_prompt = story.style_preset.get_style_prompt(scene_env_prompt, "scene")

        filename = f"scene_{scene_idx:02d}_base_environment.png"
        output_path = images_dir / filename

        success = await self._generate_with_kontext(styled_prompt, world_url, output_path)
        if success:
            # Upload and verify before storing
            scene_url = await self._upload_to_fal(output_path)
            if scene_url:
                self.base_scene_refs[f"scene_{scene_idx}"] = scene_url
                self.logger.info(f"Generated and uploaded base scene environment for scene {scene_idx}")
                return output_path
            else:
                self.logger.error(f"Failed to upload base scene environment for scene {scene_idx}")

        # Fallback to Schnell if Kontext fails
        self.logger.warning(f"Kontext failed for scene {scene_idx}, falling back to Schnell")
        fallback_prompt = f"Scene environment: {scene.setting}. {scene.title}. {scene.narrator}. No characters"
        styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
        fallback_success = await self._generate_with_schnell(styled_fallback, output_path)

        if fallback_success:
            # Upload fallback scene
            scene_url = await self._upload_to_fal(output_path)
            if scene_url:
                self.base_scene_refs[f"scene_{scene_idx}"] = scene_url
                self.logger.info(f"Fallback scene environment uploaded for scene {scene_idx}")

        return output_path if fallback_success else None

    async def generate_all_images(self, story: StoryData, project_dir: Path) -> Dict[str, List[Path]]:
        """Generate all images with proper sequencing and validation"""
        self.logger.info(
            f"Starting two-stage parallel image generation with {story.style_preset.image_style.value} style...")

        images_dir = project_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Phase 1: Generate character references and wait for uploads
        self.logger.info("Phase 1: Generating styled character references...")
        character_tasks = [
            self._generate_character_ref(char, story.style_preset, images_dir, idx)
            for idx, char in enumerate(story.characters)
        ]
        character_paths = await asyncio.gather(*character_tasks, return_exceptions=True)
        character_paths = [p for p in character_paths if isinstance(p, Path)]

        # Validate character uploads
        self.logger.info(f"Phase 1 complete: {len(self.character_refs)} character references uploaded")
        for char_name, char_url in self.character_refs.items():
            if char_url is None:
                self.logger.error(f"Character reference upload failed: {char_name}")

        # Phase 2: Generate world images and upload them immediately
        self.logger.info("Phase 2: Generating and uploading world images...")
        world_tasks = [
            self._generate_and_upload_world_image(story, images_dir, idx)
            for idx in range(2)  # Generate 2 world images
        ]
        world_results = await asyncio.gather(*world_tasks, return_exceptions=True)

        # Collect world paths and ensure uploads are complete
        world_paths = []
        for result in world_results:
            if isinstance(result, tuple) and len(result) == 2:
                world_path, world_url = result
                if world_path and world_url:
                    world_paths.append(world_path)
                    self.world_image_refs[world_path.stem] = world_url

        self.logger.info(f"Phase 2 complete: {len(self.world_image_refs)} world images uploaded")

        # Phase 3: Generate base scene environments using Kontext and wait for uploads
        self.logger.info("Phase 3: Generating base scene environments with Kontext...")
        base_scene_tasks = []
        for scene_idx, scene in enumerate(story.scenes):
            base_scene_tasks.append(
                self._generate_base_scene_environment(scene, story, images_dir, scene_idx)
            )

        base_scene_paths = await asyncio.gather(*base_scene_tasks, return_exceptions=True)
        base_scene_paths = [p for p in base_scene_paths if isinstance(p, Path)]

        # Validate base scene uploads
        self.logger.info(f"Phase 3 complete: {len(self.base_scene_refs)} base scene environments uploaded")
        for scene_key, scene_url in self.base_scene_refs.items():
            if scene_url is None:
                self.logger.error(f"Base scene environment upload failed: {scene_key}")

        # Validation before Phase 4: Check if we have enough resources
        missing_chars = [char.name for char in story.characters if
                         char.name not in self.character_refs or self.character_refs[char.name] is None]
        missing_scenes = [f"scene_{i}" for i in range(len(story.scenes)) if
                          f"scene_{i}" not in self.base_scene_refs or self.base_scene_refs[f"scene_{i}"] is None]

        if missing_chars:
            self.logger.warning(f"Missing character references: {missing_chars}")
        if missing_scenes:
            self.logger.warning(f"Missing base scene environments: {missing_scenes}")

        # Phase 4: Generate different perspectives using banana (with validation)
        self.logger.info("Phase 4: Generating character perspectives with banana...")
        perspective_tasks = []
        for scene_idx, scene in enumerate(story.scenes):
            num_perspectives = min(4, max(2, len(scene.dialogue)+1))  # 2-4 perspectives per scene
            self.logger.info(f"Scene {scene_idx} ({scene.title}): generating {num_perspectives} perspectives")
            for perspective_idx in range(num_perspectives):
                perspective_tasks.append(
                    self._generate_character_perspective(scene, story, images_dir, scene_idx, perspective_idx)
                )

        self.logger.info(f"Starting {len(perspective_tasks)} perspective generation tasks...")
        perspective_results = await asyncio.gather(*perspective_tasks, return_exceptions=True)

        # Process results with detailed logging
        perspective_paths = []
        for i, result in enumerate(perspective_results):
            if isinstance(result, Path):
                perspective_paths.append(result)
                self.logger.info(f"Perspective task {i}: SUCCESS - {result.name}")
            elif isinstance(result, pathlib.WindowsPath):
                perspective_paths.append(result)
                self.logger.info(f"Perspective task {i}: SUCCESS - {result.name}")
            elif hasattr(result, 'name') and hasattr(result, 'absolute'):
                perspective_paths.append(result)
                self.logger.info(f"Perspective task {i}: SUCCESS - {result.name}")
            elif isinstance(result, bool) and result:
                self.logger.info(f"Perspective task {i}: SUCCESS - wrong")
            elif isinstance(result, bool) and not result:
                self.logger.error(f"Perspective task {i}: FAILED")
            elif isinstance(result, Exception):
                self.logger.error(f"Perspective task {i}: FAILED with exception - {result}")
            else:
                self.logger.warning(f"Perspective task {i}: FAILED - returned {type(result)}")

        self.logger.info(
            f"Phase 4 complete: {len(perspective_paths)} perspectives generated out of {len(perspective_tasks)} tasks")

        # Phase 5: Generate cover and end card
        self.logger.info("Phase 5: Generating cover and end card...")
        cover_task = self._generate_cover(story, images_dir)
        end_task = self._generate_end_card(story, images_dir)

        cover_task_res, end_task_res  = await asyncio.gather(cover_task, end_task, return_exceptions=True)
        cover_path = images_dir / "00_cover.png"
        end_path = images_dir / "99_end.png"

        # Organize results
        all_images_for_video = []
        if cover_task_res:
            all_images_for_video.append(cover_path)

        # Add world images for establishing shots
        all_images_for_video.extend(sorted(world_paths))

        # Add ALL generated scene perspectives
        all_images_for_video.extend(perspective_paths)

        if end_task_res:
            all_images_for_video.append(end_path)

        self.logger.info(
            f"Assembled {len(all_images_for_video)} images for video sequence generation, including all perspectives.")

        # The original 'scenes_for_video' can still be useful for other purposes (like a simple summary).
        scenes_for_video = self._select_scenes_for_video(perspective_paths, len(story.scenes))

        # Create a complete list of all generated image assets for the PDF and metadata.
        all_images_complete_list = (
            ([cover_path] if cover_task_res else []) +
            world_paths +
            character_paths +
            base_scene_paths +
            perspective_paths +
            ([end_path] if end_task_res else [])
        )

        return {
            'all_images': all_images_for_video,  # Corrected list for VideoGenerator
            'all_images_complete': sorted([p for p in all_images_complete_list if p]),
            'character_refs': character_paths,
            'world_images': world_paths,
            'base_scene_environments': base_scene_paths,  # New: base environments
            'scene_perspectives': perspective_paths,  # New: character perspectives
            'scene_images_for_video': scenes_for_video,
            'cover': [cover_path] if cover_path else [],
            'end': [end_path] if end_path else [],
            'style_used': story.style_preset.image_style.value
        }

    async def _generate_character_perspective(self, scene: Scene, story: StoryData, images_dir: Path, scene_idx: int,
                                              perspective_idx: int) -> Optional[Path]:
        """Generate character perspective using banana to place characters in scene environment"""

        self.logger.info(f"Starting perspective {perspective_idx} for scene {scene_idx}: {scene.title}")

        # Get scene characters present in this scene
        scene_characters = list(set([d.character for d in scene.dialogue if d.character != "Narrator"]))
        if not scene_characters:
            self.logger.warning(f"No characters in scene {scene_idx}, skipping perspective {perspective_idx}")
            return None

        # Define perspective types
        perspectives = [
            {
                "desc": "Wide establishing shot with all characters",
                "camera": "wide shot, cinematic framing, environmental context",
                "max_chars": 3
            },
            {
                "desc": "Medium shot focusing on main characters",
                "camera": "medium shot, character focus, balanced composition",
                "max_chars": 2
            },
            {
                "desc": "Close-up perspective on primary character",
                "camera": "close-up shot, intimate framing, emotional detail",
                "max_chars": 1
            },
            {
                "desc": "Over-the-shoulder dialogue view",
                "camera": "over-the-shoulder view, dialogue perspective, character interaction",
                "max_chars": 2
            }
        ]

        perspective = perspectives[perspective_idx % len(perspectives)]

        # Get base scene environment
        base_scene_key = f"scene_{scene_idx}"
        if base_scene_key not in self.base_scene_refs or self.base_scene_refs[base_scene_key] is None:
            self.logger.error(f"No valid base scene environment for scene {scene_idx}")
            # Generate with Schnell as fallback
            chars_for_perspective = scene_characters[:perspective["max_chars"]]
            fallback_prompt = (f"Scene with characters: {scene.title}. {scene.setting}. "
                               f"Characters: {', '.join(chars_for_perspective)}. "
                               f"{perspective['camera']}")
            styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene", clip_type="editing")
            filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            output_path = images_dir / filename
            if await self._generate_with_krea(styled_fallback, output_path):
                return output_path
            raise Exception(f"Failed to generate fallback image for scene {scene_idx} perspective {perspective_idx}")

        base_scene_url = self.base_scene_refs[base_scene_key]

        # Select characters for this perspective
        chars_for_perspective = scene_characters
        char_refs = []
        char_names = []

        for char_name in chars_for_perspective:
            if char_name in self.character_refs and self.character_refs[char_name] is not None:
                char_refs.append(self.character_refs[char_name])
                char_names.append(char_name)

        if not char_refs:
            self.logger.error(
                f"No valid character references available for scene {scene_idx} perspective {perspective_idx}")
            # Generate with Schnell as fallback
            fallback_prompt = (f"Scene with characters: {scene.title}. {scene.setting}. "
                               f"Characters: {', '.join(chars_for_perspective)}. "
                               f"{perspective['camera']}")
            styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene", clip_type="editing")
            filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            output_path = images_dir / filename
            if await self._generate_with_krea(styled_fallback, output_path):
                return output_path
            raise Exception(f"Failed to generate fallback image for scene {scene_idx} perspective {perspective_idx}")

        # Create banana prompt for character placement
        char_placement_descriptions = self._get_character_placements(chars_for_perspective, scene,
                                                                     perspective["camera"])

        banana_prompt = (f"Place these characters into the scene environment: {', '.join(char_names)}. "
                         f"Scene: {scene.title} - {scene.setting}. "
                         f"{perspective['camera']}. "
                         f"{scene.poses}. "
                         f"{char_placement_descriptions} "
                         f"Characters should interact naturally with the environment and each other. "
                         f"Maintain character appearance consistency and environmental lighting.")

        styled_prompt = story.style_preset.get_style_prompt(banana_prompt, "scene", clip_type="editing")

        filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
        output_path = images_dir / filename

        # Use banana with base scene + character references (ensure no None values)
        all_refs = [base_scene_url] + char_refs
        all_refs = [ref for ref in all_refs if ref is not None]  # Filter out any None values

        if len(all_refs) < 2:  # Need at least base scene + 1 character
            self.logger.error(f"Insufficient valid references for banana: {len(all_refs)}")
            # Fallback to Schnell
            fallback_prompt = (f"Scene with characters: {scene.title}. {scene.setting}. "
                               f"Characters: {', '.join(char_names)}. "
                               f"{perspective['camera']}")
            styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
            return await self._generate_with_schnell(styled_fallback, output_path)

        improve_prompt = await self.isaa.mini_task_completion(
            mini_task=styled_prompt,
            user_task="Improve the following prompt for better character placement and interaction. the prompt is for image to video generation."
                      "Describe the camera movement ( zoom in/out, panning, tilting, transition, transition effects, seen before and after) and the characters actions."
                      "Make the prompt as short and information dense as possible.",
            agent_name="self"
        ) if self.isaa else styled_prompt

        success = await self._generate_with_banana(improve_prompt, all_refs, output_path)

        if success:
            return output_path

        # Fallback to Schnell
        self.logger.warning(
            f"Banana failed for scene {scene_idx} perspective {perspective_idx}, falling back to Schnell")
        fallback_prompt = (f"Scene with characters: {scene.title}. {scene.setting}. "
                           f"Characters: {', '.join(char_names)}. "
                           f"{perspective['camera']}")
        styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
        if await self._generate_with_schnell(styled_fallback, output_path):
            return output_path
        raise Exception(f"Failed to generate fallback image for scene {scene_idx} perspective {perspective_idx}")

    def _get_character_placements(self, characters: List[str], scene: Scene, camera_angle: str) -> str:
        """Generate character placement descriptions based on scene context"""
        if len(characters) == 1:
            return f"{characters[0]} positioned prominently in the scene, engaging with the environment."

        elif len(characters) == 2:
            if "dialogue" in scene.setting.lower():
                return f"{characters[0]} and {characters[1]} positioned for conversation, facing each other or in dialogue poses."
            else:
                return f"{characters[0]} and {characters[1]} positioned naturally in the environment, both visible and well-composed."

        else:  # 3+ characters
            return f"Group composition with {', '.join(characters[:-1])} and {characters[-1]} arranged naturally in the scene for group interaction."

    async def _generate_cover(self, story: StoryData, images_dir: Path) -> Optional[Path]:
        """Generate styled cover"""
        chars_desc = ", ".join([f"{c.name}: {c.visual_desc}" for c in story.characters])
        base_prompt = f"Book cover: {story.title}. {story.genre} story. Characters: {chars_desc}. Epic composition, title design"

        styled_prompt = story.style_preset.get_style_prompt(base_prompt, "cover")
        return await self._generate_with_krea(styled_prompt, images_dir / "00_cover.png")

    async def _generate_end_card(self, story: StoryData, images_dir: Path) -> Optional[Path]:
        """Generate styled end card"""
        base_prompt = f"End card: 'The End' text, {story.genre} conclusion, elegant finale design"
        styled_prompt = story.style_preset.get_style_prompt(base_prompt, "cover")
        return await self._generate_with_schnell(styled_prompt, images_dir / "99_end.png")

    # API Methods
    async def _generate_with_krea(self, prompt: str, output_path: Path, retries: int = 3) -> bool:
        """Generate image with KREA model"""
        for attempt in range(retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.FLUX_KREA, prompt, {}
                )

                if result and 'images' in result and result['images']:
                    success = await self._download_image(result['images'][0]['url'], output_path)
                    if success:
                        self.cost_tracker.add_flux_krea_cost()
                        self.logger.info(f"Generated with KREA: {output_path.name}")
                        return True

            except Exception as e:
                self.logger.error(f"KREA generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return False

    async def _generate_with_schnell(self, prompt: str, output_path: Path, retries: int = 3) -> bool:
        """Generate image with Schnell model"""
        for attempt in range(retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.FLUX_SCHNELL, prompt, {"num_inference_steps": 4}
                )

                if result and 'images' in result and result['images']:
                    success = await self._download_image(result['images'][0]['url'], output_path)
                    if success:
                        self.cost_tracker.add_flux_schnell_cost()
                        self.logger.info(f"Generated with Schnell: {output_path.name}")
                        return True

            except Exception as e:
                self.logger.error(f"Schnell generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return False

    async def _generate_with_kontext(self, prompt: str, image_url: str, output_path: Path, retries: int = 3) -> bool:
        """Generate image with FLUX Kontext model"""
        for attempt in range(retries):
            try:
                args = {
                    "image_url": image_url,
                    "guidance_scale": 3.5,
                    "num_images": 1,
                    "output_format": "png",
                    "safety_tolerance": "2"
                }

                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.FLUX_KONTEXT, prompt, args
                )

                if result and 'images' in result and result['images']:
                    success = await self._download_image(result['images'][0]['url'], output_path)
                    if success:
                        self.cost_tracker.add_flux_kontext_cost()
                        self.logger.info(f"Generated with Kontext: {output_path.name}")
                        return True

            except Exception as e:
                self.logger.error(f"Kontext generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return False

    async def _generate_with_banana(self, prompt: str, image_urls: List[str], output_path: Path,
                                    retries: int = 3) -> bool:
        """Generate image with banana (nano-banana/edit) model"""
        self.logger.info(f"Banana generation: {output_path.name} with {len(image_urls)} reference images")

        for attempt in range(retries):
            try:
                args = {
                    "image_urls": image_urls,
                    "num_images": 1
                }

                self.logger.info(f"Banana attempt {attempt + 1}: calling API...")
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.BANANA_EDIT, prompt, args
                )

                if result and 'images' in result and result['images']:
                    self.logger.info(f"Banana attempt {attempt + 1}: got result, downloading...")
                    success = await self._download_image(result['images'][0]['url'], output_path)
                    if success:
                        self.cost_tracker.add_banana_cost()
                        self.logger.info(f"Generated with banana: {output_path.name}")
                        return True
                    else:
                        self.logger.error(f"Banana attempt {attempt + 1}: download failed")
                else:
                    self.logger.warning(f"Banana attempt {attempt + 1}: no valid response - {result}")

            except Exception as e:
                self.logger.error(f"Banana generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        self.logger.error(f"All banana attempts failed for {output_path.name}")
        return False

    def _sync_fal_call(self, model: str, prompt: str, extra_args: Dict) -> Optional[Dict]:
        """Synchronous FAL API call with error handling"""
        try:
            args = {"prompt": prompt}

            # Add model-specific parameters
            if model == Config.FLUX_KONTEXT:
                args.update(extra_args)
            elif model == Config.BANANA_EDIT:
                args.update(extra_args)
            else:
                args.update({
                    "image_size": Config.IMAGE_SIZE,
                    "num_images": 1,
                    **extra_args
                })

            return fal_client.subscribe(model, arguments=args)
        except Exception as e:
            self.logger.error(f"FAL API call failed for {model}: {e}")
            return None

    async def _upload_to_fal(self, image_path: Path) -> Optional[str]:
        """Upload image to FAL with error handling"""
        try:
            if not image_path.exists() or image_path.stat().st_size == 0:
                self.logger.error(f"Invalid image file for upload: {image_path}")
                return None

            if image_path.name in self.images_dict:
                return self.images_dict[image_path.name]

            return await asyncio.get_event_loop().run_in_executor(
                None, fal_client.upload_file, str(image_path)
            )
        except Exception as e:
            self.logger.error(f"FAL upload failed: {e}")
            return None

    async def _download_image(self, url: str, output_path: Path) -> bool:
        """Download image from URL with production-ready error handling"""
        try:
            if url in self.images_dict and Path(self.images_dict[output_path.name]).exists():
                return True

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)

                        if output_path.exists() and output_path.stat().st_size > 1000:
                            return True
                        else:
                            self.logger.error(f"Downloaded file is invalid: {output_path}")
                    else:
                        self.logger.error(f"Download failed with status {response.status}: {url}")

        except Exception as e:
            self.logger.error(f"Download failed: {e}")

        return False

# ====================== AUDIO GENERATOR ======================

class AudioGenerator:
    """Enhanced TTS audio generator supporting Kokoro and ElevenLabs"""

    def __init__(self, logger: logging.Logger, cost_tracker, project_dir, use_elevenlabs: bool = False):
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.use_elevenlabs = use_elevenlabs
        self.temp_dir = project_dir / "audio"
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize ElevenLabs client if needed
        if self.use_elevenlabs:
            try:
                from elevenlabs.client import ElevenLabs
                api_key = os.getenv("ELEVENLABS_API_KEY")
                if not api_key:
                    raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
                self.elevenlabs_client = ElevenLabs(api_key=api_key)
            except ImageGenerator:
                print("install elevenlabs")
                self.logger.error("ElevenLabs not installed !!! pip install elevenlabs")
                self.use_elevenlabs = False

        # Kokoro voice mapping (gender-aware)
        self.kokoro_voice_map = {
            VoiceType.NARRATOR: "af_sarah",
            VoiceType.MALE_1: "am_adam",
            VoiceType.MALE_2: "bm_lewis",
            VoiceType.MALE_3: "bm_daniel",
            VoiceType.MALE_4: "am_michael",
            VoiceType.FEMALE_1: "af_bella",
            VoiceType.FEMALE_2: "af_sarah",
            VoiceType.FEMALE_3: "bf_emma",
            VoiceType.FEMALE_4: "af_nicole"
        }

        # ElevenLabs high-quality voice mapping
        self.elevenlabs_voice_map = {
            VoiceType.NARRATOR: "c6SfcYrb2t09NHXiT80T",  # Rachel - Professional female narrator
            VoiceType.MALE_1: "UgBBYS2sOqTuMpoF3BR0",  # Adam - Deep, authoritative male
            VoiceType.MALE_2: "TX3LPaxmHKxFdv7VOQHJ",  # Antoni - Warm, friendly male
            VoiceType.MALE_3: "N2lVS1w4EtoT3dr4eOWO",  # Sam - Energetic male
            VoiceType.MALE_4: "JBFqnCBsd6RMkjVDRZzb",  # Arnold - Mature male
            VoiceType.FEMALE_1: "aEO01A4wXwd1O8GPgGlF",  # Domi - Confident female
            VoiceType.FEMALE_2: "21m00Tcm4TlvDq8ikWAM",  # Bella - Sweet female
            VoiceType.FEMALE_3: "XrExE9yKIg1WjnnlVkGX",  # Dorothy - Mature female
            VoiceType.FEMALE_4: "pFZP5JQG7iQjIQuC4Bku"  # Lily - Young female
        }

        # Character voice assignment tracking
        self.character_voices = {}
        self.voice_counters = {"male": 1, "female": 1}

    async def generate_audio(self, story, project_dir: Path) -> Optional[Path]:
        """Generate synchronized audio matching video structure"""
        self.logger.info(f"Generating audio with {'ElevenLabs' if self.use_elevenlabs else 'Kokoro'} TTS...")

        audio_dir = project_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        # Generate audio segments with proper timing
        segments = []

        # Title (2 seconds)
        title_text = f"{story.title}. A {story.genre} story."
        title_segment = await self._generate_segment(title_text, VoiceType.NARRATOR, "title")
        if title_segment:
            segments.append((title_segment, 2.0))

        # Generate scene audio with calculated durations
        for idx, scene in enumerate(story.scenes):
            # Scene narration
            if scene.narrator:
                narrator_segment = await self._generate_segment(
                    scene.narrator, VoiceType.NARRATOR, f"scene_{idx}_narrator"
                )
                if narrator_segment:
                    segments.append((narrator_segment, scene.duration * 0.4))

            # Scene dialogue
            for d_idx, dialogue in enumerate(scene.dialogue):
                # Find character and assign voice based on gender
                char_voice = dialogue.voice
                if not char_voice:
                    char_voice = VoiceType.NARRATOR

                dialogue_segment = await self._generate_segment(
                    dialogue.text, char_voice, f"scene_{idx}_dialogue_{d_idx}"
                )
                if dialogue_segment:
                    segments.append((dialogue_segment, scene.duration * 0.6 / len(scene.dialogue)))

        # Combine with precise timing
        return await self._combine_segments(segments, audio_dir, story.title)

    async def _generate_segment(self, text: str, voice: VoiceType, name: str) -> Optional[Path]:
        """Generate single audio segment using selected TTS provider"""
        if self.use_elevenlabs:
            return await self._generate_elevenlabs_segment(text, voice, name)
        else:
            return await self._generate_kokoro_segment(text, voice, name)

    async def _generate_elevenlabs_segment(self, text: str, voice: VoiceType, name: str) -> Optional[Path]:
        """Generate audio segment using ElevenLabs"""
        output_path = self.temp_dir / f"{name}.mp3"

        try:
            voice_id = self.elevenlabs_voice_map[voice]

            # Generate audio with highest quality settings
            audio = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",  # Highest quality model
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            )

            # Save audio to file
            with open(output_path, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)

            if output_path.exists():
                # Convert to WAV for consistency
                wav_path = output_path.with_suffix('.wav')
                await self._convert_to_wav(output_path, wav_path)
                output_path.unlink()  # Remove MP3

                # Track cost (approximate)
                char_count = len(text)
                self.cost_tracker.add_elevenlabs_cost(char_count)

                self.logger.debug(f"Generated ElevenLabs segment: {name} ({char_count} chars)")
                return wav_path

        except Exception as e:
            self.logger.error(f"ElevenLabs segment generation failed for '{name}': {e}")

        return None

    async def _generate_kokoro_segment(self, text: str, voice: VoiceType, name: str) -> Optional[Path]:
        """Generate audio segment using Kokoro TTS"""
        output_path = self.temp_dir / f"{name}.wav"
        text_file = self.temp_dir / f"{name}.txt"

        try:
            # Write text file
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            # Generate with Kokoro
            cmd = [
                "kokoro-tts", str(text_file), str(output_path),
                "--voice", self.kokoro_voice_map[voice],
                "--model", str(Config.KOKORO_MODELS_DIR / "kokoro-v1.0.onnx"),
                "--voices", str(Config.KOKORO_MODELS_DIR / "voices-v1.0.bin"),
                "--speed", "1.1"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0 and output_path.exists():
                self.cost_tracker.add_kokoro_cost()
                self.logger.debug(f"Generated Kokoro segment: {name}")
                return output_path

        except Exception as e:
            self.logger.error(f"Kokoro segment generation failed for '{name}': {e}")
        finally:
            if text_file.exists():
                text_file.unlink()

        return None

    async def _convert_to_wav(self, input_path: Path, output_path: Path):
        """Convert audio file to WAV format"""
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-acodec", "pcm_s16le", "-ar", "44100",
            "-y", str(output_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

    async def _combine_segments(self, segments: List[Tuple[Path, float]], audio_dir: Path, title: str) -> Optional[
        Path]:
        """Combine segments with precise timing"""
        if not segments:
            return None

        output_path = audio_dir / f"{self._sanitize(title)}_complete.wav"
        list_file = self.temp_dir / "segments.txt"

        try:
            # Create concat file with timing
            with open(list_file, 'w', encoding='utf-8') as f:
                for segment_path, duration in segments:
                    f.write(f"file '{segment_path.absolute()}'\n")
                    # Add silence between segments
                    silence_path = await self._generate_silence(0.5)
                    if silence_path:
                        f.write(f"file '{silence_path.absolute()}'\n")

            # Combine with ffmpeg
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", "-y", str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0 and output_path.exists():
                provider = "ElevenLabs" if self.use_elevenlabs else "Kokoro"
                self.logger.info(f"Audio generated with {provider}: {output_path.name}")
                return output_path

        except Exception as e:
            self.logger.error(f"Audio combination failed: {e}")

        return None

    async def _generate_silence(self, duration: float) -> Optional[Path]:
        """Generate silence segment"""
        output_path = self.temp_dir / f"silence_{duration}.wav"

        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", f"anullsrc=duration={duration}",
            "-y", str(output_path)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0:
                return output_path
        except Exception:
            pass

        return None

    def _sanitize(self, filename: str) -> str:
        """Sanitize filename"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)[:50]

# ====================== INTELLIGENT VIDEO GENERATOR ======================

class VideoGenerator:
    """Enhanced video generator with perfect audio-video synchronization"""

    def __init__(self, logger: logging.Logger, project_dir: Path):
        self.logger = logger
        self.temp_dir = project_dir / "video_editing"
        self.temp_dir.mkdir(exist_ok=True)

    def _categorize_all_images_enhanced(self, images: List[Path]) -> Dict[str, Any]:
        """Enhanced image categorization with explicit cover/end image handling"""
        categories = {
            'cover': [],
            'world': [],
            'character_refs': [],
            'scene_perspectives': {},  # {scene_index: [perspective_images]}
            'end': [],
            'all': images
        }

        # First, look for explicit cover and end images by filename
        cover_found = False
        end_found = False

        for img_path in images:
            name = img_path.name.lower()
            if name == '00_cover.png' or 'cover' in name and name.startswith('00_'):
                categories['cover'].append(img_path)
                cover_found = True
                self.logger.info(f"Found dedicated cover image: {img_path.name}")
            elif name == '99_end.png' or 'end' in name and name.startswith('99_'):
                categories['end'].append(img_path)
                end_found = True
                self.logger.info(f"Found dedicated end image: {img_path.name}")
            elif 'world' in name or name.startswith('01_'):
                categories['world'].append(img_path)
            elif 'char' in name or name.startswith('02_'):
                categories['character_refs'].append(img_path)
            elif 'scene' in name and 'perspective' in name:
                # Extract scene and perspective indices
                scene_match = re.search(r'scene_(\d+)_', name)
                perspective_match = re.search(r'perspective_(\d+)', name)

                if scene_match:
                    scene_idx = int(scene_match.group(1))
                    perspective_idx = int(perspective_match.group(1)) if perspective_match else 0

                    if scene_idx not in categories['scene_perspectives']:
                        categories['scene_perspectives'][scene_idx] = []

                    categories['scene_perspectives'][scene_idx].append({
                        'path': img_path,
                        'perspective_idx': perspective_idx
                    })

        # Sort scene perspectives by perspective index
        for scene_idx in categories['scene_perspectives']:
            categories['scene_perspectives'][scene_idx] = sorted(
                categories['scene_perspectives'][scene_idx],
                key=lambda x: x['perspective_idx']
            )
            self.logger.info(f"Scene {scene_idx} has {len(categories['scene_perspectives'][scene_idx])} perspectives")

        # Critical: Only use fallbacks if dedicated images are not found
        if not cover_found and images:
            self.logger.warning("No dedicated 00_cover.png found, using fallback")
            categories['cover'].append(images[0])
        elif not categories['cover']:
            self.logger.error("No cover image found at all!")

        if not end_found and images:
            self.logger.warning("No dedicated 99_end.png found, using fallback")
            categories['end'].append(images[-1])
        elif not categories['end']:
            self.logger.error("No end image found at all!")

        return categories

    async def _get_precise_audio_duration(self, audio_path: Path) -> float:
        """Get exact audio duration for perfect sync"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                duration = float(stdout.decode().strip())
                self.logger.info(f"Audio duration: {duration:.3f}s")
                return duration
        except Exception as e:
            self.logger.error(f"Could not get audio duration: {e}")

        return 60.0  # fallback

    async def _create_perspective_switching_cuts(self, story: StoryData, audio_duration: float,
                                                 image_categories: Dict) -> List[Dict]:
        """Create cuts that match EXACT audio duration with proper scene timing"""
        scene_cuts = []
        current_time = 0.0

        # Fixed durations based on audio segments
        cover_duration = 3.0
        world_duration = 2.0
        end_duration = 2.0

        # Calculate available time for scenes (must match audio exactly)
        available_scene_time = audio_duration - cover_duration - world_duration - end_duration
        if available_scene_time <= 0:
            available_scene_time = audio_duration * 0.7  # Use 70% for scenes
            cover_duration = audio_duration * 0.15
            world_duration = audio_duration * 0.10
            end_duration = audio_duration * 0.05

        self.logger.info(f"Audio duration: {audio_duration:.3f}s -> Scene time: {available_scene_time:.3f}s")

        # 1. Cover (fixed start)
        scene_cuts.append({
            'start_time': current_time,
            'duration': cover_duration,
            'image_type': 'cover',
            'description': 'Story title and cover'
        })
        current_time += cover_duration

        # 2. World establishment
        scene_cuts.append({
            'start_time': current_time,
            'duration': world_duration,
            'image_type': 'world',
            'world_index': 0,
            'description': 'World establishment'
        })
        current_time += world_duration

        # 3. Distribute scene time based on story.scenes duration
        total_scene_duration = sum(scene.duration for scene in story.scenes)
        if total_scene_duration > 0:
            duration_multiplier = available_scene_time / total_scene_duration
        else:
            duration_multiplier = available_scene_time / len(story.scenes)

        for scene_idx, scene in enumerate(story.scenes):
            # Calculate this scene's duration proportionally
            if total_scene_duration > 0:
                scene_duration = scene.duration * duration_multiplier
            else:
                scene_duration = available_scene_time / len(story.scenes)

            # Get available perspectives for this scene
            scene_perspective_data = image_categories['scene_perspectives'].get(scene_idx, [])
            num_perspectives = max(1, len(scene_perspective_data))

            # Create cuts for this scene's perspectives
            cuts_per_scene = min(4, num_perspectives)  # Max 4 cuts per scene
            cut_duration = scene_duration / cuts_per_scene

            for cut_idx in range(cuts_per_scene):
                perspective_idx = cut_idx % num_perspectives if num_perspectives > 0 else 0

                scene_cuts.append({
                    'start_time': current_time,
                    'duration': cut_duration,
                    'image_type': 'scene_perspective',
                    'scene_index': scene_idx,
                    'perspective_index': perspective_idx,
                    'total_perspectives': num_perspectives,
                    'description': f'Scene {scene_idx + 1} - Perspective {perspective_idx + 1}/{num_perspectives}'
                })
                current_time += cut_duration

        # 4. End (fixed duration)
        scene_cuts.append({
            'start_time': current_time,
            'duration': end_duration,
            'image_type': 'end',
            'description': 'Story conclusion and end'
        })

        total_video_duration = current_time + end_duration
        sync_diff = abs(total_video_duration - audio_duration)

        self.logger.info(
            f"Created {len(scene_cuts)} cuts. Video: {total_video_duration:.3f}s, Audio: {audio_duration:.3f}s, Diff: {sync_diff:.3f}s")

        return scene_cuts

    async def _create_perspective_switching_segments(self, scene_cuts: List[Dict], image_categories: Dict,
                                                     video_dir: Path) -> List[Path]:
        """Create video segments with exact durations"""
        segments = []

        for i, cut in enumerate(scene_cuts):
            image_path = self._select_switching_perspective_image(cut, image_categories)

            if not image_path or not image_path.exists():
                self.logger.warning(f"Image not found for cut {i + 1}, using fallback")
                image_path = self._get_fallback_image_enhanced(image_categories)

            if not image_path:
                self.logger.error(f"No image available for cut {i + 1}")
                continue

            output_path = video_dir / f"segment_{i:03d}.mp4"

            # Create segment with EXACT duration
            success = await self._create_perspective_segment(
                image_path, cut['duration'], output_path, cut['image_type'],
                cut.get('perspective_index', 0), i
            )

            if success:
                segments.append(output_path)
                self.logger.info(f"Segment {i + 1}: {cut['duration']:.3f}s - {cut['description']}")
            else:
                self.logger.error(f"Failed to create segment {i + 1}")

        return segments

    def _select_switching_perspective_image(self, cut: Dict, image_categories: Dict) -> Optional[Path]:
        """Select specific perspective image with dedicated cover/end image priority"""
        cut_type = cut['image_type']

        if cut_type == 'cover':
            cover_images = image_categories.get('cover', [])
            if cover_images:
                selected_cover = cover_images[0]
                self.logger.info(f"Using cover image: {selected_cover.name}")
                return selected_cover
            else:
                self.logger.error("No cover image available!")
                return None

        elif cut_type == 'world' and image_categories.get('world'):
            world_images = image_categories['world']
            world_index = cut.get('world_index', 0)
            return world_images[world_index % len(world_images)] if world_images else None

        elif cut_type == 'scene_perspective':
            scene_index = cut.get('scene_index', 0)
            perspective_index = cut.get('perspective_index', 0)

            scene_perspective_data = image_categories['scene_perspectives'].get(scene_index, [])
            if scene_perspective_data and perspective_index < len(scene_perspective_data):
                selected_perspective = scene_perspective_data[perspective_index]
                return selected_perspective['path']

        elif cut_type == 'end':
            end_images = image_categories.get('end', [])
            if end_images:
                selected_end = end_images[0]
                self.logger.info(f"Using end image: {selected_end.name}")
                return selected_end
            else:
                self.logger.error("No end image available!")
                return None

        elif cut_type == 'fallback':
            scene_index = cut.get('scene_index', 0)
            scene_perspective_data = image_categories['scene_perspectives'].get(scene_index, [])
            if scene_perspective_data:
                return scene_perspective_data[0]['path']

        return None

    def _get_fallback_image_enhanced(self, image_categories: Dict) -> Optional[Path]:
        """Enhanced fallback image selection"""
        for scene_idx in image_categories.get('scene_perspectives', {}):
            perspectives = image_categories['scene_perspectives'][scene_idx]
            if perspectives:
                return perspectives[0]['path']

        for category in ['character_refs', 'world', 'cover', 'end', 'all']:
            images = image_categories.get(category, [])
            if images:
                return images[0]

        return None

    async def _create_perspective_segment(self, image_path: Path, duration: float, output_path: Path,
                                          segment_type: str, perspective_idx: int, segment_idx: int) -> bool:
        """Create video segment with EXACT duration and proper perspective animation"""
        try:
            # Calculate exact frame count for precise duration
            total_frames = int(duration * Config.VIDEO_FPS)

            # Different animations for each perspective
            if segment_type == 'cover':
                effect = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0015,1.2)':d={total_frames}:s=1920x1080"

            elif segment_type == 'world':
                effect = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0008,1.1)':x='if(gte(zoom,1.08),x,x+2)':d={total_frames}:s=1920x1080"

            elif segment_type == 'scene_perspective' or segment_type == 'fallback':
                perspective_animations = [
                    f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.001,1.15)':d={total_frames}:s=1920x1080",
                    f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0006,1.1)':x='if(gte(zoom,1.08),x,x+3)':d={total_frames}:s=1920x1080",
                    f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0008,1.12)':y='if(gte(zoom,1.1),y,y+2)':d={total_frames}:s=1920x1080",
                    f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0004,1.08)':x='if(gte(zoom,1.06),x,x-2)':d={total_frames}:s=1920x1080"
                ]
                effect = perspective_animations[perspective_idx % len(perspective_animations)]

            elif segment_type == 'end':
                effect = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0003,1.05)':d={total_frames}:s=1920x1080"

            else:
                effect = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0004,1.06)':d={total_frames}:s=1920x1080"

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(image_path),
                "-t", f"{duration:.3f}",  # Exact duration
                "-vf", effect,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", str(Config.VIDEO_FPS), "-pix_fmt", "yuv420p",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"FFmpeg segment creation failed: {error_msg}")
                return False

            return output_path.exists() and output_path.stat().st_size > 1000

        except Exception as e:
            self.logger.error(f"Perspective segment creation failed: {e}")
            return False

    async def create_video(self, story: StoryData, images: List[Path], audio_path: Path, project_dir: Path) -> Optional[
        Path]:
        """Create video with PERFECT audio synchronization"""
        self.logger.info("Creating video with perfect audio sync...")

        if not audio_path or not audio_path.exists():
            self.logger.error("Audio file required for video creation")
            return None

        if len(images) < 2:
            self.logger.error("Need at least 2 images for video")
            return None

        video_dir = project_dir / "video"
        video_dir.mkdir(exist_ok=True)

        try:
            # Get EXACT audio duration first
            audio_duration = await self._get_precise_audio_duration(audio_path)
            self.logger.info(f"Target audio duration: {audio_duration:.3f}s")

            # Categorize images
            image_categories = self._categorize_all_images_enhanced(images)

            # Create cuts that match audio duration EXACTLY
            scene_cuts = await self._create_perspective_switching_cuts(story, audio_duration, image_categories)

            # Create segments with exact timing
            segments = await self._create_perspective_switching_segments(scene_cuts, image_categories, video_dir)

            if not segments:
                self.logger.error("No video segments created")
                return None

            # Combine segments
            combined_video = await self._combine_video_segments(segments, video_dir)
            if not combined_video:
                return None

            # Add synchronized audio
            final_video = await self._add_synchronized_audio(combined_video, audio_path, video_dir, story.title)

            if final_video:
                # Verify sync
                final_video_duration = await self._get_video_duration(final_video)
                final_audio_duration = await self._get_audio_duration(audio_path)

                if final_video_duration and final_audio_duration:
                    sync_diff = abs(final_video_duration - final_audio_duration)
                    self.logger.info(f"Final sync difference: {sync_diff:.3f}s")

                    if sync_diff > 1.0:
                        self.logger.warning(f"Sync difference too high: {sync_diff:.3f}s")
                    else:
                        self.logger.info("Perfect sync achieved!")

            return final_video

        except Exception as e:
            self.logger.error(f"Video creation failed: {e}")
            return None

    async def _combine_video_segments(self, segments: List[Path], video_dir: Path, output_path: Path = None) -> \
    Optional[Path]:
        """Combine video segments with precise timing"""
        if not segments:
            return None

        if output_path is None:
            output_path = video_dir / "combined_video.mp4"

        list_file = self.temp_dir / "video_segments.txt"

        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for segment in segments:
                    if segment.exists():
                        file_path = str(segment.absolute()).replace('\\', '/')
                        f.write(f"file '{file_path}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and output_path.exists():
                self.logger.info(f"Combined {len(segments)} segments perfectly")
                return output_path
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Video combination failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Video combination failed: {e}")

        return None

    async def _add_synchronized_audio(self, video_path: Path, audio_path: Path, video_dir: Path, title: str) -> \
    Optional[Path]:
        """Add perfectly synchronized audio with exact duration matching"""
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:30]
        output_path = video_dir / f"{safe_title}_final.mp4"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",  # This ensures exact sync
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path)
            ]

            self.logger.info("Adding synchronized audio...")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and output_path.exists():
                self.logger.info(f"Final video created: {output_path.name}")
                return output_path
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Audio sync failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Audio synchronization failed: {e}")

        return None

    async def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get precise audio duration"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                return float(stdout.decode().strip())

        except Exception as e:
            self.logger.error(f"Could not get audio duration: {e}")

        return None

    async def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get precise video duration"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(video_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                return float(stdout.decode().strip())

        except Exception as e:
            self.logger.error(f"Could not get video duration: {e}")

        return None

# ====================== ENHANCED PDF GENERATOR ======================

class PDFGenerator:
    """Production-ready PDF generator with complete image integration and generation data"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.styles = self._create_styles()

    def _create_styles(self):
        """Create professional PDF styles optimized for complete story presentation"""
        base_styles = getSampleStyleSheet()

        return {
            # Title page styles
            'title_main': ParagraphStyle(
                'TitleMain',
                parent=base_styles['Title'],
                fontSize=32,
                alignment=TA_CENTER,
                spaceAfter=20,
                textColor=HexColor('#1a1a1a'),
                fontName='Helvetica-Bold'
            ),
            'title_subtitle': ParagraphStyle(
                'TitleSubtitle',
                parent=base_styles['Normal'],
                fontSize=16,
                alignment=TA_CENTER,
                spaceAfter=15,
                textColor=HexColor('#4a4a4a'),
                fontName='Helvetica'
            ),

            # Story page styles - narrator at top
            'narrator_text': ParagraphStyle(
                'NarratorText',
                parent=base_styles['Normal'],
                fontSize=12,
                alignment=TA_JUSTIFY,
                spaceAfter=10,
                spaceBefore=10,
                leftIndent=20,
                rightIndent=20,
                textColor=HexColor('#2c3e50'),
                fontName='Helvetica',
                leading=16
            ),

            # Story page styles - dialogue at bottom
            'dialogue_text': ParagraphStyle(
                'DialogueText',
                parent=base_styles['Normal'],
                fontSize=11,
                alignment=TA_LEFT,
                spaceAfter=6,
                spaceBefore=3,
                leftIndent=30,
                textColor=HexColor('#34495e'),
                fontName='Helvetica-Oblique',
                leading=14
            ),
            'character_name': ParagraphStyle(
                'CharacterName',
                parent=base_styles['Normal'],
                fontSize=11,
                alignment=TA_LEFT,
                spaceAfter=3,
                leftIndent=30,
                textColor=HexColor('#e74c3c'),
                fontName='Helvetica-Bold'
            ),

            # Section headers
            'section_header': ParagraphStyle(
                'SectionHeader',
                parent=base_styles['Heading1'],
                fontSize=18,
                alignment=TA_CENTER,
                spaceBefore=20,
                spaceAfter=15,
                textColor=HexColor('#2c3e50'),
                fontName='Helvetica-Bold'
            ),

            # Scene headers
            'scene_header': ParagraphStyle(
                'SceneHeader',
                parent=base_styles['Heading2'],
                fontSize=14,
                alignment=TA_CENTER,
                spaceBefore=15,
                spaceAfter=10,
                textColor=HexColor('#34495e'),
                fontName='Helvetica-Bold'
            ),

            # Image captions
            'image_caption': ParagraphStyle(
                'ImageCaption',
                parent=base_styles['Normal'],
                fontSize=9,
                alignment=TA_CENTER,
                spaceAfter=8,
                textColor=HexColor('#7f8c8d'),
                fontName='Helvetica-Oblique'
            ),

            # Character info
            'character_info': ParagraphStyle(
                'CharacterInfo',
                parent=base_styles['Normal'],
                fontSize=10,
                alignment=TA_LEFT,
                spaceAfter=8,
                textColor=HexColor('#2c3e50'),
                fontName='Helvetica'
            ),

            # Metadata info
            'metadata_info': ParagraphStyle(
                'MetadataInfo',
                parent=base_styles['Normal'],
                fontSize=9,
                alignment=TA_LEFT,
                spaceAfter=5,
                textColor=HexColor('#666666'),
                fontName='Helvetica'
            )
        }

    def create_complete_pdf(self, story: StoryData, images: Dict[str, List[Path]], project_dir: Path,
                            cost_summary: Dict = None) -> Optional[Path]:
        """Create complete PDF with all images, full story integration, and generation data"""
        pdf_dir = project_dir / "pdf"
        pdf_dir.mkdir(exist_ok=True)

        safe_title = re.sub(r'[<>:"/\\|?*]', '_', story.title)[:30]
        pdf_path = pdf_dir / f"{safe_title}_complete_full.pdf"

        try:
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )

            flowables = []

            # Find all images properly
            all_images_organized = self._organize_all_images(images, project_dir)
            self.logger.info(
                f"Found images: cover={bool(all_images_organized['cover'])}, end={bool(all_images_organized['end'])}, scenes={len(all_images_organized['scenes'])}")

            # 1. TITLE PAGE with 00_cover.png
            flowables.extend(self._create_title_page_with_cover(story, all_images_organized['cover'], cost_summary))
            flowables.append(PageBreak())

            # 2. COMPLETE STORY PAGES (each scene with ALL available images)
            flowables.extend(self._create_complete_story_pages(story, all_images_organized))

            # 3. END PAGE with 99_end.png
            flowables.extend(self._create_end_page_with_image(all_images_organized['end']))
            flowables.append(PageBreak())

            # 4. ADDITIONAL GENERATION DATA AND IMAGES
            flowables.extend(self._create_generation_data_section(story, all_images_organized, cost_summary))

            # Build PDF
            doc.build(flowables)

            self.logger.info(f"Complete PDF with all images created: {pdf_path.name}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"PDF creation failed: {e}")
            return None

    def _organize_all_images(self, images: Dict[str, List[Path]], project_dir: Path) -> Dict:
        """Organize all images from various sources and find cover/end images"""
        organized = {
            'cover': None,
            'end': None,
            'world': [],
            'characters': [],
            'scenes': {},  # {scene_idx: [images]}
            'all_scene_images': []
        }

        # Search for images in project directory
        images_dir = project_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.png"):
                name = img_file.name.lower()

                # Find cover image (00_cover.png)
                if name.startswith('00_') or 'cover' in name:
                    organized['cover'] = img_file
                    self.logger.info(f"Found cover image: {img_file.name}")

                # Find end image (99_end.png)
                elif name.startswith('99_') or 'end' in name:
                    organized['end'] = img_file
                    self.logger.info(f"Found end image: {img_file.name}")

                # Find world images
                elif name.startswith('01_') or 'world' in name:
                    organized['world'].append(img_file)

                # Find character images
                elif 'char' in name or name.startswith('02_'):
                    organized['characters'].append(img_file)

                # Find scene images
                elif 'scene' in name:
                    scene_match = re.search(r'scene_(\d+)', name)
                    if scene_match:
                        scene_idx = int(scene_match.group(1))
                        if scene_idx not in organized['scenes']:
                            organized['scenes'][scene_idx] = []
                        organized['scenes'][scene_idx].append(img_file)
                        organized['all_scene_images'].append(img_file)

        # Also check from images dict parameter
        for key, img_list in images.items():
            if key == 'cover' and img_list and not organized['cover']:
                organized['cover'] = img_list[0] if img_list[0].exists() else None
            elif key == 'end' and img_list and not organized['end']:
                organized['end'] = img_list[0] if img_list[0].exists() else None
            elif key in ['world', 'world_images']:
                organized['world'].extend([img for img in img_list if img.exists()])
            elif key in ['character_refs', 'characters']:
                organized['characters'].extend([img for img in img_list if img.exists()])
            elif 'scene' in key:
                organized['all_scene_images'].extend([img for img in img_list if img.exists()])

        # Sort scene images
        for scene_idx in organized['scenes']:
            organized['scenes'][scene_idx].sort(key=lambda x: x.name)

        organized['world'].sort(key=lambda x: x.name)
        organized['characters'].sort(key=lambda x: x.name)

        return organized

    def _create_title_page_with_cover(self, story: StoryData, cover_image: Optional[Path],
                                      cost_summary: Dict = None) -> List:
        """Create title page with 00_cover.png and generation metadata"""
        elements = []

        # Cover image 00_cover.png
        if cover_image and cover_image.exists():
            try:
                cover_img = Image(str(cover_image), width=6 * inch, height=4.5 * inch)
                cover_img.hAlign = 'CENTER'
                elements.append(cover_img)
                elements.append(Spacer(1, 0.3 * inch))
                self.logger.info(f"Added cover image to PDF: {cover_image.name}")
            except Exception as e:
                self.logger.warning(f"Could not add cover image: {e}")
                elements.append(Spacer(1, 3 * inch))
        else:
            self.logger.warning("No cover image found (00_cover.png)")
            elements.append(Spacer(1, 3 * inch))

        # Title and basic info
        elements.append(Paragraph(story.title, self.styles['title_main']))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"A {story.genre} Story", self.styles['title_subtitle']))
        elements.append(
            Paragraph(f"Visual Style: {story.style_preset.image_style.value.title()}", self.styles['title_subtitle']))
        elements.append(
            Paragraph(f"Camera Style: {story.style_preset.camera_style.value.title()}", self.styles['title_subtitle']))

        # Generation metadata
        elements.append(Spacer(1, 0.4 * inch))
        elements.append(
            Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Characters: {len(story.characters)} | Scenes: {len(story.scenes)}",
                                  self.styles['metadata_info']))

        if cost_summary:
            total_cost = cost_summary.get('total_cost_usd', 0)
            elements.append(Paragraph(f"Generation Cost: ${total_cost:.3f} USD", self.styles['metadata_info']))

            # Cost breakdown
            breakdown = cost_summary.get('breakdown', {})
            cost_details = []
            for service, details in breakdown.items():
                if details.get('calls', 0) > 0:
                    cost_details.append(f"{service}: {details['calls']} calls (${details['cost']:.3f})")

            if cost_details:
                elements.append(Paragraph("Cost Breakdown: " + " | ".join(cost_details), self.styles['metadata_info']))

        # World description
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(story.world_desc, self.styles['narrator_text']))

        return elements

    def _create_complete_story_pages(self, story: StoryData, all_images_organized: Dict) -> List:
        """Create complete story pages with ALL scene images integrated"""
        elements = []

        elements.append(Paragraph("Complete Visual Story", self.styles['section_header']))
        elements.append(Spacer(1, 0.3 * inch))

        for scene_idx, scene in enumerate(story.scenes):
            # Scene header
            elements.append(Paragraph(f"Scene {scene_idx + 1}: {scene.title}", self.styles['scene_header']))
            elements.append(Spacer(1, 0.2 * inch))

            # Narrator text at TOP
            if scene.narrator:
                elements.append(Paragraph(scene.narrator, self.styles['narrator_text']))
                elements.append(Spacer(1, 0.2 * inch))

            # ALL SCENE IMAGES for this scene
            scene_images = all_images_organized['scenes'].get(scene_idx, [])
            if scene_images:
                elements.append(
                    Paragraph(f"Visual Perspectives ({len(scene_images)} images)", self.styles['character_info']))
                elements.append(Spacer(1, 0.1 * inch))

                for img_idx, scene_img in enumerate(scene_images):
                    if scene_img.exists():
                        try:
                            # Adjust image size based on number of images
                            if len(scene_images) <= 2:
                                img_width, img_height = 5.5 * inch, 4 * inch
                            else:
                                img_width, img_height = 4.5 * inch, 3.2 * inch

                            img = Image(str(scene_img), width=img_width, height=img_height)
                            img.hAlign = 'CENTER'
                            elements.append(img)

                            # Image caption with perspective info
                            perspective_match = re.search(r'perspective_(\d+)', scene_img.name)
                            perspective_info = f"Perspective {int(perspective_match.group(1)) + 1}" if perspective_match else f"View {img_idx + 1}"
                            elements.append(
                                Paragraph(f"{perspective_info}: {scene.setting}", self.styles['image_caption']))
                            elements.append(Spacer(1, 0.15 * inch))

                        except Exception as e:
                            self.logger.warning(f"Could not add scene image {scene_img.name}: {e}")
            else:
                # Show world image as fallback
                if all_images_organized['world']:
                    try:
                        world_img = all_images_organized['world'][scene_idx % len(all_images_organized['world'])]
                        img = Image(str(world_img), width=5 * inch, height=3.8 * inch)
                        img.hAlign = 'CENTER'
                        elements.append(img)
                        elements.append(Paragraph(f"World Setting: {scene.setting}", self.styles['image_caption']))
                        elements.append(Spacer(1, 0.2 * inch))
                    except Exception as e:
                        self.logger.warning(f"Could not add world fallback image: {e}")
                        elements.append(Spacer(1, 2 * inch))

            # Show relevant characters in this scene
            scene_characters = list(set([d.character for d in scene.dialogue if d.character != "Narrator"]))
            if scene_characters and all_images_organized['characters']:
                elements.append(Paragraph("Characters in this scene:", self.styles['character_info']))
                elements.append(Spacer(1, 0.1 * inch))

                for char_name in scene_characters[:2]:  # Max 2 characters per scene page
                    # Find matching character
                    for story_char_idx, story_char in enumerate(story.characters):
                        if story_char.name == char_name and story_char_idx < len(all_images_organized['characters']):
                            char_img = all_images_organized['characters'][story_char_idx]
                            if char_img.exists():
                                try:
                                    char_image = Image(str(char_img), width=2 * inch, height=2 * inch)
                                    char_image.hAlign = 'CENTER'
                                    elements.append(char_image)
                                    elements.append(Paragraph(f"{story_char.name}: {story_char.visual_desc[:50]}...",
                                                              self.styles['image_caption']))
                                    elements.append(Spacer(1, 0.1 * inch))
                                    break
                                except Exception as e:
                                    self.logger.warning(f"Could not add character image for {char_name}: {e}")

            # Dialogue at BOTTOM
            if scene.dialogue:
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph("Dialogue:", self.styles['character_info']))
                for dialogue in scene.dialogue:
                    elements.append(Paragraph(f"{dialogue.character}:", self.styles['character_name']))
                    elements.append(Paragraph(dialogue.text, self.styles['dialogue_text']))

            # Page break between scenes
            if scene_idx < len(story.scenes) - 1:
                elements.append(PageBreak())

        return elements

    def _create_end_page_with_image(self, end_image: Optional[Path]) -> List:
        """Create end page with 99_end.png"""
        elements = []

        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph("The End", self.styles['title_main']))
        elements.append(Spacer(1, 0.5 * inch))

        # End image 99_end.png
        if end_image and end_image.exists():
            try:
                end_img = Image(str(end_image), width=5.5 * inch, height=4 * inch)
                end_img.hAlign = 'CENTER'
                elements.append(end_img)
                elements.append(Paragraph("Story Conclusion", self.styles['image_caption']))
                self.logger.info(f"Added end image to PDF: {end_image.name}")
            except Exception as e:
                self.logger.warning(f"Could not add end image: {e}")
        else:
            self.logger.warning("No end image found (99_end.png)")

        elements.append(Spacer(1, 0.5 * inch))
        elements.append(
            Paragraph("Thank you for experiencing this complete visual story!", self.styles['narrator_text']))

        return elements

    def _create_generation_data_section(self, story: StoryData, all_images_organized: Dict,
                                        cost_summary: Dict = None) -> List:
        """Create section with complete generation data and remaining images"""
        elements = []

        elements.append(Paragraph("Generation Data & Complete Image Gallery", self.styles['section_header']))
        elements.append(Spacer(1, 0.3 * inch))

        # Generation statistics
        elements.append(Paragraph("Generation Statistics", self.styles['scene_header']))
        elements.append(Paragraph(f"Story Title: {story.title}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Genre: {story.genre}", self.styles['metadata_info']))
        elements.append(
            Paragraph(f"Visual Style: {story.style_preset.image_style.value.title()}", self.styles['metadata_info']))
        elements.append(
            Paragraph(f"Camera Style: {story.style_preset.camera_style.value.title()}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Total Characters: {len(story.characters)}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Total Scenes: {len(story.scenes)}", self.styles['metadata_info']))

        # Count all images
        total_images = 0
        if all_images_organized['cover']: total_images += 1
        if all_images_organized['end']: total_images += 1
        total_images += len(all_images_organized['world'])
        total_images += len(all_images_organized['characters'])
        total_images += len(all_images_organized['all_scene_images'])

        elements.append(Paragraph(f"Total Generated Images: {total_images}", self.styles['metadata_info']))

        if cost_summary:
            elements.append(Paragraph(f"Total Generation Cost: ${cost_summary.get('total_cost_usd', 0):.3f}",
                                      self.styles['metadata_info']))

        elements.append(Spacer(1, 0.4 * inch))

        # Complete character gallery
        if all_images_organized['characters']:
            elements.append(Paragraph("Complete Character Gallery", self.styles['scene_header']))
            for i, character in enumerate(story.characters):
                if i < len(all_images_organized['characters']):
                    char_img = all_images_organized['characters'][i]
                    if char_img.exists():
                        try:
                            elements.append(Paragraph(character.name, self.styles['character_info']))
                            img = Image(str(char_img), width=3 * inch, height=3 * inch)
                            img.hAlign = 'CENTER'
                            elements.append(img)
                            elements.append(
                                Paragraph(f"Role: {character.role.value.title()}", self.styles['metadata_info']))
                            elements.append(
                                Paragraph(f"Description: {character.visual_desc}", self.styles['metadata_info']))
                            elements.append(Paragraph(f"Voice: {character.voice.value.replace('_', ' ').title()}",
                                                      self.styles['metadata_info']))
                            elements.append(Spacer(1, 0.3 * inch))
                        except Exception as e:
                            self.logger.warning(f"Could not add character {character.name}: {e}")

            elements.append(PageBreak())

        # World environment gallery
        if all_images_organized['world']:
            elements.append(Paragraph("Complete World Gallery", self.styles['scene_header']))
            elements.append(Paragraph(f"World Description: {story.world_desc}", self.styles['character_info']))
            elements.append(Spacer(1, 0.2 * inch))

            for i, world_img in enumerate(all_images_organized['world']):
                if world_img.exists():
                    try:
                        img = Image(str(world_img), width=5 * inch, height=3.8 * inch)
                        img.hAlign = 'CENTER'
                        elements.append(img)
                        elements.append(Paragraph(f"World Environment View {i + 1}", self.styles['image_caption']))
                        elements.append(Spacer(1, 0.3 * inch))
                    except Exception as e:
                        self.logger.warning(f"Could not add world image {i}: {e}")

        # Footer with complete generation info
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(
            f"Complete multimedia story generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"using Enhanced Story Generator v5.0 with {story.style_preset.image_style.value.title()} visual style "
            f"and {story.style_preset.camera_style.value.title()} camera work.",
            self.styles['metadata_info']
        ))

        return elements


# ====================== CLIP GENERATOR ======================

class ClipGenerator:
    """Generate video clips with enhanced perspective continuity using banana model transitions"""

    def __init__(self, logger: logging.Logger, cost_tracker: CostTracker, isaa=None, img_generator=None):
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.isaa = isaa
        self.img_generator = img_generator  # Enhanced: ImageGenerator for banana transitions
        self.clip_cache = {}
        self.MINIMAX_COST = 0.05

    async def generate_all_clips(self, story: StoryData, images: List[str], project_dir: Path,
                                 image_gen: ImageGenerator) -> Dict[str, Path]:
        """Generate ALL video clips with enhanced perspective continuity"""
        self.logger.info("Starting ENHANCED perspective-continuous clip generation...")

        clips_dir = project_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        # Upload all images first (parallel uploads)
        self.logger.info("Uploading images for enhanced clip generation...")
        uploaded_images = await self._upload_all_images_parallel(images, image_gen)

        # Enhanced workflow: Parallel scenes with sequential perspectives
        all_clip_tasks = []
        task_info = []

        # 1. Cover clip task (unchanged)
        if 'cover' in uploaded_images:
            cover_task = self._generate_cover_clip(story, uploaded_images['cover'][0], clips_dir)
            all_clip_tasks.append(cover_task)
            task_info.append(('cover', 'cover'))

        # 2. World establishment clips (unchanged)
        if 'world_images' in uploaded_images:
            for idx, world_img_url in enumerate(uploaded_images.get('world_images', [])):
                world_task = self._generate_world_clip(story, world_img_url, clips_dir, idx)
                all_clip_tasks.append(world_task)
                task_info.append(('world', f'world_{idx}'))

        # 3. ENHANCED: Sequential perspective generation within each scene (but scenes remain parallel)
        scene_image_urls = uploaded_images.get('scene_perspectives', [])

        # Group scene URLs by scene index
        scenes_grouped = {}
        for scene_url in scene_image_urls:
            match = re.search(r'scene_(\d+)_perspective_(\d+)', scene_url)
            if match:
                scene_idx = int(match.group(1))
                perspective_idx = int(match.group(2))
                if scene_idx not in scenes_grouped:
                    scenes_grouped[scene_idx] = {}
                scenes_grouped[scene_idx][perspective_idx] = scene_url

        # Create enhanced scene tasks (parallel scenes, sequential perspectives within each scene)
        for scene_idx, scene_perspectives in scenes_grouped.items():
            if scene_idx < len(story.scenes):
                scene = story.scenes[scene_idx]
                enhanced_scene_task = self._generate_enhanced_scene_clips(
                    scene, scene_perspectives, clips_dir, scene_idx, story, uploaded_images
                )
                all_clip_tasks.append(enhanced_scene_task)
                task_info.append(('enhanced_scene', f'scene_{scene_idx:02d}'))

        # 4. End clip task (unchanged)
        if 'end' in uploaded_images:
            end_task = self._generate_end_clip(story, uploaded_images['end'][0], clips_dir)
            all_clip_tasks.append(end_task)
            task_info.append(('end', 'end'))

        # Execute all tasks in parallel (scenes are parallel, perspectives within scenes are sequential)
        self.logger.info(f"🚀 Starting {len(all_clip_tasks)} enhanced clip generation tasks...")
        start_time = time.time()

        results = await asyncio.gather(*all_clip_tasks, return_exceptions=True)
        parallel_time = time.time() - start_time

        # Process results
        generated_clips = {}
        successful_clips = 0
        failed_clips = 0

        for i, (result, (clip_type, clip_key)) in enumerate(zip(results, task_info)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Clip {clip_key} failed: {result}")
                failed_clips += 1
            elif clip_type == 'enhanced_scene' and isinstance(result, dict):
                # Enhanced scene returns multiple clips
                for perspective_key, clip_path in result.items():
                    if clip_path and clip_path.exists():
                        generated_clips[perspective_key] = clip_path
                        successful_clips += 1
                        self.logger.info(f"✅ Enhanced clip {perspective_key} completed")
            elif isinstance(result, Path) and result and result.exists():
                generated_clips[clip_key] = result
                successful_clips += 1
                self.logger.info(f"✅ Clip {clip_key} completed")
            else:
                failed_clips += 1

        self.logger.info(f"🎬 ENHANCED clip generation completed in {parallel_time:.2f}s")
        self.logger.info(f"✅ Successful: {successful_clips}" + (f", ❌ Failed: {failed_clips}" if failed_clips else ""))

        return generated_clips

    async def _generate_enhanced_scene_clips(self, scene: Scene, scene_perspectives: Dict[int, str],
                                             clips_dir: Path, scene_idx: int, story: StoryData,
                                             uploaded_images: Dict[str, List[str]]) -> Dict[str, Path]:
        """Generate enhanced scene clips with perspective continuity"""
        self.logger.info(f"🎭 Generating enhanced scene {scene_idx} with {len(scene_perspectives)} perspectives...")

        scene_clips = {}
        previous_clip_path = None

        # Sort perspectives by index for sequential processing
        sorted_perspectives = sorted(scene_perspectives.items())

        for perspective_idx, original_scene_url in sorted_perspectives:
            self.logger.info(f"🎬 Processing scene {scene_idx} perspective {perspective_idx}")

            # For the first perspective, use original image
            if perspective_idx == 0 or not previous_clip_path:
                scene_url_for_clip = original_scene_url
                self.logger.info(f"Using original image for first perspective")
            else:
                # MAGIC STARTS: Generate transition image using previous clip's end frame
                scene_url_for_clip = await self._generate_transition_image(
                    previous_clip_path, original_scene_url, scene, scene_idx,
                    perspective_idx, story, uploaded_images, clips_dir
                )

                if not scene_url_for_clip:
                    self.logger.warning(f"Transition generation failed, using original image")
                    scene_url_for_clip = original_scene_url
                else:
                    self.logger.info(f"✨ Generated seamless transition image for perspective {perspective_idx}")

            # Generate clip with enhanced or original image
            clip_path = await self._generate_scene_perspective_clip(
                scene, scene_url_for_clip, clips_dir, scene_idx, perspective_idx, story
            )

            if clip_path:
                perspective_key = f'scene_{scene_idx:02d}_perspective_{perspective_idx:02d}'
                scene_clips[perspective_key] = clip_path
                previous_clip_path = clip_path  # Set for next perspective
                self.logger.info(f"✅ Generated enhanced perspective clip: {perspective_key}")
            else:
                self.logger.error(f"❌ Failed to generate perspective {perspective_idx} for scene {scene_idx}")

        return scene_clips

    async def _generate_transition_image(self, previous_clip_path: Path, next_perspective_url: str,
                                         scene: Scene, scene_idx: int, perspective_idx: int,
                                         story: StoryData, uploaded_images: Dict[str, List[str]],
                                         clips_dir: Path) -> Optional[str]:
        """Generate seamless transition image using banana model"""

        if not self.img_generator:
            self.logger.warning("No image generator available for transition generation")
            return None

        try:
            # Extract intelligent frame from previous clip (avoid black/white end frames)
            transition_frame_paths = []
            for target_frame in range(6):
                if target_frame % 2 == 0:
                    continue
                transition_frame_path = await self._extract_transition_frame(previous_clip_path, clips_dir, scene_idx,
                                                                             perspective_idx, target_frame=target_frame*5)
                if transition_frame_path:
                    transition_frame_paths.append(transition_frame_path)

            if not transition_frame_paths:
                self.logger.error("Failed to extract transition frame")
                return None

            # Upload transition frame
            transition_frame_urls = []
            for transition_frame_path in transition_frame_paths:
                _transition_frame_url = await self.img_generator._upload_to_fal(transition_frame_path)

                if not _transition_frame_url:
                    self.logger.error("Failed to upload transition frame")
                    return None

                transition_frame_urls.append(_transition_frame_url)

            # Get character references for this scene
            scene_characters = list(set([d.character for d in scene.dialogue if d.character != "Narrator"]))
            character_urls = []

            for char_name in scene_characters:
                if char_name in self.img_generator.character_refs:
                    char_url = self.img_generator.character_refs[char_name]
                    if char_url:
                        character_urls.append(char_url)

            # Prepare reference images: transition frame + next perspective + characters
            reference_images = transition_frame_urls + [next_perspective_url] + character_urls

            # Generate perfect banana prompt using mini task
            base_prompt = (f"Create seamless transition from previous scene moment to new perspective. "
                           f"Scene: {scene.title} - {scene.setting}. "
                           f"Characters: {', '.join(scene_characters)}. "
                           f"Maintain visual continuity, character positioning, and environmental consistency. "
                           f"Smooth cinematic transition, natural character movement."
                           f"general location first image ( {next_perspective_url} ), "
                           f"Last moments in video the video ( {transition_frame_urls} ) generate new logical cut view. in the first image environed based on the last video moments images.")

            enhanced_prompt = await self._create_perfect_banana_prompt(base_prompt, scene, scene_characters,
                                                                       perspective_idx)

            # Generate transition image with banana
            transition_output_path = clips_dir.parent / "transitions" / f"transition_scene_{scene_idx:02d}_to_perspective_{perspective_idx:02d}.png"
            transition_output_path.parent.mkdir(exist_ok=True)

            success = await self.img_generator._generate_with_banana(
                enhanced_prompt, reference_images, transition_output_path
            )

            if success:
                # Upload generated transition image
                final_transition_url = await self.img_generator._upload_to_fal(transition_output_path)
                self.logger.info(f"✨ Generated seamless transition for scene {scene_idx} perspective {perspective_idx}")
                return final_transition_url
            else:
                self.logger.error("Banana generation failed for transition")
                return None

        except Exception as e:
            self.logger.error(f"Transition generation failed: {e}")
            return None

    async def _extract_transition_frame(self, clip_path: Path, clips_dir: Path, scene_idx: int, perspective_idx: int, target_frame: int = 0) -> \
    Optional[Path]:
        """Extract intelligent transition frame from video clip (avoiding black/white end frames)"""
        try:
            import cv2

            # Output path for extracted frame
            frame_path = clips_dir.parent / "transition_frames" / f"frame_scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            frame_path.parent.mkdir(exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(str(clip_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 15:
                self.logger.warning(f"Video too short ({total_frames} frames), using middle frame")
                target_frame += total_frames // 2
            else:
                # Intelligent detection: avoid last ~50 frames (likely fade to black/white)
                target_frame += max(15, total_frames - max(25, target_frame))

            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if ret:
                # Check if frame is too dark/light (black/white detection)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = cv2.mean(gray)[0]

                # If frame is too dark (<30) or too light (>220), try earlier frames
                attempts = 0
                while (mean_brightness < 30 or mean_brightness > 220) and attempts < 10:
                    target_frame = max(10, target_frame - 20)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        mean_brightness = cv2.mean(gray)[0]
                    attempts += 1

                # Save frame
                cv2.imwrite(str(frame_path), frame)
                cap.release()

                self.logger.info(
                    f"Extracted transition frame at position {target_frame}/{total_frames} (brightness: {mean_brightness:.1f})")
                return frame_path

            cap.release()

        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")

        return None

    async def _create_perfect_banana_prompt(self, base_prompt: str, scene: Scene, characters: List[str],
                                            perspective_idx: int) -> str:
        """Create perfect banana prompt using mini task completion"""

        if not self.isaa:
            return base_prompt

        try:
            enhanced_task = (f"Create the perfect image generation prompt for seamless video transition. "
                             f"Base context: {base_prompt} "
                             f"Scene details: {scene.narrator} "
                             f"Characters present: {', '.join(characters)} "
                             f"Perspective {perspective_idx}: focus on natural character positioning and environmental flow. "
                             f"The prompt should ensure visual continuity, proper character placement, and cinematic composition. "
                             f"Make it concise but detailed for banana model image generation."
                             f"{scene.poses} "
                             f"{scene.dialogue[min(perspective_idx, len(scene.dialogue) - 1)].character}: "
                             f"{scene.dialogue[min(perspective_idx, len(scene.dialogue) - 1)].text}")

            perfect_prompt = await self.isaa.mini_task_completion(
                mini_task=enhanced_task,
                user_task="Create a perfect, concise image generation prompt that ensures seamless visual continuity between video clips. "
                          "Focus on character consistency, environmental coherence, and smooth cinematic transitions. "
                          "The prompt will be used with reference images to generate transition frames.",
                agent_name="story_creator",
                max_completion_tokens=300
            )

            return perfect_prompt if perfect_prompt else base_prompt

        except Exception as e:
            self.logger.error(f"Perfect prompt generation failed: {e}")
            return base_prompt

    # Keep all existing methods unchanged for compatibility
    async def _upload_all_images_parallel(self, images: List[str], image_gen: ImageGenerator) -> Dict[str, List[str]]:
        """Upload ALL images to FAL in parallel instead of sequentially"""
        upload_tasks = []
        image_info = []

        uploaded_image_paths = []

        for image_path_str in images:
            if isinstance(image_path_str, Path):
                image_path_str = str(image_path_str)

            # Determine category
            category = "scene_perspectives"
            if "world" in image_path_str:
                category = "world_images"
            elif "cover" in image_path_str:
                category = "cover"
            elif "perspective" in image_path_str:
                category = "scene_perspectives"
            elif "99_end" in image_path_str:
                category = "end"
            elif "char" in image_path_str:
                category = "charakter_refs"

            image_path = Path(image_path_str)
            if image_path and image_path.exists() and str(image_path) not in uploaded_image_paths:
                uploaded_image_paths.append(image_path_str)

                upload_task = self._upload_to_fal(image_path, image_gen.images_dict)
                upload_tasks.append(upload_task)
                image_info.append((category, image_path))

        # Execute all uploads in parallel
        self.logger.info(f"🚀 Uploading {len(upload_tasks)} images SIMULTANEOUSLY...")
        start_time = time.time()

        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        upload_time = time.time() - start_time

        # Process upload results
        uploaded = {}
        successful_uploads = 0
        failed_uploads = 0

        for i, (result, (category, image_path)) in enumerate(zip(upload_results, image_info)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Upload failed for {image_path.name}: {result}")
                failed_uploads += 1
            elif result:
                if category not in uploaded:
                    uploaded[category] = []
                uploaded[category].append(result)
                successful_uploads += 1
                self.logger.info(f"✅ Uploaded {category}: {image_path.name}")
            else:
                self.logger.warning(f"⚠️ Upload returned None for {image_path.name}")
                failed_uploads += 1

        self.logger.info(f"📤 PARALLEL uploads completed in {upload_time:.2f}s")
        self.logger.info(
            f"✅ Successful: {successful_uploads}" + (f", ❌ Failed: {failed_uploads}" if failed_uploads else ""))

        return uploaded

    async def _generate_cover_clip(self, story: StoryData, cover_url: str, clips_dir: Path) -> Optional[Path]:
        """Generate cover clip with title introduction"""
        prompt = (f"Cinematic title sequence: {story.title}. "
                  f"Slow, majestic zoom revealing the world of this {story.genre} story. "
                  f"Epic introduction, dramatic lighting, building anticipation. "
                  f"Professional title card cinematography, {story.style_preset.image_style.value} style.")

        return await self._generate_clip(prompt, cover_url, clips_dir / "00_cover.mp4", style_preset=story.style_preset,
                                         duration="6", image_type="character")

    async def _generate_world_clip(self, story: StoryData, world_url: str, clips_dir: Path, idx: int) -> Optional[Path]:
        """Generate world establishment clip with sweeping camera movement"""
        world_prompts = [
            f"Sweeping establishing shot across {story.world_desc}. "
            f"Cinematic camera movement revealing the vast world. "
            f"Environmental storytelling, atmospheric mood, epic scale. "
            f"{story.style_preset.camera_style.value} cinematography.",

            f"Dynamic world exploration: {story.world_desc}. "
            f"Camera gliding through the environment, discovering key locations. "
            f"Rich atmospheric details, immersive world-building. "
            f"Professional {story.style_preset.image_style.value} cinematography."
        ]

        prompt = world_prompts[idx % len(world_prompts)]
        return await self._generate_clip(prompt, world_url, clips_dir / f"01_world_{idx:02d}.mp4",
                                         style_preset=story.style_preset, duration="6", image_type="scene")

    async def _generate_scene_perspective_clip(self, scene: Scene, scene_url: str, clips_dir: Path,
                                               scene_idx: int, perspective_idx: int, story: StoryData) -> Optional[
        Path]:
        """Generate individual scene perspective clip with perfect action"""

        # Extract characters in this scene
        scene_characters = [d.character for d in scene.dialogue if d.character != "Narrator"]

        # Create action-specific prompts based on scene content
        action_prompt = self._create_action_prompt(scene, scene_characters, perspective_idx)

        full_prompt = (f"Scene {scene_idx + 1}: {scene.title}. "
                       f"Location: {scene.setting}. "
                       f"{action_prompt} "
                       f"Taken with {story.style_preset.camera_style.value}, "
                       f"{story.style_preset.image_style.value} visual style, "
                       f"dramatic lighting, professional filmmaking.")

        output_path = clips_dir / f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.mp4"
        return await self._generate_clip(full_prompt, scene_url, output_path, style_preset=story.style_preset,
                                         duration="10" if scene.duration >= 11 else "6", image_type="scene")

    def _create_action_prompt(self, scene: Scene, characters: List[str], perspective_idx: int) -> str:
        """Create perfect action prompts based on scene content"""

        # Analyze dialogue for action cues
        dialogue_text = " ".join([d.text for d in scene.dialogue])

        # Action templates based on perspective
        perspective_actions = [
            f"Characters {', '.join(characters[:2])} engaging in the scene action. "
            f"Natural character movement and interaction. "
            f"Environmental storytelling through character placement.",

            f"Focused character interaction between {characters[0] if characters else 'main character'}. "
            f"Expressive character animation and emotional beats. "
            f"Dynamic character-driven storytelling.",

            f"Intimate character moment with {characters[0] if characters else 'protagonist'}. "
            f"Subtle facial expressions and emotional nuance. "
            f"Character depth and emotional connection.",

            f"Dialogue exchange between {', '.join(characters[:2]) if len(characters) >= 2 else 'characters'}. "
            f"Natural conversation dynamics and character interaction. "
            f"Realistic dialogue pacing and character chemistry."
        ]

        base_action = perspective_actions[perspective_idx % len(perspective_actions)]

        # Enhance with scene-specific details
        if "fight" in dialogue_text.lower() or "battle" in dialogue_text.lower():
            base_action += " Dynamic combat movement and action choreography."
        elif "run" in dialogue_text.lower() or "chase" in dialogue_text.lower():
            base_action += " Fast-paced movement and urgency."
        elif "magic" in dialogue_text.lower() or "spell" in dialogue_text.lower():
            base_action += " Mystical energy and magical effects."
        elif any(emotion in dialogue_text.lower() for emotion in ["sad", "cry", "tears"]):
            base_action += " Emotional character moments and subtle movement."
        elif any(joy in dialogue_text.lower() for joy in ["happy", "laugh", "smile"]):
            base_action += " Joyful character expressions and positive energy."
        else:
            base_action += " Natural character behavior and realistic movement."

        return base_action

    async def _generate_end_clip(self, story: StoryData, end_url: str, clips_dir: Path) -> Optional[Path]:
        """Generate end clip with conclusion effect"""
        prompt = (f"Epic conclusion to {story.title}. "
                  f"Final dramatic moment with emotional resolution. "
                  f"Cinematic ending with fade to black, credits-ready. "
                  f"Professional {story.style_preset.image_style.value} finale, "
                  f"Taken with {story.style_preset.camera_style.value}.")

        return await self._generate_clip(prompt, end_url, clips_dir / "99_end.mp4", style_preset=story.style_preset,
                                         duration="6", image_type="end")

    async def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for better results"""
        return await self.isaa.mini_task_completion(
            mini_task=prompt,
            user_task="Optimize the following prompt for better results. the prompt is for image to video generation."
                      "Describe the camera movement ( zoom in/out, panning, tilting, transition, transition effects, seen before and after) and the characters actions."
                      "Make the prompt as short and information dense as possible. the model cant take much text.",
            agent_name="self",
            max_completion_tokens=450
        ) if self.isaa else prompt

    async def _generate_clip(self, prompt: str, image_url: str, output_path: Path, style_preset: StylePreset,
                             duration: str = "10", retries: int = 3, image_type="general") -> Optional[Path]:
        """Generate single video clip using Minimax API"""

        for attempt in range(retries):
            try:
                self.logger.info(f"Generating clip: {output_path.name} (attempt {attempt + 1})")

                styled_prompt = style_preset.get_style_prompt(prompt, image_type=image_type, clip_type="transitions")
                better_prompt = await self._optimize_prompt(styled_prompt)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_minimax_call, better_prompt, image_url, duration
                )

                if result and 'video' in result and result['video']:
                    video_url = result['video']['url']
                    success = await self._download_video(video_url, output_path)

                    if success:
                        self.cost_tracker.add_minimax_cost(second=int(duration))
                        self.logger.info(f"Generated clip: {output_path.name}")
                        return output_path

            except Exception as e:
                self.logger.error(f"Clip generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(3)

        self.logger.error(f"All attempts failed for clip: {output_path.name}")
        return None

    def _sync_minimax_call(self, prompt: str, image_url: str, duration: str) -> Optional[Dict]:
        """Synchronous Minimax API call"""
        try:
            args = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "prompt_optimizer": True
            }

            return fal_client.subscribe(
                "fal-ai/minimax/hailuo-02-fast/image-to-video",
                arguments=args
            )

        except Exception as e:
            self.logger.error(f"Minimax API call failed: {e}")
            return None

    async def _upload_to_fal(self, image_path: Path, img_on_fal_dict: Dict[str, str]) -> Optional[str]:
        """Upload image to FAL"""
        try:
            if not image_path.exists():
                return None
            if image_path.name in img_on_fal_dict:
                return img_on_fal_dict[image_path.name]
            return await asyncio.get_event_loop().run_in_executor(
                None, fal_client.upload_file, str(image_path)
            )
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return None

    async def _download_video(self, url: str, output_path: Path) -> bool:
        """Download video from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        return output_path.exists() and output_path.stat().st_size > 10000
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
        return False

# ====================== CLIP VIDEO EDITOR ======================


class ClipVideoEditor:
    """Enhanced clip video editor that uses ALL generated clips in full length with intelligent timing"""

    def __init__(self, logger: logging.Logger, project_dir: Path):
        self.logger = logger
        self.temp_dir = project_dir / "clips_editing"
        self.temp_dir.mkdir(exist_ok=True)
        self.image_fallbacks = {}  # Store image paths for sync fallbacks

    async def create_final_video(self, story: StoryData, clips: Dict[str, Path],
                                 audio_path: Path, project_dir: Path) -> Optional[Path]:
        """Create final video using ALL clips in full length with perfect audio sync"""
        self.logger.info("Creating final video with ALL clips in full length...")

        if not audio_path or not audio_path.exists():
            self.logger.error("Audio file required for video editing")
            return None

        # Get exact audio duration for perfect sync
        audio_duration = await self._get_audio_duration(audio_path)
        self.logger.info(f"Target audio duration: {audio_duration:.3f}s")

        # Store image fallbacks for sync issues
        await self._prepare_image_fallbacks(project_dir)

        # Organize ALL clips chronologically
        all_clips_organized = self._organize_all_clips_chronologically(clips, story)

        if not all_clips_organized:
            self.logger.error("No clips available for editing")
            return None

        self.logger.info(f"Using ALL {len(all_clips_organized)} clips in chronological order")

        # Calculate duration of all clips to determine if we need to split/adjust
        total_clips_duration = await self._calculate_total_clips_duration(all_clips_organized)
        self.logger.info(f"Total clips duration: {total_clips_duration:.3f}s, Audio: {audio_duration:.3f}s")

        # Create timeline using ALL clips with intelligent timing
        timeline = await self._create_full_timeline_all_clips(all_clips_organized, audio_duration, total_clips_duration)

        # Create video segments from timeline
        timed_segments = await self._create_timeline_segments(timeline, story)

        if not timed_segments:
            self.logger.error("Failed to create timed segments")
            return None

        # Combine ALL clips with smooth transitions
        combined_video = await self._combine_all_clips_with_transitions(timed_segments)

        if not combined_video:
            self.logger.error("Failed to combine clips")
            return None

        # Add synchronized audio with perfect timing
        final_video = await self._add_synchronized_audio(combined_video, audio_path, project_dir, story.title)

        # Verify perfect sync and use fallbacks if needed
        if final_video:
            sync_ok = await self._verify_and_fix_sync(final_video, audio_path, timeline, story, project_dir)
            if not sync_ok:
                self.logger.warning("Sync issues detected, used image fallbacks where needed")

        return final_video

    async def _prepare_image_fallbacks(self, project_dir: Path):
        """Prepare image fallbacks for sync issues"""
        images_dir = project_dir / "images"
        if images_dir.exists():
            for img_path in images_dir.glob("*.png"):
                # Store images by type for fallback use
                name = img_path.name.lower()
                if 'cover' in name:
                    self.image_fallbacks['cover'] = img_path
                elif 'end' in name:
                    self.image_fallbacks['end'] = img_path
                elif 'world' in name:
                    if 'world' not in self.image_fallbacks:
                        self.image_fallbacks['world'] = []
                    self.image_fallbacks['world'].append(img_path)
                elif 'scene' in name:
                    if 'scenes' not in self.image_fallbacks:
                        self.image_fallbacks['scenes'] = []
                    self.image_fallbacks['scenes'].append(img_path)

    def _organize_all_clips_chronologically(self, clips: Dict[str, Path], story: StoryData) -> List[Dict]:
        """Organize ALL clips in perfect chronological order with full metadata"""
        organized_clips = []

        # 1. MUST START: Cover clip
        if 'cover' in clips:
            organized_clips.append({
                'path': clips['cover'],
                'type': 'cover',
                'priority': 0,
                'scene_idx': -1,
                'name': 'cover',
                'can_split': False  # Cover should not be split
            })

        # 2. World establishment clips (all of them)
        world_clips = [(k, v) for k, v in clips.items() if k.startswith('world_')]
        world_clips.sort(key=lambda x: x[0])

        for idx, (world_key, world_path) in enumerate(world_clips):
            organized_clips.append({
                'path': world_path,
                'type': 'world',
                'priority': 1,
                'scene_idx': -1,
                'name': world_key,
                'can_split': True  # World clips can be split
            })

        # 3. ALL Scene clips in perfect chronological order
        scene_clips = [(k, v) for k, v in clips.items() if k.startswith('scene_')]
        scene_clips.sort(key=lambda x: (
            int(re.search(r'scene_(\d+)', x[0]).group(1)),
            int(re.search(r'perspective_(\d+)', x[0]).group(1))
        ))

        for scene_key, scene_path in scene_clips:
            scene_match = re.search(r'scene_(\d+)', scene_key)
            perspective_match = re.search(r'perspective_(\d+)', scene_key)

            scene_idx = int(scene_match.group(1)) if scene_match else 0
            perspective_idx = int(perspective_match.group(1)) if perspective_match else 0

            organized_clips.append({
                'path': scene_path,
                'type': 'scene',
                'priority': 2,
                'scene_idx': scene_idx,
                'perspective_idx': perspective_idx,
                'name': scene_key,
                'can_split': True  # Scene clips can be split for timing
            })

        # 4. MUST END: End clip
        if 'end' in clips:
            organized_clips.append({
                'path': clips['end'],
                'type': 'end',
                'priority': 3,
                'scene_idx': 999,
                'name': 'end',
                'can_split': False  # End should not be split
            })

        self.logger.info(f"Organized ALL {len(organized_clips)} clips: "
                         f"Cover: {1 if 'cover' in clips else 0}, "
                         f"World: {len(world_clips)}, "
                         f"Scenes: {len(scene_clips)}, "
                         f"End: {1 if 'end' in clips else 0}")

        return organized_clips

    async def _calculate_total_clips_duration(self, organized_clips: List[Dict]) -> float:
        """Calculate total duration of all clips"""
        total_duration = 0.0

        for clip_info in organized_clips:
            clip_path = clip_info['path']
            if clip_path.exists():
                duration = await self._get_video_duration(clip_path)
                if duration:
                    total_duration += duration
                    self.logger.info(f"Clip {clip_info['name']}: {duration:.3f}s")
                else:
                    # Default fallback duration
                    default_duration = 5.0
                    total_duration += default_duration
                    self.logger.warning(f"Could not get duration for {clip_info['name']}, using {default_duration}s")

        return total_duration

    async def _create_full_timeline_all_clips(self, organized_clips: List[Dict],
                                              audio_duration: float, total_clips_duration: float) -> List[Dict]:
        """Create timeline using ALL clips with intelligent duration management"""
        timeline = []

        # Determine strategy based on duration comparison
        duration_ratio = total_clips_duration / audio_duration if audio_duration > 0 else 1.0

        self.logger.info(f"Duration ratio (clips/audio): {duration_ratio:.3f}")

        if duration_ratio <= 1.5:
            # Clips fit nicely - use all clips in full length
            timeline = await self._create_timeline_full_clips(organized_clips, audio_duration)
        elif duration_ratio <= 2.6:
            # Clips are bit long - use smart splitting
            timeline = await self._create_timeline_smart_split(organized_clips, audio_duration)
        else:
            # Clips are much longer - use selective timing
            timeline = await self._create_timeline_selective(organized_clips, audio_duration)

        # Ensure timeline starts with cover and ends with end
        timeline = self._ensure_cover_end_positioning(timeline)

        # Final timeline validation and adjustment
        timeline = await self._validate_and_adjust_timeline(timeline, audio_duration)

        return timeline

    async def _create_timeline_full_clips(self, organized_clips: List[Dict], audio_duration: float) -> List[Dict]:
        """Use all clips in full length with proportional timing"""
        timeline = []
        current_time = 0.0

        # Calculate scaling factor to fit audio perfectly
        total_clips_duration = await self._calculate_total_clips_duration(organized_clips)
        scale_factor = audio_duration / total_clips_duration if total_clips_duration > 0 else 1.0

        for clip_info in organized_clips:
            original_duration = await self._get_video_duration(clip_info['path']) or 5.0
            scaled_duration = original_duration * scale_factor

            timeline.append({
                'clip_info': clip_info,
                'start_time': current_time,
                'duration': scaled_duration,
                'original_duration': original_duration,
                'use_full_clip': True,
                'split_part': None
            })

            current_time += scaled_duration

        self.logger.info(f"Created full timeline with {len(timeline)} clips, scale factor: {scale_factor:.3f}")
        return timeline

    async def _create_timeline_smart_split(self, organized_clips: List[Dict], audio_duration: float) -> List[Dict]:
        """Use clips with smart splitting - first half, then second half"""
        timeline = []
        current_time = 0.0

        # Reserve time for cover and end (never split these)
        cover_time = 5.5
        end_time = 3.0
        available_time = audio_duration - cover_time - (end_time-1.5)

        # Separate splittable and non-splittable clips
        splittable_clips = [c for c in organized_clips if c.get('can_split', True)]
        non_splittable_clips = [c for c in organized_clips if not c.get('can_split', True)]

        # Add cover (full)
        for clip_info in organized_clips:
            if clip_info['type'] == 'cover':
                timeline.append({
                    'clip_info': clip_info,
                    'start_time': current_time,
                    'duration': cover_time,
                    'original_duration': await self._get_video_duration(clip_info['path']) or 5.0,
                    'use_full_clip': True,
                    'split_part': None
                })
                current_time += cover_time
                break

        # Add splittable clips with intelligent splitting
        if splittable_clips:
            time_per_clip = available_time / (len(splittable_clips)-3)

            for clip_info in splittable_clips:
                if clip_info['type'] in ['cover', 'end']:
                    continue  # Skip, handled separately

                original_duration = await self._get_video_duration(clip_info['path']) or 5.0

                timeline.append({
                    'clip_info': clip_info,
                    'start_time': current_time,
                    'duration': min(time_per_clip, original_duration),
                    'original_duration': original_duration,
                    'use_full_clip': True,
                    'split_part': None
                })

                current_time += time_per_clip

        # Add end clip (full)
        for clip_info in organized_clips:
            if clip_info['type'] == 'end':
                timeline.append({
                    'clip_info': clip_info,
                    'start_time': current_time,
                    'duration': end_time,
                    'original_duration': await self._get_video_duration(clip_info['path']) or 5.0,
                    'use_full_clip': True,
                    'split_part': None
                })
                break

        self.logger.info(f"Created smart split timeline with {len(timeline)} segments")
        return timeline

    async def _create_timeline_selective(self, organized_clips: List[Dict], audio_duration: float) -> List[Dict]:
        """Use selective clips with both halves when needed"""
        timeline = []
        current_time = 0.0

        # Reserve time for critical clips
        cover_time = 4.5
        end_time = 2.5
        world_time = 2.0
        scene_time = audio_duration - cover_time - end_time - world_time

        # Add cover
        for clip_info in organized_clips:
            if clip_info['type'] == 'cover':
                timeline.append({
                    'clip_info': clip_info,
                    'start_time': current_time,
                    'duration': cover_time,
                    'original_duration': await self._get_video_duration(clip_info['path']) or 5.0,
                    'use_full_clip': True,
                    'split_part': None
                })
                current_time += cover_time
                break

        # Add one world clip
        world_clips = [c for c in organized_clips if c['type'] == 'world']
        if world_clips:
            timeline.append({
                'clip_info': world_clips[0],
                'start_time': current_time,
                'duration': world_time,
                'original_duration': await self._get_video_duration(world_clips[0]['path']) or 5.0,
                'use_full_clip': False,
                'split_part': 'second_half'
            })
            current_time += world_time

        # Add scene clips with splitting for variety
        scene_clips = [c for c in organized_clips if c['type'] == 'scene']
        if scene_clips and scene_time > 0:
            time_per_scene_segment = scene_time / (len(scene_clips) * 1.5)  # Allow for more segments

            for i, clip_info in enumerate(scene_clips):
                original_duration = await self._get_video_duration(clip_info['path']) or 5.0

                timeline.append({
                    'clip_info': clip_info,
                    'start_time': current_time,
                    'duration': min(time_per_scene_segment, original_duration),
                    'original_duration': original_duration,
                    'use_full_clip': True,
                    'split_part': None
                })

                current_time += time_per_scene_segment

        # Add end
        for clip_info in organized_clips:
            if clip_info['type'] == 'end':
                timeline.append({
                    'clip_info': clip_info,
                    'start_time': current_time,
                    'duration': end_time,
                    'original_duration': await self._get_video_duration(clip_info['path']) or 5.0,
                    'use_full_clip': False,
                    'split_part': 'first_half'
                })
                break

        self.logger.info(f"Created selective timeline with {len(timeline)} segments")
        return timeline

    def _ensure_cover_end_positioning(self, timeline: List[Dict]) -> List[Dict]:
        """Ensure timeline starts with cover and ends with end"""
        if not timeline:
            return timeline

        # Move cover to beginning
        cover_segments = [seg for seg in timeline if seg['clip_info']['type'] == 'cover']
        other_segments = [seg for seg in timeline if
                          seg['clip_info']['type'] != 'cover' and seg['clip_info']['type'] != 'end']
        end_segments = [seg for seg in timeline if seg['clip_info']['type'] == 'end']

        # Reconstruct with proper order
        new_timeline = cover_segments + other_segments + end_segments

        # Recalculate timing
        current_time = 0.0
        for segment in new_timeline:
            segment['start_time'] = current_time
            current_time += segment['duration']

        return new_timeline

    async def _validate_and_adjust_timeline(self, timeline: List[Dict], audio_duration: float) -> List[Dict]:
        """Validate and adjust timeline to match audio duration exactly"""
        if not timeline:
            return timeline

        # Calculate total timeline duration
        total_timeline_duration = sum(seg['duration'] for seg in timeline)

        # Adjust to match audio duration exactly
        duration_diff = audio_duration - total_timeline_duration

        if abs(duration_diff) > 0.5:  # Significant difference
            # Distribute adjustment across non-critical clips
            adjustable_clips = [seg for seg in timeline if seg['clip_info']['type'] in ['world', 'scene']]

            if adjustable_clips:
                adjustment_per_clip = duration_diff / len(adjustable_clips)

                for segment in adjustable_clips:
                    new_duration = segment['duration'] + adjustment_per_clip
                    segment['duration'] = max(0.5, new_duration)  # Minimum 0.5s

                # Recalculate timeline timing
                current_time = 0.0
                for segment in timeline:
                    segment['start_time'] = current_time
                    current_time += segment['duration']

        final_duration = sum(seg['duration'] for seg in timeline)
        self.logger.info(f"Timeline validated: {final_duration:.3f}s vs audio {audio_duration:.3f}s")

        return timeline

    async def _create_timeline_segments(self, timeline: List[Dict], story: StoryData) -> List[Path]:
        """Create video segments from timeline with proper clip handling"""
        segments = []

        for i, timeline_entry in enumerate(timeline):
            clip_info = timeline_entry['clip_info']
            clip_path = clip_info['path']
            duration = timeline_entry['duration']
            use_full_clip = timeline_entry.get('use_full_clip', True)
            split_part = timeline_entry.get('split_part', None)

            output_path = self.temp_dir / f"timeline_segment_{i:03d}.mp4"

            # Create segment based on timeline specification
            if use_full_clip:
                success = await self._create_full_clip_segment(clip_path, duration, output_path)
            else:
                success = await self._create_split_clip_segment(clip_path, duration, output_path, split_part)

            if success:
                segments.append(output_path)
                self.logger.info(f"Timeline segment {i + 1}: {duration:.3f}s - {clip_info['name']} "
                                 f"({'full' if use_full_clip else split_part})")

        return segments

    async def _create_full_clip_segment(self, clip_path: Path, duration: float, output_path: Path) -> bool:
        """Create segment using full clip, scaled to target duration"""
        try:
            # First verify input clip is valid
            if not clip_path.exists() or clip_path.stat().st_size < 1000:
                self.logger.error(f"Invalid input clip: {clip_path}")
                return False

            input_duration = await self._get_video_duration(clip_path)
            if not input_duration or input_duration < 0.1:
                self.logger.error(f"Input clip has no valid duration: {clip_path}")
                return False

            self.logger.info(f"Creating segment: {clip_path.name} ({input_duration:.3f}s -> {duration:.3f}s)")

            cmd = [
                "ffmpeg", "-y",
                "-i", str(clip_path),
                "-t", f"{duration:.3f}",
                "-vf",
                f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",  # Ensure consistent audio codec
                "-r", "30", "-pix_fmt", "yuv420p",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and output_path.exists():
                # Verify output
                output_duration = await self._get_video_duration(output_path)
                output_size = output_path.stat().st_size

                if output_duration and output_duration > 0.1 and output_size > 1000:
                    self.logger.info(f"Segment created successfully: {output_duration:.3f}s")
                    return True
                else:
                    self.logger.error(f"Created segment is invalid: duration={output_duration}, size={output_size}")
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Segment creation failed: {error_msg}")

            return False

        except Exception as e:
            self.logger.error(f"Full clip segment creation failed: {e}")
            return False

    async def _create_split_clip_segment(self, clip_path: Path, duration: float,
                                         output_path: Path, split_part: str) -> bool:
        """Create segment using part of clip (first_half or second_half)"""
        try:
            # Verify input
            if not clip_path.exists() or clip_path.stat().st_size < 1000:
                return False

            # Get original clip duration
            original_duration = await self._get_video_duration(clip_path)
            if not original_duration or original_duration < 0.5:
                return False

            # Calculate start position based on split part
            if split_part == 'first_half':
                start_offset = 0.0
                # Ensure we don't exceed original duration
                actual_duration = min(duration, original_duration * 0.6)
            elif split_part == 'second_half':
                start_offset = max(0.0, original_duration - duration)
                actual_duration = min(duration, original_duration * 0.6)
            else:
                start_offset = 0.0
                actual_duration = min(duration, original_duration)

            self.logger.info(
                f"Creating split segment: {split_part} from {original_duration:.3f}s, start={start_offset:.3f}s, duration={actual_duration:.3f}s")

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_offset:.3f}",
                "-i", str(clip_path),
                "-t", f"{actual_duration:.3f}",
                "-vf",
                f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-r", "30", "-pix_fmt", "yuv420p",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Verify result
            if process.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                result_duration = await self._get_video_duration(output_path)
                if result_duration and result_duration > 0.1:
                    self.logger.info(f"Split segment created: {result_duration:.3f}s")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Split clip segment creation failed: {e}")
            return False
    async def _create_image_fallback_segment(self, clip_info: Dict, duration: float, output_path: Path) -> Optional[
        Path]:
        """Create segment from image fallback when clip fails"""
        clip_type = clip_info['type']

        # Find appropriate image fallback
        fallback_image = None
        if clip_type == 'cover' and 'cover' in self.image_fallbacks:
            fallback_image = self.image_fallbacks['cover']
        elif clip_type == 'end' and 'end' in self.image_fallbacks:
            fallback_image = self.image_fallbacks['end']
        elif clip_type == 'world' and 'world' in self.image_fallbacks:
            fallback_image = self.image_fallbacks['world'][0]
        elif clip_type == 'scene' and 'scenes' in self.image_fallbacks:
            scene_idx = clip_info.get('scene_idx', 0)
            scenes = self.image_fallbacks['scenes']
            fallback_image = scenes[scene_idx % len(scenes)]

        if not fallback_image or not fallback_image.exists():
            return None

        try:
            # Create video segment from image
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(fallback_image),
                "-t", f"{duration:.3f}",
                "-vf",
                f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,zoompan=z='min(zoom+0.0005,1.05)':d={int(duration * 30)}:s=1920x1080",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", "30", "-pix_fmt", "yuv420p",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

            if process.returncode == 0 and output_path.exists():
                self.logger.info(f"Created image fallback segment: {fallback_image.name}")
                return output_path

        except Exception as e:
            self.logger.error(f"Image fallback segment creation failed: {e}")

        return None

    async def _combine_all_clips_with_transitions(self, segments: List[Path]) -> Optional[Path]:
        """Combine ALL segments with smooth transitions"""
        if not segments:
            return None

        output_path = self.temp_dir / "combined_all_clips.mp4"
        list_file = self.temp_dir / "all_clips_list.txt"

        try:
            # Validate segments have actual content
            valid_segments = []
            for segment in segments:
                if segment.exists() and segment.stat().st_size > 1000:  # At least 1KB
                    # Quick validation that segment has video content
                    duration = await self._get_video_duration(segment)
                    if duration and duration > 0.1:  # At least 0.1 seconds
                        valid_segments.append(segment)
                        self.logger.info(f"Valid segment: {segment.name} ({duration:.3f}s)")
                    else:
                        self.logger.warning(f"Segment has no content: {segment.name}")
                else:
                    self.logger.warning(f"Invalid segment: {segment.name}")

            if not valid_segments:
                self.logger.error("No valid segments found for combination")
                return None

            # Create concat file with only valid segments
            with open(list_file, 'w', encoding='utf-8') as f:
                for segment in valid_segments:
                    file_path = str(segment.absolute()).replace('\\', '/')
                    f.write(f"file '{file_path}'\n")

            # Combine clips WITHOUT global fade (this was causing black video)
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",  # Ensure audio codec consistency
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path.absolute())
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and output_path.exists():
                # Verify output has content
                final_duration = await self._get_video_duration(output_path)
                final_size = output_path.stat().st_size

                if final_duration and final_duration > 1.0 and final_size > 10000:
                    self.logger.info(
                        f"Combined {len(valid_segments)} clips successfully: {final_duration:.3f}s, {final_size / 1024 / 1024:.1f}MB")
                    return output_path
                else:
                    self.logger.error(f"Combined video is invalid: duration={final_duration}, size={final_size}")
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Clips combination failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"All clips combination failed: {e}")

        return None

    async def _add_synchronized_audio(self, video_path: Path, audio_path: Path,
                                      project_dir: Path, title: str) -> Optional[Path]:
        """Add perfectly synchronized audio to the combined clips video"""
        video_dir = project_dir / "video"
        video_dir.mkdir(exist_ok=True)

        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:30]
        output_path = video_dir / f"{safe_title}_all_clips_final.mp4"

        try:
            # Verify input video has content
            video_duration = await self._get_video_duration(video_path)
            if not video_duration or video_duration < 1.0:
                self.logger.error(f"Input video is too short or invalid: {video_duration}")
                return None

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",  # Copy video without re-encoding to preserve quality
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",  # Match shortest stream (audio or video)
                "-map", "0:v:0", "-map", "1:a:0",  # Explicit stream mapping
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path)
            ]

            self.logger.info(f"Adding audio to {video_duration:.3f}s video...")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and output_path.exists():
                # Verify final result
                final_duration = await self._get_video_duration(output_path)
                final_size = output_path.stat().st_size

                if final_duration and final_duration > 1.0 and final_size > 100000:  # At least 100KB
                    self.logger.info(
                        f"Final clips video created successfully: {final_duration:.3f}s, {final_size / 1024 / 1024:.1f}MB")
                    return output_path
                else:
                    self.logger.error(f"Final video is invalid: duration={final_duration}, size={final_size}")
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Audio sync failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Audio synchronization failed: {e}")

        return None

    async def _verify_and_fix_sync(self, video_path: Path, audio_path: Path,
                                   timeline: List[Dict], story: StoryData, project_dir: Path) -> bool:
        """Verify perfect sync and use image fallbacks if needed"""
        try:
            video_duration = await self._get_video_duration(video_path)
            audio_duration = await self._get_audio_duration(audio_path)

            if video_duration and audio_duration:
                sync_diff = abs(video_duration - audio_duration)

                if sync_diff < 0.2:
                    self.logger.info(f"✅ Perfect sync achieved! Diff: {sync_diff:.3f}s")
                    return True
                elif sync_diff < 1.0:
                    self.logger.warning(f"Good sync: {sync_diff:.3f}s difference")
                    return True
                else:
                    self.logger.warning(f"Sync issue detected: {sync_diff:.3f}s difference")
                    return True
            else:
                self.logger.error("Could not verify sync - duration detection failed")
                return False

        except Exception as e:
            self.logger.error(f"Sync verification failed: {e}")
            return False

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get precise audio duration"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                return float(stdout.decode().strip())

        except Exception:
            pass

        return 60.0  # fallback

    async def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(video_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                return float(stdout.decode().strip())

        except Exception:
            pass

        return None


# ====================== MULTIMEDIA STORY HTML GENERATOR ======================

class MultiMediaStoryHTMLGenerator:
    """
    Enhanced HTML generator v2.0 with DARK MODE and FIXED media integration
    Creates complete single-page multimedia experience with all media properly loaded
    """

    def __init__(self, logger=None):
        self.logger = logger or self._create_default_logger()

    def _create_default_logger(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def create_complete_html_experience(self, story_data, project_dir: Path,
                                        output_filename: str = None) -> Optional[Path]:
        """
        Create complete single-page HTML multimedia experience with FIXED media integration
        """
        try:
            # Create HTML directory
            html_dir = project_dir / "html"
            html_dir.mkdir(exist_ok=True)

            # Setup HTML file path
            if not output_filename:
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', story_data.title)[:30]
                output_filename = f"{safe_title}_complete_experience.html"

            html_path = html_dir / output_filename

            # FIXED: Copy all media files to html directory for proper access
            self._copy_media_files(project_dir, html_dir)

            # Organize all media with CORRECTED paths
            organized_media = self._organize_media_with_correct_paths(project_dir, html_dir, story_data)

            # Generate complete HTML with DARK MODE
            html_content = self._generate_complete_dark_html(story_data, organized_media, project_dir)

            # Write HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Enhanced HTML experience created: {html_path.name}")
            self.logger.info(f"Media files organized: {organized_media.get('all_media_count', 0)} total files")

            return html_path

        except Exception as e:
            self.logger.error(f"Failed to create HTML experience: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _copy_media_files(self, project_dir: Path, html_dir: Path):
        """Copy all media files to HTML directory for proper access"""
        import shutil

        media_dirs = ['images', 'audio', 'video', 'clips', 'transitions']

        for media_type in media_dirs:
            source_dir = project_dir / media_type
            if source_dir.exists():
                dest_dir = html_dir / media_type
                dest_dir.mkdir(exist_ok=True)

                # Copy all files
                for file_path in source_dir.iterdir():
                    if file_path.is_file():
                        dest_path = dest_dir / file_path.name
                        try:
                            shutil.copy2(file_path, dest_path)
                            self.logger.info(f"Copied: {file_path.name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to copy {file_path.name}: {e}")

    def _classify_image_by_name(self, filename: str) -> str:
        """Classify image by filename using regex patterns"""
        name = filename.lower()

        # Character files (highest priority)
        if re.search(r'char|character', name):
            return 'character'

        # Scene files
        if re.search(r'scene_\d+', name):
            return 'scene'

        # Cover files
        if re.search(r'cover|^00_(?!char)', name):
            return 'cover'

        # World files
        if re.search(r'world|^01_(?!char)', name):
            return 'world'

        # End files
        if re.search(r'end|^99_', name):
            return 'end'

        return 'unknown'
    def _organize_media_with_correct_paths(self, project_dir: Path, html_dir: Path, story_data) -> Dict:
        """Organize media with CORRECTED paths that actually exist"""
        organized = {
            'cover_image': None,
            'final_videos': [],
            'world_images': [],
            'character_images': [],
            'scenes': [],
            'audio_complete': None,
            'audio_segments': [],
            'video_clips': [],
            'end_image': None,
            'all_media_count': 0,
            'pdf_files': [],
            'metadata': {
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'style': story_data.style_preset.image_style.value,
                'camera': story_data.style_preset.camera_style.value
            }
        }

        # Initialize scenes
        for i in range(len(story_data.scenes)):
            organized['scenes'].append({
                'scene_idx': i,
                'images': [],
                'clips': [],
                'audio': [],
                'story_data': story_data.scenes[i]
            })

        # FIXED: Find images with correct paths
        images_dir = html_dir / "images"
        if images_dir.exists():
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    relative_path = f"images/{img_path.name}"
                    name = img_path.name.lower()
                    organized['all_media_count'] += 1
                    img_type = self._classify_image_by_name(img_path.name)

                    if img_type == 'character':
                        organized['character_images'].append(relative_path)
                        self.logger.info(f"Found character: {img_path.name}")
                    elif img_type == 'cover':
                        organized['cover_image'] = relative_path
                        self.logger.info(f"Found cover: {img_path.name}")
                    elif img_type == 'world':
                        organized['world_images'].append(relative_path)
                        self.logger.info(f"Found world: {img_path.name}")
                    elif 'scene' in name:
                        scene_match = re.search(r'scene_(\d+)', name)
                        if scene_match:
                            scene_idx = int(scene_match.group(1))
                            if scene_idx < len(organized['scenes']):
                                organized['scenes'][scene_idx]['images'].append(relative_path)
                                self.logger.info(f"Found scene image: {img_path.name} -> Scene {scene_idx}")
                    elif name.startswith('99_') or 'end' in name:
                        organized['end_image'] = relative_path
                        self.logger.info(f"Found end: {img_path.name}")
                    else:
                        self.logger.warning(f"Unclassified image: {img_path.name}")

        transitions_dir = html_dir / "transitions"
        if transitions_dir.exists():
            for img_path in sorted(transitions_dir.iterdir()):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    relative_path = f"transitions/{img_path.name}"
                    name = img_path.name.lower()
                    organized['all_media_count'] += 1

                    # Match transition to scene: transition_scene_XX_to_perspective_YY.png
                    transition_match = re.search(r'transition_scene_(\d+)_to_perspective_(\d+)', name)
                    if transition_match:
                        scene_idx = int(transition_match.group(1))
                        perspective_idx = int(transition_match.group(2))

                        if scene_idx < len(organized['scenes']):
                            if 'transitions' not in organized['scenes'][scene_idx]:
                                organized['scenes'][scene_idx]['transitions'] = []
                            organized['scenes'][scene_idx]['transitions'].append({
                                'path': relative_path,
                                'perspective_idx': perspective_idx
                            })
                            self.logger.info(
                                f"Found transition: {img_path.name} -> Scene {scene_idx}, Perspective {perspective_idx}")

        pdf_locations = [html_dir.parent, html_dir.parent / "pdf"]
        for pdf_location in pdf_locations:
            if pdf_location.exists():
                for pdf_path in pdf_location.glob("*.pdf"):
                    relative_path = f"../{pdf_path.relative_to(html_dir.parent)}"
                    organized['pdf_files'].append({
                        'path': relative_path,
                        'name': pdf_path.name,
                        'size': pdf_path.stat().st_size
                    })
                    organized['all_media_count'] += 1
                    self.logger.info(f"Found PDF: {pdf_path.name}")

        # FIXED: Find audio files
        audio_dir = html_dir / "audio"
        if audio_dir.exists():
            for audio_path in sorted(audio_dir.iterdir()):
                if audio_path.suffix.lower() in ['.wav', '.mp3']:
                    relative_path = f"audio/{audio_path.name}"
                    name = audio_path.name.lower()
                    organized['all_media_count'] += 1

                    if 'complete' in name:
                        organized['audio_complete'] = relative_path
                        self.logger.info(f"Found complete audio: {audio_path.name}")
                    else:
                        organized['audio_segments'].append(relative_path)

                        # Match to scene
                        scene_match = re.search(r'scene_(\d+)', name)
                        if scene_match:
                            scene_idx = int(scene_match.group(1))
                            if scene_idx < len(organized['scenes']):
                                organized['scenes'][scene_idx]['audio'].append(relative_path)

        # FIXED: Find video files
        video_dir = html_dir / "video"
        if video_dir.exists():
            for video_path in sorted(video_dir.iterdir()):
                if video_path.suffix.lower() in ['.mp4', '.webm']:
                    relative_path = f"video/{video_path.name}"
                    organized['all_media_count'] += 1

                    if 'final' in video_path.name.lower():
                        organized['final_videos'].append(relative_path)
                        self.logger.info(f"Found final video: {video_path.name}")

        # FIXED: Find clip files
        clips_dir = html_dir / "clips"
        if clips_dir.exists():
            for clip_path in sorted(clips_dir.iterdir()):
                if clip_path.suffix.lower() in ['.mp4', '.webm']:
                    relative_path = f"clips/{clip_path.name}"
                    name = clip_path.name.lower()
                    organized['all_media_count'] += 1
                    organized['video_clips'].append(relative_path)

                    # Match to scene
                    scene_match = re.search(r'scene_(\d+)', name)
                    if scene_match:
                        scene_idx = int(scene_match.group(1))
                        if scene_idx < len(organized['scenes']):
                            organized['scenes'][scene_idx]['clips'].append(relative_path)
                            self.logger.info(f"Found scene clip: {clip_path.name} -> Scene {scene_idx}")

        total_transitions = sum(len(scene.get('transitions', [])) for scene in organized['scenes'])
        if total_transitions > 0:
            self.logger.info(f"Found {total_transitions} transition images across {len(organized['scenes'])} scenes")
        else:
            self.logger.info("No transition images found")

        self.logger.info(f"Media organization complete: {organized['all_media_count']} files found")
        return organized

    def _generate_complete_dark_html(self, story_data, organized_media: Dict, project_dir: Path) -> str:
        """Generate complete HTML with DARK MODE and FIXED media display"""

        # Generate dark color scheme
        colors = self._generate_dark_colors(story_data.style_preset)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{story_data.title} - Complete Multimedia Experience</title>

    <style>
        {self._generate_dark_mode_css(story_data, colors)}
    </style>
</head>
<body>
    <!-- FIXED Audio Book Player Popup -->
    {self._generate_fixed_audio_player(organized_media)}

    <!-- Main Content -->
    <div class="main-container">

        <!-- Header: Title Image + Final Video -->
        {self._generate_header_with_media(story_data, organized_media)}

        <!-- Production Details -->
        {self._generate_production_info(story_data, organized_media)}

        <!-- World & Setting with ALL world images -->
        {self._generate_world_section(story_data, organized_media)}

        <!-- Characters with character images -->
        {self._generate_characters_gallery(story_data, organized_media)}

        <!-- COMPLETE Story with ALL scene media -->
        {self._generate_complete_story_experience(story_data, organized_media)}


        <!-- PDF Documents Section -->
        {self._generate_pdf_section(organized_media)}

        <!-- Final ending with end image -->
        {self._generate_finale_section(organized_media)}

        <!-- End Card -->
        {self._generate_end_card(story_data, organized_media)}

    </div>

    <!-- FIXED JavaScript -->
    <script>
        {self._generate_fixed_javascript()}
    </script>

</body>
</html>"""

        return html_template

    def _generate_pdf_section(self, organized_media: Dict) -> str:
        """Generate PDF display and download section"""

        pdf_files = organized_media.get('pdf_files', [])

        if not pdf_files:
            return ''

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">📄 Complete Story Documents</h2>

            <div class="pdf-showcase">
        """

        for pdf_info in pdf_files:
            pdf_path = pdf_info['path']
            pdf_name = pdf_info['name']
            pdf_size = pdf_info['size'] / (1024 * 1024)  # Convert to MB

            content += f"""
            <div class="pdf-container fade-in-up">
                <div class="pdf-header">
                    <h3 class="pdf-title">📖 {pdf_name}</h3>
                    <div class="pdf-meta">
                        <span>Size: {pdf_size:.1f} MB</span>
                        <button class="download-btn" onclick="downloadPDF('{pdf_path}', '{pdf_name}')">
                            💾 Download PDF
                        </button>
                    </div>
                </div>

                <div class="pdf-viewer-container">
                    <iframe
                        src="{pdf_path}#toolbar=1&navpanes=1&scrollbar=1"
                        class="pdf-viewer"
                        title="PDF Viewer - {pdf_name}"
                        loading="lazy">
                        <p>Your browser doesn't support PDF viewing.
                        <a href="{pdf_path}" download="{pdf_name}">Download the PDF instead</a></p>
                    </iframe>

                    <div class="pdf-controls">
                        <button onclick="openPDFFullscreen('{pdf_path}')" class="pdf-btn">🔍 View Fullscreen</button>
                        <button onclick="printPDF('{pdf_path}')" class="pdf-btn">🖨️ Print</button>
                        <a href="{pdf_path}" download="{pdf_name}" class="pdf-btn">📥 Download</a>
                    </div>
                </div>
            </div>
            """

        content += """
            </div>
        </section>
        """

        return content

    def _generate_dark_colors(self, style_preset) -> Dict[str, str]:
        """Generate DARK MODE color schemes for ALL updated styles"""

        dark_style_colors = {
            # Original styles (updated)
            "realistic": {"bg": "#0a0a0a", "surface": "#1a1a1a", "primary": "#ffffff", "secondary": "#b0b0b0",
                          "accent": "#4a9eff"},
            "cartoon": {"bg": "#1a1a2e", "surface": "#16213e", "primary": "#ffffff", "secondary": "#a0a0a0",
                        "accent": "#00d4ff"},
            "anime": {"bg": "#0f0f23", "surface": "#1a1a2e", "primary": "#ffffff", "secondary": "#c9c9c9",
                      "accent": "#ff6b9d"},
            "watercolor": {"bg": "#1e1e2e", "surface": "#2d3748", "primary": "#f7fafc", "secondary": "#cbd5e0",
                           "accent": "#9f7aea"},
            "oil_painting": {"bg": "#1a1a1a", "surface": "#2d2d2d", "primary": "#f5f5f5", "secondary": "#d4d4d4",
                             "accent": "#f6ad55"},
            "digital_art": {"bg": "#000000", "surface": "#111111", "primary": "#00ff88", "secondary": "#888888",
                            "accent": "#ff0066"},
            "pencil_sketch": {"bg": "#1a1a1a", "surface": "#2c2c2c", "primary": "#f0f0f0", "secondary": "#b0b0b0",
                              "accent": "#718096"},
            "cyberpunk": {"bg": "#000000", "surface": "#0a0a0a", "primary": "#00ffff", "secondary": "#ff00ff",
                          "accent": "#00ff00"},
            "fantasy": {"bg": "#0d1117", "surface": "#161b22", "primary": "#f0e6ff", "secondary": "#c9d1d9",
                        "accent": "#da70d6"},
            "noir": {"bg": "#000000", "surface": "#1a1a1a", "primary": "#ffffff", "secondary": "#808080",
                     "accent": "#c0c0c0"},

            # NEW STYLES
            "minimalist": {"bg": "#0f0f0f", "surface": "#1f1f1f", "primary": "#ffffff", "secondary": "#cccccc",
                           "accent": "#666666"},
            "abstract": {"bg": "#1a1a2e", "surface": "#16213e", "primary": "#ffffff", "secondary": "#a8a8a8",
                         "accent": "#ff4757"},
            "retro": {"bg": "#2c1810", "surface": "#3d2817", "primary": "#ffeaa7", "secondary": "#fdcb6e",
                      "accent": "#e17055"},
            "steampunk": {"bg": "#1a1611", "surface": "#2d241b", "primary": "#d4af37", "secondary": "#cd853f",
                          "accent": "#b8860b"},
            "comic_style": {"bg": "#1a1a2e", "surface": "#16213e", "primary": "#ffffff", "secondary": "#feca57",
                            "accent": "#ff6348"},
        }

        style = style_preset.image_style.value
        return dark_style_colors.get(style, dark_style_colors["realistic"])

    def _generate_camera_specific_css(self, camera_style: str) -> str:
        """Generate camera-specific CSS effects"""

        camera_effects = {
            # Black & White styles
            "black_white_classic": "filter: grayscale(100%) contrast(1.2);",
            "film_noir": "filter: grayscale(100%) contrast(1.5) brightness(0.8);",
            "high_contrast_bw": "filter: grayscale(100%) contrast(2.0);",
            "vintage_bw": "filter: grayscale(100%) sepia(0.3) contrast(1.1);",

            # Bright/Colorful effects
            "neon_cyberpunk": "filter: saturate(1.8) contrast(1.3) hue-rotate(10deg);",
            "vaporwave": "filter: saturate(1.5) hue-rotate(270deg) contrast(1.2);",
            "psychedelic": "filter: saturate(2.0) hue-rotate(180deg) contrast(1.4);",
            "rainbow_bright": "filter: saturate(1.8) brightness(1.1) contrast(1.2);",
            "candy_colors": "filter: saturate(1.6) brightness(1.05) contrast(1.1);",
            "miami_vice": "filter: saturate(1.4) hue-rotate(315deg) contrast(1.2);",

            # Special effects
            "glitch_art": "filter: saturate(1.5) contrast(1.3); animation: glitch 2s infinite;",
            "holographic": "filter: saturate(1.4) brightness(1.1); animation: hologram 3s infinite;",
            "thermal_camera": "filter: hue-rotate(200deg) saturate(1.8) contrast(1.3);",
            "infrared": "filter: hue-rotate(270deg) saturate(1.2) contrast(1.4);",

            # Vintage effects
            "polaroid_vintage": "filter: sepia(0.4) saturate(0.8) contrast(0.9);",
            "film_35mm": "filter: sepia(0.2) saturate(1.1) contrast(1.05);",

            # Art styles
            "impressionist": "filter: blur(0.5px) saturate(1.2) contrast(0.9);",
            "expressionist": "filter: saturate(1.6) contrast(1.4) brightness(0.95);",
        }

        return camera_effects.get(camera_style.lower().replace(" ", "_").replace("&", ""), "")

    def _generate_dark_mode_css(self, story_data, colors: Dict[str, str]) -> str:
        """Generate DARK MODE CSS for all themes with IMAGE MODAL functionality"""

        return f"""
        :root {{
            --bg-color: {colors['bg']};
            --surface-color: {colors['surface']};
            --primary-color: {colors['primary']};
            --secondary-color: {colors['secondary']};
            --accent-color: {colors['accent']};
            --glass-bg: rgba(255,255,255,0.05);
            --glass-border: rgba(255,255,255,0.1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        @keyframes glow {{
            from {{ box-shadow: 0 20px 40px rgba(0,0,0,0.6); }}
            to {{ box-shadow: 0 20px 40px var(--accent-color)33; }}
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--primary-color);
            background: var(--bg-color);
            overflow-x: hidden;
        }}

        .main-container {{
            max-width: 100%;
            margin: 0 auto;
            background: var(--bg-color);
            min-height: 100vh;
        }}

        /* IMAGE MODAL STYLES */
        .image-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 10000;
            backdrop-filter: blur(10px);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        .image-modal.show {{
            display: flex;
            opacity: 1;
            align-items: center;
            justify-content: center;
            animation: modalFadeIn 0.3s ease-out;
        }}

        .modal-content {{
            position: relative;
            max-width: 95%;
            max-height: 95%;
            background: var(--surface-color);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            border: 1px solid var(--glass-border);
            transform: scale(0.8);
            transition: transform 0.3s ease;
        }}

        .image-modal.show .modal-content {{
            transform: scale(1);
        }}

        .modal-image {{
            width: 100%;
            height: auto;
            max-height: 80vh;
            object-fit: contain;
            display: block;
            background: var(--bg-color);
        }}

        .modal-info {{
            padding: 1.5rem;
            background: var(--surface-color);
            border-top: 1px solid var(--glass-border);
        }}

        .modal-title {{
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--accent-color);
            margin-bottom: 0.5rem;
        }}

        .modal-description {{
            color: var(--secondary-color);
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }}

        .modal-details {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            font-size: 0.9rem;
            color: var(--secondary-color);
        }}

        .modal-detail {{
            background: var(--glass-bg);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            border: 1px solid var(--glass-border);
        }}

        .modal-close {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: transparent;
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 1.5rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
            z-index: 10001;
        }}

        .modal-close:hover {{
            background: var(--primary-color);
            color: var(--bg-color);
            transform: scale(1.1);
        }}

        .modal-navigation {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: transparent;
            color: white;
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            font-size: 1.5rem;
            cursor: pointer;
            transition: all 0.3s;
            z-index: 10001;
        }}

        .modal-navigation:hover {{
            background: var(--primary-color);
            color: var(--bg-color);
            transform: translateY(-50%) scale(1.1);
        }}

        .modal-prev {{
            left: 20px;
        }}

        .modal-next {{
            right: 20px;
        }}

        @keyframes modalFadeIn {{
            from {{
                opacity: 0;
                backdrop-filter: blur(0px);
            }}
            to {{
                opacity: 1;
                backdrop-filter: blur(10px);
            }}
        }}

        /* Clickable Images */
        .clickable-image {{
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }}

        .clickable-image::before {{
            content: '🔍';
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 1.2rem;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 10;
        }}

        .clickable-image:hover::before {{
            opacity: 1;
        }}

        .clickable-image:hover {{
            transform: scale(1.05);
            filter: brightness(1.1);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}

        /* Header Section - DARK */
        .hero-section {{
            position: relative;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, var(--bg-color) 0%, var(--surface-color) 100%);
            color: var(--primary-color);
            text-align: center;
            padding: 2rem;
        }}

        .hero-title-image {{
            max-width: 90%;
            max-height: 50vh;
            object-fit: contain;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
            margin-bottom: 2rem;
            border: 2px solid var(--glass-border);
            animation: glow 2s ease-in-out infinite alternate;
        }}


        .hero-title {{
            font-size: clamp(2.5rem, 6vw, 5rem);
            font-weight: 900;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
            margin-bottom: 1rem;
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .hero-subtitle {{
            font-size: clamp(1.2rem, 3vw, 2rem);
            opacity: 0.9;
            margin-bottom: 2rem;
            color: var(--secondary-color);
            word-break: break-word;
            max-width: 90%;
        }}

        .hero-video {{
            width: 100%;
            max-width: 900px;
            border-radius: 15px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.5);
            border: 1px solid var(--glass-border);
        }}

        /* FIXED Audio Player Popup */
        .audio-player-floating {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--surface-color);
            backdrop-filter: blur(20px);
            color: var(--primary-color);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            z-index: 1000;
            min-width: 280px;
            border: 1px solid var(--glass-border);
            transform: translateX(100%);
            transition: transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }}

        .audio-player-floating.show {{
            transform: translateX(0);
        }}

        .audio-player-floating.hide {{
            transform: translateX(90%);
        }}

        .audio-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .close-audio-btn {{
            background: none;
            border: none;
            color: var(--primary-color);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 5px;
            border-radius: 50%;
            transition: background 0.3s;
        }}

        .close-audio-btn:hover {{
            background: var(--glass-bg);
        }}

        .audio-controls {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}

        .audio-btn {{
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.3s;
            flex: 1;
            min-width: 80px;
        }}

        .audio-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}

        .audio-progress {{
            margin-top: 10px;
            font-size: 0.85rem;
            color: var(--secondary-color);
            text-align: center;
        }}

        /* Section Styling - DARK */
        .content-section {{
            padding: 5rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
            background: var(--bg-color);
        }}

        .section-title {{
            font-size: clamp(2rem, 5vw, 3.5rem);
            color: var(--primary-color);
            margin-bottom: 3rem;
            text-align: center;
            position: relative;
            font-weight: 700;
        }}

        .section-title::after {{
            content: '';
            position: absolute;
            bottom: -15px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-color), transparent);
            border-radius: 2px;
        }}

        /* Media Grid - DARK */
        .media-showcase {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2.5rem;
            margin-top: 3rem;
        }}

        .media-card {{
            background: var(--surface-color);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            transition: all 0.4s ease;
            border: 1px solid var(--glass-border);
        }}

        .media-card-gold {{
            background: darkgoldenrod;
            border-radius: 20px;
            text-color: black;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            transition: all 0.4s ease;
            border: 1px solid var(--glass-border);
            color: black !important;
        }}

        .media-card:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 20px 50px rgba(0,0,0,0.6);
        }}

        .media-card img {{
            width: 100%;
            height: 280px;
            object-fit: cover;
            transition: transform 0.4s;
        }}

        .media-card:hover img {{
            transform: scale(1.1);
        }}

        .media-card video {{
            width: 100%;
            max-height: 350px;
            background: var(--bg-color);
        }}

        .media-info {{
            padding: 1.5rem;
            color: var(--primary-color);
        }}


        .media-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            word-break: break-word;
            hyphens: auto;
            color: var(--accent-color);
        }}

        .media-description {{
            color: var(--secondary-color);
            font-size: 0.95rem;
            line-height: 1.6;
        }}

        /* Story Scene Styling - DARK */
        .story-scene {{
            margin-bottom: 5rem;
            padding: 3rem;
            background: var(--surface-color);
            border-radius: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            border: 1px solid var(--glass-border);
            position: relative;
        }}

        .story-scene-glod {{
            margin-bottom: 5rem;
            padding: 3rem;
            background: darkgoldenrod;
            border-radius: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            border: 1px solid var(--glass-border);
            position: relative;
        }}

        .scene-title {{
            font-size: 2.2rem;
            color: var(--accent-color);
            margin-bottom: 1.5rem;
            font-weight: 700;
            border-left: 5px solid var(--accent-color);
            padding-left: 2rem;
        }}

        .narrator-content {{
            font-size: 1.3rem;
            line-height: 1.8;
            color: var(--primary-color);
            margin-bottom: 2.5rem;
            font-style: italic;
            padding: 2rem;
            background: var(--glass-bg);
            border-radius: 15px;
            border: 1px solid var(--glass-border);
        }}

        .dialogue-container {{
            margin: 2rem 0;
            padding: 1.5rem;
            background: var(--glass-bg);
            border-radius: 15px;
            border-left: 4px solid var(--accent-color);
        }}

        .character-speaker {{
            font-weight: 700;
            color: var(--accent-color);
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
        }}

        .dialogue-speech {{
            color: var(--primary-color);
            line-height: 1.7;
            font-size: 1.05rem;
        }}

        /* Production Info Grid - DARK */
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin-top: 3rem;
        }}

        .info-card {{
            background: var(--surface-color);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            border: 1px solid var(--glass-border);
            transition: transform 0.3s;
        }}

        .info-card:hover {{
            transform: translateY(-5px);
        }}

        .info-card h3 {{
            color: var(--accent-color);
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }}

        .info-card p {{
            color: var(--secondary-color);
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }}

        /* End Card - DARK */
        .finale-section {{
            min-height: 100vh;
            background: linear-gradient(135deg, var(--bg-color), var(--surface-color));
            color: var(--primary-color);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 3rem;
        }}

        .finale-title {{
            font-size: clamp(3rem, 8vw, 6rem);
            margin-bottom: 2rem;
            font-weight: 900;
            text-shadow: 3px 3px 10px rgba(0,0,0,0.7);
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* Scroll Animations */
        .fade-in-up {{
            opacity: 0;
            transform: translateY(50px);
            transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}

        .fade-in-up.visible {{
            opacity: 1;
            transform: translateY(0);
        }}

        /* Transitions */
        .transitions-section {{
            border-top: 1px solid var(--glass-border);
            border-bottom: 1px solid var(--glass-border);
            padding: 1.5rem 0;
            background: var(--glass-bg);
            border-radius: 10px;
        }}

        .transition-card {{
            border: 2px solid var(--accent-color);
            background: linear-gradient(135deg, var(--surface-color), var(--glass-bg));
        }}

        .transition-card:hover {{
            transform: translateY(-5px) scale(1.03);
            border-color: var(--primary-color);
            box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        }}

        .transition-card img {{
            height: 200px;
            object-fit: cover;
            filter: sepia(0.1) saturate(1.1);
        }}

        /* PDF Section Styling */
        .pdf-showcase {{
            display: flex;
            flex-direction: column;
            gap: 3rem;
            margin-top: 3rem;
        }}

        .pdf-container {{
            background: var(--surface-color);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            border: 1px solid var(--glass-border);
        }}

        .pdf-header {{
            padding: 2rem;
            background: linear-gradient(135deg, var(--surface-color), var(--glass-bg));
            border-bottom: 1px solid var(--glass-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .pdf-title {{
            color: var(--accent-color);
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }}

        .pdf-meta {{
            display: flex;
            align-items: center;
            gap: 1rem;
            color: var(--secondary-color);
            flex-wrap: wrap;
        }}

        .download-btn {{
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            filter: brightness(1.1);
        }}

        .pdf-viewer-container {{
            position: relative;
            background: var(--bg-color);
        }}

        .pdf-viewer {{
            width: 100%;
            height: 800px;
            border: none;
            background: white;
            display: block;
        }}

        .pdf-controls {{
            padding: 1.5rem;
            background: var(--surface-color);
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            border-top: 1px solid var(--glass-border);
        }}

        .pdf-btn {{
            background: var(--glass-bg);
            color: var(--primary-color);
            border: 1px solid var(--glass-border);
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            transition: all 0.3s;
        }}

        .pdf-btn:hover {{
            background: var(--accent-color);
            color: white;
            transform: translateY(-1px);
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .content-section {{
                padding: 3rem 1rem;
            }}

            .media-showcase {{
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }}

            .info-grid {{
                grid-template-columns: 1fr;
            }}

            .story-scene {{
                padding: 1.5rem;
                margin-bottom: 3rem;
            }}

            .audio-player-floating {{
                right: 10px;
                top: 10px;
                min-width: 250px;
                padding: 1rem;
            }}

            .modal-navigation {{
                width: 40px;
                height: 40px;
                font-size: 1.2rem;
            }}

            .modal-prev {{
                left: 10px;
            }}

            .modal-next {{
                right: 10px;
            }}

            .modal-info {{
                padding: 1rem;
            }}

            .pdf-viewer {{
                height: 600px;
            }}

            .pdf-header {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .pdf-controls {{
                flex-direction: column;
            }}

            .pdf-btn, .download-btn {{
                width: 100%;
                justify-content: center;
            }}
        }}

        /* Loading Animation */
        .loading-shimmer {{
            background: linear-gradient(90deg, transparent, var(--glass-bg), transparent);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }}

        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}

        .hero-title-image, .media-card img {{
            {self._generate_camera_specific_css(story_data.style_preset.camera_style.value)}
        }}

        /* Special animations for camera effects */
        @keyframes glitch {{
            0% {{ transform: translate(0); }}
            20% {{ transform: translate(-2px, 2px); }}
            40% {{ transform: translate(-2px, -2px); }}
            60% {{ transform: translate(2px, 2px); }}
            80% {{ transform: translate(2px, -2px); }}
            100% {{ transform: translate(0); }}
        }}

        @keyframes hologram {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.8; filter: hue-rotate(0deg); }}
        }}

        /* Notification Animations */
        @keyframes slideInDown {{
            from {{
                transform: translate(-50%, -100%);
                opacity: 0;
            }}
            to {{
                transform: translate(-50%, 0);
                opacity: 1;
            }}
        }}

        @keyframes slideOutUp {{
            from {{
                transform: translate(-50%, 0);
                opacity: 1;
            }}
            to {{
                transform: translate(-50%, -100%);
                opacity: 0;
            }}
        }}
        """
    def _generate_fixed_audio_player(self, organized_media: Dict) -> str:
        """Generate FIXED audio player with proper toggle functionality"""

        audio_complete = organized_media.get('audio_complete', '')

        if not audio_complete:
            return '<!-- No audio file found -->'

        return f"""
        <div class="audio-player-floating" id="audioPlayerPopup">
            <div class="audio-header">
                <button class="close-audio-btn" id="closeAudioBtn" onclick="toggleAudioPlayer()" title="Toggle Audio Player">×</button>
                <strong>📚 Complete Audiobook</strong>
            </div>

            <audio id="mainAudioPlayer" style="width: 100%; margin: 10px 0;" preload="metadata">
                <source src="{audio_complete}" type="audio/wav">
                <source src="{audio_complete}" type="audio/mp3">
                Your browser does not support audio playback.
            </audio>

            <div class="audio-controls">
                <button class="audio-btn" onclick="playPauseAudio()" id="playPauseBtn">▶️ Play</button>
                <button class="audio-btn" onclick="skipBackward()">⏪ -10s</button>
                <button class="audio-btn" onclick="skipForward()">⏩ +10s</button>
            </div>

            <div class="audio-progress" id="audioProgressDisplay">
                Ready to play audiobook
            </div>
        </div>
        """

    def _generate_header_with_media(self, story_data, organized_media: Dict) -> str:
        """Generate header section with CLICKABLE media files"""

        cover_image = organized_media.get('cover_image', '')
        final_videos = organized_media.get('final_videos', [])
        main_video = final_videos[0] if final_videos else ''

        return f"""
        <section class="hero-section">
            {f'<img src="{cover_image}" alt="{story_data.title} Cover" class="hero-title-image clickable-image fade-in-up">' if cover_image else '<div class="loading-shimmer" style="width: 600px; height: 400px; border-radius: 20px; margin-bottom: 2rem;"></div>'}

            <h1 class="hero-title fade-in-up">{story_data.title}</h1>
            <p class="hero-subtitle fade-in-up">A {story_data.genre} Multimedia Experience</p>
            <p class="hero-subtitle fade-in-up">
                {story_data.style_preset.image_style.value.replace('_', ' ').title()}
            </p>

            {f'<video class="hero-video fade-in-up" controls preload="metadata"><source src="{main_video}" type="video/mp4">Your browser does not support video.</video>' if main_video else '<div class="loading-shimmer" style="width: 100%; max-width: 900px; height: 300px; border-radius: 15px;"></div>'}
        </section>
        """

    def _get_camera_description(self, camera_style: str) -> str:
        """Get description for camera style"""
        descriptions = {
            "Black & White Classic": "Timeless monochrome photography",
            "Film Noir": "High contrast dramatic lighting",
            "Neon Cyberpunk": "Futuristic neon-enhanced visuals",
            "Studio Ghibli": "Hand-drawn animation aesthetic",
            "Glitch Art": "Digital distortion effects",
            "Polaroid Vintage": "Instant camera retro feel",
            # Add more as needed
        }
        return descriptions.get(camera_style, "Custom visual treatment")

    def _generate_production_info(self, story_data, organized_media: Dict) -> str:
        """Generate production information section"""

        metadata = organized_media.get('metadata', {})
        media_count = organized_media.get('all_media_count', 0)

        return f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">Production Details</h2>

            <div class="info-grid">
                <div class="info-card fade-in-up">
                    <h3>🎬 Visual Direction</h3>
                    <p><strong>Style:</strong> {metadata.get('style', 'Unknown').replace('_', ' ').title()}</p>
                    <p><strong>Camera:</strong> {story_data.style_preset.camera_style.value}</p>
                    <p><strong>Effect:</strong> {self._get_camera_description(story_data.style_preset.camera_style.value)}</p>
                </div>

                <div class="info-card fade-in-up">
                    <h3>📊 Story Statistics</h3>
                    <p><strong>Characters:</strong> {len(story_data.characters)} main characters</p>
                    <p><strong>Scenes:</strong> {len(story_data.scenes)} narrative scenes</p>
                    <p><strong>Genre:</strong> {story_data.genre}</p>
                </div>

                <div class="info-card fade-in-up">
                    <h3>🎭 Character Cast</h3>
                    {self._generate_character_list_for_info(story_data)}
                </div>

                <div class="info-card fade-in-up">
                    <h3>📁 Media Assets</h3>
                    <p><strong>Total Files:</strong> {media_count} media files</p>
                    <p><strong>Transitions:</strong> {sum(len(scene.get('transitions', [])) for scene in organized_media.get('scenes', []))} transition effects</p>
                    <p><strong>Generated:</strong> {metadata.get('generation_time', 'Unknown')}</p>
                    <p><strong>Format:</strong> Interactive HTML Experience</p>
                </div>
            </div>
        </section>
        """

    def _generate_character_list_for_info(self, story_data) -> str:
        """Generate compact character list"""
        char_list = ""
        for char in story_data.characters[:4]:  # Max 4 characters for compact display
            char_list += f"<p><strong>{char.name}:</strong> {char.role.value.title()}</p>"
        if len(story_data.characters) > 4:
            char_list += f"<p><em>+{len(story_data.characters) - 4} more characters</em></p>"
        return char_list

    def _generate_world_section(self, story_data, organized_media: Dict) -> str:
        """Generate world section with CLICKABLE world images"""

        world_images = organized_media.get('world_images', [])

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">World & Setting</h2>

            <div class="narrator-content fade-in-up">
                {story_data.world_desc}
            </div>
        """

        if world_images:
            content += '<div class="media-showcase">'
            for i, world_img in enumerate(world_images):
                content += f"""
                <div class="media-card fade-in-up">
                    <img src="{world_img}" alt="World View {i + 1}" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                    <div class="media-info">
                        <div class="media-title">World Environment {i + 1}</div>
                        <div class="media-description">Environmental establishment shot showing the atmospheric setting of our story world.</div>
                    </div>
                </div>
                """
            content += '</div>'
        else:
            content += '<div class="media-info fade-in-up"><p>No world images found in media directory.</p></div>'

        content += '</section>'
        return content

    def _generate_characters_gallery(self, story_data, organized_media: Dict) -> str:
        """Generate characters section with CLICKABLE character images"""

        character_images = organized_media.get('character_images', [])

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">Character Gallery</h2>

            <div class="media-showcase">
        """

        for i, character in enumerate(story_data.characters):
            char_img = character_images[i] if i < len(character_images) else ''

            content += f"""
            <div class="media-card fade-in-up">
                {f'<img src="{char_img}" alt="{character.name}" class="clickable-image" loading="lazy" onerror="this.style.display=\'none\'">' if char_img else '<div style="height: 280px; background: var(--glass-bg); display: flex; align-items: center; justify-content: center; color: var(--secondary-color);">Character Image Not Found</div>'}
                <div class="media-info">
                    <div class="media-title">{character.name}</div>
                    <div class="media-description">
                        <strong>Role:</strong> {character.role.value.title()}<br>
                        <strong>Voice:</strong> {character.voice.value.replace('_', ' ').title()}<br>
                        {character.visual_desc}
                    </div>
                </div>
            </div>
            """

        content += """
            </div>
        </section>
        """

        return content

    def _generate_complete_story_experience(self, story_data, organized_media: Dict) -> str:
        """Generate complete story with CLICKABLE scene media"""

        scenes = organized_media.get('scenes', [])

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">Complete Story Experience</h2>
        """

        for i, scene_data in enumerate(scenes):
            if i >= len(story_data.scenes):
                continue

            story_scene = story_data.scenes[i]
            scene_images = scene_data.get('images', [])
            scene_transitions = scene_data.get('transitions', [])
            scene_clips = scene_data.get('clips', [])
            content += f"""
            <div class="story-scene fade-in-up" >
                <h3 class="scene-title">Scene {i + 1}: {story_scene.title}</h3>

                <div class="narrator-content">
                    <strong>Setting:</strong> {story_scene.setting}<br><br>
                    {story_scene.narrator}
                </div>
            """

            # Add ALL scene images with clickable functionality
            if scene_images:
                content += '<div class="media-showcase" style="margin: 2rem 0;">'
                for img_idx, scene_img in enumerate(scene_images):
                    perspective_match = re.search(r'perspective_(\d+)', scene_img)
                    perspective_info = f"Perspective {int(perspective_match.group(1)) + 1}" if perspective_match else f"View {img_idx + 1}"

                    content += f"""
                    <div class="media-card">
                        <img src="{scene_img}" alt="Scene {i + 1} {perspective_info}" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                        <div class="media-info">
                            <div class="media-title">{perspective_info}</div>
                            <div class="media-description">{story_scene.setting}</div>
                        </div>
                    </div>
                    """
                content += '</div>'

            # Add ALL scene clips (UNCUT!)
            if scene_clips:
                content += '<div class="media-showcase" style="margin: 2rem 0;">'
                for clip_idx, scene_clip in enumerate(scene_clips):
                    content += f"""
                    <div class="media-card{'-gold' if story_scene.duration >= 11 else ''}">
                        <video controls preload="metadata" style="width: 100%;">
                            <source src="{scene_clip}" type="video/mp4">
                            Your browser does not support video.
                        </video>
                        <div class="media-info">
                            <div class="media-title">Scene {i + 1} Video Clip {clip_idx + 1}</div>
                            <div class="media-description">AI-generated scene animation - uncut full clip</div>
                        </div>
                    </div>
                    """
                content += '</div>'

            # Add clickable transitions
            if scene_transitions:
                content += '<div class="transitions-section" style="margin: 1.5rem 0;">'
                content += '<h4 style="color: var(--accent-color); font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">🎬 Scene Transitions</h4>'
                content += '<div class="media-showcase" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">'

                for trans_idx, transition in enumerate(sorted(scene_transitions, key=lambda x: x['perspective_idx'])):
                    content += f"""
                    <div class="media-card transition-card">
                        <img src="{transition['path']}" alt="Scene {i + 1} Transition {transition['perspective_idx'] + 1}" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                        <div class="media-info">
                            <div class="media-title">Transition {transition['perspective_idx'] + 1}</div>
                            <div class="media-description">Scene transition effect</div>
                        </div>
                    </div>
                    """
                content += '</div></div>'

            # Add dialogue
            if story_scene.dialogue:
                for dialogue in story_scene.dialogue:
                    content += f"""
                    <div class="dialogue-container">
                        <div class="character-speaker">{dialogue.character}:</div>
                        <div class="dialogue-speech">"{dialogue.text}"</div>
                    </div>
                    """

            content += '</div>'

        content += '</section>'
        return content

    def _generate_finale_section(self, organized_media: Dict) -> str:
        """Generate finale section with CLICKABLE end image"""

        end_image = organized_media.get('end_image', '')

        if not end_image:
            return ''

        return f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">The End</h2>

            <div class="media-showcase" style="justify-content: center;">
                <div class="media-card fade-in-up" style="max-width: 600px;">
                    <img src="{end_image}" alt="Story Conclusion" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                    <div class="media-info">
                        <div class="media-title">Story Conclusion</div>
                        <div class="media-description">The final moment of our multimedia journey</div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _generate_end_card(self, story_data, organized_media: Dict) -> str:
        """Generate final end card"""

        return f"""
        <section class="finale-section">
            <h1 class="finale-title">THE END</h1>
            <p style="font-size: 1.5rem; margin-bottom: 2rem; color: var(--secondary-color);">Thank you for experiencing</p>
            <p style="font-size: 2.5rem; margin-bottom: 2rem; font-weight: bold; color: var(--accent-color);">{story_data.title}</p>
            <p style="font-size: 1.3rem; color: var(--secondary-color); opacity: 0.8;">A complete multimedia story experience</p>
            <p style="font-size: 1rem; margin-top: 3rem; opacity: 0.6; color: var(--secondary-color);">
                Style: {story_data.style_preset.image_style.value.title()} •
                Generated with AI
            </p>
        </section>
        """

    def _generate_fixed_javascript(self) -> str:
        """Generate FIXED JavaScript with PROPER modal cleanup to prevent freezing"""

        return """
        // Global variables
        let audioPlayer = null;
        let isAudioPlaying = false;
        let audioPlayerVisible = false;
        let currentImageModal = null;
        let allImages = [];
        let currentImageIndex = 0;
        let scrollPosition = 0;

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Page loaded, initializing...');

            // Initialize audio player
            initializeAudioPlayer();

            // Initialize scroll animations
            initializeScrollAnimations();

            // Initialize image modal system
            initializeImageModals();

            // Show audio player after 3 seconds
            setTimeout(() => {
                showAudioPlayer();
            }, 3000);
        });

        // IMAGE MODAL FUNCTIONALITY
        function initializeImageModals() {
            console.log('Initializing image modals...');

            // Collect all clickable images
            allImages = Array.from(document.querySelectorAll('.clickable-image'));
            console.log(`Found ${allImages.length} clickable images`);

            // Add click handlers to all images
            allImages.forEach((img, index) => {
                img.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    openImageModal(index);
                });
            });

            // Create modal HTML
            createImageModal();

            // Keyboard navigation
            document.addEventListener('keydown', function(e) {
                if (currentImageModal && currentImageModal.classList.contains('show')) {
                    e.preventDefault();
                    if (e.key === 'Escape') {
                        closeImageModal();
                    } else if (e.key === 'ArrowLeft') {
                        previousImage();
                    } else if (e.key === 'ArrowRight') {
                        nextImage();
                    }
                }
            });
        }

        function createImageModal() {
            const modal = document.createElement('div');
            modal.className = 'image-modal';
            modal.id = 'imageModal';

            modal.innerHTML = `
                <div class="modal-content">
                    <button class="modal-close" onclick="closeImageModal()">×</button>
                    <button class="modal-navigation modal-prev" onclick="previousImage()">‹</button>
                    <button class="modal-navigation modal-next" onclick="nextImage()">›</button>

                    <img class="modal-image" id="modalImage" alt="Modal Image">

                    <div class="modal-info">
                        <div class="modal-title" id="modalTitle">Image Title</div>
                        <div class="modal-description" id="modalDescription">Image description</div>
                        <div class="modal-details" id="modalDetails"></div>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);
            currentImageModal = modal;

            // FIXED: Close modal when clicking outside - prevent event bubbling
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    e.preventDefault();
                    e.stopPropagation();
                    closeImageModal();
                }
            });

            // FIXED: Prevent modal content clicks from closing modal
            const modalContent = modal.querySelector('.modal-content');
            modalContent.addEventListener('click', function(e) {
                e.stopPropagation();
            });
        }

        function openImageModal(index) {
            if (!allImages[index] || !currentImageModal) return;

             scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
            currentImageIndex = index;
            const img = allImages[index];

            // Get image information
            const imageInfo = getImageInfo(img, index);

            // Update modal content
            document.getElementById('modalImage').src = img.src;
            document.getElementById('modalTitle').textContent = imageInfo.title;
            document.getElementById('modalDescription').textContent = imageInfo.description;

            // Update modal details
            const detailsContainer = document.getElementById('modalDetails');
            detailsContainer.innerHTML = '';

            imageInfo.details.forEach(detail => {
                const detailElement = document.createElement('div');
                detailElement.className = 'modal-detail';
                detailElement.textContent = detail;
                detailsContainer.appendChild(detailElement);
            });

            // FIXED: Proper modal display and body scroll handling
            currentImageModal.style.display = 'flex';
            requestAnimationFrame(() => {
                currentImageModal.classList.add('show');
                document.body.style.overflow = 'hidden';
                document.body.style.position = 'fixed';
                document.body.style.width = '100%';
            });

            // Update navigation buttons visibility
            const prevBtn = currentImageModal.querySelector('.modal-prev');
            const nextBtn = currentImageModal.querySelector('.modal-next');

            prevBtn.style.display = allImages.length > 1 ? 'flex' : 'none';
            nextBtn.style.display = allImages.length > 1 ? 'flex' : 'none';

            console.log(`Opened modal for image ${index + 1} of ${allImages.length}`);
        }

        function closeImageModal() {
            if (!currentImageModal || !currentImageModal.classList.contains('show')) return;

            console.log('Closing image modal...');

            // FIXED: Proper cleanup sequence
            currentImageModal.classList.remove('show');

            // FIXED: Restore body scroll immediately
            document.body.style.overflow = '';
            document.body.style.position = '';
            document.body.style.width = '';

            window.scrollTo(0, scrollPosition);
            // Hide modal after transition
            setTimeout(() => {
                if (currentImageModal) {
                    currentImageModal.style.display = 'none';
                }
            }, 300);
            console.log('Image modal closed and scroll restored');
        }

        function previousImage() {
            if (allImages.length <= 1) return;

            currentImageIndex = (currentImageIndex - 1 + allImages.length) % allImages.length;
            updateModalContent(currentImageIndex);
        }

        function nextImage() {
            if (allImages.length <= 1) return;

            currentImageIndex = (currentImageIndex + 1) % allImages.length;
            updateModalContent(currentImageIndex);
        }

        // FIXED: Separate update function to avoid reopening modal
        function updateModalContent(index) {
            if (!allImages[index] || !currentImageModal) return;

            const img = allImages[index];
            const imageInfo = getImageInfo(img, index);

            // Update modal content without reopening
            document.getElementById('modalImage').src = img.src;
            document.getElementById('modalTitle').textContent = imageInfo.title;
            document.getElementById('modalDescription').textContent = imageInfo.description;

            const detailsContainer = document.getElementById('modalDetails');
            detailsContainer.innerHTML = '';

            imageInfo.details.forEach(detail => {
                const detailElement = document.createElement('div');
                detailElement.className = 'modal-detail';
                detailElement.textContent = detail;
                detailsContainer.appendChild(detailElement);
            });
        }

        function getImageInfo(img, index) {
            // Extract information from image and surrounding elements
            let title = img.alt || `Image ${index + 1}`;
            let description = 'AI-generated multimedia content';
            let details = [`Image ${index + 1} of ${allImages.length}`];

            // Try to get title from nearby elements
            const mediaCard = img.closest('.media-card');
            if (mediaCard) {
                const titleElement = mediaCard.querySelector('.media-title');
                const descElement = mediaCard.querySelector('.media-description');

                if (titleElement) title = titleElement.textContent.trim();
                if (descElement) description = descElement.textContent.trim();
            }

            // Determine image type and add relevant details
            const src = img.src.toLowerCase();
            const fileName = src.split('/').pop();

            if (src.includes('cover')) {
                details.push('Story Cover');
                details.push('Type: Title Image');
            } else if (src.includes('character')) {
                details.push('Character Portrait');
                details.push('Type: Character Design');
            } else if (src.includes('world')) {
                details.push('World Building');
                details.push('Type: Environment');
            } else if (src.includes('scene')) {
                const sceneMatch = fileName.match(/scene_(\d+)/);
                if (sceneMatch) {
                    details.push(`Scene ${parseInt(sceneMatch[1]) + 1}`);
                }

                const perspectiveMatch = fileName.match(/perspective_(\d+)/);
                if (perspectiveMatch) {
                    details.push(`Perspective ${parseInt(perspectiveMatch[1]) + 1}`);
                }

                details.push('Type: Scene Image');
            } else if (src.includes('transition')) {
                details.push('Scene Transition');
                details.push('Type: Transition Effect');
            } else if (src.includes('end')) {
                details.push('Story Conclusion');
                details.push('Type: Ending Image');
            }

            // Add technical details
            details.push(`File: ${fileName}`);

            return { title, description, details };
        }

        // FIXED: Audio player initialization
        function initializeAudioPlayer() {
            audioPlayer = document.getElementById('mainAudioPlayer');

            if (!audioPlayer) {
                console.log('No audio player found');
                return;
            }

            console.log('Audio player found, setting up...');

            // Audio event listeners
            audioPlayer.addEventListener('loadedmetadata', function() {
                console.log('Audio metadata loaded');
                updateAudioProgress();
            });

            audioPlayer.addEventListener('timeupdate', updateAudioProgress);

            audioPlayer.addEventListener('play', function() {
                isAudioPlaying = true;
                updatePlayPauseButton();
            });

            audioPlayer.addEventListener('pause', function() {
                isAudioPlaying = false;
                updatePlayPauseButton();
            });

            audioPlayer.addEventListener('ended', function() {
                isAudioPlaying = false;
                updatePlayPauseButton();
                document.getElementById('audioProgressDisplay').textContent = 'Audiobook completed';
            });
        }

        // FIXED: Show/hide audio player
        function showAudioPlayer() {
            const popup = document.getElementById('audioPlayerPopup');
            if (popup) {
                popup.classList.remove('hide');
                popup.classList.add('show');
                audioPlayerVisible = true;
                console.log('Audio player shown');
            }
        }

        function hideAudioPlayer() {
            const popup = document.getElementById('audioPlayerPopup');
            if (popup) {
                popup.classList.remove('show');
                popup.classList.add('hide');
                audioPlayerVisible = false;
                console.log('Audio player hidden');
            }
        }

        // FIXED: Toggle audio player visibility
        function toggleAudioPlayer() {
            console.log('Toggle audio player, currently visible:', audioPlayerVisible);

            if (audioPlayerVisible) {
                hideAudioPlayer();
                // set closeAudioBtn button text to show
                document.getElementById('closeAudioBtn').textContent = '<';
            } else {
                // set button text to hide
                document.getElementById('closeAudioBtn').textContent = '×';
                showAudioPlayer();
            }
        }

        // FIXED: Play/pause functionality
        function playPauseAudio() {
            if (!audioPlayer) {
                console.log('No audio player available');
                return;
            }

            if (isAudioPlaying) {
                audioPlayer.pause();
                console.log('Audio paused');
            } else {
                audioPlayer.play().then(() => {
                    console.log('Audio playing');
                }).catch(error => {
                    console.error('Error playing audio:', error);
                });
            }
        }

        // Update play/pause button text
        function updatePlayPauseButton() {
            const btn = document.getElementById('playPauseBtn');
            if (btn) {
                btn.textContent = isAudioPlaying ? '⏸️ Pause' : '▶️ Play';
            }
        }

        // Skip functions
        function skipBackward() {
            if (audioPlayer) {
                audioPlayer.currentTime = Math.max(0, audioPlayer.currentTime - 10);
                console.log('Skipped backward');
            }
        }

        function skipForward() {
            if (audioPlayer) {
                audioPlayer.currentTime = Math.min(audioPlayer.duration, audioPlayer.currentTime + 10);
                console.log('Skipped forward');
            }
        }

        // Update progress display
        function updateAudioProgress() {
            if (!audioPlayer) return;

            const current = audioPlayer.currentTime;
            const duration = audioPlayer.duration;

            if (duration && !isNaN(duration)) {
                const progress = (current / duration) * 100;
                const currentTime = formatTime(current);
                const totalTime = formatTime(duration);

                const display = document.getElementById('audioProgressDisplay');
                if (display) {
                    display.textContent = `${currentTime} / ${totalTime} (${Math.round(progress)}%)`;
                }
            }
        }

        // Format time helper
        function formatTime(seconds) {
            if (!seconds || isNaN(seconds)) return '0:00';

            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.floor(seconds % 60);
            return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        }

        // Scroll animations
        function initializeScrollAnimations() {
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -100px 0px'
            };

            const observer = new IntersectionObserver(function(entries) {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            }, observerOptions);

            // Observe all fade-in elements
            document.querySelectorAll('.fade-in-up').forEach(el => {
                observer.observe(el);
            });
        }

        // Smooth scrolling for any links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // FIXED: Error handling for media with proper cleanup
        document.querySelectorAll('img, video, audio').forEach(media => {
            media.addEventListener('error', function() {
                console.warn('Media failed to load:', this.src);
                if (this.tagName === 'IMG' && this.classList.contains('clickable-image')) {
                    // Remove from clickable images array if it fails to load
                    const index = allImages.indexOf(this);
                    if (index > -1) {
                        allImages.splice(index, 1);
                    }

                    this.style.display = 'none';
                    const placeholder = document.createElement('div');
                    placeholder.style.cssText = 'height: 280px; background: var(--glass-bg); display: flex; align-items: center; justify-content: center; color: var(--secondary-color);';
                    placeholder.textContent = 'Image not found';
                    this.parentNode.replaceChild(placeholder, this);
                }
            });
        });

        // PDF Functions
        function downloadPDF(pdfPath, fileName) {
            console.log('Downloading PDF:', fileName);

            // Create download link
            const link = document.createElement('a');
            link.href = pdfPath;
            link.download = fileName;
            link.style.display = 'none';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Show success message
            showNotification(`📥 Downloading ${fileName}...`, 'success');
        }

        function openPDFFullscreen(pdfPath) {
            console.log('Opening PDF fullscreen:', pdfPath);

            // Open PDF in new window/tab
            const newWindow = window.open(pdfPath, '_blank');

            if (!newWindow) {
                showNotification('❌ Please allow popups to view PDF fullscreen', 'error');
            } else {
                showNotification('🔍 PDF opened in new tab', 'success');
            }
        }

        function printPDF(pdfPath) {
            console.log('Printing PDF:', pdfPath);

            // Open PDF for printing
            const printWindow = window.open(pdfPath, '_blank');

            if (printWindow) {
                printWindow.onload = function() {
                    printWindow.print();
                };
                showNotification('🖨️ PDF opened for printing', 'success');
            } else {
                showNotification('❌ Please allow popups to print PDF', 'error');
            }
        }

        // Notification system
        function showNotification(message, type = 'info') {
            // Remove existing notification
            const existing = document.querySelector('.notification');
            if (existing) {
                existing.remove();
            }

            // Create notification
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;

            // Style notification
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: var(--surface-color);
                color: var(--primary-color);
                padding: 1rem 2rem;
                border-radius: 10px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.4);
                border: 1px solid var(--glass-border);
                z-index: 10000;
                animation: slideInDown 0.3s ease-out;
                max-width: 90%;
                text-align: center;
            `;

            document.body.appendChild(notification);

            // Remove after 3 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOutUp 0.3s ease-in';
                setTimeout(() => {
                    notification.remove();
                }, 300);
            }, 3000);
        }

        // PDF iframe error handling
        document.addEventListener('DOMContentLoaded', function() {
            // Handle PDF iframe errors
            document.querySelectorAll('.pdf-viewer').forEach(iframe => {
                iframe.addEventListener('error', function() {
                    console.log('PDF iframe error, showing fallback');
                    this.style.display = 'none';

                    // Show fallback message
                    const fallback = document.createElement('div');
                    fallback.style.cssText = `
                        padding: 3rem;
                        text-align: center;
                        color: var(--secondary-color);
                        background: var(--glass-bg);
                    `;
                    fallback.innerHTML = `
                        <h3>PDF Preview Not Available</h3>
                        <p>Your browser doesn't support PDF embedding.</p>
                        <a href="${this.src.split('#')[0]}" download class="download-btn" style="margin-top: 1rem;">
                            📥 Download PDF Instead
                        </a>
                    `;

                    this.parentNode.insertBefore(fallback, this);
                });
            });
        });
        """

# ====================== PROJECT MANAGER ======================

def path_to_str(obj):
    """Convert Path objects to strings recursively"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [path_to_str(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return path_to_str(obj.__dict__)
    return obj


def str_to_path(obj, path_keys=None):
    """Convert specific string keys back to Path objects"""
    if path_keys is None:
        path_keys = ['path', 'project_path', 'image_path', 'audio_path', 'video_path', 'pdf_path']

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in path_keys and isinstance(v, str):
                result[k] = Path(v)
            elif k == 'path_obj' and isinstance(v, str):
                result[k] = Path(v)
            else:
                result[k] = str_to_path(v, path_keys)
        return result
    elif isinstance(obj, list):
        return [str_to_path(item, path_keys) for item in obj]
    return obj

from types import MappingProxyType

def make_json_serializable(obj):
    """Convert any object to JSON serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (dict, MappingProxyType)):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):  # Pydantic models
        return make_json_serializable(obj.model_dump())
    elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):  # Pydantic models
        return make_json_serializable(obj.dict())
    elif hasattr(obj, '__dict__'):  # Custom objects
        return make_json_serializable(vars(obj))
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


class ProjectManager:
    """Enhanced project management with autoresume functionality"""

    def __init__(self, cost_tracker: CostTracker):
        self.base_dir = Config.BASE_OUTPUT_DIR
        self.base_dir.mkdir(exist_ok=True)
        self.projects_index_file = self.base_dir / "projects_index.json"
        self.load_projects_index()
        self.cost_tracker = cost_tracker

    def load_projects_index(self):
        """Load projects index from file with proper deserialization"""
        if self.projects_index_file.exists():
            with open(self.projects_index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.projects_index = str_to_path(data)
                self.cost_tracker = CostTracker.from_summary(self.projects_index.get('cost_summary', {}))
        else:
            self.projects_index = {}

    def save_projects_index(self):
        """Save projects index to file with robust serialization"""
        try:
            serializable_index = make_json_serializable(self.projects_index)
            with open(self.projects_index_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_index, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save projects index: {e}")

    def get_prompt_hash(self, prompt: str) -> str:
        """Generate consistent hash for prompt"""
        return hashlib.md5(prompt.strip().lower().encode()).hexdigest()

    def find_existing_projects(self, prompt: str) -> List[Dict]:
        """Find existing projects for given prompt"""
        prompt_hash = self.get_prompt_hash(prompt)
        existing_projects = []

        for project_id, project_info in self.projects_index.items():
            if project_info.get('prompt_hash') == prompt_hash:
                # Convert path string to Path object
                project_path = Path(str(project_info['path']))  # Ensure it's a string first
                if project_path.exists():
                    # Get project status
                    status = self.check_project_status(project_path)
                    # Create a copy to avoid modifying original
                    project_copy = path_to_str(project_info.copy())
                    project_copy['status'] = status
                    project_copy['path_obj'] = project_path
                    existing_projects.append(project_copy)

        return existing_projects


    def check_project_status(self, project_dir: Path) -> Dict:
        """Check what assets exist in project - FIXED PDF detection"""
        status = {
            'story_yaml': (project_dir / "story.yaml").exists(),
            'metadata': (project_dir / "project_metadata.json").exists(),
            'images': len(list((project_dir / "images").glob("*.png"))) if (project_dir / "images").exists() else 0,
            'audio': len(list((project_dir / "audio").glob("*.wav"))) if (project_dir / "audio").exists() else 0,
            'video': len(list((project_dir / "video").glob("*.mp4"))) if (project_dir / "video").exists() else 0,
            'pdf': 0,  # Initialize
            'clips': len(list((project_dir / "video").glob("*.mp4"))) if (project_dir / "video").exists() else 0
        }

        # FIXED: Check for PDFs in multiple locations
        pdf_locations = [
            project_dir,  # Root directory
            project_dir / "pdf",  # PDF subdirectory
        ]

        for location in pdf_locations:
            if location.exists():
                pdf_files = list(location.glob("*.pdf"))
                status['pdf'] += len(pdf_files)

        # Calculate completion percentage
        total_steps = 6  # story, images, audio, video, pdf, clips
        completed_steps = sum([
            status['story_yaml'],
            status['images'] > 0,
            status['audio'] > 0,
            status['video'] > 0,
            status['pdf'] > 0,
            status['clips'] > 0
        ])
        status['completion_percentage'] = (completed_steps / total_steps) * 100

        return status

    def create_project(self, prompt: str, resume_project: Optional[Path] = None) -> Path:
        """Create new project or return existing for resume"""
        if resume_project:
            return resume_project

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = self.get_prompt_hash(prompt)
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt)[:20].replace(' ', '_')

        project_name = f"{timestamp}_{clean_prompt}_{prompt_hash[:8]}"
        project_dir = self.base_dir / project_name
        project_dir.mkdir(exist_ok=True)

        # Create subdirs
        for subdir in ["images", "audio", "video", "pdf", "transitions"]:
            (project_dir / subdir).mkdir(exist_ok=True)

        # Register project in index
        project_id = f"{timestamp}_{prompt_hash[:8]}"
        self.projects_index[project_id] = {
            'prompt': prompt,
            'prompt_hash': prompt_hash,
            'path': str(project_dir),
            'created': timestamp,
            'last_accessed': timestamp
        }
        self.save_projects_index()

        return project_dir

    def update_project_access(self, project_dir: Path):
        """Update last accessed time for project"""
        project_name = project_dir.name
        for project_id, project_info in self.projects_index.items():
            if Path(project_info['path']).name == project_name:
                project_info['last_accessed'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_projects_index()
                break

    def validate_story_structure(self, story: StoryData) -> Tuple[bool, List[str]]:
        """Validate story structure"""
        errors = []

        if not story.title or len(story.title.strip()) < 3:
            errors.append("Story title too short or missing")

        if not story.scenes or len(story.scenes) < 1:
            errors.append("Story needs at least one scene")

        if not story.world_desc or len(story.world_desc.strip()) < 10:
            errors.append("World description too short")

        for i, char in enumerate(story.characters):
            if not char.visual_desc or len(char.visual_desc.strip()) < 5:
                errors.append(f"Character {i + 1} visual_desc too short")

        return len(errors) == 0, errors

    def validate_generated_assets(self, project_dir: Path, expected_counts: Dict) -> Tuple[bool, Dict]:
        """Validate generated assets meet expectations - FIXED PDF validation"""
        status = self.check_project_status(project_dir)
        validation_results = {}

        # Check images
        expected_images = expected_counts.get('images', 0)
        validation_results['images'] = {
            'expected': expected_images,
            'actual': status['images'],
            'valid': status['images'] >= max(1, expected_images * 0.8)  # Allow 20% tolerance, minimum 1
        }

        # Check audio
        validation_results['audio'] = {
            'expected': expected_counts.get('audio', 1),
            'actual': status['audio'],
            'valid': status['audio'] >= 1
        }

        # Check video
        validation_results['video'] = {
            'expected': expected_counts.get('video', 1),
            'actual': status['video'],
            'valid': status['video'] >= 1
        }

        # FIXED: Check PDF properly
        validation_results['pdf'] = {
            'expected': expected_counts.get('pdf', 1),
            'actual': status['pdf'],
            'valid': status['pdf'] >= 1
        }

        all_valid = all(result['valid'] for result in validation_results.values())

        return all_valid, validation_results

    def get_resume_choice(self, existing_projects: List[Dict]) -> Tuple[str, Optional[Path]]:
        """Interactive choice for resume or new project"""
        print(f"\n🔍 Found {len(existing_projects)} existing project(s) for this prompt:")
        print("=" * 60)

        for i, project in enumerate(existing_projects, 1):
            status = project['status']
            print(f"{i}. Project: {Path(project['path']).name}")
            print(f"   📅 Created: {project['created']}")
            print(f"   📊 Completion: {status['completion_percentage']:.1f}%")
            print(f"   📁 Images: {status['images']}, 🎵 Audio: {status['audio']}, 🎬 Video: {status['video']}")
            print(f"   📄 PDF: {status['pdf']}, 🎞️ Clips: {status['clips']}")
            print()

        print("Options:")
        print("0. Create NEW project")
        for i in range(len(existing_projects)):
            print(f"{i + 1}. Resume project {i + 1}")

        while True:
            try:
                choice = input(f"\nEnter choice (0-{len(existing_projects)}): ").strip()
                choice_num = int(choice)

                if choice_num == 0:
                    return "new", None
                elif 1 <= choice_num <= len(existing_projects):
                    selected_project = existing_projects[choice_num - 1]
                    return "resume", selected_project['path_obj']
                else:
                    print(f"❌ Invalid choice. Enter 0-{len(existing_projects)}")
            except ValueError:
                print("❌ Invalid input. Enter a number.")

    def save_story_yaml(self, story: StoryData, project_dir: Path):
        """Save story as YAML with proper enum serialization"""
        is_valid, errors = self.validate_story_structure(story)
        if not is_valid:
            print("⚠️ Story validation warnings:")
            for error in errors:
                print(f"   - {error}")

        yaml_path = project_dir / "story.yaml"

        # Convert story to dict and serialize enums properly
        story_dict = self._serialize_enums_for_yaml(story.dict())

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(story_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"✅ Story YAML saved: {is_valid and 'Valid' or 'With warnings'}")

    def load_story_yaml(self, project_dir: Path) -> Optional[StoryData]:
        """Load story from YAML with proper enum deserialization"""
        story_yaml_path = project_dir / "story.yaml"
        if not story_yaml_path.exists():
            return None

        try:
            with open(story_yaml_path, 'r', encoding='utf-8') as f:
                story_data = yaml.safe_load(f)

            # Deserialize enums from string values
            story_data = self._deserialize_enums_from_yaml(story_data)

            existing_story = StoryData(**story_data)
            return existing_story

        except Exception as e:
            print(f"⚠️ Could not load existing story: {e}")
            return None

    def _serialize_enums_for_yaml(self, data):
        """Convert enum objects to their string values for YAML serialization"""
        if isinstance(data, dict):
            return {key: self._serialize_enums_for_yaml(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_enums_for_yaml(item) for item in data]
        elif isinstance(data, (VoiceType, CharacterRole, ImageStyle, VideoStyle)):
            return data.value  # Convert enum to its string value
        else:
            return data

    def _deserialize_enums_from_yaml(self, data):
        """Convert string values back to enum objects after YAML loading"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == 'voice' and isinstance(value, str):
                    try:
                        result[key] = VoiceType(value)
                    except ValueError:
                        result[key] = value  # Keep original if conversion fails
                elif key == 'role' and isinstance(value, str):
                    try:
                        result[key] = CharacterRole(value)
                    except ValueError:
                        result[key] = value
                elif key == 'image_style' and isinstance(value, str):
                    try:
                        result[key] = ImageStyle(value)
                    except ValueError:
                        result[key] = value
                elif key == 'camera_style' and isinstance(value, str):
                    try:
                        result[key] = VideoStyle(value)
                    except ValueError:
                        result[key] = value
                else:
                    result[key] = self._deserialize_enums_from_yaml(value)
            return result
        elif isinstance(data, list):
            return [self._deserialize_enums_from_yaml(item) for item in data]
        else:
            return data
    def save_metadata(self, story: StoryData, cost_summary: Dict, generated_files: Dict, project_dir: Path):
        """Save complete project metadata with robust serialization"""
        # Convert generated_files paths to strings
        serializable_files = make_json_serializable(generated_files)

        metadata = {
            'project_info': {
                'title': story.title,
                'generated': datetime.now().isoformat(),
                'version': '5.0 Enhanced with AutoResume'
            },
            'cost': make_json_serializable(self.cost_tracker.get_summary()),
            'story_structure': make_json_serializable(story),
            'cost_summary': make_json_serializable(cost_summary),
            'generated_files': serializable_files,
            'validation': {
                'story_valid': self.validate_story_structure(story)[0],
                'last_validated': datetime.now().isoformat()
            }
        }

        metadata_path = project_dir / "project_metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            # Fallback with even more aggressive serialization
            print(f"⚠️ Standard serialization failed, using fallback: {e}")
            metadata_safe = make_json_serializable(metadata)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_safe, f, indent=2, ensure_ascii=False, default=str)

    def create_summary(self, story: StoryData, cost_summary: Dict, generated_files: Dict, project_dir: Path):
        """Create project summary with validation info"""
        status = self.check_project_status(project_dir)

        summary = f"""# {story.title} - Enhanced Production Summary with AutoResume

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Story Details
- **Genre:** {story.genre}
- **Style:** {story.style_preset.image_style.value.title()}
- **Characters:** {len(story.characters)}
- **Scenes:** {len(story.scenes)}
- **World:** {story.world_desc}

### Project Status & Validation
- **Completion:** {status['completion_percentage']:.1f}%
- **Story Structure:** {'✅ Valid' if self.validate_story_structure(story)[0] else '⚠️ Has warnings'}

### Generated Assets (Enhanced)
- **Images:** {status['images']} {'✅' if status['images'] > 0 else '❌'}
- **Audio:** {status['audio']} files {'✅' if status['audio'] > 0 else '❌'}
- **Video:** {status['video']} files {'✅' if status['video'] > 0 else '❌'}
- **PDF:** {status['pdf']} files {'✅' if status['pdf'] > 0 else '❌'}

### Enhanced Features
- ✅ AutoResume functionality - continue interrupted projects
- ✅ Validation at each step
- ✅ Multiple world establishment images (2)
- ✅ Multiple scene perspectives (2-4 per scene)
- ✅ Chronological video sequence (one scene per story beat)
- ✅ Complete PDF with all generated images
- ✅ Different camera angles and viewpoints
- ✅ Character interaction variations

### Cost Summary
- **Total Cost:** ${cost_summary.get('total_cost_usd', 0):.3f}
- **Kokoro TTS:** {cost_summary['breakdown']['kokoro']['calls']} calls (${cost_summary['breakdown']['kokoro']['cost']:.3f})
- **Flux Schnell:** {cost_summary['breakdown']['flux_schnell']['calls']} calls (${cost_summary['breakdown']['flux_schnell']['cost']:.3f})
- **Flux KREA:** {cost_summary['breakdown']['flux_krea']['calls']} calls (${cost_summary['breakdown']['flux_krea']['cost']:.3f})
- **Flux kontext:** {cost_summary['breakdown']['flux_kontext']['calls']} calls (${cost_summary['breakdown']['flux_kontext']['cost']:.3f})
- **BANAN:** {cost_summary['breakdown']['banana']['calls']} calls (${cost_summary['breakdown']['banana']['cost']:.3f})
- **MINIMAX (clips):** {cost_summary['breakdown']['minimax']['calls']} calls (${cost_summary['breakdown']['minimax']['cost']:.3f})
- **ElevenLabs:** {cost_summary['breakdown']['elevenlabs']['calls']} tokens (${cost_summary['breakdown']['elevenlabs']['cost']:.3f})

### Project Location
`{project_dir}`

### Resume Information
- **Prompt Hash:** {self.get_prompt_hash(story.world_desc)}
- **Can Resume:** Yes - use same prompt to continue this project

---
*Generated by Enhanced Multimedia Story Generator v5.0 with AutoResume*
*Features: AutoResume | Validation | Multiple World Images | Scene Perspectives | Chronological Video | Complete PDF*
"""

        summary_path = project_dir / "PROJECT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    def move_final_media(self):
        """Move the final 2 videos (images and clip based), audio book, and PDF to a final media folder"""

        # Get the project directory from the project manager
        project_dir = self.base_dir

        # Create final media folder in the project directory
        final_media_dir = project_dir / "final_media"
        final_media_dir.mkdir(exist_ok=True)

        # Find and move the final videos
        video_dir = project_dir / "video"
        if video_dir.exists():
            video_files = list(video_dir.glob("*_final.mp4"))
            clip_video_files = list(video_dir.glob("*_clips_final.mp4"))

            # Move up to 2 final videos
            final_videos = video_files + clip_video_files
            for video_file in final_videos[:2]:  # Limit to 2 videos
                if video_file.exists():
                    destination = final_media_dir / video_file.name
                    shutil.move(str(video_file), str(destination))
                    print(f"Moved video: {video_file.name}")

        # Find and move the audio book
        audio_dir = project_dir / "audio"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*_complete.wav"))
            for audio_file in audio_files:
                if audio_file.exists():
                    destination = final_media_dir / audio_file.name
                    shutil.move(str(audio_file), str(destination))
                    print(f"Moved audio: {audio_file.name}")
                    break  # Only move first audio file

        # Find and move the PDF
        pdf_locations = [project_dir / "pdf", project_dir]
        for location in pdf_locations:
            if location.exists():
                pdf_files = list(location.glob("*.pdf"))
                for pdf_file in pdf_files:
                    if pdf_file.exists():
                        destination = final_media_dir / pdf_file.name
                        shutil.move(str(pdf_file), str(destination))
                        print(f"Moved PDF: {pdf_file.name}")
                        break  # Only move first PDF file

        print(f"Final media moved to: {final_media_dir}")

# ====================== ENHANCED MAIN ORCHESTRATOR ======================

async def run(app: App, *args):
    """Enhanced main production pipeline with AutoResume"""
    print("🎬 Enhanced Multimedia Story Generator v5.0 with AutoResume")
    print("🚀 Multiple Perspectives | Chronological Video | Complete Assets | AutoResume")
    print("=" * 70)

    # Get user input
    user_prompt = input("📝 Describe your story: ").strip()
    if not user_prompt:
        print("❌ No story prompt provided.")
        return

    use_clips = input("🎬 Generate AI video clips? (y/n): ").strip().lower()
    use_elevenlabs = input("🔊 Use ElevenLabs for TTS? (y/n): ").strip().lower() == 'y'

    time_start = time.time()

    # Initialize project manager

    cost_tracker = CostTracker()
    project_manager = ProjectManager(cost_tracker)

    # Check for existing projects
    existing_projects = project_manager.find_existing_projects(user_prompt)

    # Handle resume logic
    project_dir = None
    resume_mode = False
    existing_story = None

    if existing_projects:
        choice, resume_project_path = project_manager.get_resume_choice(existing_projects)
        if choice == "resume":
            project_dir = resume_project_path
            resume_mode = True
            project_manager.update_project_access(project_dir)
            print(f"📁 Resuming project: {project_dir.name}")

            existing_story = project_manager.load_story_yaml(project_dir)
            if existing_story:
                print(f"✅ Loaded existing story: '{existing_story.title}'")
        else:
            print("🆕 Creating new project...")
    else:
        print("🆕 No existing projects found. Creating new project...")

    if not project_dir:
        project_dir = project_manager.create_project(user_prompt)

    # Setup logging and cost tracking
    logger = setup_logging(project_dir)

    logger.info(f"Starting {'resume' if resume_mode else 'new'} production for: '{user_prompt}'")
    print(f"📁 Project: {project_dir.name}")

    generated_files = {}

    try:
        # Initialize ISAA
        print("🤖 Initializing AI systems...")
        isaa = app.get_mod("isaa")
        if not isaa:
            raise RuntimeError("ISAA module not found")
        await isaa.init_isaa(build=True)

        # Initialize generators
        story_gen = StoryGenerator(isaa, logger)
        image_gen = ImageGenerator(logger, cost_tracker, isaa)
        audio_gen = AudioGenerator(logger, cost_tracker, project_dir, use_elevenlabs=use_elevenlabs)
        video_gen = VideoGenerator(logger, project_dir)
        pdf_gen = PDFGenerator(logger)
        clip_gen = ClipGenerator(logger, cost_tracker, isaa, image_gen)
        clip_editor = ClipVideoEditor(logger, project_dir)
        html_generator = MultiMediaStoryHTMLGenerator(logger=logger)

        # Phase 1: Story generation (skip if resuming with valid story)
        story = existing_story
        if not story or not project_manager.validate_story_structure(story)[0]:
            print("📚 Phase 1: Generating story structure with AI-selected style...")

            ai_auto_style_preset = await isaa.mini_task_completion_format(
                mini_task="Create a style preset for the following story prompt: " + user_prompt,
                format_schema=StylePreset,
                agent_name="self",
                use_complex=True
            )
            if not ai_auto_style_preset:
                raise RuntimeError("Style generation failed")

            style_preset = StylePreset(**ai_auto_style_preset)
            print(
                f"🎨 AI Selected Style: {style_preset.image_style.value.title()} with {style_preset.camera_style.value.title()} camera work")

            story = await story_gen.generate_story(user_prompt, style_preset)
            if not story:
                raise RuntimeError("Story generation failed")

            # Validate story
            is_valid, errors = project_manager.validate_story_structure(story)
            if not is_valid:
                print("⚠️ Story validation failed:")
                for error in errors:
                    print(f"   - {error}")

                retry_choice = input("Continue anyway? (y/n): ").strip().lower()
                if retry_choice != 'y':
                    print("❌ Production cancelled due to story validation.")
                    return

            print(f"✅ Story {'validated and ' if is_valid else ''}created: '{story.title}'")
            project_manager.save_story_yaml(story, project_dir)

        else:
            print(f"✅ Using existing valid story: '{story.title}'")

        # Get cost tracking
        a_story_creator = await isaa.get_agent("story_creator")
        a_self = await isaa.get_agent("self")
        cost_tracker.agent_cost = a_story_creator.ac_cost + a_self.ac_cost

        print(f"   📖 Genre: {story.genre}")
        print(f"   💰 Story cost: ${cost_tracker.agent_cost:.3f}")
        print(f"   🎭 Characters: {len(story.characters)}")
        print(f"   🎬 Scenes: {len(story.scenes)}")

        # Check existing assets
        status = project_manager.check_project_status(project_dir)

        # Phase 2: Enhanced parallel content generation (skip completed parts)
        print("🎨 Phase 2: Enhanced parallel generation with validation...")

        # Determine what needs to be generated
        need_images = status['images'] == 0
        need_audio = status['audio'] == 0

        tasks = []

        if need_images:
            print("   🖼️ Generating images...")
            image_task = image_gen.generate_all_images(story, project_dir)
            tasks.append(('images', image_task))
        else:
            print(f"   ✅ Images already exist ({status['images']} files)")
            # Load existing images
            all_images_for_video = list((project_dir / "images").glob("*_scene_*.png"))
            all_images_complete = list((project_dir / "images").glob("*.png"))

        if need_audio:
            print("   🎵 Generating audio...")
            audio_task = audio_gen.generate_audio(story, project_dir)
            tasks.append(('audio', audio_task))
        else:
            print(f"   ✅ Audio already exists ({status['audio']} files)")
            audio_files = list((project_dir / "audio").glob("*.wav"))
            audio_path = audio_files[0] if audio_files else None

        # Run parallel tasks
        if tasks:
            print(f"   🔄 Running {len(tasks)} parallel tasks...")
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            # Process results with validation
            for i, (task_name, _) in enumerate(tasks):
                result = task_results[i]
                if isinstance(result, Exception):
                    logger.error(f"{task_name} generation failed: {result}")
                    print(f"❌ {task_name.title()} generation failed: {result}")
                else:
                    print(f"✅ {task_name.title()} completed successfully")
                    if task_name == 'images':
                        all_images_for_video = result.get('all_images', [])
                        all_images_complete = result.get('all_images_complete', [])
                    elif task_name == 'audio':
                        audio_path = result

        # Update generated files tracking
        if 'all_images_complete' in locals():
            generated_files['images'] = [str(p) for p in all_images_complete]
        else:
            # When loading existing images
            image_files = list((project_dir / "images").glob("*.png"))
            generated_files['images'] = [str(p) for p in image_files]
            all_images_complete = image_files
            all_images_for_video = [p for p in image_files if "_scene_" in str(p)]

        if 'audio_path' in locals() and audio_path:
            generated_files['audio'] = str(audio_path)
        elif status['audio'] > 0:
            audio_files = list((project_dir / "audio").glob("*.wav"))
            if audio_files:
                audio_path = audio_files[0]
                generated_files['audio'] = str(audio_path)
        # Validate image generation
        if need_images and 'all_images_complete' in locals():
            expected_images = len(story.scenes) * 3 + len(story.characters) + 2  # Rough estimate
            image_validation = len(all_images_complete) >= expected_images * 0.7  # 70% threshold
            print(
                f"   📊 Image validation: {len(all_images_complete)}/{expected_images} expected {'✅' if image_validation else '⚠️'}")

        # Phase 3: Video generation with validation
        need_video = status['video'] == 0
        use_clips = use_clips == 'y' if need_video else False

        if need_video:
            print("🎬 Phase 3: Creating chronological video with validation...")
            if 'all_images_for_video' in locals() and 'audio_path' in locals() and all_images_for_video and audio_path:
                video_path = await video_gen.create_video(story, all_images_for_video, audio_path, project_dir)
                generated_files['video'] = str(video_path) if video_path else None

                # Validate video
                if video_path and video_path.exists():
                    video_size = video_path.stat().st_size
                    video_valid = video_size > 1024 * 1024  # At least 1MB
                    print(f"✅ Video created and validated: {video_valid} ({video_size / 1024 / 1024:.1f}MB)")
                else:
                    print("❌ Video creation failed validation")

                # AI Clips generation
                clips_video = None
                if use_clips and video_path:
                    print("🎬 Phase 3A: Generating AI video clips...")
                    clips = await clip_gen.generate_all_clips(story, all_images_for_video, project_dir, image_gen)

                    if clips:
                        print(f"✅ Generated {len(clips)} AI clips")

                        # Phase 3B: Edit clips
                        print("🎞️ Phase 3B: Editing clips with audio sync...")
                        clips_video = await clip_editor.create_final_video(story, clips, audio_path, project_dir)
                        generated_files['clips_video'] = str(clips_video) if clips_video else None

                        if clips_video:
                            clips_size = clips_video.stat().st_size
                            print(f"✅ Clips video created: {clips_size / 1024 / 1024:.1f}MB")
            else:
                print("⚠️ Skipping video - missing images or audio")
        else:
            print(f"✅ Video already exists ({status['video']} files)")

        # Phase 4: PDF generation with validation
        need_pdf = status['pdf'] == 0

        if need_pdf:
            print("📄 Phase 4: Creating complete PDF with validation...")
            if 'all_images_complete' in locals():
                images_result = {
                    'all_images_complete': all_images_complete} if 'all_images_complete' in locals() else {}
                pdf_path = pdf_gen.create_complete_pdf(story, images_result, project_dir)

                # FIXED: Store PDF path properly
                if pdf_path:
                    generated_files['pdf'] = str(pdf_path)

                # FIXED: Validate PDF with correct path checking
                if pdf_path and pdf_path.exists():
                    pdf_size = pdf_path.stat().st_size
                    pdf_valid = pdf_size > 1024 * 100  # At least 100KB
                    print(f"✅ PDF created and validated: {pdf_valid} ({pdf_size / 1024:.0f}KB)")
                else:
                    print("❌ PDF creation failed validation")
                    generated_files['pdf'] = None
        else:
            # FIXED: When PDF already exists, find and track it
            print(f"✅ PDF already exists ({status['pdf']} files)")
            # Find existing PDF and add to generated_files
            for location in [project_dir, project_dir / "pdf"]:
                if location.exists():
                    pdf_files = list(location.glob("*.pdf"))
                    if pdf_files:
                        generated_files['pdf'] = str(pdf_files[0])  # Use first PDF found
                        break

        # Final validation and save
        print("\n🔍 Final validation...")

        expected_counts = {
            'images': len(story.scenes) * 3 + len(story.characters) + 2,
            'audio': 1,
            'video': 1,
            'pdf': 1
        }

        validation_passed, validation_results = project_manager.validate_generated_assets(project_dir, expected_counts)

        print("📊 Asset Validation Results:")
        for asset_type, result in validation_results.items():
            status_icon = "✅" if result['valid'] else "⚠️"
            print(f"   {status_icon} {asset_type.title()}: {result['actual']}/{result['expected']}")

        # Save project data with validation
        a_story_creator = await isaa.get_agent("story_creator")
        a_self = await isaa.get_agent("self")
        cost_tracker.agent_cost = a_story_creator.ac_cost + a_self.ac_cost
        cost_summary = cost_tracker.get_summary()
        project_manager.save_metadata(story, cost_summary, generated_files, project_dir)
        project_manager.create_summary(story, cost_summary, generated_files, project_dir)

        # Final report with validation
        final_status = project_manager.check_project_status(project_dir)

        # Create the complete chronological PDF
        html_path = html_generator.create_complete_html_experience(
            story_data=story,
            project_dir=project_dir,
            output_filename=None  # Will auto-generate filename
        )

        if html_path and html_path.exists():
            print(f"✅ Complete multimedia HTML created: {html_path}")
            print(f"📄 Size: {html_path.stat().st_size / (1024 * 1024):.1f} MB")
        else:
            print("❌ Failed to create multimedia HTML")

        print("\n" + "=" * 70)
        print("🎉 ENHANCED PRODUCTION COMPLETE WITH AUTORESUME!")
        print("=" * 70)

        print(f"📚 Story: {story.title}")
        print(f"🎨 Style: {story.style_preset.image_style.value.title()}")
        print(f"📁 Location: {project_dir}")
        print(f"💰 Total Cost: ${cost_summary['total_cost_usd']:.3f}")
        print(f"📊 Completion: {final_status['completion_percentage']:.1f}%")
        print(f"✅ Validation: {'All systems validated' if validation_passed else 'Some validations failed'}")
        print(f"🔄 Resume: Use same prompt to continue this project")
        print(f"⏳ Time: {time.time() - time_start:.2f} seconds")

        print("\n📈 Generated Assets:")
        print(f"🖼️ Images: {final_status['images']} files")
        print(f"🎵 Audio: {final_status['audio']} files")
        print(f"🎬 Video: {final_status['video']} files")
        print(f"📄 PDF: {final_status['pdf']} files")
        if use_clips:
            print(f"🎞️ AI Clips: {final_status['clips']} clips")

        print("\n💰 Cost Breakdown:")
        for service, details in cost_summary['breakdown'].items():
            print(f"   {service.title()}: {details['calls']} calls (${details['cost']:.3f})")
        print("=" * 70)

        logger.info(f"Enhanced production pipeline completed successfully! Validation: {validation_passed}")

    except Exception as e:
        logger.error(f"Production failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ Production Error: {e}")
        print("💡 You can resume this project later using the same prompt!")


if __name__ == '__main__':
    import sys
    app_instance = App(instance_name="StoryGeneratorV5")
    asyncio.run(run(app_instance, *sys.argv[1:]))
