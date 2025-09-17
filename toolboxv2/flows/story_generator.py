"""
Production-Ready Multimedia Story Generator v5.0
Complete refactor with parallel processing, intelligent scene cuts, and perfect A-Z coherence
Enhanced with multiple world images and scene perspectives
"""

import asyncio
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
    print(f"Missing dependencies: {e}")
from toolboxv2 import App

# ====================== CORE MODELS & CONFIG ======================
NAME = "story_generator"

class VoiceType(str, Enum):
    NARRATOR = "narrator"
    CHARACTER_1 = "character_1"
    CHARACTER_2 = "character_2"
    CHARACTER_3 = "character_3"

class CharacterRole(str, Enum):
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    SIDEKICK = "sidekick"
    MYSTERIOUS = "mysterious"

class ImageStyle(str, Enum):
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

class CameraStyle(str, Enum):
    IPHONE_8 = "iPhone 8"
    IPHONE_14_PRO = "iPhone 14 Pro"
    NIKON_D3500 = "Nikon D3500"
    CANON_EOS_R5 = "Canon EOS R5"
    SONY_A7R_IV = "Sony A7R IV"
    GOPRO_HERO9 = "GoPro Hero9"
    DJI_MAVIC_3 = "DJI Mavic 3 Drone"
    FUJIFILM_XT4 = "Fujifilm X-T4"

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
    camera_style: CameraStyle
    art_style: str = Field(default="realistic 8k photography")
    quality_modifiers: str = Field(default="high quality, detailed, professional")
    lighting: str = Field(default="natural lighting")
    color_palette: str = Field(default="vibrant colors")
    texture_emphasis: str = Field(default="")

    def get_style_prompt(self, base_prompt: str, image_type: str = "general") -> str:
        """Generate style-consistent prompt"""

        style_mapping = {
            ImageStyle.REALISTIC: "Photorealistic rendering, ultra-detailed, 4K resolution, true-to-life colors",
            ImageStyle.CARTOON: "Cartoon style, vibrant colors, clean outlines, cel-shaded look",
            ImageStyle.ANIME: "Anime art style, manga-inspired, expressive characters, detailed eyes, soft shading",
            ImageStyle.WATERCOLOR: "Watercolor painting, flowing pigments, soft gradients, natural brushstrokes",
            ImageStyle.OIL_PAINTING: "Oil painting, rich textures, layered strokes, classical fine art feel",
            ImageStyle.DIGITAL_ART: "Digital artwork, modern illustration, smooth gradients, stylized design",
            ImageStyle.PENCIL_SKETCH: "Pencil sketch, graphite lines, detailed hand-drawn textures, monochrome",
            ImageStyle.CYBERPUNK: "Cyberpunk aesthetic, neon glow, futuristic cityscapes, dark and moody atmosphere",
            ImageStyle.FANTASY: "Fantasy artwork, magical elements, mythical creatures, epic scenery",
            ImageStyle.NOIR: "Film noir style, high contrast, dramatic shadows, vintage cinematic tone"
        }

        camera_mapping = {
            CameraStyle.IPHONE_8: "Apple iPhone 8, 12MP camera, f/1.8 aperture, 4K video at 60fps",
            CameraStyle.IPHONE_14_PRO: "Apple iPhone 14 Pro, 48MP camera, ProRAW, Cinematic mode, 8K video",
            CameraStyle.NIKON_D3500: "Nikon D3500 DSLR, 24.2MP APS-C sensor, ISO 100-25600, beginner-friendly DSLR",
            CameraStyle.CANON_EOS_R5: "Canon EOS R5, 45MP full-frame sensor, 8K RAW video, Dual Pixel AF II",
            CameraStyle.SONY_A7R_IV: "Sony A7R IV, 61MP full-frame sensor, 10fps, 4K HDR video, IBIS",
            CameraStyle.GOPRO_HERO9: "GoPro Hero9, 20MP sensor, 5K video, waterproof action camera",
            CameraStyle.DJI_MAVIC_3: "DJI Mavic 3, Hasselblad 20MP camera, 4/3 CMOS, 5.1K video, drone photography",
            CameraStyle.FUJIFILM_XT4: "Fujifilm X-T4, 26.1MP APS-C sensor, IBIS, 4K/60fps video, film simulation modes"
        }

        style_prompt = style_mapping.get(self.image_style, "")
        camera_prompt = camera_mapping.get(self.camera_style, "")

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
        if image_type == "character":
            components.append("character sheet, reference pose, clear details")
        elif image_type == "scene":
            components.append("scene composition, environmental storytelling")
        elif image_type == "cover":
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
        camera_style=CameraStyle.CINEMATIC
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

    # Cost per call
    COSTS = {
        'kokoro': 0.002,  # Per audio segment
        'flux_schnell': 0.003,  # Per image
        'flux_krea': 0.005,  # Per image
        'flux_kontext': 0.04,  # Per image with reference (estimated)
        'banana': 0.039  # Per edit
    }

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
                self.flux_krea_cost + self.flux_kontext_cost + self.banana_cost)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_cost_usd': round(self.total_cost, 3),
            'breakdown': {
                'agent': {'calls': 1, 'cost': round(self.agent_cost, 3)},
                'kokoro': {'calls': self.kokoro_calls, 'cost': round(self.kokoro_cost, 3)},
                'flux_schnell': {'calls': self.flux_schnell_calls, 'cost': round(self.flux_schnell_cost, 3)},
                'flux_krea': {'calls': self.flux_krea_calls, 'cost': round(self.flux_krea_cost, 3)},
                'flux_kontext': {'calls': self.flux_kontext_calls, 'cost': round(self.flux_kontext_cost, 3)},
                'banana': {'calls': self.banana_calls, 'cost': round(self.banana_cost, 3)}
            }
        }

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
                camera_style=CameraStyle.CINEMATIC
            )

        system_prompt = f"""Create a multimedia story with consistent {style_preset.image_style.value} visual styling for: "{prompt}"

Visual Style Requirements:
- All images should follow {style_preset.image_style.value} aesthetic
- Camera work: {style_preset.camera_style.value} approach
- Consistent lighting: {style_preset.lighting}
- Color scheme: {style_preset.color_palette}

Story Requirements:
- 2-3 main characters with distinct visual features optimized for {style_preset.image_style.value} style
- 3-4 scenes, each 2-3 sentences of narration + dialogue
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

    def __init__(self, logger: logging.Logger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.character_refs = {}  # Store character reference URLs
        self.world_image_refs = {}  # Store world image URLs
        self.base_scene_refs = {}  # Store base scene environment URLs


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
            num_perspectives = min(4, max(2, len(scene.dialogue) + 1))  # 2-4 perspectives per scene
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
            styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
            filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            output_path = images_dir / filename
            return await self._generate_with_schnell(styled_fallback, output_path)

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
            styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
            filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            output_path = images_dir / filename
            return await self._generate_with_schnell(styled_fallback, output_path)

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

        styled_prompt = story.style_preset.get_style_prompt(banana_prompt, "scene")

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

        success = await self._generate_with_banana(styled_prompt, all_refs, output_path)

        if success:
            return output_path

        # Fallback to Schnell
        self.logger.warning(
            f"Banana failed for scene {scene_idx} perspective {perspective_idx}, falling back to Schnell")
        fallback_prompt = (f"Scene with characters: {scene.title}. {scene.setting}. "
                           f"Characters: {', '.join(char_names)}. "
                           f"{perspective['camera']}")
        styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
        return await self._generate_with_schnell(styled_fallback, output_path)

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

            return await asyncio.get_event_loop().run_in_executor(
                None, fal_client.upload_file, str(image_path)
            )
        except Exception as e:
            self.logger.error(f"FAL upload failed: {e}")
            return None

    async def _download_image(self, url: str, output_path: Path) -> bool:
        """Download image from URL with production-ready error handling"""
        try:
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
    """Efficient Kokoro TTS audio generator"""

    def __init__(self, logger: logging.Logger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.temp_dir = Path(tempfile.mkdtemp(prefix="audio_"))

        # Voice mapping
        self.voice_map = {
            VoiceType.NARRATOR: "af_sarah",
            VoiceType.CHARACTER_1: "af_bella",
            VoiceType.CHARACTER_2: "am_adam",
            VoiceType.CHARACTER_3: "bm_daniel"
        }

    async def generate_audio(self, story: StoryData, project_dir: Path) -> Optional[Path]:
        """Generate synchronized audio matching video structure"""
        self.logger.info("Generating audio with scene timing...")

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
                narrator_segment = await self._generate_segment(scene.narrator, VoiceType.NARRATOR, f"scene_{idx}_narrator")
                if narrator_segment:
                    segments.append((narrator_segment, scene.duration * 0.4))  # 40% of scene time

            # Scene dialogue
            for d_idx, dialogue in enumerate(scene.dialogue):
                char_voice = next((c.voice for c in story.characters if c.name == dialogue.character), VoiceType.NARRATOR)
                dialogue_segment = await self._generate_segment(dialogue.text, char_voice, f"scene_{idx}_dialogue_{d_idx}")
                if dialogue_segment:
                    segments.append((dialogue_segment, scene.duration * 0.6 / len(scene.dialogue)))  # Remaining time split among dialogue

        # Combine with precise timing
        return await self._combine_segments(segments, audio_dir, story.title)

    async def _generate_segment(self, text: str, voice: VoiceType, name: str) -> Optional[Path]:
        """Generate single audio segment"""
        output_path = self.temp_dir / f"{name}.wav"
        text_file = self.temp_dir / f"{name}.txt"

        try:
            # Write text file
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            # Generate with Kokoro
            cmd = [
                "kokoro-tts", str(text_file), str(output_path),
                "--voice", self.voice_map[voice],
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
                return output_path

        except Exception as e:
            self.logger.error(f"Audio segment generation failed: {e}")
        finally:
            if text_file.exists():
                text_file.unlink()

        return None

    async def _combine_segments(self, segments: List[Tuple[Path, float]], audio_dir: Path, title: str) -> Optional[Path]:
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
                self.logger.info(f"Audio generated: {output_path.name}")
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

    def cleanup(self):
        """Cleanup temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

# ====================== INTELLIGENT VIDEO GENERATOR ======================

class VideoGenerator:
    """Enhanced video generator with perfect audio-video synchronization"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_"))

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
        finally:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)

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

# ====================== PROJECT MANAGER ======================

class ProjectManager:
    """Efficient project management"""

    def __init__(self):
        self.base_dir = Config.BASE_OUTPUT_DIR
        self.base_dir.mkdir(exist_ok=True)

    def create_project(self, prompt: str) -> Path:
        """Create project directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt)[:20].replace(' ', '_')

        project_name = f"{timestamp}_{clean_prompt}_{prompt_hash}"
        project_dir = self.base_dir / project_name
        project_dir.mkdir(exist_ok=True)

        # Create subdirs
        for subdir in ["images", "audio", "video", "pdf"]:
            (project_dir / subdir).mkdir(exist_ok=True)

        return project_dir

    def save_story_yaml(self, story: StoryData, project_dir: Path):
        """Save story as YAML for reference"""
        yaml_path = project_dir / "story.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(story.dict(), f, default_flow_style=False, allow_unicode=True)

    def save_metadata(self, story: StoryData, cost_summary: Dict, generated_files: Dict, project_dir: Path):
        """Save complete project metadata"""
        metadata = {
            'project_info': {
                'title': story.title,
                'generated': datetime.now().isoformat(),
                'version': '5.0 Enhanced'
            },
            'story_structure': story.dict(),
            'cost_summary': cost_summary,
            'generated_files': generated_files
        }

        metadata_path = project_dir / "project_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def create_summary(self, story: StoryData, cost_summary: Dict, generated_files: Dict, project_dir: Path):
        """Create project summary"""
        summary = f"""# {story.title} - Enhanced Production Summary

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Story Details
- **Genre:** {story.genre}
- **Style:** {story.style_preset.image_style.value.title()}
- **Characters:** {len(story.characters)}
- **Scenes:** {len(story.scenes)}
- **World:** {story.world_desc}

### Generated Assets (Enhanced)
- **Character References:** {len(story.characters)}
- **Audio:** {'✅' if generated_files.get('audio') else '❌'}
- **Video:** {'✅ (Chronological, one scene per sequence)' if generated_files.get('video') else '❌'}
- **PDF:** {'✅ (Contains ALL images and perspectives)' if generated_files.get('pdf') else '❌'}

### Enhanced Features
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

### Project Location
`{project_dir}`

---
*Generated by Enhanced Multimedia Story Generator v5.0*
*Features: Multiple World Images | Scene Perspectives | Chronological Video | Complete PDF*
"""

        summary_path = project_dir / "PROJECT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

# ====================== MAIN ORCHESTRATOR ======================

async def run(app: App, *args):
    """Enhanced main production pipeline"""
    print("🎬 Enhanced Multimedia Story Generator v5.0")
    print("🚀 Multiple Perspectives | Chronological Video | Complete Assets")
    print("="*70)

    # Get user input
    user_prompt = input("📝 Describe your story: ").strip()
    time_start = time.time()
    if not user_prompt:
        print("❌ No story prompt provided.")
        return

    # Initialize
    project_manager = ProjectManager()
    project_dir = project_manager.create_project(user_prompt)
    logger = setup_logging(project_dir)
    cost_tracker = CostTracker()

    logger.info(f"Starting enhanced production for: '{user_prompt}'")
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
        image_gen = ImageGenerator(logger, cost_tracker)
        audio_gen = AudioGenerator(logger, cost_tracker)
        video_gen = VideoGenerator(logger)
        pdf_gen = PDFGenerator(logger)

        # Phase 1: Generate story with auto style
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
        print(f"🎨 AI Selected Style: {style_preset.image_style.value.title()} with {style_preset.camera_style.value.title()} camera work")

        story = await story_gen.generate_story(user_prompt, style_preset)
        if not story:
            raise RuntimeError("Story generation failed")

        print(f"✅ Story created: '{story.title}'")
        a_story_creator = await isaa.get_agent("story_creator")
        a_self = await isaa.get_agent("self")
        cost_tracker.agent_cost = a_story_creator.ac_cost+a_self.ac_cost
        print(f"   📖 Genre: {story.genre}")
        print(f"   💰 Story cost: ${cost_tracker.agent_cost:.3f}")
        print(f"   🎭 Characters: {len(story.characters)}")
        print(f"   🎬 Scenes: {len(story.scenes)}")

        # Save story YAML immediately
        project_manager.save_story_yaml(story, project_dir)

        # Phase 2: Enhanced parallel content generation
        print("🎨 Phase 2: Enhanced parallel generation (multiple perspectives)...")

        # Start image and audio generation in parallel
        image_task = image_gen.generate_all_images(story, project_dir)
        audio_task = audio_gen.generate_audio(story, project_dir)

        # Wait for both to complete
        images_result, audio_path = await asyncio.gather(image_task, audio_task, return_exceptions=True)

        # Handle results
        if isinstance(images_result, Exception):
            logger.error(f"Image generation failed: {images_result}")
            images_result = {'all_images': [], 'all_images_complete': []}

        if isinstance(audio_path, Exception):
            logger.error(f"Audio generation failed: {audio_path}")
            audio_path = None

        all_images_for_video = images_result.get('all_images', [])  # Chronological for video
        all_images_complete = images_result.get('all_images_complete', [])  # All images for PDF

        generated_files['images'] = [str(p) for p in all_images_complete]
        generated_files['audio'] = str(audio_path) if audio_path else None

        print(f"✅ Generated {len(all_images_for_video)} images for video (chronological)")
        print(f"✅ Generated audio: {'Yes' if audio_path else 'No'}")

        # Phase 3: Create chronological video
        print("🎬 Phase 3: Creating chronological video (one scene per sequence)...")
        if all_images_for_video and audio_path:
            video_path = await video_gen.create_video(story, all_images_for_video, audio_path, project_dir)
            generated_files['video'] = str(video_path) if video_path else None
            print(f"✅ Chronological video created: {'Yes' if video_path else 'No'}")
        else:
            print("⚠️  Skipping video creation (missing images or audio)")
            video_path = None

        # Phase 4: Create enhanced PDF with ALL images
        print("📄 Phase 4: Creating complete PDF with all perspectives...")
        pdf_path = pdf_gen.create_complete_pdf(story, images_result, project_dir)
        generated_files['pdf'] = str(pdf_path) if pdf_path else None
        print(f"✅ Enhanced PDF created: {'Yes' if pdf_path else 'No'}")

        # Save project data
        a_story_creator = await isaa.get_agent("story_creator")
        a_self = await isaa.get_agent("self")
        cost_tracker.agent_cost = a_story_creator.ac_cost + a_self.ac_cost
        cost_summary = cost_tracker.get_summary()
        project_manager.save_metadata(story, cost_summary, generated_files, project_dir)
        project_manager.create_summary(story, cost_summary, generated_files, project_dir)

        # Final report
        print("\n" + "="*70)
        print("🎉 ENHANCED PRODUCTION COMPLETE!")
        print("="*70)
        print(f"📚 Story: {story.title}")
        print(f"🎨 Style: {story.style_preset.image_style.value.title()}")
        print(f"📁 Location: {project_dir}")
        print(f"💰 Total Cost: ${cost_summary['total_cost_usd']:.3f}")
        print(f"🎬 Video Images: {len(all_images_for_video)} (chronological)")
        print(f"🌍 World Views: 2 different perspectives")
        print(f"🎭 Scene Perspectives: 2-4 per scene")
        print(f"🎧 Audio: {'✅' if audio_path else '❌'}")
        print(f"🎬 Video: {'✅ Chronological' if video_path else '❌'}")
        print(f"📄 PDF: {'✅ Complete with all images' if pdf_path else '❌'}")
        print(f"⏳ Time: {time.time() - time_start:.2f} seconds")
        print("\n📊 Cost Breakdown:")
        for service, details in cost_summary['breakdown'].items():
            print(f"   {service.title()}: {details['calls']} calls (${details['cost']:.3f})")
        print("="*70)

        logger.info("Enhanced production pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Production failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ Production Error: {e}")

    finally:
        # Cleanup
        if 'audio_gen' in locals():
            audio_gen.cleanup()

if __name__ == '__main__':
    import sys
    app_instance = App(instance_name="StoryGeneratorV5")
    asyncio.run(run(app_instance, *sys.argv[1:]))
