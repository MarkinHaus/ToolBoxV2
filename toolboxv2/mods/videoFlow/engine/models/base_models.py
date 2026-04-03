# toolboxv2/mods/videoFlow/engine/models/base_models.py

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# Enums from demo_code.py
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
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    SIDEKICK = "sidekick"
    MYSTERIOUS = "mysterious"

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
    HOLLYWOOD_BLOCKBUSTER = "Hollywood Blockbuster"
    INDIE_FILM = "Indie Film"
    DOCUMENTARY = "Documentary"
    MUSIC_VIDEO = "Music Video"
    COMMERCIAL = "Commercial"
    DISNEY_ANIMATION = "Disney Animation"
    PIXAR_3D = "Pixar 3D"
    STUDIO_GHIBLI = "Studio Ghibli"
    STOP_MOTION = "Stop Motion"
    ANIME = "Anime"
    FILM_NOIR = "Film Noir"
    CYBERPUNK = "Cyberpunk"
    RETRO_80S = "Retro 80s"
    VINTAGE_FILM = "Vintage Film"
    BLACK_WHITE_CLASSIC = "Black & White Classic"
    TIME_LAPSE = "Time Lapse"
    SLOW_MOTION = "Slow Motion"
    GLITCH_ART = "Glitch Art"
    DOUBLE_EXPOSURE = "Double Exposure"
    SPLIT_SCREEN = "Split Screen"

# Pydantic Models from demo_code.py

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
    setting: str
    narrator: str
    dialogue: List[DialogueLine] = []
    poses: List[str] = []
    duration: float = 8.0

class StylePreset(BaseModel):
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
