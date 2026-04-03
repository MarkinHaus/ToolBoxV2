import logging
from typing import Optional

from toolboxv2 import App
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, StylePreset, ImageStyle, VideoStyle

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

        system_prompt = f'''Create a multimedia story with consistent {style_preset.image_style.value} visual styling for: "{prompt}"

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

Focus on visual storytelling that will translate effectively to {style_preset.image_style.value} images.'''

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
