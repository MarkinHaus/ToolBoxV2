# toolboxv2/mods/videoFlow/engine/generators/image_generator.py

import asyncio
import logging
import pathlib
import re
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

try:
    import fal_client
except ImportError as e:
    fal_client = None

from toolboxv2.mods.videoFlow.engine.config import Config, CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import (
    Character,
    Scene,
    StoryData,
    StylePreset,
)


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

    async def _generate_and_upload_world_image(
        self, story: StoryData, images_dir: Path, idx: int
    ) -> Optional[tuple]:
        """Generate styled world establishment image and upload it immediately"""
        world_perspectives = [
            f"Wide establishing shot: {story.world_desc}. Panoramic environmental overview, no characters, detailed landscape",
            f"Atmospheric environment: {story.world_desc}. Environmental mood, cinematic lighting, detailed setting",
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

    async def _generate_world_image(
        self, story: StoryData, images_dir: Path, idx: int
    ) -> Optional[Path]:
        """Generate styled world establishment images (kept for compatibility)"""
        result = await self._generate_and_upload_world_image(story, images_dir, idx)
        return result[0] if result else None

    def _select_scenes_for_video(
        self, scene_paths: List[Path], num_scenes: int
    ) -> List[Path]:
        """Select one scene image per scene for video (chronological order)"""
        if not scene_paths:
            return []

        # Group scene images by scene index
        scene_groups = {}
        for path in scene_paths:
            # Extract scene index from filename pattern: scene_XX_perspective_YY
            match = re.search(r"scene_(\d+)_", path.name)
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
                sorted_perspectives = sorted(
                    scene_groups[scene_idx],
                    key=lambda x: (0 if "perspective_01" in x.name else 1, x.name),
                )
                selected_scenes.extend(sorted_perspectives)

        return selected_scenes

    async def _generate_character_ref(
        self, character: Character, style: StylePreset, images_dir: Path, idx: int
    ) -> Optional[Path]:
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
                self.logger.error(
                    f"Failed to upload character reference: {character.name}"
                )

        return None

    async def _generate_base_scene_environment(
        self, scene: Scene, story: StoryData, images_dir: Path, scene_idx: int
    ) -> Optional[Path]:
        """Generate base scene environment using Kontext with world image as reference"""
        if not self.world_image_refs:
            self.logger.error(
                f"No world images available for Kontext scene generation. Available refs: {list(self.world_image_refs.keys())}"
            )
            # Fallback: generate scene environment directly
            fallback_prompt = f"Scene environment: {scene.setting}. {scene.title}. {scene.narrator}. No characters"
            styled_fallback = story.style_preset.get_style_prompt(
                fallback_prompt, "scene"
            )
            filename = f"scene_{scene_idx:02d}_base_environment.png"
            output_path = images_dir / filename
            fallback_success = await self._generate_with_schnell(
                styled_fallback, output_path
            )

            if fallback_success:
                # Upload fallback scene
                scene_url = await self._upload_to_fal(output_path)
                if scene_url:
                    self.base_scene_refs[f"scene_{scene_idx}"] = scene_url
                    self.logger.info(
                        f"Fallback scene environment uploaded for scene {scene_idx}"
                    )

            return output_path if fallback_success else None

        # Select world image (alternate between available world images)
        world_keys = list(self.world_image_refs.keys())
        world_key = world_keys[scene_idx % len(world_keys)]
        world_url = self.world_image_refs[world_key]

        self.logger.info(
            f"Using world image {world_key} for scene {scene_idx} environment"
        )

        # Create scene-specific environment prompt
        scene_env_prompt = (
            f"Transform this world into the specific scene environment: {scene.setting}. "
            f"Scene: {scene.title}. {scene.narrator}. "
            f"Create the environmental stage for character interaction, no characters present. "
            f"Maintain world consistency while adapting for scene-specific elements."
        )

        styled_prompt = story.style_preset.get_style_prompt(scene_env_prompt, "scene")

        filename = f"scene_{scene_idx:02d}_base_environment.png"
        output_path = images_dir / filename

        success = await self._generate_with_kontext(styled_prompt, world_url, output_path)
        if success:
            # Upload and verify before storing
            scene_url = await self._upload_to_fal(output_path)
            if scene_url:
                self.base_scene_refs[f"scene_{scene_idx}"] = scene_url
                self.logger.info(
                    f"Generated and uploaded base scene environment for scene {scene_idx}"
                )
                return output_path
            else:
                self.logger.error(
                    f"Failed to upload base scene environment for scene {scene_idx}"
                )

        # Fallback to Schnell if Kontext fails
        self.logger.warning(
            f"Kontext failed for scene {scene_idx}, falling back to Schnell"
        )
        fallback_prompt = f"Scene environment: {scene.setting}. {scene.title}. {scene.narrator}. No characters"
        styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
        fallback_success = await self._generate_with_schnell(styled_fallback, output_path)

        if fallback_success:
            # Upload fallback scene
            scene_url = await self._upload_to_fal(output_path)
            if scene_url:
                self.base_scene_refs[f"scene_{scene_idx}"] = scene_url
                self.logger.info(
                    f"Fallback scene environment uploaded for scene {scene_idx}"
                )

        return output_path if fallback_success else None

    async def generate_all_images(
        self, story: StoryData, project_dir: Path
    ) -> Dict[str, List[Path]]:
        """Generate all images with proper sequencing and validation"""
        self.logger.info(
            f"Starting two-stage parallel image generation with {story.style_preset.image_style.value} style..."
        )

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
        self.logger.info(
            f"Phase 1 complete: {len(self.character_refs)} character references uploaded"
        )
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

        self.logger.info(
            f"Phase 2 complete: {len(self.world_image_refs)} world images uploaded"
        )

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
        self.logger.info(
            f"Phase 3 complete: {len(self.base_scene_refs)} base scene environments uploaded"
        )
        for scene_key, scene_url in self.base_scene_refs.items():
            if scene_url is None:
                self.logger.error(f"Base scene environment upload failed: {scene_key}")

        # Validation before Phase 4: Check if we have enough resources
        missing_chars = [
            char.name
            for char in story.characters
            if char.name not in self.character_refs
            or self.character_refs[char.name] is None
        ]
        missing_scenes = [
            f"scene_{i}"
            for i in range(len(story.scenes))
            if f"scene_{i}" not in self.base_scene_refs
            or self.base_scene_refs[f"scene_{i}"] is None
        ]

        if missing_chars:
            self.logger.warning(f"Missing character references: {missing_chars}")
        if missing_scenes:
            self.logger.warning(f"Missing base scene environments: {missing_scenes}")

        # Phase 4: Generate different perspectives using banana (with validation)
        self.logger.info("Phase 4: Generating character perspectives with banana...")
        perspective_tasks = []
        for scene_idx, scene in enumerate(story.scenes):
            num_perspectives = min(
                4, max(2, len(scene.dialogue) + 1)
            )  # 2-4 perspectives per scene
            self.logger.info(
                f"Scene {scene_idx} ({scene.title}): generating {num_perspectives} perspectives"
            )
            for perspective_idx in range(num_perspectives):
                perspective_tasks.append(
                    self._generate_character_perspective(
                        scene, story, images_dir, scene_idx, perspective_idx
                    )
                )

        self.logger.info(
            f"Starting {len(perspective_tasks)} perspective generation tasks..."
        )
        perspective_results = await asyncio.gather(
            *perspective_tasks, return_exceptions=True
        )

        # Process results with detailed logging
        perspective_paths = []
        for i, result in enumerate(perspective_results):
            if isinstance(result, Path):
                perspective_paths.append(result)
                self.logger.info(f"Perspective task {i}: SUCCESS - {result.name}")
            elif isinstance(result, pathlib.WindowsPath):
                perspective_paths.append(result)
                self.logger.info(f"Perspective task {i}: SUCCESS - {result.name}")
            elif hasattr(result, "name") and hasattr(result, "absolute"):
                perspective_paths.append(result)
                self.logger.info(f"Perspective task {i}: SUCCESS - {result.name}")
            elif isinstance(result, bool) and result:
                self.logger.info(f"Perspective task {i}: SUCCESS - wrong")
            elif isinstance(result, bool) and not result:
                self.logger.error(f"Perspective task {i}: FAILED")
            elif isinstance(result, Exception):
                self.logger.error(
                    f"Perspective task {i}: FAILED with exception - {result}"
                )
            else:
                self.logger.warning(
                    f"Perspective task {i}: FAILED - returned {type(result)}"
                )

        self.logger.info(
            f"Phase 4 complete: {len(perspective_paths)} perspectives generated out of {len(perspective_tasks)} tasks"
        )

        # Phase 5: Generate cover and end card
        self.logger.info("Phase 5: Generating cover and end card...")
        cover_task = self._generate_cover(story, images_dir)
        end_task = self._generate_end_card(story, images_dir)

        cover_task_res, end_task_res = await asyncio.gather(
            cover_task, end_task, return_exceptions=True
        )
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
            f"Assembled {len(all_images_for_video)} images for video sequence generation, including all perspectives."
        )

        # The original 'scenes_for_video' can still be useful for other purposes (like a simple summary).
        scenes_for_video = self._select_scenes_for_video(
            perspective_paths, len(story.scenes)
        )

        # Create a complete list of all generated image assets for the PDF and metadata.
        all_images_complete_list = (
            ([cover_path] if cover_task_res else [])
            + world_paths
            + character_paths
            + base_scene_paths
            + perspective_paths
            + ([end_path] if end_task_res else [])
        )

        return {
            "all_images": all_images_for_video,  # Corrected list for VideoGenerator
            "all_images_complete": sorted([p for p in all_images_complete_list if p]),
            "character_refs": character_paths,
            "world_images": world_paths,
            "base_scene_environments": base_scene_paths,  # New: base environments
            "scene_perspectives": perspective_paths,  # New: character perspectives
            "scene_images_for_video": scenes_for_video,
            "cover": [cover_path] if cover_path else [],
            "end": [end_path] if end_path else [],
            "style_used": story.style_preset.image_style.value,
        }

    async def _generate_character_perspective(
        self,
        scene: Scene,
        story: StoryData,
        images_dir: Path,
        scene_idx: int,
        perspective_idx: int,
    ) -> Optional[Path]:
        """Generate character perspective using banana to place characters in scene environment"""

        self.logger.info(
            f"Starting perspective {perspective_idx} for scene {scene_idx}: {scene.title}"
        )

        # Get scene characters present in this scene
        scene_characters = list(
            set([d.character for d in scene.dialogue if d.character != "Narrator"])
        )
        if not scene_characters:
            self.logger.warning(
                f"No characters in scene {scene_idx}, skipping perspective {perspective_idx}"
            )
            return None

        # Define perspective types
        perspectives = [
            {
                "desc": "Wide establishing shot with all characters",
                "camera": "wide shot, cinematic framing, environmental context",
                "max_chars": 3,
            },
            {
                "desc": "Medium shot focusing on main characters",
                "camera": "medium shot, character focus, balanced composition",
                "max_chars": 2,
            },
            {
                "desc": "Close-up perspective on primary character",
                "camera": "close-up shot, intimate framing, emotional detail",
                "max_chars": 1,
            },
            {
                "desc": "Over-the-shoulder dialogue view",
                "camera": "over-the-shoulder view, dialogue perspective, character interaction",
                "max_chars": 2,
            },
        ]

        perspective = perspectives[perspective_idx % len(perspectives)]

        # Get base scene environment
        base_scene_key = f"scene_{scene_idx}"
        if (
            base_scene_key not in self.base_scene_refs
            or self.base_scene_refs[base_scene_key] is None
        ):
            self.logger.error(f"No valid base scene environment for scene {scene_idx}")
            # Generate with Schnell as fallback
            chars_for_perspective = scene_characters[: perspective["max_chars"]]
            fallback_prompt = (
                f"Scene with characters: {scene.title}. {scene.setting}. "
                f"Characters: {', '.join(chars_for_perspective)}. "
                f"{perspective['camera']}"
            )
            styled_fallback = story.style_preset.get_style_prompt(
                fallback_prompt, "scene", clip_type="editing"
            )
            filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            output_path = images_dir / filename
            if await self._generate_with_krea(styled_fallback, output_path):
                return output_path
            raise Exception(
                f"Failed to generate fallback image for scene {scene_idx} perspective {perspective_idx}"
            )

        base_scene_url = self.base_scene_refs[base_scene_key]

        # Select characters for this perspective
        chars_for_perspective = scene_characters
        char_refs = []
        char_names = []

        for char_name in chars_for_perspective:
            if (
                char_name in self.character_refs
                and self.character_refs[char_name] is not None
            ):
                char_refs.append(self.character_refs[char_name])
                char_names.append(char_name)

        if not char_refs:
            self.logger.error(
                f"No valid character references available for scene {scene_idx} perspective {perspective_idx}"
            )
            # Generate with Schnell as fallback
            fallback_prompt = (
                f"Scene with characters: {scene.title}. {scene.setting}. "
                f"Characters: {', '.join(chars_for_perspective)}. "
                f"{perspective['camera']}"
            )
            styled_fallback = story.style_preset.get_style_prompt(
                fallback_prompt, "scene", clip_type="editing"
            )
            filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            output_path = images_dir / filename
            if await self._generate_with_krea(styled_fallback, output_path):
                return output_path
            raise Exception(
                f"Failed to generate fallback image for scene {scene_idx} perspective {perspective_idx}"
            )

        # Create banana prompt for character placement
        char_placement_descriptions = self._get_character_placements(
            chars_for_perspective, scene, perspective["camera"]
        )

        banana_prompt = (
            f"Place these characters into the scene environment: {', '.join(char_names)}. "
            f"Scene: {scene.title} - {scene.setting}. "
            f"{perspective['camera']}. "
            f"{scene.poses}. "
            f"{char_placement_descriptions} "
            f"Characters should interact naturally with the environment and each other. "
            f"Maintain character appearance consistency and environmental lighting."
        )

        styled_prompt = story.style_preset.get_style_prompt(
            banana_prompt, "scene", clip_type="editing"
        )

        filename = f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
        output_path = images_dir / filename

        # Use banana with base scene + character references (ensure no None values)
        all_refs = [base_scene_url] + char_refs
        all_refs = [
            ref for ref in all_refs if ref is not None
        ]  # Filter out any None values

        if len(all_refs) < 2:  # Need at least base scene + 1 character
            self.logger.error(
                f"Insufficient valid references for banana: {len(all_refs)}"
            )
            # Fallback to Schnell
            fallback_prompt = (
                f"Scene with characters: {scene.title}. {scene.setting}. "
                f"Characters: {', '.join(char_names)}. "
                f"{perspective['camera']}"
            )
            styled_fallback = story.style_preset.get_style_prompt(
                fallback_prompt, "scene"
            )
            return await self._generate_with_schnell(styled_fallback, output_path)

        improve_prompt = (
            await self.isaa.mini_task_completion(
                mini_task=styled_prompt,
                user_task="Improve the following prompt for better character placement and interaction. the prompt is for image to video generation."
                "Describe the camera movement ( zoom in/out, panning, tilting, transition, transition effects, seen before and after) and the characters actions."
                "Make the prompt as short and information dense as possible.",
                agent_name="self",
            )
            if self.isaa
            else styled_prompt
        )

        success = await self._generate_with_banana(improve_prompt, all_refs, output_path)

        if success:
            return output_path

        # Fallback to Schnell
        self.logger.warning(
            f"Banana failed for scene {scene_idx} perspective {perspective_idx}, falling back to Schnell"
        )
        fallback_prompt = (
            f"Scene with characters: {scene.title}. {scene.setting}. "
            f"Characters: {', '.join(char_names)}. "
            f"{perspective['camera']}"
        )
        styled_fallback = story.style_preset.get_style_prompt(fallback_prompt, "scene")
        if await self._generate_with_schnell(styled_fallback, output_path):
            return output_path
        raise Exception(
            f"Failed to generate fallback image for scene {scene_idx} perspective {perspective_idx}"
        )

    def _get_character_placements(
        self, characters: List[str], scene: Scene, camera_angle: str
    ) -> str:
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

    async def _generate_end_card(
        self, story: StoryData, images_dir: Path
    ) -> Optional[Path]:
        """Generate styled end card"""
        base_prompt = (
            f"End card: 'The End' text, {story.genre} conclusion, elegant finale design"
        )
        styled_prompt = story.style_preset.get_style_prompt(base_prompt, "cover")
        return await self._generate_with_schnell(styled_prompt, images_dir / "99_end.png")

    # API Methods
    async def _generate_with_krea(
        self, prompt: str, output_path: Path, retries: int = 3
    ) -> bool:
        """Generate image with KREA model"""
        for attempt in range(retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.FLUX_KREA, prompt, {}
                )

                if result and "images" in result and result["images"]:
                    success = await self._download_image(
                        result["images"][0]["url"], output_path
                    )
                    if success:
                        self.cost_tracker.add_flux_krea_cost()
                        self.logger.info(f"Generated with KREA: {output_path.name}")
                        return True

            except Exception as e:
                self.logger.error(f"KREA generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

        return False

    async def _generate_with_schnell(
        self, prompt: str, output_path: Path, retries: int = 3
    ) -> bool:
        """Generate image with Schnell model"""
        for attempt in range(retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._sync_fal_call,
                    Config.FLUX_SCHNELL,
                    prompt,
                    {"num_inference_steps": 4},
                )

                if result and "images" in result and result["images"]:
                    success = await self._download_image(
                        result["images"][0]["url"], output_path
                    )
                    if success:
                        self.cost_tracker.add_flux_schnell_cost()
                        self.logger.info(f"Generated with Schnell: {output_path.name}")
                        return True

            except Exception as e:
                self.logger.error(f"Schnell generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

        return False

    async def _generate_with_kontext(
        self, prompt: str, image_url: str, output_path: Path, retries: int = 3
    ) -> bool:
        """Generate image with FLUX Kontext model"""
        for attempt in range(retries):
            try:
                args = {
                    "image_url": image_url,
                    "guidance_scale": 3.5,
                    "num_images": 1,
                    "output_format": "png",
                    "safety_tolerance": "2",
                }

                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.FLUX_KONTEXT, prompt, args
                )

                if result and "images" in result and result["images"]:
                    success = await self._download_image(
                        result["images"][0]["url"], output_path
                    )
                    if success:
                        self.cost_tracker.add_flux_kontext_cost()
                        self.logger.info(f"Generated with Kontext: {output_path.name}")
                        return True

            except Exception as e:
                self.logger.error(f"Kontext generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

        return False

    async def _generate_with_banana(
        self, prompt: str, image_urls: List[str], output_path: Path, retries: int = 3
    ) -> bool:
        """Generate image with banana (nano-banana/edit) model"""
        self.logger.info(
            f"Banana generation: {output_path.name} with {len(image_urls)} reference images"
        )

        for attempt in range(retries):
            try:
                args = {"image_urls": image_urls, "num_images": 1}

                self.logger.info(f"Banana attempt {attempt + 1}: calling API...")
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_fal_call, Config.BANANA_EDIT, prompt, args
                )

                if result and "images" in result and result["images"]:
                    self.logger.info(
                        f"Banana attempt {attempt + 1}: got result, downloading..."
                    )
                    success = await self._download_image(
                        result["images"][0]["url"], output_path
                    )
                    if success:
                        self.cost_tracker.add_banana_cost()
                        self.logger.info(f"Generated with banana: {output_path.name}")
                        return True
                    else:
                        self.logger.error(
                            f"Banana attempt {attempt + 1}: download failed"
                        )
                else:
                    self.logger.warning(
                        f"Banana attempt {attempt + 1}: no valid response - {result}"
                    )

            except Exception as e:
                self.logger.error(f"Banana generation attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

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
                args.update(
                    {"image_size": Config.IMAGE_SIZE, "num_images": 1, **extra_args}
                )

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
            if (
                url in self.images_dict
                and Path(self.images_dict[output_path.name]).exists()
            ):
                return True

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)

                        if output_path.exists() and output_path.stat().st_size > 1000:
                            return True
                        else:
                            self.logger.error(
                                f"Downloaded file is invalid: {output_path}"
                            )
                    else:
                        self.logger.error(
                            f"Download failed with status {response.status}: {url}"
                        )

        except Exception as e:
            self.logger.error(f"Download failed: {e}")

        return False
