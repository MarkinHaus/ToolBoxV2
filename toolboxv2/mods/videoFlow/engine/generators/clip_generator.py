# toolboxv2/mods/videoFlow/engine/generators/clip_generator.py

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import cv2

try:
    import fal_client
except ImportError as e:
    fal_client = None

from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.generators.image_generator import ImageGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import (
    Scene,
    StoryData,
    StylePreset,
)


class ClipGenerator:
    """Generate video clips with enhanced perspective continuity using banana model transitions"""

    def __init__(
        self,
        logger: logging.Logger,
        cost_tracker: CostTracker,
        isaa=None,
        img_generator=None,
    ):
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.isaa = isaa
        self.img_generator: ImageGenerator = (
            img_generator  # Enhanced: ImageGenerator for banana transitions
        )
        self.clip_cache = {}
        self.MINIMAX_COST = 0.05

    async def generate_all_clips(
        self,
        story: StoryData,
        images: List[str],
        project_dir: Path,
        image_gen: ImageGenerator,
    ) -> Dict[str, Path]:
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
        if "cover" in uploaded_images:
            cover_task = self._generate_cover_clip(
                story, uploaded_images["cover"][0], clips_dir
            )
            all_clip_tasks.append(cover_task)
            task_info.append(("cover", "cover"))

        # 2. World establishment clips (unchanged)
        if "world_images" in uploaded_images:
            for idx, world_img_url in enumerate(uploaded_images.get("world_images", [])):
                world_task = self._generate_world_clip(
                    story, world_img_url, clips_dir, idx
                )
                all_clip_tasks.append(world_task)
                task_info.append(("world", f"world_{idx}"))

        # 3. ENHANCED: Sequential perspective generation within each scene (but scenes remain parallel)
        scene_image_urls = uploaded_images.get("scene_perspectives", [])

        # Group scene URLs by scene index
        scenes_grouped = {}
        for scene_url in scene_image_urls:
            match = re.search(r"scene_(\d+)_perspective_(\d+)", scene_url)
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
                    scene,
                    scene_perspectives,
                    clips_dir,
                    scene_idx,
                    story,
                    uploaded_images,
                )
                all_clip_tasks.append(enhanced_scene_task)
                task_info.append(("enhanced_scene", f"scene_{scene_idx:02d}"))

        # 4. End clip task (unchanged)
        if "end" in uploaded_images:
            end_task = self._generate_end_clip(
                story, uploaded_images["end"][0], clips_dir
            )
            all_clip_tasks.append(end_task)
            task_info.append(("end", "end"))

        # Execute all tasks in parallel (scenes are parallel, perspectives within scenes are sequential)
        self.logger.info(
            f"🚀 Starting {len(all_clip_tasks)} enhanced clip generation tasks..."
        )
        start_time = asyncio.get_event_loop().time()

        results = await asyncio.gather(*all_clip_tasks, return_exceptions=True)
        parallel_time = asyncio.get_event_loop().time() - start_time

        # Process results
        generated_clips = {}
        successful_clips = 0
        failed_clips = 0

        for i, (result, (clip_type, clip_key)) in enumerate(zip(results, task_info)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Clip {clip_key} failed: {result}")
                failed_clips += 1
            elif clip_type == "enhanced_scene" and isinstance(result, dict):
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
        self.logger.info(
            f"✅ Successful: {successful_clips}"
            + (f", ❌ Failed: {failed_clips}" if failed_clips else "")
        )

        return generated_clips

    async def _generate_enhanced_scene_clips(
        self,
        scene: Scene,
        scene_perspectives: Dict[int, str],
        clips_dir: Path,
        scene_idx: int,
        story: StoryData,
        uploaded_images: Dict[str, List[str]],
    ) -> Dict[str, Path]:
        """Generate enhanced scene clips with perspective continuity"""
        self.logger.info(
            f"🎭 Generating enhanced scene {scene_idx} with {len(scene_perspectives)} perspectives..."
        )

        scene_clips = {}
        previous_clip_path = None

        # Sort perspectives by index for sequential processing
        sorted_perspectives = sorted(scene_perspectives.items())

        for perspective_idx, original_scene_url in sorted_perspectives:
            self.logger.info(
                f"🎬 Processing scene {scene_idx} perspective {perspective_idx}"
            )

            # For the first perspective, use original image
            if perspective_idx == 0 or not previous_clip_path:
                scene_url_for_clip = original_scene_url
                self.logger.info(f"Using original image for first perspective")
            else:
                # MAGIC STARTS: Generate transition image using previous clip's end frame
                scene_url_for_clip = await self._generate_transition_image(
                    previous_clip_path,
                    original_scene_url,
                    scene,
                    scene_idx,
                    perspective_idx,
                    story,
                    uploaded_images,
                    clips_dir,
                )

                if not scene_url_for_clip:
                    self.logger.warning(
                        f"Transition generation failed, using original image"
                    )
                    scene_url_for_clip = original_scene_url
                else:
                    self.logger.info(
                        f"✨ Generated seamless transition image for perspective {perspective_idx}"
                    )

            # Generate clip with enhanced or original image
            clip_path = await self._generate_scene_perspective_clip(
                scene, scene_url_for_clip, clips_dir, scene_idx, perspective_idx, story
            )

            if clip_path:
                perspective_key = (
                    f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}"
                )
                scene_clips[perspective_key] = clip_path
                previous_clip_path = clip_path  # Set for next perspective
                self.logger.info(
                    f"✅ Generated enhanced perspective clip: {perspective_key}"
                )
            else:
                self.logger.error(
                    f"❌ Failed to generate perspective {perspective_idx} for scene {scene_idx}"
                )

        return scene_clips

    async def _generate_transition_image(
        self,
        previous_clip_path: Path,
        next_perspective_url: str,
        scene: Scene,
        scene_idx: int,
        perspective_idx: int,
        story: StoryData,
        uploaded_images: Dict[str, List[str]],
        clips_dir: Path,
    ) -> Optional[str]:
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
                transition_frame_path = await self._extract_transition_frame(
                    previous_clip_path,
                    clips_dir,
                    scene_idx,
                    perspective_idx,
                    target_frame=target_frame * 5,
                )
                if transition_frame_path:
                    transition_frame_paths.append(transition_frame_path)

            if not transition_frame_paths:
                self.logger.error("Failed to extract transition frame")
                return None

            # Upload transition frame
            transition_frame_urls = []
            for transition_frame_path in transition_frame_paths:
                _transition_frame_url = await self.img_generator._upload_to_fal(
                    transition_frame_path
                )

                if not _transition_frame_url:
                    self.logger.error("Failed to upload transition frame")
                    return None

                transition_frame_urls.append(_transition_frame_url)

            # Get character references for this scene
            scene_characters = list(
                set([d.character for d in scene.dialogue if d.character != "Narrator"])
            )
            character_urls = []

            for char_name in scene_characters:
                if char_name in self.img_generator.character_refs:
                    char_url = self.img_generator.character_refs[char_name]
                    if char_url:
                        character_urls.append(char_url)

            # Prepare reference images: transition frame + next perspective + characters
            reference_images = (
                transition_frame_urls + [next_perspective_url] + character_urls
            )

            # Generate perfect banana prompt using mini task
            base_prompt = (
                f"Create seamless transition from previous scene moment to new perspective. "
                f"Scene: {scene.title} - {scene.setting}. "
                f"Characters: {', '.join(scene_characters)}. "
                f"Maintain visual continuity, character positioning, and environmental consistency. "
                f"Smooth cinematic transition, natural character movement."
                f"general location first image ( {next_perspective_url} ), "
                f"Last moments in video the video ( {transition_frame_urls} ) generate new logical cut view. in the first image environed based on the last video moments images."
            )

            enhanced_prompt = await self._create_perfect_banana_prompt(
                base_prompt, scene, scene_characters, perspective_idx
            )

            # Generate transition image with banana
            transition_output_path = (
                clips_dir.parent
                / "transitions"
                / f"transition_scene_{scene_idx:02d}_to_perspective_{perspective_idx:02d}.png"
            )
            transition_output_path.parent.mkdir(exist_ok=True)

            success = await self.img_generator._generate_with_banana(
                enhanced_prompt, reference_images, transition_output_path
            )

            if success:
                # Upload generated transition image
                final_transition_url = await self.img_generator._upload_to_fal(
                    transition_output_path
                )
                self.logger.info(
                    f"✨ Generated seamless transition for scene {scene_idx} perspective {perspective_idx}"
                )
                return final_transition_url
            else:
                self.logger.error("Banana generation failed for transition")
                return None

        except Exception as e:
            self.logger.error(f"Transition generation failed: {e}")
            return None

    async def _extract_transition_frame(
        self,
        clip_path: Path,
        clips_dir: Path,
        scene_idx: int,
        perspective_idx: int,
        target_frame: int = 0,
    ) -> Optional[Path]:
        """Extract intelligent transition frame from video clip (avoiding black/white end frames)"""
        try:
            # Output path for extracted frame
            frame_path = (
                clips_dir.parent
                / "transition_frames"
                / f"frame_scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.png"
            )
            frame_path.parent.mkdir(exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(str(clip_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 15:
                self.logger.warning(
                    f"Video too short ({total_frames} frames), using middle frame"
                )
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
                    f"Extracted transition frame at position {target_frame}/{total_frames} (brightness: {mean_brightness:.1f})"
                )
                return frame_path

            cap.release()

        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")

        return None

    async def _create_perfect_banana_prompt(
        self, base_prompt: str, scene: Scene, characters: List[str], perspective_idx: int
    ) -> str:
        """Create perfect banana prompt using mini task completion"""

        if not self.isaa:
            return base_prompt

        try:
            enhanced_task = (
                f"Create the perfect image generation prompt for seamless video transition. "
                f"Base context: {base_prompt} "
                f"Scene details: {scene.narrator} "
                f"Characters present: {', '.join(characters)} "
                f"Perspective {perspective_idx}: focus on natural character positioning and environmental flow. "
                f"The prompt should ensure visual continuity, proper character placement, and cinematic composition. "
                f"Make it concise but detailed for banana model image generation."
                f"{scene.poses} "
                f"{scene.dialogue[min(perspective_idx, len(scene.dialogue) - 1)].character}: "
                f"{scene.dialogue[min(perspective_idx, len(scene.dialogue) - 1)].text}"
            )

            perfect_prompt = await self.isaa.mini_task_completion(
                mini_task=enhanced_task,
                user_task="Create a perfect, concise image generation prompt that ensures seamless visual continuity between video clips. "
                "Focus on character consistency, environmental coherence, and smooth cinematic transitions. "
                "The prompt will be used with reference images to generate transition frames.",
                agent_name="story_creator",
                max_completion_tokens=300,
            )

            return perfect_prompt if perfect_prompt else base_prompt

        except Exception as e:
            self.logger.error(f"Perfect prompt generation failed: {e}")
            return base_prompt

    # Keep all existing methods unchanged for compatibility
    async def _upload_all_images_parallel(
        self, images: List[str], image_gen: ImageGenerator
    ) -> Dict[str, List[str]]:
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
            if (
                image_path
                and image_path.exists()
                and str(image_path) not in uploaded_image_paths
            ):
                uploaded_image_paths.append(image_path_str)

                upload_task = self._upload_to_fal(image_path, image_gen.images_dict)
                upload_tasks.append(upload_task)
                image_info.append((category, image_path))

        # Execute all uploads in parallel
        self.logger.info(f"🚀 Uploading {len(upload_tasks)} images SIMULTANEOUSLY...")
        start_time = asyncio.get_event_loop().time()

        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        upload_time = asyncio.get_event_loop().time() - start_time

        # Process upload results
        uploaded = {}
        successful_uploads = 0
        failed_uploads = 0

        for i, (result, (category, image_path)) in enumerate(
            zip(upload_results, image_info)
        ):
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
            f"✅ Successful: {successful_uploads}"
            + (f", ❌ Failed: {failed_uploads}" if failed_uploads else "")
        )

        return uploaded

    async def _generate_cover_clip(
        self, story: StoryData, cover_url: str, clips_dir: Path
    ) -> Optional[Path]:
        """Generate cover clip with title introduction"""
        prompt = (
            f"Cinematic title sequence: {story.title}. "
            f"Slow, majestic zoom revealing the world of this {story.genre} story. "
            f"Epic introduction, dramatic lighting, building anticipation. "
            f"Professional title card cinematography, {story.style_preset.image_style.value} style."
        )

        return await self._generate_clip(
            prompt,
            cover_url,
            clips_dir / "00_cover.mp4",
            style_preset=story.style_preset,
            duration="6",
            image_type="character",
        )

    async def _generate_world_clip(
        self, story: StoryData, world_url: str, clips_dir: Path, idx: int
    ) -> Optional[Path]:
        """Generate world establishment clip with sweeping camera movement"""
        world_prompts = [
            f"Sweeping establishing shot across {story.world_desc}. "
            f"Cinematic camera movement revealing the vast world. "
            f"Rich atmospheric details, immersive world-building. "
            f"{story.style_preset.camera_style.value} cinematography.",
            f"Dynamic world exploration: {story.world_desc}. "
            f"Camera gliding through the environment, discovering key locations. "
            f"Rich atmospheric details, immersive world-building. "
            f"Professional {story.style_preset.image_style.value} cinematography.",
        ]

        prompt = world_prompts[idx % len(world_prompts)]
        return await self._generate_clip(
            prompt,
            world_url,
            clips_dir / f"01_world_{idx:02d}.mp4",
            style_preset=story.style_preset,
            duration="6",
            image_type="scene",
        )

    async def _generate_scene_perspective_clip(
        self,
        scene: Scene,
        scene_url: str,
        clips_dir: Path,
        scene_idx: int,
        perspective_idx: int,
        story: StoryData,
    ) -> Optional[Path]:
        """Generate individual scene perspective clip with perfect action"""

        # Extract characters in this scene
        scene_characters = [
            d.character for d in scene.dialogue if d.character != "Narrator"
        ]

        # Create action-specific prompts based on scene content
        action_prompt = self._create_action_prompt(
            scene, scene_characters, perspective_idx
        )

        full_prompt = (
            f"Scene {scene_idx + 1}: {scene.title}. "
            f"Location: {scene.setting}. "
            f"{action_prompt} "
            f"Taken with {story.style_preset.camera_style.value}, "
            f"{story.style_preset.image_style.value} visual style, "
            f"dramatic lighting, professional filmmaking."
        )

        output_path = (
            clips_dir / f"scene_{scene_idx:02d}_perspective_{perspective_idx:02d}.mp4"
        )
        return await self._generate_clip(
            full_prompt,
            scene_url,
            output_path,
            style_preset=story.style_preset,
            duration="10" if scene.duration >= 11 else "6",
            image_type="scene",
        )

    def _create_action_prompt(
        self, scene: Scene, characters: List[str], perspective_idx: int
    ) -> str:
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
            f"Realistic dialogue pacing and character chemistry.",
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

    async def _generate_end_clip(
        self, story: StoryData, end_url: str, clips_dir: Path
    ) -> Optional[Path]:
        """Generate end clip with conclusion effect"""
        prompt = (
            f"Epic conclusion to {story.title}. "
            f"Final dramatic moment with emotional resolution. "
            f"Cinematic ending with fade to black, credits-ready. "
            f"Professional {story.style_preset.image_style.value} finale, "
            f"Taken with {story.style_preset.camera_style.value}. "
        )

        return await self._generate_clip(
            prompt,
            end_url,
            clips_dir / "99_end.mp4",
            style_preset=story.style_preset,
            duration="6",
            image_type="end",
        )

    async def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for better results"""
        return (
            await self.isaa.mini_task_completion(
                mini_task=prompt,
                user_task="Optimize the following prompt for better results. the prompt is for image to video generation."
                "Describe the camera movement ( zoom in/out, panning, tilting, transition, transition effects, seen before and after) and the characters actions."
                "Make the prompt as short and information dense as possible. the model cant take much text.",
                agent_name="self",
                max_completion_tokens=450,
            )
            if self.isaa
            else prompt
        )

    async def _generate_clip(
        self,
        prompt: str,
        image_url: str,
        output_path: Path,
        style_preset: StylePreset,
        duration: str = "10",
        retries: int = 3,
        image_type="general",
    ) -> Optional[Path]:
        """Generate single video clip using Minimax API"""

        for attempt in range(retries):
            try:
                self.logger.info(
                    f"Generating clip: {output_path.name} (attempt {attempt + 1})"
                )

                styled_prompt = style_preset.get_style_prompt(
                    prompt, image_type=image_type, clip_type="transitions"
                )
                better_prompt = await self._optimize_prompt(styled_prompt)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_minimax_call, better_prompt, image_url, duration
                )

                if result and "video" in result and result["video"]:
                    video_url = result["video"]["url"]
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

    def _sync_minimax_call(
        self, prompt: str, image_url: str, duration: str
    ) -> Optional[Dict]:
        """Synchronous Minimax API call"""
        try:
            args = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "prompt_optimizer": True,
            }

            return fal_client.subscribe(
                "fal-ai/minimax/hailuo-02-fast/image-to-video", arguments=args
            )

        except Exception as e:
            self.logger.error(f"Minimax API call failed: {e}")
            return None

    async def _upload_to_fal(
        self, image_path: Path, img_on_fal_dict: Dict[str, str]
    ) -> Optional[str]:
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
                        with open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        return output_path.exists() and output_path.stat().st_size > 10000
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
        return False
