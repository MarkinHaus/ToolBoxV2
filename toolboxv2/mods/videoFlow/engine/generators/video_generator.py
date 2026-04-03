# toolboxv2/mods/videoFlow/engine/generators/video_generator.py

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from toolboxv2.mods.videoFlow.engine.config import Config
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData


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
        safe_title = re.sub(r'[<>:"/\|?* ]', '_', title)[:30]
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
