# toolboxv2/mods/videoFlow/engine/generators/audio_generator.py

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from toolboxv2.mods.videoFlow.engine.config import Config, CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import VoiceType, StoryData


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
            except ImportError:
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

    async def generate_audio(self, story: StoryData, project_dir: Path) -> Optional[Path]:
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
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
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
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
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
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
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
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0:
                return output_path
        except Exception:
            pass

        return None

    def _sanitize(self, filename: str) -> str:
        """Sanitize filename"""
        return re.sub(r'[<>:"/\\|?* ]', '_', filename)[:50]
