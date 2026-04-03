# toolboxv2/mods/videoFlow/tests/engine/generators/test_audio_generator.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.generators.audio_generator import AudioGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, Scene, DialogueLine, VoiceType, Character, CharacterRole, StylePreset, ImageStyle, VideoStyle
from toolboxv2.mods.videoFlow.engine.config import CostTracker

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def cost_tracker():
    return CostTracker()

@pytest.fixture
def story_data():
    return StoryData(
        title="Test Audio Story",
        genre="Adventure",
        world_desc="A world of wonders.",
        characters=[
            Character(name="Narrator", visual_desc="", role=CharacterRole.MYSTERIOUS, voice=VoiceType.NARRATOR),
            Character(name="Hero", visual_desc="A brave hero", role=CharacterRole.PROTAGONIST, voice=VoiceType.MALE_1)
        ],
        scenes=[
            Scene(
                title="The Beginning",
                setting="A small village",
                narrator="Once upon a time...",
                dialogue=[
                    DialogueLine(character="Hero", text="I will save the world!", voice=VoiceType.MALE_1)
                ]
            )
        ],
        style_preset=StylePreset(image_style=ImageStyle.FANTASY, camera_style=VideoStyle.INDIE_FILM)
    )

@pytest.mark.asyncio
async def test_generate_audio_kokoro(logger, cost_tracker, story_data, tmp_path):
    # Arrange
    audio_gen = AudioGenerator(logger, cost_tracker, tmp_path, use_elevenlabs=False)
    
    async def mock_generate_segment(self, text, voice, name):
        path = self.temp_dir / f"{name}.wav"
        path.touch()
        return path

    async def mock_combine_segments(self, segments, audio_dir, title):
        path = audio_dir / f"{self._sanitize(title)}_complete.wav"
        path.touch()
        return path

    with patch('toolboxv2.mods.videoFlow.engine.generators.audio_generator.AudioGenerator._generate_segment', new=mock_generate_segment), \
         patch('toolboxv2.mods.videoFlow.engine.generators.audio_generator.AudioGenerator._combine_segments', new=mock_combine_segments):
        
        # Act
        result_path = await audio_gen.generate_audio(story_data, tmp_path)

        # Assert
        assert result_path is not None
        assert result_path.exists()
        assert result_path.name == "Test_Audio_Story_complete.wav"

@pytest.mark.asyncio
@patch('elevenlabs.client.ElevenLabs')
async def test_generate_audio_elevenlabs(mock_elevenlabs, logger, cost_tracker, story_data, tmp_path):
    # Arrange
    mock_elevenlabs_instance = mock_elevenlabs.return_value
    mock_elevenlabs_instance.text_to_speech.convert.return_value = [b'fake audio data']
    
    audio_gen = AudioGenerator(logger, cost_tracker, tmp_path, use_elevenlabs=True)

    async def mock_generate_segment(self, text, voice, name):
        path = self.temp_dir / f"{name}.wav"
        path.touch()
        return path

    async def mock_combine_segments(self, segments, audio_dir, title):
        path = audio_dir / f"{self._sanitize(title)}_complete.wav"
        path.touch()
        return path

    with patch('toolboxv2.mods.videoFlow.engine.generators.audio_generator.AudioGenerator._generate_segment', new=mock_generate_segment), \
         patch('toolboxv2.mods.videoFlow.engine.generators.audio_generator.AudioGenerator._combine_segments', new=mock_combine_segments):

        # Act
        result_path = await audio_gen.generate_audio(story_data, tmp_path)

        # Assert
        assert result_path is not None
        assert result_path.exists()
        assert result_path.name == "Test_Audio_Story_complete.wav"
