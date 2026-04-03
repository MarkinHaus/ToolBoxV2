# toolboxv2/mods/videoFlow/tests/engine/generators/test_video_generator.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.generators.video_generator import VideoGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, Scene, DialogueLine, VoiceType, Character, CharacterRole, StylePreset, ImageStyle, VideoStyle

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def story_data():
    return StoryData(
        title="Test Video Story",
        genre="Sci-Fi",
        world_desc="A futuristic city.",
        characters=[
            Character(name="Cyborg", visual_desc="A half-human, half-machine being", role=CharacterRole.PROTAGONIST, voice=VoiceType.MALE_2)
        ],
        scenes=[
            Scene(
                title="The Chase",
                setting="A neon-lit alley",
                narrator="The chase was on.",
                dialogue=[
                    DialogueLine(character="Cyborg", text="I can't let them escape!", voice=VoiceType.MALE_2)
                ]
            )
        ],
        style_preset=StylePreset(image_style=ImageStyle.CYBERPUNK, camera_style=VideoStyle.CYBERPUNK)
    )

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.engine.generators.video_generator.asyncio.create_subprocess_exec')
async def test_create_video_runs(mock_subprocess, logger, story_data, tmp_path):
    # Arrange
    async def mock_subprocess_exec(*cmd, **kwargs):
        mock_process = MagicMock()
        mock_process.returncode = 0
        
        # If it's an ffprobe command, return a duration
        if 'ffprobe' in cmd[0]:
            mock_process.communicate = AsyncMock(return_value=(b'30.0', b''))
        # If it's an ffmpeg command, create the output file
        elif 'ffmpeg' in cmd[0]:
            output_path = Path(cmd[-1])
            with open(output_path, "wb") as f:
                f.write(b"fake video data" * 100)
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
        else:
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            
        return mock_process
    
    mock_subprocess.side_effect = mock_subprocess_exec

    video_gen = VideoGenerator(logger, tmp_path)

    # Create dummy image and audio files
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "00_cover.png").touch()
    (images_dir / "scene_00_perspective_00.png").touch()
    (images_dir / "99_end.png").touch()
    
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    # Act
    result_path = await video_gen.create_video(story_data, list(images_dir.glob("*.png")), audio_path, tmp_path)

    # Assert
    assert result_path is not None
    assert result_path.exists()
    assert result_path.name == "Test_Video_Story_final.mp4"
