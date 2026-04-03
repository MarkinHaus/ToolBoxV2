# toolboxv2/mods/videoFlow/tests/engine/generators/test_clip_generator.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.generators.clip_generator import ClipGenerator
from toolboxv2.mods.videoFlow.engine.generators.image_generator import ImageGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, Scene, DialogueLine, VoiceType, Character, CharacterRole, StylePreset, ImageStyle, VideoStyle
from toolboxv2.mods.videoFlow.engine.config import CostTracker

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def cost_tracker():
    return CostTracker()

@pytest.fixture
def mock_isaa():
    mock = MagicMock()
    mock.mini_task_completion = AsyncMock(return_value="improved prompt")
    return mock

@pytest.fixture
def image_generator(logger, cost_tracker, mock_isaa):
    return ImageGenerator(logger, cost_tracker, isaa=mock_isaa)

@pytest.fixture
def story_data():
    return StoryData(
        title="Test Clip Story",
        genre="Action",
        world_desc="A city under siege.",
        characters=[
            Character(name="Rogue", visual_desc="A cunning thief", role=CharacterRole.PROTAGONIST, voice=VoiceType.FEMALE_1)
        ],
        scenes=[
            Scene(
                title="The Heist",
                setting="A high-tech vault",
                narrator="The heist was in motion.",
                dialogue=[
                    DialogueLine(character="Rogue", text="Almost there.", voice=VoiceType.FEMALE_1)
                ]
            )
        ],
        style_preset=StylePreset(image_style=ImageStyle.CYBERPUNK, camera_style=VideoStyle.HOLLYWOOD_BLOCKBUSTER)
    )

@pytest.mark.asyncio
@patch('fal_client.subscribe')
@patch('cv2.VideoCapture')
@patch('cv2.imwrite')
async def test_generate_all_clips_runs(mock_imwrite, mock_videocapture, mock_subscribe, logger, cost_tracker, mock_isaa, image_generator, story_data, tmp_path):
    # Arrange
    mock_subscribe.return_value = {'video': {'url': 'http://fakeurl.com/fake.mp4'}}
    
    # Mock VideoCapture
    mock_cap_instance = mock_videocapture.return_value
    mock_cap_instance.get.return_value = 150 # 5 seconds at 30fps
    mock_cap_instance.read.return_value = (True, MagicMock()) # Return a mock frame

    async def mock_download_video(self, url, path):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            f.write(b"fake video data" * 1000)
        return True

    async def mock_upload_image(self, path, img_dict):
        return f"http://fakeurl.com/{path.name}"

    with patch('toolboxv2.mods.videoFlow.engine.generators.clip_generator.ClipGenerator._download_video', new=mock_download_video), \
         patch('toolboxv2.mods.videoFlow.engine.generators.clip_generator.ClipGenerator._upload_to_fal', new=mock_upload_image):

        clip_gen = ClipGenerator(logger, cost_tracker, isaa=mock_isaa, img_generator=image_generator)
        
        # Create dummy image files
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        cover_path = images_dir / "00_cover.png"
        scene_path = images_dir / "scene_00_perspective_00.png"
        end_path = images_dir / "99_end.png"
        cover_path.touch()
        scene_path.touch()
        end_path.touch()
        
        images = [str(cover_path), str(scene_path), str(end_path)]

        # Act
        result = await clip_gen.generate_all_clips(story_data, images, tmp_path, image_generator)

        # Assert
        assert result is not None
        assert 'cover' in result
        assert 'scene_00_perspective_00' in result
        assert 'end' in result
        assert (tmp_path / "clips" / "00_cover.mp4").exists()
        assert (tmp_path / "clips" / "scene_00_perspective_00.mp4").exists()
        assert (tmp_path / "clips" / "99_end.mp4").exists()
