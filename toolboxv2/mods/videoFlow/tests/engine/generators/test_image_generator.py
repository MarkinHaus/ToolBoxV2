# toolboxv2/mods/videoFlow/tests/engine/generators/test_image_generator.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.generators.image_generator import ImageGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, StylePreset, ImageStyle, VideoStyle, Character, Scene, CharacterRole, VoiceType, DialogueLine
from toolboxv2.mods.videoFlow.engine.config import CostTracker

@pytest.fixture
def mock_isaa():
    mock = MagicMock()
    mock.mini_task_completion = AsyncMock(return_value="improved prompt")
    return mock

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def cost_tracker():
    return CostTracker()

@pytest.fixture
def story_data():
    return StoryData(
        title="Test Story",
        genre="Fantasy",
        world_desc="A world of dragons and magic.",
        characters=[
            Character(name="Hero", visual_desc="A brave knight", role=CharacterRole.PROTAGONIST, voice=VoiceType.MALE_1)
        ],
        scenes=[
            Scene(title="The Quest Begins", setting="A dark forest", narrator="The hero started his journey.", dialogue=[DialogueLine(character="Hero", text="I must go on.", voice=VoiceType.MALE_1)])
        ],
        style_preset=StylePreset(image_style=ImageStyle.FANTASY, camera_style=VideoStyle.INDIE_FILM)
    )

@pytest.mark.asyncio
@patch('fal_client.subscribe', return_value={'images': [{'url': 'http://fakeurl.com/fake.png'}]})
async def test_generate_all_images_runs(mock_subscribe, mock_isaa, logger, cost_tracker, story_data, tmp_path):
    # Arrange
    async def mock_download(self, url, path):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            f.write(b"fake image data")
        return True

    async def mock_upload(self, path):
        return f"http://fakeurl.com/{path.name}"

    with patch('toolboxv2.mods.videoFlow.engine.generators.image_generator.ImageGenerator._download_image', new=mock_download), \
         patch('toolboxv2.mods.videoFlow.engine.generators.image_generator.ImageGenerator._upload_to_fal', new=mock_upload):

        image_gen = ImageGenerator(logger, cost_tracker, isaa=mock_isaa)
        project_dir = tmp_path
        
        # Act
        result = await image_gen.generate_all_images(story_data, project_dir)

        # Assert
        assert result is not None
        assert 'all_images' in result
        assert 'character_refs' in result
        
        images_dir = project_dir / "images"
        assert (images_dir / "00_cover.png").exists()
        assert (images_dir / "01_world_00.png").exists()
        assert (images_dir / "00_char_hero.png").exists()
        assert (images_dir / "scene_00_base_environment.png").exists()
        assert (images_dir / "scene_00_perspective_00.png").exists()
        assert (images_dir / "99_end.png").exists()
