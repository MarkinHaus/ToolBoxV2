# toolboxv2/mods/videoFlow/tests/engine/generators/test_story_generator.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock

from toolboxv2.mods.videoFlow.engine.generators.story_generator import StoryGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, StylePreset, ImageStyle, VideoStyle, Character, Scene, CharacterRole, VoiceType

@pytest.fixture
def mock_isaa():
    mock = MagicMock()
    mock.mini_task_completion_format = AsyncMock()
    return mock

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_generate_story_success(mock_isaa, logger):
    # Arrange
    story_gen = StoryGenerator(mock_isaa, logger)
    prompt = "A space opera about a lonely robot."
    
    # Mock the return value of the isaa call
    mock_result = {
        "title": "Stardust Echoes",
        "genre": "Sci-Fi",
        "characters": [
            {"name": "Unit 734", "visual_desc": "A sleek, chrome robot with a single blue optic.", "role": "protagonist", "voice": "male_1"}
        ],
        "world_desc": "A galaxy filled with nebulae and ancient alien ruins.",
        "scenes": [
            {"title": "The Discovery", "setting": "An abandoned starship.", "narrator": "Unit 734 found a strange artifact."}
        ],
        "style_preset": {
            "image_style": "cyberpunk",
            "camera_style": "Hollywood Blockbuster"
        }
    }
    mock_isaa.mini_task_completion_format.return_value = mock_result

    # Act
    story_data = await story_gen.generate_story(prompt)

    # Assert
    assert story_data is not None
    assert isinstance(story_data, StoryData)
    assert story_data.title == "Stardust Echoes"
    assert story_data.characters[0].name == "Unit 734"
    mock_isaa.mini_task_completion_format.assert_called_once()

@pytest.mark.asyncio
async def test_generate_story_failure(mock_isaa, logger):
    # Arrange
    story_gen = StoryGenerator(mock_isaa, logger)
    prompt = "A fantasy story."
    
    # Mock the isaa call to return None
    mock_isaa.mini_task_completion_format.return_value = None

    # Act
    story_data = await story_gen.generate_story(prompt)

    # Assert
    assert story_data is None
    mock_isaa.mini_task_completion_format.assert_called_once()

@pytest.mark.asyncio
async def test_generate_story_exception(mock_isaa, logger):
    # Arrange
    story_gen = StoryGenerator(mock_isaa, logger)
    prompt = "A horror story."
    
    # Mock the isaa call to raise an exception
    mock_isaa.mini_task_completion_format.side_effect = Exception("AI model failed")

    # Act
    story_data = await story_gen.generate_story(prompt)

    # Assert
    assert story_data is None
    mock_isaa.mini_task_completion_format.assert_called_once()
