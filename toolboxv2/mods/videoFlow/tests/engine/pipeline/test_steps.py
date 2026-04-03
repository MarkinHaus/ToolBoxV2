# toolboxv2/mods/videoFlow/tests/engine/pipeline/test_steps.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.pipeline.steps import run_story_generation_step
from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, StylePreset, ImageStyle, VideoStyle, Character, Scene, CharacterRole, VoiceType, DialogueLine

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def cost_tracker():
    return CostTracker()

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.ProjectManager')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.StoryGenerator')
async def test_run_story_generation_step(MockStoryGenerator, MockProjectManager, logger, cost_tracker):
    # Arrange
    project_id = "test_project"
    prompt = "A test prompt"
    
    mock_pm_instance = MockProjectManager.return_value
    mock_pm_instance.get_project_path.return_value = Path("/fake/path")
    
    mock_story_generator_instance = MockStoryGenerator.return_value
    mock_story_generator_instance.generate_story = AsyncMock(return_value=StoryData(
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
    ))

    # Act
    story = await run_story_generation_step(project_id, prompt, cost_tracker)

    # Assert
    assert story is not None
    assert story.title == "Test Story"
    mock_pm_instance.get_project_path.assert_called_once_with(project_id)
    mock_story_generator_instance.generate_story.assert_called_once_with(prompt)
    mock_pm_instance.save_story_yaml.assert_called_once()
    mock_pm_instance.update_project_status.assert_called_once_with(project_id, "story_complete")
