# toolboxv2/mods/videoFlow/tests/engine/pipeline/test_pipeline.py

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.pipeline.steps import (
    run_story_generation_step,
    run_image_generation_step,
    run_audio_generation_step,
    run_video_generation_step,
    run_pdf_generation_step,
    run_clip_generation_step,
    run_html_generation_step,
)
from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, StylePreset, ImageStyle, VideoStyle, Character, Scene, CharacterRole, VoiceType, DialogueLine

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def cost_tracker():
    return CostTracker()

@pytest.fixture
def project_id():
    return "test_pipeline_project"

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.ProjectManager')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.StoryGenerator')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.ImageGenerator')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.AudioGenerator')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.VideoGenerator')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.PDFGenerator')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.ClipGenerator')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.MultiMediaStoryHTMLGenerator')
async def test_full_pipeline(
    MockHTMLGenerator, MockClipGenerator, MockPDFGenerator, MockVideoGenerator, 
    MockAudioGenerator, MockImageGenerator, MockStoryGenerator, MockProjectManager, 
    logger, cost_tracker, project_id, tmp_path
):
    # Arrange
    prompt = "A full pipeline test."
    
    # Mock ProjectManager
    mock_pm_instance = MockProjectManager.return_value
    
    # Create a temporary directory for the mocked project path
    mock_project_path = tmp_path / project_id
    mock_project_path.mkdir()
    (mock_project_path / "images").mkdir()
    (mock_project_path / "audio").mkdir()
    (mock_project_path / "images" / "dummy_image.png").touch()
    (mock_project_path / "audio" / "dummy_audio.wav").touch()

    mock_pm_instance.get_project_path.return_value = mock_project_path
    mock_pm_instance.load_story_yaml.return_value = StoryData(
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

    # Mock Generators
    MockStoryGenerator.return_value.generate_story = AsyncMock(return_value=mock_pm_instance.load_story_yaml.return_value)
    MockImageGenerator.return_value.generate_all_images = AsyncMock(return_value={'all_images': [Path("/fake/img.png")]})
    MockAudioGenerator.return_value.generate_audio = AsyncMock(return_value=Path("/fake/audio.wav"))
    MockVideoGenerator.return_value.create_video = AsyncMock(return_value=Path("/fake/video.mp4"))
    MockPDFGenerator.return_value.create_complete_pdf = MagicMock(return_value=Path("/fake/doc.pdf"))
    MockClipGenerator.return_value.generate_all_clips = AsyncMock(return_value={'clip': Path("/fake/clip.mp4")})
    MockHTMLGenerator.return_value.create_complete_html_experience = MagicMock(return_value=Path("/fake/index.html"))

    # Act
    await run_story_generation_step(project_id, prompt, cost_tracker)
    await run_image_generation_step(project_id, cost_tracker)
    await run_audio_generation_step(project_id, cost_tracker)
    await run_video_generation_step(project_id, cost_tracker)
    await run_pdf_generation_step(project_id, cost_tracker)
    await run_clip_generation_step(project_id, cost_tracker)
    await run_html_generation_step(project_id, cost_tracker)

    # Assert
    expected_calls = [
        call(project_id, "story_complete"),
        call(project_id, "images_complete"),
        call(project_id, "audio_complete"),
        call(project_id, "video_complete"),
        call(project_id, "pdf_complete"),
        call(project_id, "clips_complete"),
        call(project_id, "html_complete"),
    ]
    mock_pm_instance.update_project_status.assert_has_calls(expected_calls, any_order=False)
    assert mock_pm_instance.update_project_status.call_count == len(expected_calls)

