# toolboxv2/mods/videoFlow/tests/engine/generators/test_html_generator.py

import pytest
import logging
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.generators.html_generator import MultiMediaStoryHTMLGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, Scene, DialogueLine, VoiceType, Character, CharacterRole, StylePreset, ImageStyle, VideoStyle

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def story_data():
    return StoryData(
        title="Test HTML Story",
        genre="Comedy",
        world_desc="A funny world.",
        characters=[
            Character(name="Jester", visual_desc="A funny guy", role=CharacterRole.PROTAGONIST, voice=VoiceType.MALE_3)
        ],
        scenes=[
            Scene(
                title="The Joke",
                setting="A stage",
                narrator="He told a joke.",
                dialogue=[
                    DialogueLine(character="Jester", text="Why did the chicken cross the road?", voice=VoiceType.MALE_3)
                ]
            )
        ],
        style_preset=StylePreset(image_style=ImageStyle.CARTOON, camera_style=VideoStyle.MUSIC_VIDEO)
    )

def test_create_complete_html_experience_runs(logger, story_data, tmp_path):
    # Arrange
    html_gen = MultiMediaStoryHTMLGenerator(logger)
    
    # Create dummy media files
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "00_cover.png").touch()
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "complete.wav").touch()
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "final.mp4").touch()

    # Act
    result_path = html_gen.create_complete_html_experience(story_data, tmp_path)

    # Assert
    assert result_path is not None
    assert result_path.exists()
    assert result_path.name == "Test_HTML_Story_complete_experience.html"
    
    # Check if media files were copied
    assert (tmp_path / "html" / "images" / "00_cover.png").exists()
    assert (tmp_path / "html" / "audio" / "complete.wav").exists()
    assert (tmp_path / "html" / "video" / "final.mp4").exists()
