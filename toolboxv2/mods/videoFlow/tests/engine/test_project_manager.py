# toolboxv2/mods/videoFlow/tests/engine/test_project_manager.py

import pytest
import logging
from pathlib import Path
from unittest.mock import patch

from toolboxv2.mods.videoFlow.engine.project_manager import ProjectManager
from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, Scene, DialogueLine, VoiceType, Character, CharacterRole, StylePreset, ImageStyle, VideoStyle

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def cost_tracker():
    return CostTracker()

@pytest.fixture
def story_data():
    return StoryData(
        title="Test Project Story",
        genre="Mystery",
        world_desc="A city full of secrets.",
        characters=[
            Character(name="Detective", visual_desc="A sharp detective", role=CharacterRole.PROTAGONIST, voice=VoiceType.MALE_1)
        ],
        scenes=[
            Scene(
                title="The Crime Scene",
                setting="A dimly lit room",
                narrator="The detective arrived at the scene.",
                dialogue=[
                    DialogueLine(character="Detective", text="Something is not right here.", voice=VoiceType.MALE_1)
                ]
            )
        ],
        style_preset=StylePreset(image_style=ImageStyle.NOIR, camera_style=VideoStyle.FILM_NOIR)
    )

def test_create_project(cost_tracker, tmp_path):
    # Arrange
    with patch('toolboxv2.mods.videoFlow.engine.project_manager.Config.BASE_OUTPUT_DIR', tmp_path):
        pm = ProjectManager(cost_tracker)
        prompt = "A detective story."

        # Act
        project_dir = pm.create_project(prompt)

        # Assert
        assert project_dir.exists()
        assert (project_dir / "images").exists()
        assert (project_dir / "audio").exists()
        assert (project_dir / "video").exists()
        assert (project_dir / "pdf").exists()
        assert (project_dir / "transitions").exists()
        assert len(pm.projects_index) == 1

def test_save_and_load_story_yaml(cost_tracker, story_data, tmp_path):
    # Arrange
    with patch('toolboxv2.mods.videoFlow.engine.project_manager.Config.BASE_OUTPUT_DIR', tmp_path):
        pm = ProjectManager(cost_tracker)
        project_dir = pm.create_project("test prompt")

        # Act
        pm.save_story_yaml(story_data, project_dir)
        loaded_story = pm.load_story_yaml(project_dir)

        # Assert
        assert (project_dir / "story.yaml").exists()
        assert loaded_story is not None
        assert loaded_story.title == story_data.title
        assert loaded_story.scenes[0].title == story_data.scenes[0].title

def test_check_project_status(cost_tracker, tmp_path):
    # Arrange
    with patch('toolboxv2.mods.videoFlow.engine.project_manager.Config.BASE_OUTPUT_DIR', tmp_path):
        pm = ProjectManager(cost_tracker)
        project_dir = pm.create_project("test prompt")
        (project_dir / "images" / "img1.png").touch()
        (project_dir / "audio" / "audio.wav").touch()

        # Act
        status = pm.check_project_status(project_dir)

        # Assert
        assert status['images'] == 1
        assert status['audio'] == 1
        assert status['video'] == 0
        assert status['completion_percentage'] > 30 # 2/6 = 33.3%
