# toolboxv2/mods/videoFlow/tests/engine/generators/test_pdf_generator.py

import pytest
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

from toolboxv2.mods.videoFlow.engine.generators.pdf_generator import PDFGenerator
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, Scene, DialogueLine, VoiceType, Character, CharacterRole, StylePreset, ImageStyle, VideoStyle

# 1x1 transparent PNG
TINY_PNG = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

@pytest.fixture
def logger():
    return logging.getLogger(__name__)

@pytest.fixture
def story_data():
    return StoryData(
        title="Test PDF Story",
        genre="Fantasy",
        world_desc="A world of wonders.",
        characters=[
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

@patch('toolboxv2.mods.videoFlow.engine.generators.pdf_generator.Image')
@patch('toolboxv2.mods.videoFlow.engine.generators.pdf_generator.SimpleDocTemplate')
def test_create_complete_pdf_runs(mock_doc_template, mock_image, logger, story_data, tmp_path):
    # Arrange
    pdf_gen = PDFGenerator(logger)
    
    # Create dummy image files
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "00_cover.png").touch()
    (images_dir / "scene_00_perspective_00.png").touch()
    (images_dir / "99_end.png").touch()

    images_dict = {'all_images_complete': list(images_dir.glob("*.png"))}

    # Mock the build method to create the output file
    mock_build_method = MagicMock()
    def mock_build(flowables):
        pdf_path = tmp_path / "pdf" / "Test_PDF_Story_complete_full.pdf"
        pdf_path.parent.mkdir(exist_ok=True)
        pdf_path.touch()

    mock_build_method.side_effect = mock_build
    mock_doc_template.return_value.build = mock_build_method

    # Act
    result_path = pdf_gen.create_complete_pdf(story_data, images_dict, tmp_path)

    # Assert
    assert result_path is not None
    assert result_path.exists()
    assert result_path.name == "Test_PDF_Story_complete_full.pdf"
    mock_doc_template.assert_called_once()
    mock_build_method.assert_called_once()
