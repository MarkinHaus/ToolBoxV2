# toolboxv2/mods/videoFlow/tests/engine/test_models.py

import pytest
from pydantic import ValidationError
from toolboxv2.mods.videoFlow.engine.models.base_models import (
    StoryData,
    Scene,
    Character,
    DialogueLine,
    StylePreset,
    VoiceType,
    CharacterRole,
    ImageStyle,
    VideoStyle
)

def test_character_instantiation():
    char_data = {
        "name": "Gandalf",
        "visual_desc": "An old wizard with a long white beard.",
        "role": CharacterRole.MYSTERIOUS,
        "voice": VoiceType.NARRATOR
    }
    char = Character(**char_data)
    assert char.name == "Gandalf"
    assert char.role == CharacterRole.MYSTERIOUS

def test_scene_instantiation():
    scene_data = {
        "title": "The Bridge of Khazad-dûm",
        "setting": "A narrow stone bridge in the depths of Moria.",
        "narrator": "The fellowship faced the Balrog.",
        "dialogue": [
            {
                "character": "Gandalf",
                "text": "You shall not pass!",
                "voice": VoiceType.NARRATOR
            }
        ]
    }
    scene = Scene(**scene_data)
    assert scene.title == "The Bridge of Khazad-dûm"
    assert len(scene.dialogue) == 1
    assert scene.dialogue[0].character == "Gandalf"

def test_story_data_instantiation():
    story_data = {
        "title": "The Lord of the Rings",
        "genre": "Fantasy",
        "world_desc": "Middle-earth, a land of elves, dwarves, and hobbits.",
        "characters": [
            {
                "name": "Frodo",
                "visual_desc": "A small hobbit with curly brown hair.",
                "role": CharacterRole.PROTAGONIST,
                "voice": VoiceType.MALE_1
            }
        ],
        "scenes": [
            {
                "title": "A New Journey",
                "setting": "The Shire",
                "narrator": "An adventure begins."
            }
        ]
    }
    story = StoryData(**story_data)
    assert story.title == "The Lord of the Rings"
    assert len(story.characters) == 1
    assert story.style_preset.image_style == ImageStyle.DIGITAL_ART

def test_invalid_character_role():
    with pytest.raises(ValidationError):
        Character(
            name="Invalid",
            visual_desc="desc",
            role="not_a_role",
            voice=VoiceType.MALE_1
        )

def test_style_preset_defaults():
    preset = StylePreset(image_style=ImageStyle.ANIME, camera_style=VideoStyle.ANIME)
    assert preset.art_style == "realistic 8k photography"
    assert preset.color_palette == "vibrant colors"
