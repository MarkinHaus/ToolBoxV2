# Character Model

The `Character` model is a Pydantic model that represents a character in the story.

## Fields

- `name` (str): The name of the character.
- `visual_desc` (str): A concise visual description of the character, used for generating reference images.
- `role` (CharacterRole): The role of the character in the story (e.g., protagonist, antagonist).
- `voice` (VoiceType): The voice type assigned to the character for text-to-speech generation.
