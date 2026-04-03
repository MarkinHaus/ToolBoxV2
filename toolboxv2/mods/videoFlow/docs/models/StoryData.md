# StoryData Model

The `StoryData` model is a Pydantic model that represents the entire story structure.

## Fields

- `title` (str): The title of the story.
- `genre` (str): The genre of the story.
- `characters` (List[Character]): A list of `Character` models present in the story.
- `world_desc` (str): A concise description of the world or setting.
- `scenes` (List[Scene]): A list of `Scene` models that make up the story.
- `style_preset` (StylePreset): A `StylePreset` model that defines the visual style for the entire story.
