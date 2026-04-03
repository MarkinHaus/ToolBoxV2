# Scene Model

The `Scene` model is a Pydantic model that represents a single scene in the story.

## Fields

- `title` (str): The title of the scene.
- `setting` (str): A brief description of the scene's setting.
- `narrator` (str): A 2-3 sentence narration for the scene.
- `dialogue` (List[DialogueLine]): A list of `DialogueLine` models representing the dialogue in the scene.
- `poses` (List[str]): A list of character poses in this scene.
- `duration` (float): The estimated duration of the scene in seconds. Default is 8.0.
