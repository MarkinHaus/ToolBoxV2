# Pipeline Step Functions

The pipeline functions are responsible for executing the individual steps of the video generation process. They are designed to be run as background tasks.

## Functions

### `run_story_generation_step(project_id: str)`
- **Description:** Generates the story script based on a user prompt.
- **Process:**
  1. Loads the project state using `ProjectManager.get_project_state`.
  2. Retrieves the initial user prompt from the project state.
  3. Calls the `StoryGenerator` to create the `StoryData`.
  4. Updates the project state with the generated `StoryData` using `ProjectManager.update_project_data`.
  5. Sets the project status to `story_complete`.

### `run_image_generation_step(project_id: str)`
- **Description:** Generates all images required for the story.
- **Process:**
  1. Loads the project state.
  2. Retrieves the `StoryData`.
  3. Calls the `ImageGenerator` to generate all character, world, and scene images.
  4. Saves each generated image as an asset using `ProjectManager.save_asset`.
  5. Updates the project state with the paths/IDs of the generated image assets.
  6. Sets the project status to `images_complete`.

### `run_audio_generation_step(project_id: str)`
- **Description:** Generates the audio narration and dialogue.
- **Process:**
  1. Loads the project state.
  2. Retrieves the `StoryData`.
  3. Calls the `AudioGenerator` to create the full audio track.
  4. Saves the final audio track as an asset.
  5. Updates the project state with the audio asset path/ID.
  6. Sets the project status to `audio_complete`.

### `run_video_generation_step(project_id: str)`
- **Description:** Combines the generated images and audio into the final video.
- **Process:**
  1. Loads the project state.
  2. Retrieves the image and audio asset information.
  3. Calls the `VideoGenerator` to create the final video file.
  4. Saves the final video as an asset.
  5. Updates the project state with the final video path/ID.
  6. Sets the project status to `video_complete`.
