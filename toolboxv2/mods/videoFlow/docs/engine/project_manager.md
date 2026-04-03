# ProjectManager Functions

The `ProjectManager` is responsible for managing the state of video projects on the file system.

## Functions

### `create_project(user_id: str) -> dict`
- **Description:** Creates a new, empty project for a given user. It sets up the necessary directory structure and initializes a state file (e.g., `project.json`).
- **Parameters:**
  - `user_id` (str): The ID of the user creating the project.
- **Returns:** A dictionary containing the new `project_id` and other initial project metadata.

### `get_project_state(project_id: str) -> dict`
- **Description:** Retrieves the entire state of a project from its state file.
- **Parameters:**
  - `project_id` (str): The ID of the project to retrieve.
- **Returns:** A dictionary representing the current state of the project.

### `update_project_data(project_id: str, data: dict) -> bool`
- **Description:** Updates the project's state file with new data. This is used to save changes to the story, timeline, or other metadata.
- **Parameters:**
  - `project_id` (str): The ID of the project to update.
  - `data` (dict): The new data to merge into the project's state.
- **Returns:** `True` if the update was successful, `False` otherwise.

### `save_asset(project_id: str, asset_data: bytes, asset_name: str) -> str`
- **Description:** Saves a generated asset (like an image, audio clip, or video segment) to the project's asset directory.
- **Parameters:**
  - `project_id` (str): The ID of the project.
  - `asset_data` (bytes): The binary data of the asset.
  - `asset_name` (str): The desired filename for the asset.
- **Returns:** The path or a unique asset ID for the newly saved asset.
