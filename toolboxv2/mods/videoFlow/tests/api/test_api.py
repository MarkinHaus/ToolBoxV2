from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from toolboxv2.flows.story_generator import StoryData, ImageStyle, StylePreset, VideoStyle


@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.api.auth.ProjectManager')
@patch('toolboxv2.mods.videoFlow.api.auth.Code')
@patch('toolboxv2.mods.videoFlow.api.auth.json')
@patch('toolboxv2.mods.videoFlow.api.auth.uuid')
async def test_register_api(mock_uuid, mock_json, mock_code, mock_pm, mock_request_data, tmp_path, setup_api_endpoints):
    # Arrange
    mock_request_data.get.side_effect = lambda key, default=None: {
        "username": "testuser",
        "password": "testpass",
    }.get(key, default)
    mock_pm.return_value.base_dir = tmp_path
    mock_json.loads.return_value = {}
    mock_json.dumps.return_value = "{}"
    mock_uuid.uuid4.return_value = "mock_user_id"

    # Act
    register_func = setup_api_endpoints["/register"]["POST"]
    response = await register_func(mock_request_data)

    # Assert
    assert response["status_code"] == 200
    assert response["status"] == "success"
    assert response["message"] == "User registered successfully."

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.api.auth.ProjectManager')
@patch('toolboxv2.mods.videoFlow.api.auth.Code')
@patch('toolboxv2.mods.videoFlow.api.auth.json')
async def test_login_api(mock_json, mock_code, mock_pm, mock_request_data, tmp_path, setup_api_endpoints):
    # Arrange
    mock_request_data.get.side_effect = lambda key, default=None: {
        "username": "testuser",
        "password": "testpass",
    }.get(key, default)
    mock_pm.return_value.base_dir = tmp_path
    mock_json.loads.return_value = {"testuser": {"password": "hashed_password", "user_id": "test_user_id"}}
    mock_json.dumps.return_value = "{}"

    # Act
    login_func = setup_api_endpoints["/login"]["POST"]
    response = await login_func(mock_request_data)

    # Assert
    assert response["status_code"] == 200
    assert response["status"] == "success"
    assert response["message"] == "Login successful."
    assert response["user_id"] == "test_user_id"

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.api.projects.ProjectManager')
async def test_create_project_api(mock_pm, mock_request_data, setup_api_endpoints):
    # Arrange
    mock_request_data.get.side_effect = lambda key, default=None: {
        "user_id": "test_user_id",
        "projectName": "My New Project",
    }.get(key, default)
    mock_pm.return_value.create_project.return_value = Path("/fake/base_dir/new_project_id")

    # Act
    create_project_func = setup_api_endpoints["/create_project"]["POST"]
    response = await create_project_func(mock_request_data)

    # Assert
    assert response["status_code"] == 201
    assert response["status"] == "success"
    assert response["projectId"] == "new_project_id"

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.api.projects.ProjectManager')
async def test_get_project_status_api(mock_pm, mock_request_data, setup_api_endpoints):
    # Arrange
    mock_request_data.get.side_effect = lambda key, default=None: {
        "user_id": "test_user_id",
    }.get(key, default)
    mock_pm.return_value.get_project_path.return_value = MagicMock(spec=Path)
    mock_pm.return_value.get_project_path.return_value.exists.return_value = True # Ensure project exists
    mock_pm.return_value.get_project_path.return_value.name = "test_project_id"
    mock_pm.return_value.check_project_status.return_value = {'images': 1, 'audio': 1, 'video': 0, 'pdf': 0, 'clips': 0, 'story_yaml': True, 'metadata': True, 'completion_percentage': 50}
    mock_pm.return_value.load_story_yaml.return_value = StoryData(
        title="Loaded Story", genre="Sci-Fi", world_desc="A loaded world.", characters=[], scenes=[],
        style_preset=StylePreset(image_style=ImageStyle.REALISTIC, camera_style=VideoStyle.HOLLYWOOD_BLOCKBUSTER)
    )

    # Act
    get_status_func = setup_api_endpoints["/project_status/{project_id}"]["GET"]
    response = await get_status_func(mock_request_data, "test_project_id")

    # Assert
    assert response["status_code"] == 200
    assert response["projectId"] == "test_project_id"
    assert response["status"]["completion_percentage"] == 50
    assert response["storyData"]["title"] == "Loaded Story"

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.api.update_story.ProjectManager')
async def test_update_story_api(mock_pm, mock_request_data, setup_api_endpoints):
    # Arrange
    mock_request_data.get.side_effect = lambda key, default=None: {
        "user_id": "test_user_id",
        "storyData": {
            "title": "API Test Story",
            "genre": "Adventure",
            "characters": [],
            "world_desc": "A test world.",
            "scenes": [],
            "style_preset": {
                "image_style": "realistic",
                "camera_style": "hollywood_blockbuster"
            }
        },
    }.get(key, default)
    mock_pm.return_value.get_project_path.return_value = MagicMock(spec=Path)
    mock_pm.return_value.get_project_path.return_value.exists.return_value = True # Ensure project exists
    mock_pm.return_value.save_story_yaml = MagicMock()

    # Act
    update_story_func = setup_api_endpoints["/update_story/{project_id}"]["PUT"]
    response = await update_story_func(mock_request_data, "test_project_id")

    # Assert
    assert response["status_code"] == 200
    assert response["status"] == "success"
    assert response["message"] == "Story updated successfully."
    mock_pm.return_value.save_story_yaml.assert_called_once()

@pytest.mark.asyncio
@patch('toolboxv2.mods.videoFlow.api.generation.ProjectManager')
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_story_generation_step', new_callable=AsyncMock)
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_image_generation_step', new_callable=AsyncMock)
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_audio_generation_step', new_callable=AsyncMock)
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_video_generation_step', new_callable=AsyncMock)
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_pdf_generation_step', new_callable=AsyncMock)
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_clip_generation_step', new_callable=AsyncMock)
@patch('toolboxv2.mods.videoFlow.engine.pipeline.steps.run_html_generation_step', new_callable=AsyncMock)
async def test_run_step_api(
    mock_run_html,
    mock_run_clip,
    mock_run_pdf,
    mock_run_video,
    mock_run_audio,
    mock_run_image,
    mock_run_story,
    mock_pm,
    mock_request_data,
    setup_api_endpoints
):
    # Arrange
    mock_request_data.get.side_effect = lambda key, default=None: {
        "user_id": "test_user_id",
        "prompt": "A story about a hero.",
        "use_elevenlabs": False
    }.get(key, default)
    mock_pm.return_value.get_project_path.return_value = MagicMock(spec=Path)
    mock_pm.return_value.get_project_path.return_value.exists.return_value = True # Ensure project exists

    run_step_func = setup_api_endpoints["/run_step/{project_id}/{step_name}"]["POST"]

    # Test story step
    response = await run_step_func(mock_request_data, "test_project_id", "story")
    assert response["status_code"] == 202
    mock_run_story.assert_called_once()

    # Test image step
    mock_run_story.reset_mock()
    response = await run_step_func(mock_request_data, "test_project_id", "images")
    assert response["status_code"] == 202
    mock_run_image.assert_called_once()

    # Test audio step
    mock_run_image.reset_mock()
    response = await run_step_func(mock_request_data, "test_project_id", "audio")
    assert response["status_code"] == 202
    mock_run_audio.assert_called_once()

    # Test video step
    mock_run_audio.reset_mock()
    response = await run_step_func(mock_request_data, "test_project_id", "video")
    assert response["status_code"] == 202
    mock_run_video.assert_called_once()

    # Test pdf step
    mock_run_video.reset_mock()
    response = await run_step_func(mock_request_data, "test_project_id", "pdf")
    assert response["status_code"] == 202
    mock_run_pdf.assert_called_once()

    # Test clips step
    mock_run_pdf.reset_mock()
    response = await run_step_func(mock_request_data, "test_project_id", "clips")
    assert response["status_code"] == 202
    mock_run_clip.assert_called_once()

    # Test html step
    mock_run_clip.reset_mock()
    response = await run_step_func(mock_request_data, "test_project_id", "html")
    assert response["status_code"] == 202
    mock_run_html.assert_called_once()

    # Test invalid step
    response = await run_step_func(mock_request_data, "test_project_id", "invalid_step")
    assert response["status_code"] == 400
    assert response["status"] == "error"
