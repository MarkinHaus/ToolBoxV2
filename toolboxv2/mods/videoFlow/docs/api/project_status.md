# API Endpoint: `GET /api/videoFlow/project_status/{project_id}`

## Description

Retrieves the current status and state of a specific project.

## Method

`GET`

## URL

`/api/videoFlow/project_status/{project_id}`

## URL Parameters

- `project_id` (string): The ID of the project.

## Responses

- **200 OK:** Returns the full project state.
  ```json
  {
    "projectId": "unique-project-id-123",
    "status": "images_complete",
    "storyData": { ... },
    "assets": {
      "images": [ ... ],
      "audio": null
    }
  }
  ```
- **401 Unauthorized:** User is not logged in or does not own the project.
- **404 Not Found:** Project not found.
