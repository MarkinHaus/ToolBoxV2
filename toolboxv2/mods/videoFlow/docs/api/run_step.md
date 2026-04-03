# API Endpoint: `POST /api/videoFlow/run_step/{project_id}/{step_name}`

## Description

Triggers a specific generation step in the pipeline for a project. This is an asynchronous operation.

## Method

`POST`

## URL

`/api/videoFlow/run_step/{project_id}/{step_name}`

## URL Parameters

- `project_id` (string): The ID of the project.
- `step_name` (string): The name of the step to run (e.g., `story`, `images`, `audio`, `video`).

## Responses

- **202 Accepted:** The step has been accepted and is running in the background.
  ```json
  {
    "status": "accepted",
    "message": "The 'story' generation step has started."
  }
  ```
- **400 Bad Request:** Invalid step name or prerequisites for the step are not met.
- **401 Unauthorized:** User is not logged in or does not own the project.
- **402 Payment Required:** User does not have enough credits.
- **404 Not Found:** Project not found.
