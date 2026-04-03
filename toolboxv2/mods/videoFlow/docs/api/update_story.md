# API Endpoint: `PUT /api/videoFlow/update_story/{project_id}`

## Description

Updates the story data for a specific project.

## Method

`PUT`

## URL

`/api/videoFlow/update_story/{project_id}`

## URL Parameters

- `project_id` (string): The ID of the project to update.

## Body

The body should be a `StoryData` JSON object.

```json
{
  "title": "My Updated Story",
  "genre": "Sci-Fi",
  "characters": [],
  "world_desc": "A new world.",
  "scenes": [],
  "style_preset": {}
}
```

## Responses

- **200 OK:** Story updated successfully.
  ```json
  {
    "status": "success",
    "message": "Story updated."
  }
  ```
- **401 Unauthorized:** User is not logged in or does not own the project.
- **404 Not Found:** Project not found.
