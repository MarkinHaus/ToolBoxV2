# API Endpoint: `POST /api/videoFlow/create_project`

## Description

Creates a new, empty video project for the currently logged-in user.

## Method

`POST`

## URL

`/api/videoFlow/create_project`

## Body

```json
{
  "projectName": "My Awesome Video"
}
```

## Responses

- **201 Created:** Project created successfully.
  ```json
  {
    "status": "success",
    "projectId": "unique-project-id-123"
  }
  ```
- **401 Unauthorized:** User is not logged in.
