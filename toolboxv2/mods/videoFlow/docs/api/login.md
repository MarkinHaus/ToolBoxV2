# API Endpoint: `POST /api/videoFlow/login`

## Description

Logs a user in and returns a session token.

## Method

`POST`

## URL

`/api/videoFlow/login`

## Body

```json
{
  "username": "string",
  "password": "string"
}
```

## Responses

- **200 OK:** Login successful. The `toolboxv2` framework will handle the session and return a JWT in the cookies.
  ```json
  {
    "status": "success",
    "message": "Login successful."
  }
  ```
- **401 Unauthorized:** Invalid credentials.
  ```json
  {
    "status": "error",
    "message": "Invalid username or password."
  }
  ```
