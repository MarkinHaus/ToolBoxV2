# API Endpoint: `POST /api/videoFlow/register`

## Description

Registers a new user in the system.

## Method

`POST`

## URL

`/api/videoFlow/register`

## Body

```json
{
  "username": "string",
  "password": "string"
}
```

## Responses

- **200 OK:** Registration successful.
  ```json
  {
    "status": "success",
    "message": "User registered successfully."
  }
  ```
- **400 Bad Request:** Invalid input or user already exists.
  ```json
  {
    "status": "error",
    "message": "Username already exists."
  }
  ```
