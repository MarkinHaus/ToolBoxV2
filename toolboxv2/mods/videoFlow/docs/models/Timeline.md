# Timeline JSON Object

The Timeline object represents the interactive timeline of a video project. It is a JSON structure that contains all the clips, audio tracks, and effects, allowing for manipulation in the frontend editor.

## Structure

```json
{
  "projectId": "string",
  "version": "integer",
  "tracks": [
    {
      "id": "string",
      "type": "video",
      "clips": [
        {
          "id": "string",
          "assetId": "string",
          "start": 0,
          "duration": 8.5,
          "effects": [
            {
              "type": "zoom",
              "start": 0,
              "end": 1.2
            }
          ]
        }
      ]
    },
    {
      "id": "string",
      "type": "audio",
      "clips": [
        {
          "id": "string",
          "assetId": "string",
          "start": 0.5,
          "duration": 8
        }
      ]
    }
  ]
}
```

## Fields

- `projectId` (string): The unique identifier for the project this timeline belongs to.
- `version` (integer): The version number of the timeline, incremented on each save.
- `tracks` (array): An array of track objects.
  - `id` (string): A unique identifier for the track.
  - `type` (string): The type of the track, either `"video"` or `"audio"`.
  - `clips` (array): An array of clip objects within the track.
    - `id` (string): A unique identifier for the clip.
    - `assetId` (string): The ID of the asset (e.g., image, video, audio file) this clip uses. This ID is managed by the `ProjectManager`.
    - `start` (float): The start time of the clip on the timeline, in seconds.
    - `duration` (float): The duration of the clip, in seconds.
    - `effects` (array, optional): An array of effects applied to the clip.
