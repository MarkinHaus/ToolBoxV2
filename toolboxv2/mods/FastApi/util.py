import os
from pathlib import Path

from pydantic import BaseModel
from starlette.responses import FileResponse


class PostRequest(BaseModel):
    token: str
    data: dict

def serve_app_func(path: str, prefix: str = os.getcwd() + "/dist/"):
    # Default to 'index.html' if no specific path is given
    if not path or '.' not in path:  # No file extension, assume SPA route
        path = "index.html"

    # Full path to the requested file
    request_file_path = Path(prefix) / path

    # MIME types dictionary
    mime_types = {
        '.js': 'application/javascript',
        '.html': 'text/html',
        '.css': 'text/css',
    }

    # Determine MIME type based on file extension, default to 'text/html'
    content_type = mime_types.get(request_file_path.suffix, 'text/html')

    # Serve the requested file if it exists, otherwise fallback to index.html for SPA
    if request_file_path.exists():
        return FileResponse(request_file_path, media_type=content_type)

    # Fallback to a 404 page if the file does not exist
    return FileResponse(os.path.join(os.getcwd(), "dist", "web/assets/404.html"), media_type="text/html")
