# TU ISIS Adapter

A Python adapter for accessing TU Berlin's ISIS (ILIAS) platform with Shibboleth SSO authentication, course management, search, and download capabilities.

## Features

- **Authentication**: Shibboleth SSO login with session persistence
- **Course Management**: List, search, and retrieve detailed course information
- **Channel/Forum**: Access communication channels and get updates
- **Search**: Advanced course search with filters (semester, department, instructor)
- **Downloads**: Download course materials and channel attachments
- **State Management**: Save and restore sessions for password-less access

## Installation

```bash
pip install -r requirements.txt
playwright install chromium  # or firefox, webkit
```

## Quick Start

```python
import asyncio
from tu_isis_adapter import TUIsisAdapter, quick_login

async def main():
    # Method 1: Using context manager
    async with TUIsisAdapter(headless=False) as adapter:
        await adapter.create_context()
        await adapter.login("your_username", "your_password")
        
        courses = await adapter.get_courses()
        print(f"Found {len(courses)} courses")
    
    # Method 2: Quick login
    adapter = await quick_login("username", "password", headless=True)
    try:
        results = await adapter.search_courses("Machine Learning")
    finally:
        await adapter.teardown()

asyncio.run(main())
```

## API Reference

### TUIsisAdapter

#### Initialization

```python
adapter = TUIsisAdapter(
    browser_type="chromium",  # chromium, firefox, webkit
    headless=False,
    state_dir="tu_isis_states",
    downloads_dir="tu_isis_downloads",
    log_level=logging.INFO
)
```

#### Authentication

```python
# Login with credentials (first time)
await adapter.login("username", "password")

# Login using saved session
await adapter.login()

# Save current session state
await adapter.save_state("my_session")
```

#### Courses

```python
# Get all accessible courses
courses = await adapter.get_courses()

# Get detailed course information
details = await adapter.get_course_details(course_id)

# Search for courses
results = await adapter.search_courses("query")

# Search with filters
results = await adapter.search_courses("Data Science", {
    "semester": "2024Wi",
    "department": "Informatik"
})
```

#### Channels

```python
# Get all channels/forums
channels = await adapter.get_channels()

# Get channel updates
from datetime import datetime, timedelta
posts = await adapter.get_channel_updates(channel_id)

# Get only recent updates
week_ago = datetime.now() - timedelta(days=7)
recent = await adapter.get_channel_updates(channel_id, week_ago)
```

#### Downloads

```python
# Download all course materials
files = await adapter.download_course_content(course_id, "output_dir")

# Download to default location
files = await adapter.download_course_content(course_id)

# Download channel attachments
attachments = await adapter.download_channel_attachments(channel_id)
```

## Usage Examples

See `examples.py` for 16 comprehensive examples covering:

1. Basic authentication
2. Session persistence
3. Course discovery
4. Search functionality
5. Channel monitoring
6. Content downloads
7. Complete backup workflows
8. Error handling
9. Advanced configuration
10. Data analysis

Run examples:
```bash
python examples.py
```

## Output Structure

```
tu_isis_downloads/
├── channels/
│   └── {channel_id}/
│       └── attachments/
├── courses/
│   └── {course_id}/
│       ├── lectures/
│       ├── materials/
│       └── metadata.json
└── states/
    └── tu_isis_state.json
```

## Error Handling

```python
from tu_isis_adapter import (
    TUIsisAdapter,
    AuthenticationError,
    TUIsisAdapterError
)

try:
    await adapter.login("user", "pass")
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except TUIsisAdapterError as e:
    print(f"Adapter error: {e}")
```

## Notes

- **Authentication Required**: Most functions require valid TU Berlin credentials
- **MFA Support**: The adapter detects MFA prompts - you'll need to complete them manually in non-headless mode
- **Session Validity**: Saved states typically expire after 8-24 hours
- **Access Permissions**: Only content you have access to will be retrievable

## License

MIT
