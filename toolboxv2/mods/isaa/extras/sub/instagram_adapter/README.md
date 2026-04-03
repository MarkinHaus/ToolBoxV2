# Instagram Video Adapter

A Python module for downloading saved videos from Instagram using Playwright for browser automation.

## Features

- ✅ **Authentication** with username/password
- ✅ **2FA (MFA) support** for two-factor authentication
- ✅ **Persistent login sessions** - save and restore login state
- ✅ **Download all saved videos** from your Instagram account
- ✅ **Incremental downloads** - download only new videos since last run
- ✅ **Metadata extraction** (video URL, thumbnail, caption, author, timestamp, likes)
- ✅ **Scheduling support** - suitable for cron jobs
- ✅ **Error handling** for login failures, rate limiting, and network issues

## Output Structure

```
instagram_downloads/
├── videos/
│   ├── 20240115_155230_username.mp4
│   └── 20240115_155845_anotheruser.mp4
├── metadata/
│   ├── saved_videos.json
│   └── {post_id}_metadata.json
└── states/
    └── last_check.json
```

## Installation

### 1. Install Dependencies

```bash
pip install playwright aiohttp

# Install Playwright browsers
playwright install chromium
```

### 2. Clone or Copy the Module

Ensure `instagram_adapter.py` is in your Python path or current directory.

## Usage

### Basic Usage - Download All Saved Videos

```python
import asyncio
from instagram_adapter import InstagramAdapter

async def main():
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        output_dir="my_instagram_videos"
    ) as adapter:
        # Login (first run will require MFA if enabled)
        await adapter.login()
        
        # Download all saved videos
        results = await adapter.download_all_saved_videos()
        
        print(f"Downloaded {results['success']} videos")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")

asyncio.run(main())
```

### With 2FA Authentication

```python
import asyncio
from instagram_adapter import InstagramAdapter

async def main():
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        headless=False  # Set False for debugging login flow
    ) as adapter:
        # Login with 2FA code
        await adapter.login(two_factor_code="123456")
        
        # Download videos
        await adapter.download_all_saved_videos()

asyncio.run(main())
```

### Quick Start Functions

```python
import asyncio
from instagram_adapter import quick_download_saved_videos

# Download all saved videos in one call
results = asyncio.run(quick_download_saved_videos(
    username="your_username",
    password="your_password",
    output_dir="my_videos"
))

print(f"Downloaded {results['success']} videos")
```

### Incremental Downloads (Weekly)

```python
import asyncio
from instagram_adapter import InstagramAdapter

async def main():
    async with InstagramAdapter(
        username="your_username",
        password="your_password"
    ) as adapter:
        await adapter.login()
        
        # Download only new videos since last check
        results = await adapter.run_weekly_download()
        
        print(f"New videos downloaded: {results['success']}")

asyncio.run(main())
```

### Cron Job Setup

Add this to your crontab (run every Sunday at 2 AM):

```cron
0 2 * * 0 /usr/bin/python3 -c "
import asyncio
from instagram_adapter import quick_download_new_videos
asyncio.run(quick_download_new_videos('username', 'password', '/path/to/output'))
" >> /var/log/instagram_download.log 2>&1
```

Or create a script `download_videos.py`:

```python
#!/usr/bin/env python3
import asyncio
from instagram_adapter import InstagramAdapter

async def main():
    async with InstagramAdapter(
        username="your_username",
        password="your_password",
        output_dir="/path/to/downloads"
    ) as adapter:
        await adapter.login()
        results = await adapter.run_weekly_download()
        print(f"Downloaded {results['success']} new videos")

if __name__ == "__main__":
    asyncio.run(main())
```

Then add to crontab:

```cron
0 2 * * 0 /path/to/download_videos.py
```

### Get Post Details

```python
import asyncio
from instagram_adapter import InstagramAdapter

async def main():
    async with InstagramAdapter("username", "password") as adapter:
        await adapter.login()
        
        # Get details for a specific post
        post_url = "https://www.instagram.com/p/ABC123/"
        details = await adapter.get_post_details(post_url)
        
        print(f"Video URL: {details.get('video_url')}")
        print(f"Caption: {details.get('caption')}")
        print(f"Author: {details.get('author')}")
        print(f"Likes: {details.get('likes')}")

asyncio.run(main())
```

## Class Reference

### InstagramAdapter

Main class for interacting with Instagram.

#### Constructor

```python
InstagramAdapter(
    username: str,
    password: str,
    state_dir: str = "instagram_states",
    output_dir: str = "instagram_downloads",
    headless: bool = True,
    user_agent: str = None
)
```

**Parameters:**
- `username`: Instagram username
- `password`: Instagram password
- `state_dir`: Directory to save/load login state (default: "instagram_states")
- `output_dir`: Directory for downloaded videos (default: "instagram_downloads")
- `headless`: Run browser headlessly (default: True, set False for debugging)
- `user_agent`: Custom user agent string

#### Methods

##### `async login(two_factor_code: str = None) -> bool`
Login to Instagram.

**Parameters:**
- `two_factor_code`: Optional 2FA code if account has 2FA enabled

**Returns:** True if login successful

**Raises:**
- `LoginError`: If login fails
- `RateLimitError`: If rate limited by Instagram

##### `async get_saved_posts() -> list[dict]`
Get list of all saved posts.

**Returns:** List of post dictionaries with `url` and `collected_at` fields

##### `async get_post_details(post_url: str) -> dict`
Get detailed information about a specific post.

**Parameters:**
- `post_url`: URL or path to Instagram post

**Returns:** Dictionary with post metadata including:
- `video_url`: Direct download link for video
- `thumbnail_url`: URL to thumbnail image
- `caption`: Post caption text
- `author`: Post author username
- `likes`: Number of likes (as string)
- `timestamp`: Post timestamp

##### `async download_video(video_data: dict, output_dir: str = None) -> str`
Download a single video.

**Parameters:**
- `video_data`: Dictionary containing `video_url` and metadata
- `output_dir`: Custom output directory

**Returns:** Path to downloaded file

**Raises:**
- `VideoDownloadError`: If download fails

##### `async download_all_saved_videos() -> dict`
Download all videos from saved posts.

**Returns:** Dictionary with results:
- `success`: Number of successfully downloaded videos
- `failed`: Number of failed downloads
- `skipped`: Number of skipped (non-video) posts
- `videos`: List of downloaded video info

##### `async get_new_videos_since(last_check: datetime) -> list[dict]`
Get videos saved since a specific timestamp.

**Parameters:**
- `last_check`: Last check timestamp

**Returns:** List of video dictionaries

##### `async run_weekly_download(output_dir: str = None, since: datetime = None) -> dict`
Run weekly incremental download (suitable for cron jobs).

**Parameters:**
- `output_dir`: Override output directory
- `since`: Override last check timestamp

**Returns:** Dictionary with download results

##### `async teardown()`
Clean up browser resources.

## Error Handling

The adapter includes custom exceptions for different error scenarios:

```python
from instagram_adapter import (
    InstagramAdapterError,
    LoginError,
    RateLimitError,
    PrivateProfileError,
    VideoDownloadError
)

try:
    async with InstagramAdapter("user", "pass") as adapter:
        await adapter.login()
        await adapter.download_all_saved_videos()
except LoginError as e:
    print(f"Login failed: {e}")
except RateLimitError as e:
    print(f"Rate limited, wait and try again: {e}")
except VideoDownloadError as e:
    print(f"Download failed: {e}")
except InstagramAdapterError as e:
    print(f"General error: {e}")
```

## Important Notes

### Anti-Bot Measures

Instagram has strong anti-bot measures. To avoid detection:

1. **Use persistent sessions** - the adapter saves login state automatically
2. **Use realistic delays** - built-in delays between actions
3. **Don't run too frequently** - respect Instagram's rate limits
4. **Use a consistent user agent** - default provided

### Login Troubleshooting

If login fails:

1. Set `headless=False` to see what's happening:
   ```python
   async with InstagramAdapter(username, password, headless=False) as adapter:
   ```

2. Your account may have 2FA enabled - provide the code:
   ```python
   await adapter.login(two_factor_code="123456")
   ```

3. Instagram may require additional verification - complete it in the browser

4. Clear saved state and try fresh:
   ```bash
   rm -rf instagram_states/
   ```

### Rate Limiting

If you get rate-limited:

1. Wait 24-48 hours before retrying
2. Reduce download frequency
3. Consider using the incremental download feature instead of downloading everything at once

## Privacy and Terms of Service

- This tool is for **personal use only** - downloading your own saved posts
- Respect Instagram's Terms of Service
- Do not use for scraping other users' content without permission
- The authors are not responsible for misuse of this tool

## Troubleshooting

### "Playwright not found" Error

```bash
pip install playwright
playwright install chromium
```

### Login Always Asks for 2FA

The adapter saves your login state automatically. Make sure the `instagram_states/` directory is preserved between runs.

### Videos Not Found

- Some saved posts may be images, not videos
- Some videos may be from private accounts or deleted
- Check the logs for details

### Module Import Errors

Ensure `instagram_adapter.py` is in your current directory or Python path, or install it as a package.

## Requirements

- Python 3.8+
- Playwright
- aiohttp

## License

This module is provided as-is for educational and personal use.
