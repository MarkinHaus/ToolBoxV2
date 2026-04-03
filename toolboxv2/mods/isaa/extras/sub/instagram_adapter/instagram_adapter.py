"""
Instagram Video Adapter for downloading saved videos.

This module provides functionality to authenticate with Instagram,
navigate to saved posts, and download videos with metadata.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

try:
    import aiohttp
    from playwright.async_api import async_playwright
except ImportError:
    os.system("pip install aiohttp playwright")


class InstagramAdapterError(Exception):
    """Base exception for Instagram adapter errors."""
    pass


class LoginError(InstagramAdapterError):
    """Raised when login fails."""
    pass


class RateLimitError(InstagramAdapterError):
    """Raised when rate limit is hit."""
    pass


class PrivateProfileError(InstagramAdapterError):
    """Raised when trying to access private profile."""
    pass


class VideoDownloadError(InstagramAdapterError):
    """Raised when video download fails."""
    pass


class InstagramAdapter:
    """
    Adapter for downloading saved videos from Instagram.
    
    Uses Playwright for browser automation to handle Instagram's
    anti-bot measures and authentication flow.
    """

    def __init__(
        self,
        username: str,
        password: str,
        state_dir: str = "instagram_states",
        output_dir: str = "instagram_downloads",
        headless: bool = True,
        user_agent: str = None,
    ):
        """
        Initialize the Instagram adapter.
        
        :param username: Instagram username
        :param password: Instagram password
        :param state_dir: Directory to save/load login state
        :param output_dir: Directory to save downloaded videos
        :param headless: Run browser in headless mode (set False for debugging login)
        :param user_agent: Custom user agent string
        """
        self.username = username
        self.password = password
        self.state_dir = state_dir
        self.output_dir = output_dir
        self.headless = headless
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Browser components
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Auth state
        self.is_logged_in = False
        self.requires_2fa = False
        
        # Create necessary directories
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "states"), exist_ok=True)

    async def setup(self):
        """Initialize Playwright and launch browser."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                ],
                timeout=60000
            )
            self.logger.info("Browser launched successfully")
        except Exception as e:
            raise InstagramAdapterError(f"Failed to launch browser: {e}")

    async def create_context(self):
        """Create browser context with realistic settings."""
        try:
            viewport = {"width": 1920, "height": 1080}
            context_options = {
                "viewport": viewport,
                "user_agent": self.user_agent,
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "permissions": ["geolocation", "notifications"],
                "color_scheme": "light",
            }
            
            self.context = await self.browser.new_context(**context_options)
            self.page = await self.context.new_page()
            
            # Add additional headers to appear more human
            await self.page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            })
            
            self.logger.info("Browser context created")
        except Exception as e:
            raise InstagramAdapterError(f"Failed to create context: {e}")

    async def navigate(self, url: str, wait_for: str = "networkidle"):
        """
        Navigate to a URL.
        
        :param url: URL to navigate to
        :param wait_for: Wait condition ('networkidle', 'domcontentloaded', or 'load')
        """
        try:
            await self.page.goto(url, wait_until=wait_for, timeout=60000)
            # Add realistic delay after navigation
            await asyncio.sleep(2)
        except Exception as e:
            raise InstagramAdapterError(f"Failed to navigate to {url}: {e}")

    async def login(self, two_factor_code: str = None) -> bool:
        """
        Login to Instagram with username/password.
        
        :param two_factor_code: Optional 2FA code if account has 2FA enabled
        :return: True if login successful
        
        :raises LoginError: If login fails
        :raises RateLimitError: If rate limited by Instagram
        """
        try:
            # Try to load saved state first
            state_path = os.path.join(self.state_dir, "instagram_state.json")
            if os.path.exists(state_path):
                self.logger.info("Attempting to load saved login state...")
                if await self._load_saved_state(state_path):
                    # Verify we're actually logged in
                    await self.navigate("https://www.instagram.com/")
                    await asyncio.sleep(2)
                    
                    # Check if login is still valid
                    try:
                        profile_button = self.page.locator('[aria-label="Profile"]')
                        if await profile_button.count() > 0:
                            self.is_logged_in = True
                            self.logger.info("Successfully logged in from saved state")
                            return True
                    except Exception:
                        pass
            
            # Perform fresh login
            self.logger.info("Starting Instagram login process...")
            await self.navigate("https://www.instagram.com/accounts/login/")
            
            # Wait for login form
            await self.page.wait_for_selector('input[name="username"]', timeout=10000)
            await asyncio.sleep(1)
            
            # Fill credentials
            await self.page.fill('input[name="username"]', self.username)
            await asyncio.sleep(0.5)
            await self.page.fill('input[name="password"]', self.password)
            await asyncio.sleep(1)
            
            # Click login button
            await self.page.click('button[type="submit"]')
            await asyncio.sleep(3)
            
            # Check for 2FA
            two_factor_input = self.page.locator('input[aria-label="Security Code"]')
            if await two_factor_input.count() > 0:
                self.requires_2fa = True
                if two_factor_code:
                    await two_factor_input.fill(two_factor_code)
                    await asyncio.sleep(1)
                    await self.page.click('button[type="submit"]')
                    await asyncio.sleep(3)
                else:
                    self.logger.warning("2FA code required but not provided")
                    return False
            
            # Check for login challenges or errors
            error_message = await self._check_for_login_error()
            if error_message:
                if "challenge" in error_message.lower():
                    raise LoginError(f"Login challenge required: {error_message}")
                elif "rate limit" in error_message.lower() or "too many attempts" in error_message.lower():
                    raise RateLimitError(f"Rate limited by Instagram: {error_message}")
                else:
                    raise LoginError(f"Login failed: {error_message}")
            
            # Check for "Save Info" or "Turn on Notifications" prompts
            await self._handle_post_login_prompts()
            
            # Verify login success
            try:
                profile_button = self.page.locator('[aria-label="Profile"], svg[aria-label="Profile"]')
                await profile_button.wait_for(timeout=10000)
                self.is_logged_in = True
                
                # Save state for future sessions
                await self._save_state(state_path)
                
                self.logger.info("Login successful")
                return True
            except Exception as e:
                raise LoginError(f"Could not verify login: {e}")
                
        except LoginError:
            raise
        except RateLimitError:
            raise
        except Exception as e:
            raise LoginError(f"Unexpected login error: {e}")

    async def _load_saved_state(self, state_path: str) -> bool:
        """Load saved browser state."""
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.context = await self.browser.new_context(storage_state=state)
            self.page = await self.context.new_page()
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load saved state: {e}")
            return False

    async def _save_state(self, state_path: str):
        """Save current browser state."""
        try:
            state = await self.context.storage_state()
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            self.logger.info(f"State saved to {state_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save state: {e}")

    async def _check_for_login_error(self) -> str:
        """Check for error messages on login page."""
        error_selectors = [
            'p[role="alert"]',
            '#slfErrorAlert',
            '.error-message',
        ]
        
        for selector in error_selectors:
            try:
                error_element = self.page.locator(selector)
                if await error_element.count() > 0:
                    return await error_element.inner_text()
            except Exception:
                continue
        return None

    async def _handle_post_login_prompts(self):
        """Handle 'Save Info' and 'Turn on Notifications' prompts."""
        try:
            # "Save your login info?" prompt
            save_info_buttons = self.page.locator('button:has-text("Not Now")')
            if await save_info_buttons.count() > 0:
                await save_info_buttons.first.click()
                await asyncio.sleep(1)
        except Exception:
            pass
        
        try:
            # "Turn on Notifications" prompt
            notification_buttons = self.page.locator('button:has-text("Not Now")')
            if await notification_buttons.count() > 0:
                await notification_buttons.first.click()
                await asyncio.sleep(1)
        except Exception:
            pass

    async def get_saved_posts(self) -> list[dict]:
        """
        Get list of saved posts from Instagram.
        
        :return: List of post dictionaries with metadata
        """
        if not self.is_logged_in:
            raise InstagramAdapterError("Not logged in. Call login() first.")
        
        try:
            self.logger.info("Fetching saved posts...")
            
            # Navigate to saved posts page
            await self.navigate("https://www.instagram.com/")
            await asyncio.sleep(2)
            
            # Click profile menu
            profile_menu = self.page.locator('[aria-label="Profile"], svg[aria-label="Profile"]')
            await profile_menu.first.click()
            await asyncio.sleep(2)
            
            # Click saved posts (bookmark icon)
            # This requires finding the saved collection
            try:
                saved_link = self.page.locator('a[href*="/saved/"]')
                if await saved_link.count() > 0:
                    await saved_link.first.click()
                else:
                    # Alternative: click the hamburger menu then saved
                    await self.page.click('[aria-label="Menu"]')
                    await asyncio.sleep(1)
                    await self.page.click('text=Saved')
            except Exception as e:
                self.logger.warning(f"Could not navigate via menu, trying direct URL: {e}")
                await self.navigate(f"https://www.instagram.com/{self.username}/saved/")
            
            await asyncio.sleep(3)
            
            # Scroll and collect posts
            posts = []
            scroll_attempts = 0
            max_scrolls = 20
            
            while scroll_attempts < max_scrolls:
                # Get current post elements
                post_links = self.page.locator('a[href^="/p/"]')
                count = await post_links.count()
                
                # Extract unique post URLs
                current_urls = set()
                for i in range(count):
                    try:
                        href = await post_links.nth(i).get_attribute('href')
                        if href:
                            current_urls.add(href)
                    except Exception:
                        continue
                
                # Add new posts to our list
                before_count = len(posts)
                for url in current_urls:
                    if url not in [p.get('url') for p in posts]:
                        posts.append({'url': url, 'collected_at': datetime.now().isoformat()})
                
                if len(posts) == before_count:
                    # No new posts, try to scroll more or break
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                
                # Scroll down
                await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await asyncio.sleep(2)
            
            self.logger.info(f"Found {len(posts)} saved posts")
            return posts
            
        except Exception as e:
            raise InstagramAdapterError(f"Failed to get saved posts: {e}")

    async def get_post_details(self, post_url: str) -> dict:
        """
        Get detailed information about a single post.
        
        :param post_url: URL of the Instagram post
        :return: Dictionary with post metadata
        """
        if not self.is_logged_in:
            raise InstagramAdapterError("Not logged in. Call login() first.")
        
        try:
            self.logger.info(f"Fetching post details: {post_url}")
            full_url = post_url if post_url.startswith('http') else f"https://www.instagram.com{post_url}"
            await self.navigate(full_url)
            await asyncio.sleep(3)
            
            # Extract data from embedded JSON
            post_data = await self._extract_post_data_from_page()
            
            # Fallback to scraping from DOM
            if not post_data or not post_data.get('video_url'):
                post_data = await self._extract_post_data_from_dom()
            
            return post_data
            
        except Exception as e:
            raise InstagramAdapterError(f"Failed to get post details: {e}")

    async def _extract_post_data_from_page(self) -> dict:
        """Extract post data from embedded JSON in page source."""
        try:
            # Get page HTML and extract JSON data
            html = await self.page.content()
            
            # Look for additionalData or similar JSON structures
            patterns = [
                r'window\._sharedData\s*=\s*({.+?});',
                r'additionalData\s*=\s*({.+?});',
                r'<script type="application/ld\+json">(.+?)</script>',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html)
                for match in matches:
                    try:
                        data = json.loads(match)
                        # Navigate the data structure to find post info
                        post_info = self._parse_json_data(data)
                        if post_info:
                            return post_info
                    except json.JSONDecodeError:
                        continue
            
            return {}
        except Exception as e:
            self.logger.warning(f"Could not extract JSON data: {e}")
            return {}

    async def _extract_post_data_from_dom(self) -> dict:
        """Extract post data by scraping DOM elements."""
        try:
            data = {}
            
            # Try to get caption
            try:
                caption_elem = self.page.locator('h1, article div[data-testid="post-comment-root"]')
                if await caption_elem.count() > 0:
                    data['caption'] = await caption_elem.first.inner_text()
            except Exception:
                pass
            
            # Try to get video URL from video element
            try:
                video_elem = self.page.locator('video source')
                if await video_elem.count() > 0:
                    data['video_url'] = await video_elem.first.get_attribute('src')
                else:
                    # Try direct video tag
                    video_elem = self.page.locator('video')
                    if await video_elem.count() > 0:
                        data['video_url'] = await video_elem.first.get_attribute('src')
            except Exception:
                pass
            
            # Try to get image/thumbnail
            try:
                img_elem = self.page.locator('img[srcset]')
                if await img_elem.count() > 0:
                    data['thumbnail_url'] = await img_elem.first.get_attribute('src')
            except Exception:
                pass
            
            # Try to get author
            try:
                author_elem = self.page.locator('a[href^="/"]')
                if await author_elem.count() > 0:
                    author_href = await author_elem.first.get_attribute('href')
                    data['author'] = author_href.strip('/') if author_href else ''
            except Exception:
                pass
            
            # Try to get likes count
            try:
                likes_elem = self.page.locator('[class*="like-count"], span:has-text("likes")')
                if await likes_elem.count() > 0:
                    likes_text = await likes_elem.first.inner_text()
                    data['likes'] = likes_text
            except Exception:
                pass
            
            return data
        except Exception as e:
            self.logger.warning(f"Could not extract DOM data: {e}")
            return {}

    def _parse_json_data(self, data: dict) -> dict:
        """Parse JSON data to extract post information."""
        try:
            # This is a simplified parser - Instagram's data structure varies
            if 'video_url' in str(data):
                # Look for video URLs in the data
                json_str = json.dumps(data)
                video_matches = re.findall(r'"video_url":"([^"]+)"', json_str)
                if video_matches:
                    return {
                        'video_url': video_matches[0].replace('\\u0026', '&'),
                        'extracted_from': 'json',
                    }
            
            # Try to find post in the data structure
            # The actual structure depends on Instagram's current format
            return {}
        except Exception:
            return {}

    async def download_video(self, video_data: dict, output_dir: str = None) -> str:
        """
        Download a video from Instagram.
        
        :param video_data: Dictionary containing video_url and metadata
        :param output_dir: Custom output directory (default: self.output_dir/videos)
        :return: Path to downloaded file
        
        :raises VideoDownloadError: If download fails
        """
        if not video_data or not video_data.get('video_url'):
            raise VideoDownloadError("No video URL provided")
        
        video_url = video_data['video_url']
        output_dir = output_dir or os.path.join(self.output_dir, 'videos')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        author = video_data.get('author', 'unknown').replace('/', '_').replace('@', '')
        filename = f"{timestamp}_{author}.mp4"
        filepath = os.path.join(output_dir, filename)
        
        try:
            self.logger.info(f"Downloading video: {filename}")
            
            # Download using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, headers={
                    'User-Agent': self.user_agent,
                    'Referer': 'https://www.instagram.com/',
                }) as response:
                    if response.status == 200:
                        with open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        self.logger.info(f"Video downloaded: {filepath}")
                        return filepath
                    else:
                        raise VideoDownloadError(f"HTTP {response.status}: Failed to download video")
                        
        except VideoDownloadError:
            raise
        except Exception as e:
            raise VideoDownloadError(f"Download failed: {e}")

    async def get_new_videos_since(self, last_check: datetime) -> list[dict]:
        """
        Get videos saved since last check timestamp.
        
        :param last_check: Last check timestamp
        :return: List of video dictionaries
        """
        try:
            last_check_dt = datetime.fromisoformat(last_check) if isinstance(last_check, str) else last_check
            
            # Get all saved posts
            posts = await self.get_saved_posts()
            
            # Filter by collection time and extract details
            new_videos = []
            for post in posts:
                collected_at = datetime.fromisoformat(post.get('collected_at', ''))
                if collected_at > last_check_dt:
                    # Get full post details
                    details = await self.get_post_details(post['url'])
                    if details and details.get('video_url'):
                        details.update({
                            'url': post['url'],
                            'collected_at': collected_at.isoformat(),
                        })
                        new_videos.append(details)
            
            self.logger.info(f"Found {len(new_videos)} new videos since {last_check}")
            return new_videos
            
        except Exception as e:
            raise InstagramAdapterError(f"Failed to get new videos: {e}")

    async def download_all_saved_videos(self) -> dict[str, Any]:
        """
        Download all videos from saved posts.
        
        :return: Dictionary with download results
        """
        if not self.is_logged_in:
            raise InstagramAdapterError("Not logged in. Call login() first.")
        
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'videos': [],
        }
        
        try:
            posts = await self.get_saved_posts()
            self.logger.info(f"Starting download of {len(posts)} saved posts...")
            
            for i, post in enumerate(posts, 1):
                try:
                    self.logger.info(f"Processing post {i}/{len(posts)}")
                    
                    # Get post details
                    details = await self.get_post_details(post['url'])
                    
                    if not details.get('video_url'):
                        self.logger.info(f"Skipping post {post['url']} - no video found")
                        results['skipped'] += 1
                        continue
                    
                    # Download video
                    filepath = await self.download_video(details)
                    
                    # Save metadata
                    await self._save_metadata(details, filepath)
                    
                    results['videos'].append({
                        'url': post['url'],
                        'filepath': filepath,
                        'metadata': details,
                    })
                    results['success'] += 1
                    
                    # Realistic delay between downloads
                    await asyncio.sleep(2)
                    
                except VideoDownloadError as e:
                    self.logger.error(f"Failed to download {post['url']}: {e}")
                    results['failed'] += 1
                except Exception as e:
                    self.logger.error(f"Error processing {post['url']}: {e}")
                    results['failed'] += 1
            
            # Save summary
            await self._save_download_summary(results)
            
            self.logger.info(f"Download complete: {results['success']} success, {results['failed']} failed, {results['skipped']} skipped")
            return results
            
        except Exception as e:
            raise InstagramAdapterError(f"Failed to download videos: {e}")

    async def _save_metadata(self, video_data: dict, filepath: str):
        """Save metadata for a downloaded video."""
        try:
            metadata_dir = os.path.join(self.output_dir, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)
            
            # Extract post ID from URL
            post_id = re.search(r'/p/([^/]+)', video_data.get('url', ''))
            post_id = post_id.group(1) if post_id else 'unknown'
            
            # Save individual metadata file
            meta_file = os.path.join(metadata_dir, f"{post_id}_metadata.json")
            metadata = {
                **video_data,
                'downloaded_at': datetime.now().isoformat(),
                'filepath': filepath,
                'post_id': post_id,
            }
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save metadata: {e}")

    async def _save_download_summary(self, results: dict):
        """Save overall download summary."""
        try:
            summary_file = os.path.join(self.output_dir, 'metadata', 'saved_videos.json')
            
            # Load existing data
            existing = {}
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    try:
                        existing = json.load(f)
                    except json.JSONDecodeError:
                        existing = {}
            
            # Update with new results
            if 'downloads' not in existing:
                existing['downloads'] = []
            
            for video in results['videos']:
                existing['downloads'].append({
                    'url': video['url'],
                    'filepath': video['filepath'],
                    'post_id': re.search(r'/p/([^/]+)', video.get('url', ''))
                    and re.search(r'/p/([^/]+)', video['url']).group(1)
                    or 'unknown',
                })
            
            existing['last_updated'] = datetime.now().isoformat()
            existing['total_downloads'] = len(existing['downloads'])
            
            with open(summary_file, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save summary: {e}")

    async def run_weekly_download(self, output_dir: str = None, since: datetime = None) -> dict[str, Any]:
        """
        Run weekly download of new videos.
        
        Suitable for cron job scheduling.
        
        :param output_dir: Override output directory
        :param since: Override last check timestamp (defaults to saved state)
        :return: Dictionary with download results
        
        Example cron job:
            # Run every Sunday at 2 AM
            0 2 * * 0 python -c "import asyncio; from instagram_adapter import InstagramAdapter; asyncio.run(InstagramAdapter('user', 'pass').run_weekly_download())"
        """
        if output_dir:
            self.output_dir = output_dir
        
        # Load last check time from state
        state_file = os.path.join(self.output_dir, 'states', 'last_check.json')
        last_check = since
        
        if not last_check and os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_check = datetime.fromisoformat(state.get('last_check'))
        
        if not last_check:
            last_check = datetime.fromtimestamp(0)  # Download everything
        
        results = {
            'since': last_check.isoformat(),
            'success': 0,
            'failed': 0,
            'videos': [],
        }
        
        try:
            # Ensure logged in
            if not self.is_logged_in:
                await self.login()
            
            # Get new videos
            new_videos = await self.get_new_videos_since(last_check)
            
            for video_data in new_videos:
                try:
                    filepath = await self.download_video(video_data)
                    await self._save_metadata(video_data, filepath)
                    
                    results['videos'].append({
                        'url': video_data['url'],
                        'filepath': filepath,
                    })
                    results['success'] += 1
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Failed to download video: {e}")
                    results['failed'] += 1
            
            # Update last check time
            with open(state_file, 'w') as f:
                json.dump({'last_check': datetime.now().isoformat()}, f, indent=2)
            
            results['completed_at'] = datetime.now().isoformat()
            self.logger.info(f"Weekly download complete: {results['success']} success, {results['failed']} failed")
            
            return results
            
        except Exception as e:
            raise InstagramAdapterError(f"Weekly download failed: {e}")

    async def teardown(self):
        """Clean up browser resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.logger.info("Browser resources cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        await self.create_context()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.teardown()


# Convenience functions for quick usage
async def quick_download_saved_videos(username: str, password: str, output_dir: str = "instagram_downloads") -> dict:
    """
    Quick function to download all saved videos.
    
    :param username: Instagram username
    :param password: Instagram password
    :param output_dir: Output directory
    :return: Download results
    """
    async with InstagramAdapter(username, password, output_dir=output_dir) as adapter:
        await adapter.login()
        return await adapter.download_all_saved_videos()


async def quick_download_new_videos(username: str, password: str, output_dir: str = "instagram_downloads") -> dict:
    """
    Quick function to download new videos since last check.
    
    :param username: Instagram username
    :param password: Instagram password
    :param output_dir: Output directory
    :return: Download results
    """
    async with InstagramAdapter(username, password, output_dir=output_dir) as adapter:
        await adapter.login()
        return await adapter.run_weekly_download()
