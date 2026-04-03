"""
TU ISIS Adapter - Module for accessing TU Berlin's ISIS platform

This module provides a Python interface to TU Berlin's ISIS (ILIAS) platform,
including authentication via Shibboleth SSO, course/channel management,
search functionality, and content download capabilities.

Author: Generated for TU Berlin ISIS Integration
License: MIT
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

try:
    from playwright.async_api import (
        Browser as ABrowser,
        BrowserContext as ABrowserContext,
        Page as APage,
        async_playwright,
    )
except ImportError:
    os.system("pip install playwright")
    ABrowser = None
    ABrowserContext = None
    APage = None
    async_playwright = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Channel:
    """Represents a communication channel in TU ISIS"""
    channel_id: str
    name: str
    description: str = ""
    url: str = ""
    last_updated: Optional[datetime] = None


@dataclass
class Post:
    """Represents a post or announcement in a channel"""
    post_id: str
    channel_id: str
    title: str
    content: str
    author: str
    timestamp: datetime
    attachments: list[str] = None  # URLs to attachments

    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []


@dataclass
class Course:
    """Represents a course in TU ISIS"""
    course_id: str
    name: str
    description: str = ""
    instructor: str = ""
    semester: str = ""
    department: str = ""
    url: str = ""
    schedule: list[dict] = None  # List of scheduled items
    materials_count: int = 0
    announcements_count: int = 0

    def __post_init__(self):
        if self.schedule is None:
            self.schedule = []


@dataclass
class CourseMaterial:
    """Represents a material (file/video) in a course"""
    material_id: str
    course_id: str
    name: str
    type: str  # 'file', 'video', 'folder'
    url: str
    size: int = 0
    created_at: Optional[datetime] = None
    parent_folder: Optional[str] = None


# ============================================================================
# Exceptions
# ============================================================================

class TUIsisAdapterError(Exception):
    """Base exception for TUIsisAdapter errors"""
    pass


class AuthenticationError(TUIsisAdapterError):
    """Raised when authentication fails"""
    pass


class NetworkError(TUIsisAdapterError):
    """Raised for network-related errors"""
    pass


class RateLimitError(TUIsisAdapterError):
    """Raised when rate limiting is encountered"""
    pass


# ============================================================================
# Main Adapter Class
# ============================================================================

class TUIsisAdapter:
    """
    Adapter for TU Berlin's ISIS platform.
    
    Provides functionality for:
    - Shibboleth SSO authentication
    - Channel and course management
    - Content search and discovery
    - File and material downloads
    - State persistence for sessions
    """
    
    # TU Berlin ISIS URLs
    BASE_URL = "https://isis.tu-berlin.de"
    LOGIN_URL = f"{BASE_URL}/ilias.php?baseClass=ilUIHookRouterGUI&cmd=forward&lang=en"
    COURSE_URL = f"{BASE_URL}/ilias.php?baseClass=ilRepositoryGUI&cmd=frameset"
    SEARCH_URL = f"{BASE_URL}/ilias.php?baseClass=ilRepositorySearchGUI"
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = False,
        state_dir: str = "tu_isis_states",
        downloads_dir: str = "tu_isis_downloads",
        log_level: int = logging.INFO
    ):
        """
        Initialize the TUIsisAdapter.
        
        :param browser_type: Browser type ('chromium', 'firefox', 'webkit')
        :param headless: Run browser in headless mode
        :param state_dir: Directory for saving/loading session states
        :param downloads_dir: Directory for downloaded content
        :param log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.browser_type = browser_type
        self.headless = headless
        self.state_dir = state_dir
        self.downloads_dir = downloads_dir
        self.is_authenticated = False
        self.current_user = None
        
        # Playwright components
        self.playwright = None
        self.browser: Optional[ABrowser] = None
        self.context: Optional[ABrowserContext] = None
        self.page: Optional[APage] = None
        
        # Create directories
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(downloads_dir, exist_ok=True)
        os.makedirs(os.path.join(downloads_dir, "channels"), exist_ok=True)
        os.makedirs(os.path.join(downloads_dir, "courses"), exist_ok=True)
        os.makedirs(os.path.join(downloads_dir, "states"), exist_ok=True)
    
    # ========================================================================
    # Lifecycle Management
    # ========================================================================
    
    async def setup(self):
        """Initialize Playwright and launch browser"""
        if async_playwright is None:
            raise ImportError("Playwright not installed. Run: pip install playwright")
        
        self.playwright = await async_playwright().start()
        
        browser_launchers = {
            'chromium': self.playwright.chromium.launch,
            'firefox': self.playwright.firefox.launch,
            'webkit': self.playwright.webkit.launch
        }
        
        launcher = browser_launchers.get(self.browser_type, self.playwright.chromium.launch)
        self.browser = await launcher(
            headless=self.headless,
            timeout=30000
        )
        
        self.logger.info(f"Browser launched: {self.browser_type}")
    
    async def create_context(
        self,
        viewport: dict[str, int] = None,
        user_agent: str = None
    ):
        """Create a new browser context"""
        context_options = {}
        if viewport:
            context_options['viewport'] = viewport
        if user_agent:
            context_options['user_agent'] = user_agent
        
        # Set up download handling
        if 'accept_downloads' not in context_options:
            context_options['accept_downloads'] = True
        
        self.context = await self.browser.new_context(**context_options)
        self.page = await self.context.new_page()
        
        # Set default timeout
        self.page.set_default_timeout(60000)  # 60 seconds
        
        self.logger.info("Browser context created")
    
    async def teardown(self):
        """Close browser and stop Playwright"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            self.logger.error(f"Error during teardown: {e}")
        
        self.logger.info("TUIsisAdapter teardown complete")
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.teardown()
    
    # ========================================================================
    # Authentication
    # ========================================================================
    
    async def login(self, username: str = None, password: str = None) -> bool:
        """
        Authenticate with TU Berlin ISIS using Shibboleth SSO.
        
        If username and password are provided, they will be used.
        Otherwise, the method will attempt to load a saved state.
        
        :param username: TU Berlin username
        :param password: TU Berlin password
        :return: True if authentication successful
        """
        # Try to load saved state first
        if username is None and password is None:
            if await self._load_saved_state():
                self.is_authenticated = True
                self.logger.info("Authenticated using saved state")
                return True
        
        # Perform fresh login
        if username is None or password is None:
            raise AuthenticationError(
                "Username and password required for initial login, "
                "or a saved state must exist."
            )
        
        try:
            await self.navigate(self.LOGIN_URL)
            
            # Wait for and handle Shibboleth login page
            await asyncio.sleep(2)
            
            # Check if already on login page or need to follow redirect
            current_url = self.page.url
            
            # Look for username field - this is the Shibboleth login form
            try:
                await self.page.wait_for_selector('input[name="j_username"]', timeout=10000)
                self.logger.info("Shibboleth login form detected")
                
                # Fill in credentials
                await self.page.fill('input[name="j_username"]', username)
                await self.page.fill('input[name="j_password"]', password)
                
                # Click login button
                await asyncio.sleep(1)
                login_button = self.page.locator('input[type="submit"], button[type="submit"]').first
                await login_button.click()
                
                # Wait for login to complete
                await self.page.wait_for_load_state('networkidle')
                await asyncio.sleep(3)
                
                # Check for MFA or additional steps
                if 'duo' in self.page.url.lower() or 'mfa' in self.page.url.lower():
                    self.logger.warning("MFA detected - waiting for manual completion")
                    # Wait for user to complete MFA (max 2 minutes)
                    for _ in range(120):
                        if 'isis.tu-berlin.de' in self.page.url:
                            break
                        await asyncio.sleep(1)
                
                # Verify we're logged in
                if 'isis.tu-berlin.de' in self.page.url:
                    # Check for user info
                    try:
                        user_element = await self.page.query_selector('.il-user-name, .il_personal_user_name')
                        if user_element:
                            self.current_user = await user_element.text_content()
                    except:
                        self.current_user = username
                    
                    self.is_authenticated = True
                    
                    # Save state for future use
                    await self.save_state("tu_isis_session")
                    
                    self.logger.info(f"Authentication successful. User: {self.current_user}")
                    return True
                else:
                    raise AuthenticationError(f"Login failed. Current URL: {self.page.url}")
                    
            except Exception as e:
                # Try alternative login flow
                self.logger.warning(f"Primary login flow failed: {e}, trying alternative")
                return await self._alternative_login(username, password)
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Failed to authenticate: {e}")
    
    async def _alternative_login(self, username: str, password: str) -> bool:
        """Alternative login method for different Shibboleth configurations"""
        try:
            # Navigate directly to login
            await self.navigate(self.BASE_URL)
            
            # Look for login link/button
            login_selectors = [
                'a[href*="login"]',
                'button:has-text("Login")',
                'a:has-text("Login")',
                '.il-login-button',
                '#il_start_login'
            ]
            
            for selector in login_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.click()
                        await asyncio.sleep(2)
                        break
                except:
                    continue
            
            # Now try to find Shibboleth form again
            await self.page.wait_for_selector('input[name="j_username"], input[type="text"]', timeout=10000)
            
            # Fill username
            await self.page.fill('input[name="j_username"], input[type="text"]:first-of-type', username)
            await self.page.fill('input[name="j_password"], input[type="password"]', password)
            
            # Submit
            await self.page.click('input[type="submit"], button[type="submit"]')
            
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(3)
            
            if 'isis.tu-berlin.de' in self.page.url:
                self.is_authenticated = True
                await self.save_state("tu_isis_session")
                return True
            
            raise AuthenticationError("Alternative login also failed")
            
        except Exception as e:
            raise AuthenticationError(f"Alternative login failed: {e}")
    
    def _check_authentication(self):
        """Check if authenticated, raise error if not"""
        if not self.is_authenticated:
            raise AuthenticationError("Not authenticated. Please call login() first.")
    
    # ========================================================================
    # Navigation Helpers
    # ========================================================================
    
    async def navigate(self, url: str, wait_for: str = 'networkidle'):
        """
        Navigate to a URL.
        
        :param url: URL to navigate to
        :param wait_for: Wait condition ('networkidle', 'load', 'domcontentloaded')
        """
        try:
            self.logger.debug(f"Navigating to: {url}")
            await self.page.goto(url, wait_until=wait_for, timeout=60000)
        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            raise NetworkError(f"Failed to navigate to {url}: {e}")
    
    # ========================================================================
    # State Management
    # ========================================================================
    
    async def save_state(self, state_name: str = "tu_isis_session") -> bool:
        """
        Save current browser state for session persistence.
        
        :param state_name: Name of the state file (without _state.json)
        :return: True if successful
        """
        try:
            state_path = os.path.join(self.state_dir, f"{state_name}_state.json")
            state = await self.context.storage_state()
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            
            self.logger.info(f"State saved: {state_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    async def _load_saved_state(self) -> bool:
        """
        Load a previously saved browser state.
        
        :return: True if state loaded successfully
        """
        try:
            state_files = [
                f for f in os.listdir(self.state_dir)
                if f.endswith('_state.json')
            ]
            
            if not state_files:
                return False
            
            # Use the most recent state file
            state_file = sorted(state_files)[-1]
            state_path = os.path.join(self.state_dir, state_file)
            
            with open(state_path) as f:
                state = json.load(f)
            
            # Create new context with saved state
            self.context = await self.browser.new_context(storage_state=state)
            self.page = await self.context.new_page()
            
            # Verify session is still valid
            await self.navigate(self.BASE_URL)
            await asyncio.sleep(2)
            
            # Check if still logged in
            current_url = self.page.url
            if 'isis.tu-berlin.de' in current_url and 'login' not in current_url.lower():
                self.is_authenticated = True
                self.logger.info(f"State loaded and session valid: {state_path}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    # ========================================================================
    # Channel Management
    # ========================================================================
    
    async def get_channels(self) -> list[Channel]:
        """
        Get all accessible channels.
        
        :return: List of Channel objects
        """
        self._check_authentication()
        
        try:
            # Navigate to dashboard or channels page
            await self.navigate(self.COURSE_URL)
            await asyncio.sleep(2)
            
            channels = []
            
            # Try to find channels/forums
            channel_selectors = [
                'a.il-link-target[href*="frm"]',  # Forum links
                '.il-container-item > a',
                '[data-ref-id] > a',
            ]
            
            for selector in channel_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        for element in elements[:50]:  # Limit to first 50
                            try:
                                text = await element.text_content()
                                href = await element.get_attribute('href')
                                
                                if text and href:
                                    # Extract channel ID from URL
                                    channel_id = self._extract_id_from_url(href, 'ref_id')
                                    if not channel_id:
                                        channel_id = self._extract_id_from_url(href, 'frm')
                                    
                                    if channel_id:
                                        full_url = urljoin(self.BASE_URL, href) if not href.startswith('http') else href
                                        channels.append(Channel(
                                            channel_id=channel_id,
                                            name=text.strip(),
                                            url=full_url
                                        ))
                            except:
                                continue
                        break
                except:
                    continue
            
            self.logger.info(f"Found {len(channels)} channels")
            return channels
            
        except Exception as e:
            self.logger.error(f"Error getting channels: {e}")
            raise TUIsisAdapterError(f"Failed to get channels: {e}")
    
    async def get_channel_updates(self, channel_id: str, since: datetime = None) -> list[dict]:
        """
        Get updates/announcements from a specific channel.
        
        :param channel_id: Channel identifier
        :param since: Only get updates after this datetime
        :return: List of post dictionaries
        """
        self._check_authentication()
        
        try:
            # Navigate to channel
            channel_url = f"{self.BASE_URL}/ilias.php?ref_id={channel_id}&cmd=showPosts&cmdClass=ilobjforumgui"
            await self.navigate(channel_url)
            await asyncio.sleep(2)
            
            posts = []
            
            # Look for posts
            post_selectors = [
                '.il-forum-post',
                '.il-list-item',
                '[data-thread-id]',
            ]
            
            for selector in post_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        for element in elements[:100]:  # Limit to first 100 posts
                            try:
                                post_data = await self._parse_post_element(element, channel_id)
                                
                                # Filter by date if specified
                                if since and post_data.get('timestamp'):
                                    if post_data['timestamp'] <= since:
                                        continue
                                
                                posts.append(post_data)
                            except:
                                continue
                        break
                except:
                    continue
            
            self.logger.info(f"Found {len(posts)} posts in channel {channel_id}")
            return posts
            
        except Exception as e:
            self.logger.error(f"Error getting channel updates: {e}")
            raise TUIsisAdapterError(f"Failed to get channel updates: {e}")
    
    async def _parse_post_element(self, element, channel_id: str) -> dict:
        """Parse a post/announcement element and extract data"""
        try:
            # Try to get title
            title_element = await element.query_selector('.il-forum-post-title, h3, h4, .il-item-title')
            title = await title_element.text_content() if title_element else "Untitled"
            
            # Try to get content
            content_element = await element.query_selector('.il-forum-post-content, .il-description, .il-item-content')
            content = await content_element.inner_text() if content_element else ""
            
            # Try to get author
            author_element = await element.query_selector('.il-forum-post-author, .il-item-author, .il-user-name')
            author = await author_element.text_content() if author_element else "Unknown"
            
            # Try to get timestamp
            timestamp_element = await element.query_selector('.il-forum-post-date, .il-item-date, time')
            timestamp_text = await timestamp_element.text_content() if timestamp_element else ""
            timestamp = self._parse_timestamp(timestamp_text)
            
            # Try to get attachments
            attachment_links = await element.query_selector_all('a[href*="download"]')
            attachments = []
            for link in attachment_links:
                href = await link.get_attribute('href')
                if href:
                    full_url = urljoin(self.BASE_URL, href) if not href.startswith('http') else href
                    attachments.append(full_url)
            
            return {
                'post_id': self._extract_id_from_url(await element.get_attribute('data-post-id') or '', 'post'),
                'channel_id': channel_id,
                'title': title.strip(),
                'content': content.strip(),
                'author': author.strip(),
                'timestamp': timestamp,
                'attachments': attachments
            }
        except:
            return {
                'post_id': '',
                'channel_id': channel_id,
                'title': 'Unknown',
                'content': '',
                'author': 'Unknown',
                'timestamp': None,
                'attachments': []
            }
    
    # ========================================================================
    # Course Management
    # ========================================================================
    
    async def get_courses(self) -> list[Course]:
        """
        Get all accessible courses.
        
        :return: List of Course objects
        """
        self._check_authentication()
        
        try:
            # Navigate to course repository
            await self.navigate(self.COURSE_URL)
            await asyncio.sleep(2)
            
            courses = []
            
            # Look for course items
            course_selectors = [
                'a.il-link-target[href*="crs"]',
                '.il-item-title a[href*="ref_id"]',
                '[data-ref-type="crs"] > a',
            ]
            
            for selector in course_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        for element in elements[:100]:  # Limit to first 100
                            try:
                                text = await element.text_content()
                                href = await element.get_attribute('href')
                                
                                if text and href:
                                    course_id = self._extract_id_from_url(href, 'ref_id')
                                    if not course_id:
                                        course_id = self._extract_id_from_url(href, 'crs')
                                    
                                    if course_id:
                                        full_url = urljoin(self.BASE_URL, href) if not href.startswith('http') else href
                                        
                                        # Try to get additional info from parent
                                        parent = await element.evaluate('el => el.closest(".il-container-item, .il-list-item")')
                                        if parent:
                                            description = await element.evaluate('el => el.querySelector(".il-description, .il-item-desc")?.textContent?.trim() || ""')
                                        else:
                                            description = ""
                                        
                                        courses.append(Course(
                                            course_id=course_id,
                                            name=text.strip(),
                                            description=description,
                                            url=full_url
                                        ))
                            except:
                                continue
                        break
                except:
                    continue
            
            self.logger.info(f"Found {len(courses)} courses")
            return courses
            
        except Exception as e:
            self.logger.error(f"Error getting courses: {e}")
            raise TUIsisAdapterError(f"Failed to get courses: {e}")
    
    async def get_course_details(self, course_id: str) -> dict:
        """
        Get detailed information about a specific course.
        
        :param course_id: Course identifier
        :return: Dictionary with course details
        """
        self._check_authentication()
        
        try:
            course_url = f"{self.BASE_URL}/ilias.php?ref_id={course_id}&cmd=view&cmdClass=ilobjcoursegui"
            await self.navigate(course_url)
            await asyncio.sleep(2)
            
            details = {
                'course_id': course_id,
                'name': '',
                'description': '',
                'instructor': '',
                'semester': '',
                'department': '',
                'schedule': [],
                'announcements': []
            }
            
            # Get course name
            try:
                title_element = await self.page.query_selector('h1, .il-header-title, .il-object-title')
                if title_element:
                    details['name'] = await title_element.text_content()
            except:
                pass
            
            # Get description
            try:
                desc_element = await self.page.query_selector('.il-description, .il-course-description')
                if desc_element:
                    details['description'] = await desc_element.inner_text()
            except:
                pass
            
            # Get instructor
            try:
                instructor_elements = await self.page.query_selector_all('.il-member-name, .il-instructor, [data-instructor]')
                if instructor_elements:
                    details['instructor'] = await instructor_elements[0].text_content()
            except:
                pass
            
            # Get semester info
            try:
                semester_element = await self.page.query_selector('.il-semester, [data-semester]')
                if semester_element:
                    details['semester'] = await semester_element.text_content()
            except:
                pass
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error getting course details: {e}")
            raise TUIsisAdapterError(f"Failed to get course details: {e}")
    
    # ========================================================================
    # Search Functionality
    # ========================================================================
    
    async def search_courses(self, query: str, filters: dict = None) -> list[dict]:
        """
        Search for courses using the ISIS search functionality.
        
        :param query: Search query string
        :param filters: Optional filters (semester, department, instructor, time_slot)
        :return: List of course dictionaries
        """
        self._check_authentication()
        
        try:
            # Navigate to search page
            await self.navigate(self.SEARCH_URL)
            await asyncio.sleep(1)
            
            # Fill search query
            search_selectors = [
                'input[name="query"]',
                'input[type="search"]',
                '#il_search_input',
                '[data-search-input]'
            ]
            
            for selector in search_selectors:
                try:
                    search_input = await self.page.query_selector(selector)
                    if search_input:
                        await search_input.fill(query)
                        break
                except:
                    continue
            
            # Apply filters if provided
            if filters:
                await self._apply_search_filters(filters)
            
            # Submit search
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                '.il-search-button',
                '[data-search-submit]'
            ]
            
            for selector in submit_selectors:
                try:
                    submit_button = await self.page.query_selector(selector)
                    if submit_button:
                        await submit_button.click()
                        break
                except:
                    continue
            
            # Wait for results
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            # Parse results
            results = []
            result_selectors = [
                '.il-search-result-item',
                '.il-container-item',
                '[data-result-item]'
            ]
            
            for selector in result_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        for element in elements[:50]:
                            try:
                                result = await self._parse_search_result(element)
                                if result:
                                    results.append(result)
                            except:
                                continue
                        break
                except:
                    continue
            
            self.logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching courses: {e}")
            raise TUIsisAdapterError(f"Failed to search courses: {e}")
    
    async def _apply_search_filters(self, filters: dict):
        """Apply search filters if available"""
        # Semester filter
        if 'semester' in filters:
            try:
                semester_select = await self.page.query_selector('select[name="semester"], [data-filter-semester]')
                if semester_select:
                    await semester_select.select_option(filters['semester'])
            except:
                pass
        
        # Department filter
        if 'department' in filters:
            try:
                dept_select = await self.page.query_selector('select[name="department"], [data-filter-department]')
                if dept_select:
                    await dept_select.select_option(filters['department'])
            except:
                pass
    
    async def _parse_search_result(self, element) -> dict:
        """Parse a search result element"""
        try:
            link = await element.query_selector('a')
            if not link:
                return None
            
            text = await link.text_content()
            href = await link.get_attribute('href')
            
            if not text or not href:
                return None
            
            return {
                'name': text.strip(),
                'url': urljoin(self.BASE_URL, href) if not href.startswith('http') else href,
                'course_id': self._extract_id_from_url(href, 'ref_id'),
                'type': 'course'
            }
        except:
            return None
    
    # ========================================================================
    # Download Functionality
    # ========================================================================
    
    async def download_course_content(self, course_id: str, output_dir: str = None) -> list[str]:
        """
        Download all materials from a course.
        
        :param course_id: Course identifier
        :param output_dir: Output directory (defaults to downloads_dir/courses/course_id)
        :return: List of downloaded file paths
        """
        self._check_authentication()
        
        if output_dir is None:
            output_dir = os.path.join(self.downloads_dir, "courses", course_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Navigate to course
            course_url = f"{self.BASE_URL}/ilias.php?ref_id={course_id}&cmd=render&cmdClass=ilmiaslmsubmissiongui"
            await self.navigate(course_url)
            await asyncio.sleep(2)
            
            downloaded_files = []
            
            # Look for downloadable files
            file_selectors = [
                'a.il-link-target[href*="download"]',
                'a[href*="file"]',
                '.il-file-link a'
            ]
            
            for selector in file_selectors:
                try:
                    links = await self.page.query_selector_all(selector)
                    if links:
                        for link in links[:100]:  # Limit to 100 files
                            try:
                                href = await link.get_attribute('href')
                                filename = await link.text_content()
                                
                                if href and filename:
                                    file_path = await self._download_file(
                                        urljoin(self.BASE_URL, href) if not href.startswith('http') else href,
                                        output_dir,
                                        filename.strip()
                                    )
                                    if file_path:
                                        downloaded_files.append(file_path)
                            except:
                                continue
                        break
                except:
                    continue
            
            # Save metadata
            metadata = {
                'course_id': course_id,
                'downloaded_at': datetime.now().isoformat(),
                'files': downloaded_files
            }
            
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Downloaded {len(downloaded_files)} files from course {course_id}")
            return downloaded_files
            
        except Exception as e:
            self.logger.error(f"Error downloading course content: {e}")
            raise TUIsisAdapterError(f"Failed to download course content: {e}")
    
    async def _download_file(self, url: str, output_dir: str, filename: str) -> Optional[str]:
        """Download a file and save it"""
        try:
            # Clean filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # Set download path
            file_path = os.path.join(output_dir, filename)
            
            # Create download handler
            async with self.context.expect_download(timeout=60000) as download_info:
                await self.page.goto(url)
            
            download = await download_info.value
            
            # Save file
            await download.save_as(file_path)
            
            self.logger.debug(f"Downloaded: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error downloading file {filename}: {e}")
            return None
    
    async def download_channel_attachments(self, channel_id: str, since: datetime = None) -> list[str]:
        """
        Download all attachments from a channel's posts.
        
        :param channel_id: Channel identifier
        :param since: Only download from posts after this datetime
        :return: List of downloaded file paths
        """
        self._check_authentication()
        
        output_dir = os.path.join(self.downloads_dir, "channels", channel_id)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Get posts
            posts = await self.get_channel_updates(channel_id, since)
            
            downloaded_files = []
            
            # Download attachments from each post
            for post in posts:
                for attachment_url in post.get('attachments', []):
                    filename = self._extract_filename_from_url(attachment_url)
                    file_path = await self._download_file(attachment_url, output_dir, filename)
                    if file_path:
                        downloaded_files.append(file_path)
            
            self.logger.info(f"Downloaded {len(downloaded_files)} attachments from channel {channel_id}")
            return downloaded_files
            
        except Exception as e:
            self.logger.error(f"Error downloading channel attachments: {e}")
            raise TUIsisAdapterError(f"Failed to download channel attachments: {e}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _extract_id_from_url(self, url: str, param: str) -> str:
        """Extract an ID from a URL parameter"""
        if not url:
            return ""
        
        # Try to find in query parameters
        match = re.search(rf'{param}=([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)
        
        # Try to find in path
        match = re.search(r'/([a-zA-Z0-9_-]+)/?$', url)
        if match:
            return match.group(1)
        
        return ""
    
    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        try:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = "downloaded_file"
            return filename
        except:
            return "downloaded_file"
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse various timestamp formats"""
        if not timestamp_str:
            return None
        
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y %H:%M',
            '%d.%m.%Y',
            '%Y-%m-%d',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except:
                continue
        
        return None


# ============================================================================
# Convenience Functions
# ============================================================================

async def quick_login(username: str, password: str, headless: bool = True) -> TUIsisAdapter:
    """
    Quick login helper - returns an authenticated adapter instance.
    
    :param username: TU Berlin username
    :param password: TU Berlin password
    :param headless: Run in headless mode
    :return: Authenticated TUIsisAdapter instance
    """
    adapter = TUIsisAdapter(headless=headless)
    await adapter.setup()
    await adapter.create_context()
    
    if await adapter.login(username, password):
        return adapter
    else:
        await adapter.teardown()
        raise AuthenticationError("Failed to authenticate")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Demo usage of the TUIsisAdapter"""
    async with TUIsisAdapter(headless=True) as adapter:
        await adapter.create_context()
        
        # Note: You need to provide actual credentials
        # username = "your_username"
        # password = "your_password"
        
        # if await adapter.login(username, password):
        #     # Get courses
        #     courses = await adapter.get_courses()
        #     print(f"Found {len(courses)} courses")
        #     
        #     # Search for courses
        #     results = await adapter.search_courses("Machine Learning")
        #     print(f"Search results: {len(results)}")
        #     
        #     # Download course content
        #     if courses:
        #         await adapter.download_course_content(courses[0].course_id)
        
        print("TUIsisAdapter demo - provide credentials in code to test")


if __name__ == "__main__":
    asyncio.run(main())
