import json
import os
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page



class WebTestFramework:
    def __init__(self,
                 browser_type: str = 'chromium',
                 headless: bool = False,
                 state_dir: str = 'test_states',
                 auto_save_interval: int = 60,  # Auto-save every 60 seconds
                 idle_timeout: float = 30.0,  # Wait up to 30 seconds for page to be idle
                 log_level: int = logging.INFO):
        """
        Initialize the web testing framework with enhanced features

        :param browser_type: Type of browser to use
        :param headless: Run browser in headless mode
        :param state_dir: Directory to save and load browser states
        :param auto_save_interval: Interval for automatic state saving (in seconds)
        :param idle_timeout: Maximum time to wait for page to become idle
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
        self.auto_save_interval = auto_save_interval
        self.idle_timeout = idle_timeout

        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        # Auto-save threading
        self._auto_save_thread = None
        self._stop_auto_save = threading.Event()

    def setup(self):
        """
        Set up Playwright and launch browser
        """
        self.playwright = sync_playwright().start()

        # Dynamic browser launch based on type
        browser_launchers = {
            'chromium': self.playwright.chromium.launch,
            'firefox': self.playwright.firefox.launch,
            'webkit': self.playwright.webkit.launch
        }

        self.browser = browser_launchers.get(self.browser_type,
                                             self.playwright.chromium.launch)(
            headless=self.headless
        )

    def create_context(self,
                       viewport: Dict[str, int] = None,
                       user_agent: str = None):
        """
        Create a new browser context with optional configuration
        """
        context_options = {}
        if viewport:
            context_options['viewport'] = viewport
        if user_agent:
            context_options['user_agent'] = user_agent

        self.context = self.browser.new_context(**context_options)
        self.page = self.context.new_page()

        # Start auto-save thread
        self._start_auto_save()

    def navigate(self, url: str):
        """
        Navigate to a specific URL with idle waiting
        """
        self.page.goto(url, wait_until='networkidle')

        # Wait until page is truly idle
        start_time = time.time()
        while True:
            # Check network requests
            network_idle = len(self.page.request.all()) == 0

            # Check for JavaScript activity
            js_idle = self.page.evaluate('() => document.readyState === "complete"')

            if network_idle and js_idle:
                break

            if time.time() - start_time > self.idle_timeout:
                self.logger.warning(f"Page did not become idle within {self.idle_timeout} seconds")
                break

            time.sleep(0.5)

    def mimic_user_interaction(self, interactions: List[Dict[str, Any]]):
        """
        Mimic user interactions using Playwright's API

        :param interactions: List of interaction dictionaries
        Example interactions:
        [
            {"type": "click", "selector": "#login-button"},
            {"type": "type", "selector": "#username", "text": "testuser"},
            {"type": "hover", "selector": ".menu-item"},
            {"type": "select", "selector": "#dropdown", "value": "option1"},
            {"type": "check", "selector": "#checkbox"},
            {"type": "screenshot", "path": "screenshot.png"}
        ]
        """
        for interaction in interactions:
            try:
                # Locate element first
                element = self.page.locator(interaction['selector'])

                # Wait for element to be visible
                element.wait_for(state='visible')

                # Perform interaction based on type
                if interaction['type'] == 'click':
                    element.click()
                elif interaction['type'] == 'type':
                    element.fill(interaction['text'])
                elif interaction['type'] == 'hover':
                    element.hover()
                elif interaction['type'] == 'select':
                    element.select_option(interaction['value'])
                elif interaction['type'] == 'check':
                    element.check()
                elif interaction['type'] == 'screenshot':
                    self.page.screenshot(path=interaction.get('path', 'screenshot.png'))

                # Add small wait between interactions
                time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error in user interaction: {e}")

    def _start_auto_save(self):
        """
        Start automatic state saving thread
        """

        def auto_save_worker():
            while not self._stop_auto_save.is_set():
                try:
                    # Save state with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.save_state(f"auto_state_{timestamp}")

                    # Wait for next interval
                    self._stop_auto_save.wait(self.auto_save_interval)
                except Exception as e:
                    self.logger.error(f"Auto-save failed: {e}")

        self._auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self._auto_save_thread.start()

    def save_state(self, state_name: str):
        """
        Save current browser state to a file
        """
        try:
            state_path = os.path.join(self.state_dir, f"{state_name}_state.json")
            state = self.context.storage_state()

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)

            self.logger.info(f"State saved: {state_path}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load_state(self, state_name: str = None):
        """
        Load a previously saved browser state
        If no state_name is provided, load the most recent auto-state
        """
        try:
            if not state_name:
                # Find most recent auto-state
                auto_states = [f for f in os.listdir(self.state_dir) if
                               f.startswith('auto_state_') and f.endswith('_state.json')]
                if not auto_states:
                    raise FileNotFoundError("No auto-states found")

                state_name = sorted(auto_states, reverse=True)[0][:-10]

            state_path = os.path.join(self.state_dir, f"{state_name}_state.json")

            if not os.path.exists(state_path):
                raise FileNotFoundError(f"State file {state_path} not found")

            with open(state_path, 'r') as f:
                state = json.load(f)

            # Create a new context from the saved state
            self.context = self.browser.new_context(storage_state=state)
            self.page = self.context.new_page()

            self.logger.info(f"State loaded: {state_path}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")

    def teardown(self):
        """
        Close browser and stop Playwright
        """
        # Stop auto-save thread
        if self._stop_auto_save:
            self._stop_auto_save.set()

        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()


# Example Usage and Test Cases
def example_test_interaction(framework):
    """
    Example test case with user interaction mimicking
    """
    # Navigate to a test page
    framework.navigate("https://example.com")

    # Define a series of user interactions
    interactions = [
        {"type": "type", "selector": "#search-input", "text": "Playwright testing"},
        {"type": "click", "selector": "#search-button"},
        {"type": "screenshot", "path": "search_results.png"}
    ]

    # Mimic user interactions
    framework.mimic_user_interaction(interactions)


def main():
    with WebTestFramework(
        headless=False,
        auto_save_interval=30,  # Auto-save every 30 seconds
        idle_timeout=45.0  # 45 seconds max wait for idle
    ) as framework:
        # Create browser context
        framework.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )

        # Run tests with user interaction
        framework.run_tests([
            example_test_interaction
        ])

        # Optional: Manually save state
        framework.save_state("final_state")

        # Simulate long-running scenario to test auto-save
        time.sleep(60)


if __name__ == "__main__":
    main()
