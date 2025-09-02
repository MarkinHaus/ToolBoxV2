import asyncio
import json
import logging
import os
import time
from typing import Any

try:
    from playwright.async_api import Browser as ABrowser
    from playwright.async_api import BrowserContext as ABrowserContext
    from playwright.async_api import Page as APage
    from playwright.async_api import Playwright as APlaywright
    from playwright.async_api import async_playwright
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        sync_playwright,
    )
except ImportError:
    os.system("pip install playwright")

    (ABrowser, ABrowserContext, APage, APlaywright,
     async_playwright, Browser, BrowserContext, Page,
     Playwright, sync_playwright) = None, None, None, None, None, None, None, None, None, None



class AsyncWebTestFramework:
    def __init__(self,
                 browser_type: str = 'chromium',
                 headless: bool = False,
                 state_dir: str = 'tests/test_states',
                 log_level: int = logging.INFO):
        """
        Initialize the web testing framework with enhanced features

        :param browser_type: Type of browser to use
        :param headless: Run browser in headless mode
        :param state_dir: Directory to save and load browser states
        :param log_level: Logging level
        """
        self.last_url = None
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        self.browser_type = browser_type
        self.headless = headless
        self.state_dir = state_dir

        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)

        self.playwright = None
        self.browser: ABrowser | None = None
        self.context: ABrowserContext | None = None
        self.page: APage | None = None

    async def setup(self):
        """
        Set up Playwright and launch browser
        """

        self.playwright: APlaywright = await async_playwright().start()

        # Dynamic browser launch based on type
        browser_launchers = {
            'chromium': self.playwright.chromium.launch,
            'firefox': self.playwright.firefox.launch,
            'webkit': self.playwright.webkit.launch
        }

        self.browser = await browser_launchers.get(self.browser_type, self.playwright.chromium.launch)(
            headless=self.headless,
            timeout=200
        )

    async def create_context(self,
                              viewport: dict[str, int] = None,
                              user_agent: str = None):
        """
        Create a new browser context with optional configuration
        """
        context_options = {}
        if viewport:
            context_options['viewport'] = viewport
        if user_agent:
            context_options['user_agent'] = user_agent

        self.context = await self.browser.new_context(**context_options)
        self.page = await self.context.new_page()

    async def navigate(self, url: str):
        """
        Navigate to a specific URL with idle waiting
        """
        self.last_url = url
        await self.page.goto(url, wait_until='networkidle')

    async def mimic_user_interaction(self, interactions: list[dict[str, Any]]):
        """
        Mimic user interactions using Playwright's API

        :param interactions: List of interaction dictionaries
        """
        report = []
        for interaction in interactions:
            passed = True
            response = f"{interaction.get('type', 'type-missing')} - {interaction.get('selector', interaction.get('path', interaction.get('url')))} - {interaction.get('text', interaction.get('value'))}"
            try:
                # Locate element first
                element = self.page.locator(interaction['selector']) if 'selector' in interaction else None

                # Perform interaction based on type
                if interaction['type'] == 'click':
                    await element.click()
                elif interaction['type'] == 'dblclick':
                    await element.dblclick()
                elif interaction['type'] == 'type':
                    await element.fill(interaction['text'])
                elif interaction['type'] == 'hover':
                    await element.hover()
                elif interaction['type'] == 'sleep':
                    await asyncio.sleep(interaction['time'])
                elif interaction['type'] == 'select':
                    await element.select_option(**interaction['value'])
                elif interaction['type'] == 'check':
                    await element.check()
                elif interaction['type'] == 'goback':
                    await self.navigate(url=self.last_url)
                elif interaction['type'] == 'screenshot':
                    path = os.path.join('tests', 'images', interaction.get('path', 'screenshot.png'))
                    os.makedirs(os.path.join('tests', 'images'), exist_ok=True)
                    await self.page.screenshot(path=path)
                elif interaction['type'] == 'goto':
                    await self.navigate(url=interaction.get('url', '/'))
                elif interaction['type'] == 'evaluate':
                    await self.page.evaluate(interaction['expression'])
                elif interaction['type'] == 'title':
                    response = await self.page.title()
                elif interaction['type'] == 'test':
                    try:
                        await self.page.locator(interaction['selector']).wait_for()
                        response = f"found element {interaction['selector']}"
                    except Exception:
                        response = f"Could not find {interaction['selector']}"

                # Add small wait between interactions
                await asyncio.sleep(0.05)

            except Exception as e:
                self.logger.error(f"Error in user interaction: {e}")
                passed = False
                response = str(e)

            report.append((passed, response))
        return report

    async def save_state(self, state_name: str):
        """
        Save current browser state to a file
        """
        try:
            state_path = os.path.join(self.state_dir, f"{state_name}_state.json")
            state = await self.context.storage_state()

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)

            self.logger.info(f"State saved: {state_path}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    async def load_state(self, state_name: str = None):
        """
        Load a previously saved browser state
        """
        try:
            if not state_name:
                auto_states = [f for f in os.listdir(self.state_dir) if
                               f.startswith('auto_state_') and f.endswith('_state.json')]
                if not auto_states:
                    raise FileNotFoundError("No auto-states found")

                state_name = sorted(auto_states, reverse=True)[0][:-10]

            state_path = os.path.join(self.state_dir, f"{state_name}_state.json")

            if not os.path.exists(state_path):
                raise FileNotFoundError(f"State file {state_path} not found")

            with open(state_path) as f:
                state = json.load(f)

            self.context = await self.browser.new_context(storage_state=state)
            self.page = await self.context.new_page()

            self.logger.info(f"State loaded: {state_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False

    async def teardown(self):
        """
        Close browser and stop Playwright
        """
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def run_tests(self, *tests, evaluation=True):
        results = []
        for test in tests:
            if asyncio.iscoroutinefunction(test):
                interactions = await test(self)
            else:
                interactions = test(self)
            if isinstance(interactions, list) and isinstance(interactions[0], dict):
                results.append(await self.mimic_user_interaction(interactions=interactions))
            if isinstance(interactions, list) and isinstance(interactions[0], tuple):
                results.append(interactions)
            if evaluation:
                self.eval_r(results[-1])
        return results

    @staticmethod
    def eval_r(results):
        for ev, res in results:
            assert (ev, res) == (True, res)

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.teardown()


class WebTestFramework:
    def __init__(self,
                 browser_type: str = 'chromium',
                 headless: bool = False,
                 state_dir: str = 'tests/test_states',
                 log_level: int = logging.INFO):
        """
        Initialize the web testing framework with enhanced features

        :param browser_type: Type of browser to use
        :param headless: Run browser in headless mode
        :param state_dir: Directory to save and load browser states
        :param log_level: Logging level
        """
        # Setup logging
        self.last_url = None
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        self.browser_type = browser_type
        self.headless = headless
        self.state_dir = state_dir

        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)

        self.playwright:Playwright | None = None
        self.browser:Browser | None = None
        self.context:BrowserContext | None = None
        self.page:Page | None  = None

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
            headless=self.headless, timeout=200
        )

    def create_context(self,
                       viewport: dict[str, int] = None,
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

    def navigate(self, url: str):
        """
        Navigate to a specific URL with idle waiting
        """
        self.last_url = url
        self.page.goto(url, wait_until='networkidle')

    def mimic_user_interaction(self, interactions: list[dict[str, Any]]):
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
            {"type": "test", "selector": "#checkbox"}
            {"type": "sleep", "time": seconds}
        ]
        """
        report = []
        for interaction in interactions:
            passed = True
            response = f"{interaction.get('type', 'type-missing')} - {interaction.get('selector', interaction.get('path', interaction.get('url')))} - {interaction.get('text', interaction.get('value'))}"
            try:
                print(f"running interaction {interaction}")
                # Locate element first
                if 'selector' in interaction:
                    element = self.page.locator(interaction['selector'])
                    # element.scroll_into_view_if_needed(timeout=200)
                else:
                    element = None

                # Perform interaction based on type
                if interaction['type'] == 'click':
                    element.click()
                if interaction['type'] == 'dblclick':
                    element.dblclick()
                elif interaction['type'] == 'type':
                    element.fill(interaction['text'])
                elif interaction['type'] == 'hover':
                    element.hover()
                elif interaction['type'] == 'sleep':
                    time.sleep(interaction['time'])
                elif interaction['type'] == 'select':
                    element.select_option(interaction['value'])
                elif interaction['type'] == 'check':
                    element.check()
                elif interaction['type'] == 'goback':
                    self.navigate(url=self.last_url)
                elif interaction['type'] == 'screenshot':
                    path = os.path.join('tests', 'images', interaction.get('path', 'screenshot.png'))
                    os.makedirs(os.path.join('tests', 'images'), exist_ok=True)
                    self.page.screenshot(path=path)
                elif interaction['type'] == 'goto':
                    self.navigate(url=interaction.get('url', '/'))
                elif interaction['type'] == 'evaluate':
                    self.page.evaluate(interaction['expression'])
                elif interaction['type'] == 'title':
                    response = self.page.title()
                elif interaction['type'] == 'test':
                    try:
                        self.page.locator(interaction['selector'])
                        response = f"found element{interaction['selector']}",
                    except Exception:
                        response = f"Could not find {interaction['selector']}",

                # Add small wait between interactions
                time.sleep(0.05)

            except Exception as e:
                self.logger.error(f"Error in user interaction: {e}")
                passed = False
                response = str(e)

            print(passed, response)

            report.append((passed, response))
        return report

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

            with open(state_path) as f:
                state = json.load(f)

            # Create a new context from the saved state
            self.context = self.browser.new_context(storage_state=state)
            self.page = self.context.new_page()

            self.logger.info(f"State loaded: {state_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False

    def teardown(self):
        """
        Close browser and stop Playwright
        """
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def run_tests(self, *tests, evaluation=True):
        results = []
        for test in tests:
            print(f"Testing {test.__name__}")
            interactions = test(self)
            if isinstance(interactions, list) and isinstance(interactions[0], dict):
                results.append(self.mimic_user_interaction(interactions=interactions))
            if isinstance(interactions, list) and isinstance(interactions[0], tuple):
                results.append(interactions)
            if evaluation:
                self.eval_r(results[-1])
        return results

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

    @staticmethod
    def eval_r(results):
        for ev, res in results:
            assert (ev, res) == (True, res)



# Example Usage and Test Cases
def example_test_interaction(f):
    """
    Example test case with user interaction mimicking
    """
    # Define a series of user interactions
    return [
        {"type": "goto", "url": "https://example.com"},
        {"type": "screenshot", "path": "screenshot.png"}
    ]

def main():
    with WebTestFramework(
        headless=True,
    ) as framework:
        # Create browser context
        framework.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )

        # Run tests with user interaction
        framework.run_tests(*[
            example_test_interaction
        ])

        # Optional: Manually save state
        framework.save_state("final_state")


if __name__ == "__main__":
    main()
