from playwright.async_api import async_playwright, Browser, Playwright, Page
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class BrowserContext:
    """Stores browser session state"""
    cookies: List[Dict] = None
    local_storage: Dict[str, str] = None
    session_storage: Dict[str, str] = None
    viewport: Dict[str, int] = None
    user_agent: str = None


@dataclass
class JSExecutionRecord:
    """Records JavaScript execution details"""
    code: str
    result: Any
    error: Optional[str] = None
    page_state: Optional[Dict] = None
    extracted_data: Optional[Dict] = None


class BrowserSession:
    """Manages browser context and state"""

    def __init__(self, verbose=False):
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.browser: Optional[Browser] = None
        self.playwright: Optional[Playwright]  = None
        self.verbose = verbose

    async def initialize(self):
        """Initialize browser session"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=not self.verbose)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def close(self):
        """Clean up browser session"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def get_context_state(self) -> BrowserContext:
        """Capture current browser context state"""
        if not self.context:
            return BrowserContext()

        cookies = await self.context.cookies()
        storage = await self.page.evaluate("""() => {
            return {
                localStorage: Object.assign({}, window.localStorage),
                sessionStorage: Object.assign({}, window.sessionStorage)
            }
        }""")

        viewport = self.page.viewport_size
        user_agent = await self.page.evaluate('navigator.userAgent')

        return BrowserContext(
            cookies=cookies,
            local_storage=storage['localStorage'],
            session_storage=storage['sessionStorage'],
            viewport=viewport,
            user_agent=user_agent
        )

    async def restore_context(self, state: BrowserContext):
        """Restore browser context from saved state"""
        if state.cookies:
            await self.context.add_cookies(state.cookies)

        if state.local_storage or state.session_storage:
            await self.page.evaluate("""(storage) => {
                Object.assign(window.localStorage, storage.localStorage || {});
                Object.assign(window.sessionStorage, storage.sessionStorage || {});
            }""", {
                'localStorage': state.local_storage,
                'sessionStorage': state.session_storage
            })

        if state.viewport:
            await self.page.set_viewport_size(state.viewport)

        if state.user_agent:
            await self.context.set_extra_http_headers({'User-Agent': state.user_agent})


@dataclass
class ExtractionPattern:
    selector: str
    type: str = 'text'  # 'text', 'attribute', 'html'
    attribute: str = None
    multiple: bool = False


class Extractor:
    """Custom extractor using Playwright selectors"""

    def __init__(self, page):
        self.page = page

    async def extract(self, patterns: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Extract data using patterns
        patterns format: {
            'key': {
                'selector': '.css-selector',
                'type': 'text|attribute|html',
                'attribute': 'href',  # only for type='attribute'
                'multiple': False  # whether to extract multiple elements
            }
        }
        """
        results = {}
        for key, pattern in patterns.items():
            pattern = ExtractionPattern(**pattern)
            try:
                if pattern.multiple:
                    elements = await self.page.query_selector_all(pattern.selector)
                    results[key] = await self._extract_elements(elements, pattern)
                else:
                    element = await self.page.query_selector(pattern.selector)
                    results[key] = await self._extract_element(element, pattern)
            except Exception as e:
                results[key] = f"Extraction error: {str(e)}"

        return results

    async def _extract_element(self, element, pattern: ExtractionPattern) -> Any:
        """Extract data from a single element"""
        if element is None:
            return None

        if pattern.type == 'text':
            return await element.text_content()
        elif pattern.type == 'html':
            return await element.inner_html()
        elif pattern.type == 'attribute':
            return await element.get_attribute(pattern.attribute)
        else:
            raise ValueError(f"Unknown extraction type: {pattern.type}")

    async def _extract_elements(self, elements, pattern: ExtractionPattern) -> List[Any]:
        """Extract data from multiple elements"""
        return [await self._extract_element(el, pattern) for el in elements]


js_prompt = '''
JavaScript Context Usage Guide:

1. Basic Page Interaction:
{
    'content': """
const heading = document.querySelector('h1');
const text = heading.textContent;
await new Promise(r => setTimeout(r, 1000));  // Wait for 1 second
return text;
""",
context : {
    'lang': 'js',
    'url': 'https://example.com'  # Optional: page to navigate to
    }
}

2. Data Extraction Patterns:
{
    'content': """
const result = document.querySelector('button.login').click();
return result;
""",
context : {
    'lang': 'js',
    'patterns': {
        'page_title': {
        'selector': 'h1',
            'type': 'text'
        },
        'main_image': {
        'selector': '.hero-image',
            'type': 'attribute',
            'attribute': 'src'
        },
        'navigation_links': {
        'selector': 'nav a',
            'type': 'text',
            'multiple': True
        },
        'article_content': {
        'selector': '.article-body',
            'type': 'html'
        }
    }
}


3. Element Waiting and Collection:
{
    'content': """
const heading = await waitForElement('h1');
const headingText = heading.textContent;

const links = Array.from(document.querySelectorAll('a')).map(a => ({
    text: a.textContent,
    href: a.href
}));

return {
    heading: headingText,
    links: links
};
""",
context : {
    'lang': 'js',
    'url': 'https://example.com'
}

IMPORTANT NOTES:
1. Use `document.querySelector()` for single elements
2. Use `document.querySelectorAll()` for multiple elements
3. `waitForElement()` helper available for dynamic content
4. Return values are automatically formatted
5. Patterns extract data automatically after code execution
6. Async/await supported for timing operations

Context Options:
1. 'lang': 'js' (required)
2. 'url': URL to navigate to (preferred)
3. 'patterns': Data extraction patterns (preferred)

Pattern Types:
1. 'text': Extract text content
2. 'attribute': Extract specific attribute
3. 'html': Extract HTML content
4. 'multiple': Extract multiple elements'''
