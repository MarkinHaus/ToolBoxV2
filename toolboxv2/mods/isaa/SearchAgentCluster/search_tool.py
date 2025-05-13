import asyncio
import base64
import json
import os
import re
import urllib
from datetime import datetime
from typing import Any, TypeVar
from urllib.parse import urlparse

# Import BrowserAnt components
from browser_use import Controller, SystemPrompt
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from toolboxv2.mods.isaa.CodingAgent.live import BrowserWrapper

T = TypeVar('T', bound=BaseModel)
global_headless = False

class WebScraperConfig(BaseModel):
    """Configuration for web scraper operations"""
    max_concurrent_tabs: int = 5
    default_timeout: float = 30000
    scroll_delay: int = 500
    initial_delay: int = 1000
    viewport_height: int = 900
    viewport_width: int = 1600
    wait_for_selectors: bool = True
    auto_scroll: bool = True
    save_screenshots: bool = False
    screenshot_dir: str = "./screenshots"
    extract_markdown: bool = True
    extract_text: bool = True
    extract_html: bool = False
    headless: bool = False
    disable_images: bool = False
    user_agent: str | None = None


class WebScraper:
    """
    A high-performance web scraper using BrowserAnt with multi-tab parallel processing.
    Handles both structured and unstructured data collection efficiently.
import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional

# Define a structured data model
class ProductInfo(BaseModel):
    title: str
    price: str
    description: Optional[str] = None
    rating: Optional[str] = None
    availability: Optional[str] = None

async def main():
    # Initialize the scraper
    scraper = WebScraper()

    # Example 1: Simple scraping of a single URL
    result = await scraper.scrape_url("https://example.com")
    print(f"Title: {result['title']}")
    print(f"Content: {result['markdown'][:200]}...")

    # Example 2: Parallel scraping of multiple URLs
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3"
    ]
    results = await scraper.scrape_urls(urls)

    # Example 3: Structured data extraction
    products = await scraper.scrape_structured_data(
        urls=["https://example.com/product1", "https://example.com/product2"],
        model=ProductInfo,
        extraction_task="Extract product information including title, price, and availability status"
    )

    for product in products:
        if product:
            print(f"Product: {product.title}, Price: {product.price}")

    # Clean up
    await scraper.close()
    """

    def __init__(
        self,
        config: WebScraperConfig = WebScraperConfig(),
        llm: str | BaseChatModel | None = None,
        chrome_path: str | None = None,
        remote_url: str | None = None,
        browser_config: dict[str, Any] | None = None
    ):
        """
        Initialize the web scraper with configuration.

        Args:
            config: Configuration for scraper behavior
            llm: Language model for intelligent data extraction
            chrome_path: Path to Chrome executable
            remote_url: URL for remote browser connection
            browser_config: Additional browser configuration
        """
        self.config = config
        self.browser_wrapper = BrowserWrapper(
            llm=llm,
            headless=config.headless,
            chrome_path=chrome_path,
            remote_url=remote_url,
            config=browser_config
        )
        self.active_tasks = set()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_tabs)
        self._results = {}

        # Create screenshot directory if needed
        if config.save_screenshots and not os.path.exists(config.screenshot_dir):
            os.makedirs(config.screenshot_dir)

    async def initialize(self):
        """Initialize the browser if not already initialized"""
        await self.browser_wrapper.initialize()

    async def close(self):
        """Close the browser and clean up resources"""
        # Wait for all active tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks)
        await self.browser_wrapper.close()


    # Add this method to your WebScraper class
    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = True,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_links: bool = False,
        save_to_file: str | None = None
    ) -> dict[str, Any]:
        """
        Perform a comprehensive web search and return high-quality data for the given query.

        Args:
            query: Search query string
            max_results: Maximum number of results to process (default: 5)
            include_content: Whether to include full content from result pages (default: True)
            extract_images: Whether to extract images from result pages (default: False)
            extract_tables: Whether to extract tables from result pages (default: False)
            extract_links: Whether to extract links from result pages (default: False)
            save_to_file: Path to save results as JSON (optional)

        Returns:
            Dictionary containing search results and extracted information
        """
        await self.initialize()
        try:
            start_time = datetime.now()

            # Try different search engines in order
            search_engines = [
                {
                    "url": f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}",
                    "result_selector": ".g",
                    "title_selector": "h3",
                    "link_selector": "a",
                    "snippet_selector": ".VwiC3b",
                    "name": "google"
                },
                {
                    "url": f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}",
                    "result_selector": ".b_algo",
                    "title_selector": "h2",
                    "link_selector": "a",
                    "snippet_selector": ".b_caption p",
                    "name": "bing"
                },
                {
                    "url": f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}",
                    "result_selector": ".result",
                    "title_selector": "h2",
                    "link_selector": "a.result__a",
                    "snippet_selector": ".result__snippet",
                    "name": "duckduckgo"
                }
            ]

            results = []

            for engine in search_engines:
                try:
                    # Navigate to search engine
                    page = await self.browser_wrapper.navigate(engine["url"])
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(2000)  # Wait for results to load

                    # Extract search results
                    search_results = await page.evaluate(
                        """
                        (selectors) => {
                            const results = [];
                            const elements = document.querySelectorAll(selectors.result_selector);

                            for (const element of elements) {
                                const titleElement = element.querySelector(selectors.title_selector);
                                const linkElement = element.querySelector(selectors.link_selector);
                                const snippetElement = element.querySelector(selectors.snippet_selector);

                                if (titleElement && linkElement) {
                                    const url = linkElement.href;
                                    // Skip non-http links and same-domain results
                                    if (url && url.startsWith('http') &&
                                        !url.includes('google.com/search') &&
                                        !url.includes('bing.com/search') &&
                                        !url.includes('duckduckgo.com')) {
                                        results.push({
                                            title: titleElement.textContent.trim(),
                                            url: url,
                                            snippet: snippetElement ? snippetElement.textContent.trim() : '',
                                            source: selectors.name
                                        });
                                    }
                                }
                            }
                            return results;
                        }
                        """,
                        engine
                    )

                    if search_results and len(search_results) > 0:
                        # We got results, add them and break
                        results = search_results
                        break

                except Exception as e:
                    print(f"Error searching with {engine['name']}: {str(e)}")
                    continue  # Try next engine

            # Filter and limit results
            unique_urls = set()
            filtered_results = []

            for result in results:
                if result['url'] not in unique_urls and len(filtered_results) < max_results:
                    unique_urls.add(result['url'])
                    filtered_results.append(result)

            results = filtered_results

            # Get detailed content if requested
            if include_content and results:
                # Extract content from each result page
                urls_to_scrape = [result['url'] for result in results]

                # Configure what to extract
                extract_config = {}
                if extract_tables:
                    extract_config['tables'] = 'table'
                if extract_images:
                    extract_config['images'] = 'img'
                if extract_links:
                    extract_config['links'] = 'a'

                # Scrape all pages in parallel using our efficient multi-tab approach
                scraped_data = await self.scrape_urls(
                    urls_to_scrape,
                    extract_config=extract_config if extract_config else None
                )

                # Add content to results
                for i, result in enumerate(results):
                    if i < len(scraped_data) and 'error' not in scraped_data[i]:
                        result['content'] = {
                            'title': scraped_data[i].get('title', result['title']),
                            'markdown': scraped_data[i].get('markdown', ''),
                            'text': scraped_data[i].get('text', ''),
                        }

                        # Add structured data if available
                        if extract_config and 'structured_data' in scraped_data[i]:
                            structured_data = scraped_data[i]['structured_data']
                            for key, value in structured_data.items():
                                if value:  # Only add non-empty data
                                    result['content'][key] = value

            # Prepare final response
            response = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_results': len(results),
                'results': results,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }

            # Save to file if requested
            if save_to_file:
                os.makedirs(os.path.dirname(os.path.abspath(save_to_file)), exist_ok=True)
                with open(save_to_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent=2)

            return response

        finally:
            # Make sure we clean up browser resources
            await self.close()

    async def _scrape_url(self, url: str, task_id: str, extract_config: dict[str, Any] = None):
        """
        Internal method to scrape a single URL

        Args:
            url: URL to scrape
            task_id: Unique identifier for this scraping task
            extract_config: Configuration for what/how to extract
        """
        try:
            async with self._semaphore:
                # Navigate to the URL
                page = await self.browser_wrapper.navigate(url)

                # Wait for network to become idle
                await page.wait_for_load_state("networkidle")

                # Perform initial delay
                if self.config.initial_delay > 0:
                    await page.wait_for_timeout(self.config.initial_delay)

                # Auto-scroll if configured
                if self.config.auto_scroll:
                    await self._auto_scroll(page)

                # Initialize result dictionary
                result = {
                    "url": url,
                    "title": await page.title(),
                    "timestamp": datetime.now().isoformat(),
                }

                # Take screenshot if needed
                if self.config.save_screenshots:
                    file_name = f"{urlparse(url).netloc}_{task_id}.png"
                    screenshot_path = os.path.join(self.config.screenshot_dir, file_name)
                    result["screenshot"] = screenshot_path
                    await self.browser_wrapper.take_scrolling_screenshot(
                        page=page,
                        path=screenshot_path,
                        initial_delay=0,  # We've already waited
                        scroll_delay=self.config.scroll_delay
                    )

                # Extract content based on configuration
                if extract_config:
                    result["structured_data"] = await self.browser_wrapper.extract_structured_content(
                        page=page,
                        config=extract_config
                    )

                # Extract markdown if configured
                if self.config.extract_markdown:
                    result["markdown"] = await self.browser_wrapper.extract_markdown(page=page)

                # Extract text if configured
                if self.config.extract_text:
                    result["text"] = await self.browser_wrapper.extract_text(page=page)

                # Extract HTML if configured
                if self.config.extract_html:
                    result["html"] = await page.content()

                self._results[task_id] = result
                return result

        except Exception as e:
            self._results[task_id] = {"error": str(e), "url": url}
            return {"error": str(e), "url": url}

    async def _auto_scroll(self, page):
        """Automatically scroll down the page to load lazy content"""
        try:
            # Get page dimensions
            dimensions = await page.evaluate("""
                () => {
                    return {
                        width: document.documentElement.scrollWidth,
                        height: document.documentElement.scrollHeight,
                        windowHeight: window.innerHeight
                    }
                }
            """)

            # Scroll down the page gradually
            current_position = 0
            while current_position < dimensions['height']:
                await page.evaluate(f"window.scrollTo(0, {current_position})")
                await page.wait_for_timeout(self.config.scroll_delay)
                current_position += dimensions['windowHeight'] // 2

            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
        except Exception as e:
            print(f"Error during auto-scroll: {e}")

    async def scrape_url(self, url: str, extract_config: dict[str, Any] = None) -> dict[str, Any]:
        """
        Scrape a single URL and return the results

        Args:
            url: URL to scrape
            extract_config: Configuration for structured data extraction

        Returns:
            Dictionary containing scraped data
        """
        await self.initialize()
        task_id = f"{len(self._results)}_{datetime.now().timestamp()}"
        result = await self._scrape_url(url, task_id, extract_config)
        return result

    async def scrape_urls(
        self,
        urls: list[str],
        extract_config: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Scrape multiple URLs in parallel and return all results

        Args:
            urls: List of URLs to scrape
            extract_config: Configuration for structured data extraction

        Returns:
            List of dictionaries containing scraped data
        """
        await self.initialize()
        tasks = []

        for i, url in enumerate(urls):
            task_id = f"{i}_{datetime.now().timestamp()}"
            task = asyncio.create_task(self._scrape_url(url, task_id, extract_config))
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]

    async def scrape_structured_data(
        self,
        urls: list[str],
        model: type[T],
        extraction_task: str = None
    ) -> list[T]:
        """
        Scrape and parse structured data into pydantic models

        Args:
            urls: List of URLs to scrape
            model: Pydantic model class for structured data
            extraction_task: Natural language description of what to extract

        Returns:
            List of parsed data objects
        """
        await self.initialize()

        # Create intelligent extraction task if provided
        if extraction_task:
            # Create a custom system prompt for extraction
            class ExtractionPrompt(SystemPrompt):
                def important_rules(self) -> str:
                    existing_rules = super().important_rules()
                    new_rules = f"""
                    9. EXTRACTION GOAL:
                    - Your primary goal is to extract data according to this specific task: {extraction_task}
                    - You should carefully identify and extract the information as accurately as possible.
                    - Focus only on relevant information that matches the specified data structure.
                    """
                    return f'{existing_rules}\n{new_rules}'

            # Define the extraction task for each URL
            tasks = []
            for url in urls:
                # Setup intelligent extraction for each URL
                task = asyncio.create_task(self._run_extraction_agent(
                    url=url,
                    model=model,
                    extraction_task=extraction_task,
                    system_prompt_class=ExtractionPrompt
                ))
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                tasks.append(task)

            # Wait for all extractions to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r if not isinstance(r, Exception) else None for r in results]
        else:
            # Manual extraction based on model fields
            field_selectors = {}
            for field_name in model.__annotations__:
                # Convert field name to likely CSS selector
                snake_case = field_name
                selector = f".{snake_case.replace('_', '-')}, #{snake_case.replace('_', '-')}"
                field_selectors[field_name] = selector

            # Scrape with these selectors
            raw_results = await self.scrape_urls(urls, extract_config=field_selectors)

            # Convert to pydantic models
            parsed_results = []
            for result in raw_results:
                try:
                    if "structured_data" in result and "error" not in result:
                        # Map the extracted data to model fields
                        model_data = {}
                        for field_name in model.__annotations__:
                            if field_name in result["structured_data"]:
                                field_value = result["structured_data"][field_name]
                                if isinstance(field_value, list) and len(field_value) > 0:
                                    model_data[field_name] = field_value[0]  # Take first match
                                else:
                                    model_data[field_name] = field_value

                        # Create the model instance
                        parsed_results.append(model(**model_data))
                    else:
                        parsed_results.append(None)
                except Exception as e:
                    print(f"Error parsing result: {e}")
                    parsed_results.append(None)

            return parsed_results

    async def _run_extraction_agent(
        self,
        url: str,
        model: type[T],
        extraction_task: str,
        system_prompt_class: type[SystemPrompt]
    ) -> T:
        """Run an intelligent agent to extract structured data"""
        # Define output model for the agent
        controller = Controller(output_model=model)

        # Create the task description
        fields_info = "\n".join([f"- {field}: {model.__annotations__[field].__name__}"
                                 for field in model.__annotations__])

        task = f"""
        Go to {url} and extract the following information:
        {fields_info}

        Specific extraction instructions: {extraction_task}
        """

        # Create and run the agent
        agent = await self.browser_wrapper.create_agent(task=task)
        agent._controller = controller
        agent._system_prompt_class = system_prompt_class

        history = await agent.run()

        # Parse the result
        result = history.final_result()
        if result:
            try:
                return model.model_validate_json(result)
            except Exception as e:
                print(f"Error parsing agent result: {e}")
                return None
        return None


class WebContentParser:
    """Utility class for parsing web content using BrowserAnt"""

    def __init__(self, browser_wrapper: BrowserWrapper):
        """Initialize with a browser wrapper"""
        self.browser_wrapper = browser_wrapper

    async def extract_article(self, url: str) -> dict[str, Any]:
        """Extract article content with title, text, and metadata"""
        await self.browser_wrapper.initialize()
        page = await self.browser_wrapper.navigate(url)

        # Execute readability.js to extract article content
        readability_js = """
        function extractArticle() {
            // Simple article extraction logic
            const article = {
                title: document.title,
                byline: '',
                content: '',
                textContent: '',
                excerpt: '',
                siteName: '',
                publishedTime: ''
            };

            // Try to find article elements
            const articleElement = document.querySelector('article') ||
                                   document.querySelector('main') ||
                                   document.querySelector('.post-content') ||
                                   document.querySelector('.entry-content');

            if (articleElement) {
                article.content = articleElement.innerHTML;
                article.textContent = articleElement.textContent;
            } else {
                // Fallback to body content
                article.content = document.body.innerHTML;
                article.textContent = document.body.textContent;
            }

            // Try to extract metadata
            const metaTags = document.querySelectorAll('meta');
            metaTags.forEach(tag => {
                const property = tag.getAttribute('property') || tag.getAttribute('name');
                const content = tag.getAttribute('content');

                if (property && content) {
                    if (property === 'og:site_name') article.siteName = content;
                    if (property === 'og:title' && !article.title) article.title = content;
                    if (property === 'og:description' && !article.excerpt) article.excerpt = content;
                    if (property === 'article:published_time') article.publishedTime = content;
                    if (property === 'author' || property === 'article:author') article.byline = content;
                }
            });

            // Extract first paragraph as excerpt if not found
            if (!article.excerpt) {
                const paragraphs = document.querySelectorAll('p');
                if (paragraphs.length > 0) {
                    for (let i = 0; i < paragraphs.length; i++) {
                        const text = paragraphs[i].textContent.trim();
                        if (text.length > 50) {
                            article.excerpt = text;
                            break;
                        }
                    }
                }
            }

            return article;
        }

        return extractArticle();
        """

        # Extract article content
        article = await page.evaluate(readability_js)

        # Add markdown version
        article['markdown'] = await self.browser_wrapper.extract_markdown(page)

        # Take a screenshot
        screenshot_data = await self.browser_wrapper.take_scrolling_screenshot(page)
        article['screenshot'] = base64.b64encode(screenshot_data).decode('utf-8')

        return article

    async def extract_table_data(self, url: str, table_selector: str = 'table') -> list[dict[str, Any]]:
        """Extract tabular data from a webpage"""
        await self.browser_wrapper.initialize()
        page = await self.browser_wrapper.navigate(url)

        # Script to extract table data
        extract_table_js = """
        (tableSelector) => {
            const tables = document.querySelectorAll(tableSelector);
            if (tables.length === 0) return [];

            // Use the first table found
            const table = tables[0];
            const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());

            // If no headers found, try using the first row
            const headerRow = headers.length > 0 ? headers :
                            Array.from(table.querySelectorAll('tr:first-child td')).map(td => td.textContent.trim());

            const rows = Array.from(table.querySelectorAll('tr'));
            const result = [];

            // Start from 1 if we have headers, otherwise from 0
            const startIdx = headers.length > 0 ? 1 : 0;

            for (let i = startIdx; i < rows.length; i++) {
                const row = rows[i];
                const cells = Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim());

                if (cells.length > 0) {
                    const rowData = {};
                    for (let j = 0; j < Math.min(headerRow.length, cells.length); j++) {
                        // Create a valid object key from header
                        const key = headerRow[j].replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
                        rowData[key] = cells[j];
                    }
                    result.push(rowData);
                }
            }

            return result;
        }
        """

        # Extract data
        table_data = await page.evaluate(extract_table_js, table_selector)
        return table_data

    async def extract_links(self, url: str, link_selector: str = 'a') -> list[dict[str, str]]:
        """Extract all links from a webpage"""
        await self.browser_wrapper.initialize()
        page = await self.browser_wrapper.navigate(url)

        # Script to extract links
        extract_links_js = """
        (linkSelector) => {
            const links = Array.from(document.querySelectorAll(linkSelector));
            return links.map(link => {
                return {
                    text: link.textContent.trim(),
                    href: link.href,
                    title: link.getAttribute('title') || '',
                    isExternal: link.hostname !== window.location.hostname
                };
            }).filter(link => link.href && link.href.startsWith('http'));
        }
        """

        # Extract links
        links = await page.evaluate(extract_links_js, link_selector)
        return links


import logging
from urllib.parse import urljoin


async def scrape_documentation_to_markdown(
    start_url: str,
    topic: str | None = None,
    max_pages: int = 30,
    max_depth: int = 3,
    output_dir: str | None = None,
    toc_filename: str = "table_of_contents.md"
) -> dict[str, str]:
    """
    Recursively scrape documentation pages starting from a URL,
    focused on a specific topic, and convert to Markdown.

    Args:
        start_url: The documentation homepage or entry point
        topic: The topic to focus on (e.g., "streaming", "authentication")
        max_pages: Maximum number of pages to scrape
        max_depth: Maximum depth of link traversal
        output_dir: Directory to save the MD files (if None, just returns them)
        toc_filename: Filename for the table of contents

    Returns:
        Dictionary mapping page titles to markdown content
    """
    # Initialize scraper with efficient settings for docs
    scraper_config = WebScraperConfig(
        max_concurrent_tabs=5,
        headless=global_headless,
        disable_images=True,
        extract_html=False,
        auto_scroll=True,
        scroll_delay=300,
        initial_delay=500,
        save_screenshots=False
    )

    scraper = WebScraper(config=scraper_config)
    await scraper.initialize()

    # Track visited and pending URLs
    visited_urls: set[str] = set()
    pending_urls: list[dict] = [{"url": start_url, "depth": 0, "parent": None}]
    results: dict[str, dict] = {}
    domain = urlparse(start_url).netloc

    logging.info(f"Starting documentation scrape from {start_url}")
    if topic:
        logging.info(f"Focusing on topic: {topic}")

    # Create a regular expression pattern for topic if provided
    topic_pattern = re.compile(rf'\b{re.escape(topic)}\b', re.IGNORECASE) if topic else None

    try:
        # Process URLs breadth-first until we hit max pages or have no more URLs
        while pending_urls and len(results) < max_pages:
            # Get the next URL to process
            current = pending_urls.pop(0)
            current_url = current["url"]
            current_depth = current["depth"]

            # Skip if we've already visited this URL
            if current_url in visited_urls:
                continue

            logging.info(f"Scraping: {current_url} (depth: {current_depth})")
            visited_urls.add(current_url)

            # Scrape the current page
            page_result = await scraper.scrape_url(current_url)

            # Skip pages with errors
            if "error" in page_result:
                logging.warning(f"Error scraping {current_url}: {page_result['error']}")
                continue

            # Check if page is relevant to the topic
            is_relevant = True
            if topic_pattern:
                markdown_content = page_result.get("markdown", "")
                text_content = page_result.get("text", "")

                # Check if topic appears in title, URL, or content
                has_topic_in_title = topic_pattern.search(page_result.get("title", ""))
                has_topic_in_url = topic_pattern.search(current_url)
                has_topic_in_content = (
                    topic_pattern.search(markdown_content) or
                    topic_pattern.search(text_content)
                )

                is_relevant = has_topic_in_title or has_topic_in_url or has_topic_in_content

            # Process this page if it's relevant
            if is_relevant:
                # Extract title and content
                title = page_result.get("title", f"Page {len(results) + 1}")

                # Store the result
                results[current_url] = {
                    "title": title,
                    "markdown": page_result.get("markdown", ""),
                    "depth": current_depth,
                    "parent": current["parent"]
                }

                # Only proceed deeper if we haven't hit max depth
                if current_depth < max_depth:
                    # Extract links to follow
                    parser = scraper.browser_wrapper.get_parser()
                    links = await parser.extract_links(current_url)

                    # Filter links for internal documentation pages
                    doc_links = []
                    for link in links:
                        link_url = link["href"]
                        parsed_url = urlparse(link_url)

                        # Only include links to the same domain
                        if parsed_url.netloc == domain or not parsed_url.netloc:
                            # Normalize URL
                            if not parsed_url.netloc:
                                link_url = urljoin(current_url, link_url)

                            # Skip anchor links to same page
                            if link_url.split('#')[0] == current_url.split('#')[0]:
                                continue

                            # Skip non-documentation links (common patterns)
                            skip_patterns = [
                                r'(\.pdf|\.zip|\.tar|\.gz)$',  # Downloads
                                r'/search/',  # Search pages
                                r'/login/',  # Auth pages
                                r'/logout/',  # Auth pages
                                r'/tag/',  # Tag pages
                                r'/version/',  # Version switching
                                r'/latest/',  # Version switching
                                r'/download/',  # Download pages
                                r'/contact/',  # Contact pages
                                r'/blog/',  # Blog posts (unless that's what we want)
                            ]

                            should_skip = any(re.search(pattern, link_url) for pattern in skip_patterns)
                            if should_skip:
                                continue

                            # Check if it's potentially relevant to the topic
                            is_potentially_relevant = True
                            if topic_pattern:
                                has_topic_in_link_text = topic_pattern.search(link["text"])
                                has_topic_in_link_url = topic_pattern.search(link_url)
                                is_potentially_relevant = has_topic_in_link_text or has_topic_in_link_url

                            # Add to pending if it's potentially relevant and not already visited
                            if is_potentially_relevant and link_url not in visited_urls:
                                doc_links.append({
                                    "url": link_url,
                                    "depth": current_depth + 1,
                                    "parent": current_url
                                })

                    # Add the filtered links to our pending list
                    pending_urls.extend(doc_links)

        # Generate markdown output
        markdown_results = {}

        # Create a hierarchy for building a table of contents
        pages_hierarchy = {}
        for url, page_data in results.items():
            title = page_data["title"]
            markdown = page_data["markdown"]

            # Add page URL reference at the bottom
            markdown += f"\n\n---\n*Source: [{url}]({url})*"

            # Add to outputs
            markdown_results[url] = markdown

            # Track in hierarchy for TOC
            depth = page_data["depth"]
            parent = page_data["parent"]

            if depth not in pages_hierarchy:
                pages_hierarchy[depth] = []

            pages_hierarchy[depth].append({
                "url": url,
                "title": title,
                "parent": parent
            })

        # Generate table of contents
        toc = f"# Documentation: {topic if topic else 'All Topics'}\n\n"
        toc += f"*Generated from: [{start_url}]({start_url})*\n\n"
        toc += "## Table of Contents\n\n"

        # Sort by depth to build hierarchy
        for depth in sorted(pages_hierarchy.keys()):
            pages = pages_hierarchy[depth]

            for page in pages:
                # Calculate indentation based on depth
                indent = "  " * depth
                page_filename = sanitize_filename(page["title"]) + ".md"
                toc += f"{indent}- [{page['title']}]({page_filename})\n"

        # Save the results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Write the TOC file
            with open(os.path.join(output_dir, toc_filename), "w", encoding="utf-8") as f:
                f.write(toc)

            # Write each page file
            for url, content in markdown_results.items():
                page_title = results[url]["title"]
                filename = sanitize_filename(page_title) + ".md"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            logging.info(f"Saved {len(markdown_results)} documentation pages to {output_dir}")

        # Include the TOC in the results
        markdown_results["table_of_contents"] = toc

        return markdown_results

    finally:
        # Make sure we clean up browser resources
        await scraper.close()


def sanitize_filename(filename: str) -> str:
    """Convert a string to a valid filename"""
    # Replace spaces with underscores and remove invalid characters
    sanitized = re.sub(r'[^\w\s-]', '', filename).strip().lower()
    return re.sub(r'[-\s]+', '_', sanitized)


# Example usage
async def main():
    # Example: Scrape Anthropic Python SDK docs about streaming
    docs = await scrape_documentation_to_markdown(
        start_url="https://docs.anthropic.com/claude/reference/getting-started-with-the-api",
        topic="streaming",
        output_dir="./anthropic_docs"
    )

    print(f"Scraped {len(docs)} documentation pages")

    # Print the table of contents
    print("\nTable of Contents:")
    print(docs.get("table_of_contents", "No table of contents generated"))



from pydantic import BaseModel


# Example model for structured data extraction
class SearchResult(BaseModel):
    title: str
    url: str
    content_excerpt: str | None = None
    source: str

async def web_search(query: str):
    scraper = WebScraper(
        config=WebScraperConfig(
            headless=global_headless,  # Set to True in production
            auto_scroll=True,
            extract_markdown=True,
            initial_delay=1000
        )
    )

    try:
        # Simple search with content extraction
        results = await scraper.search_web(
            query="climate change latest research",
            max_results=3,
            include_content=True,
            extract_tables=True,
            save_to_file="search_results.json"
        )

    finally:
        # Clean up
        await scraper.close()

    return results


async def main2():
    # Initialize the web scraper
    scraper = WebScraper(
        chrome_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        config=WebScraperConfig(
            headless=global_headless,  # Set to True in production
            auto_scroll=True,
            extract_markdown=True,
            initial_delay=1000
        )
    )

    try:
        # Simple search with content extraction
        results = await scraper.search_web(
            query="climate change latest research",
            max_results=15,
            include_content=True,
            extract_tables=True,
            save_to_file="search_results.json"
        )

        # Print search results
        print(f"Found {results['num_results']} results for '{results['query']}'")
        for i, result in enumerate(results['results']):
            print(f"\n--- Result {i + 1} ---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Source: {result['source']}")
            if 'content' in result:
                # Print a preview of the content
                content = result['content']['markdown']
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"\nContent Preview: {preview}")

                # Check for tables
                if 'tables' in result['content']:
                    print(f"\nFound {len(result['content']['tables'])} tables in the content")

        print(f"\nSearch completed in {results['execution_time']:.2f} seconds")

    finally:
        # Clean up
        await scraper.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main2())
