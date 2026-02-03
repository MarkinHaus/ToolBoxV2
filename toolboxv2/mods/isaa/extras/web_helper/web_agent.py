"""
WebAgent - Erweitertes Web-Scraping mit SearXNG, Agent-Tools und Preview
=========================================================================

Features:
- SearXNG Integration (Google Dorks, Multi-Engine Search)
- Clevere Agent-Tools (Navigation, Extraktion, Interaktion)
- Detaillierte strukturierte Logs
- Preview-Modus (headless=False für Debugging)
- Strukturierter Markdown Output

Basiert auf AsyncWebTestFramework.
"""

import asyncio
import json
import logging
import os
import re
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any
from urllib.parse import urljoin, urlparse, quote_plus

try:
    from playwright.async_api import Browser as ABrowser
    from playwright.async_api import BrowserContext as ABrowserContext
    from playwright.async_api import Page as APage
    from playwright.async_api import Playwright as APlaywright
    from playwright.async_api import async_playwright
except ImportError:
    os.system("pip install playwright && playwright install chromium")
    from playwright.async_api import (
        Browser as ABrowser, BrowserContext as ABrowserContext,
        Page as APage, Playwright as APlaywright, async_playwright
    )


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    ACTION = "ACTION"
    SEARCH = "SEARCH"
    EXTRACT = "EXTRACT"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """Strukturierter Log-Eintrag."""
    timestamp: str
    level: LogLevel
    component: str
    message: str
    data: dict = field(default_factory=dict)
    duration_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "data": self.data,
            "duration_ms": self.duration_ms
        }

    def __str__(self) -> str:
        icon = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.ACTION: "▶️",
            LogLevel.SEARCH: "🔎",
            LogLevel.EXTRACT: "📄",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌"
        }.get(self.level, "•")

        duration = f" [{self.duration_ms:.0f}ms]" if self.duration_ms > 0 else ""
        return f"{icon} [{self.level.value}] {self.component}: {self.message}{duration}"


class AgentLogger:
    """Detaillierter strukturierter Logger für den WebAgent."""

    def __init__(self, name: str = "WebAgent", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.entries: list[LogEntry] = []
        self._start_times: dict[str, float] = {}

        # Python logger für Konsole
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    def _log(self, level: LogLevel, component: str, message: str,
             data: dict = None, duration_ms: float = 0):
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            component=component,
            message=message,
            data=data or {},
            duration_ms=duration_ms
        )
        self.entries.append(entry)

        if self.verbose or level in (LogLevel.WARNING, LogLevel.ERROR, LogLevel.ACTION):
            self._logger.info(str(entry))

    def start_timer(self, key: str):
        """Timer starten für Duration-Messung."""
        self._start_times[key] = asyncio.get_event_loop().time() * 1000

    def stop_timer(self, key: str) -> float:
        """Timer stoppen und Duration zurückgeben."""
        if key in self._start_times:
            duration = (asyncio.get_event_loop().time() * 1000) - self._start_times[key]
            del self._start_times[key]
            return duration
        return 0

    def debug(self, component: str, message: str, data: dict = None):
        self._log(LogLevel.DEBUG, component, message, data)

    def info(self, component: str, message: str, data: dict = None):
        self._log(LogLevel.INFO, component, message, data)

    def action(self, component: str, message: str, data: dict = None, duration_ms: float = 0):
        self._log(LogLevel.ACTION, component, message, data, duration_ms)

    def search(self, component: str, message: str, data: dict = None, duration_ms: float = 0):
        self._log(LogLevel.SEARCH, component, message, data, duration_ms)

    def extract(self, component: str, message: str, data: dict = None, duration_ms: float = 0):
        self._log(LogLevel.EXTRACT, component, message, data, duration_ms)

    def warning(self, component: str, message: str, data: dict = None):
        self._log(LogLevel.WARNING, component, message, data)

    def error(self, component: str, message: str, data: dict = None):
        self._log(LogLevel.ERROR, component, message, data)

    def get_summary(self) -> dict:
        """Log-Zusammenfassung."""
        by_level = {}
        for entry in self.entries:
            by_level[entry.level.value] = by_level.get(entry.level.value, 0) + 1

        total_duration = sum(e.duration_ms for e in self.entries)

        return {
            "total_entries": len(self.entries),
            "by_level": by_level,
            "total_duration_ms": total_duration,
            "errors": [e.to_dict() for e in self.entries if e.level == LogLevel.ERROR]
        }

    def export(self, filepath: str):
        """Logs als JSON exportieren."""
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2, ensure_ascii=False)


# ============================================================================
# SEARXNG CLIENT
# ============================================================================

@dataclass
class SearchResult:
    """Einzelnes Suchergebnis."""
    title: str
    url: str
    snippet: str
    engine: str = ""
    position: int = 0

    def to_markdown(self) -> str:
        return f"### [{self.title}]({self.url})\n{self.snippet}\n*via {self.engine}*"


@dataclass
class SearchResponse:
    """Gesamte Suchantwort."""
    query: str
    results: list[SearchResult]
    engines_used: list[str]
    total_results: int
    search_time_ms: float

    def to_markdown(self) -> str:
        lines = [
            f"# Suchergebnisse: {self.query}",
            f"*{self.total_results} Ergebnisse in {self.search_time_ms:.0f}ms via {', '.join(self.engines_used)}*\n"
        ]
        for r in self.results:
            lines.append(r.to_markdown())
        return "\n\n".join(lines)


class SearXNGClient:
    """
    Client für SearXNG Meta-Suchmaschine.

    Unterstützt:
    - Google Dorks (site:, filetype:, inurl:, intitle:, etc.)
    - Multiple Engines (google, bing, duckduckgo, brave, etc.)
    - Kategorien (general, images, news, science, files, it)
    - Time Range Filter
    """

    # Öffentliche SearXNG Instanzen (Fallbacks)
    PUBLIC_INSTANCES = [
        "http://localhost:8072",
        #"https://search.sapti.me",
        #"https://searx.tiekoetter.com",
        "https://search.bus-hit.me",
        #"https://searx.be",
    ]

    def __init__(
        self,
        base_url: str = "",
        timeout: int = 30,
        logger: AgentLogger = None
    ):
        """
        Args:
            base_url: SearXNG Instanz URL (leer = Public Instance)
            timeout: Request Timeout in Sekunden
            logger: AgentLogger Instanz
        """
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.timeout = timeout
        self.logger = logger or AgentLogger("SearXNG", verbose=False)
        self._client = httpx.AsyncClient(timeout=timeout)
        self._working_instance = None

    async def _find_working_instance(self) -> str:
        """Findet eine funktionierende öffentliche Instanz."""
        if self._working_instance:
            return self._working_instance

        for instance in self.PUBLIC_INSTANCES:
            try:
                print(f"Testing instance: {instance}")
                resp = await self._client.get(f"{instance}/search", params={"q": "test", "format": "json"}, timeout=10)
                print(f"Response: {resp.status_code} {resp.text[:100]}", resp)
                if resp.status_code == 200:
                    self._working_instance = instance
                    self.logger.info("SearXNG", f"Using instance: {instance}")
                    return instance
                print(f"Instance {instance} not working", resp.status_code, resp)
            except Exception as e:
                continue

        raise Exception("No working SearXNG instance found")

    async def search(
        self,
        query: str,
        engines: list[str] = None,
        categories: list[str] = None,
        language: str = "de",
        time_range: str = "",
        safesearch: int = 0,
        pageno: int = 1,
        max_results: int = 20
    ) -> SearchResponse:
        """
        Führt eine Suche durch.

        Args:
            query: Suchquery (unterstützt Google Dorks)
            engines: Liste von Engines ["google", "bing", "duckduckgo", "brave"]
            categories: Kategorien ["general", "images", "news", "science", "files", "it"]
            language: Sprachcode
            time_range: "day", "week", "month", "year"
            safesearch: 0=off, 1=moderate, 2=strict
            pageno: Seitennummer
            max_results: Max Anzahl Ergebnisse

        Returns:
            SearchResponse mit Ergebnissen
        """
        self.logger.start_timer("search")

        base = self.base_url or await self._find_working_instance()

        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": safesearch,
            "pageno": pageno,
        }

        if engines:
            params["engines"] = ",".join(engines)
        if categories:
            params["categories"] = ",".join(categories)
        if time_range:
            params["time_range"] = time_range

        self.logger.search("SearXNG", f"Searching: {query}", {
            "engines": engines,
            "categories": categories,
            "time_range": time_range
        })

        try:
            resp = await self._client.get(f"{base}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for i, r in enumerate(data.get("results", [])[:max_results]):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    engine=r.get("engine", ""),
                    position=i + 1
                ))

            duration = self.logger.stop_timer("search")

            engines_used = list(set(r.engine for r in results if r.engine))

            response = SearchResponse(
                query=query,
                results=results,
                engines_used=engines_used,
                total_results=len(results),
                search_time_ms=duration
            )

            self.logger.search("SearXNG", f"Found {len(results)} results", duration_ms=duration)

            return response

        except Exception as e:
            self.logger.error("SearXNG", f"Search failed: {e}")
            raise

    # ========================================================================
    # GOOGLE DORKS HELPER
    # ========================================================================

    def dork_site(self, query: str, site: str) -> str:
        """site:example.com query"""
        return f"site:{site} {query}"

    def dork_filetype(self, query: str, filetype: str) -> str:
        """filetype:pdf query"""
        return f"filetype:{filetype} {query}"

    def dork_inurl(self, query: str, inurl: str) -> str:
        """inurl:admin query"""
        return f"inurl:{inurl} {query}"

    def dork_intitle(self, query: str, intitle: str) -> str:
        """intitle:login query"""
        return f"intitle:{intitle} {query}"

    def dork_intext(self, query: str, intext: str) -> str:
        """intext:password query"""
        return f"intext:{intext} {query}"

    def dork_exclude(self, query: str, exclude: str) -> str:
        """-site:example.com query"""
        return f"-{exclude} {query}"

    def dork_exact(self, phrase: str) -> str:
        '''"exact phrase"'''
        return f'"{phrase}"'

    def dork_or(self, term1: str, term2: str) -> str:
        """term1 OR term2"""
        return f"{term1} OR {term2}"

    def build_dork(self, base_query: str, **kwargs) -> str:
        """
        Baut eine komplexe Dork-Query.

        Example:
            build_dork("password", site="github.com", filetype="env", intitle="config")
            -> "site:github.com filetype:env intitle:config password"
        """
        parts = []

        if kwargs.get("site"):
            parts.append(f"site:{kwargs['site']}")
        if kwargs.get("filetype"):
            parts.append(f"filetype:{kwargs['filetype']}")
        if kwargs.get("inurl"):
            parts.append(f"inurl:{kwargs['inurl']}")
        if kwargs.get("intitle"):
            parts.append(f"intitle:{kwargs['intitle']}")
        if kwargs.get("intext"):
            parts.append(f"intext:{kwargs['intext']}")
        if kwargs.get("exclude"):
            for ex in (kwargs["exclude"] if isinstance(kwargs["exclude"], list) else [kwargs["exclude"]]):
                parts.append(f"-{ex}")
        if kwargs.get("exact"):
            parts.append(f'"{kwargs["exact"]}"')

        parts.append(base_query)
        return " ".join(parts)

    async def close(self):
        await self._client.aclose()


# ============================================================================
# SCRAPED CONTENT (aus async_web_scraper.py)
# ============================================================================

@dataclass
class ScrapedHeading:
    level: int
    text: str
    anchor: str = ""

    def to_markdown(self) -> str:
        return f"{'#' * self.level} {self.text}"


@dataclass
class ScrapedLink:
    text: str
    href: str
    context: str = ""

    def to_markdown(self) -> str:
        return f"[{self.text}]({self.href})"


@dataclass
class ScrapedContent:
    url: str
    title: str
    markdown: str
    headings: list[ScrapedHeading] = field(default_factory=list)
    links: list[ScrapedLink] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_toc(self) -> str:
        if not self.headings:
            return ""
        lines = ["## Table of Contents\n"]
        for h in self.headings:
            indent = "  " * (h.level - 1)
            anchor = h.anchor or h.text.lower().replace(" ", "-")
            lines.append(f"{indent}- [{h.text}](#{anchor})")
        return "\n".join(lines)

    def get_internal_links(self, base_domain: str = "") -> list[ScrapedLink]:
        if not base_domain:
            base_domain = urlparse(self.url).netloc
        return [l for l in self.links if base_domain in l.href]

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "markdown": self.markdown,
            "headings": [{"level": h.level, "text": h.text} for h in self.headings],
            "links": [{"text": l.text, "href": l.href} for l in self.links],
            "meta": self.meta,
            "scraped_at": self.scraped_at
        }


# ============================================================================
# WEB AGENT - HAUPTKLASSE
# ============================================================================

class WebAgent:
    """
    Erweiterter Web-Agent mit SearXNG, Tools und Preview.

    Features:
    - SearXNG Meta-Suche mit Google Dorks
    - Clevere Tools (navigate, click, type, extract, screenshot)
    - Detaillierte Logs
    - Preview-Modus (headless=False)
    - Strukturierter Markdown Output
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        searxng_url: str = "",
        viewport: dict = None,
        user_agent: str = "",
        state_dir: str = "agent_states",
        verbose: bool = True
    ):
        """
        Args:
            headless: False für Preview/Debugging
            browser_type: chromium, firefox, webkit
            searxng_url: SearXNG Instanz (leer = Public)
            viewport: {"width": 1920, "height": 1080}
            user_agent: Custom User-Agent
            state_dir: Verzeichnis für Session-States
            verbose: Ausführliche Logs
        """
        self.headless = headless
        self.browser_type = browser_type
        self.viewport = viewport or {"width": 1920, "height": 1080}
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.state_dir = state_dir

        os.makedirs(state_dir, exist_ok=True)

        # Logger
        self.logger = AgentLogger("WebAgent", verbose=verbose)

        # SearXNG Client
        self.search = SearXNGClient(base_url=searxng_url, logger=self.logger)

        # Playwright
        self._playwright: Optional[APlaywright] = None
        self._browser: Optional[ABrowser] = None
        self._context: Optional[ABrowserContext] = None
        self._page: Optional[APage] = None

        # Navigation History
        self._history: list[str] = []
        self._current_url: str = ""

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    async def __aenter__(self) -> "WebAgent":
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    async def start(self):
        """Browser starten."""
        self.logger.info("Browser", f"Starting {self.browser_type} (headless={self.headless})")
        self.logger.start_timer("browser_start")

        self._playwright = await async_playwright().start()

        launcher = {
            "chromium": self._playwright.chromium,
            "firefox": self._playwright.firefox,
            "webkit": self._playwright.webkit
        }.get(self.browser_type, self._playwright.chromium)

        self._browser = await launcher.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"] if self.browser_type == "chromium" else []
        )

        self._context = await self._browser.new_context(
            viewport=self.viewport,
            user_agent=self.user_agent
        )

        self._page = await self._context.new_page()

        # Event Listeners für Logging
        self._page.on("console", lambda msg: self.logger.debug("Console", msg.text))
        self._page.on("pageerror", lambda err: self.logger.error("PageError", str(err)))

        duration = self.logger.stop_timer("browser_start")
        self.logger.info("Browser", f"Started successfully {duration:.1f}")

    async def stop(self):
        """Browser stoppen."""
        self.logger.info("Browser", "Shutting down...")

        await self.search.close()

        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        self.logger.info("Browser", "Shutdown complete")

        # Log Summary
        summary = self.logger.get_summary()
        self.logger.info("Summary", f"Session finished", summary)

    # ========================================================================
    # NAVIGATION TOOLS
    # ========================================================================

    async def goto(self, url: str, wait_until: str = "networkidle") -> bool:
        """
        Navigiert zu URL.

        Args:
            url: Ziel-URL
            wait_until: "load", "domcontentloaded", "networkidle"
        """
        self.logger.start_timer("goto")
        self.logger.action("Navigate", f"Going to: {url}")

        try:
            await self._page.goto(url, wait_until=wait_until)
            self._history.append(url)
            self._current_url = url

            duration = self.logger.stop_timer("goto")
            title = await self._page.title()
            self.logger.action("Navigate", f"Loaded: {title}", {"url": url}, duration)
            return True

        except Exception as e:
            self.logger.error("Navigate", f"Failed: {e}")
            return False

    async def back(self) -> bool:
        """Zurück zur vorherigen Seite."""
        if len(self._history) < 2:
            self.logger.warning("Navigate", "No history to go back")
            return False

        self._history.pop()
        prev_url = self._history[-1]
        return await self.goto(prev_url)

    async def refresh(self):
        """Seite neu laden."""
        self.logger.action("Navigate", "Refreshing page")
        await self._page.reload(wait_until="networkidle")

    async def wait(self, selector: str = "", timeout: int = 30000):
        """Warten auf Element oder Zeit."""
        if selector:
            self.logger.debug("Wait", f"Waiting for: {selector}")
            await self._page.wait_for_selector(selector, timeout=timeout)
        else:
            await self._page.wait_for_load_state("networkidle")

    # ========================================================================
    # INTERACTION TOOLS
    # ========================================================================

    async def click(self, selector: str, wait_after: bool = True) -> bool:
        """Element klicken."""
        self.logger.start_timer("click")
        self.logger.action("Click", f"Clicking: {selector}")

        try:
            await self._page.locator(selector).click()
            if wait_after:
                await asyncio.sleep(0.5)

            duration = self.logger.stop_timer("click")
            self.logger.action("Click", "Clicked successfully", duration_ms=duration)
            return True
        except Exception as e:
            self.logger.error("Click", f"Failed: {e}", {"selector": selector})
            return False

    async def type(self, selector: str, text: str, clear: bool = True) -> bool:
        """Text eingeben."""
        self.logger.action("Type", f"Typing into: {selector}", {"text_length": len(text)})

        try:
            element = self._page.locator(selector)
            if clear:
                await element.fill(text)
            else:
                await element.type(text)
            return True
        except Exception as e:
            self.logger.error("Type", f"Failed: {e}")
            return False

    async def select(self, selector: str, value: str = "", label: str = "", index: int = -1) -> bool:
        """Dropdown auswählen."""
        self.logger.action("Select", f"Selecting in: {selector}")

        try:
            element = self._page.locator(selector)
            if value:
                await element.select_option(value=value)
            elif label:
                await element.select_option(label=label)
            elif index >= 0:
                await element.select_option(index=index)
            return True
        except Exception as e:
            self.logger.error("Select", f"Failed: {e}")
            return False

    async def hover(self, selector: str) -> bool:
        """Über Element hovern."""
        try:
            await self._page.locator(selector).hover()
            return True
        except Exception as e:
            self.logger.error("Hover", f"Failed: {e}")
            return False

    async def scroll(self, direction: str = "down", amount: int = 500):
        """Scrollen."""
        delta = amount if direction == "down" else -amount
        await self._page.mouse.wheel(0, delta)
        self.logger.debug("Scroll", f"Scrolled {direction} by {amount}px")

    async def scroll_to_bottom(self):
        """Zum Ende scrollen (für Lazy-Load)."""
        self.logger.action("Scroll", "Scrolling to bottom")

        await self._page.evaluate("""
            async () => {
                const delay = ms => new Promise(r => setTimeout(r, ms));
                let lastHeight = 0;
                let scrolls = 0;

                while (scrolls < 50) {
                    window.scrollBy(0, 500);
                    await delay(200);
                    scrolls++;

                    const newHeight = document.body.scrollHeight;
                    if (newHeight === lastHeight) break;
                    lastHeight = newHeight;
                }
            }
        """)

    # ========================================================================
    # EXTRACTION TOOLS
    # ========================================================================

    async def extract_text(self, selector: str = "body") -> str:
        """Text aus Element extrahieren."""
        try:
            return await self._page.locator(selector).inner_text()
        except:
            return ""

    async def extract_html(self, selector: str = "body") -> str:
        """HTML aus Element extrahieren."""
        try:
            return await self._page.locator(selector).inner_html()
        except:
            return ""

    async def extract_attribute(self, selector: str, attribute: str) -> str:
        """Attribut aus Element extrahieren."""
        try:
            return await self._page.locator(selector).get_attribute(attribute) or ""
        except:
            return ""

    async def extract_all(self, selector: str, attribute: str = "") -> list[str]:
        """Alle Elemente extrahieren."""
        elements = self._page.locator(selector)
        count = await elements.count()

        results = []
        for i in range(count):
            el = elements.nth(i)
            if attribute:
                results.append(await el.get_attribute(attribute) or "")
            else:
                results.append(await el.inner_text())

        return results

    async def extract_markdown(self) -> ScrapedContent:
        """Seite als strukturiertes Markdown extrahieren."""
        self.logger.start_timer("extract")
        self.logger.extract("Extract", "Extracting page content")

        title = await self._page.title()
        url = self._page.url

        # Markdown extraction (aus async_web_scraper.py)
        markdown = await self._page.evaluate(r"""
            () => {
                const removeSelectors = [
                    'script', 'style', 'noscript', 'iframe', 'svg',
                    'nav', 'header', 'footer', 'aside',
                    '.advertisement', '.ads', '.sidebar', '.menu',
                    '.cookie-banner', '.popup', '.modal'
                ];

                const clone = document.body.cloneNode(true);
                removeSelectors.forEach(sel => {
                    clone.querySelectorAll(sel).forEach(el => el.remove());
                });

                const mainSelectors = ['main', 'article', '[role="main"]', '.content', '#content'];
                let root = null;
                for (const sel of mainSelectors) {
                    root = clone.querySelector(sel);
                    if (root) break;
                }
                root = root || clone;

                function nodeToMd(node) {
                    if (node.nodeType === Node.TEXT_NODE) {
                        return node.textContent.trim();
                    }
                    if (node.nodeType !== Node.ELEMENT_NODE) return '';

                    const tag = node.tagName.toLowerCase();
                    const children = Array.from(node.childNodes).map(c => nodeToMd(c)).join('').trim();

                    if (!children && !['img', 'br', 'hr'].includes(tag)) return '';

                    switch (tag) {
                        case 'h1': return `\n# ${children}\n`;
                        case 'h2': return `\n## ${children}\n`;
                        case 'h3': return `\n### ${children}\n`;
                        case 'h4': return `\n#### ${children}\n`;
                        case 'p': return `\n${children}\n`;
                        case 'br': return '\n';
                        case 'hr': return '\n---\n';
                        case 'strong': case 'b': return `**${children}**`;
                        case 'em': case 'i': return `*${children}*`;
                        case 'code': return `\`${children}\``;
                        case 'pre': return `\n\`\`\`\n${children}\n\`\`\`\n`;
                        case 'a':
                            const href = node.getAttribute('href') || '';
                            return href ? `[${children}](${href})` : children;
                        case 'img':
                            const src = node.getAttribute('src') || '';
                            const alt = node.getAttribute('alt') || 'image';
                            return src ? `![${alt}](${src})` : '';
                        case 'ul': case 'ol':
                            return '\n' + Array.from(node.children).map((li, i) => {
                                const bullet = tag === 'ol' ? `${i + 1}.` : '-';
                                return `${bullet} ${nodeToMd(li)}`;
                            }).join('\n') + '\n';
                        case 'li': return children;
                        default: return children;
                    }
                }

                return nodeToMd(root);
            }
        """)

        # Headings extrahieren
        headings_data = await self._page.evaluate("""
            () => {
                const headings = [];
                document.querySelectorAll('h1, h2, h3, h4').forEach(h => {
                    headings.push({
                        level: parseInt(h.tagName[1]),
                        text: h.textContent.trim(),
                        anchor: h.id || ''
                    });
                });
                return headings;
            }
        """)

        # Links extrahieren
        links_data = await self._page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.getAttribute('href');
                    const text = a.textContent.trim();
                    if (href && text && !href.startsWith('javascript:')) {
                        links.push({ text: text.slice(0, 100), href });
                    }
                });
                return links;
            }
        """)

        # Meta extrahieren
        meta = await self._page.evaluate("""
            () => {
                const meta = {};
                const desc = document.querySelector('meta[name="description"]');
                if (desc) meta.description = desc.getAttribute('content');
                meta.lang = document.documentElement.lang || '';
                return meta;
            }
        """)

        # Cleanup
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        markdown = markdown.strip()

        content = ScrapedContent(
            url=url,
            title=title,
            markdown=markdown,
            headings=[ScrapedHeading(**h) for h in headings_data],
            links=[ScrapedLink(text=l['text'], href=urljoin(url, l['href'])) for l in links_data],
            meta=meta
        )

        duration = self.logger.stop_timer("extract")
        self.logger.extract("Extract",
                            f"Extracted {len(markdown)} chars, {len(content.headings)} headings, {len(content.links)} links",
                            duration_ms=duration)

        return content

    # ========================================================================
    # UTILITY TOOLS
    # ========================================================================

    async def screenshot(self, path: str = "", full_page: bool = False) -> str:
        """Screenshot machen."""
        if not path:
            path = os.path.join(self.state_dir, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        await self._page.screenshot(path=path, full_page=full_page)

        self.logger.action("Screenshot", f"Saved: {path}")
        return path

    async def save_state(self, name: str = ""):
        """Session-State speichern."""
        if not name:
            name = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        path = os.path.join(self.state_dir, f"{name}.json")
        state = await self._context.storage_state()

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.action("State", f"Saved: {path}")
        return path

    async def load_state(self, name: str) -> bool:
        """Session-State laden."""
        path = os.path.join(self.state_dir, f"{name}.json")

        if not os.path.exists(path):
            self.logger.error("State", f"Not found: {path}")
            return False

        with open(path) as f:
            state = json.load(f)

        # Neuen Context mit State erstellen
        if self._context:
            await self._context.close()

        self._context = await self._browser.new_context(
            storage_state=state,
            viewport=self.viewport,
            user_agent=self.user_agent
        )
        self._page = await self._context.new_page()

        self.logger.action("State", f"Loaded: {path}")
        return True

    async def evaluate(self, js_code: str) -> Any:
        """JavaScript ausführen."""
        return await self._page.evaluate(js_code)

    def current_url(self) -> str:
        """Aktuelle URL."""
        return self._page.url if self._page else ""

    async def title(self) -> str:
        """Aktueller Titel."""
        return await self._page.title() if self._page else ""

    # ========================================================================
    # HIGH-LEVEL ACTIONS
    # ========================================================================

    async def search_and_scrape(
        self,
        query: str,
        max_results: int = 5,
        engines: list[str] = None,
        **dork_kwargs
    ) -> list[ScrapedContent]:
        """
        Suchen und Top-Ergebnisse scrapen.

        Args:
            query: Suchquery
            max_results: Anzahl zu scrapender Seiten
            engines: SearXNG Engines
            **dork_kwargs: Google Dork Parameter (site=, filetype=, etc.)
        """
        # Dork bauen wenn nötig
        if dork_kwargs:
            query = self.search.build_dork(query, **dork_kwargs)

        # Suchen
        search_results = await self.search.search(
            query=query,
            engines=engines,
            max_results=max_results
        )

        self.logger.info("SearchAndScrape", f"Scraping top {max_results} results for: {query}")

        # Top-Ergebnisse scrapen
        scraped = []
        for result in search_results.results[:max_results]:
            try:
                await self.goto(result.url)
                content = await self.extract_markdown()
                scraped.append(content)
            except Exception as e:
                self.logger.error("SearchAndScrape", f"Failed to scrape {result.url}: {e}")

        return scraped

    async def login(
        self,
        url: str,
        username_selector: str,
        password_selector: str,
        submit_selector: str,
        username: str,
        password: str,
        success_indicator: str = ""
    ) -> bool:
        """
        Login durchführen.
        """
        self.logger.action("Login", f"Logging in at: {url}")

        await self.goto(url)
        await self.type(username_selector, username)
        await self.type(password_selector, password)
        await self.click(submit_selector)

        await asyncio.sleep(2)

        if success_indicator:
            try:
                await self._page.wait_for_selector(success_indicator, timeout=10000)
                self.logger.action("Login", "Login successful")
                return True
            except:
                self.logger.error("Login", "Login failed - indicator not found")
                return False

        self.logger.action("Login", "Login completed (unverified)")
        return True

    def export_logs(self, filepath: str = ""):
        """Logs exportieren."""
        if not filepath:
            filepath = os.path.join(self.state_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.logger.export(filepath)
        return filepath


# ============================================================================
# CONVENIENCE
# ============================================================================

async def quick_search(query: str, **dork_kwargs) -> SearchResponse:
    """Schnelle Suche ohne Browser."""
    client = SearXNGClient()
    try:
        if dork_kwargs:
            query = client.build_dork(query, **dork_kwargs)
        return await client.search(query)
    finally:
        await client.close()


async def quick_scrape(url: str, headless: bool = True) -> ScrapedContent:
    """Schnelles Scraping einer URL."""
    async with WebAgent(headless=headless) as agent:
        await agent.goto(url)
        return await agent.extract_markdown()


"""
WebAgent - Nutzungsbeispiele
============================

Praktische Beispiele für alle Features des WebAgent.
"""

# ============================================================================
# BEISPIEL 1: Einfache Suche (ohne Browser)
# ============================================================================

async def beispiel_suche():
    """
    SearXNG Suche ohne Browser - schnell und leichtgewichtig.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 1: Einfache SearXNG Suche")
    print("=" * 60)

    client = SearXNGClient()

    # Standard Suche
    results = await client.search("Python asyncio tutorial", max_results=5)

    print(f"\n🔎 Query: {results.query}")
    print(f"📊 Gefunden: {results.total_results} Ergebnisse")
    print(f"🔧 Engines: {', '.join(results.engines_used)}")
    print(f"⏱️  Zeit: {results.search_time_ms:.0f}ms\n")

    for r in results.results:
        print(f"  [{r.position}] {r.title}")
        print(f"      {r.url}")
        print(f"      {r.snippet[:100]}...")
        print()

    await client.close()


# ============================================================================
# BEISPIEL 2: Google Dorks
# ============================================================================

async def beispiel_dorks():
    """
    Google Dorks Syntax für gezielte Suche.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 2: Google Dorks")
    print("=" * 60)

    client = SearXNGClient()

    # Dork Builder
    print("\n--- Dork Builder ---")

    dork = client.build_dork(
        "API documentation",
        site="github.com",
        filetype="md"
    )
    print(f"Built: {dork}")

    # Verschiedene Dork-Typen
    print("\n--- Dork Varianten ---")

    # Site-spezifisch
    site_dork = client.dork_site("web scraping", "python.org")
    print(f"Site: {site_dork}")

    # Filetype
    file_dork = client.dork_filetype("machine learning", "pdf")
    print(f"File: {file_dork}")

    # Combined Dork
    combined = client.build_dork(
        "REST API",
        site="github.com",
        filetype="md",
        intitle="documentation",
        exclude=["deprecated", "old"]
    )
    print(f"Combined: {combined}")

    # Suche ausführen
    results = await client.search(combined, max_results=3)
    print(f"\nResults: {results.total_results}")

    await client.close()


# ============================================================================
# BEISPIEL 3: Vollständiger Agent (mit Browser)
# ============================================================================

async def beispiel_agent():
    """
    Vollständiger WebAgent mit Suche + Navigation + Extraktion.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 3: WebAgent mit Browser")
    print("=" * 60)

    async with WebAgent(
        headless=True,  # Für Debugging: False
        verbose=True  # Detaillierte Logs
    ) as agent:
        # Suchen
        results = await agent.search.search("FastAPI tutorial", max_results=3)
        print(f"\n🔎 Gefunden: {results.total_results} Ergebnisse")

        # Erste Seite öffnen
        if results.results:
            url = results.results[0].url
            print(f"\n🌐 Öffne: {url}")

            await agent.goto(url)

            # Inhalt extrahieren
            content = await agent.extract_markdown()

            print(f"\n📄 Extrahiert:")
            print(f"   Titel: {content.title}")
            print(f"   Headings: {len(content.headings)}")
            print(f"   Links: {len(content.links)}")
            print(f"   Content: {len(content.markdown)} Zeichen")

            # Table of Contents
            print(f"\n📑 Table of Contents:")
            for h in content.headings[:5]:
                indent = "  " * (h.level - 1)
                print(f"   {indent}{'#' * h.level} {h.text[:50]}")


# ============================================================================
# BEISPIEL 4: Debug-Modus (Browser sichtbar)
# ============================================================================

async def beispiel_debug():
    """
    Debug-Modus: Browser ist sichtbar für Debugging.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 4: Debug-Modus (Browser sichtbar)")
    print("=" * 60)

    async with WebAgent(
        headless=False,  # 👀 Browser sichtbar!
        verbose=True
    ) as agent:
        # Navigate
        await agent.goto("https://news.ycombinator.com")

        print("\n👀 Browser ist jetzt sichtbar!")
        print("   Du kannst sehen was der Agent macht.")

        # Screenshot
        path = await agent.screenshot(full_page=True)
        print(f"\n📸 Screenshot: {path}")

        # Scrollen
        await agent.scroll("down", 500)
        await asyncio.sleep(1)
        await agent.scroll("down", 500)

        # Extrahieren
        content = await agent.extract_markdown()
        print(f"\n📄 Titel: {content.title}")
        print(f"   Links: {len(content.links)}")


# ============================================================================
# BEISPIEL 5: Interaktionen
# ============================================================================

async def beispiel_interaktionen():
    """
    Interaktive Aktionen: Click, Type, Select.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 5: Interaktionen")
    print("=" * 60)

    async with WebAgent(headless=True, verbose=True) as agent:
        # DuckDuckGo Suche
        await agent.goto("https://duckduckgo.com")

        # Suchfeld ausfüllen
        await agent.type('input[name="q"]', "Python web scraping")

        # Submit (Enter-Key simulieren)
        await agent.evaluate('document.querySelector("input[name=\\"q\\"]").form.submit()')

        # Warten auf Ergebnisse
        await asyncio.sleep(2)

        # Screenshot
        await agent.screenshot("duckduckgo_results.png")

        print("\n✅ Suche auf DuckDuckGo durchgeführt!")


# ============================================================================
# BEISPIEL 6: Search & Scrape (Kombiniert)
# ============================================================================

async def beispiel_search_and_scrape():
    """
    Suchen und mehrere Ergebnisse automatisch scrapen.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 6: Search & Scrape")
    print("=" * 60)

    async with WebAgent(headless=True, verbose=True) as agent:
        # Suchen und Top-3 scrapen
        scraped_pages = await agent.search_and_scrape(
            query="REST API best practices",
            max_results=3,
            site="github.com"  # Google Dork
        )

        print(f"\n📄 {len(scraped_pages)} Seiten gescraped:")

        for i, page in enumerate(scraped_pages, 1):
            print(f"\n   [{i}] {page.title}")
            print(f"       URL: {page.url}")
            print(f"       Content: {len(page.markdown)} Zeichen")
            print(f"       Preview: {page.markdown[:200]}...")


# ============================================================================
# BEISPIEL 7: Session Persistence
# ============================================================================

async def beispiel_session():
    """
    Session speichern und wiederverwenden.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 7: Session Persistence")
    print("=" * 60)

    # Session 1: State speichern
    async with WebAgent(headless=True) as agent:
        await agent.goto("https://github.com")

        # State speichern
        path = await agent.save_state("github_session")
        print(f"\n💾 Session gespeichert: {path}")

    # Session 2: State laden
    async with WebAgent(headless=True) as agent:
        success = await agent.load_state("github_session")
        print(f"\n📂 Session geladen: {success}")

        # Jetzt haben wir die Cookies/LocalStorage von vorher


# ============================================================================
# BEISPIEL 8: Logs exportieren
# ============================================================================

async def beispiel_logs():
    """
    Detaillierte Logs für Analyse exportieren.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 8: Log Export")
    print("=" * 60)

    async with WebAgent(headless=True, verbose=True) as agent:
        # Einige Aktionen
        await agent.search.search("test query", max_results=2)
        await agent.goto("https://example.com")
        await agent.extract_markdown()

        # Logs exportieren
        log_path = agent.export_logs()
        print(f"\n📋 Logs exportiert: {log_path}")

        # Summary
        summary = agent.logger.get_summary()
        print(f"\n📊 Summary:")
        print(f"   Total Entries: {summary['total_entries']}")
        print(f"   By Level: {summary['by_level']}")
        print(f"   Total Duration: {summary['total_duration_ms']:.0f}ms")


# ============================================================================
# BEISPIEL 9: Quick Functions
# ============================================================================

async def beispiel_quick():
    """
    Schnelle Hilfsfunktionen ohne Setup.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 9: Quick Functions")
    print("=" * 60)

    # Quick Search (ohne Browser)
    print("\n--- Quick Search ---")
    results = await quick_search("Python tutorial", site="realpython.com")
    print(f"Found: {results.total_results}")

    # Quick Scrape (mit temporärem Browser)
    print("\n--- Quick Scrape ---")
    content = await quick_scrape("https://example.com")
    print(f"Title: {content.title}")
    print(f"Content: {len(content.markdown)} chars")


# ============================================================================
# BEISPIEL 10: ToolBoxV2 Integration Pattern
# ============================================================================

async def minimal_web_agent_integration():
    """
    So würdest du den WebAgent in ToolBoxV2 integrieren.
    """
    print("\n" + "=" * 60)
    print("📝 BEISPIEL 10: ToolBoxV2 Integration")
    print("=" * 60)

    # Als Tool-Funktionen:

    async def tool_web_search(query: str, site: str = "", filetype: str = "", max_results: int = 5) -> dict:
        """🔎 Web-Suche mit Google Dorks Support."""
        async with WebAgent(headless=True, verbose=False) as agent:
            q = agent.search.build_dork(query, site=site, filetype=filetype) if site or filetype else query
            results = await agent.search.search(q, max_results=max_results)
            return {
                "query": results.query,
                "total": results.total_results,
                "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results.results]
            }

    async def tool_web_scrape(url: str) -> dict:
        """📄 URL scrapen und als Markdown zurückgeben."""
        async with WebAgent(headless=True, verbose=False) as agent:
            await agent.goto(url)
            content = await agent.extract_markdown()
            return content.to_dict()

    async def tool_web_navigate(url: str, actions: list[dict]) -> dict:
        """🧭 Navigieren und Aktionen ausführen."""
        async with WebAgent(headless=True, verbose=False) as agent:
            await agent.goto(url)

            for action in actions:
                if action["type"] == "click":
                    await agent.click(action["selector"])
                elif action["type"] == "type":
                    await agent.type(action["selector"], action["text"])
                elif action["type"] == "scroll":
                    await agent.scroll(action.get("direction", "down"))
                elif action["type"] == "wait":
                    await agent.wait(action.get("selector", ""))

            content = await agent.extract_markdown()
            return content.to_dict()

    return [
        {
            "name": "web_search",
            "description": "Web-Suche mit Google Dorks Support.",
            "func": tool_web_search,
            "category": "web",
            "flags": {"read": True, "write": False}
        },
        {
            "name": "web_scrape",
            "description": "URL scrapen und als Markdown zurückgeben.",
            "func": tool_web_scrape,
            "category": "web",
            "flags": {"read": True, "write": False}
        },
        {
            "name": "web_navigate",
            "description": "Navigieren und Aktionen ausführen.",
            "func": tool_web_navigate,
            "category": "web",
            "flags": {"read": True, "write": True}
        },
    ]


async def google_navigation_test():
    # Wir starten den Agenten mit headless=False, um die Aktion live zu sehen
    async with WebAgent(
        headless=False,
        verbose=True,
        browser_type="chromium"
    ) as agent:

        # 1. Zu Google navigieren
        print("\n--- Schritt 1: Navigation zu Google ---")
        await agent.goto("https://www.google.com")

        # 2. Cookie-Banner akzeptieren (typisch für EU/Deutschland)
        # Google nutzt oft IDs wie 'L2AGLb' für den "Alle akzeptieren" Button
        print("\n--- Schritt 2: Cookies akzeptieren ---")
        try:
            # Wir versuchen den Button per Text oder ID zu finden
            cookie_button = "button:has-text('Alle akzeptieren'), #L2AGLb"
            await agent.click(cookie_button)
            print("✅ Cookie-Banner weggeklickt.")
        except:
            print("ℹ️ Kein Cookie-Banner gefunden oder bereits akzeptiert.")

        # 3. Suchbegriff eingeben
        print("\n--- Schritt 3: Suche ausführen ---")
        search_query = "Playwright Python Automatisierung Tutorial"
        # Google Suchfeld ist meist ein 'textarea' oder 'input' mit Name 'q'
        await agent.type('textarea[name="q"]', search_query)
        await agent._page.keyboard.press("Enter")  # Enter-Taste simulieren

        # Kurz warten, bis die Ergebnisse geladen sind
        await agent.wait(selector="#search")

        # 4. Ergebnisse inspizieren & Scrollen
        print("\n--- Schritt 4: Ergebnisse scannen ---")
        await agent.scroll("down", 500)
        await agent.screenshot("google_suche_ergebnisse.png")

        # Extrahiere alle h3-Titel der Suchergebnisse
        titles = await agent.extract_all("h3")
        print(f"Gefundene Titel auf Seite 1: {titles[:5]}")

        # 5. Das erste echte Suchergebnis anklicken
        print("\n--- Schritt 5: Erstes Ergebnis öffnen ---")
        # Wir klicken auf den ersten h3-Titel innerhalb des Suchcontainers
        await agent.click("#search h3")

        # Warten bis die Zielseite geladen ist
        await agent.wait()
        print(f"📍 Aktuelle URL: {agent.current_url()}")

        # 6. Inhalt der Zielseite extrahieren (Markdown)
        print("\n--- Schritt 6: Content Extraktion ---")
        content = await agent.extract_markdown()

        print(f"📄 Titel der Seite: {content.title}")
        print(f"📊 Anzahl Headings: {len(content.headings)}")
        print(f"🔗 Anzahl Links: {len(content.links)}")

        # Eine kleine Vorschau des Markdowns ausgeben
        print("\n--- Content Vorschau (erste 300 Zeichen) ---")
        print(content.markdown[:300] + "...")

        # 7. Session beenden & Log-Pfad ausgeben
        log_file = agent.export_logs()
        print(f"\n✅ Test abgeschlossen. Logs gespeichert unter: {log_file}")

# ============================================================================
# RUN
# ============================================================================
async def quick_search_test():
    print("🔍 Testing WebAgent...\n")

    # Test SearXNG
    print("=" * 60)
    print("Testing SearXNG Search")
    print("=" * 60)

    try:
        results = await quick_search("Python web scraping", site="github.com")
        print(f"Found {len(results.results)} results")
        for r in results.results[:3]:
            print(f"  - {r.title}")
            print(f"    {r.url}")
    except Exception as e:
        print(f"Search test skipped (no network): {e}")

    print("\n✅ WebAgent module loaded successfully!")
    print("Use: async with WebAgent(headless=False) as agent: ...")

async def run_all():
    """Alle Beispiele ausführen."""

    # Wähle welche Beispiele laufen sollen:
    # await beispiel_suche()
    # await beispiel_dorks()
    # await quick_search_test()
    # await beispiel_agent()           # Braucht Netzwerk
    # await beispiel_debug()           # Öffnet sichtbaren Browser!
    # await beispiel_interaktionen()   # Interaktiv
    # await beispiel_search_and_scrape()
    # await beispiel_session()
    # await beispiel_logs()
    await google_navigation_test()
    # await beispiel_quick()


if __name__ == "__main__":
    asyncio.run(run_all())
