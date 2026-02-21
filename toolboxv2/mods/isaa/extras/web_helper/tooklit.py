"""
WebAgent ToolBoxV2 Integration
==============================

Vollständige Integration des WebAgent als ToolBoxV2 Tools.

Features:
- Managed Browser Lifecycle (auto-start, keep-open, on-exit close)
- Full Tool Pack (alle Features)
- Minimal Tool Pack (nur Basics)
- Session Management
- Headless/Headful Toggle

Usage:
    from web_agent_tools import WebAgentToolkit, get_minimal_tools, get_full_tools

    # Option 1: Managed Toolkit
    toolkit = WebAgentToolkit()
    tools = toolkit.get_tools()

    # Option 2: Quick Tools
    tools = get_minimal_tools()  # Nur basics
    tools = get_full_tools()     # Alles
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from toolboxv2.mods.isaa.extras.web_helper.web_agent import WebAgent


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

class ToolCategory(Enum):
    """Tool-Kategorien."""
    BROWSER = "browser"
    SEARCH = "search"
    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    EXTRACTION = "extraction"
    SESSION = "session"
    UTILITY = "utility"


@dataclass
class ToolDefinition:
    """Definition eines Tools."""
    name: str
    description: str
    func: Callable
    category: ToolCategory
    parameters: dict = field(default_factory=dict)
    returns: str = "dict"
    flags: dict = field(default_factory=lambda: {"read": True, "write": False})
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary für ToolBoxV2."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_func": self.func,
            "category": self.category.value,
            "parameters": self.parameters,
            "returns": self.returns,
            "flags": self.flags,
            "examples": self.examples
        }


# ============================================================================
# MANAGED BROWSER
# ============================================================================

class ManagedBrowser:
    """
    Managed Browser Instance mit Lifecycle-Management.

    Modes:
    - auto_start: Browser startet automatisch beim ersten Tool-Call
    - keep_open: Browser bleibt offen zwischen Calls
    - on_exit_close: Browser schließt beim Programm-Ende
    """

    _instance: Optional["ManagedBrowser"] = None

    def __init__(
        self,
        headless: bool = True,
        auto_start: bool = True,
        keep_open: bool = True,
        verbose: bool = False
    ):
        self.headless = headless
        self.auto_start = auto_start
        self.keep_open = keep_open
        self.verbose = verbose

        self._agent: Optional[WebAgent] = None
        self._started = False
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls, **kwargs) -> "ManagedBrowser":
        """Singleton Instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset Singleton (für Tests)."""
        if cls._instance and cls._instance._agent:
            asyncio.create_task(cls._instance.stop())
        cls._instance = None

    async def start(self):
        """Browser starten."""
        async with self._lock:
            if self._started:
                return

            self._agent = WebAgent(
                headless=self.headless,
                verbose=self.verbose
            )
            await self._agent.start()
            self._started = True

    async def stop(self):
        """Browser stoppen."""
        async with self._lock:
            if self._agent and self._started:
                await self._agent.stop()
                self._started = False
                self._agent = None

    async def get_agent(self) -> WebAgent:
        """Agent holen (startet automatisch wenn nötig)."""
        if self.auto_start and not self._started:
            await self.start()

        if not self._agent:
            raise RuntimeError("Browser nicht gestartet! Rufe start() auf oder setze auto_start=True")

        return self._agent

    @property
    def is_running(self) -> bool:
        return self._started

    async def set_headless(self, headless: bool):
        """Headless-Modus ändern (erfordert Neustart)."""
        was_running = self._started
        if was_running:
            await self.stop()

        self.headless = headless

        if was_running:
            await self.start()


# ============================================================================
# TOOLKIT CLASS
# ============================================================================

class WebAgentToolkit:
    """
    WebAgent Toolkit für ToolBoxV2.

    Verwaltet Browser-Lifecycle und stellt Tools bereit.

    Usage:
        toolkit = WebAgentToolkit(headless=True, keep_open=True)

        # Alle Tools holen
        tools = toolkit.get_tools()

        # Oder nur bestimmte Kategorien
        tools = toolkit.get_tools(categories=[ToolCategory.SEARCH, ToolCategory.EXTRACTION])

        # Browser manuell starten/stoppen
        await toolkit.start_browser()
        await toolkit.stop_browser()
    """

    def __init__(
        self,
        headless: bool = True,
        auto_start: bool = True,
        keep_open: bool = True,
        verbose: bool = False,
        state_dir: str = "agent_states"
    ):
        self.browser = ManagedBrowser(
            headless=headless,
            auto_start=auto_start,
            keep_open=keep_open,
            verbose=verbose
        )
        self.state_dir = state_dir
        self._tools: list[ToolDefinition] = []
        self._build_tools()

    def _build_tools(self):
        """Baut alle Tool-Definitionen."""

        # ====================================================================
        # BROWSER CONTROL TOOLS
        # ====================================================================

        async def tool_browser_start(headless: bool = True) -> dict:
            """
            🚀 Startet den Browser.

            Args:
                headless: True für unsichtbar, False für sichtbar (Debug)

            Returns:
                Status des Browsers
            """
            try:
                await self.browser.set_headless(headless)
                await self.browser.start()
                return {
                    "status": "started",
                    "headless": headless,
                    "message": f"Browser gestartet ({'headless' if headless else 'sichtbar'})"
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"status": "error", "message": f"Failed to start browser: {e}"}

        async def tool_browser_stop() -> dict:
            """
            🛑 Stoppt den Browser.

            Returns:
                Status
            """
            await self.browser.stop()
            return {"status": "stopped", "message": "Browser gestoppt"}

        async def tool_browser_status() -> dict:
            """
            📊 Zeigt Browser-Status.

            Returns:
                Aktueller Status des Browsers
            """
            return {
                "running": self.browser.is_running,
                "headless": self.browser.headless,
                "auto_start": self.browser.auto_start,
                "keep_open": self.browser.keep_open
            }

        async def tool_browser_set_headless(headless: bool) -> dict:
            """
            👁️ Wechselt zwischen Headless und sichtbarem Modus.

            Args:
                headless: True=unsichtbar, False=sichtbar

            Returns:
                Neuer Status
            """
            await self.browser.set_headless(headless)
            return {
                "headless": headless,
                "message": f"Browser ist jetzt {'unsichtbar' if headless else 'sichtbar'}"
            }

        self._tools.extend([
            ToolDefinition(
                name="browser_start",
                description="Startet den Browser. headless=False für Debug/Preview.",
                func=tool_browser_start,
                category=ToolCategory.BROWSER,
                parameters={"headless": {"type": "bool", "default": True}},
                flags={"read": False, "write": True,"no_thread": True},
                examples=["browser_start()", "browser_start(headless=False)"]
            ),
            ToolDefinition(
                name="browser_stop",
                description="Stoppt den Browser und gibt Ressourcen frei.",
                func=tool_browser_stop,
                category=ToolCategory.BROWSER,
                flags={"read": False, "write": True,"no_thread": True}
            ),
            ToolDefinition(
                name="browser_status",
                description="Zeigt ob Browser läuft und in welchem Modus.",
                func=tool_browser_status,
                category=ToolCategory.BROWSER,
                flags={"read": True, "write": False,"no_thread": True}
            ),
            ToolDefinition(
                name="browser_set_headless",
                description="Wechselt zwischen sichtbar/unsichtbar. Für Debugging: headless=False",
                func=tool_browser_set_headless,
                category=ToolCategory.BROWSER,
                parameters={"headless": {"type": "bool", "required": True}},
                flags={"read": False, "write": True,"no_thread": True}
            ),
        ])

        # ====================================================================
        # SEARCH TOOLS
        # ====================================================================

        async def tool_web_search(
            query: str,
            site: str = "",
            filetype: str = "",
            inurl: str = "",
            intitle: str = "",
            exclude: str = "",
            max_results: int = 10,
            engines: str = ""
        ) -> dict:
            """
            🔎 Web-Suche mit Google Dorks Support.

            Args:
                query: Suchbegriff
                site: Nur diese Domain (z.B. "github.com")
                filetype: Nur dieser Dateityp (z.B. "pdf", "md")
                inurl: URL muss enthalten
                intitle: Titel muss enthalten
                exclude: Ausschließen (comma-separated)
                max_results: Max Ergebnisse (default: 10)
                engines: Engines comma-separated (z.B. "google,bing,duckduckgo")

            Returns:
                Suchergebnisse mit Titel, URL, Snippet

            Examples:
                web_search("Python tutorial")
                web_search("API docs", site="github.com", filetype="md")
                web_search("machine learning", exclude="beginner,basic")
            """
            agent = await self.browser.get_agent()

            # Dork bauen
            dork_kwargs = {}
            if site:
                dork_kwargs["site"] = site
            if filetype:
                dork_kwargs["filetype"] = filetype
            if inurl:
                dork_kwargs["inurl"] = inurl
            if intitle:
                dork_kwargs["intitle"] = intitle
            if exclude:
                dork_kwargs["exclude"] = [e.strip() for e in exclude.split(",")]

            if dork_kwargs:
                query = agent.search.build_dork(query, **dork_kwargs)

            # Engines parsen
            engine_list = [e.strip() for e in engines.split(",")] if engines else None

            results = await agent.search.search(
                query=query,
                engines=engine_list,
                max_results=max_results
            )

            return {
                "query": results.query,
                "total": results.total_results,
                "engines": results.engines_used,
                "results": [
                    {"title": r.title, "url": r.url, "snippet": r.snippet, "engine": r.engine}
                    for r in results.results
                ]
            }

        async def tool_search_site(site: str, query: str = "", max_results: int = 10) -> dict:
            """
            🌐 Suche innerhalb einer Website.

            Args:
                site: Domain (z.B. "docs.python.org")
                query: Suchbegriff (optional)
                max_results: Max Ergebnisse

            Returns:
                Suchergebnisse von dieser Site
            """
            return await tool_web_search(query=query, site=site, max_results=max_results)

        async def tool_search_files(filetype: str, query: str, max_results: int = 10) -> dict:
            """
            📁 Suche nach Dateitypen.

            Args:
                filetype: Dateityp (pdf, doc, xls, ppt, md, txt, etc.)
                query: Suchbegriff
                max_results: Max Ergebnisse

            Returns:
                Gefundene Dateien
            """
            return await tool_web_search(query=query, filetype=filetype, max_results=max_results)

        self._tools.extend([
            ToolDefinition(
                name="web_search",
                description="Web-Suche mit Google Dorks (site:, filetype:, inurl:, intitle:, -exclude).",
                func=tool_web_search,
                category=ToolCategory.SEARCH,
                parameters={
                    "query": {"type": "str", "required": True},
                    "site": {"type": "str", "default": ""},
                    "filetype": {"type": "str", "default": ""},
                    "inurl": {"type": "str", "default": ""},
                    "intitle": {"type": "str", "default": ""},
                    "exclude": {"type": "str", "default": ""},
                    "max_results": {"type": "int", "default": 10},
                    "engines": {"type": "str", "default": ""}
                },
                flags={"read": True, "write": False,"no_thread": True},
                examples=[
                    'web_search("Python async")',
                    'web_search("REST API", site="github.com")',
                    'web_search("tutorial", filetype="pdf")'
                ]
            ),
            ToolDefinition(
                name="search_site",
                description="Suche nur innerhalb einer bestimmten Website.",
                func=tool_search_site,
                category=ToolCategory.SEARCH,
                parameters={
                    "site": {"type": "str", "required": True},
                    "query": {"type": "str", "default": ""},
                    "max_results": {"type": "int", "default": 10}
                },
                flags={"read": True, "write": False,"no_thread": True}
            ),
            ToolDefinition(
                name="search_files",
                description="Suche nach bestimmten Dateitypen (PDF, DOC, etc.).",
                func=tool_search_files,
                category=ToolCategory.SEARCH,
                parameters={
                    "filetype": {"type": "str", "required": True},
                    "query": {"type": "str", "required": True},
                    "max_results": {"type": "int", "default": 10}
                },
                flags={"read": True, "write": False,"no_thread": True}
            ),
        ])

        # ====================================================================
        # NAVIGATION TOOLS
        # ====================================================================

        async def tool_goto(url: str, wait_until: str = "networkidle") -> dict:
            """
            🧭 Navigiert zu einer URL.

            Args:
                url: Ziel-URL
                wait_until: "load", "domcontentloaded", "networkidle"

            Returns:
                Seitentitel und URL
            """
            agent = await self.browser.get_agent()
            await agent.goto(url, wait_until=wait_until)
            return {
                "url": agent.current_url(),
                "title": await agent.title()
            }

        async def tool_back() -> dict:
            """
            ⬅️ Geht zur vorherigen Seite zurück.

            Returns:
                Neue URL
            """
            agent = await self.browser.get_agent()
            await agent.back()
            return {"url": agent.current_url()}

        async def tool_refresh() -> dict:
            """
            🔄 Lädt die aktuelle Seite neu.

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.refresh()
            return {"status": "refreshed", "url": agent.current_url()}

        async def tool_current_url() -> dict:
            """
            📍 Gibt die aktuelle URL zurück.

            Returns:
                Aktuelle URL und Titel
            """
            agent = await self.browser.get_agent()
            return {
                "url": agent.current_url(),
                "title": await agent.title()
            }

        self._tools.extend([
            ToolDefinition(
                name="goto",
                description="Navigiert zu einer URL und wartet bis geladen.",
                func=tool_goto,
                category=ToolCategory.NAVIGATION,
                parameters={
                    "url": {"type": "str", "required": True},
                    "wait_until": {"type": "str", "default": "networkidle"}
                },
                flags={"read": True, "write": True}
            ),
            ToolDefinition(
                name="back",
                description="Geht zur vorherigen Seite zurück.",
                func=tool_back,
                category=ToolCategory.NAVIGATION,
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="refresh",
                description="Lädt die aktuelle Seite neu.",
                func=tool_refresh,
                category=ToolCategory.NAVIGATION,
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="current_url",
                description="Zeigt die aktuelle URL an.",
                func=tool_current_url,
                category=ToolCategory.NAVIGATION,
                flags={"read": True, "write": False}
            ),
        ])

        # ====================================================================
        # INTERACTION TOOLS
        # ====================================================================

        async def tool_click(selector: str, wait_after: float = 0.5) -> dict:
            """
            🖱️ Klickt auf ein Element.

            Args:
                selector: CSS Selector (z.B. "#button", ".submit", "button[type='submit']")
                wait_after: Wartezeit nach Klick in Sekunden

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.click(selector, wait_after=wait_after > 0)
            if wait_after > 0:
                await asyncio.sleep(wait_after)
            return {"status": "clicked", "selector": selector}

        async def tool_type(selector: str, text: str, clear: bool = True) -> dict:
            """
            ⌨️ Gibt Text in ein Eingabefeld ein.

            Args:
                selector: CSS Selector des Eingabefelds
                text: Einzugebender Text
                clear: True=Feld vorher leeren, False=Text anhängen

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.type(selector, text, clear=clear)
            return {"status": "typed", "selector": selector, "length": len(text)}

        async def tool_select(selector: str, value: str = "", label: str = "", index: int = -1) -> dict:
            """
            📋 Wählt Option in einem Dropdown.

            Args:
                selector: CSS Selector des Dropdowns
                value: Option value (oder)
                label: Option label (oder)
                index: Option index

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.select(selector, value=value, label=label, index=index)
            return {"status": "selected", "selector": selector}

        async def tool_scroll(direction: str = "down", amount: int = 500) -> dict:
            """
            📜 Scrollt die Seite.

            Args:
                direction: "down" oder "up"
                amount: Pixel

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.scroll(direction, amount)
            return {"status": "scrolled", "direction": direction, "amount": amount}

        async def tool_scroll_to_bottom() -> dict:
            """
            ⬇️ Scrollt zum Ende der Seite (für Lazy-Loading).

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.scroll_to_bottom()
            return {"status": "scrolled_to_bottom"}

        async def tool_wait(selector: str = "", timeout: int = 30) -> dict:
            """
            ⏳ Wartet auf ein Element oder bis Seite geladen.

            Args:
                selector: CSS Selector (leer = auf networkidle warten)
                timeout: Max Wartezeit in Sekunden

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.wait(selector, timeout=timeout * 1000)
            return {"status": "ready", "selector": selector or "page"}

        async def tool_hover(selector: str) -> dict:
            """
            🎯 Bewegt Maus über ein Element.

            Args:
                selector: CSS Selector

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            await agent.hover(selector)
            return {"status": "hovered", "selector": selector}

        self._tools.extend([
            ToolDefinition(
                name="click",
                description="Klickt auf ein Element (Button, Link, etc.).",
                func=tool_click,
                category=ToolCategory.INTERACTION,
                parameters={
                    "selector": {"type": "str", "required": True},
                    "wait_after": {"type": "float", "default": 0.5}
                },
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="type",
                description="Gibt Text in ein Eingabefeld ein.",
                func=tool_type,
                category=ToolCategory.INTERACTION,
                parameters={
                    "selector": {"type": "str", "required": True},
                    "text": {"type": "str", "required": True},
                    "clear": {"type": "bool", "default": True}
                },
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="select",
                description="Wählt Option in Dropdown (value, label oder index).",
                func=tool_select,
                category=ToolCategory.INTERACTION,
                parameters={
                    "selector": {"type": "str", "required": True},
                    "value": {"type": "str", "default": ""},
                    "label": {"type": "str", "default": ""},
                    "index": {"type": "int", "default": -1}
                },
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="scroll",
                description="Scrollt die Seite hoch/runter.",
                func=tool_scroll,
                category=ToolCategory.INTERACTION,
                parameters={
                    "direction": {"type": "str", "default": "down"},
                    "amount": {"type": "int", "default": 500}
                },
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="scroll_to_bottom",
                description="Scrollt zum Seitenende (lädt Lazy-Content).",
                func=tool_scroll_to_bottom,
                category=ToolCategory.INTERACTION,
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="wait",
                description="Wartet auf Element oder bis Seite geladen.",
                func=tool_wait,
                category=ToolCategory.INTERACTION,
                parameters={
                    "selector": {"type": "str", "default": ""},
                    "timeout": {"type": "int", "default": 30}
                },
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="hover",
                description="Bewegt Maus über Element (für Tooltips/Dropdowns).",
                func=tool_hover,
                category=ToolCategory.INTERACTION,
                parameters={"selector": {"type": "str", "required": True}},
                flags={"read": False, "write": True}
            ),
        ])

        # ====================================================================
        # EXTRACTION TOOLS
        # ====================================================================

        async def tool_extract(include_links: bool = True, include_headings: bool = True) -> dict:
            """
            📄 Extrahiert Seiteninhalt als strukturiertes Markdown.

            Args:
                include_links: Links mit extrahieren
                include_headings: Überschriften mit extrahieren

            Returns:
                Strukturierter Inhalt (title, markdown, headings, links, meta)
            """
            agent = await self.browser.get_agent()
            content = await agent.extract_markdown()

            result = {
                "url": content.url,
                "title": content.title,
                "markdown": content.markdown,
                "meta": content.meta
            }

            if include_headings:
                result["headings"] = [{"level": h.level, "text": h.text} for h in content.headings]

            if include_links:
                result["links"] = [{"text": l.text, "href": l.href} for l in content.links[:100]]

            return result

        async def tool_extract_text(selector: str = "body") -> dict:
            """
            📝 Extrahiert Text aus einem Element.

            Args:
                selector: CSS Selector

            Returns:
                Text-Inhalt
            """
            agent = await self.browser.get_agent()
            text = await agent.extract_text(selector)
            return {"text": text, "length": len(text)}

        async def tool_extract_html(selector: str = "body") -> dict:
            """
            🔧 Extrahiert HTML aus einem Element.

            Args:
                selector: CSS Selector

            Returns:
                HTML-Inhalt
            """
            agent = await self.browser.get_agent()
            html = await agent.extract_html(selector)
            return {"html": html, "length": len(html)}

        async def tool_extract_links(filter_pattern: str = "") -> dict:
            """
            🔗 Extrahiert alle Links von der Seite.

            Args:
                filter_pattern: Regex-Pattern zum Filtern (optional)

            Returns:
                Liste von Links
            """
            agent = await self.browser.get_agent()
            content = await agent.extract_markdown()

            links = content.links
            if filter_pattern:
                import re
                pattern = re.compile(filter_pattern, re.IGNORECASE)
                links = [l for l in links if pattern.search(l.href) or pattern.search(l.text)]

            return {
                "total": len(links),
                "links": [{"text": l.text, "href": l.href} for l in links[:100]]
            }

        async def tool_extract_attribute(selector: str, attribute: str) -> dict:
            """
            🏷️ Extrahiert Attribut aus Element.

            Args:
                selector: CSS Selector
                attribute: Attributname (href, src, data-*, etc.)

            Returns:
                Attributwert
            """
            agent = await self.browser.get_agent()
            value = await agent.extract_attribute(selector, attribute)
            return {"attribute": attribute, "value": value}

        async def tool_scrape_url(url: str) -> dict:
            """
            🌐 Navigiert zu URL und extrahiert Inhalt in einem Schritt.

            Args:
                url: Ziel-URL

            Returns:
                Vollständiger Seiteninhalt
            """
            agent = await self.browser.get_agent()
            await agent.goto(url)
            content = await agent.extract_markdown()
            return content.to_dict()

        self._tools.extend([
            ToolDefinition(
                name="extract",
                description="Extrahiert aktuellen Seiteninhalt als Markdown mit Struktur.",
                func=tool_extract,
                category=ToolCategory.EXTRACTION,
                parameters={
                    "include_links": {"type": "bool", "default": True},
                    "include_headings": {"type": "bool", "default": True}
                },
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="extract_text",
                description="Extrahiert reinen Text aus einem Element.",
                func=tool_extract_text,
                category=ToolCategory.EXTRACTION,
                parameters={"selector": {"type": "str", "default": "body"}},
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="extract_html",
                description="Extrahiert HTML aus einem Element.",
                func=tool_extract_html,
                category=ToolCategory.EXTRACTION,
                parameters={"selector": {"type": "str", "default": "body"}},
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="extract_links",
                description="Extrahiert alle Links (mit optionalem Filter).",
                func=tool_extract_links,
                category=ToolCategory.EXTRACTION,
                parameters={"filter_pattern": {"type": "str", "default": ""}},
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="extract_attribute",
                description="Extrahiert ein Attribut (href, src, data-*, etc.).",
                func=tool_extract_attribute,
                category=ToolCategory.EXTRACTION,
                parameters={
                    "selector": {"type": "str", "required": True},
                    "attribute": {"type": "str", "required": True}
                },
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="scrape_url",
                description="Navigiert zu URL und extrahiert Inhalt (All-in-One).",
                func=tool_scrape_url,
                category=ToolCategory.EXTRACTION,
                parameters={"url": {"type": "str", "required": True}},
                flags={"read": True, "write": True}
            ),
        ])

        # ====================================================================
        # SESSION TOOLS
        # ====================================================================

        async def tool_session_save(name: str) -> dict:
            """
            💾 Speichert Browser-Session (Cookies, LocalStorage).

            Args:
                name: Name der Session

            Returns:
                Pfad zur Session-Datei
            """
            agent = await self.browser.get_agent()
            path = await agent.save_state(name)
            return {"status": "saved", "name": name, "path": path}

        async def tool_session_load(name: str) -> dict:
            """
            📂 Lädt gespeicherte Browser-Session.

            Args:
                name: Name der Session

            Returns:
                Status
            """
            agent = await self.browser.get_agent()
            success = await agent.load_state(name)
            return {
                "status": "loaded" if success else "failed",
                "name": name,
                "success": success
            }

        async def tool_session_list() -> dict:
            """
            📋 Listet alle gespeicherten Sessions.

            Returns:
                Liste der Sessions
            """
            from pathlib import Path

            state_dir = Path(self.state_dir)
            if not state_dir.exists():
                return {"sessions": []}

            sessions = [
                f.stem for f in state_dir.glob("*.json")
                if not f.name.startswith(".")
            ]
            return {"sessions": sessions}

        async def tool_login(
            url: str,
            username_selector: str,
            password_selector: str,
            submit_selector: str,
            username: str,
            password: str,
            success_indicator: str = "",
            save_as: str = ""
        ) -> dict:
            """
            🔐 Führt Login durch und speichert optional Session.

            Args:
                url: Login-URL
                username_selector: Selector für Username-Feld
                password_selector: Selector für Passwort-Feld
                submit_selector: Selector für Submit-Button
                username: Benutzername
                password: Passwort
                success_indicator: Element das nach Login erscheint (optional)
                save_as: Session-Name zum Speichern (optional)

            Returns:
                Login-Status
            """
            agent = await self.browser.get_agent()
            success = await agent.login(
                url=url,
                username_selector=username_selector,
                password_selector=password_selector,
                submit_selector=submit_selector,
                username=username,
                password=password,
                success_indicator=success_indicator
            )

            result = {"success": success, "url": agent.current_url()}

            if success and save_as:
                await agent.save_state(save_as)
                result["session_saved"] = save_as

            return result

        self._tools.extend([
            ToolDefinition(
                name="session_save",
                description="Speichert Browser-Session (Cookies, LocalStorage) für späteren Login.",
                func=tool_session_save,
                category=ToolCategory.SESSION,
                parameters={"name": {"type": "str", "required": True}},
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="session_load",
                description="Lädt gespeicherte Session - kein erneuter Login nötig.",
                func=tool_session_load,
                category=ToolCategory.SESSION,
                parameters={"name": {"type": "str", "required": True}},
                flags={"read": False, "write": True}
            ),
            ToolDefinition(
                name="session_list",
                description="Zeigt alle gespeicherten Sessions.",
                func=tool_session_list,
                category=ToolCategory.SESSION,
                flags={"read": True, "write": False}
            ),
            ToolDefinition(
                name="login",
                description="Automatischer Login mit optionalem Session-Speichern.",
                func=tool_login,
                category=ToolCategory.SESSION,
                parameters={
                    "url": {"type": "str", "required": True},
                    "username_selector": {"type": "str", "required": True},
                    "password_selector": {"type": "str", "required": True},
                    "submit_selector": {"type": "str", "required": True},
                    "username": {"type": "str", "required": True},
                    "password": {"type": "str", "required": True},
                    "success_indicator": {"type": "str", "default": ""},
                    "save_as": {"type": "str", "default": ""}
                },
                flags={"read": False, "write": True}
            ),
        ])

        # ====================================================================
        # UTILITY TOOLS
        # ====================================================================

        async def tool_screenshot(name: str = "", full_page: bool = False) -> dict:
            """
            📸 Macht Screenshot der aktuellen Seite.

            Args:
                name: Dateiname (ohne .png)
                full_page: True für komplette Seite

            Returns:
                Pfad zum Screenshot
            """
            agent = await self.browser.get_agent()
            path = await agent.screenshot(name + ".png" if name else "", full_page=full_page)
            return {"path": path}

        async def tool_execute_js(script: str) -> dict:
            """
            🔧 Führt JavaScript auf der Seite aus.

            Args:
                script: JavaScript-Code

            Returns:
                Rückgabewert des Scripts
            """
            agent = await self.browser.get_agent()
            result = await agent.evaluate(script)
            return {"result": result}

        async def tool_get_logs() -> dict:
            """
            📋 Gibt Agent-Logs zurück.

            Returns:
                Log-Summary
            """
            agent = await self.browser.get_agent()
            return agent.logger.get_summary()

        self._tools.extend([
            ToolDefinition(
                name="screenshot",
                description="Macht Screenshot (für Debugging/Dokumentation).",
                func=tool_screenshot,
                category=ToolCategory.UTILITY,
                parameters={
                    "name": {"type": "str", "default": ""},
                    "full_page": {"type": "bool", "default": False}
                },
                flags={"read": True, "write": True,"no_thread": True}
            ),
            ToolDefinition(
                name="execute_js",
                description="Führt JavaScript auf der Seite aus.",
                func=tool_execute_js,
                category=ToolCategory.UTILITY,
                parameters={"script": {"type": "str", "required": True}},
                flags={"read": True, "write": True,"no_thread": True}
            ),
            ToolDefinition(
                name="get_logs",
                description="Zeigt Agent-Logs und Statistiken.",
                func=tool_get_logs,
                category=ToolCategory.UTILITY,
                flags={"read": True, "write": False,"no_thread": True}
            ),
        ])

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_tools(
        self,
        categories: list[ToolCategory] = None,
        include_browser_control: bool = True
    ) -> list[dict]:
        """
        Gibt Tools als Liste von Dictionaries zurück.

        Args:
            categories: Nur diese Kategorien (None = alle)
            include_browser_control: Browser-Start/Stop Tools einschließen

        Returns:
            Liste von Tool-Dictionaries für ToolBoxV2
        """
        tools = self._tools

        if categories:
            tools = [t for t in tools if t.category in categories]

        if not include_browser_control:
            tools = [t for t in tools if t.category != ToolCategory.BROWSER]

        return [t.to_dict() for t in tools]

    def get_tool_names(self) -> list[str]:
        """Liste aller Tool-Namen."""
        return [t.name for t in self._tools]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Einzelnes Tool by Name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    async def start_browser(self, headless: bool = True):
        """Browser manuell starten."""
        await self.browser.set_headless(headless)
        await self.browser.start()

    async def stop_browser(self):
        """Browser manuell stoppen."""
        await self.browser.stop()

    def __del__(self):
        """Cleanup beim Löschen."""
        if self.browser.is_running:
            try:
                asyncio.get_event_loop().run_until_complete(self.browser.stop())
            except:
                pass


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_full_tools(
    headless: bool = True,
    auto_start: bool = True,
    keep_open: bool = True,
    verbose: bool = False
) -> tuple[WebAgentToolkit, list[dict]]:
    """
    🔧 Gibt ALLE WebAgent Tools zurück (Full Pack).

    Enthält:
    - Browser Control (start, stop, status, set_headless)
    - Search (web_search, search_site, search_files)
    - Navigation (goto, back, refresh, current_url)
    - Interaction (click, type, select, scroll, wait, hover)
    - Extraction (extract, extract_text, extract_html, extract_links, scrape_url)
    - Session (session_save, session_load, session_list, login)
    - Utility (screenshot, execute_js, get_logs)

    Returns:
        Liste von 26 Tools
    """
    toolkit = WebAgentToolkit(
        headless=headless,
        auto_start=auto_start,
        keep_open=keep_open,
        verbose=verbose
    )
    return toolkit, toolkit.get_tools()


def get_minimal_tools(
    headless: bool = True,
    auto_start: bool = True,
    keep_open: bool = True,
    verbose: bool = False
) -> tuple[WebAgentToolkit, list[dict]]:
    """
    🔧 Gibt minimale WebAgent Tools zurück (Minimal Pack).

    Enthält nur:
    - web_search
    - scrape_url
    - extract
    - goto
    - click
    - type
    - session_load
    - session_save

    Returns:
        Liste von 8 essentiellen Tools
    """
    toolkit = WebAgentToolkit(
        headless=headless,
        auto_start=auto_start,
        keep_open=keep_open,
        verbose=verbose
    )

    minimal_names = [
        "web_search",
        "scrape_url",
        "extract",
        "goto",
        "click",
        "type",
        "session_load",
        "session_save"
    ]

    return toolkit, [t for t in toolkit.get_tools() if t["name"] in minimal_names]


def get_search_only_tools() -> list[dict]:
    """
    🔍 Nur Such-Tools (kein Browser nötig).

    Enthält:
    - web_search
    - search_site
    - search_files
    """
    toolkit = WebAgentToolkit(auto_start=False)
    return toolkit.get_tools(categories=[ToolCategory.SEARCH])


import multiprocessing as mp
import asyncio
import traceback
import threading
from typing import Any
from dataclasses import dataclass, field

# ================================================================
# Protocol: Messages zwischen Proxy ↔ Worker
# ================================================================

MSG_READY = "ready"
MSG_CALL = "call"
MSG_RESULT = "result"
MSG_SHUTDOWN = "shutdown"
MSG_STOPPED = "stopped"

# ================================================================
# Worker – eigener Prozess, eigener Main-Thread
# ================================================================

def _playwright_worker_main(
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict,
):
    """Entry point für den Browser-Prozess."""

    async def _async_main():
        # ── Setup ──
        if config.get("full"):
            from toolboxv2.mods.isaa.extras.web_helper.tooklit import WebAgentToolkit
            toolkit = WebAgentToolkit(
                headless=config.get("headless", True),
                auto_start=True,
                keep_open=True,
            )
            await toolkit.start_browser()
            tool_map = {t.name: t.func for t in toolkit._tools}
            tool_meta = [t.to_dict() for t in toolkit._tools]
            # Entferne nicht-serialisierbare Felder
            for meta in tool_meta:
                meta.pop("tool_func", None)
        else:
            from toolboxv2.mods.isaa.extras.web_helper.web_agent import (
                minimal_web_agent_integration,
            )
            raw = await minimal_web_agent_integration()
            tool_map = {t["name"]: t["func"] for t in raw}
            tool_meta = [{k: v for k, v in t.items() if k != "func"} for t in raw]
            toolkit = None

        # ── Ready Signal mit Metadaten ──
        result_queue.put({
            "type": MSG_READY,
            "tools": tool_meta,
        })

        # ── Command Loop ──
        while True:
            try:
                cmd = cmd_queue.get(timeout=0.5)
            except Exception:
                continue

            if cmd.get("type") == MSG_SHUTDOWN:
                break

            req_id = cmd["id"]
            tool_name = cmd["tool"]
            kwargs = cmd.get("kwargs", {})

            try:
                result = await tool_map[tool_name](**kwargs)
                result_queue.put({
                    "type": MSG_RESULT,
                    "id": req_id,
                    "data": result,
                })
            except Exception as e:
                result_queue.put({
                    "type": MSG_RESULT,
                    "id": req_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        # ── Cleanup ──
        if toolkit:
            await toolkit.stop_browser()
        result_queue.put({"type": MSG_STOPPED})

    asyncio.run(_async_main())


# ================================================================
# Proxy – läuft im Agent-Prozess, beliebiger Thread
# ================================================================

class PlaywrightProxy:
    """Thread-safe proxy zum Browser-Worker-Prozess."""

    def __init__(self, full: bool = True, headless: bool = True):
        self._config = {"full": full, "headless": headless}
        self._cmd_q = mp.Queue()
        self._result_q = mp.Queue()
        self._proc: mp.Process | None = None
        self._tool_meta: list[dict] = []
        self._lock = threading.Lock()
        self._counter = 0

    # ── Lifecycle ──

    def start(self, timeout: float = 30):
        self._proc = mp.Process(
            target=_playwright_worker_main,
            args=(self._cmd_q, self._result_q, self._config),
            daemon=True,
            name="playwright-worker",
        )
        self._proc.start()

        msg = self._result_q.get(timeout=timeout)
        if msg.get("type") != MSG_READY:
            raise RuntimeError(f"Worker startup failed: {msg}")

        self._tool_meta = msg["tools"]

    def shutdown(self):
        if not self._proc or not self._proc.is_alive():
            return
        self._cmd_q.put({"type": MSG_SHUTDOWN})
        try:
            self._result_q.get(timeout=15)
        except Exception:
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()
        self._proc = None

    # ── Tool Calls ──

    def call_tool(self, name: str, **kwargs) -> Any:
        """Thread-safe synchroner Tool-Call."""
        with self._lock:
            self._counter += 1
            req_id = self._counter

        self._cmd_q.put({
            "type": MSG_CALL,
            "id": req_id,
            "tool": name,
            "kwargs": kwargs,
        })

        resp = self._result_q.get(timeout=120)

        if resp.get("error"):
            raise RuntimeError(
                f"Tool '{name}' failed: {resp['error']}\n{resp.get('traceback', '')}"
            )
        return resp["data"]

    # ── Tool Definitions für Agent ──

    def build_agent_tools(self) -> list[dict]:
        """Baut vollständige Tool-Dicts mit sync callables."""
        tools = []
        for meta in self._tool_meta:
            name = meta["name"]

            def _make_fn(tool_name: str):
                def fn(**kw):
                    return self.call_tool(tool_name, **kw)
                fn.__name__ = tool_name
                fn.__doc__ = meta.get("description", "")
                return fn

            tool_dict = {
                **meta,
                "tool_func": _make_fn(name),
            }
            tools.append(tool_dict)

        return tools




# ============================================================================
# CLI / TEST
# ============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  WebAgent Tools - Full Pack")
    print("=" * 70)

    kit, full = get_full_tools()
    print(f"\n📦 Full Pack: {len(full)} Tools\n")

    for tool in full:
        flags = "R" if tool["flags"]["read"] else "-"
        flags += "W" if tool["flags"]["write"] else "-"
        print(f"  [{flags}] {tool['name']:20} - {tool['description'][:50]}...")

    print("\n" + "=" * 70)
    print("  WebAgent Tools - Minimal Pack")
    print("=" * 70)

    _, minimal = get_minimal_tools()
    print(f"\n📦 Minimal Pack: {len(minimal)} Tools\n")

    for tool in minimal:
        print(f"  • {tool['name']:20} - {tool['description'][:50]}...")

    print("\n" + "=" * 70)
    print("  Categories")
    print("=" * 70)

    toolkit = kit
    for cat in ToolCategory:
        tools = [t for t in toolkit._tools if t.category == cat]
        if tools:
            print(f"\n  {cat.value.upper()} ({len(tools)} tools):")
            for t in tools:
                print(f"    - {t.name}")
    async def test():
        await toolkit.start_browser(False)
        await asyncio.sleep(8000)
        await toolkit.stop_browser()
    print("\n" + "=" * 70)
    print("  Usage Example")
    print("=" * 70)
    print("""
    # Option 1: Full Tools
    from web_agent_tools import get_full_tools
    tools = get_full_tools()

    # Option 2: Minimal Tools
    from web_agent_tools import get_minimal_tools
    tools = get_minimal_tools()

    # Option 3: Toolkit mit Kontrolle
    from web_agent_tools import WebAgentToolkit
    toolkit = WebAgentToolkit(headless=False, keep_open=True)
    tools = toolkit.get_tools()
    await toolkit.start_browser()
    ...
    await toolkit.stop_browser()

    # Option 4: ToolBoxV2 Registration
    from web_agent_tools import register_with_toolbox
    register_with_toolbox(app, get_full_tools())
    """)
    asyncio.run(test())
