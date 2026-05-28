"""
web_shell_tool.py
=================
Unix-like shell interface for web interaction with learning memory.

Provides ONE primary agent tool:

  web_shell(reason, command)  — All web operations via unix-like commands

Memory layers (persisted in /global/web/ via VFS):
  Layer 1 — Site Templates:  known selectors per domain (auto-learned)
  Layer 2 — Flows:           multi-step sequences as named commands

Usage in init_session_tools():
    from toolboxv2.mods.isaa.base.patch.web_shell_tool import make_web_shell

    web_shell = make_web_shell(session, headless=False)

Commands:
    goto <url>                     Navigate to URL
    click <selector>               Click element
    type <selector> <text>         Type into element
    fill <selector1>=<val1> ...    Fill multiple fields
    select <selector> <value>      Select dropdown option
    extract [selector] [--md]      Extract content (default: markdown)
    screenshot [path] [--ocr]      Screenshot, optional OCR
    ocr <path_or_url> [--tier T]   OCR a file/screenshot (fast|balanced|accurate|api)
    eval <js_code>                 Execute JavaScript
    wait [selector] [--timeout N]  Wait for element or networkidle
    scroll <direction> [amount]    Scroll page
    back                           Navigate back
    refresh                        Reload page
    search <query> [--site S]      SearXNG search (no browser needed)
    login <flow_or_url>            Login using saved credentials
    status                         Current URL, title, session info
    tabs                           List open tabs
    save_template [domain]         Save current site's selectors
    save_flow <name> <steps_json>  Save a multi-step flow
    run_flow <name> [params_json]  Execute a saved flow
    list_flows [domain]            List saved flows
    list_templates                 List known site templates
    close_browser                  Explicitly close (auto-closed on session end)
"""

from __future__ import annotations

import json
import os
import shlex
import time
from datetime import datetime, UTC
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2

from toolboxv2 import get_logger

logger = get_logger()


# =============================================================================
# HELPERS
# =============================================================================

def _ok(stdout: str = "", stderr: str = "", returncode: int = 0) -> dict:
    return {"success": returncode == 0, "stdout": stdout, "stderr": stderr, "returncode": returncode}


def _err(stderr: Any, returncode: int = 1) -> dict:
    return {"success": False, "stdout": "", "stderr": str(stderr), "returncode": returncode}


def _domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def _truncate(text: str, max_chars: int = 4000) -> str:
    """Token-efficient truncation with indicator."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[... {len(text) - max_chars} chars truncated ...]\n\n{text[-half:]}"


# =============================================================================
# SITE TEMPLATE (Layer 1) — Low-level per-domain selector memory
# =============================================================================

class SiteTemplate:
    """Known selectors and patterns for a specific domain."""

    __slots__ = ("domain", "selectors", "meta_selectors", "cookie_selectors",
                 "login_selectors", "content_selectors", "updated_at")

    def __init__(self, domain: str, data: dict | None = None):
        self.domain = domain
        data = data or {}
        self.selectors: dict[str, str] = data.get("selectors", {})
        self.meta_selectors: dict[str, str] = data.get("meta_selectors", {})
        self.cookie_selectors: list[str] = data.get("cookie_selectors", [])
        self.login_selectors: dict[str, str] = data.get("login_selectors", {})
        self.content_selectors: dict[str, str] = data.get("content_selectors", {})
        self.updated_at: str = data.get("updated_at", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "selectors": self.selectors,
            "meta_selectors": self.meta_selectors,
            "cookie_selectors": self.cookie_selectors,
            "login_selectors": self.login_selectors,
            "content_selectors": self.content_selectors,
            "updated_at": self.updated_at,
        }

    def learn_selector(self, category: str, name: str, selector: str):
        """Learn a new selector for this domain."""
        if category == "cookie":
            if selector not in self.cookie_selectors:
                self.cookie_selectors.append(selector)
        elif category == "login":
            self.login_selectors[name] = selector
        elif category == "content":
            self.content_selectors[name] = selector
        else:
            self.selectors[name] = selector
        self.updated_at = datetime.now(UTC).isoformat()


# =============================================================================
# FLOW (Layer 2) — High-level multi-step sequences
# =============================================================================

class Flow:
    """A named sequence of web_shell commands for a domain."""

    __slots__ = ("name", "domain", "description", "steps", "params",
                 "created_at", "run_count", "last_run")

    def __init__(self, name: str, domain: str, data: dict | None = None):
        self.name = name
        self.domain = domain
        data = data or {}
        self.description: str = data.get("description", "")
        self.steps: list[str] = data.get("steps", [])
        self.params: list[str] = data.get("params", [])  # placeholder names like {username}
        self.created_at: str = data.get("created_at", datetime.now(UTC).isoformat())
        self.run_count: int = data.get("run_count", 0)
        self.last_run: str = data.get("last_run", "")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "steps": self.steps,
            "params": self.params,
            "created_at": self.created_at,
            "run_count": self.run_count,
            "last_run": self.last_run,
        }


# =============================================================================
# WEB MEMORY MANAGER — VFS-backed persistence for templates + flows
# =============================================================================

class WebMemoryManager:
    """Manages site templates and flows via VFS at /global/web/."""

    TEMPLATES_BASE = "/global/web/templates"
    FLOWS_BASE = "/global/web/flows"
    SESSIONS_BASE = "/global/web/sessions"
    GUIDE_PATH = "/global/web/_guide.md"

    def __init__(self, vfs):
        self.vfs = vfs
        self._templates: dict[str, SiteTemplate] = {}
        self._flows: dict[str, dict[str, Flow]] = {}  # domain -> {name -> Flow}
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create base directories if they don't exist."""
        for path in (self.TEMPLATES_BASE, self.FLOWS_BASE, self.SESSIONS_BASE):
            if not self.vfs._is_directory(path):
                self.vfs.mkdir(path, parents=True)

    # ── Template (Layer 1) ────────────────────────────────────────────

    def load_template(self, domain: str) -> SiteTemplate | None:
        """Load a site template from VFS."""
        if domain in self._templates:
            return self._templates[domain]

        path = f"{self.TEMPLATES_BASE}/{domain}/selectors.json"
        r = self.vfs.read(path)
        if not r.get("success"):
            return None

        try:
            data = json.loads(r["content"])
            t = SiteTemplate(domain, data)
            self._templates[domain] = t
            return t
        except (json.JSONDecodeError, KeyError):
            return None

    def save_template(self, template: SiteTemplate):
        """Save a site template to VFS."""
        domain = template.domain
        dir_path = f"{self.TEMPLATES_BASE}/{domain}"
        if not self.vfs._is_directory(dir_path):
            self.vfs.mkdir(dir_path, parents=True)

        path = f"{dir_path}/selectors.json"
        content = json.dumps(template.to_dict(), indent=2, ensure_ascii=False)

        if self.vfs._is_file(path):
            self.vfs.write(path, content)
        else:
            self.vfs.create(path, content)

        self._templates[domain] = template
        self._update_guide()

    def get_or_create_template(self, domain: str) -> SiteTemplate:
        """Get existing or create new template for domain."""
        t = self.load_template(domain)
        if t is None:
            t = SiteTemplate(domain)
            self._templates[domain] = t
        return t

    def list_templates(self) -> list[str]:
        """List all known domain templates."""
        domains = []
        r = self.vfs.ls(self.TEMPLATES_BASE)
        if r.get("success"):
            for item in r.get("contents", []):
                if item["type"] == "directory":
                    domains.append(item["name"])
        return sorted(domains)

    # ── Flow (Layer 2) ────────────────────────────────────────────────

    def load_flow(self, domain: str, name: str) -> Flow | None:
        """Load a flow from VFS."""
        cache = self._flows.get(domain, {})
        if name in cache:
            return cache[name]

        path = f"{self.FLOWS_BASE}/{domain}/{name}.json"
        r = self.vfs.read(path)
        if not r.get("success"):
            return None

        try:
            data = json.loads(r["content"])
            f = Flow(name, domain, data)
            self._flows.setdefault(domain, {})[name] = f
            return f
        except (json.JSONDecodeError, KeyError):
            return None

    def save_flow(self, flow: Flow):
        """Save a flow to VFS."""
        dir_path = f"{self.FLOWS_BASE}/{flow.domain}"
        if not self.vfs._is_directory(dir_path):
            self.vfs.mkdir(dir_path, parents=True)

        path = f"{dir_path}/{flow.name}.json"
        content = json.dumps(flow.to_dict(), indent=2, ensure_ascii=False)

        if self.vfs._is_file(path):
            self.vfs.write(path, content)
        else:
            self.vfs.create(path, content)

        self._flows.setdefault(flow.domain, {})[flow.name] = flow
        self._update_guide()

    def list_flows(self, domain: str | None = None) -> dict[str, list[str]]:
        """List all flows, optionally filtered by domain."""
        result: dict[str, list[str]] = {}
        r = self.vfs.ls(self.FLOWS_BASE)
        if not r.get("success"):
            return result

        for item in r.get("contents", []):
            if item["type"] != "directory":
                continue
            d = item["name"]
            if domain and d != domain:
                continue
            flow_r = self.vfs.ls(f"{self.FLOWS_BASE}/{d}")
            if flow_r.get("success"):
                names = [
                    i["name"].replace(".json", "")
                    for i in flow_r.get("contents", [])
                    if i["type"] == "file" and i["name"].endswith(".json")
                ]
                if names:
                    result[d] = sorted(names)

        return result

    # ── Session State ─────────────────────────────────────────────────

    def load_session_state(self, domain: str) -> dict | None:
        """Load Playwright storage_state for domain."""
        path = f"{self.SESSIONS_BASE}/{domain}/state.json"
        r = self.vfs.read(path)
        if not r.get("success"):
            return None
        try:
            return json.loads(r["content"])
        except json.JSONDecodeError:
            return None

    def save_session_state(self, domain: str, state: dict):
        """Save Playwright storage_state for domain."""
        dir_path = f"{self.SESSIONS_BASE}/{domain}"
        if not self.vfs._is_directory(dir_path):
            self.vfs.mkdir(dir_path, parents=True)

        path = f"{dir_path}/state.json"
        content = json.dumps(state, indent=2, ensure_ascii=False)

        if self.vfs._is_file(path):
            self.vfs.write(path, content)
        else:
            self.vfs.create(path, content)

    # ── Guide Generator ───────────────────────────────────────────────

    def _update_guide(self):
        """Regenerate /global/web/_guide.md with current templates + flows."""
        lines = [
            "# Web Shell Guide",
            "",
            f"*Auto-generated: {datetime.now(UTC).isoformat()}*",
            "",
            "## Known Sites (Layer 1 — Templates)",
            "",
        ]

        templates = self.list_templates()
        if templates:
            for d in templates:
                t = self.load_template(d)
                sel_count = 0
                if t:
                    sel_count = (len(t.selectors) + len(t.cookie_selectors) +
                                 len(t.login_selectors) + len(t.content_selectors))
                lines.append(f"- **{d}** — {sel_count} selectors")
        else:
            lines.append("*(no templates yet)*")

        lines.extend(["", "## Learned Flows (Layer 2 — Sequences)", ""])

        all_flows = self.list_flows()
        if all_flows:
            for domain, names in all_flows.items():
                lines.append(f"### {domain}")
                for name in names:
                    f = self.load_flow(domain, name)
                    desc = f.description if f else ""
                    steps = len(f.steps) if f else 0
                    runs = f.run_count if f else 0
                    lines.append(f"- `{name}` — {steps} steps, {runs} runs. {desc}")
                lines.append("")
        else:
            lines.append("*(no flows yet)*")

        lines.extend([
            "",
            "## Session Data",
            "",
        ])
        r = self.vfs.ls(self.SESSIONS_BASE)
        if r.get("success"):
            domains = [i["name"] for i in r.get("contents", []) if i["type"] == "directory"]
            if domains:
                for d in sorted(domains):
                    lines.append(f"- {d} (stored cookies/state)")
            else:
                lines.append("*(no saved sessions)*")

        lines.extend([
            "",
            "## Quick Reference",
            "",
            "```",
            'web_shell "reason" "goto https://example.com"',
            'web_shell "reason" "extract main --md"',
            'web_shell "reason" "click .submit"',
            'web_shell "reason" "fill #user=admin #pass=secret"',
            'web_shell "reason" "screenshot --ocr"',
            'web_shell "reason" "search Python tutorial --site realpython.com"',
            'web_shell "reason" "run_flow github_login"',
            "```",
        ])

        guide_content = "\n".join(lines)

        # Write via VFS — this gets picked up by set_gen_system_file for context visibility
        if self.vfs._is_file(self.GUIDE_PATH):
            self.vfs.write(self.GUIDE_PATH, guide_content)
        else:
            self.vfs.create(self.GUIDE_PATH, guide_content)


# =============================================================================
# WEB SHELL FACTORY
# =============================================================================

def make_web_shell(
    session: "AgentSessionV2",
    headless: bool = False,
    single_site: str | None = None,
    trusted_sites: list[str] | None = None,
    enable_ocr: bool = True,
    ocr_default_tier: str = "fast",
    searxng_url: str = "",
):
    """
    Factory — returns async web_shell closure bound to *session*.

    Args:
        session:        AgentSessionV2 instance
        headless:       False = debug mode (visible browser), True = headless
        single_site:    If set, restrict navigation to this domain only
        trusted_sites:  List of domains that may use saved credentials
        enable_ocr:     Enable OCR integration for screenshots
        ocr_default_tier: Default OCR tier (fast|balanced|accurate|api)
        searxng_url:    Custom SearXNG instance URL
    """
    vfs = session.vfs
    memory = WebMemoryManager(vfs)

    from toolboxv2.mods.isaa.extras.web_helper.web_agent import WebAgent
    agent = WebAgent(headless=headless, searxng_url=searxng_url, verbose=False)
    _started = False

    async def _ensure_agent():
       nonlocal _started
       if _started:
           return True
       try:
            await agent.start()
       except Exception as exc:
           logger.error("WebAgent start failed: %s", exc)
           import traceback
           traceback.print_exc()
           _started = False # ← Erlaubt Retry bei nächstem Aufruf
           return False

      # Validierung: Sind die Methoden tatsächlich verfügbar?
       if not callable(getattr(agent, "goto", None)):
           logger.error("agent.goto nicht verfügbar nach Start")
           return False # ← Erlaubt Retry bei nächstem Aufruf

       _started = True
       return True

    # Write initial guide to VFS
    memory._update_guide()

    # Make guide visible in agent context via set_gen_system_file
    guide_r = vfs.read(memory.GUIDE_PATH)
    if guide_r.get("success"):
        vfs.set_gen_system_file(guide_r["content"], path=memory.GUIDE_PATH)

    async def web_shell(reason: str, command: str) -> dict:
        """
        Unix-like shell for web interaction with learning memory.

        Layer 1 (Site Templates): Auto-learns selectors per domain.
        Layer 2 (Flows): Saves multi-step sequences as named commands.

        Commands
        --------
        NAVIGATION  goto <url>  |  back  |  refresh  |  wait [selector]
        INTERACT    click <sel>  |  type <sel> <text>  |  fill <sel>=<val> ...
                    select <sel> <value>  |  scroll <dir> [amount]
        EXTRACT     extract [selector] [--md|--text|--links|--headings]
        CAPTURE     screenshot [path] [--ocr] [--tier T]
                    ocr <path_or_url> [--tier T]
        EXECUTE     eval <js_code>
        SEARCH      search <query> [--site S] [--max N]
        AUTH        login <domain_or_flow>
        MEMORY      save_template [domain]  |  list_templates
                    save_flow <name> <steps_json>  |  run_flow <name> [params]
                    list_flows [domain]
        SESSION     status  |  tabs  |  save_session [domain]  |  load_session <domain>
                    close_browser

        Returns
        -------
        {"success": bool, "stdout": str, "stderr": str, "returncode": int}
        """
        nonlocal _started
        command = command.strip()
        if not command:
            return _err("empty command")

        # Parse
        try:
            args = shlex.split(command)
        except ValueError:
            args = command.split()

        if not args:
            return _err("empty command")

        cmd, rest = args[0].lower(), args[1:]

        # ── Flag extraction helpers ───────────────────────────────────
        def _pop_flag(name: str, default: str | None = None) -> str | None:
            """Extract --name VALUE from rest, return value."""
            nonlocal rest
            for i, a in enumerate(rest):
                if a == f"--{name}" and i + 1 < len(rest):
                    val = rest[i + 1]
                    rest = rest[:i] + rest[i + 2:]
                    return val
                if a.startswith(f"--{name}="):
                    val = a.split("=", 1)[1]
                    rest = rest[:i] + rest[i + 1:]
                    return val
            return default

        def _has_flag(name: str) -> bool:
            nonlocal rest
            for i, a in enumerate(rest):
                if a == f"--{name}":
                    rest = rest[:i] + rest[i + 1:]
                    return True
            return False

        # ═══════════════════════════════════════════════════════════════
        # goto <url>
        # ═══════════════════════════════════════════════════════════════
        if cmd == "goto":
            if not rest:
                return _err("goto: missing URL")
            url = rest[0]
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            domain = _domain(url)

            # Single-site restriction
            if single_site and domain != _domain(single_site):
                return _err(f"goto: restricted to {single_site}, cannot navigate to {domain}")

            if not await _ensure_agent():
                return _err("Browser konnte nicht starten – goto nicht möglich")

            # Load saved session state for this domain
            saved_state = memory.load_session_state(domain)
            if saved_state:
                # WebAgent.load_state expects a name, but we manage state via VFS.
                # Write temp state file and load it through WebAgent's mechanism.
                import tempfile, json as _json
                _state_path = os.path.join(agent.state_dir, f"{domain}.json")
                with open(_state_path, 'w') as _f:
                    _json.dump(saved_state, _f)
                await agent.load_state(domain)

            try:
                wait_until = _pop_flag("wait", "domcontentloaded")
                success = await agent.goto(url, wait_until=wait_until)
                if not success:
                    return _err(f"goto failed: navigation error")

                title = await agent.title()

                # Auto-learn: try to dismiss cookie banner with known selectors
                template = memory.get_or_create_template(domain)
                if template.cookie_selectors:
                    for sel in template.cookie_selectors:
                        try:
                            await agent.click(sel)
                            break
                        except Exception:
                            continue

                return _ok(f"→ {url}\nTitle: {title}")

            except Exception as e:
                return _err(f"goto failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # back
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "back":
            if not _started:
                return _err("no browser session")
            success = await agent.back()
            if not success:
                return _err("no history to go back")
            title = await agent.title()
            return _ok(f"← {agent.current_url()}\nTitle: {title}")

        # ═══════════════════════════════════════════════════════════════
        # refresh
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "refresh":
            if not _started:
                return _err("no browser session")
            await agent.refresh()
            return _ok(f"Refreshed: {agent.current_url()}")

        # ═══════════════════════════════════════════════════════════════
        # wait [selector] [--timeout N]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "wait":
            if not _started:
                return _err("no browser session")
            timeout = int(_pop_flag("timeout", "30000"))
            selector = rest[0] if rest else ""
            await agent.wait(selector, timeout=timeout)
            return _ok(f"Found: {selector}" if selector else "networkidle reached")

        # ═══════════════════════════════════════════════════════════════
        # click <selector>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "click":
            if not _started:
                return _err("no browser session")
            if not rest:
                return _err("click: missing selector")
            selector = rest[0]
            success = await agent.click(selector)
            if not success:
                return _err(f"click failed on {selector}")

            # Auto-learn selector
            domain = _domain(agent.current_url())
            template = memory.get_or_create_template(domain)
            template.learn_selector("general", f"clicked_{len(template.selectors)}", selector)

            return _ok(f"Clicked: {selector}")

        # ═══════════════════════════════════════════════════════════════
        # type <selector> <text>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "type":
            if not _started:
                return _err("no browser session")
            if len(rest) < 2:
                return _err("type: usage: type <selector> <text>")
            selector = rest[0]
            text = " ".join(rest[1:])
            success = await agent.type(selector, text)
            if not success:
                return _err(f"type failed on {selector}")
            return _ok(f"Typed {len(text)} chars into {selector}")

        # ═══════════════════════════════════════════════════════════════
        # fill <sel1>=<val1> <sel2>=<val2> ...
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "fill":
            if not _started:
                return _err("no browser session")
            if not rest:
                return _err("fill: usage: fill <selector>=<value> ...")
            filled = []
            for pair in rest:
                if "=" not in pair:
                    return _err(f"fill: invalid pair '{pair}' — use selector=value")
                sel, val = pair.split("=", 1)
                success = await agent.type(sel, val)
                if not success:
                    return _err(f"fill failed on {sel}")
                filled.append(sel)

            # Auto-learn login selectors if filling typical auth fields
            domain = _domain(agent.current_url())
            template = memory.get_or_create_template(domain)
            for pair in rest:
                sel, _ = pair.split("=", 1)
                for kw in ("user", "email", "login", "name"):
                    if kw in sel.lower():
                        template.learn_selector("login", "username_field", sel)
                for kw in ("pass", "pwd", "secret"):
                    if kw in sel.lower():
                        template.learn_selector("login", "password_field", sel)

            return _ok(f"Filled: {', '.join(filled)}")

        # ═══════════════════════════════════════════════════════════════
        # select <selector> <value>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "select":
            if not _started:
                return _err("no browser session")
            if len(rest) < 2:
                return _err("select: usage: select <selector> <value>")
            selector, value = rest[0], rest[1]
            success = await agent.select(selector, value=value)
            if not success:
                return _err(f"select failed on {selector}")
            return _ok(f"Selected '{value}' in {selector}")

        # ═══════════════════════════════════════════════════════════════
        # scroll <direction> [amount]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "scroll":
            if not _started:
                return _err("no browser session")
            direction = rest[0] if rest else "down"
            amount = int(rest[1]) if len(rest) > 1 else 500
            await agent.scroll(direction, amount)
            return _ok(f"Scrolled {direction} {amount}px")

        # ═══════════════════════════════════════════════════════════════
        # extract [selector] [--md|--text|--links|--headings]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "extract":
            if not _started:
                return _err("no browser session")

            mode = "md"
            if _has_flag("text"):
                mode = "text"
            elif _has_flag("links"):
                mode = "links"
            elif _has_flag("headings"):
                mode = "headings"
            elif _has_flag("md"):
                mode = "md"

            selector = rest[0] if rest else None

            try:
                if mode == "text":
                    text = await agent.extract_text(selector or "body")
                    return _ok(_truncate(text))

                elif mode == "links":
                    content = await agent.extract_markdown()
                    lines = [l.to_markdown() for l in content.links[:50]]
                    return _ok("\n".join(lines) if lines else "(no links found)")

                elif mode == "headings":
                    content = await agent.extract_markdown()
                    lines = [h.to_markdown() for h in content.headings]
                    return _ok("\n".join(lines) if lines else "(no headings found)")

                else:  # md
                    from toolboxv2.mods.isaa.extras.web_helper.web_agent import scraped_content2md_str
                    content = await agent.extract_markdown()
                    output = _truncate(scraped_content2md_str(content, include_toc=False))

                    # Auto-learn content selectors
                    domain = _domain(agent.current_url())
                    template = memory.get_or_create_template(domain)
                    for sel_name in ("main", "article", '[role="main"]', ".content", "#content"):
                        try:
                            found = await agent._page.query_selector(sel_name)
                            if found:
                                template.learn_selector("content", "main_content", sel_name)
                                break
                        except Exception:
                            continue

                    return _ok(output)

            except Exception as e:
                return _err(f"extract failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # screenshot [path] [--ocr] [--tier T]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "screenshot":
            if not _started:
                return _err("no browser session")

            do_ocr = _has_flag("ocr") or enable_ocr
            tier = _pop_flag("tier", ocr_default_tier)
            full_page = _has_flag("full")

            path = rest[0] if rest else f"/tmp/screenshot_{int(time.time())}.png"

            try:
                saved_path = await agent.screenshot(path, full_page=full_page)
                output = f"Screenshot saved: {saved_path}"

                if do_ocr:
                    try:
                        from toolboxv2.mods.isaa.extras.ocr_engine import OCRRouter, IsaaOCRConfig
                        router = OCRRouter(IsaaOCRConfig())
                        result = await router.ocr(path, tier=tier)
                        ocr_text = result.to_markdown()
                        output += f"\n\n--- OCR ({result.engine}, {result.elapsed_s:.1f}s) ---\n{_truncate(ocr_text)}"
                    except Exception as ocr_err:
                        output += f"\nOCR failed: {ocr_err}"

                return _ok(output)

            except Exception as e:
                return _err(f"screenshot failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # ocr <path_or_url> [--tier T]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "ocr":
            if not rest:
                return _err("ocr: missing path or URL")
            source = rest[0]
            tier = _pop_flag("tier", ocr_default_tier)

            try:
                from toolboxv2.mods.isaa.extras.ocr_engine import OCRRouter, IsaaOCRConfig
                router = OCRRouter(IsaaOCRConfig())
                result = await router.ocr(source, tier=tier)
                output = (
                    f"Engine: {result.engine} | Tier: {result.tier.value} | "
                    f"Pages: {len(result.pages)} | Time: {result.elapsed_s:.1f}s\n\n"
                    f"{_truncate(result.to_markdown())}"
                )
                return _ok(output)
            except Exception as e:
                return _err(f"ocr failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # eval <js_code>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "eval":
            if not _started:
                return _err("no browser session")
            if not rest:
                return _err("eval: missing JavaScript code")
            js = " ".join(rest)
            try:
                result = await agent.evaluate(js)
                return _ok(str(result) if result is not None else "(undefined)")
            except Exception as e:
                return _err(f"eval failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # search <query> [--site S] [--max N]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "search":
            if not rest:
                return _err("search: missing query")

            site = _pop_flag("site")
            max_results = int(_pop_flag("max", "5"))
            query = " ".join(rest)

            try:

                if not await _ensure_agent():
                    return _err("Browser konnte nicht starten – goto nicht möglich")

                if site:
                    query = agent.search.build_dork(query, site=site)
                results = await agent.search.search(query, max_results=max_results)

                lines = [f"Search: {results.query} ({results.total_results} results)"]
                for r in results.results:
                    lines.append(f"  [{r.position}] {r.title}")
                    lines.append(f"      {r.url}")
                    if r.snippet:
                        lines.append(f"      {r.snippet[:120]}")
                    lines.append("")

                return _ok("\n".join(lines))
            except Exception as e:
                return _err(f"search failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # status
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "status":
            if not _started:
                return _ok("Browser: not started\nTemplates: " +
                           str(len(memory.list_templates())) +
                           "\nFlows: " + str(sum(len(v) for v in memory.list_flows().values())))

            title = await agent.title()
            url = agent.current_url()
            domain = _domain(url)
            template = memory.load_template(domain)
            t_info = f"{len(template.selectors)} selectors" if template else "no template"

            return _ok(
                f"URL: {url}\n"
                f"Title: {title}\n"
                f"Domain: {domain} ({t_info})\n"
                f"History: {len(agent._history)} pages\n"
                f"Headless: {agent.headless}\n"
                f"Templates: {len(memory.list_templates())}\n"
                f"Flows: {sum(len(v) for v in memory.list_flows().values())}"
            )

        # ═══════════════════════════════════════════════════════════════
        # save_template [domain]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "save_template":
            domain = rest[0] if rest else _domain(agent.current_url()) if _started else None
            if not domain:
                return _err("save_template: no domain (navigate first or specify domain)")

            template = memory.get_or_create_template(domain)
            memory.save_template(template)

            # Update guide in agent context
            guide_r = vfs.read(memory.GUIDE_PATH)
            if guide_r.get("success"):
                vfs.set_gen_system_file(guide_r["content"], path=memory.GUIDE_PATH)

            return _ok(f"Template saved for {domain} ({len(template.selectors)} selectors)")

        # ═══════════════════════════════════════════════════════════════
        # list_templates
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "list_templates":
            templates = memory.list_templates()
            if not templates:
                return _ok("(no templates)")
            lines = []
            for d in templates:
                t = memory.load_template(d)
                count = len(t.selectors) + len(t.content_selectors) if t else 0
                lines.append(f"  {d}: {count} selectors")
            return _ok("\n".join(lines))

        # ═══════════════════════════════════════════════════════════════
        # save_flow <name> <steps_json_or_quoted>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "save_flow":
            if len(rest) < 2:
                return _err("save_flow: usage: save_flow <name> <json_steps>")

            name = rest[0]
            steps_raw = " ".join(rest[1:])
            domain = _pop_flag("domain") or (_domain(agent.current_url()) if _started else "global")

            try:
                data = json.loads(steps_raw)
            except json.JSONDecodeError:
                return _err("save_flow: steps must be valid JSON (list of command strings or dict)")

            if isinstance(data, list):
                flow = Flow(name, domain, {"steps": data})
            elif isinstance(data, dict):
                flow = Flow(name, domain, data)
            else:
                return _err("save_flow: steps must be a JSON list or dict")

            memory.save_flow(flow)

            # Update guide
            guide_r = vfs.read(memory.GUIDE_PATH)
            if guide_r.get("success"):
                vfs.set_gen_system_file(guide_r["content"], path=memory.GUIDE_PATH)

            return _ok(f"Flow '{name}' saved for {domain} ({len(flow.steps)} steps)")

        # ═══════════════════════════════════════════════════════════════
        # run_flow <name> [params_json]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "run_flow":
            if not rest:
                return _err("run_flow: missing flow name")

            name = rest[0]
            domain = _pop_flag("domain") or (_domain(agent.current_url()) if _started else None)

            # Find flow — try current domain first, then global, then all domains
            flow = None
            if domain:
                flow = memory.load_flow(domain, name)
            if not flow:
                flow = memory.load_flow("global", name)
            if not flow:
                # Search all domains
                all_flows = memory.list_flows()
                for d, names in all_flows.items():
                    if name in names:
                        flow = memory.load_flow(d, name)
                        break

            if not flow:
                return _err(f"run_flow: flow '{name}' not found")

            # Parse params if provided
            params = {}
            if len(rest) > 1:
                params_raw = " ".join(rest[1:])
                try:
                    params = json.loads(params_raw)
                except json.JSONDecodeError:
                    return _err("run_flow: params must be valid JSON dict")

            # Execute steps sequentially
            outputs: list[str] = []
            for i, step in enumerate(flow.steps):
                # Substitute params
                for k, v in params.items():
                    step = step.replace(f"{{{k}}}", str(v))

                result = await web_shell(f"flow:{name} step {i+1}/{len(flow.steps)}", step)
                outputs.append(f"[{i+1}] {step[:60]}... → {'✓' if result.get('success') else '✗'}")

                if not result.get("success"):
                    outputs.append(f"    Error: {result.get('stderr', '')[:100]}")
                    break

            # Update run count
            flow.run_count += 1
            flow.last_run = datetime.now(UTC).isoformat()
            memory.save_flow(flow)

            return _ok("\n".join(outputs))

        # ═══════════════════════════════════════════════════════════════
        # list_flows [domain]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "list_flows":
            domain = rest[0] if rest else None
            all_flows = memory.list_flows(domain)
            if not all_flows:
                return _ok("(no flows)")
            lines = []
            for d, names in all_flows.items():
                lines.append(f"  {d}/")
                for n in names:
                    f = memory.load_flow(d, n)
                    runs = f.run_count if f else 0
                    steps = len(f.steps) if f else 0
                    lines.append(f"    {n}: {steps} steps, {runs} runs")
            return _ok("\n".join(lines))

        # ═══════════════════════════════════════════════════════════════
        # login <domain_or_flow>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "login":
            if not rest:
                return _err("login: missing domain or flow name")
            target = rest[0]
            domain = _domain(target) if "." in target else target

            # Check trusted sites
            if trusted_sites and domain not in trusted_sites:
                return _err(f"login: {domain} is not in trusted_sites list")

            # Try to find a login flow first
            flow = memory.load_flow(domain, "login")
            if flow:
                return await web_shell(reason, f"run_flow login --domain {domain}")

            # Otherwise check for saved session
            state = memory.load_session_state(domain)
            if state:

                if not await _ensure_agent():
                    return _err("Browser konnte nicht starten – goto nicht möglich")

                _state_path = os.path.join(agent.state_dir, f"{domain}.json")
                with open(_state_path, 'w') as _f:
                    json.dump(state, _f)
                await agent.load_state(domain)
                return _ok(f"Loaded saved session for {domain}")

            return _err(f"login: no login flow or saved session for {domain}. "
                        f"Navigate manually and use 'save_session {domain}' to persist.")

        # ═══════════════════════════════════════════════════════════════
        # save_session [domain]
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "save_session":
            if not _started:
                return _err("no browser session")
            domain = rest[0] if rest else _domain(agent.current_url())
            state_path = await agent.save_state(domain)
            # Read the state file WebAgent wrote and persist to VFS
            try:
                with open(state_path) as _f:
                    state = json.load(_f)
                memory.save_session_state(domain, state)
            except Exception as e:
                return _err(f"save_session: failed to read state: {e}")
            return _ok(f"Session state saved for {domain}")

        # ═══════════════════════════════════════════════════════════════
        # load_session <domain>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "load_session":
            if not rest:
                return _err("load_session: missing domain")
            domain = rest[0]
            state = memory.load_session_state(domain)
            if not state:
                return _err(f"No saved session for {domain}")

            if not await _ensure_agent():
                return _err("Browser konnte nicht starten – goto nicht möglich")

            _state_path = os.path.join(agent.state_dir, f"{domain}.json")
            with open(_state_path, 'w') as _f:
                json.dump(state, _f)
            await agent.load_state(domain)
            return _ok(f"Session loaded for {domain}")

        # ═══════════════════════════════════════════════════════════════
        # close_browser
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "close_browser":
            if not _started:
                return _ok("Browser already closed")
            # Save session state for current domain before closing
            url = agent.current_url()
            if url:
                domain = _domain(url)
                state_path = await agent.save_state(domain)
                try:
                    with open(state_path) as _f:
                        state = json.load(_f)
                    memory.save_session_state(domain, state)
                except Exception:
                    pass
            await agent.stop()
            _started = False
            return _ok("Browser closed. Session state saved.")

        # ═══════════════════════════════════════════════════════════════
        # learn_cookie <selector>
        # ═══════════════════════════════════════════════════════════════
        elif cmd == "learn_cookie":
            if not rest:
                return _err("learn_cookie: missing selector")
            if not _started:
                return _err("no browser session")
            selector = rest[0]
            domain = _domain(agent.current_url())
            template = memory.get_or_create_template(domain)
            template.learn_selector("cookie", "", selector)
            memory.save_template(template)
            return _ok(f"Cookie selector learned for {domain}: {selector}")

        # ═══════════════════════════════════════════════════════════════
        # Unknown
        # ═══════════════════════════════════════════════════════════════
        else:
            return _err(
                f"web_shell: {cmd}: command not found\n"
                "Commands: goto back refresh wait click type fill select scroll "
                "extract screenshot ocr eval search status "
                "save_template list_templates save_flow run_flow list_flows "
                "login save_session load_session close_browser learn_cookie"
            )

    # Store agent ref on closure for cleanup access
    web_shell._web_agent = agent
    web_shell._memory_manager = memory

    return web_shell


# =============================================================================
# CLEANUP HELPER — called from session.close()
# =============================================================================

async def cleanup_web_shell(web_shell_fn):
    """Close browser session. Called from AgentSessionV2.close()."""
    agent = getattr(web_shell_fn, "_web_agent", None)
    if agent:
        memory = getattr(web_shell_fn, "_memory_manager", None)
        if memory and agent._page and agent.current_url():
            domain = _domain(agent.current_url())
            try:
                state_path = await agent.save_state(domain)
                with open(state_path) as _f:
                    state = json.load(_f)
                memory.save_session_state(domain, state)
            except Exception:
                pass
        await agent.stop()
