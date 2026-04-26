"""
ISIS TU Berlin Scraper Toolkit
==============================

Builds on WebAgentToolkit — no direct Playwright access.

Usage:
    isis = IsisToolkit(headless=True, out_of_process=True)
    await isis.start()
    courses = await isis.list_courses()
    await isis.stop()

    # Or as agent tools:
    tools = isis.get_tools()
    agent.add_tools(tools)

Credentials: ISIS_USERNAME, ISIS_PASSWORD (env vars)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ============================================================================
# CONSTANTS
# ============================================================================

ISIS_BASE = "https://isis.tu-berlin.de"
ISIS_LOGIN_URL = ISIS_BASE + "/auth/shibboleth/index.php"

USERNAME_SELECTORS = [
    "input#username",
    'input[name="j_username"]',
    'input[name="username"]',
    'input[type="text"]',
]
PASSWORD_SELECTORS = [
    "input#password",
    'input[name="j_password"]',
    'input[name="password"]',
    'input[type="password"]',
]
SUBMIT_SELECTORS = [
    'button[type="submit"]',
    'input[type="submit"]',
    'button[name="_eventId_proceed"]',
]

# ============================================================================
# JS SNIPPETS — reusable extraction logic
# ============================================================================

# nodeToMd: converts a DOM subtree to markdown (same as WebAgent's version)
_JS_NODE_TO_MD = r"""
function nodeToMd(node) {
    if (!node) return '';
    if (node.nodeType === Node.TEXT_NODE) return node.textContent.trim();
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    const tag = node.tagName.toLowerCase();
    const children = Array.from(node.childNodes).map(c => nodeToMd(c)).join('').trim();
    if (!children && !['img', 'br', 'hr'].includes(tag)) return '';
    switch (tag) {
        case 'h1': return '\n# ' + children + '\n';
        case 'h2': return '\n## ' + children + '\n';
        case 'h3': return '\n### ' + children + '\n';
        case 'h4': return '\n#### ' + children + '\n';
        case 'p': return '\n' + children + '\n';
        case 'br': return '\n';
        case 'hr': return '\n---\n';
        case 'strong': case 'b': return '**' + children + '**';
        case 'em': case 'i': return '*' + children + '*';
        case 'code': return '`' + children + '`';
        case 'pre': return '\n```\n' + children + '\n```\n';
        case 'a':
            const href = node.getAttribute('href') || '';
            return href ? '[' + children + '](' + href + ')' : children;
        case 'img':
            const src = node.getAttribute('src') || '';
            const alt = node.getAttribute('alt') || 'image';
            return src ? '![' + alt + '](' + src + ')' : '';
        case 'ul': case 'ol':
            return '\n' + Array.from(node.children).map((li, i) => {
                const bullet = tag === 'ol' ? (i + 1) + '.' : '-';
                return bullet + ' ' + nodeToMd(li);
            }).join('\n') + '\n';
        case 'li': return children;
        default: return children;
    }
}
"""

_JS_CLEAN_CLONE = """
function cleanClone(root, removeNav) {
    const removeSels = ['script', 'style', 'noscript', 'iframe', 'svg'];
    if (removeNav) removeSels.push('nav','header','footer','aside',
        '.advertisement','.ads','.sidebar','.menu','.cookie-banner','.popup','.modal');
    const clone = root.cloneNode(true);
    removeSels.forEach(s => clone.querySelectorAll(s).forEach(el => el.remove()));
    return clone;
}
"""

def _js_extract_md(selector: str, remove_nav: bool = False) -> str:
    """Build JS that extracts markdown from a selector."""
    return f"""
    (() => {{
        {_JS_NODE_TO_MD}
        {_JS_CLEAN_CLONE}
        const el = document.querySelector({json.dumps(selector)});
        if (!el) return '';
        return nodeToMd(cleanClone(el, {str(remove_nav).lower()}));
    }})()
    """


# ============================================================================
# ISIS TOOLKIT
# ============================================================================

class IsisToolkit:
    """ISIS scraper built on WebAgentToolkit primitives."""

    def __init__(
        self,
        headless: bool = True,
        out_of_process: bool = False,
        state_name: str = "isis_session",
        verbose: bool = False,
    ):
        from toolboxv2.mods.isaa.extras.web_helper.tooklit import WebAgentToolkit
        self._tk = WebAgentToolkit(
            headless=headless,
            auto_start=False,
            keep_open=True,
            verbose=verbose,
            out_of_process=out_of_process,
        )
        self._state_name = state_name
        self._started = False

    # ── helpers: route through toolkit tools ──

    async def _goto(self, url: str, wait_until: str = "domcontentloaded") -> dict:
        tool = self._tk.get_tool("goto")
        return await tool.func(url=url, wait_until=wait_until)

    async def _click(self, selector: str) -> dict:
        tool = self._tk.get_tool("click")
        return await tool.func(selector=selector)

    async def _type(self, selector: str, text: str) -> dict:
        tool = self._tk.get_tool("type")
        return await tool.func(selector=selector, text=text)

    async def _js(self, script: str) -> Any:
        tool = self._tk.get_tool("execute_js")
        r = await tool.func(script=script)
        return r.get("result") if isinstance(r, dict) else r

    async def _url(self) -> str:
        tool = self._tk.get_tool("current_url")
        r = await tool.func()
        return r.get("url", "")

    async def _wait(self, selector: str = "", timeout: int = 8) -> dict:
        tool = self._tk.get_tool("wait")
        return await tool.func(selector=selector, timeout=timeout)

    async def _screenshot(self, name: str) -> dict:
        tool = self._tk.get_tool("screenshot")
        return await tool.func(name=name)

    async def _scroll(self, direction: str = "down", amount: int = 500) -> dict:
        tool = self._tk.get_tool("scroll")
        return await tool.func(direction=direction, amount=amount)

    async def _session_save(self, name: str = "") -> dict:
        tool = self._tk.get_tool("session_save")
        return await tool.func(name=name or self._state_name)

    async def _session_load(self, name: str = "") -> dict:
        tool = self._tk.get_tool("session_load")
        return await tool.func(name=name or self._state_name)

    async def _extract_md(self, selector: str, remove_nav: bool = False) -> str:
        """Extract markdown from a DOM element via selector."""
        md = await self._js(_js_extract_md(selector, remove_nav))
        return re.sub(r'\n{3,}', '\n\n', (md or '')).strip()

    # ================================================================
    # LIFECYCLE
    # ================================================================

    async def start(self, username: str = "", password: str = ""):
        """Start browser, load session, auto-relogin if expired."""
        await self._tk.start_browser()
        self._started = True

        # Try loading existing session
        load_result = await self._session_load()
        if load_result.get("success"):
            if await self._session_valid():
                return {"status": "ready", "login": "cached"}

        # Session missing or expired → fresh login
        ok = await self.login(username=username, password=password)
        if ok:
            await self._session_save()
            return {"status": "ready", "login": "fresh"}

        return {"status": "login_failed"}

    async def stop(self):
        """Stop browser."""
        if self._started:
            await self._tk.stop_browser()
            self._started = False
        return {"status": "stopped"}

    # ================================================================
    # LOGIN
    # ================================================================

    async def login(self, username: str = "", password: str = "") -> bool:
        """Shibboleth login with selector discovery + consent handling."""
        username = username or os.environ.get("ISIS_USERNAME", "")
        password = password or os.environ.get("ISIS_PASSWORD", "")
        if not username or not password:
            raise ValueError("ISIS_USERNAME and ISIS_PASSWORD required (env or args)")

        await self._goto(ISIS_LOGIN_URL, wait_until="networkidle")

        # Find visible selectors
        u_sel = await self._find_first_visible(USERNAME_SELECTORS)
        p_sel = await self._find_first_visible(PASSWORD_SELECTORS)
        s_sel = await self._find_first_visible(SUBMIT_SELECTORS)

        if not all([u_sel, p_sel, s_sel]):
            await self._screenshot("isis_login_debug")
            raise RuntimeError(
                f"Login form not found. Selectors: user={u_sel} pass={p_sel} submit={s_sel}"
            )

        await self._type(u_sel, username)
        await self._type(p_sel, password)
        await self._click(s_sel)

        # Wait for redirect back to ISIS
        try:
            await self._wait(timeout=15)
        except Exception:
            pass

        await self._handle_consent()
        return await self._session_valid()

    async def _find_first_visible(self, selectors: list[str]) -> Optional[str]:
        """Find first visible selector from a list."""
        for sel in selectors:
            visible = await self._js(f"""
                (() => {{
                    const el = document.querySelector({json.dumps(sel)});
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && style.visibility !== 'hidden'
                           && el.offsetParent !== null;
                }})()
            """)
            if visible:
                return sel
        return None

    async def _handle_consent(self):
        """Handle Shibboleth consent page if present."""
        url = await self._url()
        if "isis" not in url:
            has_submit = await self._js("""
                document.querySelectorAll('button[type="submit"]').length
            """)
            if has_submit and has_submit > 0:
                await self._click('button[type="submit"]')
                try:
                    await self._wait(timeout=5)
                except Exception:
                    pass

    async def _session_valid(self) -> bool:
        """Check if current session is authenticated."""
        await self._goto(ISIS_BASE + "/my/", wait_until="domcontentloaded")
        try:
            await self._wait(selector='a[href*="/course/view.php"]', timeout=6)
        except Exception:
            pass

        url = (await self._url()).lower()
        if "shibboleth" in url or "/login/" in url or "idp" in url:
            return False
        if "/my" not in url and "isis.tu-berlin.de" not in url:
            return False

        count = await self._js("""
            document.querySelectorAll('.usermenu, [data-userid], .user-picture').length
        """)
        return (count or 0) > 0

    # ================================================================
    # COURSE LISTING
    # ================================================================

    async def list_courses(self) -> list[dict]:
        """List all courses from dashboard."""
        await self._goto(ISIS_BASE + "/my/", wait_until="domcontentloaded")
        try:
            await self._wait(selector='a[href*="/course/view.php"]', timeout=8)
        except Exception:
            pass

        return await self._js("""
            (() => {
                const seen = new Set();
                return Array.from(document.querySelectorAll('a[href*="/course/view.php"]'))
                    .filter(a => a.href.includes('id='))
                    .map(a => ({
                        title: a.textContent.trim(),
                        url: a.href,
                        id: a.href.split('id=').pop().split('&')[0]
                    }))
                    .filter(c => c.title && !seen.has(c.id) && seen.add(c.id));
            })()
        """) or []

    # ================================================================
    # CHAT MESSAGES
    # ================================================================

    async def list_chat_messages(self) -> list[dict]:
        """List chat conversations from Moodle message drawer."""
        await self._goto(ISIS_BASE + "/message/index.php", wait_until="domcontentloaded")
        try:
            await self._wait(selector=".view-overview-body", timeout=8)
        except Exception:
            pass

        # Expand collapsed sections + extract in one JS call
        return await self._js("""
            (async () => {
                // expand sections
                const toggles = document.querySelectorAll(
                    '.view-overview-body button.overview-section-toggle[aria-expanded="false"]'
                );
                for (const t of Array.from(toggles).slice(0, 2)) {
                    try { t.click(); await new Promise(r => setTimeout(r, 500)); } catch {}
                }
                await new Promise(r => setTimeout(r, 800));

                const seen = new Set();
                const msgs = [];
                document.querySelectorAll('.view-overview-body [data-conversation-id]')
                    .forEach(c => {
                        const id = c.getAttribute('data-conversation-id');
                        if (!id || seen.has(id)) return;
                        seen.add(id);
                        const sec = c.closest('[data-region^="view-overview-"]');
                        const section = sec
                            ? (sec.getAttribute('data-region') || '').replace('view-overview-', '')
                            : 'unknown';
                        const nameEl = c.querySelector(
                            '[data-region="header-name-content"], .name, strong, h4'
                        );
                        const textEl = c.querySelector(
                            '[data-region="last-message"], .text-truncate, .small'
                        );
                        const unreadEl = c.querySelector('[data-region="unread-count"]');
                        msgs.push({
                            id,
                            from: (nameEl?.textContent || c.textContent.trim().slice(0, 60)).trim(),
                            text: (textEl?.textContent || '').trim(),
                            section,
                            unread: parseInt((unreadEl?.textContent || '0').trim(), 10) || 0,
                        });
                    });
                return msgs;
            })()
        """) or []

    # ================================================================
    # COURSE OVERVIEW
    # ================================================================

    async def get_course_overview(self, course_id: str) -> dict:
        """Course main page: title + sections with activity counts."""
        url = f"{ISIS_BASE}/course/view.php?id={course_id}"
        await self._goto(url, wait_until="domcontentloaded")
        try:
            await self._wait(selector=".course-content, h1", timeout=8)
        except Exception:
            pass

        return await self._js(f"""
            (() => {{
                const titleEl = document.querySelector('h1, .page-header-headings h2');
                const sections = [];
                document.querySelectorAll('.course-content .section.main').forEach(sec => {{
                    const nameEl = sec.querySelector('.sectionname, .content .sectionname');
                    sections.push({{
                        name: (nameEl?.textContent || 'Unbenannt').trim(),
                        activity_count: sec.querySelectorAll('.activity').length,
                    }});
                }});
                const partLink = document.querySelector('a[href*="user/index.php"]');
                return {{
                    id: {json.dumps(str(course_id))},
                    url: {json.dumps(url)},
                    title: (titleEl?.textContent || '').trim(),
                    sections,
                    participants_url: partLink?.href || null,
                }};
            }})()
        """)

    # ================================================================
    # COURSE ACTIVITIES (overview.php with expand)
    # ================================================================

    async def get_course_activities(self, course_id: str) -> list[dict]:
        """All activities from overview.php after expanding all collapsibles."""
        url = f"{ISIS_BASE}/course/overview.php?id={course_id}"
        await self._goto(url, wait_until="domcontentloaded")

        # Expand all collapsibles sequentially in JS
        await self._js("""
            (async () => {
                const colls = document.querySelectorAll('[id$="_overview_collapsible"]');
                for (const coll of colls) {
                    const toggle = coll.querySelector(
                        'a[data-bs-toggle="collapse"][aria-expanded="false"]'
                    );
                    if (!toggle) continue;
                    toggle.click();
                    // wait for content to load
                    await new Promise(r => setTimeout(r, 1500));
                }
            })()
        """)

        # Extract all activities
        return await self._js(f"""
            (() => {{
                {_JS_NODE_TO_MD}
                const activities = [];
                document.querySelectorAll('[id$="_overview_collapsible"]').forEach(coll => {{
                    const region = coll.querySelector('[data-region="loading-icon-container"]');
                    const modname = region?.getAttribute('data-modname') || 'unknown';
                    const h3 = coll.querySelector('h3');
                    const sectionLabel = h3
                        ? h3.textContent.replace(/\\n/g, ' ').trim() || modname
                        : modname;
                    coll.querySelectorAll('tr[data-mdl-overview-cmid]').forEach(row => {{
                        const link = row.querySelector('a.activityname');
                        if (!link) return;
                        const typeEl = row.querySelector('td[data-mdl-overview-item="type"]');
                        activities.push({{
                            name: link.textContent.trim(),
                            url: link.href,
                            type: modname,
                            cmid: row.getAttribute('data-mdl-overview-cmid'),
                            extra: (typeEl?.textContent || '').trim(),
                            section: sectionLabel,
                        }});
                    }});
                }});
                return activities;
            }})()
        """) or []

    # ================================================================
    # COURSE SECTIONS MARKDOWN
    # ================================================================

    async def scrape_course_sections_markdown(self, course_id: str) -> list[dict]:
        """Each section of view.php as markdown."""
        url = f"{ISIS_BASE}/course/view.php?id={course_id}"
        await self._goto(url, wait_until="domcontentloaded")
        try:
            await self._wait(selector=".course-content .section.main", timeout=8)
        except Exception:
            pass

        # Expand collapsed sections
        await self._js("""
            (() => {
                const sels = [
                    '.course-content a.section-toggler[aria-expanded="false"]',
                    '.course-content a[data-toggler="collapse"][aria-expanded="false"]',
                    '.course-content button[aria-expanded="false"][data-bs-toggle="collapse"]',
                ];
                sels.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        try { el.click(); } catch {}
                    });
                });
            })()
        """)
        try:
            await self._wait(timeout=2)
        except Exception:
            pass

        return await self._js(f"""
            (() => {{
                {_JS_NODE_TO_MD}
                {_JS_CLEAN_CLONE}
                const sections = [];
                document.querySelectorAll('.course-content .section.main').forEach((sec, idx) => {{
                    const nameEl = sec.querySelector(
                        '.sectionname, h3.sectionname, h4.sectionname, .section-title'
                    );
                    const contentEl = sec.querySelector('.content') || sec;
                    const secId = sec.id || 'section-' + idx;
                    sections.push({{
                        idx,
                        id: secId,
                        name: (nameEl?.textContent || 'Section ' + idx).trim(),
                        url: {json.dumps(url)} + '#' + secId,
                        markdown: nodeToMd(cleanClone(contentEl, false)),
                    }});
                }});
                return sections;
            }})()
        """) or []

    # ================================================================
    # SHALLOW SCRAPE (overview + activities, no detail visits)
    # ================================================================

    async def scrape_course_shallow(self, course_id: str) -> dict:
        """Overview + activity list without visiting individual URLs."""
        overview = await self.get_course_overview(course_id)
        activities = await self.get_course_activities(course_id)
        type_counts = {}
        for a in activities:
            type_counts[a["type"]] = type_counts.get(a["type"], 0) + 1
        return {
            **overview,
            "activities": activities,
            "activity_count": len(activities),
            "type_counts": type_counts,
        }

    # ================================================================
    # ACTIVITY SCRAPERS (type-dispatched)
    # ================================================================

    async def scrape_activity(self, activity: dict) -> dict:
        """Type-based dispatcher for individual activities."""
        url = activity.get("url", "")
        atype = activity.get("type", "unknown")

        scrapers = {
            "forum": self._scrape_forum,
            "resource": self._scrape_resource,
            "folder": self._scrape_folder,
            "assign": self._scrape_assign,
            "quiz": self._scrape_quiz,
            "choicegroup": self._scrape_choicegroup,
            "choice": self._scrape_choice,
            "feedback": self._scrape_feedback,
            "questionnaire": self._scrape_feedback,
        }

        scraper = scrapers.get(atype, self._scrape_generic)
        try:
            detail = await scraper(url)
            return {**activity, "scraped": True, **detail}
        except Exception as e:
            return {**activity, "scraped": False, "error": str(e)}

    async def _scrape_forum(self, url: str, max_threads: int = 0) -> dict:
        """Forum: thread list + posts per thread."""
        await self._goto(url, wait_until="domcontentloaded")

        meta = await self._js("""
            (() => {
                const intro = document.querySelector('.activity-description .no-overflow');
                return {
                    title: document.title,
                    description: (intro?.textContent || '').trim(),
                };
            })()
        """)

        thread_meta = await self._js("""
            Array.from(document.querySelectorAll('table.discussion-list tr.discussion'))
                .map(row => {
                    const link = row.querySelector('a[href*="discuss.php"]');
                    return link ? { url: link.href, title: link.textContent.trim() } : null;
                })
                .filter(Boolean)
        """) or []

        if max_threads > 0:
            thread_meta = thread_meta[:max_threads]

        threads = []
        for tm in thread_meta:
            try:
                await self._goto(tm["url"], wait_until="domcontentloaded")
                posts = await self._scrape_forum_posts()
                threads.append({**tm, "post_count": len(posts), "posts": posts})
            except Exception as e:
                threads.append({**tm, "error": str(e)})

        return {
            "type": "forum", "url": url,
            "title": meta.get("title", ""), "description": meta.get("description", ""),
            "thread_count": len(threads), "threads": threads,
        }

    async def _scrape_forum_posts(self) -> list[dict]:
        """All posts on current thread page."""
        return await self._js(f"""
            (() => {{
                {_JS_NODE_TO_MD}
                {_JS_CLEAN_CLONE}
                return Array.from(document.querySelectorAll('article[data-post-id]')).map(art => {{
                    const subj = art.querySelector('h3, [data-region-content="forum-post-core-subject"]');
                    const authorA = art.querySelector('header a[href*="user/view.php"]');
                    const t = art.querySelector('time[datetime]');
                    const body = art.querySelector('.post-content-container, [id^="post-content-"]');
                    return {{
                        post_id: art.getAttribute('data-post-id'),
                        subject: (subj?.textContent || '').trim(),
                        author: (authorA?.textContent || '').trim(),
                        author_url: authorA?.href || '',
                        datetime: t?.getAttribute('datetime') || '',
                        datetime_text: (t?.textContent || '').trim(),
                        body_markdown: body ? nodeToMd(cleanClone(body, false)) : '',
                    }};
                }});
            }})()
        """) or []

    async def _scrape_resource(self, url: str) -> dict:
        """Resource: file redirect or HTML page."""
        await self._goto(url, wait_until="domcontentloaded")
        final_url = await self._url()

        # Check if it's a direct file download (non-HTML)
        is_html = await self._js("document.contentType?.includes('text/html') ?? true")
        if not is_html:
            return {
                "type": "resource", "url": url, "final_url": final_url,
                "is_file": True, "download_url": final_url,
                "title": url.rsplit("/", 1)[-1],
            }

        title = await self._js("document.title")
        md = await self._extract_md(
            '#region-main .box.generalbox .no-overflow, '
            '#region-main .box.generalbox, #region-main'
        )
        return {
            "type": "resource", "url": url, "final_url": final_url,
            "is_file": False, "title": title, "markdown": md,
        }

    async def _scrape_folder(self, url: str) -> dict:
        """Folder: list of downloadable files."""
        await self._goto(url, wait_until="domcontentloaded")
        title = await self._js("document.title")

        files = await self._js("""
            (() => {
                const seen = new Set();
                return Array.from(
                    (document.querySelector('#region-main') || document.body)
                        .querySelectorAll('a[href*="/pluginfile.php/"]')
                ).map(a => ({
                    name: (a.textContent || a.href.split('/').pop() || '').trim(),
                    url: a.href,
                })).filter(f => !seen.has(f.url) && seen.add(f.url));
            })()
        """) or []

        return {
            "type": "folder", "url": url, "title": title,
            "file_count": len(files), "files": files,
        }

    async def _scrape_assign(self, url: str) -> dict:
        """Assignment: dates + submission status."""
        await self._goto(url, wait_until="domcontentloaded")
        title = await self._js("document.title")

        info = await self._js("""
            (() => {
                const dates = {};
                const dateRoot = document.querySelector('[data-region="activity-dates"]');
                if (dateRoot) dateRoot.querySelectorAll('div').forEach(d => {
                    const s = d.querySelector('strong');
                    if (s) dates[s.textContent.replace(':','').trim()] =
                        d.textContent.replace(s.textContent, '').trim();
                });
                const status = {};
                const tbl = document.querySelector('.submissionstatustable table');
                if (tbl) tbl.querySelectorAll('tr').forEach(tr => {
                    const th = tr.querySelector('th'), td = tr.querySelector('td');
                    if (th && td) status[th.textContent.trim()] = td.textContent.trim();
                });
                return { dates, status };
            })()
        """)

        description = await self._extract_md(".activity-description .no-overflow")

        return {
            "type": "assign", "url": url, "title": title,
            "dates": info.get("dates", {}),
            "submission_status": info.get("status", {}),
            "description": description,
        }

    async def _scrape_quiz(self, url: str) -> dict:
        """Quiz: dates + info + attempts."""
        await self._goto(url, wait_until="domcontentloaded")
        title = await self._js("document.title")

        info = await self._js("""
            (() => {
                const dates = {};
                const dateRoot = document.querySelector('[data-region="activity-dates"]');
                if (dateRoot) dateRoot.querySelectorAll('div').forEach(d => {
                    const s = d.querySelector('strong');
                    if (s) dates[s.textContent.replace(':','').trim()] =
                        d.textContent.replace(s.textContent, '').trim();
                });
                const quizInfo = document.querySelector('.quizinfo');
                return { dates, info: (quizInfo?.textContent || '').trim() };
            })()
        """)

        attempts_md = await self._extract_md(
            ".quizattempt table, .generaltable.quizattemptsummary"
        )
        description = await self._extract_md(".activity-description .no-overflow")

        return {
            "type": "quiz", "url": url, "title": title,
            "dates": info.get("dates", {}), "info": info.get("info", ""),
            "attempts_markdown": attempts_md, "description": description,
        }

    async def _scrape_choicegroup(self, url: str) -> dict:
        """Choicegroup: options with member counts."""
        await self._goto(url, wait_until="domcontentloaded")
        title = await self._js("document.title")

        data = await self._js("""
            (() => {
                const dates = {};
                const dateRoot = document.querySelector('[data-region="activity-dates"]');
                if (dateRoot) dateRoot.querySelectorAll('div').forEach(d => {
                    const s = d.querySelector('strong');
                    if (s) dates[s.textContent.replace(':','').trim()] =
                        d.textContent.replace(s.textContent, '').trim();
                });
                const options = [];
                document.querySelectorAll('table.choicegroups tr.option').forEach(tr => {
                    const tds = tr.querySelectorAll('td');
                    if (tds.length < 3) return;
                    const radio = tr.querySelector('input[name="answer"]');
                    const label = tr.querySelector('label');
                    options.push({
                        id: radio?.value || '',
                        name: (label?.textContent || '').trim(),
                        checked: !!radio?.checked,
                        members_count: (tds[2]?.textContent || '').trim(),
                        members: (tds[3]?.textContent || '').trim(),
                    });
                });
                const own = options.find(o => o.checked);
                return { dates, options, own_choice: own?.name || null };
            })()
        """)

        return {"type": "choicegroup", "url": url, "title": title, **data}

    async def _scrape_choice(self, url: str) -> dict:
        """Standard Moodle choice."""
        await self._goto(url, wait_until="domcontentloaded")
        title = await self._js("document.title")

        data = await self._js("""
            (() => {
                const dates = {};
                const dateRoot = document.querySelector('[data-region="activity-dates"]');
                if (dateRoot) dateRoot.querySelectorAll('div').forEach(d => {
                    const s = d.querySelector('strong');
                    if (s) dates[s.textContent.replace(':','').trim()] =
                        d.textContent.replace(s.textContent, '').trim();
                });
                const options = [];
                document.querySelectorAll('.choices .option, form .option').forEach(el => {
                    const input = el.querySelector('input[type="radio"], input[type="checkbox"]');
                    const label = el.querySelector('label');
                    options.push({
                        id: input?.value || '',
                        name: (label?.textContent || '').trim(),
                        checked: !!input?.checked,
                    });
                });
                return { dates, options };
            })()
        """)

        return {"type": "choice", "url": url, "title": title, **data}

    async def _scrape_feedback(self, url: str) -> dict:
        """Feedback/Questionnaire."""
        await self._goto(url, wait_until="domcontentloaded")
        title = await self._js("document.title")

        dates = await self._js("""
            (() => {
                const dates = {};
                const dateRoot = document.querySelector('[data-region="activity-dates"]');
                if (dateRoot) dateRoot.querySelectorAll('div').forEach(d => {
                    const s = d.querySelector('strong');
                    if (s) dates[s.textContent.replace(':','').trim()] =
                        d.textContent.replace(s.textContent, '').trim();
                });
                return dates;
            })()
        """)

        description = await self._extract_md(".activity-description .no-overflow")
        content = await self._extract_md(
            '#region-main .box.generalbox, #region-main [role="main"]'
        )

        return {
            "type": "feedback", "url": url, "title": title,
            "dates": dates, "description": description,
            "content_markdown": content,
        }

    async def _scrape_generic(self, url: str) -> dict:
        """Fallback for unknown activity types."""
        await self._goto(url, wait_until="domcontentloaded")
        final_url = await self._url()
        title = await self._js("document.title")
        md = await self._extract_md("#region-main", remove_nav=True)
        return {
            "type": "generic", "url": url, "final_url": final_url,
            "title": title, "markdown": md,
        }

    # ================================================================
    # DEEP SCRAPE
    # ================================================================

    async def scrape_course_deep(
        self, course_id: str, skip_types: list[str] = None
    ) -> dict:
        """Shallow + scrape each activity. Slow (many page loads)."""
        skip = set(skip_types or ["videoservice"])
        shallow = await self.scrape_course_shallow(course_id)

        detailed = []
        for act in shallow["activities"]:
            if act.get("type") in skip:
                detailed.append({**act, "scraped": False, "skipped": True})
                continue
            detail = await self.scrape_activity(act)
            detailed.append(detail)

        shallow["activities"] = detailed
        return shallow

    # ================================================================
    # TOOL EXPORT (compatible with WebAgentToolkit)
    # ================================================================

    def get_tools(self) -> list[dict]:
        """Export as tool dicts compatible with agent.add_tools()."""
        tools = []
        for spec in _TOOL_SPECS:
            method = getattr(self, spec["method"])
            tools.append({
                "name": spec["name"],
                "description": spec["description"],
                "tool_func": method,
                "category": "isis",
                "args_schema": spec.get("args_schema", {}),
                "flags": spec.get("flags", {"read": True, "write": False, "no_thread": True}),
                "source": "isis-toolkit",
            })
        return tools


# ============================================================================
# TOOL SPECS
# ============================================================================

_TOOL_SPECS = [
    {
        "name": "isis_start",
        "method": "start",
        "description": (
            "Start ISIS browser session. Loads cached session, auto-relogins if expired. "
            "Credentials from ISIS_USERNAME/ISIS_PASSWORD env vars."
        ),
        "args_schema": {
            "username": {"type": "str", "default": ""},
            "password": {"type": "str", "default": ""},
        },
        "flags": {"read": False, "write": True, "no_thread": True},
    },
    {
        "name": "isis_stop",
        "method": "stop",
        "description": "Stop ISIS browser session.",
        "flags": {"read": False, "write": True, "no_thread": True},
    },
    {
        "name": "isis_list_courses",
        "method": "list_courses",
        "description": "List all enrolled courses from dashboard. Returns [{id, title, url}].",
    },
    {
        "name": "isis_list_chat",
        "method": "list_chat_messages",
        "description": "List Moodle chat conversations. Returns [{id, from, text, section, unread}].",
    },
    {
        "name": "isis_course_overview",
        "method": "get_course_overview",
        "description": "Course main page: title + sections with activity counts.",
        "args_schema": {"course_id": {"type": "str", "required": True}},
    },
    {
        "name": "isis_course_activities",
        "method": "get_course_activities",
        "description": (
            "All activities from course overview.php (expanded). "
            "Returns [{name, url, type, cmid, section}]."
        ),
        "args_schema": {"course_id": {"type": "str", "required": True}},
    },
    {
        "name": "isis_course_sections_md",
        "method": "scrape_course_sections_markdown",
        "description": "Each section of course view.php as markdown.",
        "args_schema": {"course_id": {"type": "str", "required": True}},
    },
    {
        "name": "isis_course_shallow",
        "method": "scrape_course_shallow",
        "description": "Overview + activity list without visiting activity URLs.",
        "args_schema": {"course_id": {"type": "str", "required": True}},
    },
    {
        "name": "isis_course_deep",
        "method": "scrape_course_deep",
        "description": (
            "Shallow + detail-scrape per activity. SLOW. "
            "Supported types: forum, resource, folder, assign, quiz, choicegroup, choice, feedback."
        ),
        "args_schema": {
            "course_id": {"type": "str", "required": True},
            "skip_types": {"type": "list", "default": ["videoservice"]},
        },
    },
    {
        "name": "isis_scrape_activity",
        "method": "scrape_activity",
        "description": "Scrape single activity (type-dispatched). Input: {url, type}.",
        "args_schema": {"activity": {"type": "dict", "required": True}},
    },
]
