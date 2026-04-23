"""ISIS TU Berlin Scraper Toolkit - Minimal Login & Session
Credentials via Env-Vars: ISIS_USERNAME, ISIS_PASSWORD
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from toolboxv2 import get_logger

try:
    from playwright.async_api import async_playwright
except ImportError:
    os.system("pip install playwright && playwright install chromium")
    from playwright.async_api import async_playwright

import re

logger = get_logger()

HEADLESS = False
ISIS_BASE = "https://isis.tu-berlin.de"
ISIS_LOGIN_URL = ISIS_BASE + "/auth/shibboleth/index.php"
STATE_DIR = Path(__file__).parent / "isis_states"
STATE_DIR.mkdir(exist_ok=True)


USERNAME_SELECTORS = [
    "input#username",
    "input[name=\"j_username\"]",
    "input[name=\"username\"]",
    "input[type=\"text\"]",
]
PASSWORD_SELECTORS = [
    "input#password",
    "input[name=\"j_password\"]",
    "input[name=\"password\"]",
    "input[type=\"password\"]",
]
SUBMIT_SELECTORS = [
    "button[type=\"submit\"]",
    "input[type=\"submit\"]",
    "button[name=\"_eventId_proceed\"]",
]


class IsisSession:

    def __init__(self, headless: bool = HEADLESS):
        self.headless = headless
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    async def start(self):
        logger.info("Starting browser headless=%s", self.headless)
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        logger.info("Browser ready")

    async def _find_first_visible(self, selectors):
        for sel in selectors:
            try:
                loc = self._page.locator(sel)
                if await loc.count() > 0 and await loc.first.is_visible():
                    return sel
            except Exception:
                continue
        return None

    async def login(self, username=None, password=None):
        username = username or os.environ.get("ISIS_USERNAME")
        password = password or os.environ.get("ISIS_PASSWORD")
        if not username or not password:
            raise ValueError("ISIS_USERNAME und ISIS_PASSWORD fehlen")

        logger.info("Navigiere zu ISIS Login: %s", ISIS_LOGIN_URL)
        await self._page.goto(ISIS_LOGIN_URL, wait_until="networkidle")
        logger.info("URL nach Redirect: %s", self._page.url)
        await self._page.screenshot(path=str(STATE_DIR / "01_after_redirect.png"))

        u_sel = await self._find_first_visible(USERNAME_SELECTORS)
        p_sel = await self._find_first_visible(PASSWORD_SELECTORS)
        s_sel = await self._find_first_visible(SUBMIT_SELECTORS)

        logger.info("Selectors: user=%s pass=%s submit=%s", u_sel, p_sel, s_sel)

        if not all([u_sel, p_sel, s_sel]):
            html_path = STATE_DIR / "02_login_page_debug.html"
            html_path.write_text(await self._page.content())
            raise RuntimeError("Login-Formular nicht erkannt. Debug: " + str(html_path))

        await self._page.fill(u_sel, username)
        await self._page.fill(p_sel, password)
        await self._page.click(s_sel)
        logger.info("Credentials submitted, warte auf Redirect...")

        try:
            await self._page.wait_for_url(ISIS_BASE + "/**", timeout=300)
        except Exception:
            await asyncio.sleep(5)

        await self._page.screenshot(path=str(STATE_DIR / "03_after_login.png"))
        logger.info("URL nach Login: %s", self._page.url)

        await self._handle_consent_page()
        return await self.is_authenticated()

    async def _handle_consent_page(self):
        try:
            if "isis" not in self._page.url:
                btn = self._page.locator("button[type=\"submit\"]")
                if await btn.count() > 0:
                    await btn.first.click()
                    await asyncio.sleep(3)
                    logger.info("Consent-Seite uebersprungen")
        except Exception as e:
            logger.debug("Keine Consent-Seite: %s", e)

    async def is_authenticated(self):
        if not self._page:
            return False
        on_isis = "isis.tu-berlin.de" in self._page.url
        on_idp = "shibboleth" in self._page.url.lower() or "idp" in self._page.url.lower()
        if on_isis and not on_idp:
            try:
                cnt = await self._page.locator(".usermenu, [data-userid], .user-picture").count()
                if cnt > 0:
                    logger.info("Auth bestaetigt: Moodle User-Menu")
                    return True
            except Exception:
                pass
            if "/my/" in self._page.url or "/dashboard" in self._page.url:
                logger.info("Auth bestaetigt: Dashboard URL")
                return True
        logger.warning("Auth fehlgeschlagen. URL: %s", self._page.url)
        return False

    async def _session_valid(self):
        """Prueft ob aktueller state eine valide ISIS-Session hat.
        Geht zu /my/ und checkt auf Shibboleth-Redirect.
        """
        await self._page.goto(ISIS_BASE + '/my/', wait_until='domcontentloaded')
        try:
            await self._page.wait_for_selector(
                'a[href*="/course/view.php"]', timeout=600, state='attached'
            )
        except Exception:
            logger.debug('list_courses: course links attached-wait timeout, continuing')
            return False
        url = self._page.url.lower()
        if 'shibboleth' in url or '/login/' in url or 'idp' in url:
            return False
        # Moodle redirect auf /login/index.php bei expired
        if '/my' not in url and 'isis.tu-berlin.de' not in url:
            return False
        try:
            cnt = await self._page.locator('.usermenu, [data-userid], .user-picture').count()
            return cnt > 0
        except Exception:
            return False

    async def save_state(self, name="isis_session"):
        path = STATE_DIR / (name + ".json")
        state = await self._context.storage_state()
        path.write_text(json.dumps(state, indent=2))
        logger.info("State gespeichert: %s", path)
        return str(path)

    async def load_state(self, name='isis_session', auto_relogin=True):
        path = STATE_DIR / (name + '.json')
        if not path.exists():
            if auto_relogin:
                logger.info('State fehlt, starte frischen Login: %s', name)
                return await self._login_and_save(name)
            return False
        if self._context:
            await self._context.close()
        self._context = await self._browser.new_context(
            storage_state=str(path), viewport={'width': 960, 'height': 540}
        )
        self._page = await self._context.new_page()
        logger.info('State geladen: %s', path)
        if not auto_relogin:
            return True
        if await self._session_valid():
            logger.info('Session valid.')
            return True
        logger.warning('Session expired, re-login...')
        return await self._login_and_save(name)

    async def _login_and_save(self, state_name):
        """Fresh login + save. Context muss bereits existieren (von start())."""
        if not self._context:
            self._context = await self._browser.new_context(
                viewport={'width': 960, 'height': 540},
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
            )
            self._page = await self._context.new_page()
        else:
            # Frischer Context ohne expired cookies
            await self._context.close()
            self._context = await self._browser.new_context(
                viewport={'width': 960, 'height': 540},
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
            )
            self._page = await self._context.new_page()
        ok = await self.login()
        if not ok:
            logger.error('Re-login fehlgeschlagen')
            return False
        await self.save_state(state_name)
        logger.info('Re-login + save erfolgreich: %s', state_name)
        return True

    async def close(self):
        if self._page:
            try:
                await self._page.goto('about:blank', timeout=3000)
            except Exception as e:
                logger.debug('about:blank failed: %s', e)
        if self._context:
            try:
                await asyncio.wait_for(self._context.close(), timeout=5)
            except Exception as e:
                logger.debug('context close timeout/err: %s', e)
        if self._browser:
            try:
                await asyncio.wait_for(self._browser.close(), timeout=5)
            except Exception as e:
                logger.debug('browser close timeout/err: %s', e)
        if self._pw:
            try:
                await self._pw.stop()
            except Exception as e:
                logger.debug('pw stop err: %s', e)
        logger.info('Session geschlossen')

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ============================================================
    # Scraping: Kurse, Chat, Course Overview, Activities
    # ============================================================

    async def list_courses(self):
        logger.info('list_courses: Lade Dashboard...')
        await self._page.goto(ISIS_BASE + '/my/', wait_until='domcontentloaded')
        try:
            await self._page.wait_for_selector('a[href*="/course/view.php"]', timeout=1000)
        except Exception:
            logger.warning('list_courses: course links selector timeout')
        await self._page.screenshot(path=str(STATE_DIR / '04_dashboard.png'))
        courses = []
        seen = set()
        links = await self._page.query_selector_all('a[href*="/course/view.php"]')
        for link in links:
            href = await link.get_attribute('href')
            if not href or 'id=' not in href:
                continue
            cid = href.split('id=')[-1].split('&')[0]
            if cid in seen:
                continue
            seen.add(cid)
            title = (await link.inner_text()).strip()
            if title:
                courses.append({'title': title, 'url': href, 'id': cid})
        logger.info('Kurse gefunden: %d', len(courses))
        return courses

    async def list_chat_messages(self):
        logger.info('list_chat_messages: Lade Nachrichten...')
        await self._page.goto(ISIS_BASE + '/message/index.php', wait_until='domcontentloaded')

        # drawer init is async, wait for body
        try:
            await self._page.wait_for_selector('.view-overview-body', timeout=8000)
        except Exception:
            logger.warning('view-overview-body not found')
        await self._page.wait_for_timeout(1200)
        # expand all collapsed sections (favourites / group-messages / messages)
        toggles = await self._page.query_selector_all(
            '.view-overview-body button.overview-section-toggle[aria-expanded="false"]'
        )
        for t in toggles[:2]:
            try:
                await t.click()

                await self._page.wait_for_timeout(500)  # per-section ajax

            except Exception as e:
                logger.debug('msg toggle click failed: %s', e)
        logger.info('chat: %d sections expanded', len(toggles))
        await self._page.screenshot(path=str(STATE_DIR / '05_messages.png'))
        # collect conversations across all sections (Favoriten, Gruppen, Privat)
        messages = []
        seen = set()
        convs = await self._page.query_selector_all('.view-overview-body [data-conversation-id]')
        for c in convs:
            try:
                conv_id = await c.get_attribute('data-conversation-id')
                if not conv_id or conv_id in seen:
                    continue
                # vor close() in list_chat_messages einbauen
                pending = await self._page.evaluate("""() => performance.getEntriesByType('resource')
                    .filter(r => r.duration === 0 && r.name.includes('service.php'))
                    .map(r => r.name)""")
                logger.info('pending service.php requests: %s', pending)
                seen.add(conv_id)
                parsed = await c.evaluate("""el => {
                    const sec = el.closest('[data-region^="view-overview-"]');
                    const section = sec
                        ? (sec.getAttribute('data-region') || '').replace('view-overview-', '')
                        : 'unknown';
                    const nameEl = el.querySelector(
                        '[data-region="header-name-content"], .name, strong, h4'
                    );
                    const textEl = el.querySelector(
                        '[data-region="last-message"], .text-truncate, .small'
                    );
                    const unreadEl = el.querySelector('[data-region="unread-count"]');
                    return {
                        section,
                        name: (nameEl?.textContent || '').trim(),
                        text: (textEl?.textContent || '').trim(),
                        unread: parseInt((unreadEl?.textContent || '0').trim(), 10) || 0,
                        raw: el.textContent.trim().replace(/\\s+/g, ' ').slice(0, 200),
                    };
                }""")
                messages.append({
                    'id': conv_id,
                    'from': parsed['name'] or parsed['raw'][:60],
                    'text': parsed['text'],
                    'section': parsed['section'],
                    'unread': parsed['unread'],
                })
            except Exception as e:
                logger.debug('conv parse failed: %s', e)
                continue
        logger.info('Chat-Nachrichten: %d (sections: %s)',
                    len(messages), {m['section'] for m in messages})
        return messages

    async def get_course_overview(self, course_id):
        url = ISIS_BASE + '/course/view.php?id=' + str(course_id)
        logger.info('get_course_overview: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        try:
            await self._page.wait_for_selector('.course-content, h1, .page-header-headings', timeout=8000)
        except Exception:
            logger.warning('get_course_overview: content selector timeout')
        await self._page.screenshot(path=str(STATE_DIR / '06_course_overview.png'))
        overview = {
            'id': str(course_id),
            'url': url,
            'title': '',
            'sections': [],
            'participants_count': None,
            'participants_url': None,
        }
        title_el = await self._page.query_selector('h1, .page-header-headings h2')
        if title_el:
            overview['title'] = (await title_el.inner_text()).strip()
        section_els = await self._page.query_selector_all('.course-content .section.main')
        for sec in section_els:
            try:
                name_el = await sec.query_selector('.sectionname, .content .sectionname')
                name = (await name_el.inner_text()).strip() if name_el else 'Unbenannt'
                act_els = await sec.query_selector_all('.activity')
                overview['sections'].append({'name': name, 'activity_count': len(act_els)})
            except Exception:
                continue
        part_link = await self._page.query_selector('a[href*="user/index.php"]')
        if part_link:
            overview['participants_url'] = await part_link.get_attribute('href')
        logger.info('Course overview: %s (%d sections)', overview['title'], len(overview['sections']))
        return overview

    async def get_course_activities(self, course_id):
        url = ISIS_BASE + '/course/overview.php?id=' + str(course_id)
        logger.info('get_course_activities: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        pending = await self._page.evaluate("""() => performance.getEntriesByType('resource')
                                            .filter(r => r.duration === 0 && r.name.includes('service.php'))
                                            .map(r => r.name)""")
        logger.info('pending service.php requests: %s', pending)
        await self._expand_all_sections(timeout_per=1500)
        await self._page.screenshot(path=str(STATE_DIR / '07_overview_expanded.png'))
        activities = []
        pending = await self._page.evaluate("""() => performance.getEntriesByType('resource')
                                            .filter(r => r.duration === 0 && r.name.includes('service.php'))
                                            .map(r => r.name)""")
        logger.info('pending service.php requests: %s', pending)
        collapsibles = await self._page.query_selector_all('[id$="_overview_collapsible"]')
        for coll in collapsibles:
            region = await coll.query_selector('[data-region="loading-icon-container"]')
            modname = (await region.get_attribute('data-modname')) if region else 'unknown'
            heading_el = await coll.query_selector('h3')
            if heading_el:
                raw = (await heading_el.inner_text()).strip()
                # h3 enthaelt Icon + Label, icon hat oft ein newline
                section_label = raw.replace('\n', ' ').strip() or modname
            else:
                section_label = modname
            rows = await coll.query_selector_all('tr[data-mdl-overview-cmid]')
            for row in rows:
                try:
                    cmid = await row.get_attribute('data-mdl-overview-cmid')
                    link = await row.query_selector('a.activityname')
                    if not link:
                        continue
                    href = (await link.get_attribute('href')) or ''
                    name = (await link.inner_text()).strip()
                    extra_el = await row.query_selector('td[data-mdl-overview-item="type"]')
                    extra = (await extra_el.inner_text()).strip() if extra_el else ''
                    activities.append({
                        'name': name, 'url': href, 'type': modname,
                        'cmid': cmid, 'extra': extra, 'section': section_label,
                    })
                except Exception as e:
                    logger.debug('row parse failed: %s', e)
        logger.info('Activities gefunden: %d (types: %s)',
                    len(activities), {a['type'] for a in activities})
        return activities

    async def _expand_all_sections(self, timeout_per=6000):
        """overview.php: click + sofort auf content warten, pro collapsible sequentiell."""
        collapsibles = await self._page.query_selector_all('[id$="_overview_collapsible"]')
        clicked = 0
        for coll in collapsibles:
            coll_id = await coll.get_attribute('id') or ''
            if not coll_id:
                continue
            toggle = await coll.query_selector(
                'a[data-bs-toggle="collapse"][aria-expanded="false"]'
            )
            if not toggle:
                # schon expanded -> skip
                continue
            try:
                await toggle.click()
            except Exception as e:
                logger.debug('expand click failed %s: %s', coll_id, e)
                continue
            # Warte JETZT auf content BEVOR naechster click (sonst race)
            content_sel = (
                '#' + coll_id + ' table.course-overview-table tr[data-mdl-overview-cmid], '
                                '#' + coll_id + ' [data-region="loading-icon-container"][data-loaded="true"]'
            )
            try:
                if clicked >= 1:
                    await self._page.wait_for_selector(content_sel, timeout=timeout_per)
                else:
                    clicked += 1
                    raise Exception
                clicked += 1
            except Exception:
                logger.warning('expand: %s content timeout -> retry click', coll_id)
                # Re-try: manchmal wird collapsible durch click auf naechsten wieder zu
                try:
                    still_toggle = await coll.query_selector(
                        'a[data-bs-toggle="collapse"][aria-expanded="false"]'
                    )
                    if still_toggle:
                        await still_toggle.click()
                        await self._page.wait_for_selector(content_sel, timeout=timeout_per)
                        clicked += 1
                except Exception:
                    logger.warning('expand: %s retry also failed', coll_id)
        logger.info('expand_all_sections: %d collapsibles expanded', clicked)
        return clicked

    async def _expand_course_sections(self):
        """view.php: collapsed section-header expanden (defensiv, mehrere Selektor-Varianten)."""
        selectors = [
            '.course-content a.section-toggler[aria-expanded="false"]',
            '.course-content a[data-toggler="collapse"][aria-expanded="false"]',
            '.course-content .section-header a[aria-expanded="false"]',
            '.course-content button[aria-expanded="false"][data-bs-toggle="collapse"]',
        ]
        clicked = 0
        for sel in selectors:
            els = await self._page.query_selector_all(sel)
            for el in els:
                try:
                    await el.click()
                    clicked += 1
                except Exception as e:
                    logger.debug('course section click failed: %s', e)
        if clicked:
            await self._page.wait_for_timeout(700)
        logger.info('course sections expanded: %d', clicked)
        return clicked

    async def _extract_markdown_from_element(self, element_handle, remove_nav=False):
        """Deinen nodeToMd-Extraktor auf ein ElementHandle anwenden."""
        md = await self._page.evaluate(r"""
            ([root, removeNav]) => {
                const removeSelectors = ['script', 'style', 'noscript', 'iframe', 'svg'];
                if (removeNav) {
                    removeSelectors.push('nav', 'header', 'footer', 'aside',
                        '.advertisement', '.ads', '.sidebar', '.menu',
                        '.cookie-banner', '.popup', '.modal');
                }
                const clone = root.cloneNode(true);
                removeSelectors.forEach(sel => {
                    clone.querySelectorAll(sel).forEach(el => el.remove());
                });
                function nodeToMd(node) {
                    if (node.nodeType === Node.TEXT_NODE) return node.textContent.trim();
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
                return nodeToMd(clone);
            }
        """, [element_handle, bool(remove_nav)])
        md = re.sub(r'\n{3,}', '\n\n', md or '').strip()
        return md

    async def _extract_activity_dates(self):
        """Parsed [data-region=activity-dates] -> {label: value, ...}."""
        return await self._page.evaluate("""() => {
            const out = {};
            const root = document.querySelector('[data-region="activity-dates"]');
            if (!root) return out;
            root.querySelectorAll('div').forEach(d => {
                const strong = d.querySelector('strong');
                if (!strong) return;
                const key = strong.textContent.replace(':', '').trim();
                const val = d.textContent.replace(strong.textContent, '').trim();
                if (key) out[key] = val;
            });
            return out;
        }""")

    async def _extract_kv_table(self, table_selector):
        """th/td-Table -> dict. Fuer Abgabestatus etc."""
        return await self._page.evaluate("""(sel) => {
            const out = {};
            const t = document.querySelector(sel);
            if (!t) return out;
            t.querySelectorAll('tr').forEach(tr => {
                const th = tr.querySelector('th');
                const td = tr.querySelector('td');
                if (th && td) out[th.textContent.trim()] = td.textContent.trim();
            });
            return out;
        }""", table_selector)

    async def scrape_course_shallow(self, course_id):
        """Aggregat: overview + activity-liste. Keine activity-URLs besucht."""
        logger.info('scrape_course_shallow: id=%s', course_id)
        overview = await self.get_course_overview(course_id)
        activities = await self.get_course_activities(course_id)
        types = {}
        for a in activities:
            types[a['type']] = types.get(a['type'], 0) + 1
        return {
            'id': overview['id'],
            'url': overview['url'],
            'title': overview['title'],
            'sections': overview['sections'],
            'activities': activities,
            'activity_count': len(activities),
            'type_counts': types,
        }

    async def _scrape_forum(self, url, max_threads=None):
        """Forum -> thread-liste + pro thread alle posts rekursiv."""
        logger.info('_scrape_forum: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(800)
        meta = await self._page.evaluate("""() => {
            const main = document.querySelector('#region-main');
            const intro = main?.querySelector('.activity-description .no-overflow');
            return {
                title: document.title,
                description: (intro?.textContent || '').trim(),
            };
        }""")
        thread_rows = await self._page.query_selector_all(
            'table.discussion-list tr.discussion'
        )
        thread_meta = []
        for row in thread_rows:
            link = await row.query_selector('a[href*="discuss.php"]')
            if not link:
                continue
            thread_meta.append({
                'url': await link.get_attribute('href'),
                'title': (await link.inner_text()).strip(),
            })
        if max_threads:
            thread_meta = thread_meta[:max_threads]
        threads = []
        for tm in thread_meta:
            try:
                await self._page.goto(tm['url'], wait_until='domcontentloaded')
                await self._page.wait_for_timeout(600)
                posts = await self._scrape_forum_posts()
                threads.append({
                    'url': tm['url'],
                    'title': tm['title'],
                    'post_count': len(posts),
                    'posts': posts,
                })
            except Exception as e:
                logger.warning('forum thread failed %s: %s', tm['url'], e)
                threads.append({'url': tm['url'], 'title': tm['title'], 'error': str(e)})
        return {
            'type': 'forum', 'url': url,
            'title': meta['title'], 'description': meta['description'],
            'thread_count': len(threads), 'threads': threads,
        }

    async def _scrape_forum_posts(self):
        """Alle article[data-post-id] der aktuellen Thread-Seite."""
        post_ids = await self._page.evaluate("""() => {
            return Array.from(document.querySelectorAll('article[data-post-id]'))
                .map(a => a.getAttribute('data-post-id'));
        }""")
        posts = []
        for pid in post_ids:
            art = await self._page.query_selector('article[data-post-id="' + pid + '"]')
            if not art:
                continue
            header = await art.evaluate("""a => {
                const subj = a.querySelector('h3, [data-region-content="forum-post-core-subject"]');
                const authorA = a.querySelector('header a[href*="user/view.php"]');
                const t = a.querySelector('time[datetime]');
                const replyLabel = a.querySelector('[aria-label*="Anzahl Antworten"]');
                const replyMatch = replyLabel?.textContent?.match(/\\d+/);
                return {
                    subject: (subj?.textContent || '').trim(),
                    author: (authorA?.textContent || '').trim(),
                    author_url: authorA?.href || '',
                    datetime: t?.getAttribute('datetime') || '',
                    datetime_text: (t?.textContent || '').trim(),
                    reply_count: replyMatch ? parseInt(replyMatch[0], 10) : 0,
                };
            }""")
            body_el = await art.query_selector('.post-content-container, [id^="post-content-"]')
            body_md = ''
            if body_el:
                body_md = await self._extract_markdown_from_element(body_el)
            posts.append({'post_id': pid, **header, 'body_markdown': body_md})
        return posts

    async def _scrape_resource(self, url):
        """resource: entweder direkter PDF-redirect oder HTML-seite."""
        logger.info('_scrape_resource: %s', url)
        try:
            resp = await self._page.goto(url, wait_until='domcontentloaded')
        except Exception as e:
            return {'type': 'resource', 'url': url, 'error': str(e)}
        final_url = self._page.url
        ct = resp.headers.get('content-type', '') if resp else ''
        if resp and 'text/html' not in ct:
            return {
                'type': 'resource', 'url': url, 'final_url': final_url,
                'content_type': ct, 'download_url': final_url,
                'is_file': True, 'title': url.rsplit('/', 1)[-1],
            }
        # HTML (page-type resource)
        title = await self._page.title()
        main = await self._page.query_selector(
            '#region-main .box.generalbox .no-overflow, #region-main .box.generalbox, #region-main'
        )
        md = await self._extract_markdown_from_element(main) if main else ''
        modified = await self._page.evaluate("""() => {
            const el = document.querySelector('.modified');
            return (el?.textContent || '').trim();
        }""")
        return {
            'type': 'resource', 'url': url, 'final_url': final_url,
            'content_type': ct, 'is_file': False,
            'title': title, 'markdown': md, 'modified': modified,
        }

    async def _scrape_folder(self, url):
        """folder: datei-liste mit resolved pluginfile-URLs."""
        logger.info('_scrape_folder: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(500)
        title = await self._page.title()
        files = await self._page.evaluate("""() => {
            const main = document.querySelector('#region-main') || document.body;
            const out = [];
            main.querySelectorAll('a[href*="/pluginfile.php/"]').forEach(a => {
                out.push({
                    name: (a.textContent || a.getAttribute('href').split('/').pop() || '').trim(),
                    url: a.href,
                });
            });
            return out;
        }""")
        # dedup by url
        seen, uniq = set(), []
        for f in files:
            if f['url'] in seen:
                continue
            seen.add(f['url'])
            uniq.append(f)
        return {
            'type': 'folder', 'url': url, 'title': title,
            'file_count': len(uniq), 'files': uniq,
        }

    async def _scrape_assign(self, url):
        """assign: dates + Abgabestatus-table."""
        logger.info('_scrape_assign: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(500)
        title = await self._page.title()
        dates = await self._extract_activity_dates()
        status = await self._extract_kv_table('.submissionstatustable table')
        desc_el = await self._page.query_selector('.activity-description .no-overflow')
        description = ''
        if desc_el:
            description = await self._extract_markdown_from_element(desc_el)
        return {
            'type': 'assign', 'url': url, 'title': title,
            'dates': dates, 'submission_status': status,
            'description': description,
        }

    async def _scrape_quiz(self, url):
        """quiz: dates + quizinfo + attempts."""
        logger.info('_scrape_quiz: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(500)
        title = await self._page.title()
        dates = await self._extract_activity_dates()
        info = await self._page.evaluate("""() => {
            const el = document.querySelector('.quizinfo');
            return (el?.textContent || '').trim();
        }""")
        attempts_el = await self._page.query_selector('.quizattempt table, .generaltable.quizattemptsummary')
        attempts_md = ''
        if attempts_el:
            attempts_md = await self._extract_markdown_from_element(attempts_el)
        desc_el = await self._page.query_selector('.activity-description .no-overflow')
        description = ''
        if desc_el:
            description = await self._extract_markdown_from_element(desc_el)
        return {
            'type': 'quiz', 'url': url, 'title': title,
            'dates': dates, 'info': info,
            'attempts_markdown': attempts_md, 'description': description,
        }

    async def _scrape_choicegroup(self, url):
        """choicegroup: options + member-counts + eigene auswahl."""
        logger.info('_scrape_choicegroup: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(500)
        title = await self._page.title()
        dates = await self._extract_activity_dates()
        options = await self._page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('table.choicegroups tr.option').forEach(tr => {
                const tds = tr.querySelectorAll('td');
                if (tds.length < 3) return;
                const radio = tr.querySelector('input[name="answer"]');
                const label = tr.querySelector('label');
                out.push({
                    id: radio?.value || '',
                    name: (label?.textContent || '').trim(),
                    checked: !!radio?.checked,
                    members_count: (tds[2]?.textContent || '').trim(),
                    members: (tds[3]?.textContent || '').trim(),
                });
            });
            return out;
        }""")
        own = next((o for o in options if o['checked']), None)
        return {
            'type': 'choicegroup', 'url': url, 'title': title,
            'dates': dates, 'options': options,
            'own_choice': own['name'] if own else None,
        }

    async def _scrape_choice(self, url):
        """Standard moodle choice (nicht choicegroup)."""
        logger.info('_scrape_choice: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(500)
        title = await self._page.title()
        dates = await self._extract_activity_dates()
        options = await self._page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('.choices .option, form .option').forEach(el => {
                const input = el.querySelector('input[type="radio"], input[type="checkbox"]');
                const label = el.querySelector('label');
                out.push({
                    id: input?.value || '',
                    name: (label?.textContent || '').trim(),
                    checked: !!input?.checked,
                });
            });
            return out;
        }""")
        return {
            'type': 'choice', 'url': url, 'title': title,
            'dates': dates, 'options': options,
        }

    async def _scrape_feedback(self, url):
        """feedback/questionnaire: dates + intro + status ob beantwortet."""
        logger.info('_scrape_feedback: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        await self._page.wait_for_timeout(500)
        title = await self._page.title()
        dates = await self._extract_activity_dates()
        desc_el = await self._page.query_selector('.activity-description .no-overflow')
        description = ''
        if desc_el:
            description = await self._extract_markdown_from_element(desc_el)
        # feedback-/questionnaire-main content
        main_el = await self._page.query_selector(
            '#region-main .box.generalbox, #region-main [role="main"]'
        )
        main_md = ''
        if main_el:
            main_md = await self._extract_markdown_from_element(main_el)
        return {
            'url': url, 'title': title,
            'dates': dates, 'description': description,
            'content_markdown': main_md,
        }

    async def scrape_activity(self, activity):
        """Type-basierter dispatcher. activity = dict aus get_course_activities."""
        url = activity['url']
        t = activity.get('type', 'unknown')
        try:
            if t == 'forum':
                detail = await self._scrape_forum(url)
            elif t == 'resource':
                detail = await self._scrape_resource(url)
            elif t == 'folder':
                detail = await self._scrape_folder(url)
            elif t == 'assign':
                detail = await self._scrape_assign(url)
            elif t == 'quiz':
                detail = await self._scrape_quiz(url)
            elif t == 'choicegroup':
                detail = await self._scrape_choicegroup(url)
            elif t == 'choice':
                detail = await self._scrape_choice(url)
            elif t in ('feedback', 'questionnaire'):
                detail = await self._scrape_feedback(url)
            else:
                detail = await self._scrape_generic(url, t)
        except Exception as e:
            logger.warning('scrape_activity %s failed: %s', url, e)
            return {**activity, 'scraped': False, 'error': str(e)}
        return {**activity, 'scraped': True, **detail}

    async def _scrape_generic(self, url, type_hint):
        """Fallback fuer unbekannte types. Generisches MD vom region-main."""
        try:
            resp = await self._page.goto(url, wait_until='domcontentloaded')
        except Exception as e:
            return {'type': type_hint, 'url': url, 'error': str(e)}
        final_url = self._page.url
        ct = resp.headers.get('content-type', '') if resp else ''
        if resp and 'text/html' not in ct:
            return {
                'type': type_hint, 'url': url, 'final_url': final_url,
                'content_type': ct, 'download_url': final_url, 'is_file': True,
            }
        title = await self._page.title()
        main = await self._page.query_selector('#region-main') \
               or await self._page.query_selector('main')
        md = await self._extract_markdown_from_element(main, remove_nav=True) if main else ''
        return {
            'type': type_hint, 'url': url, 'final_url': final_url,
            'content_type': ct, 'title': title, 'markdown': md,
        }

    async def scrape_course_deep(self, course_id, skip_types=None):
        """shallow + pro activity scrape_activity. skip_types: liste von types.
        'videoservice' defaultmaessig geskippt (liefert nichts sinnvolles).
        """
        skip_types = set(skip_types or ['videoservice'])
        shallow = await self.scrape_course_shallow(course_id)
        detailed = []
        for act in shallow['activities']:
            if act.get('type') in skip_types:
                detailed.append({**act, 'scraped': False, 'skipped': True})
                continue
            detail = await self.scrape_activity(act)
            detailed.append(detail)
        shallow['activities'] = detailed
        return shallow

    async def scrape_course_sections_markdown(self, course_id):
        """Haupt-Kursseite (view.php) pro Section als Markdown.
        Returns: [{idx, id, name, url, markdown}]
        """
        url = ISIS_BASE + '/course/view.php?id=' + str(course_id)
        logger.info('scrape_course_sections_markdown: %s', url)
        await self._page.goto(url, wait_until='domcontentloaded')
        try:
            await self._page.wait_for_selector('.course-content .section.main', timeout=8000)
        except Exception:
            logger.warning('course-content sections not found')
        await self._expand_course_sections()
        await self._page.screenshot(path=str(STATE_DIR / '08_view_expanded.png'))
        sections = []
        section_els = await self._page.query_selector_all('.course-content .section.main')
        for idx, sec in enumerate(section_els):
            try:
                name_el = await sec.query_selector(
                    '.sectionname, h3.sectionname, h4.sectionname, .section-title'
                )
                name = (await name_el.inner_text()).strip() if name_el else 'Section ' + str(idx)
                content_el = await sec.query_selector('.content') or sec
                markdown = await self._extract_markdown_from_element(content_el)
                sec_id = await sec.get_attribute('id') or 'section-' + str(idx)
                sections.append({
                    'idx': idx,
                    'id': sec_id,
                    'name': name,
                    'url': url + '#' + sec_id,
                    'markdown': markdown,
                })
            except Exception as e:
                logger.debug('section %d parse failed: %s', idx, e)
                continue
        logger.info('Sections scraped: %d', len(sections))
        return sections


# ============================================================
# Agent Toolkit: Singleton-Session + freie tool-Funktionen
# ============================================================

_session = None
_session_lock = asyncio.Lock()

async def _get_session(headless=None, state_name='isis_session'):
    global _session
    async with _session_lock:
        if _session is None:
            _session = IsisSession(headless=HEADLESS if headless is None else headless)
            await _session.start()
            await _session.load_state(state_name)
        return _session

async def tool_isis_start(headless=True, state_name='isis_session'):
    """Startet (oder returnt) Session. headless default True fuer Agent-Usage."""
    s = await _get_session(headless=headless, state_name=state_name)
    valid = await s._session_valid()
    return {'started': True, 'headless': s.headless, 'session_valid': valid}

async def tool_isis_stop():
    """Schliesst Session."""
    global _session
    async with _session_lock:
        if _session is None:
            return {'stopped': False, 'reason': 'not running'}
        await _session.close()
        _session = None
    return {'stopped': True}

async def tool_list_courses():
    s = await _get_session()
    return await s.list_courses()

async def tool_list_chat_messages():
    s = await _get_session()
    return await s.list_chat_messages()

async def tool_get_course_overview(course_id):
    s = await _get_session()
    return await s.get_course_overview(course_id)

async def tool_get_course_activities(course_id):
    s = await _get_session()
    return await s.get_course_activities(course_id)

async def tool_scrape_course_sections_markdown(course_id):
    s = await _get_session()
    return await s.scrape_course_sections_markdown(course_id)

async def tool_scrape_course_shallow(course_id):
    s = await _get_session()
    return await s.scrape_course_shallow(course_id)

async def tool_scrape_course_deep(course_id, skip_types=None):
    s = await _get_session()
    return await s.scrape_course_deep(course_id, skip_types=skip_types)

async def tool_scrape_activity(activity):
    """activity: dict mit mind. 'url' und 'type'."""
    s = await _get_session()
    return await s.scrape_activity(activity)


TOOLKIT_SPECS = [
    {
        'tool_func': tool_isis_start,
        'name': 'tool_isis_start',
        'description': (
            'Startet die ISIS-Browser-Session (Singleton). Laedt gespeicherten '
            'State und prueft session_valid; bei expired erfolgt Auto-Relogin '
            '(ISIS_USERNAME/ISIS_PASSWORD aus Env). '
            'kwargs: headless=True (bool), state_name="isis_session". '
            'Returns: {started, headless, session_valid}.'
        ),
        'flags': {"no_thread": True},
        'category': ['isis', 'session'],
        'live_test_inputs': [{'headless': True}],
        'result_contract': {
            'expected_type': dict,
            'semantic_check_hint': "Dict mit 'started' True/False, 'headless' bool, 'session_valid' bool.",
        },
    },
    {
        'tool_func': tool_isis_stop,
        'name': 'tool_isis_stop',
        'description': 'Schliesst die ISIS-Session. Keine kwargs. Returns: {stopped, reason?}.',
        'category': ['isis', 'session'],
        'live_test_inputs': [{}],
        'flags': {"no_thread": True},
        'result_contract': {
            'expected_type': dict,
            'semantic_check_hint': "Dict mit 'stopped' bool.",
        },
    },
    {
        'tool_func': tool_list_courses,
        'name': 'tool_list_courses',
        'description': (
            'Listet alle Kurse des Users vom Dashboard. Keine kwargs. '
            'Returns: list[{id, title, url}].'
        ),
        'category': ['isis', 'courses'],
        'live_test_inputs': [{}],
        'result_contract': {
            'expected_type': list,
            'semantic_check_hint': "Liste von dicts mit 'id', 'title', 'url'.",
        },
    },
    {
        'tool_func': tool_list_chat_messages,
        'name': 'tool_list_chat_messages',
        'description': (
            'Listet Chat-Konversationen aus allen Moodle-message-drawer-sections '
            '(favourites, group-messages, messages). '
            'Returns: list[{id, from, text, section, unread}].'
        ),
        'category': ['isis', 'messages'],
        'live_test_inputs': [{}],
        'result_contract': {
            'expected_type': list,
            'semantic_check_hint': "Liste von Message-dicts mit Feldern id, from, text, section, unread.",
        },
    },
    {
        'tool_func': tool_get_course_overview,
        'name': 'tool_get_course_overview',
        'description': (
            'Haupt-Kursseite (view.php): Titel + section-Namen + activity_count je section. '
            'kwargs: course_id (str/int). Returns: {id, url, title, sections, ...}.'
        ),
        'category': ['isis', 'courses'],
        'live_test_inputs': [{'course_id': '45731'}],
        'result_contract': {
            'expected_type': dict,
            'semantic_check_hint': "Dict mit 'id', 'title', 'sections' (liste).",
        },
    },
    {
        'tool_func': tool_get_course_activities,
        'name': 'tool_get_course_activities',
        'description': (
            'Liefert alle Activities eines Kurses von overview.php (nach Expand '
            'aller collapsibles). kwargs: course_id. '
            'Returns: list[{name, url, type, cmid, extra, section}].'
        ),
        'category': ['isis', 'activities'],
        'live_test_inputs': [{'course_id': '45731'}],
        'result_contract': {
            'expected_type': list,
            'semantic_check_hint': "Liste von Activity-dicts. type in {resource, forum, videoservice, assign, quiz, folder, choicegroup, feedback, ...}.",
        },
    },
    {
        'tool_func': tool_scrape_course_sections_markdown,
        'name': 'tool_scrape_course_sections_markdown',
        'description': (
            'Pro section der view.php-Hauptseite: Markdown des .content-Blocks. '
            'kwargs: course_id. Returns: list[{idx, id, name, url, markdown}].'
        ),
        'category': ['isis', 'courses', 'markdown'],
        'live_test_inputs': [{'course_id': '45731'}],
        'result_contract': {
            'expected_type': list,
            'semantic_check_hint': "Liste von Section-dicts mit 'markdown' (str).",
        },
    },
    {
        'tool_func': tool_scrape_course_shallow,
        'name': 'tool_scrape_course_shallow',
        'description': (
            'Aggregat: overview + activity-liste (ohne Besuch der activity-URLs). '
            'kwargs: course_id. '
            'Returns: {id, title, sections, activities, activity_count, type_counts}.'
        ),
        'category': ['isis', 'courses'],
        'live_test_inputs': [{'course_id': '45731'}],
        'result_contract': {
            'expected_type': dict,
            'semantic_check_hint': "Dict mit 'activities' (liste) und 'type_counts' (dict).",
        },
    },
    {
        'tool_func': tool_scrape_course_deep,
        'name': 'tool_scrape_course_deep',
        'description': (
            'Shallow + pro Activity wird scrape_activity aufgerufen. '
            'kwargs: course_id, skip_types=["videoservice"] default. '
            'SLOW (viele goto-calls). Returns: shallow-dict mit detailiertem activities[].'
        ),
        'category': ['isis', 'courses', 'deep'],
        'live_test_inputs': [{'course_id': '45731', 'skip_types': ['videoservice', 'forum']}],
        'result_contract': {
            'expected_type': dict,
            'semantic_check_hint': "Dict wie shallow; jede activity hat 'scraped' bool und type-spezifische Felder.",
        },
    },
    {
        'tool_func': tool_scrape_activity,
        'name': 'tool_scrape_activity',
        'description': (
            'Type-spezifischer Scraper fuer EINE Activity. '
            'kwargs: activity (dict mit mind. url und type). '
            'Supported types: forum (rekursiv threads+posts), resource, folder, '
            'assign (dates+submission-status), quiz (dates+info+attempts), '
            'choicegroup, choice, feedback, questionnaire. Unbekannte types: generic MD.'
        ),
        'category': ['isis', 'activities', 'scrape'],
        'live_test_inputs': [
            {'activity': {'url': 'https://isis.tu-berlin.de/mod/forum/view.php?id=2153498', 'type': 'forum'}},
        ],
        'result_contract': {
            'expected_type': dict,
            'semantic_check_hint': "Dict mit 'scraped' bool und type-spezifischen Feldern (z.B. 'threads' bei forum, 'dates'+'submission_status' bei assign).",
        },
    },
]


# ==========================================
# 3. HELPER FUNKTIONEN FÜR CLI ARGUMENTE
# ==========================================

def parse_bool(value: str) -> bool:
    """Helper, um CLI Strings sicher in Booleans umzuwandeln."""
    if isinstance(value, bool): return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean erwartet (True/False).')


def parse_json(value: str):
    """Helper, um Listen und Dictionaries aus CLI Strings zu parsen."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Ungültiges JSON-Format: {e}")


def build_parser() -> argparse.ArgumentParser:
    """Baut den ArgumentParser dynamisch aus den TOOLKIT_SPECS auf."""
    parser = argparse.ArgumentParser(
        description="ISIS Toolkit CLI - Pures I/O Tool für asynchrone ISIS Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="tool", required=True, help="Verfügbare Tools")

    for spec in TOOLKIT_SPECS:
        tool_name = spec['name']
        tool_desc = spec.get('description', '')
        tool_func = spec['tool_func']

        # Erstelle Subparser für das spezifische Tool
        p = subparsers.add_parser(tool_name, help=tool_desc)
        p.set_defaults(func=tool_func)

        # Analysiere live_test_inputs um Felder automatisch als Argumente bereitzustellen
        test_inputs = spec.get('live_test_inputs', [{}])[0]

        # Generiere Argumente für bekannte Felder
        for key, default_val in test_inputs.items():
            arg_name = f"--{key.replace('_', '-')}"

            # Typ-Bestimmung für den Helper
            if isinstance(default_val, bool):
                p.add_argument(arg_name, type=parse_bool, default=default_val,
                               help=f"Boolescher Wert (Default: {default_val})")
            elif isinstance(default_val, (list, dict)):
                p.add_argument(arg_name, type=parse_json, default=default_val,
                               help=f"JSON-String erwartet (Default: {json.dumps(default_val)})")
            elif isinstance(default_val, int):
                p.add_argument(arg_name, type=int, default=default_val,
                               help=f"Ganzzahl (Default: {default_val})")
            else:
                p.add_argument(arg_name, type=str, default=str(default_val),
                               help=f"String (Default: {default_val})")

        # Optionales Argument für beliebige weitere kwargs (falls in description erwähnt, aber nicht in live_test_inputs)
        p.add_argument("--extra-kwargs", type=parse_json, default={},
                       help="Zusätzliche Argumente als JSON-Dict, z.B.: '{\"state_name\": \"test\"}'")

    return parser


# ==========================================
# 4. ASYNC RUNNER & __MAIN__ BLOCK
# ==========================================

async def main_async(args, parsed_args_dict):
    """Führt die asynchrone Tool-Funktion aus."""
    func = args.func

    # Entferne interne argparse-Felder
    kwargs_for_tool = {k.replace('-', '_'): v for k, v in parsed_args_dict.items() if
                       k not in ['tool', 'func', 'extra_kwargs']}

    # Extra-Kwargs mergen (überschreiben defaults)
    if args.extra_kwargs:
        kwargs_for_tool.update(args.extra_kwargs)

    try:
        # Purer I/O Call der ausgewählten Funktion
        result = await func(**kwargs_for_tool)

        # Gebe das Ergebnis als sauberes JSON auf Stdout aus
        print(json.dumps({
            "status": "success",
            "tool": args.tool,
            "result": result
        }, indent=2, ensure_ascii=False))

    except Exception as e:
        # Bei Fehlern sauberes JSON über Stderr (damit Stdout pur bleibt für Pipes)
        error_msg = json.dumps({
            "status": "error",
            "tool": args.tool,
            "error_type": type(e).__name__,
            "message": str(e)
        }, indent=2)
        print(error_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Parser initialisieren und Argumente einlesen
    parser = build_parser()
    args = parser.parse_args()

    # Aus ArgumentNamespace ein Dictionary machen
    parsed_args_dict = vars(args)

    # Event-Loop starten
    try:
        asyncio.run(main_async(args, parsed_args_dict))
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "message": "Prozess durch User abgebrochen (SIGINT)"}), file=sys.stderr)
        sys.exit(130)
