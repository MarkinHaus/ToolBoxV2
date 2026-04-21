"""ISIS TU Berlin Scraper Toolkit - Minimal Login & Session
Credentials via Env-Vars: ISIS_USERNAME, ISIS_PASSWORD
"""

import asyncio
import json
import os
import logging
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    os.system("pip install playwright && playwright install chromium")
    from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

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

    async def save_state(self, name="isis_session"):
        path = STATE_DIR / (name + ".json")
        state = await self._context.storage_state()
        path.write_text(json.dumps(state, indent=2))
        logger.info("State gespeichert: %s", path)
        return str(path)

    async def load_state(self, name="isis_session"):
        path = STATE_DIR / (name + ".json")
        if not path.exists():
            return False
        if self._context:
            await self._context.close()
        self._context = await self._browser.new_context(
            storage_state=str(path), viewport={"width": 1920, "height": 1080}
        )
        self._page = await self._context.new_page()
        logger.info("State geladen: %s", path)
        return True

    async def close(self):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        logger.info("Session geschlossen")

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
        await self._page.goto(ISIS_BASE + '/my/', wait_until='networkidle')
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
        await self._page.goto(ISIS_BASE + '/message/index.php', wait_until='networkidle')
        await self._page.screenshot(path=str(STATE_DIR / '05_messages.png'))
        messages = []
        convos = await self._page.query_selector_all('[data-region="contact"]')
        if not convos:
            convos = await self._page.query_selector_all('.conversation')
        for conv in convos:
            try:
                name_el = await conv.query_selector('[data-region="contact-name"], .name')
                text_el = await conv.query_selector('[data-region="last-message"], .lastmessage')
                name = (await name_el.inner_text()).strip() if name_el else '?'
                text = (await text_el.inner_text()).strip() if text_el else ''
                messages.append({'from': name, 'text': text})
            except Exception:
                continue
        logger.info('Chat-Nachrichten: %d', len(messages))
        return messages

    async def get_course_overview(self, course_id):
        url = ISIS_BASE + '/course/view.php?id=' + str(course_id)
        logger.info('get_course_overview: %s', url)
        await self._page.goto(url, wait_until='networkidle')
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
        url = ISIS_BASE + '/course/view.php?id=' + str(course_id)
        logger.info('get_course_activities: %s', url)
        await self._page.goto(url, wait_until='networkidle')
        activities = []
        section_els = await self._page.query_selector_all('.course-content .section.main')
        for sec in section_els:
            try:
                name_el = await sec.query_selector('.sectionname, .content .sectionname')
                sec_name = (await name_el.inner_text()).strip() if name_el else '?'
            except Exception:
                sec_name = '?'
            act_els = await sec.query_selector_all('.activity .activityinstance a')
            for act in act_els:
                try:
                    href = await act.get_attribute('href') or ''
                    name = (await act.inner_text()).strip()
                    mod_type = 'unknown'
                    for t in ['assign', 'forum', 'resource', 'url', 'quiz', 'folder', 'page', 'scorm', 'choice', 'wiki', 'glossary', 'lesson', 'workshop', 'data']:
                        if t in href:
                            mod_type = t
                            break
                    activities.append({'name': name, 'url': href, 'type': mod_type, 'section': sec_name})
                except Exception:
                    continue
        logger.info('Activities gefunden: %d', len(activities))
        return activities
