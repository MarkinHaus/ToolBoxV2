"""TDD Tests: ISIS Scraping Functions
Requires: test_session.json from test_isis_login.py
Usage: python test_scraping.py
"""

import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))

from isis_toolkit import IsisSession, HEADLESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


async def test_list_courses():
    async with IsisSession(headless=HEADLESS) as s:
        loaded = await s.load_state("test_session")
        assert loaded, "State fehlt - erst test_isis_login.py laufen lassen"
        courses = await s.list_courses()
        assert isinstance(courses, list), "list_courses soll Liste zurueckgeben"
        if courses:
            first = courses[0]
            assert "title" in first, "Kurs-Eintrag braucht title"
            assert "id" in first, "Kurs-Eintrag braucht id"
            assert "url" in first, "Kurs-Eintrag braucht url"
            print("  " + first["title"] + " (id=" + first["id"] + ")")
        print("  Kurse gefunden: " + str(len(courses)))
    return True


async def test_list_chat_messages():
    async with IsisSession(headless=HEADLESS) as s:
        loaded = await s.load_state("test_session")
        assert loaded, "State fehlt"
        msgs = await s.list_chat_messages()
        assert isinstance(msgs, list), "list_chat_messages soll Liste zurueckgeben"
        assert len(msgs) > 0, "Keine Nachrichten gefunden - Expand-Strategie kaputt?"
        first = msgs[0]
        for key in ("id", "from", "text", "section", "unread"):
            assert key in first, "Nachricht braucht " + key
        sections = {m["section"] for m in msgs}
        print("  Sections: " + str(sections))
        print("  Neueste: " + first["from"] + " - " + first["text"][:50])
        print("  Nachrichten: " + str(len(msgs)))
    return True

async def test_scrape_course_shallow():
    async with IsisSession(headless=HEADLESS) as s:
        loaded = await s.load_state("test_session")
        assert loaded, "State fehlt"
        cid = await _get_first_course_id(s)
        if not cid:
            print("  SKIP"); return True
        shallow = await s.scrape_course_shallow(cid)
        assert shallow["id"] == str(cid)
        assert isinstance(shallow["title"], str) and shallow["title"]
        assert isinstance(shallow["activities"], list)
        assert shallow["activity_count"] == len(shallow["activities"])
        assert isinstance(shallow["type_counts"], dict)
        print("  " + shallow["title"])
        print("  Sections: " + str(len(shallow["sections"])))
        print("  Activities: " + str(shallow["activity_count"]))
        print("  Types: " + str(shallow["type_counts"]))
    return True


async def _get_first_course_id(session):
    courses = await session.list_courses()
    if not courses:
        return None
    return courses[0]["id"]


async def test_get_course_overview():
    async with IsisSession(headless=HEADLESS) as s:
        loaded = await s.load_state("test_session")
        assert loaded, "State fehlt"
        cid = await _get_first_course_id(s)
        if not cid:
            print("  SKIP: Keine Kurse verfuegbar")
            return True
        overview = await s.get_course_overview(cid)
        assert overview["id"] == str(cid), "Course ID mismatch"
        assert isinstance(overview["title"], str), "title soll string sein"
        assert isinstance(overview["sections"], list), "sections soll liste sein"
        print("  Kurs: " + overview["title"])
        print("  Abschnitte: " + str(len(overview["sections"])))
        for sec in overview["sections"]:
            print("    - " + sec["name"] + " (" + str(sec["activity_count"]) + " Aktivitaeten)")
    return True


async def test_get_course_activities():
    async with IsisSession(headless=HEADLESS) as s:
        loaded = await s.load_state("test_session")
        assert loaded, "State fehlt"
        cid = await _get_first_course_id(s)
        if not cid:
            print("  SKIP: Keine Kurse verfuegbar")
            return True
        acts = await s.get_course_activities(cid)
        assert isinstance(acts, list), "activities soll liste sein"
        assert len(acts) > 0, "Kurs " + str(cid) + " hat keine Aktivitaeten gefunden (overview.php leer?)"
        for key in ("name", "url", "type", "section", "cmid", "extra"):
            assert key in acts[0], "Aktivitaet braucht " + key
        assert acts[0]["cmid"] and acts[0]["cmid"].isdigit(), "cmid soll numerisch sein"
        types = {}
        for a in acts:
            types[a["type"]] = types.get(a["type"], 0) + 1
            print(a)
        print("  Aktivitaeten gesamt: " + str(len(acts)))
        for t, c in types.items():
            print("    " + t + ": " + str(c))
    return True

async def test_scrape_course_sections_markdown():
    async with IsisSession(headless=HEADLESS) as s:
        loaded = await s.load_state("test_session")
        assert loaded, "State fehlt"
        cid = await _get_first_course_id(s)
        if not cid:
            print("  SKIP"); return True
        sections = await s.scrape_course_sections_markdown(cid)
        assert isinstance(sections, list), "sections soll liste sein"
        assert len(sections) > 0, "Keine Sections gefunden"
        for key in ("idx", "id", "name", "url", "markdown"):
            assert key in sections[0], "section braucht " + key
        non_empty = [s for s in sections if s["markdown"]]
        assert len(non_empty) > 0, "Alle sections haben leeres markdown - Expand nicht geklappt?"
        total_chars = sum(len(s["markdown"]) for s in sections)
        print("  Sections: " + str(len(sections)) + " (non-empty: " + str(len(non_empty)) + ")")
        print("  MD total: " + str(total_chars) + " chars")
        for sec in sections[:3]:
            preview = sec["markdown"].replace("\n", " | ")
            print("    [" + str(sec["idx"]) + "] " + sec["name"][:40] + " -> " + preview)
    return True


async def test_auto_relogin():
    # erzwinge expired state durch umbenennen eines dummy-files
    import shutil
    from isis_toolkit import STATE_DIR
    real = STATE_DIR / "test_session.json"
    if not real.exists():
        print("  SKIP: test_session.json fehlt"); return True
    bak = STATE_DIR / "test_session_bak.json"
    shutil.copy(real, bak)
    # corrupt state: leere cookies simulieren expired
    import json
    state = json.loads(real.read_text())
    for c in state.get('cookies', []):
        c['value'] = 'expired'
    real.write_text(json.dumps(state))
    try:
        async with IsisSession(headless=HEADLESS) as s:
            loaded = await s.load_state("test_session")
            assert loaded, "Auto-Relogin fehlgeschlagen"
            # nach relogin: list_courses muss funktionieren
            courses = await s.list_courses()
            assert len(courses) > 0, "Nach Relogin keine Kurse"
            print("  Relogin + " + str(len(courses)) + " Kurse")
    finally:
        shutil.copy(bak, real)  # restore
        bak.unlink()
    return True
async def test_scrape_activity_types():
    async with IsisSession(headless=HEADLESS) as s:
        assert await s.load_state("test_session")
        # kurs 46889 hat forum+assign+quiz+choicegroup+questionnaire+resource(page)
        targets = [
            {'url': 'https://isis.tu-berlin.de/mod/forum/view.php?id=2230845', 'type': 'forum'},
            {'url': 'https://isis.tu-berlin.de/mod/assign/view.php?id=2311345', 'type': 'assign'},
            {'url': 'https://isis.tu-berlin.de/mod/quiz/view.php?id=2308914', 'type': 'quiz'},
            {'url': 'https://isis.tu-berlin.de/mod/choicegroup/view.php?id=2310574', 'type': 'choicegroup'},
            {'url': 'https://isis.tu-berlin.de/mod/page/view.php?id=2230847', 'type': 'resource'},
            {'url': 'https://isis.tu-berlin.de/mod/questionnaire/view.php?id=2230850', 'type': 'questionnaire'},
        ]
        for t in targets:
            r = await s.scrape_activity(t)
            assert r.get('scraped'), t['type'] + ' not scraped: ' + str(r.get('error'))
            assert r.get('type') == t['type'], t['type'] + ' type mismatch'
            print('  [' + t['type'] + '] keys: ' + ','.join(k for k in r.keys() if k not in ('url','type','scraped')))
            if t['type'] == 'forum':
                assert 'threads' in r
                print('    threads: ' + str(r['thread_count']))
                if r['threads']:
                    print('    first thread posts: ' + str(r['threads'][0].get('post_count', 0)))
            if t['type'] == 'assign':
                assert 'dates' in r and 'submission_status' in r
                print('    dates: ' + str(r['dates']))
            if t['type'] == 'choicegroup':
                assert 'options' in r
                print('    options: ' + str(len(r['options'])) + ' own: ' + str(r.get('own_choice')))
    return True


async def test_toolkit_module_api():
    """Singleton-API Test. Stoppt am Ende um anderen Tests nicht zu stoeren."""
    from isis_toolkit import tool_isis_start, tool_isis_stop, tool_list_courses
    started = await tool_isis_start(headless=HEADLESS)
    assert started['started'], 'start failed'
    assert started['session_valid'], 'session not valid'
    courses = await tool_list_courses()
    assert isinstance(courses, list) and len(courses) > 0
    print('  singleton courses: ' + str(len(courses)))
    stopped = await tool_isis_stop()
    assert stopped['stopped']
    return True

def run_tests():
    tests = [
        #("Kurse auflisten", test_list_courses),
        #("Auto-Relogin", test_auto_relogin),
        #("Chat-Nachrichten", test_list_chat_messages),
        #("Kurs-Overview", test_get_course_overview),
        #("Kurs-Aktivitaeten", test_get_course_activities),
        ("Activity types", test_scrape_activity_types),
        ("Toolkit Singleton API", test_toolkit_module_api),
        # ("Course Sections MD", test_scrape_course_sections_markdown),
    ]
    results = []
    for name, test_fn in tests:
        sep = "=" * 60
        print("")
        print(sep)
        print("TEST: " + name)
        print(sep)
        try:
            asyncio.run(test_fn())
            results.append((name, "PASS"))
        except AssertionError as e:
            print("FAIL: " + str(e))
            results.append((name, "FAIL: " + str(e)))
        except Exception as e:
            print("ERROR: " + str(e))
            results.append((name, "ERROR: " + str(e)))

    sep = "=" * 60
    print("")
    print(sep)
    print("ERGEBNISSE")
    print(sep)
    for name, status in results:
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print("  " + icon + " " + name + ": " + status)

    return all(s == "PASS" for _, s in results)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    success = run_tests()
    sys.exit(0 if success else 1)
