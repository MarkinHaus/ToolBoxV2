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
        if msgs:
            first = msgs[0]
            assert "from" in first, "Nachricht braucht from"
            assert "text" in first, "Nachricht braucht text"
            print("  Neueste: " + first["from"] + " - " + first["text"][:50])
        print("  Nachrichten: " + str(len(msgs)))
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
        if acts:
            first = acts[0]
            assert "name" in first, "Aktivitaet braucht name"
            assert "url" in first, "Aktivitaet braucht url"
            assert "type" in first, "Aktivitaet braucht type"
            assert "section" in first, "Aktivitaet braucht section"
        print("  Aktivitaeten gesamt: " + str(len(acts)))
        types = {}
        for a in acts:
            t = a["type"]
            types[t] = types.get(t, 0) + 1
        for t, c in types.items():
            print("    " + t + ": " + str(c))
    return True


def run_tests():
    tests = [
        ("Kurse auflisten", test_list_courses),
        ("Chat-Nachrichten", test_list_chat_messages),
        ("Kurs-Overview", test_get_course_overview),
        ("Kurs-Aktivitaeten", test_get_course_activities),
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
