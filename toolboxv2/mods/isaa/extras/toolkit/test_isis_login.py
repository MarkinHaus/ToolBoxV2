"""TDD Test: ISIS TU Berlin Login Validierung
Usage: python test_isis_login.py
"""

import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))

from isis_toolkit import IsisSession, HEADLESS, STATE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

HEADLESS_MODE = HEADLESS


async def test_login_returns_true():
    assert os.environ.get("ISIS_USERNAME"), "ISIS_USERNAME env var fehlt"
    assert os.environ.get("ISIS_PASSWORD"), "ISIS_PASSWORD env var fehlt"

    async with IsisSession(headless=HEADLESS_MODE) as session:
        result = await session.login()
        assert result is True, "Login fehlgeschlagen. Screenshots in " + str(STATE_DIR)

        auth = await session.is_authenticated()
        assert auth is True, "Session nicht authentifiziert nach Login"

        state_path = await session.save_state("test_session")
        assert os.path.exists(state_path), "State nicht gespeichert: " + state_path

    print("PASS: Login + Auth + State-Save erfolgreich")
    return True


async def test_load_saved_state():
    async with IsisSession(headless=HEADLESS_MODE) as session:
        loaded = await session.load_state("test_session")
        assert loaded is True, "State laden fehlgeschlagen"

        await session._page.goto("https://isis.tu-berlin.de/my/", wait_until="networkidle")
        auth = await session.is_authenticated()
        assert auth is True, "Geladener State nicht mehr authentifiziert"

    print("PASS: State laden + Re-Auth erfolgreich")
    return True


async def test_login_with_wrong_credentials():
    async with IsisSession(headless=HEADLESS_MODE) as session:
        result = await session.login(username="wrong_user_12345", password="wrong_pass_12345")
        assert result is False, "Login mit falschen Credentials sollte False sein"

    print("PASS: Falsche Credentials korrekt abgelehnt")
    return True


def run_tests():
    tests = [
        ("Login + Auth", test_login_returns_true),
        ("State laden", test_load_saved_state),
        ("Falsche Credentials", test_login_with_wrong_credentials),
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
    if not os.environ.get("ISIS_USERNAME") or not os.environ.get("ISIS_PASSWORD"):
        print("FEHLER: Setze ISIS_USERNAME und ISIS_PASSWORD:")
        print("  $env:ISIS_USERNAME=\"dein_username\"  # PowerShell")
        print("  $env:ISIS_PASSWORD=\"dein_passwort\"")
        sys.exit(1)
    print(1)
    success = run_tests()
    sys.exit(0 if success else 1)
