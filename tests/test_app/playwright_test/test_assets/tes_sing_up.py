import time

import faker
from playwright.sync_api import Playwright, sync_playwright

from toolboxv2 import get_app, TBEF


def test_run_sing_up(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    user_name = faker.Faker().word()

    page.goto("http://localhost:5000/web/signup")
    page.wait_for_load_state("networkidle")
    page.get_by_placeholder("Username").click()
    page.get_by_placeholder("Username").fill(user_name)
    page.get_by_placeholder("Email").click()
    page.get_by_placeholder("Email").fill("Markinhaus@gmail.com")
    page.get_by_placeholder("Initiation-Key").click()
    page.get_by_placeholder("Initiation-Key").fill("00#inv")
    page.get_by_role("button", name="Sign Up").click()
    page.wait_for_load_state("networkidle")
    time.sleep(10)
    page.get_by_role("button", name="Retry").click()

    time.sleep(300)
    page.get_by_role("button", name="Only Device").click()
    result = get_app().run_any(TBEF.DB.DELETE, query=f"USER::{user_name}::*", matching=True,
                               get_results=True).print()
    assert not result.is_error(), f"User Creation was not successfully {result.print(show=False)}"
    context.close()
    browser.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        test_run_sing_up(playwright)
