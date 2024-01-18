import time

from playwright.sync_api import Playwright, sync_playwright, expect


def test_run_sing_in(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    page.goto("http://127.0.0.1:5000/app/signup")
    page.wait_for_load_state("networkidle")
    page.get_by_placeholder("Username").click()
    page.get_by_placeholder("Username").fill("Kinr3")
    page.get_by_placeholder("Email").click()
    page.get_by_placeholder("Email").fill("Markinhaus@gmail.com")
    page.get_by_placeholder("Initiation-Key").click()
    page.get_by_placeholder("Initiation-Key").fill("00#inv")
    page.get_by_role("button", name="Sign Up").click()
    page.wait_for_load_state("networkidle")
    time.sleep(10)
    page.get_by_role("button", name="Retry").click()
    context.close()
    browser.close()


def test_run_email(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("http://127.0.0.1:5000/app/assets/waiting_list.html")
    page.wait_for_load_state("networkidle")
    page.get_by_placeholder("test@test.com").click()
    page.get_by_placeholder("test@test.com").fill("Test2@email.com")
    page.get_by_role("button", name="Subscribe").click()
    page.get_by_text("You will receive an invitation email in a few days").click()
    page.get_by_role("button", name="Subscribe").click()
    page.get_by_text("You are already in the list, please do not try to add yourself more than once.").click()
    context.close()
    browser.close()


def test_run_links(playwright: Playwright) -> None:

    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("http://127.0.0.1:5000/app/core0/index.html")
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Main Idea").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Routenplanung").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Kontaktwege").click()
    page.wait_for_load_state("networkidle")

    page.goto("http://127.0.0.1:5000/app/core0/roadmap.html")
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Home").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Installation").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Home").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="App").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Start").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Email waiting list").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Start").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Email waiting list").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Terms and Conditions").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("heading", name="404 Page not found").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Start", exact=True).click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Sign Up").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Login").click()
    page.wait_for_load_state("networkidle")

    page.get_by_label("Remember me").check()
    page.wait_for_load_state("networkidle")

    page.get_by_text("Remember me").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Terms and Conditions").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("heading", name="Welcome to simple").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Start").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Terms and Conditions").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Sign Up").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Terms and Conditions").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("heading", name="404 Page not found").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Start", exact=True).click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Terms and Conditions").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Login").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Terms and Conditions").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Start").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("link", name="Dashboard").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="LogOut").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="startMD").click()
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="Start Dashboard").click()
    page.wait_for_load_state("networkidle")

    page.locator("#main").click()
    page.wait_for_load_state("networkidle")

    # ---------------------
    context.close()
    browser.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        test_run_sing_in(playwright)
        # test_run(playwright)
