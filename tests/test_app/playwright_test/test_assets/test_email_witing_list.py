from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("http://127.0.0.1:5000/app/assets/waiting_list.html")
    page.wait_for_load_state("networkidle")
    page.get_by_label("Email:").click()
    page.get_by_label("Email:").fill("test@email")
    page.get_by_role("button", name="Subscribe").click()

    # ---------------------
    context.close()
    browser.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
