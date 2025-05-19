
def main_page_interactions(self):
    """
    Test interactions for the ToolBoxV2 Main Page

    Returns a list of interaction dictionaries covering key page elements
    """
    return [
        # Navigate to the main page
        {"type": "goto", "url": "http://localhost:5000/web/"},

        # Take initial screenshot
        {"type": "screenshot", "path": "main_page_interactions/main_page_initial.png"},

        # Test navigation to App
        {"type": "click", "selector": "a[href='/web/mainContent.html']"},
        {"type": "screenshot", "path": "main_page_interactions/nav_to_app.png"},
        {"type": "goBack"},

        # Test navigation to Main Idea
        {"type": "click", "selector": "a[href='/web/core0/MainIdea.html']"},
        {"type": "screenshot", "path": "main_page_interactions/nav_to_main_idea.png"},
        {"type": "goBack"},

        # Test navigation to Installation
        {"type": "click", "selector": "a[href='/web/core0/Installer.html']"},
        {"type": "screenshot", "path": "main_page_interactions/nav_to_installer.png"},
        {"type": "goBack"},

        # Test external links
        {"type": "click", "selector": "a[href='https://github.com/MarkinHaus/ToolBoxV2']"},
        {"type": "screenshot", "path": "main_page_interactions/github_link.png"},
        {"type": "goBack"},

        {"type": "click", "selector": "a[href='https://markinhaus.github.io/ToolBoxV2/']"},
        {"type": "screenshot", "path": "main_page_interactions/docs_link.png"},
        {"type": "goBack"},

        # Test contact and impressum links
        {"type": "click", "selector": "a[href='https://github.com/MarkinHaus/ToolBoxV2']", "index": 1},
        {"type": "screenshot", "path": "main_page_interactions/contact_link.png"},
        {"type": "goBack"},

        {"type": "click", "selector": "a[href='https://markinhaus.github.io/ToolBoxV2/']", "index": 1},
        {"type": "screenshot", "path": "main_page_interactions/impressum_link.png"},
        {"type": "goBack"},

        # Verify welcome message text
        {"type": "test", "selector": ".welcome-message", "exists": True},
        {"type": "test", "selector": ".words span", "count": 5},
    ]


def installer_interactions(self):
    """
    Test interactions for the ToolBoxV2 Installer page

    Returns a list of interaction dictionaries covering key page elements
    """
    return [
        # Navigate to the installer page
        {"type": "goto", "url": "http://localhost:5000/web/core0/Installer.html"},

        # Check OS selection dropdown
        {"type": "click", "selector": "#os-selection"},
        {"type": "select", "selector": "#os-selection", "value": "Python Runtime"},
        {"type": "screenshot", "path": "installer_interactions/os_selection_python.png"},

        # Check Windows exe option
        {"type": "click", "selector": "#os-selection"},
        {"type": "select", "selector": "#os-selection", "value": "exe"},
        {"type": "screenshot", "path": "installer_interactions/os_selection_windows.png"},

        # Check Mac dmg option
        {"type": "click", "selector": "#os-selection"},
        {"type": "select", "selector": "#os-selection", "value": "dmg"},
        {"type": "screenshot", "path": "installer_interactions/os_selection_mac.png"},

        # Check Android apk option
        {"type": "click", "selector": "#os-selection"},
        {"type": "select", "selector": "#os-selection", "value": "apk"},
        {"type": "screenshot", "path": "installer_interactions/os_selection_android.png"},

        # Check iOS option
        {"type": "click", "selector": "#os-selection"},
        {"type": "select", "selector": "#os-selection", "value": "iOS-IPA"},
        {"type": "screenshot", "path": "installer_interactions/os_selection_ios.png"},

        # Check Web option
        {"type": "click", "selector": "#os-selection"},
        {"type": "select", "selector": "#os-selection", "value": "Web"},
        {"type": "screenshot", "path": "installer_interactions/os_selection_web.png"},

        # Check home link navigation
        {"type": "click", "selector": "a[href='/index.html']"},
        {"type": "screenshot", "path": "installer_interactions/home_link_click.png"},

        {"type": "goback"},

        # Check roadmap link
        {"type": "click", "selector": "a[href='/web/core0/roadmap.html']"},
        {"type": "screenshot", "path": "installer_interactions/roadmap_link_click.png"},

        {"type": "goback"},

        # Check support links
        {"type": "click", "selector": "a[href='https://www.buymeacoffee.com/markinhaus']"},
        {"type": "screenshot", "path": "installer_interactions/buymeacoffee_link.png"},

        {"type": "goback"},

        {"type": "click", "selector": "a[href='https://www.patreon.com/de-DE']"},
        {"type": "screenshot", "path": "installer_interactions/patreon_link.png"},
    ]


def contact_page_interactions(self):
    """
    Test interactions for the Contact Page

    Returns a list of interaction dictionaries covering key page elements
    """
    return [
        # Navigate to the contact page
        {"type": "goto", "url": "http://localhost:5000/web/core0/kontakt.html"},

        # Take initial screenshot
        {"type": "screenshot", "path": "contact_page_interactions/contact_page_initial.png"},

        # Verify form elements exist
        {"type": "test", "selector": "#name"},
        {"type": "test", "selector": "#email"},
        {"type": "test", "selector": "#subject"},
        {"type": "test", "selector": "#message"},

        # Additional form validation tests
        # Test form validation by leaving fields empty
        {"type": "click", "selector": "button[type='submit']"},
        {"type": "screenshot", "path": "contact_page_interactions/contact_form_empty_submission.png"},

        # Test form input
        {"type": "type", "selector": "#name", "text": "Test User"},
        {"type": "type", "selector": "#email", "text": "testuser@example.com"},
        {"type": "type", "selector": "#subject", "text": "Test Inquiry"},
        {"type": "type", "selector": "#message",
         "text": "This is a test message to verify the contact form functionality."},

        # Take screenshot after filling form
        {"type": "screenshot", "path": "contact_page_interactions/contact_form_filled.png"},

        # Validate form submission (note: this will trigger email client)
        {"type": "click", "selector": "button[type='submit']"},
        {"type": "screenshot", "path": "contact_page_interactions/contact_form_submitted.png"},

        # Test invalid email
        {"type": "type", "selector": "#name", "text": "Test User"},
        {"type": "type", "selector": "#email", "text": "invalid-email"},
        {"type": "type", "selector": "#subject", "text": "Test Inquiry"},
        {"type": "type", "selector": "#message", "text": "This is a test message."},
        {"type": "click", "selector": "button[type='submit']"},
        {"type": "screenshot", "path": "contact_page_interactions/contact_form_invalid_email.png"},
    ]

