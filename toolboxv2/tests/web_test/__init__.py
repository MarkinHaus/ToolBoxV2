# toolboxv2/tests/web_test/__init__.py
"""
ToolBoxV2 E2E Web Tests Package

This package contains end-to-end tests for the ToolBoxV2 web interface.
"""

# Import test interaction functions
from .test_main_page import (
    main_page_interactions,
    contact_page_interactions,
    installer_page_interactions,
    login_page_interactions,
    signup_page_interactions,
    roadmap_page_interactions,
    main_idea_page_interactions,
    toolbox_infos_page_interactions,
)

# Test categorization for different session states
in_valid_session_tests = []
valid_session_tests = [contact_page_interactions, installer_page_interactions]
loot_session_tests = []
