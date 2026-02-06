# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyautogui",
#   "pygetwindow",
#   "mss",
#   "opencv-python",
#   "numpy",
# ]
# ///
import os
import platform
import re
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import pyautogui
import pygetwindow as gw


# ============================================================================
# ENHANCED TYPES
# ============================================================================

class UserLevel(Enum):
    """User-Profile für kontextabhängige Skills"""
    BASIC = "basic"  # Endanwender
    POWER = "power"  # Power User
    BUSINESS = "business"  # Unternehmens-Mitarbeiter
    DEVELOPER = "developer"  # Entwickler


@dataclass
class ActionStep:
    """Atomarer Schritt in einer Skill-Action"""
    action_type: str  # "click", "type", "hotkey", "wait", "navigate"
    target: Optional[str] = None
    value: Optional[str] = None
    wait_ms: int = 200
    description: str = ""


@dataclass
class ApplicationContext:
    """Vollständiger Kontext einer App"""
    name: str
    title: str
    handle: any
    geometry: Tuple[int, int, int, int]
    is_browser: bool = False
    browser_url: Optional[str] = None
    detected_page: Optional[str] = None
    skill_available: bool = False
    available_actions: List[str] = field(default_factory=list)


# ============================================================================
# UI-TREE EXTRACTOR (von vorher, gekürzt)
# ============================================================================

class UITreeExtractor:
    """Plattform-unabhängiger UI-Tree Extraktor"""

    def __init__(self):
        self.platform = platform.system()
        # Implementation wie vorher (Windows/Linux/macOS)
        pass

    def get_windows(self) -> List[Dict]:
        """Gibt alle offenen Fenster zurück"""
        windows = []
        for win in gw.getAllWindows():
            if win.title and win.visible:
                windows.append({
                    "title": win.title,
                    "handle": win._hWnd if hasattr(win, '_hWnd') else win,
                    "app": self._extract_app_name(win.title),
                    "geometry": (win.left, win.top, win.width, win.height)
                })
        return windows

    def _extract_app_name(self, window_title: str) -> str:
        """Intelligente App-Erkennung"""
        patterns = {
            r'chrome|chromium': 'chrome',
            r'firefox': 'firefox',
            r'spotify': 'spotify',
            r'pycharm|intellij': 'pycharm',
            r'terminal|konsole|gnome-terminal': 'terminal',
            r'powershell': 'powershell',
            r'explorer\.exe|file explorer': 'explorer',
            r'finder': 'finder',
            r'outlook': 'outlook',
            r'slack': 'slack',
            r'code|visual studio code': 'vscode',
        }

        title_lower = window_title.lower()
        for pattern, app_name in patterns.items():
            if re.search(pattern, title_lower):
                return app_name

        return title_lower.split()[0] if title_lower.split() else "unknown"


# ============================================================================
# BROWSER URL DETECTION
# ============================================================================

class BrowserURLDetector:
    """Erkennt aktuelle URL in Browser-Fenstern"""

    def detect_url(self, window_title: str, app_name: str) -> Optional[str]:
        """
        Extrahiert URL aus Browser-Titel.
        Format: "Page Title - Google Chrome" oder "URL - Firefox"
        """
        if app_name not in ['chrome', 'firefox', 'edge', 'safari']:
            return None

        # Chrome: URL oft im Titel oder via Clipboard-Trick
        # Vereinfachte Version: Pattern-Matching
        url_patterns = [
            r'(https?://[^\s]+)',
            r'([a-z0-9.-]+\.[a-z]{2,})',
        ]

        for pattern in url_patterns:
            match = re.search(pattern, window_title.lower())
            if match:
                return match.group(1)

        return None

    def detect_page_type(self, url: str) -> Optional[str]:
        """Erkennt spezifische Seiten (Instagram, WhatsApp, etc.)"""
        if not url:
            return None

        page_patterns = {
            'instagram': r'instagram\.com',
            'whatsapp': r'web\.whatsapp\.com',
            'personio': r'personio\.',
            'wheniwork': r'wheniwork\.com',
            'slack': r'slack\.com',
            'gmail': r'mail\.google\.com',
            'tu_berlin': r'tu-berlin\.de',
            'youtube': r'youtube\.com',
            'netflix': r'netflix\.com',
            'notion': r'notion\.so',
            'trello': r'trello\.com',
            'jira': r'atlassian\.net',
        }

        for page_type, pattern in page_patterns.items():
            if re.search(pattern, url):
                return page_type

        return None


# ============================================================================
# SKILL SYSTEM - BASE CLASS
# ============================================================================

class AppSkill:
    """Basis-Klasse für alle App-Skills - FIXED VERSION"""

    def __init__(self, app_name: str, user_levels: List[UserLevel] = None):
        self.app_name = app_name
        self.user_levels = user_levels or [UserLevel.BASIC]
        self.actions = {}
        self.description = ""

    def register_action(self, name: str, steps: List[ActionStep],
                        description: str = "", user_level: UserLevel = UserLevel.BASIC):
        """Registriert neue Action"""
        self.actions[name] = {
            "steps": steps,
            "description": description,
            "user_level": user_level
        }

    def get_available_actions(self, user_level: UserLevel = UserLevel.BASIC) -> List[str]:
        """Gibt verfügbare Actions für User-Level zurück"""
        return [
            name for name, action in self.actions.items()
            if action["user_level"].value <= user_level.value
        ]

    def get_action_plan(self, intent: str) -> Optional[List[ActionStep]]:
        """
        Findet passende Action für Intent.

        FIXED: Unterstützt jetzt:
        - Exakte Matches: "play_pause" → "play_pause"
        - Spaced Matches: "play pause" → "play_pause"
        - Partial Matches: "play" → "play_pause"
        """
        intent_clean = intent.lower().strip()

        # 1. Exakter Match (höchste Priorität)
        if intent_clean in self.actions:
            return self.actions[intent_clean]["steps"]

        # 2. Underscore/Space-Varianten
        intent_normalized = intent_clean.replace(" ", "_")
        if intent_normalized in self.actions:
            return self.actions[intent_normalized]["steps"]

        # 3. Spaced-Version des Action-Names
        for action_name, action_data in self.actions.items():
            action_spaced = action_name.replace("_", " ")
            if intent_clean == action_spaced:
                return action_data["steps"]

        # 4. Partial Match (niedrigste Priorität)
        for action_name, action_data in self.actions.items():
            action_variants = [
                action_name,
                action_name.replace("_", " "),
                action_name.replace("_", "")
            ]
            if any(intent_clean in variant for variant in action_variants):
                return action_data["steps"]

        return None


# ============================================================================
# BROWSER SKILLS
# ============================================================================

class BrowserSkill(AppSkill):
    """Allgemeine Browser-Skills (Chrome, Firefox, etc.)"""

    def __init__(self, browser_name: str = "chrome"):
        super().__init__(browser_name, [UserLevel.BASIC, UserLevel.POWER])
        self.description = "General web browser automation"
        self._register_basic_actions()
        self._register_power_actions()

    def _register_basic_actions(self):
        """Basic User Actions"""

        # Neuer Tab
        self.register_action("new_tab", [
            ActionStep("hotkey", value="ctrl+t", description="Open new tab")
        ], "Open a new browser tab", UserLevel.BASIC)

        # Tab schließen
        self.register_action("close_tab", [
            ActionStep("hotkey", value="ctrl+w", description="Close current tab")
        ], "Close active tab", UserLevel.BASIC)

        # URL öffnen
        self.register_action("navigate_url", [
            ActionStep("hotkey", value="ctrl+l", description="Focus address bar"),
            ActionStep("type", value="{url}", description="Type URL"),
            ActionStep("hotkey", value="enter", description="Navigate")
        ], "Navigate to URL", UserLevel.BASIC)

        # Suche
        self.register_action("search", [
            ActionStep("hotkey", value="ctrl+l"),
            ActionStep("type", value="{query}"),
            ActionStep("hotkey", value="enter")
        ], "Search via address bar", UserLevel.BASIC)

        # Verlauf
        self.register_action("open_history", [
            ActionStep("hotkey", value="ctrl+h")
        ], "Open browsing history", UserLevel.BASIC)

        # Downloads
        self.register_action("open_downloads", [
            ActionStep("hotkey", value="ctrl+j")
        ], "Open downloads page", UserLevel.BASIC)

        # Lesezeichen
        self.register_action("bookmark_page", [
            ActionStep("hotkey", value="ctrl+d")
        ], "Bookmark current page", UserLevel.BASIC)

        # Inkognito
        self.register_action("new_incognito", [
            ActionStep("hotkey", value="ctrl+shift+n")
        ], "Open incognito window", UserLevel.BASIC)

    def _register_power_actions(self):
        """Power User Actions"""

        # DevTools
        self.register_action("open_devtools", [
            ActionStep("hotkey", value="f12")
        ], "Open Developer Tools", UserLevel.POWER)

        # Hard Reload
        self.register_action("hard_reload", [
            ActionStep("hotkey", value="ctrl+shift+r")
        ], "Hard reload (clear cache)", UserLevel.POWER)

        # Extensions
        self.register_action("manage_extensions", [
            ActionStep("hotkey", value="ctrl+shift+delete")
        ], "Clear browsing data", UserLevel.POWER)

        # Zoom
        self.register_action("zoom_in", [
            ActionStep("hotkey", value="ctrl+plus")
        ], "Zoom in", UserLevel.POWER)

        self.register_action("zoom_out", [
            ActionStep("hotkey", value="ctrl+minus")
        ], "Zoom out", UserLevel.POWER)


# ============================================================================
# PAGE-SPECIFIC SKILLS (Browser)
# ============================================================================

class InstagramSkill(AppSkill):
    """Instagram Web-Spezifische Skills"""

    def __init__(self):
        super().__init__("instagram", [UserLevel.BASIC])
        self.description = "Instagram web interface automation"

        self.register_action("open_direct_messages", [
            ActionStep("click", target="dm_icon", description="Click DM icon"),
        ], "Open Instagram Direct Messages")

        self.register_action("search_user", [
            ActionStep("click", target="search_box"),
            ActionStep("type", value="{username}"),
            ActionStep("wait", wait_ms=500)
        ], "Search for a user")

        self.register_action("like_post", [
            ActionStep("click", target="like_button")
        ], "Like current post")


class WhatsAppWebSkill(AppSkill):
    """WhatsApp Web Skills"""

    def __init__(self):
        super().__init__("whatsapp", [UserLevel.BASIC])
        self.description = "WhatsApp Web automation"

        self.register_action("send_message", [
            ActionStep("click", target="message_input"),
            ActionStep("type", value="{message}"),
            ActionStep("hotkey", value="enter")
        ], "Send message in active chat")

        self.register_action("search_chat", [
            ActionStep("hotkey", value="ctrl+f"),
            ActionStep("type", value="{query}")
        ], "Search chats")

        self.register_action("attach_file", [
            ActionStep("click", target="attach_button"),
            ActionStep("click", target="document_option")
        ], "Attach file to message")


class PersonioSkill(AppSkill):
    """Personio HR Platform Skills"""

    def __init__(self):
        super().__init__("personio", [UserLevel.BUSINESS])
        self.description = "Personio HR automation"

        self.register_action("request_absence", [
            ActionStep("click", target="absence_menu"),
            ActionStep("click", target="new_request"),
            ActionStep("type", value="{dates}")
        ], "Request time off", UserLevel.BUSINESS)

        self.register_action("view_documents", [
            ActionStep("click", target="documents_tab")
        ], "View personal documents", UserLevel.BUSINESS)


class WhenIWorkSkill(AppSkill):
    """WhenIWork Scheduling Skills"""

    def __init__(self):
        super().__init__("wheniwork", [UserLevel.BUSINESS])

        self.register_action("view_schedule", [
            ActionStep("click", target="schedule_tab")
        ], "View work schedule")

        self.register_action("clock_in", [
            ActionStep("click", target="clock_in_button")
        ], "Clock in for shift")


class SlackSkill(AppSkill):
    """Slack Communication Skills"""

    def __init__(self):
        super().__init__("slack", [UserLevel.BUSINESS, UserLevel.POWER])

        self.register_action("quick_switcher", [
            ActionStep("hotkey", value="ctrl+k")
        ], "Open quick switcher")

        self.register_action("set_status", [
            ActionStep("click", target="profile_icon"),
            ActionStep("click", target="set_status")
        ], "Set status message")

        self.register_action("search_messages", [
            ActionStep("hotkey", value="ctrl+f"),
            ActionStep("type", value="{query}")
        ], "Search messages")


class GmailSkill(AppSkill):
    """Gmail Skills"""

    def __init__(self):
        super().__init__("gmail", [UserLevel.BASIC, UserLevel.POWER])

        # Basic
        self.register_action("compose", [
            ActionStep("hotkey", value="c")
        ], "Compose new email", UserLevel.BASIC)

        self.register_action("search", [
            ActionStep("hotkey", value="/"),
            ActionStep("type", value="{query}"),
            ActionStep("hotkey", value="enter")
        ], "Search emails", UserLevel.BASIC)

        # Power
        self.register_action("archive", [
            ActionStep("hotkey", value="e")
        ], "Archive selected email", UserLevel.POWER)

        self.register_action("apply_label", [
            ActionStep("hotkey", value="l"),
            ActionStep("type", value="{label_name}"),
            ActionStep("hotkey", value="enter")
        ], "Apply label", UserLevel.POWER)


# ============================================================================
# APPLICATION SKILLS (Non-Browser)
# ============================================================================

class SpotifySkill(AppSkill):
    """Spotify Desktop Skills"""

    def __init__(self):
        super().__init__("spotify", [UserLevel.BASIC])

        self.register_action("play_pause", [
            ActionStep("hotkey", value="space")
        ], "Toggle play/pause")

        self.register_action("next_track", [
            ActionStep("hotkey", value="ctrl+right")
        ], "Skip to next track")

        self.register_action("previous_track", [
            ActionStep("hotkey", value="ctrl+left")
        ], "Go to previous track")

        self.register_action("search", [
            ActionStep("hotkey", value="ctrl+l"),
            ActionStep("type", value="{query}")
        ], "Search Spotify")


class TerminalSkill(AppSkill):
    """Unix-based Terminal Skills"""

    def __init__(self):
        super().__init__("terminal", [UserLevel.POWER, UserLevel.DEVELOPER])

        self.register_action("new_tab", [
            ActionStep("hotkey", value="ctrl+shift+t")
        ], "New terminal tab", UserLevel.POWER)

        self.register_action("clear_screen", [
            ActionStep("type", value="clear"),
            ActionStep("hotkey", value="enter")
        ], "Clear terminal", UserLevel.POWER)

        self.register_action("list_files", [
            ActionStep("type", value="ls -la"),
            ActionStep("hotkey", value="enter")
        ], "List files (detailed)", UserLevel.DEVELOPER)

        self.register_action("change_directory", [
            ActionStep("type", value="cd {path}"),
            ActionStep("hotkey", value="enter")
        ], "Change directory", UserLevel.POWER)


class PowerShellSkill(AppSkill):
    """Windows PowerShell Skills"""

    def __init__(self):
        super().__init__("powershell", [UserLevel.POWER])

        self.register_action("get_processes", [
            ActionStep("type", value="Get-Process"),
            ActionStep("hotkey", value="enter")
        ], "List running processes")

        self.register_action("get_services", [
            ActionStep("type", value="Get-Service"),
            ActionStep("hotkey", value="enter")
        ], "List system services")

        self.register_action("clear", [
            ActionStep("type", value="Clear-Host"),
            ActionStep("hotkey", value="enter")
        ], "Clear console")


class FileExplorerSkill(AppSkill):
    """Windows File Explorer Skills"""

    def __init__(self):
        super().__init__("explorer", [UserLevel.BASIC])

        self.register_action("new_folder", [
            ActionStep("hotkey", value="ctrl+shift+n")
        ], "Create new folder")

        self.register_action("search", [
            ActionStep("hotkey", value="ctrl+f"),
            ActionStep("type", value="{query}")
        ], "Search files")

        self.register_action("copy", [
            ActionStep("hotkey", value="ctrl+c")
        ], "Copy selected items")

        self.register_action("paste", [
            ActionStep("hotkey", value="ctrl+v")
        ], "Paste items")


class TaskManagerSkill(AppSkill):
    """Windows Task Manager Skills"""

    def __init__(self):
        super().__init__("taskmgr", [UserLevel.POWER])

        self.register_action("open", [
            ActionStep("hotkey", value="ctrl+shift+esc")
        ], "Open Task Manager")

        self.register_action("end_task", [
            ActionStep("hotkey", value="delete")
        ], "End selected task")


# ============================================================================
# OS-SPECIFIC SETTINGS SKILLS
# ============================================================================

class WindowsSettingsSkill(AppSkill):
    """Windows System Settings"""

    def __init__(self):
        super().__init__("windows_settings", [UserLevel.BASIC, UserLevel.POWER])

        self.register_action("open_settings", [
            ActionStep("hotkey", value="win+i")
        ], "Open Windows Settings")

        self.register_action("check_updates", [
            ActionStep("hotkey", value="win+i"),
            ActionStep("type", value="update"),
            ActionStep("hotkey", value="enter")
        ], "Check for updates")


class LinuxSettingsSkill(AppSkill):
    """Linux System Settings"""

    def __init__(self):
        super().__init__("linux_settings", [UserLevel.POWER])

        self.register_action("open_settings", [
            ActionStep("click", target="settings_icon")
        ], "Open system settings")


class MacOSSettingsSkill(AppSkill):
    """macOS System Preferences"""

    def __init__(self):
        super().__init__("macos_settings", [UserLevel.BASIC])

        self.register_action("open_preferences", [
            ActionStep("hotkey", value="cmd+,")
        ], "Open System Preferences")


# ============================================================================
# SKILL REGISTRY
# ============================================================================

class SkillRegistry:
    """Zentrale Verwaltung aller Skills"""

    def __init__(self):
        self.app_skills = {}
        self.page_skills = {}
        self.os_skills = {}
        self.url_detector = BrowserURLDetector()

        self._register_all_skills()

    def _register_all_skills(self):
        """Registriert alle verfügbaren Skills"""

        # Browser Skills
        self.app_skills['chrome'] = BrowserSkill('chrome')
        self.app_skills['firefox'] = BrowserSkill('firefox')
        self.app_skills['edge'] = BrowserSkill('edge')

        # Page-specific Skills
        self.page_skills['instagram'] = InstagramSkill()
        self.page_skills['whatsapp'] = WhatsAppWebSkill()
        self.page_skills['personio'] = PersonioSkill()
        self.page_skills['wheniwork'] = WhenIWorkSkill()
        self.page_skills['slack'] = SlackSkill()
        self.page_skills['gmail'] = GmailSkill()

        # Application Skills
        self.app_skills['spotify'] = SpotifySkill()
        self.app_skills['terminal'] = TerminalSkill()
        self.app_skills['powershell'] = PowerShellSkill()
        self.app_skills['explorer'] = FileExplorerSkill()
        self.app_skills['finder'] = FileExplorerSkill()  # macOS
        self.app_skills['taskmgr'] = TaskManagerSkill()

        # OS Settings
        current_os = platform.system()
        if current_os == "Windows":
            self.os_skills['settings'] = WindowsSettingsSkill()
        elif current_os == "Linux":
            self.os_skills['settings'] = LinuxSettingsSkill()
        elif current_os == "Darwin":
            self.os_skills['settings'] = MacOSSettingsSkill()

    def get_skill_for_app(self, app_name: str, url: str = None) -> Optional[AppSkill]:
        """Gibt passenden Skill zurück (priorisiert Page-Skills)"""

        # Prüfe Page-Skills bei Browsern
        if url and app_name in ['chrome', 'firefox', 'edge', 'safari']:
            page_type = self.url_detector.detect_page_type(url)
            if page_type and page_type in self.page_skills:
                return self.page_skills[page_type]

        # Fallback: App-Skill
        return self.app_skills.get(app_name)

    def get_all_registered_skills(self) -> Dict[str, List[str]]:
        """Gibt Übersicht aller Skills zurück"""
        return {
            "applications": list(self.app_skills.keys()),
            "web_pages": list(self.page_skills.keys()),
            "os_settings": list(self.os_skills.keys())
        }


# ============================================================================
# HAUPTKLASSE: ENHANCED DESKTOP AUTOMATION
# ============================================================================

class EnhancedDesktopAutomation:
    """
    Verbesserte Desktop-Automation mit:
    - Übersichtlicher Einsicht
    - Auto-Switch
    - Umfangreicher Skill-Library
    """

    def __init__(self, user_level: UserLevel = UserLevel.BASIC):
        self.ui_extractor = UITreeExtractor()
        self.skill_registry = SkillRegistry()
        self.url_detector = BrowserURLDetector()
        self.user_level = user_level

        # Session State
        self.active_app: Optional[ApplicationContext] = None
        self.open_applications: List[ApplicationContext] = []
        self.last_scan_time = 0

    # ========================================================================
    # TOOL 1: SCOUT INTERFACE (ENHANCED)
    # ========================================================================

    def scout_interface(self,
                        app_name: Optional[str] = None,
                        window_title: Optional[str] = None,
                        auto_switch: bool = True) -> Dict:
        """
        Erweiterte Scout-Funktion mit:
        - Übersicht aller offenen Apps
        - Automatischer App-Wechsel wenn gewünscht
        - Mini-Preview der wichtigsten Details

        Returns:
            {
                "open_applications": [...],  # Alle offenen Apps
                "active_application": {...}, # Aktuell aktive App
                "possible_actions": {
                    "switch_to": [...],      # App-Wechsel möglich
                    "interact": [...]        # Verfügbare Interactions
                },
                "preview": {...}             # Fokussierte Details der aktiven App
            }
        """

        # Step 1: Alle offenen Fenster scannen
        windows = self.ui_extractor.get_windows()

        self.open_applications = []
        for win in windows:
            url = None
            page_type = None

            # Browser URL Detection
            if win['app'] in ['chrome', 'firefox', 'edge', 'safari']:
                url = self.url_detector.detect_url(win['title'], win['app'])
                if url:
                    page_type = self.url_detector.detect_page_type(url)

            # Skill-Check
            skill = self.skill_registry.get_skill_for_app(win['app'], url)

            app_ctx = ApplicationContext(
                name=win['app'],
                title=win['title'],
                handle=win['handle'],
                geometry=win['geometry'],
                is_browser=win['app'] in ['chrome', 'firefox', 'edge', 'safari'],
                browser_url=url,
                detected_page=page_type,
                skill_available=skill is not None,
                available_actions=skill.get_available_actions(self.user_level) if skill else []
            )

            self.open_applications.append(app_ctx)

        # Step 2: Aktive App bestimmen/wechseln
        if app_name or window_title:
            # User hat spezifische App gewählt
            target_app = self._find_application(app_name, window_title)

            if not target_app:
                return {
                    "status": "error",
                    "message": f"App nicht gefunden: {app_name or window_title}",
                    "open_applications": self._serialize_apps_overview()
                }

            # Auto-Switch aktivieren
            if auto_switch:
                self._activate_window(target_app)

            self.active_app = target_app
        else:
            # Keine Auswahl: Zeige aktuell aktive App
            active_win = gw.getActiveWindow()
            if active_win:
                self.active_app = next(
                    (app for app in self.open_applications if app.title == active_win.title),
                    None
                )

        # Step 3: Preview generieren
        preview = self._generate_preview()

        return {
            "status": "success",
            "open_applications": self._serialize_apps_overview(),
            "active_application": self._serialize_app_detail(self.active_app) if self.active_app else None,
            "possible_actions": {
                "switch_to": [app.name for app in self.open_applications if app != self.active_app],
                "interact": self.active_app.available_actions if self.active_app else []
            },
            "preview": preview
        }

    def _find_application(self, app_name: Optional[str], window_title: Optional[str]) -> Optional[ApplicationContext]:
        """Findet App in offenen Anwendungen"""
        for app in self.open_applications:
            if app_name and app.name == app_name.lower():
                return app
            if window_title and window_title.lower() in app.title.lower():
                return app
        return None

    def _activate_window(self, app: ApplicationContext):
        """Aktiviert Fenster (App-Switch)"""
        try:
            windows = gw.getWindowsWithTitle(app.title)
            if windows:
                windows[0].activate()
                time.sleep(0.3)  # Warte auf Fokus
        except Exception as e:
            print(f"Window activation failed: {e}")

    def _serialize_apps_overview(self) -> List[Dict]:
        """Kompakte Übersicht aller Apps"""
        return [
            {
                "name": app.name,
                "title": app.title[:50] + "..." if len(app.title) > 50 else app.title,
                "type": "browser" if app.is_browser else "application",
                "page": app.detected_page,
                "has_skill": app.skill_available
            }
            for app in self.open_applications
        ]

    def _serialize_app_detail(self, app: ApplicationContext) -> Dict:
        """Detaillierte App-Informationen"""
        return {
            "name": app.name,
            "title": app.title,
            "geometry": app.geometry,
            "is_browser": app.is_browser,
            "browser_url": app.browser_url,
            "detected_page": app.detected_page,
            "skill_available": app.skill_available,
            "available_actions": app.available_actions
        }

    def _generate_preview(self) -> Dict:
        """
        Mini-Preview der wichtigsten Details der aktiven App.
        Fokussiert auf scout_interface Args.
        """
        if not self.active_app:
            return {"message": "No active application"}

        skill = self.skill_registry.get_skill_for_app(
            self.active_app.name,
            self.active_app.browser_url
        )

        preview = {
            "app_name": self.active_app.name,
            "quick_summary": f"{self.active_app.name.upper()} - {self.active_app.detected_page or 'General'}"
        }

        # Skill-basierte Highlights
        if skill:
            top_actions = skill.get_available_actions(self.user_level)[:5]
            preview["key_actions"] = top_actions
            preview["skill_description"] = skill.description
        else:
            preview["key_actions"] = ["No predefined skills available"]
            preview["note"] = "Use UI-Tree or Vision for interaction"

        # Browser-spezifische Infos
        if self.active_app.is_browser and self.active_app.browser_url:
            preview["current_url"] = self.active_app.browser_url
            if self.active_app.detected_page:
                preview["page_type"] = self.active_app.detected_page.upper()

        return preview

    # ========================================================================
    # TOOL 2: EXECUTE ACTION
    # ========================================================================

    def execute_action(self, command: str) -> Dict:
        """
        FIXED VERSION: Führt Action aus mit intelligenterem Matching.
        """

        if not self.active_app:
            return {
                "status": "error",
                "message": "No active application. Call scout_interface first."
            }

        # Step 1: Prüfe Skill
        skill = self.skill_registry.get_skill_for_app(
            self.active_app.name,
            self.active_app.browser_url
        )

        if skill:
            # NEUE MATCHING-LOGIK
            action_plan = self._find_best_action_match(skill, command)

            if action_plan:
                return self._execute_skill_action(action_plan, command)

        # Step 2: Fallback
        return {
            "status": "error",
            "message": f"No skill action found for: '{command}'",
            "suggestion": f"Available actions: {self.active_app.available_actions[:3]}",
            "debug": {
                "command_received": command,
                "skill_available": skill is not None,
                "skill_actions": list(skill.actions.keys()) if skill else []
            }
        }

    def _find_best_action_match(self, skill: AppSkill, command: str) -> Optional[List[ActionStep]]:
        """
        Intelligentes Action-Matching mit mehreren Strategien.

        Returns:
            ActionStep-Liste oder None
        """
        command_clean = command.lower().strip()

        # Strategie 1: Exakter Match
        if command_clean in skill.actions:
            return skill.actions[command_clean]["steps"]

        # Strategie 2: Normalisierung (space ↔ underscore)
        command_with_underscore = command_clean.replace(" ", "_")
        command_with_space = command_clean.replace("_", " ")

        for action_name, action_data in skill.actions.items():
            action_normalized = action_name.replace("_", " ")

            # Check alle Varianten
            if any([
                command_clean == action_name,
                command_with_underscore == action_name,
                command_with_space == action_normalized,
                command_clean == action_normalized
            ]):
                return action_data["steps"]

        # Strategie 3: Fuzzy Match (falls nichts anderes funktioniert)
        best_match = None
        best_score = 0

        for action_name, action_data in skill.actions.items():
            # Berechne Ähnlichkeits-Score
            score = self._similarity_score(command_clean, action_name)
            if score > best_score and score > 0.7:  # Mindestens 70% Ähnlichkeit
                best_score = score
                best_match = action_data["steps"]

        return best_match

    def _similarity_score(self, s1: str, s2: str) -> float:
        """
        Berechnet Ähnlichkeits-Score zwischen zwei Strings.

        Returns:
            0.0 - 1.0 (1.0 = identisch)
        """
        # Normalisierung
        s1_norm = s1.replace("_", " ").replace("-", " ")
        s2_norm = s2.replace("_", " ").replace("-", " ")

        # Exact Match
        if s1_norm == s2_norm:
            return 1.0

        # Substring Match
        if s1_norm in s2_norm or s2_norm in s1_norm:
            shorter = min(len(s1_norm), len(s2_norm))
            longer = max(len(s1_norm), len(s2_norm))
            return shorter / longer

        # Levenshtein-basiert (vereinfacht)
        distance = self._levenshtein(s1_norm, s2_norm)
        max_len = max(len(s1_norm), len(s2_norm))
        return 1 - (distance / max_len)

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Berechnet Levenshtein-Distanz"""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _execute_skill_action(self, action_plan: List[ActionStep], original_command: str) -> Dict:
        """Führt Skill-Action aus"""
        executed_steps = []

        for step in action_plan:
            if step.action_type == "hotkey":
                keys = step.value.split('+')
                pyautogui.hotkey(*keys)
                executed_steps.append(f"Hotkey: {step.value}")

            elif step.action_type == "type":
                # Variable Replacement
                text = step.value
                if "{" in text:
                    var_match = re.search(r'["\'](.+?)["\']', original_command)
                    if var_match:
                        text = text.replace("{url}", var_match.group(1))
                        text = text.replace("{query}", var_match.group(1))
                        text = text.replace("{message}", var_match.group(1))

                pyautogui.write(text, interval=0.05)
                executed_steps.append(f"Typed: {text}")

            elif step.action_type == "wait":
                time.sleep(step.wait_ms / 1000.0)

            time.sleep(0.1)

        return {
            "status": "success",
            "action_taken": original_command,
            "executed_steps": executed_steps,
            "message": f"✓ Completed: {original_command}"
        }


# ============================================================================
# TOOL REGISTRATION
# ============================================================================

def register_enhanced_tools(user_level: UserLevel | str = UserLevel.BASIC):
    """Registriert Enhanced Tools"""

    if isinstance(user_level, str):
        user_level = UserLevel[user_level]

    toolkit = EnhancedDesktopAutomation(user_level)

    tools = [
        {
            "tool_func": toolkit.scout_interface,
            "name": "scout_interface",
            "flags": {"no_thread": True},
            "category": ["desktop", "overview", "context"],
            "description": """
Scannt Desktop und gibt umfassende Übersicht:

RETURNS:
- open_applications: Liste aller offenen Apps (Name, Typ, Skill-Status)
- active_application: Details zur aktiven App
- possible_actions:
  * switch_to: Apps zu denen gewechselt werden kann
  * interact: Verfügbare Skill-Actions
- preview: Fokussierte Mini-Ansicht der wichtigsten Infos

PARAMETERS:
- app_name: Wechselt zu dieser App (auto_switch=True)
- window_title: Alternative Auswahl via Titel
- auto_switch: Aktiviert App automatisch (default: True)

EXAMPLE CALLS:
scout_interface()                          # Übersicht aller Apps
scout_interface(app_name="chrome")         # Wechsle zu Chrome
scout_interface(window_title="Gmail")      # Wechsle zu Gmail-Tab
            """.strip(),
            "parameters": {
                "app_name": {"type": "string", "required": False},
                "window_title": {"type": "string", "required": False},
                "auto_switch": {"type": "boolean", "default": True}
            }
        },
        {
            "tool_func": toolkit.execute_action,
            "name": "execute_action",
            "flags": {"no_thread": True},
            "category": ["desktop", "action", "automation"],
            "description": """
Führt Aktion in aktiver App aus.

SKILL-BASIERT: Nutzt vordefinierte Actions wenn verfügbar.

EXAMPLES:
- execute_action("new tab")           # Browser: Neuer Tab
- execute_action("search python")     # Browser: Suche
- execute_action("play pause")        # Spotify: Toggle
- execute_action("send message hello")# WhatsApp: Nachricht

Verfügbare Actions siehe scout_interface['preview']['key_actions']
            """.strip(),
            "parameters": {
                "command": {"type": "string", "required": True}
            }
        }
    ]

    return toolkit, tools


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    toolkit, tools = register_enhanced_tools(UserLevel.POWER)

    print("=" * 80)
    print("ENHANCED DESKTOP AUTOMATION TOOLKIT")
    print("=" * 80)

    # Skill-Übersicht
    skills = toolkit.skill_registry.get_all_registered_skills()
    print("\n📦 REGISTERED SKILLS:")
    print(f"  Applications: {len(skills['applications'])}")
    for app in sorted(skills['applications']):
        print(f"    - {app}")
    print(f"\n  Web Pages: {len(skills['web_pages'])}")
    for page in sorted(skills['web_pages']):
        print(f"    - {page}")

    # Scout Demo
    print("\n" + "=" * 80)
    print("DEMO: Scout Interface")
    print("=" * 80)

    result = toolkit.scout_interface()
    print(f"\nStatus: {result['status']}")
    print(f"\nOpen Applications ({len(result['open_applications'])}):")
    for app in result['open_applications']:
        skill_marker = "✓" if app['has_skill'] else "✗"
        print(f"  {skill_marker} {app['name']:15} - {app['title'][:40]}")

    if result.get('active_application'):
        print(f"\nActive: {result['active_application']['name']}")
        print(f"Preview: {result['preview']}")

    print("\n" + "=" * 80)
    result = toolkit.scout_interface()  # Übersicht aller Apps
    print(f"\nStatus: {result['status']}")
    print(f"\nOpen Applications ({len(result['open_applications'])}):")
    for app in result['open_applications']:
        skill_marker = "✓" if app['has_skill'] else "✗"
        print(f"  {skill_marker} {app['name']:15} - {app['title'][:40]}")
    print(result)
    print("\n" + "=" * 80)
    result = toolkit.scout_interface(app_name="chrome")
    print(f"\nStatus: {result['status']}")
    print(result)
    print("\n" + "=" * 80)

    res = toolkit.execute_action("new tab")
    print(res)
