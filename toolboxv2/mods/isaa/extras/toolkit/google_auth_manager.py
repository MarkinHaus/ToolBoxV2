"""
Google Auth Manager — Multi-Account, Cross-Platform Credential Manager.

Ersetzt session_id-basierte Token-Files mit account_id-basierten.
Unterstützt: Desktop (Browser), Server (Device Flow), Mobile (Device Flow).

Usage:
    mgr = GoogleAuthManager()
    creds = mgr.ensure_authenticated("privat", GmailToolkit.SCOPES)

CLI:
    python -m toolboxv2.mods.isaa.extras.toolkit.google_auth_manager login <account_id>
    python -m toolboxv2.mods.isaa.extras.toolkit.google_auth_manager list
    python -m toolboxv2.mods.isaa.extras.toolkit.google_auth_manager status <account_id>
"""

import json
import logging
import os
import sys
import webbrowser
from pathlib import Path

from toolboxv2 import get_logger, get_app

logger = get_logger()

# Default paths — cross-platform via pathlib
DEFAULT_DATA_DIR = Path(get_app().appdata)
DEFAULT_TOKEN_DIR = DEFAULT_DATA_DIR / "tokens"
DEFAULT_CONFIG_FILE = DEFAULT_DATA_DIR / "google_accounts.json"
DEFAULT_CREDS_PATH = str(DEFAULT_DATA_DIR / "google_credentials.json")


# Unified scopes: ein Login für Gmail + Calendar + Tasks
ALL_GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]


class GoogleAuthManager:
    """Centralized Multi-Account Manager for Google Services.

    Handles:
    - Multiple accounts (privat, arbeit, bot, ...)
    - Persistent token storage keyed by account_id (NOT session_id)
    - Auto-refresh of expired tokens
    - Auto-login trigger when token missing
    - Platform detection: Desktop (Browser) vs Server (Device/Manual Flow)
    - Unified scopes: one token works for Gmail + Calendar + Tasks
    """

    def __init__(
        self,
        credentials_path: str | None = None,
        token_dir: str | None = None,
        config_file: str | None = None,
    ):
        self.credentials_path = (
            credentials_path
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or DEFAULT_CREDS_PATH
        )
        self.token_dir = Path(token_dir or os.getenv("GOOGLE_TOKEN_DIR", "") or DEFAULT_TOKEN_DIR)
        self.config_file = Path(config_file or DEFAULT_CONFIG_FILE)

        # Ensure directories exist
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Init config if missing
        if not self.config_file.exists():
            self._save_config({"accounts": {}, "default_account": None})

    # ── Config Management ──────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            return json.loads(self.config_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"accounts": {}, "default_account": None}

    def _save_config(self, config: dict):
        self.config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def list_accounts(self) -> list[str]:
        config = self._load_config()
        return list(config.get("accounts", {}).keys())

    def get_account_info(self, account_id: str) -> dict | None:
        config = self._load_config()
        return config.get("accounts", {}).get(account_id)

    # ── Token Storage ──────────────────────────────────────────────

    def _token_path(self, account_id: str) -> Path:
        return self.token_dir / f"{account_id}.json"

    def get_credentials(self, account_id: str, scopes: list[str]):
        """Load credentials for account. Returns Credentials or None.

        Auto-refreshes expired tokens.
        """
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        token_path = self._token_path(account_id)
        if not token_path.exists():
            return None

        try:
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self._save_token(account_id, creds)
                logger.info("Refreshed token for account '%s'", account_id)
            return creds
        except Exception as e:
            logger.warning("Failed to load credentials for '%s': %s", account_id, e)
            return None

    def _save_token(self, account_id: str, creds):
        token_path = self._token_path(account_id)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    def is_authenticated(self, account_id: str) -> bool:
        return self._token_path(account_id).exists()

    # ── Login Flow ─────────────────────────────────────────────────

    def login(self, account_id: str, scopes: list[str] | None = None) -> str:
        """Login a Google account. Auto-detects platform.

        Desktop → Browser opens automatically.
        Server  → Prints URL, user pastes code back.

        Returns status message string.
        """
        if scopes is None:
            scopes = ALL_GOOGLE_SCOPES

        # Already authenticated?
        existing = self.get_credentials(account_id, scopes)
        if existing and existing.valid:
            return f"✅ Account '{account_id}' already authenticated."

        if not Path(self.credentials_path).exists():
            return (
                f"❌ Credentials file not found at {self.credentials_path}\n"
                f"   Download client_secret.json from Google Cloud Console\n"
                f"   and save as {DEFAULT_CREDS_PATH}\n"
                f"   or set GOOGLE_APPLICATION_CREDENTIALS env var."
            )

        if self._is_headless():
            return self._login_headless(account_id, scopes)
        else:
            return self._login_desktop(account_id, scopes)

    def _is_headless(self) -> bool:
        """Detect headless environment (no display / no interactive session)."""
        # Windows: never headless (always has GUI)
        if os.name == "nt":
            return False
        # Linux/Mac: check DISPLAY and terminal
        if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
            return True
        # Check if stdin is a TTY (non-interactive = headless)
        try:
            return not sys.stdin.isatty()
        except Exception:
            return True

    def _login_desktop(self, account_id: str, scopes: list[str]) -> str:
        """Desktop login: opens local browser, auto-captures callback."""
        from google_auth_oauthlib.flow import InstalledAppFlow

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_path, scopes=scopes
            )
            creds = flow.run_local_server(port=0, open_browser=True)
            self._save_token(account_id, creds)
            self._update_config(account_id, scopes, status="authenticated")
            return f"✅ Account '{account_id}' authenticated via browser."
        except Exception as e:
            # Fallback to headless if local server fails (firewall, etc.)
            logger.warning("Desktop login failed (%s), trying manual flow", e)
            return self._login_headless(account_id, scopes)

    def _login_headless(self, account_id: str, scopes: list[str]) -> str:
        """Server/Mobile login: prints URL, user pastes code.

        Uses OOB redirect — works everywhere (no port needed).
        """
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=scopes,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
        auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")

        print(f"\n{'='*60}")
        print(f"  Google Login für Account '{account_id}'")
        print(f"{'='*60}")
        print(f"\n  1. Öffne diese URL im Browser:\n")
        print(f"  {auth_url}\n")
        print(f"  2. Erlaube den Zugriff")
        print(f"  3. Kopiere den Autorisierungs-Code\n")

        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        code = input("  Code hier einfügen: ").strip()
        if not code:
            return "❌ No authorization code provided."

        try:
            flow.fetch_token(code=code)
            self._save_token(account_id, flow.credentials)
            self._update_config(account_id, scopes, status="authenticated")
            return f"✅ Account '{account_id}' authenticated successfully."
        except Exception as e:
            return f"❌ Auth failed: {e}"

    def _update_config(self, account_id: str, scopes: list[str], status: str = "authenticated"):
        config = self._load_config()
        config.setdefault("accounts", {})[account_id] = {
            "scopes": scopes,
            "status": status,
        }
        if not config.get("default_account"):
            config["default_account"] = account_id
        self._save_config(config)

    # ── Auto-Login (called by toolkits) ────────────────────────────

    def ensure_authenticated(self, account_id: str, scopes: list[str]):
        """Ensure credentials exist for account. Auto-login if missing.

        This is the main entry point for toolkits.
        Returns valid Credentials object.
        """
        creds = self.get_credentials(account_id, scopes)
        if creds and creds.valid:
            return creds

        # Auto-login trigger
        logger.info("Account '%s' not authenticated. Starting login flow...", account_id)
        result = self.login(account_id, scopes)
        logger.info("Login result: %s", result)

        creds = self.get_credentials(account_id, scopes)
        if creds and creds.valid:
            return creds

        raise RuntimeError(
            f"Google account '{account_id}' authentication failed. "
            f"Run: python -m toolboxv2.mods.isaa.extras.toolkit.google_auth_manager login {account_id}"
        )

    def get_auth_url(self, account_id: str, scopes: list[str] | None = None) -> str:
        """Generate OAuth URL for external/programmatic login (Discord, Web).

        Returns URL string. Pair with complete_auth().
        """
        if scopes is None:
            scopes = ALL_GOOGLE_SCOPES
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=scopes,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
        url, _ = flow.authorization_url(access_type="offline", prompt="consent")
        # Store flow for callback
        self._pending_flows = getattr(self, "_pending_flows", {})
        self._pending_flows[account_id] = flow
        return url

    def complete_auth(self, account_id: str, code: str, scopes: list[str] | None = None) -> str:
        """Complete external OAuth login with authorization code."""
        if scopes is None:
            scopes = ALL_GOOGLE_SCOPES
        self._pending_flows = getattr(self, "_pending_flows", {})
        flow = self._pending_flows.get(account_id)
        if flow is None:
            return "❌ No pending auth flow. Call get_auth_url first."
        try:
            flow.fetch_token(code=code)
            self._save_token(account_id, flow.credentials)
            self._update_config(account_id, scopes, status="authenticated")
            return f"✅ Account '{account_id}' authenticated successfully."
        except Exception as e:
           return f"❌ Auth failed: {e}"

    def _update_config(self, account_id: str, scopes: list[str], status: str = "authenticated"):
       config = self._load_config()
       config.setdefault("accounts", {})[account_id] = {
           "scopes": scopes,
           "status": status,
       }
       if not config.get("default_account"):
           config["default_account"] = account_id
       self._save_config(config)

    # ── Auto-Login (called by toolkits) ────────────────────────────

    def ensure_authenticated(self, account_id: str, scopes: list[str]):
       """Ensure credentials exist for account. Auto-login if missing.

       This is the main entry point for toolkits.
       Returns valid Credentials object.
       """
       creds = self.get_credentials(account_id, scopes)
       if creds and creds.valid:
           return creds

       # Auto-login trigger
       logger.info("Account '%s' not authenticated. Starting login flow...", account_id)
       result = self.login(account_id, scopes)
       logger.info("Login result: %s", result)

       creds = self.get_credentials(account_id, scopes)
       if creds and creds.valid:
           return creds

       raise RuntimeError(
           f"Google account '{account_id}' authentication failed. "
           f"Run: python -m toolboxv2.mods.isaa.extras.toolkit.google_auth_manager login {account_id}"
       )

    def get_auth_url(self, account_id: str, scopes: list[str] | None = None) -> str:
       """Generate OAuth URL for external/programmatic login (Discord, Web).

       Returns URL string. Pair with complete_auth().
       """
       if scopes is None:
           scopes = ALL_GOOGLE_SCOPES
       from google_auth_oauthlib.flow import Flow

       flow = Flow.from_client_secrets_file(
           self.credentials_path,
           scopes=scopes,
           redirect_uri="urn:ietf:wg:oauth:2.0:oob",
       )
       url, _ = flow.authorization_url(access_type="offline", prompt="consent")
       # Store flow for callback
       self._pending_flows = getattr(self, "_pending_flows", {})
       self._pending_flows[account_id] = flow
       return url

    def complete_auth(self, account_id: str, code: str, scopes: list[str] | None = None) -> str:
       """Complete external OAuth login with authorization code."""
       if scopes is None:
           scopes = ALL_GOOGLE_SCOPES
       self._pending_flows = getattr(self, "_pending_flows", {})
       flow = self._pending_flows.get(account_id)
       if flow is None:
           return "❌ No pending auth flow. Call get_auth_url first."
       try:
           flow.fetch_token(code=code)
           self._save_token(account_id, flow.credentials)
           self._update_config(account_id, scopes, status="authenticated")
           del self._pending_flows[account_id]
           return f"✅ Account '{account_id}' authenticated."
       except Exception as e:
           return f"❌ Auth failed: {e}"


# ── CLI Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
   mgr = GoogleAuthManager()
   cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

   if cmd == "login":
       if len(sys.argv) < 3:
           print("Usage: python -m ...google_auth_manager login <account_id>")
           sys.exit(1)
       account = sys.argv[2]
       print(mgr.login(account))
   elif cmd == "list":
       accounts = mgr.list_accounts()
       if not accounts:
           print("No accounts registered.")
       else:
           for acc in accounts:
               info = mgr.get_account_info(acc)
               auth_status = "✅" if mgr.is_authenticated(acc) else "❌"
               print(f"  {auth_status} {acc} — {info.get('status', 'unknown')}")
   elif cmd == "status":
       if len(sys.argv) < 3:
           print("Usage: python -m ...google_auth_manager status <account_id>")
           sys.exit(1)
       account = sys.argv[2]
       print(f"Account '{account}': {'authenticated' if mgr.is_authenticated(account) else 'not authenticated'}")
   else:
       print("Commands: login <id>, list, status <id>")
