import binascii
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

from toolboxv2 import FileHandler, MainTool, Style, get_app
from toolboxv2.utils.extras.registry_client import RegistryClient
from toolboxv2.utils.system.state_system import find_highest_zip_version
from .UserInstances import UserInstances

Name = 'CloudM'
version = "0.0.5"  # Bumped for Registry integration
export = get_app(f"{Name}.EXPORT").tb
no_test = export(mod_name=Name, test=False, version=version)
to_api = export(mod_name=Name, api=True, version=version)


class Tools(MainTool, FileHandler):
    version = version

    # Default registry URL
    DEFAULT_REGISTRY_URL = "https://registry.simplecore.app"

    def __init__(self, app=None):
        t0 = time.perf_counter()
        self.version = version
        self.api_version = "404"
        self.name = "CloudM"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "CYAN"
        if app is None:
            app = get_app()
        self.user_instances = UserInstances()

        # Registry client instance (lazy initialized)
        self._registry_client: Optional[RegistryClient] = None
        self._registry_authenticated: bool = False

        self.keys = {
            "URL": "comm-vcd~~",
            "URLS": "comm-vcds~",
            "TOKEN": "comm-tok~~",
            "REGISTRY_URL": "registry-url~",
        }
        self.tools = {
            "all": [
                ["Version", "Shows current Version"],
                ["api_Version", "Shows current Version"],
                ["registry", "Access TB Registry for package management"],
            ],
            "name": "cloudM",
            "Version": self.get_version,
            "show_version": self.s_version,
            "get_mod_snapshot": self.get_mod_snapshot,
        }

        self.logger.info("init FileHandler cloudM")
        t1 = time.perf_counter()
        FileHandler.__init__(self, "modules.config", app.id if app else __name__,
                             self.keys, {
                                 "URL": '"https://simpelm.com/api"',
                                 "TOKEN": '"~tok~"',
                                 "REGISTRY_URL": f'"{self.DEFAULT_REGISTRY_URL}"',
                             })
        self.logger.info(f"Time to initialize FileHandler {time.perf_counter() - t1}")
        t1 = time.perf_counter()
        self.logger.info("init MainTool cloudM")
        MainTool.__init__(self,
                          load=self.load_open_file,
                          v=self.version,
                          tool=self.tools,
                          name=self.name,
                          logs=self.logger,
                          color=self.color,
                          on_exit=self.on_exit)

        self.logger.info(f"Time to initialize MainTool {time.perf_counter() - t1}")
        self.logger.info(
            f"Time to initialize Tools {self.name} {time.perf_counter() - t0}")

    async def load_open_file(self):
        self.logger.info("Starting cloudM")
        self.load_file_handler()
        from toolboxv2.mods.Minu.examples import initialize
        initialize(self.app)
        await self.app.session.login()

    def s_version(self):
        return self.version

    def on_exit(self):
        self.save_file_handler()

    def get_version(self):  # Add root and upper and controll comander pettern
        version_command = self.app.config_fh.get_file_handler("provider::")

        url = version_command + "/api/Cloudm/show_version"

        try:
            self.api_version = requests.get(url, timeout=5).json()["res"]
            self.print(f"API-Version: {self.api_version}")
        except Exception as e:
            self.logger.error(Style.YELLOW(str(e)))
            self.print(
                Style.RED(
                    f" Error retrieving version from {url}\n\t run : cloudM first-web-connection\n"
                ))
            self.logger.error(f"Error retrieving version from {url}")
        return self.version

    def get_mod_snapshot(self, mod_name):
        if mod_name is None:
            return None
        self.print("")
        return find_highest_zip_version(mod_name, version_only=True)

    # =================== Registry Integration ===================

    def get_registry_url(self) -> str:
        """Get the configured registry URL."""
        try:
            url = self.get_file_handler("REGISTRY_URL")
            if url and url != "~":
                return url.strip('"')
        except Exception:
            pass
        return self.DEFAULT_REGISTRY_URL

    @property
    def registry(self) -> RegistryClient:
        """
        Get the Registry Client instance.

        The client is lazily initialized on first access.
        Uses the configured registry URL and caches in .tb-registry/cache.

        Returns:
            RegistryClient instance
        """
        if self._registry_client is None:
            cache_dir = Path(self.app.start_dir) / ".tb-registry" / "cache"
            self._registry_client = RegistryClient(
                registry_url=self.get_registry_url(),
                cache_dir=cache_dir,
            )
            self.logger.info(f"Initialized RegistryClient for {self.get_registry_url()}")
        return self._registry_client

    async def ensure_registry_auth(self) -> bool:
        """
        Ensure the registry client is authenticated.

        Attempts to authenticate using the session's Clerk token if available.
        This enables access to unlisted packages and publishing.

        Returns:
            True if authenticated, False otherwise
        """
        if self._registry_authenticated:
            return True

        # Check if already authenticated
        if await self.registry.is_authenticated():
            self._registry_authenticated = True
            return True

        # Try to get Clerk token from session
        if hasattr(self.app, 'session') and self.app.session.valid:
            try:
                # Get Clerk token from session
                clerk_token = await self._get_clerk_token_from_session()
                if clerk_token:
                    if await self.registry.login(clerk_token):
                        self._registry_authenticated = True
                        self.logger.info("Registry authenticated via session token")
                        return True
            except Exception as e:
                self.logger.warning(f"Failed to authenticate registry: {e}")

        return False

    async def _get_clerk_token_from_session(self) -> Optional[str]:
        """
        Get Clerk JWT token from the current session.

        Returns:
            Clerk JWT token or None if not available
        """
        if not hasattr(self.app, 'session') or not self.app.session.valid:
            return None

        # Try to get token from session's auth_clerk
        if hasattr(self.app.session, 'auth_clerk') and self.app.session.auth_clerk:
            auth_clerk = self.app.session.auth_clerk
            if hasattr(auth_clerk, 'get_token'):
                try:
                    token = await auth_clerk.get_token()
                    return token
                except Exception as e:
                    self.logger.debug(f"Failed to get Clerk token: {e}")

        return None

    def invalidate_registry_auth(self):
        """Invalidate the registry authentication state."""
        self._registry_authenticated = False
        if self._registry_client:
            self._registry_client.auth_token = None


# Create a hashed password
def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt,
                                  100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')


# Check hashed password validity
def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'),
                                  salt.encode('ascii'), 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password
