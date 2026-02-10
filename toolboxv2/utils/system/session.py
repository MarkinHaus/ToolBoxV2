"""
ToolBox V2 - Session Management
Handles CLI and API sessions with Custom Auth (CloudM.Auth)
"""

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Optional

import requests
from aiohttp import ClientResponse, ClientSession, MultipartWriter
from aiohttp import ClientConnectorError, ClientError
from requests import Response

from toolboxv2.utils.singelton_class import Singleton
from toolboxv2.utils.system.getting_and_closing_app import get_app, get_logger
from toolboxv2.utils.system.types import Result


class RequestSession:
    """Wrapper for request session data"""

    def __init__(self, session, body, json_data, row):
        super().__init__()
        self.session = session
        self._body = body
        self._json = json_data
        self.row = row

    def body(self):
        return self._body

    def json(self):
        if isinstance(self._json, dict):
            return self._json
        return self._json()


class Session(metaclass=Singleton):
    """
    Session manager for ToolBox V2 with Custom Auth (CloudM.Auth).
    Handles JWT tokens and API communication.
    Token storage: local JSON file in user config dir (CLI has no DB access).
    """

    def __init__(self, username=None, base=None):
        self.username = username
        self._session: Optional[ClientSession] = None
        self._event_loop = None
        self.valid = False
        self.user_id: Optional[str] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

        # Backwards compat aliases
        self.clerk_user_id: Optional[str] = None
        self.clerk_session_token: Optional[str] = None

        # Set base URL
        if base is None:
            base = os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app")
        if base is not None and base.endswith("/api/"):
            base = base.replace("api/", "")
        self.base = base.rstrip('/')

    @property
    def session(self):
        self._ensure_session()
        return self._session

    def _ensure_session(self):
        """Ensure session is valid for current event loop"""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._session is not None:
                self._session = None
                self._event_loop = None
            return

        if self._session is None or self._event_loop != current_loop:
            if self._session is not None:
                try:
                    if not self._session.closed:
                        asyncio.create_task(self._session.close())
                except:
                    pass
            self._session = ClientSession()
            self._event_loop = current_loop

    # =================== Token Storage (local JSON file) ===================

    def _get_token_dir(self) -> Path:
        """Get directory for CLI session storage."""
        config_dir = Path(os.environ.get("TB_DATA_DIR", ".")) / ".data" / "cli_sessions"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _get_token_path(self) -> Path:
        """Get file path for session token."""
        safe_name = self.username or "default"
        # Simple safe filename (no crypto dependency needed)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)[:32]
        return self._get_token_dir() / f"{safe_name}_session.json"

    def _save_session_token(self, access_token: str, refresh_token: str = "", user_id: str = None):
        """Save session tokens to local JSON file."""
        try:
            path = self._get_token_path()
            session_data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_id": user_id or self.user_id or "",
                "username": self.username or "",
            }
            path.write_text(json.dumps(session_data, indent=2))
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.user_id = user_id
            # Backwards compat
            self.clerk_session_token = access_token
            self.clerk_user_id = user_id
            return True
        except Exception as e:
            get_logger().error(f"Failed to save session token: {e}")
            return False

    def _load_session_token(self) -> Optional[dict]:
        """Load session tokens from local JSON file."""
        try:
            path = self._get_token_path()
            if not path.exists():
                return None
            session_data = json.loads(path.read_text())
            self.access_token = session_data.get("access_token")
            self.refresh_token = session_data.get("refresh_token")
            self.user_id = session_data.get("user_id")
            # Backwards compat
            self.clerk_session_token = self.access_token
            self.clerk_user_id = self.user_id
            return session_data
        except Exception as e:
            get_logger().debug(f"No session token found: {e}")
        return None

    def _clear_session_token(self):
        """Clear session token file."""
        try:
            path = self._get_token_path()
            if path.exists():
                path.unlink()
            self.access_token = None
            self.refresh_token = None
            self.user_id = None
            self.clerk_session_token = None
            self.clerk_user_id = None
            return True
        except:
            return False

    # =================== Authentication ===================

    async def login(self, verbose=False) -> bool:
        """
        Login using stored JWT access token.
        Returns True if session is valid.
        """
        self._ensure_session()

        # Try to load existing session
        session_data = self._load_session_token()

        if not session_data or not session_data.get("access_token"):
            if verbose:
                print("No stored session token. Please run 'tb login' first.")
            return False

        token = session_data.get("access_token")

        try:
            # Verify session with backend
            async with self.session.request(
                "POST",
                url=f"{self.base}/api/CloudM.Auth/validate_session",
                json={"token": token}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("result", {}).get("authenticated"):
                        get_logger().info("Session validated successfully")
                        self.valid = True
                        self.username = session_data.get("username") or result.get("result", {}).get("username")
                        return True

                # Try token refresh
                refresh = session_data.get("refresh_token")
                if refresh:
                    refreshed = await self._try_refresh(refresh)
                    if refreshed:
                        return True

                # Session invalid
                get_logger().warning("Session validation failed")
                self._clear_session_token()
                self.valid = False
                return False

        except ClientConnectorError as e:
            if verbose:
                print(f"Server not reachable: {e}")
            return False
        except Exception as e:
            if verbose:
                print(f"Connection error: {e}")
            return False

    async def _try_refresh(self, refresh_token: str) -> bool:
        """Try to refresh the access token."""
        try:
            async with self.session.request(
                "POST",
                url=f"{self.base}/api/CloudM.Auth/refresh_token",
                json={"refresh_token": refresh_token}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("result", {})
                    new_access = data.get("access_token")
                    new_refresh = data.get("refresh_token", refresh_token)
                    if new_access:
                        self._save_session_token(new_access, new_refresh, self.user_id)
                        self.valid = True
                        return True
        except Exception as e:
            get_logger().debug(f"Token refresh failed: {e}")
        return False

    async def login_with_magic_link(self, email: str) -> Result:
        """
        Request a magic link for CLI login.
        User receives email, clicks link, pastes token back.
        """
        self._ensure_session()

        try:
            async with self.session.request(
                "POST",
                url=f"{self.base}/api/CloudM.Auth/request_magic_link",
                json={"email": email}
            ) as response:
                if response.status != 200:
                    return Result.default_user_error("Failed to request magic link")

                result = await response.json()
                if result.get("error", 0) != 0:
                    return Result.default_user_error(
                        result.get("info", {}).get("help_text", "Unknown error")
                    )

                return Result.ok("Magic link sent", data=result.get("result", {}))

        except Exception as e:
            get_logger().error(f"Magic link request error: {e}")
            return Result.default_internal_error(str(e))

    async def verify_magic_link(self, token: str) -> Result:
        """Verify a magic link token and complete login."""
        self._ensure_session()

        try:
            async with self.session.request(
                "POST",
                url=f"{self.base}/api/CloudM.Auth/verify_magic_link",
                json={"token": token}
            ) as response:
                if response.status != 200:
                    return Result.default_user_error("Verification failed")

                result = await response.json()
                if result.get("error", 0) != 0:
                    return Result.default_user_error(
                        result.get("info", {}).get("help_text", "Invalid or expired link")
                    )

                data = result.get("result", {})
                self._save_session_token(
                    data.get("access_token", ""),
                    data.get("refresh_token", ""),
                    data.get("user_id", ""),
                )
                self.username = data.get("username", "")
                self.valid = True
                return Result.ok("Login successful", data=data)

        except Exception as e:
            get_logger().error(f"Magic link verify error: {e}")
            return Result.default_internal_error(str(e))

    async def login_with_invite_code(self, code: str) -> Result:
        """Login with a device invite code."""
        self._ensure_session()

        try:
            async with self.session.request(
                "POST",
                url=f"{self.base}/api/CloudM.Auth/verify_device_invite",
                json={"code": code}
            ) as response:
                if response.status != 200:
                    return Result.default_user_error("Verification failed")

                result = await response.json()
                if result.get("error", 0) != 0:
                    return Result.default_user_error(
                        result.get("info", {}).get("help_text", "Invalid or expired code")
                    )

                data = result.get("result", {})
                self._save_session_token(
                    data.get("access_token", ""),
                    data.get("refresh_token", ""),
                    data.get("user_id", ""),
                )
                self.username = data.get("username", "")
                self.valid = True
                return Result.ok("Login successful", data=data)

        except Exception as e:
            get_logger().error(f"Invite code login error: {e}")
            return Result.default_internal_error(str(e))

    # Backwards compat alias
    async def login_with_code(self, email: str, code: str) -> Result:
        """Legacy alias: tries magic link flow (request + verify)."""
        req = await self.login_with_magic_link(email)
        if req.is_error():
            return req
        return await self.verify_magic_link(code)

    async def logout(self) -> bool:
        """Logout and clear session."""
        self._ensure_session()

        # Notify server (blacklist token)
        if self.session and not self.session.closed and self.access_token:
            try:
                headers = self._get_auth_headers()
                await self.session.post(
                    f'{self.base}/api/CloudM.Auth/logout',
                    json={"token": self.access_token},
                    headers=headers,
                )
            except:
                pass

        # Clear local session
        self._clear_session_token()
        self.valid = False
        self.username = None

        # Close HTTP session
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except:
                pass
            self._session = None
            self._event_loop = None

        return True

    def init(self):
        """Initialize session (legacy compatibility)."""
        self._ensure_session()

    def set_token(self, token: str):
        """Set session token (for web login callback)."""
        self._save_session_token(token)

    # =================== HTTP Methods ===================

    def _get_auth_headers(self) -> dict:
        """Get authentication headers for API requests."""
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif self.clerk_session_token:
            # Backwards compat fallback
            headers["Authorization"] = f"Bearer {self.clerk_session_token}"
        return headers

    async def fetch(
        self,
        url: str,
        method: str = 'GET',
        data=None,
        json=None,
        **kwargs
    ) -> bool | ClientResponse | Response:
        """Fetch URL with authentication"""
        self._ensure_session()

        if isinstance(url, str) and not url.startswith(('http://', 'https://')):
            url = self.base + url

        data = json or data
        # Add auth headers
        headers = kwargs.pop('headers', {})
        headers.update(self._get_auth_headers())

        if self.session:
            try:
                if method.upper() == 'POST':
                    return await self.session.post(url, json=data, headers=headers, **kwargs)
                else:
                    return await self.session.get(url, headers=headers, **kwargs)
            except ClientConnectorError as e:
                print(f"Server not reachable: {e}")
                return False
            except ClientError as e:
                print(f"Client error: {e}")
                return False
            except Exception as e:
                print(f"Error: {e}")
                return requests.request(method, url, json=data if method.upper() == 'POST' else None, headers=headers)
        else:
            return requests.request(
                method,
                url,
                json=data if method.upper() == 'POST' else None,
                headers=headers
            )

    async def download_file(self, url: str, dest_folder: str = "mods_sto") -> bool:
        """Download file from URL"""
        self._ensure_session()

        if not self.session:
            raise Exception("Session not initialized")

        os.makedirs(dest_folder, exist_ok=True)

        filename = url.split('/')[-1]
        valid_chars = '-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        filename = ''.join(char for char in filename if char in valid_chars)
        file_path = os.path.join(dest_folder, filename)

        if isinstance(url, str) and not url.startswith(('http://', 'https://')):
            url = self.base + url

        headers = self._get_auth_headers()

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    with open(file_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    print(f'File downloaded: {file_path}')
                    return True
                else:
                    print(f'Failed to download: {url} (Status: {response.status})')
        except Exception as e:
            print(f"Download error: {e}")
        return False

    async def upload_file(self, file_path: str, upload_url: str):
        """Upload file to URL"""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self._ensure_session()

        upload_url = self.base + upload_url
        headers = self._get_auth_headers()

        with open(file_path, 'rb') as f:
            file_data = f.read()

        with MultipartWriter('form-data') as mpwriter:
            part = mpwriter.append(file_data)
            part.set_content_disposition('form-data', name='file', filename=os.path.basename(file_path))

            try:
                async with self.session.post(upload_url, data=mpwriter, headers=headers, timeout=20000) as response:
                    if response.status == 200:
                        print(f"File uploaded: {file_path}")
                        return await response.json()
                    else:
                        print(f"Upload failed: {response.status}")
                        return None
            except Exception as e:
                print(f"Upload error: {e}")
                return None

    async def cleanup(self):
        """Cleanup session resources"""
        try:
            if self._session is not None and not self._session.closed:
                await self._session.close()
        except:
            pass
        finally:
            self._session = None
            self._event_loop = None

    def exit(self):
        """Exit and clear session (legacy compatibility)"""
        self._clear_session_token()


# =================== Utility Functions ===================

def get_public_ip() -> Optional[str]:
    """Get public IP address"""
    try:
        response = requests.get('https://api.ipify.org?format=json')
        return response.json()['ip']
    except Exception as e:
        print(f"Error getting public IP: {e}")
        return None


def get_local_ip() -> Optional[str]:
    """Get local IP address"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return None


# =================== Tests ===================

async def _test_session_login():
    """Test session login (requires valid session)"""
    s = Session('test')
    result = await s.login(verbose=True)
    print(f"Login result: {result}")
    return result


def test_session():
    """Run session tests"""
    asyncio.run(_test_session_login())
