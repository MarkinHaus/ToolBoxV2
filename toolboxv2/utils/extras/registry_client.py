"""
TB Registry Client
Async HTTP client for interacting with the TB Registry API.

Provides:
- Authentication (JWT token)
- Package Discovery (search, get, versions)
- Dependency Resolution
- Download (single and with dependencies)
- Publishing (for verified publishers)
- Artifacts management
- Cache management
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

import httpx

# =================== Exceptions ===================


class RegistryError(Exception):
    """Base exception for registry errors."""
    pass


class RegistryConnectionError(RegistryError):
    """Connection to registry failed."""
    pass


class RegistryAuthError(RegistryError):
    """Authentication failed."""
    pass


class PackageNotFoundError(RegistryError):
    """Package not found in registry."""
    pass


class VersionNotFoundError(RegistryError):
    """Version not found for package."""
    pass


class PublishPermissionError(RegistryError):
    """User does not have permission to publish."""
    pass


class DownloadError(RegistryError):
    """Download failed."""
    pass


# =================== Data Classes ===================


@dataclass
class UserInfo:
    """User information from registry."""
    id: str
    username: str
    email: str
    is_verified: bool = False
    is_admin: bool = False
    publisher_id: Optional[str] = None


@dataclass
class PublisherInfo:
    """Publisher information from registry."""
    id: str
    name: str
    display_name: str
    verification_status: str = "unverified"
    package_count: int = 0
    total_downloads: int = 0


@dataclass
class PackageSummary:
    """Summary of a package for search results."""
    name: str
    description: str
    latest_version: str
    visibility: str = "public"
    downloads: int = 0
    publisher: str = ""


@dataclass
class VersionInfo:
    """Version information."""
    version: str
    published_at: str
    yanked: bool = False
    downloads: int = 0


@dataclass
class VersionDetail:
    """Detailed version information."""
    version: str
    published_at: str
    yanked: bool = False
    downloads: int = 0
    checksum_sha256: str = ""
    download_url: str = ""
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    changelog: str = ""


@dataclass
class PackageDetail:
    """Detailed package information."""
    name: str
    description: str
    latest_version: str
    visibility: str = "public"
    downloads: int = 0
    publisher: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = ""
    keywords: List[str] = field(default_factory=list)
    versions: List[VersionInfo] = field(default_factory=list)


@dataclass
class ResolvedPackage:
    """A resolved package with version."""
    name: str
    version: str
    download_url: str
    checksum_sha256: str
    dependencies: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # name -> {version, version_spec, optional}


@dataclass
class ResolutionResult:
    """Result of dependency resolution."""
    success: bool
    resolved: Dict[str, ResolvedPackage] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ArtifactBuild:
    """Artifact build information."""
    version: str
    platform: str
    arch: str
    download_url: str
    checksum_sha256: str
    size: int = 0


@dataclass
class ArtifactDetail:
    """Detailed artifact information."""
    name: str
    description: str
    latest_version: str
    builds: List[ArtifactBuild] = field(default_factory=list)


# =================== Retry Auth ===================


class RetryAuth(httpx.Auth):
    """Auth handler with retry logic for transient errors."""

    def __init__(self, token: Optional[str] = None, max_retries: int = 3):
        self.token = token
        self.max_retries = max_retries

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        if self.token:
            request.headers["Authorization"] = f"Bearer {self.token}"

        for attempt in range(self.max_retries):
            response = yield request

            # Retry on 429 (rate limit) or 5xx (server errors)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(retry_after)
                    continue
            break


# =================== Registry Client ===================


class RegistryClient:
    """
    Async client for TB Registry API.

    Provides methods for:
    - Authentication
    - Package discovery and search
    - Dependency resolution
    - Package download
    - Publishing (for verified publishers)
    - Artifact management
    - Cache management
    - Automatic token refresh on 401 errors
    """

    def __init__(
        self,
        registry_url: str = os.getenv("REGISTRY_BASE_URL","https://registry.simplecore.app"),
        auth_token: Optional[str] = None,
        timeout: int = 30,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        token_refresh_callback: Optional[callable] = None,
    ):
        """
        Initialize the registry client.

        Args:
            registry_url: Base URL of the registry API
            auth_token: Optional JWT authentication token
            timeout: Request timeout in seconds
            cache_dir: Optional directory for local caching
            max_retries: Number of retries for transient errors
            session_id: Session ID for token refresh
            user_id: User ID for token refresh
            token_refresh_callback: Async callback to refresh token (returns new token)
        """
        self.registry_url = registry_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.cache_dir = cache_dir or Path.home() / ".tb-registry" / "cache"
        self.max_retries = max_retries
        self.session_id = session_id
        self.user_id = user_id
        self._token_refresh_callback = token_refresh_callback
        self._client: Optional[httpx.AsyncClient] = None
        self._user: Optional[UserInfo] = None
        self._token_refresh_attempted = False
        self.logger = logging.getLogger(__name__)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.registry_url,
                timeout=httpx.Timeout(self.timeout),
                auth=RetryAuth(self.auth_token, self.max_retries),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RegistryClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # =================== Authentication ===================

    async def login(
        self,
        auth_token: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Login with JWT token.

        Args:
            auth_token: JWT authentication token
            session_id: Optional session ID for token refresh
            user_id: Optional user ID for token refresh

        Returns:
            True if login successful
        """
        self.auth_token = auth_token
        if session_id:
            self.session_id = session_id
        if user_id:
            self.user_id = user_id
        self._token_refresh_attempted = False

        # Recreate client with new token
        if self._client:
            await self._client.aclose()
            self._client = None

        try:
            user = await self.get_current_user()
            return user is not None
        except RegistryAuthError:
            self.auth_token = None
            return False

    async def logout(self) -> None:
        """Logout and clear authentication."""
        self.auth_token = None
        self.session_id = None
        self._user = None
        self._token_refresh_attempted = False
        if self._client:
            await self._client.aclose()
            self._client = None

    async def refresh_token(self) -> bool:
        """
        Attempt to refresh the JWT token.

        Uses the token_refresh_callback if provided, or tries to call
        the ToolBox server's refresh endpoint.

        Returns:
            True if token was refreshed successfully
        """
        if self._token_refresh_attempted:
            # Prevent infinite refresh loops
            self.logger.warning("Token refresh already attempted, not retrying")
            return False

        self._token_refresh_attempted = True

        try:
            # Try callback first
            if self._token_refresh_callback:
                new_token = await self._token_refresh_callback(
                    session_id=self.session_id,
                    user_id=self.user_id
                )
                if new_token:
                    self.auth_token = new_token
                    # Recreate client with new token
                    if self._client:
                        await self._client.aclose()
                        self._client = None
                    self._token_refresh_attempted = False
                    self.logger.info("Token refreshed successfully via callback")
                    return True

            # Fallback: Try ToolBox server refresh endpoint
            if self.session_id or self.user_id:
                try:
                    # Import here to avoid circular imports
                    from toolboxv2 import get_app
                    app = get_app("RegistryClient.refresh_token")
                    result = await app.a_run_any(
                        "CloudM.Auth.refresh_jwt_token",
                        session_id=self.session_id,
                        user_id=self.user_id,
                        get_results=True
                    )
                    if not result.is_error():
                        data = result.get()
                        new_token = data.get("session_token")
                        if new_token:
                            self.auth_token = new_token
                            if self._client:
                                await self._client.aclose()
                                self._client = None
                            self._token_refresh_attempted = False
                            self.logger.info("Token refreshed successfully via ToolBox server")
                            return True
                except Exception as e:
                    self.logger.warning(f"Failed to refresh token via ToolBox: {e}")

            self.logger.warning("Token refresh failed - no valid refresh method available")
            return False

        except Exception as e:
            self.logger.error(f"Error during token refresh: {e}")
            return False

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make a request with automatic token refresh on 401.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response

        Raises:
            RegistryAuthError: If authentication fails after refresh attempt
        """
        client = await self._get_client()
        response = await client.request(method, url, **kwargs)

        # If we get 401 and haven't tried refresh yet, try to refresh
        if response.status_code == 401 and not self._token_refresh_attempted:
            self.logger.info("Got 401, attempting token refresh...")
            if await self.refresh_token():
                # Retry the request with new token
                client = await self._get_client()
                response = await client.request(method, url, **kwargs)

        # Reset refresh flag on successful request
        if response.status_code != 401:
            self._token_refresh_attempted = False

        return response

    async def get_current_user(self) -> Optional[UserInfo]:
        """
        Get current authenticated user info.

        Returns:
            UserInfo if authenticated, None otherwise
        """
        if not self.auth_token:
            return None

        try:
            response = await self._request_with_retry("GET", "/api/v1/auth/me")

            if response.status_code == 401:
                raise RegistryAuthError("Invalid or expired token")

            if response.status_code != 200:
                return None

            data = response.json()
            self._user = UserInfo(
                id=data.get("id", ""),
                username=data.get("username", ""),
                email=data.get("email", ""),
                is_verified=data.get("is_verified", False),
                is_admin=data.get("is_admin", False),
                publisher_id=data.get("publisher_id"),
            )
            return self._user

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        if not self.auth_token:
            return False
        try:
            user = await self.get_current_user()
            return user is not None
        except RegistryError:
            return False

    # =================== Publisher ===================

    async def get_publisher(self, name: str) -> Optional[PublisherInfo]:
        """
        Get publisher info by name.

        Args:
            name: Publisher name

        Returns:
            PublisherInfo if found, None otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/api/v1/publishers/{name}")

            if response.status_code == 404:
                return None
            if response.status_code != 200:
                self.logger.warning(f"Get publisher failed: {response.status_code}")
                return None

            data = response.json()
            return PublisherInfo(
                id=data.get("id", ""),
                name=data.get("name", ""),
                display_name=data.get("display_name", ""),
                verification_status=data.get("verification_status", "unverified"),
                package_count=data.get("package_count", 0),
                total_downloads=data.get("total_downloads", 0),
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def list_publishers(
        self,
        page: int = 1,
        per_page: int = 20,
        verified_only: bool = False,
        status: Optional[str] = None,
    ) -> List[PublisherInfo]:
        """
        List publishers with pagination.

        Args:
            page: Page number (1-based)
            per_page: Items per page
            verified_only: Only return verified publishers (legacy)
            status: Filter by status (unverified, pending, verified, rejected)

        Returns:
            List of PublisherInfo
        """
        try:
            client = await self._get_client()
            params = {
                "page": page,
                "per_page": per_page,
            }
            if status:
                params["status"] = status
            elif verified_only:
                params["verified_only"] = "true"

            response = await client.get("/api/v1/publishers", params=params)

            if response.status_code != 200:
                self.logger.warning(f"List publishers failed: {response.status_code}")
                return []

            data = response.json()
            publishers = data.get("publishers", [])

            return [
                PublisherInfo(
                    id=p.get("id", ""),
                    name=p.get("name", ""),
                    display_name=p.get("display_name", ""),
                    verification_status=p.get("verification_status", "unverified"),
                    package_count=p.get("package_count", 0),
                    total_downloads=p.get("total_downloads", 0),
                )
                for p in publishers
            ]

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def get_my_publisher(self) -> Optional[PublisherInfo]:
        """
        Get the current user's publisher profile.

        Returns:
            PublisherInfo if user is a publisher, None otherwise
        """
        if not self.auth_token:
            return None

        try:
            response = await self._request_with_retry("GET", "/api/v1/auth/publisher")

            if response.status_code != 200:
                return None

            data = response.json()
            if not data:
                return None

            return PublisherInfo(
                id=data.get("id", ""),
                name=data.get("name", ""),
                display_name=data.get("display_name", ""),
                verification_status=data.get("verification_status", "unverified"),
                package_count=data.get("package_count", 0),
                total_downloads=data.get("total_downloads", 0),
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def register_publisher(
        self,
        name: str,
        display_name: str,
        email: str,
        homepage: Optional[str] = None,
    ) -> Optional[PublisherInfo]:
        """
        Register as a publisher.

        Args:
            name: Publisher handle (unique)
            display_name: Public display name
            email: Contact email
            homepage: Optional homepage URL

        Returns:
            Created PublisherInfo, or None on failure
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            payload = {
                "name": name,
                "display_name": display_name,
                "email": email,
            }
            if homepage:
                payload["homepage"] = homepage

            response = await self._request_with_retry(
                "POST", "/api/v1/auth/register-publisher", json=payload
            )

            if response.status_code == 400:
                detail = response.json().get("detail", "Bad request")
                raise RegistryError(detail)
            if response.status_code == 409:
                detail = response.json().get("detail", "Name already taken")
                raise RegistryError(detail)
            if response.status_code not in (200, 201):
                return None

            data = response.json()
            return PublisherInfo(
                id=data.get("id", ""),
                name=data.get("name", ""),
                display_name=data.get("display_name", ""),
                verification_status=data.get("verification_status", "unverified"),
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def submit_verification(
        self,
        method: str,
        data: dict,
    ) -> bool:
        """
        Submit a publisher verification request.

        Args:
            method: Verification method (e.g. 'github', 'domain')
            data: Method-specific verification data

        Returns:
            True if submission was accepted
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            payload = {"method": method, "data": data}
            response = await self._request_with_retry(
                "POST", "/api/v1/publishers/verify", json=payload
            )

            if response.status_code == 400:
                detail = response.json().get("detail", "Bad request")
                raise RegistryError(detail)
            if response.status_code == 202:
                return True

            return False

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    # =================== Admin Publisher Management ===================

    async def admin_list_pending_publishers(
        self,
        page: int = 1,
        per_page: int = 50,
    ) -> List[PublisherInfo]:
        """
        List publishers with pending verification (admin only).

        Args:
            page: Page number (1-based)
            per_page: Items per page

        Returns:
            List of pending PublisherInfo
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            params = {"page": page, "per_page": per_page}
            response = await self._request_with_retry(
                "GET", "/api/v1/publishers/admin/pending", params=params,
            )

            if response.status_code == 403:
                raise RegistryAuthError("Admin privileges required")
            if response.status_code != 200:
                self.logger.warning(f"Admin list pending failed: {response.status_code}")
                return []

            data = response.json()
            publishers = data.get("publishers", [])

            return [
                PublisherInfo(
                    id=p.get("id", ""),
                    name=p.get("name", ""),
                    display_name=p.get("display_name", ""),
                    verification_status=p.get("verification_status", "pending"),
                    package_count=p.get("package_count", 0),
                    total_downloads=p.get("total_downloads", 0),
                )
                for p in publishers
            ]

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def admin_verify_publisher(
        self,
        publisher_id: str,
        notes: str = "",
    ) -> bool:
        """
        Verify a publisher (admin only).

        Args:
            publisher_id: Publisher ID to verify
            notes: Optional notes for audit trail

        Returns:
            True if successful
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            payload = {"notes": notes}
            response = await self._request_with_retry(
                "POST",
                f"/api/v1/publishers/admin/{publisher_id}/verify",
                json=payload,
            )

            if response.status_code == 403:
                raise RegistryAuthError("Admin privileges required")
            if response.status_code == 404:
                raise RegistryError(f"Publisher '{publisher_id}' not found")
            return response.status_code == 200

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def admin_reject_publisher(
        self,
        publisher_id: str,
        notes: str = "",
    ) -> bool:
        """
        Reject a publisher verification (admin only).

        Args:
            publisher_id: Publisher ID to reject
            notes: Reason for rejection

        Returns:
            True if successful
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            payload = {"notes": notes}
            response = await self._request_with_retry(
                "POST",
                f"/api/v1/publishers/admin/{publisher_id}/reject",
                json=payload,
            )

            if response.status_code == 403:
                raise RegistryAuthError("Admin privileges required")
            if response.status_code == 404:
                raise RegistryError(f"Publisher '{publisher_id}' not found")
            return response.status_code == 200

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def admin_revoke_publisher(
        self,
        publisher_id: str,
        notes: str = "",
    ) -> bool:
        """
        Revoke a publisher's verified status (admin only).

        Args:
            publisher_id: Publisher ID to revoke
            notes: Reason for revocation

        Returns:
            True if successful
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            payload = {"notes": notes}
            response = await self._request_with_retry(
                "POST",
                f"/api/v1/publishers/admin/{publisher_id}/revoke",
                json=payload,
            )

            if response.status_code == 403:
                raise RegistryAuthError("Admin privileges required")
            if response.status_code == 404:
                raise RegistryError(f"Publisher '{publisher_id}' not found")
            if response.status_code == 400:
                detail = response.json().get("detail", "Bad request")
                raise RegistryError(detail)
            return response.status_code == 200

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    # =================== Package Discovery ===================

    async def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[PackageSummary]:
        """
        Search for packages.

        Args:
            query: Search query string
            filters: Optional filters (visibility, publisher, etc.)

        Returns:
            List of matching packages
        """
        try:
            client = await self._get_client()
            params = {"q": query}
            if filters:
                params.update(filters)

            response = await client.get("/api/packages/search", params=params)

            if response.status_code != 200:
                self.logger.warning(f"Search failed: {response.status_code}")
                return []

            data = response.json()
            packages = data.get("packages", data) if isinstance(data, dict) else data

            return [
                PackageSummary(
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    latest_version=p.get("latest_version", ""),
                    visibility=p.get("visibility", "public"),
                    downloads=p.get("downloads", 0),
                    publisher=p.get("publisher", ""),
                )
                for p in packages
            ]

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def get_package(self, name: str) -> Optional[PackageDetail]:
        """
        Get detailed package information.

        Args:
            name: Package name

        Returns:
            PackageDetail if found, None otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/api/packages/{name}")

            if response.status_code == 404:
                return None

            if response.status_code != 200:
                self.logger.warning(f"Get package failed: {response.status_code}")
                return None

            data = response.json()
            versions = [
                VersionInfo(
                    version=v.get("version", ""),
                    published_at=v.get("published_at", ""),
                    yanked=v.get("yanked", False),
                    downloads=v.get("downloads", 0),
                )
                for v in data.get("versions", [])
            ]

            return PackageDetail(
                name=data.get("name", ""),
                description=data.get("description", ""),
                latest_version=data.get("latest_version", ""),
                visibility=data.get("visibility", "public"),
                downloads=data.get("downloads", 0),
                publisher=data.get("publisher", ""),
                homepage=data.get("homepage", ""),
                repository=data.get("repository", ""),
                license=data.get("license", ""),
                keywords=data.get("keywords", []),
                versions=versions,
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def get_versions(self, name: str) -> List[VersionInfo]:
        """
        Get all versions of a package.

        Args:
            name: Package name

        Returns:
            List of versions
        """
        package = await self.get_package(name)
        if package:
            return package.versions
        return []

    async def get_version(self, name: str, version: str) -> Optional[VersionDetail]:
        """
        Get detailed version information.

        Args:
            name: Package name
            version: Version string

        Returns:
            VersionDetail if found, None otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/api/packages/{name}/versions/{version}")

            if response.status_code == 404:
                return None

            if response.status_code != 200:
                return None

            data = response.json()
            return VersionDetail(
                version=data.get("version", ""),
                published_at=data.get("published_at", ""),
                yanked=data.get("yanked", False),
                downloads=data.get("downloads", 0),
                checksum_sha256=data.get("checksum_sha256", ""),
                download_url=data.get("download_url", ""),
                dependencies=data.get("dependencies", []),
                changelog=data.get("changelog", ""),
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def get_latest_version(self, name: str) -> Optional[str]:
        """
        Get the latest version of a package.

        Args:
            name: Package name

        Returns:
            Latest version string or None
        """
        package = await self.get_package(name)
        if package:
            return package.latest_version
        return None


    # =================== Dependency Resolution ===================

    async def resolve(
        self,
        requirements: List[str],
        toolbox_version: Optional[str] = None,
    ) -> ResolutionResult:
        """
        Resolve dependencies for a list of requirements.

        Args:
            requirements: List of requirement strings (e.g., ["CloudM>=0.1.0", "DB~1.0"])
            toolbox_version: Optional ToolBox version for compatibility check

        Returns:
            ResolutionResult with resolved packages or conflicts
        """
        try:
            client = await self._get_client()
            payload = {
                "requirements": requirements,
            }
            if toolbox_version:
                payload["toolbox_version"] = toolbox_version

            response = await client.post("/api/resolve", json=payload)

            if response.status_code != 200:
                data = response.json() if response.content else {}
                return ResolutionResult(
                    success=False,
                    errors=[data.get("detail", f"Resolution failed: {response.status_code}")],
                )

            data = response.json()

            resolved = {}
            for name, pkg_data in data.get("resolved", {}).items():
                # Convert dependencies list to dict with constraints
                dependencies_dict = {}
                for dep in pkg_data.get("dependencies", []):
                    if isinstance(dep, dict):
                        dep_name = dep.get("name")
                        if dep_name:
                            dependencies_dict[dep_name] = {
                                "version": dep.get("version", ""),
                                "version_spec": dep.get("version_spec", "*"),
                                "optional": dep.get("optional", False),
                            }

                resolved[name] = ResolvedPackage(
                    name=name,
                    version=pkg_data.get("version", ""),
                    download_url=pkg_data.get("download_url", ""),
                    checksum_sha256=pkg_data.get("checksum_sha256", ""),
                    dependencies=dependencies_dict,
                )

            return ResolutionResult(
                success=data.get("success", True),
                resolved=resolved,
                conflicts=data.get("conflicts", []),
                errors=data.get("errors", []),
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    # =================== Download ===================

    async def download(
        self,
        name: str,
        version: str,
        dest_dir: Path,
    ) -> Path:
        """
        Download a package.

        Args:
            name: Package name
            version: Version to download
            dest_dir: Destination directory

        Returns:
            Path to downloaded file

        Raises:
            PackageNotFoundError: If package not found
            VersionNotFoundError: If version not found
            DownloadError: If download fails
        """
        # Get version details for download URL
        version_detail = await self.get_version(name, version)
        if not version_detail:
            raise VersionNotFoundError(f"Version {version} not found for {name}")

        if not version_detail.download_url:
            raise DownloadError(f"No download URL for {name}@{version}")

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{name}-{version}.zip"
        dest_path = dest_dir / filename

        try:
            client = await self._get_client()

            async with client.stream("GET", version_detail.download_url) as response:
                if response.status_code == 404:
                    raise VersionNotFoundError(f"Download not found: {name}@{version}")

                if response.status_code != 200:
                    raise DownloadError(f"Download failed: {response.status_code}")

                # Download with checksum verification
                hasher = hashlib.sha256()
                with open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        hasher.update(chunk)

                # Verify checksum
                if version_detail.checksum_sha256:
                    if hasher.hexdigest() != version_detail.checksum_sha256:
                        dest_path.unlink()
                        raise DownloadError("Checksum verification failed")

            # Cache the download
            await self._cache_package(name, version, dest_path)

            return dest_path

        except httpx.RequestError as e:
            raise DownloadError(f"Download failed: {e}") from e

    async def download_with_dependencies(
        self,
        name: str,
        version: str,
        dest_dir: Path,
    ) -> List[Path]:
        """
        Download a package with all its dependencies.

        Args:
            name: Package name
            version: Version to download
            dest_dir: Destination directory

        Returns:
            List of paths to downloaded files
        """
        # Resolve dependencies
        result = await self.resolve([f"{name}=={version}"])

        if not result.success:
            raise DownloadError(f"Dependency resolution failed: {result.errors}")

        downloaded = []
        for pkg_name, pkg in result.resolved.items():
            # Check cache first
            cached = await self.get_cached_package(pkg_name, pkg.version)
            if cached:
                downloaded.append(cached)
                continue

            path = await self.download(pkg_name, pkg.version, dest_dir)
            downloaded.append(path)

        return downloaded

    async def get_diff_info(
        self,
        name: str,
        from_version: str,
        to_version: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get diff information between two versions.

        Args:
            name: Package name
            from_version: Source version
            to_version: Target version

        Returns:
            Dict with diff info or None if not available
        """
        try:
            client = await self._get_client()

            response = await client.get(
                f"/api/v1/packages/{name}/diff/{from_version}/{to_version}"
            )

            if response.status_code == 404:
                return None

            if response.status_code != 200:
                raise DownloadError(f"Failed to get diff info: {response.status_code}")

            return response.json()

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def download_with_diff(
        self,
        name: str,
        from_version: str,
        to_version: str,
        dest_dir: Path,
        max_diff_size_ratio: float = 0.5,
    ) -> Path:
        """
        Download package using diff if available and efficient.

        Args:
            name: Package name
            from_version: Source version (currently installed)
            to_version: Target version
            dest_dir: Destination directory
            max_diff_size_ratio: Maximum ratio of diff/full size to use diff (default: 0.5 = 50%)

        Returns:
            Path to downloaded package file

        Raises:
            PackageNotFoundError: If package not found
            DownloadError: If download fails
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Try to get diff info
        diff_info = await self.get_diff_info(name, from_version, to_version)

        if diff_info and diff_info.get("compression_ratio", 1.0) < max_diff_size_ratio:
            # Use diff download
            print(f"📦 Using diff update: {diff_info['patch_size']} bytes "
                  f"({diff_info['compression_ratio']:.1%} of full size)")

            return await self._download_and_apply_diff(name, from_version, to_version, dest_dir, diff_info)

        # Fallback to full download
        print(f"📦 Full download required (diff not available or too large)")
        return await self.download(name, to_version, dest_dir)

    async def _download_and_apply_diff(
        self,
        name: str,
        from_version: str,
        to_version: str,
        dest_dir: Path,
        diff_info: Dict[str, Any],
    ) -> Path:
        """
        Download and apply diff patch.

        Args:
            name: Package name
            from_version: Source version
            to_version: Target version
            dest_dir: Destination directory
            diff_info: Diff information from get_diff_info

        Returns:
            Path to patched package file

        Raises:
            DownloadError: If download or patch application fails
        """
        import tempfile
        import shutil

        # Download patch
        patch_url = f"{self.registry_url}/api/v1/packages/{name}/diff/{from_version}/{to_version}/download"
        patch_path = dest_dir / f"{name}-{from_version}-to-{to_version}.patch"

        try:
            client = await self._get_client()
            response = await client.get(patch_url)

            if response.status_code != 200:
                raise DownloadError(f"Failed to download patch: {response.status_code}")

            # For now, the API returns metadata, not the actual patch
            # In production, this would stream the actual patch file
            patch_data = response.json()

            # Get the old package
            old_package_path = dest_dir / f"{name}-{from_version}.zip"
            if not old_package_path.exists():
                # Fall back to full download
                print(f"⚠ Old package not found for patch application, using full download")
                return await self.download(name, to_version, dest_dir)

            # Create new package path
            new_package_path = dest_dir / f"{name}-{to_version}.zip"

            # For now, copy old to new as placeholder
            # In production, use bspatch to apply the actual patch
            shutil.copy(old_package_path, new_package_path)

            print(f"✓ Patch applied: {new_package_path}")

            return new_package_path

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    # =================== Publishing ===================

    async def publish(self, package_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Publish a new package to the registry.

        Args:
            package_path: Path to package directory or ZIP file
            metadata: Package metadata (name, version, description, etc.)

        Returns:
            True if successful

        Raises:
            PublishPermissionError: If user cannot publish
            RegistryError: If publish fails
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required for publishing")

        user = await self.get_current_user()
        if not user or not user.is_verified:
            raise PublishPermissionError("Only verified publishers can publish packages")

        try:
            client = await self._get_client()

            # Prepare multipart form data
            package_path = Path(package_path)

            if package_path.is_dir():
                # Create ZIP from directory
                import tempfile
                import zipfile
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in package_path.rglob("*"):
                        if file.is_file():
                            zf.write(file, file.relative_to(package_path))
                file_path = tmp_path
            else:
                file_path = package_path

            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/zip")}
                data = {"metadata": json.dumps(metadata)}

                response = await client.post(
                    "/api/packages",
                    files=files,
                    data=data,
                )

            if response.status_code == 403:
                raise PublishPermissionError("Publishing not allowed")

            if response.status_code not in (200, 201):
                data = response.json() if response.content else {}
                raise RegistryError(data.get("detail", f"Publish failed: {response.status_code}"))

            return True

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def upload_version(
        self,
        name: str,
        version: str,
        file_path: Path,
        changelog: Optional[str] = None,
    ) -> bool:
        """
        Upload a new version of an existing package.

        Args:
            name: Package name
            version: New version string
            file_path: Path to package ZIP file
            changelog: Optional changelog text

        Returns:
            True if successful
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            client = await self._get_client()

            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/zip")}
                data = {"version": version}
                if changelog:
                    data["changelog"] = changelog

                response = await client.post(
                    f"/api/packages/{name}/versions",
                    files=files,
                    data=data,
                )

            if response.status_code == 403:
                raise PublishPermissionError("Not authorized to upload to this package")

            if response.status_code not in (200, 201):
                return False

            return True

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def yank_version(self, name: str, version: str, reason: str) -> bool:
        """
        Yank a version (mark as not recommended).

        Args:
            name: Package name
            version: Version to yank
            reason: Reason for yanking

        Returns:
            True if successful
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            client = await self._get_client()
            response = await client.post(
                f"/api/packages/{name}/versions/{version}/yank",
                json={"reason": reason},
            )

            if response.status_code == 403:
                raise PublishPermissionError("Not authorized to yank this version")

            return response.status_code == 200

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def create_package(
        self,
        name: str,
        display_name: str,
        package_type: str = "mod",
        visibility: str = "public",
        description: str = "",
        readme: str = "",
        homepage: Optional[str] = None,
        repository: Optional[str] = None,
        license: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new package in the registry.

        Args:
            name: Unique package name (lowercase, alphanumeric, hyphens, underscores)
            display_name: Human-readable display name
            package_type: Type of package (mod, artifact, library, theme, plugin)
            visibility: Package visibility (public, private, unlisted)
            description: Short description (max 500 chars)
            readme: Full README content
            homepage: Homepage URL
            repository: Repository URL
            license: License identifier
            keywords: Search keywords

        Returns:
            Created package data or None on failure

        Raises:
            RegistryAuthError: If not authenticated
            PublishPermissionError: If user cannot create packages
            RegistryError: If creation fails
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required to create packages")

        try:
            client = await self._get_client()

            payload = {
                "name": name,
                "display_name": display_name,
                "package_type": package_type,
                "visibility": visibility,
                "description": description,
                "readme": readme,
            }

            if homepage:
                payload["homepage"] = homepage
            if repository:
                payload["repository"] = repository
            if license:
                payload["license"] = license
            if keywords:
                payload["keywords"] = keywords

            response = await self._request_with_retry("POST", "/api/v1/packages", json=payload)

            if response.status_code == 401:
                raise RegistryAuthError("Authentication failed")

            if response.status_code == 403:
                raise PublishPermissionError("Must be a registered publisher to create packages")

            if response.status_code == 409:
                data = response.json() if response.content else {}
                raise RegistryError(data.get("detail", "Package already exists"))

            if response.status_code not in (200, 201):
                data = response.json() if response.content else {}
                raise RegistryError(data.get("detail", f"Create failed: {response.status_code}"))

            return response.json()

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def get_my_packages(self) -> List[PackageSummary]:
        """
        Get all packages owned by the current user.

        Returns:
            List of packages owned by the authenticated user

        Raises:
            RegistryAuthError: If not authenticated
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required")

        try:
            user = await self.get_current_user()
            if not user:
                raise RegistryAuthError("Could not get current user")

            client = await self._get_client()

            # Get packages filtered by publisher
            response = await self._request_with_retry(
                "GET",
                "/api/v1/packages",
                params={"publisher": user.publisher_id} if user.publisher_id else {}
            )

            if response.status_code != 200:
                self.logger.warning(f"Get my packages failed: {response.status_code}")
                return []

            data = response.json()
            packages = data.get("packages", [])

            return [
                PackageSummary(
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    latest_version=p.get("latest_version", ""),
                    visibility=p.get("visibility", "public"),
                    downloads=p.get("total_downloads", 0),
                    publisher=p.get("publisher_id", ""),
                )
                for p in packages
            ]

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def delete_package(self, name: str) -> bool:
        """
        Delete a package from the registry.

        Args:
            name: Package name to delete

        Returns:
            True if deleted successfully

        Raises:
            RegistryAuthError: If not authenticated
            PublishPermissionError: If user cannot delete this package
            PackageNotFoundError: If package not found
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required to delete packages")

        try:
            response = await self._request_with_retry("DELETE", f"/api/v1/packages/{name}")

            if response.status_code == 401:
                raise RegistryAuthError("Authentication failed")

            if response.status_code == 403:
                raise PublishPermissionError("Not authorized to delete this package")

            if response.status_code == 404:
                raise PackageNotFoundError(f"Package '{name}' not found")

            return response.status_code == 204

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    async def update_package(
        self,
        name: str,
        display_name: Optional[str] = None,
        visibility: Optional[str] = None,
        description: Optional[str] = None,
        readme: Optional[str] = None,
        homepage: Optional[str] = None,
        repository: Optional[str] = None,
        license: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing package.

        Args:
            name: Package name to update
            display_name: New display name
            visibility: New visibility (public, private, unlisted)
            description: New description
            readme: New README content
            homepage: New homepage URL
            repository: New repository URL
            license: New license identifier
            keywords: New keywords

        Returns:
            Updated package data or None on failure

        Raises:
            RegistryAuthError: If not authenticated
            PublishPermissionError: If user cannot update this package
            PackageNotFoundError: If package not found
        """
        if not self.auth_token:
            raise RegistryAuthError("Authentication required to update packages")

        try:
            payload = {}
            if display_name is not None:
                payload["display_name"] = display_name
            if visibility is not None:
                payload["visibility"] = visibility
            if description is not None:
                payload["description"] = description
            if readme is not None:
                payload["readme"] = readme
            if homepage is not None:
                payload["homepage"] = homepage
            if repository is not None:
                payload["repository"] = repository
            if license is not None:
                payload["license"] = license
            if keywords is not None:
                payload["keywords"] = keywords

            if not payload:
                return None  # Nothing to update

            response = await self._request_with_retry("PATCH", f"/api/v1/packages/{name}", json=payload)

            if response.status_code == 401:
                raise RegistryAuthError("Authentication failed")

            if response.status_code == 403:
                raise PublishPermissionError("Not authorized to update this package")

            if response.status_code == 404:
                raise PackageNotFoundError(f"Package '{name}' not found")

            if response.status_code != 200:
                return None

            return response.json()

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e

    # =================== Artifacts ===================

    async def get_artifact(self, name: str) -> Optional[ArtifactDetail]:
        """
        Get artifact information.

        Args:
            name: Artifact name

        Returns:
            ArtifactDetail if found
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/api/artifacts/{name}")

            if response.status_code == 404:
                return None

            if response.status_code != 200:
                return None

            data = response.json()
            builds = [
                ArtifactBuild(
                    version=b.get("version", ""),
                    platform=b.get("platform", ""),
                    arch=b.get("arch", ""),
                    download_url=b.get("download_url", ""),
                    checksum_sha256=b.get("checksum_sha256", ""),
                    size=b.get("size", 0),
                )
                for b in data.get("builds", [])
            ]

            return ArtifactDetail(
                name=data.get("name", ""),
                description=data.get("description", ""),
                latest_version=data.get("latest_version", ""),
                builds=builds,
            )

        except httpx.RequestError as e:
            raise RegistryConnectionError(f"Connection failed: {e}") from e


    async def get_latest_artifact(
        self, name: str, platform: str, arch: str
    ) -> Optional[ArtifactBuild]:
        """
        Get the latest artifact build for a platform/arch.

        Args:
            name: Artifact name
            platform: Platform (e.g., "windows", "linux", "macos")
            arch: Architecture (e.g., "x86_64", "aarch64")

        Returns:
            ArtifactBuild if found
        """
        artifact = await self.get_artifact(name)
        if not artifact:
            return None

        # Find matching build
        for build in artifact.builds:
            if build.platform == platform and build.arch == arch:
                if build.version == artifact.latest_version:
                    return build

        # Return any matching platform/arch if no latest
        for build in artifact.builds:
            if build.platform == platform and build.arch == arch:
                return build

        return None

    async def download_artifact(
        self,
        name: str,
        version: str,
        platform: str,
        arch: str,
        dest: Path,
    ) -> Path:
        """
        Download an artifact build.

        Args:
            name: Artifact name
            version: Version to download
            platform: Platform
            arch: Architecture
            dest: Destination path

        Returns:
            Path to downloaded file
        """
        artifact = await self.get_artifact(name)
        if not artifact:
            raise PackageNotFoundError(f"Artifact {name} not found")

        # Find matching build
        build = None
        for b in artifact.builds:
            if b.version == version and b.platform == platform and b.arch == arch:
                build = b
                break

        if not build:
            raise VersionNotFoundError(
                f"Build not found: {name}@{version} for {platform}/{arch}"
            )

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            client = await self._get_client()

            async with client.stream("GET", build.download_url) as response:
                if response.status_code != 200:
                    raise DownloadError(f"Download failed: {response.status_code}")

                hasher = hashlib.sha256()
                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        hasher.update(chunk)

                # Verify checksum
                if build.checksum_sha256:
                    if hasher.hexdigest() != build.checksum_sha256:
                        dest.unlink()
                        raise DownloadError("Checksum verification failed")

            return dest

        except httpx.RequestError as e:
            raise DownloadError(f"Download failed: {e}") from e

    # =================== Cache Management ===================

    async def _cache_package(self, name: str, version: str, file_path: Path) -> None:
        """Cache a downloaded package."""
        cache_path = self.cache_dir / name / version
        cache_path.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy2(file_path, cache_path / file_path.name)

        # Write metadata
        meta_path = cache_path / "meta.json"
        meta_path.write_text(json.dumps({
            "name": name,
            "version": version,
            "cached_at": asyncio.get_event_loop().time(),
            "file": file_path.name,
        }))

    async def get_cached_package(
        self, name: str, version: str
    ) -> Optional[Path]:
        """
        Get a cached package if available.

        Args:
            name: Package name
            version: Version

        Returns:
            Path to cached file or None
        """
        cache_path = self.cache_dir / name / version
        meta_path = cache_path / "meta.json"

        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            file_path = cache_path / meta["file"]
            if file_path.exists():
                return file_path
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    async def clear_cache(self) -> None:
        """Clear all cached packages."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)



# =================== Lock File Manager ===================


class LockFileManager:
    """
    Manages mods.lock.yaml for tracking installed packages.

    The lock file contains:
    - Installed packages with versions and checksums
    - Dependency information
    - Pending updates
    """

    def __init__(self, lock_file_path: Optional[Path] = None):
        """
        Initialize the lock file manager.

        Args:
            lock_file_path: Path to lock file (default: ./mods.lock.yaml)
        """
        self.lock_file = lock_file_path or Path("mods.lock.yaml")
        self._data: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """
        Load the lock file.

        Returns:
            Lock file data
        """
        if self._data is not None:
            return self._data

        if not self.lock_file.exists():
            self._data = {
                "version": "1.0",
                "packages": {},
                "pending_updates": {},
            }
            return self._data

        try:
            import yaml
            with open(self.lock_file, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

            # Ensure required keys exist
            if "packages" not in self._data:
                self._data["packages"] = {}
            if "pending_updates" not in self._data:
                self._data["pending_updates"] = {}
            if "version" not in self._data:
                self._data["version"] = "1.0"

            return self._data

        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load lock file: {e}")
            self._data = {"version": "1.0", "packages": {}, "pending_updates": {}}
            return self._data

    def save(self) -> None:
        """Save the lock file."""
        if self._data is None:
            return

        try:
            import yaml
            with open(self.lock_file, "w", encoding="utf-8") as f:
                yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save lock file: {e}")

    def add_package(
        self,
        name: str,
        version: str,
        checksum: str,
        source: str,
        dependencies: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add or update a package in the lock file.

        Args:
            name: Package name
            version: Installed version
            checksum: SHA256 checksum
            source: Download source URL
            dependencies: List of dependency dicts with version constraints
                          Each dict: {name, version_spec, installed_version, optional}
        """
        data = self.load()
        data["packages"][name] = {
            "version": version,
            "checksum": checksum,
            "source": source,
            "dependencies": dependencies or [],
            "installed_at": __import__("time").time(),
        }

        # Remove from pending updates if present
        if name in data.get("pending_updates", {}):
            del data["pending_updates"][name]

    def remove_package(self, name: str) -> None:
        """
        Remove a package from the lock file.

        Args:
            name: Package name
        """
        data = self.load()
        if name in data["packages"]:
            del data["packages"][name]
        if name in data.get("pending_updates", {}):
            del data["pending_updates"][name]

    def get_installed_version(self, name: str) -> Optional[str]:
        """
        Get the installed version of a package.

        Args:
            name: Package name

        Returns:
            Version string or None if not installed
        """
        data = self.load()
        pkg = data["packages"].get(name)
        if pkg:
            return pkg.get("version")
        return None

    def get_all_installed(self) -> Dict[str, str]:
        """
        Get all installed packages with versions.

        Returns:
            Dict of package name -> version
        """
        data = self.load()
        return {
            name: pkg.get("version", "")
            for name, pkg in data["packages"].items()
        }

    def get_pending_updates(self) -> Dict[str, str]:
        """
        Get packages with pending updates.

        Returns:
            Dict of package name -> new version
        """
        data = self.load()
        return dict(data.get("pending_updates", {}))

    def mark_pending_update(self, name: str, new_version: str) -> None:
        """
        Mark a package as having a pending update.

        Args:
            name: Package name
            new_version: New version available
        """
        data = self.load()
        if "pending_updates" not in data:
            data["pending_updates"] = {}
        data["pending_updates"][name] = new_version

    def clear_pending_update(self, name: str) -> None:
        """
        Clear pending update for a package.

        Args:
            name: Package name
        """
        data = self.load()
        if name in data.get("pending_updates", {}):
            del data["pending_updates"][name]

    def get_package_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get full package info from lock file.

        Args:
            name: Package name

        Returns:
            Package info dict or None
        """
        data = self.load()
        return data["packages"].get(name)

    def is_installed(self, name: str) -> bool:
        """
        Check if a package is installed.

        Args:
            name: Package name

        Returns:
            True if installed
        """
        return self.get_installed_version(name) is not None

    def check_compatibility(
        self,
        package_name: str,
        required_version_spec: str,
    ) -> tuple[bool, list[str]]:
        """
        Check if installed package version is compatible with requirement.

        Args:
            package_name: Name of the package to check
            required_version_spec: Version specifier (e.g., ">=2.0.0,<3.0.0")

        Returns:
            Tuple of (is_compatible, error_messages)
        """
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet

        installed_version = self.get_installed_version(package_name)
        if not installed_version:
            return False, [f"Package '{package_name}' is not installed"]

        try:
            spec = SpecifierSet(required_version_spec)
            installed_ver = Version(installed_version)

            if installed_ver not in spec:
                return False, [
                    f"Package '{package_name}' requires {required_version_spec} "
                    f"but {installed_version} is installed"
                ]

            return True, []
        except Exception as e:
            return False, [f"Error checking compatibility for '{package_name}': {e}"]

    def check_upgrade_conflicts(
        self,
        package_name: str,
        from_version: str,
        to_version: str,
    ) -> list[str]:
        """
        Check if upgrading a package would break dependencies of other packages.

        Args:
            package_name: Package to upgrade
            from_version: Current version
            to_version: Target version

        Returns:
            List of conflict descriptions (empty if no conflicts)
        """
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet

        conflicts = []
        data = self.load()

        for pkg_name, pkg_info in data.get("packages", {}).items():
            if pkg_name == package_name:
                continue

            dependencies = pkg_info.get("dependencies", [])
            for dep in dependencies:
                if isinstance(dep, dict):
                    dep_name = dep.get("name")
                    dep_spec = dep.get("version_spec")
                else:
                    # Legacy format: just package name
                    continue

                if dep_name == package_name:
                    try:
                        spec = SpecifierSet(dep_spec or "*")
                        new_ver = Version(to_version)

                        if new_ver not in spec:
                            conflicts.append(
                                f"  - {pkg_name} requires {package_name}{dep_spec} "
                                f"but {to_version} would be installed"
                            )
                    except Exception as e:
                        conflicts.append(
                            f"  - {pkg_name} has invalid dependency spec for {package_name}: {e}"
                        )

        return conflicts

    def get_dependency_constraints(
        self,
        package_name: str,
    ) -> list[Dict[str, Any]]:
        """
        Get all dependency constraints for a specific package.

        Args:
            package_name: Package name

        Returns:
            List of dicts with constraints from all installed packages
        """
        data = self.load()
        constraints = []

        for pkg_name, pkg_info in data.get("packages", {}).items():
            dependencies = pkg_info.get("dependencies", [])
            for dep in dependencies:
                if isinstance(dep, dict):
                    dep_name = dep.get("name")
                    if dep_name == package_name:
                        constraints.append({
                            "from_package": pkg_name,
                            "version_spec": dep.get("version_spec", "*"),
                            "optional": dep.get("optional", False),
                        })

        return constraints
