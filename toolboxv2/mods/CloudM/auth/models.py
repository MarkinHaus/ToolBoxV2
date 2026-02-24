"""
Data models for the auth system.
"""

import time
from typing import Dict, List
from dataclasses import dataclass, asdict, field


@dataclass
class OAuthProvider:
    """OAuth Provider data for a user"""
    provider_id: str
    provider: str       # "discord", "google"
    access_token: str = ""
    refresh_token: str = ""
    token_expires: float = 0.0
    username: str = ""
    email: str = ""
    avatar: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthProvider":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Passkey:
    """WebAuthn Passkey credential"""
    credential_id: str
    public_key: str           # base64-encoded
    sign_count: int = 0
    name: str = "Passkey"
    transports: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Passkey":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MinIOCredentials:
    """MinIO User Credentials stored in UserData"""
    access_key: str = ""        # MinIO Access Key (tb_xxx)
    secret_key: str = ""        # MinIO Secret Key (encrypted when stored)
    created_at: float = 0.0     # When credentials were created
    last_rotated: float = 0.0   # When credentials were last rotated
    policies: List[str] = field(default_factory=list)  # List of attached policy names
    enabled: bool = True        # Whether MinIO access is enabled

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MinIOCredentials":
        safe_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**safe_data)


@dataclass
class UserData:
    """User profile stored in AUTH_USER::{user_id}"""
    user_id: str
    username: str
    email: str
    level: int = 1
    created_at: float = field(default_factory=time.time)
    last_login: float = 0.0
    settings: dict = field(default_factory=dict)
    mod_data: dict = field(default_factory=dict)
    oauth_providers: Dict[str, dict] = field(default_factory=dict)
    passkeys: List[dict] = field(default_factory=list)

    # MinIO Credentials - Unified User System
    minio_credentials: MinIOCredentials = field(default_factory=MinIOCredentials)

    # MinIO Policy Fields for Data Sharing
    minio_policy: dict = field(default_factory=dict)  # Custom MinIO policy for this user
    shared_with: Dict[str, List[str]] = field(default_factory=dict)  # user_id -> [paths] they can access

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserData":
        # Handle migration from old UserData without new fields
        safe_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        # Ensure new fields have defaults if not present
        if "minio_policy" not in safe_data:
            safe_data["minio_policy"] = {}
        if "shared_with" not in safe_data:
            safe_data["shared_with"] = {}
        if "minio_credentials" not in safe_data:
            safe_data["minio_credentials"] = MinIOCredentials()
        elif isinstance(safe_data["minio_credentials"], dict):
            safe_data["minio_credentials"] = MinIOCredentials.from_dict(safe_data["minio_credentials"])
        return cls(**safe_data)

    def grant_access_to(self, target_user_id: str, paths: List[str]) -> None:
        """
        Grant another user access to specific paths.

        Args:
            target_user_id: User ID to grant access to
            paths: List of paths this user can access
        """
        if target_user_id not in self.shared_with:
            self.shared_with[target_user_id] = []
        self.shared_with[target_user_id].extend(p for p in paths if p not in self.shared_with[target_user_id])

    def revoke_access_from(self, target_user_id: str, paths: List[str] = None) -> None:
        """
        Revoke access from another user.

        Args:
            target_user_id: User ID to revoke access from
            paths: Specific paths to revoke (None = all paths)
        """
        if target_user_id not in self.shared_with:
            return

        if paths is None:
            del self.shared_with[target_user_id]
        else:
            self.shared_with[target_user_id] = [p for p in self.shared_with[target_user_id] if p not in paths]
            if not self.shared_with[target_user_id]:
                del self.shared_with[target_user_id]

    def list_shared_access(self) -> Dict[str, List[str]]:
        """
        Get list of all users with access to this user's data.

        Returns:
            Dict mapping user_id to list of accessible paths
        """
        return self.shared_with.copy()

    def get_accessible_paths_for(self, target_user_id: str) -> List[str]:
        """
        Get paths accessible to a specific user.

        Args:
            target_user_id: User ID to check

        Returns:
            List of paths the user can access
        """
        return self.shared_with.get(target_user_id, []).copy()
