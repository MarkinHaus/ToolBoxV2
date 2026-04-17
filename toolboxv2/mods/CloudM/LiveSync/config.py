"""
LiveSync Configuration
======================
- SyncConfig: runtime configuration for a sync session
- ShareToken: encode/decode share tokens (NO MinIO credentials inside!)
- load_env_config: read MinIO + WS settings from environment
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field


@dataclass
class SyncConfig:
    """Runtime configuration for one sync share."""

    share_id: str
    vault_path: str
    minio_endpoint: str
    ws_endpoint: str
    encryption_key: str  # Base64-encoded AES-256 key

    # Defaults
    bucket: str = "livesync"
    prefix: str = ""  # defaults to share_id if empty
    max_file_size: int = 50 * 1024 * 1024  # 50 MB
    debounce_seconds: float = 2.0
    max_concurrent_transfers: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    healthcheck_interval: float = 30.0

    def __post_init__(self):
        if not self.prefix:
            self.prefix = self.share_id


@dataclass
class ShareToken:
    """
    Encodes everything a client needs to join a share.

    SECURITY: Token contains the AES encryption key and endpoints,
    but NEVER MinIO access/secret keys. Those are provisioned via
    WebSocket after authentication.
    """

    share_id: str
    minio_endpoint: str
    bucket: str
    prefix: str
    encryption_key: str  # Base64 AES key
    ws_endpoint: str
    version: int = 1

    def encode(self) -> str:
        """Encode token as URL-safe Base64 string."""
        data = {
            "v": self.version,
            "share_id": self.share_id,
            "minio_endpoint": self.minio_endpoint,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "enc_key": self.encryption_key,
            "ws_endpoint": self.ws_endpoint,
        }
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

    @classmethod
    def decode(cls, token: str) -> ShareToken:
        """Decode token from Base64 string. Raises ValueError on bad input."""
        try:
            raw = base64.urlsafe_b64decode(token)
            data = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"Invalid share token: {exc}") from exc

        return cls(
            share_id=data["share_id"],
            minio_endpoint=data["minio_endpoint"],
            bucket=data["bucket"],
            prefix=data["prefix"],
            encryption_key=data["enc_key"],
            ws_endpoint=data["ws_endpoint"],
            version=data.get("v", 1),
        )

    def to_sync_config(self, vault_path: str) -> SyncConfig:
        """Convert token data into a SyncConfig for the client."""
        return SyncConfig(
            share_id=self.share_id,
            vault_path=vault_path,
            minio_endpoint=self.minio_endpoint,
            ws_endpoint=self.ws_endpoint,
            encryption_key=self.encryption_key,
            bucket=self.bucket,
            prefix=self.prefix,
        )


def load_env_config() -> dict:
    """
    Load MinIO + LiveSync configuration from environment variables.

    Env vars:
        MINIO_ENDPOINT       (default: 127.0.0.1:9000)
        MINIO_ROOT_USER      (default: admin)
        MINIO_ROOT_PASSWORD   (default: minioadmin)
        MINIO_SECURE         (default: false)
        LIVESYNC_WS_HOST     (default: 0.0.0.0)
        LIVESYNC_WS_PORT     (default: 8765)
        LIVESYNC_BUCKET      (default: livesync)
    """
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        "access_key": os.getenv("MINIO_ROOT_USER", "admin"),
        "secret_key": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "secure": os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes"),
        "ws_host": os.getenv("LIVESYNC_WS_HOST", "0.0.0.0"),
        "ws_port": int(os.getenv("LIVESYNC_WS_PORT", "8765")),
        "bucket": os.getenv("LIVESYNC_BUCKET", "livesync"),
    }
