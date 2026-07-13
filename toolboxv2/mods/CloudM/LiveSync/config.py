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
    bucket: str = "tb-shared"
    prefix: str = ""  # defaults to share_id if empty
    max_file_size: int = 50 * 1024 * 1024  # 50 MB
    debounce_seconds: float = 2.0
    max_concurrent_transfers: int = 5
    share_token: str = ""  # raw share token, sent in AUTH (server validates it)
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

    SECURITY (FIX 3b): Token payload is encrypted with the TB Device Key
    via Code.encode_code(). The AES file-encryption key, endpoints, and
    all metadata are opaque to the client. Only the server (which has the
    Device Key) can decode the token.

    Token format: v2 = "v2:" + Code.encode_code(json_payload)
                   v1 = urlsafe_b64(json_payload)  [legacy, deprecated]
    """

    share_id: str
    minio_endpoint: str
    bucket: str
    prefix: str
    encryption_key: str  # Base64 AES key
    ws_endpoint: str
    version: int = 3
    expires_at: float = 0  # Unix timestamp, 0 = no expiry

    def encode(self) -> str:
        """Encode token v3 = split: endpoints Klartext, Secrets verschlüsselt."""
        # SECRET part (verschlüsselt mit Device Key)
        if not self.expires_at:
            import time as _time
            self.expires_at = _time.time() + 86400  # 24 Stunden
        secret = {
            "share_id": self.share_id,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "enc_key": self.encryption_key,
            "exp": self.expires_at,
        }

        # PUBLIC part (Klartext — Endpoints, sonst kann Client nicht verbinden)
        public = {
            "ws": self.ws_endpoint,
            "minio": self.minio_endpoint,
            "v": 3,
        }

        try:
            from toolboxv2.utils.security.cryp import Code
            encrypted_secret = Code.encode_code(json.dumps(secret))
        except Exception:
            # Standalone fallback
            encrypted_secret = base64.urlsafe_b64encode(
                json.dumps(secret).encode()).decode()

        # Format: v3:<public_b64>.<encrypted_secret>
        public_b64 = base64.urlsafe_b64encode(
            json.dumps(public).encode()).decode()
        return f"v3:{public_b64}.{encrypted_secret}"

    @classmethod
    def decode(cls, token: str) -> ShareToken:
        """Decode token v3 (split), v2 (encrypted), v1 (legacy)."""
        try:
            if token.startswith("v3:"):
                # v3 = split: <public_b64>.<encrypted_secret>
                payload = token[3:]
                public_b64, encrypted_secret = payload.split(".", 1)
                public = json.loads(
                    base64.urlsafe_b64decode(public_b64).decode())

                try:
                    from toolboxv2.utils.security.cryp import Code
                    secret_json = Code.decode_code(encrypted_secret)
                    secret = json.loads(secret_json)
                except Exception:
                    # Server-side: Device Key verfügbar
                    secret = json.loads(base64.urlsafe_b64decode(
                        encrypted_secret).decode())

                return cls(
                    share_id=secret["share_id"],
                    minio_endpoint=public["minio"],
                    bucket=secret["bucket"],
                    prefix=secret["prefix"],
                    encryption_key=secret["enc_key"],
                    ws_endpoint=public["ws"],
                    expires_at=secret.get("exp", 0),
                    version=3,
                )

        except Exception as exc:
            raise ValueError(f"Invalid share token: {exc}") from exc

    def to_sync_config(self, vault_path: str, raw_token: str = "") -> SyncConfig:
        """Convert token data into a SyncConfig for the client.

        raw_token: the original encoded token string. It is passed through to
        the AUTH message unchanged — the client cannot re-encode it because the
        secret part is encrypted with the *server's* device key.
        """
        return SyncConfig(
            share_id=self.share_id,
            vault_path=vault_path,
            minio_endpoint=self.minio_endpoint,
            ws_endpoint=self.ws_endpoint,
            encryption_key=self.encryption_key,
            bucket=self.bucket,
            prefix=self.prefix,
            share_token=raw_token,
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
        LIVESYNC_BUCKET      (default: tb-shared)
    """
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        "access_key": os.getenv("MINIO_ROOT_USER", "admin"),
        "secret_key": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "secure": os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes"),
        "ws_host": os.getenv("LIVESYNC_WS_HOST", "0.0.0.0"),
        "ws_port": int(os.getenv("LIVESYNC_WS_PORT", "8765")),
        "bucket": os.getenv("LIVESYNC_BUCKET", "tb-shared"),
        "ws_secure": os.getenv("LIVESYNC_WSS", "false").lower() in ("true", "1", "yes"),
        "ws_ssl_cert": os.getenv("LIVESYNC_SSL_CERT", ""),
        "ws_ssl_key": os.getenv("LIVESYNC_SSL_KEY", ""),
    }
