"""
LiveSync Configuration
======================
- SyncConfig: runtime configuration for a sync session
- ShareToken: encode/verify share tokens (v4, HMAC-SHA256 signed)
- load_env_config: read MinIO + WS settings from environment

SECURITY MODEL (v4)
-------------------
A share token carries the AES-256 file-encryption key in clear text inside
its payload. That is unavoidable: the client encrypts and decrypts locally,
so it must be able to read the key, and it does not have the server's device
key. **Treat a share token like a password.** Anyone holding it can decrypt
the share's contents.

What the HMAC signature adds: only the node that owns the TB device key can
*mint* a token. A token that was not signed by this server is rejected at
AUTH, so nobody can fabricate a share membership. There is no unsigned
fallback path - an unverifiable token is an error, never a downgrade.

Token wire format:
    v4:<payload_b64url>.<hmac_sha256_b64url>
    payload   = canonical JSON (sorted keys, no spaces)
    signature = HMAC-SHA256(device_key, payload_b64url)
Both parts are unpadded URL-safe base64.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

TOKEN_VERSION = 4
DEFAULT_TOKEN_TTL = 86400  # 24 h


# -- base64 helpers (unpadded, URL-safe) --

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signing_key() -> bytes:
    """
    Return the HMAC signing key: the ToolBox device key of this node.

    The device key lives in ``<tb_root>/.info/device.enc`` and is derived from
    the ``TB_R_KEY`` environment variable. Only the node that created a share
    can sign tokens for it, and only that node can verify them.

    Raises:
        RuntimeError: if the device key is unavailable.
    """
    try:
        from toolboxv2.utils.security.cryp import DEVICE_KEY
        key = DEVICE_KEY()
    except Exception as exc:
        raise RuntimeError(
            f"LiveSync token signing needs the TB device key: {exc}"
        ) from exc

    if isinstance(key, str):
        key = key.encode("utf-8")
    if not key:
        raise RuntimeError("LiveSync token signing needs the TB device key: empty key")
    return key


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
    share_token: str = ""  # raw share token, sent in AUTH (server verifies it)
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    healthcheck_interval: float = 30.0

    def __post_init__(self):
        if not self.prefix:
            self.prefix = self.share_id


@dataclass
class ShareToken:
    """
    Everything a client needs to join a share, signed by the issuing node.

    :meth:`encode` mints one (needs the device key), :meth:`decode` reads one
    without verifying (client side, which has no device key), :meth:`verify`
    authenticates one (server side).
    """

    share_id: str
    minio_endpoint: str
    bucket: str
    prefix: str
    encryption_key: str  # Base64 AES key
    ws_endpoint: str
    version: int = TOKEN_VERSION
    expires_at: float = 0  # Unix ts; 0 -> DEFAULT_TOKEN_TTL is applied on encode

    # -- payload --

    def _payload(self) -> dict:
        return {
            "v": TOKEN_VERSION,
            "sid": self.share_id,
            "bkt": self.bucket,
            "pfx": self.prefix,
            "key": self.encryption_key,
            "ws": self.ws_endpoint,
            "s3": self.minio_endpoint,
            "exp": self.expires_at,
        }

    @staticmethod
    def _canonical(payload: dict) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _sign(payload_b64: str) -> str:
        mac = hmac.new(_signing_key(), payload_b64.encode("ascii"), hashlib.sha256)
        return _b64e(mac.digest())

    # -- encode / decode / verify --

    def encode(self) -> str:
        """
        Mint a signed v4 token. Requires the device key of this node.

        Raises:
            RuntimeError: if the device key is unavailable.
        """
        if not self.expires_at:
            self.expires_at = time.time() + DEFAULT_TOKEN_TTL
        payload_b64 = _b64e(self._canonical(self._payload()).encode("utf-8"))
        return f"v{TOKEN_VERSION}:{payload_b64}.{self._sign(payload_b64)}"

    @classmethod
    def _split(cls, token: str) -> tuple:
        prefix = f"v{TOKEN_VERSION}:"
        if not isinstance(token, str) or not token.startswith(prefix):
            raise ValueError(
                f"Invalid share token: expected a v{TOKEN_VERSION} token"
            )
        body = token[len(prefix):]
        if "." not in body:
            raise ValueError("Invalid share token: missing signature")
        payload_b64, signature = body.split(".", 1)
        if not payload_b64 or not signature:
            raise ValueError("Invalid share token: empty payload or signature")
        return payload_b64, signature

    @classmethod
    def _from_payload(cls, payload: dict) -> ShareToken:
        try:
            return cls(
                share_id=payload["sid"],
                minio_endpoint=payload["s3"],
                bucket=payload["bkt"],
                prefix=payload["pfx"],
                encryption_key=payload["key"],
                ws_endpoint=payload["ws"],
                expires_at=float(payload.get("exp", 0)),
                version=int(payload.get("v", TOKEN_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid share token: malformed payload ({exc})") from exc

    @classmethod
    def decode(cls, token: str) -> ShareToken:
        """
        Read a token WITHOUT verifying its signature.

        This is the client path: a client has no device key and therefore
        cannot verify anything. It only needs endpoints and the AES key to
        connect; the server decides whether the token is genuine.

        Raises:
            ValueError: on any malformed or non-v4 token.
        """
        payload_b64, _signature = cls._split(token)
        try:
            payload = json.loads(_b64d(payload_b64).decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Invalid share token: undecodable payload ({exc})") from exc
        return cls._from_payload(payload)

    @classmethod
    def verify(cls, token: str) -> ShareToken:
        """
        Verify signature and expiry, then return the token.

        Server path, and the only function that authenticates a token.
        Requires the device key of the node that issued the share.

        Raises:
            ValueError: on a malformed, unsigned, forged or expired token.
            RuntimeError: if the device key is unavailable.
        """
        payload_b64, signature = cls._split(token)
        if not hmac.compare_digest(signature, cls._sign(payload_b64)):
            raise ValueError("Invalid share token: signature mismatch")

        try:
            payload = json.loads(_b64d(payload_b64).decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Invalid share token: undecodable payload ({exc})") from exc

        tok = cls._from_payload(payload)
        if tok.expires_at and time.time() > tok.expires_at:
            raise ValueError("Invalid share token: expired")
        return tok

    def to_sync_config(self, vault_path: str, raw_token: str = "") -> SyncConfig:
        """
        Convert token data into a SyncConfig for the client.

        raw_token: the original encoded token string. It MUST be passed
        through, because the client cannot re-sign the token - only the
        issuing node can. Without it the AUTH message carries no token and
        the server rejects the connection.
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
        MINIO_ROOT_PASSWORD  (default: minioadmin)
        MINIO_SECURE         (default: false)
        LIVESYNC_WS_HOST     (default: 0.0.0.0)
        LIVESYNC_WS_PORT     (default: 8765)
        LIVESYNC_BUCKET      (default: tb-shared)
        LIVESYNC_URL_TTL     (default: 900) seconds a presigned URL stays valid
    """
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        "access_key": os.getenv("MINIO_ROOT_USER", "admin"),
        "secret_key": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "secure": os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes"),
        "ws_host": os.getenv("LIVESYNC_WS_HOST", "0.0.0.0"),
        "ws_port": int(os.getenv("LIVESYNC_WS_PORT", "8765")),
        "bucket": os.getenv("LIVESYNC_BUCKET", "tb-shared"),
        "url_ttl": int(os.getenv("LIVESYNC_URL_TTL", "900")),
        "ws_secure": os.getenv("LIVESYNC_WSS", "false").lower() in ("true", "1", "yes"),
        "ws_ssl_cert": os.getenv("LIVESYNC_SSL_CERT", ""),
        "ws_ssl_key": os.getenv("LIVESYNC_SSL_KEY", ""),
    }
