"""
LiveSync MinIO Helper
=====================
All MinIO I/O operations. Clients upload/download directly to MinIO —
WebSocket NEVER carries file content.

Bucket layout:
    tb-shared/
    └── {share_id}/
        ├── {rel_path}.enc          ← encrypted + compressed file
        └── .meta/
            └── {rel_path}.json     ← metadata (checksum, mtime, source_client)

Access model:
    The SERVER holds the MinIO credentials. Clients never do. Every client
    transfer runs against a short-lived presigned URL that the server issues
    for exactly one object (see presign_get / presign_put) and that the client
    fetches over plain HTTP (http_download / http_upload).
"""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None  # type: ignore

logger = logging.getLogger("LiveSync")


# ── Client Factory ──

def create_minio_client(creds: Dict[str, Any]) -> "Minio":
    """
    Create a MinIO client from credentials dict.

    Args:
        creds: {endpoint, access_key, secret_key, secure}

    Returns:
        Minio client instance.
    """
    if not MINIO_AVAILABLE:
        raise RuntimeError("minio package required: pip install minio")
    return Minio(
        creds["endpoint"],
        access_key=creds["access_key"],
        secret_key=creds["secret_key"],
        secure=creds.get("secure", False),
    )


# ── Key Helpers ──

def make_object_key(share_prefix: str, rel_path: str) -> str:
    """Build the MinIO object key for an encrypted file."""
    return f"{share_prefix}/{rel_path}.enc"


def make_meta_key(share_prefix: str, rel_path: str) -> str:
    """Build the MinIO object key for file metadata."""
    return f"{share_prefix}/.meta/{rel_path}.json"


def rel_path_from_object_key(
    share_prefix: str, object_key: str
) -> Optional[str]:
    """
    Extract relative path from an object key.
    Returns None for non-.enc files or .meta/ paths.
    """
    prefix = f"{share_prefix}/"
    if not object_key.startswith(prefix):
        return None
    remainder = object_key[len(prefix):]
    # Skip metadata objects
    if remainder.startswith(".meta/"):
        return None
    if not remainder.endswith(".enc"):
        return None
    return remainder[:-4]  # strip .enc


# ── Bucket Management ──

def ensure_bucket(client: "Minio", bucket: str) -> None:
    """Create bucket if it doesn't exist and enable versioning."""
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    # Enable versioning to protect against accidental deletes (FIX)
    try:
        from minio.versioningconfig import VersioningConfig, ENABLED
        status = client.get_bucket_versioning(bucket)
        if status.status != "Enabled":
            client.set_bucket_versioning(bucket, VersioningConfig(ENABLED))
            logger.info(f"[LiveSync] Bucket versioning enabled: {bucket}")
    except ImportError:
        logger.warning("[LiveSync] minio.versioningconfig not available - versioning skipped")
    except Exception as e:
        logger.warning(f"[LiveSync] Versioning enable failed for {bucket}: {e}")


# ── Upload ──

def upload_bytes(
    client: "Minio",
    bucket: str,
    key: str,
    data: bytes,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Upload raw bytes to MinIO.

    Args:
        client: MinIO client
        bucket: bucket name
        key: object key
        data: raw bytes to upload
        metadata: optional S3 metadata headers
    """
    client.put_object(
        bucket,
        key,
        io.BytesIO(data),
        len(data),
        metadata=metadata,
    )


def upload_metadata(
    client: "Minio",
    bucket: str,
    share_prefix: str,
    rel_path: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Upload file metadata JSON to the .meta/ prefix.

    Args:
        client: MinIO client
        bucket: bucket name
        share_prefix: share ID prefix
        rel_path: relative file path
        metadata: dict with checksum, mtime, source_client, etc.
    """
    key = make_meta_key(share_prefix, rel_path)
    data = json.dumps(metadata).encode("utf-8")
    client.put_object(bucket, key, io.BytesIO(data), len(data))


# ── Download ──

def download_bytes(client: "Minio", bucket: str, key: str) -> bytes:
    """
    Download raw bytes from MinIO.

    Returns:
        The raw object bytes.

    Raises:
        S3Error on missing object or auth failure.
    """
    resp = client.get_object(bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def download_metadata(
    client: "Minio", bucket: str, share_prefix: str, rel_path: str
) -> Optional[Dict[str, Any]]:
    """Download and parse file metadata JSON. Returns None if not found."""
    key = make_meta_key(share_prefix, rel_path)
    try:
        data = download_bytes(client, bucket, key)
        return json.loads(data)
    except Exception:
        return None


# ── Delete ──

def delete_object(client: "Minio", bucket: str, key: str) -> None:
    """Delete a single object from MinIO."""
    client.remove_object(bucket, key)


def delete_file_and_meta(
    client: "Minio", bucket: str, share_prefix: str, rel_path: str
) -> None:
    """Delete both the encrypted file and its metadata."""
    try:
        client.remove_object(bucket, make_object_key(share_prefix, rel_path))
    except Exception:
        pass
    try:
        client.remove_object(bucket, make_meta_key(share_prefix, rel_path))
    except Exception:
        pass


# ── List ──

def list_remote_files(
    client: "Minio", bucket: str, share_prefix: str
) -> Dict[str, Dict[str, Any]]:
    """
    List all encrypted files in a share prefix.

    Returns:
        {rel_path: {"minio_key": ..., "mtime": ..., "size": ...}}
    """
    result: Dict[str, Dict[str, Any]] = {}
    objects = client.list_objects(bucket, prefix=f"{share_prefix}/", recursive=True)

    for obj in objects:
        rel_path = rel_path_from_object_key(share_prefix, obj.object_name)
        if rel_path is None:
            continue
        result[rel_path] = {
            "minio_key": obj.object_name,
            "mtime": obj.last_modified.timestamp() if obj.last_modified else 0,
            "size": obj.size,
        }
    return result


# ── Presigned URLs (server issues, client consumes) ──

def presign_get(
    client: "Minio", bucket: str, key: str, ttl_seconds: int = 900
) -> str:
    """
    Presigned download URL for exactly one object.

    Args:
        client: MinIO client holding the server credentials
        bucket: bucket name
        key: object key
        ttl_seconds: how long the URL stays valid

    Returns:
        Absolute HTTP(S) URL usable without any credentials.
    """
    return client.presigned_get_object(
        bucket, key, expires=timedelta(seconds=ttl_seconds)
    )


def presign_put(
    client: "Minio", bucket: str, key: str, ttl_seconds: int = 900
) -> str:
    """
    Presigned upload URL for exactly one object.

    Args:
        client: MinIO client holding the server credentials
        bucket: bucket name
        key: object key
        ttl_seconds: how long the URL stays valid

    Returns:
        Absolute HTTP(S) URL usable without any credentials.
    """
    return client.presigned_put_object(
        bucket, key, expires=timedelta(seconds=ttl_seconds)
    )


def object_exists(client: "Minio", bucket: str, key: str) -> bool:
    """True if the object is present in the bucket."""
    try:
        client.stat_object(bucket, key)
        return True
    except Exception:
        return False


# ── Credential-free HTTP transfer (client side) ──

class TransferError(RuntimeError):
    """A presigned transfer failed. Carries the HTTP status when known."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def http_download(url: str, timeout: float = 120.0) -> bytes:
    """
    Download an object through a presigned URL.

    Blocking on purpose - call it from a worker thread
    (``asyncio.to_thread``) so the event loop stays free.

    Raises:
        TransferError: on any HTTP or transport failure.
    """
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise TransferError(
            f"download failed: HTTP {exc.code} {exc.reason}", exc.code
        ) from exc
    except Exception as exc:
        raise TransferError(f"download failed: {exc}") from exc


def http_upload(url: str, data: bytes, timeout: float = 120.0) -> None:
    """
    Upload an object through a presigned URL.

    Blocking on purpose - call it from a worker thread
    (``asyncio.to_thread``) so the event loop stays free.

    Raises:
        TransferError: on any HTTP or transport failure.
    """
    req = urllib.request.Request(
        url, data=data, method="PUT",
        headers={"Content-Length": str(len(data)),
                 "Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise TransferError(
            f"upload failed: HTTP {exc.code} {exc.reason}", exc.code
        ) from exc
    except Exception as exc:
        raise TransferError(f"upload failed: {exc}") from exc

    if status not in (200, 201, 204):
        raise TransferError(f"upload failed: unexpected HTTP {status}", status)


# ── Healthcheck ──

def healthcheck(client: "Minio") -> Tuple[bool, str]:
    """
    Check MinIO connectivity.

    Returns:
        (ok: bool, message: str)
    """
    try:
        client.list_buckets()
        return True, "MinIO OK"
    except Exception as e:
        return False, f"MinIO unreachable: {e}"


# ── CredentialBroker Integration ──

def vend_user_credentials_for_user(user_id: str, env_config: dict) -> Dict[str, Any]:
    """
    Mint scoped MinIO credentials for a user using CredentialBroker.

    Uses the user-scoped policy (``tb-users-private/{user_id}/*`` + RO on
    ``tb-users-public/*``). See
    ``toolboxv2.mods.CloudM.auth.minio_policy.CredentialBroker``.

    This is the dashboard path for a user's OWN storage - unrelated to
    LiveSync shares, which use presigned URLs and hand out no credentials.
    It has no admin fallback: a broker failure raises.

    Args:
        user_id: the user identifier
        env_config: dict from ``load_env_config()`` with endpoint,
            access_key, secret_key, secure

    Returns:
        Credential dict: {endpoint, access_key, secret_key, secure,
            buckets, user_prefix, policy_applied, expires_in}

    Raises:
        ValueError: if required env fields are missing
        RuntimeError: if minio package is unavailable
    """
    if not user_id:
        raise ValueError("user_id required")
    for field in ("endpoint", "access_key", "secret_key"):
        if not env_config.get(field):
            raise ValueError(f"env_config.{field} required")

    from toolboxv2.mods.CloudM.auth.minio_policy import (
        CredentialBroker,
        MinIOPolicyConfig,
    )

    config = MinIOPolicyConfig(
        endpoint=env_config["endpoint"],
        access_key=env_config["access_key"],
        secret_key=env_config["secret_key"],
        secure=env_config.get("secure", False),
    )
    broker = CredentialBroker(config)
    creds = broker.vend_user_credentials(user_id)
    logger.info(f"Minted scoped credentials for user {user_id}")
    logger.info(
        f"Minted scoped credentials for user {user_id}",
        extra={"audit_action": "CREDENTIAL_VEND", "user_id": user_id, "scoped": True}
    )
    return creds
