"""
LiveSync MinIO Helper
=====================
All MinIO I/O operations. Clients upload/download directly to MinIO —
WebSocket NEVER carries file content.

Bucket layout:
    livesync/
    └── {share_id}/
        ├── {rel_path}.enc          ← encrypted + compressed file
        └── .meta/
            └── {rel_path}.json     ← metadata (checksum, mtime, source_client)

CredentialBroker integration:
    Uses toolboxv2.mods.CloudM.auth.minio_policy.CredentialBroker
    to mint scoped per-client MinIO service accounts.
"""

from __future__ import annotations

import io
import json
import logging
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
    """Create bucket if it doesn't exist."""
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


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

def vend_credentials_for_share(share_id: str, env_config: dict) -> Dict[str, Any]:
    """
    Mint scoped MinIO credentials for a share using CredentialBroker.

    Falls back to admin credentials if broker is unavailable
    (with a warning — client will have broader access).

    Args:
        share_id: the share identifier
        env_config: dict from load_env_config() with endpoint, access_key, secret_key, secure

    Returns:
        Credential dict: {endpoint, access_key, secret_key, secure, bucket, prefix, ...}
    """
    try:
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
        creds = broker.vend_share_credentials(share_id)
        logger.info(f"Minted scoped credentials for share {share_id}")
        return creds

    except ImportError:
        logger.warning(
            "CredentialBroker not available — using admin credentials as fallback"
        )
        return {
            "endpoint": env_config["endpoint"],
            "access_key": env_config["access_key"],
            "secret_key": env_config["secret_key"],
            "secure": env_config.get("secure", False),
            "bucket": env_config.get("bucket", "livesync"),
            "prefix": share_id,
            "policy_applied": False,
            "warning": "CredentialBroker not available, using admin credentials",
        }
    except Exception as e:
        logger.error(f"CredentialBroker failed for share {share_id}: {e}")
        return {
            "endpoint": env_config["endpoint"],
            "access_key": env_config["access_key"],
            "secret_key": env_config["secret_key"],
            "secure": env_config.get("secure", False),
            "bucket": env_config.get("bucket", "livesync"),
            "prefix": share_id,
            "policy_applied": False,
            "warning": f"CredentialBroker error: {e}",
        }
