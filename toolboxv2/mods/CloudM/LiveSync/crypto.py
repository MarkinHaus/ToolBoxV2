"""
LiveSync Crypto Layer
=====================
Client-side encryption: zlib compress → AES-256-GCM encrypt → MinIO.
Server never sees plaintext.

Wraps primitives from toolboxv2.utils.security.cryp (AESGCM path).
Falls back to direct cryptography usage if toolboxv2 not available.

Wire format:  [12-byte nonce][ciphertext + GCM tag]
    inner:    zlib-compressed original data
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import zlib
from pathlib import Path

# Try to import from toolboxv2; fall back to direct cryptography usage
try:
    from toolboxv2.utils.security.cryp import encrypt_with_key, decrypt_with_key
    _TB_CRYPTO = True
except ImportError:
    _TB_CRYPTO = False
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RuntimeError(
            "cryptography package required: pip install cryptography"
        )


# ── Key Generation ──

def generate_encryption_key() -> str:
    """
    Generate a new AES-256 key.

    Returns:
        URL-safe Base64-encoded 32-byte key string.
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def _decode_key(key_b64: str) -> bytes:
    """Decode a Base64-encoded key to raw bytes."""
    raw = base64.urlsafe_b64decode(key_b64)
    if len(raw) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(raw)}")
    return raw


# ── Low-level encrypt/decrypt ──

def _encrypt_raw(data: bytes, key: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns nonce (12B) + ciphertext."""
    if _TB_CRYPTO:
        return encrypt_with_key(data, key)
    else:
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, data, None)
        return nonce + ct


def _decrypt_raw(encrypted: bytes, key: bytes) -> bytes:
    """AES-256-GCM decrypt. Input: nonce (12B) + ciphertext."""
    if _TB_CRYPTO:
        return decrypt_with_key(encrypted, key)
    else:
        nonce = encrypted[:12]
        ct = encrypted[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None)


# ── Public API: bytes ──

def encrypt_bytes(data: bytes, key_b64: str) -> bytes:
    """
    Compress + encrypt arbitrary bytes.

    Args:
        data: raw plaintext bytes
        key_b64: Base64-encoded AES-256 key

    Returns:
        Encrypted blob (nonce + ciphertext of zlib-compressed data).
    """
    key = _decode_key(key_b64)
    compressed = zlib.compress(data, level=6)
    return _encrypt_raw(compressed, key)


def decrypt_bytes(encrypted: bytes, key_b64: str) -> bytes:
    """
    Decrypt + decompress bytes.

    Args:
        encrypted: blob produced by encrypt_bytes
        key_b64: Base64-encoded AES-256 key

    Returns:
        Original plaintext bytes.

    Raises:
        Exception on wrong key, tampered data, or corrupt zlib stream.
    """
    key = _decode_key(key_b64)
    compressed = _decrypt_raw(encrypted, key)
    return zlib.decompress(compressed)


# ── Public API: files ──

def encrypt_file(path: str, key_b64: str) -> bytes:
    """
    Read file from disk, compress + encrypt.

    Args:
        path: path to the file to encrypt
        key_b64: Base64-encoded AES-256 key

    Returns:
        Encrypted blob ready for MinIO upload.
    """
    with open(path, "rb") as f:
        data = f.read()
    return encrypt_bytes(data, key_b64)


def decrypt_to_file(encrypted: bytes, key_b64: str, dest_path: str) -> None:
    """
    Decrypt blob and write to disk atomically.

    Uses a .sync-tmp intermediate file + rename to prevent
    partial writes on crash / network interruption.

    Args:
        encrypted: blob from encrypt_bytes / encrypt_file
        key_b64: Base64-encoded AES-256 key
        dest_path: where to write the decrypted file
    """
    data = decrypt_bytes(encrypted, key_b64)

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest.with_suffix(dest.suffix + ".sync-tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        # Atomic rename (same filesystem)
        if dest.exists():
            dest.unlink()
        os.rename(tmp_path, dest)
    finally:
        # Clean up tmp if rename failed
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ── Checksum ──

def compute_checksum(data: bytes) -> str:
    """
    SHA-256 checksum, truncated to first 16 hex chars.

    Matches the spec: short enough for WS messages,
    long enough to avoid collisions in practice.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def compute_checksum_file(path: str) -> str:
    """Compute checksum of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]
