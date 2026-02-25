"""
Secure ZIP extraction with path traversal and symlink protection.

This module provides secure ZIP extraction that protects against:
- Path traversal attacks (.. in filenames)
- Symlink attacks (symlinks escaping destination)
- ZIP bombs (compression ratio, size limits)

Classes:
    ZipSecurityError: Base exception for security errors
    ZipPathTraversalError: Path traversal attempt detected
    ZipSymlinkError: Suspicious symlink detected
    ZipSizeError: Size limit exceeded

Functions:
    is_path_within_base(path: Path, base: Path) -> bool
        Check if path is within base directory
    extract_zip_securely(zip_path: Path, dest_dir: Path, ...) -> None
        Extract ZIP with all security checks
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)


class ZipSecurityError(Exception):
    """Base class for ZIP security errors."""
    pass


class ZipPathTraversalError(ZipSecurityError):
    """Raised when ZIP member path escapes destination."""
    pass


class ZipSymlinkError(ZipSecurityError):
    """Raised when ZIP symlink is suspicious."""
    pass


class ZipSizeError(ZipSecurityError):
    """Raised when ZIP extraction exceeds size limit."""
    pass


def is_path_within_base(path: Path, base: Path) -> bool:
    """Check if path is within base directory.

    Args:
        path: Path to check
        base: Base directory path

    Returns:
        True if path is within base, False otherwise
    """
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_zip_securely(
    zip_path: Path,
    dest_dir: Path,
    max_size_bytes: Optional[int] = None,
    max_compression_ratio: int = 100,  # 100x compressed vs uncompressed
    allowed_members: Optional[Set[str]] = None,
) -> None:
    """
    Extract ZIP file with security checks.

    Security measures:
    1. Path traversal protection
    2. Symlink validation
    3. Size limits
    4. Compression ratio checks (ZIP bomb protection)

    Args:
        zip_path: Path to ZIP file
        dest_dir: Destination directory (must exist)
        max_size_bytes: Maximum total extraction size (default: 1GB)
        max_compression_ratio: Max compression ratio (default: 100)
        allowed_members: Optional whitelist of member paths

    Raises:
        ZipSecurityError: If security check fails
    """
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if max_size_bytes is None:
        max_size_bytes = 1 * 1024 * 1024 * 1024  # 1GB default

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Phase 1: Validate all members before extraction
        total_uncompressed_size = 0
        total_compressed_size = 0

        for member in zip_ref.infolist():
            # Check filename length (ZIP bomb protection)
            if len(member.filename) > 255:
                raise ZipSecurityError(
                    f"Suspicious filename length: {len(member.filename)}"
                )

            # Check for absolute paths (path traversal)
            if os.path.isabs(member.filename):
                raise ZipPathTraversalError(
                    f"Absolute path not allowed: {member.filename}"
                )

            # Check against whitelist if provided
            if allowed_members is not None:
                if member.filename not in allowed_members:
                    logger.debug(f"Skipping non-whitelisted member: {member.filename}")
                    continue

            # Path traversal check
            member_path = (dest_dir / member.filename).resolve()
            if not is_path_within_base(member_path, dest_dir):
                raise ZipPathTraversalError(
                    f"Path traversal attempt: {member.filename}"
                )

            # Check for suspicious symlinks (Unix symlinks have specific mode in external_attr)
            # Symlink check: Unix file type bits are in high 16 bits of external_attr
            # 0xA000 is the file type for symlink (S_IFLNK)
            is_symlink = (member.external_attr >> 16) & 0xF000 == 0xA000

            if is_symlink:
                target = zip_ref.read(member.filename).decode('utf-8', errors='replace')
                if os.path.isabs(target):
                    raise ZipSymlinkError(
                        f"Absolute symlink target: {member.filename} -> {target}"
                    )
                target_path = (dest_dir / target).resolve()
                if not is_path_within_base(target_path, dest_dir):
                    raise ZipSymlinkError(
                        f"Suspicious symlink: {member.filename} -> {target}"
                    )

            # Size checks (ZIP bomb protection)
            total_uncompressed_size += member.file_size
            total_compressed_size += member.compress_size

            # Check single file size
            if member.file_size > 100 * 1024 * 1024:  # 100MB per file limit
                raise ZipSizeError(
                    f"File too large: {member.filename} ({member.file_size} bytes)"
                )

            # Check compression ratio
            if member.compress_size > 0:
                ratio = member.file_size / member.compress_size
                if ratio > max_compression_ratio:
                    raise ZipSizeError(
                        f"Compression ratio too high: {member.filename} ({ratio:.1f}x)"
                    )

            # Check total size
            if total_uncompressed_size > max_size_bytes:
                raise ZipSizeError(
                    f"Total extraction size exceeds limit: "
                    f"{total_uncompressed_size} > {max_size_bytes}"
                )

        # Phase 2: Extract (safe after validation)
        for member in zip_ref.infolist():
            if allowed_members is not None and member.filename not in allowed_members:
                continue

            # Extract member safely
            zip_ref.extract(member, dest_dir)

            # Set safe permissions (no executable bits from ZIP)
            extracted_path = dest_dir / member.filename
            if extracted_path.is_file():
                os.chmod(extracted_path, 0o644)

        logger.info(f"Securely extracted {len(zip_ref.infolist())} members to {dest_dir}")
