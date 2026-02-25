"""
bsdiff integration for binary diff creation and application.

This module provides a Python wrapper around bsdiff/bspatch for
creating and applying binary diffs between package versions.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BsdiffError(Exception):
    """Base exception for bsdiff operations."""
    pass


class BsdiffNotFoundError(BsdiffError):
    """Raised when bsdiff executable is not found."""
    pass


def check_bsdiff_available() -> bool:
    """
    Check if bsdiff/bspatch executables are available.

    Returns:
        True if both bsdiff and bspatch are available
    """
    try:
        result = subprocess.run(
            ["bsdiff", "--help"],
            capture_output=True,
            timeout=1,
        )
        return result.returncode == 0 or b"bsdiff" in result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_patch(
    old_file: Path,
    new_file: Path,
    patch_file: Path,
) -> int:
    """
    Create a binary patch from old_file to new_file.

    Args:
        old_file: Path to original file
        new_file: Path to new file
        patch_file: Path where patch will be written

    Returns:
        Size of the created patch in bytes

    Raises:
        BsdiffNotFoundError: If bsdiff is not installed
        BsdiffError: If patch creation fails
    """
    if not check_bsdiff_available():
        raise BsdiffNotFoundError(
            "bsdiff not found. Install with: apt-get install bsdiff (Linux) "
            "or brew install bsdiff (macOS)"
        )

    if not old_file.exists():
        raise BsdiffError(f"Old file not found: {old_file}")
    if not new_file.exists():
        raise BsdiffError(f"New file not found: {new_file}")

    try:
        # Create parent directories for patch file
        patch_file.parent.mkdir(parents=True, exist_ok=True)

        # Run bsdiff
        result = subprocess.run(
            [
                "bsdiff",
                str(old_file),
                str(new_file),
                str(patch_file),
            ],
            capture_output=True,
            timeout=300,  # 5 minute timeout for large files
        )

        if result.returncode != 0:
            raise BsdiffError(
                f"bsdiff failed with code {result.returncode}: {result.stderr.decode()}"
            )

        patch_size = patch_file.stat().st_size
        logger.debug(f"Created patch: {patch_file} ({patch_size} bytes)")

        return patch_size

    except subprocess.TimeoutExpired:
        raise BsdiffError("bsdiff timed out after 5 minutes")
    except Exception as e:
        raise BsdiffError(f"Failed to create patch: {e}")


def apply_patch(
    old_file: Path,
    patch_file: Path,
    new_file: Path,
) -> int:
    """
    Apply a binary patch to old_file to create new_file.

    Args:
        old_file: Path to original file
        patch_file: Path to patch file
        new_file: Path where patched file will be written

    Returns:
        Size of the created new file in bytes

    Raises:
        BsdiffNotFoundError: If bspatch is not installed
        BsdiffError: If patch application fails
    """
    if not check_bsdiff_available():
        raise BsdiffNotFoundError(
            "bspatch not found. Install with: apt-get install bsdiff (Linux) "
            "or brew install bsdiff (macOS)"
        )

    if not old_file.exists():
        raise BsdiffError(f"Old file not found: {old_file}")
    if not patch_file.exists():
        raise BsdiffError(f"Patch file not found: {patch_file}")

    try:
        # Create parent directories for new file
        new_file.parent.mkdir(parents=True, exist_ok=True)

        # Run bspatch
        result = subprocess.run(
            [
                "bspatch",
                str(old_file),
                str(new_file),
                str(patch_file),
            ],
            capture_output=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise BsdiffError(
                f"bspatch failed with code {result.returncode}: {result.stderr.decode()}"
            )

        new_size = new_file.stat().st_size
        logger.debug(f"Applied patch: {old_file} + {patch_file} -> {new_file} ({new_size} bytes)")

        return new_size

    except subprocess.TimeoutExpired:
        raise BsdiffError("bspatch timed out after 5 minutes")
    except Exception as e:
        raise BsdiffError(f"Failed to apply patch: {e}")


def estimate_patch_size(old_file: Path, new_file: Path) -> Optional[int]:
    """
    Estimate patch size without creating the full patch.

    This is a fast heuristic for deciding whether to use diff upload.
    For actual patch, use create_patch().

    Args:
        old_file: Path to original file
        new_file: Path to new file

    Returns:
        Estimated patch size in bytes, or None if cannot estimate
    """
    try:
        import hashlib

        # Calculate file sizes
        old_size = old_file.stat().st_size
        new_size = new_file.stat().st_size

        # If sizes are very different, patch will be larger
        size_ratio = min(old_size, new_size) / max(old_size, new_size)

        if size_ratio < 0.5:
            # Very different sizes - patch will be ~50% of new file
            return new_size // 2

        # Similar sizes - estimate based on content difference
        # Read first and last 1KB of each file
        sample_size = 1024

        with open(old_file, 'rb') as f:
            old_start = f.read(sample_size)
            f.seek(-sample_size, 2)
            old_end = f.read(sample_size)

        with open(new_file, 'rb') as f:
            new_start = f.read(sample_size)
            f.seek(-sample_size, 2)
            new_end = f.read(sample_size)

        # Calculate difference in samples
        diff_bytes = 0
        total_bytes = len(old_start) + len(old_end)

        for a, b in [(old_start, new_start), (old_end, new_end)]:
            for byte_a, byte_b in zip(a, b):
                if byte_a != byte_b:
                    diff_bytes += 1

        diff_ratio = diff_bytes / total_bytes if total_bytes > 0 else 0

        # Estimate: ~20% base + diff_ratio * new_size
        estimated = int(new_size * (0.2 + diff_ratio * 0.8))

        return estimated

    except Exception as e:
        logger.warning(f"Failed to estimate patch size: {e}")
        return None


# Fallback implementation using pure Python (slower but always available)
class PurePythonDiff:
    """
    Pure Python diff implementation as fallback when bsdiff is not available.

    This is slower than bsdiff but doesn't require external dependencies.
    Uses a simple byte-level diffing approach.
    """

    @staticmethod
    def create_patch(
        old_file: Path,
        new_file: Path,
        patch_file: Path,
    ) -> int:
        """
        Create a patch using pure Python.

        The patch format is simple:
        - 4 bytes: number of chunks
        - For each chunk:
            - 8 bytes: offset in old file
            - 8 bytes: length to copy
            - 4 bytes: new data length
            - N bytes: new data
        """
        old_data = old_file.read_bytes()
        new_data = new_file.read_bytes()

        # Simple diff: find matching chunks
        chunk_size = 4096
        chunks = []

        old_pos = 0
        new_pos = 0

        while new_pos < len(new_data):
            # Look for matching chunk in old file
            best_match = None
            best_match_len = 0
            best_match_pos = 0

            for i in range(max(0, old_pos - 10000), min(len(old_data), old_pos + 10000)):
                match_len = 0
                while (
                    new_pos + match_len < len(new_data) and
                    i + match_len < len(old_data) and
                    new_data[new_pos + match_len] == old_data[i + match_len]
                ):
                    match_len += 1
                    if match_len > chunk_size:
                        break

                if match_len > best_match_len:
                    best_match_len = match_len
                    best_match_pos = i
                    if match_len >= 64:  # Good enough match
                        break

            if best_match_len >= 64:
                # Found good match - use copy operation
                if new_pos > old_pos:
                    # Insert new data before this match
                    chunks.append({
                        'type': 'insert',
                        'data': new_data[old_pos:new_pos]
                    })

                chunks.append({
                    'type': 'copy',
                    'offset': best_match_pos,
                    'length': best_match_len,
                })

                new_pos += best_match_len
                old_pos = new_pos
            else:
                # No good match - insert byte
                chunks.append({
                    'type': 'insert',
                    'data': new_data[new_pos:new_pos + 1],
                })
                new_pos += 1
                old_pos = new_pos

        # Write patch file
        import struct

        with open(patch_file, 'wb') as f:
            # Write magic and version
            f.write(b'PYD1')  # Python Diff v1

            # Write number of chunks
            f.write(struct.pack('<I', len(chunks)))

            # Write chunks
            for chunk in chunks:
                if chunk['type'] == 'copy':
                    f.write(struct.pack('<B', 1))  # Copy operation
                    f.write(struct.pack('<Q', chunk['offset']))
                    f.write(struct.pack('<Q', chunk['length']))
                else:
                    f.write(struct.pack('<B', 2))  # Insert operation
                    data = chunk['data']
                    f.write(struct.pack('<I', len(data)))
                    f.write(data)

        return patch_file.stat().st_size

    @staticmethod
    def apply_patch(
        old_file: Path,
        patch_file: Path,
        new_file: Path,
    ) -> int:
        """Apply a patch created with pure Python diff."""
        import struct

        old_data = old_file.read_bytes()
        patch_data = patch_file.read_bytes()

        # Verify magic
        if patch_data[:4] != b'PYD1':
            raise BsdiffError("Invalid patch format (magic mismatch)")

        pos = 4

        # Read number of chunks
        num_chunks = struct.unpack('<I', patch_data[pos:pos + 4])[0]
        pos += 4

        # Apply chunks
        result = bytearray()

        for _ in range(num_chunks):
            op_type = struct.unpack('<B', patch_data[pos:pos + 1])[0]
            pos += 1

            if op_type == 1:  # Copy
                offset = struct.unpack('<Q', patch_data[pos:pos + 8])[0]
                pos += 8
                length = struct.unpack('<Q', patch_data[pos:pos + 8])[0]
                pos += 8

                result.extend(old_data[offset:offset + length])

            elif op_type == 2:  # Insert
                data_len = struct.unpack('<I', patch_data[pos:pos + 4])[0]
                pos += 4

                result.extend(patch_data[pos:pos + data_len])
                pos += data_len

        # Write result
        new_file.write_bytes(result)

        return len(result)


def create_patch_auto(
    old_file: Path,
    new_file: Path,
    patch_file: Path,
    fallback: bool = True,
) -> int:
    """
    Create patch, using bsdiff if available, otherwise pure Python.

    Args:
        old_file: Path to original file
        new_file: Path to new file
        patch_file: Path where patch will be written
        fallback: If True, use pure Python when bsdiff unavailable

    Returns:
        Size of created patch

    Raises:
        BsdiffError: If patch creation fails
    """
    if check_bsdiff_available():
        logger.debug("Using bsdiff for patch creation")
        return create_patch(old_file, new_file, patch_file)

    if fallback:
        logger.warning("bsdiff unavailable, using pure Python diff (slower)")
        return PurePythonDiff.create_patch(old_file, new_file, patch_file)

    raise BsdiffNotFoundError("bsdiff not available and fallback disabled")


def apply_patch_auto(
    old_file: Path,
    patch_file: Path,
    new_file: Path,
    fallback: bool = True,
) -> int:
    """
    Apply patch, using bsdiff if available, otherwise pure Python.

    Args:
        old_file: Path to original file
        patch_file: Path to patch file
        new_file: Path where patched file will be written
        fallback: If True, use pure Python when bsdiff unavailable

    Returns:
        Size of created new file

    Raises:
        BsdiffError: If patch application fails
    """
    # Check patch format
    with open(patch_file, 'rb') as f:
        magic = f.read(4)

    if magic == b'PYD1':
        # Pure Python patch format
        return PurePythonDiff.apply_patch(old_file, patch_file, new_file)

    if check_bsdiff_available():
        logger.debug("Using bspatch for patch application")
        return apply_patch(old_file, patch_file, new_file)

    if fallback:
        raise BsdiffError("Cannot apply bsdiff patch without bspatch (fallback not supported)")

    raise BsdiffNotFoundError("bspatch not available and fallback disabled")
