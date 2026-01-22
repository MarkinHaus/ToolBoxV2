"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Optional

from registry.models.package import StorageLocation


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage implementations must inherit from this class
    and implement the abstract methods.

    Attributes:
        name: Unique name for this storage backend.
    """

    name: str = "base"

    @abstractmethod
    async def upload(
        self,
        local_path: Path,
        remote_path: str,
    ) -> StorageLocation:
        """Upload a file to storage.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path in storage.

        Returns:
            StorageLocation with upload details.

        Raises:
            StorageError: If upload fails.
        """
        ...

    @abstractmethod
    async def download(
        self,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Download a file from storage.

        Args:
            remote_path: Path in storage.
            local_path: Destination local path.

        Returns:
            True if successful, False otherwise.

        Raises:
            StorageError: If download fails.
        """
        ...

    @abstractmethod
    async def get_url(
        self,
        remote_path: str,
        expires_in: int = 3600,
    ) -> Optional[str]:
        """Get a presigned URL for downloading.

        Args:
            remote_path: Path in storage.
            expires_in: URL expiration time in seconds.

        Returns:
            Presigned URL or None if not available.
        """
        ...

    @abstractmethod
    async def exists(self, remote_path: str) -> bool:
        """Check if a file exists in storage.

        Args:
            remote_path: Path in storage.

        Returns:
            True if file exists, False otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, remote_path: str) -> bool:
        """Delete a file from storage.

        Args:
            remote_path: Path in storage.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def stream(self, remote_path: str) -> AsyncIterator[bytes]:
        """Stream file contents.

        Default implementation downloads to temp file and streams.
        Override for more efficient streaming.

        Args:
            remote_path: Path in storage.

        Yields:
            File content chunks.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if await self.download(remote_path, tmp_path):
                with open(tmp_path, "rb") as f:
                    while chunk := f.read(8192):
                        yield chunk
        finally:
            tmp_path.unlink(missing_ok=True)

