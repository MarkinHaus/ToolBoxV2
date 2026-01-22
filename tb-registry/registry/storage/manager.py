"""Storage manager for coordinating multiple backends."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from registry.config import Settings
from registry.models.package import StorageLocation
from registry.storage.base import StorageBackend
from registry.storage.minio_mirror import MinioMirrorBackend
from registry.storage.minio_primary import MinioPrimaryBackend

logger = logging.getLogger(__name__)


@dataclass
class SyncTask:
    """Task for syncing a file to mirrors.

    Attributes:
        remote_path: Path in storage.
        source_backend: Source backend name.
    """

    remote_path: str
    source_backend: str


class StorageManager:
    """Manager for coordinating storage backends.

    Handles uploads to primary storage and background sync to mirrors.

    Attributes:
        primary: Primary storage backend.
        mirrors: List of mirror backends.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the storage manager.

        Args:
            config: Application settings.
        """
        self.primary = MinioPrimaryBackend(
            endpoint=config.minio_primary_endpoint,
            access_key=config.minio_primary_access_key,
            secret_key=config.minio_primary_secret_key,
            bucket=config.minio_primary_bucket,
            secure=config.minio_primary_secure,
        )

        self.mirrors: list[StorageBackend] = []
        if config.has_mirror:
            self.mirrors.append(
                MinioMirrorBackend(
                    endpoint=config.minio_mirror_endpoint,
                    access_key=config.minio_mirror_access_key,
                    secret_key=config.minio_mirror_secret_key,
                    bucket=config.minio_mirror_bucket,
                    secure=config.minio_mirror_secure,
                )
            )

        self._sync_queue: asyncio.Queue[SyncTask] = asyncio.Queue()
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the storage manager and sync worker."""
        self._running = True
        if self.mirrors:
            self._sync_task = asyncio.create_task(self._sync_worker())
            logger.info("Storage sync worker started")

    async def stop(self) -> None:
        """Stop the storage manager and sync worker."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Storage sync worker stopped")

    def health_check(self) -> dict:
        """Check storage connectivity.

        Returns:
            Dict with health status for primary and mirrors.
        """
        result = {
            "healthy": True,
            "primary": {"name": self.primary.name, "healthy": False},
            "mirrors": [],
        }

        # Check primary storage
        try:
            bucket_exists = self.primary.client.bucket_exists(self.primary.bucket)
            result["primary"]["healthy"] = bucket_exists
            if not bucket_exists:
                result["healthy"] = False
        except Exception as e:
            logger.error(f"Primary storage health check failed: {e}")
            result["primary"]["healthy"] = False
            result["primary"]["error"] = str(e)
            result["healthy"] = False

        # Check mirrors
        for mirror in self.mirrors:
            mirror_status = {"name": mirror.name, "healthy": False}
            try:
                bucket_exists = mirror.client.bucket_exists(mirror.bucket)
                mirror_status["healthy"] = bucket_exists
            except Exception as e:
                logger.warning(f"Mirror storage health check failed: {e}")
                mirror_status["error"] = str(e)
            result["mirrors"].append(mirror_status)

        return result

    async def upload(
        self,
        local_path: Path,
        remote_path: str,
        sync_to_mirrors: bool = True,
    ) -> list[StorageLocation]:
        """Upload a file to storage.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path in storage.
            sync_to_mirrors: Whether to sync to mirrors.

        Returns:
            List of storage locations.
        """
        locations = []

        # Upload to primary
        primary_loc = await self.primary.upload(local_path, remote_path)
        locations.append(primary_loc)

        # Queue mirror sync
        if sync_to_mirrors and self.mirrors:
            await self._sync_queue.put(
                SyncTask(remote_path=remote_path, source_backend=self.primary.name)
            )

        return locations

    async def download(
        self,
        remote_path: str,
        local_path: Path,
        locations: Optional[list[StorageLocation]] = None,
    ) -> bool:
        """Download a file from storage.

        Tries primary first, then mirrors on failure.

        Args:
            remote_path: Path in storage.
            local_path: Destination local path.
            locations: Optional list of known locations.

        Returns:
            True if successful, False otherwise.
        """
        # Try primary first
        if await self.primary.download(remote_path, local_path):
            return True

        # Try mirrors
        for mirror in self.mirrors:
            if await mirror.download(remote_path, local_path):
                logger.info(f"Downloaded from mirror: {mirror.name}")
                return True

        return False

    async def get_download_url(
        self,
        remote_path: str,
        expires_in: int = 3600,
        prefer_mirror: bool = False,
    ) -> Optional[str]:
        """Get a presigned download URL.

        Args:
            remote_path: Path in storage.
            expires_in: URL expiration time in seconds.
            prefer_mirror: Prefer mirror URL if available.

        Returns:
            Presigned URL or None.
        """
        if prefer_mirror and self.mirrors:
            for mirror in self.mirrors:
                url = await mirror.get_url(remote_path, expires_in)
                if url:
                    return url

        return await self.primary.get_url(remote_path, expires_in)

    async def exists(self, remote_path: str) -> bool:
        """Check if a file exists in storage.

        Args:
            remote_path: Path in storage.

        Returns:
            True if file exists in primary.
        """
        return await self.primary.exists(remote_path)

    async def delete(self, remote_path: str) -> bool:
        """Delete a file from all storage backends.

        Args:
            remote_path: Path in storage.

        Returns:
            True if deleted from primary.
        """
        result = await self.primary.delete(remote_path)

        # Also delete from mirrors
        for mirror in self.mirrors:
            await mirror.delete(remote_path)

        return result

    async def _sync_worker(self) -> None:
        """Background worker for syncing files to mirrors."""
        import tempfile

        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._sync_queue.get(),
                    timeout=1.0,
                )

                # Download from primary to temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    if await self.primary.download(task.remote_path, tmp_path):
                        # Upload to all mirrors
                        for mirror in self.mirrors:
                            try:
                                await mirror.upload(tmp_path, task.remote_path)
                                logger.info(
                                    f"Synced {task.remote_path} to {mirror.name}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to sync to {mirror.name}: {e}"
                                )
                finally:
                    tmp_path.unlink(missing_ok=True)

                self._sync_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync worker error: {e}")

    @staticmethod
    async def calculate_checksum(path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Args:
            path: Path to the file.

        Returns:
            Hex-encoded SHA256 checksum.
        """
        return await MinioPrimaryBackend._calculate_checksum(path)

