"""Sync service for background storage synchronization."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from registry.storage.manager import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class SyncJob:
    """Sync job information.

    Attributes:
        id: Job ID.
        remote_path: Path to sync.
        status: Job status.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        error: Error message if failed.
        retries: Number of retries.
    """

    id: str
    remote_path: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retries: int = 0


class SyncService:
    """Service for managing storage synchronization.

    Handles background sync between primary and mirror storage.

    Attributes:
        storage: Storage manager.
        max_retries: Maximum retry attempts.
    """

    def __init__(
        self,
        storage: StorageManager,
        max_retries: int = 3,
    ) -> None:
        """Initialize the service.

        Args:
            storage: Storage manager.
            max_retries: Maximum retry attempts.
        """
        self.storage = storage
        self.max_retries = max_retries
        self._jobs: dict[str, SyncJob] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the sync service."""
        self._running = True
        logger.info("Sync service started")

    async def stop(self) -> None:
        """Stop the sync service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Sync service stopped")

    async def queue_sync(self, remote_path: str) -> str:
        """Queue a file for synchronization.

        Args:
            remote_path: Path to sync.

        Returns:
            Job ID.
        """
        import uuid

        job_id = str(uuid.uuid4())
        job = SyncJob(id=job_id, remote_path=remote_path)
        self._jobs[job_id] = job

        logger.info(f"Queued sync job {job_id} for {remote_path}")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[SyncJob]:
        """Get status of a sync job.

        Args:
            job_id: Job ID.

        Returns:
            SyncJob or None if not found.
        """
        return self._jobs.get(job_id)

    async def get_pending_jobs(self) -> list[SyncJob]:
        """Get all pending sync jobs.

        Returns:
            List of pending jobs.
        """
        return [j for j in self._jobs.values() if j.status == "pending"]

    async def get_failed_jobs(self) -> list[SyncJob]:
        """Get all failed sync jobs.

        Returns:
            List of failed jobs.
        """
        return [j for j in self._jobs.values() if j.status == "failed"]

    async def retry_failed_jobs(self) -> int:
        """Retry all failed jobs.

        Returns:
            Number of jobs queued for retry.
        """
        count = 0
        for job in await self.get_failed_jobs():
            if job.retries < self.max_retries:
                job.status = "pending"
                job.retries += 1
                count += 1
        return count

    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs.

        Args:
            max_age_hours: Maximum age in hours.

        Returns:
            Number of jobs cleaned up.
        """
        cutoff = datetime.utcnow()
        count = 0
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.status == "completed" and job.completed_at:
                age = (cutoff - job.completed_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(job_id)
                    count += 1

        for job_id in to_remove:
            del self._jobs[job_id]

        return count

