"""MinIO primary storage backend."""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from minio import Minio
from minio.error import S3Error

from registry.exceptions import StorageError
from registry.models.package import StorageLocation
from registry.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class MinioPrimaryBackend(StorageBackend):
    """MinIO primary storage backend.

    Implements storage operations using MinIO S3-compatible storage.

    Attributes:
        name: Backend name identifier.
        client: MinIO client instance.
        bucket: Bucket name.
    """

    name: str = "minio-primary"

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        """Initialize the MinIO backend.

        Args:
            endpoint: MinIO server endpoint.
            access_key: Access key for authentication.
            secret_key: Secret key for authentication.
            bucket: Bucket name to use.
            secure: Use HTTPS connection.
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Ensure the bucket exists, create if not."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket: {e}")
            raise StorageError(str(e), "bucket_check")

    async def upload(
        self,
        local_path: Path,
        remote_path: str,
    ) -> StorageLocation:
        """Upload a file to MinIO.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path in storage.

        Returns:
            StorageLocation with upload details.

        Raises:
            StorageError: If upload fails.
        """
        try:
            # Calculate checksum
            checksum = await self._calculate_checksum(local_path)
            size = local_path.stat().st_size

            # Upload file
            self.client.fput_object(
                self.bucket,
                remote_path,
                str(local_path),
            )

            logger.info(f"Uploaded {local_path} to {self.bucket}/{remote_path}")

            return StorageLocation(
                backend=self.name,
                bucket=self.bucket,
                path=remote_path,
                checksum_sha256=checksum,
                size_bytes=size,
                uploaded_at=datetime.utcnow(),
            )
        except S3Error as e:
            logger.error(f"Upload failed: {e}")
            raise StorageError(str(e), "upload")

    @staticmethod
    async def _calculate_checksum(path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Args:
            path: Path to the file.

        Returns:
            Hex-encoded SHA256 checksum.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def download(
        self,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Download a file from MinIO.

        Args:
            remote_path: Path in storage.
            local_path: Destination local path.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.fget_object(
                self.bucket,
                remote_path,
                str(local_path),
            )
            logger.info(f"Downloaded {self.bucket}/{remote_path} to {local_path}")
            return True
        except S3Error as e:
            logger.error(f"Download failed: {e}")
            return False

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
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                remote_path,
                expires=timedelta(seconds=expires_in),
            )
            return url
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    async def exists(self, remote_path: str) -> bool:
        """Check if a file exists in MinIO.

        Args:
            remote_path: Path in storage.

        Returns:
            True if file exists, False otherwise.
        """
        try:
            self.client.stat_object(self.bucket, remote_path)
            return True
        except S3Error:
            return False

    async def delete(self, remote_path: str) -> bool:
        """Delete a file from MinIO.

        Args:
            remote_path: Path in storage.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self.client.remove_object(self.bucket, remote_path)
            logger.info(f"Deleted {self.bucket}/{remote_path}")
            return True
        except S3Error as e:
            logger.error(f"Delete failed: {e}")
            return False

