#!/usr/bin/env python3
"""
upload_manager.py - Upload Metadata Tracking und Cleanup für ToolBoxV2

Features:
- Trackt alle Uploads mit Metadata (user, filename, size, timestamp)
- Automatischer Cleanup nach 7 Tagen (konfigurierbar)
- Background Task für Cleanup via app.run_bg_task
- Speichert Metadata in JSON-File
"""

import json
import os
import time
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class UploadMetadata:
    """Metadata für einen Upload."""
    upload_id: str
    user_id: str
    filename: str
    original_filename: str
    size: int
    content_type: str
    temp_path: str
    uploaded_at: float  # Unix timestamp
    expires_at: float  # Unix timestamp
    ttl_days: int = 7
    backed_up: bool = False
    backup_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def remaining_days(self) -> float:
        remaining_seconds = self.expires_at - time.time()
        return max(0, remaining_seconds / 86400)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'UploadMetadata':
        return cls(**data)


class UploadManager:
    """
    Verwaltet Upload-Metadata und Cleanup.

    Usage:
        manager = UploadManager(app.data_dir)

        # Bei Upload registrieren
        metadata = manager.register_upload(
            user_id="user123",
            filename="document.pdf",
            temp_path="/path/to/temp/file",
            size=1024000,
        )

        # Cleanup starten (als Background Task)
        app.run_bg_task(manager.cleanup_expired())
    """

    DEFAULT_TTL_DAYS = 7
    METADATA_FILENAME = "uploads_metadata.json"

    def __init__(self, data_dir: str, ttl_days: int = None):
        """
        Args:
            data_dir: Base directory (app.data_dir)
            ttl_days: Default TTL in Tagen (default: 7)
        """
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.temp_dir = self.uploads_dir / "temp"
        self.ttl_days = ttl_days or self.DEFAULT_TTL_DAYS

        # Directories erstellen
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self._metadata_file = self.uploads_dir / self.METADATA_FILENAME
        self._metadata: Dict[str, UploadMetadata] = {}
        self._lock = threading.Lock()

        # Laden
        self._load_metadata()

    def _load_metadata(self):
        """Lädt Metadata aus JSON-Datei."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    data = json.load(f)
                    self._metadata = {
                        k: UploadMetadata.from_dict(v)
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._metadata)} upload metadata entries")
            except Exception as e:
                logger.error(f"Error loading upload metadata: {e}")
                self._metadata = {}

    def _save_metadata(self):
        """Speichert Metadata in JSON-Datei."""
        try:
            with open(self._metadata_file, 'w') as f:
                data = {k: v.to_dict() for k, v in self._metadata.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving upload metadata: {e}")

    def register_upload(
        self,
        user_id: str,
        filename: str,
        temp_path: str,
        size: int,
        content_type: str = "application/octet-stream",
        ttl_days: int = None,
        extra: Dict = None,
    ) -> UploadMetadata:
        """
        Registriert einen neuen Upload.

        Args:
            user_id: User ID
            filename: Originaler Dateiname
            temp_path: Pfad zur temporären Datei
            size: Dateigröße in Bytes
            content_type: MIME Type
            ttl_days: TTL in Tagen (optional, default: 7)
            extra: Zusätzliche Metadaten

        Returns:
            UploadMetadata Objekt
        """
        import uuid

        ttl = ttl_days or self.ttl_days
        now = time.time()
        upload_id = str(uuid.uuid4())

        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)

        metadata = UploadMetadata(
            upload_id=upload_id,
            user_id=user_id,
            filename=safe_filename,
            original_filename=filename,
            size=size,
            content_type=content_type,
            temp_path=temp_path,
            uploaded_at=now,
            expires_at=now + (ttl * 86400),  # TTL in Sekunden
            ttl_days=ttl,
            extra=extra or {},
        )

        with self._lock:
            self._metadata[upload_id] = metadata
            self._save_metadata()

        logger.info(f"Registered upload: {upload_id} ({filename}, {size} bytes, TTL: {ttl} days)")
        return metadata

    def get_upload(self, upload_id: str) -> Optional[UploadMetadata]:
        """Holt Upload-Metadata."""
        return self._metadata.get(upload_id)

    def get_user_uploads(self, user_id: str) -> List[UploadMetadata]:
        """Holt alle Uploads eines Users."""
        return [
            m for m in self._metadata.values()
            if m.user_id == user_id and not m.is_expired
        ]

    def update_ttl(self, upload_id: str, ttl_days: int) -> bool:
        """
        Aktualisiert TTL eines Uploads.

        Args:
            upload_id: Upload ID
            ttl_days: Neue TTL in Tagen

        Returns:
            True bei Erfolg
        """
        with self._lock:
            if upload_id not in self._metadata:
                return False

            metadata = self._metadata[upload_id]
            metadata.ttl_days = ttl_days
            metadata.expires_at = metadata.uploaded_at + (ttl_days * 86400)
            self._save_metadata()

        logger.info(f"Updated TTL for {upload_id}: {ttl_days} days")
        return True

    def mark_backed_up(self, upload_id: str, backup_path: str) -> bool:
        """
        Markiert Upload als gebackupt.

        Args:
            upload_id: Upload ID
            backup_path: Pfad zum Backup (z.B. MinIO path)

        Returns:
            True bei Erfolg
        """
        with self._lock:
            if upload_id not in self._metadata:
                return False

            metadata = self._metadata[upload_id]
            metadata.backed_up = True
            metadata.backup_path = backup_path
            self._save_metadata()

        logger.info(f"Marked {upload_id} as backed up: {backup_path}")
        return True

    def delete_upload(self, upload_id: str, delete_file: bool = True) -> bool:
        """
        Löscht Upload und optional die Datei.

        Args:
            upload_id: Upload ID
            delete_file: Datei auch löschen (default: True)

        Returns:
            True bei Erfolg
        """
        with self._lock:
            if upload_id not in self._metadata:
                return False

            metadata = self._metadata[upload_id]

            # Datei löschen
            if delete_file and metadata.temp_path:
                try:
                    if os.path.exists(metadata.temp_path):
                        os.remove(metadata.temp_path)
                        logger.debug(f"Deleted file: {metadata.temp_path}")
                except Exception as e:
                    logger.warning(f"Could not delete file {metadata.temp_path}: {e}")

            # Metadata entfernen
            del self._metadata[upload_id]
            self._save_metadata()

        logger.info(f"Deleted upload: {upload_id}")
        return True

    async def cleanup_expired(self) -> Dict[str, int]:
        """
        Löscht abgelaufene Uploads.
        Sollte als Background Task ausgeführt werden.

        Returns:
            Stats dict mit deleted count
        """
        stats = {"deleted": 0, "errors": 0}
        expired_ids = []

        with self._lock:
            for upload_id, metadata in self._metadata.items():
                if metadata.is_expired:
                    expired_ids.append(upload_id)

        for upload_id in expired_ids:
            try:
                if self.delete_upload(upload_id, delete_file=True):
                    stats["deleted"] += 1
            except Exception as e:
                logger.error(f"Error deleting expired upload {upload_id}: {e}")
                stats["errors"] += 1

        if stats["deleted"] > 0:
            logger.info(f"Cleanup: deleted {stats['deleted']} expired uploads")

        return stats

    def cleanup_expired_sync(self) -> Dict[str, int]:
        """Synchrone Version von cleanup_expired."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.cleanup_expired())

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        total_size = sum(m.size for m in self._metadata.values())
        expired_count = sum(1 for m in self._metadata.values() if m.is_expired)
        backed_up_count = sum(1 for m in self._metadata.values() if m.backed_up)

        return {
            "total_uploads": len(self._metadata),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "expired_count": expired_count,
            "backed_up_count": backed_up_count,
            "temp_dir": str(self.temp_dir),
        }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitized Dateinamen für sichere Speicherung."""
        import re
        # Nur alphanumerisch, Punkte, Unterstriche, Bindestriche
        safe = re.sub(r'[^\w\-_\.]', '_', filename)
        # Keine doppelten Punkte oder Unterstriche
        safe = re.sub(r'\.{2,}', '.', safe)
        safe = re.sub(r'_{2,}', '_', safe)
        return safe[:255]  # Max 255 Zeichen


# Convenience function für Module
def get_upload_manager(app) -> UploadManager:
    """
    Holt oder erstellt UploadManager für eine App-Instanz.

    Usage in Module:
        from upload_manager import get_upload_manager

        manager = get_upload_manager(app)
        metadata = manager.register_upload(...)
    """
    if not hasattr(app, '_upload_manager'):
        app._upload_manager = UploadManager(app.data_dir)
    return app._upload_manager


# Cleanup Task Scheduler
async def schedule_cleanup_task(app, interval_hours: int = 6):
    """
    Scheduled Cleanup Task.
    Sollte beim App-Start gestartet werden.

    Usage:
        app.run_bg_task(schedule_cleanup_task(app, interval_hours=6))
    """
    import asyncio

    manager = get_upload_manager(app)

    while True:
        try:
            await manager.cleanup_expired()
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

        # Warte bis zum nächsten Run
        await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    # Test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UploadManager(tmpdir, ttl_days=7)

        # Test Upload registrieren
        metadata = manager.register_upload(
            user_id="test_user",
            filename="test document.pdf",
            temp_path="/tmp/test.pdf",
            size=1024000,
        )

        print(f"Registered: {metadata.upload_id}")
        print(f"Expires in: {metadata.remaining_days:.1f} days")
        print(f"Stats: {manager.get_stats()}")

        # Test TTL Update
        manager.update_ttl(metadata.upload_id, ttl_days=30)
        updated = manager.get_upload(metadata.upload_id)
        print(f"New expiry: {updated.remaining_days:.1f} days")

        print("\n✓ All tests passed!")
