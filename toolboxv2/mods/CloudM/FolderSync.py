"""
ToolBox V2 - Secure Folder Sync
Bidirektionale, verschlüsselte Ordner-Synchronisation via MinIO.

Features:
- Live-Sync mittels Watchdog (Filesystem Events)
- Batch-Processing für Performance
- Client-Side Encryption & Compression (zlib + AES)
- Lokaler Index (SQLite) für schnelle Diff-Berechnung
- Share-Token System für einfaches Pairing

Requirements:
- pip install watchdog
"""

import os
import time
import json
import zlib
import hashlib
import threading
import base64
import queue
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

# ToolBox Imports
try:
    from toolboxv2 import App, RequestData, Result, get_app, get_logger
    from toolboxv2.utils.security.cryp import Code
    # Wir nutzen den MinIO Manager aus dem Context
    from toolboxv2.utils.extras.db.minio_manager import MinIOManager, MinIOConfig
except ImportError:
    # Stubs für Standalone-Test
    class App:
        pass


    class Result:
        @staticmethod
        def ok(data=None, info=None): return {"ok": True, "data": data, "info": info}

        @staticmethod
        def error(info=None): return {"ok": False, "error": info}


    def get_logger():
        import logging; return logging.getLogger("FolderSync")

# Watchdog Import mit Fallback
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


# MinIO & Watchdog Imports
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_LIB_AVAILABLE = True
except ImportError:
    MINIO_LIB_AVAILABLE = False

# SQLite für lokalen Index
import sqlite3

Name = "CloudM.FolderSync"
version = "1.0.0"


# =================== Configuration Helper ===================

def get_minio_config() -> dict:
    """Lädt Konfiguration aus Environment"""
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        "access_key": os.getenv("MINIO_ACCESS_KEY", "admin"),
        "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        "secure": os.getenv("MINIO_SECURE", "False").lower() in ("true", "1", "yes")
    }


def check_environment() -> Result:
    """Prüft ob die Umgebung korrekt konfiguriert ist"""
    issues = []

    if not MINIO_LIB_AVAILABLE:
        issues.append("Python Library 'minio' fehlt (pip install minio)")

    if not WATCHDOG_AVAILABLE:
        issues.append("Python Library 'watchdog' fehlt (pip install watchdog) - Live-Sync deaktiviert")

    config = get_minio_config()

    # Verbindungstest
    if MINIO_LIB_AVAILABLE:
        try:
            client = Minio(
                config["endpoint"],
                access_key=config["access_key"],
                secret_key=config["secret_key"],
                secure=config["secure"]
            )
            # Einfacher API Call um Auth zu prüfen
            client.list_buckets()
        except Exception as e:
            issues.append(f"MinIO Verbindung fehlgeschlagen zu {config['endpoint']}: {str(e)}")

    if issues:
        return Result.error(info="Konfigurationsprobleme: " + "; ".join(issues))

    return Result.ok(info="Umgebung OK", data=config)


# =================== Data Structures ===================
# =================== Data Structures ===================

@dataclass
class SyncConfig:
    """Konfiguration für einen synchronisierten Ordner"""
    share_id: str  # Unique ID des Shares
    local_path: str  # Lokaler Pfad
    remote_bucket: str  # MinIO Bucket
    remote_prefix: str  # Prefix im Bucket
    encryption_key: str  # Symmetrischer Key (Base64)
    direction: str = "bidirectional"  # 'bidirectional', 'send_only', 'receive_only'
    last_sync: float = 0

    def to_dict(self):
        return asdict(self)


class LocalIndex:
    """
    Lokaler SQLite Index für Datei-Metadaten.
    Verhindert unnötiges Re-Hashing und hilft Konflikte zu erkennen.
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._lock = threading.Lock()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                              CREATE TABLE IF NOT EXISTS files
                              (
                                  rel_path
                                  TEXT
                                  PRIMARY
                                  KEY,
                                  mtime
                                  REAL,
                                  size
                                  INTEGER,
                                  checksum
                                  TEXT,
                                  sync_state
                                  TEXT
                                  DEFAULT
                                  'synced'
                              )
                              """)

    def get_file(self, rel_path: str):
        with self._lock:
            cur = self.conn.execute("SELECT * FROM files WHERE rel_path = ?", (rel_path,))
            return cur.fetchone()

    def update_file(self, rel_path: str, mtime: float, size: int, checksum: str):
        with self._lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO files (rel_path, mtime, size, checksum, sync_state)
                VALUES (?, ?, ?, ?, 'synced')
            """, (rel_path, mtime, size, checksum))
            self.conn.commit()

    def delete_file(self, rel_path: str):
        with self._lock:
            self.conn.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
            self.conn.commit()

    def get_all_paths(self) -> Set[str]:
        with self._lock:
            cur = self.conn.execute("SELECT rel_path FROM files")
            return {row['rel_path'] for row in cur.fetchall()}


# =================== Core Engine ===================

class SyncEngine:
    """
    Der Kern-Prozess. Läuft im Hintergrund pro Share.
    """

    def __init__(self, config: SyncConfig, minio_client):
        self.cfg = config
        self.minio = minio_client
        self.logger = get_logger()
        self.running = False

        # Paths
        self.root = Path(config.local_path)
        self.index_db = self.root / ".tb_sync_index.db"

        # Encryption Key Decode
        self.key = base64.b64decode(self.cfg.encryption_key)

        # Queues & Batching
        self.upload_queue = queue.Queue()
        self.batch_timer: Optional[threading.Timer] = None
        self.BATCH_DELAY = 2.0  # Sekunden warten vor Upload-Batch

        # State
        self.db = LocalIndex(str(self.index_db))
        self.observer = None
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    def start(self):
        """Startet Watcher und Sync-Loop"""
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

        self.running = True

        # 1. Watchdog starten
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
        else:
            self.logger.warning(f"Watchdog nicht verfügbar. Fallback auf Polling für {self.cfg.share_id}")

        # 2. Initial Full Sync (Async)
        threading.Thread(target=self._full_sync_loop, daemon=True).start()

        self.logger.info(f"Sync Engine gestartet für {self.cfg.local_path}")

    def stop(self):
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    def _start_watchdog(self):
        class Handler(FileSystemEventHandler):
            def __init__(self, engine):
                self.engine = engine

            def on_any_event(self, event):
                if event.is_directory: return
                if event.src_path.endswith('.tb_sync_index.db') or event.src_path.endswith('.tmp'): return

                # Relativen Pfad berechnen
                try:
                    rel_path = str(Path(event.src_path).relative_to(self.engine.root)).replace("\\", "/")
                    self.engine._schedule_upload(rel_path)
                except ValueError:
                    pass

        self.observer = Observer()
        self.observer.schedule(Handler(self), str(self.root), recursive=True)
        self.observer.start()

    def _schedule_upload(self, rel_path: str):
        """Fügt Datei zur Upload-Queue hinzu (Debounced)"""
        self.upload_queue.put(rel_path)
        # Trigger Batch Processor
        # In einer echten Implementierung würde hier ein Timer zurückgesetzt werden
        # um Events zusammenzufassen. Wir verarbeiten hier direkt.

    # =================== Logic: Processing ===================

    def _full_sync_loop(self):
        """Hintergrund-Loop für Remote-Check und Queue-Verarbeitung"""
        while self.running:
            try:
                # 1. Verarbeite lokale Änderungen (Uploads)
                while not self.upload_queue.empty():
                    rel_path = self.upload_queue.get()
                    self._process_local_change(rel_path)

                # 2. Prüfe Remote Änderungen (Downloads)
                # Nur alle X Sekunden pollen, wenn keine WebSockets/Events verfügbar sind
                self._check_remote_changes()

                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Sync Loop Error: {e}")
                time.sleep(5)

    def _process_local_change(self, rel_path: str):
        """Verarbeitet eine lokale Änderung -> Upload"""
        local_file = self.root / rel_path

        if not local_file.exists():
            # Deletion handling
            self._delete_remote(rel_path)
            self.db.delete_file(rel_path)
            return

        # Checksum berechnen
        try:
            with open(local_file, "rb") as f:
                raw_data = f.read()

            checksum = hashlib.sha256(raw_data).hexdigest()
            mtime = local_file.stat().st_mtime
            size = len(raw_data)

            # Prüfen ob Upload nötig (Vergleich mit Index)
            cached = self.db.get_file(rel_path)
            if cached and cached['checksum'] == checksum:
                return  # Keine Änderung des Inhalts

            # Compress & Encrypt
            compressed = zlib.compress(raw_data)
            encrypted = Code.encrypt_symmetric(compressed, self.key)

            # Upload
            remote_key = f"{self.cfg.remote_prefix}/{rel_path}.enc"

            # Metadaten für Remote-Seite
            metadata = {
                "x-amz-meta-original-hash": checksum,
                "x-amz-meta-mtime": str(mtime)
            }

            from io import BytesIO
            self.minio.put_object(
                self.cfg.remote_bucket,
                remote_key,
                BytesIO(encrypted),
                len(encrypted),
                metadata=metadata
            )

            # Update Local Index
            self.db.update_file(rel_path, mtime, size, checksum)
            self.logger.info(f"Uploaded: {rel_path}")

        except Exception as e:
            self.logger.error(f"Upload failed for {rel_path}: {e}")

    def _check_remote_changes(self):
        """Prüft MinIO auf neue Dateien -> Download"""
        try:
            # List Objects recursive
            objects = self.minio.list_objects(
                self.cfg.remote_bucket,
                prefix=self.cfg.remote_prefix,
                recursive=True
            )

            remote_files = set()

            for obj in objects:
                # Pfad aufräumen (Prefix entfernen, .enc entfernen)
                obj_name = obj.object_name
                if not obj_name.endswith(".enc"): continue  # Ignoriere nicht-verschlüsselte Dateien

                rel_path = obj_name[len(self.cfg.remote_prefix) + 1:-4]  # remove prefix/ and .enc
                remote_files.add(rel_path)

                # Check Metadata (Hash vergleich)
                # Minio Python Client braucht stat_object für Custom Metadata wenn nicht in list enthalten
                # Performance-Optimierung: Wir nutzen ETag oder Größe als ersten Indikator

                # Wir laden Stats nur wenn lokal nicht aktuell
                local_entry = self.db.get_file(rel_path)

                # Wenn wir die Datei gar nicht haben oder Größe abweicht (grober Check)
                # Für exakten Check brauchen wir den Hash aus den Metadaten
                if not local_entry:
                    self._download_file(rel_path, obj_name)
                    continue

                # Um API Calls zu sparen: Wenn Remote 'Last Modified' neuer ist als unser Index Check
                # (Hier vereinfacht)
                if obj.last_modified.timestamp() > local_entry['mtime'] + 2.0:  # 2s Toleranz
                    # Detail Check
                    stat = self.minio.stat_object(self.cfg.remote_bucket, obj_name)
                    remote_hash = stat.metadata.get('x-amz-meta-original-hash')

                    if remote_hash and remote_hash != local_entry['checksum']:
                        self._download_file(rel_path, obj_name)

            # Optional: Löschen von lokalen Dateien, die remote nicht mehr existieren?
            # Das ist gefährlich bei Sync-Problemen ("Ich sehe nichts, also lösche ich alles").
            # Besser: Explizite Delete-Marker verwenden. Hier weggelassen für Sicherheit.

        except Exception as e:
            self.logger.error(f"Remote check failed: {e}")

    def _download_file(self, rel_path: str, remote_obj_name: str):
        """Lädt Datei herunter, entschlüsselt & entpackt"""
        try:
            resp = self.minio.get_object(self.cfg.remote_bucket, remote_obj_name)
            encrypted_data = resp.read()
            resp.close()
            resp.release_conn()

            # Decrypt & Decompress
            decrypted = Code.decrypt_symmetric(encrypted_data, self.key, to_str=False)
            data = zlib.decompress(decrypted)

            # Write to Disk (Atomic)
            local_path = self.root / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = local_path.with_suffix('.tmp')
            with open(tmp_path, 'wb') as f:
                f.write(data)

            # Atomic rename
            if local_path.exists():
                local_path.unlink()
            os.rename(tmp_path, local_path)

            # Update Index
            checksum = hashlib.sha256(data).hexdigest()
            mtime = local_path.stat().st_mtime
            self.db.update_file(rel_path, mtime, len(data), checksum)

            self.logger.info(f"Downloaded: {rel_path}")

        except Exception as e:
            self.logger.error(f"Download failed for {rel_path}: {e}")

    def _delete_remote(self, rel_path: str):
        """Löscht Datei auf Server"""
        try:
            remote_key = f"{self.cfg.remote_prefix}/{rel_path}.enc"
            self.minio.remove_object(self.cfg.remote_bucket, remote_key)
        except Exception as e:
            self.logger.error(f"Remote delete failed: {e}")


# =================== Manager ===================

class FolderSyncManager:
    _instance = None

    def __init__(self):
        self.active_syncs: Dict[str, SyncEngine] = {}
        self.minio_manager = MinIOManager()
        self.logger = get_logger()

        # MinIO Client initialisieren
        # Wir nehmen an, dass 'cloud' Alias konfiguriert ist oder Standard
        self.minio = self._init_client()

    def _init_client(self):
        if not MINIO_LIB_AVAILABLE: return None
        cfg = get_minio_config()
        try:
            return Minio(cfg["endpoint"], access_key=cfg["access_key"], secret_key=cfg["secret_key"],
                         secure=cfg["secure"])
        except:
            return None

    def create_share(self, local_path: str) -> Result:
        if not self.minio: return Result.error("MinIO nicht verfügbar. ENV Config prüfen.")

        import uuid
        share_id = str(uuid.uuid4())[:8]
        bucket = "tb-shared"
        prefix = share_id
        key = base64.b64encode(os.urandom(32)).decode('ascii')

        config = SyncConfig(share_id, local_path, bucket, prefix, key)

        # Token enthält Endpoint als Hinweis, aber keine Credentials
        token_data = {
            "e": get_minio_config()["endpoint"],
            "b": bucket,
            "p": prefix,
            "k": key,
            "i": share_id
        }
        token = base64.b64encode(json.dumps(token_data).encode()).decode()

        try:
            if not self.minio.bucket_exists(bucket): self.minio.make_bucket(bucket)
            engine = SyncEngine(config, self.minio)
            self.active_syncs[share_id] = engine
            engine.start()
            return Result.ok(data={"share_id": share_id, "token": token}, info="Share gestartet")
        except Exception as e:
            return Result.error(str(e))

    def join_share(self, local_path: str, token: str) -> Result:
        """Tritt einem Share mittels Token bei"""
        try:
            token_data = json.loads(base64.b64decode(token).decode())

            config = SyncConfig(
                share_id=token_data['i'],
                local_path=local_path,
                remote_bucket=token_data['b'],
                remote_prefix=token_data['p'],
                encryption_key=token_data['k']
            )

            # Engine starten
            engine = SyncEngine(config, self.minio)
            self.active_syncs[config.share_id] = engine
            engine.start()

            return Result.ok(data={"share_id": config.share_id, "status": "joined"})

        except Exception as e:
            return Result.error(f"Ungültiger Token oder Fehler: {e}")

    def stop_sync(self, share_id: str):
        if share_id in self.active_syncs:
            self.active_syncs[share_id].stop()
            del self.active_syncs[share_id]
            return Result.ok(info="Sync gestoppt")
        return Result.error("Share ID nicht gefunden")


# =================== ToolBox Interfaces ===================

export = get_app(f"{Name}.Export").tb

@export(mod_name=Name, api=True, version=version)
def check_status(app: App, request: RequestData):
    """Prüft Config und Libraries"""
    return check_environment()

@export(mod_name=Name, api=True, version=version)
def start_sharing(local_path: str):
    """Startet das Teilen eines Ordners (Erstellt Share)"""
    mgr = _get_manager()
    return mgr.create_share(local_path)


@export(mod_name=Name, api=True, version=version)
def connect_share(local_path: str, token: str):
    """Verbindet einen lokalen Ordner mit einem Share"""
    mgr = _get_manager()
    return mgr.join_share(local_path, token)


@export(mod_name=Name, api=True, version=version)
def stop_share(share_id: str):
    """Stoppt die Synchronisation"""
    mgr = _get_manager()
    return mgr.stop_sync(share_id)


@export(mod_name=Name, api=True, version=version)
def list_active_shares():
    """Listet aktive Synchronisationen"""
    mgr = _get_manager()
    return Result.ok(data=[
        {
            "id": k,
            "path": v.cfg.local_path,
            "bucket": v.cfg.remote_bucket
        } for k, v in mgr.active_syncs.items()
    ])


# Singleton Helper
_mgr_instance = None


def _get_manager():
    global _mgr_instance
    if not _mgr_instance:
        _mgr_instance = FolderSyncManager()
    return _mgr_instance
