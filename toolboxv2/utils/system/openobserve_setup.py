#!/usr/bin/env python3
# file: toolboxv2/utils/system/openobserve_setup.py
# Manages OpenObserve deployment via Docker and optional historical log import from MinIO.
#
# ENV Configuration:
#   OPENOBSERVE_ENDPOINT      = http://localhost:5080       (API endpoint)
#   OPENOBSERVE_USER          = root@example.com            (root user email)
#   OPENOBSERVE_PASSWORD      = <password>                  (root user password)
#   OPENOBSERVE_ORG           = default                     (organization)
#   OPENOBSERVE_DATA_DIR      = ./openobserve-data          (host volume mount)
#   OPENOBSERVE_PORT          = 5080                        (exposed port)
#   OPENOBSERVE_DOCKER_IMAGE  = public.ecr.aws/zinclabs/openobserve:latest
#   OPENOBSERVE_CONTAINER     = toolbox-openobserve         (container name)
#
#   # Storage backend — default: local disk. Set these for MinIO-backed storage:
#   OPENOBSERVE_STORAGE       = disk | s3                   (default: disk)
#   OPENOBSERVE_S3_ENDPOINT   = http://localhost:9000
#   OPENOBSERVE_S3_ACCESS_KEY = <key>
#   OPENOBSERVE_S3_SECRET_KEY = <secret>
#   OPENOBSERVE_S3_BUCKET     = openobserve-data            (must pre-exist)
#
#   # Historical import — set to enable:
#   OPENOBSERVE_IMPORT_ENABLED = false | true
#   OPENOBSERVE_IMPORT_BUCKET  = system-audit-logs          (source MinIO bucket)
#   OPENOBSERVE_IMPORT_PREFIX  =                            (e.g. "toolbox-instance-01/")
#   OPENOBSERVE_IMPORT_FROM    =                            (e.g. "2026-01-01")
#   OPENOBSERVE_IMPORT_TO      =                            (e.g. "2026-02-23")
#
#   # MinIO connection for import (reuses S3 keys if not set):
#   MINIO_ENDPOINT             = localhost:9000
#   MINIO_ACCESS_KEY           = <key>
#   MINIO_SECRET_KEY           = <secret>
#
# Usage:
#   # Deploy + optional import
#   python -m toolboxv2.utils.system.openobserve_setup deploy
#
#   # Import historical logs only (OpenObserve must be running)
#   python -m toolboxv2.utils.system.openobserve_setup import
#
#   # Check status
#   python -m toolboxv2.utils.system.openobserve_setup status
#
#   # Stop
#   python -m toolboxv2.utils.system.openobserve_setup stop
#
#   # Programmatic:
#   from .openobserve_setup import OpenObserveManager
#   mgr = OpenObserveManager()
#   mgr.deploy()
#   mgr.import_historical_logs()

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("toolboxV2.openobserve_setup")


class OpenObserveManager:
    """Manages OpenObserve Docker lifecycle and historical log import."""

    def __init__(self):
        self.endpoint = os.environ.get("OPENOBSERVE_ENDPOINT", "http://localhost:5080")
        self.user = os.environ.get("OPENOBSERVE_USER", "root@example.com")
        self.password = os.environ.get("OPENOBSERVE_PASSWORD", "")
        self.org = os.environ.get("OPENOBSERVE_ORG", "default")
        self.data_dir = os.environ.get("OPENOBSERVE_DATA_DIR", "./openobserve-data")
        self.port = os.environ.get("OPENOBSERVE_PORT", "5080")
        self.image = os.environ.get(
            "OPENOBSERVE_DOCKER_IMAGE",
            "public.ecr.aws/zinclabs/openobserve:latest",
        )
        self.container_name = os.environ.get("OPENOBSERVE_CONTAINER", "toolbox-openobserve")

        # Storage backend
        self.storage = os.environ.get("OPENOBSERVE_STORAGE", "disk")
        self.s3_endpoint = os.environ.get("OPENOBSERVE_S3_ENDPOINT", "")
        self.s3_access_key = os.environ.get("OPENOBSERVE_S3_ACCESS_KEY", "")
        self.s3_secret_key = os.environ.get("OPENOBSERVE_S3_SECRET_KEY", "")
        self.s3_bucket = os.environ.get("OPENOBSERVE_S3_BUCKET", "openobserve-data")

        # Import config
        self.import_enabled = os.environ.get("OPENOBSERVE_IMPORT_ENABLED", "false").lower() == "true"
        self.import_bucket = os.environ.get("OPENOBSERVE_IMPORT_BUCKET", "system-audit-logs")
        self.import_prefix = os.environ.get("OPENOBSERVE_IMPORT_PREFIX", "")
        self.import_from = os.environ.get("OPENOBSERVE_IMPORT_FROM", "")
        self.import_to = os.environ.get("OPENOBSERVE_IMPORT_TO", "")

        # MinIO connection for import
        self.minio_endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        self.minio_access_key = os.environ.get("MINIO_ACCESS_KEY", self.s3_access_key)
        self.minio_secret_key = os.environ.get("MINIO_SECRET_KEY", self.s3_secret_key)

        # Auth header (reusable)
        if self.user and self.password:
            token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
            self._auth = f"Basic {token}"
        else:
            self._auth = ""

    # -----------------------------------------------------------------------
    # Docker Lifecycle
    # -----------------------------------------------------------------------

    def deploy(self, pull: bool = True, wait: bool = True) -> bool:
        """
        Deploy OpenObserve via Docker.

        Args:
            pull:  Pull latest image before starting
            wait:  Wait until health endpoint responds (max 30s)

        Returns:
            True if container is running and healthy
        """
        if not self.password:
            logger.error("OPENOBSERVE_PASSWORD is required. Set it via env var.")
            return False

        if not self._docker_available():
            logger.error("Docker is not available. Install Docker first.")
            return False

        # Stop existing container if present
        self._docker_run(["docker", "rm", "-f", self.container_name], check=False)

        if pull:
            logger.info(f"Pulling {self.image} ...")
            self._docker_run(["docker", "pull", self.image])

        # Ensure data dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--restart", "unless-stopped",
            "-v", f"{os.path.abspath(self.data_dir)}:/data",
            "-p", f"{self.port}:5080",
            "-e", f"ZO_ROOT_USER_EMAIL={self.user}",
            "-e", f"ZO_ROOT_USER_PASSWORD={self.password}",
            "-e", "ZO_DATA_DIR=/data",
        ]

        # S3/MinIO storage backend
        if self.storage == "s3" and self.s3_endpoint:
            cmd.extend([
                "-e", "ZO_LOCAL_MODE_STORAGE=s3",
                "-e", f"ZO_S3_SERVER_URL={self.s3_endpoint}",
                "-e", f"ZO_S3_ACCESS_KEY={self.s3_access_key}",
                "-e", f"ZO_S3_SECRET_KEY={self.s3_secret_key}",
                "-e", f"ZO_S3_BUCKET_NAME={self.s3_bucket}",
                "-e", "ZO_S3_REGION_NAME=us-east-1",
                "-e", "ZO_S3_PROVIDER=minio",
            ])

        cmd.append(self.image)

        logger.info(f"Starting container '{self.container_name}' on port {self.port} ...")
        result = self._docker_run(cmd)
        if result.returncode != 0:
            logger.error(f"Docker run failed: {result.stderr}")
            return False

        if wait:
            return self._wait_healthy(timeout=30)

        return True

    def stop(self) -> bool:
        """Stop and remove the OpenObserve container."""
        result = self._docker_run(
            ["docker", "rm", "-f", self.container_name], check=False
        )
        return result.returncode == 0

    def status(self) -> Dict[str, Any]:
        """Return container status and health info."""
        info: Dict[str, Any] = {
            "container": self.container_name,
            "running": False,
            "healthy": False,
            "endpoint": f"http://localhost:{self.port}",
        }

        result = self._docker_run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
            check=False, capture=True,
        )
        if result.returncode == 0:
            info["running"] = result.stdout.strip() == "true"

        if info["running"]:
            info["healthy"] = self._health_check()

        return info

    def logs(self, tail: int = 50) -> str:
        """Return recent container logs."""
        result = self._docker_run(
            ["docker", "logs", "--tail", str(tail), self.container_name],
            check=False, capture=True,
        )
        return result.stdout + result.stderr

    # -----------------------------------------------------------------------
    # Historical Log Import (MinIO → OpenObserve)
    # -----------------------------------------------------------------------

    def import_historical_logs(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        system_stream: str = "system_logs",
        audit_stream: str = "audit_logs",
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """
        Import historical JSONL logs from MinIO into OpenObserve.

        Reads JSONL files from MinIO, parses them, and POSTs to OpenObserve
        ingestion API. Respects the path convention:
            {app_id}/logs/{node_id}/{YYYY-MM-DD}/{type}_{ts}.jsonl

        Args:
            bucket:         Source MinIO bucket (default: env OPENOBSERVE_IMPORT_BUCKET)
            prefix:         Object prefix filter (default: env OPENOBSERVE_IMPORT_PREFIX)
            date_from:      Only import logs >= this date "YYYY-MM-DD"
            date_to:        Only import logs <= this date "YYYY-MM-DD"
            system_stream:  OpenObserve stream for system logs
            audit_stream:   OpenObserve stream for audit logs
            batch_size:     Entries per POST request

        Returns:
            {"system_sent": int, "audit_sent": int, "files_processed": int,
             "files_skipped": int, "errors": list}
        """
        try:
            from minio import Minio
        except ImportError:
            logger.error("minio package required for import. pip install minio")
            return {"error": "minio not installed"}

        bucket = bucket or self.import_bucket
        prefix = prefix or self.import_prefix
        date_from = date_from or self.import_from or None
        date_to = date_to or self.import_to or None

        if not self.minio_access_key or not self.minio_secret_key:
            logger.error("MinIO credentials required. Set MINIO_ACCESS_KEY + MINIO_SECRET_KEY.")
            return {"error": "missing minio credentials"}

        if not self._health_check():
            logger.error("OpenObserve is not reachable. Deploy first.")
            return {"error": "openobserve not reachable"}

        client = Minio(
            self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=False,
        )

        if not client.bucket_exists(bucket):
            logger.error(f"Bucket '{bucket}' does not exist in MinIO.")
            return {"error": f"bucket {bucket} not found"}

        stats: Dict[str, Any] = {
            "system_sent": 0,
            "audit_sent": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "errors": [],
        }

        objects = client.list_objects(bucket, prefix=prefix, recursive=True)

        system_batch: List[Dict[str, Any]] = []
        audit_batch: List[Dict[str, Any]] = []

        for obj in objects:
            if not obj.object_name.endswith(".jsonl"):
                stats["files_skipped"] += 1
                continue

            # Extract date from path: .../YYYY-MM-DD/type_ts.jsonl
            date_folder = self._extract_date_from_path(obj.object_name)
            if date_folder:
                if date_from and date_folder < date_from:
                    stats["files_skipped"] += 1
                    continue
                if date_to and date_folder > date_to:
                    stats["files_skipped"] += 1
                    continue

            fname = obj.object_name.rsplit("/", 1)[-1]
            is_audit = fname.startswith("audit_")

            # Read JSONL from MinIO
            try:
                response = client.get_object(bucket, obj.object_name)
                raw = response.read().decode("utf-8", errors="replace")
                response.close()
                response.release_conn()
            except Exception as e:
                stats["errors"].append(f"read:{obj.object_name}:{e}")
                continue

            stats["files_processed"] += 1

            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                if is_audit or "audit_action" in entry:
                    audit_batch.append(entry)
                else:
                    system_batch.append(entry)

                # Flush when batch is full
                if len(system_batch) >= batch_size:
                    sent = self._ingest_batch(system_batch, system_stream)
                    stats["system_sent"] += sent
                    system_batch = []

                if len(audit_batch) >= batch_size:
                    sent = self._ingest_batch(audit_batch, audit_stream)
                    stats["audit_sent"] += sent
                    audit_batch = []

        # Flush remaining
        if system_batch:
            sent = self._ingest_batch(system_batch, system_stream)
            stats["system_sent"] += sent
        if audit_batch:
            sent = self._ingest_batch(audit_batch, audit_stream)
            stats["audit_sent"] += sent

        logger.info(
            f"Import complete: {stats['files_processed']} files, "
            f"{stats['system_sent']} system + {stats['audit_sent']} audit entries"
        )
        return stats

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _ingest_batch(self, entries: List[Dict[str, Any]], stream: str) -> int:
        """POST a batch of entries to OpenObserve ingestion API."""
        if not entries:
            return 0

        url = f"{self.endpoint}/api/{self.org}/{stream}/_json"
        payload = json.dumps(entries).encode("utf-8")

        try:
            req = Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._auth,
                },
                method="POST",
            )
            with urlopen(req, timeout=15) as resp:
                if resp.status in (200, 204):
                    return len(entries)
                else:
                    body = resp.read().decode()[:200]
                    logger.warning(f"Ingest HTTP {resp.status}: {body}")
                    return 0
        except (HTTPError, URLError, OSError) as e:
            logger.warning(f"Ingest failed for stream={stream}: {e}")
            return 0

    def _extract_date_from_path(self, path: str) -> Optional[str]:
        """
        Extract YYYY-MM-DD from path like:
            app_id/logs/node_id/2026-02-23/system_123.jsonl
        """
        parts = path.split("/")
        for part in parts:
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                try:
                    int(part[:4])
                    int(part[5:7])
                    int(part[8:10])
                    return part
                except ValueError:
                    continue
        return None

    def _health_check(self) -> bool:
        try:
            req = Request(f"{self.endpoint}/healthz", method="GET")
            with urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _wait_healthy(self, timeout: int = 30) -> bool:
        """Poll healthz endpoint until ready."""
        start = time.time()
        while time.time() - start < timeout:
            if self._health_check():
                logger.info(f"OpenObserve ready at {self.endpoint}")
                return True
            time.sleep(1)
        logger.warning(f"OpenObserve not ready after {timeout}s")
        return False

    @staticmethod
    def _docker_available() -> bool:
        try:
            r = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=5
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _docker_run(
        cmd: List[str],
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                capture_output=capture,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 1, "", "timeout")
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, 1, "", "docker not found")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenObserve Docker management + historical log import"
    )
    parser.add_argument(
        "action",
        choices=["deploy", "stop", "status", "logs", "import"],
        help="Action to perform",
    )
    parser.add_argument("--no-pull", action="store_true", help="Skip docker pull")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for healthz")
    parser.add_argument("--tail", type=int, default=50, help="Log tail lines")
    parser.add_argument("--date-from", type=str, default="", help="Import: start date YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default="", help="Import: end date YYYY-MM-DD")
    parser.add_argument("--prefix", type=str, default="", help="Import: MinIO object prefix")
    parser.add_argument("--bucket", type=str, default="", help="Import: MinIO source bucket")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mgr = OpenObserveManager()

    if args.action == "deploy":
        ok = mgr.deploy(pull=not args.no_pull, wait=not args.no_wait)
        if ok and mgr.import_enabled:
            print("\n── Historical import enabled, starting ──")
            stats = mgr.import_historical_logs()
            print(json.dumps(stats, indent=2))
        sys.exit(0 if ok else 1)

    elif args.action == "stop":
        ok = mgr.stop()
        print("Stopped." if ok else "Failed to stop.")
        sys.exit(0 if ok else 1)

    elif args.action == "status":
        info = mgr.status()
        print(json.dumps(info, indent=2))
        sys.exit(0 if info["running"] else 1)

    elif args.action == "logs":
        print(mgr.logs(tail=args.tail))

    elif args.action == "import":
        stats = mgr.import_historical_logs(
            bucket=args.bucket or None,
            prefix=args.prefix or None,
            date_from=args.date_from or None,
            date_to=args.date_to or None,
        )
        print(json.dumps(stats, indent=2))
        sys.exit(0 if not stats.get("error") else 1)


if __name__ == "__main__":
    main()
