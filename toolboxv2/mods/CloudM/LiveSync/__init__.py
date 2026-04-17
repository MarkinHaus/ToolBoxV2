"""
LiveSync — ToolBoxV2 Supervisor Interface
==========================================
Manages the SyncService as a subprocess, provides API for
creating/joining/stopping shares, healthcheck, and sync log.

ToolBoxV2 Integration:
  - @export decorators for all API functions
  - Subprocess management with auto-restart
  - Healthcheck every 30s

Standalone usage (without ToolBox):
  - Direct function calls work without @export
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import SyncConfig, ShareToken, load_env_config
from .crypto import generate_encryption_key

# ToolBox integration (graceful fallback)
try:
    from toolboxv2 import App, RequestData, Result, get_app
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

    class Result:
        @staticmethod
        def ok(data=None, info=None):
            return {"ok": True, "data": data, "info": info}
        @staticmethod
        def error(info=None):
            return {"ok": False, "error": info}

    class App:
        pass

logger = logging.getLogger("LiveSync")

Name = "CloudM.LiveSync"
version = "1.0.0"

# ── Subprocess State ──

_sync_process: Optional[subprocess.Popen] = None
_share_registry: Dict[str, Dict[str, Any]] = {}


# ── Share Token ──

def create_share_token(
    share_id: str,
    encryption_key: str,
    minio_endpoint: str,
    ws_endpoint: str,
    bucket: str = "livesync",
) -> str:
    """Create a share token (base64 string) for distribution."""
    tok = ShareToken(
        share_id=share_id,
        minio_endpoint=minio_endpoint,
        bucket=bucket,
        prefix=share_id,
        encryption_key=encryption_key,
        ws_endpoint=ws_endpoint,
    )
    return tok.encode()


# ── Share Registry ──

def register_share(share_id: str, vault_path: str, token: str):
    """Register a share in the local registry."""
    _share_registry[share_id] = {
        "share_id": share_id,
        "vault_path": vault_path,
        "token": token,
        "created_at": time.time(),
    }


def list_shares() -> List[Dict[str, Any]]:
    """List all registered shares."""
    return list(_share_registry.values())


def stop_share(share_id: str) -> dict:
    """Stop and deregister a share."""
    if share_id in _share_registry:
        del _share_registry[share_id]
        logger.info(f"[LiveSync] Share stopped: {share_id}")
        return {"ok": True, "info": f"Share {share_id} stopped"}
    return {"ok": False, "error": f"Share {share_id} not found"}


# ── Subprocess Management ──

def start_sync(vault_path: str, share_id: str = "default", port: int = 8765) -> dict:
    """
    Start SyncService as a subprocess.

    The server runs independently — if ToolBox crashes, the sync continues.
    """
    global _sync_process

    if _sync_process is not None and _sync_process.poll() is None:
        return {
            "ok": True,
            "info": "Already running",
            "pid": _sync_process.pid,
        }

    try:
        _sync_process = subprocess.Popen(
            [
                sys.executable, "-m",
                "toolboxv2.mods.CloudM.LiveSync.server",
                "--vault", vault_path,
                "--share-id", share_id,
                "--port", str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"[LiveSync] Server started: PID {_sync_process.pid}, port {port}")
        return {
            "ok": True,
            "info": "Server started",
            "pid": _sync_process.pid,
            "port": port,
        }
    except Exception as e:
        logger.error(f"[LiveSync] Failed to start server: {e}")
        return {"ok": False, "error": str(e)}


def stop_sync() -> dict:
    """Stop the SyncService subprocess gracefully."""
    global _sync_process

    if _sync_process is None or _sync_process.poll() is not None:
        _sync_process = None
        return {"ok": True, "info": "Not running"}

    try:
        _sync_process.terminate()
        _sync_process.wait(timeout=10)
        logger.info("[LiveSync] Server stopped")
    except subprocess.TimeoutExpired:
        _sync_process.kill()
        logger.warning("[LiveSync] Server killed (timeout)")
    except Exception as e:
        logger.error(f"[LiveSync] Stop error: {e}")

    pid = _sync_process.pid if _sync_process else None
    _sync_process = None
    return {"ok": True, "info": "Stopped", "pid": pid}


def restart_sync(vault_path: str, share_id: str = "default", port: int = 8765) -> dict:
    """Hard restart the SyncService."""
    stop_sync()
    time.sleep(1)
    return start_sync(vault_path, share_id, port)


def get_sync_status() -> dict:
    """Check if the SyncService is running."""
    global _sync_process

    if _sync_process is not None and _sync_process.poll() is None:
        return {
            "running": True,
            "pid": _sync_process.pid,
            "shares": list_shares(),
        }
    return {
        "running": False,
        "pid": None,
        "shares": list_shares(),
    }


# ── Create / Join Share ──

def create_share(vault_path: str, ws_host: str = "0.0.0.0", ws_port: int = 8765) -> dict:
    """
    Create a new share for a vault folder.

    Steps:
      1. Generate share_id + encryption key
      2. Create share token
      3. Start SyncService if not running
      4. Return token for distribution
    """
    env = load_env_config()

    share_id = uuid.uuid4().hex[:8]
    enc_key = generate_encryption_key()
    ws_endpoint = f"ws://{ws_host}:{ws_port}"

    token = create_share_token(
        share_id=share_id,
        encryption_key=enc_key,
        minio_endpoint=env["endpoint"],
        ws_endpoint=ws_endpoint,
        bucket=env.get("bucket", "livesync"),
    )

    register_share(share_id, vault_path, token)
    start_sync(vault_path, share_id, ws_port)

    logger.info(f"[LiveSync] Share created: {share_id}")
    return {
        "ok": True,
        "share_id": share_id,
        "token": token,
        "info": "Share created. Distribute token to join.",
    }


def join_share(vault_path: str, token: str) -> dict:
    """Join an existing share using a token."""
    try:
        tok = ShareToken.decode(token)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    config = tok.to_sync_config(vault_path)
    register_share(config.share_id, vault_path, token)

    logger.info(f"[LiveSync] Joined share: {config.share_id}")
    return {
        "ok": True,
        "share_id": config.share_id,
        "info": f"Joined share {config.share_id}. Start client to sync.",
        "config": {
            "ws_endpoint": config.ws_endpoint,
            "vault_path": config.vault_path,
        },
    }


# ── Selftest / Healthcheck ──

def run_selftest() -> Dict[str, bool]:
    """Check all dependencies are available."""
    report = {}
    for name, mod in [
        ("websockets", "websockets"),
        ("watchdog", "watchdog.observers"),
        ("minio", "minio"),
        ("cryptography", "cryptography.hazmat.primitives.ciphers.aead"),
        ("aiosqlite", "aiosqlite"),
        ("pydantic", "pydantic"),
    ]:
        try:
            __import__(mod)
            report[name] = True
        except ImportError:
            report[name] = False
    return report


def check_minio_health() -> dict:
    """Check MinIO connectivity."""
    try:
        from .minio_helper import create_minio_client, healthcheck
        env = load_env_config()
        client = create_minio_client(env)
        ok, msg = healthcheck(client)
        return {"ok": ok, "message": msg}
    except Exception as e:
        return {"ok": False, "message": f"MinIO check failed: {e}"}


# ── Sync Log ──

def get_sync_log(share_id: str, vault_path: str = "", limit: int = 50) -> dict:
    """Read recent sync log entries from the server index."""
    import asyncio
    from .index import LocalIndex

    if share_id in _share_registry:
        vault_path = _share_registry[share_id].get("vault_path", vault_path)

    if not vault_path:
        return {"ok": False, "error": "vault_path required"}

    db_path = os.path.join(vault_path, ".livesync_server.db")
    if not os.path.exists(db_path):
        return {"ok": True, "data": [], "info": "No sync log yet"}

    async def _read():
        idx = LocalIndex(db_path)
        await idx.init()
        try:
            return await idx.get_sync_log(limit=limit)
        finally:
            await idx.close()

    try:
        loop = asyncio.new_event_loop()
        logs = loop.run_until_complete(_read())
        loop.close()
        return {"ok": True, "data": logs}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── ToolBoxV2 @export Decorators ──

if _TB_AVAILABLE:
    export = get_app(f"{Name}.Export").tb

    @export(mod_name=Name, api=True, version=version)
    def tb_start_sync(app: App, request: RequestData = None, vault_path: str = "", port: int = 8765):
        """Start LiveSync server."""
        if not vault_path:
            return Result.error("vault_path required")
        return start_sync(vault_path, port=port)

    @export(mod_name=Name, api=True, version=version)
    def tb_stop_sync(app: App, request: RequestData = None):
        """Stop LiveSync server."""
        return stop_sync()

    @export(mod_name=Name, api=True, version=version)
    def tb_sync_status(app: App, request: RequestData = None):
        """Get sync status."""
        return get_sync_status()

    @export(mod_name=Name, api=True, version=version)
    def tb_create_share(app: App, request: RequestData = None, vault_path: str = "", port: int = 8765):
        """Create a new share."""
        if not vault_path:
            return Result.error("vault_path required")
        return create_share(vault_path, ws_port=port)

    @export(mod_name=Name, api=True, version=version)
    def tb_join_share(app: App, request: RequestData = None, vault_path: str = "", token: str = ""):
        """Join an existing share."""
        if not vault_path or not token:
            return Result.error("vault_path and token required")
        return join_share(vault_path, token)

    @export(mod_name=Name, api=True, version=version)
    def tb_list_shares(app: App, request: RequestData = None):
        """List active shares."""
        return list_shares()

    @export(mod_name=Name, api=True, version=version)
    def tb_stop_share(app: App, request: RequestData = None, share_id: str = ""):
        """Stop a specific share."""
        if not share_id:
            return Result.error("share_id required")
        return stop_share(share_id)

    @export(mod_name=Name, api=True, version=version)
    def tb_get_sync_log(app: App, request: RequestData = None, share_id: str = "", limit: int = 50):
        """Get sync log entries."""
        return get_sync_log(share_id, limit=limit)

    @export(mod_name=Name, api=True, version=version)
    def tb_selftest(app: App, request: RequestData = None):
        """Run dependency selftest."""
        deps = run_selftest()
        minio = check_minio_health()
        return {"dependencies": deps, "minio": minio}

    @export(mod_name=Name, api=True, version=version)
    def tb_restart_sync(app: App, request: RequestData = None, vault_path: str = "", port: int = 8765):
        """Hard restart LiveSync server."""
        if not vault_path:
            return Result.error("vault_path required")
        return restart_sync(vault_path, port=port)
