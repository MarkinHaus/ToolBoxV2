"""
Shared test doubles for ContainerManager tests.

FakeDockerOps — in-memory Docker replacement
FakeDB        — in-memory TBEF.DB replacement
FakeApp       — minimal App that routes a_run_any to FakeDB
Factories     — make_container_spec, make_container_info
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from toolboxv2.mods.ContainerManager.docker_ops import (
    DockerOps, ContainerInfo, NetworkInfo,
)


# ============================================================================
# Minimal Result stand-in for tests that don't import the full TB stack
# ============================================================================

class _FakeToolBoxError(Enum):
    none = "none"
    input_error = "input_error"
    internal_error = "internal_error"
    custom_error = "custom_error"


class _FakeInfo:
    def __init__(self, exec_code=0, help_text=""):
        self.exec_code = exec_code
        self.help_text = help_text


class _FakeResultPayload:
    def __init__(self, data=None, data_to=None, data_info="", data_type=""):
        self.data = data
        self.data_to = data_to
        self.data_info = data_info
        self.data_type = data_type


class FakeResult:
    """Minimal Result compatible with ContainerManager's usage patterns."""

    def __init__(self, error, info, result, origin=None):
        self.error = error
        self.info = info
        self.result = result
        self.origin = origin

    def is_error(self):
        if self.error == _FakeToolBoxError.none:
            return False
        if self.info.exec_code == 0:
            return False
        return self.info.exec_code != 200

    def is_ok(self):
        return not self.is_error()

    def get(self, key=None, default=None):
        data = self.result.data
        if key is not None and isinstance(data, dict):
            return data.get(key, default)
        return data if data is not None else default

    @classmethod
    def ok(cls, data=None, data_info="", info="OK"):
        return cls(
            error=_FakeToolBoxError.none,
            info=_FakeInfo(exec_code=0, help_text=info),
            result=_FakeResultPayload(data=data, data_info=data_info),
        )

    @classmethod
    def json(cls, data=None, info="OK", exec_code=0, status_code=None):
        return cls(
            error=_FakeToolBoxError.none,
            info=_FakeInfo(exec_code=status_code or exec_code, help_text=info),
            result=_FakeResultPayload(data=data, data_info="JSON response", data_type="json"),
        )

    @classmethod
    def default_user_error(cls, info="", exec_code=-3, data=None):
        return cls(
            error=_FakeToolBoxError.input_error,
            info=_FakeInfo(exec_code=exec_code, help_text=info),
            result=_FakeResultPayload(data=data),
        )

    @classmethod
    def default_internal_error(cls, info="", exec_code=-2, data=None):
        return cls(
            error=_FakeToolBoxError.internal_error,
            info=_FakeInfo(exec_code=exec_code, help_text=info),
            result=_FakeResultPayload(data=data),
        )


# ============================================================================
# FakeDockerOps
# ============================================================================

class FakeDockerOps(DockerOps):
    """In-memory Docker replacement for unit tests."""

    def __init__(self):
        # Skip parent __init__ — we don't want a real client
        self._containers: dict[str, dict] = {}  # id -> raw state dict
        self._volumes: set[str] = set()
        self._networks: list[NetworkInfo] = []
        self._images: set[str] = set()
        self._available = True
        self._exec_results: dict[str, tuple[int, str]] = {}  # container_id -> (code, output)

    def set_available(self, available: bool):
        self._available = available

    def add_container(self, container_id: str, name: str, image: str,
                      status: str = "running", labels: dict = None,
                      ports: dict = None, networks: list = None):
        """Seed a container into the fake."""
        self._containers[container_id] = {
            "container_id": container_id,
            "name": name,
            "image": image,
            "status": status,
            "labels": labels or {},
            "ports": ports or {},
            "networks": networks or ["bridge"],
            "created_at": "2025-01-01T00:00:00Z",
        }

    def set_exec_result(self, container_id: str, exit_code: int, output: str):
        """Pre-program what exec_run returns for a container."""
        self._exec_results[container_id] = (exit_code, output)

    # -- DockerOps interface -------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def list_all_containers(self, include_stopped: bool = True) -> list[ContainerInfo]:
        if not self._available:
            return []
        result = []
        for raw in self._containers.values():
            if not include_stopped and raw["status"] != "running":
                continue
            result.append(self._raw_to_info(raw))
        return result

    def get_container_status(self, container_id: str) -> str:
        if not self._available:
            return "docker_offline"
        raw = self._containers.get(container_id)
        if raw is None:
            return "not_found"
        return raw["status"]

    def get_container_stats(self, container_id: str) -> Optional[dict]:
        if not self._available or container_id not in self._containers:
            return None
        return {
            "cpu_percent": 2.5,
            "memory_mb": 128.0,
            "memory_percent": 25.0,
            "network_rx_bytes": 1024,
            "network_tx_bytes": 512,
        }

    def get_container_networks(self, container_id: str) -> list[str]:
        if not self._available:
            return []
        raw = self._containers.get(container_id)
        return raw["networks"] if raw else []

    def create_container(self, **kwargs) -> str:
        if not self._available:
            raise RuntimeError("Docker not available")
        cid = f"fake_{len(self._containers):04d}"
        name = kwargs.get("name", cid)
        image = kwargs.get("image", "unknown")
        labels = kwargs.get("labels", {})
        ports = {}
        for internal, external in (kwargs.get("ports") or {}).items():
            ports[internal] = external
        volume_binds = kwargs.get("volumes", {})
        for vol_name in volume_binds:
            self._volumes.add(vol_name)
        self._containers[cid] = {
            "container_id": cid,
            "name": name,
            "image": image,
            "status": "running",
            "labels": labels,
            "ports": ports,
            "networks": ["bridge"],
            "created_at": "2025-01-01T00:00:00Z",
        }
        return cid

    def start(self, container_id: str) -> bool:
        if not self._available or container_id not in self._containers:
            return False
        self._containers[container_id]["status"] = "running"
        return True

    def stop(self, container_id: str, timeout: int = 30) -> bool:
        if not self._available or container_id not in self._containers:
            return False
        self._containers[container_id]["status"] = "exited"
        return True

    def restart(self, container_id: str, timeout: int = 30) -> bool:
        if not self._available or container_id not in self._containers:
            return False
        self._containers[container_id]["status"] = "running"
        return True

    def remove(self, container_id: str, force: bool = False) -> bool:
        if not self._available:
            return False
        if container_id not in self._containers:
            return False
        if self._containers[container_id]["status"] == "running" and not force:
            return False
        del self._containers[container_id]
        return True

    def remove_volume(self, volume_name: str) -> bool:
        if not self._available:
            return False
        if volume_name in self._volumes:
            self._volumes.discard(volume_name)
            return True
        return False

    def pull_image(self, image: str) -> bool:
        if not self._available:
            return False
        self._images.add(image)
        return True

    def exec_run(self, container_id: str, cmd: list[str],
                 user: str = "", timeout: int = 60) -> tuple[int, str]:
        if not self._available:
            return (-1, "Docker not available")
        if container_id not in self._containers:
            return (-1, "Container not found")
        if container_id in self._exec_results:
            return self._exec_results[container_id]
        return (0, "ok")

    def logs(self, container_id: str, tail: int = 100) -> str:
        if not self._available or container_id not in self._containers:
            return ""
        return f"[fake log output for {container_id}]"

    def list_networks(self) -> list[NetworkInfo]:
        if not self._available:
            return []
        return list(self._networks)

    def health_check_http(self, host: str, port: int,
                          path: str = "/health", timeout: int = 5) -> bool:
        # Default: healthy for running containers
        return True

    @staticmethod
    def get_server_ip() -> str:
        return "10.0.0.1"

    @staticmethod
    def _raw_to_info(raw: dict) -> ContainerInfo:
        labels = raw.get("labels", {})
        return ContainerInfo(
            container_id=raw["container_id"],
            name=raw["name"],
            image=raw["image"],
            status=raw["status"],
            labels=labels,
            ports=raw.get("ports", {}),
            networks=raw.get("networks", []),
            created_at=raw.get("created_at", ""),
            is_tb_managed=labels.get("managed-by") == "ContainerManager",
        )


# ============================================================================
# FakeDB
# ============================================================================

class FakeDB:
    """In-memory TBEF.DB replacement with wildcard support."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, query: str):
        if "*" in query:
            prefix = query.replace("*", "")
            matches = [v for k, v in self._store.items() if k.startswith(prefix)]
            return FakeResult.ok(data=matches)
        if query in self._store:
            return FakeResult.ok(data=self._store[query])
        return FakeResult.default_user_error(info=f"'{query}' not found")

    def set(self, query: str, data: str):
        self._store[query] = data
        return FakeResult.ok()

    def delete(self, query: str):
        self._store.pop(query, None)
        return FakeResult.ok()


# ============================================================================
# FakeApp
# ============================================================================

class FakeApp:
    """Minimal App mock that routes a_run_any to FakeDB."""

    def __init__(self, db: FakeDB = None):
        self._db = db or FakeDB()
        self.logger = _FakeLogger()

    async def a_run_any(self, action, query="", data=None, get_results=False):
        action_str = str(action)
        if "GET" in action_str:
            return self._db.get(query)
        elif "SET" in action_str:
            return self._db.set(query, data)
        elif "DELETE" in action_str:
            return self._db.delete(query)
        return FakeResult.default_internal_error(info=f"Unknown action: {action_str}")


class _FakeLogger:
    """Minimal logger that doesn't crash."""
    def debug(self, *a, **kw): pass
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def critical(self, *a, **kw): pass


# ============================================================================
# Factory Functions
# ============================================================================

def make_container_spec(
    container_id="test-abc123",
    user_id="usr_test",
    container_type="cli_v4",
    port=9001,
    internal_port=8080,
    ssh_port=22001,
    status="running",
    image="toolboxv2:latest",
    **overrides,
) -> dict:
    """Create a ContainerSpec-compatible dict.  Use for seeding FakeDB."""
    defaults = {
        "container_id": container_id,
        "container_name": f"{user_id}_{container_type}_test",
        "container_type": container_type,
        "user_id": user_id,
        "port": port,
        "internal_port": internal_port,
        "image": image,
        "volume_name": f"container_{user_id}_{container_type}_test",
        "status": status,
        "created_at": time.time(),
        "last_heartbeat": 0.0,
        "restart_count": 0,
        "env": {"USER_ID": user_id, "MODE": container_type},
        "ssh_port": ssh_port,
    }
    defaults.update(overrides)
    return defaults


def make_container_info(
    container_id="test-abc123",
    name="test_container",
    image="toolboxv2:latest",
    status="running",
    is_tb_managed=True,
    ports=None,
    networks=None,
    labels=None,
) -> ContainerInfo:
    """Create a ContainerInfo for direct use in tests."""
    if labels is None:
        labels = {"managed-by": "ContainerManager"} if is_tb_managed else {}
    return ContainerInfo(
        container_id=container_id,
        name=name,
        image=image,
        status=status,
        labels=labels,
        ports=ports or {},
        networks=networks or ["bridge"],
        created_at="2025-01-01T00:00:00Z",
        is_tb_managed=is_tb_managed,
    )


def seed_container_in_db(db: FakeDB, spec_dict: dict):
    """Write a ContainerSpec dict into FakeDB with proper keys."""
    cid = spec_dict["container_id"]
    uid = spec_dict["user_id"]
    db.set(f"CONTAINER::{cid}", json.dumps(spec_dict))
    # Update user container list
    existing = db.get(f"CONTAINER_USER::{uid}")
    if existing.is_ok():
        ids = json.loads(existing.get()) if isinstance(existing.get(), str) else existing.get()
    else:
        ids = []
    if cid not in ids:
        ids.append(cid)
    db.set(f"CONTAINER_USER::{uid}", json.dumps(ids))
