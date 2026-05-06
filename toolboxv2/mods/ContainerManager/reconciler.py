"""
ContainerReconciler — syncs DB state with Docker reality.

Design:
- reconcile_next() checks ONE container per call (round-robin)
- Frontend calls it every 5s when visible, every 60s when hidden
- Load is distributed: never all containers at once
"""

import json
from typing import Optional, Callable, Awaitable

from toolboxv2.mods.ContainerManager.docker_ops import DockerOps, get_docker_ops


class ContainerReconciler:
    """Periodically syncs DB container state with Docker reality."""

    def __init__(self, docker_ops: DockerOps = None):
        if docker_ops is not None:
            self._ops = docker_ops
        else:
            self._ops = get_docker_ops()
        self._container_ids: list[str] = []
        self._current_index: int = 0

    def set_container_ids(self, ids: list[str]):
        """Update the list of known container IDs.  Resets round-robin pointer."""
        self._container_ids = list(ids)
        self._current_index = 0

    @property
    def container_count(self) -> int:
        return len(self._container_ids)

    def reconcile_next_sync(self) -> Optional[dict]:
        """
        Reconcile ONE container (round-robin).

        Returns:
            dict with {container_id, old_status, new_status, changed}
            or None if no containers to check.
        """
        if not self._container_ids:
            return None

        # Round-robin
        if self._current_index >= len(self._container_ids):
            self._current_index = 0

        cid = self._container_ids[self._current_index]
        self._current_index += 1

        # Get live status from Docker
        new_status = self._ops.get_container_status(cid)

        return {
            "container_id": cid,
            "new_status": new_status,
        }

    def reconcile_all_sync(self) -> list[dict]:
        """Full reconciliation pass.  Returns list of status updates."""
        results = []
        for cid in self._container_ids:
            new_status = self._ops.get_container_status(cid)
            results.append({
                "container_id": cid,
                "new_status": new_status,
            })
        return results

    def docker_health(self) -> dict:
        """Return Docker daemon health status."""
        available = self._ops.is_available()
        return {
            "docker_available": available,
            "status": "online" if available else "offline",
            "container_count": len(self._container_ids),
        }
