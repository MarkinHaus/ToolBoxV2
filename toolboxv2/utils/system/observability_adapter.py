# file: toolboxv2/utils/system/observability_adapter.py
# Non-invasive adapter layer for forwarding logs to observability backends.
# Plugs into LogSyncManager without modifying its core sync logic.
#
# Architecture:
#   LogSyncManager._do_sync()
#       ├── MinIO upload (existing, unchanged)
#       └── adapter.send_batch(entries)  ← NEW, optional
#
# Usage:
#   from .observability_adapter import OpenObserveAdapter
#
#   adapter = OpenObserveAdapter(
#       endpoint="http://ryzen.local:5080",
#       org="default",
#       credentials=("admin@toolbox.local", "secret"),
#   )
#   sync_manager.set_observability_adapter(adapter)

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin

logger = logging.getLogger("toolboxV2.observability")


# ---------------------------------------------------------------------------
# Abstract Base — implement this for any backend
# ---------------------------------------------------------------------------

class ObservabilityAdapter(ABC):
    """
    Abstract interface for observability backends.

    Any backend (OpenObserve, Loki, OpenSearch, custom) implements:
      - send_batch()   : push a list of parsed JSONL entries
      - health_check() : verify connectivity
      - close()        : cleanup resources

    The adapter receives already-parsed dicts (from JSONL lines),
    NOT raw bytes — so each backend can transform/enrich as needed.
    """

    @abstractmethod
    def send_batch(
        self,
        entries: List[Dict[str, Any]],
        stream: str = "default",
    ) -> Dict[str, Any]:
        """
        Send a batch of log entries to the backend.

        Args:
            entries: List of parsed log dicts (from JSONL)
            stream:  Target stream/index name

        Returns:
            {"sent": int, "failed": int, "errors": list}
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and accepting data."""
        ...

    def close(self):
        """Optional cleanup (connections, buffers, etc.)."""
        pass

    def send_audit_batch(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Send audit entries to a dedicated audit stream.
        Override for backends that handle audit differently.
        Default: send to stream "audit_logs".
        """
        return self.send_batch(entries, stream="audit_logs")


# ---------------------------------------------------------------------------
# OpenObserve Adapter — stdlib only (urllib), no requests dependency
# ---------------------------------------------------------------------------

class OpenObserveAdapter(ObservabilityAdapter):
    """
    Pushes JSONL log batches to OpenObserve via its JSON ingestion API.

    Endpoint: POST /api/{org}/{stream}/_json
    Auth:     Basic Auth (user:password)

    Env-override (all optional, constructor args take precedence):
        OPENOBSERVE_ENDPOINT    = http://localhost:5080
        OPENOBSERVE_ORG         = default
        OPENOBSERVE_USER        = root@example.com
        OPENOBSERVE_PASSWORD    = <password>
        OPENOBSERVE_STREAM      = system_logs
        OPENOBSERVE_AUDIT_STREAM = audit_logs
        OPENOBSERVE_VERIFY_SSL  = true
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        org: Optional[str] = None,
        credentials: Optional[Tuple[str, str]] = None,
        stream: str = "",
        audit_stream: str = "",
        timeout: float = 10.0,
        max_retries: int = 2,
        verify_ssl: bool = True,
    ):
        import os
        import base64

        self.endpoint = (
            endpoint
            or os.environ.get("OPENOBSERVE_ENDPOINT", "http://localhost:5080")
        ).rstrip("/")

        self.org = org or os.environ.get("OPENOBSERVE_ORG", "default")
        self.default_stream = stream or os.environ.get("OPENOBSERVE_STREAM", "system_logs")
        self.audit_stream_name = audit_stream or os.environ.get("OPENOBSERVE_AUDIT_STREAM", "audit_logs")
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

        user = os.environ.get("OPENOBSERVE_USER", "")
        password = os.environ.get("OPENOBSERVE_PASSWORD", "")
        if credentials:
            user, password = credentials

        if not user or not password:
            raise ValueError(
                "OpenObserveAdapter requires credentials. "
                "Pass credentials=(user, password) or set "
                "OPENOBSERVE_USER + OPENOBSERVE_PASSWORD env vars."
            )

        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._auth_header = f"Basic {token}"
        self._lock = threading.Lock()

    def _build_url(self, stream: str) -> str:
        return f"{self.endpoint}/api/{self.org}/{stream}/_json"

    def send_batch(
        self,
        entries: List[Dict[str, Any]],
        stream: str = "",
    ) -> Dict[str, Any]:
        if not entries:
            return {"sent": 0, "failed": 0, "errors": []}

        target_stream = stream or self.default_stream
        url = self._build_url(target_stream)
        payload = json.dumps(entries).encode("utf-8")

        stats = {"sent": 0, "failed": 0, "errors": []}

        for attempt in range(1, self.max_retries + 1):
            try:
                req = Request(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": self._auth_header,
                    },
                    method="POST",
                )

                ctx = None
                if not self.verify_ssl:
                    import ssl
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE

                with urlopen(req, timeout=self.timeout, context=ctx) as resp:
                    if resp.status in (200, 204):
                        stats["sent"] = len(entries)
                        return stats

                    body = resp.read().decode("utf-8", errors="replace")
                    stats["errors"].append(
                        f"HTTP {resp.status}: {body[:200]}"
                    )

            except HTTPError as e:
                msg = f"attempt={attempt} HTTP {e.code}: {e.reason}"
                stats["errors"].append(msg)
                if e.code in (400, 401, 403):
                    # Don't retry auth/schema errors
                    break

            except (URLError, OSError, TimeoutError) as e:
                stats["errors"].append(f"attempt={attempt} {type(e).__name__}: {e}")

            if attempt < self.max_retries:
                time.sleep(min(attempt * 0.5, 2.0))

        stats["failed"] = len(entries)
        return stats

    def send_audit_batch(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self.send_batch(entries, stream=self.audit_stream_name)

    def health_check(self) -> bool:
        try:
            url = f"{self.endpoint}/healthz"
            req = Request(url, method="GET")
            ctx = None
            if not self.verify_ssl:
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            with urlopen(req, timeout=5, context=ctx) as resp:
                return resp.status == 200
        except Exception:
            return False

    def close(self):
        pass

    def __repr__(self):
        return (
            f"OpenObserveAdapter(endpoint={self.endpoint!r}, "
            f"org={self.org!r}, stream={self.default_stream!r})"
        )
