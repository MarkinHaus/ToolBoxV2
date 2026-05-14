"""
o2_intel — OpenObserve Intelligence Flow for ISAA

TB Flow that gives FlowAgent SQL-level access to OpenObserve data.
Provides tools for: stream discovery, error reports, usage intel,
performance metrics, anomaly detection.

All results are persisted via BlobFile for cross-session access.

Usage (CLI):
    tb o2_intel                          # auto-config from manifest
    tb o2_intel --endpoint http://... --org default --user x --pass y

Usage (Agent):
    # registered as cli_tool, other agents call via:
    agent.add_tool(o2_intel_tool, name="o2_intel", ...)
"""

NAME = 'o2_intel'

import asyncio
import base64
import json
import logging
import os
import time
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("o2_intel")

# ═══════════════════════════════════════════════════════════════════════════
# OpenObserve SQL Query Client
# ═══════════════════════════════════════════════════════════════════════════

class O2QueryClient:
    """Thin SQL query client for OpenObserve _search API."""

    def __init__(
        self,
        endpoint: str,
        org: str,
        user: str,
        password: str,
        verify_ssl: bool = True,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.org = org
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._auth = f"Basic {token}"

    @classmethod
    def from_manifest(cls, app) -> "O2QueryClient":
        """Build client from app.manifest.observability.dashboard config."""
        obs = app.manifest.observability.dashboard
        return cls(
            endpoint=obs.endpoint,
            org=obs.org,
            user=obs.user,
            password=obs.password,
            verify_ssl=obs.verify_ssl,
        )

    @classmethod
    def from_adapter(cls, adapter) -> "O2QueryClient":
        """Build client reusing an existing OpenObserveAdapter's auth."""
        # The adapter stores endpoint, org, and _auth_header
        client = cls.__new__(cls)
        client.endpoint = adapter.endpoint
        client.org = adapter.org
        client.timeout = adapter.timeout
        client.verify_ssl = adapter.verify_ssl
        client._auth = adapter._auth_header
        return client

    def _ssl_ctx(self):
        if not self.verify_ssl:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None

    def query(
        self,
        sql: str,
        stream_type: str = "logs",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        size: int = 1000,
    ) -> dict:
        """
        Execute SQL query against OpenObserve.

        Args:
            sql:          SQL query string (e.g. "SELECT * FROM system_logs")
            stream_type:  "logs" | "metrics" | "traces"
            start_time:   Microsecond epoch (default: 24h ago)
            end_time:     Microsecond epoch (default: now)
            size:         Max rows to return

        Returns:
            dict with "hits" (list of rows) and "total" (count)
        """
        now_us = int(time.time() * 1_000_000)
        if end_time is None:
            end_time = now_us
        if start_time is None:
            start_time = now_us - (24 * 3600 * 1_000_000)  # 24h ago

        url = f"{self.endpoint}/api/{self.org}/_search?type={stream_type}"
        payload = json.dumps({
            "query": {
                "sql": sql,
                "start_time": start_time,
                "end_time": end_time,
                "from": 0,
                "size": size,
            }
        }).encode("utf-8")

        req = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": self._auth,
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout, context=self._ssl_ctx()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"O2 query failed HTTP {e.code}: {body}")
        except (URLError, OSError, TimeoutError) as e:
            raise RuntimeError(f"O2 query connection error: {e}")

    def list_streams(self, stream_type: str = "logs") -> list[dict]:
        """List all available streams."""
        url = f"{self.endpoint}/api/{self.org}/streams?type={stream_type}"
        req = Request(url, headers={"Authorization": self._auth}, method="GET")
        try:
            with urlopen(req, timeout=self.timeout, context=self._ssl_ctx()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("list", data.get("streams", []))
        except Exception as e:
            raise RuntimeError(f"O2 list_streams failed: {e}")

    def health(self) -> bool:
        """Check if OpenObserve is reachable."""
        url = f"{self.endpoint}/healthz"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=3, context=self._ssl_ctx()) as resp:
                return resp.status == 200
        except Exception:
            return False

    def time_range(self, hours: float = 24) -> tuple[int, int]:
        """Helper: return (start_us, end_us) for last N hours."""
        now_us = int(time.time() * 1_000_000)
        start_us = now_us - int(hours * 3600 * 1_000_000)
        return start_us, now_us


# ═══════════════════════════════════════════════════════════════════════════
# Persistent Report Storage (BlobFile)
# ═══════════════════════════════════════════════════════════════════════════

BLOB_ID = "o2_intel_reports"


def _get_storage(app=None):
    """Get or create BlobStorage instance."""
    try:
        if app:
            # Try to get from app's blob storage
            from toolboxv2 import get_app
            a = app or get_app()
            if hasattr(a, 'blob_storage'):
                return a.blob_storage
    except Exception:
        pass

    # Fallback: create standalone offline storage
    from toolboxv2.utils.extras.blobs import BlobStorage, StorageMode
    return BlobStorage(mode=StorageMode.OFFLINE)


def save_report(storage, report_type: str, data: dict) -> str:
    """Save a report to BlobFile. Returns the blob path."""
    from toolboxv2.utils.extras.blobs import BlobFile

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = f"{BLOB_ID}/reports/{report_type}/{ts}.json"

    with BlobFile(path, "w", storage=storage) as f:
        f.write_json(data)

    return path


def load_latest_report(storage, report_type: str) -> Optional[dict]:
    """Load the most recent report of a given type."""
    from toolboxv2.utils.extras.blobs import BlobFile

    # Read the blob's folder structure
    try:
        raw = storage.read_blob(BLOB_ID)
        if not raw:
            return None
        content = pickle.loads(raw)

        reports_folder = content.get("reports", {}).get(report_type, {})
        if not reports_folder:
            return None

        # Get latest by filename (ISO timestamp → lexicographic sort)
        latest_key = sorted(reports_folder.keys())[-1]
        data = reports_folder[latest_key]
        if isinstance(data, bytes):
            return json.loads(data.decode())
        return data
    except Exception:
        return None


def load_report_history(storage, report_type: str, limit: int = 10) -> list[dict]:
    """Load last N reports for trend comparison."""
    try:
        raw = storage.read_blob(BLOB_ID)
        if not raw:
            return []
        content = pickle.loads(raw)
        reports_folder = content.get("reports", {}).get(report_type, {})
        if not reports_folder:
            return []

        keys = sorted(reports_folder.keys())[-limit:]
        results = []
        for k in keys:
            data = reports_folder[k]
            if isinstance(data, bytes):
                data = json.loads(data.decode())
            results.append({"timestamp": k.replace(".json", ""), "data": data})
        return results
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Agent Tool Functions
# ═══════════════════════════════════════════════════════════════════════════

def _make_tools(client: O2QueryClient, storage) -> list[dict]:
    """Create all o2_intel tool definitions for FlowAgent registration."""

    # ── Tool 1: Discover Streams ──────────────────────────────────────
    def o2_discover(stream_type: str = "logs") -> str:
        """Discover available OpenObserve streams and their schemas.
        Args:
            stream_type: 'logs', 'metrics', or 'traces'
        Returns:
            JSON with stream names, doc counts, and field lists.
        """
        try:
            streams = client.list_streams(stream_type)
            summary = []
            for s in streams:
                info = {
                    "name": s.get("name", "?"),
                    "doc_count": s.get("stats", {}).get("doc_num", s.get("doc_num", 0)),
                    "storage_size": s.get("stats", {}).get("storage_size", 0),
                    "schema": [
                        {"name": f.get("name"), "type": f.get("type", "?")}
                        for f in s.get("schema", s.get("settings", {}).get("fields", []))[:30]
                    ],
                }
                summary.append(info)
            result = {"stream_type": stream_type, "count": len(summary), "streams": summary}
            save_report(storage, "discover", result)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool 2: Error Report ─────────────────────────────────────────
    def o2_errors(
        stream: str = "system_logs",
        hours: float = 24,
        top_n: int = 20,
    ) -> str:
        """Query OpenObserve for error patterns and generate error report.
        Args:
            stream: Stream name to query
            hours:  Lookback window in hours (default 24)
            top_n:  Max error groups to return
        Returns:
            JSON error report with counts, patterns, timeline.
        """
        start, end = client.time_range(hours)
        report = {"stream": stream, "hours": hours, "generated_at": _now_iso()}

        try:
            # Error count by level
            r1 = client.query(
                f"SELECT level, count(*) as cnt FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'WARNING', 'error', 'critical', 'fatal', 'warning') "
                f"GROUP BY level ORDER BY cnt DESC",
                start_time=start, end_time=end, size=50,
            )
            report["level_counts"] = r1.get("hits", [])

            # Top error messages
            r2 = client.query(
                f"SELECT message, level, count(*) as cnt FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'error', 'critical', 'fatal') "
                f"GROUP BY message, level ORDER BY cnt DESC",
                start_time=start, end_time=end, size=top_n,
            )
            report["top_errors"] = r2.get("hits", [])

            # Error timeline (hourly buckets)
            r3 = client.query(
                f"SELECT histogram(_timestamp) as ts, count(*) as cnt FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'error', 'critical', 'fatal') "
                f"GROUP BY ts ORDER BY ts",
                start_time=start, end_time=end, size=500,
            )
            report["timeline"] = r3.get("hits", [])

            # Total vs error ratio
            r4 = client.query(
                f"SELECT count(*) as total FROM \"{stream}\"",
                start_time=start, end_time=end, size=1,
            )
            total = r4.get("hits", [{}])[0].get("total", 0) if r4.get("hits") else 0
            error_count = sum(
                h.get("cnt", 0) for h in report.get("level_counts", [])
                if h.get("level", "").upper() in ("ERROR", "CRITICAL", "FATAL")
            )
            report["summary"] = {
                "total_logs": total,
                "total_errors": error_count,
                "error_rate_pct": round(error_count / max(total, 1) * 100, 2),
            }

            # Compare with previous report
            prev = load_latest_report(storage, "errors")
            if prev and "summary" in prev:
                prev_errors = prev["summary"].get("total_errors", 0)
                report["trend"] = {
                    "prev_errors": prev_errors,
                    "delta": error_count - prev_errors,
                    "direction": "up" if error_count > prev_errors else ("down" if error_count < prev_errors else "stable"),
                }

            save_report(storage, "errors", report)
            return json.dumps(report, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool 3: Usage Intel ──────────────────────────────────────────
    def o2_usage(
        stream: str = "audit_logs",
        hours: float = 24,
    ) -> str:
        """Analyze agent/tool usage patterns from audit logs.
        Args:
            stream: Audit stream name
            hours:  Lookback window in hours
        Returns:
            JSON usage report: actions, agents, tools, hourly distribution.
        """
        start, end = client.time_range(hours)
        report = {"stream": stream, "hours": hours, "generated_at": _now_iso()}

        try:
            # Top actions
            r1 = client.query(
                f"SELECT audit_action, count(*) as cnt FROM \"{stream}\" "
                f"WHERE audit_action IS NOT NULL "
                f"GROUP BY audit_action ORDER BY cnt DESC",
                start_time=start, end_time=end, size=50,
            )
            report["top_actions"] = r1.get("hits", [])

            # Top users/agents
            r2 = client.query(
                f"SELECT user_id, count(*) as cnt FROM \"{stream}\" "
                f"WHERE user_id IS NOT NULL "
                f"GROUP BY user_id ORDER BY cnt DESC",
                start_time=start, end_time=end, size=30,
            )
            report["top_agents"] = r2.get("hits", [])

            # Tool usage (from llm.call.success entries)
            r3 = client.query(
                f"SELECT resource, count(*) as cnt FROM \"{stream}\" "
                f"WHERE audit_action = 'llm.call.success' "
                f"GROUP BY resource ORDER BY cnt DESC",
                start_time=start, end_time=end, size=30,
            )
            report["llm_models"] = r3.get("hits", [])

            # Tool invocations
            r4 = client.query(
                f"SELECT resource, count(*) as cnt FROM \"{stream}\" "
                f"WHERE audit_action = 'tool.execute' OR audit_action LIKE '%tool%' "
                f"GROUP BY resource ORDER BY cnt DESC",
                start_time=start, end_time=end, size=50,
            )
            report["tool_usage"] = r4.get("hits", [])

            # Hourly distribution
            r5 = client.query(
                f"SELECT histogram(_timestamp) as ts, count(*) as cnt FROM \"{stream}\" "
                f"GROUP BY ts ORDER BY ts",
                start_time=start, end_time=end, size=500,
            )
            report["hourly_activity"] = r5.get("hits", [])

            save_report(storage, "usage", report)
            return json.dumps(report, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool 4: Performance Metrics ──────────────────────────────────
    def o2_performance(
        stream: str = "audit_logs",
        hours: float = 24,
    ) -> str:
        """Analyze system performance from audit logs (schema-adaptive).
        Args:
            stream: Stream with performance data (audit_logs)
            hours:  Lookback window
        Returns:
            JSON performance report.
        """
        start, end = client.time_range(hours)
        report = {"stream": stream, "hours": hours, "generated_at": _now_iso()}

        try:
            # ── Step 1: Discover actual schema ──
            available_fields = set()
            try:
                streams = client.list_streams("logs")
                for s in streams:
                    if s.get("name") == stream:
                        for f in s.get("schema", s.get("settings", {}).get("fields", [])):
                            available_fields.add(f.get("name", ""))
                        break
            except Exception:
                pass

            has = lambda f: f in available_fields or not available_fields  # noqa: if discovery failed, try anyway

            # ── Step 2: LLM call volume per model (always works) ──
            r1 = client.query(
                f"SELECT resource, count(*) as calls FROM \"{stream}\" "
                f"WHERE audit_action = 'llm.call.success' "
                f"GROUP BY resource ORDER BY calls DESC",
                start_time=start, end_time=end, size=30,
            )
            report["llm_calls_per_model"] = r1.get("hits", [])

            # ── Step 3: Latency/tokens if fields exist ──
            if has("duration"):
                try:
                    r2 = client.query(
                        f"SELECT resource, count(*) as calls, "
                        f"avg(duration) as avg_dur, min(duration) as min_dur, max(duration) as max_dur "
                        f"FROM \"{stream}\" WHERE audit_action = 'llm.call.success' AND duration IS NOT NULL "
                        f"GROUP BY resource ORDER BY calls DESC",
                        start_time=start, end_time=end, size=30,
                    )
                    report["llm_latency"] = r2.get("hits", [])
                except Exception as e:
                    report["llm_latency"] = {"skipped": str(e)}

            if has("input_tokens"):
                try:
                    r3 = client.query(
                        f"SELECT resource, sum(input_tokens) as total_in, sum(output_tokens) as total_out, "
                        f"avg(input_tokens) as avg_in, avg(output_tokens) as avg_out "
                        f"FROM \"{stream}\" WHERE audit_action = 'llm.call.success' AND input_tokens IS NOT NULL "
                        f"GROUP BY resource",
                        start_time=start, end_time=end, size=30,
                    )
                    report["token_throughput"] = r3.get("hits", [])
                except Exception as e:
                    report["token_throughput"] = {"skipped": str(e)}

            # ── Step 4: Action breakdown (always works) ──
            r4 = client.query(
                f"SELECT audit_action, count(*) as cnt FROM \"{stream}\" "
                f"GROUP BY audit_action ORDER BY cnt DESC",
                start_time=start, end_time=end, size=50,
            )
            report["action_breakdown"] = r4.get("hits", [])

            # ── Step 5: Error vs success ratio ──
            r5 = client.query(
                f"SELECT audit_action, count(*) as cnt FROM \"{stream}\" "
                f"WHERE audit_action IN ('llm.call.success', 'llm.call.error', 'llm.rate_limit.wait') "
                f"GROUP BY audit_action",
                start_time=start, end_time=end, size=10,
            )
            hits = r5.get("hits", [])
            success = sum(h.get("cnt", 0) for h in hits if h.get("audit_action") == "llm.call.success")
            errors = sum(h.get("cnt", 0) for h in hits if h.get("audit_action") == "llm.call.error")
            waits = sum(h.get("cnt", 0) for h in hits if h.get("audit_action") == "llm.rate_limit.wait")
            total = success + errors
            report["llm_reliability"] = {
                "success": success, "errors": errors, "rate_limit_waits": waits,
                "success_rate_pct": round(success / max(total, 1) * 100, 2),
            }

            # ── Step 6: Agent creation overhead ──
            r6 = client.query(
                f"SELECT audit_action, count(*) as cnt FROM \"{stream}\" "
                f"WHERE audit_action IN ('agent.tool_added', 'agent.session.create', 'agent.session.close') "
                f"GROUP BY audit_action",
                start_time=start, end_time=end, size=10,
            )
            agent_hits = r6.get("hits", [])
            tool_adds = sum(h.get("cnt", 0) for h in agent_hits if h.get("audit_action") == "agent.tool_added")
            session_creates = sum(h.get("cnt", 0) for h in agent_hits if h.get("audit_action") == "agent.session.create")
            session_closes = sum(h.get("cnt", 0) for h in agent_hits if h.get("audit_action") == "agent.session.close")
            report["agent_overhead"] = {
                "tool_registrations": tool_adds,
                "session_creates": session_creates,
                "session_closes": session_closes,
                "tools_per_session": round(tool_adds / max(session_creates, 1), 1),
                "unclosed_sessions": session_creates - session_closes,
                "warning": "HIGH agent churn — agents are recreated per call instead of reused"
                    if session_creates > 100 and tool_adds / max(session_creates, 1) > 5
                    else None,
            }

            # ── Step 7: Hotspot agents (who creates the most overhead?) ──
            if has("user_id"):
                try:
                    r7 = client.query(
                        f"SELECT resource, count(*) as cnt FROM \"{stream}\" "
                        f"WHERE audit_action = 'agent.tool_added' "
                        f"GROUP BY resource ORDER BY cnt DESC",
                        start_time=start, end_time=end, size=10,
                    )
                    report["tool_registration_hotspots"] = r7.get("hits", [])
                except Exception:
                    pass

            # ── Step 8: Available schema (for debugging / o2_sql) ──
            if available_fields:
                report["_schema_fields"] = sorted(available_fields - {"_timestamp"})

            save_report(storage, "performance", report)
            return json.dumps(report, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool 5: Anomaly Detection ────────────────────────────────────
    def o2_anomalies(
        stream: str = "system_logs",
        hours: float = 24,
        baseline_hours: float = 168,  # 7 days
    ) -> str:
        """Detect anomalies by comparing recent activity against baseline.
        Args:
            stream:         Stream to analyze
            hours:          Recent window to check
            baseline_hours: Baseline window for comparison (default 7d)
        Returns:
            JSON anomaly report with spikes, new errors, disappeared patterns.
        """
        recent_start, recent_end = client.time_range(hours)
        baseline_start, _ = client.time_range(baseline_hours)
        report = {"stream": stream, "hours": hours, "baseline_hours": baseline_hours, "generated_at": _now_iso()}

        try:
            # Recent error rate per hour
            r_recent = client.query(
                f"SELECT histogram(_timestamp) as ts, count(*) as cnt FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'error', 'critical', 'fatal') "
                f"GROUP BY ts ORDER BY ts",
                start_time=recent_start, end_time=recent_end, size=500,
            )

            # Baseline average error rate per hour
            r_baseline = client.query(
                f"SELECT count(*) as total_errors FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'error', 'critical', 'fatal')",
                start_time=baseline_start, end_time=recent_start, size=1,
            )

            baseline_total = r_baseline.get("hits", [{}])[0].get("total_errors", 0) if r_baseline.get("hits") else 0
            baseline_hours_actual = max(baseline_hours - hours, 1)
            baseline_avg_per_hour = baseline_total / baseline_hours_actual

            # Check for spikes
            recent_buckets = r_recent.get("hits", [])
            spikes = []
            for bucket in recent_buckets:
                cnt = bucket.get("cnt", 0)
                if baseline_avg_per_hour > 0 and cnt > baseline_avg_per_hour * 3:
                    spikes.append({
                        "timestamp": bucket.get("ts"),
                        "count": cnt,
                        "baseline_avg": round(baseline_avg_per_hour, 1),
                        "factor": round(cnt / baseline_avg_per_hour, 1),
                    })

            report["baseline_avg_errors_per_hour"] = round(baseline_avg_per_hour, 2)
            report["spikes"] = spikes
            report["spike_count"] = len(spikes)

            # New error messages (in recent but not in baseline)
            r_recent_msgs = client.query(
                f"SELECT message, count(*) as cnt FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'error', 'critical', 'fatal') "
                f"GROUP BY message ORDER BY cnt DESC",
                start_time=recent_start, end_time=recent_end, size=50,
            )
            r_baseline_msgs = client.query(
                f"SELECT message, count(*) as cnt FROM \"{stream}\" "
                f"WHERE level IN ('ERROR', 'CRITICAL', 'FATAL', 'error', 'critical', 'fatal') "
                f"GROUP BY message ORDER BY cnt DESC",
                start_time=baseline_start, end_time=recent_start, size=200,
            )

            baseline_msgs = {h.get("message", "") for h in r_baseline_msgs.get("hits", [])}
            new_errors = [
                h for h in r_recent_msgs.get("hits", [])
                if h.get("message", "") and h.get("message", "") not in baseline_msgs
            ]
            report["new_errors"] = new_errors[:20]
            report["new_error_count"] = len(new_errors)

            # Volume anomaly (total log volume)
            r_vol_recent = client.query(
                f"SELECT count(*) as cnt FROM \"{stream}\"",
                start_time=recent_start, end_time=recent_end, size=1,
            )
            r_vol_baseline = client.query(
                f"SELECT count(*) as cnt FROM \"{stream}\"",
                start_time=baseline_start, end_time=recent_start, size=1,
            )
            recent_vol = r_vol_recent.get("hits", [{}])[0].get("cnt", 0) if r_vol_recent.get("hits") else 0
            baseline_vol = r_vol_baseline.get("hits", [{}])[0].get("cnt", 0) if r_vol_baseline.get("hits") else 0
            baseline_vol_per_window = baseline_vol / max(baseline_hours_actual / hours, 1)

            report["volume"] = {
                "recent": recent_vol,
                "baseline_avg_per_window": round(baseline_vol_per_window, 0),
                "ratio": round(recent_vol / max(baseline_vol_per_window, 1), 2),
                "anomaly": recent_vol > baseline_vol_per_window * 2 or recent_vol < baseline_vol_per_window * 0.3,
            }

            save_report(storage, "anomalies", report)
            return json.dumps(report, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool 6: Raw SQL Query ────────────────────────────────────────
    def o2_sql(
        sql: str,
        stream_type: str = "logs",
        hours: float = 24,
        size: int = 100,
    ) -> str:
        """Execute arbitrary SQL query against OpenObserve.
        Args:
            sql:         Raw SQL query
            stream_type: 'logs', 'metrics', 'traces'
            hours:       Lookback window
            size:        Max rows
        Returns:
            JSON query results.
        """
        start, end = client.time_range(hours)
        try:
            result = client.query(sql, stream_type=stream_type, start_time=start, end_time=end, size=size)
            return json.dumps({
                "sql": sql,
                "total": result.get("total", 0),
                "hits": result.get("hits", []),
                "scan_size": result.get("scan_size", 0),
            }, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "sql": sql})

    # ── Tool 7: Full Report ──────────────────────────────────────────
    def o2_full_report(
        system_stream: str = "system_logs",
        audit_stream: str = "audit_logs",
        hours: float = 24,
    ) -> str:
        """Run all analyses and produce a combined intel report.
        Args:
            system_stream: System log stream name
            audit_stream:  Audit log stream name
            hours:         Lookback window
        Returns:
            JSON combined report summary.
        """
        results = {}
        results["errors"] = json.loads(o2_errors(stream=system_stream, hours=hours))
        results["usage"] = json.loads(o2_usage(stream=audit_stream, hours=hours))
        results["performance"] = json.loads(o2_performance(stream=audit_stream, hours=hours))
        results["anomalies"] = json.loads(o2_anomalies(stream=system_stream, hours=hours))

        # Build executive summary
        summary = {
            "generated_at": _now_iso(),
            "hours": hours,
            "error_rate": results["errors"].get("summary", {}).get("error_rate_pct", "?"),
            "total_errors": results["errors"].get("summary", {}).get("total_errors", "?"),
            "error_trend": results["errors"].get("trend", {}).get("direction", "unknown"),
            "spike_count": results["anomalies"].get("spike_count", 0),
            "new_error_count": results["anomalies"].get("new_error_count", 0),
            "volume_anomaly": results["anomalies"].get("volume", {}).get("anomaly", False),
            "top_action": results["usage"].get("top_actions", [{}])[0].get("audit_action", "?") if results["usage"].get("top_actions") else "?",
        }
        results["summary"] = summary
        save_report(storage, "full_report", results)
        return json.dumps(results, indent=2, default=str)

    # ── Tool 8: Report History ───────────────────────────────────────
    def o2_history(
        report_type: str = "errors",
        limit: int = 10,
    ) -> str:
        """Load historical reports for trend analysis.
        Args:
            report_type: 'errors', 'usage', 'performance', 'anomalies', 'full_report'
            limit:       Number of past reports to load
        Returns:
            JSON list of historical reports.
        """
        history = load_report_history(storage, report_type, limit)
        return json.dumps({"report_type": report_type, "count": len(history), "reports": history}, indent=2, default=str)

    return [
        {"tool_func": o2_discover, "name": "o2_discover",
         "description": "Discover available OpenObserve streams, their schemas, and doc counts. Args: stream_type='logs'|'metrics'|'traces'",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_errors, "name": "o2_errors",
         "description": "Generate error report from OpenObserve logs: error counts, top patterns, timeline, trend. Args: stream, hours, top_n",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_usage, "name": "o2_usage",
         "description": "Analyze agent/tool/LLM usage from audit logs: top actions, agents, models, hourly activity. Args: stream, hours",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_performance, "name": "o2_performance",
         "description": "Performance analysis: LLM reliability, agent creation overhead, call volume per model, schema-adaptive latency/tokens. Args: stream, hours",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_anomalies, "name": "o2_anomalies",
         "description": "Anomaly detection: error spikes vs baseline, new error patterns, volume anomalies. Args: stream, hours, baseline_hours",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_sql, "name": "o2_sql",
         "description": "Execute arbitrary SQL against OpenObserve. Args: sql (the query), stream_type, hours, size",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_full_report, "name": "o2_full_report",
         "description": "Run ALL analyses (errors+usage+performance+anomalies) and produce combined intel report. Args: system_stream, audit_stream, hours",
         "category": ["observability", "o2_intel"]},
        {"tool_func": o2_history, "name": "o2_history",
         "description": "Load historical reports for trend comparison. Args: report_type='errors'|'usage'|'performance'|'anomalies'|'full_report', limit",
         "category": ["observability", "o2_intel"]},
    ]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ═══════════════════════════════════════════════════════════════════════════
# FlowAgent Builder Integration
# ═══════════════════════════════════════════════════════════════════════════

def register_o2_tools(
    builder,
    client: O2QueryClient,
    storage=None,
    app=None,
) -> None:
    """
    Register all o2_intel tools on a FlowAgentBuilder.

    Usage:
        from toolboxv2.mods.isaa.flows.o2_intel import register_o2_tools, O2QueryClient

        client = O2QueryClient.from_manifest(app)
        register_o2_tools(builder, client)
        agent = await builder.build()
    """
    if storage is None:
        storage = _get_storage(app)

    tools = _make_tools(client, storage)
    for tool_def in tools:
        builder.add_tool(**tool_def)


def create_o2_intel_agent(
    app,
    name: str = "O2IntelAgent",
    system_stream: str = "",
    audit_stream: str = "",
) -> "FlowAgentBuilder":
    """
    Factory: Create a pre-configured FlowAgent with o2_intel tools.

    Usage:
        builder = create_o2_intel_agent(app)
        agent = await builder.build()
        result = await agent.a_run("Generate error report for last 24h")
    """
    from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder

    obs = app.manifest.observability.dashboard
    sys_stream = system_stream or obs.system_stream or "system_logs"
    aud_stream = audit_stream or obs.audit_stream or "audit_logs"

    client = O2QueryClient.from_manifest(app)
    storage = _get_storage(app)

    builder = (
        FlowAgentBuilder()
        .with_name(name)
        .with_analyst_persona()
        .with_system_message(
            f"You are an observability analyst for the ToolBoxV2 platform. "
            f"You have SQL access to OpenObserve via your tools. "
            f"System logs are in stream '{sys_stream}', audit logs in '{aud_stream}'. "
            f"Use o2_discover first if unsure about available streams/fields. "
            f"Use o2_sql for custom queries not covered by the predefined tools. "
            f"Always save results via the built-in persistence. "
            f"Compare with historical data when available (o2_history). "
            f"Be specific about numbers, timestamps, and actionable findings."
        )
        .with_checkpointing(enabled=True)
    )

    register_o2_tools(builder, client, storage, app)
    return builder


# ═══════════════════════════════════════════════════════════════════════════
# TB Flow Entry Point
# ═══════════════════════════════════════════════════════════════════════════

async def run(app, args_namespace=None, help=False):
    """
    TB Flow entry point.

    CLI usage:
        tb o2_intel                                  # full report, 24h
        tb o2_intel --kwargs report=errors hours=6   # error report, 6h
        tb o2_intel --kwargs report=agent             # FlowAgent mode
        tb o2_intel --kwargs report=discover stream_type=metrics
    """
    if help:
        HELP_STRING = """
        [o2_intel] Observability Intelligence Helper

        Parameter:
          report=<type>         Report type to generate
          hours=<float>         Time window in hours (default: 24)
          stream_type=<type>    Stream type for discovery mode (default: logs)

        Available report types:
          full           Generate complete observability analysis
          errors         Analyze system errors and failures
          usage          Analyze API/user usage metrics
          performance    Analyze latency and performance metrics
          anomalies      Detect unusual patterns and anomalies
          discover       Discover available streams/fields
          agent          Launch interactive FlowAgent observability assistant

        Examples:
          report=full
          report=errors hours=6
          report=usage hours=168
          report=performance hours=12
          report=anomalies hours=48
          report=discover stream_type=metrics
          report=agent hours=24

        Manifest requirements:
          observability.dashboard.endpoint
          observability.dashboard.password

        Optional manifest fields:
          observability.dashboard.system_stream
          observability.dashboard.audit_stream

        Default streams:
          system_stream = "system_logs"
          audit_stream  = "audit_logs"
        """
        print(HELP_STRING)
    # ── Parse kwargs from TB Namespace ──
    raw_kwargs = {}
    if args_namespace is not None:
        kw_list = getattr(args_namespace, "kwargs", None) or []
        for item in kw_list:
            if isinstance(item, dict):
                raw_kwargs.update(item)
            elif isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                raw_kwargs[k.strip()] = v.strip()

    report = raw_kwargs.get("report", "full")
    try:
        hours = float(raw_kwargs.get("hours", 24))
    except (ValueError, TypeError):
        hours = 24.0
    stream_type = raw_kwargs.get("stream_type", "logs")

    app.print(f"[o2_intel] Starting {report} report (last {hours}h)...")

    # ── Auto-configure from manifest ──
    obs = app.manifest.observability.dashboard
    if not obs.password:
        app.print("[o2_intel] ERROR: No OpenObserve credentials in manifest. "
                  "Set observability.dashboard.password in manifest.")
        return

    client = O2QueryClient.from_manifest(app)

    if not client.health():
        app.print(f"[o2_intel] WARNING: OpenObserve not reachable at {obs.endpoint}")
        # Try anyway — might be a healthz issue

    storage = _get_storage(app)
    tools = {t["name"]: t["tool_func"] for t in _make_tools(client, storage)}

    sys_stream = obs.system_stream or "system_logs"
    aud_stream = obs.audit_stream or "audit_logs"

    if report == "agent":
        # Launch full FlowAgent with tools — for interactive use by other agents
        builder = create_o2_intel_agent(app, system_stream=sys_stream, audit_stream=aud_stream)
        agent = await builder.build()
        result = await agent.a_run(
            f"Generate a comprehensive observability report for the last {hours} hours. "
            f"Include errors, usage patterns, performance metrics, and anomaly detection. "
            f"Compare with historical data if available.",
        )
        app.print(result)
        return agent  # Return agent for further use

    # Direct tool execution (no LLM needed)
    if report == "discover":
        result = tools["o2_discover"](stream_type=stream_type)
    elif report == "errors":
        result = tools["o2_errors"](stream=sys_stream, hours=hours)
    elif report == "usage":
        result = tools["o2_usage"](stream=aud_stream, hours=hours)
    elif report == "performance":
        result = tools["o2_performance"](stream=aud_stream, hours=hours)
    elif report == "anomalies":
        result = tools["o2_anomalies"](stream=sys_stream, hours=hours)
    elif report == "full":
        result = tools["o2_full_report"](system_stream=sys_stream, audit_stream=aud_stream, hours=hours)
    else:
        app.print(f"[o2_intel] Unknown report type: {report}")
        return

    data = json.loads(result)
    app.print(json.dumps(data, indent=2))
    app.print(f"\n[o2_intel] Report saved to BlobStorage ({BLOB_ID})")
