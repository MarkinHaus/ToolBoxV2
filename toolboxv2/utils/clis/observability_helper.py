#!/usr/bin/env python3
# file: toolboxv2/utils/system/observability_helper.py
# End-to-End CLI helper for all observability scenarios.
#
# Scenarios:
#   local-debug   — Start OpenObserve (Docker or connect to existing), send test logs, open dashboard
#   server-setup  — Deploy OpenObserve on server with MinIO backend + import historical logs
#   ci-push       — Send a single log/audit entry from CI (GitHub Actions, etc.)
#   status        — Check OpenObserve health + pending sync stats
#   stop          — Stop local OpenObserve container
#
# Usage:
#   python -m toolboxv2.utils.system.observability_helper local-debug
#   python -m toolboxv2.utils.system.observability_helper server-setup --import-from 2026-01-01
#   python -m toolboxv2.utils.system.observability_helper ci-push --message "Deploy OK" --level INFO
#   python -m toolboxv2.utils.system.observability_helper status
#   python -m toolboxv2.utils.system.observability_helper stop

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("toolboxV2.observability_helper")

# Defaults
DEFAULT_ENDPOINT = "http://localhost:5080"
DEFAULT_USER = "root@example.com"
DEFAULT_PASSWORD = "toolbox-dev-2026"
DEFAULT_ORG = "default"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _ensure_password() -> str:
    """Ensure a password is set, using env or default for dev."""
    pw = _get_env("OPENOBSERVE_PASSWORD", "")
    if not pw:
        pw = DEFAULT_PASSWORD
        os.environ["OPENOBSERVE_PASSWORD"] = pw
        logger.info(f"Using default dev password (set OPENOBSERVE_PASSWORD to override)")
    return pw


def _make_adapter(endpoint: str = "", user: str = "", password: str = ""):
    from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter

    endpoint = endpoint or _get_env("OPENOBSERVE_ENDPOINT", DEFAULT_ENDPOINT)
    user = user or _get_env("OPENOBSERVE_USER", DEFAULT_USER)
    password = password or _get_env("OPENOBSERVE_PASSWORD", DEFAULT_PASSWORD)

    return OpenObserveAdapter(
        endpoint=endpoint,
        credentials=(user, password),
        verify_ssl=False,
    )


def _make_manager() -> "OpenObserveManager":
    from toolboxv2.utils.system.openobserve_setup import OpenObserveManager
    return OpenObserveManager()


def _open_browser(url: str):
    """Best-effort browser open."""
    try:
        import webbrowser
        webbrowser.open(url)
        logger.info(f"Browser opened: {url}")
    except Exception:
        logger.info(f"Open manually: {url}")


def _send_test_logs(adapter, count: int = 5):
    """Send a few test entries so the dashboard isn't empty."""
    now = datetime.datetime.now(datetime.timezone.utc)
    node = platform.node()

    system_entries = []
    for i in range(count):
        ts = (now - datetime.timedelta(seconds=count - i)).isoformat()
        level = ["DEBUG", "INFO", "INFO", "WARNING", "ERROR"][i % 5]
        system_entries.append({
            "timestamp": ts,
            "level": level,
            "name": "observability-test",
            "message": f"Test log entry #{i + 1} — {level}",
            "app_id": "toolbox-test",
            "node_id": node,
            "filename": "observability_helper.py",
            "funcName": "_send_test_logs",
        })

    audit_entries = [
        {
            "timestamp": now.isoformat(),
            "level": "INFO",
            "name": "observability-test",
            "message": "AUDIT: LOGIN on /auth by test_user",
            "audit_action": "LOGIN",
            "user_id": "test_user",
            "resource": "/auth",
            "status": "SUCCESS",
            "app_id": "toolbox-test",
            "node_id": node,
        },
        {
            "timestamp": now.isoformat(),
            "level": "INFO",
            "name": "observability-test",
            "message": "AUDIT: API_CALL on /api/test by test_user",
            "audit_action": "API_CALL",
            "user_id": "test_user",
            "resource": "/api/test",
            "status": "FAILURE",
            "details": {"reason": "permission_denied"},
            "app_id": "toolbox-test",
            "node_id": node,
        },
    ]

    r1 = adapter.send_batch(system_entries, stream="system_logs")
    r2 = adapter.send_audit_batch(audit_entries)

    return {
        "system": r1,
        "audit": r2,
    }


# ---------------------------------------------------------------------------
# Scenario 1: local-debug
# ---------------------------------------------------------------------------

def cmd_local_debug(args):
    """Start OpenObserve locally and send test logs."""
    password = _ensure_password()
    endpoint = _get_env("OPENOBSERVE_ENDPOINT", DEFAULT_ENDPOINT)
    user = _get_env("OPENOBSERVE_USER", DEFAULT_USER)

    if not args.no_docker:
        print("── Starting OpenObserve via Docker ──")
        mgr = _make_manager()

        if not mgr._docker_available():
            print("Docker nicht verfügbar.")
            print("Entweder Docker installieren oder --no-docker verwenden.")
            print()
            print("Ohne Docker:")
            print("  1. OpenObserve Binary herunterladen:")
            print("     https://github.com/openobserve/openobserve/releases/latest")
            print(f"  2. ZO_ROOT_USER_EMAIL={user} ZO_ROOT_USER_PASSWORD=*** ./openobserve")
            print(f"  3. python -m toolboxv2.utils.system.observability_helper local-debug --no-docker")
            return 1

        ok = mgr.deploy(pull=not args.no_pull, wait=True)
        if not ok:
            print("Deploy fehlgeschlagen. Logs:")
            print(mgr.logs(tail=20))
            return 1

        print(f"✓ OpenObserve läuft auf {endpoint}")
    else:
        print(f"── Verbinde mit bestehendem OpenObserve auf {endpoint} ──")

    # Connect adapter + test
    adapter = _make_adapter(endpoint, user, password)

    if not adapter.health_check():
        print(f"✗ OpenObserve nicht erreichbar auf {endpoint}")
        if args.no_docker:
            print("  Stelle sicher, dass OpenObserve läuft:")
            print(f"  ZO_ROOT_USER_EMAIL={user} ZO_ROOT_USER_PASSWORD=*** ./openobserve")
        return 1

    print("✓ Health-Check OK")

    # Send test data
    print("── Sende Test-Logs ──")
    results = _send_test_logs(adapter, count=args.test_count)
    print(f"  System: {results['system'].get('sent', 0)} gesendet")
    print(f"  Audit:  {results['audit'].get('sent', 0)} gesendet")

    if results["system"].get("failed") or results["audit"].get("failed"):
        print(f"  ⚠ Fehler: {results['system'].get('errors', [])} {results['audit'].get('errors', [])}")

    # Open dashboard
    if not args.no_browser:
        _open_browser(endpoint)

    print()
    print("── Bereit ──")
    print(f"  Dashboard:  {endpoint}")
    print(f"  Login:      {user}")
    print(f"  Streams:    system_logs, audit_logs")
    print()
    print("  Live-Logs aktivieren (in deinem Code oder toolbox.py):")
    print()
    print("    from toolboxv2.utils.system.observability_adapter import OpenObserveAdapter")
    print("    from toolboxv2.utils.system.tb_logger import enable_live_observability")
    print()
    print(f'    adapter = OpenObserveAdapter(endpoint="{endpoint}", credentials=("{user}", "***"))')
    print("    enable_live_observability(adapter)")
    print()
    print("  Oder den Helper nochmal mit --attach für laufende Instanzen.")
    print()
    print("  Stoppen:")
    print("    python -m toolboxv2.utils.system.observability_helper stop")

    return 0


# ---------------------------------------------------------------------------
# Scenario 2: server-setup
# ---------------------------------------------------------------------------

def cmd_server_setup(args):
    """Deploy OpenObserve on server with MinIO + historical import."""
    password = _ensure_password()

    # Set MinIO env vars from args if provided
    if args.minio_endpoint:
        os.environ["MINIO_ENDPOINT"] = args.minio_endpoint
    if args.minio_access_key:
        os.environ["MINIO_ACCESS_KEY"] = args.minio_access_key
    if args.minio_secret_key:
        os.environ["MINIO_SECRET_KEY"] = args.minio_secret_key

    # S3 storage backend
    if args.minio_endpoint:
        s3_url = args.minio_endpoint
        if not s3_url.startswith("http"):
            s3_url = f"http://{s3_url}"
        os.environ.setdefault("OPENOBSERVE_STORAGE", "s3")
        os.environ.setdefault("OPENOBSERVE_S3_ENDPOINT", s3_url)
        os.environ.setdefault("OPENOBSERVE_S3_ACCESS_KEY", args.minio_access_key or "")
        os.environ.setdefault("OPENOBSERVE_S3_SECRET_KEY", args.minio_secret_key or "")

    print("── Server Setup ──")

    mgr = _make_manager()

    if not mgr._docker_available():
        print("✗ Docker ist erforderlich für den Server-Setup.")
        return 1

    # Deploy
    print("Deploying OpenObserve ...")
    ok = mgr.deploy(pull=not args.no_pull, wait=True)
    if not ok:
        print("✗ Deploy fehlgeschlagen:")
        print(mgr.logs(tail=30))
        return 1

    endpoint = f"http://localhost:{mgr.port}"
    print(f"✓ OpenObserve läuft auf {endpoint}")

    # Historical import
    if args.import_bucket or mgr.import_enabled:
        print()
        print("── Historischer Import ──")
        stats = mgr.import_historical_logs(
            bucket=args.import_bucket or None,
            prefix=args.import_prefix or None,
            date_from=args.import_from or None,
            date_to=args.import_to or None,
        )

        if stats.get("error"):
            print(f"✗ Import-Fehler: {stats['error']}")
        else:
            print(f"✓ {stats.get('files_processed', 0)} Dateien verarbeitet")
            print(f"  System: {stats.get('system_sent', 0)} Einträge")
            print(f"  Audit:  {stats.get('audit_sent', 0)} Einträge")
            if stats.get("errors"):
                print(f"  ⚠ {len(stats['errors'])} Fehler: {stats['errors'][:3]}")
    else:
        print("  (Kein historischer Import — verwende --import-bucket zum Aktivieren)")

    print()
    print("── Server bereit ──")
    print(f"  Dashboard:   {endpoint}")
    print(f"  Login:       {mgr.user}")
    print()
    print("  Nächste Schritte:")
    print("    1. Nginx Reverse Proxy konfigurieren (siehe observability_guide.md)")
    print("    2. In toolbox.py: sync_manager.set_observability_adapter(adapter)")
    print("    3. Auto-Sync aktivieren: sync_manager.start_auto_sync(interval_seconds=60)")

    return 0


# ---------------------------------------------------------------------------
# Scenario 3: ci-push
# ---------------------------------------------------------------------------

def cmd_ci_push(args):
    """Send a single log entry from CI/CD pipeline."""
    adapter = _make_adapter()

    if not adapter.health_check():
        # CI: don't fail the build if observability is down
        print(f"⚠ OpenObserve nicht erreichbar, überspringe Log-Push", file=sys.stderr)
        return 0 if args.soft_fail else 1

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    node = _get_env("GITHUB_RUNNER_NAME", _get_env("HOSTNAME", platform.node()))

    # Parse optional metadata
    meta = {}
    if args.meta:
        try:
            meta = json.loads(args.meta)
        except json.JSONDecodeError:
            print(f"⚠ --meta ist kein gültiges JSON: {args.meta}", file=sys.stderr)

    stream = args.stream or "ci_logs"

    # System log entry
    entry = {
        "timestamp": now,
        "level": args.level.upper(),
        "name": "ci-pipeline",
        "message": args.message,
        "app_id": _get_env("GITHUB_REPOSITORY", "ci"),
        "node_id": node,
        "ci_run_id": _get_env("GITHUB_RUN_ID", ""),
        "ci_workflow": _get_env("GITHUB_WORKFLOW", ""),
        "ci_actor": _get_env("GITHUB_ACTOR", ""),
        **meta,
    }

    result = adapter.send_batch([entry], stream=stream)
    sent = result.get("sent", 0)

    # Optional audit entry
    if args.audit_action:
        audit = {
            "timestamp": now,
            "level": "INFO",
            "name": "ci-pipeline",
            "message": f"AUDIT: {args.audit_action} by {_get_env('GITHUB_ACTOR', 'ci')}",
            "audit_action": args.audit_action,
            "user_id": _get_env("GITHUB_ACTOR", "ci"),
            "resource": _get_env("GITHUB_REPOSITORY", "unknown"),
            "status": args.audit_status or "SUCCESS",
            "details": meta,
            "app_id": _get_env("GITHUB_REPOSITORY", "ci"),
            "node_id": node,
        }
        r2 = adapter.send_audit_batch([audit])
        if r2.get("sent"):
            print(f"✓ Audit: {args.audit_action} ({args.audit_status or 'SUCCESS'})")

    if sent:
        print(f"✓ [{args.level.upper()}] {args.message} → {stream}")
    else:
        print(f"⚠ Push fehlgeschlagen: {result.get('errors', [])}", file=sys.stderr)
        return 0 if args.soft_fail else 1

    return 0


# ---------------------------------------------------------------------------
# Status + Stop
# ---------------------------------------------------------------------------

def cmd_status(args):
    """Check OpenObserve health."""
    endpoint = _get_env("OPENOBSERVE_ENDPOINT", DEFAULT_ENDPOINT)

    adapter = _make_adapter()
    healthy = adapter.health_check()

    info = {
        "endpoint": endpoint,
        "healthy": healthy,
    }

    # Try Docker status
    try:
        mgr = _make_manager()
        docker_info = mgr.status()
        info.update(docker_info)
    except Exception:
        pass

    print(json.dumps(info, indent=2))
    return 0 if healthy else 1


def cmd_stop(args):
    """Stop OpenObserve container."""
    mgr = _make_manager()
    ok = mgr.stop()
    print("✓ Gestoppt." if ok else "✗ Stoppen fehlgeschlagen.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ToolBoxV2 Observability Helper — End-to-End Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Szenarien:
  local-debug    Lokales Debugging mit Dashboard
  server-setup   Server-Deployment + historischer Import
  ci-push        Einzelnen Log-Eintrag aus CI/CD senden
  status         Health-Check
  stop           Container stoppen

Beispiele:
  %(prog)s local-debug
  %(prog)s local-debug --no-docker
  %(prog)s server-setup --minio-endpoint localhost:9000 --import-from 2026-01-01
  %(prog)s ci-push --message "Build OK" --level INFO --audit-action DEPLOY
  %(prog)s status
  %(prog)s stop
""",
    )

    sub = parser.add_subparsers(dest="scenario", required=True)

    # ── local-debug ──
    p_local = sub.add_parser("local-debug", help="Lokales Debugging mit Dashboard")
    p_local.add_argument("--no-docker", action="store_true",
                         help="Kein Docker — verbinde mit bestehendem OpenObserve")
    p_local.add_argument("--no-pull", action="store_true",
                         help="Docker-Image nicht pullen")
    p_local.add_argument("--no-browser", action="store_true",
                         help="Browser nicht öffnen")
    p_local.add_argument("--test-count", type=int, default=5,
                         help="Anzahl Test-Logs (default: 5)")

    # ── server-setup ──
    p_server = sub.add_parser("server-setup", help="Server-Deployment mit MinIO + Import")
    p_server.add_argument("--no-pull", action="store_true")
    p_server.add_argument("--minio-endpoint", type=str, default="",
                          help="MinIO Endpoint (z.B. localhost:9000)")
    p_server.add_argument("--minio-access-key", type=str, default="")
    p_server.add_argument("--minio-secret-key", type=str, default="")
    p_server.add_argument("--import-bucket", type=str, default="",
                          help="MinIO Bucket mit historischen Logs")
    p_server.add_argument("--import-prefix", type=str, default="",
                          help="Objekt-Prefix (z.B. toolbox-instance-01/)")
    p_server.add_argument("--import-from", type=str, default="",
                          help="Import ab Datum YYYY-MM-DD")
    p_server.add_argument("--import-to", type=str, default="",
                          help="Import bis Datum YYYY-MM-DD")

    # ── ci-push ──
    p_ci = sub.add_parser("ci-push", help="Log aus CI/CD senden")
    p_ci.add_argument("--message", "-m", type=str, required=True,
                      help="Log-Nachricht")
    p_ci.add_argument("--level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Log-Level (default: INFO)")
    p_ci.add_argument("--stream", type=str, default="ci_logs",
                      help="Ziel-Stream (default: ci_logs)")
    p_ci.add_argument("--meta", type=str, default="",
                      help='JSON-Metadaten, z.B. \'{"commit":"abc"}\'')
    p_ci.add_argument("--audit-action", type=str, default="",
                      help="Optionales Audit-Event (z.B. DEPLOY, TEST)")
    p_ci.add_argument("--audit-status", type=str, default="SUCCESS",
                      choices=["SUCCESS", "FAILURE"],
                      help="Audit-Status (default: SUCCESS)")
    p_ci.add_argument("--soft-fail", action="store_true",
                      help="Exit 0 auch wenn Push fehlschlägt")

    # ── status ──
    sub.add_parser("status", help="Health-Check")

    # ── stop ──
    sub.add_parser("stop", help="Container stoppen")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    handlers = {
        "local-debug": cmd_local_debug,
        "server-setup": cmd_server_setup,
        "ci-push": cmd_ci_push,
        "status": cmd_status,
        "stop": cmd_stop,
    }

    handler = handlers.get(args.scenario)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
