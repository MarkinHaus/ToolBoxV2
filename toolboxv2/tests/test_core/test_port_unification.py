"""Port-Unifizierung: Kanonische Ports ueber alle Config-Quellen (Port-Guide).

Live-UI = 8467, WS = 8468 (= live_ui+1, Rust worker_manager.rs DEFAULT_WS_PORT),
Manager = 9010. Bricht, wenn eine Config-Quelle vom Kanon abweicht.
"""
import os
import sys
from pathlib import Path

CANON_LIVE_UI = 8467
CANON_WS = 8468
CANON_MANAGER = 9010


def test_ws_worker_config_default():
    from toolboxv2.utils.workers.config import WSWorkerConfig
    assert WSWorkerConfig().port == CANON_WS


def test_manifest_ws_instance_default():
    from toolboxv2.utils.manifest.schema import WSWorkerInstance
    assert WSWorkerInstance().port == CANON_WS


def test_manifest_live_ui_and_manager_ports():
    from toolboxv2.utils.manifest.schema import ManagerConfig as ManifestManager
    assert ManifestManager().live_ui_port == CANON_LIVE_UI
    assert ManifestManager().web_ui_port == CANON_MANAGER


def test_workers_config_manager_defaults():
    from toolboxv2.utils.workers.config import ManagerConfig
    assert ManagerConfig().live_ui_port == CANON_LIVE_UI
    assert ManagerConfig().web_ui_port == CANON_MANAGER


def test_endpoint_default_matches_canon():
    from toolboxv2.utils.workers.fast.endpoint import DEFAULT_LOCAL_UI_PORT
    assert DEFAULT_LOCAL_UI_PORT == CANON_LIVE_UI


def test_embedded_js_fallback_matches_canon():
    """Embedded JS: __TB_WS_PORT__ Fallback muss kanonischen WS-Port nutzen."""
    base = Path(__file__).resolve().parents[3]
    for rel in ("fast_tb.py", "fast_tb_defaults.py"):
        text = (base / "utils" / "workers" / rel).read_text(encoding="utf-8")
        assert "'8100'" not in text, f"{rel}: alter WS-Fallback 8100 vorhanden"
        assert "'8468'" in text, f"{rel}: kanonischer WS-Fallback 8468 fehlt"


def test_ws_floor_logic_yields_canon():
    """cli_worker_manager Floor: ws_base = max(ws.port, http.port+100) -> 8468."""
    from toolboxv2.utils.workers.config import HTTPWorkerConfig, WSWorkerConfig
    ws_port = WSWorkerConfig().port
    http_port = HTTPWorkerConfig().port
    ws_base = ws_port
    if ws_base < http_port + 100:
        ws_base = http_port + 100
    assert ws_base == CANON_WS
