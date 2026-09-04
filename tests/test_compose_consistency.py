"""Compose-Stack Konsistenz: Ports + Health-Endpoints +WS-Referenzen.

Port-Kanon (Port-Guide): Live-UI 8467, WS 8468, Manager 9010.
Bricht, wenn Compose vom Kanon abweicht (z.B. WS 8001 statt 8468).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGINX = ROOT / "docker" / "nginx" / "default.conf"
COMPOSE = ROOT / "compose.yaml"


def test_files_exist():
    assert COMPOSE.exists(), "compose.yaml fehlt"
    assert NGINX.exists(), "docker/nginx/default.conf fehlt"
    assert (ROOT / "Dockerfile.toolbox").exists()
    assert (ROOT / "Dockerfile.base").exists()


def test_worker_healthcheck_target():
    s = COMPOSE.read_text(encoding="utf-8")
    assert "localhost:8000/health" in s, "tb-worker healthcheck muss /health auf 8000 treffen"


def test_nginx_ws_upstream_matches_canon():
    """nginx WS-upstream muss auf den kanonischen WS-Port zeigen (8468)."""
    s = NGINX.read_text(encoding="utf-8")
    m = re.search(r"upstream tb_ws\s*\{(.+?)\}", s, re.S)
    assert m, "upstream tb_ws fehlt"
    assert "8001" not in m.group(1), f"WS-upstream auf altem Port 8001: {m.group(1)}"


def test_nginx_ws_location_uses_upgrade():
    s = NGINX.read_text(encoding="utf-8")
    ws_block = re.search(r"location /ws \{.*?\}", s, re.S)
    assert ws_block, "location /ws fehlt"
    assert "Upgrade" in ws_block.group(0), "WS-Location ohne Upgrade-Header"


def test_no_stray_8001_in_nginx():
    s = NGINX.read_text(encoding="utf-8")
    assert "8001" not in s, "nginx referenziert alten WS-Port 8001"


def test_nginx_conf_has_no_envsubst_vars():
    """Gemountete .conf wird von nginx NICHT via envsubst behandelt (nur templates/*.template).
    ${VAR} in server_name => [emerg] invalid host name => Restart-Loop (tb-nginx Log-Spam)."""
    s = NGINX.read_text(encoding="utf-8")
    assert "${" not in s, "envsubst-Variablen gehoeren in templates/*.template, nicht in gemountete .conf"
    assert "server_name _;" in s, "Catch-all server_name fehlt"


def test_nginx_serves_dist_static():
    """Compose-Mode: nginx servet dist direkt (root /usr/share/nginx/html/tb + SPA-Fallback),
    API bleibt proxied. UI muss laden, auch wenn tb-worker down ist (sonst nginx-502-Seite)."""
    s = NGINX.read_text(encoding="utf-8")
    assert "root /usr/share/nginx/html/tb;" in s, "Static-root fuer dist fehlt"
    assert "try_files" in s, "SPA-Fallback fehlt"
    c = COMPOSE.read_text(encoding="utf-8")
    assert "/usr/share/nginx/html/tb:ro" in c, "dist-Mount in compose nginx fehlt"
    assert "toolboxv2/dist/web:/usr/share/nginx/html/tb" in c, "dist-Quellpfad falsch"


def test_nginx_health_local_not_proxied():
    """/health muss lokal beantwortet werden — proxyt sie an einen downen Worker,
    failt der nginx-Container-Healthcheck (wget http://localhost/health) -> unhealthy-Spirale."""
    s = NGINX.read_text(encoding="utf-8")
    health_block = re.search(r"location /health \{.*?\}", s, re.S)
    assert health_block, "location /health fehlt"
    assert "proxy_pass" not in health_block.group(0), "/health darf nicht proxien (Healthcheck-Spirale)"
    assert "return 200" in health_block.group(0), "/health muss lokal 200 liefern"
