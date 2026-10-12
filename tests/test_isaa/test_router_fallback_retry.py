"""Regression: Router-Fallback-Retry nach Provider-Fehlern.

Deckt den Live-Crash vom 12.10. ab:
  AttributeError: 'IntelligentRateLimiter' object has no attribute 'get_active_model'
Der Router darf die Fallback-Ermittlung NUR ueber Strukturen loesen, die am
echten IntelligentRateLimiter existieren (fallback_manager / handle_rate_limit_error-
Rueckgabe) — nicht an einer Wunsch-API.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "/root/ToolBoxV2")

from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter import (
    create_handler_from_config,
)


class _Usage:
    total_tokens = 5
    prompt_tokens = 2
    completion_tokens = 3


class _Result:
    def __init__(self, content):
        self.content = content
        self.usage = _Usage()


class FailPrimaryAdapter:
    """Primary wirft 500, Backup antwortet normal (zwei Praefixe)."""

    def __init__(self):
        self.calls = []

    async def complete(self, session, api_key, model, messages, tools, **kw):
        self.calls.append(model)
        if model.rstrip() == "primary" or model == "prov/primary":
            raise RuntimeError('HTTP 500: {"error":{"code":"1234","message":"Internal network failure"}}')
        return _Result("OK-backup" if "backup" in model else f"OK-{model}")

    async def stream(self, session, api_key, model, messages, tools, metrics=None, **kw):
        self.calls.append(model)
        if model.rstrip() == "primary" or model == "prov/primary":
            raise RuntimeError("HTTP 500: socket-timeout-sim")
        yield _Result("OK-backup" if "backup" in model else f"OK-{model}")


class _NoBudget:
    def check(self, *a, **k):
        return None

    def track(self, *a, **k):
        return None


def _build_router(config, fail_adapter):
    r = CompletionRouter(strict_mode=False)
    r.register("prov", fail_adapter)
    handler = create_handler_from_config(config)
    r._rate_limiter = handler.rate_limiter  # exakt wie flow_agent.router-property
    r._budget = _NoBudget()
    return r


def _config():
    return {
        "features": {"rate_limiting": True, "model_fallback": True,
                     "key_rotation": True, "key_rotation_mode": "balance"},
        "api_keys": {},
        "fallback_chains": {"prov/primary": ["prov/backup"]},
        "limits": {},
    }


def test_complete_falls_back_with_real_rate_limiter():
    """500 am Primary -> complete() liefert Backup-Ergebnis, kein AttributeError."""
    adapter = FailPrimaryAdapter()
    router = _build_router(_config(), adapter)
    res = asyncio.run(router.complete("prov/primary", [{"role": "user", "content": "hi"}]))
    assert res.content == "OK-backup"
    assert any("backup" in c for c in adapter.calls)


def test_stream_falls_back_with_real_rate_limiter():
    """Stream-Fehler am Primary -> Retry ueber Chain, keine AttributeError."""

    async def _run():
        adapter = FailPrimaryAdapter()
        router = _build_router(_config(), adapter)
        out = []
        async for chunk in router.stream("prov/primary", [{"role": "user", "content": "hi"}]):
            out.append(chunk)
        return out, adapter.calls

    out, calls = asyncio.run(_run())
    assert any(getattr(c, "content", "") == "OK-backup" for c in out)
    assert any("backup" in c for c in calls)


def test_real_icli_config_chains_do_not_crash():
    """Echte rate_limiter_config.json (zglm/9rou-Chains) am echten Handler laden
    und einen Fehler-Retry durchlaufen lassen — regrediert den Live-Crash vom
    12.10.11:09 (get_active_model AttributeError)."""
    import json
    cfg_path = Path("/root/.config/ToolBoxV2/icli/rate_limiter_config.json")
    if not cfg_path.exists():
        print("skip: rate_limiter_config.json nicht vorhanden")
        return
    cfg = json.loads(cfg_path.read_text())
    cfg["fallback_chains"]["prov/primary"] = ["prov/backup"]  # Test-Praefix ergaenzen
    adapter = FailPrimaryAdapter()
    router = _build_router(cfg, adapter)
    res = asyncio.run(router.complete("prov/primary", [{"role": "user", "content": "hi"}]))
    assert res.content == "OK-backup"


def test_router_without_limiter_raises_normally():
    """Ohne Rate-Limiter: Fehler wird normal weitergereicht (kein neuer Crash)."""
    router = CompletionRouter(strict_mode=False)
    adapter = FailPrimaryAdapter()
    router.register("prov", adapter)
    router._budget = _NoBudget()
    try:
        asyncio.run(router.complete("prov/primary", [{"role": "user", "content": "hi"}]))
    except RuntimeError as e:
        assert "HTTP 500" in str(e)
    else:
        raise AssertionError("RuntimeError erwartet")
