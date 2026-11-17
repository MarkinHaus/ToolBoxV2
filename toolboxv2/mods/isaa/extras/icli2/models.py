"""Modellliste fuer den Modell-Picker der icli2.

Quellen (jede einzeln abgesichert - ein Fehler entfernt nie die anderen):
  1. static_models():  MODEL_MAPPING aus flows/isaa/icli.py (als Text gelesen, nicht importiert)
  2. router_models():  Custom-Router zur Laufzeit
       - Gateway: IntelligentRateLimiter/gateway.py (nur mit TB_LLM_GATEWAY_URL, wie llm_router)
       - litellm.custom_provider_map: zur Laufzeit registrierte Handler mit get_available_models()
"""
from __future__ import annotations

try:
    from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.registry import (
        REGISTRY as _IRL_REGISTRY,
    )
except Exception:  # Quelle einzeln gesichert - IRL fehlt: Picker lebt weiter
    _IRL_REGISTRY = ()

import ast
import asyncio
import contextlib
import io
import os
import sys
import types
from pathlib import Path

GATEWAY_ENV = "TB_LLM_GATEWAY_URL"


def static_models() -> dict[str, str]:
    """alias -> voller Modellname aus MODEL_MAPPING in flows/isaa/icli.py.
    Die Datei wird als TEXT gelesen und per `ast` ausgewertet (ast.parse, die Zuweisung
    `MODEL_MAPPING = {...}` suchen, ast.literal_eval auf den Wert). Fehler -> {}."""
    try:
        import toolboxv2.flows.isaa as flows_isaa
        source = (Path(flows_isaa.__file__).parent / "icli.py").read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MODEL_MAPPING" for t in node.targets):
                value = ast.literal_eval(node.value)
                return {str(k): str(v) for k, v in value.items()}
    except Exception:
        pass
    return {}


def _fresh_names(handler) -> list[str]:
    """Modellnamen eines Handlers holen. Ein internes Cache-Attribut wird vorher
    zurueckgesetzt, damit neu hinzugekommene Modelle sichtbar werden. Stdout wird
    unterdrueckt, weil Handler bei Fehlern direkt drucken und die TUI stoeren wuerden."""
    if hasattr(handler, "_available_models"):
        handler._available_models = None
    with contextlib.redirect_stdout(io.StringIO()):
        return [str(n) for n in handler.get_available_models()]


def _from_gateway() -> list[tuple[str, str, str]]:
    """Gateway aus IntelligentRateLimiter/gateway.py. Aktiv nur mit TB_LLM_GATEWAY_URL.
    Das Modul exportiert keine Instanz, daher wird pro Aufruf eine frische GatewayProvider-Instanz gebaut."""
    if not os.getenv(GATEWAY_ENV):
        return []
    try:
        from toolboxv2.mods.isaa.base.IntelligentRateLimiter import gateway as gw
        if isinstance(gw, types.ModuleType):
            gw = gw.GatewayProvider()
        names = _fresh_names(gw)
    except Exception:
        return []
    return [(f"gateway/{n}", n, "gateway") for n in names]


def _from_custom_providers() -> list[tuple[str, str, str]]:
    """Handler aus litellm.custom_provider_map, die get_available_models() anbieten.
    litellm wird nur gelesen, wenn es schon geladen ist: ohne Import gibt es keine Registrierung."""
    litellm = sys.modules.get("litellm")
    rows: list[tuple[str, str, str]] = []
    for entry in list(getattr(litellm, "custom_provider_map", None) or []):
        try:
            prov = str(entry["provider"])
            names = _fresh_names(entry["custom_handler"])
        except Exception:
            continue
        rows += [(f"{prov}/{n}", n, prov) for n in names]
    return rows


def router_models() -> list[tuple[str, str, str]]:
    """Alle Modelle aus den Custom-Routern/Gateways.
    Rueckgabe: Liste von (voller_modellname, anzeigename, router_name). Wird bei JEDEM Aufruf
    neu gelesen, damit neue Router sofort erscheinen. Ein Fehler in einer Quelle wird uebersprungen."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for source in (_from_gateway, _from_custom_providers):
        try:
            rows = source()
        except Exception:
            continue
        for full, name, router in rows:
            if full not in seen:
                seen.add(full)
                out.append((full, name, router))
    return out


def _fetch_available_models(active_providers, extra_env=None, use_warm_fallback=True):
    """Netzgrenze: dünn gekapselt für Test-Fakes (async)."""
    from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.fetch_models import (
        fetch_available_models as _fetch,
    )
    return _fetch(active_providers, extra_env=extra_env, use_warm_fallback=use_warm_fallback)


_REAL_FETCH = _fetch_available_models


def irl_models(active_providers: dict | None = None) -> list[tuple[str, str, str]]:
    """Modellzeilen aller IRL-Provider: [(litellm_prefix+model_id, display, provider_id)].
    Quelle 1: fetch_models (aktive Provider mit Key, Netzgrenze gefaked in Tests).
    Quelle 2: REGISTRY warm_models (offline-Fallback, immer verfügbar)."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(prefix: str, model_id: str, display: str, provider: str) -> None:
        full = prefix + model_id
        if full not in seen:
            seen.add(full)
            rows.append((full, display, provider))

    fetched = None
    if active_providers:
        try:
            loop = asyncio.new_event_loop()
            try:
                fetched = loop.run_until_complete(
                    _fetch_available_models(active_providers))
            finally:
                loop.close()
        except Exception:
            fetched = None

    specs = {s.id: s for s in _IRL_REGISTRY}
    if fetched:
        for provider, model_list in fetched.items():
            spec = specs.get(provider)
            if spec is None:
                continue
            for mid in model_list:
                add(spec.litellm_prefix, str(mid), spec.display_name, provider)

    for spec in _IRL_REGISTRY:                      # offline-Fallback immer dazu
        for mid in getattr(spec, "warm_models", ()) or ():
            add(spec.litellm_prefix, mid, spec.display_name, spec.id)
    return rows


def all_models(current: list[tuple[str, str]] = ()) -> list[dict]:
    """Zeilen fuer den Picker: {"value": voller_modellname, "label": anzeigename, "hint": text}.
    Reihenfolge: 1. `current` (hint = aktuell schnell/komplex), 2. Router-Modelle, 3. statische Modelle.
    Werte aus `current` bleiben immer erhalten. Router-Modelle gewinnen gegen statische
    Modelle. Werte, die schon in `current` stehen, erscheinen in 2. und 3. nicht noch einmal."""
    rows: list[dict] = [{"value": str(full), "label": str(full), "hint": str(hint)} for full, hint in current]
    used = {r["value"] for r in rows}

    def add(value: str, label: str, hint: str) -> None:
        if value not in used:
            used.add(value)
            rows.append({"value": value, "label": label, "hint": hint})

    for full, name, router in router_models():
        add(full, name, f"Router: {router}")
    for alias, full in static_models().items():
        add(full, alias, full)
    for full, display, provider in irl_models():
        add(full, f"{display}: {full}", f"IRL ({provider})")
    return rows
