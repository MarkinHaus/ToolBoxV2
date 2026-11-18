# -*- coding: utf-8 -*-
"""T2: Modellliste aller IRL-Provider im icli2-Modell-Picker (statt nur Router/litellm)."""
import asyncio

import pytest

from toolboxv2.mods.isaa.extras.icli2 import models as m


def test_irl_models_function_exists():
    assert hasattr(m, "irl_models")


def test_irl_models_offline_registry():
    """Ohne Keys/Netz: warm_models der REGISTRY (offline-Quelle, kein Aufruf von litellm)."""
    rows = m.irl_models()
    assert rows, "Registry darf nicht leer sein"
    full, label, provider = rows[0]
    assert isinstance(full, str) and full
    assert provider in {s.id for s in m._IRL_REGISTRY}


def test_irl_models_prefix_is_litellm():
    """Voller Name = litellm_prefix + model_id (direkt im Agenten nutzbar)."""
    rows = m.irl_models()
    prefixes = {s.litellm_prefix for s in m._IRL_REGISTRY}
    full, label, provider = rows[0]
    spec = {s.id: s for s in m._IRL_REGISTRY}[provider]
    assert full.startswith(spec.litellm_prefix)


def test_irl_models_active_providers_preferred():
    """fetch_models-Ergebnis (aktive Provider) gewinnt ueber warm_models, Netzgrenze gefaked."""
    async def fake_fetch(active_providers, extra_env=None, use_warm_fallback=True):
        return {"groq": ["groq-test-model-x"]}

    m._fetch_available_models = fake_fetch
    try:
        rows = m.irl_models(active_providers={"groq": ["key"]})
    finally:
        m._fetch_available_models = m._REAL_FETCH
    by_provider = {}
    for full, label, prov in rows:
        by_provider.setdefault(prov, []).append(full)
    assert any(f.endswith("groq-test-model-x") for f in by_provider.get("groq", []))
    # warm-Modelle von groq verschwinden nicht (Dedup behaelt fetch-Ergebnis + warm)
    assert by_provider["groq"]


def test_all_models_contains_irl():
    """Picker-Zeilen enthalten IRL-Modelle, dedupliziert gegen current/router/static."""
    rows = m.all_models()
    values = [r["value"] for r in rows]
    assert any(v.startswith("groq/") or v.startswith("gemini/") for v in values)


def test_all_models_dedup_irl_vs_router():
    """Gleicher voller Name aus Router- und IRL-Quelle erscheint nur einmal."""
    rows = m.all_models()
    values = [r["value"] for r in rows]
    assert len(values) == len(set(values))
