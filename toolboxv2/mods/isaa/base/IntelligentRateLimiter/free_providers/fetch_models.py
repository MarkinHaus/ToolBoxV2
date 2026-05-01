# free_providers/fetch_models.py
import asyncio
import os
import httpx

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.registry import REGISTRY_BY_ID, ProviderSpec


def _build_request(spec: ProviderSpec, key: str, account_id: str | None) -> tuple[str, dict, dict]:
    """Returns (url, headers, params)."""
    url, headers, params = spec.models_endpoint or "", {}, {}
    if spec.models_auth == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif spec.models_auth == "query_param":
        params["key"] = key
    elif spec.models_auth == "x-api-key":
        headers["x-api-key"] = key
    elif spec.models_auth == "cf-headers":
        if not account_id:
            raise RuntimeError("CF_ACCOUNT_ID required for cloudflare")
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
        headers["Authorization"] = f"Bearer {key}"
    return url, headers, params

def _filter_openrouter(payload: dict) -> list[str]:
    """
    Filter OpenRouter models to match:
      https://openrouter.ai/models?fmt=cards&input_modalities=text&max_price=0&supported_parameters=tools

    Criteria (AND):
      - id ends with ':free'
      - input modalities contain 'text' (may contain more)
      - prompt price == 0 AND completion price == 0
      - supported_parameters contains 'tools'
    """
    data = payload.get("data", []) if isinstance(payload, dict) else []
    out: list[str] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        mid = m.get("id", "")
        if not mid.endswith(":free"):
            continue

        arch = m.get("architecture") or {}
        modalities = arch.get("input_modalities") or []
        if "text" not in modalities:
            continue

        pricing = m.get("pricing") or {}
        try:
            prompt_p = float(pricing.get("prompt", "0") or 0)
            compl_p = float(pricing.get("completion", "0") or 0)
        except (TypeError, ValueError):
            continue
        if prompt_p != 0 or compl_p != 0:
            continue

        supported = m.get("supported_parameters") or []
        if "tools" not in supported:
            continue

        out.append(mid)
    return out

def _parse_models(spec: ProviderSpec, payload: dict | list) -> list[str]:
    # OpenAI-compatible: {"data": [{"id": "..."}]}
    if isinstance(payload, dict) and "data" in payload:
        return [m["id"] for m in payload["data"] if isinstance(m, dict) and "id" in m]
    # Gemini: {"models": [{"name": "models/gemini-..."}]}
    if isinstance(payload, dict) and "models" in payload:
        out = []
        for m in payload["models"]:
            name = m.get("name", "")
            out.append(name.split("/", 1)[1] if "/" in name else name)
        return out
    # Cloudflare: {"result": [{"name": "@cf/..."}]}
    if isinstance(payload, dict) and "result" in payload:
        return [m["name"] for m in payload["result"] if isinstance(m, dict) and "name" in m]
    return []


async def _fetch_one(
    client: httpx.AsyncClient, spec: ProviderSpec, key: str, account_id: str | None
) -> list[str]:
    if spec.models_endpoint is None and spec.models_auth != "cf-headers":
        return list(spec.warm_models)
    try:
        url, headers, params = _build_request(spec, key, account_id)
        r = await client.get(url, headers=headers, params=params, timeout=15.0)
        r.raise_for_status()
        payload = r.json()
        if spec.id == "openrouter":                      # <-- new branch
            models = _filter_openrouter(payload)
        else:
            models = _parse_models(spec, payload)
        return models or list(spec.warm_models)
    except Exception:
        return list(spec.warm_models)

async def fetch_available_models(
    active_providers: dict[str, list[str]],
    extra_env: dict[str, str] | None = None,
    use_warm_fallback: bool = True,
) -> dict[str, list[str]]:
    """Returns raw model IDs per provider (no litellm-prefix)."""
    extra_env = extra_env or {}
    account_id = extra_env.get("CF_ACCOUNT_ID") or os.environ.get("CF_ACCOUNT_ID")

    async with httpx.AsyncClient() as client:
        tasks = {
            pid: _fetch_one(client, REGISTRY_BY_ID[pid], keys[0], account_id)
            for pid, keys in active_providers.items()
            if pid in REGISTRY_BY_ID
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    out: dict[str, list[str]] = {}
    for pid, res in zip(tasks.keys(), results):
        if isinstance(res, Exception):
            out[pid] = list(REGISTRY_BY_ID[pid].warm_models) if use_warm_fallback else []
        else:
            out[pid] = res
    return out
