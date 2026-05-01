# free_providers/build_config.py
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.paid_providers.paid_coding_registry import CODING_PLAN_BY_ID
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.paid_providers.paid_payg_registry import PAYG_BY_ID
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.registry import REGISTRY_BY_ID
from toolboxv2.mods.isaa.base.IntelligentRateLimiter.onboarding import OnboardingResult



def build_config(
    onboarding: OnboardingResult,
    models_raw: dict[str, list[str]],
    key_rotation_mode: str = "drain",
) -> dict:
    assert key_rotation_mode in ("drain", "balance")

    api_keys: dict[str, list[str]] = {}
    limits: dict[str, dict] = {}
    selected_prefixed: list[str] = []

    # --- FREE ---
    for pid, keys in onboarding.active_providers.items():
        spec = REGISTRY_BY_ID[pid]
        api_keys[pid] = list(keys)
        available = models_raw.get(pid) or list(spec.warm_models)
        chosen = onboarding.selected_models.get(pid, available)
        for m in chosen:
            key = f"{spec.litellm_prefix}{m}"
            selected_prefixed.append(key)
            limits[key] = {
                "rpm": spec.free_rpm * len(keys),
                "input_tpm": max(spec.free_tpd_low // (24 * 60), 0),
            }

    # --- CODING PLANS ---
    for pid, keys in onboarding.coding_providers.items():
        spec = CODING_PLAN_BY_ID[pid]
        api_keys[pid] = list(keys)
        for m in spec.warm_models:
            key = f"{spec.litellm_prefix}{m}"
            selected_prefixed.append(key)
            limits[key] = {
                "rpm": 0,
                "input_tpm": 0,
                "plan_type": "coding_plan",
                "monthly_flat_usd": spec.monthly_flat_usd,
                "plan_tier": spec.plan_tier,
            }

    # --- PAYG ---
    for pid, keys in onboarding.payg_providers.items():
        spec = PAYG_BY_ID[pid]
        api_keys[pid] = list(keys)
        available = models_raw.get(pid) or list(spec.price_table.keys())
        for m in available:
            if m not in spec.price_table:
                continue
            in_p, out_p, cache_d = spec.price_table[m]
            key = f"{spec.litellm_prefix}{m}"
            selected_prefixed.append(key)
            limits[key] = {
                "rpm": 0,
                "input_tpm": 0,
                "plan_type": "payg",
                "price_input_per_mtok": in_p,
                "price_output_per_mtok": out_p,
                "cache_discount": cache_d,
            }

    fallback_chains = _derive_fallbacks(selected_prefixed)

    return {
        "features": {
            "rate_limiting": True,
            "model_fallback": bool(fallback_chains),
            "key_rotation": any(len(v) > 1 for v in api_keys.values()),
            "key_rotation_mode": key_rotation_mode,
        },
        "api_keys": api_keys,
        "fallback_chains": fallback_chains,
        "limits": limits,
    }

def _derive_fallbacks(models: list[str]) -> dict[str, list[str]]:
    """Primitive heuristic: bigger model falls back to smaller model of same provider."""
    by_provider: dict[str, list[str]] = {}
    for m in models:
        prov = m.split("/", 1)[0]
        by_provider.setdefault(prov, []).append(m)

    order_hints = ("pro", "large", "70b", "120b", "405b", "235b", "maverick",
                   "flash", "small", "mini", "scout", "8b", "lite")

    def rank(name: str) -> int:
        lo = name.lower()
        for i, h in enumerate(order_hints):
            if h in lo:
                return i
        return len(order_hints)

    out: dict[str, list[str]] = {}
    for prov, items in by_provider.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=rank)
        for i, m in enumerate(items_sorted[:-1]):
            out[m] = items_sorted[i + 1:]
    return out
