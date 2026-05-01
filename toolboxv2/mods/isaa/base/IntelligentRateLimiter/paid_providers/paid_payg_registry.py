# toolboxv2/mods/isaa/base/IntelligentRateLimiter/paid_payg_registry.py
from dataclasses import dataclass, field

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.registry import ProviderSpec


@dataclass(frozen=True)
class PaygProviderSpec(ProviderSpec):
    plan_type: str = "payg"
    has_open_api: bool = True
    # dict: model_id -> (input_usd_per_mtok, output_usd_per_mtok, cache_discount_0_to_1)
    price_table: dict[str, tuple[float, float, float]] = field(default_factory=dict)


PAYG_REGISTRY: tuple[PaygProviderSpec, ...] = (
    PaygProviderSpec(
        id="openai", display_name="OpenAI (PAYG)",
        signup_url="https://platform.openai.com/signup",
        key_url="https://platform.openai.com/api-keys",
        env_key="OPENAI_API_KEY", litellm_prefix="openai/",
        multi_account_risk="red",
        multi_account_note="Sofortige Sperre bei Multi-Acc.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.openai.com/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "gpt-5.4":        (2.50, 15.00, 0.90),
            "gpt-5.4-mini":   (0.75,  3.00, 0.90),
            "gpt-5.2":        (1.75, 14.00, 0.90),
            "gpt-4.1":        (2.00,  8.00, 0.90),
            "gpt-4.1-nano":   (0.10,  0.40, 0.90),
            "o4-mini":        (1.10,  4.40, 0.90),
        },
    ),
    PaygProviderSpec(
        id="anthropic", display_name="Anthropic Claude (PAYG)",
        signup_url="https://console.anthropic.com/",
        key_url="https://console.anthropic.com/settings/keys",
        env_key="ANTHROPIC_API_KEY", litellm_prefix="anthropic/",
        multi_account_risk="red",
        multi_account_note="Multi-Acc strikt verboten.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.anthropic.com/v1/models", models_auth="x-api-key",
        warm_models=(),
        price_table={
            "claude-opus-4-7":    (5.00, 25.00, 0.90),
            "claude-opus-4-6":    (5.00, 25.00, 0.90),
            "claude-sonnet-4-6":  (3.00, 15.00, 0.90),
            "claude-haiku-4-5":   (1.00,  5.00, 0.90),
            "claude-haiku-3":     (0.25,  1.25, 0.90),
        },
    ),
    PaygProviderSpec(
        id="gemini_payg", display_name="Google Gemini (PAYG)",
        signup_url="https://aistudio.google.com/",
        key_url="https://aistudio.google.com/apikey",
        env_key="GEMINI_API_KEY", litellm_prefix="gemini/",
        multi_account_risk="red",
        multi_account_note="Per-project billing.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://generativelanguage.googleapis.com/v1beta/models",
        models_auth="query_param",
        warm_models=(),
        price_table={
            "gemini-3.1-pro":        (2.00, 12.00, 0.0),
            "gemini-3-flash":        (0.50,  3.00, 0.0),
            "gemini-2.5-pro":        (1.25, 10.00, 0.0),
            "gemini-2.5-flash":      (0.30,  2.50, 0.0),
            "gemini-2.5-flash-lite": (0.10,  0.40, 0.0),
        },
    ),
    PaygProviderSpec(
        id="xai", display_name="xAI Grok (PAYG)",
        signup_url="https://console.x.ai/",
        key_url="https://console.x.ai/",
        env_key="XAI_API_KEY", litellm_prefix="xai/",
        multi_account_risk="yellow",
        multi_account_note="$25 free credits bei signup.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.x.ai/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "grok-4":           (3.00, 15.00, 0.0),
            "grok-4.1-fast":    (0.20,  0.50, 0.0),
            "grok-3":           (3.00, 15.00, 0.0),
            "grok-3-mini":      (0.30,  0.50, 0.0),
            "grok-code-fast-1": (0.20,  1.50, 0.0),
        },
    ),
    PaygProviderSpec(
        id="deepseek", display_name="DeepSeek (PAYG)",
        signup_url="https://platform.deepseek.com/",
        key_url="https://platform.deepseek.com/api_keys",
        env_key="DEEPSEEK_API_KEY", litellm_prefix="deepseek/",
        multi_account_risk="yellow",
        multi_account_note="CN-platform, gelegentliche Kapazitaetsengpaesse.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.deepseek.com/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "deepseek-chat":       (0.30, 0.50, 0.90),
            "deepseek-reasoner":   (0.55, 2.19, 0.90),
            "deepseek-v3.2":       (0.14, 0.28, 0.90),
        },
    ),
    PaygProviderSpec(
        id="mistral_payg", display_name="Mistral (PAYG)",
        signup_url="https://console.mistral.ai/",
        key_url="https://console.mistral.ai/api-keys/",
        env_key="MISTRAL_API_KEY", litellm_prefix="mistral/",
        multi_account_risk="yellow", multi_account_note="EU-KYC.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.mistral.ai/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "mistral-large-latest":  (2.00, 6.00, 0.0),
            "mistral-medium-latest": (1.00, 3.00, 0.0),
            "mistral-small-latest":  (0.20, 0.60, 0.0),
            "codestral-latest":      (0.20, 0.60, 0.0),
            "open-mistral-nemo":     (0.02, 0.02, 0.0),
        },
    ),
    PaygProviderSpec(
        id="zai_payg", display_name="Z.AI (PAYG)",
        signup_url="https://z.ai/", key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zai/",
        multi_account_risk="yellow", multi_account_note="Separate vom Coding Plan.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint=None, models_auth="custom",
        warm_models=(),
        price_table={
            "glm-5":   (1.00, 3.00, 0.0),
            "glm-4.7": (0.60, 2.00, 0.0),
            "glm-4.6": (0.50, 1.75, 0.0),
        },
    ),
    PaygProviderSpec(
        id="groq_payg", display_name="Groq (PAYG)",
        signup_url="https://console.groq.com/", key_url="https://console.groq.com/keys",
        env_key="GROQ_API_KEY", litellm_prefix="groq/",
        multi_account_risk="yellow", multi_account_note="Developer tier 25% token discount.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.groq.com/openai/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "llama-3.3-70b-versatile": (0.59, 0.79, 0.0),
            "llama-3.1-8b-instant":    (0.05, 0.08, 0.0),
        },
    ),
    PaygProviderSpec(
        id="cerebras_payg", display_name="Cerebras (PAYG)",
        signup_url="https://cloud.cerebras.ai/", key_url="https://cloud.cerebras.ai/platform/",
        env_key="CEREBRAS_API_KEY", litellm_prefix="cerebras/",
        multi_account_risk="yellow", multi_account_note="Paid tier; wafer capacity.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.cerebras.ai/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "llama-3.3-70b":               (0.65, 0.85, 0.0),
            "llama-4-scout-17b-16e-instruct": (0.65, 0.85, 0.0),
        },
    ),
)

PAYG_BY_ID: dict[str, PaygProviderSpec] = {p.id: p for p in PAYG_REGISTRY}


# =========================================================================
# Cost Calculator
# =========================================================================

INPUT_OUTPUT_RATIO = (3, 1)   # 3 parts input : 1 part output
TOKEN_STEP = 500_000          # 0.5M


def _resolve_price(provider_id: str, model_id: str) -> tuple[float, float, float]:
    spec = PAYG_BY_ID.get(provider_id)
    if spec is None or model_id not in spec.price_table:
        raise KeyError(f"No price for {provider_id}/{model_id}")
    return spec.price_table[model_id]


def cost_for_tokens(
    provider_id: str,
    model_id: str,
    total_tokens: int,
    ratio: tuple[int, int] = INPUT_OUTPUT_RATIO,
    cache_hit_fraction: float = 0.0,
) -> dict:
    """
    Split total_tokens by ratio (input:output). Return cost breakdown in USD.
    cache_hit_fraction: 0.0-1.0 = share of INPUT tokens that hit the cache.
    """
    in_p, out_p, cache_disc = _resolve_price(provider_id, model_id)
    r_in, r_out = ratio
    total_ratio = r_in + r_out
    input_tokens = total_tokens * r_in // total_ratio
    output_tokens = total_tokens - input_tokens

    cached_in = int(input_tokens * cache_hit_fraction)
    fresh_in = input_tokens - cached_in

    input_cost = (fresh_in / 1_000_000) * in_p + \
                 (cached_in / 1_000_000) * in_p * (1 - cache_disc)
    output_cost = (output_tokens / 1_000_000) * out_p

    return {
        "provider": provider_id,
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_in,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def cost_table_for_config(
    config: dict,
    max_tokens: int = 5_000_000,
    step: int = TOKEN_STEP,
    ratio: tuple[int, int] = INPUT_OUTPUT_RATIO,
    cache_hit_fraction: float = 0.0,
) -> dict:
    """
    Build a cost table for every PAYG model in a built config.
    Returns: {model_key: [(total_tokens, total_usd), ...]}
    model_key format: "provider_id/model_id"
    """
    out: dict[str, list[tuple[int, float]]] = {}
    steps = list(range(step, max_tokens + 1, step))

    for full_model_key in config.get("limits", {}):
        # full_model_key looks like "openai/gpt-5.4" — split on first slash
        prov_prefix, _, model_id = full_model_key.partition("/")
        # map litellm prefix back to provider id
        prov_id = _prefix_to_id(prov_prefix)
        if prov_id is None or prov_id not in PAYG_BY_ID:
            continue
        spec = PAYG_BY_ID[prov_id]
        if model_id not in spec.price_table:
            continue
        rows: list[tuple[int, float]] = []
        for t in steps:
            r = cost_for_tokens(prov_id, model_id, t, ratio, cache_hit_fraction)
            rows.append((t, r["total_cost_usd"]))
        out[full_model_key] = rows
    return out


def _prefix_to_id(prefix: str) -> str | None:
    for pid, spec in PAYG_BY_ID.items():
        if spec.litellm_prefix.rstrip("/") == prefix:
            return pid
    return None


def format_cost_table(table: dict, step: int = TOKEN_STEP) -> str:
    """Plain-text renderer, monospace-safe."""
    if not table:
        return "(no payg models in config)"
    models = sorted(table.keys())
    header_cols = [f"{t/1_000_000:>5.1f}M" for t, _ in table[models[0]]]
    lines = ["Model".ljust(40) + "".join(c.rjust(10) for c in header_cols)]
    lines.append("-" * len(lines[0]))
    for m in models:
        row = m.ljust(40)
        for _, cost in table[m]:
            row += f"${cost:>8.2f} "
        lines.append(row.rstrip())
    lines.append("")
    lines.append(f"Ratio input:output = {INPUT_OUTPUT_RATIO[0]}:{INPUT_OUTPUT_RATIO[1]}, step = {step:,} tokens")
    return "\n".join(lines)
