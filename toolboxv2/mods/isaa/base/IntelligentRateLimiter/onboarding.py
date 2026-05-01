# toolboxv2/mods/isaa/base/IntelligentRateLimiter/onboarding.py
"""
Free + paid provider onboarding. Produces a rate_limiter config dict
matching the schema consumed by IntelligentRateLimiter.

api_keys maps provider -> list of ENV VAR NAMES (not values).
Values stay in .env and are resolved at runtime.

Prices for PAYG models encoded in ModelRateLimit.notes as:
    "price_in=<float>,price_out=<float>,cache_discount=<float>"
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv, set_key
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text as pft

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.provider_limits import ModelRateLimit

# -----------------------------------------------------------------------------
# Registry — minimal metadata. Limits come from provider_limits._init_known_limits
# -----------------------------------------------------------------------------
ZAI_FREE_BASE_URL = "https://api.z.ai/api/paas/v4/"

RISK_COLOR = {"green": "ansigreen", "yellow": "ansiyellow", "red": "ansired"}

GLOBAL_WARNING = (
    "<ansired><b>[!] MULTI-ACCOUNT-WARNUNG</b></ansired>\n"
    "Die meisten Anbieter verbieten Multi-Account-Nutzung zur Umgehung von\n"
    "Rate-Limits. Account-Sperrung, Billing-Verweigerung und rechtliche Folgen\n"
    "moeglich. <b>Nutzung ausschliesslich auf eigenes Risiko. Wir raten ab.</b>\n"
)


@dataclass(frozen=True)
class ProviderMeta:
    id: str
    display_name: str
    category: str                       # "free" | "coding" | "payg"
    signup_url: str
    key_url: str
    env_key: str
    litellm_prefix: str
    multi_account_risk: str
    multi_account_note: str
    models_endpoint: Optional[str]
    models_auth: str
    warm_models: tuple[str, ...]
    extra_env: tuple[str, ...] = ()
    tool_calling_reliable: bool = True
    # PAYG price_table: model_id -> (input_usd_per_mtok, output_usd_per_mtok, cache_discount)
    price_table: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Coding plan monthly price
    monthly_flat_usd: float = 0.0
    plan_tier: str = ""


REGISTRY: tuple[ProviderMeta, ...] = (
    # -------------------- FREE --------------------
    ProviderMeta(
        id="gemini", display_name="Google Gemini (AI Studio)", category="free",
        signup_url="https://aistudio.google.com/",
        key_url="https://aistudio.google.com/apikey",
        env_key="GEMINI_API_KEY", litellm_prefix="gemini/",
        multi_account_risk="red",
        multi_account_note="Per-project quota, account tracking aggressiv.",
        models_endpoint="https://generativelanguage.googleapis.com/v1beta/models",
        models_auth="query_param",
        warm_models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
    ),
    ProviderMeta(
        id="groq", display_name="Groq", category="free",
        signup_url="https://console.groq.com/", key_url="https://console.groq.com/keys",
        env_key="GROQ_API_KEY", litellm_prefix="groq/",
        multi_account_risk="red",
        multi_account_note="ToS verbietet Multi-Account zur Limit-Umgehung explizit.",
        models_endpoint="https://api.groq.com/openai/v1/models", models_auth="bearer",
        warm_models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    ),
    ProviderMeta(
        id="cerebras", display_name="Cerebras", category="free",
        signup_url="https://cloud.cerebras.ai/", key_url="https://cloud.cerebras.ai/platform/",
        env_key="CEREBRAS_API_KEY", litellm_prefix="cerebras/",
        multi_account_risk="yellow",
        multi_account_note="Keine explizite ToS-Aussage, wafer capacity knapp.",
        models_endpoint="https://api.cerebras.ai/v1/models", models_auth="bearer",
        warm_models=("llama-3.3-70b", "llama-4-scout-17b-16e-instruct", "qwen-3-32b"),
    ),
    ProviderMeta(
        id="mistral", display_name="Mistral La Plateforme (Experiment)", category="free",
        signup_url="https://console.mistral.ai/", key_url="https://console.mistral.ai/api-keys/",
        env_key="MISTRAL_API_KEY", litellm_prefix="mistral/",
        multi_account_risk="yellow", multi_account_note="EU-KYC.",
        models_endpoint="https://api.mistral.ai/v1/models", models_auth="bearer",
        warm_models=("mistral-large-latest", "mistral-small-latest", "codestral-latest"),
    ),
    ProviderMeta(
        id="openrouter", display_name="OpenRouter (:free)", category="free",
        signup_url="https://openrouter.ai/", key_url="https://openrouter.ai/keys",
        env_key="OPENROUTER_API_KEY", litellm_prefix="openrouter/",
        multi_account_risk="green",
        multi_account_note="Multi-Acc bringt fuer :free nichts — gleiche limits.",
        models_endpoint="https://openrouter.ai/api/v1/models", models_auth="bearer",
        warm_models=(),
    ),
    ProviderMeta(
        id="cloudflare", display_name="Cloudflare Workers AI", category="free",
        signup_url="https://dash.cloudflare.com/sign-up",
        key_url="https://dash.cloudflare.com/profile/api-tokens",
        env_key="CLOUDFLARE_API_KEY", litellm_prefix="cloudflare/",
        extra_env=("CF_ACCOUNT_ID",),
        multi_account_risk="yellow", multi_account_note="Account-bound.",
        models_endpoint=None, models_auth="cf-headers",
        warm_models=(
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/mistralai/mistral-small-3.1-24b-instruct",
        ),
    ),
    ProviderMeta(
        id="nvidia_nim", display_name="NVIDIA NIM", category="free",
        signup_url="https://build.nvidia.com/",
        key_url="https://build.nvidia.com/settings/api-keys",
        env_key="NVIDIA_NIM_API_KEY", litellm_prefix="nvidia_nim/",
        multi_account_risk="yellow", multi_account_note="Phone-verif; credits einmalig.",
        models_endpoint="https://integrate.api.nvidia.com/v1/models", models_auth="bearer",
        warm_models=("meta/llama-3.1-70b-instruct", "deepseek-ai/deepseek-r1"),
    ),
    ProviderMeta(
        id="cohere", display_name="Cohere (Trial)", category="free",
        signup_url="https://dashboard.cohere.com/",
        key_url="https://dashboard.cohere.com/api-keys",
        env_key="COHERE_API_KEY", litellm_prefix="cohere/",
        multi_account_risk="red",
        multi_account_note="Non-commercial + Multi-Acc = double violation.",
        models_endpoint="https://api.cohere.com/v1/models", models_auth="bearer",
        warm_models=("command-r-plus", "command-r"),
    ),
    ProviderMeta(
        id="together_ai", display_name="Together AI (Trial)", category="free",
        signup_url="https://api.together.xyz/",
        key_url="https://api.together.xyz/settings/api-keys",
        env_key="TOGETHER_API_KEY", litellm_prefix="together_ai/",
        multi_account_risk="yellow", multi_account_note="Trial credit.",
        models_endpoint="https://api.together.xyz/v1/models", models_auth="bearer",
        warm_models=(),
    ),
    ProviderMeta(
        id="huggingface", display_name="HuggingFace Inference", category="free",
        signup_url="https://huggingface.co/join",
        key_url="https://huggingface.co/settings/tokens",
        env_key="HUGGINGFACE_API_KEY", litellm_prefix="huggingface/",
        multi_account_risk="green", multi_account_note="Multi-Acc in Community ueblich.",
        models_endpoint=None, models_auth="bearer",
        warm_models=(
            "meta-llama/Llama-3.3-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ),
        tool_calling_reliable=False,
    ),
    ProviderMeta(
        id="zai", display_name="Z.AI Free Flash / PAYG", category="free",
        signup_url="https://z.ai/", key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zai/",
        multi_account_risk="yellow", multi_account_note="Free Flash; account-bound.",
        models_endpoint=None, models_auth="custom",
        warm_models=("glm-4.7-flash", "glm-4.5-flash"),
    ),
    # -------------------- CODING PLANS --------------------
    ProviderMeta(
        id="zglm_lite", display_name="Z.AI GLM Coding — Lite", category="coding",
        signup_url="https://z.ai/subscribe",
        key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zglm/",
        multi_account_risk="red",
        multi_account_note="ToS: nur supported tools; SDK-Zugriff kann gedrosselt werden.",
        models_endpoint=None, models_auth="custom",
        warm_models=("glm-4.7", "glm-4.6", "glm-4.5"),
        monthly_flat_usd=10.0, plan_tier="lite",
    ),
    ProviderMeta(
        id="zglm_pro", display_name="Z.AI GLM Coding — Pro", category="coding",
        signup_url="https://z.ai/subscribe",
        key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zglm/",
        multi_account_risk="red",
        multi_account_note="ToS: nur supported tools.",
        models_endpoint=None, models_auth="custom",
        warm_models=("glm-5", "glm-4.7", "glm-4.6"),
        monthly_flat_usd=30.0, plan_tier="pro",
    ),
    ProviderMeta(
        id="alibaba_bailian", display_name="Alibaba Bailian Coding — Pro", category="coding",
        signup_url="https://bailian.console.aliyun.com/",
        key_url="https://bailian.console.aliyun.com/?apiKey=1",
        env_key="BAILIAN_API_KEY", litellm_prefix="openai/",
        multi_account_risk="red", multi_account_note="Region-locked.",
        models_endpoint="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        models_auth="bearer",
        warm_models=("qwen3.6-plus", "qwen3-coder-plus", "kimi-k2.5", "glm-5"),
        monthly_flat_usd=28.0, plan_tier="pro",
    ),
    ProviderMeta(
        id="kimi_code", display_name="Kimi Code Plan", category="coding",
        signup_url="https://platform.moonshot.ai/",
        key_url="https://platform.moonshot.ai/console/api-keys",
        env_key="KIMI_API_KEY", litellm_prefix="moonshot/",
        multi_account_risk="yellow", multi_account_note="CN-platform.",
        models_endpoint="https://api.moonshot.ai/v1/models", models_auth="bearer",
        warm_models=("kimi-k2.5", "kimi-k2.6-code-preview"),
        monthly_flat_usd=15.0, plan_tier="pro",
    ),
    ProviderMeta(
        id="minimax_code", display_name="MiniMax Coding", category="coding",
        signup_url="https://www.minimax.io/",
        key_url="https://www.minimax.io/user-center/basic-information/interface-key",
        env_key="MINIMAX_API_KEY", litellm_prefix="minimax/",
        multi_account_risk="yellow", multi_account_note="Custom provider present.",
        models_endpoint=None, models_auth="custom",
        warm_models=("minimax-m2.5",),
        monthly_flat_usd=15.0, plan_tier="pro",
    ),
    # -------------------- PAYG --------------------
    ProviderMeta(
        id="openai", display_name="OpenAI (PAYG)", category="payg",
        signup_url="https://platform.openai.com/signup",
        key_url="https://platform.openai.com/api-keys",
        env_key="OPENAI_API_KEY", litellm_prefix="openai/",
        multi_account_risk="red", multi_account_note="Sofortige Sperre bei Multi-Acc.",
        models_endpoint="https://api.openai.com/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "gpt-5.4":      (2.50, 15.00, 0.90),
            "gpt-5.4-mini": (0.75,  3.00, 0.90),
            "gpt-5.2":      (1.75, 14.00, 0.90),
            "gpt-4.1":      (2.00,  8.00, 0.90),
            "gpt-4.1-nano": (0.10,  0.40, 0.90),
            "o4-mini":      (1.10,  4.40, 0.90),
        },
    ),
    ProviderMeta(
        id="anthropic", display_name="Anthropic Claude (PAYG)", category="payg",
        signup_url="https://console.anthropic.com/",
        key_url="https://console.anthropic.com/settings/keys",
        env_key="ANTHROPIC_API_KEY", litellm_prefix="anthropic/",
        multi_account_risk="red", multi_account_note="Multi-Acc strikt verboten.",
        models_endpoint="https://api.anthropic.com/v1/models", models_auth="x-api-key",
        warm_models=(),
        price_table={
            "claude-opus-4-7":   (5.00, 25.00, 0.90),
            "claude-opus-4-6":   (5.00, 25.00, 0.90),
            "claude-sonnet-4-6": (3.00, 15.00, 0.90),
            "claude-haiku-4-5":  (1.00,  5.00, 0.90),
            "claude-haiku-3":    (0.25,  1.25, 0.90),
        },
    ),
    ProviderMeta(
        id="xai", display_name="xAI Grok (PAYG)", category="payg",
        signup_url="https://console.x.ai/", key_url="https://console.x.ai/",
        env_key="XAI_API_KEY", litellm_prefix="xai/",
        multi_account_risk="yellow", multi_account_note="$25 free credits bei signup.",
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
    ProviderMeta(
        id="deepseek", display_name="DeepSeek (PAYG)", category="payg",
        signup_url="https://platform.deepseek.com/",
        key_url="https://platform.deepseek.com/api_keys",
        env_key="DEEPSEEK_API_KEY", litellm_prefix="deepseek/",
        multi_account_risk="yellow", multi_account_note="CN-platform.",
        models_endpoint="https://api.deepseek.com/v1/models", models_auth="bearer",
        warm_models=(),
        price_table={
            "deepseek-chat":     (0.30, 0.50, 0.90),
            "deepseek-reasoner": (0.55, 2.19, 0.90),
            "deepseek-v3.2":     (0.14, 0.28, 0.90),
        },
    ),
)

REGISTRY_BY_ID: dict[str, ProviderMeta] = {p.id: p for p in REGISTRY}


# -----------------------------------------------------------------------------
# .env Persistence (kritisch — append-only, verified, 0600)
# -----------------------------------------------------------------------------
def _persist_key(env_path: Path, key: str, value: str) -> None:
    if not key or "=" in key or any(c in key for c in "\n\r \t"):
        raise ValueError(f"invalid env key: {key!r}")
    if any(c in value for c in "\n\r\x00"):
        raise ValueError("env value contains control chars")
    if not value:
        raise ValueError("empty value refused")

    env_path = env_path.expanduser().resolve()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.touch(mode=0o600)
    else:
        try:
            env_path.chmod(0o600)
        except OSError:
            pass

    existing = dotenv_values(env_path)
    if key in existing:
        raise ValueError(f"{key} already present — use _next_slot() for fresh slot")

    success, _, _ = set_key(str(env_path), key, value, quote_mode="always", encoding="utf-8")
    if not success:
        raise RuntimeError(f"dotenv set_key failed for {key}")

    reread = dotenv_values(env_path)
    if reread.get(key) != value:
        raise RuntimeError(f"verification failed for {key} at {env_path}")


def _next_slot(env_key: str, env_path: Path) -> str:
    env_path = env_path.expanduser().resolve()
    existing = dotenv_values(env_path) if env_path.exists() else {}
    if env_key not in existing:
        return env_key
    pattern = re.compile(rf"^{re.escape(env_key)}_(\d+)$")
    used = {int(m.group(1)) for k in existing if (m := pattern.match(k))}
    n = 2
    while n in used:
        n += 1
    return f"{env_key}_{n}"


def _env_var_names_for(spec: ProviderMeta, env_path: Path) -> list[str]:
    """Return all env var NAMES set for this provider (primary + _[n])."""
    existing = dotenv_values(env_path) if env_path.exists() else {}
    names: list[tuple[int, str]] = []
    if spec.env_key in existing and existing[spec.env_key]:
        names.append((0, spec.env_key))
    pattern = re.compile(rf"^{re.escape(spec.env_key)}_(\d+)$")
    for k in existing:
        m = pattern.match(k)
        if m and existing[k]:
            names.append((int(m.group(1)), k))
    names.sort(key=lambda x: x[0])
    return [n for _, n in names]


# -----------------------------------------------------------------------------
# Onboarding Result
# -----------------------------------------------------------------------------
@dataclass
class OnboardingResult:
    # provider_id -> list of ENV VAR NAMES (not values!)
    active_env_names: dict[str, list[str]] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)
    selected_models: dict[str, list[str]] = field(default_factory=dict)
    openrouter_credit_tier: str = "none"
    env_path: Path = field(default_factory=lambda: Path(".env"))

    def active_by_category(self, category: str) -> dict[str, list[str]]:
        return {
            pid: names for pid, names in self.active_env_names.items()
            if REGISTRY_BY_ID[pid].category == category
        }


# -----------------------------------------------------------------------------
# Scan
# -----------------------------------------------------------------------------
def scan_env(env_path: Path) -> dict[str, list[str]]:
    """provider_id -> [env var names]"""
    env_path = env_path.expanduser().resolve()
    out: dict[str, list[str]] = {}
    seen_env_keys: set[str] = set()
    for spec in REGISTRY:
        # for coding plans that share ZGLM_API_KEY, only register first one found
        if spec.env_key in seen_env_keys:
            continue
        names = _env_var_names_for(spec, env_path)
        if names:
            out[spec.id] = names
            seen_env_keys.add(spec.env_key)
    return out


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def _render_status_line(spec: ProviderMeta, active_names: list[str]) -> HTML:
    risk = RISK_COLOR[spec.multi_account_risk]
    status = (f"<ansigreen>[{len(active_names)} keys]</ansigreen>"
              if active_names else "<ansibrightblack>[-]</ansibrightblack>")
    extra = ""
    if spec.category == "coding" and spec.monthly_flat_usd:
        extra = f" <ansibrightblack>${spec.monthly_flat_usd:.0f}/mo</ansibrightblack>"
    elif spec.category == "payg" and spec.price_table:
        ps = [p[0] for p in spec.price_table.values()]
        extra = f" <ansibrightblack>${min(ps):.2f}-${max(ps):.2f}/M in</ansibrightblack>"
    tool = "" if spec.tool_calling_reliable else " <ansiyellow>(tools: unreliable)</ansiyellow>"
    return HTML(
        f"  <{risk}>●</{risk}> <b>{spec.display_name:<40}</b> {status}{extra}{tool}"
    )


def _render_all(active: dict[str, list[str]], include_paid: str) -> None:
    pft(HTML("<b>Free tier:</b>"))
    for spec in REGISTRY:
        if spec.category == "free":
            pft(_render_status_line(spec, active.get(spec.id, [])))
    if include_paid in ("coding", "all"):
        pft(HTML("\n<b>Coding plans:</b>"))
        for spec in REGISTRY:
            if spec.category == "coding":
                pft(_render_status_line(spec, active.get(spec.id, [])))
    if include_paid in ("payg", "all"):
        pft(HTML("\n<b>Pay-as-you-go:</b>"))
        for spec in REGISTRY:
            if spec.category == "payg":
                pft(_render_status_line(spec, active.get(spec.id, [])))


# -----------------------------------------------------------------------------
# Stats (reads limits from provider_limits.py via IntelligentRateLimiter)
# -----------------------------------------------------------------------------
# Mapping: Provider-Anzeigename in all_providers  →  Provider-IDs in REGISTRY
_PROVIDER_NAME_TO_IDS: dict[str, tuple[str, ...]] = {
    "OpenAI":                   ("openai",),
    "Anthropic":                ("anthropic",),
    "Google/Vertex":            ("gemini",),
    "Groq":                     ("groq",),
    "Together AI":              ("together_ai",),
    "Mistral":                  ("mistral",),
    "Cohere":                   ("cohere",),
    "MiniMax":                  ("minimax_code",),
    "Zhipu AI / GLM (Z.AI)":    ("zai", "zglm_lite", "zglm_pro"),
    "Inception Labs / Mercury": (),  # kein LiteLLM-Provider in REGISTRY
}

# Tier-Preferenz pro REGISTRY-Kategorie
_CATEGORY_TIER_PREFERENCE: dict[str, tuple[str, ...]] = {
    "free":   ("free",),
    "payg":   ("pay_as_you_go", "tier_1", "tier_2"),
    "coding": ("pay_as_you_go", "tier_1"),
}


def _resolve_limits_from_rate_limiter() -> dict[str, ModelRateLimit]:
    """
    Build {f"{provider_id}::{model_pattern}": ModelRateLimit} from all_providers.

    all_providers is a nested dict: {display_name: {Tier: [ModelRateLimit]}}
    We map display_name -> one or more provider_ids via _PROVIDER_NAME_TO_IDS
    and pick the tier matching the provider_id's category.
    """
    from .provider_limits import all_providers

    flat: dict[str, ModelRateLimit] = {}
    for display_name, tier_map in all_providers.items():
        provider_ids = _PROVIDER_NAME_TO_IDS.get(display_name, ())
        if not provider_ids:
            continue
        # normalize tier keys: could be Tier enum or str
        tier_by_value: dict[str, list[ModelRateLimit]] = {}
        for tier_key, models in tier_map.items():
            value = tier_key.value if hasattr(tier_key, "value") else str(tier_key)
            tier_by_value[value] = models

        for pid in provider_ids:
            spec = REGISTRY_BY_ID.get(pid)
            if spec is None:
                continue
            preferred_tiers = _CATEGORY_TIER_PREFERENCE.get(spec.category, ("free",))
            chosen: list[ModelRateLimit] = []
            for t in preferred_tiers:
                if t in tier_by_value:
                    chosen = tier_by_value[t]
                    break
            for mrl in chosen:
                flat[f"{pid}::{mrl.model_pattern}"] = mrl
    return flat


def _match_limit_for(provider_id: str, model_id: str,
                     flat: dict[str, ModelRateLimit]) -> Optional[ModelRateLimit]:
    """Find the first ModelRateLimit whose regex matches the model_id."""
    for k, mrl in flat.items():
        prov, pattern = k.split("::", 1)
        if prov != provider_id:
            continue
        if re.search(pattern, model_id):
            return mrl
    return None


def _render_final_stats(result: OnboardingResult, models_raw: dict[str, list[str]]) -> None:
    try:
        flat = _resolve_limits_from_rate_limiter()
    except Exception as e:
        pft(HTML(f"<ansired>  (could not load rate_limiter limits: {e})</ansired>"))
        flat = {}

    total_rpm = 0
    total_tpm_low = 0
    total_rpd = 0

    per_cat = {"free": (0, 0, 0), "coding": (0, 0, 0), "payg": (0, 0, 0)}

    for pid, names in result.active_env_names.items():
        spec = REGISTRY_BY_ID[pid]
        n_keys = len(names)
        models = models_raw.get(pid) or list(spec.warm_models)
        cat_rpm, cat_tpm, cat_rpd = per_cat[spec.category]

        for m in models:
            mrl = _match_limit_for(pid, m, flat)
            if mrl is None:
                continue
            if mrl.rpm:
                total_rpm += mrl.rpm * n_keys
                cat_rpm += mrl.rpm * n_keys
            if mrl.tpm:
                total_tpm_low += mrl.tpm * n_keys
                cat_tpm += mrl.tpm * n_keys
            if mrl.rpd:
                total_rpd += mrl.rpd * n_keys
                cat_rpd += mrl.rpd * n_keys
        per_cat[spec.category] = (cat_rpm, cat_tpm, cat_rpd)

    pft(HTML("\n<b>============ Summary ============</b>"))
    done = 0
    for cat in ("free", "coding", "payg"):
        rpm, tpm, rpd = per_cat[cat]
        if rpm == 0 and tpm == 0 and rpd == 0:
            continue
        done += 1
        label = {"free": "Free tier", "coding": "Coding plans", "payg": "Pay-as-you-go"}[cat]
        pft(HTML(f"\n<b>{label}</b>"))
        pft(HTML(f"  RPM (sum):     <ansigreen>{rpm:>12,}</ansigreen>"))
        pft(HTML(f"  TPM (sum):     <ansigreen>{tpm:>12,}</ansigreen>"))
        pft(HTML(f"  RPD (sum):     <ansigreen>{rpd:>12,}</ansigreen>"))
        if tpm:
            pft(HTML(f"  Tokens/day est: <ansibrightblack>{tpm*60*24:>12,}</ansibrightblack>"
                     " <ansibrightblack>(if TPM held constantly)</ansibrightblack>"))
    if done > 1:
        pft(HTML("\n<b>Total</b>"))
        pft(HTML(f"  Combined RPM:  <ansigreen>{total_rpm:>12,}</ansigreen>"))
        pft(HTML(f"  Combined TPM:  <ansigreen>{total_tpm_low:>12,}</ansigreen>"))
        pft(HTML(f"  Combined RPD:  <ansigreen>{total_rpd:>12,}</ansigreen>"))
        pft(HTML(""))


# -----------------------------------------------------------------------------
# Interactive flow
# -----------------------------------------------------------------------------
async def onboarding_flow(
    env_path: Path = Path(".env"),
    session: PromptSession | None = None,
    include_paid: str = "none",
) -> OnboardingResult:
    env_path = env_path.expanduser().resolve()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    session = session or PromptSession()

    loaded = load_dotenv(env_path, override=True)
    pft(HTML(f"<ansibrightblack>  .env: {env_path} "
             f"({'loaded' if loaded else 'empty/missing'})</ansibrightblack>"))

    active = scan_env(env_path)
    extra_env = {k: os.environ[k] for k in ("CF_ACCOUNT_ID",) if k in os.environ}

    pft(HTML("\n<b>=== Provider Onboarding ===</b>\n"))
    pft(HTML(GLOBAL_WARNING))
    pft(HTML("<b>Legende:</b> <ansigreen>●</ansigreen> ok  "
             "<ansiyellow>●</ansiyellow> grau  <ansired>●</ansired> ToS-Risiko\n"))
    _render_all(active, include_paid)

    while True:
        pft(HTML("\n<b>Optionen:</b> [r]escan  [a]dd provider  [s]kip to summary"))
        choice = (await session.prompt_async("> ")).strip().lower()

        if choice in ("s", "", "skip"):
            break
        if choice in ("r", "rescan"):
            load_dotenv(env_path, override=True)
            active = scan_env(env_path)
            _render_all(active, include_paid)
            continue
        if choice in ("a", "add"):
            await _add_provider(session, env_path, active, extra_env, include_paid)
            continue
        pft(HTML("<ansired>unknown option</ansired>"))

    or_tier = "none"
    if "openrouter" in active:
        pft(HTML("\n<b>OpenRouter:</b> credits eingezahlt?"))
        pft(HTML("  [1] unter $10 (→ 50 RPD)"))
        pft(HTML("  [2] $10 oder mehr (→ 1000 RPD)"))
        ans = (await session.prompt_async("> ")).strip()
        or_tier = "over_10" if ans == "2" else "under_10"

    selected = {pid: list(REGISTRY_BY_ID[pid].warm_models) for pid in active}

    return OnboardingResult(
        active_env_names=active,
        extra_env=extra_env,
        selected_models=selected,
        openrouter_credit_tier=or_tier,
        env_path=env_path,
    )


async def _add_provider(
    session: PromptSession, env_path: Path,
    active: dict[str, list[str]], extra_env: dict[str, str],
    include_paid: str,
) -> None:
    if include_paid == "none":
        pool = [s for s in REGISTRY if s.category == "free"]
    else:
        pft(HTML("\n<b>Kategorie:</b> [1] free  [2] coding plan  [3] payg"))
        cat = (await session.prompt_async("> ")).strip()
        cat_map = {"1": "free", "2": "coding", "3": "payg"}
        want = cat_map.get(cat, "free")
        pool = [s for s in REGISTRY if s.category == want]

    pft(HTML("\n<b>Verfuegbare Provider:</b>"))
    for i, spec in enumerate(pool, 1):
        pft(HTML(f"  [{i:>2}] {spec.display_name}"))
    raw = (await session.prompt_async("\nProvider-Nummer > ")).strip()
    try:
        spec = pool[int(raw) - 1]
    except (ValueError, IndexError):
        pft(HTML("<ansired>invalid</ansired>"))
        return

    pft(HTML(f"\n<b>{spec.display_name}</b>"))
    pft(HTML(f"  Signup:   <u>{spec.signup_url}</u>"))
    pft(HTML(f"  API Key:  <u>{spec.key_url}</u>"))
    pft(HTML(f"  Risk:     <{RISK_COLOR[spec.multi_account_risk]}>"
             f"{spec.multi_account_risk.upper()}</{RISK_COLOR[spec.multi_account_risk]}>"
             f" — {spec.multi_account_note}"))

    key = (await session.prompt_async(
        "API Key (empty = abort) > ", is_password=True,
    )).strip()
    if not key:
        pft(HTML("<ansibrightblack>  (aborted)</ansibrightblack>"))
        return

    slot = _next_slot(spec.env_key, env_path)
    try:
        _persist_key(env_path, slot, key)
    except Exception as e:
        pft(HTML(f"<ansired>✗ persist FAILED: {e}</ansired>"))
        return

    os.environ[slot] = key
    active.setdefault(spec.id, []).append(slot)
    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 10 else "***"
    pft(HTML(f"<ansigreen>✓ {slot} = {masked} saved to .env</ansigreen>"))

    for extra in spec.extra_env:
        if extra in os.environ:
            continue
        val = (await session.prompt_async(f"{extra} > ")).strip()
        if not val:
            continue
        try:
            _persist_key(env_path, extra, val)
            os.environ[extra] = val
            extra_env[extra] = val
            pft(HTML(f"<ansigreen>✓ {extra} saved</ansigreen>"))
        except Exception as e:
            pft(HTML(f"<ansired>✗ {extra} failed: {e}</ansired>"))


# -----------------------------------------------------------------------------
# Config builder — target schema: "rate_limiter": {...}
# -----------------------------------------------------------------------------
def build_config(
    result: OnboardingResult,
    models_raw: dict[str, list[str]],
    key_rotation_mode: str = "balance",
    enable_key_rotation: bool = True,
    enable_model_fallback: bool = True,
    max_retries: int = 3,
) -> dict:
    assert key_rotation_mode in ("drain", "balance")

    # api_keys: provider -> [env var NAMES]
    api_keys: dict[str, list[str]] = {}
    for pid, names in result.active_env_names.items():
        # merge across multiple specs sharing env_key (e.g. zglm_lite + zglm_pro)
        prefix = REGISTRY_BY_ID[pid].litellm_prefix.rstrip("/")
        api_keys.setdefault(prefix, list(names))

    custom_limits: dict[str, dict] = {}
    try:
        flat = _resolve_limits_from_rate_limiter()
    except Exception:
        flat = {}

    for pid, names in result.active_env_names.items():
        spec = REGISTRY_BY_ID[pid]
        models = models_raw.get(pid) or list(spec.warm_models)
        selected = result.selected_models.get(pid) or models

        for m in selected:
            if m not in models:
                continue
            full_key = f"{spec.litellm_prefix}{m}"
            entry: dict = {"is_free_tier": spec.category == "free"}

            mrl = _match_limit_for(pid, m, flat)
            if mrl:
                for field_name in ("rpm", "rpd", "tpm", "tpd",
                                   "input_tpm", "output_tpm", "context_window"):
                    v = getattr(mrl, field_name, None)
                    if v is not None:
                        entry[field_name] = v

            # PAYG prices → notes (Option A, respects ModelRateLimit schema)
            if spec.category == "payg" and m in spec.price_table:
                in_p, out_p, cache_d = spec.price_table[m]
                entry["notes"] = (
                    f"price_in={in_p},price_out={out_p},cache_discount={cache_d}"
                )
            elif spec.category == "coding" and spec.monthly_flat_usd:
                entry["notes"] = (
                    f"plan_tier={spec.plan_tier},monthly_flat_usd={spec.monthly_flat_usd}"
                )

            custom_limits[full_key] = entry

    fallback_chains = _derive_fallbacks(list(custom_limits.keys()))

    return {
        "rate_limiter": {
            "enable_rate_limiting": True,
            "enable_model_fallback": enable_model_fallback,
            "enable_key_rotation": enable_key_rotation,
            "key_rotation_mode": key_rotation_mode,
            "api_keys": api_keys,
            "fallback_chains": fallback_chains,
            "custom_limits": custom_limits,
            "max_retries": max_retries,
            "wait_if_all_exhausted": True,
        }
    }


def _derive_fallbacks(models: list[str]) -> dict[str, list[str]]:
    by_provider: dict[str, list[str]] = {}
    for m in models:
        prov = m.split("/", 1)[0]
        by_provider.setdefault(prov, []).append(m)

    order_hints = ("pro", "opus", "large", "70b", "120b", "405b", "235b",
                   "maverick", "sonnet", "flash", "small", "mini", "nano",
                   "scout", "8b", "lite", "haiku")

    def rank(name: str) -> int:
        lo = name.lower()
        for i, h in enumerate(order_hints):
            if h in lo:
                return i
        return len(order_hints)

    out: dict[str, list[str]] = {}
    for items in by_provider.values():
        if len(items) < 2:
            continue
        s = sorted(items, key=rank)
        for i, m in enumerate(s[:-1]):
            out[m] = s[i + 1:]
    return out


# -----------------------------------------------------------------------------
# Cost Calculator (reads prices from notes)
# -----------------------------------------------------------------------------
INPUT_OUTPUT_RATIO = (3, 1)
TOKEN_STEP = 500_000


def _parse_price_notes(notes: str) -> Optional[tuple[float, float, float]]:
    if not notes or "price_in=" not in notes:
        return None
    try:
        parts = dict(kv.split("=", 1) for kv in notes.split(","))
        return (
            float(parts["price_in"]),
            float(parts["price_out"]),
            float(parts.get("cache_discount", "0")),
        )
    except (ValueError, KeyError):
        return None


def cost_for_tokens(full_model_key: str, total_tokens: int, config: dict,
                    ratio: tuple[int, int] = INPUT_OUTPUT_RATIO,
                    cache_hit_fraction: float = 0.0) -> dict:
    entry = config["rate_limiter"]["custom_limits"].get(full_model_key)
    if entry is None:
        raise KeyError(f"{full_model_key} not in config")
    price = _parse_price_notes(entry.get("notes", ""))
    if price is None:
        raise KeyError(f"{full_model_key} is free tier, no price info")
    in_p, out_p, cache_d = price

    r_in, r_out = ratio
    total_ratio = r_in + r_out
    input_tokens = total_tokens * r_in // total_ratio
    output_tokens = total_tokens - input_tokens
    cached_in = int(input_tokens * cache_hit_fraction)
    fresh_in = input_tokens - cached_in

    input_cost = (fresh_in / 1_000_000) * in_p + \
                 (cached_in / 1_000_000) * in_p * (1 - cache_d)
    output_cost = (output_tokens / 1_000_000) * out_p
    return {
        "model": full_model_key,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cached_input_tokens": cached_in,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def cost_table(config: dict, max_tokens: int = 5_000_000, step: int = TOKEN_STEP,
               ratio: tuple[int, int] = INPUT_OUTPUT_RATIO,
               cache_hit_fraction: float = 0.0) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    steps = list(range(step, max_tokens + 1, step))
    for key, entry in config["rate_limiter"]["custom_limits"].items():
        if _parse_price_notes(entry.get("notes", "")) is None:
            continue
        out[key] = [(t, cost_for_tokens(key, t, config, ratio, cache_hit_fraction)["total_cost_usd"])
                    for t in steps]
    return out


def format_cost_table(table: dict, step: int = TOKEN_STEP) -> str:
    if not table:
        return "(no payg models)"
    models = sorted(table.keys())
    headers = [f"{t/1_000_000:>5.1f}M" for t, _ in table[models[0]]]
    lines = ["Model".ljust(50) + "".join(c.rjust(10) for c in headers)]
    lines.append("-" * len(lines[0]))
    for m in models:
        row = m.ljust(50)
        for _, cost in table[m]:
            row += f"${cost:>8.2f} "
        lines.append(row.rstrip())
    lines.append("")
    lines.append(f"Ratio in:out = {INPUT_OUTPUT_RATIO[0]}:{INPUT_OUTPUT_RATIO[1]}, step = {step:,}")
    return "\n".join(lines)
