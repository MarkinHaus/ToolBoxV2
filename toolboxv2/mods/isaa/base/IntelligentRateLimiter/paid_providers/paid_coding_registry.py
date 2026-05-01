# toolboxv2/mods/isaa/base/IntelligentRateLimiter/paid_coding_registry.py
from dataclasses import dataclass

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.free_providers.registry import ProviderSpec


@dataclass(frozen=True)
class CodingPlanSpec(ProviderSpec):
    plan_type: str = "coding_plan"
    monthly_flat_usd: float = 0.0
    has_open_api: bool = True
    plan_tier: str = ""                   # "lite" | "pro" | "max"


CODING_PLAN_REGISTRY: tuple[CodingPlanSpec, ...] = (
    CodingPlanSpec(
        id="zglm_lite", display_name="Z.AI GLM Coding Plan — Lite",
        signup_url="https://z.ai/subscribe",
        key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zglm/",
        multi_account_risk="red",
        multi_account_note="ToS erlaubt nur supported tools; SDK/third-party kann gedrosselt werden.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint=None, models_auth="custom",
        warm_models=("glm-4.7", "glm-4.6", "glm-4.5"),
        monthly_flat_usd=10.0, plan_tier="lite",
    ),
    CodingPlanSpec(
        id="zglm_pro", display_name="Z.AI GLM Coding Plan — Pro",
        signup_url="https://z.ai/subscribe",
        key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zglm/",
        multi_account_risk="red",
        multi_account_note="ToS erlaubt nur supported tools.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint=None, models_auth="custom",
        warm_models=("glm-5", "glm-4.7", "glm-4.6"),
        monthly_flat_usd=30.0, plan_tier="pro",
    ),
    CodingPlanSpec(
        id="alibaba_bailian", display_name="Alibaba Bailian Coding Plan (Pro)",
        signup_url="https://bailian.console.aliyun.com/",
        key_url="https://bailian.console.aliyun.com/?apiKey=1",
        env_key="BAILIAN_API_KEY", litellm_prefix="openai/",   # via custom base_url
        multi_account_risk="red",
        multi_account_note="Region-locked, CN account tracking strikt.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        models_auth="bearer",
        warm_models=("qwen3.6-plus", "qwen3-coder-plus", "kimi-k2.5", "glm-5", "minimax-m2.5"),
        monthly_flat_usd=28.0, plan_tier="pro",
    ),
    CodingPlanSpec(
        id="kimi_code", display_name="Kimi Code Plan",
        signup_url="https://platform.moonshot.ai/",
        key_url="https://platform.moonshot.ai/console/api-keys",
        env_key="KIMI_API_KEY", litellm_prefix="moonshot/",
        multi_account_risk="yellow",
        multi_account_note="Moonshot Plattform, CN-account.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.moonshot.ai/v1/models", models_auth="bearer",
        warm_models=("kimi-k2.5", "kimi-k2.6-code-preview"),
        monthly_flat_usd=15.0, plan_tier="pro",
    ),
    CodingPlanSpec(
        id="minimax_code", display_name="MiniMax Coding Plan",
        signup_url="https://www.minimax.io/",
        key_url="https://www.minimax.io/user-center/basic-information/interface-key",
        env_key="MINIMAX_API_KEY", litellm_prefix="minimax/",
        multi_account_risk="yellow",
        multi_account_note="CN-platform; you already run custom LiteLLM provider.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint=None, models_auth="custom",
        warm_models=("minimax-m2.5",),
        monthly_flat_usd=15.0, plan_tier="pro",
    ),
)

CODING_PLAN_BY_ID: dict[str, CodingPlanSpec] = {p.id: p for p in CODING_PLAN_REGISTRY}
