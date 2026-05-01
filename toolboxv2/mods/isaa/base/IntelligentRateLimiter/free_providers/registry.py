# free_providers/registry.py
from dataclasses import dataclass

ZAI_FREE_BASE_URL = "https://api.z.ai/api/paas/v4/"  # hard-coded, user override via setup_zai_provider

@dataclass(frozen=True)
class ProviderSpec:
    id: str
    display_name: str
    signup_url: str
    key_url: str
    env_key: str
    litellm_prefix: str
    multi_account_risk: str          # green | yellow | red
    multi_account_note: str
    free_rpm: int
    free_tpd_low: int
    free_tpd_high: int
    models_endpoint: str | None
    models_auth: str                 # bearer | query_param | x-goog-api-key | cf-headers | custom
    warm_models: tuple[str, ...]
    tool_calling_reliable: bool = True
    extra_env: tuple[str, ...] = ()


REGISTRY: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="gemini", display_name="Google Gemini (AI Studio)",
        signup_url="https://aistudio.google.com/",
        key_url="https://aistudio.google.com/apikey",
        env_key="GEMINI_API_KEY", litellm_prefix="gemini/",
        multi_account_risk="red",
        multi_account_note="Per-project quota, Account-Tracking aggressiv. ToS-Verstoss.",
        free_rpm=15, free_tpd_low=50_000, free_tpd_high=8_000_000,
        models_endpoint="https://generativelanguage.googleapis.com/v1beta/models",
        models_auth="query_param",
        warm_models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
    ),
    ProviderSpec(
        id="groq", display_name="Groq",
        signup_url="https://console.groq.com/", key_url="https://console.groq.com/keys",
        env_key="GROQ_API_KEY", litellm_prefix="groq/",
        multi_account_risk="red",
        multi_account_note="ToS verbietet Multi-Account zur Limit-Umgehung explizit.",
        free_rpm=30, free_tpd_low=500_000, free_tpd_high=1_000_000,
        models_endpoint="https://api.groq.com/openai/v1/models", models_auth="bearer",
        warm_models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    ),
    ProviderSpec(
        id="cerebras", display_name="Cerebras",
        signup_url="https://cloud.cerebras.ai/", key_url="https://cloud.cerebras.ai/platform/",
        env_key="CEREBRAS_API_KEY", litellm_prefix="cerebras/",
        multi_account_risk="yellow",
        multi_account_note="Keine explizite ToS-Aussage, aber wafer-capacity knapp.",
        free_rpm=30, free_tpd_low=1_000_000, free_tpd_high=1_000_000,
        models_endpoint="https://api.cerebras.ai/v1/models", models_auth="bearer",
        warm_models=("llama-3.3-70b", "llama-4-scout-17b-16e-instruct", "qwen-3-32b"),
    ),
    ProviderSpec(
        id="mistral", display_name="Mistral La Plateforme (Experiment)",
        signup_url="https://console.mistral.ai/", key_url="https://console.mistral.ai/api-keys/",
        env_key="MISTRAL_API_KEY", litellm_prefix="mistral/",
        multi_account_risk="yellow",
        multi_account_note="EU-KYC; Multi-Acc schwierig, nicht ToS-explizit.",
        free_rpm=2, free_tpd_low=500_000, free_tpd_high=33_000_000,
        models_endpoint="https://api.mistral.ai/v1/models", models_auth="bearer",
        warm_models=("mistral-large-latest", "mistral-small-latest", "codestral-latest"),
    ),
    ProviderSpec(
        id="openrouter", display_name="OpenRouter (:free models)",
        signup_url="https://openrouter.ai/", key_url="https://openrouter.ai/keys",
        env_key="OPENROUTER_API_KEY", litellm_prefix="openrouter/",
        multi_account_risk="green",
        multi_account_note="Multi-Acc bringt fuer :free nichts - gleiche Limits.",
        free_rpm=20, free_tpd_low=100_000, free_tpd_high=2_000_000,
        models_endpoint="https://openrouter.ai/api/v1/models", models_auth="bearer",
        warm_models=(),
    ),
    # ProviderSpec(
    #     id="cloudflare", display_name="Cloudflare Workers AI",
    #     signup_url="https://dash.cloudflare.com/sign-up",
    #     key_url="https://dash.cloudflare.com/profile/api-tokens",
    #     env_key="CLOUDFLARE_API_KEY", litellm_prefix="cloudflare/",
    #     extra_env=("CLOUDFLARE_ACCOUNT_ID",),
    #     multi_account_risk="yellow",
    #     multi_account_note="Account-bound, payment-tracking moderat.",
    #     free_rpm=300, free_tpd_low=5_000, free_tpd_high=50_000,
    #     models_endpoint=None, models_auth="cf-headers",
    #     warm_models=(
    #         "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    #         "@cf/meta/llama-3.1-8b-instruct",
    #         "@cf/mistralai/mistral-small-3.1-24b-instruct",
    #     ),
    # ),
    ProviderSpec(
        id="nvidia_nim", display_name="NVIDIA NIM (build.nvidia.com)",
        signup_url="https://build.nvidia.com/",
        key_url="https://build.nvidia.com/settings/api-keys",
        env_key="NVIDIA_NIM_API_KEY", litellm_prefix="nvidia_nim/",
        multi_account_risk="yellow",
        multi_account_note="Phone-verif + dev-program; credits einmalig.",
        free_rpm=40, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://integrate.api.nvidia.com/v1/models", models_auth="bearer",
        warm_models=("meta/llama-3.1-70b-instruct", "deepseek-ai/deepseek-r1"),
    ),
    ProviderSpec(
        id="cohere", display_name="Cohere (Trial)",
        signup_url="https://dashboard.cohere.com/",
        key_url="https://dashboard.cohere.com/api-keys",
        env_key="COHERE_API_KEY", litellm_prefix="cohere/",
        multi_account_risk="red",
        multi_account_note="Non-commercial Klausel + Multi-Acc = double violation.",
        free_rpm=20, free_tpd_low=16_000, free_tpd_high=66_000,
        models_endpoint="https://api.cohere.com/v1/models", models_auth="bearer",
        warm_models=("command-r-plus", "command-r"),
    ),
    ProviderSpec(
        id="together_ai", display_name="Together AI (Trial credit)",
        signup_url="https://api.together.xyz/",
        key_url="https://api.together.xyz/settings/api-keys",
        env_key="TOGETHER_API_KEY", litellm_prefix="together_ai/",
        multi_account_risk="yellow",
        multi_account_note="Trial-credit-basiert.",
        free_rpm=60, free_tpd_low=0, free_tpd_high=0,
        models_endpoint="https://api.together.xyz/v1/models", models_auth="bearer",
        warm_models=(),
    ),
    ProviderSpec(
        id="huggingface", display_name="HuggingFace Inference API",
        signup_url="https://huggingface.co/join",
        key_url="https://huggingface.co/settings/tokens",
        env_key="HUGGINGFACE_API_KEY", litellm_prefix="huggingface/",
        multi_account_risk="green",
        multi_account_note="Multi-Acc in Community ueblich.",
        free_rpm=0, free_tpd_low=0, free_tpd_high=0,
        models_endpoint=None, models_auth="bearer",
        warm_models=(
            "meta-llama/Llama-3.3-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ),
        tool_calling_reliable=False,
    ),
    ProviderSpec(
        id="zai", display_name="Z.AI Free Flash / PAYG",
        signup_url="https://z.ai/", key_url="https://z.ai/manage-apikey/apikey-list",
        env_key="ZAI_API_KEY", litellm_prefix="zai/",
        multi_account_risk="yellow",
        multi_account_note="Free Flash-Modelle; account-bound.",
        free_rpm=10, free_tpd_low=500_000, free_tpd_high=1_000_000,
        models_endpoint=None, models_auth="custom",
        warm_models=("glm-4.7-flash", "glm-4.5-flash"),
    ),
)

REGISTRY_BY_ID: dict[str, ProviderSpec] = {p.id: p for p in REGISTRY}
