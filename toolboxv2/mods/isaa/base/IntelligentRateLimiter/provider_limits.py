"""
Provider-spezifische Rate Limit Konfigurationen.

Diese Datei enthält detaillierte, aktuelle Rate Limits für verschiedene LLM Provider.
Stand: 2024 (aktualisiere bei Bedarf)

Quellen:
- OpenAI: https://platform.openai.com/docs/guides/rate-limits
- Anthropic: https://docs.anthropic.com/en/api/rate-limits
- Google/Vertex: https://ai.google.dev/gemini-api/docs/rate-limits
- Groq: https://console.groq.com/docs/rate-limits
- Together: https://docs.together.ai/docs/rate-limits
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    """API-Plan Tiers"""

    FREE = "free"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"
    PAY_AS_YOU_GO = "pay_as_you_go"
    ENTERPRISE = "enterprise"


@dataclass
class ModelRateLimit:
    """Rate Limits für ein spezifisches Model"""

    model_pattern: str  # Regex oder exakter Name
    rpm: int  # Requests per Minute
    rpd: int = None  # Requests per Day
    tpm: int = None  # Tokens per Minute
    tpd: int = None  # Tokens per Day
    input_tpm: int = None  # Input Tokens per Minute
    output_tpm: int = None  # Output Tokens per Minute
    context_window: int = None
    notes: str = ""


# ===== OPENAI RATE LIMITS =====
# https://platform.openai.com/docs/guides/rate-limits

OPENAI_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit("gpt-3.5-turbo", rpm=3, rpd=200, tpm=40000),
        ModelRateLimit("gpt-4o-mini", rpm=3, rpd=200, tpm=40000),
        ModelRateLimit("gpt-4o", rpm=3, rpd=200, tpm=30000),
    ],
    Tier.TIER_1: [  # $5+ spend
        ModelRateLimit("gpt-3.5-turbo", rpm=3500, tpm=200000),
        ModelRateLimit("gpt-4o-mini", rpm=500, tpm=200000),
        ModelRateLimit("gpt-4o", rpm=500, tpm=30000),
        ModelRateLimit("gpt-4-turbo", rpm=500, tpm=30000),
        ModelRateLimit("gpt-4", rpm=500, tpm=10000),
    ],
    Tier.TIER_2: [  # $50+ spend
        ModelRateLimit("gpt-3.5-turbo", rpm=3500, tpm=2000000),
        ModelRateLimit("gpt-4o-mini", rpm=5000, tpm=2000000),
        ModelRateLimit("gpt-4o", rpm=5000, tpm=450000),
        ModelRateLimit("gpt-4-turbo", rpm=5000, tpm=450000),
        ModelRateLimit("gpt-4", rpm=5000, tpm=40000),
    ],
    Tier.TIER_3: [  # $100+ spend
        ModelRateLimit("gpt-3.5-turbo", rpm=3500, tpm=4000000),
        ModelRateLimit("gpt-4o-mini", rpm=5000, tpm=4000000),
        ModelRateLimit("gpt-4o", rpm=5000, tpm=800000),
        ModelRateLimit("gpt-4-turbo", rpm=5000, tpm=800000),
        ModelRateLimit("gpt-4", rpm=5000, tpm=80000),
    ],
}


# ===== ANTHROPIC RATE LIMITS =====
# https://docs.anthropic.com/en/api/rate-limits

ANTHROPIC_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.TIER_1: [  # $0 - Build Tier
        ModelRateLimit(
            "claude-3-5-sonnet", rpm=50, tpm=40000, input_tpm=40000, output_tpm=8000
        ),
        ModelRateLimit(
            "claude-3-opus", rpm=50, tpm=20000, input_tpm=20000, output_tpm=4000
        ),
        ModelRateLimit(
            "claude-3-sonnet", rpm=50, tpm=40000, input_tpm=40000, output_tpm=8000
        ),
        ModelRateLimit(
            "claude-3-haiku", rpm=50, tpm=50000, input_tpm=50000, output_tpm=10000
        ),
    ],
    Tier.TIER_2: [  # $40-160 credit balance
        ModelRateLimit(
            "claude-3-5-sonnet", rpm=1000, tpm=80000, input_tpm=80000, output_tpm=16000
        ),
        ModelRateLimit(
            "claude-3-opus", rpm=1000, tpm=40000, input_tpm=40000, output_tpm=8000
        ),
        ModelRateLimit(
            "claude-3-sonnet", rpm=1000, tpm=80000, input_tpm=80000, output_tpm=16000
        ),
        ModelRateLimit(
            "claude-3-haiku", rpm=1000, tpm=100000, input_tpm=100000, output_tpm=20000
        ),
    ],
    Tier.TIER_3: [  # $160-400 credit balance
        ModelRateLimit(
            "claude-3-5-sonnet", rpm=2000, tpm=160000, input_tpm=160000, output_tpm=32000
        ),
        ModelRateLimit(
            "claude-3-opus", rpm=2000, tpm=80000, input_tpm=80000, output_tpm=16000
        ),
        ModelRateLimit(
            "claude-3-sonnet", rpm=2000, tpm=160000, input_tpm=160000, output_tpm=32000
        ),
        ModelRateLimit(
            "claude-3-haiku", rpm=2000, tpm=200000, input_tpm=200000, output_tpm=40000
        ),
    ],
    Tier.TIER_4: [  # $400+ credit balance
        ModelRateLimit(
            "claude-3-5-sonnet", rpm=4000, tpm=400000, input_tpm=400000, output_tpm=80000
        ),
        ModelRateLimit(
            "claude-3-opus", rpm=4000, tpm=400000, input_tpm=400000, output_tpm=80000
        ),
        ModelRateLimit(
            "claude-3-sonnet", rpm=4000, tpm=400000, input_tpm=400000, output_tpm=80000
        ),
        ModelRateLimit(
            "claude-3-haiku", rpm=4000, tpm=400000, input_tpm=400000, output_tpm=80000
        ),
    ],
}


# ===== GOOGLE GEMINI / VERTEX AI RATE LIMITS =====
# https://ai.google.dev/gemini-api/docs/rate-limits

GOOGLE_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit(
            "gemini-2.0-flash",
            rpm=10,
            rpd=1500,
            input_tpm=4000000,  # 4M input TPM
            notes="Free tier, resets daily",
        ),
        ModelRateLimit(
            "gemini-2.5-flash",
            rpm=10,
            rpd=500,
            input_tpm=250000,  # 250K input TPM - Das ist dein Limit!
            notes="Free tier, preview model",
        ),
        ModelRateLimit(
            "gemini-1.5-flash", rpm=15, rpd=1500, input_tpm=1000000, notes="Free tier"
        ),
        ModelRateLimit(
            "gemini-1.5-pro",
            rpm=2,
            rpd=50,
            input_tpm=32000,
            notes="Free tier, very limited",
        ),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit(
            "gemini-2.0-flash", rpm=2000, input_tpm=4000000, notes="Pay-as-you-go tier"
        ),
        ModelRateLimit(
            "gemini-2.5-flash", rpm=2000, input_tpm=4000000, notes="Pay-as-you-go tier"
        ),
        ModelRateLimit(
            "gemini-1.5-flash",
            rpm=2000,
            input_tpm=4000000,
        ),
        ModelRateLimit(
            "gemini-1.5-pro",
            rpm=1000,
            input_tpm=4000000,
        ),
    ],
}


# ===== GROQ RATE LIMITS =====
# https://console.groq.com/docs/rate-limits

GROQ_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit(
            "llama-3.1-70b-versatile", rpm=30, rpd=14400, tpm=6000, tpd=500000
        ),
        ModelRateLimit("llama-3.1-8b-instant", rpm=30, rpd=14400, tpm=6000, tpd=500000),
        ModelRateLimit(
            "llama-3.2-90b-vision-preview", rpm=30, rpd=7000, tpm=6000, tpd=250000
        ),
        ModelRateLimit("mixtral-8x7b-32768", rpm=30, rpd=14400, tpm=5000, tpd=500000),
        ModelRateLimit("gemma2-9b-it", rpm=30, rpd=14400, tpm=15000, tpd=500000),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit("llama-3.1-70b-versatile", rpm=1000, tpm=100000),
        ModelRateLimit("llama-3.1-8b-instant", rpm=1000, tpm=100000),
        ModelRateLimit("mixtral-8x7b-32768", rpm=1000, tpm=100000),
    ],
}


# ===== TOGETHER AI RATE LIMITS =====
# https://docs.together.ai/docs/rate-limits

TOGETHER_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit("*", rpm=60, notes="All models, free tier"),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit("*", rpm=600, notes="All models, paid tier"),
    ],
}


# ===== MISTRAL RATE LIMITS =====
# https://docs.mistral.ai/getting-started/rate-limiting/

MISTRAL_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit("mistral-tiny", rpm=5, notes="Free tier"),
        ModelRateLimit("mistral-small", rpm=5, notes="Free tier"),
        ModelRateLimit("mistral-medium", rpm=5, notes="Free tier"),
        ModelRateLimit("mistral-large", rpm=5, notes="Free tier"),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit("mistral-tiny", rpm=500, notes="Paid tier"),
        ModelRateLimit("mistral-small", rpm=500, notes="Paid tier"),
        ModelRateLimit("mistral-medium", rpm=500, notes="Paid tier"),
        ModelRateLimit("mistral-large", rpm=500, notes="Paid tier"),
    ],
}


# ===== COHERE RATE LIMITS =====

COHERE_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit("command-r", rpm=20, notes="Trial API key"),
        ModelRateLimit("command-r-plus", rpm=20, notes="Trial API key"),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit("command-r", rpm=10000, notes="Production API key"),
        ModelRateLimit("command-r-plus", rpm=10000, notes="Production API key"),
    ],
}

# ===== MINIMAX RATE LIMITS =====
# https://platform.minimax.io/docs/guides/rate-limits
# Hinweis: MiniMax hat kein separates Free/Paid Tier in den Docs --
# die unten stehenden Limits gelten pro Account (Pay-as-you-go).

MINIMAX_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit(
            "MiniMax-M2.5",
            rpm=500,
            tpm=20_000_000,
            notes="Gilt auch für MiniMax-M2.5-highspeed",
        ),
        ModelRateLimit(
            "MiniMax-M2.1",
            rpm=500,
            tpm=20_000_000,
            notes="Gilt auch für MiniMax-M2.1-highspeed",
        ),
        ModelRateLimit(
            "MiniMax-M2",
            rpm=500,
            tpm=20_000_000,
        ),
        ModelRateLimit(
            "speech-2",
            rpm=60,
            tpm=20_000,
            notes="T2A: speech-2.8-turbo/hd, speech-2.6-turbo/hd, speech-02-turbo/hd",
        ),
        ModelRateLimit(
            "MiniMax-Hailuo",
            rpm=5,
            notes="Video Generation: Hailuo-2.3, Hailuo-2.3-Fast, Hailuo-02",
        ),
        ModelRateLimit(
            "image-01",
            rpm=10,
            tpm=60,
            notes="Image Generation",
        ),
        ModelRateLimit(
            "Music",
            rpm=120,
            notes="Music-2.5, Music-2.0",
        ),
    ],
}


# ===== ZHIPU AI / GLM RATE LIMITS (Z.AI) =====
# https://docs.z.ai / https://bigmodel.cn
# Stand: März 2026
# WICHTIG: Zhipu veröffentlicht keine konkreten RPM/TPM-Zahlen in den Docs.
# Bekannte Facts:
#   - GLM-4.7-Flash und GLM-4.5-Flash sind vollständig kostenlos (kein RPD-Cap).
#   - Free-Tier: ~5-10 concurrent requests beobachtet, höhere Parallelität führt zu 429.
#   - Coding Plan (Subscription): 5-Stunden-Zyklus + Weekly-Quota, kein direktes RPM-Limit.
#   - Paid API (Pay-as-you-go): Concurrency-Limits je nach Account-Level, nicht öffentlich.
# Für Production-Nutzung: Limits über das Dashboard abfragen oder Zhipu kontaktieren.
# TODO fix wit real limits
ZHIPU_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit(
            "glm-4.7-flash",
            rpm=60,   # Geschätzt basierend auf Community-Beobachtungen (5-10 concurrent)
            notes="Vollständig kostenlos, kein RPD-Cap. Offizielles RPM nicht dokumentiert.",
        ),
        ModelRateLimit(
            "glm-4.5-flash",
            rpm=60,
            notes="Vollständig kostenlos, kein RPD-Cap. Offizielles RPM nicht dokumentiert.",
        ),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit(
            "glm-5",
            rpm=5,
            context_window=200_000,
            notes="Flagship, ~Opus-Level. Offizielles RPM/TPM nicht veröffentlicht. Kontakt: Zhipu Sales.",
        ),
        ModelRateLimit(
            "glm-4.7",
            rpm=10,
            context_window=200_000,
            notes="~Sonnet-Level. $0.60/$2.20 per 1M tokens. RPM nicht öffentlich.",
        ),
        ModelRateLimit(
            "glm-4.7-flashx",
            rpm=10,
            context_window=200_000,
            notes="Paid-Flash-Variante mit besseren Latenz-Garantien.",
        ),
        ModelRateLimit(
            "glm-4.5",
            rpm=10,
            context_window=128_000,
            notes="355B/32B active MoE. RPM nicht öffentlich dokumentiert.",
        ),
        ModelRateLimit(
            "glm-4.5-air",
            rpm=10,
            context_window=128_000,
            notes="106B/12B active MoE, günstiger.",
        ),
    ],
}


# ===== INCEPTION LABS / MERCURY RATE LIMITS =====
# https://docs.inceptionlabs.ai
# Stand: März 2026
# WICHTIG: Inception veröffentlicht KEINE öffentlichen RPM/TPM-Zahlen.
# Bekannte Facts:
#   - API-Endpoint: https://api.inceptionlabs.ai/v1 (OpenAI-kompatibel)
#   - Neue API-Keys erhalten 10M kostenlose Token als Startguthaben.
#   - Beta-Phase: Kein Kredit erforderlich für initiale Tests.
#   - Context Window: 16K (mercury-coder-small), höher bei mercury-2 / mercury-edit.
#   - Bei Rate-Limit-Errors: HTTP 429, Exponential Backoff empfohlen.
# Für genaue Limits: Dashboard unter platform.inceptionlabs.ai prüfen.

INCEPTION_LIMITS: Dict[Tier, List[ModelRateLimit]] = {
    Tier.FREE: [
        ModelRateLimit(
            model_pattern="mercury-coder-small",
            rpm=100,
            input_tpm=100_000,
            output_tpm=10_000,
            context_window=16_000,
            notes="Free tier rate limits: 100 RPM, 100k input TPM, 10k output TPM.",
        ),
    ],
    Tier.PAY_AS_YOU_GO: [
        ModelRateLimit(
            "mercury-2",
            rpm=100,
            context_window=16_000,
            notes="$0.25/$0.75 per 1M tokens. ~1000 t/s Throughput. RPM nicht öffentlich.",
        ),
        ModelRateLimit(
            "mercury-edit",
            rpm=100,
            notes="Code-Edit/Apply/FIM-Modell. RPM nicht öffentlich dokumentiert.",
        ),
    ],
}
# ===== HELPER FUNCTIONS =====


def get_limits_for_model(
    provider: str, model: str, tier: Tier = Tier.FREE
) -> ModelRateLimit:
    """
    Hole die Rate Limits für einen Provider/Model.

    Args:
        provider: Provider-Name (openai, anthropic, google, etc.)
        model: Model-Name
        tier: API-Tier

    Returns:
        ModelRateLimit oder None
    """
    provider = provider.lower()
    model = model.lower()

    provider_limits = {
        "openai": OPENAI_LIMITS,
        "anthropic": ANTHROPIC_LIMITS,
        "google": GOOGLE_LIMITS,
        "vertex_ai": GOOGLE_LIMITS,
        "groq": GROQ_LIMITS,
        "together": TOGETHER_LIMITS,
        "together_ai": TOGETHER_LIMITS,
        "mistral": MISTRAL_LIMITS,
        "cohere": COHERE_LIMITS,
        "minimax": MINIMAX_LIMITS,
        "mm": MINIMAX_LIMITS,
        "zhipu": ZHIPU_LIMITS,
        "glm": ZHIPU_LIMITS,
        "zglm": ZHIPU_LIMITS,
        "z.ai": ZHIPU_LIMITS,
        "inception": INCEPTION_LIMITS,
        "inceptionlabs": INCEPTION_LIMITS,
        "mercury": INCEPTION_LIMITS,
    }

    limits = provider_limits.get(provider, {})
    tier_limits = limits.get(tier, [])

    for limit in tier_limits:
        # Exakte Match oder Wildcard
        if limit.model_pattern == "*" or model.startswith(limit.model_pattern.lower()):
            return limit

    return None

all_providers = {
    "OpenAI": OPENAI_LIMITS,
    "Anthropic": ANTHROPIC_LIMITS,
    "Google/Vertex": GOOGLE_LIMITS,
    "Groq": GROQ_LIMITS,
    "Together AI": TOGETHER_LIMITS,
    "Mistral": MISTRAL_LIMITS,
    "Cohere": COHERE_LIMITS,
    "MiniMax": MINIMAX_LIMITS,
    "Zhipu AI / GLM (Z.AI)": ZHIPU_LIMITS,
    "Inception Labs / Mercury": INCEPTION_LIMITS,
}
def print_all_limits():
    """Drucke alle bekannten Limits übersichtlich"""


    for provider, limits in all_providers.items():
        print(f"\n{'=' * 60}")
        print(f" {provider}")
        print(f"{'=' * 60}")

        for tier, models in limits.items():
            print(f"\n  {tier.value.upper()}:")
            print(f"  {'-' * 50}")

            for model in models:
                print(f"    {model.model_pattern}:")
                print(f"      RPM: {model.rpm}", end="")
                if model.rpd:
                    print(f", RPD: {model.rpd}", end="")
                if model.tpm:
                    print(f", TPM: {model.tpm:,}", end="")
                if model.input_tpm:
                    print(f", Input TPM: {model.input_tpm:,}", end="")
                print()
                if model.notes:
                    print(f"      Note: {model.notes}")


if __name__ == "__main__":
    print_all_limits()

    # Beispiel: Hole Limits für dein Model
    print("\n" + "=" * 60)
    print(" Dein spezifisches Limit:")
    print("=" * 60)

    limit = get_limits_for_model("google", "gemini-2.5-flash", Tier.FREE)
    if limit:
        print(f"\nModel: {limit.model_pattern}")
        print(f"RPM: {limit.rpm}")
        print(f"RPD: {limit.rpd}")
        print(f"Input TPM: {limit.input_tpm:,}")
        print(f"Notes: {limit.notes}")
