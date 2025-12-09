"""
Intelligenter, selbst-adaptierender LLM Rate Limiter v2

Features:
- Automatische Extraktion von Rate-Limit-Informationen aus Fehlerantworten
- Provider- und modellspezifische Konfiguration
- Token-basiertes Rate Limiting (nicht nur Request-basiert)
- Exponential Backoff mit Jitter
- Persistente Limit-Datenbank für bekannte Provider/Modelle
- Dynamische Anpassung basierend auf tatsächlichem Verhalten

NEW v2:
- Model Fallback Chains: Automatischer Wechsel zu Fallback-Modellen bei Limit
- Multi-API-Key Management mit Drain/Balance Modi
- Kombinierbare Strategien: Key-Rotation + Model-Fallback
- Minimale Konfiguration erforderlich
"""

import asyncio
import time
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
from enum import Enum
from pathlib import Path
import random
from collections import defaultdict
from contextlib import asynccontextmanager
import hashlib

logger = logging.getLogger(__name__)


class QuotaType(Enum):
    """Verschiedene Quota-Typen die Provider verwenden"""
    REQUESTS_PER_MINUTE = "rpm"
    REQUESTS_PER_SECOND = "rps"
    REQUESTS_PER_DAY = "rpd"
    TOKENS_PER_MINUTE = "tpm"
    TOKENS_PER_DAY = "tpd"
    INPUT_TOKENS_PER_MINUTE = "input_tpm"
    OUTPUT_TOKENS_PER_MINUTE = "output_tpm"


class KeyRotationMode(Enum):
    """Modi für API-Key Rotation"""
    DRAIN = "drain"      # Ein Key bis Limit, dann nächster
    BALANCE = "balance"  # Gleichmäßige Verteilung über alle Keys


class FallbackReason(Enum):
    """Grund für Fallback"""
    RATE_LIMIT = "rate_limit"
    KEY_EXHAUSTED = "key_exhausted"
    MODEL_UNAVAILABLE = "model_unavailable"
    ERROR = "error"


@dataclass
class ProviderModelLimits:
    """Rate Limits für ein spezifisches Provider/Model Paar"""
    provider: str
    model: str

    # Request-basierte Limits
    requests_per_minute: int = 60
    requests_per_second: int = 10
    requests_per_day: Optional[int] = None

    # Token-basierte Limits
    tokens_per_minute: Optional[int] = None
    tokens_per_day: Optional[int] = None
    input_tokens_per_minute: Optional[int] = None
    output_tokens_per_minute: Optional[int] = None

    # Metadata
    is_free_tier: bool = False
    last_updated: float = field(default_factory=time.time)
    confidence: float = 0.5

    # Dynamisch gelernte Werte
    observed_retry_delays: list = field(default_factory=list)
    rate_limit_hits: int = 0
    successful_requests: int = 0


@dataclass
class RateLimitState:
    """Aktueller Zustand für ein Provider/Model Paar"""
    minute_window: list = field(default_factory=list)
    second_window: list = field(default_factory=list)
    day_window: list = field(default_factory=list)
    tokens_minute_window: list = field(default_factory=list)
    tokens_day_window: list = field(default_factory=list)
    backoff_until: float = 0.0
    consecutive_failures: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class APIKeyInfo:
    """Information über einen API-Key"""
    key: str
    key_hash: str  # Für Logging ohne Key-Exposure
    provider: str

    # Usage Tracking
    requests_made: int = 0
    tokens_used: int = 0
    rate_limit_hits: int = 0
    last_used: float = 0.0

    # State
    exhausted_until: float = 0.0  # Timestamp bis wann Key exhausted ist
    is_active: bool = True
    priority: int = 0  # Niedrigere Zahl = höhere Priorität

    # Per-Key Limits (falls unterschiedlich)
    custom_rpm: Optional[int] = None
    custom_tpm: Optional[int] = None


@dataclass
class ModelFallbackConfig:
    """Konfiguration für Model-Fallback"""
    primary_model: str
    fallback_models: List[str] = field(default_factory=list)

    # Timing
    fallback_duration: float = 60.0  # Wie lange Fallback aktiv bleibt
    cooldown_check_interval: float = 10.0  # Wie oft Primary gecheckt wird

    # Behavior
    auto_recover: bool = True  # Automatisch zu Primary zurück wenn verfügbar
    inherit_api_keys: bool = True  # Fallback nutzt gleiche Keys wie Primary


@dataclass
class FallbackState:
    """Aktueller Fallback-Zustand für ein Model"""
    is_in_fallback: bool = False
    current_fallback_index: int = 0
    fallback_started: float = 0.0
    reason: Optional[FallbackReason] = None
    original_model: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ===== API KEY MANAGER =====

class APIKeyManager:
    """
    Verwaltet mehrere API-Keys pro Provider.

    Features:
    - Drain Mode: Ein Key bis Limit, dann nächster
    - Balance Mode: Round-Robin über alle Keys
    - Automatische Key-Rotation bei Limits
    - Per-Key Tracking
    """

    def __init__(self, default_mode: KeyRotationMode = KeyRotationMode.BALANCE):
        # provider -> [APIKeyInfo]
        self._keys: Dict[str, List[APIKeyInfo]] = defaultdict(list)
        # provider -> current index (für Drain Mode)
        self._current_index: Dict[str, int] = defaultdict(int)
        # provider -> rotation counter (für Balance Mode)
        self._rotation_counter: Dict[str, int] = defaultdict(int)
        # Global mode
        self._mode = default_mode
        # Lock für Thread-Safety
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> KeyRotationMode:
        return self._mode

    @mode.setter
    def mode(self, value: Union[KeyRotationMode, str]):
        if isinstance(value, str):
            value = KeyRotationMode(value)
        self._mode = value
        logger.info(f"Key rotation mode set to: {value.value}")

    def add_key(
        self,
        provider: str,
        key: str,
        priority: int = 0,
        custom_rpm: Optional[int] = None,
        custom_tpm: Optional[int] = None,
    ) -> str:
        """
        Füge einen API-Key hinzu.

        Args:
            provider: Provider-Name (z.B. "vertex_ai", "openai")
            key: Der API-Key
            priority: Niedrigere Zahl = höhere Priorität
            custom_rpm: Optionales custom RPM-Limit für diesen Key
            custom_tpm: Optionales custom TPM-Limit für diesen Key

        Returns:
            Key-Hash für Referenz
        """
        provider = provider.lower().strip()
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]

        # Prüfe ob Key bereits existiert
        for existing in self._keys[provider]:
            if existing.key_hash == key_hash:
                logger.warning(f"Key {key_hash} already exists for {provider}")
                return key_hash

        key_info = APIKeyInfo(
            key=key,
            key_hash=key_hash,
            provider=provider,
            priority=priority,
            custom_rpm=custom_rpm,
            custom_tpm=custom_tpm,
        )

        self._keys[provider].append(key_info)
        # Sortiere nach Priorität
        self._keys[provider].sort(key=lambda k: k.priority)

        logger.info(f"Added API key {key_hash} for {provider}")
        return key_hash

    def remove_key(self, provider: str, key_hash: str) -> bool:
        """Entferne einen API-Key"""
        provider = provider.lower().strip()

        for i, key_info in enumerate(self._keys[provider]):
            if key_info.key_hash == key_hash:
                del self._keys[provider][i]
                logger.info(f"Removed API key {key_hash} from {provider}")
                return True
        return False

    async def get_next_key(self, provider: str) -> Optional[APIKeyInfo]:
        """
        Hole den nächsten verfügbaren API-Key.

        Berücksichtigt:
        - Globalen Key-Modus (Drain/Balance)
        - Exhausted Status
        - Priorität
        """
        async with self._lock:
            provider = provider.lower().strip()
            keys = self._keys.get(provider, [])

            if not keys:
                return None

            now = time.time()
            available_keys = [k for k in keys if k.is_active and k.exhausted_until < now]

            if not available_keys:
                # Alle Keys exhausted - finde den mit kürzester Wartezeit
                return min(keys, key=lambda k: k.exhausted_until) if keys else None

            if self._mode == KeyRotationMode.DRAIN:
                # Verwende aktuellen Key bis exhausted
                idx = self._current_index[provider] % len(available_keys)
                return available_keys[idx]
            else:
                # Balance: Round-Robin
                idx = self._rotation_counter[provider] % len(available_keys)
                self._rotation_counter[provider] += 1
                return available_keys[idx]

    def mark_key_exhausted(
        self,
        provider: str,
        key_hash: str,
        duration: float = 60.0,
        advance_to_next: bool = True
    ):
        """
        Markiere einen Key als temporär exhausted.

        Args:
            provider: Provider-Name
            key_hash: Hash des Keys
            duration: Wie lange der Key exhausted ist (Sekunden)
            advance_to_next: Bei Drain-Mode zum nächsten Key wechseln
        """
        provider = provider.lower().strip()

        for i, key_info in enumerate(self._keys[provider]):
            if key_info.key_hash == key_hash:
                key_info.exhausted_until = time.time() + duration
                key_info.rate_limit_hits += 1

                if advance_to_next and self._mode == KeyRotationMode.DRAIN:
                    self._current_index[provider] = (i + 1) % len(self._keys[provider])

                logger.info(f"Key {key_hash} exhausted for {duration:.0f}s")
                break

    def mark_key_used(self, provider: str, key_hash: str, tokens: int = 0):
        """Registriere Key-Nutzung"""
        provider = provider.lower().strip()

        for key_info in self._keys[provider]:
            if key_info.key_hash == key_hash:
                key_info.requests_made += 1
                key_info.tokens_used += tokens
                key_info.last_used = time.time()
                break

    def get_all_keys(self, provider: str) -> List[APIKeyInfo]:
        """Hole alle Keys für einen Provider"""
        return self._keys.get(provider.lower().strip(), [])

    def get_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Hole Statistiken über alle Keys"""
        if provider:
            keys = self._keys.get(provider.lower().strip(), [])
            return self._stats_for_keys(keys)

        return {p: self._stats_for_keys(k) for p, k in self._keys.items()}

    def _stats_for_keys(self, keys: List[APIKeyInfo]) -> Dict[str, Any]:
        now = time.time()
        return {
            "total_keys": len(keys),
            "active_keys": sum(1 for k in keys if k.is_active and k.exhausted_until < now),
            "exhausted_keys": sum(1 for k in keys if k.exhausted_until >= now),
            "total_requests": sum(k.requests_made for k in keys),
            "total_tokens": sum(k.tokens_used for k in keys),
            "total_rate_limits": sum(k.rate_limit_hits for k in keys),
            "rotation_mode": self._mode.value,
            "keys": [
                {
                    "hash": k.key_hash,
                    "requests": k.requests_made,
                    "tokens": k.tokens_used,
                    "rate_limits": k.rate_limit_hits,
                    "exhausted": k.exhausted_until >= now,
                    "exhausted_remaining": max(0, k.exhausted_until - now),
                }
                for k in keys
            ]
        }


# ===== MODEL FALLBACK MANAGER =====

class ModelFallbackManager:
    """
    Verwaltet Model-Fallback-Chains.

    Features:
    - Automatischer Wechsel zu Fallback bei Rate-Limit
    - Timed Recovery zu Primary Model
    - Kaskadierender Fallback (Primary -> Fallback1 -> Fallback2 -> ...)
    """

    # Bekannte Fallback-Chains (sinnvolle Defaults)
    DEFAULT_FALLBACK_CHAINS: Dict[str, List[str]] = {
        # Vertex AI / Google
        "vertex_ai/gemini-2.5-pro": [
            "vertex_ai/gemini-2.5-flash",
            "vertex_ai/gemini-2.0-flash",
        ],
        "vertex_ai/gemini-1.5-pro": [
            "vertex_ai/gemini-1.5-flash",
            "vertex_ai/gemini-2.5-flash",
        ],
        "google/gemini-2.5-pro": [
            "google/gemini-2.5-flash",
            "google/gemini-2.0-flash",
        ],
        # OpenAI
        "openai/gpt-4": [
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
        ],
        "openai/gpt-4o": [
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
        ],
        # Anthropic
        "anthropic/claude-3-opus": [
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
        ],
        "anthropic/claude-3-sonnet": [
            "anthropic/claude-3-haiku",
        ],
    }

    def __init__(self, use_defaults: bool = True):
        # model -> ModelFallbackConfig
        self._configs: Dict[str, ModelFallbackConfig] = {}
        # model -> FallbackState
        self._states: Dict[str, FallbackState] = defaultdict(FallbackState)

        if use_defaults:
            self._init_default_chains()

    def _init_default_chains(self):
        """Initialisiere Default-Fallback-Chains"""
        for primary, fallbacks in self.DEFAULT_FALLBACK_CHAINS.items():
            self.add_fallback_chain(primary, fallbacks)

    def add_fallback_chain(
        self,
        primary_model: str,
        fallback_models: List[str],
        fallback_duration: float = 60.0,
        cooldown_check_interval: float = 10.0,
        auto_recover: bool = True,
    ) -> None:
        """
        Füge eine Fallback-Chain hinzu.

        Args:
            primary_model: Das primäre Model
            fallback_models: Liste von Fallback-Models (in Prioritätsreihenfolge)
            fallback_duration: Wie lange bleibt Fallback aktiv (Sekunden)
            cooldown_check_interval: Wie oft wird Primary gecheckt
            auto_recover: Automatisch zu Primary zurück wenn verfügbar
        """
        primary_model = self._normalize_model(primary_model)
        fallback_models = [self._normalize_model(m) for m in fallback_models]

        self._configs[primary_model] = ModelFallbackConfig(
            primary_model=primary_model,
            fallback_models=fallback_models,
            fallback_duration=fallback_duration,
            cooldown_check_interval=cooldown_check_interval,
            auto_recover=auto_recover,
        )

        logger.info(f"Added fallback chain: {primary_model} -> {fallback_models}")

    def add_fallback_model(self, primary_model: str, fallback_model: str) -> None:
        """Füge ein einzelnes Fallback-Model hinzu"""
        primary_model = self._normalize_model(primary_model)
        fallback_model = self._normalize_model(fallback_model)

        if primary_model not in self._configs:
            self._configs[primary_model] = ModelFallbackConfig(
                primary_model=primary_model,
                fallback_models=[fallback_model],
            )
        else:
            if fallback_model not in self._configs[primary_model].fallback_models:
                self._configs[primary_model].fallback_models.append(fallback_model)

        logger.info(f"Added fallback: {primary_model} -> {fallback_model}")

    def _normalize_model(self, model: str) -> str:
        """Normalisiere Model-Namen"""
        return model.lower().strip()

    async def get_active_model(self, requested_model: str) -> Tuple[str, bool]:
        """
        Hole das aktuell zu verwendende Model.

        Returns:
            (active_model, is_fallback)
        """
        requested_model = self._normalize_model(requested_model)

        if requested_model not in self._configs:
            return requested_model, False

        config = self._configs[requested_model]
        state = self._states[requested_model]

        async with state.lock:
            now = time.time()

            # Prüfe ob Fallback noch aktiv sein sollte
            if state.is_in_fallback:
                elapsed = now - state.fallback_started

                if elapsed > config.fallback_duration and config.auto_recover:
                    # Versuche Recovery zu Primary
                    state.is_in_fallback = False
                    state.current_fallback_index = 0
                    logger.info(f"Auto-recovering to primary model: {requested_model}")
                    return requested_model, False

                # Noch in Fallback - gib aktuelles Fallback-Model zurück
                if state.current_fallback_index < len(config.fallback_models):
                    fallback = config.fallback_models[state.current_fallback_index]
                    return fallback, True

            return requested_model, False

    async def trigger_fallback(
        self,
        model: str,
        reason: FallbackReason = FallbackReason.RATE_LIMIT,
        duration: Optional[float] = None,
    ) -> Optional[str]:
        """
        Trigger Fallback für ein Model.

        Returns:
            Das neue aktive Model oder None wenn kein Fallback verfügbar
        """
        model = self._normalize_model(model)

        if model not in self._configs:
            return None

        config = self._configs[model]
        state = self._states[model]

        async with state.lock:
            if not config.fallback_models:
                return None

            # Wenn bereits in Fallback, versuche nächstes Fallback-Model
            if state.is_in_fallback:
                state.current_fallback_index += 1
                if state.current_fallback_index >= len(config.fallback_models):
                    # Alle Fallbacks erschöpft
                    logger.warning(f"All fallbacks exhausted for {model}")
                    return None
            else:
                state.is_in_fallback = True
                state.current_fallback_index = 0
                state.fallback_started = time.time()
                state.reason = reason
                state.original_model = model

            if duration:
                # Override duration wenn angegeben
                config.fallback_duration = duration

            fallback = config.fallback_models[state.current_fallback_index]
            logger.info(f"Fallback triggered: {model} -> {fallback} (reason: {reason.value})")

            return fallback

    async def reset_fallback(self, model: str):
        """Setze Fallback-State zurück (manuell oder nach erfolgreicher Recovery)"""
        model = self._normalize_model(model)
        state = self._states[model]

        async with state.lock:
            state.is_in_fallback = False
            state.current_fallback_index = 0
            state.reason = None
            logger.info(f"Fallback reset for {model}")

    def get_fallback_chain(self, model: str) -> Optional[List[str]]:
        """Hole die Fallback-Chain für ein Model"""
        model = self._normalize_model(model)
        config = self._configs.get(model)
        return config.fallback_models if config else None

    def get_state(self, model: str) -> Dict[str, Any]:
        """Hole den aktuellen Fallback-State"""
        model = self._normalize_model(model)
        state = self._states.get(model)
        config = self._configs.get(model)

        if not state or not config:
            return {"configured": False}

        now = time.time()
        return {
            "configured": True,
            "is_in_fallback": state.is_in_fallback,
            "current_fallback": (
                config.fallback_models[state.current_fallback_index]
                if state.is_in_fallback and state.current_fallback_index < len(config.fallback_models)
                else None
            ),
            "fallback_index": state.current_fallback_index,
            "fallback_chain": config.fallback_models,
            "fallback_started": state.fallback_started,
            "fallback_elapsed": now - state.fallback_started if state.is_in_fallback else 0,
            "reason": state.reason.value if state.reason else None,
        }


# ===== INTELLIGENT RATE LIMITER (CORE) =====

class IntelligentRateLimiter:
    """
    Intelligenter Rate Limiter der sich automatisch an Provider-Limits anpasst.

    v2 Features:
    - Model Fallback Chains
    - Multi-API-Key Management
    - Kombinierbare Strategien
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        default_rpm: int = 60,
        default_rps: int = 10,
        safety_margin: float = 0.9,
        enable_token_tracking: bool = True,
        persist_learned_limits: bool = True,
        # v2 Options
        enable_model_fallback: bool = True,
        enable_key_rotation: bool = True,
        use_default_fallback_chains: bool = True,
        key_rotation_mode: str = "balance",  # "drain" or "balance"
    ):
        self.config_path = config_path or Path.home() / ".llm_rate_limits.json"
        self.default_rpm = default_rpm
        self.default_rps = default_rps
        self.safety_margin = safety_margin
        self.enable_token_tracking = enable_token_tracking
        self.persist_learned_limits = persist_learned_limits

        # v2 Feature Flags
        self.enable_model_fallback = enable_model_fallback
        self.enable_key_rotation = enable_key_rotation

        # Core State
        self.limits: Dict[str, ProviderModelLimits] = {}
        self.states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._global_lock = asyncio.Lock()

        # v2 Managers
        mode = KeyRotationMode(key_rotation_mode)
        self.key_manager = APIKeyManager(default_mode=mode)
        self.fallback_manager = ModelFallbackManager(use_defaults=use_default_fallback_chains)

        # Load & Initialize
        self._load_limits()
        self._init_known_limits()

    def _init_known_limits(self):
        """Initialisiere bekannte Default-Limits für gängige Provider"""
        known_limits = [
            # OpenAI
            ProviderModelLimits(
                provider="openai", model="gpt-4",
                requests_per_minute=500, tokens_per_minute=40000, confidence=0.8,
            ),
            ProviderModelLimits(
                provider="openai", model="gpt-4-turbo",
                requests_per_minute=500, tokens_per_minute=150000, confidence=0.8,
            ),
            ProviderModelLimits(
                provider="openai", model="gpt-4o",
                requests_per_minute=500, tokens_per_minute=150000, confidence=0.8,
            ),
            ProviderModelLimits(
                provider="openai", model="gpt-4o-mini",
                requests_per_minute=500, tokens_per_minute=200000, confidence=0.8,
            ),
            ProviderModelLimits(
                provider="openai", model="gpt-3.5-turbo",
                requests_per_minute=3500, tokens_per_minute=90000, confidence=0.8,
            ),
            # Anthropic
            ProviderModelLimits(
                provider="anthropic", model="claude-3-opus",
                requests_per_minute=50, tokens_per_minute=40000, confidence=0.8,
            ),
            ProviderModelLimits(
                provider="anthropic", model="claude-3-sonnet",
                requests_per_minute=50, tokens_per_minute=80000, confidence=0.8,
            ),
            ProviderModelLimits(
                provider="anthropic", model="claude-3-haiku",
                requests_per_minute=50, tokens_per_minute=100000, confidence=0.8,
            ),
            # Google/Vertex AI - Free Tier
            ProviderModelLimits(
                provider="vertex_ai", model="gemini-2.5-pro",
                requests_per_minute=2, input_tokens_per_minute=32000,
                is_free_tier=True, confidence=0.9,
            ),
            ProviderModelLimits(
                provider="vertex_ai", model="gemini-2.5-flash",
                requests_per_minute=15, input_tokens_per_minute=250000,
                is_free_tier=True, confidence=0.9,
            ),
            ProviderModelLimits(
                provider="vertex_ai", model="gemini-2.0-flash",
                requests_per_minute=15, input_tokens_per_minute=250000,
                is_free_tier=True, confidence=0.9,
            ),
            ProviderModelLimits(
                provider="vertex_ai", model="gemini-1.5-pro",
                requests_per_minute=2, input_tokens_per_minute=32000,
                is_free_tier=True, confidence=0.9,
            ),
            ProviderModelLimits(
                provider="vertex_ai", model="gemini-1.5-flash",
                requests_per_minute=15, input_tokens_per_minute=250000,
                is_free_tier=True, confidence=0.9,
            ),
            ProviderModelLimits(
                provider="google", model="gemini-2.5-flash",
                requests_per_minute=15, input_tokens_per_minute=250000,
                is_free_tier=True, confidence=0.9,
            ),
            # Groq
            ProviderModelLimits(
                provider="groq", model="llama-3.1-70b",
                requests_per_minute=30, tokens_per_minute=6000, confidence=0.7,
            ),
            ProviderModelLimits(
                provider="groq", model="mixtral-8x7b",
                requests_per_minute=30, tokens_per_minute=5000, confidence=0.7,
            ),
            # Together AI
            ProviderModelLimits(
                provider="together_ai", model="*",
                requests_per_minute=600, requests_per_second=10, confidence=0.6,
            ),
            # Mistral
            ProviderModelLimits(
                provider="mistral", model="*",
                requests_per_second=5, confidence=0.6,
            ),
        ]

        for limit in known_limits:
            key = self._get_key(limit.provider, limit.model)
            if key not in self.limits:
                self.limits[key] = limit

    def _get_key(self, provider: str, model: str) -> str:
        """Generiere einen eindeutigen Key für Provider/Model"""
        provider = self._normalize_provider(provider)
        model = self._normalize_model(model)
        return f"{provider}::{model}"

    def _normalize_provider(self, provider: str) -> str:
        """Normalisiere Provider-Namen"""
        provider = provider.lower().strip()
        aliases = {
            "vertex_ai": ["vertexai", "vertex-ai", "google_vertex", "gemini"],
            "openai": ["azure", "azure_openai", "openai_azure"],
            "anthropic": ["claude"],
            "together_ai": ["together", "togetherai"],
        }
        for canonical, variants in aliases.items():
            if provider in variants or provider == canonical:
                return canonical
        return provider

    def _normalize_model(self, model: str) -> str:
        """Normalisiere Model-Namen"""
        model = model.lower().strip()
        patterns_to_strip = [r"-\d{8}$", r"-preview$", r"-latest$"]
        for pattern in patterns_to_strip:
            model = re.sub(pattern, "", model)
        return model

    def _extract_provider_from_model_string(self, model_string: str) -> Tuple[str, str]:
        """Extrahiere Provider und Model aus litellm Model-String"""
        if "/" in model_string:
            parts = model_string.split("/", 1)
            return parts[0], parts[1]

        model_lower = model_string.lower()
        if model_lower.startswith("gpt-") or model_lower.startswith("o1"):
            return "openai", model_string
        elif model_lower.startswith("claude"):
            return "anthropic", model_string
        elif model_lower.startswith("gemini"):
            return "vertex_ai", model_string
        elif "llama" in model_lower or "mixtral" in model_lower:
            return "groq", model_string
        elif model_lower.startswith("mistral"):
            return "mistral", model_string

        return "unknown", model_string

    def _get_limits_for_model(self, provider: str, model: str) -> ProviderModelLimits:
        """Hole die Limits für ein Provider/Model Paar"""
        key = self._get_key(provider, model)

        if key in self.limits:
            return self.limits[key]

        wildcard_key = self._get_key(provider, "*")
        if wildcard_key in self.limits:
            return self.limits[wildcard_key]

        new_limits = ProviderModelLimits(
            provider=provider,
            model=model,
            requests_per_minute=self.default_rpm,
            requests_per_second=self.default_rps,
            confidence=0.3,
        )
        self.limits[key] = new_limits
        return new_limits

    # ===== v2 PUBLIC API =====

    def add_api_key(
        self,
        provider: str,
        key: str,
        priority: int = 0,
        custom_rpm: Optional[int] = None,
        custom_tpm: Optional[int] = None,
    ) -> str:
        """
        Füge einen API-Key hinzu.

        Args:
            provider: z.B. "vertex_ai", "openai", "anthropic"
            key: Der API-Key
            priority: Niedrigere Zahl = höhere Priorität
            custom_rpm: Optionales custom RPM-Limit
            custom_tpm: Optionales custom TPM-Limit

        Returns:
            Key-Hash für Referenz
        """
        return self.key_manager.add_key(
            provider=provider,
            key=key,
            priority=priority,
            custom_rpm=custom_rpm,
            custom_tpm=custom_tpm,
        )

    def set_key_rotation_mode(self, mode: str) -> None:
        """
        Setze den Key-Rotation-Modus.

        Args:
            mode: "drain" (ein Key bis Limit) oder "balance" (Round-Robin)
        """
        self.key_manager.mode = mode

    def add_fallback_model(
        self,
        primary_model: str,
        fallback_model: str,
    ) -> None:
        """
        Füge ein Fallback-Model hinzu.

        Args:
            primary_model: z.B. "vertex_ai/gemini-2.5-pro"
            fallback_model: z.B. "vertex_ai/gemini-2.5-flash"
        """
        self.fallback_manager.add_fallback_model(primary_model, fallback_model)

    def add_fallback_chain(
        self,
        primary_model: str,
        fallback_models: List[str],
        fallback_duration: float = 60.0,
    ) -> None:
        """
        Füge eine komplette Fallback-Chain hinzu.

        Args:
            primary_model: Das primäre Model
            fallback_models: Liste von Fallbacks in Prioritätsreihenfolge
            fallback_duration: Wie lange bleibt Fallback aktiv (Sekunden)
        """
        self.fallback_manager.add_fallback_chain(
            primary_model=primary_model,
            fallback_models=fallback_models,
            fallback_duration=fallback_duration,
        )

    async def acquire(
        self,
        model: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
    ) -> Tuple[str, Optional[str]]:
        """
        Warte bis ein Request erlaubt ist.

        Args:
            model: Model-String (kann Provider enthalten wie "vertex_ai/gemini-1.5-pro")
            estimated_input_tokens: Geschätzte Input-Tokens
            estimated_output_tokens: Geschätzte Output-Tokens

        Returns:
            (active_model, api_key) - Das tatsächlich zu verwendende Model und ggf. API-Key
        """
        original_model = model

        # Check Model Fallback
        if self.enable_model_fallback:
            model, is_fallback = await self.fallback_manager.get_active_model(model)
            if is_fallback:
                logger.debug(f"Using fallback model: {model} (original: {original_model})")

        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)
        state = self.states[key]

        # Get API Key if key rotation enabled
        api_key = None
        if self.enable_key_rotation:
            key_info = await self.key_manager.get_next_key(provider)
            if key_info:
                api_key = key_info.key
                # Apply custom limits from key if set
                if key_info.custom_rpm:
                    limits.requests_per_minute = key_info.custom_rpm
                if key_info.custom_tpm:
                    limits.tokens_per_minute = key_info.custom_tpm

        async with state.lock:
            now = time.time()

            # Check Backoff
            if state.backoff_until > now:
                wait_time = state.backoff_until - now
                logger.info(f"[RateLimiter] In backoff for {key}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                now = time.time()

            self._cleanup_windows(state, now)

            effective_rpm = int(limits.requests_per_minute * self.safety_margin)
            effective_rps = (
                int(limits.requests_per_second * self.safety_margin)
                if limits.requests_per_second
                else None
            )

            while True:
                self._cleanup_windows(state, now)

                rpm_ok = len(state.minute_window) < effective_rpm
                rps_ok = effective_rps is None or len(state.second_window) < effective_rps

                tpm_ok = True
                if self.enable_token_tracking and limits.input_tokens_per_minute:
                    current_tokens = sum(t[1] for t in state.tokens_minute_window)
                    effective_tpm = int(limits.input_tokens_per_minute * self.safety_margin)
                    tpm_ok = (current_tokens + estimated_input_tokens) < effective_tpm

                if rpm_ok and rps_ok and tpm_ok:
                    break

                wait_time = self._calculate_wait_time(state, limits, now)
                logger.debug(f"[RateLimiter] {key} rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                now = time.time()

            # Register Request
            state.minute_window.append(now)
            if effective_rps:
                state.second_window.append(now)

            if self.enable_token_tracking and estimated_input_tokens > 0:
                state.tokens_minute_window.append((now, estimated_input_tokens))

        return model, api_key

    def _cleanup_windows(self, state: RateLimitState, now: float):
        """Entferne abgelaufene Einträge aus den Sliding Windows"""
        state.minute_window = [t for t in state.minute_window if now - t < 60]
        state.second_window = [t for t in state.second_window if now - t < 1]
        state.day_window = [t for t in state.day_window if now - t < 86400]
        state.tokens_minute_window = [(t, c) for t, c in state.tokens_minute_window if now - t < 60]
        state.tokens_day_window = [(t, c) for t, c in state.tokens_day_window if now - t < 86400]

    def _calculate_wait_time(
        self, state: RateLimitState, limits: ProviderModelLimits, now: float
    ) -> float:
        """Berechne die optimale Wartezeit"""
        wait_times = []

        if len(state.minute_window) >= limits.requests_per_minute:
            oldest = state.minute_window[0]
            wait_times.append(60.0 - (now - oldest) + 0.1)

        if limits.requests_per_second and len(state.second_window) >= limits.requests_per_second:
            oldest = state.second_window[0]
            wait_times.append(1.0 - (now - oldest) + 0.01)

        if limits.input_tokens_per_minute and state.tokens_minute_window:
            current_tokens = sum(t[1] for t in state.tokens_minute_window)
            if current_tokens >= limits.input_tokens_per_minute:
                oldest = state.tokens_minute_window[0][0]
                wait_times.append(60.0 - (now - oldest) + 0.1)

        if wait_times:
            return min(max(wait_times), 60.0)
        return 0.1

    async def handle_rate_limit_error(
        self,
        model: str,
        error: Exception,
        response_body: Optional[str] = None,
        api_key_hash: Optional[str] = None,
    ) -> Tuple[float, Optional[str]]:
        """
        Verarbeite einen Rate-Limit-Fehler.

        Returns:
            (wait_time, fallback_model) - Wartezeit und ggf. Fallback-Model
        """
        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)
        state = self.states[key]

        # Extract info from error
        error_str = str(error)
        if response_body:
            error_str += " " + response_body

        retry_delay = self._extract_retry_delay(error_str)
        quota_info = self._extract_quota_info(error_str)

        if quota_info:
            self._update_limits_from_quota(limits, quota_info)

        # Calculate backoff
        state.consecutive_failures += 1
        backoff_time = self._calculate_backoff(retry_delay, state.consecutive_failures)
        state.backoff_until = time.time() + backoff_time

        limits.rate_limit_hits += 1
        if retry_delay:
            limits.observed_retry_delays.append(retry_delay)
            limits.observed_retry_delays = limits.observed_retry_delays[-10:]

        # Mark API key as exhausted if applicable
        if self.enable_key_rotation and api_key_hash:
            self.key_manager.mark_key_exhausted(
                provider=provider,
                key_hash=api_key_hash,
                duration=backoff_time,
            )

        # Try model fallback
        fallback_model = None
        if self.enable_model_fallback:
            fallback_model = await self.fallback_manager.trigger_fallback(
                model=model,
                reason=FallbackReason.RATE_LIMIT,
                duration=backoff_time * 2,  # Fallback bleibt länger aktiv
            )

        if self.persist_learned_limits:
            self._save_limits()

        logger.warning(
            f"[RateLimiter] Rate limit hit for {key}. "
            f"Retry delay: {retry_delay}s, Backoff: {backoff_time:.1f}s, "
            f"Fallback: {fallback_model}"
        )

        return backoff_time, fallback_model

    def _extract_retry_delay(self, error_str: str) -> Optional[float]:
        """Extrahiere retry delay aus Fehlertext"""
        patterns = [
            r"retry[_ ]?(?:in|after|delay)[:\s]*(\d+\.?\d*)\s*s",
            r'retryDelay["\s:]+(\d+)',
            r"Please retry in (\d+\.?\d*)",
            r"try again in (\d+)",
            r'"retry_after":\s*(\d+\.?\d*)',
            r"Retry-After:\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_quota_info(self, error_str: str) -> Optional[Dict[str, Any]]:
        """Extrahiere Quota-Informationen aus Fehlertext"""
        quota_info = {}

        try:
            json_match = re.search(r'\{[^{}]*"error"[^{}]*\{.*?\}\s*\}', error_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "details" in data.get("error", {}):
                    for detail in data["error"]["details"]:
                        if detail.get("@type", "").endswith("QuotaFailure"):
                            for violation in detail.get("violations", []):
                                metric = violation.get("quotaMetric", "")
                                value = violation.get("quotaValue")
                                if "input_token" in metric.lower():
                                    quota_info["input_tokens_per_minute"] = int(value)
                                elif "output_token" in metric.lower():
                                    quota_info["output_tokens_per_minute"] = int(value)
                                elif "request" in metric.lower():
                                    quota_info["requests_per_minute"] = int(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        patterns = {
            "requests_per_minute": [
                r"limit:\s*(\d+).*?requests?\s*per\s*minute",
                r"(\d+)\s*requests?\s*per\s*minute",
                r"rpm[:\s]+(\d+)",
            ],
            "tokens_per_minute": [
                r"limit:\s*(\d+).*?tokens?\s*per\s*minute",
                r"(\d+)\s*tokens?\s*per\s*minute",
                r"tpm[:\s]+(\d+)",
            ],
            "input_tokens_per_minute": [
                r"input_token.*?limit:\s*(\d+)",
                r'quotaValue["\s:]+(\d+).*?input',
            ],
        }

        for field, field_patterns in patterns.items():
            if field not in quota_info:
                for pattern in field_patterns:
                    match = re.search(pattern, error_str, re.IGNORECASE)
                    if match:
                        quota_info[field] = int(match.group(1))
                        break

        return quota_info if quota_info else None

    def _update_limits_from_quota(
        self, limits: ProviderModelLimits, quota_info: Dict[str, Any]
    ):
        """Update Limits basierend auf extrahierten Quota-Informationen"""
        updated = False
        for field, value in quota_info.items():
            if hasattr(limits, field):
                current = getattr(limits, field)
                if current is None or value < current:
                    setattr(limits, field, value)
                    updated = True
                    logger.info(f"[RateLimiter] Updated {field} to {value}")

        if updated:
            limits.last_updated = time.time()
            limits.confidence = min(limits.confidence + 0.1, 1.0)

    def _calculate_backoff(
        self, retry_delay: Optional[float], consecutive_failures: int
    ) -> float:
        """Berechne Backoff-Zeit mit Exponential Backoff und Jitter"""
        if retry_delay:
            base = retry_delay
        else:
            base = min(2 ** (consecutive_failures - 1), 60)

        jitter = base * 0.2 * (random.random() * 2 - 1)
        return max(base + jitter, 0.5)

    def report_success(
        self,
        model: str,
        tokens_used: Optional[int] = None,
        api_key_hash: Optional[str] = None,
    ):
        """Melde einen erfolgreichen Request"""
        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)
        state = self.states[key]

        state.consecutive_failures = 0
        limits.successful_requests += 1

        if tokens_used and self.enable_token_tracking:
            now = time.time()
            if state.tokens_minute_window:
                state.tokens_minute_window[-1] = (now, tokens_used)

        if self.enable_key_rotation and api_key_hash:
            self.key_manager.mark_key_used(provider, api_key_hash, tokens_used or 0)

    def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Hole Statistiken"""
        stats = {}

        if model:
            provider, model_name = self._extract_provider_from_model_string(model)
            key = self._get_key(provider, model_name)
            stats["limits"] = self._get_stats_for_key(key)
            stats["fallback"] = self.fallback_manager.get_state(model)
            stats["keys"] = self.key_manager.get_stats(provider)
        else:
            stats["limits"] = {key: self._get_stats_for_key(key) for key in self.limits.keys()}
            stats["keys"] = self.key_manager.get_stats()

        return stats

    def _get_stats_for_key(self, key: str) -> Dict[str, Any]:
        """Hole Statistiken für einen spezifischen Key"""
        if key not in self.limits:
            return {}

        limits = self.limits[key]
        state = self.states[key]
        now = time.time()

        self._cleanup_windows(state, now)

        return {
            "provider": limits.provider,
            "model": limits.model,
            "limits": {
                "rpm": limits.requests_per_minute,
                "rps": limits.requests_per_second,
                "tpm": limits.tokens_per_minute,
                "input_tpm": limits.input_tokens_per_minute,
            },
            "current_usage": {
                "requests_last_minute": len(state.minute_window),
                "requests_last_second": len(state.second_window),
                "tokens_last_minute": sum(t[1] for t in state.tokens_minute_window),
            },
            "metadata": {
                "is_free_tier": limits.is_free_tier,
                "confidence": limits.confidence,
                "rate_limit_hits": limits.rate_limit_hits,
                "successful_requests": limits.successful_requests,
                "avg_retry_delay": (
                    sum(limits.observed_retry_delays) / len(limits.observed_retry_delays)
                    if limits.observed_retry_delays
                    else None
                ),
            },
            "backoff": {
                "active": state.backoff_until > now,
                "remaining_seconds": max(0, state.backoff_until - now),
                "consecutive_failures": state.consecutive_failures,
            },
        }

    def set_limits(
        self,
        model: str,
        rpm: Optional[int] = None,
        rps: Optional[int] = None,
        tpm: Optional[int] = None,
        input_tpm: Optional[int] = None,
        is_free_tier: bool = False,
    ):
        """Setze Limits manuell für ein Model"""
        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)

        if rpm is not None:
            limits.requests_per_minute = rpm
        if rps is not None:
            limits.requests_per_second = rps
        if tpm is not None:
            limits.tokens_per_minute = tpm
        if input_tpm is not None:
            limits.input_tokens_per_minute = input_tpm

        limits.is_free_tier = is_free_tier
        limits.confidence = 1.0
        limits.last_updated = time.time()

        if self.persist_learned_limits:
            self._save_limits()

    def _load_limits(self):
        """Lade persistierte Limits aus Datei"""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)

            for key, limit_data in data.get("limits", data).items():
                self.limits[key] = ProviderModelLimits(**limit_data)

            logger.info(f"[RateLimiter] Loaded {len(self.limits)} limit configurations")
        except Exception as e:
            logger.warning(f"[RateLimiter] Failed to load limits: {e}")

    def _save_limits(self):
        """Speichere gelernte Limits in Datei"""
        try:
            data = {"limits": {}}
            for key, limits in self.limits.items():
                data["limits"][key] = {
                    "provider": limits.provider,
                    "model": limits.model,
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_second": limits.requests_per_second,
                    "requests_per_day": limits.requests_per_day,
                    "tokens_per_minute": limits.tokens_per_minute,
                    "tokens_per_day": limits.tokens_per_day,
                    "input_tokens_per_minute": limits.input_tokens_per_minute,
                    "output_tokens_per_minute": limits.output_tokens_per_minute,
                    "is_free_tier": limits.is_free_tier,
                    "last_updated": limits.last_updated,
                    "confidence": limits.confidence,
                    "observed_retry_delays": limits.observed_retry_delays,
                    "rate_limit_hits": limits.rate_limit_hits,
                    "successful_requests": limits.successful_requests,
                }

            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"[RateLimiter] Failed to save limits: {e}")


# ===== LITELLM INTEGRATION =====

class LiteLLMRateLimitHandler:
    """
    Intelligenter Handler für LiteLLM mit automatischem Rate Limiting,
    Model Fallback und Multi-API-Key Support.

    Features (alle togglebar):
    - Rate Limiting mit automatischer Anpassung
    - Model Fallback bei Limits (z.B. pro -> flash)
    - Multi-API-Key mit Drain/Balance Modi
    - Kombinierbare Strategien
    """

    def __init__(
        self,
        rate_limiter: Optional[IntelligentRateLimiter] = None,
        max_retries: int = 3,
        # Feature Toggles
        enable_rate_limiting: bool = True,
        enable_model_fallback: bool = True,
        enable_key_rotation: bool = True,
        key_rotation_mode: str = "balance",  # "drain" or "balance"
        # Fallback Behavior
        fallback_on_any_error: bool = False,  # Auch bei non-rate-limit Errors
        wait_if_all_exhausted: bool = True,    # Warten wenn alles erschöpft
    ):
        self.rate_limiter = rate_limiter or IntelligentRateLimiter(
            enable_model_fallback=enable_model_fallback,
            enable_key_rotation=enable_key_rotation,
            key_rotation_mode=key_rotation_mode,
        )
        self.max_retries = max_retries

        # Feature Toggles
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_model_fallback = enable_model_fallback
        self.enable_key_rotation = enable_key_rotation
        self.fallback_on_any_error = fallback_on_any_error
        self.wait_if_all_exhausted = wait_if_all_exhausted

        # Request Tracking
        self._active_requests: Dict[str, int] = defaultdict(int)

    # ===== CONVENIENCE METHODS =====

    def add_api_key(
        self,
        provider: str,
        key: str,
        **kwargs,
    ) -> str:
        """
        Füge einen API-Key hinzu.

        Args:
            provider: "vertex_ai", "openai", "anthropic", etc.
            key: Der API-Key

        Example:
            handler.add_api_key("vertex_ai", "AIza...")
        """
        return self.rate_limiter.add_api_key(provider, key, **kwargs)

    def set_key_rotation_mode(self, mode: str) -> None:
        """
        Setze den Key-Rotation-Modus.

        Args:
            mode: "drain" (ein Key bis Limit) oder "balance" (Round-Robin)

        Example:
            handler.set_key_rotation_mode("drain")
        """
        self.rate_limiter.set_key_rotation_mode(mode)

    def add_fallback(
        self,
        primary_model: str,
        fallback_model: str,
    ) -> None:
        """
        Füge ein Fallback-Model hinzu.

        Example:
            handler.add_fallback("vertex_ai/gemini-2.5-pro", "vertex_ai/gemini-2.5-flash")
        """
        self.rate_limiter.add_fallback_model(primary_model, fallback_model)

    def add_fallback_chain(
        self,
        primary_model: str,
        fallback_models: List[str],
        fallback_duration: float = 60.0,
    ) -> None:
        """
        Füge eine Fallback-Chain hinzu.

        Example:
            handler.add_fallback_chain(
                "vertex_ai/gemini-2.5-pro",
                ["vertex_ai/gemini-2.5-flash", "vertex_ai/gemini-2.0-flash"],
                fallback_duration=120.0
            )
        """
        self.rate_limiter.add_fallback_chain(primary_model, fallback_models, fallback_duration)

    def set_limits(self, model: str, **kwargs) -> None:
        """Setze Model-Limits manuell"""
        self.rate_limiter.set_limits(model, **kwargs)

    def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Hole Statistiken"""
        return self.rate_limiter.get_stats(model)

    async def completion_with_rate_limiting(
        self,
        litellm_module,
        wait_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Wrapper für litellm.acompletion mit allen intelligenten Features.

        Features (alle automatisch):
        - Rate Limiting
        - Model Fallback bei Limits
        - API Key Rotation
        - Automatische Retries

        Example:
            response = await handler.completion_with_rate_limiting(
                litellm,
                model="vertex_ai/gemini-2.5-pro",
                messages=[{"role": "user", "content": "Hello!"}],
            )
        """
        original_model = kwargs.get("model", "")
        estimated_tokens = self._estimate_input_tokens(kwargs.get("messages", []))

        current_api_key_hash = None
        current_model = original_model

        for attempt in range(self.max_retries + 1):
            try:
                # Acquire rate limit slot and get active model/key
                if self.enable_rate_limiting:
                    current_model, api_key = await self.rate_limiter.acquire(
                        model=current_model,
                        estimated_input_tokens=estimated_tokens,
                    )

                    # Track key hash for error handling
                    if api_key:
                        current_api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12]
                        # Inject API key based on provider
                        kwargs = self._inject_api_key(kwargs, current_model, api_key)

                # Update model in kwargs if changed by fallback
                kwargs["model"] = current_model

                # Execute request
                response = await litellm_module.acompletion(**kwargs)

                # Report success
                if self.enable_rate_limiting:
                    self.rate_limiter.report_success(
                        model=current_model,
                        api_key_hash=current_api_key_hash,
                    )

                return response

            except Exception as e:
                error_str = str(e).lower()

                is_rate_limit = any(
                    x in error_str
                    for x in [
                        "rate_limit", "ratelimit", "429", "quota",
                        "resource_exhausted", "too many requests",
                    ]
                )

                should_fallback = is_rate_limit or (self.fallback_on_any_error and attempt < self.max_retries)

                if should_fallback and attempt < self.max_retries:
                    # Handle rate limit error
                    wait_time, fallback_model = await self.rate_limiter.handle_rate_limit_error(
                        model=current_model,
                        error=e,
                        api_key_hash=current_api_key_hash,
                    )

                    # Try fallback model if available
                    if fallback_model and self.enable_model_fallback:
                        logger.info(
                            f"[Handler] Switching to fallback: {current_model} -> {fallback_model}"
                        )
                        current_model = fallback_model
                        continue

                    # No fallback available - wait or fail
                    if self.wait_if_all_exhausted:
                        logger.warning(
                            f"[Handler] Rate limit (attempt {attempt + 1}/{self.max_retries}), "
                            f"waiting {wait_time:.1f}s"
                        )
                        await wait_callback(wait_time) if wait_callback else None
                        await asyncio.sleep(wait_time)
                        current_model = original_model  # Try original again
                    else:
                        raise
                else:
                    raise

    def _inject_api_key(
        self,
        kwargs: Dict[str, Any],
        model: str,
        api_key: str,
    ) -> Dict[str, Any]:
        """
        Injiziere API-Key in kwargs basierend auf Provider.

        LiteLLM unterstützt verschiedene Methoden je nach Provider.
        """
        kwargs = kwargs.copy()
        provider, _ = self.rate_limiter._extract_provider_from_model_string(model)

        if provider in ("openai", "azure"):
            kwargs["api_key"] = api_key
        elif provider == "anthropic":
            kwargs["api_key"] = api_key
        elif provider in ("vertex_ai", "google"):
            # Vertex AI verwendet normalerweise Credentials, nicht API Keys
            # Für API Key-basierte Nutzung:
            kwargs.setdefault("vertex_credentials", api_key)
        elif provider == "together_ai":
            kwargs["api_key"] = api_key
        elif provider == "groq":
            kwargs["api_key"] = api_key
        elif provider == "mistral":
            kwargs["api_key"] = api_key
        else:
            # Generic fallback
            kwargs["api_key"] = api_key

        return kwargs

    def _estimate_input_tokens(self, messages: list) -> int:
        """Grobe Schätzung der Input-Tokens"""
        if not messages:
            return 0
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    @asynccontextmanager
    async def rate_limited(self, model: str, estimated_tokens: int = 0):
        """
        Context Manager für manuelles Rate Limiting.

        Example:
            async with handler.rate_limited("vertex_ai/gemini-2.5-pro", 1000):
                # Your API call here
                pass
        """
        active_model, api_key = await self.rate_limiter.acquire(
            model=model,
            estimated_input_tokens=estimated_tokens,
        )
        try:
            yield active_model, api_key
            self.rate_limiter.report_success(model=active_model)
        except Exception as e:
            await self.rate_limiter.handle_rate_limit_error(model=active_model, error=e)
            raise


# ===== CONFIG-BASED SETUP =====

def create_handler_from_config(config: Dict[str, Any]) -> LiteLLMRateLimitHandler:
    """
    Erstelle Handler aus Konfiguration.

    Config Format:
    {
        "features": {
            "rate_limiting": true,
            "model_fallback": true,
            "key_rotation": true,
            "key_rotation_mode": "drain"  // or "balance"
        },
        "api_keys": {
            "vertex_ai": ["AIza...", "AIzb..."],
            "openai": ["sk-..."]
        },
        "fallback_chains": {
            "vertex_ai/gemini-2.5-pro": ["vertex_ai/gemini-2.5-flash"],
            "openai/gpt-4o": ["openai/gpt-4o-mini"]
        },
        "limits": {
            "vertex_ai/gemini-2.5-pro": {"rpm": 2, "input_tpm": 32000}
        }
    }
    """
    features = config.get("features", {})

    handler = LiteLLMRateLimitHandler(
        enable_rate_limiting=features.get("rate_limiting", True),
        enable_model_fallback=features.get("model_fallback", True),
        enable_key_rotation=features.get("key_rotation", True),
        key_rotation_mode=features.get("key_rotation_mode", "balance"),
        wait_if_all_exhausted=features.get("wait_if_all_exhausted", True),
    )

    # Add API Keys
    for provider, keys in config.get("api_keys", {}).items():
        for key_config in keys:
            if isinstance(key_config, str):
                handler.add_api_key(provider, key_config)
            else:
                handler.add_api_key(
                    provider=provider,
                    key=key_config["key"],
                    priority=key_config.get("priority", 0),
                    custom_rpm=key_config.get("rpm"),
                    custom_tpm=key_config.get("tpm"),
                )

    # Add Fallback Chains
    for primary, fallbacks in config.get("fallback_chains", {}).items():
        handler.add_fallback_chain(primary, fallbacks)

    # Set Limits
    for model, limits in config.get("limits", {}).items():
        handler.set_limits(model, **limits)

    return handler


def load_handler_from_file(path: Union[str, Path]) -> LiteLLMRateLimitHandler:
    """Lade Handler-Konfiguration aus JSON/YAML Datei"""
    path = Path(path)

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    return create_handler_from_config(config)


# ===== EXAMPLE USAGE =====

async def example_usage():
    """Beispiel für die Verwendung des intelligenten Rate Limiters v2"""

    # Option 1: Minimal Setup (alles automatisch)
    handler = LiteLLMRateLimitHandler()

    # Option 2: Mit Custom Config
    handler = LiteLLMRateLimitHandler(
        enable_model_fallback=True,
        enable_key_rotation=True,
        key_rotation_mode="drain",  # Global: "drain" oder "balance"
        max_retries=3,
    )

    # API Keys hinzufügen (Mode wird global gesetzt)
    handler.add_api_key("vertex_ai", "AIza_KEY_1")
    handler.add_api_key("vertex_ai", "AIza_KEY_2")
    handler.add_api_key("openai", "sk-KEY_1")
    handler.add_api_key("openai", "sk-KEY_2")

    # Mode kann auch später geändert werden
    handler.set_key_rotation_mode("balance")

    # Custom Fallback Chain
    handler.add_fallback_chain(
        primary_model="vertex_ai/gemini-2.5-pro",
        fallback_models=[
            "vertex_ai/gemini-2.5-flash",
            "vertex_ai/gemini-2.0-flash",
        ],
        fallback_duration=120.0,
    )

    # Custom Limits
    handler.set_limits(
        model="vertex_ai/gemini-2.5-pro",
        rpm=2,
        input_tpm=32000,
        is_free_tier=True,
    )

    # Verwendung
    import litellm

    try:
        response = await handler.completion_with_rate_limiting(
            litellm,
            model="vertex_ai/gemini-2.5-pro",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response)
    except Exception as e:
        print(f"Request failed: {e}")

    # Stats
    print(json.dumps(handler.get_stats(), indent=2))


async def example_from_config():
    """Beispiel mit Config-Datei"""

    config = {
        "features": {
            "rate_limiting": True,
            "model_fallback": True,
            "key_rotation": True,
            "key_rotation_mode": "drain",  # Global mode
        },
        "api_keys": {
            "vertex_ai": ["AIza_KEY_1", "AIza_KEY_2"],
            "openai": ["sk-KEY_1"],
        },
        "fallback_chains": {
            "vertex_ai/gemini-2.5-pro": [
                "vertex_ai/gemini-2.5-flash",
                "vertex_ai/gemini-2.0-flash",
            ],
        },
        "limits": {
            "vertex_ai/gemini-2.5-pro": {"rpm": 2, "input_tpm": 32000},
        },
    }

    handler = create_handler_from_config(config)

    import litellm

    response = await handler.completion_with_rate_limiting(
        litellm,
        model="vertex_ai/gemini-2.5-pro",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    return response


if __name__ == "__main__":
    asyncio.run(example_usage())
