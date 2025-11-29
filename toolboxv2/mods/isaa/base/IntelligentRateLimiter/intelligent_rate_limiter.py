"""
Intelligenter, selbst-adaptierender LLM Rate Limiter

Features:
- Automatische Extraktion von Rate-Limit-Informationen aus Fehlerantworten
- Provider- und modellspezifische Konfiguration
- Token-basiertes Rate Limiting (nicht nur Request-basiert)
- Exponential Backoff mit Jitter
- Persistente Limit-Datenbank für bekannte Provider/Modelle
- Dynamische Anpassung basierend auf tatsächlichem Verhalten
"""

import asyncio
import time
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
import random
from collections import defaultdict

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
    confidence: float = 0.5  # Wie sicher sind wir über diese Limits? 0-1

    # Dynamisch gelernte Werte
    observed_retry_delays: list = field(default_factory=list)
    rate_limit_hits: int = 0
    successful_requests: int = 0


@dataclass
class RateLimitState:
    """Aktueller Zustand für ein Provider/Model Paar"""

    # Sliding Windows für Requests
    minute_window: list = field(default_factory=list)
    second_window: list = field(default_factory=list)
    day_window: list = field(default_factory=list)

    # Token Tracking
    tokens_minute_window: list = field(default_factory=list)  # (timestamp, token_count)
    tokens_day_window: list = field(default_factory=list)

    # Backoff State
    backoff_until: float = 0.0
    consecutive_failures: int = 0

    # Lock für Thread-Safety
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class IntelligentRateLimiter:
    """
    Intelligenter Rate Limiter der sich automatisch an Provider-Limits anpasst.

    Features:
    - Automatische Erkennung von Rate-Limits aus Fehlerantworten
    - Provider/Model-spezifische Konfiguration
    - Token-basiertes Tracking
    - Exponential Backoff mit Jitter
    - Persistente Speicherung gelernter Limits
    """

    # Bekannte Default-Limits für gängige Provider (Stand: 2024)
    DEFAULT_LIMITS: Dict[str, Dict[str, ProviderModelLimits]] = {}

    def __init__(
        self,
        config_path: Optional[Path] = None,
        default_rpm: int = 60,
        default_rps: int = 10,
        safety_margin: float = 0.9,  # Nutze nur 90% des Limits
        enable_token_tracking: bool = True,
        persist_learned_limits: bool = True,
    ):
        self.config_path = config_path or Path.home() / ".llm_rate_limits.json"
        self.default_rpm = default_rpm
        self.default_rps = default_rps
        self.safety_margin = safety_margin
        self.enable_token_tracking = enable_token_tracking
        self.persist_learned_limits = persist_learned_limits

        # Provider/Model -> Limits
        self.limits: Dict[str, ProviderModelLimits] = {}

        # Provider/Model -> State
        self.states: Dict[str, RateLimitState] = defaultdict(RateLimitState)

        # Global Lock für Limit-Updates
        self._global_lock = asyncio.Lock()

        # Lade persistierte Limits
        self._load_limits()

        # Initialisiere bekannte Provider-Defaults
        self._init_known_limits()

    def _init_known_limits(self):
        """Initialisiere bekannte Default-Limits für gängige Provider"""

        known_limits = [
            # OpenAI
            ProviderModelLimits(
                provider="openai",
                model="gpt-4",
                requests_per_minute=500,
                tokens_per_minute=40000,
                confidence=0.8,
            ),
            ProviderModelLimits(
                provider="openai",
                model="gpt-4-turbo",
                requests_per_minute=500,
                tokens_per_minute=150000,
                confidence=0.8,
            ),
            ProviderModelLimits(
                provider="openai",
                model="gpt-3.5-turbo",
                requests_per_minute=3500,
                tokens_per_minute=90000,
                confidence=0.8,
            ),
            # Anthropic
            ProviderModelLimits(
                provider="anthropic",
                model="claude-3-opus",
                requests_per_minute=50,
                tokens_per_minute=40000,
                confidence=0.8,
            ),
            ProviderModelLimits(
                provider="anthropic",
                model="claude-3-sonnet",
                requests_per_minute=50,
                tokens_per_minute=80000,
                confidence=0.8,
            ),
            ProviderModelLimits(
                provider="anthropic",
                model="claude-3-haiku",
                requests_per_minute=50,
                tokens_per_minute=100000,
                confidence=0.8,
            ),
            # Google/Vertex AI - Free Tier
            ProviderModelLimits(
                provider="vertex_ai",
                model="gemini-2.5-flash",
                requests_per_minute=15,
                input_tokens_per_minute=250000,
                is_free_tier=True,
                confidence=0.9,
            ),
            ProviderModelLimits(
                provider="vertex_ai",
                model="gemini-1.5-pro",
                requests_per_minute=2,
                input_tokens_per_minute=32000,
                is_free_tier=True,
                confidence=0.9,
            ),
            ProviderModelLimits(
                provider="google",
                model="gemini-2.5-flash",
                requests_per_minute=15,
                input_tokens_per_minute=250000,
                is_free_tier=True,
                confidence=0.9,
            ),
            # Groq
            ProviderModelLimits(
                provider="groq",
                model="llama-3.1-70b",
                requests_per_minute=30,
                tokens_per_minute=6000,
                confidence=0.7,
            ),
            ProviderModelLimits(
                provider="groq",
                model="mixtral-8x7b",
                requests_per_minute=30,
                tokens_per_minute=5000,
                confidence=0.7,
            ),
            # Together AI
            ProviderModelLimits(
                provider="together_ai",
                model="*",
                requests_per_minute=600,
                requests_per_second=10,
                confidence=0.6,
            ),
            # Mistral
            ProviderModelLimits(
                provider="mistral", model="*", requests_per_second=5, confidence=0.6
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

        # Bekannte Aliase
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
        """Normalisiere Model-Namen (entferne Versions-Suffixe etc.)"""
        model = model.lower().strip()

        # Entferne common Suffixe
        patterns_to_strip = [
            r"-\d{8}$",  # Datums-Suffixe wie -20240101
            r"-preview$",
            r"-latest$",
        ]

        for pattern in patterns_to_strip:
            model = re.sub(pattern, "", model)

        return model

    def _extract_provider_from_model_string(self, model_string: str) -> Tuple[str, str]:
        """
        Extrahiere Provider und Model aus litellm Model-String.

        Beispiele:
        - "gpt-4" -> ("openai", "gpt-4")
        - "anthropic/claude-3-opus" -> ("anthropic", "claude-3-opus")
        - "vertex_ai/gemini-1.5-pro" -> ("vertex_ai", "gemini-1.5-pro")
        """
        if "/" in model_string:
            parts = model_string.split("/", 1)
            return parts[0], parts[1]

        # Inferiere Provider aus Model-Namen
        model_lower = model_string.lower()

        if model_lower.startswith("gpt-") or model_lower.startswith("o1"):
            return "openai", model_string
        elif model_lower.startswith("claude"):
            return "anthropic", model_string
        elif model_lower.startswith("gemini"):
            return "vertex_ai", model_string
        elif "llama" in model_lower or "mixtral" in model_lower:
            return "groq", model_string  # Default, könnte auch together sein
        elif model_lower.startswith("mistral"):
            return "mistral", model_string

        return "unknown", model_string

    def _get_limits_for_model(self, provider: str, model: str) -> ProviderModelLimits:
        """Hole die Limits für ein Provider/Model Paar"""
        key = self._get_key(provider, model)

        if key in self.limits:
            return self.limits[key]

        # Versuche Wildcard-Match
        wildcard_key = self._get_key(provider, "*")
        if wildcard_key in self.limits:
            return self.limits[wildcard_key]

        # Erstelle neue Default-Limits
        new_limits = ProviderModelLimits(
            provider=provider,
            model=model,
            requests_per_minute=self.default_rpm,
            requests_per_second=self.default_rps,
            confidence=0.3,
        )
        self.limits[key] = new_limits
        return new_limits

    async def acquire(
        self,
        model: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
    ) -> None:
        """
        Warte bis ein Request erlaubt ist.

        Args:
            model: Model-String (kann Provider enthalten wie "vertex_ai/gemini-1.5-pro")
            estimated_input_tokens: Geschätzte Input-Tokens
            estimated_output_tokens: Geschätzte Output-Tokens
        """
        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)
        state = self.states[key]

        async with state.lock:
            now = time.time()

            # Check Backoff
            if state.backoff_until > now:
                wait_time = state.backoff_until - now
                logger.info(
                    f"[RateLimiter] In backoff for {key}, waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                now = time.time()

            # Cleanup alte Einträge
            self._cleanup_windows(state, now)

            # Berechne effektive Limits mit Safety Margin
            effective_rpm = int(limits.requests_per_minute * self.safety_margin)
            effective_rps = (
                int(limits.requests_per_second * self.safety_margin)
                if limits.requests_per_second
                else None
            )

            # Warte bis Limits erfüllt sind
            while True:
                self._cleanup_windows(state, now)

                # Check Request-Limits
                rpm_ok = len(state.minute_window) < effective_rpm
                rps_ok = effective_rps is None or len(state.second_window) < effective_rps

                # Check Token-Limits (wenn aktiviert)
                tpm_ok = True
                if self.enable_token_tracking and limits.input_tokens_per_minute:
                    current_tokens = sum(t[1] for t in state.tokens_minute_window)
                    effective_tpm = int(
                        limits.input_tokens_per_minute * self.safety_margin
                    )
                    tpm_ok = (current_tokens + estimated_input_tokens) < effective_tpm

                if rpm_ok and rps_ok and tpm_ok:
                    break

                # Berechne Wartezeit
                wait_time = self._calculate_wait_time(state, limits, now)
                logger.debug(
                    f"[RateLimiter] {key} rate limited, waiting {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                now = time.time()

            # Registriere Request
            state.minute_window.append(now)
            if effective_rps:
                state.second_window.append(now)

            if self.enable_token_tracking and estimated_input_tokens > 0:
                state.tokens_minute_window.append((now, estimated_input_tokens))

    def _cleanup_windows(self, state: RateLimitState, now: float):
        """Entferne abgelaufene Einträge aus den Sliding Windows"""
        state.minute_window = [t for t in state.minute_window if now - t < 60]
        state.second_window = [t for t in state.second_window if now - t < 1]
        state.day_window = [t for t in state.day_window if now - t < 86400]
        state.tokens_minute_window = [
            (t, c) for t, c in state.tokens_minute_window if now - t < 60
        ]
        state.tokens_day_window = [
            (t, c) for t, c in state.tokens_day_window if now - t < 86400
        ]

    def _calculate_wait_time(
        self, state: RateLimitState, limits: ProviderModelLimits, now: float
    ) -> float:
        """Berechne die optimale Wartezeit"""
        wait_times = []

        # RPM
        if len(state.minute_window) >= limits.requests_per_minute:
            oldest = state.minute_window[0]
            wait_times.append(60.0 - (now - oldest) + 0.1)

        # RPS
        if (
            limits.requests_per_second
            and len(state.second_window) >= limits.requests_per_second
        ):
            oldest = state.second_window[0]
            wait_times.append(1.0 - (now - oldest) + 0.01)

        # TPM
        if limits.input_tokens_per_minute and state.tokens_minute_window:
            current_tokens = sum(t[1] for t in state.tokens_minute_window)
            if current_tokens >= limits.input_tokens_per_minute:
                oldest = state.tokens_minute_window[0][0]
                wait_times.append(60.0 - (now - oldest) + 0.1)

        if wait_times:
            return min(max(wait_times), 60.0)  # Max 60s warten

        return 0.1  # Minimal wait

    def handle_rate_limit_error(
        self, model: str, error: Exception, response_body: Optional[str] = None
    ) -> float:
        """
        Verarbeite einen Rate-Limit-Fehler und extrahiere Informationen.

        Returns:
            Empfohlene Wartezeit in Sekunden
        """
        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)
        state = self.states[key]

        # Extrahiere Informationen aus dem Fehler
        error_str = str(error)
        if response_body:
            error_str += " " + response_body

        retry_delay = self._extract_retry_delay(error_str)
        quota_info = self._extract_quota_info(error_str)

        # Update Limits basierend auf extrahierten Informationen
        if quota_info:
            self._update_limits_from_quota(limits, quota_info)

        # Berechne Backoff
        state.consecutive_failures += 1
        state.backoff_until = time.time() + self._calculate_backoff(
            retry_delay, state.consecutive_failures
        )

        # Statistiken
        limits.rate_limit_hits += 1
        if retry_delay:
            limits.observed_retry_delays.append(retry_delay)
            # Halte nur die letzten 10
            limits.observed_retry_delays = limits.observed_retry_delays[-10:]

        # Persistiere gelernte Limits
        if self.persist_learned_limits:
            self._save_limits()

        logger.warning(
            f"[RateLimiter] Rate limit hit for {key}. "
            f"Retry delay: {retry_delay}s, Backoff until: {state.backoff_until - time.time():.1f}s"
        )

        return state.backoff_until - time.time()

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

        # Versuche JSON zu parsen
        try:
            # Suche nach JSON in der Fehlermeldung
            json_match = re.search(
                r'\{[^{}]*"error"[^{}]*\{.*?\}\s*\}', error_str, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group())

                # Google/Vertex AI Format
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

        # Regex-basierte Extraktion als Fallback
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
            # Provider hat uns gesagt wie lange wir warten sollen
            base = retry_delay
        else:
            # Exponential Backoff: 1s, 2s, 4s, 8s, ... bis max 60s
            base = min(2 ** (consecutive_failures - 1), 60)

        # Jitter hinzufügen (±20%)
        jitter = base * 0.2 * (random.random() * 2 - 1)

        return max(base + jitter, 0.5)

    def report_success(self, model: str, tokens_used: Optional[int] = None):
        """
        Melde einen erfolgreichen Request.

        Args:
            model: Model-String
            tokens_used: Tatsächlich verwendete Tokens
        """
        provider, model_name = self._extract_provider_from_model_string(model)
        key = self._get_key(provider, model_name)

        limits = self._get_limits_for_model(provider, model_name)
        state = self.states[key]

        # Reset Backoff bei Erfolg
        state.consecutive_failures = 0

        # Update Statistiken
        limits.successful_requests += 1

        # Token-Tracking aktualisieren wenn echte Werte vorliegen
        if tokens_used and self.enable_token_tracking:
            now = time.time()
            # Ersetze Schätzung durch echten Wert (letzter Eintrag)
            if state.tokens_minute_window:
                state.tokens_minute_window[-1] = (now, tokens_used)

    def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Hole Statistiken für ein oder alle Models"""
        if model:
            provider, model_name = self._extract_provider_from_model_string(model)
            key = self._get_key(provider, model_name)
            return self._get_stats_for_key(key)

        return {key: self._get_stats_for_key(key) for key in self.limits.keys()}

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
        """
        Setze Limits manuell für ein Model.

        Nützlich wenn du die Limits deines API-Plans kennst.
        """
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
        limits.confidence = 1.0  # Manuell gesetzte Limits sind sicher
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

            for key, limit_data in data.items():
                self.limits[key] = ProviderModelLimits(**limit_data)

            logger.info(f"[RateLimiter] Loaded {len(data)} limit configurations")
        except Exception as e:
            logger.warning(f"[RateLimiter] Failed to load limits: {e}")

    def _save_limits(self):
        """Speichere gelernte Limits in Datei"""
        try:
            data = {}
            for key, limits in self.limits.items():
                data[key] = {
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


# ===== INTEGRATION MIT LITELLM =====


class LiteLLMRateLimitHandler:
    """
    Handler-Klasse für die Integration mit LiteLLM.

    Fängt Rate-Limit-Fehler ab und verwaltet Retries automatisch.
    """

    def __init__(self, rate_limiter: IntelligentRateLimiter, max_retries: int = 3):
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

    async def completion_with_rate_limiting(self, litellm_module, **kwargs):
        """
        Wrapper für litellm.acompletion mit automatischem Rate Limiting.

        Usage:
            handler = LiteLLMRateLimitHandler(rate_limiter)
            response = await handler.completion_with_rate_limiting(
                litellm,
                model="vertex_ai/gemini-1.5-pro",
                messages=[...],
                stream=True
            )
        """
        model = kwargs.get("model", "")

        # Schätze Tokens (grob)
        estimated_tokens = self._estimate_input_tokens(kwargs.get("messages", []))

        for attempt in range(self.max_retries + 1):
            try:
                # Warte auf Rate Limit
                await self.rate_limiter.acquire(
                    model=model, estimated_input_tokens=estimated_tokens
                )

                # Führe Request aus
                response = await litellm_module.acompletion(**kwargs)

                # Melde Erfolg
                self.rate_limiter.report_success(model)

                return response

            except Exception as e:
                error_str = str(e).lower()

                # Prüfe ob es ein Rate-Limit-Fehler ist
                is_rate_limit = any(
                    x in error_str
                    for x in [
                        "rate_limit",
                        "ratelimit",
                        "429",
                        "quota",
                        "resource_exhausted",
                        "too many requests",
                    ]
                )

                if is_rate_limit and attempt < self.max_retries:
                    # Verarbeite Fehler und warte
                    wait_time = self.rate_limiter.handle_rate_limit_error(
                        model=model, error=e
                    )

                    logger.warning(
                        f"[RateLimitHandler] Rate limit hit (attempt {attempt + 1}/{self.max_retries}), "
                        f"waiting {wait_time:.1f}s before retry"
                    )

                    await asyncio.sleep(wait_time)
                else:
                    # Kein Rate-Limit-Fehler oder max retries erreicht
                    raise

    def _estimate_input_tokens(self, messages: list) -> int:
        """Grobe Schätzung der Input-Tokens"""
        if not messages:
            return 0

        total_chars = sum(len(str(m.get("content", ""))) for m in messages)

        # Grobe Schätzung: 4 Zeichen pro Token
        return total_chars // 4


# ===== BEISPIEL USAGE =====


async def example_usage():
    """Beispiel für die Verwendung des intelligenten Rate Limiters"""
    import litellm

    # Initialisiere Rate Limiter
    rate_limiter = IntelligentRateLimiter(
        safety_margin=0.85,  # Nutze nur 85% des Limits
        enable_token_tracking=True,
        persist_learned_limits=True,
    )

    # Optional: Setze bekannte Limits für deinen API-Plan
    rate_limiter.set_limits(
        model="vertex_ai/gemini-2.5-flash", rpm=15, input_tpm=250000, is_free_tier=True
    )

    # Handler für LiteLLM
    handler = LiteLLMRateLimitHandler(rate_limiter, max_retries=3)

    # Beispiel-Request
    try:
        response = await handler.completion_with_rate_limiting(
            litellm,
            model="vertex_ai/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            stream=False,
        )
        print(response)
    except Exception as e:
        print(f"Request failed: {e}")

    # Statistiken anzeigen
    stats = rate_limiter.get_stats("vertex_ai/gemini-2.5-flash")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(example_usage())
