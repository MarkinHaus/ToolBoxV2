import os
import logging

# Lazy — no provider setup at import time.
# Provider registration now happens via CompletionRouter.setup_default_adapters()

from .intelligent_rate_limiter import (
    IntelligentRateLimiter,
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
    FallbackReason,
)


def setup_legacy_providers():
    """Call explicitly if legacy litellm code paths are still needed."""
    from .gateway import setup_gateway_provider, setup_custom_provider
    from .minimax_provider import register_minimax
    from .zai_litellm_provider import setup_zai_provider

    gateway = setup_gateway_provider()
    provider = setup_zai_provider(debug=False)
    if os.getenv("MINIMAX_API_BASE") and os.getenv("MINIMAX_API_KEY"):
        register_minimax(provider_name="mm", debug=False)

    try:
        from .inception_litellm_provider import setup_inception_provider
        setup_inception_provider()
    except Exception:
        logging.getLogger(__name__).debug("inception provider not available")

    return gateway, provider
