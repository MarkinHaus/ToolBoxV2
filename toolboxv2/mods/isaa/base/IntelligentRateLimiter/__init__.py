from .gateway import setup_gateway_provider
from .intelligent_rate_limiter import (
    IntelligentRateLimiter,
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
)
gateway = setup_gateway_provider()
print(f"{gateway=} GATEWAY SETUP COMPLETE")
