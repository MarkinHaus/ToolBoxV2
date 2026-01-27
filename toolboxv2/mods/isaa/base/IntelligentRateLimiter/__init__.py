import os

from .gateway import setup_gateway_provider, setup_custom_provider
from .intelligent_rate_limiter import (
    IntelligentRateLimiter,
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
)
gateway = setup_gateway_provider()
# zai_provider = setup_custom_provider("zglm", os.getenv("ZAI_API_BASE"), os.getenv("ZAI_API_KEY"))
