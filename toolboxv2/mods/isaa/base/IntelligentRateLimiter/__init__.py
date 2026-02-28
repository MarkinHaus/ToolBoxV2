import os

from .gateway import setup_gateway_provider, setup_custom_provider
from .intelligent_rate_limiter import (
    IntelligentRateLimiter,
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
)
from .minimax_provider import register_minimax

gateway = setup_gateway_provider()
# zai_provider = setup_custom_provider("zglm", os.getenv("ZAI_API_BASE"), os.getenv("ZAI_API_KEY"))
from .zai_litellm_provider import setup_zai_provider
provider = setup_zai_provider(debug=False)
if os.getenv("MINIMAX_API_BASE") and os.getenv("MINIMAX_API_KEY"):
    mm_provider = register_minimax(provider_name="mm", debug=False)
