"""
LiteLLM Custom Provider for your LLM Gateway

Installation:
1. Place this file in your project directory
2. Update litellm's openai_compatible_providers.json OR
3. Use the CustomLLM class approach for full control

Usage:
    import litellm
    from gateway_provider import setup_gateway_provider

    # Setup provider
    setup_gateway_provider()

    # Use it
    response = litellm.completion(
        model="gateway/qwen3-4b-instruct-2507.q4_k_m",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import litellm
from litellm import CustomLLM, completion
from litellm.types.utils import ModelResponse

# === Configuration ===
GW_URL = os.getenv("TB_LLM_GATEWAY_URL", "http://localhost:4000/v1")
GW_KEY = os.getenv("TB_LLM_GATEWAY_KEY", "sk-admin-change-me-on-first-run")


# === Method 1: JSON-based Provider (Simplest) ===
def add_to_openai_providers_json():
    """
    Add gateway provider to LiteLLM's openai_compatible_providers.json

    This is the SIMPLEST approach - just add your config to the JSON file.

    File location (usually):
    - ~/.local/lib/python3.x/site-packages/litellm/llms/openai_compatible_providers.json

    Add this entry:
    {
      "gateway": {
        "base_url": "http://localhost:4000/v1",
        "api_key_env": "TB_LLM_GATEWAY_KEY"
      }
    }
    """
    provider_config = {
        "gateway": {
            "base_url": GW_URL,
            "api_key_env": "TB_LLM_GATEWAY_KEY"
        }
    }

    print("📝 Add this to litellm/llms/openai_compatible_providers.json:")
    print(json.dumps(provider_config, indent=2))
    print("\nThen use: litellm.completion(model='gateway/your-model', ...)")


# === Method 2: CustomLLM Class (Full Control) ===
class GatewayProvider(CustomLLM):
    """
    Custom LLM Provider for your Gateway

    This gives you full control over request/response handling.
    Supports both sync and async operations.
    """

    def __init__(self):
        super().__init__()
        self.gateway_url = GW_URL.rstrip('/v1')  # Remove /v1 if present
        self.api_key = GW_KEY
        self._available_models = None

    def get_available_models(self) -> List[str]:
        """Fetch available models from gateway"""
        if self._available_models is not None:
            return self._available_models

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.gateway_url}/v1/models",
                headers=headers,
                timeout=2
            )

            if response.status_code == 200:
                data = response.json()
                self._available_models = [m['id'] for m in data.get('data', [])]
                print(f"✅ Gateway models: {self._available_models}")
            else:
                print(f"⚠️ Gateway responded with status {response.status_code}")
                self._available_models = []

        except Exception as e:
            print(f"⚠️ Could not fetch models from gateway: {e}")
            self._available_models = []

        return self._available_models

    async def aget_available_models(self) -> List[str]:
        """Async: Fetch available models from gateway"""
        if self._available_models is not None:
            return self._available_models

        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(
                    f"{self.gateway_url}/v1/models",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    self._available_models = [m['id'] for m in data.get('data', [])]
                    print(f"✅ Gateway models: {self._available_models}")
                else:
                    print(f"⚠️ Gateway responded with status {response.status_code}")
                    self._available_models = []

        except Exception as e:
            print(f"⚠️ Could not fetch models from gateway: {e}")
            self._available_models = []

        return self._available_models

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ):
        """
        Handle completion requests (sync)

        LiteLLM will call this with model="gateway/model-name"
        We need to:
        1. Extract actual model name
        2. Forward to gateway
        3. Return raw response (LiteLLM will parse it)
        """

        # Filter out LiteLLM internal params and non-serializable objects
        # Only keep OpenAI-compatible parameters
        allowed_params = {
            'temperature', 'top_p', 'max_tokens', 'stream',
            'stop', 'n', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'seed', 'tools', 'tool_choice',
            'response_format', 'logprobs', 'top_logprobs'
        }

        # Build clean payload with only allowed params
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k in allowed_params and v is not None
        }

        # Extract model name (remove "gateway/" prefix if present)
        actual_model = model.split("/", 1)[-1] if "/" in model else model

        # Use gateway URL
        base_url = api_base or f"{self.gateway_url}/v1"

        # Make direct HTTP request instead of using litellm.completion
        # This avoids double-wrapping the response
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": actual_model,
            "messages": messages,
            **clean_kwargs
        }

        # Check if streaming
        is_streaming = clean_kwargs.get("stream", False)

        if is_streaming:
            # For streaming, we need to return the raw stream
            # LiteLLM will handle the parsing
            client = httpx.Client(timeout=300.0)
            response = client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True
            )
            return response
        else:
            # For non-streaming, return the JSON response
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return ModelResponse(**response.json())

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ):
        """
        Handle completion requests (async)

        Async version of completion() for better performance
        in async applications.
        """

        # Filter out LiteLLM internal params
        allowed_params = {
            'temperature', 'top_p', 'max_tokens', 'stream',
            'stop', 'n', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'seed', 'tools', 'tool_choice',
            'response_format', 'logprobs', 'top_logprobs'
        }

        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k in allowed_params and v is not None
        }

        actual_model = model.split("/", 1)[-1] if "/" in model else model
        base_url = api_base or f"{self.gateway_url}/v1"

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": actual_model,
            "messages": messages,
            **clean_kwargs
        }

        is_streaming = clean_kwargs.get("stream", False)

        if is_streaming:
            # Async streaming
            client = httpx.AsyncClient(timeout=300.0)
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            # Return async iterator for streaming
            return response.aiter_lines()
        else:
            # Async non-streaming
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return ModelResponse(**response.json())

    def embedding(
        self,
        model: str,
        input: Any,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """Handle embedding requests (sync)"""

        # Filter allowed params
        allowed_params = {'encoding_format', 'user', 'dimensions'}
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k in allowed_params and v is not None
        }

        actual_model = model.split("/", 1)[-1] if "/" in model else model
        base_url = api_base or f"{self.gateway_url}/v1"

        # Direct HTTP request
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": actual_model,
            "input": input,
            **clean_kwargs
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url}/embeddings",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return ModelResponse(**response.json())

    async def aembedding(
        self,
        model: str,
        input: Any,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """Handle embedding requests (async)"""

        # Filter allowed params
        allowed_params = {'encoding_format', 'user', 'dimensions'}
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k in allowed_params and v is not None
        }

        actual_model = model.split("/", 1)[-1] if "/" in model else model
        base_url = api_base or f"{self.gateway_url}/v1"

        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": actual_model,
            "input": input,
            **clean_kwargs
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/embeddings",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return ModelResponse(**response.json())


# === Method 3: Direct OpenAI-compatible calls ===
def use_gateway_directly():
    """
    Use gateway directly as OpenAI-compatible endpoint

    This is the EASIEST if you don't need special provider handling.
    Just use openai/ prefix with custom api_base.
    """

    response = litellm.completion(
        model="openai/qwen3-4b-instruct-2507.q4_k_m",  # or any model name
        messages=[{"role": "user", "content": "Hello!"}],
        api_base=GW_URL,
        api_key=GW_KEY,
    )

    return response


# === Setup Function ===
def setup_gateway_provider():
    """
    Register gateway provider with LiteLLM

    Call this at the start of your application.
    """

    # Create provider instance
    gateway = GatewayProvider()

    # Register with LiteLLM
    litellm.custom_provider_map = [
        {
            "provider": "gateway",
            "custom_handler": gateway
        }
    ]

    # Fetch available models
    models = gateway.get_available_models()

    print("✅ Gateway provider registered!")
    print(f"📡 URL: {GW_URL}")
    print(f"🔑 Key: {GW_KEY[:12]}...")
    print(f"🤖 Models: {models if models else 'Could not fetch (using fallback)'}")

    return gateway


# === Usage Examples ===
if __name__ == "__main__":
    print("🚀 LiteLLM Gateway Provider Setup\n")

    # Method 1: Show JSON config (manual addition)
    print("=" * 60)
    print("METHOD 1: JSON-based Provider (Simplest)")
    print("=" * 60)
    add_to_openai_providers_json()

    # Method 2: Custom Provider (recommended)
    print("\n" + "=" * 60)
    print("METHOD 2: Custom Provider Class (Recommended)")
    print("=" * 60)
    gateway = setup_gateway_provider()

    # Test it
    print("\n📝 Testing completion...")
    try:
        # Get first available model or use fallback
        models = gateway.get_available_models()
        test_model = models[0] if models else "qwen3-4b-instruct-2507.q4_k_m"

        response = litellm.completion(
            model=f"gateway/{test_model}",
            messages=[{"role": "user", "content": "Say 'Hello from Gateway!'"}],
            max_tokens=50
        )

        print(f"✅ Success! Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Method 3: Direct approach
    print("\n" + "=" * 60)
    print("METHOD 3: Direct OpenAI-compatible (Easiest)")
    print("=" * 60)
    print("""
# Just use OpenAI with custom api_base:

response = litellm.completion(
    model="openai/your-model-name",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="{GW_URL}",
    api_key="{GW_KEY}",
)
""".format(GW_URL=GW_URL, GW_KEY=GW_KEY[:12] + "..."))

    print("\n✅ All methods ready to use!")
