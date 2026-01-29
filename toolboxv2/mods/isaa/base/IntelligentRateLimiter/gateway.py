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
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator, Union
import requests
import httpx
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse


class CustomLLMError(Exception):
    """Custom exception for LLM errors"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class GatewayProvider(CustomLLM):
    """
    Custom LLM Provider for your Gateway

    This gives you full control over request/response handling.
    Supports both sync and async operations including streaming.
    """

    def __init__(self, gateway_url: str=None, api_key: str=None):
        super().__init__()
        self.gateway_url = gateway_url or GW_URL # Remove /v1 if present
        self.api_key = api_key or GW_KEY
        self._available_models = None

    def get_available_models(self) -> List[str]:
        """Fetch available models from gateway"""
        if self._available_models is not None:
            return self._available_models

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.gateway_url}/models",
                headers=headers,
                timeout=2
            )

            if response.status_code == 200:
                data = response.json()
                self._available_models = [m['id'] for m in data.get('data', [])]
                # print(f"✅ Gateway models: {self._available_models}")
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
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(
                    f"{self.gateway_url}/models",
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

    def _get_allowed_params(self) -> set:
        """Return set of allowed OpenAI-compatible parameters"""
        return {
            'temperature', 'top_p', 'max_tokens', 'stream',
            'stop', 'n', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'seed', 'tools', 'tool_choice',
            'response_format', 'logprobs', 'top_logprobs'
        }

    def _clean_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to only include allowed params"""
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})

        allowed = self._get_allowed_params()

        if optional_params:
            optional_params = {k: v for k, v in optional_params.items() if k in allowed and v is not None}
        if litellm_params:
            litellm_params = {k: v for k, v in litellm_params.items() if k in allowed and v is not None}

        print(f"{litellm_params=}")
        print(f"{optional_params=}")
        print(f"{kwargs=}")
        # kwargs.update(optional_params)
        # kwargs.update(litellm_params)

        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    def _extract_model(self, model: str) -> str:
        """Extract actual model name (remove provider prefix)"""
        return model.split("/", 1)[-1] if "/" in model else model

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single SSE line and return data dict or None"""
        import json

        line = line.strip()
        if not line or line.startswith(':'):
            return None

        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                return None
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return None
        return None

    def _chunk_to_generic(self, chunk_data: Dict[str, Any]) -> GenericStreamingChunk:
        """Convert OpenAI chunk format to GenericStreamingChunk"""
        choices = chunk_data.get('choices', [{}])
        choice = choices[0] if choices else {}
        delta = choice.get('delta', {})
        finish_reason = choice.get('finish_reason')

        # Extract usage if present
        usage_data = chunk_data.get('usage', {})
        usage = {
            "completion_tokens": usage_data.get('completion_tokens', 0),
            "prompt_tokens": usage_data.get('prompt_tokens', 0),
            "total_tokens": usage_data.get('total_tokens', 0)
        }

        return {
            "text": delta.get('content', '') or '',
            "is_finished": finish_reason is not None,
            "finish_reason": finish_reason,
            "index": choice.get('index', 0),
            "tool_use": delta.get('tool_calls'),
            "usage": usage
        }

    # ========== NON-STREAMING METHODS ==========

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Handle non-streaming completion requests (sync)"""

        clean_kwargs = self._clean_kwargs(kwargs)
        clean_kwargs.pop('stream', None)  # Ensure no streaming

        actual_model = self._extract_model(model)
        base_url = api_base or f"{self.gateway_url}"

        payload = {
            "model": actual_model,
            "messages": messages,
            **clean_kwargs
        }

        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
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
    ) -> ModelResponse:
        """Handle non-streaming completion requests (async)"""

        clean_kwargs = self._clean_kwargs(kwargs)
        clean_kwargs.pop('stream', None)  # Ensure no streaming

        actual_model = self._extract_model(model)
        base_url = api_base or f"{self.gateway_url}"

        payload = {
            "model": actual_model,
            "messages": messages,
            **clean_kwargs
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return ModelResponse(**response.json())

    # ========== STREAMING METHODS ==========

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> Iterator[GenericStreamingChunk]:
        """
        Handle streaming completion requests (sync)

        Returns an iterator that yields GenericStreamingChunk objects.
        LiteLLM will wrap this in CustomStreamWrapper.
        """

        clean_kwargs = self._clean_kwargs(kwargs)
        clean_kwargs['stream'] = True  # Ensure streaming is enabled

        actual_model = self._extract_model(model)
        base_url = api_base or f"{self.gateway_url}"

        payload = {
            "model": actual_model,
            "messages": messages,
            **clean_kwargs
        }

        # Use streaming request
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    chunk_data = self._parse_sse_line(line)
                    if chunk_data:
                        yield self._chunk_to_generic(chunk_data)

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[GenericStreamingChunk]:
        """
        Handle streaming completion requests (async)

        Returns an async iterator that yields GenericStreamingChunk objects.
        LiteLLM will wrap this in CustomStreamWrapper.

        IMPORTANT: This must be an async generator (use 'yield', not 'return')
        """

        clean_kwargs = self._clean_kwargs(kwargs)
        clean_kwargs['stream'] = True  # Ensure streaming is enabled

        actual_model = self._extract_model(model)
        base_url = api_base or f"{self.gateway_url}"

        payload = {
            "model": actual_model,
            "messages": messages,
            **clean_kwargs
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    chunk_data = self._parse_sse_line(line)
                    if chunk_data:
                        yield self._chunk_to_generic(chunk_data)

    # ========== EMBEDDING METHODS ==========

    def embedding(
        self,
        model: str,
        input: Any,
        api_base: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Handle embedding requests (sync)"""

        allowed_params = {'encoding_format', 'user', 'dimensions'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params and v is not None}

        actual_model = self._extract_model(model)
        base_url = api_base or f"{self.gateway_url}"

        payload = {
            "model": actual_model,
            "input": input,
            **clean_kwargs
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url}/embeddings",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return ModelResponse(**response.json())

    async def aembedding(
        self,
        model: str,
        input: Any,
        api_base: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Handle embedding requests (async)"""

        allowed_params = {'encoding_format', 'user', 'dimensions'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params and v is not None}

        actual_model = self._extract_model(model)
        base_url = api_base or f"{self.gateway_url}"

        payload = {
            "model": actual_model,
            "input": input,
            **clean_kwargs
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/embeddings",
                json=payload,
                headers=self._get_headers()
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

    return setup_custom_provider("gateway", GW_URL, GW_KEY)

# === Setup Function ===
def setup_custom_provider(prefix: str, api_base: str, api_key: str):
    """
    Register gateway provider with LiteLLM

    Call this at the start of your application.
    """
    provider = GatewayProvider(api_base, api_key)
    if litellm.custom_provider_map is None:
        litellm.custom_provider_map = []
    litellm.custom_provider_map.append(
        {
            "provider": prefix,
            "custom_handler": provider
        }
    )
    print(f"Registered custom provider {prefix} with api_base {api_base}")
    return provider


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

    # Tool definitions in different formats
    tools_openai = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]


    try:

        models = gateway.get_available_models()
        test_model = models[0] if models else "qwen3-4b-instruct-2507.q4_k_m"
        response = litellm.completion(
            model=f"gateway/{test_model}",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools_openai,
            tool_choice="auto",
            max_tokens=200
        )
        msg = response.choices[0].message
        print(f"✅ Content: {msg.content}")
        print(f"   Tool calls: {msg.tool_calls}")
    except Exception as e:
        print(f"❌ Error: {e}")


    # Method 3: Direct approach
    print("\n" + "=" * 60)
    print("METHOD 3: Direct OpenAI-compatible (Easiest)")
    print("=" * 60)

    print("\n✅ All methods ready to use!")
