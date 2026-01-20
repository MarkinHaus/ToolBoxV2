"""
Einfacher Gateway Wrapper für LiteLLM

WARUM DIESER ANSATZ BESSER IST:
- Kein Custom Provider nötig
- Keine Serialisierungs-Probleme
- Funktioniert out-of-the-box
- Einfacher zu warten
- Vollständiger Async Support

Dein Gateway ist bereits OpenAI-kompatibel, also nutzen wir das!
"""

import os
import requests
from typing import List, Optional, Dict, Any
import litellm


class GatewayClient:
    """
    Simple Gateway Client für LiteLLM

    Unterstützt sync und async operations.

    Sync Usage:
        gw = GatewayClient()
        response = gw.completion("qwen3-4b-instruct", [{"role": "user", "content": "Hi"}])

    Async Usage:
        gw = GatewayClient()
        response = await gw.acompletion("qwen3-4b-instruct", [{"role": "user", "content": "Hi"}])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url or os.getenv("TB_LLM_GATEWAY_URL", "http://localhost:4000/v1")
        self.api_key = api_key or os.getenv("TB_LLM_GATEWAY_KEY", "sk-admin-change-me-on-first-run")
        self._models_cache = None

    # === Sync Methods ===

    def get_models(self, force_refresh: bool = False) -> List[str]:
        """Hole verfügbare Modelle vom Gateway (sync)"""
        if self._models_cache and not force_refresh:
            return self._models_cache

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.base_url.rstrip('/v1')}/v1/models",
                headers=headers,
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                self._models_cache = [m['id'] for m in data.get('data', [])]
                return self._models_cache
            else:
                print(f"⚠️ Gateway Status: {response.status_code}")
                return []

        except Exception as e:
            print(f"⚠️ Gateway nicht erreichbar: {e}")
            return []

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Completion via Gateway (sync, mit LiteLLM)

        Args:
            model: Model name (ohne prefix)
            messages: Chat messages
            **kwargs: LiteLLM parameters (temperature, max_tokens, etc.)

        Returns:
            LiteLLM ModelResponse
        """

        # Use LiteLLM's openai/ provider with custom base_url
        # Dies ist der EINFACHSTE und ROBUSTESTE Weg!
        return litellm.completion(
            model=f"openai/{model}",
            messages=messages,
            api_base=self.base_url,
            api_key=self.api_key,
            **kwargs
        )

    def embedding(
        self,
        model: str,
        input: Any,
        **kwargs
    ):
        """Embeddings via Gateway (sync)"""
        return litellm.embedding(
            model=f"openai/{model}",
            input=input,
            api_base=self.base_url,
            api_key=self.api_key,
            **kwargs
        )

    # === Async Methods ===

    async def aget_models(self, force_refresh: bool = False) -> List[str]:
        """Hole verfügbare Modelle vom Gateway (async)"""
        if self._models_cache and not force_refresh:
            return self._models_cache

        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url.rstrip('/v1')}/v1/models",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    self._models_cache = [m['id'] for m in data.get('data', [])]
                    return self._models_cache
                else:
                    print(f"⚠️ Gateway Status: {response.status_code}")
                    return []

        except Exception as e:
            print(f"⚠️ Gateway nicht erreichbar: {e}")
            return []

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Async completion via Gateway

        Args:
            model: Model name (ohne prefix)
            messages: Chat messages
            **kwargs: LiteLLM parameters

        Returns:
            LiteLLM ModelResponse
        """
        return await litellm.acompletion(
            model=f"openai/{model}",
            messages=messages,
            api_base=self.base_url,
            api_key=self.api_key,
            **kwargs
        )

    async def aembedding(
        self,
        model: str,
        input: Any,
        **kwargs
    ):
        """Async embeddings via Gateway"""
        return await litellm.aembedding(
            model=f"openai/{model}",
            input=input,
            api_base=self.base_url,
            api_key=self.api_key,
            **kwargs
        )

    # === Batch Operations (Async) ===

    async def batch_completion(
        self,
        model: str,
        messages_list: List[List[Dict[str, Any]]],
        **kwargs
    ):
        """
        Process multiple completions in parallel (async)

        Args:
            model: Model name
            messages_list: List of message lists to process
            **kwargs: Shared parameters for all requests

        Returns:
            List of ModelResponse objects
        """
        import asyncio

        tasks = [
            self.acompletion(model, messages, **kwargs)
            for messages in messages_list
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def batch_embedding(
        self,
        model: str,
        inputs: List[Any],
        **kwargs
    ):
        """
        Process multiple embeddings in parallel (async)

        Args:
            model: Model name
            inputs: List of inputs to embed
            **kwargs: Shared parameters

        Returns:
            List of EmbeddingResponse objects
        """
        import asyncio

        tasks = [
            self.aembedding(model, input_text, **kwargs)
            for input_text in inputs
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)


# === Convenience Functions ===

def create_gateway_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> GatewayClient:
    """Factory function"""
    return GatewayClient(base_url, api_key)


# === Usage Examples ===

if __name__ == "__main__":
    import asyncio

    print("🚀 Simple Gateway Client Test\n")

    # Create client
    gw = GatewayClient()

    # === SYNC TESTS ===
    print("=" * 70)
    print("SYNC TESTS")
    print("=" * 70)

    # Get models
    print("\n📡 Fetching models (sync)...")
    models = gw.get_models()

    if not models:
        print("❌ No models found. Is gateway running?")
        exit(1)

    print(f"✅ Found {len(models)} models: {models}\n")

    # Test completion
    test_model = models[0]
    print(f"🧪 Testing completion with {test_model}...")

    try:
        response = gw.completion(
            model=test_model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": "Sage 'Hello from Gateway!' auf Deutsch."}
            ],
            max_tokens=50,
            temperature=0.7
        )

        print(f"✅ Success!")
        print(f"📝 Response: {response.choices[0].message.content}")
        print(f"📊 Tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test streaming
    print("\n🌊 Testing streaming (sync)...")
    try:
        response = gw.completion(
            model=test_model,
            messages=[{"role": "user", "content": "Zähle von 1 bis 3."}],
            stream=True,
            max_tokens=50
        )

        print("📡 Streaming: ", end="", flush=True)
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n✅ Streaming successful!")

    except Exception as e:
        print(f"❌ Error: {e}")

    # === ASYNC TESTS ===
    async def async_tests():
        print("\n" + "=" * 70)
        print("ASYNC TESTS")
        print("=" * 70)

        # Async get models
        print("\n📡 Fetching models (async)...")
        models = await gw.aget_models(force_refresh=True)
        print(f"✅ Found {len(models)} models: {models}")

        # Async completion
        print(f"\n🧪 Testing async completion with {test_model}...")
        try:
            response = await gw.acompletion(
                model=test_model,
                messages=[
                    {"role": "user", "content": "Sage 'Hello from async!' auf Deutsch."}
                ],
                max_tokens=50
            )

            print(f"✅ Success!")
            print(f"📝 Response: {response.choices[0].message.content}")
            print(f"📊 Tokens: {response.usage.total_tokens}")

        except Exception as e:
            print(f"❌ Error: {e}")

        # Batch processing
        print(f"\n🔄 Testing batch completion (parallel)...")
        try:
            messages_batch = [
                [{"role": "user", "content": "Zähle von 1 bis 3."}],
                [{"role": "user", "content": "Was ist 2+2?"}],
                [{"role": "user", "content": "Nenne 3 Farben."}]
            ]

            responses = await gw.batch_completion(
                model=test_model,
                messages_list=messages_batch,
                max_tokens=30
            )

            print(f"✅ Processed {len(responses)} requests in parallel!")
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    print(f"  {i+1}. Error: {resp}")
                else:
                    print(f"  {i+1}. {resp.choices[0].message.content[:50]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Run async tests
    asyncio.run(async_tests())

    print("\n" + "=" * 70)
    print("🎉 All tests completed!")
    print("=" * 70)
    print("""
USAGE IN YOUR CODE:

# === SYNC ===
from gateway_client import GatewayClient

gw = GatewayClient()

# Simple completion
response = gw.completion(
    "your-model-name",
    [{"role": "user", "content": "Hello!"}]
)

# With all LiteLLM features
response = gw.completion(
    "your-model-name",
    messages=[...],
    temperature=0.7,
    max_tokens=100,
    stream=True,
    tools=[...],  # Function calling
    response_format={"type": "json_object"}  # JSON mode
)

# === ASYNC ===
import asyncio

async def main():
    gw = GatewayClient()

    # Single request
    response = await gw.acompletion(
        "your-model-name",
        [{"role": "user", "content": "Hello!"}]
    )

    # Batch processing (parallel)
    responses = await gw.batch_completion(
        "your-model-name",
        messages_list=[
            [{"role": "user", "content": "Query 1"}],
            [{"role": "user", "content": "Query 2"}],
            [{"role": "user", "content": "Query 3"}],
        ],
        max_tokens=100
    )

asyncio.run(main())
""")
