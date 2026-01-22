"""
Test Script für LiteLLM Gateway Integration

Testet alle 3 Methoden:
1. Direct OpenAI-compatible
2. Custom Provider
3. LiteLLM Proxy
"""

import os
import sys
import requests
from typing import List
import dotenv

dotenv.load_dotenv(r"C:\Users\Markin\Workspace\ToolBoxV2\.env")
# Setup Environment
GW_URL = os.getenv("TB_LLM_GATEWAY_URL", "http://localhost:4000/v1")
GW_KEY = os.getenv("TB_LLM_GATEWAY_KEY", "sk-admin-change-me-on-first-run")

print("🔧 Configuration:")
print(f"   Gateway URL: {GW_URL}")
print(f"   API Key: {GW_KEY[:12]}...\n")


# === Fetch Available Models ===
def get_gateway_models() -> List[str]:
    """Hole Modelle vom Gateway"""
    try:
        headers = {"Authorization": f"Bearer {GW_KEY}"}
        response = requests.get(f"{GW_URL.rstrip('/v1')}/v1/models", headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = [m['id'] for m in data.get('data', [])]
            print(f"✅ Gateway Models gefunden: {models}\n")
            return models
        else:
            print(f"⚠️ Gateway Status: {response.status_code}")
            return ["qwen3-4b-instruct-2507.q4_k_m"]

    except Exception as e:
        print(f"⚠️ Gateway nicht erreichbar: {e}")
        return ["qwen3-4b-instruct-2507.q4_k_m"]


available_models = get_gateway_models()
test_model = available_models[0] if available_models else "qwen3-4b-instruct-2507.q4_k_m"


# === Method 1: Direct OpenAI-compatible (EASIEST) ===
def test_direct_openai():
    """Direkter Aufruf als OpenAI-compatible endpoint"""
    print("=" * 70)
    print("METHOD 1: Direct OpenAI-compatible (Empfohlen für schnellen Start)")
    print("=" * 70)

    try:
        import litellm

        response = litellm.completion(
            model=f"openai/{test_model}",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": "Sage 'Hallo von Method 1!' auf Deutsch."}
            ],
            api_base=GW_URL,
            api_key=GW_KEY,
            max_tokens=50,
            temperature=0.7
        )

        content = response.choices[0].message.content
        print(f"✅ Success!")
        print(f"📝 Response: {content}")
        print(f"📊 Tokens: {response.usage.total_tokens}")
        print()
        return True

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


# === Method 2: Custom Provider ===
def test_custom_provider():
    """Custom Provider mit voller Kontrolle"""
    print("=" * 70)
    print("METHOD 2: Custom Provider (Empfohlen für Production)")
    print("=" * 70)

    try:
        import litellm
        from litellm import CustomLLM
        import httpx
        import json as json_module

        from toolboxv2.mods.isaa.base.IntelligentRateLimiter.gateway import GatewayProvider

        # Register Provider
        gateway = GatewayProvider()
        litellm.custom_provider_map = [
            {"provider": "gateway", "custom_handler": gateway}
        ]

        # Test
        response = litellm.completion(
            model=f"gateway/{test_model}",
            messages=[
                {"role": "user", "content": "Sage 'Hallo von Method 2!' auf Deutsch."}
            ],
            max_tokens=50
        )

        content = response.choices[0].message.content
        print(f"✅ Success!")
        print(f"📝 Response: {content}")
        print(f"📊 Tokens: {response.usage.total_tokens}")
        print()
        return True

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(f"📋 Details: {traceback.format_exc()}")
        print()
        return False

def test_custom_provider_streaming():
    """Custom Provider mit voller Kontrolle"""
    print("=" * 70)
    print("METHOD 2: Custom Provider (Empfohlen für Production)")
    print("=" * 70)

    try:
        import litellm
        from litellm import CustomLLM
        import httpx
        import json as json_module

        from toolboxv2.mods.isaa.base.IntelligentRateLimiter.gateway import GatewayProvider

        # Register Provider
        gateway = GatewayProvider()
        litellm.custom_provider_map = [
            {"provider": "gateway", "custom_handler": gateway}
        ]

        # Test
        response = litellm.completion(
            model=f"gateway/{test_model}",
            messages=[
                {"role": "user", "content": "Sage 'Hallo von Method 2!' auf Deutsch."}
            ],
            max_tokens=50,
            stream=True,
            stream_options={"include_usage": True}
        )
        print("📡 Streaming: ", end="")

        # Sammle Chunks für stream_chunk_builder
        chunks = []
        for chunk in response:
            chunks.append(chunk)
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end="", flush=True)

        print()  # Newline nach Stream

        # Baue vollständige Response aus Chunks
        complete_response = litellm.stream_chunk_builder(
            chunks,
            messages=[{"role": "user", "content": "..."}]
        )

        print(f"✅ Success!")
        print(f"📝 Full Response: {complete_response.choices[0].message.content}")

        if complete_response.usage:
            print(f"📊 Tokens: prompt={complete_response.usage.prompt_tokens}, "
                  f"completion={complete_response.usage.completion_tokens}, "
                  f"total={complete_response.usage.total_tokens}")

        print()
        return True

    except Exception as e:
            import traceback
            print(f"❌ Error: {e}")
            print(f"📋 Details: {traceback.format_exc()}")
            print()
            return False


# === Method 3: Test streaming ===
def test_streaming():
    """Test Streaming-Antworten"""
    print("=" * 70)
    print("METHOD 3: Streaming Test")
    print("=" * 70)

    try:
        import litellm

        print("📡 Streaming response:")

        response = litellm.completion(
            model=f"openai/{test_model}",
            messages=[
                {"role": "user", "content": "Zähle von 1 bis 5 auf Deutsch."}
            ],
            api_base=GW_URL,
            api_key=GW_KEY,
            max_tokens=100,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print(f"\n\n✅ Streaming successful!")
        print(f"📝 Full response: {full_response}")
        print()
        return True

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


# === Method 4: Test with OpenAI SDK ===
def test_openai_sdk():
    """Test mit Official OpenAI SDK"""
    print("=" * 70)
    print("METHOD 4: OpenAI SDK (Drop-in Replacement)")
    print("=" * 70)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=GW_KEY,
            base_url=GW_URL
        )

        response = client.chat.completions.create(
            model=test_model,
            messages=[
                {"role": "user", "content": "Sage 'Hallo von OpenAI SDK!' auf Deutsch."}
            ],
            max_tokens=50
        )

        content = response.choices[0].message.content
        print(f"✅ Success!")
        print(f"📝 Response: {content}")
        print(f"📊 Tokens: {response.usage.total_tokens}")
        print()
        return True

    except ImportError:
        print("⚠️ OpenAI SDK nicht installiert (pip install openai)\n")
        return None
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


# === Run All Tests ===
def main():
    print("🧪 Starting LiteLLM Gateway Integration Tests\n")

    results = {
        "Direct OpenAI": test_direct_openai(),
        "Custom Provider": test_custom_provider(),
        "Custom Provider Streaming": test_custom_provider_streaming(),
        "Streaming": test_streaming(),
        "OpenAI SDK": test_openai_sdk(),
    }

    # Summary
    print("=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️ SKIP"
        print(f"{status:12} {name}")

    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)

    print(f"\nErgebnis: {passed}/{total} Tests bestanden")

    if passed == total:
        print("\n🎉 Alle Tests erfolgreich! Gateway ist bereit für LiteLLM.")
    else:
        print("\n⚠️ Einige Tests fehlgeschlagen. Prüfe Gateway und API Key.")
        sys.exit(1)


if __name__ == "__main__":
    main()
