import os
import requests
import litellm
import json

# ==========================================
# KONFIGURATION
# ==========================================
# Setze hier deinen API Key ein oder nutze die Umgebungsvariable
API_KEY = os.getenv("ZAI_API_KEY", "DEIN_API_KEY_HIER")

# Verwende glm-5 oder glm-4.7 je nach deinem Plan
MODEL_NAME = "glm-4.7"


# ==========================================
# TEST 1: Direkter HTTP-Aufruf (OpenAI Format)
# ==========================================
def test_direct_openai():
    print("\n--- TEST 1: Direkter HTTP-Aufruf (OpenAI Protocol auf Coding Endpoint) ---")
    # WICHTIG: Das ist der spezifische Endpunkt für den Coding Plan!
    url = "https://api.z.ai/api/coding/paas/v4/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hallo, antworte mit genau 3 Wörtern."}],
        "max_tokens": 50
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Erfolg! Antwort:", response.json()["choices"][0]["message"]["content"])
    else:
        print("❌ Fehler:", response.text)


# ==========================================
# TEST 2: Nativer LiteLLM OpenAI Provider (Der beste Weg!)
# ==========================================
def test_litellm_native_openai():
    print("\n--- TEST 2: LiteLLM nativer OpenAI Provider ---")
    print("Nutzt LiteLLMs eingebautes OpenAI Format ohne Custom Adapter.")

    try:
        # Indem wir "openai/" voranstellen, nutzt LiteLLM den nativen OpenAI Code
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",
            api_base="https://api.z.ai/api/coding/paas/v4/",  # Coding Endpunkt!
            api_key=API_KEY,
            messages=[{"role": "user", "content": "Hallo Z.AI, wie geht es dir?"}],
            max_tokens=50
        )
        print("✅ Erfolg! Antwort:", response.choices[0].message.content)
    except Exception as e:
        print("❌ Fehler bei LiteLLM OpenAI:", str(e))


# ==========================================
# TEST 3: Direkter HTTP-Aufruf (Anthropic Format)
# ==========================================
def test_direct_anthropic():
    print("\n--- TEST 3: Direkter HTTP-Aufruf (Anthropic Protocol) ---")
    print("Dieser Test prüft, ob der Anthropic Endpoint (den dein Adapter nutzt) dich blockiert.")

    url = "https://api.z.ai/api/anthropic/messages"

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        # Manchmal fordert Z.AI hier einen spezifischen User-Agent wie Claude Code
        # "User-Agent": "claude-code/1.0"
    }
    data = {
        "model": MODEL_NAME,  # Bei Anthropic evtl. nur "GLM-4.7" erlaubt
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Test"}]
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Erfolg! Antwort:", response.json())
    else:
        print("❌ Fehler (Vermutlich 401 wegen Restriktionen):", response.text)


if __name__ == "__main__":
    if API_KEY == "DEIN_API_KEY_HIER":
        print("⚠️ Bitte trage deinen Z.AI API-Key in das Skript ein.")
    else:
        print("🚀 Starte Z.AI Coding Plan API Tests...")
        test_direct_openai()
        test_litellm_native_openai()
        test_direct_anthropic()
