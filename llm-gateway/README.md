# LLM Gateway

**OpenAI-kompatible API für lokale LLM-Modelle mit llama.cpp**

Ein leichtgewichtiger Gateway-Server, der lokale GGUF-Modelle über eine OpenAI-kompatible REST-API bereitstellt. Unterstützt Text, Vision, Audio (Omni), Embeddings und TTS.

## Features

### 🚀 Core Features
- **OpenAI-kompatible API** - Drop-in Ersatz für OpenAI API
- **Multi-Model Slots** - Bis zu 7 Modelle gleichzeitig (Ports 4801-4807)
- **Smart Routing** - Automatische Modellauswahl basierend auf Request-Typ
- **Streaming** - Server-Sent Events für Chat Completions
- **Rate Limiting** - Konfigurierbare Limits pro User-Tier

### 🎯 Modell-Typen
| Typ | Beschreibung | Capabilities |
|-----|--------------|--------------|
| `text` | Standard Chat-Modelle | Text |
| `vision` | Vision-Language Modelle (VL) | Text + Bild |
| `omni` | Multimodale Modelle | Text + Bild + Audio |
| `embedding` | Embedding-Modelle | Vektorisierung |
| `vision-embedding` | Vision + Embedding | Bild-Vektorisierung |
| `audio` | Whisper (Legacy) | Transkription |
| `tts` | Text-to-Speech | Sprachsynthese |

### 🔐 User Management
- API-Key basierte Authentifizierung
- User-Tiers: `payg` (Pay-as-you-go), `sub` (Subscription), `admin`
- Balance-Tracking und Usage-Logging
- Signup-Request System

### 🎙️ Live Voice API
- WebSocket-basierte Echtzeit-Sprachkonversation
- Wake-Word Detection (pre/post/mid Modi)
- Interrupt-Handling
- Paralleles LLM + TTS Streaming

## Installation

### Voraussetzungen
- Python 3.12+
- CMake, Git, Build Tools
- ~48GB RAM empfohlen für große Modelle

### Linux/macOS
```bash
cd llm-gateway
chmod +x setup.sh
./setup.sh
```

### Windows
```powershell
cd llm-gateway
.\win_setup.ps1
```

### Mit ToolBoxV2 CLI
```bash
tb llm-gateway setup
tb llm-gateway start
```

## Konfiguration

Die Konfiguration liegt in `data/config.json`:

```json
{
  "slots": {
    "4801": null,
    "4802": null,
    "4803": null,
    "4804": null,
    "4805": null,
    "4806": null,
    "4807": null
  },
  "hf_token": "hf_xxx",
  "admin_key": "sk-admin-xxx",
  "default_threads": 10,
  "default_ctx_size": 8192,
  "pricing": {
    "input_per_1k": 0.0001,
    "output_per_1k": 0.0002
  },
  "rate_limits": {
    "payg": 25,
    "sub": 100
  },
  "performance": {
    "flash_attention": true,
    "mlock": true,
    "kv_cache_quantization": "q8_0",
    "batch_size": 512
  }
}
```

## API Endpoints

### OpenAI-kompatibel

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/v1/models` | GET | Liste verfügbarer Modelle |
| `/v1/chat/completions` | POST | Chat Completion (Streaming) |
| `/v1/embeddings` | POST | Text Embeddings |
| `/v1/audio/transcriptions` | POST | Audio Transkription |
| `/v1/audio/speech` | POST | Text-to-Speech |

### Live Voice API

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/v1/audio/live` | POST | Session erstellen |
| `/v1/audio/live/ws/{token}` | WS | WebSocket Verbindung |
| `/v1/audio/live/{token}` | GET | Session Info |
| `/v1/audio/live/{token}` | DELETE | Session beenden |

### Admin Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/admin/api/slots` | GET | Slot-Status |
| `/admin/api/slots/load` | POST | Modell laden |
| `/admin/api/slots/{slot}/unload` | POST | Modell entladen |
| `/admin/api/models/local` | GET | Lokale Modelle |
| `/admin/api/models/search` | GET | HuggingFace suchen |
| `/admin/api/models/download` | POST | Modell herunterladen |
| `/admin/api/users` | GET/POST | User-Verwaltung |
| `/admin/api/system` | GET | System-Stats |

### User Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/user/api/me` | GET | User-Info |
| `/user/api/usage` | GET | Usage-Statistiken |
| `/user/api/models` | GET | Verfügbare Modelle |
| `/user/api/ratelimit` | GET | Rate-Limit Status |

### Public Endpoints (ohne Auth)

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/health` | GET | Health Check |
| `/api/models` | GET | Modell-Liste |
| `/api/uptime` | GET | Uptime-Historie |
| `/api/signup` | POST | Signup-Request |

## Web Interfaces

- **Landing Page**: `http://localhost:4000/`
- **Admin Panel**: `http://localhost:4000/admin/`
- **User Dashboard**: `http://localhost:4000/user/`
- **Playground**: `http://localhost:4000/playground/`
- **Live Voice**: `http://localhost:4000/live`

## Verwendung

### Server starten
```bash
# Mit venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

uvicorn server:app --host 0.0.0.0 --port 4000

# Mit ToolBoxV2
tb llm-gateway start
```

### Modell laden (Admin)
```bash
curl -X POST http://localhost:4000/admin/api/slots/load \
  -H "Authorization: Bearer sk-admin-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "slot": 4801,
    "model_path": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "model_type": "text"
  }'
```

### Chat Completion
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-xxx"
)

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Gateway (Port 4000)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   FastAPI   │  │ Rate Limit  │  │   Auth (API Keys)   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │            │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐ │
│  │                    Smart Router                         │ │
│  │  (Model Selection based on capabilities)                │ │
│  └─────────────────────────┬───────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
│  4801   │  │  4802   │  │  4803   │  │  ...    │
│  text   │  │ vision  │  │  omni   │  │         │
│ llama-  │  │ llama-  │  │ llama-  │  │         │
│ server  │  │ server  │  │ server  │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

## Verzeichnisstruktur

```
llm-gateway/
├── server.py           # Haupt-Server (FastAPI)
├── model_manager.py    # Modell-Verwaltung
├── live_handler.py     # Live Voice API
├── main.py             # Entry Point
├── requirements.txt    # Python Dependencies
├── setup.sh            # Linux Setup
├── win_setup.ps1       # Windows Setup
├── data/
│   ├── config.json     # Konfiguration
│   ├── gateway.db      # SQLite Datenbank
│   └── models/         # GGUF Modelle
├── static/
│   ├── index.html      # Landing Page
│   ├── admin.html      # Admin Panel
│   ├── user.html       # User Dashboard
│   ├── playground.html # Chat Playground
│   └── live-playground.html  # Voice Playground
└── build/              # llama.cpp Build
```

## Performance-Optimierungen

- **Flash Attention**: 20-30% Speedup
- **mlock**: Verhindert Swapping
- **KV-Cache Quantization**: Spart RAM bei großem Context
- **Continuous Batching**: Parallele Request-Verarbeitung

## Troubleshooting

### Modell lädt nicht
- Prüfe RAM-Verfügbarkeit (`/admin/api/system`)
- Erhöhe Timeout für große Modelle
- Prüfe mmproj für Vision-Modelle

### Rate Limit erreicht
- Warte 60 Sekunden
- Upgrade auf höheren Tier
- Admin hat kein Limit

### WebSocket Verbindung fehlgeschlagen
- Prüfe Session-Token Gültigkeit
- Session läuft nach 15-20 Minuten ab

## Lizenz

Teil des ToolBoxV2 Projekts.
