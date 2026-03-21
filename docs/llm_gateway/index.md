# LLM Gateway

> **Location**: `llm-gateway/` (separate directory)
> **Default Port**: 4000
> **Backend**: Ollama (local model runtime)

OpenAI-compatible API gateway powered by **Ollama**. Provides user management, API key auth, usage tracking, rate limiting, and a web admin UI — fully self-hosted.

## Architecture

```
Clients (OpenAI SDK, curl, tbjs, etc.)
        │
        ▼
┌──────────────────────┐
│   LLM Gateway :4000  │  FastAPI · Auth · Rate Limiting · Usage Tracking
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Ollama :11434      │  Text · Vision · Embedding · Audio · TTS
└──────────────────────┘
```

## Features

- **OpenAI-compatible API** — drop-in replacement for `openai` SDK
- **Multi-modal** — text, vision (llava, qwen-vl), audio, embeddings, TTS
- **Smart routing** — auto-detects content type (image/audio) → routes to capable model
- **Live Voice** — real-time voice conversation via WebSocket
- **User management** — API keys, tiers (PAYG / subscription), balance tracking
- **HuggingFace GGUF import** — search and download models directly
- **Admin UI** — model management, user management, system monitoring
- **Playground** — chat with streaming, image upload, audio recording
- **Two deployment modes** — bare metal (native Ollama) or Docker

## Quick Start

### 1. Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from ollama.com/download
```

### 2. Setup Gateway

```bash
cd llm-gateway
bash setup.sh          # Linux/macOS
# or:
powershell -ExecutionPolicy Bypass -File win_setup.ps1   # Windows
```

### 3. Start

```bash
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 4000
```

### 4. Via ToolBoxV2

```bash
tb llm-gateway setup
tb llm-gateway start
tb llm-gateway status
tb llm-gateway stop
tb llm-gateway restart
```

## OpenAI-Compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-admin-..."   # from data/config.json
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Endpoints

### Chat / Completions

```
POST /v1/chat/completions          → Chat (streaming supported)
POST /v1/embeddings                → Embeddings
POST /v1/audio/speech              → TTS
POST /v1/audio/transcriptions      → STT
GET  /v1/models                    → List loaded models
```

### Live Voice (WebSocket)

```
POST /v1/audio/live                → Create session
WS   /v1/audio/live/ws/{token}    → Stream voice
DEL  /v1/audio/live/{token}       → Close session
```

### Admin

```
GET/POST /admin/api/users          → User management
GET/POST /admin/api/models/*       → Model load/unload
GET      /admin/api/system         → System stats
GET/PUT  /admin/api/config         → Configuration
```

## Model Types

| Type | Capabilities | Example Models |
|------|-------------|----------------|
| `text` | Chat + tool calling | llama3.2, qwen2.5, mistral |
| `vision` | Chat + images | llava, qwen2-vl, bakllava |
| `omni` | Chat + image + audio | qwen2-audio |
| `embedding` | Text vectors | nomic-embed-text, mxbai-embed-large |
| `tts` | Text-to-speech | kokoro |
| `audio` | Speech-to-text | whisper |

## Configuration (`data/config.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_url` | `http://127.0.0.1:11434` | Ollama server URL |
| `backend_mode` | `bare` | `bare` (native) or `docker` |
| `admin_key` | generated | Admin API key |
| `hf_token` | null | HuggingFace token |
| `rate_limits.payg` | 5 | Req/min for PAYG users |
| `rate_limits.sub` | 10 | Req/min for subscribers |

## Docker

```bash
# Gateway only (Ollama on host)
OLLAMA_URL=http://host.docker.internal:11434 docker compose up gateway

# Full stack (Gateway + Ollama)
docker compose --profile ollama up
```

GPU support: uncomment `deploy.resources.reservations.devices` in `compose.yaml`.

## Web UIs

| URL | Description |
|-----|-------------|
| `/` | Landing page |
| `/admin/` | Admin panel |
| `/playground/` | Chat playground |
| `/live` | Live voice |
| `/user/` | User dashboard |

## Testing

```bash
# 99 tests total
python -m unittest discover tests/

# Individual modules
python -m unittest tests.test_server          # 38 tests
python -m unittest tests.test_model_manager   # 38 tests
python -m unittest tests.test_live_handler    # 23 tests
```

## Related

- [ISAA Agent Framework](../mods/isaa/README.md) — Uses LLM Gateway as a backend provider
- [System Stack](../new/analysis/installation/stack.md) — How gateway fits into the overall architecture
