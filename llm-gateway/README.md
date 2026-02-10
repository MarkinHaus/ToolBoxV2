# LLM Gateway

OpenAI-compatible API gateway powered by **Ollama**. Provides user management, API key auth, usage tracking, rate limiting, and a web UI — all self-hosted.

## Architecture

```
Clients (OpenAI SDK, curl, etc.)
        │
        ▼
┌──────────────────────┐
│   LLM Gateway :4000  │  FastAPI server
│   (OpenAI-compat API)│  Auth, rate limiting, usage tracking
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Ollama :11434      │  Native or Docker
│   (Model runtime)    │  Text, Vision, Embedding, Audio, TTS
└──────────────────────┘
```

## Features

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech`, `/v1/audio/transcriptions`
- **Tool calling** — Full function calling support via Ollama
- **Multi-modal** — Text, vision (llava, qwen-vl), audio, embeddings, TTS
- **Smart routing** — Auto-detects content type (image/audio) and routes to capable model
- **Admin UI** — Load/unload models, manage users, monitor system
- **Playground** — Chat with streaming, image upload, audio recording
- **Live Voice** — Real-time voice conversation via WebSocket
- **User management** — API keys, tiers (PAYG/subscription), balance tracking
- **HuggingFace integration** — Search and download GGUF models, import into Ollama
- **Two deployment modes** — Bare metal (native Ollama) or Docker
- **Cross-platform** — Linux + Windows, with systemd service support

## Quick Start

### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download)

### 2. Setup Gateway

```bash
cd llm-gateway

# Linux/macOS
bash setup.sh

# Windows
powershell -ExecutionPolicy Bypass -File win_setup.ps1
```

### 3. Start

```bash
# Activate venv
source venv/bin/activate  # Linux
# or: venv\Scripts\activate  # Windows

# Start server
uvicorn server:app --host 0.0.0.0 --port 4000
```

### 4. Open Admin UI

Navigate to `http://localhost:4000/admin/` and login with admin key from `data/config.json`.

Pull and load a model:
```
Model name: llama3.2
Type: auto
→ Load
```

### 5. Use the API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-admin-..."  # your API key
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Docker Deployment

### Gateway only (Ollama on host)

```bash
# Ollama running natively on the host
OLLAMA_URL=http://host.docker.internal:11434 docker compose up gateway
```

### Full stack (Gateway + Ollama in Docker)

```bash
docker compose --profile ollama up
```

### Environment Variables

Override config values via environment variables (useful for Docker/CI):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://ollama:11434` (Docker) | Ollama server URL |
| `ADMIN_KEY` | from `config.json` | Admin API key |

### GPU Support

Uncomment in `compose.yaml`:
```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

## Configuration

`data/config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_url` | `http://127.0.0.1:11434` | Ollama server URL |
| `backend_mode` | `bare` | `bare` (native) or `docker` |
| `admin_key` | generated | Admin API key |
| `hf_token` | null | HuggingFace token for private models |
| `pricing.input_per_1k` | 0.0001 | Input token price |
| `pricing.output_per_1k` | 0.0002 | Output token price |
| `rate_limits.payg` | 5 | Requests/min for PAYG users |
| `rate_limits.sub` | 10 | Requests/min for subscribers |

## Model Types

| Type | Capabilities | Examples |
|------|-------------|----------|
| `text` | Chat, tool calling | llama3.2, qwen2.5, mistral |
| `vision` | Chat + image understanding | llava, qwen2-vl, bakllava |
| `omni` | Chat + image + audio | qwen2-audio |
| `embedding` | Text embeddings | nomic-embed-text, mxbai-embed-large |
| `tts` | Text-to-speech | kokoro |
| `audio` | Speech-to-text | whisper |

## API Endpoints

### OpenAI-Compatible
- `POST /v1/chat/completions` — Chat (streaming supported)
- `POST /v1/embeddings` — Embeddings
- `POST /v1/audio/speech` — TTS
- `POST /v1/audio/transcriptions` — STT
- `GET /v1/models` — List models

### Live Voice
- `POST /v1/audio/live` — Create session
- `WS /v1/audio/live/ws/{token}` — WebSocket
- `DELETE /v1/audio/live/{token}` — Close session

### Admin
- `GET/POST /admin/api/users` — User management
- `GET/POST /admin/api/models/*` — Model management
- `GET /admin/api/system` — System stats
- `GET/PATCH /admin/api/config` — Configuration

### Public
- `GET /health` — Health check
- `GET /api/models` — List models (no auth)
- `GET /api/uptime` — Uptime history
- `POST /api/signup` — Request access

## Web UIs

| URL | Description |
|-----|-------------|
| `/` | Landing page |
| `/admin/` | Admin panel |
| `/playground/` | Chat playground |
| `/live` | Live voice playground |
| `/user/` | User dashboard |

## ToolBoxV2 CLI Integration

```bash
tb llm-gateway setup     # Install dependencies + check Ollama
tb llm-gateway start     # Start gateway server
tb llm-gateway stop      # Stop gateway
tb llm-gateway status    # Show status
tb llm-gateway restart   # Restart
```

## Importing GGUF Models

Download GGUF from HuggingFace (via Admin UI or manually), then import:

```bash
# Via Admin UI: Models → Import GGUF
# Or via API:
curl -X POST http://localhost:4000/admin/api/models/import-gguf \
  -H "Authorization: Bearer sk-admin-..." \
  -H "Content-Type: application/json" \
  -d '{"gguf_path": "data/models/model.gguf", "model_name": "my-model"}'
```

## Linux Service

```bash
# Copy service file
sudo cp llm-gateway.service /etc/systemd/system/
# Edit paths and user
sudo systemctl daemon-reload
sudo systemctl enable llm-gateway
sudo systemctl start llm-gateway
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests (99 tests)
python -m unittest discover tests/

# Run specific test modules
python -m unittest tests.test_server          # 38 tests — API, auth, rate limiting
python -m unittest tests.test_model_manager   # 38 tests — model types, routing, Ollama API
python -m unittest tests.test_live_handler    # 23 tests — WebSocket, TTS, sessions

# Run server in dev mode
uvicorn server:app --reload --port 4000
```

## License

Part of the ToolBoxV2 project.
