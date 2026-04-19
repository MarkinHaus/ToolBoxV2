# icli_web

FastAPI server + icli-side WebSocket client. Browser ↔ server ↔ icli.

```
Browser ──WS──► server.py ◄──WS── icli (client.py wrapping run_agent_for_web)
```

## Layout

```
toolboxv2/mods/icli_web/
├── server.py            ← FastAPI router (run standalone)
├── client.py            ← icli-side WS client
├── _icli_patch.py       ← run_agent_for_web to paste into ICli
├── __init__.py          ← mod registration
├── web/
│   ├── orb.html
│   └── monitor.html
└── test_icli_web.py
```

## Setup

```bash
pip install fastapi uvicorn websockets
```

### 1. Start the server

```bash
python -m toolboxv2.mods.icli_web.server
```

Logs the URL including API key on first run (key stored at
`~/.toolbox/icli_web.key`).

### 2. Wire icli — two edits to `ICli`

**In `ICli.__init__`** (after `self._task_views = {}` is ready):

```python
from toolboxv2.mods.icli_web.client import IcliWebClient
IcliWebClient.get().attach(self)
```

**As a new method on `ICli`** — copy from `_icli_patch.py`:

```python
async def run_agent_for_web(self, agent_name: str, query: str):
    # ... see _icli_patch.py for full body
```

The method uses your existing `_create_execution(kind="web", ...)` factory,
so web queries appear in Zen+ / monitor just like chat queries. Focus is
not taken (terminal stays interactive), and TTS is handled web-side.

### 3. Open the orb

```
http://127.0.0.1:5055/?key=<KEY>
```

## Reconnection behavior

Both client (icli-side) and orb (browser) reconnect automatically.

**Client (icli → server):**
- Exponential backoff: 1s → 1.5s → 2.25s → … capped at 30s
- ±20% jitter to avoid thundering herd
- Backoff resets to 1s after staying connected >30s
- Server's `ping_interval=20s` / `ping_timeout=10s` (websockets library)
- On reconnect: fresh `hello` with capabilities, all in-flight sessions
  cancelled cleanly (agent tasks cancelled, TTS players stopped)

**Orb (browser → server):**
- Same backoff strategy, client-side JS
- Server sends `{type:"ping"}` every 15s; orb replies with `pong`
- Recv timeout on server: 45s — closes socket if orb is silent that long
- Auth failure (code 4401) stops retries

**Server-side disconnects:**
- If icli drops: all orbs get `{type:"error",cid}` + `{type:"done",cid}`
  for their in-flight queries, then `{type:"status", icli_connected:false}`
- If an orb drops: server sends `{type:"cancel",cid}` to icli for each of
  that orb's active cids; client cancels the agent tasks

**ExecutionTask cleanup:**
- `run_agent_for_web` wraps the stream in a tee; on browser disconnect,
  `CancelledError` propagates → the drain task is cancelled → the
  underlying agent stream aborts → `_on_agent_task_done` runs as usual

## Message protocol

### Orb → server

```json
{"action": "query", "query": "hello", "agent": "self", "tts": {...}, "stt": {...}}
{"action": "audio_start", "agent": "self", "tts": {...}, "stt": {...}}
<binary audio chunks>
{"action": "audio_end"}
{"action": "stop_tts"}
{"type": "ping"} / {"type": "pong"}
```

### Server → orb

```json
{"type": "status",        "icli_connected": true}
{"type": "cid_assigned",  "cid": "abc123"}
{"type": "transcription", "cid": "abc", "text": "hello"}
{"type": "text_chunk",    "cid": "abc", "text": "Hi "}
{"type": "reasoning",     "cid": "abc", "text": "Let me think…"}
{"type": "tool_start",    "cid": "abc", "name": "search"}
{"type": "tool_result",   "cid": "abc", "name": "search", "success": true, "info": "…"}
{"type": "narrator",      "cid": "abc", "text": "searching web"}
{"type": "response",      "cid": "abc", "text": "Hi there!"}
{"type": "audio",         "cid": "abc", "text": "Hi there!", "chunk_index": 0}
<binary WAV frame>
{"type": "done",          "cid": "abc"}
{"type": "error",         "cid": "abc", "error": "..."}
{"type": "ping"}
```

### icli → server

```json
{"type": "hello", "capabilities": {...}}
{"type": "task",  "data": {...}}
{"type": "text_chunk" | "reasoning" | "tool_start" | "tool_result" |
         "narrator" | "response" | "audio" | "done" | "error", "cid": "...", ...}
<binary WAV frame after audio meta>
```

### server → icli

```json
{"type": "query",         "cid": "abc", "agent": "self", "query": "...",
 "tts": {...}, "stt": {...}}
{"type": "audio_start",   "cid": "abc", ...}
{"type": "audio_chunk_in", "cid": "abc"}   <then binary follows>
{"type": "audio_end",     "cid": "abc"}
{"type": "stop_tts",      "cid": "abc"}
{"type": "cancel",        "cid": "abc"}   # orb disconnected
```

## Orb transcript shows

- `> user`: transcribed speech / typed queries
- `$ agent`: streamed text + final answer
- `sys`: reasoning (`› thought…`), tool calls (`→ tool_name`), tool
  results (`● tool_name — info` or `✕ tool_name — error`), narrator
  messages, errors

## Environment

| Var | Default | Purpose |
|---|---|---|
| `ICLI_WEB_HOST` | `127.0.0.1` | Bind / target host |
| `ICLI_WEB_PORT` | `5055` | Port |
| `ICLI_WEB_API_KEY` | auto-gen | Overrides key file |
| `ICLI_WEB_LANG` | `de` | Default TTS/STT language |

## Nginx

```nginx
location / {
    proxy_pass http://127.0.0.1:5055;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_buffering off;
    proxy_read_timeout 3600s;
}
```

## Qwen3-TTS

When you add the backend to `Tts.py`:

1. `QWEN3_TTS = "qwen3_tts"` in `TTSBackend`
2. `style_prompt: Optional[str] = None` in `TTSConfig`
3. `_synthesize_qwen3_tts()` reads `config.style_prompt`

The orb style-prompt field auto-appears — `client.py` already reports
`supports_style_prompt = ["qwen3_tts"]` in the capabilities hello.
