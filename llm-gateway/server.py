"""
LLM Gateway - OpenAI Compatible API Server (Ollama Backend)
Gateway port: 4000 | Ollama: 11434 (native or Docker)
"""

import asyncio
import hashlib
import json
import os
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import aiosqlite
import httpx
import psutil
from fastapi import (
    FastAPI, HTTPException, Header, Request, Depends,
    BackgroundTasks, WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_manager import ModelManager, detect_model_type
from live_handler import (
    LiveHandler, LiveSessionRequest, LiveSessionResponse,
    create_live_handler,
)

# === Config ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "gateway.db"
CONFIG_PATH = DATA_DIR / "config.json"
STATIC_DIR = BASE_DIR / "static"

DATA_DIR.mkdir(exist_ok=True)


# === Rate Limiter ===

class RateLimiter:
    """Simple in-memory rate limiter per user."""

    def __init__(self):
        self.requests: Dict[int, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: int, tier: str, config: Dict) -> bool:
        now = time.time()
        window = 60
        rate_limits = config.get("rate_limits", {"payg": 5, "sub": 10})
        max_requests = rate_limits.get(tier, 5)
        if tier == "admin":
            return True
        self.requests[user_id] = [ts for ts in self.requests[user_id] if now - ts < window]
        if len(self.requests[user_id]) >= max_requests:
            return False
        self.requests[user_id].append(now)
        return True

    def get_remaining(self, user_id: int, tier: str, config: Dict) -> int:
        now = time.time()
        window = 60
        rate_limits = config.get("rate_limits", {"payg": 5, "sub": 10})
        max_requests = rate_limits.get(tier, 5)
        if tier == "admin":
            return 999
        recent = [ts for ts in self.requests[user_id] if now - ts < window]
        return max(0, max_requests - len(recent))


rate_limiter = RateLimiter()


# === Pydantic Models ===

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list for vision

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    n: Optional[int] = 1
    response_format: Optional[Dict] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: Any  # str or list
    encoding_format: Optional[str] = "float"

class UserCreate(BaseModel):
    email: str
    tier: str = "payg"

class UserUpdate(BaseModel):
    balance: Optional[float] = None
    tier: Optional[str] = None
    active: Optional[bool] = None

class ApiKeyCreate(BaseModel):
    user_id: int
    name: str = "default"

class ModelLoadRequest(BaseModel):
    model_name: str
    model_type: str = "auto"
    keep_alive: str = "-1"

class ModelImportRequest(BaseModel):
    gguf_path: str
    model_name: str

class DownloadRequest(BaseModel):
    repo_id: str
    filename: str

class DeleteModelRequest(BaseModel):
    model_name: str

class SignupRequest(BaseModel):
    email: str
    tier: str = "payg"
    message: Optional[str] = None

class TTSRequest(BaseModel):
    model: str = "tts"
    input: str
    voice: str = "default"
    speed: float = 1.0
    response_format: str = "opus"


# === Database ===

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                balance REAL DEFAULT 0.0,
                tier TEXT DEFAULT 'payg',
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT DEFAULT 'default',
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY,
                api_key_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                type TEXT NOT NULL,
                note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS signup_requests (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                tier TEXT DEFAULT 'payg',
                message TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS uptime_checks (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                active_models INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_uptime_created ON uptime_checks(created_at);
        """)
        await db.commit()

        # Create admin user if not exists
        cursor = await db.execute("SELECT id FROM users WHERE email = 'admin'")
        if not await cursor.fetchone():
            await db.execute(
                "INSERT INTO users (email, balance, tier) VALUES (?, ?, ?)",
                ("admin", 999999.0, "admin"),
            )
            await db.commit()

        config = load_config()
        admin_key = config.get("admin_key", "sk-admin-change-me-on-first-run")
        admin_key_hash = hash_key(admin_key)

        cursor = await db.execute("SELECT id FROM users WHERE email = 'admin'")
        admin_row = await cursor.fetchone()
        admin_id = admin_row[0]

        cursor = await db.execute(
            "SELECT id FROM api_keys WHERE key_hash = ?", (admin_key_hash,)
        )
        if not await cursor.fetchone():
            await db.execute(
                "DELETE FROM api_keys WHERE user_id = ? AND name = 'admin-key'",
                (admin_id,),
            )
            await db.execute(
                "INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES (?, ?, ?, ?)",
                (admin_id, admin_key_hash, admin_key[:10], "admin-key"),
            )
            await db.commit()


# === Auth ===

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(authorization: Optional[str] = Header(None)) -> Dict:
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Authorization format")

    key = authorization[7:]
    key_hash = hash_key(key)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT ak.id as key_id, ak.user_id, u.email, u.balance, u.tier, u.active
            FROM api_keys ak
            JOIN users u ON ak.user_id = u.id
            WHERE ak.key_hash = ? AND ak.active = 1
        """, (key_hash,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(401, "Invalid API key")
        if not row["active"]:
            raise HTTPException(403, "User account disabled")
        return dict(row)


async def verify_admin(auth: Dict = Depends(verify_api_key)) -> Dict:
    if auth["tier"] != "admin":
        raise HTTPException(403, "Admin access required")
    return auth


# === Config ===

def load_config() -> Dict:
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
        config["_config_path"] = str(CONFIG_PATH)
    else:
        config = {
            "ollama_url": "http://127.0.0.1:11434",
            "backend_mode": "bare",  # "bare" or "docker"
            "hf_token": None,
            "admin_key": "sk-admin-change-me-on-first-run",
            "loaded_models": {},
            "pricing": {"input_per_1k": 0.0001, "output_per_1k": 0.0002},
            "rate_limits": {"payg": 5, "sub": 10},
            "_config_path": str(CONFIG_PATH),
        }
    # Environment variables override config file (for Docker)
    if os.environ.get("OLLAMA_URL"):
        config["ollama_url"] = os.environ["OLLAMA_URL"]
    if os.environ.get("ADMIN_KEY"):
        config["admin_key"] = os.environ["ADMIN_KEY"]
    return config


def save_config(config: Dict):
    c = {k: v for k, v in config.items() if not k.startswith("_")}
    CONFIG_PATH.write_text(json.dumps(c, indent=2))


# === Global instances ===
model_manager: ModelManager = None
live_handler: LiveHandler = None
uptime_task: asyncio.Task = None


async def uptime_checker():
    while True:
        try:
            start = time.time()
            status = "up"
            active_models = 0
            if model_manager:
                active_models = len(model_manager.get_active_models())
                if not await model_manager._ollama_health():
                    status = "degraded"
            latency = int((time.time() - start) * 1000)
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO uptime_checks (status, active_models, latency_ms) VALUES (?, ?, ?)",
                    (status, active_models, latency),
                )
                await db.execute(
                    "DELETE FROM uptime_checks WHERE datetime(created_at) < datetime('now', '-30 days')"
                )
                await db.commit()
        except Exception as e:
            print(f"Uptime check error: {e}")
            try:
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute(
                        "INSERT INTO uptime_checks (status, active_models, latency_ms) VALUES (?, ?, ?)",
                        ("down", 0, 0),
                    )
                    await db.commit()
            except Exception:
                pass
        await asyncio.sleep(300)


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, live_handler, uptime_task
    await init_db()
    config = load_config()
    model_manager = ModelManager(config=config)
    live_handler = create_live_handler(str(DB_PATH), model_manager)
    await live_handler.init_db()

    async def restore_in_background():
        try:
            print("Restoring models from config...")
            await model_manager.restore_models()
            print("Models restored successfully")
        except Exception as e:
            print(f"Warning: Failed to restore some models: {e}")

    asyncio.create_task(restore_in_background())
    uptime_task = asyncio.create_task(uptime_checker())

    async with aiosqlite.connect(DB_PATH) as db:
        active = len(model_manager.get_active_models())
        await db.execute(
            "INSERT INTO uptime_checks (status, active_models, latency_ms) VALUES (?, ?, ?)",
            ("up", active, 0),
        )
        await db.commit()

    print("LLM Gateway started successfully! (Ollama backend)")
    yield

    if uptime_task:
        uptime_task.cancel()
        try:
            await uptime_task
        except asyncio.CancelledError:
            pass
    await model_manager.shutdown_all()


# === App ===

app = FastAPI(title="LLM Gateway", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Content Detection ===

def detect_content_requirements(messages: List[ChatMessage]) -> tuple:
    needs_audio = False
    needs_vision = False
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    pt = part.get("type", "")
                    if pt == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if "data:audio/" in url:
                            needs_audio = True
                        else:
                            needs_vision = True
                    if pt == "image":
                        needs_vision = True
                    if pt in ("audio", "input_audio", "audio_url"):
                        needs_audio = True
        elif isinstance(content, str):
            if "data:image/" in content:
                needs_vision = True
            if "data:audio/" in content:
                needs_audio = True
    return needs_audio, needs_vision


# === OpenAI Compatible Endpoints ===

@app.get("/v1/models")
async def list_models(auth: Dict = Depends(verify_api_key)):
    models = model_manager.get_active_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": m["name"],
                "parent": None,
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key),
):
    config = load_config()
    if not rate_limiter.is_allowed(auth["user_id"], auth["tier"], config):
        raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")

    if auth["tier"] == "payg" and auth["balance"] <= 0:
        raise HTTPException(402, "Insufficient balance")

    needs_audio, needs_vision = detect_content_requirements(request.messages)
    model_info = model_manager.find_model_for_request(
        model_name=request.model,
        needs_audio=needs_audio,
        needs_vision=needs_vision,
    )
    if not model_info:
        if needs_audio:
            raise HTTPException(404, "No model with audio capability loaded. Load an 'omni' model.")
        elif needs_vision:
            raise HTTPException(404, "No model with vision capability loaded. Load a 'vision' or 'omni' model.")
        else:
            raise HTTPException(404, f"Model '{request.model}' not loaded")

    # Proxy to Ollama OpenAI-compatible endpoint
    ollama_url = f"{model_manager.ollama_url}/v1/chat/completions"
    payload = {
        "model": model_info["name"],
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream,
        "n": request.n,
    }
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens
    if request.tools:
        payload["tools"] = request.tools
    if request.tool_choice:
        payload["tool_choice"] = request.tool_choice
    if request.stop:
        payload["stop"] = request.stop
    if request.response_format:
        payload["response_format"] = request.response_format

    start_time = time.time()

    if request.stream:
        return StreamingResponse(
            stream_chat_response(ollama_url, payload, auth, start_time, model_info["name"], background_tasks),
            media_type="text/event-stream",
        )
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(ollama_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency = int((time.time() - start_time) * 1000)
                tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
                tokens_out = data.get("usage", {}).get("completion_tokens", 0)
                background_tasks.add_task(
                    log_usage, auth["key_id"], model_info["name"], tokens_in, tokens_out, latency,
                )
                return data
            except httpx.HTTPStatusError as e:
                raise HTTPException(e.response.status_code, str(e))
            except Exception as e:
                raise HTTPException(500, str(e))


async def stream_chat_response(
    url: str,
    payload: Dict,
    auth: Dict,
    start_time: float,
    model: str,
    background_tasks: BackgroundTasks,
) -> AsyncGenerator[str, None]:
    tokens_out = 0
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                        if line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                    tokens_out += 1
                            except Exception:
                                pass
                    elif line:
                        yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
            latency = int((time.time() - start_time) * 1000)
            background_tasks.add_task(log_usage, auth["key_id"], model, 0, tokens_out, latency)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key),
):
    model_info = model_manager.find_model_for_request(
        model_name=request.model, needs_embedding=True,
    )
    if not model_info:
        model_info = model_manager.find_embedding_model()
    if not model_info:
        model_info = model_manager.find_text_model()
    if not model_info:
        raise HTTPException(404, "No embedding model loaded.")

    ollama_url = f"{model_manager.ollama_url}/v1/embeddings"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(ollama_url, json={
                "model": model_info["name"],
                "input": request.input,
                "encoding_format": request.encoding_format,
            })
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"Backend error: {e.response.text}")
        except Exception as e:
            raise HTTPException(500, str(e))


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key),
):
    model_info = model_manager.find_audio_model()
    if not model_info:
        raise HTTPException(
            404,
            "No audio model loaded. Use /v1/chat/completions with an 'omni' model for audio processing.",
        )
    # Forward multipart form data to Ollama (if it supports whisper-style endpoint)
    form = await request.form()
    files = {}
    data = {}
    for key, value in form.items():
        if hasattr(value, "read"):
            files[key] = (value.filename, await value.read(), value.content_type)
        else:
            data[key] = value

    ollama_url = f"{model_manager.ollama_url}/v1/audio/transcriptions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(ollama_url, files=files, data=data)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"Backend error: {e.response.text}")
        except Exception as e:
            raise HTTPException(500, str(e))


# === TTS ===

@app.post("/v1/audio/speech")
async def text_to_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key),
):
    tts_model = model_manager.find_tts_model()
    if not tts_model:
        raise HTTPException(503, "No TTS model loaded.")

    ollama_url = f"{model_manager.ollama_url}/v1/audio/speech"
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            payload = {
                "model": tts_model["name"],
                "input": request.input,
                "voice": request.voice,
                "speed": request.speed,
                "response_format": request.response_format,
            }
            resp = await client.post(ollama_url, json=payload)
            resp.raise_for_status()
            content_type = {
                "opus": "audio/opus", "mp3": "audio/mpeg",
                "wav": "audio/wav", "pcm": "audio/pcm",
            }.get(request.response_format, "audio/opus")
            return StreamingResponse(
                iter([resp.content]),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"},
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"TTS error: {e.response.text}")
        except Exception as e:
            raise HTTPException(500, str(e))


# === Live Voice API ===

@app.post("/v1/audio/live", response_model=LiveSessionResponse)
async def create_live_session(
    request: LiveSessionRequest,
    auth: Dict = Depends(verify_api_key),
):
    return await live_handler.create_session(request, auth["user_id"])


@app.websocket("/v1/audio/live/ws/{session_token}")
async def live_websocket(websocket: WebSocket, session_token: str):
    await live_handler.handle_websocket(websocket, session_token)


@app.delete("/v1/audio/live/{session_token}")
async def close_live_session(session_token: str, auth: Dict = Depends(verify_api_key)):
    session = await live_handler.get_session(session_token)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.user_id != auth["user_id"] and auth["tier"] != "admin":
        raise HTTPException(403, "Not authorized")
    await live_handler.close_session(session_token)
    return {"status": "closed", "session_token": session_token}


@app.get("/v1/audio/live/{session_token}")
async def get_live_session_info(session_token: str, auth: Dict = Depends(verify_api_key)):
    session = await live_handler.get_session(session_token)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.user_id != auth["user_id"] and auth["tier"] != "admin":
        raise HTTPException(403, "Not authorized")
    return {
        "session_token": session.token,
        "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
        "expires_at": datetime.fromtimestamp(session.expires_at).isoformat(),
        "is_active": session.is_active,
        "history_length": len(session.history),
        "config": {
            "audio": session.audio_config.model_dump(),
            "wake_word": session.wake_word_config.model_dump(),
            "voice": session.voice_config.model_dump(),
            "llm": {"model": session.llm_config.model, "history_length": session.llm_config.history_length},
        },
    }


# === Usage Logging ===

async def log_usage(key_id: int, model: str, tokens_in: int, tokens_out: int, latency_ms: int):
    config = load_config()
    pricing = config.get("pricing", {})
    cost = (
        tokens_in / 1000 * pricing.get("input_per_1k", 0.0001)
        + tokens_out / 1000 * pricing.get("output_per_1k", 0.0002)
    )
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO usage (api_key_id, model, tokens_in, tokens_out, cost, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (key_id, model, tokens_in, tokens_out, cost, latency_ms),
        )
        await db.execute(
            "UPDATE users SET balance = balance - ? WHERE id = (SELECT user_id FROM api_keys WHERE id = ?) AND tier = 'payg'",
            (cost, key_id),
        )
        await db.commit()


# === Stats ===

@app.get("/v1/stats/summary")
async def stats_summary(auth: Dict = Depends(verify_api_key)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT COUNT(DISTINCT api_key_id) as unique_keys, COUNT(*) as total_requests,
                   SUM(tokens_in + tokens_out) as total_tokens, AVG(latency_ms) as avg_latency,
                   SUM(cost) as total_cost
            FROM usage WHERE datetime(created_at) > datetime('now', '-24 hours')
        """)
        stats = dict(await cursor.fetchone())
        cursor = await db.execute("SELECT COUNT(*) as new_users FROM users WHERE date(created_at) = date('now')")
        new_users = (await cursor.fetchone())[0]
        active_models = model_manager.get_active_models()
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        return {
            "period": "24h",
            "summary": f"{new_users} neue User, {stats['total_requests'] or 0} Anfragen, "
                       f"{stats['total_tokens'] or 0} Tokens, Ø {int(stats['avg_latency'] or 0)}ms",
            "stats": {
                "new_users": new_users,
                "total_requests": stats["total_requests"] or 0,
                "total_tokens": stats["total_tokens"] or 0,
                "avg_latency_ms": int(stats["avg_latency"] or 0),
                "total_cost": round(stats["total_cost"] or 0, 4),
            },
            "system": {
                "cpu_percent": cpu,
                "ram_used_gb": round(mem.used / (1024**3), 1),
                "ram_total_gb": round(mem.total / (1024**3), 1),
                "ram_percent": mem.percent,
            },
            "active_models": len(active_models),
            "models": active_models,
        }


# === Admin Endpoints ===

@app.get("/admin/", response_class=HTMLResponse)
async def admin_panel():
    return (STATIC_DIR / "admin.html").read_text(encoding="utf-8")


@app.get("/admin/api/users")
async def admin_list_users(auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT u.*, (SELECT COUNT(*) FROM api_keys WHERE user_id = u.id) as key_count,
                   (SELECT SUM(cost) FROM usage WHERE api_key_id IN (SELECT id FROM api_keys WHERE user_id = u.id)) as total_usage
            FROM users u ORDER BY created_at DESC
        """)
        return [dict(row) for row in await cursor.fetchall()]


@app.post("/admin/api/users")
async def admin_create_user(user: UserCreate, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            cursor = await db.execute(
                "INSERT INTO users (email, tier) VALUES (?, ?) RETURNING id",
                (user.email, user.tier),
            )
            row = await cursor.fetchone()
            await db.commit()
            return {"id": row[0], "email": user.email}
        except aiosqlite.IntegrityError:
            raise HTTPException(400, "Email already exists")


@app.patch("/admin/api/users/{user_id}")
async def admin_update_user(user_id: int, update: UserUpdate, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        updates = []
        values = []
        if update.balance is not None:
            updates.append("balance = ?")
            values.append(update.balance)
            await db.execute(
                "INSERT INTO transactions (user_id, amount, type, note) VALUES (?, ?, ?, ?)",
                (user_id, update.balance, "admin_topup", "Admin balance update"),
            )
        if update.tier is not None:
            updates.append("tier = ?")
            values.append(update.tier)
        if update.active is not None:
            updates.append("active = ?")
            values.append(1 if update.active else 0)
        if updates:
            values.append(user_id)
            await db.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", values)
            await db.commit()
        return {"status": "updated"}


@app.post("/admin/api/users/{user_id}/apikey")
async def admin_create_apikey(user_id: int, data: ApiKeyCreate, auth: Dict = Depends(verify_admin)):
    key = f"sk-{secrets.token_hex(24)}"
    key_hash = hash_key(key)
    key_prefix = key[:8]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES (?, ?, ?, ?)",
            (user_id, key_hash, key_prefix, data.name),
        )
        await db.commit()
    return {"key": key, "prefix": key_prefix}


@app.delete("/admin/api/apikeys/{key_id}")
async def admin_delete_apikey(key_id: int, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))
        await db.commit()
    return {"status": "deleted"}


# === Model Management (Admin) ===

@app.get("/admin/api/models")
async def admin_get_models(auth: Dict = Depends(verify_admin)):
    return model_manager.get_models_status()


@app.get("/admin/api/models/ollama")
async def admin_list_ollama_models(auth: Dict = Depends(verify_admin)):
    return await model_manager.list_ollama_models()


@app.get("/admin/api/models/running")
async def admin_list_running(auth: Dict = Depends(verify_admin)):
    return await model_manager.list_running_models()


@app.post("/admin/api/models/load")
async def admin_load_model(request: ModelLoadRequest, auth: Dict = Depends(verify_admin)):
    return await model_manager.load_model(
        model_name=request.model_name,
        model_type=request.model_type,
        keep_alive=request.keep_alive,
    )


@app.post("/admin/api/models/pull")
async def admin_pull_model(request: ModelLoadRequest, auth: Dict = Depends(verify_admin)):
    return await model_manager.pull_model(request.model_name)


@app.post("/admin/api/models/{model_name}/unload")
async def admin_unload_model(model_name: str, auth: Dict = Depends(verify_admin)):
    return await model_manager.unload_model(model_name)


@app.post("/admin/api/models/delete")
async def admin_delete_model(request: DeleteModelRequest, auth: Dict = Depends(verify_admin)):
    return await model_manager.delete_model(request.model_name)


@app.post("/admin/api/models/import-gguf")
async def admin_import_gguf(request: ModelImportRequest, auth: Dict = Depends(verify_admin)):
    return await model_manager.import_gguf(request.gguf_path, request.model_name)


@app.get("/admin/api/models/local")
async def admin_list_local_models(auth: Dict = Depends(verify_admin)):
    return model_manager.list_local_gguf()


@app.get("/admin/api/models/search-hf")
async def admin_search_hf(q: str, auth: Dict = Depends(verify_admin)):
    return await model_manager.search_hf_models(q)


@app.post("/admin/api/models/download-hf")
async def admin_download_hf(request: DownloadRequest, auth: Dict = Depends(verify_admin)):
    return await model_manager.download_hf_model(request.repo_id, request.filename)


# === System Stats (Admin) ===

@app.get("/admin/api/system")
async def admin_system_stats(auth: Dict = Depends(verify_admin)):
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
    ollama_healthy = await model_manager._ollama_health()
    running = await model_manager.list_running_models()
    return {
        "cpu": {
            "percent_per_core": cpu_percent,
            "total_percent": sum(cpu_percent) / max(len(cpu_percent), 1),
            "cores": len(cpu_percent),
        },
        "memory": {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        },
        "ollama": {
            "url": model_manager.ollama_url,
            "healthy": ollama_healthy,
            "running_models": running,
        },
        "loaded_models": model_manager.get_models_status(),
        "backend_mode": model_manager.config.get("backend_mode", "bare"),
    }


@app.get("/admin/api/config")
async def admin_get_config(auth: Dict = Depends(verify_admin)):
    config = load_config()
    safe = {k: v for k, v in config.items() if not k.startswith("_")}
    if safe.get("hf_token"):
        safe["hf_token"] = safe["hf_token"][:8] + "..."
    if safe.get("admin_key"):
        key = safe["admin_key"]
        safe["admin_key_display"] = key[:12] + "..." + key[-4:]
    return safe


@app.patch("/admin/api/config")
async def admin_update_config(updates: Dict, auth: Dict = Depends(verify_admin)):
    config = load_config()
    # Merge non-internal keys
    for k, v in updates.items():
        if not k.startswith("_"):
            config[k] = v
    save_config(config)
    model_manager.config = config
    model_manager.ollama_url = config.get("ollama_url", model_manager.ollama_url)
    return {"status": "updated"}


# === User Endpoints ===

@app.get("/user/", response_class=HTMLResponse)
async def user_dashboard():
    return (STATIC_DIR / "user.html").read_text(encoding="utf-8")


@app.get("/playground/", response_class=HTMLResponse)
async def playground():
    return (STATIC_DIR / "playground.html").read_text(encoding="utf-8")


@app.get("/playground-live/", response_class=HTMLResponse)
async def playground_live():
    return (STATIC_DIR / "live-playground.html").read_text(encoding="utf-8")


@app.get("/user/api/me")
async def user_me(auth: Dict = Depends(verify_api_key)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, email, balance, tier, created_at FROM users WHERE id = ?",
            (auth["user_id"],),
        )
        user = dict(await cursor.fetchone())
        cursor = await db.execute(
            "SELECT id, key_prefix, name, created_at FROM api_keys WHERE user_id = ? AND active = 1",
            (auth["user_id"],),
        )
        keys = [dict(row) for row in await cursor.fetchall()]
        return {**user, "api_keys": keys}


@app.get("/user/api/usage")
async def user_usage(days: int = 7, auth: Dict = Depends(verify_api_key)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT date(created_at) as date, SUM(tokens_in) as tokens_in, SUM(tokens_out) as tokens_out,
                   SUM(cost) as cost, COUNT(*) as requests, AVG(latency_ms) as avg_latency
            FROM usage WHERE api_key_id IN (SELECT id FROM api_keys WHERE user_id = ?)
            AND datetime(created_at) > datetime('now', ? || ' days')
            GROUP BY date(created_at) ORDER BY date DESC
        """, (auth["user_id"], -days))
        return [dict(row) for row in await cursor.fetchall()]


@app.get("/user/api/models")
async def user_models(auth: Dict = Depends(verify_api_key)):
    return model_manager.get_active_models()


@app.get("/user/api/ratelimit")
async def user_rate_limit(auth: Dict = Depends(verify_api_key)):
    config = load_config()
    rate_limits = config.get("rate_limits", {"payg": 5, "sub": 10})
    max_requests = rate_limits.get(auth["tier"], 5)
    remaining = rate_limiter.get_remaining(auth["user_id"], auth["tier"], config)
    return {"tier": auth["tier"], "max_per_minute": max_requests, "remaining": remaining, "resets_in": 60}


# === Health / Public ===

@app.get("/health")
async def health():
    ollama_ok = await model_manager._ollama_health() if model_manager else False
    return {
        "status": "ok" if ollama_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "active_models": len(model_manager.get_active_models()) if model_manager else 0,
        "ollama": ollama_ok,
    }


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
async def public_models():
    if not model_manager:
        return {"models": []}
    return {
        "models": [
            {"id": m["name"], "type": m["type"], "status": "online"}
            for m in model_manager.get_active_models()
        ]
    }


@app.get("/api/uptime")
async def public_uptime(days: int = 30):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT date(created_at) as date, COUNT(*) as total_checks,
                   SUM(CASE WHEN status = 'up' THEN 1 ELSE 0 END) as up_checks,
                   SUM(CASE WHEN status = 'degraded' THEN 1 ELSE 0 END) as degraded_checks,
                   SUM(CASE WHEN status = 'down' THEN 1 ELSE 0 END) as down_checks,
                   AVG(latency_ms) as avg_latency, AVG(active_models) as avg_models
            FROM uptime_checks WHERE datetime(created_at) > datetime('now', ? || ' days')
            GROUP BY date(created_at) ORDER BY date DESC
        """, (f"-{days}",))
        daily_stats = []
        for row in await cursor.fetchall():
            total = row["total_checks"] or 1
            up_pct = ((row["up_checks"] or 0) / total) * 100
            degraded_pct = ((row["degraded_checks"] or 0) / total) * 100
            status = "up" if up_pct >= 99 else ("degraded" if up_pct >= 90 or degraded_pct > 0 else "down")
            daily_stats.append({
                "date": row["date"], "status": status,
                "uptime_percent": round(up_pct, 2),
                "avg_latency_ms": int(row["avg_latency"] or 0),
                "avg_models": round(row["avg_models"] or 0, 1),
            })
        cursor = await db.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('up', 'degraded') THEN 1 ELSE 0 END) as available
            FROM uptime_checks WHERE datetime(created_at) > datetime('now', ? || ' days')
        """, (f"-{days}",))
        overall = await cursor.fetchone()
        total = overall["total"] or 1
        overall_uptime = ((overall["available"] or 0) / total) * 100
        return {"days": days, "overall_uptime_percent": round(overall_uptime, 2), "daily": daily_stats}


# === Public Signup ===

@app.post("/api/signup")
async def public_signup(request: SignupRequest):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (request.email,))
        if await cursor.fetchone():
            raise HTTPException(400, "Email already registered")
        cursor = await db.execute(
            "SELECT id FROM signup_requests WHERE email = ? AND status = 'pending'",
            (request.email,),
        )
        if await cursor.fetchone():
            raise HTTPException(400, "Signup request already pending")
        await db.execute(
            "INSERT INTO signup_requests (email, tier, message) VALUES (?, ?, ?)",
            (request.email, request.tier, request.message),
        )
        await db.commit()
    return {"status": "pending", "message": "Your request has been submitted"}


# === Admin Signup Requests ===

@app.get("/admin/api/signups")
async def admin_list_signups(auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM signup_requests ORDER BY created_at DESC")
        return [dict(row) for row in await cursor.fetchall()]


@app.post("/admin/api/signups/{request_id}/approve")
async def admin_approve_signup(request_id: int, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM signup_requests WHERE id = ?", (request_id,))
        req = await cursor.fetchone()
        if not req:
            raise HTTPException(404, "Request not found")
        if req["status"] != "pending":
            raise HTTPException(400, "Request already processed")
        try:
            cursor = await db.execute(
                "INSERT INTO users (email, tier) VALUES (?, ?) RETURNING id",
                (req["email"], req["tier"]),
            )
            user_row = await cursor.fetchone()
            user_id = user_row[0]
        except aiosqlite.IntegrityError:
            raise HTTPException(400, "User already exists")
        key = f"sk-{secrets.token_hex(24)}"
        key_hash = hash_key(key)
        key_prefix = key[:8]
        await db.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES (?, ?, ?, ?)",
            (user_id, key_hash, key_prefix, "default"),
        )
        await db.execute(
            "UPDATE signup_requests SET status = 'approved' WHERE id = ?",
            (request_id,),
        )
        await db.commit()
        return {"status": "approved", "user_id": user_id, "email": req["email"], "tier": req["tier"], "api_key": key}


@app.post("/admin/api/signups/{request_id}/reject")
async def admin_reject_signup(request_id: int, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE signup_requests SET status = 'rejected' WHERE id = ?", (request_id,))
        await db.commit()
    return {"status": "rejected"}


# === Live Voice Playground ===

@app.get("/live", response_class=HTMLResponse)
async def live_playground_page():
    live_page = STATIC_DIR / "live-playground.html"
    if live_page.exists():
        return live_page.read_text(encoding="utf-8")
    raise HTTPException(404, "Live playground not found")


# === Static Files ===
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
