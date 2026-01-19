"""
LLM Gateway - OpenAI Compatible API Server
Ports: Gateway 4000, Model Slots 4801-4807
"""

import asyncio
import hashlib
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import aiosqlite
import httpx
import psutil
from fastapi import FastAPI, HTTPException, Header, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_manager import ModelManager

# === Config ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "gateway.db"
CONFIG_PATH = DATA_DIR / "config.json"
STATIC_DIR = BASE_DIR / "static"

DATA_DIR.mkdir(exist_ok=True)

# === Rate Limiter ===
from collections import defaultdict
import time as time_module

class RateLimiter:
    """Simple in-memory rate limiter per user"""

    def __init__(self):
        self.requests: Dict[int, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: int, tier: str, config: Dict) -> bool:
        """Check if request is allowed based on tier rate limit"""
        now = time_module.time()
        window = 60  # 1 minute window

        # Get rate limit from config
        rate_limits = config.get("rate_limits", {"payg": 5, "sub": 10})
        max_requests = rate_limits.get(tier, 5)

        # Admin has no limit
        if tier == "admin":
            return True

        # Clean old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if now - ts < window
        ]

        # Check limit
        if len(self.requests[user_id]) >= max_requests:
            return False

        # Record request
        self.requests[user_id].append(now)
        return True

    def get_remaining(self, user_id: int, tier: str, config: Dict) -> int:
        """Get remaining requests in current window"""
        now = time_module.time()
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

class TranscriptionRequest(BaseModel):
    model: str = "whisper"
    language: Optional[str] = None
    temperature: Optional[float] = 0.0

class UserCreate(BaseModel):
    email: str
    tier: str = "payg"  # payg or sub

class UserUpdate(BaseModel):
    balance: Optional[float] = None
    tier: Optional[str] = None
    active: Optional[bool] = None

class ApiKeyCreate(BaseModel):
    user_id: int
    name: str = "default"

class ModelLoadRequest(BaseModel):
    slot: int  # 4801-4807
    model_path: str  # local path or HF repo
    model_type: str = "text"  # text, vision, audio
    ctx_size: Optional[int] = None
    threads: Optional[int] = None

class SignupRequest(BaseModel):
    email: str
    tier: str = "payg"
    message: Optional[str] = None

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
                ("admin", 999999.0, "admin")
            )
            await db.commit()

        # Ensure admin has API key from config
        config = load_config()
        admin_key = config.get("admin_key", "sk-admin-change-me-on-first-run")
        admin_key_hash = hash_key(admin_key)

        cursor = await db.execute("SELECT id FROM users WHERE email = 'admin'")
        admin_row = await cursor.fetchone()
        admin_id = admin_row[0]

        # Check if this specific admin key exists
        cursor = await db.execute(
            "SELECT id FROM api_keys WHERE key_hash = ?", (admin_key_hash,)
        )
        if not await cursor.fetchone():
            # Remove old admin keys and add the configured one
            await db.execute(
                "DELETE FROM api_keys WHERE user_id = ? AND name = 'admin-key'",
                (admin_id,)
            )
            await db.execute(
                "INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES (?, ?, ?, ?)",
                (admin_id, admin_key_hash, admin_key[:10], "admin-key")
            )
            await db.commit()

async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db

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

# === Config Management ===

def load_config() -> Dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "slots": {str(p): None for p in range(4801, 4808)},
        "hf_token": None,
        "default_threads": 10,
        "default_ctx_size": 8192,
        "pricing": {"input_per_1k": 0.0001, "output_per_1k": 0.0002}
    }

def save_config(config: Dict):
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

# === Model Manager Instance ===
model_manager: ModelManager = None
uptime_task: asyncio.Task = None

async def uptime_checker():
    """Background task to record uptime every 5 minutes"""
    while True:
        try:
            start = time.time()

            # Check if server is responding
            status = "up"
            active_models = 0

            if model_manager:
                models = model_manager.get_active_models()
                active_models = len(models)

                # If we have models configured but none running, it's degraded
                slots = model_manager.get_slots_status()
                crashed = [s for s in slots if s.get("status") == "crashed"]
                if crashed:
                    status = "degraded"

            latency = int((time.time() - start) * 1000)

            # Record to database
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO uptime_checks (status, active_models, latency_ms) VALUES (?, ?, ?)",
                    (status, active_models, latency)
                )

                # Clean up old records (keep 30 days)
                await db.execute(
                    "DELETE FROM uptime_checks WHERE datetime(created_at) < datetime('now', '-30 days')"
                )
                await db.commit()

        except Exception as e:
            # Log error but don't crash
            print(f"Uptime check error: {e}")
            try:
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute(
                        "INSERT INTO uptime_checks (status, active_models, latency_ms) VALUES (?, ?, ?)",
                        ("down", 0, 0)
                    )
                    await db.commit()
            except:
                pass

        await asyncio.sleep(300)  # 5 minutes

# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, uptime_task
    await init_db()
    config = load_config()
    model_manager = ModelManager(
        base_dir=BASE_DIR,
        models_dir=DATA_DIR / "models",
        config=config
    )
    await model_manager.restore_slots()

    # Start uptime checker
    uptime_task = asyncio.create_task(uptime_checker())

    # Record initial uptime
    async with aiosqlite.connect(DB_PATH) as db:
        active = len(model_manager.get_active_models()) if model_manager else 0
        await db.execute(
            "INSERT INTO uptime_checks (status, active_models, latency_ms) VALUES (?, ?, ?)",
            ("up", active, 0)
        )
        await db.commit()

    yield

    # Cancel uptime checker
    if uptime_task:
        uptime_task.cancel()
        try:
            await uptime_task
        except asyncio.CancelledError:
            pass

    await model_manager.shutdown_all()

# === App ===

app = FastAPI(
    title="LLM Gateway",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow all origins for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === OpenAI Compatible Endpoints ===

@app.get("/v1/models")
async def list_models(auth: Dict = Depends(verify_api_key)):
    """List available models (OpenAI compatible)"""
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
                "parent": None
            }
            for m in models
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key)
):
    """Chat completions endpoint (OpenAI compatible)"""

    # Check rate limit
    config = load_config()
    if not rate_limiter.is_allowed(auth["user_id"], auth["tier"], config):
        remaining = rate_limiter.get_remaining(auth["user_id"], auth["tier"], config)
        raise HTTPException(
            429,
            f"Rate limit exceeded. Try again in a minute. Remaining: {remaining}"
        )

    # Check balance for payg users
    if auth["tier"] == "payg" and auth["balance"] <= 0:
        raise HTTPException(402, "Insufficient balance")

    # Find model slot
    slot = model_manager.find_model_slot(request.model)
    if not slot:
        raise HTTPException(404, f"Model '{request.model}' not loaded")

    backend_url = f"http://127.0.0.1:{slot['port']}/v1/chat/completions"

    # Prepare request for llama-server
    payload = {
        "model": request.model,
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
            stream_chat_response(
                backend_url, payload, auth, start_time, request.model, background_tasks
            ),
            media_type="text/event-stream"
        )
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(backend_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                # Track usage
                latency = int((time.time() - start_time) * 1000)
                tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
                tokens_out = data.get("usage", {}).get("completion_tokens", 0)

                background_tasks.add_task(
                    log_usage, auth["key_id"], request.model,
                    tokens_in, tokens_out, latency
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
    background_tasks: BackgroundTasks
) -> AsyncGenerator[str, None]:
    """Stream response from llama-server"""
    tokens_out = 0

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"

                        # Count tokens from chunks
                        if line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                    tokens_out += 1  # Approximate
                            except:
                                pass
                    elif line:
                        yield f"data: {line}\n\n"

            yield "data: [DONE]\n\n"

            # Log usage after stream
            latency = int((time.time() - start_time) * 1000)
            background_tasks.add_task(
                log_usage, auth["key_id"], model, 0, tokens_out, latency
            )

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key)
):
    """Embeddings endpoint (OpenAI compatible)"""
    slot = model_manager.find_model_slot(request.model)
    if not slot:
        raise HTTPException(404, f"Model '{request.model}' not loaded")

    backend_url = f"http://127.0.0.1:{slot['port']}/v1/embeddings"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(backend_url, json=request.model_dump())
        resp.raise_for_status()
        return resp.json()

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_api_key)
):
    """Audio transcription endpoint (OpenAI compatible)"""
    # Find whisper slot
    slot = model_manager.find_audio_slot()
    if not slot:
        raise HTTPException(404, "No audio model loaded")

    backend_url = f"http://127.0.0.1:{slot['port']}/inference"

    # Forward multipart form data
    form = await request.form()
    files = {}
    data = {}

    for key, value in form.items():
        if hasattr(value, 'read'):
            files[key] = (value.filename, await value.read(), value.content_type)
        else:
            data[key] = value

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(backend_url, files=files, data=data)
        resp.raise_for_status()
        return resp.json()

# === Usage Logging ===

async def log_usage(key_id: int, model: str, tokens_in: int, tokens_out: int, latency_ms: int):
    config = load_config()
    pricing = config.get("pricing", {})
    cost = (tokens_in / 1000 * pricing.get("input_per_1k", 0.0001) +
            tokens_out / 1000 * pricing.get("output_per_1k", 0.0002))

    async with aiosqlite.connect(DB_PATH) as db:
        # Log usage
        await db.execute("""
            INSERT INTO usage (api_key_id, model, tokens_in, tokens_out, cost, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (key_id, model, tokens_in, tokens_out, cost, latency_ms))

        # Deduct from balance (for payg users)
        await db.execute("""
            UPDATE users SET balance = balance - ?
            WHERE id = (SELECT user_id FROM api_keys WHERE id = ?)
            AND tier = 'payg'
        """, (cost, key_id))

        await db.commit()

# === Stats Endpoint ===

@app.get("/v1/stats/summary")
async def stats_summary(auth: Dict = Depends(verify_api_key)):
    """NLP-style stats summary"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Get stats for last 24h
        cursor = await db.execute("""
            SELECT
                COUNT(DISTINCT api_key_id) as unique_keys,
                COUNT(*) as total_requests,
                SUM(tokens_in + tokens_out) as total_tokens,
                AVG(latency_ms) as avg_latency,
                SUM(cost) as total_cost
            FROM usage
            WHERE datetime(created_at) > datetime('now', '-24 hours')
        """)
        stats = dict(await cursor.fetchone())

        # Get new users today
        cursor = await db.execute("""
            SELECT COUNT(*) as new_users
            FROM users
            WHERE date(created_at) = date('now')
        """)
        new_users = (await cursor.fetchone())[0]

        # Get active models
        active_models = model_manager.get_active_models()

        # System stats
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)

        return {
            "period": "24h",
            "summary": f"{new_users} neue User, {stats['total_requests'] or 0} Anfragen verarbeitet, "
                      f"{stats['total_tokens'] or 0} Tokens, Ø Latency {int(stats['avg_latency'] or 0)}ms",
            "stats": {
                "new_users": new_users,
                "total_requests": stats["total_requests"] or 0,
                "total_tokens": stats["total_tokens"] or 0,
                "avg_latency_ms": int(stats["avg_latency"] or 0),
                "total_cost": round(stats["total_cost"] or 0, 4)
            },
            "system": {
                "cpu_percent": cpu,
                "ram_used_gb": round(mem.used / (1024**3), 1),
                "ram_total_gb": round(mem.total / (1024**3), 1),
                "ram_percent": mem.percent
            },
            "active_models": len(active_models),
            "models": active_models
        }

# === Admin Endpoints ===

@app.get("/admin/", response_class=HTMLResponse)
async def admin_panel():
    """Serve admin panel"""
    return (STATIC_DIR / "admin.html").read_text(encoding="utf-8")

@app.get("/admin/api/users")
async def admin_list_users(auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT u.*,
                   (SELECT COUNT(*) FROM api_keys WHERE user_id = u.id) as key_count,
                   (SELECT SUM(cost) FROM usage WHERE api_key_id IN
                    (SELECT id FROM api_keys WHERE user_id = u.id)) as total_usage
            FROM users u
            ORDER BY created_at DESC
        """)
        return [dict(row) for row in await cursor.fetchall()]

@app.post("/admin/api/users")
async def admin_create_user(user: UserCreate, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            cursor = await db.execute(
                "INSERT INTO users (email, tier) VALUES (?, ?) RETURNING id",
                (user.email, user.tier)
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
            # Log transaction
            await db.execute(
                "INSERT INTO transactions (user_id, amount, type, note) VALUES (?, ?, ?, ?)",
                (user_id, update.balance, "admin_topup", "Admin balance update")
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
            (user_id, key_hash, key_prefix, data.name)
        )
        await db.commit()

    return {"key": key, "prefix": key_prefix}  # Only time full key is shown

@app.delete("/admin/api/apikeys/{key_id}")
async def admin_delete_apikey(key_id: int, auth: Dict = Depends(verify_admin)):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))
        await db.commit()
    return {"status": "deleted"}

# === Model Management ===

@app.get("/admin/api/slots")
async def admin_get_slots(auth: Dict = Depends(verify_admin)):
    """Get all slot statuses"""
    return model_manager.get_slots_status()

@app.post("/admin/api/slots/load")
async def admin_load_model(request: ModelLoadRequest, auth: Dict = Depends(verify_admin)):
    """Load model into slot"""
    result = await model_manager.load_model(
        slot=request.slot,
        model_path=request.model_path,
        model_type=request.model_type,
        ctx_size=request.ctx_size,
        threads=request.threads
    )
    return result

@app.post("/admin/api/slots/{slot}/unload")
async def admin_unload_model(slot: int, auth: Dict = Depends(verify_admin)):
    """Unload model from slot"""
    result = await model_manager.unload_model(slot)
    return result

@app.get("/admin/api/models/local")
async def admin_list_local_models(auth: Dict = Depends(verify_admin)):
    """List locally downloaded models"""
    return model_manager.list_local_models()

@app.get("/admin/api/models/search")
async def admin_search_hf_models(q: str, auth: Dict = Depends(verify_admin)):
    """Search HuggingFace for GGUF models"""
    return await model_manager.search_hf_models(q)
class DownloadRequest(BaseModel):
    repo_id: str
    filename: str
@app.post("/admin/api/models/download")
async def admin_download_model(request: DownloadRequest, auth: Dict = Depends(verify_admin)):
    """Download model from HuggingFace"""
    return await model_manager.download_hf_model(request.repo_id, request.filename)

@app.get("/admin/api/system")
async def admin_system_stats(auth: Dict = Depends(verify_admin)):
    """Get system stats"""
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)

    # Per-process memory for model slots
    slot_processes = model_manager.get_process_stats()

    return {
        "cpu": {
            "percent_per_core": cpu_percent,
            "total_percent": sum(cpu_percent) / len(cpu_percent),
            "cores": len(cpu_percent)
        },
        "memory": {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent
        },
        "slots": slot_processes
    }

@app.get("/admin/api/config")
async def admin_get_config(auth: Dict = Depends(verify_admin)):
    config = load_config()
    # Mask sensitive values
    if config.get("hf_token"):
        config["hf_token"] = config["hf_token"][:8] + "..."
    if config.get("admin_key"):
        key = config["admin_key"]
        config["admin_key_display"] = key[:12] + "..." + key[-4:]
    return config

@app.patch("/admin/api/config")
async def admin_update_config(updates: Dict, auth: Dict = Depends(verify_admin)):
    config = load_config()
    config.update(updates)
    save_config(config)
    model_manager.config = config
    return {"status": "updated"}

# === User Endpoints ===

@app.get("/user/", response_class=HTMLResponse)
async def user_dashboard():
    """Serve user dashboard"""
    return (STATIC_DIR / "user.html").read_text(encoding="utf-8")

@app.get("/playground/", response_class=HTMLResponse)
async def playground():
    """Serve playground"""
    return (STATIC_DIR / "playground.html").read_text(encoding="utf-8")

@app.get("/user/api/me")
async def user_me(auth: Dict = Depends(verify_api_key)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT id, email, balance, tier, created_at FROM users WHERE id = ?",
            (auth["user_id"],)
        )
        user = dict(await cursor.fetchone())

        cursor = await db.execute(
            "SELECT id, key_prefix, name, created_at FROM api_keys WHERE user_id = ? AND active = 1",
            (auth["user_id"],)
        )
        keys = [dict(row) for row in await cursor.fetchall()]

        return {**user, "api_keys": keys}

@app.get("/user/api/usage")
async def user_usage(
    days: int = 7,
    auth: Dict = Depends(verify_api_key)
):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT
                date(created_at) as date,
                SUM(tokens_in) as tokens_in,
                SUM(tokens_out) as tokens_out,
                SUM(cost) as cost,
                COUNT(*) as requests,
                AVG(latency_ms) as avg_latency
            FROM usage
            WHERE api_key_id IN (SELECT id FROM api_keys WHERE user_id = ?)
            AND datetime(created_at) > datetime('now', ? || ' days')
            GROUP BY date(created_at)
            ORDER BY date DESC
        """, (auth["user_id"], -days))

        return [dict(row) for row in await cursor.fetchall()]

@app.get("/user/api/models")
async def user_models(auth: Dict = Depends(verify_api_key)):
    """List available models for user"""
    return model_manager.get_active_models()

@app.get("/user/api/ratelimit")
async def user_rate_limit(auth: Dict = Depends(verify_api_key)):
    """Get user's rate limit status"""
    config = load_config()
    rate_limits = config.get("rate_limits", {"payg": 5, "sub": 10})
    max_requests = rate_limits.get(auth["tier"], 5)
    remaining = rate_limiter.get_remaining(auth["user_id"], auth["tier"], config)

    return {
        "tier": auth["tier"],
        "max_per_minute": max_requests,
        "remaining": remaining,
        "resets_in": 60  # seconds
    }

# === Health ===

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_slots": len(model_manager.get_active_models()) if model_manager else 0
    }

# === Landing Page ===

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Serve landing page"""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

# === Public Models Endpoint (no auth) ===

@app.get("/api/models")
async def public_models():
    """Public endpoint to list available models (no auth required)"""
    if not model_manager:
        return {"models": []}
    models = model_manager.get_active_models()
    return {
        "models": [
            {
                "id": m["name"],
                "type": m["type"],
                "status": "online"
            }
            for m in models
        ]
    }

@app.get("/api/uptime")
async def public_uptime(days: int = 30):
    """Public endpoint for uptime history (no auth required)"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Get daily aggregated uptime for last N days
        cursor = await db.execute("""
            SELECT
                date(created_at) as date,
                COUNT(*) as total_checks,
                SUM(CASE WHEN status = 'up' THEN 1 ELSE 0 END) as up_checks,
                SUM(CASE WHEN status = 'degraded' THEN 1 ELSE 0 END) as degraded_checks,
                SUM(CASE WHEN status = 'down' THEN 1 ELSE 0 END) as down_checks,
                AVG(latency_ms) as avg_latency,
                AVG(active_models) as avg_models
            FROM uptime_checks
            WHERE datetime(created_at) > datetime('now', ? || ' days')
            GROUP BY date(created_at)
            ORDER BY date DESC
        """, (f"-{days}",))

        daily_stats = []
        for row in await cursor.fetchall():
            total = row["total_checks"] or 1
            up_pct = ((row["up_checks"] or 0) / total) * 100
            degraded_pct = ((row["degraded_checks"] or 0) / total) * 100

            # Determine day status
            if up_pct >= 99:
                status = "up"
            elif up_pct >= 90 or degraded_pct > 0:
                status = "degraded"
            else:
                status = "down"

            daily_stats.append({
                "date": row["date"],
                "status": status,
                "uptime_percent": round(up_pct, 2),
                "avg_latency_ms": int(row["avg_latency"] or 0),
                "avg_models": round(row["avg_models"] or 0, 1)
            })

        # Calculate overall uptime
        cursor = await db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status IN ('up', 'degraded') THEN 1 ELSE 0 END) as available
            FROM uptime_checks
            WHERE datetime(created_at) > datetime('now', ? || ' days')
        """, (f"-{days}",))
        overall = await cursor.fetchone()

        total = overall["total"] or 1
        overall_uptime = ((overall["available"] or 0) / total) * 100

        return {
            "days": days,
            "overall_uptime_percent": round(overall_uptime, 2),
            "daily": daily_stats
        }

# === Public Signup ===

@app.post("/api/signup")
async def public_signup(request: SignupRequest):
    """Public endpoint for signup requests (no auth required)"""
    async with aiosqlite.connect(DB_PATH) as db:
        # Check if email already exists as user
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (request.email,))
        if await cursor.fetchone():
            raise HTTPException(400, "Email already registered")

        # Check if pending request exists
        cursor = await db.execute(
            "SELECT id FROM signup_requests WHERE email = ? AND status = 'pending'",
            (request.email,)
        )
        if await cursor.fetchone():
            raise HTTPException(400, "Signup request already pending")

        # Create request
        await db.execute(
            "INSERT INTO signup_requests (email, tier, message) VALUES (?, ?, ?)",
            (request.email, request.tier, request.message)
        )
        await db.commit()

    return {"status": "pending", "message": "Your request has been submitted"}

# === Admin Signup Requests ===

@app.get("/admin/api/signups")
async def admin_list_signups(auth: Dict = Depends(verify_admin)):
    """List pending signup requests"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM signup_requests ORDER BY created_at DESC"
        )
        return [dict(row) for row in await cursor.fetchall()]

@app.post("/admin/api/signups/{request_id}/approve")
async def admin_approve_signup(request_id: int, auth: Dict = Depends(verify_admin)):
    """Approve signup request and create user with API key"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Get request
        cursor = await db.execute(
            "SELECT * FROM signup_requests WHERE id = ?", (request_id,)
        )
        req = await cursor.fetchone()
        if not req:
            raise HTTPException(404, "Request not found")

        if req["status"] != "pending":
            raise HTTPException(400, "Request already processed")

        # Create user
        try:
            cursor = await db.execute(
                "INSERT INTO users (email, tier) VALUES (?, ?) RETURNING id",
                (req["email"], req["tier"])
            )
            user_row = await cursor.fetchone()
            user_id = user_row[0]
        except aiosqlite.IntegrityError:
            raise HTTPException(400, "User already exists")

        # Create API key
        key = f"sk-{secrets.token_hex(24)}"
        key_hash = hash_key(key)
        key_prefix = key[:8]

        await db.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES (?, ?, ?, ?)",
            (user_id, key_hash, key_prefix, "default")
        )

        # Update request status
        await db.execute(
            "UPDATE signup_requests SET status = 'approved' WHERE id = ?",
            (request_id,)
        )

        await db.commit()

        return {
            "status": "approved",
            "user_id": user_id,
            "email": req["email"],
            "tier": req["tier"],
            "api_key": key  # Admin should send this to user
        }

@app.post("/admin/api/signups/{request_id}/reject")
async def admin_reject_signup(request_id: int, auth: Dict = Depends(verify_admin)):
    """Reject signup request"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE signup_requests SET status = 'rejected' WHERE id = ?",
            (request_id,)
        )
        await db.commit()
    return {"status": "rejected"}

# === Static Files ===
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
