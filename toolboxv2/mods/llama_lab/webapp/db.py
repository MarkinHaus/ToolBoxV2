# file: toolboxv2/mods/llama_lab/webapp/db.py
"""Persistence + auth for the llama_lab web app (ported from llm-gateway).

Same product layer as the old gateway — users, API keys, usage, signups,
uptime — minus the Ollama bits. Web UI auth rides on the TB session; the API
keys here are what external OpenAI-compatible clients use against /v1/*.

New vs gateway: chat_folders + chat_sessions for the file-tree playground.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import aiosqlite

_DIR = Path(__file__).resolve().parent / "data"
_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(_DIR / "gateway.db")
CONFIG_PATH = _DIR / "config.json"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                balance REAL DEFAULT 0.0, tier TEXT DEFAULT 'payg',
                active INTEGER DEFAULT 1, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL, key_prefix TEXT NOT NULL,
                name TEXT DEFAULT 'default', active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id));
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY, api_key_id INTEGER NOT NULL, model TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0, latency_ms INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (api_key_id) REFERENCES api_keys(id));
            CREATE TABLE IF NOT EXISTS signup_requests (
                id INTEGER PRIMARY KEY, email TEXT NOT NULL, tier TEXT DEFAULT 'payg',
                message TEXT, status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS uptime_checks (
                id INTEGER PRIMARY KEY, status TEXT NOT NULL, active_models INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            -- file-tree playground (new)
            CREATE TABLE IF NOT EXISTS chat_folders (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL,
                parent_id INTEGER, sort INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, folder_id INTEGER,
                title TEXT DEFAULT 'New chat', messages TEXT DEFAULT '[]',
                sort INTEGER DEFAULT 0, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_key ON usage(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_sess_user ON chat_sessions(user_id);
        """)
        await db.commit()

        cur = await db.execute("SELECT id FROM users WHERE email='admin'")
        if not await cur.fetchone():
            await db.execute("INSERT INTO users (email,balance,tier) VALUES (?,?,?)",
                             ("admin", 999999.0, "admin"))
            await db.commit()

        config = load_config()
        admin_key = config.get("admin_key", "sk-admin-change-me-on-first-run")
        ah = hash_key(admin_key)
        cur = await db.execute("SELECT id FROM users WHERE email='admin'")
        admin_id = (await cur.fetchone())[0]
        cur = await db.execute("SELECT id FROM api_keys WHERE key_hash=?", (ah,))
        if not await cur.fetchone():
            await db.execute("DELETE FROM api_keys WHERE user_id=? AND name='admin-key'", (admin_id,))
            await db.execute(
                "INSERT INTO api_keys (user_id,key_hash,key_prefix,name) VALUES (?,?,?,?)",
                (admin_id, ah, admin_key[:10], "admin-key"))
            await db.commit()


# --- auth -------------------------------------------------------------------

async def verify_api_key(authorization: str | None):
    """Resolve a 'Bearer <key>' header to a user/key row, or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT ak.id AS key_id, ak.user_id, u.email, u.balance, u.tier, u.active
            FROM api_keys ak JOIN users u ON ak.user_id = u.id
            WHERE ak.key_hash = ? AND ak.active = 1
        """, (hash_key(authorization[7:]),))
        row = await cur.fetchone()
        return dict(row) if row and row["active"] else None


async def user_by_email(email: str, create: bool = False):
    """Map a TB-session identity to a gateway user row (create on first login)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE email=?", (email,))
        row = await cur.fetchone()
        if row:
            return dict(row)
        if not create:
            return None
        await db.execute("INSERT INTO users (email,tier) VALUES (?,?)", (email, "payg"))
        await db.commit()
        cur = await db.execute("SELECT * FROM users WHERE email=?", (email,))
        return dict(await cur.fetchone())


# --- config -----------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
    else:
        config = {
            "admin_key": "sk-admin-change-me-on-first-run",
            "pricing": {"input_per_1k": 0.0001, "output_per_1k": 0.0002},
            "rate_limits": {"payg": 5, "sub": 10, "admin": 1000},
            "live": {"enabled": True, "webcam": True, "screen": True},
        }
    if os.environ.get("ADMIN_KEY"):
        config["admin_key"] = os.environ["ADMIN_KEY"]
    return config


def save_config(config: dict):
    CONFIG_PATH.write_text(json.dumps({k: v for k, v in config.items()
                                       if not k.startswith("_")}, indent=2))


# --- usage ------------------------------------------------------------------

async def log_usage(key_id: int, model: str, tin: int, tout: int, latency: int):
    cfg = load_config().get("pricing", {})
    cost = tin / 1000 * cfg.get("input_per_1k", 0) + tout / 1000 * cfg.get("output_per_1k", 0)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO usage (api_key_id,model,tokens_in,tokens_out,cost,latency_ms) "
            "VALUES (?,?,?,?,?,?)", (key_id, model, tin, tout, cost, latency))
        await db.execute("UPDATE users SET balance = balance - ? WHERE id = "
                         "(SELECT user_id FROM api_keys WHERE id=?)", (cost, key_id))
        await db.commit()


async def record_uptime(status: str, active: int, latency: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO uptime_checks (status,active_models,latency_ms) "
                         "VALUES (?,?,?)", (status, active, latency))
        await db.execute("DELETE FROM uptime_checks WHERE datetime(created_at) "
                         "< datetime('now','-30 days')")
        await db.commit()
