"""
Interactive Configuration Wizard for ToolBoxV2

A beautiful, minimal, and repeatable CLI wizard that:
1. Configures tb-manifest.yaml settings
2. Sets up missing environment variables from env-template
3. Can be run multiple times to update configuration
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cli_printing import (
    print_box_header,
    print_box_footer,
    print_box_content,
    print_status,
    print_separator,
    c_print,
    Colors,
)
from .cli_input import menu_select


# =================== LLM Provider Registry ===================

LLM_PROVIDERS: Dict[str, dict] = {
    "9router": {
        "name": "9Router (Local Proxy / Gateway)",
        "default": True,
        "key_vars": ["NINEROUTER_KEY"],
        "url_var": "NINEROUTER_URL",
        "default_url": "http://localhost:20128/v1",
        "extra_vars": ["NINEROUTER_USER", "NINEROUTER_PASSWORD"],
        "validate_path": "/models",
        "models": {
            "FAST": "9rou/fast",
            "BLITZ": "cerebras/gpt-oss-120b",
            "COMPLEX": "9rou/complex",
            "SUMMARY": "openrouter/mistralai/ministral-8b",
        },
    },
    "ollama": {
        "name": "Ollama (Local AI)",
        "default": False,
        "key_vars": [],
        "url_var": "OLLAMA_BASE_URL",
        "default_url": "http://localhost:11434",
        "extra_vars": [],
        "validate_path": "/api/tags",
        "models": {
            "FAST": "ollama/llama3.1",
            "BLITZ": "ollama/llama3.1",
            "COMPLEX": "ollama/llama3.1",
            "SUMMARY": "ollama/llama2",
        },
    },
    "openrouter": {
        "name": "OpenRouter (All models via one key)",
        "default": False,
        "key_vars": ["OPENROUTER_API_KEY"],
        "url_var": None,
        "default_url": "https://openrouter.ai/api/v1",
        "extra_vars": [],
        "validate_path": "/models",
        "models": {
            "FAST": "openrouter/google/gemini-2.5-flash-lite",
            "BLITZ": "openrouter/google/gemini-2.5-flash-lite",
            "COMPLEX": "openrouter/google/gemini-2.5-flash",
            "SUMMARY": "openrouter/mistralai/ministral-8b",
        },
    },
    "groq": {
        "name": "Groq (Free, fast)",
        "default": False,
        "key_vars": ["GROQ_API_KEY"],
        "url_var": None,
        "default_url": "https://api.groq.com/openai/v1",
        "extra_vars": [],
        "validate_path": "/models",
        "models": {
            "FAST": "groq/llama-3.1-8b-instant",
            "BLITZ": "groq/llama-3.1-8b-instant",
            "COMPLEX": "groq/llama-3.1-70b-versatile",
            "SUMMARY": "groq/llama-3.1-8b-instant",
        },
    },
    "gemini": {
        "name": "Google Gemini",
        "default": False,
        "key_vars": ["GEMINI_API_KEY"],
        "url_var": None,
        "default_url": "https://generativelanguage.googleapis.com/v1",
        "extra_vars": [],
        "validate_path": "/models",
        "models": {
            "FAST": "gemini/gemini-2.0-flash-lite",
            "BLITZ": "gemini/gemini-2.0-flash-lite",
            "COMPLEX": "gemini/gemini-2.5-flash",
            "SUMMARY": "gemini/gemini-2.0-flash-lite",
        },
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "default": False,
        "key_vars": ["ANTHROPIC_API_KEY"],
        "url_var": None,
        "default_url": "https://api.anthropic.com/v1",
        "extra_vars": [],
        "validate_path": "/models",
        "models": {
            "FAST": "anthropic/claude-3-5-haiku-20241022",
            "BLITZ": "anthropic/claude-3-5-haiku-20241022",
            "COMPLEX": "anthropic/claude-3-5-sonnet-20241022",
            "SUMMARY": "anthropic/claude-3-5-haiku-20241022",
        },
    },
    "openai": {
        "name": "OpenAI",
        "default": False,
        "key_vars": ["OPENAI_API_KEY"],
        "url_var": None,
        "default_url": "https://api.openai.com/v1",
        "extra_vars": [],
        "validate_path": "/models",
        "models": {
            "FAST": "openai/gpt-4.1-mini",
            "BLITZ": "openai/gpt-4.1-mini",
            "COMPLEX": "openai/gpt-4.1",
            "SUMMARY": "openai/gpt-4.1-mini",
        },
    },
    "deepseek": {
        "name": "DeepSeek",
        "default": False,
        "key_vars": ["DEEPSEEK_API_KEY"],
        "url_var": None,
        "default_url": "https://api.deepseek.com/v1",
        "extra_vars": [],
        "validate_path": "/models",
        "models": {
            "FAST": "deepseek/deepseek-chat",
            "BLITZ": "deepseek/deepseek-chat",
            "COMPLEX": "deepseek/deepseek-reasoner",
            "SUMMARY": "deepseek/deepseek-chat",
        },
    },
}


def validate_llm_key(provider_key: str, api_key: str, base_url: str = "") -> Tuple[bool, str, list]:
    """Validate an LLM API key by fetching the /models endpoint.

    Returns (success, message, list_of_model_ids).
    Uses urllib (stdlib) — no external dependency.
    """
    p = LLM_PROVIDERS.get(provider_key)
    if not p:
        return False, f"Unknown provider: {provider_key}", []

    url = base_url or p["default_url"]
    path = p.get("validate_path", "/models")
    full_url = f"{url.rstrip('/')}{path}"

    headers: Dict[str, str] = {}
    if api_key:
        if provider_key == "9router":
            # 9Router uses basic auth via user:password
            import base64 as b64mod
            user = os.getenv("NINEROUTER_USER", "")
            pw = os.getenv("NINEROUTER_PASSWORD", "")
            if user and pw:
                cred = b64mod.b64encode(f"{user}:{pw}".encode()).decode()
                headers["Authorization"] = f"Basic {cred}"
            elif api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        elif provider_key == "gemini":
            full_url = f"{url.rstrip('/')}/models?key={api_key}&pageSize=100"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = Request(full_url, headers=headers)
        with urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}", []
            data = json.loads(resp.read().decode())
    except HTTPError as e:
        if e.code == 401:
            return False, "Invalid API key (401 Unauthorized)", []
        if e.code == 403:
            return False, "Forbidden (403) — check key permissions", []
        return False, f"HTTP error: {e.code}", []
    except URLError as e:
        return False, f"Connection failed: {e.reason}", []
    except Exception as e:
        return False, f"Error: {e}", []

    # Parse models
    models: list = []
    if isinstance(data, dict):
        model_list = data.get("data") or data.get("models") or []
        for m in model_list:
            if isinstance(m, dict):
                mid = m.get("id") or m.get("name") or ""
                if mid:
                    models.append(mid)
            elif isinstance(m, str):
                models.append(m)

    return True, f"OK — {len(models)} models available", models


# =================== Profile-based Core Defaults ===================

PROFILE_CORE_DEFAULTS: Dict[str, dict] = {
    "local": {
        "APP_BASE_URL": "http://localhost:8000",
        "TOOLBOXV2_BASE": "localhost",
        "TOOLBOXV2_REMOTE_BASE": "https://simplecore.app",
        "IS_OFFLINE_DB": "true",
        "DB_MODE_KEY": "LC",
    },
    "consumer": {
        "APP_BASE_URL": "http://localhost:8000",
        "TOOLBOXV2_BASE": "localhost",
        "TOOLBOXV2_REMOTE_BASE": "https://simplecore.app",
        "IS_OFFLINE_DB": "true",
        "DB_MODE_KEY": "LC",
    },
    "homelab": {
        "APP_BASE_URL": "http://localhost:8000",
        "TOOLBOXV2_BASE": "localhost",
        "TOOLBOXV2_REMOTE_BASE": "https://simplecore.app",
        "IS_OFFLINE_DB": "false",
        "DB_MODE_KEY": "LR",
    },
    "server": {
        "APP_BASE_URL": "https://simplecore.app",
        "TOOLBOXV2_BASE": "0.0.0.0",
        "TOOLBOXV2_REMOTE_BASE": "https://simplecore.app",
        "IS_OFFLINE_DB": "false",
        "DB_MODE_KEY": "RR",
    },
    "business": {
        "APP_BASE_URL": "https://simplecore.app",
        "TOOLBOXV2_BASE": "0.0.0.0",
        "TOOLBOXV2_REMOTE_BASE": "https://simplecore.app",
        "IS_OFFLINE_DB": "false",
        "DB_MODE_KEY": "CB",
    },
    "developer": {
        "APP_BASE_URL": "http://localhost:8000",
        "TOOLBOXV2_BASE": "localhost",
        "TOOLBOXV2_REMOTE_BASE": "https://simplecore.app",
        "IS_OFFLINE_DB": "false",
        "DB_MODE_KEY": "LC",
    },
}


def get_profile_defaults(profile: Optional[str]) -> dict:
    """Return core env defaults for the given profile."""
    if not profile:
        return PROFILE_CORE_DEFAULTS["local"]
    return PROFILE_CORE_DEFAULTS.get(profile, PROFILE_CORE_DEFAULTS["local"])


# =================== Environment Template Parser ===================

def parse_env_template(template_path: Path) -> Dict[str, Tuple[str, str]]:
    """Parse env-template file and return dict of {VAR_NAME: (current_value, comment)}."""
    env_vars: Dict[str, Tuple[str, str]] = {}

    if not template_path.exists():
        return env_vars

    content = template_path.read_text(encoding="utf-8")
    current_comment = ""

    for line in content.splitlines():
        line = line.strip()

        # Collect comments
        if line.startswith("#"):
            current_comment = line[1:].strip()
            continue

        # Skip empty lines
        if not line:
            current_comment = ""
            continue

        # Parse VAR=value or VAR=
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env_vars[key] = (value, current_comment)
            current_comment = ""

    return env_vars


def load_existing_env(env_path: Path) -> Dict[str, str]:
    """Load existing .env file values."""
    existing: Dict[str, str] = {}

    if not env_path.exists():
        return existing

    content = env_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            existing[key.strip()] = value.strip().strip('"').strip("'")

    return existing


# =================== Interactive Input Helpers ===================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def prompt_input(prompt: str, default: str = "", password: bool = False,
                 required: bool = False) -> str:
    """Get user input with optional default value."""
    if default:
        display_default = "***" if password and default else default
        prompt_text = f"{Colors.CYAN}❯{Colors.RESET} {prompt} [{Colors.DIM}{display_default}{Colors.RESET}]: "
    else:
        prompt_text = f"{Colors.CYAN}❯{Colors.RESET} {prompt}: "

    try:
        if password:
            import getpass
            value = getpass.getpass(prompt_text)
        else:
            value = input(prompt_text)

        value = value.strip()

        if not value and default:
            return default

        if required and not value:
            print_status("This field is required", "error")
            return prompt_input(prompt, default, password, required)

        return value
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def prompt_choice(prompt: str, choices: List[str], default: int = 0) -> str:
    """Let user choose from a list of options."""
    sel = menu_select(
        choices,
        title=prompt,
        start=default,
        hint="↑/↓ or W/S · Enter · q to back"
    )
    if sel is None:
        return choices[default]
    return sel


def prompt_bool(prompt: str, default: bool = True) -> bool:
    """Ask yes/no question."""
    default_str = "Y/n" if default else "y/N"
    try:
        answer = input(f"{Colors.CYAN}❯{Colors.RESET} {prompt} [{default_str}]: ").strip().lower()

        if not answer:
            return default

        return answer in ("y", "yes", "ja", "j", "1", "true")
    except (KeyboardInterrupt, EOFError):
        return default


# =================== Environment Variable Categories ===================

ENV_CATEGORIES = {
    "Core Settings": [
        "TOOLBOXV2_BASE", "TOOLBOXV2_BASE_PORT", "TOOLBOXV2_REMOTE_BASE",
        "APP_BASE_URL", "ADMIN_UI_PASSWORD", "TOOLBOX_LOGGING_LEVEL",
    ],
    "Database": [
        "DB_CONNECTION_URI", "DB_MODE_KEY", "IS_OFFLINE_DB", "SERVER_ID", "DB_CACHE_TTL",
    ],
    "Storage (MinIO/S3)": [
        "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY",
        "CLOUD_ENDPOINT", "CLOUD_ACCESS_KEY", "CLOUD_SECRET_KEY",
    ],
    "AI/LLM APIs": [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
        "OLLAMA_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN",
    ],
    "AI Extras": [
        "ELEVENLABS_API_KEY", "REPLICATE_API_TOKEN",
        "DEEPGRAM_API_KEY", "WOLFRAM_ALPHA_APPID", "PINECONE_API_KEY",
    ],
    "LLM Gateway & Providers": [
        "TB_LLM_GATEWAY_URL", "TB_LLM_GATEWAY_KEY", "OLLAMA_BASE_URL",
        "MINIMAX_API_KEY", "ZAI_API_KEY", "INCEPTION_API_KEY",
        "CEREBRAS_API_KEY", "NINEROUTER_KEY", "NINEROUTER_URL",
    ],
    "Agent Tuning": [
        "DEFAULT_MAX_ITERATIONS", "DEFAULT_MAX_HISTORY_LENGTH",
        "AGENT_VERBOSE", "DREAMER_FAST_MODEL", "DREAMER_COMPLEX_MODEL",
        "NARRATOR_ENABLED", "NARRATOR_LANG",
        "SUB_AGENT_MAX_TOKENS", "SUB_AGENT_MAX_ITERATIONS",
    ],
    "Search & Web": [
        "SERP_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID",
        "BING_API_KEY", "FIRECRAWL_API_KEY",
    ],
    "Messaging & Bots": [
        "DISCORD_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "DISCORD_AID",
        "TELEGRAM_AID",
        "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_API_TOKEN",
    ],
    "Email": [
        "GMAIL_EMAIL", "GMAIL_PASSWORD", "GOOGLE_APPLICATION_CREDENTIALS",
    ],
    "Security & Auth": [
        "TB_JWT_SECRET", "TB_COOKIE_SECRET", "TOKEN_SECRET",
        "CLUSTER_SECRET", "TB_R_KEY",
    ],
    "Development": [
        "GITHUB_TOKEN", "GITHUB_TOKEN_GIST", "DEV_MODULES",
    ],
}


# =================== Wizard Sections ===================

def section_header(title: str, step: int, total: int):
    """Display section header."""
    print()
    print_separator("═")
    c_print(f"  {Colors.CYAN}[{step}/{total}]{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET}")
    print_separator("═")
    print()


def wizard_app_settings(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Configure app settings."""
    app = manifest_data.get("app", {})

    app["name"] = prompt_input("Application name", app.get("name", "ToolBoxV2"))
    app["debug"] = prompt_bool("Enable debug mode?", app.get("debug", False))

    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    current_level = app.get("log_level", "INFO")
    default_idx = log_levels.index(current_level) if current_level in log_levels else 1
    app["log_level"] = prompt_choice("Log level", log_levels, default_idx)

    manifest_data["app"] = app
    return manifest_data


def wizard_database_settings(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Configure database settings based on DB module modes."""
    db = manifest_data.get("database", {})

    # Initialize sub-configs if not present
    if "local" not in db:
        db["local"] = {}
    if "redis" not in db:
        db["redis"] = {}
    if "minio" not in db:
        db["minio"] = {}

    modes = ["LC", "LR", "RR", "CB"]
    mode_descriptions = [
        "LC - Local Dict (JSON file, no external dependencies)",
        "LR - Local Redis (requires local Redis server)",
        "RR - Remote Redis (requires remote Redis server)",
        "CB - Cluster Blob (MinIO/S3 encrypted storage)",
    ]

    current_mode = db.get("mode", "LC").upper() if isinstance(db.get("mode"), str) else "LC"
    default_idx = modes.index(current_mode) if current_mode in modes else 0

    selected = prompt_choice("Database mode", mode_descriptions, default_idx)
    db["mode"] = selected.split()[0]  # Extract mode code (LC, LR, RR, CB)

    # Configure based on selected mode
    if db["mode"] == "LC":
        # Local Dict - just needs a path
        local = db.get("local", {})
        local["path"] = prompt_input(
            "Local DB path",
            local.get("path", ".data/MiniDictDB.json")
        )
        db["local"] = local

    elif db["mode"] in ("LR", "RR"):
        # Redis configuration
        redis = db.get("redis", {})

        if db["mode"] == "LR":
            c_print(f"  {Colors.DIM}Local Redis - typically redis://localhost:6379{Colors.RESET}")
        else:
            c_print(f"  {Colors.DIM}Remote Redis - use full connection URI{Colors.RESET}")

        redis["url"] = prompt_input(
            "Redis URL",
            redis.get("url", "${DB_CONNECTION_URI:-redis://localhost:6379}")
        )

        if db["mode"] == "RR":
            # Remote might need auth
            if prompt_bool("Configure Redis authentication?", default=False):
                redis["username"] = prompt_input("Redis username", redis.get("username", ""))
                redis["password"] = prompt_input("Redis password", redis.get("password", ""), password=True)

        redis["db_index"] = int(prompt_input("Redis DB index", str(redis.get("db_index", 0))))
        db["redis"] = redis

    elif db["mode"] == "CB":
        # MinIO/S3 Blob configuration
        minio = db.get("minio", {})

        c_print(f"  {Colors.DIM}Cluster Blob uses MinIO/S3 for encrypted storage{Colors.RESET}")

        minio["endpoint"] = prompt_input(
            "MinIO endpoint",
            minio.get("endpoint", "${MINIO_ENDPOINT:-localhost:9000}")
        )
        minio["access_key"] = prompt_input(
            "MinIO access key",
            minio.get("access_key", "${MINIO_ACCESS_KEY:-minioadmin}")
        )
        minio["secret_key"] = prompt_input(
            "MinIO secret key",
            minio.get("secret_key", "${MINIO_SECRET_KEY:-minioadmin}"),
            password=True
        )
        minio["bucket"] = prompt_input(
            "MinIO bucket name",
            minio.get("bucket", "toolbox-data")
        )
        minio["use_ssl"] = prompt_bool("Use SSL/TLS?", minio.get("use_ssl", False))

        # Optional cloud sync
        if prompt_bool("Configure cloud sync (backup to remote S3)?", default=False):
            minio["cloud_endpoint"] = prompt_input("Cloud endpoint", minio.get("cloud_endpoint", ""))
            minio["cloud_access_key"] = prompt_input("Cloud access key", minio.get("cloud_access_key", ""))
            minio["cloud_secret_key"] = prompt_input("Cloud secret key", minio.get("cloud_secret_key", ""), password=True)

        db["minio"] = minio

    manifest_data["database"] = db
    return manifest_data


def wizard_workers_settings(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Configure worker settings."""
    workers = manifest_data.get("workers", {})

    # HTTP Worker
    http_s = workers.get("http", [{}])
    w = []
    for http in http_s:
        http["enabled"] = prompt_bool("Enable HTTP worker?", http.get("enabled", True))
        if http["enabled"]:
            http["instances"] = int(prompt_input("HTTP worker instances", str(http.get("instances", 2))))
            http["port"] = int(prompt_input("HTTP port", str(http.get("port", 5000))))
        w.append(http)
    workers["http"] = w

    # WebSocket Worker
    ws = workers.get("ws", {})
    ws["enabled"] = prompt_bool("Enable WebSocket worker?", ws.get("enabled", True))
    if ws["enabled"]:
        ws["instances"] = int(prompt_input("WebSocket instances", str(ws.get("instances", 1))))
        ws["port"] = int(prompt_input("WebSocket port", str(ws.get("port", 6587))))
    workers["ws"] = ws

    manifest_data["workers"] = workers
    return manifest_data


def wizard_services_settings(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Configure external services."""
    services = manifest_data.get("services", {})

    # Redis
    redis = services.get("redis", {})
    redis["enabled"] = prompt_bool("Enable Redis?", redis.get("enabled", False))
    if redis["enabled"]:
        redis["host"] = prompt_input("Redis host", redis.get("host", "${REDIS_HOST:-localhost}"))
        redis["port"] = int(prompt_input("Redis port", str(redis.get("port", 6379))))
    services["redis"] = redis

    # MinIO
    minio = services.get("minio", {})
    minio["enabled"] = prompt_bool("Enable MinIO/S3?", minio.get("enabled", False))
    if minio["enabled"]:
        minio["endpoint"] = prompt_input("MinIO endpoint", minio.get("endpoint", "${MINIO_ENDPOINT:-localhost:9000}"))
    services["minio"] = minio

    manifest_data["services"] = services
    return manifest_data


def wizard_llm_providers(manifest_data: Dict[str, Any],
                         existing_env: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Interactive LLM provider selection with live API key validation.

    Returns (updated manifest_data, updated env_dict).
    """
    updated_env = existing_env.copy()

    # 1. Provider selection (multi-select via individual prompts)
    c_print(f"  {Colors.CYAN}LLM Provider Setup{Colors.RESET}")
    c_print(f"  {Colors.DIM}Select which providers you want to configure.{Colors.RESET}")
    c_print(f"  {Colors.DIM}9Router is the default — it proxies to all other providers.{Colors.RESET}")
    print()

    provider_keys = list(LLM_PROVIDERS.keys())
    selected_providers: list = []

    for pk in provider_keys:
        p = LLM_PROVIDERS[pk]
        is_default = p.get("default", False)
        already_set = any(os.getenv(v) or existing_env.get(v) for v in p["key_vars"])
        if pk == "ollama":
            already_set = bool(os.getenv("OLLAMA_BASE_URL") or existing_env.get("OLLAMA_BASE_URL"))

        label = p["name"]
        if already_set:
            label += " (already configured)"
        if is_default:
            label += " [recommended]"

        if prompt_bool(f"Configure {label}?", default=is_default):
            selected_providers.append(pk)

    if not selected_providers:
        print_status("No providers selected — skipping LLM setup", "warning")
        return manifest_data, updated_env

    # 2. Per-provider: collect keys, validate, gather models
    all_available_models: Dict[str, list] = {}  # provider_key -> [model_ids]
    validated_keys: Dict[str, str] = {}          # env_var -> key_value

    for pk in selected_providers:
        p = LLM_PROVIDERS[pk]
        print()
        c_print(f"  {Colors.CYAN}--- {p['name']} ---{Colors.RESET}")

        # URL if applicable
        base_url = p["default_url"]
        url_var = p.get("url_var")
        if url_var:
            current_url = os.getenv(url_var) or existing_env.get(url_var, "")
            base_url = prompt_input(
                f"{url_var} (base URL)",
                current_url or p["default_url"],
            )
            if base_url:
                updated_env[url_var] = base_url

        # Extra vars (e.g. 9Router user/password) — leave empty by default
        for ev in p.get("extra_vars", []):
            current = os.getenv(ev) or existing_env.get(ev, "")
            is_secret = "PASSWORD" in ev or "SECRET" in ev
            val = prompt_input(ev, current, password=is_secret)
            if val:
                updated_env[ev] = val

        # API Key(s)
        api_key = ""
        for kv in p["key_vars"]:
            current = os.getenv(kv) or existing_env.get(kv, "")
            api_key = prompt_input(kv, current, password=True)
            if api_key:
                updated_env[kv] = api_key

        # Validate
        if p["key_vars"] and not api_key:
            print_status(f"No API key for {pk} — skipping validation", "info")
            all_available_models[pk] = []
            continue

        # Live validate
        if prompt_bool(f"Validate {pk} API key now?", default=True):
            ok, msg, models = validate_llm_key(pk, api_key, base_url)
            if ok:
                print_status(f"✓ {pk}: {msg}", "success")
                all_available_models[pk] = models
            else:
                print_status(f"✗ {pk}: {msg}", "error")
                if not prompt_bool("Continue anyway?", default=True):
                    selected_providers.remove(pk)
                    continue
        else:
            all_available_models[pk] = []

    # 3. Model selection — merge defaults from all selected providers
    print()
    c_print(f"  {Colors.CYAN}Model Configuration{Colors.RESET}")
    c_print(f"  {Colors.DIM}Choose models for each task type.{Colors.RESET}")
    print()

    model_roles = ["FAST", "BLITZ", "COMPLEX", "SUMMARY"]

    # Build combined model options per role
    combined_defaults: Dict[str, str] = {}
    for pk in selected_providers:
        p = LLM_PROVIDERS[pk]
        for role in model_roles:
            if role in p["models"] and role not in combined_defaults:
                combined_defaults[role] = p["models"][role]

    # If we have live model lists, offer them as choices
    isaa = manifest_data.get("isaa") or {}
    models_cfg = isaa.get("models", {})

    env_model_map = {"FAST": "FASTMODEL", "BLITZ": "BLITZMODEL",
                     "COMPLEX": "COMPLEXMODEL", "SUMMARY": "SUMMARYMODEL"}

    for role in model_roles:
        default_model = combined_defaults.get(role, "")

        # Build choices: defaults from selected providers + any live models
        choices: list = []
        seen: set = set()

        for pk in selected_providers:
            p = LLM_PROVIDERS[pk]
            m = p["models"].get(role, "")
            if m and m not in seen:
                choices.append(m)
                seen.add(m)

        # Add live models if available
        for pk, live_models in all_available_models.items():
            prefix = pk
            for lm in live_models[:20]:
                full = f"{prefix}/{lm}" if not lm.startswith(prefix) else lm
                if full not in seen:
                    choices.append(full)
                    seen.add(full)

        current_env = os.getenv(env_model_map[role]) or existing_env.get(env_model_map[role], "")

        if len(choices) == 0:
            chosen = prompt_input(f"{role} model", current_env or default_model)
        elif len(choices) == 1:
            chosen = choices[0]
            c_print(f"  {Colors.DIM}{role}: {chosen}{Colors.RESET}")
        else:
            default_idx = 0
            if current_env in choices:
                default_idx = choices.index(current_env)
            elif default_model in choices:
                default_idx = choices.index(default_model)
            chosen = prompt_choice(f"{role} model", choices, default_idx)

        # Write to env
        if chosen:
            updated_env[env_model_map[role]] = chosen
            # Also write to manifest isaa.models
            role_key = role.lower()
            models_cfg[role_key] = chosen

    # Embedding model
    emb_choices = []
    for pk in selected_providers:
        p = LLM_PROVIDERS[pk]
        # providers don't have explicit embedding models, use default
    emb_default = os.getenv("DEFAULTMODELEMBEDDING") or existing_env.get(
        "DEFAULTMODELEMBEDDING", "gemini/text-embedding-004")
    emb = prompt_input("Embedding model", emb_default)
    if emb:
        updated_env["DEFAULTMODELEMBEDDING"] = emb
        models_cfg["embedding"] = emb

    # Persist to manifest
    isaa["models"] = models_cfg
    isaa["enabled"] = True

    # Set the primary API key env var for the agent
    primary_pk = selected_providers[0]
    primary_p = LLM_PROVIDERS[primary_pk]
    if primary_p["key_vars"]:
        primary_key_var = primary_p["key_vars"][0]
    else:
        primary_key_var = ""

    self_agent = isaa.get("self_agent", {})
    self_agent["api_key_env_var"] = primary_key_var
    isaa["self_agent"] = self_agent

    manifest_data["isaa"] = isaa

    return manifest_data, updated_env


def wizard_env_variables(root_dir: Path, existing_env: Dict[str, str],
                         template_vars: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Configure environment variables interactively."""
    updated_env = existing_env.copy()

    for category, var_names in ENV_CATEGORIES.items():
        # Check if any vars in this category need configuration
        vars_to_configure = []
        for var in var_names:
            if var in template_vars:
                current = existing_env.get(var, "")
                if not current:
                    vars_to_configure.append(var)

        if not vars_to_configure:
            continue

        # Ask if user wants to configure this category
        print()
        c_print(f"  {Colors.YELLOW}📁 {category}{Colors.RESET}")
        c_print(f"     {len(vars_to_configure)} unconfigured variable(s)")

        if not prompt_bool(f"Configure {category}?", default=False):
            continue

        print()
        for var in vars_to_configure:
            default_val, comment = template_vars[var]
            if comment:
                c_print(f"     {Colors.DIM}# {comment}{Colors.RESET}")

            # Detect if this is a secret/password
            is_secret = any(x in var.upper() for x in ["KEY", "SECRET", "PASSWORD", "TOKEN"])

            value = prompt_input(var, default_val, password=is_secret)
            if value:
                updated_env[var] = value

    return updated_env


def save_env_file(env_path: Path, env_vars: Dict[str, str]):
    """Save environment variables to .env file."""
    lines = []

    # Read existing file to preserve comments and order
    if env_path.exists():
        existing_content = env_path.read_text(encoding="utf-8")
        existing_keys = set()

        for line in existing_content.splitlines():
            if line.strip() and not line.strip().startswith("#") and "=" in line:
                key = line.split("=")[0].strip()
                existing_keys.add(key)
                # Update value if we have a new one
                if key in env_vars:
                    lines.append(f"{key}={env_vars[key]}")
                else:
                    lines.append(line)
            else:
                lines.append(line)

        # Add new variables
        new_vars = set(env_vars.keys()) - existing_keys
        if new_vars:
            lines.append("")
            lines.append("# Added by config wizard")
            for key in sorted(new_vars):
                if env_vars[key]:
                    lines.append(f"{key}={env_vars[key]}")
    else:
        # Create new file
        lines.append("# ToolBoxV2 Environment Configuration")
        lines.append("# Generated by config wizard")
        lines.append("")
        for key, value in sorted(env_vars.items()):
            if value:
                lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wizard_advanced_agent(manifest_data: Dict[str, Any],
                          existing_env: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Optional advanced agent tuning. User can skip entirely to keep schema defaults.

    Covers: Temperature, Max Tokens, Checkpoints, MCP, A2A,
            DEFAULT_MAX_ITERATIONS, DREAMER_*, NARRATOR_*.
    Returns (updated manifest_data, updated env_dict).
    """
    updated_env = existing_env.copy()

    isaa = manifest_data.get("isaa") or {}
    self_agent = isaa.get("self_agent", {})

    c_print(f"  {Colors.CYAN}Agent Behavior{Colors.RESET}")
    print()

    # Temperature
    cur_temp = str(self_agent.get("temperature", 0.7))
    val = prompt_input("Temperature (0.0-1.0)", cur_temp)
    try:
        self_agent["temperature"] = float(val)
    except ValueError:
        pass

    # Max output tokens
    cur_tok = str(self_agent.get("max_tokens_output", 2048))
    val = prompt_input("Max output tokens", cur_tok)
    try:
        self_agent["max_tokens_output"] = int(val)
    except ValueError:
        pass

    # Streaming
    self_agent["stream"] = prompt_bool(
        "Enable streaming responses?",
        self_agent.get("stream", True))

    self_agent["name"] = prompt_input(
        "Agent name",
        self_agent.get("name", "self"))

    print()
    c_print(f"  {Colors.CYAN}Iteration Limits{Colors.RESET}")
    print()

    # DEFAULT_MAX_ITERATIONS
    cur_iter = os.getenv("DEFAULT_MAX_ITERATIONS") or updated_env.get("DEFAULT_MAX_ITERATIONS", "30")
    val = prompt_input("Max iterations per task", cur_iter)
    try:
        updated_env["DEFAULT_MAX_ITERATIONS"] = str(int(val))
    except ValueError:
        pass

    # DEFAULT_MAX_HISTORY_LENGTH
    cur_hist = os.getenv("DEFAULT_MAX_HISTORY_LENGTH") or updated_env.get("DEFAULT_MAX_HISTORY_LENGTH", "100")
    val = prompt_input("Max history turns", cur_hist)
    try:
        updated_env["DEFAULT_MAX_HISTORY_LENGTH"] = str(int(val))
    except ValueError:
        pass

    print()
    c_print(f"  {Colors.CYAN}Checkpoints{Colors.RESET}")
    print()

    checkpoint = self_agent.get("checkpoint", {})
    checkpoint["enabled"] = prompt_bool(
        "Enable checkpoints (auto-save agent state)?",
        checkpoint.get("enabled", False))
    if checkpoint["enabled"]:
        cur_int = str(checkpoint.get("interval_seconds", 300))
        val = prompt_input("Checkpoint interval (seconds)", cur_int)
        try:
            checkpoint["interval_seconds"] = int(val)
        except ValueError:
            pass
        cur_max = str(checkpoint.get("max_checkpoints", 10))
        val = prompt_input("Max checkpoints to keep", cur_max)
        try:
            checkpoint["max_checkpoints"] = int(val)
        except ValueError:
            pass
    self_agent["checkpoint"] = checkpoint

    print()
    c_print(f"  {Colors.CYAN}Protocols{Colors.RESET}")
    print()

    # MCP
    mcp = isaa.get("mcp", {})
    mcp["enabled"] = prompt_bool(
        "Enable MCP (Model Context Protocol)?",
        mcp.get("enabled", True))
    isaa["mcp"] = mcp

    # A2A
    a2a = isaa.get("a2a", {})
    a2a["enabled"] = prompt_bool(
        "Enable A2A (Agent-to-Agent)?",
        a2a.get("enabled", False))
    isaa["a2a"] = a2a

    print()
    c_print(f"  {Colors.CYAN}Dreamer{Colors.RESET}")
    print()

    cur_dream_fast = os.getenv("DREAMER_FAST_MODEL") or updated_env.get("DREAMER_FAST_MODEL", "")
    val = prompt_input("Dreamer fast model (empty = inherit FASTMODEL)", cur_dream_fast)
    if val:
        updated_env["DREAMER_FAST_MODEL"] = val

    cur_dream_complex = os.getenv("DREAMER_COMPLEX_MODEL") or updated_env.get("DREAMER_COMPLEX_MODEL", "")
    val = prompt_input("Dreamer complex model (empty = inherit COMPLEXMODEL)", cur_dream_complex)
    if val:
        updated_env["DREAMER_COMPLEX_MODEL"] = val

    cur_dream_budget = os.getenv("DREAMER_BUDGET") or updated_env.get("DREAMER_BUDGET", "160000")
    val = prompt_input("Dreamer token budget", cur_dream_budget)
    try:
        updated_env["DREAMER_BUDGET"] = str(int(val))
    except ValueError:
        pass

    print()
    c_print(f"  {Colors.CYAN}Narrator{Colors.RESET}")
    print()

    cur_narr = os.getenv("NARRATOR_ENABLED") or updated_env.get("NARRATOR_ENABLED", "true")
    updated_env["NARRATOR_ENABLED"] = "true" if prompt_bool(
        "Enable Narrator?", cur_narr.lower() == "true") else "false"

    cur_lang = os.getenv("NARRATOR_LANG") or updated_env.get("NARRATOR_LANG", "auto")
    updated_env["NARRATOR_LANG"] = prompt_input("Narrator language (auto|de|en)", cur_lang)

    isaa["self_agent"] = self_agent
    manifest_data["isaa"] = isaa

    return manifest_data, updated_env


# =================== Main Wizard Function ===================

def _wizard_addon_categories(categories: Dict[str, list],
                             existing_env: Dict[str, str],
                             template_vars: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Configure add-on env categories — filtered category set."""
    updated_env = existing_env.copy()

    for category, var_names in categories.items():
        vars_to_configure = []
        for var in var_names:
            if var in template_vars:
                current = existing_env.get(var, "")
                if not current:
                    vars_to_configure.append(var)

        if not vars_to_configure:
            continue

        print()
        c_print(f"  {Colors.YELLOW}?? {category}{Colors.RESET}")
        c_print(f"     {len(vars_to_configure)} unconfigured variable(s)")

        if not prompt_bool(f"Configure {category}?", default=False):
            continue

        print()
        for var in vars_to_configure:
            default_val, comment = template_vars[var]
            if comment:
                c_print(f"     {Colors.DIM}# {comment}{Colors.RESET}")

            is_secret = any(x in var.upper() for x in ["KEY", "SECRET", "PASSWORD", "TOKEN"])

            value = prompt_input(var, default_val, password=is_secret)
            if value:
                updated_env[var] = value

    return updated_env


def run_config_wizard(root_dir: Optional[Path] = None) -> int:
    """Run the interactive configuration wizard.

    Profile is read from manifest (set during first-run onboarding).
    Never re-asks "who are you" — uses manifest.app.profile.
    """
    if root_dir is None:
        from toolboxv2 import tb_root_dir
        root_dir = tb_root_dir

    clear_screen()

    # Welcome
    print_box_header("ToolBoxV2 Configuration Wizard", "⚙️")
    print_box_content("Interactive setup for manifest and environment", "info")
    print_box_content("Press Ctrl+C at any time to cancel", "info")
    print_box_footer()

    try:
        # Load existing manifest or create default
        from toolboxv2.utils.manifest.loader import ManifestLoader
        from toolboxv2.utils.manifest.schema import TBManifest

        loader = ManifestLoader(root_dir)

        if loader.exists():
            manifest = loader.load()
            print_status(f"Loaded existing manifest: {loader.manifest_path}", "success")
        else:
            manifest = TBManifest()
            print_status("Creating new manifest configuration", "info")

        # Convert to dict for editing
        # Read profile from manifest — NEVER ask the user
        profile = None
        app_data = manifest.app
        if app_data and app_data.profile:
            profile = app_data.profile.value if hasattr(app_data.profile, 'value') else str(app_data.profile)
        if profile:
            c_print(f"  {Colors.DIM}Profile: {profile}{Colors.RESET}")
        else:
            c_print(f"  {Colors.YELLOW}No profile set (first-run incomplete). Using local defaults.{Colors.RESET}")
            profile = "local"

        core_defaults = get_profile_defaults(profile)

        manifest_data = manifest.model_dump()

        is_server = profile in ("server", "business")
        total_steps = 6 if not is_server else 7

        # Step 1: App Settings
        section_header("Application Settings", 1, total_steps)
        manifest_data = wizard_app_settings(manifest_data)

        # Step 2: Database
        section_header("Database Configuration", 2, total_steps)
        manifest_data = wizard_database_settings(manifest_data)

        # Step 3: LLM Providers (replaces old ISAA settings)
        section_header("LLM Provider Setup", 3, total_steps)

        template_path = root_dir / "env-template"
        env_path = root_dir / ".env"
        existing_env = load_existing_env(env_path)

        for k, v in core_defaults.items():
            if k not in existing_env and not os.getenv(k):
                existing_env[k] = v

        manifest_data, updated_env = wizard_llm_providers(manifest_data, existing_env)

        # Step 4: Optional Add-Ons
        section_header("Add-On Configuration", 4, total_steps)

        template_vars = parse_env_template(template_path)
        if template_vars:
            addon_categories = {
                k: v for k, v in ENV_CATEGORIES.items()
                if k not in ("AI/LLM APIs", "LLM Gateway & Providers")
            }
            if prompt_bool("Configure any add-on categories? (Bots, Search, Storage, etc.)", default=False):
                updated_env = _wizard_addon_categories(addon_categories, updated_env, template_vars)

        # Step 5: Advanced Agent Settings (optional — skippable)
        section_header("Advanced Agent Settings", 5, total_steps)
        if prompt_bool("Configure advanced agent settings? (Temperature, Checkpoints, Dreamer, Narrator)\n  Skip = keep schema defaults.",
                       default=False):
            manifest_data, updated_env = wizard_advanced_agent(manifest_data, updated_env)
        else:
            c_print(f"  {Colors.DIM}Skipped — using schema defaults.{Colors.RESET}")

        # Step 6 (server only): Workers & Services
        if is_server:
            section_header("Worker Processes", 5, total_steps)
            manifest_data = wizard_workers_settings(manifest_data)

            section_header("External Services", 6, total_steps)
            manifest_data = wizard_services_settings(manifest_data)

        # Features
        step_num = 6 if not is_server else 7
        section_header("Features", step_num, total_steps)
        from toolboxv2.feature_loader import (
            list_available_features, is_feature_installed, get_required_features
        )
        available = list_available_features()
        required = get_required_features()
        if available:
            c_print(f"  Available features: {', '.join(available)}")
            c_print(f"  Already required:   {', '.join(required) or 'core only'}")
            print()
            features_to_load = []
            for feat in available:
                if feat in ("core",) or is_feature_installed(feat):
                    continue
                if prompt_bool(f"Install feature '{feat}'?", default=False):
                    features_to_load.append(feat)
            if features_to_load:
                from toolboxv2.feature_loader import unpack_feature
                for feat in features_to_load:
                    ok = unpack_feature(feat)
                    print_status(f"Feature '{feat}': {'loaded' if ok else 'failed'}", "success" if ok else "error")
        else:
            print_status("No additional features available", "info")

        # Step 7: Environment Variables
        section_header("Environment Variables", 6, total_steps)

        template_path = root_dir / "env-template"
        env_path = root_dir / ".env"

        template_vars = parse_env_template(template_path)
        existing_env = load_existing_env(env_path)

        if template_vars:
            c_print(f"  Found {len(template_vars)} variables in env-template")
            missing = sum(1 for v in template_vars if v not in existing_env or not existing_env[v])
            c_print(f"  {missing} variables not yet configured")
            print()

            if prompt_bool("Configure environment variables?", default=missing > 0):
                updated_env = wizard_env_variables(root_dir, existing_env, template_vars)

                if updated_env != existing_env:
                    save_env_file(env_path, updated_env)
                    print_status(f"Saved environment to {env_path}", "success")
        else:
            print_status("No env-template found, skipping", "warning")

        # Step 8: Save and Apply
        section_header("Save Configuration", total_steps, total_steps)

        # Update manifest
        manifest = TBManifest(**manifest_data)
        loader.save(manifest)
        print_status(f"Saved manifest: {loader.manifest_path}", "success")

        # Ask to apply
        if prompt_bool("Generate sub-config files now?", default=True):
            from toolboxv2.utils.manifest.converter import ConfigConverter
            converter = ConfigConverter(manifest, root_dir)
            results = converter.apply_all()

            for path in results:
                print_status(f"Generated: {path}", "success")

        # Done
        print()
        print_separator("═")
        c_print(f"  {Colors.GREEN}✓ Configuration complete!{Colors.RESET}")
        print_separator("═")
        print()

        c_print("  Next steps:")
        c_print("    • Run 'tb manifest show' to view configuration")
        c_print("    • Run 'tb manifest validate' to check for errors")
        c_print("    • Run 'tb' to start the application")
        print()

        return 0

    except KeyboardInterrupt:
        print()
        print_status("Configuration cancelled", "warning")
        return 1
    except Exception as e:
        print()
        import traceback
        traceback.print_exc()
        print_status(f"Error: {e}", "error")
        return 1
