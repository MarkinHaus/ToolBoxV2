"""
Interactive Configuration Wizard for ToolBoxV2

A beautiful, minimal, and repeatable CLI wizard that:
1. Configures tb-manifest.yaml settings
2. Sets up missing environment variables from env-template
3. Can be run multiple times to update configuration
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cli_printing import (
    print_box_header,
    print_box_footer,
    print_box_content,
    print_status,
    print_separator,
    c_print,
    Colors,
)


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
    print()
    c_print(f"  {prompt}")
    print()

    for idx, choice in enumerate(choices):
        marker = f"{Colors.CYAN}▶{Colors.RESET}" if idx == default else " "
        c_print(f"  {marker} {idx + 1}. {choice}")

    print()

    try:
        selection = input(f"{Colors.CYAN}❯{Colors.RESET} Choose [1-{len(choices)}] (default: {default + 1}): ").strip()

        if not selection:
            return choices[default]

        idx = int(selection) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

        print_status(f"Please enter 1-{len(choices)}", "warning")
        return prompt_choice(prompt, choices, default)
    except (ValueError, KeyboardInterrupt, EOFError):
        return choices[default]


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
        "DB_CONNACTION_URI", "DB_MODE_KEY", "IS_OFFLINE_DB", "SERVER_ID", "DB_CACHE_TTL",
    ],
    "Storage (MinIO/S3)": [
        "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY",
        "CLOUD_ENDPOINT", "CLOUD_ACCESS_KEY", "CLOUD_SECRET_KEY",
    ],
    "AI/LLM APIs": [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
        "GROQ_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY",
    ],
    "AI Extras": [
        "HUGGINGFACEHUB_API_TOKEN", "ELEVENLABS_API_KEY", "REPLICATE_API_TOKEN",
        "DEEPGRAM_API_KEY", "WOLFRAM_ALPHA_APPID",
    ],
    "Search & Web": [
        "SERP_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID",
        "BING_SUBSCRIPTION_KEY", "BING_SEARCH_URL",
    ],
    "Messaging & Bots": [
        "DISCORD_BOT_TOKEN", "TELEGRAM_BOT_TOKEN","DISCORD_AID",
        "TELEGRAM_AID",
        "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_API_TOKEN",
    ],
    "Email": [
        "GMAIL_EMAIL", "GMAIL_PASSWORD", "GOOGLE_CREDENTIALS_FILE",
    ],
    "Security & Auth": [
        "TB_COOKIE_SECRET", "TOKEN_SECRET", "CLUSTER_SECRET", "CLOUDM_JWT_SECRET", "CLOUDM_AUTH_URL",
        "TB_JWT_SECRET","TB_COOKIE_SECRET"
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
    http = workers.get("http", {})
    http["enabled"] = prompt_bool("Enable HTTP worker?", http.get("enabled", True))
    if http["enabled"]:
        http["instances"] = int(prompt_input("HTTP worker instances", str(http.get("instances", 2))))
        http["port"] = int(prompt_input("HTTP port", str(http.get("port", 5000))))
    workers["http"] = http

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


def wizard_isaa_settings(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Configure ISAA agent settings."""
    # Check if ISAA should be configured
    if not prompt_bool("Configure ISAA AI Agent?", default=True):
        manifest_data["isaa"] = None
        return manifest_data

    isaa = manifest_data.get("isaa") or {}
    if not isaa:
        isaa = {"enabled": True}

    isaa["enabled"] = True

    # Models configuration
    models = isaa.get("models", {})
    c_print(f"  {Colors.CYAN}LLM Models{Colors.RESET}")
    c_print(f"  {Colors.DIM}Format: provider/model (e.g., openrouter/anthropic/claude-3-haiku){Colors.RESET}")
    print()

    models["fast"] = prompt_input(
        "Fast model (quick responses)",
        models.get("fast", "${FASTMODEL:-openrouter/anthropic/claude-3-haiku}")
    )
    models["complex"] = prompt_input(
        "Complex model (reasoning)",
        models.get("complex", "${COMPLEXMODEL:-openrouter/openai/gpt-4o}")
    )
    isaa["models"] = models

    # Self-agent configuration
    self_agent = isaa.get("self_agent", {})
    print()
    c_print(f"  {Colors.CYAN}Agent Settings{Colors.RESET}")

    self_agent["name"] = prompt_input(
        "Agent name",
        self_agent.get("name", "self")
    )
    self_agent["temperature"] = float(prompt_input(
        "Temperature (0.0-1.0)",
        str(self_agent.get("temperature", 0.7))
    ))
    self_agent["max_tokens_output"] = int(prompt_input(
        "Max output tokens",
        str(self_agent.get("max_tokens_output", 2048))
    ))
    self_agent["api_key_env_var"] = prompt_input(
        "API key environment variable",
        self_agent.get("api_key_env_var", "OPENROUTER_API_KEY")
    )
    self_agent["stream"] = prompt_bool(
        "Enable streaming responses?",
        self_agent.get("stream", True)
    )

    # Checkpoint configuration
    checkpoint = self_agent.get("checkpoint", {})
    if prompt_bool("Configure checkpoints (auto-save agent state)?", default=False):
        checkpoint["enabled"] = prompt_bool("Enable checkpoints?", checkpoint.get("enabled", True))
        if checkpoint["enabled"]:
            checkpoint["interval_seconds"] = int(prompt_input(
                "Checkpoint interval (seconds)",
                str(checkpoint.get("interval_seconds", 300))
            ))
            checkpoint["max_checkpoints"] = int(prompt_input(
                "Max checkpoints to keep",
                str(checkpoint.get("max_checkpoints", 10))
            ))
    self_agent["checkpoint"] = checkpoint

    isaa["self_agent"] = self_agent

    # MCP configuration
    mcp = isaa.get("mcp", {})
    mcp["enabled"] = prompt_bool("Enable MCP (Model Context Protocol)?", mcp.get("enabled", True))
    isaa["mcp"] = mcp

    # A2A configuration
    a2a = isaa.get("a2a", {})
    a2a["enabled"] = prompt_bool("Enable A2A (Agent-to-Agent)?", a2a.get("enabled", False))
    isaa["a2a"] = a2a

    manifest_data["isaa"] = isaa
    return manifest_data


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


# =================== Main Wizard Function ===================

def run_config_wizard(root_dir: Optional[Path] = None) -> int:
    """Run the interactive configuration wizard."""
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
        manifest_data = manifest.model_dump()

        total_steps = 7

        # Step 1: App Settings
        section_header("Application Settings", 1, total_steps)
        manifest_data = wizard_app_settings(manifest_data)

        # Step 2: Database
        section_header("Database Configuration", 2, total_steps)
        manifest_data = wizard_database_settings(manifest_data)

        # Step 3: Workers
        section_header("Worker Processes", 3, total_steps)
        manifest_data = wizard_workers_settings(manifest_data)

        # Step 4: Services
        section_header("External Services", 4, total_steps)
        manifest_data = wizard_services_settings(manifest_data)

        # Step 5: ISAA Agent
        section_header("ISAA AI Agent", 5, total_steps)
        manifest_data = wizard_isaa_settings(manifest_data)

        # Step 6: Environment Variables
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

        # Step 7: Save and Apply
        section_header("Save Configuration", 7, total_steps)

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
        print_status(f"Error: {e}", "error")
        return 1
