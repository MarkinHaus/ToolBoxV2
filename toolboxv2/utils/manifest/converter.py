"""
ConfigConverter - Generate sub-configs from tb-manifest.yaml
=============================================================

Generates:
- .config.yaml (Python workers)
- bin/config.toml (Rust server)
- services.json (auto-start services)
- nginx site config

IMPORTANT: NEVER generates or overwrites .env!
Only appends missing values to .env if they don't exist.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from .schema import TBManifest, DatabaseMode


class ConfigConverter:
    """
    Convert tb-manifest.yaml to various sub-config formats.

    CRITICAL: .env is NEVER overwritten or regenerated!
    Only missing values can be appended.
    """

    def __init__(self, manifest: TBManifest, base_dir: Optional[Path] = None):
        """
        Initialize converter.

        Args:
            manifest: The loaded TBManifest
            base_dir: Base directory for output files
        """
        self.manifest = manifest
        self.base_dir = base_dir or Path.cwd()
        self._generated_files: List[Path] = []

    @property
    def generated_files(self) -> List[Path]:
        """List of files generated in last apply() call."""
        return self._generated_files

    def apply_all(self) -> List[Path]:
        """
        Apply manifest to all sub-configs.

        Returns:
            List of generated/updated file paths
        """
        self._generated_files = []

        # Generate each config
        self._generate_worker_config()
        self._generate_rust_config()
        self._generate_services_json()

        # Generate ISAA agent config if enabled
        if self.manifest.isaa and self.manifest.isaa.enabled:
            self._generate_agent_config()

        # Only suggest missing env vars, never overwrite
        self._suggest_env_vars()

        return self._generated_files

    def _generate_worker_config(self) -> Path:
        """Generate .config.yaml for Python workers."""
        m = self.manifest

        # Get first HTTP worker for primary config
        http_worker = m.workers.http[0] if m.workers.http else None
        ws_worker = m.workers.websocket[0] if m.workers.websocket else None

        config = {
            "environment": m.app.environment.value,
            "debug": m.app.debug,
            "log_level": m.app.log_level.value,
            "data_dir": m.paths.data_dir,

            "zmq": {
                "pub_endpoint": m.services.zmq.pub_endpoint,
                "sub_endpoint": m.services.zmq.sub_endpoint,
                "req_endpoint": m.services.zmq.req_endpoint,
                "http_to_ws_endpoint": m.services.zmq.http_to_ws_endpoint,
                "hwm_send": m.services.zmq.hwm_send,
                "hwm_recv": m.services.zmq.hwm_recv,
            },

            "session": {
                "cookie_name": m.auth.session.cookie_name,
                "cookie_secret": m.auth.session.cookie_secret,
                "cookie_max_age": m.auth.session.cookie_max_age,
                "cookie_secure": m.auth.session.cookie_secure,
                "cookie_httponly": m.auth.session.cookie_httponly,
                "cookie_samesite": m.auth.session.cookie_samesite,
            },

            "auth": {
                "auth_enabled": m.auth.provider.value in ("custom", "clerk"),
                "ws_require_auth": m.auth.ws_require_auth,
                "ws_allow_anonymous": m.auth.ws_allow_anonymous,
            },
        }

        # HTTP worker config
        if http_worker:
            config["http_worker"] = {
                "host": http_worker.host,
                "port": http_worker.port,
                "workers": http_worker.workers,
                "max_concurrent": http_worker.max_concurrent,
                "timeout": http_worker.timeout,
            }

        # WS worker config
        if ws_worker:
            config["ws_worker"] = {
                "host": ws_worker.host,
                "port": ws_worker.port,
                "max_connections": ws_worker.max_connections,
                "ping_interval": ws_worker.ping_interval,
                "ping_timeout": ws_worker.ping_timeout,
                "compression": ws_worker.compression,
            }

        # Nginx config
        config["nginx"] = {
            "enabled": m.nginx.enabled,
            "server_name": m.nginx.server_name,
            "listen_port": m.nginx.listen_port,
            "ssl_enabled": m.nginx.ssl_enabled,
            "static_root": m.paths.dist_dir,  # Use dist_dir from paths!
            "static_enabled": m.nginx.static_enabled,
            "rate_limit_enabled": m.nginx.rate_limit_enabled,
            "rate_limit_zone": m.nginx.rate_limit_zone,
            "rate_limit_rate": m.nginx.rate_limit_rate,
            "rate_limit_burst": m.nginx.rate_limit_burst,
        }

        # Manager config
        config["manager"] = {
            "web_ui_enabled": m.services.manager.web_ui_enabled,
            "web_ui_host": m.services.manager.web_ui_host,
            "web_ui_port": m.services.manager.web_ui_port,
            "health_check_interval": m.services.manager.health_check_interval,
            "restart_delay": m.services.manager.restart_delay,
            "max_restart_attempts": m.services.manager.max_restart_attempts,
        }

        # ToolBox integration
        config["toolbox"] = {
            "instance_id": m.app.instance_id,
            "modules_preload": m.mods.init_modules,
            "api_prefix": m.toolbox.api_prefix,
            "api_allowed_mods": m.mods.open_modules,
        }

        # Write file
        output_path = self.base_dir / ".config.yaml"

        header = "# AUTO-GENERATED from tb-manifest.yaml - DO NOT EDIT DIRECTLY\n"
        header += "# Regenerate with: tb manifest apply\n\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        self._generated_files.append(output_path)
        return output_path

    def _generate_rust_config(self) -> Path:
        """Generate bin/config.toml for Rust server."""
        m = self.manifest

        # Resolve dist_dir to absolute path
        dist_dir = m.paths.dist_dir
        if dist_dir.startswith("${"):
            # Keep env var reference
            pass
        elif not os.path.isabs(dist_dir):
            dist_dir = str((self.base_dir / dist_dir).resolve())

        # Build TOML content manually for better formatting
        lines = [
            "# AUTO-GENERATED from tb-manifest.yaml - DO NOT EDIT DIRECTLY",
            "# Regenerate with: tb manifest apply",
            "",
            "[server]",
            f'ip = "0.0.0.0"',
            f'port = 8080',
            f'dist_path = "{dist_dir}"',
            f'open_modules = {json.dumps(m.mods.open_modules)}',
            f'init_modules = {json.dumps(m.mods.init_modules)}',
            f'watch_modules = {json.dumps(m.mods.watch_modules)}',
            "",
            "[toolbox]",
            f'client_prefix = "{m.toolbox.client_prefix}"',
            f'timeout_seconds = {m.toolbox.timeout_seconds}',
            f'max_instances = {m.toolbox.max_instances}',
            "",
            "[session]",
            f'secret_key = "{m.auth.session.cookie_secret}"',
            f'duration_minutes = {m.auth.session.cookie_max_age // 60}',
            "",
        ]

        content = "\n".join(lines)

        # Write to bin/config.toml
        output_path = self.base_dir / "bin" / "config.toml"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._generated_files.append(output_path)
        return output_path

    def _generate_services_json(self) -> Path:
        """Generate services.json for auto-start configuration."""
        m = self.manifest

        services_config = {
            "version": "1.0.0",
            "autostart": {
                "enabled": m.autostart.enabled,
                "services": m.autostart.services,
                "commands": m.autostart.commands,
            },
            "services": {
                "enabled": m.services.enabled,
            },
            "workers": {
                "http": [
                    {
                        "name": w.name,
                        "host": w.host,
                        "port": w.port,
                        "workers": w.workers,
                        "ssl": w.ssl,
                    }
                    for w in m.workers.http
                ],
                "websocket": [
                    {
                        "name": w.name,
                        "host": w.host,
                        "port": w.port,
                        "max_connections": w.max_connections,
                    }
                    for w in m.workers.websocket
                ],
            },
            "database": {
                "mode": m.database.mode.value,
            },
        }

        output_path = self.base_dir / "services.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(services_config, f, indent=2)

        self._generated_files.append(output_path)
        return output_path

    def _generate_agent_config(self) -> Path:
        """
        Generate agent.json for ISAA FlowAgent.

        This creates the agent configuration in the standard location:
        .data/{app_id}/Agents/{agent_name}/agent.json
        """
        m = self.manifest
        isaa = m.isaa
        if not isaa:
            return None

        agent = isaa.self_agent
        models = isaa.models

        # Build agent config matching FlowAgentBuilder.AgentConfig format
        agent_config = {
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "fast_llm_model": models.fast,
            "complex_llm_model": models.complex,
            "system_message": agent.system_message,
            "temperature": agent.temperature,
            "max_tokens_output": agent.max_tokens_output,
            "max_tokens_input": agent.max_tokens_input,
            "vfs_max_window_lines": agent.vfs_max_window_lines,
            "api_key_env_var": agent.api_key_env_var,
            "use_fast_response": agent.use_fast_response,
            "mcp": {
                "enabled": isaa.mcp.enabled,
                "config_path": None,
                "server_name": None,
                "host": "0.0.0.0",
                "port": 8000,
                "auto_expose_tools": True,
            },
            "a2a": {
                "enabled": isaa.a2a.enabled,
                "host": "0.0.0.0",
                "port": 5000,
                "agent_name": None,
                "agent_description": None,
                "agent_version": "1.0.0",
                "expose_tools_as_skills": True,
            },
            "checkpoint": {
                "enabled": agent.checkpoint.enabled,
                "interval_seconds": agent.checkpoint.interval_seconds,
                "max_checkpoints": agent.checkpoint.max_checkpoints,
                "checkpoint_dir": agent.checkpoint.checkpoint_dir,
                "auto_save_on_exit": agent.checkpoint.auto_save_on_exit,
                "auto_load_on_start": agent.checkpoint.auto_load_on_start,
                "max_age_hours": agent.checkpoint.max_age_hours,
            },
            "rate_limiter": {
                "enable_rate_limiting": agent.rate_limiter.enable_rate_limiting,
                "enable_model_fallback": agent.rate_limiter.enable_model_fallback,
                "enable_key_rotation": agent.rate_limiter.enable_key_rotation,
                "key_rotation_mode": agent.rate_limiter.key_rotation_mode,
                "api_keys": {},
                "fallback_chains": {},
                "custom_limits": {},
                "max_retries": agent.rate_limiter.max_retries,
                "wait_if_all_exhausted": agent.rate_limiter.wait_if_all_exhausted,
            },
            "max_parallel_tasks": agent.max_parallel_tasks,
            "verbose_logging": agent.verbose_logging,
            "stream": agent.stream,
            "active_persona": None,
            "persona_profiles": {},
            "world_model": {},
            "rule_config_path": None,
        }

        # Determine output path - use .data directory structure
        # Try to get app data_dir, fallback to .data/main
        try:
            from toolboxv2 import get_app
            data_dir = Path(get_app().data_dir)
        except Exception:
            data_dir = self.base_dir / ".data" / "main"

        agents_dir = data_dir / "Agents" / agent.name
        agents_dir.mkdir(parents=True, exist_ok=True)

        output_path = agents_dir / "agent.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(agent_config, f, indent=2)

        self._generated_files.append(output_path)
        return output_path

    def _suggest_env_vars(self) -> Dict[str, str]:
        """
        Check for missing environment variables and suggest them.

        IMPORTANT: This method NEVER overwrites .env!
        It only returns suggestions for missing values.

        Returns:
            Dict of suggested env var name -> suggested value
        """
        m = self.manifest
        suggestions: Dict[str, str] = {}

        # Check required env vars based on config
        required_vars = []

        # Database mode requirements
        if m.database.mode in (DatabaseMode.LR, DatabaseMode.RR):
            required_vars.append(("DB_CONNECTION_URI", "redis://localhost:6379"))

        if m.database.mode == DatabaseMode.CB:
            required_vars.extend([
                ("MINIO_ENDPOINT", "localhost:9000"),
                ("MINIO_ACCESS_KEY", "minioadmin"),
                ("MINIO_SECRET_KEY", "minioadmin"),
            ])

        # Auth requirements
        if m.auth.provider.value in ("custom", "clerk"):
            required_vars.extend([
                ("TB_JWT_SECRET", ""),
            ])

        # Session secret
        required_vars.append(("TB_COOKIE_SECRET", ""))

        # Check which are missing
        for var_name, default in required_vars:
            if not os.environ.get(var_name):
                suggestions[var_name] = default

        return suggestions

    def append_missing_env_vars(self, env_path: Optional[Path] = None) -> List[str]:
        """
        Append ONLY missing environment variables to .env file.

        CRITICAL: This method NEVER overwrites existing values!
        It only appends new variables that don't exist.

        Args:
            env_path: Path to .env file. Defaults to base_dir/.env

        Returns:
            List of variable names that were added
        """
        if env_path is None:
            env_path = self.base_dir / ".env"

        # Get suggestions
        suggestions = self._suggest_env_vars()
        if not suggestions:
            return []

        # Read existing .env if it exists
        existing_vars: Set[str] = set()
        existing_content = ""

        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
                for line in existing_content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        var_name = line.split("=", 1)[0].strip()
                        existing_vars.add(var_name)

        # Find truly missing vars
        missing = {k: v for k, v in suggestions.items() if k not in existing_vars}

        if not missing:
            return []

        # Append missing vars
        with open(env_path, "a", encoding="utf-8") as f:
            if existing_content and not existing_content.endswith("\n"):
                f.write("\n")

            f.write("\n# Added by tb manifest apply\n")
            for var_name, value in missing.items():
                f.write(f"{var_name}={value}\n")

        return list(missing.keys())

