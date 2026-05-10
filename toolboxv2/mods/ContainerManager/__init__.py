"""
ContainerManager - Docker Container Management via ToolBox REST API

Features:
- User-spezifische Container-Verwaltung
- Persistente Speicherung in TBEF.DB
- HTTP(S) Exposition via nginx
- Auto-Restart & Absturz-Sicherheit
- Integration mit CloudM Auth

Admin Key: CONTAINER_ADMIN_KEY Environment Variable
"""

import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from pathlib import Path

from toolboxv2 import Result, get_app, TBEF
from toolboxv2.utils.extras.base_widget import get_user_from_request
from toolboxv2.utils.system.types import ToolBoxInterfaces

export = get_app(from_="ContainerManager.EXPORT").tb
Name, version = "ContainerManager", "1.0.0"

# Admin Key (aus Environment Variable)
ADMIN_KEY = os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me")

# Docker abstraction layer (singleton)
from toolboxv2.mods.ContainerManager.docker_ops import get_docker_ops, DockerOps


@dataclass
class ContainerSpec:
    """Container-Spezifikation"""
    container_id: str
    container_name: str
    container_type: str  # cli_v4, project_dev, custom
    user_id: str
    port: int
    internal_port: int
    image: str
    volume_name: str
    status: str = "stopped"
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = 0.0
    restart_count: int = 0
    env: dict = field(default_factory=dict)
    ssh_port: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ContainerSpec":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ContainerStats:
    """Laufzeit-Statistiken"""
    container_id: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    uptime_seconds: float = 0.0
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)
# ============================================================================
# CONFIG
# ============================================================================

# Container-Typen mit ihren Default-Konfigurationen
CONTAINER_TYPES = {
    "cli_v4": {
        "image": "toolboxv2:latest",
        "internal_port": 8080,
        "command": "tb -m cli",
        "environment": {"MODE": "cli_v4"},
        "memory_limit": "512m",
        "cpu_limit": "0.5",
    },
    "custom": {
        "image": "toolboxv2:latest",
        "internal_port": 8080,
        "command": None,  # User-spezifisch
        "environment": {},
        "memory_limit": "512m",
        "cpu_limit": "0.5",
    },
    "isaa": {
        "image": "toolboxv2:latest",
        "internal_port": 5055,
        "command": "sh -c 'tb -c icli_web server & sleep 3 && tb -m icli'",
        "environment": {"MODE": "isaa"},
        "memory_limit": "1g",
        "cpu_limit": "1.0",
    },
    "tb_external": {
        "image": "toolboxv2:latest",
        "internal_port": 8080,
        "command": None,  # must be provided by user
        "environment": {"MODE": "external"},
        "memory_limit": "512m",
        "cpu_limit": "0.5",
    },
}

# Port-Pool für Container (exklusiv zur Toolbox)
PORT_POOL_START = 9000
PORT_POOL_END = 9500
SSH_PORT_POOL_START = 22000
SSH_PORT_POOL_END = 22500

# ============================================================================
# DOCKER HELPER
# ============================================================================

def get_docker():
    """Compatibility wrapper — returns DockerOps if available, else None."""
    ops = get_docker_ops()
    if ops.is_available():
        return ops
    return None


def check_admin_key(key: str) -> bool:
    """Validate admin key"""
    return key == ADMIN_KEY


def err(msg: str) -> Result:
    """Convenient error return"""
    return Result.default_user_error(info=msg)


# ============================================================================
# TBEF.DB HELPER
# ============================================================================

async def db_get_container(app, container_id: str) -> Optional[ContainerSpec]:
    """Lade Container aus TBEF.DB"""
    result = await app.a_run_any(
        TBEF.DB.GET,
        query=f"CONTAINER::{container_id}",
        get_results=True
    )
    if result.is_error() or not result.get():
        return None
    try:
        data = result.get()
        if isinstance(data, str):
            import json
            data = json.loads(data)
        return ContainerSpec.from_dict(data)
    except Exception:
        return None

async def db_resolve_container(app, container_id: str) -> tuple[Optional[ContainerSpec], str]:
    """Resolve a (possibly truncated) container_id to spec + full ID."""
    spec = await db_get_container(app, container_id)
    if spec:
        return spec, container_id
    if len(container_id) <= 12:
        result = await app.a_run_any(TBEF.DB.GET, query="CONTAINER::*", get_results=True)
        if not result.is_error() and result.get():
            import json
            for item in result.get():
                try:
                    data = json.loads(item) if isinstance(item, str) else item
                    if isinstance(data, dict) and data.get("container_id", "").startswith(container_id):
                        return ContainerSpec.from_dict(data), data["container_id"]
                except (json.JSONDecodeError, TypeError):
                    continue
    return None, container_id

async def db_set_container(app, container: ContainerSpec) -> Result:
    """Speichere Container in TBEF.DB"""
    import json
    return await app.a_run_any(
        TBEF.DB.SET,
        query=f"CONTAINER::{container.container_id}",
        data=json.dumps(container.to_dict()),
        get_results=True
    )


async def db_delete_container(app, container_id: str) -> Result:
    """Lösche Container aus TBEF.DB"""
    return await app.a_run_any(
        TBEF.DB.DELETE,
        query=f"CONTAINER::{container_id}",
        get_results=True
    )


async def db_get_user_containers(app, user_id: str) -> List[str]:
    """Lade Liste von Container-IDs für einen User"""
    result = await app.a_run_any(
        TBEF.DB.GET,
        query=f"CONTAINER_USER::{user_id}",
        get_results=True
    )
    if result.is_error() or not result.get():
        return []
    data = result.get()
    if isinstance(data, str):
        import json
        data = json.loads(data)
    return data if isinstance(data, list) else []


async def db_add_user_container(app, user_id: str, container_id: str) -> Result:
    """Füge Container-ID zur User-Liste hinzu"""
    containers = await db_get_user_containers(app, user_id)
    if container_id not in containers:
        containers.append(container_id)
    import json
    return await app.a_run_any(
        TBEF.DB.SET,
        query=f"CONTAINER_USER::{user_id}",
        data=json.dumps(containers),
        get_results=True
    )


async def db_remove_user_container(app, user_id: str, container_id: str) -> Result:
    """Entferne Container-ID aus User-Liste"""
    containers = await db_get_user_containers(app, user_id)
    if container_id in containers:
        containers.remove(container_id)
    import json
    return await app.a_run_any(
        TBEF.DB.SET,
        query=f"CONTAINER_USER::{user_id}",
        data=json.dumps(containers),
        get_results=True
    )


async def db_get_port_pool(app) -> List[int]:
    """Lade belegte Ports aus TBEF.DB"""
    result = await app.a_run_any(
        TBEF.DB.GET,
        query="CONTAINER_PORT_POOL",
        get_results=True
    )
    if result.is_error() or not result.get():
        return []
    import json
    try:
        data = result.get()
        if isinstance(data, str):
            data = json.loads(data)
        return data if isinstance(data, list) else []
    except Exception:
        return []


async def db_set_port_pool(app, ports: List[int]) -> Result:
    """Speichere belegte Ports in TBEF.DB"""
    import json
    return await app.a_run_any(
        TBEF.DB.SET,
        query="CONTAINER_PORT_POOL",
        data=json.dumps(ports),
        get_results=True
    )


async def db_allocate_port(app) -> Optional[int]:
    """Alloziere einen freien Port aus dem Pool"""
    used_ports = await db_get_port_pool(app)
    for port in range(PORT_POOL_START, PORT_POOL_END):
        if port not in used_ports:
            used_ports.append(port)
            await db_set_port_pool(app, used_ports)
            return port
    return None

async def db_release_port(app, port: int) -> Result:
    """Gebe einen Port an den Pool zurück"""
    used_ports = await db_get_port_pool(app)
    if port in used_ports:
        used_ports.remove(port)
        return await db_set_port_pool(app, used_ports)
    return Result.ok()


async def db_allocate_ssh_port(app) -> Optional[int]:
    result = await app.a_run_any(TBEF.DB.GET, query="CONTAINER_SSH_PORT_POOL", get_results=True)
    used = []
    if not result.is_error() and result.get():
        import json
        data = result.get()
        used = json.loads(data) if isinstance(data, str) else (data if isinstance(data, list) else [])
    for port in range(SSH_PORT_POOL_START, SSH_PORT_POOL_END):
        if port not in used:
            used.append(port)
            import json
            await app.a_run_any(TBEF.DB.SET, query="CONTAINER_SSH_PORT_POOL",
                                data=json.dumps(used), get_results=True)
            return port
    return None

async def db_release_ssh_port(app, port: int) -> Result:
    result = await app.a_run_any(TBEF.DB.GET, query="CONTAINER_SSH_PORT_POOL", get_results=True)
    if result.is_error() or not result.get():
        return Result.ok()
    import json
    data = result.get()
    used = json.loads(data) if isinstance(data, str) else data
    if port in used:
        used.remove(port)
        return await app.a_run_any(TBEF.DB.SET, query="CONTAINER_SSH_PORT_POOL",
                                   data=json.dumps(used), get_results=True)
    return Result.ok()

@export(mod_name=Name, version=version)
async def start_ui(port=8123, host="localhost", **_):
    from toolboxv2.mods.ContainerManager.web_ui import container_ui_app
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler

    # With waitress:
    from waitress import serve
    print(f"Starting UI at http://{host}:{port}")
    serve(FastTBHandler(container_ui_app).as_wsgi_app(enable_ws=True), host=host, port=port)

# ============================================================================
# CONTAINER MANAGEMENT API
# ============================================================================

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def create_container(
    app=None,
    request=None,
    container_type: str = "cli_v4",
    user_id: str = None,
    container_name: str = None,
    admin_key: str = None,
    image: str = None,
    command: str = None,
    environment: dict = None,
    memory_limit: str = None,
    cpu_limit: str = None,
    ssh_public_key: str = None,
    # tb_external params
    ext_source: str = None,
    ext_git_url: str = None,
    ext_git_branch: str = "main",
    ext_folder_path: str = None,
    ext_zip_url: str = None,
    # manifest
    manifest_yaml: str = None,
) -> Result:
    """
    Create a new container for a user.

    For tb_external containers, ext_source determines how the app code is loaded:
    - "image": use the specified Docker image directly
    - "git": clone a git repo into the container
    - "folder": mount a host folder into the container
    - "zip": download and extract a ZIP archive

    manifest_yaml: optional tb-manifest.yaml content to configure the container.
    """
    # Admin Key Check
    if not check_admin_key(admin_key):
        return err("Invalid admin key")

    # User Validation
    from toolboxv2.mods.CloudM.auth.user_store import _load_user
    user = await _load_user(app, user_id)
    if not user:
        return err(f"User '{user_id}' not found in CloudM Auth")

    # Container-Typ Validierung
    if container_type not in CONTAINER_TYPES:
        return err(f"Unknown container type: {container_type}. Valid: {list(CONTAINER_TYPES.keys())}")

    # tb_external: command is required
    if container_type == "tb_external" and not command:
        return err("tb_external requires a start command")

    # tb_external source validation
    if container_type == "tb_external":
        ext_source = ext_source or "image"
        if ext_source == "git" and not ext_git_url:
            return err("ext_source=git requires ext_git_url")
        if ext_source == "folder" and not ext_folder_path:
            return err("ext_source=folder requires ext_folder_path")
        if ext_source == "zip" and not ext_zip_url:
            return err("ext_source=zip requires ext_zip_url")

    # Docker Check
    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    # Config zusammenstellen
    config = CONTAINER_TYPES[container_type]
    image = image or config["image"]
    internal_port = config["internal_port"]
    default_command = config.get("command")
    default_env = config.get("environment", {}).copy()
    default_memory = config.get("memory_limit", "512m")
    default_cpu = config.get("cpu_limit", "0.5")

    # Umgebung zusammenführen
    env = {**default_env, "USER_ID": user_id}
    if environment:
        env.update(environment)

    # tb_external: build the actual command with source setup
    actual_command = command or default_command
    if container_type == "tb_external" and ext_source:
        setup_cmds = []
        if ext_source == "git":
            branch = ext_git_branch or "main"
            setup_cmds.append(f"git clone --depth 1 -b {branch} {ext_git_url} /app/external")
            setup_cmds.append("cd /app/external")
        elif ext_source == "zip":
            setup_cmds.append(f"mkdir -p /app/external && cd /app/external")
            setup_cmds.append(
                f"wget -qO /tmp/app.zip '{ext_zip_url}' && unzip -qo /tmp/app.zip -d /app/external && rm /tmp/app.zip")
        # folder mount is handled via volumes below, no setup command needed

        if setup_cmds:
            # Prepend source setup to user command
            actual_command = f"sh -c '{' && '.join(setup_cmds)} && {command}'"
        env["TB_EXT_SOURCE"] = ext_source

    # Manifest: inject as environment variable (container reads on startup)
    if manifest_yaml:
        env["TB_MANIFEST_YAML"] = manifest_yaml

    # Port allozieren
    port = await db_allocate_port(app)
    if not port:
        return err("No free port available in pool")

    ssh_port = await db_allocate_ssh_port(app)
    if not ssh_port:
        await db_release_port(app, port)
        return err("No free SSH port available (22000-22500)")

    if ssh_public_key:
        import re
        if not re.match(r'^ssh-(ed25519|rsa|ecdsa|dss) [A-Za-z0-9+/=]+ ?\S*$', ssh_public_key):
            await db_release_port(app, port)
            await db_release_ssh_port(app, ssh_port)
            return err("Invalid SSH public key format")
        if len(ssh_public_key) > 2048:
            await db_release_port(app, port)
            await db_release_ssh_port(app, ssh_port)
            return err("SSH public key too long")
        env["SSH_PUBLIC_KEY"] = ssh_public_key

    name = container_name or f"{user_id}_{container_type}_{secrets.token_hex(4)}"
    volume_name = f"container_{user_id}_{container_type}_{secrets.token_hex(4)}"

    port_bindings = {f"{internal_port}/tcp": port}
    if ssh_port:
        port_bindings["2222/tcp"] = ssh_port

    # Volumes
    volumes = {volume_name: {"bind": "/data", "mode": "rw"}}

    # tb_external folder mount
    if container_type == "tb_external" and ext_source == "folder" and ext_folder_path:
        volumes[ext_folder_path] = {"bind": "/app/external", "mode": "rw"}

    try:
        container_id = ops.create_container(
            image=image,
            name=name,
            command=actual_command,
            ports=port_bindings,
            environment=env,
            volumes=volumes,
            mem_limit=memory_limit or default_memory,
            cpu_quota=int(float(cpu_limit or default_cpu) * 100000),
            restart_policy={"Name": "unless-stopped"},
            tty=True,
            stdin_open=True,
            labels={
                "managed-by": "ContainerManager",
                "user-id": user_id,
                "container-type": container_type,
                "port": str(port),
                **({"ssh-port": str(ssh_port)} if ssh_port else {}),
                **({"ext-source": ext_source} if ext_source else {}),
            }
        )
        container_status = ops.get_container_status(container_id)

        spec = ContainerSpec(
            container_id=container_id,
            container_name=name,
            container_type=container_type,
            user_id=user_id,
            port=port,
            internal_port=internal_port,
            image=image,
            volume_name=volume_name,
            status=container_status,
            env=env,
            ssh_port=ssh_port or 0,
        )

        await db_set_container(app, spec)
        await db_add_user_container(app, user_id, container_id)
        await deploy_nginx_config(user_id, container_type, port)

        result_data = {
            "container_id": container_id[:12],
            "container_name": name,
            "port": port,
            "url": f"/container/{user_id}/{container_type}",
            "status": container_status,
            "image": image,
        }
        if ssh_port:
            server_ip = DockerOps.get_server_ip()
            result_data["ssh_port"] = ssh_port
            result_data["ssh_connection"] = (
                f"ssh -i ~/.ssh/docksh_id_ed25519 -p {ssh_port} cli@{server_ip}"
            )
        return Result.json(data=result_data)

    except Exception as e:
        await db_release_port(app, port)
        if ssh_port:
            await db_release_ssh_port(app, ssh_port)
        return Result.default_internal_error(info=str(e))


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def list_containers(
    app=None,
    request=None,
    user_id: str = None,
    admin_key: str = None,
    all: bool = False
) -> Result:
    """
    Liste Container auf.

    Args:
        user_id: User ID (optional, wenn all=True und admin_key)
        admin_key: Admin Key (für all=True)
        all: Wenn True, liste alle Container auf (nur mit admin_key)
    """
    # Wenn alle Container angefordert -> Admin Key Check
    if all:
        if not check_admin_key(admin_key):
            return err("Admin key required for listing all containers")
    else:
        # Sonst user_id erforderlich
        if not user_id:
            return err("user_id required")

    # User Container-IDs laden
    if all:
        result = await app.a_run_any(
            TBEF.DB.GET,
            query="CONTAINER::*",
            get_results=True
        )
        container_ids = []
        if not result.is_error() and result.get():
            import json
            for item in result.get():
                try:
                    data = json.loads(item) if isinstance(item, str) else item
                    if isinstance(data, dict) and "container_id" in data:
                        container_ids.append(data["container_id"])
                except (json.JSONDecodeError, TypeError):
                    continue
    else:
        container_ids = await db_get_user_containers(app, user_id)

    containers = []
    ops = get_docker_ops()
    for cid in container_ids:
        spec = await db_get_container(app, cid)
        if spec:
            # Live status from Docker via DockerOps
            spec.status = ops.get_container_status(cid)

            containers.append({
                "container_id": spec.container_id[:12],
                "container_name": spec.container_name,
                "container_type": spec.container_type,
                "user_id": spec.user_id,
                "port": spec.port,
                "url": f"/container/{spec.user_id}/{spec.container_type}",
                "status": spec.status,
                "created_at": spec.created_at,
                "uptime": time.time() - spec.created_at if spec.created_at else 0
            })

    return Result.json(data={"containers": containers})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def get_container(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None
) -> Result:
    """
    Hole Container-Details.
    """
    if not container_id:
        return err("container_id required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung: Nur Admin oder der Besitzer
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized to access this container")

    # Live status + stats via DockerOps
    ops = get_docker_ops()
    status = ops.get_container_status(container_id)
    if status == "docker_offline":
        status = spec.status  # fallback to DB
    raw_stats = ops.get_container_stats(container_id)
    stats = None
    if raw_stats:
        stats = ContainerStats(
            container_id=container_id,
            cpu_percent=raw_stats["cpu_percent"],
            memory_mb=raw_stats["memory_mb"],
            memory_percent=raw_stats["memory_percent"],
            network_rx_bytes=raw_stats["network_rx_bytes"],
            network_tx_bytes=raw_stats["network_tx_bytes"],
            last_update=time.time(),
        )

    return Result.json(data={
        "container": spec.to_dict(),
        "docker_status": status,
        "stats": stats.to_dict() if stats else None
    })


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def delete_container(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None,
    force: bool = False
) -> Result:
    """
    Lösche einen Container.
    """
    if not container_id:
        return err("container_id required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized to delete this container")

    ops = get_docker_ops()
    if ops.is_available():
        live_status = ops.get_container_status(container_id)
        if live_status == "running" and not force:
            return err("Container is running. Use force=True to stop and delete.")
        if not ops.remove(container_id, force=force):
            if live_status != "not_found":
                return Result.default_internal_error(info="Failed to remove container")
        # Clean up associated volume
        ops.remove_volume(spec.volume_name)

    # Port freigeben
    await db_release_port(app, spec.port)
    if spec.ssh_port:
        await db_release_ssh_port(app, spec.ssh_port)

    # Aus DB löschen
    await db_delete_container(app, container_id)
    await db_remove_user_container(app, spec.user_id, container_id)

    # Nginx Config entfernen
    await remove_nginx_config(spec.user_id, spec.container_type)

    return Result.json(data={
        "message": f"Container '{spec.container_name}' deleted",
        "container_id": container_id[:12]
    })


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def start_container(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None
) -> Result:
    """Starte einen gestoppten Container."""
    if not container_id:
        return err("container_id required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    if not ops.start(container_id):
        return Result.default_internal_error(info="Failed to start container")
    new_status = ops.get_container_status(container_id)
    spec.status = new_status
    await db_set_container(app, spec)
    return Result.json(data={"status": new_status})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def stop_container(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None
) -> Result:
    """Stoppe einen laufenden Container."""
    if not container_id:
        return err("container_id required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    if not ops.stop(container_id, timeout=30):
        return Result.default_internal_error(info="Failed to stop container")
    new_status = ops.get_container_status(container_id)
    spec.status = new_status
    await db_set_container(app, spec)
    return Result.json(data={"status": new_status})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def restart_container(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None
) -> Result:
    """Restarte einen Container."""
    if not container_id:
        return err("container_id required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    if not ops.restart(container_id, timeout=30):
        return Result.default_internal_error(info="Failed to restart container")
    new_status = ops.get_container_status(container_id)
    spec.status = new_status
    await db_set_container(app, spec)
    return Result.json(data={"status": new_status})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def container_logs(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None,
    tail: int = 100,
    follow: bool = False
) -> Result:
    """Hole Container-Logs."""
    if not container_id:
        return err("container_id required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    logs = ops.logs(container_id, tail=tail)
    return Result.json(data={"logs": logs, "container_id": container_id[:12]})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def container_exec(
    app=None,
    request=None,
    container_id: str = None,
    user_id: str = None,
    admin_key: str = None,
    command: str = None,
    timeout: int = 60
) -> Result:
    """Führe ein Command im Container aus."""
    if not container_id or not command:
        return err("container_id and command required")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    exit_code, output = ops.exec_run(container_id, ["sh", "-c", command], timeout=timeout)
    if exit_code == -1 and output == "Docker not available":
        return Result.default_internal_error(info=output)
    return Result.json(data={
        "exit_code": exit_code,
        "output": output,
        "container_id": container_id[:12]
    })


# ============================================================================
# NGINX CONFIG MANAGEMENT
# ============================================================================

async def deploy_nginx_config(user_id: str, container_type: str, port: int) -> Result:
    """
    Erstelle nginx Config für einen Container (nur auf Linux).
    Schreibt nach /etc/nginx/box-available/ und symlinked nach box-enabled/
    """
    import platform
    import subprocess

    if platform.system().lower() != "linux":
        return Result.ok(info="nginx config skipped (not Linux)")

    config = _generate_nginx_location(user_id, container_type, port)

    sites_available = "/etc/nginx/box-available"
    sites_enabled = "/etc/nginx/box-enabled"

    # Verzeichnisse erstellen
    import os
    os.makedirs(sites_available, exist_ok=True)
    os.makedirs(sites_enabled, exist_ok=True)

    # Config schreiben
    config_file = f"{sites_available}/container-{user_id}-{container_type}.conf"
    try:
        with open(config_file, "w") as f:
            f.write(config)
    except PermissionError:
        return Result.default_internal_error(info="Permission denied writing nginx config")

    # Symlink erstellen
    symlink = f"{sites_enabled}/container-{user_id}-{container_type}.conf"
    try:
        if not os.path.exists(symlink):
            os.symlink(config_file, symlink)
    except PermissionError:
        return Result.default_internal_error(info="Permission denied creating symlink")

    # Nginx testen und reloaden
    try:
        subprocess.run(["nginx", "-t"], check=True, capture_output=True)
        subprocess.run(["nginx", "-s", "reload"], check=False, capture_output=True)
    except FileNotFoundError:
        return Result.default_internal_error(info="nginx not found")
    except subprocess.CalledProcessError as e:
        return Result.default_internal_error(info=f"nginx reload failed: {e}")

    return Result.ok(info="nginx config deployed")


async def remove_nginx_config(user_id: str, container_type: str) -> Result:
    """Entferne nginx Config für einen Container."""
    import platform
    import os

    if platform.system().lower() != "linux":
        return Result.ok()

    sites_available = "/etc/nginx/box-available"
    sites_enabled = "/etc/nginx/box-enabled"
    symlink = f"{sites_enabled}/container-{user_id}-{container_type}.conf"
    config_file = f"{sites_available}/container-{user_id}-{container_type}.conf"

    for path in (symlink, config_file):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # Nginx reloaden
    try:
        import subprocess
        subprocess.run(["nginx", "-s", "reload"], check=False, capture_output=True)
    except Exception:
        pass

    return Result.ok()


def _generate_nginx_location(user_id: str, container_type: str, port: int) -> str:
    """Generiere nginx location-Block für einen Container."""
    return f"""
# Container {container_type} for user {user_id}
location /container/{user_id}/{container_type}/ {{
    proxy_pass http://127.0.0.1:{port}/;
    proxy_http_version 1.1;

    # WebSocket Support
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    # Headers
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Timeouts für langlaufende Prozesse (z.B. Streamlit, CLI)
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    proxy_connect_timeout 60s;

    # Buffering
    proxy_buffering off;
    proxy_request_buffering off;

    # CORS — restrict to own origin
    add_header 'Access-Control-Allow-Origin' '$scheme://$host' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type' always;
    add_header 'Access-Control-Allow-Credentials' 'true' always;

    if ($request_method = 'OPTIONS') {{
        return 204;
    }}
}}
"""


# ============================================================================
# NEW API ENDPOINTS — docker health, all containers, update
# ============================================================================

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def docker_health(app=None, request=None, admin_key: str = None) -> Result:
    """Return Docker daemon status and container count."""
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")

    ops = get_docker_ops()
    available = ops.is_available()
    all_containers = ops.list_all_containers() if available else []
    tb_count = sum(1 for c in all_containers if c.is_tb_managed)
    ext_count = len(all_containers) - tb_count

    return Result.json(data={
        "docker_available": available,
        "status": "online" if available else "offline",
        "total_containers": len(all_containers),
        "tb_managed": tb_count,
        "external": ext_count,
    })


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def list_all_docker_containers(app=None, request=None, admin_key: str = None) -> Result:
    """
    List ALL Docker containers (not just TB-managed).
    Each container is tagged as 'tb_managed' or 'external'.
    """
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")

    ops = get_docker_ops()
    if not ops.is_available():
        return Result.json(data={"containers": [], "docker_available": False})

    all_containers = ops.list_all_containers()
    result = []
    for c in all_containers:
        entry = {
            "container_id": c.container_id[:12],
            "container_id_full": c.container_id,
            "name": c.name,
            "image": c.image,
            "status": c.status,
            "ports": c.ports,
            "networks": c.networks,
            "is_tb_managed": c.is_tb_managed,
            "labels": c.labels,
        }
        result.append(entry)

    return Result.json(data={"containers": result, "docker_available": True})


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def update_container(
    app=None,
    request=None,
    container_id: str = None,
    admin_key: str = None,
    pull: bool = True,
) -> Result:
    """
    Update a container to the latest image without losing config.

    1. Read spec from DB (ports, volumes, env, image)
    2. Pull latest image (if pull=True)
    3. Stop + remove old container (keep volume!)
    4. Create new container with same spec
    5. Update DB with new container_id
    """
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    # Step 1: Pull new image (skip for local-only images)
    if pull:
        if not ops.pull_image(spec.image):
            # Image might be local-only (built via tb docker-image)
            # Check if it exists locally before failing
            try:
                client = ops._get_client()
                client.images.get(spec.image)
            except Exception:
                return err(f"Failed to pull image: {spec.image}. Build locally with: tb docker-image")

    # Step 2: Stop and remove old container (keep volume!)
    ops.stop(container_id, timeout=30)
    if not ops.remove(container_id, force=True):
        live_status = ops.get_container_status(container_id)
        if live_status != "not_found":
            return err("Failed to remove old container")

    # Step 3: Recreate with same config
    port_bindings = {f"{spec.internal_port}/tcp": spec.port}
    if spec.ssh_port:
        port_bindings["2222/tcp"] = spec.ssh_port

    try:
        config = CONTAINER_TYPES.get(spec.container_type, CONTAINER_TYPES["custom"])
        new_id = ops.create_container(
            image=spec.image,
            name=spec.container_name,
            command=config.get("command"),
            ports=port_bindings,
            environment=spec.env,
            volumes={spec.volume_name: {"bind": "/data", "mode": "rw"}},
            mem_limit=config.get("memory_limit", "512m"),
            cpu_quota=int(float(config.get("cpu_limit", "0.5")) * 100000),
            restart_policy={"Name": "unless-stopped"},
            labels={
                "managed-by": "ContainerManager",
                "user-id": spec.user_id,
                "container-type": spec.container_type,
                "port": str(spec.port),
                **({"ssh-port": str(spec.ssh_port)} if spec.ssh_port else {}),
            },
        )
    except Exception as e:
        return Result.default_internal_error(info=f"Failed to recreate container: {e}")

    # Step 4: Update DB — delete old, save new
    await db_delete_container(app, container_id)
    new_status = ops.get_container_status(new_id)
    new_spec = ContainerSpec(
        container_id=new_id,
        container_name=spec.container_name,
        container_type=spec.container_type,
        user_id=spec.user_id,
        port=spec.port,
        internal_port=spec.internal_port,
        image=spec.image,
        volume_name=spec.volume_name,
        status=new_status,
        env=spec.env,
        ssh_port=spec.ssh_port,
    )
    await db_set_container(app, new_spec)

    # Update user container list: remove old, add new
    await db_remove_user_container(app, spec.user_id, container_id)
    await db_add_user_container(app, spec.user_id, new_id)

    # Re-deploy nginx config (port unchanged, just in case)
    await deploy_nginx_config(spec.user_id, spec.container_type, spec.port)

    return Result.json(data={
        "message": f"Container '{spec.container_name}' updated",
        "old_container_id": container_id[:12],
        "new_container_id": new_id[:12],
        "status": new_status,
        "image": spec.image,
    })


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def reconcile_status(
    app=None,
    request=None,
    container_id: str = None,
    admin_key: str = None,
) -> Result:
    """
    Reconcile a single container's status: read live from Docker, update DB.
    Used by the frontend polling loop (one container per call).
    """
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")
    if not container_id:
        return err("container_id required")

    ops = get_docker_ops()
    new_status = ops.get_container_status(container_id)

    # Update DB if we have a spec
    spec, container_id = await db_resolve_container(app, container_id)
    if spec and spec.status != new_status:
        spec.status = new_status
        await db_set_container(app, spec)

    return Result.json(data={
        "container_id": container_id[:12],
        "status": new_status,
        "docker_available": ops.is_available(),
    })


# ============================================================================
# CLI COMMANDS
# ============================================================================

@export(mod_name=Name, version=version)
async def container_create_cli(
    app=None,
    container_type: str = "cli_v4",
    user_id: str = None,
    admin_key: str = None,
    **kwargs
) -> Result:
    """CLI Wrapper für create_container"""
    key = admin_key or os.getenv("CONTAINER_ADMIN_KEY")
    if not key:
        print("❌ CONTAINER_ADMIN_KEY not set")
        return err("Admin key missing")

    result = await create_container(
        app=app,
        container_type=container_type,
        user_id=user_id,
        admin_key=key,
        **kwargs
    )

    if result.is_error():
        print(f"❌ {result.info}")
    else:
        data = result.get()
        print(f"✅ Container created: {data['container_id']}")
        print(f"   URL: {data['url']}")
        print(f"   Port: {data['port']}")

    return result


@export(mod_name=Name, version=version)
async def container_list_cli(
    app=None,
    user_id: str = None,
    admin_key: str = None
) -> Result:
    """CLI Wrapper für list_containers"""
    key = admin_key or os.getenv("CONTAINER_ADMIN_KEY")
    if not key:
        print("❌ CONTAINER_ADMIN_KEY not set")
        return err("Admin key missing")

    result = await list_containers(app=app, user_id=user_id, admin_key=key)

    if result.is_error():
        print(f"❌ {result.info}")
    else:
        containers = result.get().get("containers", [])
        if not containers:
            print("📭 No containers found")
        else:
            for c in containers:
                status_icon = "🟢" if c["status"] == "running" else "⚫"
                print(f"{status_icon} {c['container_id']} {c['container_name']} ({c['container_type']}) - {c['url']}")

    return result


@export(mod_name=Name, version=version)
async def container_delete_cli(
    app=None,
    container_id: str = None,
    admin_key: str = None,
    force: bool = False
) -> Result:
    """CLI Wrapper für delete_container"""
    key = admin_key or os.getenv("CONTAINER_ADMIN_KEY")
    if not key:
        print("❌ CONTAINER_ADMIN_KEY not set")
        return err("Admin key missing")

    result = await delete_container(
        app=app,
        container_id=container_id,
        admin_key=key,
        force=force
    )

    if result.is_error():
        print(f"❌ {result.info}")
    else:
        print(f"✅ {result.get().get('message')}")

    return result


@export(mod_name=Name, version=version)
async def generate_admin_key(app=None, name: str = "admin") -> Result:
    """
    Generiere einen neuen Admin Key.

    Dies sollte beim ersten Start ausgeführt werden und das Ergebnis
    in die Environment Variable CONTAINER_ADMIN_KEY gesetzt werden.
    """
    import secrets
    key = f"cm-{secrets.token_urlsafe(32)}"
    print(f"\n{'='*60}")
    print(f"GENERATED ADMIN KEY FOR {name.upper()}:")
    print(f"{'='*60}")
    print(f"export CONTAINER_ADMIN_KEY={key}")
    print(f"{'='*60}\n")
    return Result.json(data={"admin_key": key})


# ============================================================================
# DOCKSH SSH INTEGRATION
# ============================================================================

@export(mod_name=Name, version=version, api=True)
async def add_ssh_key_to_container(
    app=None,
    container_id: str = None,
    ssh_public_key: str = None,
    admin_key: str = None
) -> Result:
    """
    Füge einen SSH Public Key zu einem Container hinzu.
    Ermöglicht SSH-Zugriff via Docksh für autorisierte User.

    Args:
        container_id: Die Container ID
        ssh_public_key: Der SSH Public Key (ssh-ed25519 AAAA...)
        admin_key: Admin Key für Authentifizierung

    Returns:
        Result mit Informationen über den SSH-Zugang
    """
    # Admin Key prüfen
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")

    if not spec.ssh_port:
        return err("Container has no SSH port allocated. Re-create with ssh_public_key.")

    live_status = ops.get_container_status(container_id)
    if live_status != "running":
        return err("Container must be running to inject SSH key via exec.")

    import base64
    key_b64 = base64.b64encode(ssh_public_key.encode()).decode()
    exit_code, output = ops.exec_run(
        container_id,
        ["sh", "-c",
         f"mkdir -p /home/cli/.ssh && "
         f"echo {key_b64} | base64 -d >> /home/cli/.ssh/authorized_keys && "
         f"chmod 700 /home/cli/.ssh && chmod 600 /home/cli/.ssh/authorized_keys && "
         f"chown -R cli:cli /home/cli/.ssh"],
        user="root",
    )
    if exit_code != 0:
        return err(f"exec failed: {output}")

    server_ip = DockerOps.get_server_ip()
    return Result.json(data={
        "message": "SSH key added",
        "container_id": container_id,
        "ssh_connection": f"ssh -i ~/.ssh/docksh_id_ed25519 -p {spec.ssh_port} cli@{server_ip}",
        "server_ip": server_ip,
        "ssh_port": spec.ssh_port,
        "username": "cli",
    })


@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def register_ssh_key(
    app=None,
    request=None,
    ssh_public_key: str = None,
) -> Result:
    """
    User-facing: registriert den eigenen SSH Public Key am zugewiesenen Container.
    Auth über CloudM Session (kein admin_key nötig).
    Gibt SSH-Verbindungsinfos zurück.
    """
    import re
    if not ssh_public_key or not re.match(r'^ssh-(ed25519|rsa|ecdsa|dss) [A-Za-z0-9+/=]+ ?\S*$', ssh_public_key):
        return err("Invalid SSH public key format")
    if len(ssh_public_key) > 2048:
        return err("ssh_public_key required and must start with 'ssh-'")

    # CloudM Auth: user_id aus dem Request-Token holen
    user_id = await get_user_from_request(app, request)
    if not user_id:
        return err("Not authenticated")

    # Container des Users laden
    container_ids = await db_get_user_containers(app, user_id)
    if not container_ids:
        return err("No container assigned to this user. Contact admin.")

    # Ersten laufenden Container nehmen (oder gezielt per container_id filtern)
    spec = None
    for cid in container_ids:
        s = await db_get_container(app, cid)
        if s and s.ssh_port:
            spec = s
            break

    if not spec:
        return err("No SSH-enabled container found. Admin must create container with ssh_public_key support.")
    if spec.user_id != user_id:
        return err("Not authorized")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    try:
        container_status = ops.get_container_status(spec.container_id)
        if container_status != "running":
            return err("Container is not running")

        import base64
        key_b64 = base64.b64encode(ssh_public_key.encode()).decode()
        exit_code, output = ops.exec_run(
            spec.container_id,
            ["sh", "-c",
             f"mkdir -p /home/cli/.ssh && "
             f"echo {key_b64} | base64 -d >> /home/cli/.ssh/authorized_keys && "
             f"chmod 700 /home/cli/.ssh && "
             f"chmod 600 /home/cli/.ssh/authorized_keys && "
             f"chown -R cli:cli /home/cli/.ssh"],
            user="root",
        )
        if exit_code != 0:
            return err(f"Key injection failed: {output}")

        server_ip = DockerOps.get_server_ip()
        return Result.json(data={
            "message": "SSH key registered",
            "ssh_port": spec.ssh_port,
            "server_ip": server_ip,
            "username": "cli",
            "connection_string": (
                f"ssh -i ~/.ssh/docksh_id_ed25519 -p {spec.ssh_port} cli@{server_ip}"
            ),
        })

    except Exception as e:
        return err(str(e))

@export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def get_my_ssh_info(
    app=None,
    request=None,
) -> Result:
    """
    User-facing: gibt SSH-Verbindungsinfos für den eigenen Container zurück.
    Kein admin_key nötig — nur gültige CloudM Session.
    """
    user_id = await get_user_from_request(app, request)
    if not user_id:
        return err("Not authenticated")

    container_ids = await db_get_user_containers(app, user_id)
    if not container_ids:
        return err("No container assigned")

    results = []
    server_ip = DockerOps.get_server_ip()

    for cid in container_ids:
        spec = await db_get_container(app, cid)
        if not spec or spec.user_id != user_id:
            continue
        entry = {
            "container_id": spec.container_id[:12],
            "container_name": spec.container_name,
            "container_type": spec.container_type,
            "status": spec.status,
            "http_url": f"/container/{user_id}/{spec.container_type}",
            "ssh_enabled": bool(spec.ssh_port),
        }
        if spec.ssh_port:
            entry["ssh_port"] = spec.ssh_port
            entry["connection_string"] = (
                f"ssh -i ~/.ssh/docksh_id_ed25519 -p {spec.ssh_port} cli@{server_ip}"
            )
        results.append(entry)

    return Result.json(data={"containers": results, "server_ip": server_ip})

@export(mod_name=Name, version=version, api=True)
async def get_container_ssh_info(
    app=None,
    container_id: str = None,
    admin_key: str = None
) -> Result:
    """
    Hole SSH-Verbindungsinfos für einen Container.

    Args:
        container_id: Die Container ID
        admin_key: Admin Key für Authentifizierung

    Returns:
        Result mit SSH-Verbindungsinformationen
    """
    # Admin Key prüfen
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")

    ops = get_docker_ops()
    if not ops.is_available():
        return err("Docker not available")

    spec, container_id = await db_resolve_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")
    if not spec.ssh_port:
        return err("Container has no SSH port. Re-create with ssh_public_key parameter.")

    server_ip = DockerOps.get_server_ip()
    return Result.json(data={
        "container_id": container_id,
        "container_name": spec.container_name,
        "user_id": spec.user_id,
        "container_type": spec.container_type,
        "ssh_enabled": True,
        "ssh_port": spec.ssh_port,
        "server_ip": server_ip,
        "connection_string": (
            f"ssh -i ~/.ssh/docksh_id_ed25519 -p {spec.ssh_port} cli@{server_ip}"
        ),
        "username": "cli",
    })

@export(mod_name=Name, version=version, api=True)
async def list_ssh_containers(
    app=None,
    user_id: str = None,
    admin_key: str = None
) -> Result:
    """
    Liste alle Container mit SSH-Unterstützung.

    Args:
        user_id: Optional - nur Container eines Users
        admin_key: Admin Key für Authentifizierung

    Returns:
        Result mit Liste der SSH-fähigen Container
    """
    # Admin Key prüfen
    if not check_admin_key(admin_key or ""):
        return err("Invalid admin key")

    result = await list_containers(app=app, user_id=user_id, admin_key=admin_key, all=True)
    if result.is_error():
        return result

    containers = result.get().get("containers", [])

    # Filter for containers with SSH from DB specs
    ssh_containers = []
    ops = get_docker_ops()
    server_ip = DockerOps.get_server_ip()
    for c in containers:
        cid_full = c["container_id"]
        # Look up spec in DB for ssh_port info
        spec = await db_get_container(app, cid_full)
        if spec and spec.ssh_port:
            ssh_containers.append({
                **c,
                "ssh_enabled": True,
                "ssh_port": spec.ssh_port,
                "server_ip": server_ip,
                "connection_string": f"ssh -p {spec.ssh_port} cli@{server_ip}"
            })

    return Result.json(data={
        "containers": ssh_containers,
        "count": len(ssh_containers)
    })


@export(mod_name=Name, version=version)
async def container_ssh_cli(
    app=None,
    container_id: str = None,
    admin_key: str = None
) -> Result:
    """
    CLI Wrapper für SSH-Zugriff auf Container.
    Zeigt Verbindungsinformationen und öffnet SSH.

    Args:
        container_id: Die Container ID
        admin_key: Admin Key für Authentifizierung
    """
    key = admin_key or os.getenv("CONTAINER_ADMIN_KEY")
    if not key:
        print("❌ CONTAINER_ADMIN_KEY not set")
        return err("Admin key missing")

    result = await get_container_ssh_info(app=app, container_id=container_id, admin_key=key)
    if result.is_error():
        print(f"❌ {result.info}")
        return result

    info = result.get()
    print(f"\n{'='*60}")
    print(f"🔑 SSH ACCESS FOR CONTAINER: {info['container_name']}")
    print(f"{'='*60}")
    print(f"Container ID: {info['container_id']}")
    print(f"User ID: {info['user_id']}")
    print(f"SSH Port: {info['ssh_port']}")
    print(f"Server: {info['server_ip']}")
    print(f"Username: {info['username']}")
    print(f"\n📋 Connection String:")
    print(f"   {info['connection_string']}")
    print(f"{'='*60}\n")

    # Versuche SSH zu öffnen (wenn ssh-client verfügbar)
    import shutil
    if shutil.which("ssh"):
        import subprocess
        try:
            print("🔗 Opening SSH connection... (Exit with 'Ctrl+b, d' or 'exit')")
            ssh_cmd = [
                "ssh",
                "-p", str(info['ssh_port']),
                f"cli@{info['server_ip']}"
            ]
            subprocess.run(ssh_cmd)
        except KeyboardInterrupt:
            print("\n✋ SSH connection closed")
    else:
        print("⚠️  SSH client not found. Please install OpenSSH to connect directly.")

    return result

