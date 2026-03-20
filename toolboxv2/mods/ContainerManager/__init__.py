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

# Docker SDK (lazy import)
_docker_client = None
_docker_available = None


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


# ============================================================================
# CONFIG
# ============================================================================

# Container-Typen mit ihren Default-Konfigurationen
CONTAINER_TYPES = {
    "cli_v4": {
        "image": "toolboxv2:latest",
        "internal_port": 8080,
        "command": "python -m toolboxv2.flows.minicli",
        "environment": {"MODE": "cli_v4"},
        "memory_limit": "512m",
        "cpu_limit": "0.5",
    },
    "project_dev": {
        "image": "toolboxv2:dev",
        "internal_port": 8501,
        "command": "streamlit run app.py",
        "environment": {"MODE": "project_dev"},
        "memory_limit": "1g",
        "cpu_limit": "1.0",
    },
    "preview_server": {
        "image": "toolboxv2:latest",
        "internal_port": 8600,
        "command": "python -m preview_server",
        "environment": {"MODE": "preview"},
        "memory_limit": "256m",
        "cpu_limit": "0.3",
    },
    "custom": {
        "image": "toolboxv2:latest",
        "internal_port": 8080,
        "command": None,  # User-spezifisch
        "environment": {},
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
    """Get Docker SDK client or return None if unavailable"""
    global _docker_client, _docker_available

    if _docker_available is False:
        return None

    try:
        if _docker_client is None:
            import docker
            _docker_client = docker.from_env()
            _docker_available = True
        return _docker_client
    except Exception:
        _docker_available = False
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
    import subprocess, sys
    from toolboxv2 import tb_root_dir
    # Check dependencies
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.40.0"])

    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(tb_root_dir /"mods"/"ContainerManager"/"ui.py"),
        "--server.port", port,
        "--server.address", host,
        "--theme.base", "dark",
        "--theme.primaryColor", "#6366f1",
        "--theme.backgroundColor", "#0a0e17",
        "--theme.secondaryBackgroundColor", "#1a2332",
        "--theme.textColor", "#f1f5f9",
    ]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 ProjectDeveloper Studio                                 ║
║                                                              ║
║   Starting on http://{host}:{port}                          ║
║                                                              ║
║   Press Ctrl+C to stop                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Run Streamlit
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

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
) -> Result:
    """
    Erstelle einen neuen Container für einen User.

    Args:
        container_type: Typ des Containers (cli_v4, project_dev, preview_server, custom)
        user_id: User ID aus CloudM Auth
        container_name: Optionale Name für den Container
        admin_key: Admin Key für Autorisierung
        image: Optionales Image (überschreibt Default)
        command: Optionaler Command (überschreibt Default)
        environment: Optionale Environment Variables
        memory_limit: Memory Limit (z.B. "512m", "1g")
        cpu_limit: CPU Limit (z.B. "0.5", "1.0")

    Returns:
        Result mit container_id, port, url
    """
    # Admin Key Check
    if not check_admin_key(admin_key):
        return err("Invalid admin key")

    # User Validation (über CloudM Auth)
    from toolboxv2.mods.CloudM.auth.user_store import _load_user
    user = await _load_user(app, user_id)
    if not user:
        return err(f"User '{user_id}' not found in CloudM Auth")

    # Container-Typ Validierung
    if container_type not in CONTAINER_TYPES:
        return err(f"Unknown container type: {container_type}. Valid: {list(CONTAINER_TYPES.keys())}")

    # Docker Check
    docker = get_docker()
    if not docker:
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

    # Port allozieren
    port = await db_allocate_port(app)
    if not port:
        return err("No free port available in pool")

    port = await db_allocate_port(app)
    if not port:
        return err("No free port available in pool")

    ssh_port = None
    ssh_port = await db_allocate_ssh_port(app)
    if not ssh_port:
        await db_release_port(app, port)
        return err("No free SSH port available (22000-22500)")

    if ssh_public_key:
        if not ssh_public_key.startswith("ssh-"):
            await db_release_port(app, port)
            await db_release_ssh_port(app, ssh_port)
            return err("ssh_public_key must start with 'ssh-'")
        env["SSH_PUBLIC_KEY"] = ssh_public_key

    name = container_name or f"{user_id}_{container_type}_{secrets.token_hex(4)}"
    volume_name = f"container_{user_id}_{container_type}_{secrets.token_hex(4)}"

    port_bindings = {f"{internal_port}/tcp": port}
    if ssh_port:
        port_bindings["2222/tcp"] = ssh_port

    try:
        container = docker.containers.run(
            image=image,
            name=name,
            command=command or default_command,
            detach=True,
            ports=port_bindings,
            environment=env,
            volumes={volume_name: {"bind": "/data", "mode": "rw"}},
            mem_limit=memory_limit or default_memory,
            cpu_quota=int(float(cpu_limit or default_cpu) * 100000),
            restart_policy={"Name": "unless-stopped"},
            labels={
                "managed-by": "ContainerManager",
                "user-id": user_id,
                "container-type": container_type,
                "port": str(port),
                **({"ssh-port": str(ssh_port)} if ssh_port else {}),
            }
        )
        container.reload()

        spec = ContainerSpec(
            container_id=container.id,
            container_name=name,
            container_type=container_type,
            user_id=user_id,
            port=port,
            internal_port=internal_port,
            image=image,
            volume_name=volume_name,
            status=container.status,
            env=env,
            ssh_port=ssh_port or 0,
        )

        await db_set_container(app, spec)
        await db_add_user_container(app, user_id, container.id)
        await deploy_nginx_config(user_id, container_type, port)

        result_data = {
            "container_id": container.id[:12],
            "container_name": name,
            "port": port,
            "url": f"/container/{user_id}/{container_type}",
            "status": container.status,
            "image": image,
        }
        if ssh_port:
            import socket
            result_data["ssh_port"] = ssh_port
            result_data["ssh_connection"] = (
                f"ssh -i ~/.ssh/docksh_id_ed25519 -p {ssh_port} "
                f"cli@{socket.gethostbyname(socket.gethostname())}"
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
        # Alle Container aus DB laden (durch Prefix-Scan)
        # Dies erfordert eine DB-Iteration - vorerst User-spezifisch
        container_ids = []
        # TODO: Implementiere DB-Scan für CONTAINER::*
    else:
        container_ids = await db_get_user_containers(app, user_id)

    containers = []
    for cid in container_ids:
        spec = await db_get_container(app, cid)
        if spec:
            # Status von Docker aktualisieren
            docker = get_docker()
            if docker:
                try:
                    c = docker.containers.get(cid)
                    spec.status = c.status
                except Exception:
                    spec.status = "unknown"

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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung: Nur Admin oder der Besitzer
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized to access this container")

    # Status von Docker aktualisieren
    docker = get_docker()
    status = spec.status
    stats = None
    if docker:
        try:
            c = docker.containers.get(container_id)
            c.reload()
            status = c.status

            # Stats sammeln
            c_stats = c.stats(stream=False)
            if c_stats:
                stats = ContainerStats(
                    container_id=container_id,
                    last_update=time.time()
                )
                # CPU
                cpu_delta = c_stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
                system_delta = c_stats.get("cpu_stats", {}).get("system_cpu_usage", 0)
                online_cpus = c_stats.get("cpu_stats", {}).get("online_cpus", 1)
                if system_delta > 0:
                    stats.cpu_percent = (cpu_delta / system_delta) * online_cpus * 100

                # Memory
                mem_usage = c_stats.get("memory_stats", {}).get("usage", 0)
                mem_limit = c_stats.get("memory_stats", {}).get("limit", 1)
                stats.memory_mb = mem_usage / (1024 * 1024)
                stats.memory_percent = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0

                # Network
                net_rx = c_stats.get("networks", {})
                for iface, data in net_rx.items():
                    stats.network_rx_bytes += data.get("rx_bytes", 0)
                    stats.network_tx_bytes += data.get("tx_bytes", 0)
        except Exception:
            pass

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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized to delete this container")

    docker = get_docker()
    if docker:
        try:
            c = docker.containers.get(container_id)
            if c.status == "running" and not force:
                return err("Container is running. Use force=True to stop and delete.")
            c.remove(force=force)
        except Exception as e:
            return Result.default_internal_error(info=str(e))

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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    try:
        c = docker.containers.get(container_id)
        c.start()
        c.reload()
        spec.status = c.status
        await db_set_container(app, spec)
        return Result.json(data={"status": c.status})
    except Exception as e:
        return Result.default_internal_error(info=str(e))


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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    try:
        c = docker.containers.get(container_id)
        c.stop(timeout=30)
        c.reload()
        spec.status = c.status
        await db_set_container(app, spec)
        return Result.json(data={"status": c.status})
    except Exception as e:
        return Result.default_internal_error(info=str(e))


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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    try:
        c = docker.containers.get(container_id)
        c.restart(timeout=30)
        c.reload()
        spec.status = c.status
        await db_set_container(app, spec)
        return Result.json(data={"status": c.status})
    except Exception as e:
        return Result.default_internal_error(info=str(e))


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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    try:
        c = docker.containers.get(container_id)
        logs = c.logs(tail=tail, timestamps=True).decode("utf-8")
        return Result.json(data={"logs": logs, "container_id": container_id[:12]})
    except Exception as e:
        return Result.default_internal_error(info=str(e))


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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err("Container not found")

    # Authorisierung
    if not check_admin_key(admin_key) and spec.user_id != user_id:
        return err("Not authorized")

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    try:
        c = docker.containers.get(container_id)
        exit_code, output = c.exec_run(f"sh -c '{command}'", timeout=timeout)
        return Result.json(data={
            "exit_code": exit_code,
            "output": output.decode("utf-8", errors="replace"),
            "container_id": container_id[:12]
        })
    except Exception as e:
        return Result.default_internal_error(info=str(e))


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

    sites_enabled = "/etc/nginx/box-enabled"
    symlink = f"{sites_enabled}/container-{user_id}-{container_type}.conf"

    try:
        if os.path.exists(symlink):
            os.remove(symlink)
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

    # CORS (für Preview Server)
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' '*' always;

    if ($request_method = 'OPTIONS') {{
        return 204;
    }}
}}
"""


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

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    spec = await db_get_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")

    spec = await db_get_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")
    if not spec.ssh_port:
        return err("Container has no SSH port allocated. Re-create with ssh_public_key.")

    try:
        container = docker.containers.get(container_id)
        if container.status != "running":
            return err("Container must be running to inject SSH key via exec.")

        exec_result = container.exec_run(
            f"sh -c 'mkdir -p /home/cli/.ssh && "
            f"echo \"{ssh_public_key}\" >> /home/cli/.ssh/authorized_keys && "
            f"chmod 700 /home/cli/.ssh && chmod 600 /home/cli/.ssh/authorized_keys && "
            f"chown -R cli:cli /home/cli/.ssh'",
            user="root"
        )
        if exec_result.exit_code != 0:
            return err(f"exec failed: {exec_result.output.decode()}")

        import socket
        server_ip = socket.gethostbyname(socket.gethostname())
        return Result.json(data={
            "message": "SSH key added",
            "container_id": container_id,
            "ssh_connection": f"ssh -i ~/.ssh/docksh_id_ed25519 -p {spec.ssh_port} cli@{server_ip}",
            "server_ip": server_ip,
            "ssh_port": spec.ssh_port,
            "username": "cli",
        })
    except Exception as e:
        return err(f"Failed to add SSH key: {str(e)}")


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
    if not ssh_public_key or not ssh_public_key.startswith("ssh-"):
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

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    try:
        container = docker.containers.get(spec.container_id)
        if container.status != "running":
            return err("Container is not running")

        exec_result = container.exec_run(
            f"sh -c 'mkdir -p /home/cli/.ssh && "
            f"echo \"{ssh_public_key}\" >> /home/cli/.ssh/authorized_keys && "
            f"chmod 700 /home/cli/.ssh && "
            f"chmod 600 /home/cli/.ssh/authorized_keys && "
            f"chown -R cli:cli /home/cli/.ssh'",
            user="root"
        )
        if exec_result.exit_code != 0:
            return err(f"Key injection failed: {exec_result.output.decode()}")

        import socket
        server_ip = socket.gethostbyname(socket.gethostname())
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
    import socket
    server_ip = socket.gethostbyname(socket.gethostname())

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

    spec = await db_get_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")

    docker = get_docker()
    if not docker:
        return err("Docker not available")

    spec = await db_get_container(app, container_id)
    if not spec:
        return err(f"Container {container_id} not found")
    if not spec.ssh_port:
        return err("Container has no SSH port. Re-create with ssh_public_key parameter.")

    import socket
    server_ip = socket.gethostbyname(socket.gethostname())
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

    # Filter für Container mit SSH (Port 2222)
    ssh_containers = []
    for c in containers:
        try:
            docker = get_docker()
            container = docker.containers.get(c["container_id"])
            if "2222/tcp" in container.ports:
                # SSH Port holen
                ssh_port = None
                for port_binding in container.ports.get("2222/tcp", []):
                    if "HostPort" in port_binding:
                        ssh_port = int(port_binding["HostPort"])
                        break

                if ssh_port:
                    import socket
                    server_ip = socket.gethostbyname(socket.gethostname())

                    ssh_containers.append({
                        **c,
                        "ssh_enabled": True,
                        "ssh_port": ssh_port,
                        "server_ip": server_ip,
                        "connection_string": f"ssh -p {ssh_port} cli@{server_ip}"
                    })
        except Exception:
            # Container nicht verfügbar oder keine SSH
            pass

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
