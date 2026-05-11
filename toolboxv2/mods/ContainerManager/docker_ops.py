"""
Docker Abstraction Layer for ContainerManager.

Single point of contact with the Docker daemon.
All container operations go through DockerOps — nothing else imports `docker` directly.
"""

import socket
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

# Singleton instance
_ops: Optional["DockerOps"] = None


@dataclass(frozen=True)
class ContainerInfo:
    """Snapshot of a Docker container's state."""
    container_id: str
    name: str
    image: str
    status: str  # running / exited / created / dead / paused / restarting
    labels: dict = field(default_factory=dict)
    ports: dict = field(default_factory=dict)   # {"8080/tcp": 9001, ...}
    networks: list = field(default_factory=list)
    created_at: str = ""
    is_tb_managed: bool = False  # True when label managed-by == ContainerManager


@dataclass(frozen=True)
class NetworkInfo:
    """Snapshot of a Docker network."""
    network_id: str
    name: str
    driver: str
    containers: list = field(default_factory=list)  # list of container_ids

@dataclass(frozen=True)
class ServiceInfo:
    """Auto-discovered service from running Docker containers."""
    name: str
    service_type: str  # redis, minio, searxng, opentelemetry, openobserve
    internal_url: str  # Docker network URL (e.g. redis://redis:6379)
    external_url: str  # Host-reachable URL (e.g. redis://localhost:6379)
    container_id: str
    container_name: str
    is_tb_managed: bool


def get_docker_ops() -> "DockerOps":
    """Return the module-level DockerOps singleton.

    Never blocks — if Docker SDK is not installed or daemon unreachable,
    returns a DockerOps whose is_available() returns False.
    """
    global _ops
    if _ops is None:
        _ops = DockerOps()
    return _ops


def set_docker_ops(ops: "DockerOps"):
    """Replace the singleton — used by tests to inject FakeDockerOps."""
    global _ops
    _ops = ops


class DockerOps:
    """Thin wrapper around Docker SDK.  The ONLY place that talks to Docker."""

    def __init__(self):
        self._client = None
        self._available: Optional[bool] = None

    # -- connection ----------------------------------------------------------

    def _get_client(self):
        if self._available is False:
            self._available = None
        if self._client is not None:
            return self._client
        try:
            import docker
            self._client = docker.from_env(timeout=5)
            self._available = True
            return self._client
        except Exception:
            self._available = False
            return None

    def is_available(self) -> bool:
        """Check whether the Docker daemon is reachable."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.ping()
            return True
        except Exception:
            self._available = False
            self._client = None
            return False

    # -- list / inspect ------------------------------------------------------

    def list_all_containers(self, include_stopped: bool = True) -> list[ContainerInfo]:
        """Return every container the daemon knows about."""
        client = self._get_client()
        if client is None:
            return []
        try:
            raw = client.containers.list(all=include_stopped)
            return [self._to_info(c) for c in raw]
        except Exception:
            return []

    def get_container_status(self, container_id: str) -> str:
        """Return live status string or 'not_found'."""
        client = self._get_client()
        if client is None:
            return "docker_offline"
        try:
            c = client.containers.get(container_id)
            c.reload()
            return c.status
        except Exception:
            return "not_found"

    def get_container_stats(self, container_id: str) -> Optional[dict]:
        """Return one-shot stats snapshot or None."""
        client = self._get_client()
        if client is None:
            return None
        try:
            c = client.containers.get(container_id)
            raw = c.stats(stream=False)
            return self._parse_stats(raw)
        except Exception:
            return None

    def get_container_networks(self, container_id: str) -> list[str]:
        """Return list of network names this container is attached to."""
        client = self._get_client()
        if client is None:
            return []
        try:
            c = client.containers.get(container_id)
            c.reload()
            net_settings = c.attrs.get("NetworkSettings", {}).get("Networks", {})
            return list(net_settings.keys())
        except Exception:
            return []

    # -- lifecycle -----------------------------------------------------------

    def create_container(self, **kwargs) -> str:
        """Create and start a container.  Returns container_id."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Docker not available")
        c = client.containers.run(detach=True, **kwargs)
        c.reload()
        return c.id

    def start(self, container_id: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            c = client.containers.get(container_id)
            c.start()
            return True
        except Exception:
            return False

    def stop(self, container_id: str, timeout: int = 30) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            c = client.containers.get(container_id)
            c.stop(timeout=timeout)
            return True
        except Exception:
            return False

    def restart(self, container_id: str, timeout: int = 30) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            c = client.containers.get(container_id)
            c.restart(timeout=timeout)
            return True
        except Exception:
            return False

    def remove(self, container_id: str, force: bool = False) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            c = client.containers.get(container_id)
            c.remove(force=force)
            return True
        except Exception:
            return False

    def remove_volume(self, volume_name: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            client.volumes.get(volume_name).remove()
            return True
        except Exception:
            return False

    def pull_image(self, image: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            # Split image:tag
            parts = image.rsplit(":", 1)
            repo = parts[0]
            tag = parts[1] if len(parts) > 1 else "latest"
            client.images.pull(repo, tag=tag)
            return True
        except Exception:
            return False

    # -- exec / logs ---------------------------------------------------------

    def exec_run(self, container_id: str, cmd: list[str],
                 user: str = "", timeout: int = 60) -> tuple[int, str]:
        """Execute a command inside a running container.  Returns (exit_code, output)."""
        client = self._get_client()
        if client is None:
            return (-1, "Docker not available")
        try:
            c = client.containers.get(container_id)
            kwargs = {"cmd": cmd}
            if user:
                kwargs["user"] = user
            exit_code, output = c.exec_run(**kwargs)
            return (exit_code, output.decode("utf-8", errors="replace"))
        except Exception as e:
            return (-1, str(e))

    def logs(self, container_id: str, tail: int = 100) -> str:
        client = self._get_client()
        if client is None:
            return ""
        try:
            c = client.containers.get(container_id)
            return c.logs(tail=tail, timestamps=True).decode("utf-8", errors="replace")
        except Exception:
            return ""

    # -- networks ------------------------------------------------------------

    def list_networks(self) -> list[NetworkInfo]:
        client = self._get_client()
        if client is None:
            return []
        try:
            raw = client.networks.list()
            result = []
            for n in raw:
                n.reload()
                containers = list(n.attrs.get("Containers", {}).keys())
                result.append(NetworkInfo(
                    network_id=n.id,
                    name=n.name,
                    driver=n.attrs.get("Driver", ""),
                    containers=containers,
                ))
            return result
        except Exception:
            return []

    # -- health probing ------------------------------------------------------

    def health_check_http(self, host: str, port: int,
                          path: str = "/health", timeout: int = 5) -> bool:
        """HTTP GET probe — returns True if status 2xx."""
        try:
            url = f"http://{host}:{port}{path}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    # -- server IP -----------------------------------------------------------

    @staticmethod
    def get_server_ip() -> str:
        """Return the configured server IP or best-effort guess."""
        env_ip = os.getenv("CONTAINER_SERVER_IP")
        if env_ip:
            return env_ip
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _to_info(container) -> ContainerInfo:
        """Convert a docker.models.containers.Container to ContainerInfo."""
        labels = container.labels or {}
        # Parse port bindings
        ports = {}
        port_data = container.ports or {}
        for internal, bindings in port_data.items():
            if bindings:
                ports[internal] = int(bindings[0].get("HostPort", 0))

        # Networks
        net_settings = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        networks = list(net_settings.keys())

        # Image name — handle deleted images gracefully
        try:
            image = container.image.tags[0] if container.image.tags else str(container.image.id)[:19]
        except Exception:
            image = container.attrs.get("Config", {}).get("Image", "unknown")

        return ContainerInfo(
            container_id=container.id,
            name=container.name,
            image=image,
            status=container.status,
            labels=labels,
            ports=ports,
            networks=networks,
            created_at=container.attrs.get("Created", ""),
            is_tb_managed=labels.get("managed-by") == "ContainerManager",
        )

    @staticmethod
    def _parse_stats(raw: dict) -> dict:
        """Parse Docker stats into a clean dict with correct CPU delta calculation."""
        # CPU — delta between current and previous sample
        cpu_delta = (
            raw.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
            - raw.get("precpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
        )
        system_delta = (
            raw.get("cpu_stats", {}).get("system_cpu_usage", 0)
            - raw.get("precpu_stats", {}).get("system_cpu_usage", 0)
        )
        online_cpus = raw.get("cpu_stats", {}).get("online_cpus", 1)
        cpu_percent = 0.0
        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * online_cpus * 100

        # Memory
        mem_usage = raw.get("memory_stats", {}).get("usage", 0)
        mem_limit = raw.get("memory_stats", {}).get("limit", 1)
        memory_mb = mem_usage / (1024 * 1024)
        memory_percent = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0

        # Network
        net_rx = 0
        net_tx = 0
        for _iface, data in raw.get("networks", {}).items():
            net_rx += data.get("rx_bytes", 0)
            net_tx += data.get("tx_bytes", 0)

        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_mb": round(memory_mb, 1),
            "memory_percent": round(memory_percent, 2),
            "network_rx_bytes": net_rx,
            "network_tx_bytes": net_tx,
        }

    # -- service discovery ---------------------------------------------------

    def discover_services(self) -> list["ServiceInfo"]:
        """Scan running containers for known service types."""
        containers = self.list_all_containers(include_stopped=False)
        services = []
        for c in containers:
            svc_type = self.detect_service_type(c)
            if svc_type:
                host = self.get_server_ip()
                ext_port = list(c.ports.values())[0] if c.ports else 0
                port_key = list(c.ports.keys())[0].split('/')[0] if c.ports else 'unknown'
                if svc_type == 'redis':
                    internal = f"redis://{c.name}:6379"
                    external = f"redis://{host}:{ext_port}"
                elif svc_type in ('minio', 'openobserve'):
                    internal = f"http://{c.name}:{port_key}"
                    external = f"http://{host}:{ext_port}"
                else:
                    internal = f"{svc_type}://{c.name}:{port_key}"
                    external = f"{svc_type}://{host}:{ext_port}"
                services.append(ServiceInfo(
                    name=c.name,
                    service_type=svc_type,
                    internal_url=internal,
                    external_url=external,
                    container_id=c.container_id,
                    container_name=c.name,
                    is_tb_managed=c.is_tb_managed,
                ))
        return services

    def detect_service_type(self, info) -> "Optional[str]":
        """Detect service type from image name and ports."""
        img = (info.image or "").lower()
        if "redis" in img:
            return "redis"
        if "minio" in img:
            return "minio"
        if "searxng" in img:
            return "searxng"
        if "otel" in img or "opentelemetry" in img:
            return "opentelemetry"
        if "openobserve" in img:
            return "openobserve"
        port_keys = list(info.ports.keys())
        for pk in port_keys:
            port_num = int(pk.split("/")[0])
            if port_num == 6379:
                return "redis"
            if port_num in (9000, 9001):
                return "minio"
            if port_num == 8080 and "searxng" in img:
                return "searxng"
            if port_num in (4317, 4318):
                return "opentelemetry"
            if port_num == 5080:
                return "openobserve"
        return None

    def get_container_env(self, container_id: str) -> dict:
        """Read environment variables of a running container."""
        client = self._get_client()
        if client is None:
            return {}
        try:
            c = client.containers.get(container_id)
            env_list = c.attrs.get("Config", {}).get("Env", [])
            return {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in env_list if "=" in kv}
        except Exception:
            return {}
