"""
DockerVFS - Docker-based execution environment for VFS

Uses ContainerManager's DockerOps as the single Docker interface.
Containers appear in the ContainerManager dashboard.

Provides:
- Isolated container execution with bidirectional VFS sync
- Safe script execution (no TB context)
- Full TB context execution (with manifest injection)
- Sub-agent launching inside containers

Author: FlowAgent V2
"""
from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DockerConfig:
    """Docker configuration"""
    base_image: str = "toolboxv2:latest"  # Dockerfile.toolbox image
    workspace_dir: str = "/workspace"
    toolboxv2_wheel_path: str | None = None  # Not needed with toolboxv2:latest
    container_name_prefix: str = "vfs_session"
    network_mode: str = "bridge"
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    auto_remove: bool = True
    port_range_start: int = 6080
    port_range_end: int = 6100
    timeout_seconds: int = 300  # 5 minutes default
    # Manifest injection for full TB context
    inject_manifest: bool = False
    # Service overrides for manifest translation (same format as ContainerManager)
    service_overrides: dict | None = None


@dataclass
class CommandResult:
    """Result of a command execution"""
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    command: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "command": self.command,
            "timestamp": self.timestamp,
            "success": self.success
        }


# =============================================================================
# DOCKER VFS
# =============================================================================

class DockerVFS:
    """
    Docker-based execution environment for VFS.

    Uses ContainerManager's DockerOps for all Docker interaction.
    Containers are labeled and visible in the ContainerManager dashboard.

    Features:
    - Container per session (non-persistent)
    - Bidirectional file sync with VFS
    - Command execution in isolated environment
    - Optional TBManifest injection for full TB context
    - Sub-agent launching
    - Web app port exposure
    """

    def __init__(
        self,
        vfs: 'VirtualFileSystemV2',
        config: DockerConfig | None = None,
        on_output: Callable[[str], None] | None = None
    ):
        """
        Initialize DockerVFS.

        Args:
            vfs: VirtualFileSystemV2 instance to sync with
            config: Docker configuration
            on_output: Callback for streaming output
        """
        self.vfs = vfs
        self.config = config or DockerConfig()
        self.on_output = on_output

        # Container state
        self._container_id: str | None = None
        self._container_name: str | None = None
        self._exposed_ports: dict[int, int] = {}  # container_port -> host_port
        self._is_running: bool = False

        # Execution history
        self._history: list[CommandResult] = []

        # Port allocation
        self._used_ports: set[int] = set()

        # Sub-agent tracking
        self._sub_agent_pids: dict[str, int] = {}  # agent_name -> PID in container

    # =========================================================================
    # DOCKER OPS ACCESS
    # =========================================================================

    @staticmethod
    def _get_ops():
        """Get the shared DockerOps singleton."""
        from toolboxv2.mods.ContainerManager.docker_ops import get_docker_ops
        return get_docker_ops()

    @staticmethod
    def _get_client():
        """Get the Docker SDK client for low-level ops (sync, exec with streams)."""
        ops = DockerVFS._get_ops()
        client = ops._get_client()
        if client is None:
            raise RuntimeError("Docker not available")
        return client

    # =========================================================================
    # CONTAINER LIFECYCLE
    # =========================================================================

    def _get_container_name(self) -> str:
        """Generate unique container name"""
        return f"{self.config.container_name_prefix}_{self.vfs.session_id}"

    def _allocate_port(self) -> int | None:
        """Allocate an available port from the configured range"""
        for port in range(self.config.port_range_start, self.config.port_range_end):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        return None

    def _release_port(self, port: int):
        """Release a previously allocated port"""
        self._used_ports.discard(port)

    async def create_container(self) -> dict:
        """
        Create and start a new Docker container.

        Returns:
            Result dict with container info
        """
        ops = self._get_ops()
        if not ops.is_available():
            return {"success": False, "error": "Docker is not available"}

        if self._is_running:
            return {"success": False, "error": "Container already running"}

        self._container_name = self._get_container_name()

        # Port allocation
        host_port = self._allocate_port()
        port_bindings = {}
        if host_port:
            port_bindings["8080/tcp"] = host_port
            self._exposed_ports[8080] = host_port

        # Volumes
        volumes = {}
        if self.config.toolboxv2_wheel_path:
            import os
            if os.path.exists(self.config.toolboxv2_wheel_path):
                wheel_name = os.path.basename(self.config.toolboxv2_wheel_path)
                volumes[self.config.toolboxv2_wheel_path] = {
                    "bind": f"/mnt/{wheel_name}", "mode": "ro"
                }

        # Optional: manifest injection for full TB context
        manifest_volume = self._prepare_manifest()
        if manifest_volume:
            volumes.update(manifest_volume)

        # Labels — makes container visible in ContainerManager dashboard
        labels = {
            "managed-by": "DockerVFS",
            "vfs-session": self.vfs.session_id,
            "container-type": "vfs_agent",
        }

        try:
            container_id = ops.create_container(
                image=self.config.base_image,
                name=self._container_name,
                command="tail -f /dev/null",  # Keep alive, exec into it
                ports=port_bindings,
                environment={"WORKSPACE": self.config.workspace_dir},
                volumes=volumes,
                mem_limit=self.config.memory_limit,
                cpu_quota=int(self.config.cpu_limit * 100000),
                working_dir=self.config.workspace_dir,
                tty=True,
                stdin_open=True,
                # Allow container to reach host services (Redis, MinIO, etc.)
                extra_hosts={"host.docker.internal": "host-gateway"},
                labels=labels,
            )

            self._container_id = container_id
            self._is_running = True

            # Install ToolboxV2 from wheel if provided (not needed with toolboxv2:latest)
            if self.config.toolboxv2_wheel_path:
                import os
                wheel_name = os.path.basename(self.config.toolboxv2_wheel_path)
                install_result = await self._exec_in_container(
                    f"pip install /mnt/{wheel_name} --quiet --break-system-packages"
                )
                if not install_result.success:
                    print(f"Warning: Failed to install ToolboxV2: {install_result.stderr}")

            # Ensure workspace dir exists
            await self._exec_in_container(f"mkdir -p {self.config.workspace_dir}")

            # Sync VFS to container
            sync_result = await self._sync_to_container()
            if not sync_result["success"]:
                return {"success": False, "error": f"Failed to sync VFS: {sync_result['error']}"}

            return {
                "success": True,
                "container_id": self._container_id,
                "container_name": self._container_name,
                "exposed_ports": self._exposed_ports,
                "manifest_injected": manifest_volume is not None,
                "message": "Container created and VFS synced"
            }

        except Exception as e:
            # Cleanup on failure
            if self._container_id:
                ops.remove(self._container_id, force=True)
                self._container_id = None
            self._is_running = False
            for port in list(self._exposed_ports.values()):
                self._release_port(port)
            self._exposed_ports.clear()
            return {"success": False, "error": str(e)}

    def _prepare_manifest(self) -> dict | None:
        """
        Prepare manifest volume mount for TB context containers.

        Returns volume dict for DockerOps.create_container() or None.
        """
        if not self.config.inject_manifest:
            return None

        try:
            from toolboxv2.mods.ContainerManager import (
                load_host_manifest,
                translate_manifest_for_container,
                write_container_manifest,
            )

            host_manifest, err = load_host_manifest()
            if host_manifest is None:
                print(f"[DockerVFS] Cannot inject manifest: {err}")
                return None

            svc_overrides = self.config.service_overrides or {}
            container_manifest = translate_manifest_for_container(
                host_manifest, svc_overrides, self._container_name,
            )
            manifest_path = write_container_manifest(
                container_manifest, self._container_name,
            )

            return {
                str(manifest_path): {"bind": "/app/tb-manifest.yaml", "mode": "ro"}
            }
        except Exception as e:
            print(f"[DockerVFS] Manifest injection failed: {e}")
            return None

    async def destroy_container(self) -> dict:
        """
        Stop and remove the container.

        Returns:
            Result dict
        """
        if not self._container_id:
            return {"success": True, "message": "No container to destroy"}

        try:
            # Sync back to VFS before destroying
            await self._sync_from_container()

            # Stop and remove via DockerOps
            ops = self._get_ops()
            ops.stop(self._container_id, timeout=10)
            ops.remove(self._container_id, force=True)

            # Release ports
            for port in list(self._exposed_ports.values()):
                self._release_port(port)
            self._exposed_ports.clear()

            # Clear sub-agent tracking
            self._sub_agent_pids.clear()

            self._container_id = None
            self._container_name = None
            self._is_running = False

            return {"success": True, "message": "Container destroyed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # COMMAND EXECUTION (INTERNAL)
    # =========================================================================

    async def _exec_in_container(
        self, command: str, timeout: int | None = None
    ) -> CommandResult:
        """Execute a command inside the container via Docker SDK."""
        if not self._container_id:
            return CommandResult(
                exit_code=-1, stdout="", stderr="Container not running",
                duration=0, command=command,
            )

        timeout = timeout or self.config.timeout_seconds

        import asyncio
        import time
        start = time.monotonic()

        try:
            # Use Docker SDK exec_create + exec_start for separate stdout/stderr
            client = self._get_client()
            container = client.containers.get(self._container_id)

            # Run in thread pool (SDK is synchronous)
            loop = asyncio.get_running_loop()

            def _run():
                exec_id = client.api.exec_create(
                    container.id,
                    ["sh", "-c", command],
                    stdout=True, stderr=True,
                    workdir=self.config.workspace_dir,
                )
                output = client.api.exec_start(exec_id["Id"], demux=True)
                inspect = client.api.exec_inspect(exec_id["Id"])
                return output, inspect

            output, inspect = await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=timeout,
            )

            duration = time.monotonic() - start
            stdout_bytes, stderr_bytes = output or (b"", b"")

            return CommandResult(
                exit_code=inspect.get("ExitCode", -1),
                stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
                stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
                duration=duration,
                command=command,
            )

        except asyncio.TimeoutError:
            return CommandResult(
                exit_code=-1, stdout="",
                stderr=f"Command timed out after {timeout}s",
                duration=timeout, command=command,
            )
        except Exception as e:
            return CommandResult(
                exit_code=-1, stdout="", stderr=str(e),
                duration=time.monotonic() - start, command=command,
            )

    # =========================================================================
    # FILE SYNCHRONIZATION (via Docker SDK put_archive / get_archive)
    # =========================================================================

    async def _sync_to_container(self) -> dict:
        """Sync all VFS files to the container."""
        if not self._container_id:
            return {"success": False, "error": "Container not running"}

        try:
            # Create tar archive of VFS contents
            tar_buffer = io.BytesIO()
            file_count = 0
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                for path, vfs_file in self.vfs.files.items():
                    if vfs_file.readonly:
                        continue
                    rel_path = path.lstrip('/')
                    if not rel_path:
                        continue
                    content = vfs_file.content.encode('utf-8')
                    tarinfo = tarfile.TarInfo(name=rel_path)
                    tarinfo.size = len(content)
                    tar.addfile(tarinfo, io.BytesIO(content))
                    file_count += 1

                for dir_path in self.vfs.directories:
                    if dir_path == "/" or self.vfs.directories[dir_path].readonly:
                        continue
                    rel_path = dir_path.lstrip('/')
                    tarinfo = tarfile.TarInfo(name=rel_path + "/")
                    tarinfo.type = tarfile.DIRTYPE
                    tar.addfile(tarinfo)

            tar_buffer.seek(0)

            # Use Docker SDK put_archive (no subprocess)
            import asyncio
            loop = asyncio.get_running_loop()
            client = self._get_client()
            container = client.containers.get(self._container_id)

            await loop.run_in_executor(
                None,
                container.put_archive, self.config.workspace_dir, tar_buffer.read()
            )

            return {"success": True, "message": f"Synced {file_count} files to container"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_from_container(self) -> dict:
        """Sync all files from the container back to VFS."""
        if not self._container_id:
            return {"success": False, "error": "Container not running"}

        try:
            import asyncio
            loop = asyncio.get_running_loop()
            client = self._get_client()
            container = client.containers.get(self._container_id)

            # Use Docker SDK get_archive
            def _get():
                bits, stat = container.get_archive(self.config.workspace_dir + "/.")
                return b"".join(bits)

            tar_bytes = await loop.run_in_executor(None, _get)

            # Extract tar and update VFS
            tar_buffer = io.BytesIO(tar_bytes)
            with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            try:
                                content = f.read().decode('utf-8')
                                vfs_path = "/" + member.name
                                self.vfs.write(vfs_path, content)
                            except UnicodeDecodeError:
                                pass  # Skip binary files
                    elif member.isdir():
                        vfs_path = "/" + member.name.rstrip('/')
                        if not self.vfs._is_directory(vfs_path):
                            self.vfs.mkdir(vfs_path, parents=True)

            return {"success": True, "message": "Container synced to VFS"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # COMMAND EXECUTION (EXPORTED TOOL — unchanged API)
    # =========================================================================

    async def run_command(
        self,
        command: str,
        timeout: int | None = None,
        sync_before: bool = True,
        sync_after: bool = True
    ) -> dict:
        """
        Run a command in the Docker container.

        This is the primary tool exported for agent use.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds
            sync_before: Sync VFS to container before execution
            sync_after: Sync container to VFS after execution

        Returns:
            Result dict with command output
        """
        # Ensure container is running
        if not self._is_running:
            create_result = await self.create_container()
            if not create_result["success"]:
                return create_result

        # Sync VFS to container
        if sync_before:
            sync_result = await self._sync_to_container()
            if not sync_result["success"]:
                return {"success": False, "error": f"Pre-sync failed: {sync_result['error']}"}

        # Execute command
        result = await self._exec_in_container(command, timeout)

        # Record in history
        self._history.append(result)

        # Stream output if callback provided
        if self.on_output:
            if result.stdout:
                self.on_output(result.stdout)
            if result.stderr:
                self.on_output(f"[STDERR] {result.stderr}")

        # Sync container to VFS
        if sync_after:
            sync_result = await self._sync_from_container()
            if not sync_result["success"]:
                return {
                    **result.to_dict(),
                    "success": result.success,
                    "sync_warning": f"Post-sync failed: {sync_result['error']}"
                }

        return {
            **result.to_dict(),
            "success": result.success
        }

    # =========================================================================
    # TB CONTEXT EXECUTION
    # =========================================================================

    async def run_with_tb_context(
        self,
        command: str,
        timeout: int | None = None,
        sync_before: bool = True,
        sync_after: bool = True,
    ) -> dict:
        """
        Run a command with full ToolBoxV2 context.

        Ensures manifest is injected and TB is available.
        Use for commands that import from toolboxv2 or use `tb` CLI.

        Args:
            command: Shell command (e.g. "python -m unittest test_module")
            timeout: Command timeout in seconds
            sync_before: Sync VFS to container before execution
            sync_after: Sync container to VFS after execution

        Returns:
            Result dict with command output
        """
        if not self._is_running:
            # Force manifest injection for TB context
            old_inject = self.config.inject_manifest
            self.config.inject_manifest = True
            create_result = await self.create_container()
            self.config.inject_manifest = old_inject
            if not create_result["success"]:
                return create_result
        elif not self.config.inject_manifest:
            # Container already running without manifest — warn
            return {
                "success": False,
                "error": "Container was created without manifest. "
                         "Destroy and recreate with inject_manifest=True, "
                         "or use run_command() for non-TB execution."
            }

        # Verify tb is available
        check = await self._exec_in_container("which tb || echo 'TB_NOT_FOUND'", timeout=10)
        if "TB_NOT_FOUND" in check.stdout:
            return {
                "success": False,
                "error": "ToolBoxV2 not installed in container. "
                         "Use base_image='toolboxv2:latest' (Dockerfile.toolbox)."
            }

        return await self.run_command(command, timeout, sync_before, sync_after)

    # =========================================================================
    # SUB-AGENT MANAGEMENT
    # =========================================================================

    async def start_sub_agent(
        self,
        agent_name: str,
        command: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:
        """
        Start a sub-agent process inside the container.

        The container must have TB context (inject_manifest=True).

        Args:
            agent_name: Identifier for this sub-agent
            command: Custom start command. Default: tb agent run
            env: Extra environment variables

        Returns:
            Result dict with agent status
        """
        if not self._is_running:
            self.config.inject_manifest = True
            create_result = await self.create_container()
            if not create_result["success"]:
                return create_result

        if agent_name in self._sub_agent_pids:
            return {
                "success": False,
                "error": f"Sub-agent '{agent_name}' already running (PID {self._sub_agent_pids[agent_name]})"
            }

        # Sync latest VFS state
        await self._sync_to_container()

        # Build command
        cmd = command or f"tb -m isaa --agent {agent_name}"
        env_str = ""
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items()) + " "

        # Start in background, capture PID
        bg_cmd = (
            f"nohup {env_str}{cmd} "
            f"> /tmp/agent_{agent_name}.log 2>&1 & echo $!"
        )
        result = await self._exec_in_container(bg_cmd, timeout=30)

        if not result.success:
            return {"success": False, "error": result.stderr}

        try:
            pid = int(result.stdout.strip())
            self._sub_agent_pids[agent_name] = pid
        except ValueError:
            return {"success": False, "error": f"Could not get PID: {result.stdout}"}

        return {
            "success": True,
            "agent_name": agent_name,
            "pid": pid,
            "log_file": f"/tmp/agent_{agent_name}.log",
            "message": f"Sub-agent '{agent_name}' started",
        }

    async def get_sub_agent_status(self, agent_name: str) -> dict:
        """
        Check status of a running sub-agent.

        Returns:
            Dict with running status, recent log output
        """
        if agent_name not in self._sub_agent_pids:
            return {"success": False, "error": f"Sub-agent '{agent_name}' not tracked"}

        pid = self._sub_agent_pids[agent_name]

        # Check if process is still running
        check = await self._exec_in_container(
            f"kill -0 {pid} 2>/dev/null && echo RUNNING || echo STOPPED",
            timeout=10,
        )
        is_running = "RUNNING" in check.stdout

        # Get recent logs
        logs = await self._exec_in_container(
            f"tail -n 50 /tmp/agent_{agent_name}.log 2>/dev/null || echo '(no logs)'",
            timeout=10,
        )

        if not is_running:
            del self._sub_agent_pids[agent_name]

        return {
            "success": True,
            "agent_name": agent_name,
            "running": is_running,
            "pid": pid,
            "recent_logs": logs.stdout,
        }

    async def stop_sub_agent(self, agent_name: str) -> dict:
        """Stop a running sub-agent."""
        if agent_name not in self._sub_agent_pids:
            return {"success": True, "message": f"Sub-agent '{agent_name}' not running"}

        pid = self._sub_agent_pids[agent_name]
        result = await self._exec_in_container(f"kill {pid} 2>/dev/null || true", timeout=10)
        del self._sub_agent_pids[agent_name]

        return {"success": True, "message": f"Sub-agent '{agent_name}' stopped (PID {pid})"}

    async def list_sub_agents(self) -> dict:
        """List all tracked sub-agents with their status."""
        agents = {}
        for name in list(self._sub_agent_pids.keys()):
            status = await self.get_sub_agent_status(name)
            agents[name] = status
        return {"success": True, "agents": agents}

    # =========================================================================
    # WEB APP SUPPORT
    # =========================================================================

    async def start_web_app(
        self,
        entrypoint: str,
        port: int = 8080,
        env: dict[str, str] | None = None
    ) -> dict:
        """
        Start a web application in the container.

        Args:
            entrypoint: Command to start the app (e.g., "python app.py")
            port: Port the app listens on inside container
            env: Environment variables

        Returns:
            Result dict with access URL
        """
        if not self._is_running:
            create_result = await self.create_container()
            if not create_result["success"]:
                return create_result

        # Sync VFS first
        await self._sync_to_container()

        # Build environment string
        env_str = ""
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items()) + " "

        # Start app in background
        bg_command = f"nohup {env_str}{entrypoint} > /tmp/app.log 2>&1 &"
        result = await self._exec_in_container(bg_command)

        if not result.success:
            return {"success": False, "error": result.stderr}

        # Wait a moment for app to start
        import asyncio
        await asyncio.sleep(2)

        # Check if app is running
        check_result = await self._exec_in_container(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/ || echo 'not_ready'"
        )

        host_port = self._exposed_ports.get(port)
        if host_port:
            return {
                "success": True,
                "url": f"http://localhost:{host_port}",
                "container_port": port,
                "host_port": host_port,
                "status": check_result.stdout.strip(),
                "message": f"Web app started on port {host_port}"
            }
        else:
            return {
                "success": True,
                "message": f"App started but no port mapping for {port}",
                "internal_url": f"http://localhost:{port}"
            }

    async def stop_web_app(self) -> dict:
        """Stop running web app"""
        if not self._is_running:
            return {"success": True, "message": "No container running"}

        await self._exec_in_container("pkill -f 'python|node' || true")
        return {"success": True, "message": "Web app stopped"}

    async def get_app_logs(self, lines: int = 100) -> dict:
        """Get web app logs"""
        if not self._is_running:
            return {"success": False, "error": "Container not running"}

        result = await self._exec_in_container(
            f"tail -n {lines} /tmp/app.log 2>/dev/null || echo 'No logs'"
        )
        return {"success": True, "logs": result.stdout}

    # =========================================================================
    # STATUS & INFO
    # =========================================================================

    def get_status(self) -> dict:
        """Get container status"""
        return {
            "is_running": self._is_running,
            "container_id": self._container_id,
            "container_name": self._container_name,
            "exposed_ports": self._exposed_ports,
            "command_history_count": len(self._history),
            "workspace_dir": self.config.workspace_dir,
            "manifest_injected": self.config.inject_manifest,
            "sub_agents": list(self._sub_agent_pids.keys()),
        }

    def get_history(self, last_n: int | None = None) -> list[dict]:
        """Get command execution history"""
        history = self._history if last_n is None else self._history[-last_n:]
        return [r.to_dict() for r in history]

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize DockerVFS state for checkpoint"""
        return {
            "config": {
                "base_image": self.config.base_image,
                "workspace_dir": self.config.workspace_dir,
                "toolboxv2_wheel_path": self.config.toolboxv2_wheel_path,
                "container_name_prefix": self.config.container_name_prefix,
                "network_mode": self.config.network_mode,
                "memory_limit": self.config.memory_limit,
                "cpu_limit": self.config.cpu_limit,
                "port_range_start": self.config.port_range_start,
                "port_range_end": self.config.port_range_end,
                "timeout_seconds": self.config.timeout_seconds,
                "inject_manifest": self.config.inject_manifest,
            },
            "history": [r.to_dict() for r in self._history[-50:]]  # Keep last 50 commands
        }

    def from_checkpoint(self, data: dict):
        """Restore from checkpoint (history only, container is not persistent)"""
        if "config" in data:
            cfg = data["config"]
            self.config = DockerConfig(**cfg)
        self._history = [
            CommandResult(**h) for h in data.get("history", [])
        ]

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def cleanup(self):
        """Clean up resources"""
        if self._is_running:
            await self.destroy_container()


# =============================================================================
# TOOL EXPORT HELPERS
# =============================================================================

def create_docker_vfs_tool(docker_vfs: DockerVFS) -> dict:
    """
    Create a tool definition for the agent — safe script execution.

    Returns a dict that can be used with add_tool.
    """
    async def run_command(command: str, timeout: int = 300) -> dict:
        """
        Execute a command in the Docker container.

        The container has your VFS files synced to /workspace.
        Changes made in the container are synced back to VFS.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (default: 300)

        Returns:
            Result with stdout, stderr, exit_code
        """
        return await docker_vfs.run_command(command, timeout=timeout)

    return {
        "function": run_command,
        "name": "docker_exec",
        "description": "Execute commands in isolated Docker container with VFS files",
        "category": ["system", "docker"],
        "flags": {"requires_container": True}
    }


def create_tb_exec_tool(docker_vfs: DockerVFS) -> dict:
    """
    Create a tool for executing commands with full TB context.

    The container has ToolBoxV2 installed and configured with
    the host's manifest (translated for Docker networking).
    """
    async def run_tb_command(command: str, timeout: int = 300) -> dict:
        """
        Execute a command with full ToolBoxV2 context in Docker.

        TB is installed, manifest is configured, DB connections are set up.
        Use for: running tests, tb CLI commands, importing toolboxv2 modules.

        Args:
            command: Shell command (e.g. "python -m unittest discover")
            timeout: Timeout in seconds

        Returns:
            Result with stdout, stderr, exit_code
        """
        return await docker_vfs.run_with_tb_context(command, timeout=timeout)

    return {
        "function": run_tb_command,
        "name": "tb_exec",
        "description": "Execute commands with full ToolBoxV2 context in Docker",
        "category": ["system", "docker", "toolbox"],
        "flags": {"requires_container": True, "requires_tb": True}
    }


def create_sub_agent_tool(docker_vfs: DockerVFS) -> dict:
    """
    Create a tool for managing sub-agents in the container.
    """
    async def manage_sub_agent(
        action: str,
        agent_name: str = "worker",
        command: str | None = None,
        env: dict | None = None,
    ) -> dict:
        """
        Manage sub-agents running inside the Docker container.

        Actions:
        - start: Launch a sub-agent process
        - status: Check if agent is running + get recent logs
        - stop: Kill the sub-agent process
        - list: List all tracked sub-agents

        Args:
            action: start | status | stop | list
            agent_name: Name identifier for the sub-agent
            command: Custom start command (for 'start' action)
            env: Extra environment variables (for 'start' action)

        Returns:
            Result dict with agent status
        """
        if action == "start":
            return await docker_vfs.start_sub_agent(agent_name, command, env)
        elif action == "status":
            return await docker_vfs.get_sub_agent_status(agent_name)
        elif action == "stop":
            return await docker_vfs.stop_sub_agent(agent_name)
        elif action == "list":
            return await docker_vfs.list_sub_agents()
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    return {
        "function": manage_sub_agent,
        "name": "sub_agent",
        "description": "Start, monitor, and stop sub-agents in Docker container",
        "category": ["system", "docker", "agent"],
        "flags": {"requires_container": True, "requires_tb": True}
    }
