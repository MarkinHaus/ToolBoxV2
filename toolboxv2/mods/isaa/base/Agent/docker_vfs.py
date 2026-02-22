"""
DockerVFS - Docker-based execution environment for VFS

Provides isolated container execution with bidirectional sync
between VFS and Docker container.

Author: FlowAgent V2
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tarfile
import tempfile
import io
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
    base_image: str = "toolboxv2:latest"  # Use local ToolBoxV2 image (built with Dockerfile.toolbox)
    workspace_dir: str = "/workspace"
    toolboxv2_wheel_path: str | None = None  # Path to ToolboxV2 wheel on host (not needed with toolboxv2:latest)
    container_name_prefix: str = "vfs_session"
    network_mode: str = "bridge"
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    auto_remove: bool = True
    port_range_start: int = 6080
    port_range_end: int = 6100
    timeout_seconds: int = 300  # 5 minutes default


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

    Features:
    - Container per session (non-persistent)
    - Bidirectional file sync with VFS
    - Command execution in isolated environment
    - ToolboxV2 pre-installed
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

    # =========================================================================
    # CONTAINER LIFECYCLE
    # =========================================================================

    async def _check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

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
        if not await self._check_docker_available():
            return {"success": False, "error": "Docker is not available"}

        if self._is_running:
            return {"success": False, "error": "Container already running"}

        self._container_name = self._get_container_name()

        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", self._container_name,
            "--network", self.config.network_mode,
            "--memory", self.config.memory_limit,
            f"--cpus={self.config.cpu_limit}",
            "-w", self.config.workspace_dir,
        ]

        # Allocate and expose ports
        host_port = self._allocate_port()
        if host_port:
            cmd.extend(["-p", f"{host_port}:8080"])
            self._exposed_ports[8080] = host_port

        # Mount ToolboxV2 wheel if provided
        if self.config.toolboxv2_wheel_path and os.path.exists(self.config.toolboxv2_wheel_path):
            wheel_name = os.path.basename(self.config.toolboxv2_wheel_path)
            cmd.extend(["-v", f"{self.config.toolboxv2_wheel_path}:/mnt/{wheel_name}:ro"])

        # Use base image
        cmd.append(self.config.base_image)

        # Keep container running
        cmd.extend(["tail", "-f", "/dev/null"])

        try:
            # Create container
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )

            if process.returncode != 0:
                return {"success": False, "error": f"Failed to create container: {stderr.decode()}"}

            self._container_id = stdout.decode().strip()
            self._is_running = True

            # Install ToolboxV2 if wheel is provided
            if self.config.toolboxv2_wheel_path:
                wheel_name = os.path.basename(self.config.toolboxv2_wheel_path)
                install_result = await self._exec_in_container(
                    f"pip install /mnt/{wheel_name} --quiet"
                )
                if not install_result.success:
                    print(f"Warning: Failed to install ToolboxV2: {install_result.stderr}")

            # Sync VFS to container
            sync_result = await self._sync_to_container()
            if not sync_result["success"]:
                return {"success": False, "error": f"Failed to sync VFS: {sync_result['error']}"}

            return {
                "success": True,
                "container_id": self._container_id,
                "container_name": self._container_name,
                "exposed_ports": self._exposed_ports,
                "message": f"Container created and VFS synced"
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout creating container"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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

            # Stop and remove container
            process = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", self._container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            # Release ports
            for port in list(self._exposed_ports.values()):
                self._release_port(port)
            self._exposed_ports.clear()

            self._container_id = None
            self._container_name = None
            self._is_running = False

            return {"success": True, "message": "Container destroyed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _exec_in_container(self, command: str, timeout: int | None = None) -> CommandResult:
        """Execute a command inside the container"""
        if not self._container_id:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr="Container not running",
                duration=0,
                command=command
            )

        timeout = timeout or self.config.timeout_seconds
        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "exec", self._container_id,
                "sh", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            duration = asyncio.get_event_loop().time() - start_time

            return CommandResult(
                exit_code=process.returncode or 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                duration=duration,
                command=command
            )

        except asyncio.TimeoutError:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                duration=timeout,
                command=command
            )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=asyncio.get_event_loop().time() - start_time,
                command=command
            )

    # =========================================================================
    # FILE SYNCHRONIZATION
    # =========================================================================

    async def _sync_to_container(self) -> dict:
        """Sync all VFS files to the container"""
        if not self._container_id:
            return {"success": False, "error": "Container not running"}

        try:
            # Create tar archive of VFS contents
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                for path, vfs_file in self.vfs.files.items():
                    if vfs_file.readonly:
                        continue

                    # Convert VFS path to relative path
                    rel_path = path.lstrip('/')
                    if not rel_path:
                        continue

                    # Add file to tar
                    content = vfs_file.content.encode('utf-8')
                    tarinfo = tarfile.TarInfo(name=rel_path)
                    tarinfo.size = len(content)
                    tar.addfile(tarinfo, io.BytesIO(content))

                # Create directories
                for dir_path in self.vfs.directories:
                    if dir_path == "/" or self.vfs.directories[dir_path].readonly:
                        continue

                    rel_path = dir_path.lstrip('/')
                    tarinfo = tarfile.TarInfo(name=rel_path + "/")
                    tarinfo.type = tarfile.DIRTYPE
                    tar.addfile(tarinfo)

            tar_buffer.seek(0)

            # Copy tar to container
            process = await asyncio.create_subprocess_exec(
                "docker", "cp", "-", f"{self._container_id}:{self.config.workspace_dir}",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate(input=tar_buffer.read())

            if process.returncode != 0:
                return {"success": False, "error": "Failed to copy files to container"}

            return {"success": True, "message": "VFS synced to container"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_from_container(self) -> dict:
        """Sync all files from the container back to VFS"""
        if not self._container_id:
            return {"success": False, "error": "Container not running"}

        try:
            # Get tar archive of workspace
            process = await asyncio.create_subprocess_exec(
                "docker", "cp", f"{self._container_id}:{self.config.workspace_dir}/.", "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return {"success": False, "error": f"Failed to copy from container: {stderr.decode()}"}

            # Extract tar and update VFS
            tar_buffer = io.BytesIO(stdout)
            with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        # Read file content
                        f = tar.extractfile(member)
                        if f:
                            try:
                                content = f.read().decode('utf-8')
                                vfs_path = "/" + member.name

                                # Update or create file in VFS
                                self.vfs.write(vfs_path, content)
                            except UnicodeDecodeError:
                                # Skip binary files
                                pass
                    elif member.isdir():
                        vfs_path = "/" + member.name.rstrip('/')
                        if not self.vfs._is_directory(vfs_path):
                            self.vfs.mkdir(vfs_path, parents=True)

            return {"success": True, "message": "Container synced to VFS"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # COMMAND EXECUTION (EXPORTED TOOL)
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
        await asyncio.sleep(2)

        # Check if app is running
        check_result = await self._exec_in_container(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/ || echo 'not_ready'")

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

        # Kill python/node processes
        await self._exec_in_container("pkill -f 'python|node' || true")

        return {"success": True, "message": "Web app stopped"}

    async def get_app_logs(self, lines: int = 100) -> dict:
        """Get web app logs"""
        if not self._is_running:
            return {"success": False, "error": "Container not running"}

        result = await self._exec_in_container(f"tail -n {lines} /tmp/app.log 2>/dev/null || echo 'No logs'")

        return {
            "success": True,
            "logs": result.stdout
        }

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
            "workspace_dir": self.config.workspace_dir
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
                "timeout_seconds": self.config.timeout_seconds
            },
            "history": [r.to_dict() for r in self._history[-50:]]  # Keep last 50 commands
        }

    def from_checkpoint(self, data: dict):
        """Restore from checkpoint (history only, container is not persistent)"""
        # Restore config
        if "config" in data:
            cfg = data["config"]
            self.config = DockerConfig(**cfg)

        # Restore history
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
# TOOL EXPORT HELPER
# =============================================================================

def create_docker_vfs_tool(docker_vfs: DockerVFS) -> dict:
    """
    Create a tool definition for the agent.

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
