"""
LSP Manager - Language Server Protocol integration for VFS

Handles automatic download, installation, and management of LSP servers
for code diagnostics, hints, and error detection.

Author: FlowAgent V2
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# DIAGNOSTICS
# =============================================================================

class DiagnosticSeverity(Enum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class Position:
    """Position in a text document (0-indexed)"""
    line: int
    character: int


@dataclass
class Range:
    """A range in a text document"""
    start: Position
    end: Position


@dataclass
class Diagnostic:
    """A diagnostic, such as an error or warning"""
    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    code: str | int | None = None
    source: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "range": {
                "start": {"line": self.range.start.line, "character": self.range.start.character},
                "end": {"line": self.range.end.line, "character": self.range.end.character}
            },
            "message": self.message,
            "severity": self.severity.name.lower(),
            "code": self.code,
            "source": self.source
        }
    
    def to_display_string(self, content_lines: list[str] | None = None) -> str:
        """Format diagnostic for display"""
        severity_icons = {
            DiagnosticSeverity.ERROR: "❌",
            DiagnosticSeverity.WARNING: "⚠️",
            DiagnosticSeverity.INFORMATION: "ℹ️",
            DiagnosticSeverity.HINT: "💡"
        }
        
        icon = severity_icons.get(self.severity, "•")
        line_num = self.range.start.line + 1
        col = self.range.start.character + 1
        
        result = f"{icon} Line {line_num}:{col} - {self.message}"
        
        if self.source:
            result += f" [{self.source}]"
        
        if content_lines and 0 <= self.range.start.line < len(content_lines):
            line_content = content_lines[self.range.start.line]
            result += f"\n   │ {line_content}"
            # Add caret
            caret_pos = self.range.start.character
            caret_len = max(1, self.range.end.character - self.range.start.character)
            result += f"\n   │ {' ' * caret_pos}{'^' * caret_len}"
        
        return result


# =============================================================================
# LSP SERVER DEFINITIONS
# =============================================================================

@dataclass
class LSPServerConfig:
    """Configuration for an LSP server"""
    name: str
    language_ids: list[str]
    install_command: list[str]  # Command to install the server
    start_command: list[str]    # Command to start the server
    install_check: str          # Command/path to check if installed
    package_manager: str = "pip"  # pip, npm, cargo, etc.
    requires_workspace: bool = False
    initialization_options: dict[str, Any] = field(default_factory=dict)


# LSP Server Registry
LSP_SERVERS: dict[str, LSPServerConfig] = {
    "pylsp": LSPServerConfig(
        name="Python Language Server",
        language_ids=["python"],
        install_command=[sys.executable, "-m", "pip", "install", "python-lsp-server[all]"],
        start_command=[sys.executable, "-m", "pylsp"],
        install_check="pylsp",
        package_manager="pip"
    ),
    "typescript-language-server": LSPServerConfig(
        name="TypeScript Language Server",
        language_ids=["javascript", "javascriptreact", "typescript", "typescriptreact"],
        install_command=["npm", "install", "-g", "typescript-language-server", "typescript"],
        start_command=["typescript-language-server", "--stdio"],
        install_check="typescript-language-server",
        package_manager="npm"
    ),
    "rust-analyzer": LSPServerConfig(
        name="Rust Analyzer",
        language_ids=["rust"],
        install_command=["rustup", "component", "add", "rust-analyzer"],
        start_command=["rust-analyzer"],
        install_check="rust-analyzer",
        package_manager="rustup",
        requires_workspace=True
    ),
    "gopls": LSPServerConfig(
        name="Go Language Server",
        language_ids=["go"],
        install_command=["go", "install", "golang.org/x/tools/gopls@latest"],
        start_command=["gopls"],
        install_check="gopls",
        package_manager="go"
    ),
    "clangd": LSPServerConfig(
        name="Clang Language Server",
        language_ids=["c", "cpp"],
        install_command=["apt-get", "install", "-y", "clangd"],  # Linux, adjust for other OS
        start_command=["clangd"],
        install_check="clangd",
        package_manager="system"
    ),
    "yaml-language-server": LSPServerConfig(
        name="YAML Language Server",
        language_ids=["yaml"],
        install_command=["npm", "install", "-g", "yaml-language-server"],
        start_command=["yaml-language-server", "--stdio"],
        install_check="yaml-language-server",
        package_manager="npm"
    ),
    "vscode-html-language-server": LSPServerConfig(
        name="HTML Language Server",
        language_ids=["html"],
        install_command=["npm", "install", "-g", "vscode-langservers-extracted"],
        start_command=["vscode-html-language-server", "--stdio"],
        install_check="vscode-html-language-server",
        package_manager="npm"
    ),
    "vscode-css-language-server": LSPServerConfig(
        name="CSS Language Server",
        language_ids=["css", "scss", "less"],
        install_command=["npm", "install", "-g", "vscode-langservers-extracted"],
        start_command=["vscode-css-language-server", "--stdio"],
        install_check="vscode-css-language-server",
        package_manager="npm"
    ),
    "vscode-json-language-server": LSPServerConfig(
        name="JSON Language Server",
        language_ids=["json", "jsonc"],
        install_command=["npm", "install", "-g", "vscode-langservers-extracted"],
        start_command=["vscode-json-language-server", "--stdio"],
        install_check="vscode-json-language-server",
        package_manager="npm"
    ),
    "bash-language-server": LSPServerConfig(
        name="Bash Language Server",
        language_ids=["shellscript"],
        install_command=["npm", "install", "-g", "bash-language-server"],
        start_command=["bash-language-server", "start"],
        install_check="bash-language-server",
        package_manager="npm"
    ),
    "taplo": LSPServerConfig(
        name="Taplo TOML Language Server",
        language_ids=["toml"],
        install_command=["cargo", "install", "taplo-cli", "--features", "lsp"],
        start_command=["taplo", "lsp", "stdio"],
        install_check="taplo",
        package_manager="cargo"
    ),
}


# =============================================================================
# LSP MANAGER
# =============================================================================

class LSPManager:
    """
    Manages LSP servers for code diagnostics.
    
    Features:
    - Automatic server installation
    - Server lifecycle management
    - Diagnostic retrieval
    - Caching for performance
    """
    
    def __init__(
        self,
        cache_dir: str | None = None,
        auto_install: bool = True,
        timeout: float = 30.0
    ):
        """
        Initialize LSP Manager.
        
        Args:
            cache_dir: Directory for caching LSP data
            auto_install: Automatically install missing LSP servers
            timeout: Timeout for LSP operations in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "vfs_lsp"
        self.auto_install = auto_install
        self.timeout = timeout
        
        # Server processes
        self._servers: dict[str, asyncio.subprocess.Process] = {}
        self._server_locks: dict[str, asyncio.Lock] = {}
        
        # Installation status cache
        self._installed: dict[str, bool] = {}
        
        # Diagnostic cache: (file_path, content_hash) -> diagnostics
        self._diagnostic_cache: dict[tuple[str, str], list[Diagnostic]] = {}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_server_for_language(self, language_id: str) -> str | None:
        """Get the LSP server name for a language"""
        for server_name, config in LSP_SERVERS.items():
            if language_id in config.language_ids:
                return server_name
        return None
    
    def _is_installed(self, server_name: str) -> bool:
        """Check if LSP server is installed"""
        if server_name in self._installed:
            return self._installed[server_name]
        
        config = LSP_SERVERS.get(server_name)
        if not config:
            return False
        
        # Check if command exists
        installed = shutil.which(config.install_check) is not None
        self._installed[server_name] = installed
        
        return installed
    
    async def _install_server(self, server_name: str) -> bool:
        """Install an LSP server"""
        config = LSP_SERVERS.get(server_name)
        if not config:
            return False
        
        try:
            # Run install command
            process = await asyncio.create_subprocess_exec(
                *config.install_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minute timeout for installation
            )
            
            if process.returncode == 0:
                self._installed[server_name] = True
                return True
            else:
                print(f"Failed to install {server_name}: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            print(f"Timeout installing {server_name}")
            return False
        except Exception as e:
            print(f"Error installing {server_name}: {e}")
            return False
    
    async def _ensure_server_available(self, server_name: str) -> bool:
        """Ensure LSP server is available, installing if needed"""
        if self._is_installed(server_name):
            return True
        
        if self.auto_install:
            return await self._install_server(server_name)
        
        return False
    
    async def _start_server(self, server_name: str) -> asyncio.subprocess.Process | None:
        """Start an LSP server process"""
        if server_name not in self._server_locks:
            self._server_locks[server_name] = asyncio.Lock()
        
        async with self._server_locks[server_name]:
            # Check if already running
            if server_name in self._servers:
                proc = self._servers[server_name]
                if proc.returncode is None:  # Still running
                    return proc
            
            config = LSP_SERVERS.get(server_name)
            if not config:
                return None
            
            if not await self._ensure_server_available(server_name):
                return None
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *config.start_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                self._servers[server_name] = process
                return process
                
            except Exception as e:
                print(f"Error starting {server_name}: {e}")
                return None
    
    async def _send_lsp_request(
        self,
        server_name: str,
        method: str,
        params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC request to an LSP server"""
        process = await self._start_server(server_name)
        if not process or not process.stdin or not process.stdout:
            return None
        
        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        
        try:
            process.stdin.write(message.encode())
            await process.stdin.drain()
            
            # Read response
            # First read headers
            headers = {}
            while True:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=self.timeout
                )
                line = line.decode().strip()
                if not line:
                    break
                if ": " in line:
                    key, value = line.split(": ", 1)
                    headers[key] = value
            
            # Read content
            content_length = int(headers.get("Content-Length", 0))
            if content_length > 0:
                content = await asyncio.wait_for(
                    process.stdout.read(content_length),
                    timeout=self.timeout
                )
                return json.loads(content.decode())
            
            return None
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"LSP request error: {e}")
            return None
    
    def _content_hash(self, content: str) -> str:
        """Generate hash for content caching"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_diagnostics(
        self,
        file_path: str,
        content: str,
        language_id: str
    ) -> list[Diagnostic]:
        """
        Get diagnostics for a file.
        
        Args:
            file_path: Virtual file path
            content: File content
            language_id: Language identifier (e.g., "python", "javascript")
            
        Returns:
            List of Diagnostic objects
        """
        # Check cache
        cache_key = (file_path, self._content_hash(content))
        if cache_key in self._diagnostic_cache:
            return self._diagnostic_cache[cache_key]
        
        # Get appropriate server
        server_name = self._get_server_for_language(language_id)
        if not server_name:
            return []
        
        # Use simple diagnostic approach for common cases
        diagnostics = await self._get_simple_diagnostics(content, language_id, server_name)
        
        # Cache results
        self._diagnostic_cache[cache_key] = diagnostics
        
        return diagnostics
    
    async def _get_simple_diagnostics(
        self,
        content: str,
        language_id: str,
        server_name: str
    ) -> list[Diagnostic]:
        """
        Get diagnostics using simple approach (subprocess for specific tools).
        This is faster and more reliable than full LSP for basic diagnostics.
        """
        diagnostics = []
        
        if language_id == "python" and server_name == "pylsp":
            diagnostics = await self._python_diagnostics(content)
        elif language_id in ("javascript", "typescript", "typescriptreact", "javascriptreact"):
            diagnostics = await self._js_ts_diagnostics(content, language_id)
        # Add more language-specific handlers as needed
        
        return diagnostics
    
    async def _python_diagnostics(self, content: str) -> list[Diagnostic]:
        """Get Python diagnostics using pyflakes/pylint"""
        diagnostics = []
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Try pyflakes first (fast)
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pyflakes", temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
                
                for line in stdout.decode().strip().split('\n'):
                    if not line:
                        continue
                    # Parse pyflakes output: filename:line:col: message
                    parts = line.split(':', 3)
                    if len(parts) >= 3:
                        try:
                            line_num = int(parts[1]) - 1
                            message = parts[-1].strip()
                            diagnostics.append(Diagnostic(
                                range=Range(
                                    start=Position(line_num, 0),
                                    end=Position(line_num, 100)
                                ),
                                message=message,
                                severity=DiagnosticSeverity.WARNING,
                                source="pyflakes"
                            ))
                        except ValueError:
                            pass
            except Exception:
                pass
            
            # Also run basic syntax check
            try:
                compile(content, '<string>', 'exec')
            except SyntaxError as e:
                line_num = (e.lineno or 1) - 1
                col = (e.offset or 1) - 1
                diagnostics.append(Diagnostic(
                    range=Range(
                        start=Position(line_num, col),
                        end=Position(line_num, col + 1)
                    ),
                    message=str(e.msg),
                    severity=DiagnosticSeverity.ERROR,
                    source="python"
                ))
        
        finally:
            os.unlink(temp_path)
        
        return diagnostics
    
    async def _js_ts_diagnostics(self, content: str, language_id: str) -> list[Diagnostic]:
        """Get JavaScript/TypeScript diagnostics"""
        diagnostics = []
        
        # Determine file extension
        ext = {
            "javascript": ".js",
            "javascriptreact": ".jsx",
            "typescript": ".ts",
            "typescriptreact": ".tsx"
        }.get(language_id, ".js")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Try tsc for TypeScript
            if language_id in ("typescript", "typescriptreact"):
                if shutil.which("tsc"):
                    try:
                        process = await asyncio.create_subprocess_exec(
                            "tsc", "--noEmit", "--pretty", "false", temp_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15)
                        
                        for line in stdout.decode().strip().split('\n'):
                            if not line:
                                continue
                            # Parse tsc output
                            import re
                            match = re.match(r'.*\((\d+),(\d+)\): (error|warning) TS\d+: (.+)', line)
                            if match:
                                line_num = int(match.group(1)) - 1
                                col = int(match.group(2)) - 1
                                severity = DiagnosticSeverity.ERROR if match.group(3) == "error" else DiagnosticSeverity.WARNING
                                message = match.group(4)
                                
                                diagnostics.append(Diagnostic(
                                    range=Range(
                                        start=Position(line_num, col),
                                        end=Position(line_num, col + 1)
                                    ),
                                    message=message,
                                    severity=severity,
                                    source="tsc"
                                ))
                    except Exception:
                        pass
        
        finally:
            os.unlink(temp_path)
        
        return diagnostics
    
    async def stop_server(self, server_name: str):
        """Stop a running LSP server"""
        if server_name in self._servers:
            process = self._servers[server_name]
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
            del self._servers[server_name]
    
    async def stop_all_servers(self):
        """Stop all running LSP servers"""
        for server_name in list(self._servers.keys()):
            await self.stop_server(server_name)
    
    def clear_cache(self):
        """Clear diagnostic cache"""
        self._diagnostic_cache.clear()
    
    def get_server_status(self) -> dict[str, dict]:
        """Get status of all LSP servers"""
        status = {}
        for server_name, config in LSP_SERVERS.items():
            status[server_name] = {
                "name": config.name,
                "languages": config.language_ids,
                "installed": self._is_installed(server_name),
                "running": server_name in self._servers and self._servers[server_name].returncode is None,
                "package_manager": config.package_manager
            }
        return status
    
    def get_available_servers(self) -> list[str]:
        """Get list of available (installed) LSP servers"""
        return [name for name in LSP_SERVERS if self._is_installed(name)]
    
    async def ensure_server_for_language(self, language_id: str) -> bool:
        """Ensure LSP server is available for a language"""
        server_name = self._get_server_for_language(language_id)
        if not server_name:
            return False
        return await self._ensure_server_available(server_name)
