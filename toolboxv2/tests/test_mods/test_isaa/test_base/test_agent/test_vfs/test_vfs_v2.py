"""
Tests for VFS V2, LSP Manager, Docker VFS, and Web Display

Uses unittest as per project standards.

Author: FlowAgent V2
"""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Import components
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    VirtualFileSystemV2,
    VFSFile,
    VFSDirectory,
    FileCategory,
    FileTypeInfo,
    get_file_type
)
from toolboxv2.mods.isaa.base.Agent.lsp_manager import (
    LSPManager,
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range
)
from toolboxv2.mods.isaa.base.Agent.docker_vfs import (
    DockerVFS,
    DockerConfig,
    CommandResult
)
from toolboxv2.mods.isaa.base.Agent.web_display import (
    WebAppDisplay,
    WebDisplayConfig,
    DisplaySession
)
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2
from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_shell, make_vfs_view


# =============================================================================
# HELPERS
# =============================================================================

def _make_session(tmp_dir: str | None = None) -> MagicMock:
    """Build a minimal AgentSession mock with a real VFS instance."""
    vfs = VirtualFileSystemV2(session_id="test_shell", agent_name="test_agent")
    session = MagicMock()
    session.vfs = vfs
    session.agent_name = "test_agent"
    return session


def _prep(session, paths: dict[str, str] | None = None, dirs: list[str] | None = None):
    """Populate a session's VFS with directories and files."""
    for d in dirs or []:
        session.vfs.mkdir(d, parents=True)
    for path, content in (paths or {}).items():
        parent = os.path.dirname(path)
        if parent and parent != "/":
            session.vfs.mkdir(parent, parents=True)
        session.vfs.create(path, content)



class TestFileTypes(unittest.TestCase):
    """Tests for file type detection"""

    def test_python_file(self):
        """Test Python file detection"""
        info = get_file_type("test.py")
        self.assertEqual(info.category, FileCategory.CODE)
        self.assertEqual(info.language_id, "python")
        self.assertTrue(info.is_executable)
        self.assertEqual(info.lsp_server, "pylsp")

    def test_typescript_file(self):
        """Test TypeScript file detection"""
        info = get_file_type("app.ts")
        self.assertEqual(info.category, FileCategory.CODE)
        self.assertEqual(info.language_id, "typescript")
        self.assertTrue(info.is_executable)

    def test_html_file(self):
        """Test HTML file detection"""
        info = get_file_type("index.html")
        self.assertEqual(info.category, FileCategory.WEB)
        self.assertEqual(info.language_id, "html")
        self.assertFalse(info.is_executable)

    def test_yaml_config(self):
        """Test YAML config detection"""
        info = get_file_type("config.yaml")
        self.assertEqual(info.category, FileCategory.CONFIG)
        self.assertEqual(info.language_id, "yaml")

    def test_unknown_file(self):
        """Test unknown file type"""
        info = get_file_type("data.xyz")
        self.assertEqual(info.category, FileCategory.UNKNOWN)

    def test_dockerfile(self):
        """Test Dockerfile detection"""
        info = get_file_type("Dockerfile")
        self.assertEqual(info.category, FileCategory.CONFIG)
        self.assertEqual(info.language_id, "dockerfile")

    def test_tb_language(self):
        """Test TB language file"""
        info = get_file_type("script.tb")
        self.assertEqual(info.category, FileCategory.CODE)
        self.assertEqual(info.language_id, "tb")
        self.assertTrue(info.is_executable)


class TestVFSV2Directories(unittest.TestCase):
    """Tests for VFS V2 directory operations"""

    def setUp(self):
        self.vfs = VirtualFileSystemV2(
            session_id="test_session",
            agent_name="test_agent"
        )

    def test_mkdir_simple(self):
        """Test simple directory creation"""
        result = self.vfs.mkdir("/src")
        self.assertTrue(result["success"])
        self.assertTrue(self.vfs._is_directory("/src"))

    def test_mkdir_nested(self):
        """Test nested directory creation with parents=True"""
        result = self.vfs.mkdir("/src/components/ui", parents=True)
        self.assertTrue(result["success"])
        self.assertTrue(self.vfs._is_directory("/src"))
        self.assertTrue(self.vfs._is_directory("/src/components"))
        self.assertTrue(self.vfs._is_directory("/src/components/ui"))

    def test_mkdir_no_parent(self):
        """Test mkdir fails without parent"""
        result = self.vfs.mkdir("/nonexistent/child")
        self.assertFalse(result["success"])
        self.assertIn("Parent directory does not exist", result["error"])

    def test_rmdir_empty(self):
        """Test removing empty directory"""
        self.vfs.mkdir("/empty_dir")
        result = self.vfs.rmdir("/empty_dir")
        self.assertTrue(result["success"])
        self.assertFalse(self.vfs._is_directory("/empty_dir"))

    def test_rmdir_nonempty(self):
        """Test removing non-empty directory fails"""
        self.vfs.mkdir("/parent")
        self.vfs.create("/parent/file.txt", "content")

        result = self.vfs.rmdir("/parent")
        self.assertFalse(result["success"])
        self.assertIn("not empty", result["error"])

    def test_rmdir_force(self):
        """Test force removing non-empty directory"""
        self.vfs.mkdir("/parent")
        self.vfs.create("/parent/file.txt", "content")

        result = self.vfs.rmdir("/parent", force=True)
        self.assertTrue(result["success"])
        self.assertFalse(self.vfs._is_directory("/parent"))
        self.assertFalse(self.vfs._is_file("/parent/file.txt"))

    def test_ls(self):
        """Test listing directory contents"""
        self.vfs.mkdir("/project")
        self.vfs.mkdir("/project/src")
        self.vfs.create("/project/README.md", "# Project")

        result = self.vfs.ls("/project")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["contents"]), 2)

        # Check ordering (directories first)
        self.assertEqual(result["contents"][0]["name"], "src")
        self.assertEqual(result["contents"][0]["type"], "directory")
        self.assertEqual(result["contents"][1]["name"], "README.md")
        self.assertEqual(result["contents"][1]["type"], "file")

    def test_ls_recursive(self):
        """Test recursive listing"""
        self.vfs.mkdir("/project/src/utils", parents=True)
        self.vfs.create("/project/src/utils/helper.py", "# helper")

        result = self.vfs.ls("/project", recursive=True)
        self.assertTrue(result["success"])

        # Should have nested items
        paths = [c["path"] for c in result["contents"]]
        self.assertIn("/project/src", paths)
        self.assertIn("/project/src/utils", paths)
        self.assertIn("/project/src/utils/helper.py", paths)

    def test_mv_file(self):
        """Test moving a file"""
        self.vfs.mkdir("/src")
        self.vfs.mkdir("/dest")
        self.vfs.create("/src/file.py", "content")

        result = self.vfs.mv("/src/file.py", "/dest/file.py")
        self.assertTrue(result["success"])
        self.assertFalse(self.vfs._is_file("/src/file.py"))
        self.assertTrue(self.vfs._is_file("/dest/file.py"))

    def test_mv_directory(self):
        """Test moving a directory"""
        self.vfs.mkdir("/old/subdir", parents=True)
        self.vfs.create("/old/subdir/file.txt", "content")

        result = self.vfs.mv("/old", "/new")
        self.assertTrue(result["success"])
        self.assertFalse(self.vfs._is_directory("/old"))
        self.assertTrue(self.vfs._is_directory("/new"))
        self.assertTrue(self.vfs._is_directory("/new/subdir"))
        self.assertTrue(self.vfs._is_file("/new/subdir/file.txt"))


class TestVFSV2Files(unittest.TestCase):
    """Tests for VFS V2 file operations"""

    def setUp(self):
        self.vfs = VirtualFileSystemV2(
            session_id="test_session",
            agent_name="test_agent"
        )

    def test_create_file(self):
        """Test file creation"""
        result = self.vfs.create("/test.py", "print('hello')")
        self.assertTrue(result["success"])
        self.assertIn("Python", result["file_type"])

    def test_create_file_in_directory(self):
        """Test file creation in directory"""
        self.vfs.mkdir("/src")
        result = self.vfs.create("/src/main.py", "# main")
        self.assertTrue(result["success"])

    def test_create_file_no_parent(self):
        """Test file creation without parent directory NOT fails"""
        result = self.vfs.create("/nonexistent/file.py", "content")
        self.assertTrue(result["success"])

    def test_read_write(self):
        """Test read and write operations"""
        self.vfs.create("/data.txt", "initial")

        # Write
        self.vfs.write("/data.txt", "updated")

        # Read
        result = self.vfs.read("/data.txt")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "updated")

    def test_edit_lines(self):
        """Test line-based editing"""
        self.vfs.create("/code.py", "line1\nline2\nline3")

        result = self.vfs.edit("/code.py", 2, 2, "new_line2")
        self.assertTrue(result["success"])

        content = self.vfs.read("/code.py")["content"]
        self.assertEqual(content, "line1\nnew_line2\nline3")

    def test_file_info(self):
        """Test file info retrieval"""
        self.vfs.create("/info_test.py", "# Python file")

        info = self.vfs.get_file_info("/info_test.py")
        self.assertTrue(info["success"])
        self.assertEqual(info["type"], "file")
        self.assertEqual(info["category"], "CODE")
        self.assertTrue(info["is_executable"])
        self.assertTrue(info["lsp_enabled"])


class TestVFSV2Context(unittest.TestCase):
    """Tests for VFS V2 context building"""

    def setUp(self):
        self.vfs = VirtualFileSystemV2(
            session_id="test_session",
            agent_name="test_agent"
        )

    def test_context_string(self):
        """Test context string generation"""
        self.vfs.mkdir("/src")
        self.vfs.create("/src/main.py", "print('hello')")
        self.vfs.open("/src/main.py")

        context = self.vfs.build_context_string()

        self.assertIn("VFS", context)
        self.assertIn("main.py", context)
        self.assertIn("OPEN", context)

    def test_context_includes_tree(self):
        """Test context includes directory tree"""
        self.vfs.mkdir("/project/src", parents=True)
        self.vfs.create("/project/README.md", "# Readme")

        context = self.vfs.build_context_string()

        self.assertIn("Structure:", context)


class TestVFSV2OpenClose(unittest.TestCase):
    """Tests for VFS V2 open/close operations"""

    def setUp(self):
        self.vfs = VirtualFileSystemV2(
            session_id="test_session",
            agent_name="test_agent"
        )

    def test_open_file(self):
        """Test opening a file"""
        self.vfs.create("/test.py", "line1\nline2\nline3")

        result = self.vfs.open("/test.py")
        self.assertTrue(result["success"])
        self.assertEqual(self.vfs.files["/test.py"].state, "open")

    def test_open_with_range(self):
        """Test opening file with line range"""
        self.vfs.create("/long.py", "\n".join(f"line{i}" for i in range(100)))

        result = self.vfs.open("/long.py", line_start=10, line_end=20)
        self.assertTrue(result["success"])

        f = self.vfs.files["/long.py"]
        self.assertEqual(f.view_start, 9)  # 0-indexed
        self.assertEqual(f.view_end, 20)

    def test_close_file(self):
        """Test closing a file"""
        self.vfs.create("/test.py", "content")
        self.vfs.open("/test.py")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.vfs.close("/test.py"))
        loop.close()

        self.assertTrue(result["success"])
        self.assertEqual(self.vfs.files["/test.py"].state, "closed")
        self.assertIn("summary", result)


class TestLSPManager(unittest.TestCase):
    """Tests for LSP Manager"""

    def setUp(self):
        self.lsp = LSPManager(auto_install=False)

    def test_server_for_python(self):
        """Test getting LSP server for Python"""
        server = self.lsp._get_server_for_language("python")
        self.assertEqual(server, "pylsp")

    def test_server_for_typescript(self):
        """Test getting LSP server for TypeScript"""
        server = self.lsp._get_server_for_language("typescript")
        self.assertEqual(server, "typescript-language-server")

    def test_server_for_unknown(self):
        """Test getting LSP server for unknown language"""
        server = self.lsp._get_server_for_language("unknown_lang")
        self.assertIsNone(server)

    def test_diagnostic_creation(self):
        """Test Diagnostic creation and serialization"""
        diag = Diagnostic(
            range=Range(
                start=Position(10, 5),
                end=Position(10, 15)
            ),
            message="Undefined variable 'x'",
            severity=DiagnosticSeverity.ERROR,
            source="pylsp"
        )

        d = diag.to_dict()
        self.assertEqual(d["message"], "Undefined variable 'x'")
        self.assertEqual(d["severity"], "error")
        self.assertEqual(d["range"]["start"]["line"], 10)

    def test_diagnostic_display(self):
        """Test Diagnostic display string"""
        diag = Diagnostic(
            range=Range(
                start=Position(5, 0),
                end=Position(5, 10)
            ),
            message="Unused import",
            severity=DiagnosticSeverity.WARNING,
            source="pyflakes"
        )

        display = diag.to_display_string()
        self.assertIn("Line 6", display)  # 1-indexed for display
        self.assertIn("⚠️", display)
        self.assertIn("Unused import", display)

    def test_get_server_status(self):
        """Test getting server status"""
        status = self.lsp.get_server_status()

        self.assertIn("pylsp", status)
        self.assertIn("typescript-language-server", status)
        self.assertEqual(status["pylsp"]["name"], "Python Language Server")


class TestDiagnosticPythonIntegration(unittest.TestCase):
    """Integration tests for Python diagnostics"""

    def setUp(self):
        self.lsp = LSPManager(auto_install=False)

    def test_python_syntax_error(self):
        """Test Python syntax error detection"""
        code = """
def broken_function(
    print("missing closing parenthesis"
"""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        diagnostics = loop.run_until_complete(
            self.lsp._python_diagnostics(code)
        )
        loop.close()

        # Should detect syntax error
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertTrue(len(errors) > 0)


class TestDockerConfig(unittest.TestCase):
    """Tests for Docker configuration"""

    def test_default_config(self):
        """Test default Docker configuration"""
        config = DockerConfig()

        self.assertEqual(config.base_image, "toolboxv2:latest")
        self.assertEqual(config.workspace_dir, "/workspace")
        self.assertEqual(config.memory_limit, "2g")

    def test_custom_config(self):
        """Test custom Docker configuration"""
        config = DockerConfig(
            base_image="python:3.11-alpine",
            memory_limit="4g",
            port_range_start=9000
        )

        self.assertEqual(config.base_image, "python:3.11-alpine")
        self.assertEqual(config.memory_limit, "4g")
        self.assertEqual(config.port_range_start, 9000)


class TestCommandResult(unittest.TestCase):
    """Tests for CommandResult"""

    def test_success_result(self):
        """Test successful command result"""
        result = CommandResult(
            exit_code=0,
            stdout="Hello World",
            stderr="",
            duration=0.5,
            command="echo 'Hello World'"
        )

        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "Hello World")

    def test_failed_result(self):
        """Test failed command result"""
        result = CommandResult(
            exit_code=1,
            stdout="",
            stderr="Error: file not found",
            duration=0.1,
            command="cat missing.txt"
        )

        self.assertFalse(result.success)
        self.assertIn("not found", result.stderr)

    def test_to_dict(self):
        """Test result serialization"""
        result = CommandResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration=1.5,
            command="test"
        )

        d = result.to_dict()
        self.assertEqual(d["exit_code"], 0)
        self.assertEqual(d["success"], True)
        self.assertIn("timestamp", d)


class TestDockerVFSMocked(unittest.TestCase):
    """Tests for DockerVFS with mocked Docker"""

    def setUp(self):
        self.vfs = VirtualFileSystemV2(
            session_id="test_docker",
            agent_name="test_agent"
        )
        self.docker = DockerVFS(
            vfs=self.vfs,
            config=DockerConfig()
        )

    def test_container_name_generation(self):
        """Test container name generation"""
        name = self.docker._get_container_name()
        self.assertIn("test_docker", name)
        self.assertIn("vfs_session", name)

    def test_port_allocation(self):
        """Test port allocation"""
        port1 = self.docker._allocate_port()
        port2 = self.docker._allocate_port()

        self.assertIsNotNone(port1)
        self.assertIsNotNone(port2)
        self.assertNotEqual(port1, port2)

    def test_port_release(self):
        """Test port release"""
        port = self.docker._allocate_port()
        self.docker._release_port(port)

        self.assertNotIn(port, self.docker._used_ports)

    def test_status_not_running(self):
        """Test status when not running"""
        status = self.docker.get_status()

        self.assertFalse(status["is_running"])
        self.assertIsNone(status["container_id"])

    def test_checkpoint(self):
        """Test checkpoint serialization"""
        checkpoint = self.docker.to_checkpoint()

        self.assertIn("config", checkpoint)
        self.assertIn("history", checkpoint)
        self.assertEqual(checkpoint["config"]["base_image"], "toolboxv2:latest")


class TestWebAppDisplay(unittest.TestCase):
    """Tests for WebAppDisplay"""

    def setUp(self):
        self.display = WebAppDisplay(
            config=WebDisplayConfig(
                proxy_port_start=19000,
                proxy_port_end=19100
            )
        )

    def tearDown(self):
        self.display.cleanup()

    def test_create_session(self):
        """Test creating a display session"""
        result = self.display.create_session("http://localhost:8080")

        self.assertTrue(result["success"])
        self.assertIn("session_id", result)
        self.assertIn("token", result)
        self.assertIn("access_url", result)
        self.assertIn("iframe_html", result)

    def test_session_token_unique(self):
        """Test session tokens are unique"""
        result1 = self.display.create_session("http://localhost:8080")
        result2 = self.display.create_session("http://localhost:8081")

        self.assertNotEqual(result1["token"], result2["token"])
        self.assertNotEqual(result1["session_id"], result2["session_id"])

    def test_get_session(self):
        """Test retrieving a session"""
        create_result = self.display.create_session("http://localhost:8080")
        session = self.display.get_session(create_result["session_id"])

        self.assertIsNotNone(session)
        self.assertEqual(session.container_url, "http://localhost:8080")

    def test_close_session(self):
        """Test closing a session"""
        create_result = self.display.create_session("http://localhost:8080")
        close_result = self.display.close_session(create_result["session_id"])

        self.assertTrue(close_result["success"])
        self.assertIsNone(self.display.get_session(create_result["session_id"]))

    def test_list_sessions(self):
        """Test listing sessions"""
        self.display.create_session("http://localhost:8080")
        self.display.create_session("http://localhost:8081")

        sessions = self.display.list_sessions()

        self.assertEqual(len(sessions), 2)

    def test_max_sessions(self):
        """Test max sessions limit"""
        self.display.config.max_sessions = 2

        self.display.create_session("http://localhost:8080")
        self.display.create_session("http://localhost:8081")
        result = self.display.create_session("http://localhost:8082")

        self.assertFalse(result["success"])
        self.assertIn("Maximum sessions", result["error"])

    def test_iframe_html(self):
        """Test iframe HTML generation"""
        result = self.display.create_session("http://localhost:8080")

        iframe = result["iframe_html"]

        self.assertIn("<iframe", iframe)
        self.assertIn("src=", iframe)
        self.assertIn("frameborder", iframe)

    def test_full_html_page(self):
        """Test full HTML page generation"""
        result = self.display.create_session("http://localhost:8080")
        html = self.display.generate_full_html_page(result["session_id"], "Test App")

        self.assertIsNotNone(html)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Test App", html)
        self.assertIn("<iframe", html)

    def test_simplecor_config(self):
        """Test SimpleCor config generation"""
        result = self.display.create_session("http://localhost:8080")
        config = self.display.generate_simplecor_config(result["session_id"])

        self.assertIsNotNone(config)
        self.assertIn("routing_rules", config)
        self.assertIn("iframe_url", config)


class TestVFSV2Checkpoint(unittest.TestCase):
    """Tests for VFS V2 checkpoint/restore"""

    def test_checkpoint_and_restore(self):
        """Test full checkpoint and restore cycle"""
        # Create VFS with content
        vfs1 = VirtualFileSystemV2(
            session_id="test_checkpoint",
            agent_name="test_agent"
        )
        vfs1.mkdir("/src", parents=True)
        vfs1.create("/src/main.py", "print('hello')")
        vfs1.create("/README.md", "# Project")

        # Checkpoint
        checkpoint = vfs1.to_checkpoint()

        # Create new VFS and restore
        vfs2 = VirtualFileSystemV2(
            session_id="test_checkpoint",
            agent_name="test_agent"
        )
        vfs2.from_checkpoint(checkpoint)

        # Verify
        self.assertTrue(vfs2._is_directory("/src"))
        self.assertTrue(vfs2._is_file("/src/main.py"))
        self.assertTrue(vfs2._is_file("/README.md"))

        content = vfs2.read("/src/main.py")
        print(content)
        self.assertEqual(content["content"], "print('hello')")


class TestExecutableFiles(unittest.TestCase):
    """Tests for executable file detection"""

    def setUp(self):
        self.vfs = VirtualFileSystemV2(
            session_id="test_exec",
            agent_name="test_agent"
        )

    def test_get_executable_files(self):
        """Test getting list of executable files"""
        self.vfs.create("/main.py", "# Python")
        self.vfs.create("/app.js", "// JS")
        self.vfs.create("/README.md", "# Docs")
        self.vfs.create("/config.yaml", "key: value")

        executables = self.vfs.get_executable_files()

        paths = [e["path"] for e in executables]
        self.assertIn("/main.py", paths)
        self.assertIn("/app.js", paths)
        self.assertNotIn("/README.md", paths)
        self.assertNotIn("/config.yaml", paths)

    def test_can_execute(self):
        """Test can_execute method"""
        self.vfs.create("/script.py", "# script")
        self.vfs.create("/notes.txt", "notes")

        self.assertTrue(self.vfs.can_execute("/script.py"))
        self.assertFalse(self.vfs.can_execute("/notes.txt"))
        self.assertFalse(self.vfs.can_execute("/nonexistent.py"))



# =============================================================================
# VFS_SHELL — BASE BEHAVIOUR
# =============================================================================

class TestVfsShellReturnSchema(unittest.TestCase):
    """Every command must return the standard {success, stdout, stderr, returncode} dict."""

    def setUp(self):
        self.session = _make_session()
        self.shell = make_vfs_shell(self.session)

    def _assert_schema(self, result):
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("stdout", result)
        self.assertIn("stderr", result)
        self.assertIn("returncode", result)

    def test_ls_schema(self):        self._assert_schema(self.shell("ls /"))

    def test_cat_schema(self):
        _prep(self.session, {"/f.txt": "x"})
        self._assert_schema(self.shell("cat /f.txt"))

    def test_mkdir_schema(self):     self._assert_schema(self.shell("mkdir /new"))

    def test_unknown_schema(self):   self._assert_schema(self.shell("notacmd"))

    def test_empty_schema(self):     self._assert_schema(self.shell(""))


# =============================================================================
# VFS_SHELL — NAVIGATION
# =============================================================================

class TestVfsShellNavigation(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        _prep(self.session,
              paths={"/src/main.py": "print('hi')", "/README.md": "# R"},
              dirs=["/src/util"])
        self.shell = make_vfs_shell(self.session)

    # ── ls ──────────────────────────────────────────────────────────────────

    def test_ls_root(self):
        r = self.shell("ls /")
        self.assertTrue(r["success"])
        self.assertIn("src", r["stdout"])
        self.assertIn("README.md", r["stdout"])

    def test_ls_subdir(self):
        r = self.shell("ls /src")
        self.assertTrue(r["success"])
        self.assertIn("main.py", r["stdout"])

    def test_ls_long(self):
        r = self.shell("ls -la /src")
        self.assertTrue(r["success"])
        # Long format shows size column + state
        self.assertIn("main.py", r["stdout"])

    def test_ls_recursive(self):
        r = self.shell("ls -R /src")
        self.assertTrue(r["success"])
        self.assertIn("util", r["stdout"])

    def test_ls_nonexistent(self):
        r = self.shell("ls /does_not_exist")
        self.assertFalse(r["success"])

    # ── pwd ─────────────────────────────────────────────────────────────────

    def test_pwd(self):
        r = self.shell("pwd")
        self.assertTrue(r["success"])
        self.assertEqual(r["stdout"].strip(), "/")

    # ── tree ────────────────────────────────────────────────────────────────

    def test_tree_root(self):
        r = self.shell("tree /")
        self.assertTrue(r["success"])
        self.assertIn("src", r["stdout"])

    def test_tree_depth(self):
        r = self.shell("tree / -L 1")
        self.assertTrue(r["success"])
        # At depth 1 we should see src but not src/main.py
        self.assertNotIn("main.py", r["stdout"])

    def test_tree_subpath(self):
        r = self.shell("tree /src")
        self.assertTrue(r["success"])
        self.assertIn("main.py", r["stdout"])


# =============================================================================
# VFS_SHELL — READ COMMANDS
# =============================================================================

class TestVfsShellRead(unittest.TestCase):
    CONTENT = "\n".join(f"line{i}" for i in range(1, 21))  # 20 lines

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {"/big.txt": self.CONTENT})
        self.shell = make_vfs_shell(self.session)

    def test_cat(self):
        r = self.shell("cat /big.txt")
        self.assertTrue(r["success"])
        self.assertEqual(r["stdout"], self.CONTENT)

    def test_cat_missing(self):
        r = self.shell("cat /missing.txt")
        # cat reports per-file errors in stdout (like real cat)
        self.assertIn("missing.txt", r["stdout"])

    def test_head_default(self):
        r = self.shell("head /big.txt")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 10)
        self.assertEqual(lines[0], "line1")

    def test_head_n5(self):
        r = self.shell("head -n 5 /big.txt")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["stdout"].splitlines()), 5)

    def test_tail_default(self):
        r = self.shell("tail /big.txt")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 10)
        self.assertEqual(lines[-1], "line20")

    def test_tail_n3(self):
        r = self.shell("tail -n 3 /big.txt")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[-1], "line20")

    def test_wc_l(self):
        r = self.shell("wc -l /big.txt")
        self.assertTrue(r["success"])
        self.assertIn("20", r["stdout"])

    def test_wc_default(self):
        r = self.shell("wc /big.txt")
        self.assertTrue(r["success"])
        # Should show lines words chars
        parts = r["stdout"].split()
        self.assertGreaterEqual(len(parts), 3)

    def test_stat(self):
        r = self.shell("stat /big.txt")
        self.assertTrue(r["success"])
        self.assertIn("big.txt", r["stdout"])
        self.assertIn("size", r["stdout"].lower())

    def test_info_alias(self):
        r = self.shell("info /big.txt")
        self.assertTrue(r["success"])


# =============================================================================
# VFS_SHELL — SEARCH
# =============================================================================

class TestVfsShellSearch(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {
            "/src/models.py": "class UserModel:\n    id = 1\n",
            "/src/views.py": "class UserView:\n    pass\n",
            "/src/config.yml": "host: localhost\n",
            "/tests/test_models.py": "import models\ndef test_user(): pass\n",
        })
        self.shell = make_vfs_shell(self.session)

    # ── find ────────────────────────────────────────────────────────────────

    def test_find_by_name(self):
        r = self.shell("find / -name *.py")
        self.assertTrue(r["success"])
        self.assertIn("/src/models.py", r["stdout"])
        self.assertIn("/src/views.py", r["stdout"])
        self.assertNotIn(".yml", r["stdout"])

    def test_find_type_f(self):
        r = self.shell("find /src -type f -name *.py")
        self.assertTrue(r["success"])
        self.assertIn("models.py", r["stdout"])

    def test_find_type_d(self):
        r = self.shell("find / -type d -name src")
        self.assertTrue(r["success"])
        self.assertIn("src", r["stdout"])

    def test_find_no_matches(self):
        r = self.shell("find / -name *.go")
        print(r)
        self.assertFalse(r["success"])  # returncode 1, no matches
        self.assertIn("no matches", r["stdout"])

    # ── grep ────────────────────────────────────────────────────────────────

    def test_grep_recursive(self):
        r = self.shell("grep -rn UserModel /")
        self.assertTrue(r["success"])
        self.assertIn("models.py", r["stdout"])
        self.assertNotIn("views.py", r["stdout"])

    def test_grep_case_insensitive(self):
        r = self.shell("grep -ri usermodel /")
        self.assertTrue(r["success"])
        self.assertIn("models.py", r["stdout"])

    def test_grep_show_line_number(self):
        r = self.shell("grep -rn class /src")
        self.assertTrue(r["success"])
        # Line-number output: file:line:content
        self.assertRegex(r["stdout"], r":\d+:")

    def test_grep_files_only(self):
        r = self.shell("grep -rl class /src")
        self.assertTrue(r["success"])
        lines = r["stdout"].splitlines()
        # Only filenames, no line content
        for line in lines:
            self.assertNotIn("class", line)

    def test_grep_single_file(self):
        r = self.shell("grep -n id /src/models.py")
        self.assertTrue(r["success"])
        self.assertIn("id", r["stdout"])

    def test_grep_no_matches(self):
        r = self.shell("grep -r zzznomatch /")
        print(r)
        self.assertFalse(r["success"])
        self.assertIn("no matches", r["stdout"])

    def test_grep_context_lines(self):
        r = self.shell("grep -rn -C 1 class /src/models.py")
        self.assertTrue(r["success"])
        # Context separator "--" should appear
        self.assertIn("--", r["stdout"])


# =============================================================================
# VFS_SHELL — WRITE COMMANDS
# =============================================================================

class TestVfsShellWrite(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        self.shell = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    # ── touch ───────────────────────────────────────────────────────────────

    def test_touch_creates_file(self):
        r = self.shell("touch /new.txt")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/new.txt"))

    def test_touch_existing_file(self):
        self.vfs.create("/exists.txt", "content")
        before = self.vfs.files["/exists.txt"].updated_at
        r = self.shell("touch /exists.txt")
        self.assertTrue(r["success"])
        # Content unchanged
        self.assertEqual(self.vfs.read("/exists.txt")["content"], "content")

    # ── echo redirect ───────────────────────────────────────────────────────

    def test_echo_overwrite(self):
        self.vfs.create("/out.txt", "old")
        r = self.shell('echo "new content" > /out.txt')
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.read("/out.txt")["content"], "new content")

    def test_echo_append(self):
        self.vfs.create("/log.txt", "first\n")
        r = self.shell('echo "second" >> /log.txt')
        self.assertTrue(r["success"])
        self.assertIn("second", self.vfs.read("/log.txt")["content"])

    def test_echo_creates_file(self):
        r = self.shell('echo "hello" > /brand_new.txt')
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.read("/brand_new.txt")["content"], "hello")

    # ── write ───────────────────────────────────────────────────────────────

    def test_write_simple(self):
        r = self.shell("write /data.txt hello")
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.read("/data.txt")["content"], "hello")

    def test_write_multiline(self):
        r = self.shell(r'write /multi.py line1\nline2\nline3')
        self.assertTrue(r["success"])
        content = self.vfs.read("/multi.py")["content"]
        print(content)
        self.assertEqual(content.count("\n"), 2)
        self.assertIn("line2", content)

    def test_write_overwrite(self):
        self.vfs.create("/f.txt", "old")
        self.shell("write /f.txt new")
        self.assertEqual(self.vfs.read("/f.txt")["content"], "new")

    def test_write_missing_args(self):
        r = self.shell("write /only_path")
        self.assertFalse(r["success"])

    # ── edit ────────────────────────────────────────────────────────────────

    def test_edit_single_line(self):
        _prep(self.session, {"/code.py": "a\nb\nc\nd"})
        r = self.shell("edit /code.py 2 2 B")
        self.assertTrue(r["success"])
        content = self.vfs.read("/code.py")["content"]
        self.assertEqual(content, "a\nB\nc\nd")

    def test_edit_range(self):
        _prep(self.session, {"/code.py": "a\nb\nc\nd"})
        r = self.shell("edit /code.py 2 3 X")
        self.assertTrue(r["success"])
        content = self.vfs.read("/code.py")["content"]
        self.assertNotIn("b", content)
        self.assertNotIn("c", content)
        self.assertIn("X", content)

    def test_edit_multiline_replacement(self):
        _prep(self.session, {"/code.py": "a\nb\nc"})
        r = self.shell(r"edit /code.py 2 2 new1\nnew2")
        self.assertTrue(r["success"])
        content = self.vfs.read("/code.py")["content"]
        self.assertIn("new1", content)
        self.assertIn("new2", content)

    def test_edit_missing_args(self):
        _prep(self.session, {"/code.py": "a"})
        r = self.shell("edit /code.py 1")
        self.assertFalse(r["success"])

    def test_edit_nonexistent_file(self):
        r = self.shell("edit /ghost.py 1 1 x")
        self.assertFalse(r["success"])

    # ── mkdir ───────────────────────────────────────────────────────────────

    def test_mkdir_simple(self):
        r = self.shell("mkdir /newdir")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_directory("/newdir"))

    def test_mkdir_parents(self):
        r = self.shell("mkdir -p /a/b/c")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_directory("/a/b/c"))

    def test_mkdir_no_parent_fails(self):
        r = self.shell("mkdir /no/parent/path")
        self.assertFalse(r["success"])

    # ── rm ──────────────────────────────────────────────────────────────────

    def test_rm_file(self):
        _prep(self.session, {"/del.txt": "bye"})
        r = self.shell("rm /del.txt")
        self.assertTrue(r["success"])
        self.assertFalse(self.vfs._is_file("/del.txt"))

    def test_rm_directory_requires_rf(self):
        self.vfs.mkdir("/dir_with_file")
        self.vfs.create("/dir_with_file/f.txt", "x")
        r = self.shell("rm /dir_with_file")
        self.assertFalse(r["success"])

    def test_rm_rf_directory(self):
        self.vfs.mkdir("/dir_to_kill")
        self.vfs.create("/dir_to_kill/f.txt", "x")
        r = self.shell("rm -rf /dir_to_kill")
        self.assertTrue(r["success"])
        self.assertFalse(self.vfs._is_directory("/dir_to_kill"))

    def test_rm_force_nonexistent(self):
        # -f should not raise an error on missing file
        r = self.shell("rm -f /ghost.txt")
        # with -rf the result depends on the implementation: success or silent
        # Acceptable either way — should not crash
        self.assertIsInstance(r, dict)

    # ── mv ──────────────────────────────────────────────────────────────────

    def test_mv_file(self):
        _prep(self.session, {"/old.txt": "data"})
        r = self.shell("mv /old.txt /new.txt")
        self.assertTrue(r["success"])
        self.assertFalse(self.vfs._is_file("/old.txt"))
        self.assertTrue(self.vfs._is_file("/new.txt"))
        self.assertEqual(self.vfs.read("/new.txt")["content"], "data")

    def test_mv_into_directory(self):
        _prep(self.session, {"/file.txt": "data"}, dirs=["/dest"])
        r = self.shell("mv /file.txt /dest")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/dest/file.txt"))

    def test_mv_missing_source(self):
        r = self.shell("mv /ghost.txt /any.txt")
        self.assertFalse(r["success"])

    # ── cp ──────────────────────────────────────────────────────────────────

    def test_cp_file(self):
        _prep(self.session, {"/orig.txt": "hello"})
        r = self.shell("cp /orig.txt /copy.txt")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/orig.txt"))  # original kept
        self.assertTrue(self.vfs._is_file("/copy.txt"))
        self.assertEqual(self.vfs.read("/copy.txt")["content"], "hello")

    def test_cp_into_directory(self):
        _prep(self.session, {"/file.py": "pass"}, dirs=["/backup"])
        r = self.shell("cp /file.py /backup")
        self.assertTrue(r["success"])
        self.assertTrue(self.vfs._is_file("/backup/file.py"))

    def test_cp_missing_source(self):
        r = self.shell("cp /nothing.py /dst.py")
        self.assertFalse(r["success"])


# =============================================================================
# VFS_SHELL — CONTEXT CONTROL (close)
# =============================================================================

class TestVfsShellClose(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {"/a.py": "x", "/b.py": "y"})
        self.vfs = self.session.vfs
        self.shell = make_vfs_shell(self.session)

    def test_close_open_file(self):
        self.vfs.open("/a.py")
        self.assertEqual(self.vfs.files["/a.py"].state, "open")
        r = self.shell("close /a.py")
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.files["/a.py"].state, "closed")

    def test_close_already_closed(self):
        # Closing a closed file should still succeed
        r = self.shell("close /b.py")
        self.assertTrue(r["success"])

    def test_close_nonexistent(self):
        r = self.shell("close /ghost.py")
        self.assertFalse(r["success"])

    def test_close_system_file(self):
        # System files must not be closable
        r = self.shell("close /system_context.md")
        self.assertFalse(r["success"])


# =============================================================================
# VFS_SHELL — EXECUTE
# =============================================================================

class TestVfsShellExec(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        self.shell = make_vfs_shell(self.session)
        self.vfs = self.session.vfs

    def test_exec_python(self):
        _prep(self.session, {"/hello.py": "print('hello world')"})
        r = self.shell("exec /hello.py")
        self.assertTrue(r["success"], r["stderr"])
        self.assertIn("hello world", r["stdout"])

    def test_exec_with_args(self):
        _prep(self.session, {"/echo_arg.py": "import sys; print(sys.argv[1])"})
        r = self.shell("exec /echo_arg.py myarg")
        self.assertTrue(r["success"], r["stderr"])
        self.assertIn("myarg", r["stdout"])

    def test_exec_non_executable(self):
        _prep(self.session, {"/data.json": '{"key": "value"}'})
        r = self.shell("exec /data.json")
        self.assertFalse(r["success"])

    def test_exec_nonexistent(self):
        r = self.shell("exec /ghost.py")
        self.assertFalse(r["success"])


# =============================================================================
# VFS_SHELL — EDGE CASES & PARSE ROBUSTNESS
# =============================================================================

class TestVfsShellEdgeCases(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        self.shell = make_vfs_shell(self.session)

    def test_unknown_command(self):
        r = self.shell("notacommand")
        self.assertFalse(r["success"])
        self.assertIn("command not found", r["stderr"])

    def test_empty_string(self):
        r = self.shell("")
        self.assertFalse(r["success"])

    def test_whitespace_only(self):
        r = self.shell("   ")
        self.assertFalse(r["success"])

    def test_quoted_content_with_spaces(self):
        r = self.shell('write /f.txt "hello world today"')
        self.assertTrue(r["success"])
        self.assertIn("hello world today", self.session.vfs.read("/f.txt")["content"])

    def test_echo_single_quotes(self):
        r = self.shell("echo 'single quoted' > /sq.txt")
        self.assertTrue(r["success"])
        self.assertIn("single quoted", self.session.vfs.read("/sq.txt")["content"])

    def test_path_normalisation(self):
        # Double slash / no leading slash should still work via vfs normalise
        _prep(self.session, {"/src/a.py": "x"})
        r = self.shell("cat /src/a.py")
        self.assertTrue(r["success"])


# =============================================================================
# VFS_VIEW — BASE BEHAVIOUR
# =============================================================================

class TestVfsViewReturnSchema(unittest.TestCase):
    CONTENT = "\n".join(f"line{i}" for i in range(1, 51))  # 50 lines

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {"/big.py": self.CONTENT})
        self.view = make_vfs_view(self.session)

    def _assert_schema(self, r):
        self.assertIsInstance(r, dict)
        self.assertIn("success", r)
        if r["success"]:
            self.assertIn("content", r)
            self.assertIn("showing", r)
            self.assertIn("total_lines", r)
            self.assertIn("file_type", r)

    def test_basic_open(self):
        self._assert_schema(self.view("/big.py"))

    def test_scroll_to(self):
        self._assert_schema(self.view("/big.py", scroll_to="line25"))

    def test_nonexistent(self):
        r = self.view("/ghost.py")
        self.assertFalse(r["success"])


# =============================================================================
# VFS_VIEW — LINE RANGE
# =============================================================================

class TestVfsViewLineRange(unittest.TestCase):
    CONTENT = "\n".join(f"line{i}" for i in range(1, 101))  # 100 lines

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {"/doc.py": self.CONTENT})
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs

    def test_open_full_file(self):
        r = self.view("/doc.py")
        self.assertTrue(r["success"])
        self.assertEqual(r["total_lines"], 100)

    def test_open_range(self):
        r = self.view("/doc.py", line_start=10, line_end=20)
        self.assertTrue(r["success"])
        lines = r["content"].splitlines()
        self.assertEqual(lines[0], "line10")
        self.assertEqual(lines[-1], "line20")

    def test_open_range_updates_vfs_state(self):
        self.view("/doc.py", line_start=5, line_end=15)
        f = self.vfs.files["/doc.py"]
        self.assertEqual(f.state, "open")
        self.assertEqual(f.view_start, 4)  # 0-indexed
        self.assertEqual(f.view_end, 15)

    def test_open_range_clamps_start(self):
        # line_start < 1 should not crash
        r = self.view("/doc.py", line_start=-5, line_end=5)
        self.assertTrue(r["success"])

    def test_open_range_clamps_end(self):
        # line_end beyond EOF should return up to EOF
        r = self.view("/doc.py", line_start=95, line_end=999)
        self.assertTrue(r["success"])
        lines = r["content"].splitlines()
        self.assertGreaterEqual(len(lines), 1)
        self.assertLessEqual(len(lines), 10)


# =============================================================================
# VFS_VIEW — scroll_to
# =============================================================================

class TestVfsViewScrollTo(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        code = (
            "# header\n"
            "import os\n"
            "\n"
            "class UserModel:\n"  # line 4
            "    id = 1\n"
            "    name = 'test'\n"
            "\n"
            "def get_user(user_id):\n"  # line 8
            "    return UserModel()\n"
            "\n"
            "class OrderModel:\n"  # line 11
            "    user_id = 1\n"
        )
        _prep(self.session, {"/models.py": code})
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs

    def test_scroll_to_finds_line(self):
        r = self.view("/models.py", scroll_to="UserModel")
        self.assertTrue(r["success"])
        self.assertIn("match", r)
        self.assertEqual(r["match"]["matched_line"], 4)
        self.assertIn("UserModel", r["content"])

    def test_scroll_to_case_insensitive(self):
        r = self.view("/models.py", scroll_to="usermodel")
        self.assertTrue(r["success"])
        self.assertIn("UserModel", r["content"])

    def test_scroll_to_context_lines(self):
        r = self.view("/models.py", scroll_to="get_user", context_lines=4)
        self.assertTrue(r["success"])
        # With context_lines=4 we get ±2 lines around the match
        self.assertLessEqual(len(r["content"].splitlines()), 5)

    def test_scroll_to_missing_pattern(self):
        r = self.view("/models.py", scroll_to="zzznomatch")
        self.assertFalse(r["success"])
        self.assertIn("not found", r["error"])
        self.assertIn("hint", r)

    def test_scroll_to_regex(self):
        r = self.view("/models.py", scroll_to=r"class \w+Model")
        self.assertTrue(r["success"])
        # Should find first class
        self.assertIn("UserModel", r["content"])

    def test_scroll_to_overrides_line_range(self):
        # Even if line_start/end are given, scroll_to wins
        r = self.view("/models.py", line_start=1, line_end=2, scroll_to="OrderModel")
        self.assertTrue(r["success"])
        self.assertIn("OrderModel", r["content"])

    def test_scroll_to_sets_state_open(self):
        self.view("/models.py", scroll_to="UserModel")
        self.assertEqual(self.vfs.files["/models.py"].state, "open")


# =============================================================================
# VFS_VIEW — close_others
# =============================================================================

class TestVfsViewCloseOthers(unittest.TestCase):

    def setUp(self):
        self.session = _make_session()
        _prep(self.session, {
            "/a.py": "alpha",
            "/b.py": "beta",
            "/c.py": "gamma",
        })
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs

    def test_close_others_closes_open_files(self):
        self.vfs.open("/a.py")
        self.vfs.open("/b.py")
        # Open c.py and close others
        r = self.view("/c.py", close_others=True)
        self.assertTrue(r["success"])
        self.assertEqual(self.vfs.files["/c.py"].state, "open")
        self.assertEqual(self.vfs.files["/a.py"].state, "closed")
        self.assertEqual(self.vfs.files["/b.py"].state, "closed")

    def test_close_others_note_in_result(self):
        r = self.view("/c.py", close_others=True)
        self.assertTrue(r["success"])
        self.assertIn("note", r)

    def test_close_others_does_not_close_system_files(self):
        r = self.view("/a.py", close_others=True)
        self.assertTrue(r["success"])
        # System context should still be open
        self.assertEqual(self.vfs.files["/system_context.md"].state, "open")

    def test_close_others_false_leaves_files_open(self):
        self.vfs.open("/a.py")
        self.vfs.open("/b.py")
        self.view("/c.py", close_others=False)
        self.assertEqual(self.vfs.files["/a.py"].state, "open")
        self.assertEqual(self.vfs.files["/b.py"].state, "open")


# =============================================================================
# VFS_VIEW — workflow integration (the x+y focus pattern)
# =============================================================================

class TestVfsViewWorkflow(unittest.TestCase):
    """
    End-to-end simulation of the find → focus → multi-open → answer workflow
    described in vfs_guide.md.
    """

    def setUp(self):
        self.session = _make_session()
        models_code = "\n".join([
            "# models.py",
            "class Base:",
            "    pass",
            "",
            "class UserModel(Base):",  # line 5
            "    id   = 1",
            "    name = 'anon'",
            "",
        ])
        services_code = "\n".join([
            "# services.py",
            "from models import UserModel",
            "",
            "class UserService:",
            "    def get_user(self, uid):",  # line 5
            "        return UserModel()",
            "",
            "    def create_user(self, name):",
            "        u = UserModel()",
            "        u.name = name",
            "        return u",
        ])
        _prep(self.session, {
            "/src/models.py": models_code,
            "/src/services.py": services_code,
            "/src/utils.py": "def helper(): pass",
        })
        self.shell = make_vfs_shell(self.session)
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs

    def test_full_xy_workflow(self):
        # 1. Locate x (UserModel) via grep
        g1 = self.shell("grep -rn UserModel /src")
        self.assertTrue(g1["success"])
        self.assertIn("models.py", g1["stdout"])

        # 2. Focus on x
        r1 = self.view("/src/models.py", scroll_to="UserModel", context_lines=6)
        self.assertTrue(r1["success"])
        self.assertIn("UserModel", r1["content"])

        # 3. Locate y (get_user) via grep
        g2 = self.shell("grep -rn get_user /src")
        self.assertTrue(g2["success"])
        self.assertIn("services.py", g2["stdout"])

        # 4. Add y to context (without closing x)
        r2 = self.view("/src/services.py", scroll_to="get_user", context_lines=6)
        self.assertTrue(r2["success"])
        self.assertIn("get_user", r2["content"])

        # 5. Both files must be open simultaneously
        self.assertEqual(self.vfs.files["/src/models.py"].state, "open")
        self.assertEqual(self.vfs.files["/src/services.py"].state, "open")
        # utils.py was never opened
        self.assertEqual(self.vfs.files["/src/utils.py"].state, "closed")

        # 6. Clean up — close_others=True on next task
        r3 = self.view("/src/utils.py", scroll_to="helper", close_others=True)
        self.assertTrue(r3["success"])
        self.assertEqual(self.vfs.files["/src/models.py"].state, "closed")
        self.assertEqual(self.vfs.files["/src/services.py"].state, "closed")
        self.assertEqual(self.vfs.files["/src/utils.py"].state, "open")


if __name__ == "__main__":
    unittest.main()
