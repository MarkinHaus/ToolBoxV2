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
        """Test file creation without parent directory fails"""
        result = self.vfs.create("/nonexistent/file.py", "content")
        self.assertFalse(result["success"])
    
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
        
        self.assertEqual(config.base_image, "python:3.12-slim")
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
        self.assertEqual(checkpoint["config"]["base_image"], "python:3.12-slim")


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


if __name__ == "__main__":
    unittest.main()
