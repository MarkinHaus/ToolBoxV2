"""
Unit Tests for Database Layer

Uses unittest as per project preference.
"""

import unittest
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Add parent to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import (
    DatabaseManager, get_db, Project, ProjectStatus,
    ChatMessage, GeneratedFile, TestResult
)


class TestDatabaseManager(unittest.TestCase):
    """Test suite for DatabaseManager"""

    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        """Clean up test database"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_create_project(self):
        """Test project creation"""
        project = self.db.create_project(
            project_id="test123",
            name="Test Project",
            description="A test project",
            workspace_path=self.temp_dir,
            config={"key": "value"}
        )

        self.assertEqual(project.id, "test123")
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.status, ProjectStatus.IDLE)
        self.assertEqual(project.config, {"key": "value"})

    def test_get_project(self):
        """Test retrieving a project"""
        self.db.create_project(
            project_id="test456",
            name="Fetch Test",
            description="",
            workspace_path=self.temp_dir
        )

        project = self.db.get_project("test456")
        self.assertIsNotNone(project)
        self.assertEqual(project.name, "Fetch Test")

        missing = self.db.get_project("nonexistent")
        self.assertIsNone(missing)

    def test_list_projects(self):
        """Test listing all projects"""
        self.db.create_project("p1", "Project 1", "", self.temp_dir)
        self.db.create_project("p2", "Project 2", "", self.temp_dir)
        self.db.create_project("p3", "Project 3", "", self.temp_dir)

        projects = self.db.list_projects()
        self.assertEqual(len(projects), 3)

        # Should be ordered by updated_at DESC
        names = [p.name for p in projects]
        self.assertIn("Project 1", names)
        self.assertIn("Project 2", names)
        self.assertIn("Project 3", names)

    def test_update_project_status(self):
        """Test updating project status"""
        self.db.create_project("status_test", "Status Test", "", self.temp_dir)

        self.db.update_project_status("status_test", ProjectStatus.RUNNING)
        project = self.db.get_project("status_test")
        self.assertEqual(project.status, ProjectStatus.RUNNING)

        self.db.update_project_status("status_test", ProjectStatus.COMPLETED)
        project = self.db.get_project("status_test")
        self.assertEqual(project.status, ProjectStatus.COMPLETED)

    def test_delete_project(self):
        """Test deleting a project"""
        self.db.create_project("delete_me", "To Delete", "", self.temp_dir)
        self.db.add_chat_message("delete_me", "user", "Hello")

        self.db.delete_project("delete_me")

        project = self.db.get_project("delete_me")
        self.assertIsNone(project)

        messages = self.db.get_chat_history("delete_me")
        self.assertEqual(len(messages), 0)


class TestChatMessages(unittest.TestCase):
    """Test suite for chat message operations"""

    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_chat.db")
        self.db = DatabaseManager(self.db_path)
        self.db.create_project("chat_project", "Chat Test", "", self.temp_dir)

    def tearDown(self):
        """Clean up"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_add_chat_message(self):
        """Test adding a chat message"""
        msg_id = self.db.add_chat_message(
            project_id="chat_project",
            role="user",
            content="Hello, world!",
            message_type="chat",
            metadata={"test": True}
        )

        self.assertIsInstance(msg_id, int)
        self.assertGreater(msg_id, 0)

    def test_get_chat_history(self):
        """Test retrieving chat history"""
        self.db.add_chat_message("chat_project", "user", "Message 1")
        self.db.add_chat_message("chat_project", "assistant", "Response 1")
        self.db.add_chat_message("chat_project", "user", "Message 2")

        messages = self.db.get_chat_history("chat_project")

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].content, "Message 1")
        self.assertEqual(messages[1].role, "assistant")

    def test_clear_chat_history(self):
        """Test clearing chat history"""
        self.db.add_chat_message("chat_project", "user", "To be cleared")
        self.db.add_chat_message("chat_project", "assistant", "Also cleared")

        self.db.clear_chat_history("chat_project")

        messages = self.db.get_chat_history("chat_project")
        self.assertEqual(len(messages), 0)


class TestGeneratedFiles(unittest.TestCase):
    """Test suite for generated file operations"""

    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_files.db")
        self.db = DatabaseManager(self.db_path)
        self.db.create_project("file_project", "File Test", "", self.temp_dir)

    def tearDown(self):
        """Clean up"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_save_generated_file(self):
        """Test saving a generated file"""
        file_id = self.db.save_generated_file(
            project_id="file_project",
            file_path="main.py",
            content="print('hello')",
            language="python",
            validation_status="passed"
        )

        self.assertIsInstance(file_id, int)
        self.assertGreater(file_id, 0)

    def test_file_versioning(self):
        """Test that file versions increment"""
        self.db.save_generated_file("file_project", "test.py", "v1", "python")
        self.db.save_generated_file("file_project", "test.py", "v2", "python")
        self.db.save_generated_file("file_project", "test.py", "v3", "python")

        # Get all files should return latest version only
        files = self.db.get_generated_files("file_project")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].version, 3)
        self.assertEqual(files[0].content, "v3")

    def test_get_file_content(self):
        """Test getting specific file content"""
        self.db.save_generated_file("file_project", "utils.py", "def helper(): pass")

        content = self.db.get_file_content("file_project", "utils.py")
        self.assertEqual(content, "def helper(): pass")

        missing = self.db.get_file_content("file_project", "nonexistent.py")
        self.assertIsNone(missing)


class TestTestResults(unittest.TestCase):
    """Test suite for test result operations"""

    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_results.db")
        self.db = DatabaseManager(self.db_path)
        self.db.create_project("test_project", "Test Results", "", self.temp_dir)

    def tearDown(self):
        """Clean up"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_save_test_result(self):
        """Test saving a test result"""
        result_id = self.db.save_test_result(
            project_id="test_project",
            file_path="main.py",
            test_name="test_main",
            success=True,
            output="All tests passed",
            error=None,
            execution_time_ms=123.45
        )

        self.assertIsInstance(result_id, int)

    def test_get_test_results(self):
        """Test retrieving test results"""
        self.db.save_test_result("test_project", "a.py", "test_a", True, "OK")
        self.db.save_test_result("test_project", "b.py", "test_b", False, "", "Error")

        # Get all results
        results = self.db.get_test_results("test_project")
        self.assertEqual(len(results), 2)

        # Filter by file
        results_a = self.db.get_test_results("test_project", "a.py")
        self.assertEqual(len(results_a), 1)
        self.assertTrue(results_a[0].success)


class TestExecutionState(unittest.TestCase):
    """Test suite for execution state persistence"""

    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_state.db")
        self.db = DatabaseManager(self.db_path)
        self.db.create_project("state_project", "State Test", "", self.temp_dir)

    def tearDown(self):
        """Clean up"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_save_and_get_execution_state(self):
        """Test saving and retrieving execution state"""
        self.db.save_execution_state(
            project_id="state_project",
            execution_id="exec123",
            phase="generation",
            state_data={"progress": 0.5, "files": ["main.py"]}
        )

        state = self.db.get_latest_execution_state("state_project")

        self.assertIsNotNone(state)
        self.assertEqual(state["execution_id"], "exec123")
        self.assertEqual(state["phase"], "generation")
        self.assertEqual(state["state_data"]["progress"], 0.5)

    def test_latest_state(self):
        """Test that latest state is returned"""
        self.db.save_execution_state("state_project", "exec1", "analysis", {"step": 1})
        self.db.save_execution_state("state_project", "exec2", "generation", {"step": 2})
        self.db.save_execution_state("state_project", "exec3", "validation", {"step": 3})

        state = self.db.get_latest_execution_state("state_project")
        self.assertEqual(state["execution_id"], "exec3")
        self.assertEqual(state["phase"], "validation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
