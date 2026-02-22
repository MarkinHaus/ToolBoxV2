"""
Database Layer - SQLite persistence for ProjectDeveloperEngine UI
Handles: Projects, Sessions, Chat History, Generated Files, Test Results
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


class ProjectStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Project:
    id: str
    name: str
    description: str
    workspace_path: str
    status: ProjectStatus
    created_at: str
    updated_at: str
    config: Dict[str, Any]

    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_row(cls, row: tuple) -> 'Project':
        return cls(
            id=row[0],
            name=row[1],
            description=row[2],
            workspace_path=row[3],
            status=ProjectStatus(row[4]),
            created_at=row[5],
            updated_at=row[6],
            config=json.loads(row[7]) if row[7] else {}
        )


@dataclass
class ChatMessage:
    id: int
    project_id: str
    role: str  # 'user', 'assistant', 'system', 'developer'
    content: str
    message_type: str  # 'chat', 'status', 'error', 'code', 'result'
    metadata: Dict[str, Any]
    timestamp: str

    @classmethod
    def from_row(cls, row: tuple) -> 'ChatMessage':
        return cls(
            id=row[0],
            project_id=row[1],
            role=row[2],
            content=row[3],
            message_type=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
            timestamp=row[6]
        )


@dataclass
class GeneratedFile:
    id: int
    project_id: str
    file_path: str
    content: str
    language: str
    version: int
    validation_status: str
    created_at: str

    @classmethod
    def from_row(cls, row: tuple) -> 'GeneratedFile':
        return cls(
            id=row[0],
            project_id=row[1],
            file_path=row[2],
            content=row[3],
            language=row[4],
            version=row[5],
            validation_status=row[6],
            created_at=row[7]
        )


@dataclass
class TestResult:
    id: int
    project_id: str
    file_path: str
    test_name: str
    success: bool
    output: str
    error: Optional[str]
    execution_time_ms: float
    timestamp: str

    @classmethod
    def from_row(cls, row: tuple) -> 'TestResult':
        return cls(
            id=row[0],
            project_id=row[1],
            file_path=row[2],
            test_name=row[3],
            success=bool(row[4]),
            output=row[5],
            error=row[6],
            execution_time_ms=row[7],
            timestamp=row[8]
        )


class DatabaseManager:
    """SQLite Database Manager for Project Persistence"""

    def __init__(self, db_path: str = "project_dev.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Projects table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS projects
                           (
                               id
                               TEXT
                               PRIMARY
                               KEY,
                               name
                               TEXT
                               NOT
                               NULL,
                               description
                               TEXT,
                               workspace_path
                               TEXT
                               NOT
                               NULL,
                               status
                               TEXT
                               DEFAULT
                               'idle',
                               created_at
                               TEXT
                               NOT
                               NULL,
                               updated_at
                               TEXT
                               NOT
                               NULL,
                               config
                               TEXT
                               DEFAULT
                               '{}'
                           )
                           """)

            # Chat history table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS chat_messages
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               project_id
                               TEXT
                               NOT
                               NULL,
                               role
                               TEXT
                               NOT
                               NULL,
                               content
                               TEXT
                               NOT
                               NULL,
                               message_type
                               TEXT
                               DEFAULT
                               'chat',
                               metadata
                               TEXT
                               DEFAULT
                               '{}',
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               FOREIGN
                               KEY
                           (
                               project_id
                           ) REFERENCES projects
                           (
                               id
                           )
                               )
                           """)

            # Generated files table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS generated_files
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               project_id
                               TEXT
                               NOT
                               NULL,
                               file_path
                               TEXT
                               NOT
                               NULL,
                               content
                               TEXT
                               NOT
                               NULL,
                               language
                               TEXT
                               DEFAULT
                               'unknown',
                               version
                               INTEGER
                               DEFAULT
                               1,
                               validation_status
                               TEXT
                               DEFAULT
                               'pending',
                               created_at
                               TEXT
                               NOT
                               NULL,
                               FOREIGN
                               KEY
                           (
                               project_id
                           ) REFERENCES projects
                           (
                               id
                           )
                               )
                           """)

            # Test results table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS test_results
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               project_id
                               TEXT
                               NOT
                               NULL,
                               file_path
                               TEXT
                               NOT
                               NULL,
                               test_name
                               TEXT
                               NOT
                               NULL,
                               success
                               INTEGER
                               NOT
                               NULL,
                               output
                               TEXT,
                               error
                               TEXT,
                               execution_time_ms
                               REAL
                               DEFAULT
                               0,
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               FOREIGN
                               KEY
                           (
                               project_id
                           ) REFERENCES projects
                           (
                               id
                           )
                               )
                           """)

            # Execution state table (for continuing executions)
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS execution_state
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               project_id
                               TEXT
                               NOT
                               NULL,
                               execution_id
                               TEXT
                               NOT
                               NULL,
                               phase
                               TEXT
                               NOT
                               NULL,
                               state_data
                               TEXT
                               NOT
                               NULL,
                               created_at
                               TEXT
                               NOT
                               NULL,
                               FOREIGN
                               KEY
                           (
                               project_id
                           ) REFERENCES projects
                           (
                               id
                           )
                               )
                           """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_project ON chat_messages(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_project ON generated_files(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tests_project ON test_results(project_id)")

    # ==========================================================================
    # PROJECT OPERATIONS
    # ==========================================================================

    def create_project(self, project_id: str, name: str, description: str,
                       workspace_path: str, config: Dict = None) -> Project:
        """Create a new project"""
        now = datetime.now().isoformat()
        config = config or {}

        # Create workspace directory
        Path(workspace_path).mkdir(parents=True, exist_ok=True)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO projects (id, name, description, workspace_path, status, created_at, updated_at,
                                                 config)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           """, (project_id, name, description, workspace_path, ProjectStatus.IDLE.value, now, now,
                                 json.dumps(config)))

        return Project(
            id=project_id,
            name=name,
            description=description,
            workspace_path=workspace_path,
            status=ProjectStatus.IDLE,
            created_at=now,
            updated_at=now,
            config=config
        )

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            if row:
                return Project.from_row(tuple(row))
        return None

    def list_projects(self) -> List[Project]:
        """List all projects"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")
            return [Project.from_row(tuple(row)) for row in cursor.fetchall()]

    def update_project_status(self, project_id: str, status: ProjectStatus):
        """Update project status"""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           UPDATE projects
                           SET status     = ?,
                               updated_at = ?
                           WHERE id = ?
                           """, (status.value, now, project_id))

    def delete_project(self, project_id: str):
        """Delete a project and all related data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM generated_files WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM test_results WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM execution_state WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    # ==========================================================================
    # CHAT OPERATIONS
    # ==========================================================================

    def add_chat_message(self, project_id: str, role: str, content: str,
                         message_type: str = "chat", metadata: Dict = None) -> int:
        """Add a chat message"""
        now = datetime.now().isoformat()
        metadata = metadata or {}

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO chat_messages (project_id, role, content, message_type, metadata, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?)
                           """, (project_id, role, content, message_type, json.dumps(metadata), now))
            return cursor.lastrowid

    def get_chat_history(self, project_id: str, limit: int = 100) -> List[ChatMessage]:
        """Get chat history for a project"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT *
                           FROM chat_messages
                           WHERE project_id = ?
                           ORDER BY timestamp ASC LIMIT ?
                           """, (project_id, limit))
            return [ChatMessage.from_row(tuple(row)) for row in cursor.fetchall()]

    def clear_chat_history(self, project_id: str):
        """Clear chat history for a project"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE project_id = ?", (project_id,))

    # ==========================================================================
    # FILE OPERATIONS
    # ==========================================================================

    def save_generated_file(self, project_id: str, file_path: str, content: str,
                            language: str = "unknown", validation_status: str = "pending") -> int:
        """Save a generated file"""
        now = datetime.now().isoformat()

        # Get current version
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT MAX(version)
                           FROM generated_files
                           WHERE project_id = ?
                             AND file_path = ?
                           """, (project_id, file_path))
            result = cursor.fetchone()
            version = (result[0] or 0) + 1

            cursor.execute("""
                           INSERT INTO generated_files (project_id, file_path, content, language, version,
                                                        validation_status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           """, (project_id, file_path, content, language, version, validation_status, now))
            return cursor.lastrowid

    def get_generated_files(self, project_id: str) -> List[GeneratedFile]:
        """Get all generated files for a project (latest versions only)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT gf.*
                           FROM generated_files gf
                                    INNER JOIN (SELECT file_path, MAX(version) as max_version
                                                FROM generated_files
                                                WHERE project_id = ?
                                                GROUP BY file_path) latest
                                               ON gf.file_path = latest.file_path AND gf.version = latest.max_version
                           WHERE gf.project_id = ?
                           """, (project_id, project_id))
            return [GeneratedFile.from_row(tuple(row)) for row in cursor.fetchall()]

    def get_file_content(self, project_id: str, file_path: str) -> Optional[str]:
        """Get latest content of a specific file"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT content
                           FROM generated_files
                           WHERE project_id = ?
                             AND file_path = ?
                           ORDER BY version DESC LIMIT 1
                           """, (project_id, file_path))
            row = cursor.fetchone()
            return row[0] if row else None

    # ==========================================================================
    # TEST RESULTS
    # ==========================================================================

    def save_test_result(self, project_id: str, file_path: str, test_name: str,
                         success: bool, output: str = "", error: str = None,
                         execution_time_ms: float = 0) -> int:
        """Save a test result"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO test_results (project_id, file_path, test_name, success, output, error,
                                                     execution_time_ms, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           """, (project_id, file_path, test_name, int(success), output, error, execution_time_ms, now))
            return cursor.lastrowid

    def get_test_results(self, project_id: str, file_path: str = None) -> List[TestResult]:
        """Get test results for a project"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if file_path:
                cursor.execute("""
                               SELECT *
                               FROM test_results
                               WHERE project_id = ?
                                 AND file_path = ?
                               ORDER BY timestamp DESC
                               """, (project_id, file_path))
            else:
                cursor.execute("""
                               SELECT *
                               FROM test_results
                               WHERE project_id = ?
                               ORDER BY timestamp DESC
                               """, (project_id,))
            return [TestResult.from_row(tuple(row)) for row in cursor.fetchall()]

    # ==========================================================================
    # EXECUTION STATE
    # ==========================================================================

    def save_execution_state(self, project_id: str, execution_id: str,
                             phase: str, state_data: Dict) -> int:
        """Save execution state for resumption"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO execution_state (project_id, execution_id, phase, state_data, created_at)
                           VALUES (?, ?, ?, ?, ?)
                           """, (project_id, execution_id, phase, json.dumps(state_data), now))
            return cursor.lastrowid

    def get_latest_execution_state(self, project_id: str) -> Optional[Dict]:
        """Get the latest execution state for a project"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT execution_id, phase, state_data
                           FROM execution_state
                           WHERE project_id = ?
                           ORDER BY created_at DESC LIMIT 1
                           """, (project_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "execution_id": row[0],
                    "phase": row[1],
                    "state_data": json.loads(row[2])
                }
        return None


# Singleton instance
_db_instance: Optional[DatabaseManager] = None


def get_db(db_path: str = "project_dev.db") -> DatabaseManager:
    """Get or create database manager instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance
