"""
Tests for ExecutionEngine V3, Skills System, and Sub-Agent System

Uses unittest (not pytest) as per project conventions.

Run with:
    python test_execution_engine_v3.py -v

Author: FlowAgent V3 Tests
"""

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from dataclasses import dataclass
import json
import sys
import os

# Setup imports for standalone testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util

# def load_module(name, path):
#     """Load module from file path"""
#     spec = importlib.util.spec_from_file_location(name, path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[name] = module
#     sys.modules[f'toolboxv2.mods.isaa.base.Agent.{name}'] = module
#     spec.loader.exec_module(module)
#     return module
#
# # Load modules
# _dir = os.path.dirname(os.path.abspath(__file__))
# skills = load_module('skills', os.path.join(_dir, 'skills.py'))
# sub_agent = load_module('sub_agent', os.path.join(_dir, 'sub_agent.py'))
# execution_engine = load_module('execution_engine', os.path.join(_dir, 'execution_engine.py'))

from toolboxv2.mods.isaa.base.Agent import skills, sub_agent, execution_engine
import numpy as np


# =============================================================================
# MOCK CLASSES (für isolierte Tests ohne echte Abhängigkeiten)
# =============================================================================

class MockMemory:
    """Mock für AISemanticMemory"""

    async def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Return fake embeddings"""
        return np.random.rand(len(texts), 384).astype(np.float32)

    async def add_data(self, space_name, content, metadata, direct=True):
        pass

    async def query(self, text, space_name, **kwargs):
        return []


class MockVDB:
    """Mock für AbstractVectorStore"""

    def add_embeddings(self, embeddings, chunks):
        pass

    def search(self, query_embedding, k=5, min_similarity=0.7):
        return []

    def save(self):
        return b""

    def load(self, data):
        return self

    def clear(self):
        pass

    def rebuild_index(self):
        pass


class MockVFS:
    """Mock für VirtualFileSystemV2"""

    def __init__(self):
        self.files = {}
        self.directories = {"/", "/sub"}

    def read(self, path: str) -> dict:
        if path in self.files:
            return {"success": True, "content": self.files[path]}
        return {"success": False, "error": "File not found"}

    def write(self, path: str, content: str) -> dict:
        self.files[path] = content
        return {"success": True, "path": path}

    def create(self, path: str, content: str = "") -> dict:
        return self.write(path, content)

    def list_files(self) -> dict:
        return {"success": True, "files": list(self.files.keys())}

    def ls(self, path: str = "/", recursive: bool = False) -> dict:
        files = [{"path": p} for p in self.files.keys() if p.startswith(path)]
        return {"success": True, "files": files}

    def mkdir(self, path: str, parents: bool = False) -> dict:
        self.directories.add(path)
        return {"success": True, "path": path}

    def rmdir(self, path: str, force: bool = False) -> dict:
        if path in self.directories:
            self.directories.remove(path)
            return {"success": True}
        return {"success": False, "error": "Directory not found"}

    def mv(self, source: str, destination: str) -> dict:
        if source in self.files:
            self.files[destination] = self.files.pop(source)
            return {"success": True}
        return {"success": False, "error": "Source not found"}

    def build_context_string(self) -> str:
        return "\n".join(self.files.keys())


class MockChatSession:
    """Mock für ChatSession"""

    def __init__(self):
        self.history = []
        self.space_name = "test_session"
        self.mem = MockMemory()

    async def add_message(self, message, **kwargs):
        self.history.append(message)

    def get_past_x(self, x):
        return self.history[-x:] if x <= len(self.history) else self.history

    def get_start_with_last_user(self, x=None):
        return self.history[-x:] if x else self.history

    def clear_history(self):
        self.history.clear()

    def on_exit(self):
        pass

    def get_volume(self):
        return len(self.history)


class MockSession:
    """Mock für AgentSessionV2"""

    def __init__(self, session_id="test_session"):
        self.session_id = session_id
        self.vfs = MockVFS()
        self._chat_session = MockChatSession()
        self._initialized = True

    async def add_message(self, message, **kwargs):
        await self._chat_session.add_message(message, **kwargs)

    def get_history_for_llm(self, last_n=10):
        return self._chat_session.get_start_with_last_user(last_n)

    def build_vfs_context(self):
        return self.vfs.build_context_string()


class MockSessionManager:
    """Mock für SessionManager"""

    def __init__(self):
        self.sessions = {}

    async def get_or_create(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = MockSession(session_id)
        return self.sessions[session_id]

    def _get_memory(self):
        return MockMemory()


@dataclass
class MockToolEntry:
    """Mock für ToolEntry"""
    name: str
    description: str = "Test tool"
    args_schema: str = "{}"
    category: list = None
    flags: dict = None
    source: str = "local"
    function: callable = None
    litellm_schema: dict = None

    def __post_init__(self):
        if self.category is None:
            self.category = ["test"]
        if self.flags is None:
            self.flags = {}
        if self.litellm_schema is None:
            self.litellm_schema = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {}}
                }
            }


class MockToolManager:
    """Mock für ToolManager"""

    def __init__(self):
        self.tools = {
            "vfs_read": MockToolEntry("vfs_read", "Read file", category=["vfs"]),
            "vfs_write": MockToolEntry("vfs_write", "Write file", category=["vfs"]),
            "vfs_list": MockToolEntry("vfs_list", "List files", category=["vfs"]),
            "discord_send": MockToolEntry("discord_send", "Send Discord message", category=["discord"]),
            "discord_edit": MockToolEntry("discord_edit", "Edit Discord message", category=["discord"]),
            "http_request": MockToolEntry("http_request", "Make HTTP request", category=["http"]),
        }

    def get_all(self):
        return list(self.tools.values())

    def get_all_litellm(self):
        return [t.litellm_schema for t in self.tools.values()]

    def get(self, name: str):
        return self.tools.get(name)

    def list_names(self):
        return list(self.tools.keys())

    def list_categories(self):
        cats = set()
        for t in self.tools.values():
            if t.category:
                cats.update(t.category)
        return list(cats)


@dataclass
class MockAMD:
    """Mock für Agent Metadata"""
    name: str = "test_agent"


class MockAgent:
    """Mock für FlowAgent"""

    def __init__(self):
        self.amd = MockAMD()
        self.session_manager = MockSessionManager()
        self.tool_manager = MockToolManager()
        self.skills_manager = None

    async def a_run_llm_completion(self, messages, tools=None, **kwargs):
        """Mock LLM completion - returns a mock response"""
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.tool_calls = None
        return mock_response

    async def arun_function(self, name: str, **kwargs):
        """Mock function execution"""
        return f"Executed {name} with {kwargs}"


# =============================================================================
# SKILLS TESTS
# =============================================================================

class TestSkill(unittest.TestCase):
    """Tests für Skill Dataclass"""

    def test_skill_creation(self):
        """Test Skill wird korrekt erstellt"""
        Skill = skills.Skill

        skill = Skill(
            id="test_skill",
            name="Test Skill",
            triggers=["test", "example"],
            instruction="Do something"
        )

        self.assertEqual(skill.id, "test_skill")
        self.assertEqual(skill.name, "Test Skill")
        self.assertEqual(skill.triggers, ["test", "example"])
        self.assertEqual(skill.source, "predefined")
        self.assertEqual(skill.confidence, 1.0)

    def test_skill_matches_keywords(self):
        """Test Keyword Matching"""
        Skill = skills.Skill

        skill = Skill(
            id="test",
            name="Test",
            triggers=["flask", "api", "web server"],
            instruction="Test instruction"
        )

        self.assertTrue(skill.matches_keywords("Erstelle eine Flask App"))
        self.assertTrue(skill.matches_keywords("Ich brauche einen API Server"))
        self.assertFalse(skill.matches_keywords("Schreibe ein Python Script"))

    def test_skill_is_active(self):
        """Test Activation Threshold"""
        Skill = skills.Skill

        # Predefined skill - immer aktiv
        skill1 = Skill(id="s1", name="S1", triggers=[], instruction="Test", source="predefined")
        self.assertTrue(skill1.is_active())

        # Learned skill mit niedriger confidence
        skill2 = Skill(id="s2", name="S2", triggers=[], instruction="Test", source="learned", confidence=0.3)
        self.assertFalse(skill2.is_active())

        # Learned skill mit hoher confidence
        skill3 = Skill(id="s3", name="S3", triggers=[], instruction="Test", source="learned", confidence=0.7)
        self.assertTrue(skill3.is_active())

    def test_skill_record_usage(self):
        """Test Confidence Update bei Verwendung"""
        Skill = skills.Skill

        skill = Skill(id="test", name="Test", triggers=[], instruction="Test", confidence=0.5)

        # Success erhöht confidence
        skill.record_usage(success=True)
        self.assertEqual(skill.confidence, 0.6)
        self.assertEqual(skill.success_count, 1)

        # Failure verringert confidence
        skill.record_usage(success=False)
        self.assertAlmostEqual(skill.confidence, 0.45, places=5)  # 0.6 - 0.15
        self.assertEqual(skill.failure_count, 1)

    def test_skill_serialization(self):
        """Test to_dict und from_dict"""
        Skill = skills.Skill

        skill = Skill(
            id="test",
            name="Test",
            triggers=["a", "b"],
            instruction="Do X",
            tools_used=["tool1"],
            source="learned",
            confidence=0.8
        )

        data = skill.to_dict()
        restored = Skill.from_dict(data)

        self.assertEqual(restored.id, skill.id)
        self.assertEqual(restored.name, skill.name)
        self.assertEqual(restored.triggers, skill.triggers)
        self.assertEqual(restored.confidence, skill.confidence)


class TestSkillsManager(unittest.TestCase):
    """Tests für SkillsManager"""

    def test_init_predefined_skills(self):
        """Test dass predefined Skills geladen werden"""
        SkillsManager = skills.SkillsManager

        manager = SkillsManager(agent_name="test")

        # Sollte predefined Skills haben
        self.assertGreater(len(manager.skills), 0)

        # Check spezifische Skills
        self.assertIn("user_preference_save", manager.skills)
        self.assertIn("habits_tracking", manager.skills)
        self.assertIn("multi_step_task", manager.skills)
        self.assertIn("vfs_info_persist", manager.skills)
        self.assertIn("vfs_task_planning", manager.skills)
        self.assertIn("vfs_knowledge_base", manager.skills)

    def test_match_skills_keyword(self):
        """Test Keyword-basiertes Matching"""
        SkillsManager = skills.SkillsManager

        manager = SkillsManager(agent_name="test")

        # Sollte user_preference_save matchen
        matches = manager.match_skills("Merke dir dass ich Python bevorzuge")
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].id, "user_preference_save")

    def test_match_skills_no_match(self):
        """Test wenn kein Keyword matcht"""
        SkillsManager = skills.SkillsManager

        manager = SkillsManager(agent_name="test")

        matches = manager.match_skills("xyz abc 123 random text")
        # Könnte leer sein oder embedding fallback
        # Ohne memory sollte es leer sein
        self.assertEqual(len(matches), 0)

    def test_score_tool_relevance(self):
        """Test Tool Relevanz Scoring"""
        SkillsManager = skills.SkillsManager

        manager = SkillsManager(agent_name="test")

        # Discord query sollte discord tools hoch scoren
        score = manager.score_tool_relevance(
            query="Sende eine Discord Nachricht",
            tool_name="discord_send",
            tool_description="Send a message to Discord channel"
        )
        self.assertGreater(score, 0.3)

        # Unrelated tool sollte niedrig scoren
        score2 = manager.score_tool_relevance(
            query="Sende eine Discord Nachricht",
            tool_name="database_query",
            tool_description="Query SQL database"
        )
        self.assertLess(score2, score)

    def test_checkpoint_serialization(self):
        """Test to_checkpoint und from_checkpoint"""
        SkillsManager = skills.SkillsManager
        Skill = skills.Skill

        manager = SkillsManager(agent_name="test")

        # Add a learned skill
        learned = Skill(
            id="learned_test",
            name="Learned Test",
            triggers=["test"],
            instruction="Test instruction",
            source="learned",
            confidence=0.7
        )
        manager.skills["learned_test"] = learned

        # Serialize
        checkpoint = manager.to_checkpoint()

        # Restore in new manager
        manager2 = SkillsManager(agent_name="test")
        manager2.from_checkpoint(checkpoint)

        # Learned skill sollte wiederhergestellt sein
        self.assertIn("learned_test", manager2.skills)
        self.assertEqual(manager2.skills["learned_test"].confidence, 0.7)


class TestToolGroup(unittest.TestCase):
    """Tests für ToolGroup"""

    def test_tool_group_creation(self):
        """Test ToolGroup wird korrekt erstellt"""
        ToolGroup = skills.ToolGroup

        group = ToolGroup(
            name="discord_tools",
            display_name="Discord Tools",
            description="Tools for Discord",
            tool_names=["discord_send", "discord_edit"],
            trigger_keywords=["discord", "message"]
        )

        self.assertEqual(group.name, "discord_tools")
        self.assertEqual(len(group.tool_names), 2)

    def test_tool_group_matches_query(self):
        """Test Query Matching"""
        ToolGroup = skills.ToolGroup

        group = ToolGroup(
            name="discord_tools",
            display_name="Discord Tools",
            description="",
            tool_names=[],
            trigger_keywords=["discord", "server", "bot"]
        )

        self.assertTrue(group.matches_query("Erstelle einen Discord Bot"))
        self.assertFalse(group.matches_query("Schreibe eine Email"))


# =============================================================================
# EXECUTION ENGINE TESTS
# =============================================================================

class TestAutoFocusTracker(unittest.TestCase):
    """Tests für AutoFocusTracker"""

    def test_record_action(self):
        """Test Action Recording"""
        AutoFocusTracker = execution_engine.AutoFocusTracker

        tracker = AutoFocusTracker(max_actions=3)

        tracker.record("vfs_write", {"path": "/test.py"}, "Created file")
        self.assertEqual(len(tracker.actions), 1)
        self.assertIn("✏️", tracker.actions[0])  # Write icon

    def test_max_actions_limit(self):
        """Test dass max_actions eingehalten wird"""
        AutoFocusTracker = execution_engine.AutoFocusTracker

        tracker = AutoFocusTracker(max_actions=2)

        tracker.record("tool1", {}, "result1")
        tracker.record("tool2", {}, "result2")
        tracker.record("tool3", {}, "result3")

        self.assertEqual(len(tracker.actions), 2)
        # Älteste sollte entfernt sein
        self.assertNotIn("tool1", tracker.actions[0])

    def test_get_focus_message(self):
        """Test Focus Message Generation"""
        AutoFocusTracker = execution_engine.AutoFocusTracker

        tracker = AutoFocusTracker()

        # Leer sollte None zurückgeben
        self.assertIsNone(tracker.get_focus_message())

        # Mit Actions sollte Message kommen
        tracker.record("vfs_read", {"path": "/test.py"}, "content")
        msg = tracker.get_focus_message()

        self.assertIsNotNone(msg)
        self.assertEqual(msg["role"], "system")
        self.assertIn("LETZTE AKTIONEN", msg["content"])


class TestLoopDetector(unittest.TestCase):
    """Tests für LoopDetector"""

    def test_no_loop_detected(self):
        """Test normale Sequenz erkennt keinen Loop"""
        LoopDetector = execution_engine.LoopDetector

        detector = LoopDetector()

        self.assertFalse(detector.record("tool1", {"a": 1}))
        self.assertFalse(detector.record("tool2", {"b": 2}))
        self.assertFalse(detector.record("tool3", {"c": 3}))

    def test_exact_repeat_loop(self):
        """Test erkennt exakten Wiederholungs-Loop"""
        LoopDetector = execution_engine.LoopDetector

        detector = LoopDetector(max_repeats=3)

        self.assertFalse(detector.record("tool1", {"a": 1}))
        self.assertFalse(detector.record("tool1", {"a": 1}))
        self.assertTrue(detector.record("tool1", {"a": 1}))  # 3. Wiederholung

    def test_ping_pong_loop(self):
        """Test erkennt Ping-Pong Pattern"""
        LoopDetector = execution_engine.LoopDetector

        detector = LoopDetector()

        detector.record("toolA", {"x": 1})
        detector.record("toolB", {"y": 2})
        detector.record("toolA", {"x": 1})
        loop = detector.record("toolB", {"y": 2})

        self.assertTrue(loop)

    def test_intervention_message(self):
        """Test Intervention Message"""
        LoopDetector = execution_engine.LoopDetector

        detector = LoopDetector()
        detector.record("stuck_tool", {})

        msg = detector.get_intervention_message()
        self.assertIn("LOOP ERKANNT", msg)
        self.assertIn("stuck_tool", msg)


class TestHistoryCompressor(unittest.TestCase):
    """Tests für HistoryCompressor"""

    def test_compress_to_summary(self):
        """Test Full Compression"""
        HistoryCompressor = execution_engine.HistoryCompressor

        working_history = [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Create a file"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "name": "vfs_write", "content": "Created /test.py"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "name": "vfs_read", "content": "content..."},
        ]

        summary = HistoryCompressor.compress_to_summary(working_history, "run123")

        self.assertIsNotNone(summary)
        self.assertEqual(summary["role"], "system")
        self.assertIn("ABGESCHLOSSENE AKTIONEN", summary["content"])
        self.assertEqual(summary["metadata"]["run_id"], "run123")

    def test_compress_partial(self):
        """Test Partial Compression"""
        HistoryCompressor = execution_engine.HistoryCompressor

        working_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "tool", "name": "tool1", "content": "r1"},
            {"role": "tool", "name": "tool2", "content": "r2"},
            {"role": "tool", "name": "tool3", "content": "r3"},
            {"role": "tool", "name": "tool4", "content": "r4"},
            {"role": "tool", "name": "tool5", "content": "r5"},
        ]

        summary, remaining = HistoryCompressor.compress_partial(
            working_history, keep_last_n=2
        )

        self.assertIsNotNone(summary)
        # Should keep system + summary + last 2
        self.assertEqual(len(remaining), 4)
        self.assertEqual(remaining[0]["role"], "system")


class TestExecutionContext(unittest.TestCase):
    """Tests für ExecutionContext"""

    def test_context_creation(self):
        """Test Context wird korrekt erstellt"""
        ExecutionContext = execution_engine.ExecutionContext

        ctx = ExecutionContext()

        self.assertIsNotNone(ctx.run_id)
        self.assertEqual(len(ctx.dynamic_tools), 0)
        self.assertEqual(ctx.max_dynamic_tools, 5)

    def test_add_remove_tool(self):
        """Test Tool Slot Management"""
        ExecutionContext = execution_engine.ExecutionContext

        ctx = ExecutionContext()

        # Add tool
        self.assertTrue(ctx.add_tool("tool1", 0.8, "cat1"))
        self.assertEqual(len(ctx.dynamic_tools), 1)

        # Duplicate sollte False zurückgeben
        self.assertFalse(ctx.add_tool("tool1", 0.8, "cat1"))

        # Remove tool
        self.assertTrue(ctx.remove_tool("tool1"))
        self.assertEqual(len(ctx.dynamic_tools), 0)

    def test_get_least_relevant_tool(self):
        """Test findet Tool mit niedrigster Relevanz"""
        ExecutionContext = execution_engine.ExecutionContext

        ctx = ExecutionContext()
        ctx.add_tool("high", 0.9, "cat")
        ctx.add_tool("low", 0.2, "cat")
        ctx.add_tool("mid", 0.5, "cat")

        least = ctx.get_least_relevant_tool()
        self.assertEqual(least, "low")


# =============================================================================
# SUB-AGENT TESTS
# =============================================================================

class TestRestrictedVFSWrapper(unittest.TestCase):
    """Tests für RestrictedVFSWrapper"""

    def test_read_allowed_everywhere(self):
        """Test Read funktioniert überall"""
        RestrictedVFSWrapper = sub_agent.RestrictedVFSWrapper

        mock_vfs = MockVFS()
        mock_vfs.files["/other/file.txt"] = "content"

        wrapper = RestrictedVFSWrapper(mock_vfs, "/sub/task1")

        result = wrapper.read("/other/file.txt")
        self.assertTrue(result["success"])

    def test_write_restricted(self):
        """Test Write nur im erlaubten Verzeichnis"""
        RestrictedVFSWrapper = sub_agent.RestrictedVFSWrapper

        mock_vfs = MockVFS()
        wrapper = RestrictedVFSWrapper(mock_vfs, "/sub/task1")

        # Erlaubt
        result1 = wrapper.write("/sub/task1/result.md", "content")
        self.assertTrue(result1["success"])

        # Nicht erlaubt
        result2 = wrapper.write("/other/file.txt", "content")
        self.assertFalse(result2["success"])
        self.assertIn("Sub-agent can only write", result2["error"])

    def test_mkdir_restricted(self):
        """Test Mkdir nur im erlaubten Verzeichnis"""
        RestrictedVFSWrapper = sub_agent.RestrictedVFSWrapper

        mock_vfs = MockVFS()
        wrapper = RestrictedVFSWrapper(mock_vfs, "/sub/task1")

        # Erlaubt
        result1 = wrapper.mkdir("/sub/task1/subdir")
        self.assertTrue(result1["success"])

        # Nicht erlaubt
        result2 = wrapper.mkdir("/other/dir")
        self.assertFalse(result2["success"])


class TestSubAgentState(unittest.TestCase):
    """Tests für SubAgentState"""

    def test_state_creation(self):
        """Test State wird korrekt erstellt"""
        SubAgentState = sub_agent.SubAgentState
        SubAgentConfig = sub_agent.SubAgentConfig
        SubAgentStatus = sub_agent.SubAgentStatus

        config = SubAgentConfig(max_tokens=3000)
        state = SubAgentState(
            id="sub_123",
            task="Do something",
            output_dir="/sub/task",
            config=config
        )

        self.assertEqual(state.id, "sub_123")
        self.assertEqual(state.status, SubAgentStatus.PENDING)
        self.assertEqual(state.config.max_tokens, 3000)


class TestSubAgentManager(unittest.TestCase):
    """Tests für SubAgentManager"""

    def test_can_spawn_main_agent(self):
        """Test Main Agent kann spawnen"""
        SubAgentManager = sub_agent.SubAgentManager

        manager = SubAgentManager(
            parent_engine=MagicMock(),
            parent_session=MockSession(),
            is_sub_agent=False
        )

        self.assertTrue(manager.can_spawn())

    def test_cannot_spawn_sub_agent(self):
        """Test Sub-Agent kann NICHT spawnen"""
        SubAgentManager = sub_agent.SubAgentManager

        manager = SubAgentManager(
            parent_engine=MagicMock(),
            parent_session=MockSession(),
            is_sub_agent=True  # Das ist ein Sub-Agent
        )

        self.assertFalse(manager.can_spawn())

    def test_spawn_raises_for_sub_agent(self):
        """Test spawn() wirft Error für Sub-Agent"""
        SubAgentManager = sub_agent.SubAgentManager

        manager = SubAgentManager(
            parent_engine=MagicMock(),
            parent_session=MockSession(),
            is_sub_agent=True
        )

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(manager.spawn("task", "/sub/out"))

        self.assertIn("cannot spawn other sub-agents", str(ctx.exception))

    def test_format_results_for_auto_focus(self):
        """Test AutoFocus Formatting"""
        SubAgentManager = sub_agent.SubAgentManager
        SubAgentResult = sub_agent.SubAgentResult
        SubAgentStatus = sub_agent.SubAgentStatus

        manager = SubAgentManager(
            parent_engine=MagicMock(),
            parent_session=MockSession(),
            is_sub_agent=False
        )

        results = {
            "sub_1": SubAgentResult(
                id="sub_1",
                success=True,
                status=SubAgentStatus.COMPLETED,
                result="Done",
                error=None,
                output_dir="/sub/task1",
                files_written=["/sub/task1/result.md"],
                tokens_used=1000,
                duration_seconds=5.0
            ),
            "sub_2": SubAgentResult(
                id="sub_2",
                success=False,
                status=SubAgentStatus.TIMEOUT,
                result=None,
                error="Timeout after 300s",
                output_dir="/sub/task2",
                files_written=[],
                tokens_used=500,
                duration_seconds=300.0
            )
        }

        formatted = manager.format_results_for_auto_focus(results)

        self.assertIn("SUB-AGENT ERGEBNISSE", formatted)
        self.assertIn("✅", formatted)
        self.assertIn("❌", formatted)
        self.assertIn("sub_1", formatted)
        self.assertIn("sub_2", formatted)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestExecutionEngineIntegration(unittest.TestCase):
    """Integration Tests für ExecutionEngine"""

    def test_engine_initialization(self):
        """Test Engine wird korrekt initialisiert"""
        ExecutionEngine = execution_engine.ExecutionEngine

        agent = MockAgent()
        engine = ExecutionEngine(agent)

        self.assertIsNotNone(engine.skills_manager)
        self.assertFalse(engine.is_sub_agent)
        self.assertIn("parallel_subtasks", engine.skills_manager.skills)

    def test_sub_agent_engine_initialization(self):
        """Test Sub-Agent Engine hat keine spawn tools"""
        ExecutionEngine = execution_engine.ExecutionEngine
        ExecutionContext = execution_engine.ExecutionContext

        agent = MockAgent()
        engine = ExecutionEngine(
            agent,
            is_sub_agent=True,
            sub_agent_output_dir="/sub/test"
        )

        self.assertTrue(engine.is_sub_agent)

        # Tool definitions sollten keine sub-agent tools haben
        ctx = ExecutionContext()
        tools = engine._get_tool_definitions(ctx)
        tool_names = [t["function"]["name"] for t in tools]

        self.assertNotIn("spawn_sub_agent", tool_names)
        self.assertNotIn("wait_for", tool_names)

    def test_main_agent_has_sub_agent_tools(self):
        """Test Main Agent hat spawn/wait_for tools"""
        ExecutionEngine = execution_engine.ExecutionEngine
        ExecutionContext = execution_engine.ExecutionContext

        agent = MockAgent()
        engine = ExecutionEngine(agent, is_sub_agent=False)

        ctx = ExecutionContext()
        tools = engine._get_tool_definitions(ctx)
        tool_names = [t["function"]["name"] for t in tools]

        self.assertIn("spawn_sub_agent", tool_names)
        self.assertIn("wait_for", tool_names)

    def test_build_system_prompt_main(self):
        """Test System Prompt für Main Agent"""
        ExecutionEngine = execution_engine.ExecutionEngine
        ExecutionContext = execution_engine.ExecutionContext

        agent = MockAgent()
        engine = ExecutionEngine(agent)

        ctx = ExecutionContext()
        session = MockSession()

        prompt = engine._build_system_prompt(ctx, session)

        self.assertIn("IDENTITY: You are FlowAgent, an autonomous execution unit capable of", prompt)
        self.assertIn("vfs", prompt)

    def test_build_system_prompt_sub_agent(self):
        """Test System Prompt für Sub-Agent"""
        ExecutionEngine = execution_engine.ExecutionEngine
        ExecutionContext = execution_engine.ExecutionContext

        agent = MockAgent()
        engine = ExecutionEngine(
            agent,
            is_sub_agent=True,
            sub_agent_output_dir="/sub/task"
        )

        ctx = ExecutionContext()
        session = MockSession()

        prompt = engine._build_system_prompt(ctx, session)

        self.assertIn("SUB-AGENT", prompt)
        self.assertIn("/sub/task", prompt)
        self.assertIn("You are a focused SUB-AGENT", prompt)


# =============================================================================
# ASYNC TESTS
# =============================================================================

class TestAsyncOperations(unittest.TestCase):
    """Tests für async Operationen"""

    def test_async_skill_matching(self):
        """Test async Skill Matching mit Embedding"""
        SkillsManager = skills.SkillsManager

        async def run_test():
            memory = MockMemory()
            manager = SkillsManager(
                agent_name="test",
                memory_instance=memory
            )

            matches = await manager.match_skills_async("Speichere meine Präferenzen")
            self.assertGreater(len(matches), 0)

        asyncio.run(run_test())

    def test_async_wait_for_unknown_id(self):
        """Test wait_for mit unbekannter ID"""
        SubAgentManager = sub_agent.SubAgentManager
        SubAgentStatus = sub_agent.SubAgentStatus

        async def run_test():
            manager = SubAgentManager(
                parent_engine=MagicMock(),
                parent_session=MockSession(),
                is_sub_agent=False
            )

            results = await manager.wait_for("unknown_id")

            self.assertIn("unknown_id", results)
            self.assertFalse(results["unknown_id"].success)
            self.assertEqual(results["unknown_id"].status, SubAgentStatus.FAILED)

        asyncio.run(run_test())


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
