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
    system_message: str = """# ROLE: Isaa (spoken Asa)"""


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
        self.assertIn("Total: 2 tool calls, 2 unique tools", summary["content"])

    def test_compress_partial(self):
        """Test Partial Compression"""
        HistoryCompressor = execution_engine.HistoryCompressor

        working_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "System prompt"},
            {"role": "tool", "name": "tool1", "content": "r1"*10000},
            {"role": "assistant", "content": "System prompt"},
            {"role": "tool", "name": "tool2", "content": "r2"*10000},
            {"role": "assistant", "content": "System prompt"},
            {"role": "tool", "name": "tool3", "content": "r3"*10000},
            {"role": "tool", "name": "tool4", "content": "r4"*10000},
            {"role": "assistant", "content": "System prompt"},
            {"role": "tool", "name": "tool5", "content": "r5"*10000},
        ]

        summary, remaining = HistoryCompressor.compress_partial(
            working_history, keep_last_n=1
        )

        self.assertIsNotNone(summary)
        # Should keep system + summary + last 2
        from litellm import token_counter
        b,a = token_counter(messages=working_history),token_counter(messages=remaining)
        print(b,'#',a)
        self.assertLess(a,b//2)
        self.assertEqual(len(remaining), len(working_history)+1)
        self.assertEqual(len(remaining), len(working_history)+1)
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


# =============================================================================
# TOOL_CALL_ID PAIRING INTEGRITY  (MiniMax 400-Fehler Regression)
# =============================================================================

def _collect_orphan_tool_ids(history: list) -> list:
    """
    Findet tool_call_ids die in assistant-Messages referenziert werden,
    aber keine passende tool-Message als Antwort haben.
    Leere Liste = valide History.
    """
    answered = set()
    for msg in history:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                answered.add(tc_id)

    orphans = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                tc_id = tc.get("id", "")
            else:
                tc_id = getattr(tc, "id", "")
            if tc_id and tc_id not in answered:
                orphans.append(tc_id)
    return orphans


def _make_tool_pair(tc_id: str, tool_name: str = "my_tool", result: str = "ok"):
    """Erzeugt ein valides (assistant, tool) Message-Paar."""
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": tc_id, "type": "function",
             "function": {"name": tool_name, "arguments": "{}"}},
        ],
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": tool_name,
        "content": result,
    }
    return assistant_msg, tool_msg


def _build_realistic_history(n_pairs: int, system_prompt: str = "You are FlowAgent.") -> list:
    """Baut eine realistische working_history mit n_pairs tool-call-Runden."""
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Do something complex."},
    ]
    for i in range(n_pairs):
        tc_id = f"call_test_{i:04d}"
        asst, tool = _make_tool_pair(tc_id, f"tool_{i % 6}", f"result_{i}")
        history.append(asst)
        history.append(tool)
    return history


class TestToolCallIdPairingAfterCompressPartial(unittest.TestCase):
    """compress_partial darf niemals orphaned tool_call_ids hinterlassen (MiniMax 400 Regression)."""

    def setUp(self):
        self.HC = execution_engine.HistoryCompressor

    def test_clean_history_has_no_orphans(self):
        """Baseline: frisch aufgebaute History hat keine orphan IDs."""
        history = _build_realistic_history(5)
        self.assertEqual(_collect_orphan_tool_ids(history), [])

    def test_compress_partial_no_orphans_small(self):
        """compress_partial mit kleiner History hinterlässt keine orphan IDs."""
        history = _build_realistic_history(8)
        _, compressed = self.HC.compress_partial(history, keep_last_n=3)
        self.assertEqual(_collect_orphan_tool_ids(compressed), [],
            "compress_partial erzeugt orphan tool_call_ids")

    def test_compress_partial_no_orphans_large(self):
        """compress_partial mit 50 Paaren hinterlässt keine orphan IDs."""
        history = _build_realistic_history(50)
        _, compressed = self.HC.compress_partial(history, keep_last_n=5)
        self.assertEqual(_collect_orphan_tool_ids(compressed), [],
            "compress_partial (50 pairs) erzeugt orphan IDs")

    def test_compress_partial_no_orphans_keep_1(self):
        """Extremfall keep_last_n=1: immer noch kein orphan."""
        history = _build_realistic_history(10)
        _, compressed = self.HC.compress_partial(history, keep_last_n=1)
        self.assertEqual(_collect_orphan_tool_ids(compressed), [],
            "compress_partial(keep=1) erzeugt orphan IDs")

    def test_compress_partial_with_read_tools_no_orphans(self):
        """P3-Tools (read/list/view) werden summarized - kein orphan durch Drop."""
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        for name in ("vfs_read", "vfs_list", "vfs_view", "vfs_open"):
            asst, tool = _make_tool_pair(f"call_{name}", name, "data")
            history.append(asst)
            history.append(tool)
        for i in range(5):
            asst, tool = _make_tool_pair(f"call_pad_{i}", "think", "thought")
            history.append(asst)
            history.append(tool)

        _, compressed = self.HC.compress_partial(history, keep_last_n=3)
        self.assertEqual(_collect_orphan_tool_ids(compressed), [],
            "P3-Tool-Summarizing erzeugt orphan IDs")

    def test_compress_partial_parallel_tool_calls_no_orphans(self):
        """Mehrere tool_calls in einer assistant-Message bleiben alle beantwortet."""
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "parallel task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_A", "type": "function",
                     "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_B", "type": "function",
                     "function": {"name": "tool_b", "arguments": "{}"}},
                    {"id": "call_C", "type": "function",
                     "function": {"name": "tool_c", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_A", "name": "tool_a", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_B", "name": "tool_b", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_C", "name": "tool_c", "content": "ok"},
        ]
        for i in range(5):
            asst, tool = _make_tool_pair(f"pad_{i}", "think", "t")
            history.append(asst)
            history.append(tool)

        _, compressed = self.HC.compress_partial(history, keep_last_n=3)
        self.assertEqual(_collect_orphan_tool_ids(compressed), [],
            "Parallele tool_calls erzeugen orphan IDs nach compress")


class TestToolCallIdPairingOver1000Steps(unittest.TestCase):
    """Stress-Test: 1000 Iterations ohne einen einzigen orphan tool_call_id."""

    def setUp(self):
        self.HC = execution_engine.HistoryCompressor

    def test_repeated_compress_1000_steps_no_orphans(self):
        """1000 Steps mit compress alle 10: kein einziger orphan."""
        history = [
            {"role": "system", "content": "You are FlowAgent."},
            {"role": "user", "content": "Long task."},
        ]
        violations = []

        for step in range(1000):
            asst, tool = _make_tool_pair(f"call_{step:04d}", f"tool_{step % 8}", f"r_{step}")
            history.append(asst)
            history.append(tool)

            if step > 0 and step % 10 == 0:
                _, history = self.HC.compress_partial(history, keep_last_n=4)
                orphans = _collect_orphan_tool_ids(history)
                if orphans:
                    violations.append((step, orphans[:3]))

        self.assertEqual(violations, [],
            f"Orphan IDs entstanden bei Steps: {violations[:5]}")

    def test_history_format_consistent_after_1000_steps(self):
        """Alle Messages behalten konsistentes Format nach 1000 Steps."""
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        for step in range(1000):
            asst, tool = _make_tool_pair(f"call_{step:04d}", "any_tool", "result")
            history.append(asst)
            history.append(tool)
            if step % 15 == 0 and step > 0:
                _, history = self.HC.compress_partial(history, keep_last_n=5)

        for i, msg in enumerate(history):
            role = msg.get("role")
            self.assertIn(role, ("system", "user", "assistant", "tool"),
                f"Msg {i} hat ungültigen role: {role!r}")
            if role == "tool":
                self.assertIn("tool_call_id", msg,
                    f"tool-Message {i} fehlt tool_call_id (MiniMax 400)")
                self.assertIn("content", msg,
                    f"tool-Message {i} fehlt content")
            if role == "assistant":
                self.assertIn("content", msg,
                    f"assistant-Message {i} fehlt content-Key")

    def test_retroactive_offload_preserves_tool_call_id(self):
        """_retroactive_offload ersetzt Content aber muss tool_call_id erhalten."""
        ExecutionEngine = execution_engine.ExecutionEngine
        ExecutionContext = execution_engine.ExecutionContext

        agent = MockAgent()
        engine = ExecutionEngine(agent)
        ctx = ExecutionContext()
        ctx.run_id = "test_offload"
        engine._current_session = MockSession()

        big_content = "X" * 5000
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        for i in range(10):
            asst, tool = _make_tool_pair(f"call_off_{i}", "big_tool", big_content)
            history.append(asst)
            history.append(tool)

        ctx.working_history = history
        engine._retroactive_offload(ctx, engine._current_session, tokens_needed=1000)

        self.assertEqual(_collect_orphan_tool_ids(ctx.working_history), [],
            "_retroactive_offload erzeugt orphan IDs")
        for msg in ctx.working_history:
            if msg.get("role") == "tool":
                self.assertIn("tool_call_id", msg,
                    "_retroactive_offload entfernt tool_call_id (MiniMax 400!)")
                self.assertNotEqual(msg["tool_call_id"], "",
                    "_retroactive_offload leert tool_call_id")


class TestInternalToolExecution(unittest.IsolatedAsyncioTestCase):
    """_execute_tool_call für statische interne Tools: think, final_answer, list_tools, shift_focus."""

    def _tc(self, name: str, args: dict, tc_id: str = "call_001"):
        tc = MagicMock()
        tc.id = tc_id
        tc.function.name = name
        tc.function.arguments = json.dumps(args)
        return tc

    def _engine(self, is_sub=False):
        agent = MockAgent()
        return execution_engine.ExecutionEngine(agent, is_sub_agent=is_sub)

    def _ctx(self):
        ctx = execution_engine.ExecutionContext()
        ctx.run_id = "test_run"
        return ctx

    async def test_think_not_final(self):
        e, ctx = self._engine(), self._ctx()
        _, is_final = await e._execute_tool_call(ctx, self._tc("think", {"thought": "..."}))
        self.assertFalse(is_final)

    async def test_think_en(self):
        e, ctx = self._engine(), self._ctx()
        res, is_final = await e._execute_tool_call(ctx, self._tc("think", {"thought": "..."}))
        print(res)
        self.assertIsNotNone(res)

    async def test_think_live(self):
        e, ctx = self._engine(), self._ctx()
        res, is_final = await e._execute_tool_call(ctx, self._tc("think", {"thought": "Two parallel analyses needed:\n1. Worker-System + Worker..."}))
        print(res)
        self.assertIsNotNone(res)

    async def test_think_decrements_iteration(self):
        e, ctx = self._engine(), self._ctx()
        ctx.max_iterations = 5
        await e._execute_tool_call(ctx, self._tc("think", {"thought": "plan"}))
        self.assertEqual(ctx.max_iterations, 6,
            "think muss max_iterations incrementer")

    async def test_think_records_in_auto_focus(self):
        e, ctx = self._engine(), self._ctx()
        await e._execute_tool_call(ctx, self._tc("think", {"thought": "insight"}))
        self.assertGreater(len(ctx.auto_focus.actions), 0,
            "think muss in AutoFocus aufgezeichnet werden")

    async def test_final_answer_is_final(self):
        e, ctx = self._engine(), self._ctx()
        result, is_final = await e._execute_tool_call(
            ctx, self._tc("final_answer", {"answer": "Done!", "success": True}))
        self.assertTrue(is_final)
        self.assertEqual(result, "Done!")

    async def test_final_answer_not_in_history(self):
        e, ctx = self._engine(), self._ctx()
        init_len = len(ctx.working_history)
        await e._execute_tool_call(
            ctx, self._tc("final_answer", {"answer": "Done!"}))
        self.assertEqual(len(ctx.working_history), init_len,
            "final_answer darf nicht in working_history landen")

    async def test_list_tools_decrements_iteration(self):
        e, ctx = self._engine(), self._ctx()
        ctx.max_iterations = 5
        await e._execute_tool_call(ctx, self._tc("list_tools", {}))
        self.assertEqual(ctx.max_iterations, 6,
            "list_tools muss max_iterations incrementer")

    async def test_list_tools_returns_string(self):
        e, ctx = self._engine(), self._ctx()
        result, is_final = await e._execute_tool_call(ctx, self._tc("list_tools", {}))
        self.assertIsInstance(result, str)
        self.assertFalse(is_final)

    async def test_shift_focus_decrements_5(self):
        e, ctx = self._engine(), self._ctx()
        ctx.max_iterations = 10
        await e._execute_tool_call(ctx, self._tc("shift_focus", {
            "summary_of_achievements": "done reading",
            "next_objective": "write output",
        }))
        self.assertEqual(ctx.max_iterations, 20,
            "shift_focus muss max_iterations um 10 incrementer")

    async def test_tool_result_has_correct_structure(self):
        """Jede non-final tool-Message muss korrekte Struktur für API haben."""
        e, ctx = self._engine(), self._ctx()
        tc_id = "call_struct_01"
        await e._execute_tool_call(ctx, self._tc("think", {"thought": "x"}, tc_id))
        last = ctx.working_history[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("tool_call_id", last, "tool_call_id fehlt (MiniMax 400!)")
        self.assertEqual(last["tool_call_id"], tc_id,
            "tool_call_id muss mit tc.id übereinstimmen")
        self.assertIn("content", last)

    async def test_minimax_style_id_preserved_exactly(self):
        """MiniMax-style IDs wie 'call_function_hn4sxvjzffmj_1' bleiben exakt erhalten."""
        e, ctx = self._engine(), self._ctx()
        minimax_id = "call_function_hn4sxvjzffmj_1"
        await e._execute_tool_call(ctx, self._tc("think", {"thought": "t"}, minimax_id))
        last = ctx.working_history[-1]
        self.assertEqual(last["tool_call_id"], minimax_id,
            f"MiniMax-ID nicht exakt erhalten, got: {last.get('tool_call_id')!r}")

    async def test_tool_call_sequence_no_orphans_in_history(self):
        """Nach think → list_tools → think: working_history hat keine orphan IDs."""
        e, ctx = self._engine(), self._ctx()
        calls = [
            ("think", {"thought": "step 1"}, "call_001"),
            ("list_tools", {}, "call_002"),
            ("think", {"thought": "step 3"}, "call_003"),
        ]
        for name, args, tc_id in calls:
            # Simuliere was die Engine macht: assistant-Message zuerst hinzufügen
            ctx.working_history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": tc_id, "type": "function",
                                "function": {"name": name, "arguments": "{}"}}],
            })
            await e._execute_tool_call(ctx, self._tc(name, args, tc_id))

        orphans = _collect_orphan_tool_ids(ctx.working_history)
        self.assertEqual(orphans, [],
            f"Sequenz think→list_tools→think hat orphan IDs: {orphans}")

    async def test_unknown_tool_no_exception(self):
        """Unbekannte Tools werfen keine Exception."""
        agent = MockAgent()
        agent.arun_function = AsyncMock(side_effect=Exception("Tool not loaded"))
        e = execution_engine.ExecutionEngine(agent)
        ctx = self._ctx()
        try:
            result, is_final = await e._execute_tool_call(
                ctx, self._tc("nonexistent_tool_xyz", {"x": 1}))
            self.assertFalse(is_final)
            self.assertIsInstance(result, str)
        except Exception as ex:
            self.fail(f"_execute_tool_call soll keine Exception werfen: {ex}")


class TestManageContextBudgetPairing(unittest.TestCase):
    """_manage_context_budget muss in allen Szenarios (A/B/C) tool_call_id exakt erhalten."""

    def _make(self):
        engine = execution_engine.ExecutionEngine(MockAgent())
        ctx = execution_engine.ExecutionContext()
        ctx.run_id = "budget_test"
        engine._current_session = MockSession()
        return engine, ctx

    def test_scenario_a_small_content(self):
        """Szenario A: content passt rein → tool_call_id erhalten."""
        e, ctx = self._make()
        msg = e._manage_context_budget(ctx, "vfs_read", "small result", "call_A01")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "call_A01")
        self.assertIn("content", msg)
        self.assertTrue(msg["content"])

    def test_scenario_c_large_offload(self):
        """Szenario C: Sofort-Offload → tool_call_id bleibt trotzdem exakt erhalten."""
        e, ctx = self._make()
        big = "A" * 100_000
        msg = e._manage_context_budget(ctx, "big_tool", big, "call_C01")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "call_C01",
            "Szenario C Offload verliert tool_call_id (MiniMax 400!)")

    def test_duplicate_hash_uses_current_call_id(self):
        """Dedup-Pfad setzt tool_call_id des AKTUELLEN Calls, nicht des gecachten."""
        e, ctx = self._make()
        content = "repeated content"
        e._manage_context_budget(ctx, "tool", content, "call_first")
        msg2 = e._manage_context_budget(ctx, "tool", content, "call_second")
        self.assertEqual(msg2["tool_call_id"], "call_second",
            "Dedup muss tool_call_id des zweiten Calls setzen, nicht des ersten")

    def test_all_paths_return_valid_structure(self):
        """Alle Pfade geben role=tool, tool_call_id, content zurück."""
        e, ctx = self._make()
        cases = [
            ("tiny", "x", "call_s"),
            ("medium", "M" * 1000, "call_m"),
            ("large", "L" * 50_000, "call_l"),
        ]
        for label, content, tc_id in cases:
            with self.subTest(label=label):
                msg = e._manage_context_budget(ctx, "t", content, tc_id)
                self.assertEqual(msg["role"], "tool", f"{label}: role falsch")
                self.assertEqual(msg["tool_call_id"], tc_id,
                    f"{label}: tool_call_id falsch")
                self.assertIn("content", msg, f"{label}: kein content-Key")
                self.assertTrue(msg["content"], f"{label}: content leer")
