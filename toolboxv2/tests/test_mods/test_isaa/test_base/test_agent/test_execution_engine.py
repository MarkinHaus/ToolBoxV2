"""
Test Suite für ExecutionEngine V3

Demonstriert die Kernverbesserungen:
1. ChatHistoryManager - Strict ChatML Compliance
2. Auto-Focus Tracker - Dynamic Context Injection
3. Loop Detection - Semantic Matching
4. Tool Discovery - Dynamic Tool Loading
"""

import unittest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass

# Import from V3
# Import from V3
from toolboxv2.mods.isaa.base.Agent.execution_engine import (
    ChatHistoryManager,
    AutoFocusTracker,
    LoopDetector,
    ToolDiscoveryManager,
    ExecutionEngine,
    ExecutionConfig,
    ExecutionStatus,
    TerminationReason,
)


class TestChatHistoryManager(unittest.TestCase):
    """
    Tests für den ChatHistoryManager - Der Kern des WTF-Bug-Fixes
    """

    def setUp(self):
        self.manager = ChatHistoryManager()

    def test_basic_message_flow(self):
        """Test: Einfacher Nachrichtenfluss"""
        self.manager.add_system("System prompt")
        self.manager.add_user("Hello")
        self.manager.add_assistant_text("Hi there!")

        messages = self.manager.get_messages()

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")

    def test_tool_call_cycle_strict_compliance(self):
        """
        KRITISCHER TEST: ChatML Compliance

        Prüft, dass der Zyklus korrekt ist:
        1. Assistant mit tool_calls
        2. Tool mit tool_call_id

        Dies ist der Fix für den WTF-Bug in V2!
        """
        self.manager.add_system("System")
        self.manager.add_user("Create a file")

        # Simuliere LLM Response mit Tool Call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "vfs_create"
        mock_tool_call.function.arguments = '{"filename": "test.txt"}'

        # KRITISCH: Assistant Message MIT tool_calls hinzufügen
        self.manager.add_assistant_with_tools(
            content="I'll create that file.",
            tool_calls=[mock_tool_call]
        )

        # Tool Result hinzufügen
        self.manager.add_tool_result(
            tool_call_id="call_123",
            content="File created successfully",
            name="vfs_create"
        )

        messages = self.manager.get_messages()

        # Prüfe Struktur
        self.assertEqual(len(messages), 4)

        # Assistant Message MUSS tool_calls enthalten
        assistant_msg = messages[2]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIn("tool_calls", assistant_msg)
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        self.assertEqual(assistant_msg["tool_calls"][0]["id"], "call_123")
        self.assertEqual(assistant_msg["tool_calls"][0]["function"]["name"], "vfs_create")

        # Tool Message MUSS tool_call_id haben
        tool_msg = messages[3]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_123")

    def test_multiple_tool_calls_in_sequence(self):
        """Test: Mehrere Tool Calls nacheinander"""
        self.manager.add_system("System")
        self.manager.add_user("Do multiple things")

        # Erste Tool-Call-Runde
        tc1 = Mock(id="call_1", function=Mock(name="tool_a", arguments="{}"))
        self.manager.add_assistant_with_tools(None, [tc1])
        self.manager.add_tool_result("call_1", "Result 1")

        # Zweite Tool-Call-Runde
        tc2 = Mock(id="call_2", function=Mock(name="tool_b", arguments="{}"))
        self.manager.add_assistant_with_tools("Thinking...", [tc2])
        self.manager.add_tool_result("call_2", "Result 2")

        messages = self.manager.get_messages()

        # Prüfe, dass beide Zyklen korrekt sind
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("tool_calls", messages[2])
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_1")

        self.assertEqual(messages[4]["role"], "assistant")
        self.assertIn("tool_calls", messages[4])
        self.assertEqual(messages[5]["role"], "tool")
        self.assertEqual(messages[5]["tool_call_id"], "call_2")

    def test_context_injection(self):
        """Test: Dynamic Context Injection (Auto-Focus)"""
        self.manager.add_system("System")
        self.manager.add_user("Do something")

        # Injeziere Context VOR dem letzten User Message
        self.manager.inject_context("Last action: Created file.txt", "before_last_user")

        messages = self.manager.get_messages()

        # Context sollte vor User sein
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "system")  # Injected context
        self.assertIn("Last action", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "user")

    def test_history_trimming(self):
        """Test: History Trimming behält System Message"""
        manager = ChatHistoryManager(max_history=5)
        manager.add_system("System prompt")

        # Füge viele Messages hinzu
        for i in range(10):
            manager.add_user(f"Message {i}")

        messages = manager.get_messages()

        # System Message sollte immer erhalten bleiben
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")


class TestAutoFocusTracker(unittest.TestCase):
    """Tests für Auto-Focus Tracker"""

    def setUp(self):
        self.tracker = AutoFocusTracker(max_entries=3)

    def test_vfs_recording(self):
        """Test: VFS Operationen werden aufgezeichnet"""
        self.tracker.record_vfs("test.txt", "created", "Hello World")

        context = self.tracker.build_context()

        self.assertIn("test.txt", context)
        self.assertIn("CREATED", context)
        self.assertIn("Hello World", context)

    def test_tool_recording(self):
        """Test: Tool Ergebnisse werden aufgezeichnet"""
        self.tracker.record_tool("web_search", "Found 10 results")

        context = self.tracker.build_context()

        self.assertIn("web_search", context)
        self.assertIn("Found 10 results", context)

    def test_max_entries_limit(self):
        """Test: Max Entries werden respektiert"""
        self.tracker.record_vfs("file1.txt", "created", "1")
        self.tracker.record_vfs("file2.txt", "created", "2")
        self.tracker.record_vfs("file3.txt", "created", "3")
        self.tracker.record_vfs("file4.txt", "created", "4")  # Sollte file1 verdrängen

        context = self.tracker.build_context()

        self.assertNotIn("file1.txt", context)
        self.assertIn("file4.txt", context)

    def test_empty_context(self):
        """Test: Leerer Tracker gibt leeren String zurück"""
        context = self.tracker.build_context()
        self.assertEqual(context, "")


class TestLoopDetector(unittest.TestCase):
    """Tests für Loop Detection"""

    def setUp(self):
        self.detector = LoopDetector(threshold=3)

    def test_no_loop_initially(self):
        """Test: Kein Loop am Anfang"""
        is_loop, reason = self.detector.detect()
        self.assertFalse(is_loop)

    def test_detects_repeated_tool_calls(self):
        """Test: Erkennt wiederholte Tool Calls"""
        # Gleicher Tool Call 3x
        for _ in range(3):
            self.detector.record("vfs_create", {"filename": "test.txt"})

        is_loop, reason = self.detector.detect()

        self.assertTrue(is_loop)
        self.assertIn("vfs_create", reason)

    def test_ignores_variable_fields(self):
        """Test: Ignoriert variable Felder wie UUIDs"""
        # Calls mit verschiedenen UUIDs sollten als gleich erkannt werden
        self.detector.record("tool", {"id": "550e8400-e29b-41d4-a716-446655440000"})
        self.detector.record("tool", {"id": "550e8400-e29b-41d4-a716-446655440001"})
        self.detector.record("tool", {"id": "550e8400-e29b-41d4-a716-446655440002"})

        is_loop, reason = self.detector.detect()

        self.assertTrue(is_loop)

    def test_different_args_no_loop(self):
        """Test: Verschiedene Args = kein Loop"""
        self.detector.record("vfs_create", {"filename": "file1.txt"})
        self.detector.record("vfs_create", {"filename": "file2.txt"})
        self.detector.record("vfs_create", {"filename": "file3.txt"})

        is_loop, reason = self.detector.detect()

        self.assertFalse(is_loop)

    def test_reset(self):
        """Test: Reset löscht alle Daten"""
        for _ in range(3):
            self.detector.record("tool", {"x": 1})

        self.detector.reset()

        is_loop, _ = self.detector.detect()
        self.assertFalse(is_loop)


class TestToolDiscoveryManager(unittest.TestCase):
    """Tests für Tool Discovery Manager - Dynamic Tool Loading"""

    def setUp(self):
        # Mock Agent mit Tool Manager
        self.mock_agent = Mock()
        self.mock_agent.tool_manager = Mock()

        # Mock Tools
        mock_tool_entry_1 = Mock()
        mock_tool_entry_1.name = "discord_send"
        mock_tool_entry_1.category = ["discord", "messaging"]

        mock_tool_entry_2 = Mock()
        mock_tool_entry_2.name = "web_search"
        mock_tool_entry_2.category = ["web"]

        mock_tool_entry_3 = Mock()
        mock_tool_entry_3.name = "discord_read"
        mock_tool_entry_3.category = ["discord"]

        self.mock_agent.tool_manager.get_all.return_value = [
            mock_tool_entry_1, mock_tool_entry_2, mock_tool_entry_3
        ]

        self.mock_agent.tool_manager.get_all_litellm.return_value = [
            {"type": "function", "function": {"name": "discord_send", "description": "Send Discord message"}},
            {"type": "function", "function": {"name": "web_search", "description": "Search the web"}},
            {"type": "function", "function": {"name": "discord_read", "description": "Read Discord messages"}},
        ]

        self.discovery = ToolDiscoveryManager(self.mock_agent, max_active=3)

    def test_discover_by_keyword(self):
        """Test: Finde Tools nach Keyword"""
        results = self.discovery.discover("discord")

        self.assertEqual(len(results), 2)
        names = [r['name'] for r in results]
        self.assertIn("discord_send", names)
        self.assertIn("discord_read", names)

    def test_discover_by_category(self):
        """Test: Finde Tools nach Kategorie"""
        results = self.discovery.discover("message", category="discord")

        # Nur discord Tools
        for r in results:
            self.assertIn("discord", r['category'])

    def test_load_tools(self):
        """Test: Tools laden"""
        result = self.discovery.load(["discord_send", "web_search"])

        self.assertEqual(len(result['loaded']), 2)
        self.assertIn("discord_send", result['loaded'])
        self.assertIn("web_search", result['loaded'])
        self.assertEqual(result['active_count'], 2)

    def test_max_active_limit(self):
        """Test: Max 3 aktive Tools"""
        self.discovery.load(["discord_send", "web_search", "discord_read"])

        # Versuche ein 4. Tool zu laden (sollte fehlschlagen)
        # Erst ein Mock-Tool hinzufügen
        self.mock_agent.tool_manager.get_all_litellm.return_value.append(
            {"type": "function", "function": {"name": "extra_tool", "description": "Extra"}}
        )
        self.discovery._tool_cache["extra_tool"] = {"type": "function", "function": {"name": "extra_tool"}}

        result = self.discovery.load(["extra_tool"])

        self.assertIn("extra_tool", result['failed'][0])
        self.assertEqual(result['active_count'], 3)

    def test_unload_tools(self):
        """Test: Tools entladen"""
        self.discovery.load(["discord_send", "web_search"])

        result = self.discovery.unload(["discord_send"])

        self.assertIn("discord_send", result['unloaded'])
        self.assertEqual(result['active_count'], 1)
        self.assertFalse(self.discovery.is_tool_active("discord_send"))
        self.assertTrue(self.discovery.is_tool_active("web_search"))

    def test_is_tool_active(self):
        """Test: Prüfe ob Tool aktiv ist"""
        self.assertFalse(self.discovery.is_tool_active("discord_send"))

        self.discovery.load(["discord_send"])

        self.assertTrue(self.discovery.is_tool_active("discord_send"))

    def test_get_active_tools_litellm(self):
        """Test: Hole aktive Tools im LiteLLM Format"""
        self.discovery.load(["discord_send"])

        tools = self.discovery.get_active_tools_litellm()

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]['function']['name'], "discord_send")

    def test_discover_shows_loaded_status(self):
        """Test: Discover zeigt an ob Tool geladen ist"""
        self.discovery.load(["discord_send"])

        results = self.discovery.discover("discord")

        for r in results:
            if r['name'] == "discord_send":
                self.assertTrue(r['loaded'])
            else:
                self.assertFalse(r['loaded'])


class TestExecutionEngineIntegration(unittest.TestCase):
    """Integration Tests für ExecutionEngine"""

    def setUp(self):
        # Mock Agent
        self.mock_agent = Mock()
        self.mock_agent.tool_manager = Mock()
        self.mock_agent.tool_manager.get_all_litellm.return_value = []
        self.mock_agent.tool_manager.get_all.return_value = []
        self.mock_agent.session_manager = Mock()

        # Mock Session
        self.mock_session = Mock()
        self.mock_session.vfs = Mock()
        self.mock_session.vfs.read.return_value = {"success": True, "content": "test content"}
        self.mock_session.vfs.create.return_value = {"success": True}
        self.mock_session.vfs.write.return_value = {"success": True}
        self.mock_session.vfs.list_files.return_value = ["file1.txt", "file2.txt"]

        self.mock_agent.session_manager.get_or_create = AsyncMock(return_value=self.mock_session)

    def test_engine_initialization(self):
        """Test: Engine kann initialisiert werden"""
        engine = ExecutionEngine(
            agent=self.mock_agent,
            human_online=False,
            callback=print,
            max_active_tools=5
        )

        self.assertIsNotNone(engine)
        self.assertEqual(engine.max_active_tools, 5)

    def test_tool_preparation_includes_discovery(self):
        """Test: Tools enthalten Discovery Tools"""
        engine = ExecutionEngine(agent=self.mock_agent)

        config = ExecutionConfig()
        state = Mock()
        state.session_id = "test"
        state.config = config

        discovery = engine._get_discovery("test_exec")
        tools = engine._prepare_tools(state, discovery)

        tool_names = [t['function']['name'] for t in tools]

        # System Tools
        self.assertIn("vfs_read", tool_names)
        self.assertIn("final_answer", tool_names)

        # Discovery Tools
        self.assertIn("discover_tools", tool_names)
        self.assertIn("load_tools", tool_names)

    def test_active_tools_status(self):
        """Test: Status String für aktive Tools"""
        engine = ExecutionEngine(agent=self.mock_agent, max_active_tools=5)
        discovery = engine._get_discovery("test")

        # Keine Tools geladen
        status = engine._build_active_tools_status(discovery)
        self.assertIn("Keine", status)

        # Tools laden (manuell für Test)
        discovery._tool_cache["test_tool"] = {"type": "function", "function": {"name": "test_tool"}}
        discovery.load(["test_tool"])

        status = engine._build_active_tools_status(discovery)
        self.assertIn("test_tool", status)
        self.assertIn("1/5", status)


class TestChatMLComplianceScenarios(unittest.TestCase):
    """
    Spezifische Szenarien, die den WTF-Bug in V2 ausgelöst haben
    """

    def test_scenario_v2_bug_recreation(self):
        """
        Szenario: Der Bug in V2

        V2 hat oft nur das Tool-Result hinzugefügt, ohne den
        vorhergehenden Assistant-Message mit tool_calls.

        Das führte dazu, dass das Modell nicht wusste, welche
        Aktion es gerade ausgeführt hatte.
        """
        manager = ChatHistoryManager()
        manager.add_system("System")
        manager.add_user("Create test.txt")

        # FALSCH (V2 Bug): Nur Result ohne vorhergehenden Call
        # manager._messages.append({"role": "tool", "content": "Created"})

        # RICHTIG (V3): Erst Assistant mit tool_calls, dann Result
        tc = Mock(id="call_1", function=Mock(name="vfs_create", arguments='{}'))
        manager.add_assistant_with_tools("Creating...", [tc])
        manager.add_tool_result("call_1", "Created successfully")

        messages = manager.get_messages()

        # Validiere, dass die Reihenfolge stimmt
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])

        # Assistant MUSS tool_calls haben
        self.assertIn("tool_calls", messages[2])

    def test_scenario_multi_tool_parallel(self):
        """
        Szenario: Mehrere Tools in einem Response

        LLMs können mehrere tool_calls in einer Response senden.
        Alle müssen korrekt in der History landen.
        """
        manager = ChatHistoryManager()
        manager.add_system("System")
        manager.add_user("Search and create file")

        # Ein Assistant-Call mit mehreren Tools
        tc1 = Mock(id="call_1", function=Mock(name="web_search", arguments='{}'))
        tc2 = Mock(id="call_2", function=Mock(name="vfs_create", arguments='{}'))

        manager.add_assistant_with_tools("I'll do both", [tc1, tc2])

        # Beide Results
        manager.add_tool_result("call_1", "Search results")
        manager.add_tool_result("call_2", "File created")

        messages = manager.get_messages()

        # Ein Assistant mit 2 tool_calls, dann 2 Tool Results
        self.assertEqual(len(messages[2]["tool_calls"]), 2)
        self.assertEqual(messages[3]["tool_call_id"], "call_1")
        self.assertEqual(messages[4]["tool_call_id"], "call_2")


class TestDiscoveryWorkflow(unittest.TestCase):
    """Tests für den kompletten Discovery Workflow"""

    def test_discover_load_use_unload_workflow(self):
        """Test: Kompletter Workflow discover → load → use → unload"""
        mock_agent = Mock()
        mock_agent.tool_manager.get_all.return_value = [
            Mock(name="email_send", category=["email"]),
        ]
        mock_agent.tool_manager.get_all_litellm.return_value = [
            {"type": "function", "function": {"name": "email_send", "description": "Send email"}}
        ]

        discovery = ToolDiscoveryManager(mock_agent, max_active=3)

        # 1. Discover
        results = discovery.discover("email")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], "email_send")
        self.assertFalse(results[0]['loaded'])

        # 2. Load
        load_result = discovery.load(["email_send"])
        self.assertIn("email_send", load_result['loaded'])
        self.assertTrue(discovery.is_tool_active("email_send"))

        # 3. Check discover shows loaded
        results = discovery.discover("email")
        self.assertTrue(results[0]['loaded'])

        # 4. Unload
        unload_result = discovery.unload(["email_send"])
        self.assertIn("email_send", unload_result['unloaded'])
        self.assertFalse(discovery.is_tool_active("email_send"))


def run_async_test(coro):
    """Helper für async Tests"""
    return asyncio.get_event_loop().run_until_complete(coro)


if __name__ == "__main__":
    unittest.main(verbosity=2)
