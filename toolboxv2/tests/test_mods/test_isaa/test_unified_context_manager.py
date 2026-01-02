# C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\tests\test_mods\test_isaa\test_unified_context_manager.py

import unittest
import asyncio
import sys
import time
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta


# ============================================================================
# MOCKS - Unabhängig von toolboxv2 Imports
# ============================================================================

class MockVariableManager:
    """Mock für VariableManager"""

    def __init__(self):
        self.scopes = {}
        self._data = {}

    def register_scope(self, name: str, data: dict):
        self.scopes[name] = data

    def set(self, key: str, value):
        self._data[key] = value

    def get(self, key: str, default=None):
        return self._data.get(key, default)


class MockChatSession:
    """Mock für ChatSession"""

    def __init__(self):
        self.history = []

    async def add_message(self, message: dict):
        self.history.append(message)

    def get_past_x(self, count: int, last_u: bool = False) -> list:
        return self.history[-count:] if self.history else []

    async def get_reference(self, query: str) -> str:
        return f"Reference for: {query}" if query else ""

    def on_exit(self):
        pass


class MockAgentModelData:
    """Mock für AgentModelData"""

    def __init__(self, name: str = "TestAgent"):
        self.name = name


class MockAgent:
    """Mock für FlowAgent"""

    def __init__(self, amd: MockAgentModelData = None):
        self.amd = amd or MockAgentModelData()
        self.shared = {
            "tasks": {},
            "results": {},
            "system_status": "idle"
        }


# ============================================================================
# UnifiedContextManager - Standalone für Tests
# ============================================================================

def rprint(*args, **kwargs):
    """Mock rprint"""
    pass


def eprint(*args, **kwargs):
    """Mock eprint"""
    pass


class UnifiedContextManager:
    """
    Standalone Version für Tests - Copy aus dem Original.
    """

    def __init__(self, agent):
        self.agent = agent
        self.session_managers: dict = {}
        self.variable_manager = None
        self.compression_threshold = 15
        self._context_cache: dict = {}
        self.cache_ttl = 300
        self._memory_instance = None

        self._history_cache = {}
        self._variables_cache = {}
        self._execution_cache = {}

        self.HISTORY_TTL = 30
        self.VARIABLES_TTL = 10
        self.EXECUTION_TTL = 5

        self._cache_stats = {"hits": 0, "misses": 0}

    async def initialize_session(self, session_id: str, max_history: int = 200):
        """Initialisiere oder lade existierende ChatSession"""
        if session_id not in self.session_managers:
            try:
                # Fallback: Create minimal session manager
                self.session_managers[session_id] = {
                    "history": [],
                    "session_id": session_id,
                    "fallback_mode": True,
                }

                if self.variable_manager:
                    self.variable_manager.register_scope(
                        f"session_{session_id}",
                        {
                            "chat_session_active": True,
                            "history_length": 0,
                            "last_interaction": None,
                            "session_id": session_id,
                        },
                    )

                return self.session_managers[session_id]

            except Exception as e:
                self.session_managers[session_id] = {
                    "history": [],
                    "session_id": session_id,
                    "fallback_mode": True,
                }
                return self.session_managers[session_id]

        return self.session_managers[session_id]

    async def add_interaction(
        self, session_id: str, role: str, content: str, metadata: dict = None
    ) -> None:
        """Einheitlicher Weg um Interaktionen zu speichern"""
        session = await self.initialize_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "metadata": metadata or {},
        }

        if hasattr(session, "add_message"):
            await session.add_message(message)
        elif isinstance(session, dict) and "history" in session:
            session["history"].append(message)
            max_len = 200
            if len(session["history"]) > max_len:
                session["history"] = session["history"][-max_len:]

        if self.variable_manager:
            self.variable_manager.set(f"session_{session_id}.last_interaction", message)
            if hasattr(session, "history"):
                self.variable_manager.set(
                    f"session_{session_id}.history_length", len(session.history)
                )
            elif isinstance(session, dict):
                self.variable_manager.set(
                    f"session_{session_id}.history_length",
                    len(session.get("history", [])),
                )

        self._invalidate_cache(session_id)

    async def get_contextual_history(
        self, session_id: str, query: str = "", max_entries: int = 10
    ) -> list:
        """Intelligente Auswahl relevanter Geschichte"""
        session = self.session_managers.get(session_id)
        if not session:
            return []

        try:
            if hasattr(session, 'get_past_x'):
                recent_history = session.get_past_x(max_entries, last_u=False)
                c = await session.get_reference(query)
                return recent_history[:max_entries] + ([] if not c else [{'role': 'system', 'content': c,
                                                                          'timestamp': datetime.now().isoformat(),
                                                                          'metadata': {
                                                                              'source': 'contextual_history'}}])

            elif isinstance(session, dict) and 'history' in session:
                history = session['history']
                result = []
                for msg in reversed(history[-max_entries:]):
                    result.append(msg)
                    if msg.get('role') == 'user' and len(result) >= max_entries:
                        break
                return list(reversed(result))[:max_entries]

        except Exception as e:
            pass

        return []

    async def build_unified_context(self, session_id: str, query: str = None, context_type: str = "full") -> dict:
        """Optimierte Context-Building Methode"""
        cache_key = f"{session_id}_{context_type}"

        cached = self._get_cached_context(cache_key)
        if cached:
            self._cache_stats["hits"] = self._cache_stats.get("hits", 0) + 1
            return cached

        self._cache_stats["misses"] = self._cache_stats.get("misses", 0) + 1

        context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "context_type": context_type,
        }

        current_time = time.time()

        try:
            if self._is_component_cache_valid("history", session_id, self.HISTORY_TTL):
                context["chat_history"] = self._history_cache[session_id][1]
            else:
                context["chat_history"] = await self.get_contextual_history(
                    session_id, query or "", max_entries=10
                )
                self._history_cache[session_id] = (current_time, context["chat_history"])

            if self._is_component_cache_valid("variables", session_id, self.VARIABLES_TTL):
                context["variables"] = self._variables_cache[session_id][1]
            else:
                context["variables"] = self._build_minimal_variables_snapshot()
                self._variables_cache[session_id] = (current_time, context["variables"])

            if query and self.variable_manager:
                world_model = self.variable_manager.get("world", {})
                if world_model:
                    context["relevant_facts"] = self._extract_relevant_facts(world_model, query)[:3]

            context["execution_state"] = {
                "active_tasks": len(self._get_active_tasks()),
                "recent_completions": len(self._get_recent_completions(2)),
                "recent_results": self._get_recent_results(2),
                "system_status": self.agent.shared.get("system_status", "idle"),
            }

            context["session_stats"] = {
                "history_length": len(context.get("chat_history", [])),
                "cache_hit_rate": self._get_cache_hit_rate(),
            }

        except Exception as e:
            context["error"] = str(e)
            context["fallback_mode"] = True

        self._cache_context(cache_key, context)
        return context

    def _is_component_cache_valid(self, component: str, session_id: str, ttl: float) -> bool:
        """Prüft ob ein Komponenten-Cache noch gültig ist"""
        cache_map = {
            'history': self._history_cache,
            'variables': self._variables_cache,
            'execution': self._execution_cache
        }

        cache = cache_map.get(component, {})
        if session_id not in cache:
            return False

        timestamp, _ = cache[session_id]
        return time.time() - timestamp < ttl

    def _build_minimal_variables_snapshot(self) -> dict:
        """Minimaler Variable-Snapshot"""
        if not self.variable_manager:
            return {'status': 'unavailable'}

        snapshot = {}
        priority_scopes = ['results', 'delegation', 'files', 'user']

        for scope_name in priority_scopes:
            scope = self.variable_manager.scopes.get(scope_name, {})
            if isinstance(scope, dict) and scope:
                snapshot[scope_name] = {
                    'count': len(scope),
                    'keys': list(scope.keys())[:3]
                }

        return snapshot

    def _get_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate"""
        hits = self._cache_stats.get('hits', 0)
        misses = self._cache_stats.get('misses', 0)
        total = hits + misses
        return round(hits / total, 2) if total > 0 else 0.0

    def get_formatted_context_for_llm(self, unified_context: dict) -> str:
        """Optimierter formatierter Context für LLM"""
        try:
            parts = []

            session_id = unified_context.get('session_id', '?')[:20]
            parts.append(f"## Context [{session_id}]")

            chat_history = unified_context.get('chat_history', [])
            if chat_history:
                parts.append("\n### Recent")
                for msg in chat_history[-3:]:
                    role = msg.get('role', '?')[0].upper()
                    content = msg.get('content', '')
                    preview = content[:150] + "..." if len(content) > 150 else content
                    parts.append(f"{role}: {preview}")

            variables = unified_context.get('variables', {})
            if variables and variables.get('status') != 'unavailable':
                var_info = []
                for scope, data in variables.items():
                    if isinstance(data, dict) and 'count' in data:
                        var_info.append(f"{scope}({data['count']})")
                if var_info:
                    parts.append(f"\n### Vars: {', '.join(var_info)}")

            execution = unified_context.get('execution_state', {})
            active = execution.get('active_tasks', 0)
            if active > 0:
                parts.append(f"\n### Active: {active} tasks")

            facts = unified_context.get('relevant_facts', [])
            if facts:
                parts.append("\n### Facts")
                for fact in facts[:2]:
                    if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                        key, value = fact[0], str(fact[1])[:80]
                        parts.append(f"- {key}: {value}")

            return "\n".join(parts)

        except Exception as e:
            return f"Context error: {str(e)}"

    def _extract_relevant_facts(self, world_model: dict, query: str) -> list:
        """Extrahiere relevante Facts basierend auf Query"""
        try:
            query_words = set(query.lower().split())
            relevant_facts = []

            for key, value in world_model.items():
                key_words = set(key.lower().split())
                value_words = set(str(value).lower().split())

                key_overlap = len(query_words.intersection(key_words))
                value_overlap = len(query_words.intersection(value_words))

                if key_overlap > 0 or value_overlap > 0:
                    relevance_score = key_overlap * 2 + value_overlap
                    relevant_facts.append((relevance_score, key, value))

            relevant_facts.sort(key=lambda x: x[0], reverse=True)
            return [(key, value) for _, key, value in relevant_facts[:5]]
        except:
            return list(world_model.items())[:5]

    def _get_active_tasks(self) -> list:
        """Hole aktive Tasks"""
        try:
            tasks = self.agent.shared.get("tasks", {})
            return [
                {"id": task_id, "description": task.description, "status": task.status}
                for task_id, task in tasks.items()
                if task.status == "running"
            ]
        except:
            return []

    def _get_recent_results(self, limit: int = 3) -> list:
        try:
            results_store = self.agent.shared.get("results", {})
            if not results_store:
                return []

            recent_results = []
            for task_id, result_data in list(results_store.items())[-limit:]:
                if result_data and result_data.get("data"):
                    data = result_data["data"]
                    if isinstance(data, str):
                        preview = data[:200] + "..." if len(data) > 200 else data
                    elif isinstance(data, dict):
                        preview = f"Dict({len(data)} keys)"
                    else:
                        preview = str(data)[:80]

                    recent_results.append({
                        "task_id": task_id[:30],
                        "preview": preview,
                        "success": result_data.get("metadata", {}).get("success", False)
                    })

            return recent_results
        except:
            return []

    def get_minimal_context_for_reasoning(self, session_id: str) -> str:
        """Ultra-minimaler Context für Reasoning Loops"""
        try:
            parts = []

            session = self.session_managers.get(session_id)
            if session:
                if hasattr(session, 'history'):
                    history_len = len(session.history)
                elif isinstance(session, dict):
                    history_len = len(session.get('history', []))
                else:
                    history_len = 0
                parts.append(f"History: {history_len} msgs")

            if self.variable_manager:
                non_empty = []
                for name in ['results', 'delegation', 'files']:
                    scope = self.variable_manager.scopes.get(name, {})
                    if isinstance(scope, dict) and scope:
                        non_empty.append(f"{name}({len(scope)})")
                if non_empty:
                    parts.append(f"Data: {', '.join(non_empty)}")

            active_count = len(self._get_active_tasks())
            if active_count > 0:
                parts.append(f"Active: {active_count}")

            return " | ".join(parts) if parts else "No context"

        except:
            return "Context unavailable"

    def _get_recent_completions(self, limit: int = 3) -> list:
        """Hole recent completions"""
        try:
            tasks = self.agent.shared.get("tasks", {})
            completed = [
                {"id": task_id, "description": task.description, "completed_at": task.completed_at}
                for task_id, task in tasks.items()
                if task.status == "completed" and hasattr(task, 'completed_at') and task.completed_at
            ]
            completed.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
            return completed[:limit]
        except:
            return []

    def _get_cached_context(self, cache_key: str):
        """Hole Context aus Cache wenn noch gültig"""
        if cache_key in self._context_cache:
            timestamp, data = self._context_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                del self._context_cache[cache_key]
        return None

    def _cache_context(self, cache_key: str, context: dict):
        """Speichere Context in Cache"""
        self._context_cache[cache_key] = (time.time(), context.copy())

        if len(self._context_cache) > 50:
            oldest_key = min(self._context_cache.keys(),
                             key=lambda k: self._context_cache[k][0])
            del self._context_cache[oldest_key]

    def _invalidate_cache(self, session_id: str = None):
        """Gezieltes Invalidieren"""
        if session_id:
            for cache in [self._context_cache, self._history_cache,
                          self._variables_cache, self._execution_cache]:
                keys_to_remove = [k for k in cache if session_id in str(k)]
                for key in keys_to_remove:
                    del cache[key]
        else:
            self._context_cache.clear()
            self._history_cache.clear()
            self._variables_cache.clear()
            self._execution_cache.clear()

    def get_session_statistics(self) -> dict:
        """Hole Statistiken über alle Sessions"""
        stats = {
            "total_sessions": len(self.session_managers),
            "active_sessions": [],
            "cache_entries": len(self._context_cache),
            "cache_hit_rate": 0.0
        }

        for session_id, session in self.session_managers.items():
            session_info = {
                "session_id": session_id,
                "fallback_mode": isinstance(session, dict) and session.get('fallback_mode', False)
            }

            if hasattr(session, 'history'):
                session_info["message_count"] = len(session.history)
            elif isinstance(session, dict) and 'history' in session:
                session_info["message_count"] = len(session['history'])

            stats["active_sessions"].append(session_info)

        return stats

    async def cleanup_old_sessions(self, max_age_hours: int = 168) -> int:
        """Cleanup alte Sessions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0

            sessions_to_remove = []
            for session_id, session in self.session_managers.items():
                should_remove = False

                if hasattr(session, 'history') and session.history:
                    last_msg = session.history[-1]
                    last_timestamp = last_msg.get('timestamp')
                    if last_timestamp:
                        try:
                            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            if last_time < cutoff_time:
                                should_remove = True
                        except:
                            pass
                elif isinstance(session, dict) and session.get('history'):
                    last_msg = session['history'][-1]
                    last_timestamp = last_msg.get('timestamp')
                    if last_timestamp:
                        try:
                            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            if last_time < cutoff_time:
                                should_remove = True
                        except:
                            pass

                if should_remove:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                session = self.session_managers[session_id]
                if hasattr(session, 'on_exit'):
                    session.on_exit()
                del self.session_managers[session_id]
                removed_count += 1

                if self.variable_manager:
                    scope_name = f'session_{session_id}'
                    if scope_name in self.variable_manager.scopes:
                        del self.variable_manager.scopes[scope_name]

            self._invalidate_cache()

            return removed_count
        except Exception as e:
            return 0


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestUnifiedContextManagerUnit(unittest.TestCase):
    """Unit Tests für UnifiedContextManager - isolierte Komponenten-Tests"""

    def setUp(self):
        """Setup für jeden Test"""
        self.agent = MockAgent()
        self.manager = UnifiedContextManager(self.agent)

    def tearDown(self):
        """Cleanup nach jedem Test"""
        self.manager._context_cache.clear()
        self.manager._history_cache.clear()
        self.manager._variables_cache.clear()
        self.manager._execution_cache.clear()

    # -------------------------------------------------------------------------
    # Cache Tests
    # -------------------------------------------------------------------------

    def test_cache_context_stores_data(self):
        """Test: Context wird korrekt gecacht"""
        cache_key = "test_session_full"
        context = {"session_id": "test", "data": "test_data"}

        self.manager._cache_context(cache_key, context)

        self.assertIn(cache_key, self.manager._context_cache)
        cached_timestamp, cached_data = self.manager._context_cache[cache_key]
        self.assertEqual(cached_data["data"], "test_data")

    def test_get_cached_context_returns_valid_cache(self):
        """Test: Gültiger Cache wird zurückgegeben"""
        cache_key = "test_session_full"
        context = {"session_id": "test", "value": 42}

        self.manager._cache_context(cache_key, context)
        result = self.manager._get_cached_context(cache_key)

        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 42)

    def test_get_cached_context_returns_none_for_expired(self):
        """Test: Abgelaufener Cache wird nicht zurückgegeben"""
        cache_key = "test_session_full"
        context = {"session_id": "test"}

        old_timestamp = time.time() - self.manager.cache_ttl - 10
        self.manager._context_cache[cache_key] = (old_timestamp, context)

        result = self.manager._get_cached_context(cache_key)

        self.assertIsNone(result)
        self.assertNotIn(cache_key, self.manager._context_cache)

    def test_cache_cleanup_on_overflow(self):
        """Test: Älteste Cache-Einträge werden bei Overflow entfernt"""
        for i in range(55):
            self.manager._cache_context(f"key_{i}", {"data": i})

        self.assertLessEqual(len(self.manager._context_cache), 50)

    def test_invalidate_cache_specific_session(self):
        """Test: Gezieltes Invalidieren einer Session"""
        self.manager._context_cache["session1_full"] = (time.time(), {})
        self.manager._context_cache["session2_full"] = (time.time(), {})
        self.manager._history_cache["session1"] = (time.time(), [])

        self.manager._invalidate_cache("session1")

        self.assertNotIn("session1_full", self.manager._context_cache)
        self.assertIn("session2_full", self.manager._context_cache)

    def test_invalidate_cache_all(self):
        """Test: Komplett-Invalidierung aller Caches"""
        self.manager._context_cache["key1"] = (time.time(), {})
        self.manager._history_cache["key2"] = (time.time(), [])

        self.manager._invalidate_cache()

        self.assertEqual(len(self.manager._context_cache), 0)
        self.assertEqual(len(self.manager._history_cache), 0)

    def test_is_component_cache_valid_true(self):
        """Test: Gültiger Komponenten-Cache wird erkannt"""
        self.manager._history_cache["session1"] = (time.time(), [])

        result = self.manager._is_component_cache_valid("history", "session1", 30)

        self.assertTrue(result)

    def test_is_component_cache_valid_false_expired(self):
        """Test: Abgelaufener Komponenten-Cache wird erkannt"""
        old_time = time.time() - 100
        self.manager._history_cache["session1"] = (old_time, [])

        result = self.manager._is_component_cache_valid("history", "session1", 30)

        self.assertFalse(result)

    def test_is_component_cache_valid_false_missing(self):
        """Test: Fehlender Cache wird als ungültig erkannt"""
        result = self.manager._is_component_cache_valid("history", "nonexistent", 30)

        self.assertFalse(result)

    # -------------------------------------------------------------------------
    # Variable Snapshot Tests
    # -------------------------------------------------------------------------

    def test_build_minimal_variables_snapshot_unavailable(self):
        """Test: Snapshot ohne VariableManager"""
        self.manager.variable_manager = None

        result = self.manager._build_minimal_variables_snapshot()

        self.assertEqual(result, {'status': 'unavailable'})

    def test_build_minimal_variables_snapshot_with_data(self):
        """Test: Snapshot mit VariableManager und Daten"""
        vm = MockVariableManager()
        vm.scopes = {
            'results': {'task1': 'done', 'task2': 'done', 'task3': 'pending', 'task4': 'x'},
            'files': {'file1': '/path/to/file'},
            'empty_scope': {}
        }
        self.manager.variable_manager = vm

        result = self.manager._build_minimal_variables_snapshot()

        self.assertIn('results', result)
        self.assertEqual(result['results']['count'], 4)
        self.assertEqual(len(result['results']['keys']), 3)
        self.assertIn('files', result)

    # -------------------------------------------------------------------------
    # Facts Extraction Tests
    # -------------------------------------------------------------------------

    def test_extract_relevant_facts_with_matches(self):
        """Test: Relevante Facts werden extrahiert"""
        world_model = {
            "user_name": "Alice",
            "current_project": "ToolBox",
            "weather": "sunny",
            "toolbox_version": "2.0"
        }
        query = "What is the ToolBox version?"

        result = self.manager._extract_relevant_facts(world_model, query)
        print(result)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        keys = [v for k, v in result]
        self.assertTrue(any("toolbox" in k.lower() for k in keys))

    def test_extract_relevant_facts_no_matches(self):
        """Test: Fallback wenn keine Matches"""
        world_model = {"key1": "value1", "key2": "value2"}
        query = "completely unrelated query xyz123"

        result = self.manager._extract_relevant_facts(world_model, query)

        self.assertIsInstance(result, list)

    def test_extract_relevant_facts_empty_model(self):
        """Test: Leeres World Model"""
        result = self.manager._extract_relevant_facts({}, "any query")

        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # Task/Results Helpers Tests
    # -------------------------------------------------------------------------

    def test_get_active_tasks_empty(self):
        """Test: Keine aktiven Tasks"""
        result = self.manager._get_active_tasks()

        self.assertEqual(result, [])

    def test_get_active_tasks_with_running(self):
        """Test: Aktive Tasks werden gefunden"""
        mock_task = Mock()
        mock_task.description = "Test Task"
        mock_task.status = "running"

        self.agent.shared["tasks"] = {"task1": mock_task}

        result = self.manager._get_active_tasks()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "task1")
        self.assertEqual(result[0]["status"], "running")

    def test_get_recent_results_empty(self):
        """Test: Keine Results"""
        result = self.manager._get_recent_results(3)

        self.assertEqual(result, [])

    def test_get_recent_results_with_data(self):
        """Test: Results werden korrekt formatiert"""
        self.agent.shared["results"] = {
            "task1": {
                "data": "Short result",
                "metadata": {"success": True}
            },
            "task2": {
                "data": "A" * 300,
                "metadata": {"success": False}
            }
        }

        result = self.manager._get_recent_results(2)

        self.assertEqual(len(result), 2)
        self.assertTrue(result[1]["preview"].endswith("...") or len(result[1]["preview"]) <= 200)

    def test_get_recent_completions_empty(self):
        """Test: Keine Completions"""
        result = self.manager._get_recent_completions(3)

        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # Cache Statistics Tests
    # -------------------------------------------------------------------------

    def test_get_cache_hit_rate_zero(self):
        """Test: Hit Rate bei 0 Requests"""
        self.manager._cache_stats = {"hits": 0, "misses": 0}

        result = self.manager._get_cache_hit_rate()

        self.assertEqual(result, 0.0)

    def test_get_cache_hit_rate_calculation(self):
        """Test: Hit Rate Berechnung"""
        self.manager._cache_stats = {"hits": 7, "misses": 3}

        result = self.manager._get_cache_hit_rate()

        self.assertEqual(result, 0.7)

    # -------------------------------------------------------------------------
    # Formatted Context Tests
    # -------------------------------------------------------------------------

    def test_get_formatted_context_minimal(self):
        """Test: Minimaler Context wird formatiert"""
        unified_context = {
            "session_id": "test_session_12345",
            "chat_history": [],
            "variables": {"status": "unavailable"},
            "execution_state": {"active_tasks": 0}
        }

        result = self.manager.get_formatted_context_for_llm(unified_context)

        self.assertIn("Context", result)
        self.assertIn("test_session", result)

    def test_get_formatted_context_with_history(self):
        """Test: Context mit Chat History"""
        unified_context = {
            "session_id": "test",
            "chat_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "variables": {},
            "execution_state": {"active_tasks": 0}
        }

        result = self.manager.get_formatted_context_for_llm(unified_context)

        self.assertIn("Recent", result)
        self.assertIn("U:", result)
        self.assertIn("A:", result)

    def test_get_formatted_context_truncates_long_content(self):
        """Test: Langer Content wird gekürzt"""
        long_content = "X" * 200
        unified_context = {
            "session_id": "test",
            "chat_history": [{"role": "user", "content": long_content}],
            "variables": {},
            "execution_state": {"active_tasks": 0}
        }

        result = self.manager.get_formatted_context_for_llm(unified_context)

        self.assertIn("...", result)

    def test_get_formatted_context_with_active_tasks(self):
        """Test: Active Tasks werden angezeigt"""
        unified_context = {
            "session_id": "test",
            "chat_history": [],
            "variables": {},
            "execution_state": {"active_tasks": 5}
        }

        result = self.manager.get_formatted_context_for_llm(unified_context)

        self.assertIn("Active", result)
        self.assertIn("5", result)

    def test_get_formatted_context_error_handling(self):
        """Test: Error Handling bei ungültigem Context"""
        result = self.manager.get_formatted_context_for_llm(None)

        self.assertIn("error", result.lower())

    # -------------------------------------------------------------------------
    # Minimal Context for Reasoning Tests
    # -------------------------------------------------------------------------

    def test_get_minimal_context_no_session(self):
        """Test: Minimal Context ohne Session"""
        result = self.manager.get_minimal_context_for_reasoning("nonexistent")

        self.assertIn("No context", result)

    def test_get_minimal_context_with_session(self):
        """Test: Minimal Context mit Session"""
        mock_session = MockChatSession()
        mock_session.history = [{"role": "user", "content": "test"}] * 5
        self.manager.session_managers["test_session"] = mock_session

        result = self.manager.get_minimal_context_for_reasoning("test_session")

        self.assertIn("History", result)
        self.assertIn("5", result)

    def test_get_minimal_context_with_variables(self):
        """Test: Minimal Context mit VariableManager"""
        vm = MockVariableManager()
        vm.scopes = {"results": {"a": 1, "b": 2}}
        self.manager.variable_manager = vm

        mock_session = MockChatSession()
        self.manager.session_managers["test"] = mock_session

        result = self.manager.get_minimal_context_for_reasoning("test")

        self.assertIn("Data", result)
        self.assertIn("results", result)

    # -------------------------------------------------------------------------
    # Session Statistics Tests
    # -------------------------------------------------------------------------

    def test_get_session_statistics_empty(self):
        """Test: Statistiken ohne Sessions"""
        result = self.manager.get_session_statistics()

        self.assertEqual(result["total_sessions"], 0)
        self.assertEqual(result["active_sessions"], [])

    def test_get_session_statistics_with_sessions(self):
        """Test: Statistiken mit Sessions"""
        mock_session = MockChatSession()
        mock_session.history = [{"msg": 1}, {"msg": 2}]
        self.manager.session_managers["session1"] = mock_session
        self.manager.session_managers["session2"] = {"history": [{"msg": 1}], "fallback_mode": True}

        result = self.manager.get_session_statistics()

        self.assertEqual(result["total_sessions"], 2)
        self.assertEqual(len(result["active_sessions"]), 2)

        fallback_session = next(s for s in result["active_sessions"] if s["session_id"] == "session2")
        self.assertTrue(fallback_session["fallback_mode"])


# ============================================================================
# ASYNC UNIT TESTS
# ============================================================================

class TestUnifiedContextManagerAsyncUnit(unittest.IsolatedAsyncioTestCase):
    """Async Unit Tests für UnifiedContextManager"""

    async def asyncSetUp(self):
        """Async Setup"""
        self.agent = MockAgent()
        self.manager = UnifiedContextManager(self.agent)

    async def asyncTearDown(self):
        """Async Cleanup"""
        self.manager._invalidate_cache()

    async def test_initialize_session_creates_fallback(self):
        """Test: Session wird im Fallback-Modus erstellt"""
        session = await self.manager.initialize_session("test_session")

        self.assertIn("test_session", self.manager.session_managers)
        self.assertTrue(session.get("fallback_mode"))

    async def test_initialize_session_reuses_existing(self):
        """Test: Existierende Session wird wiederverwendet"""
        mock_session = MockChatSession()
        self.manager.session_managers["existing"] = mock_session

        session = await self.manager.initialize_session("existing")

        self.assertIs(session, mock_session)

    async def test_add_interaction_to_fallback_session(self):
        """Test: Interaktion wird zu Fallback-Session hinzugefügt"""
        self.manager.session_managers["test"] = {"history": [], "fallback_mode": True}

        await self.manager.add_interaction("test", "user", "Hello!")

        session = self.manager.session_managers["test"]
        self.assertEqual(len(session["history"]), 1)
        self.assertEqual(session["history"][0]["role"], "user")
        self.assertEqual(session["history"][0]["content"], "Hello!")

    async def test_add_interaction_respects_max_length(self):
        """Test: History wird auf max_length begrenzt"""
        self.manager.session_managers["test"] = {"history": [], "fallback_mode": True}

        for i in range(250):
            await self.manager.add_interaction("test", "user", f"Message {i}")

        session = self.manager.session_managers["test"]
        self.assertLessEqual(len(session["history"]), 200)

    async def test_add_interaction_invalidates_cache(self):
        """Test: Cache wird nach Interaktion invalidiert"""
        self.manager.session_managers["test"] = {"history": [], "fallback_mode": True}
        self.manager._context_cache["test_full"] = (time.time(), {"old": "data"})

        await self.manager.add_interaction("test", "user", "New message")

        self.assertNotIn("test_full", self.manager._context_cache)

    async def test_add_interaction_updates_variable_manager(self):
        """Test: VariableManager wird aktualisiert"""
        vm = MockVariableManager()
        self.manager.variable_manager = vm
        self.manager.session_managers["test"] = {"history": [], "fallback_mode": True}

        await self.manager.add_interaction("test", "user", "Test message")

        self.assertIn("session_test.last_interaction", vm._data)
        self.assertIn("session_test.history_length", vm._data)

    async def test_get_contextual_history_empty_session(self):
        """Test: Leere History bei nicht-existierender Session"""
        result = await self.manager.get_contextual_history("nonexistent")

        self.assertEqual(result, [])

    async def test_get_contextual_history_fallback_mode(self):
        """Test: History aus Fallback-Session"""
        history = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        self.manager.session_managers["test"] = {"history": history, "fallback_mode": True}

        result = await self.manager.get_contextual_history("test", max_entries=5)

        self.assertGreater(len(result), 0)

    async def test_build_unified_context_uses_cache(self):
        """Test: Cached Context wird verwendet"""
        cached_context = {"session_id": "test", "cached": True, "timestamp": "x"}
        self.manager._context_cache["test_full"] = (time.time(), cached_context)

        result = await self.manager.build_unified_context("test", context_type="full")

        self.assertTrue(result.get("cached"))
        self.assertEqual(self.manager._cache_stats["hits"], 1)

    async def test_build_unified_context_creates_new(self):
        """Test: Neuer Context wird erstellt wenn kein Cache"""
        self.manager.session_managers["test"] = {"history": [], "fallback_mode": True}

        result = await self.manager.build_unified_context("test", "test query", "full")

        self.assertIn("timestamp", result)
        self.assertIn("session_id", result)
        self.assertIn("chat_history", result)
        self.assertIn("execution_state", result)

    async def test_build_unified_context_handles_errors(self):
        """Test: Fehler werden abgefangen"""
        self.manager.session_managers["test"] = "Exception"
        result = await self.manager.build_unified_context("test")

        self.assertIn("test", result.get("session_id"))

    async def test_cleanup_old_sessions_removes_old(self):
        """Test: Alte Sessions werden entfernt"""
        old_timestamp = (datetime.now() - timedelta(hours=200)).isoformat()
        self.manager.session_managers["old_session"] = {
            "history": [{"timestamp": old_timestamp}],
            "fallback_mode": True
        }

        new_timestamp = datetime.now().isoformat()
        self.manager.session_managers["new_session"] = {
            "history": [{"timestamp": new_timestamp}],
            "fallback_mode": True
        }

        removed = await self.manager.cleanup_old_sessions(max_age_hours=168)

        self.assertEqual(removed, 1)
        self.assertNotIn("old_session", self.manager.session_managers)
        self.assertIn("new_session", self.manager.session_managers)

    async def test_cleanup_old_sessions_calls_on_exit(self):
        """Test: on_exit wird für ChatSessions aufgerufen"""
        mock_session = MockChatSession()
        mock_session.on_exit = Mock()
        old_timestamp = (datetime.now() - timedelta(hours=200)).isoformat()
        mock_session.history = [{"timestamp": old_timestamp}]

        self.manager.session_managers["test"] = mock_session

        await self.manager.cleanup_old_sessions(max_age_hours=168)

        mock_session.on_exit.assert_called_once()


# ============================================================================
# E2E TESTS (Standalone - keine toolboxv2 Imports)
# ============================================================================

class TestUnifiedContextManagerE2E(unittest.IsolatedAsyncioTestCase):
    """
    End-to-End Tests für UnifiedContextManager.
    Testet vollständige Workflows mit Mocks.
    """

    async def asyncSetUp(self):
        """Setup für jeden E2E Test"""
        self.amd = MockAgentModelData(name="E2ETestAgent")
        self.agent = MockAgent(self.amd)
        self.manager = UnifiedContextManager(self.agent)
        self.manager.variable_manager = MockVariableManager()

    async def asyncTearDown(self):
        """Cleanup nach jedem E2E Test"""
        await self.manager.cleanup_old_sessions(max_age_hours=0)
        self.manager._invalidate_cache()

    # -------------------------------------------------------------------------
    # Full Workflow Tests
    # -------------------------------------------------------------------------

    async def test_e2e_complete_conversation_workflow(self):
        """E2E: Kompletter Konversations-Workflow"""
        session_id = "e2e_conversation_test"

        # 1. Initialize Session
        session = await self.manager.initialize_session(session_id)
        self.assertIsNotNone(session)

        # 2. Add multiple interactions
        conversations = [
            ("user", "Hello, I need help with Python"),
            ("assistant", "Sure! What would you like to know about Python?"),
            ("user", "How do I use list comprehensions?"),
            ("assistant", "List comprehensions are a concise way to create lists..."),
            ("user", "Can you show me an example?"),
        ]

        for role, content in conversations:
            await self.manager.add_interaction(session_id, role, content)

        # 3. Build unified context
        context = await self.manager.build_unified_context(
            session_id,
            query="list comprehension example",
            context_type="full"
        )

        # 4. Verify context structure
        self.assertEqual(context["session_id"], session_id)
        self.assertIn("chat_history", context)
        self.assertIn("execution_state", context)
        self.assertGreater(len(context["chat_history"]), 0)

        # 5. Get formatted context
        formatted = self.manager.get_formatted_context_for_llm(context)
        self.assertIsInstance(formatted, str)
        self.assertIn("Recent", formatted)

        # 6. Verify statistics
        stats = self.manager.get_session_statistics()
        self.assertEqual(stats["total_sessions"], 1)

    async def test_e2e_multi_session_management(self):
        """E2E: Verwaltung mehrerer gleichzeitiger Sessions"""
        sessions = ["session_a", "session_b", "session_c"]

        for session_id in sessions:
            await self.manager.initialize_session(session_id)
            await self.manager.add_interaction(session_id, "user", f"Hello from {session_id}")

        self.assertEqual(len(self.manager.session_managers), 3)

        contexts = {}
        for session_id in sessions:
            contexts[session_id] = await self.manager.build_unified_context(session_id)

        for session_id, context in contexts.items():
            self.assertEqual(context["session_id"], session_id)

        stats = self.manager.get_session_statistics()
        self.assertEqual(stats["total_sessions"], 3)

    async def test_e2e_cache_performance(self):
        """E2E: Cache-Performance bei wiederholten Anfragen"""
        session_id = "cache_perf_test"

        await self.manager.initialize_session(session_id)
        await self.manager.add_interaction(session_id, "user", "Test message")

        self.manager._cache_stats = {"hits": 0, "misses": 0}

        await self.manager.build_unified_context(session_id, context_type="full")

        for _ in range(5):
            await self.manager.build_unified_context(session_id, context_type="full")

        hit_rate = self.manager._get_cache_hit_rate()
        self.assertGreater(hit_rate, 0.8)

    async def test_e2e_context_with_variable_manager(self):
        """E2E: Context-Building mit VariableManager Integration"""
        session_id = "var_manager_test"

        self.manager.variable_manager._data["world"] = {
            "user_preference": "detailed explanations",
            "current_topic": "Python programming",
            "expertise_level": "intermediate"
        }

        await self.manager.initialize_session(session_id)
        await self.manager.add_interaction(
            session_id, "user",
            "Tell me more about Python programming"
        )

        context = await self.manager.build_unified_context(
            session_id,
            query="Python programming",
            context_type="full"
        )
        print(context)
        if "relevant_facts" in context:
            self.assertIsInstance(context["relevant_facts"], list)
            self.assertGreater(len(context["relevant_facts"]), 0)

    async def test_e2e_session_cleanup_workflow(self):
        """E2E: Session Cleanup nach Inaktivität"""
        old_session = "old_session"
        new_session = "new_session"

        await self.manager.initialize_session(old_session)
        await self.manager.initialize_session(new_session)

        old_time = (datetime.now() - timedelta(hours=200)).isoformat()
        self.manager.session_managers[old_session]["history"] = [{"timestamp": old_time}]

        await self.manager.add_interaction(new_session, "user", "Recent message")

        removed_count = await self.manager.cleanup_old_sessions(max_age_hours=168)

        self.assertEqual(removed_count, 1)
        self.assertNotIn(old_session, self.manager.session_managers)
        self.assertIn(new_session, self.manager.session_managers)

    async def test_e2e_error_recovery(self):
        """E2E: Wiederherstellung nach Fehlern"""
        session_id = "error_recovery_test"

        await self.manager.initialize_session(session_id)
        await self.manager.add_interaction(session_id, "user", "Normal message")

        original_session = self.manager.session_managers[session_id]
        self.manager.session_managers[session_id] = None

        context = await self.manager.build_unified_context(session_id)
        self.assertEqual(0, len(context.get("chat_history")))

        self.manager.session_managers[session_id] = original_session
        self.manager._invalidate_cache(session_id)
        context = await self.manager.build_unified_context(session_id)
        self.assertNotIn("error", context)

    async def test_e2e_concurrent_access(self):
        """E2E: Gleichzeitiger Zugriff auf Sessions"""
        session_id = "concurrent_test"
        await self.manager.initialize_session(session_id)

        async def add_messages(start_idx: int, count: int):
            for i in range(count):
                await self.manager.add_interaction(
                    session_id,
                    "user" if i % 2 == 0 else "assistant",
                    f"Message {start_idx + i}"
                )

        await asyncio.gather(
            add_messages(0, 10),
            add_messages(100, 10),
            add_messages(200, 10)
        )

        session = self.manager.session_managers[session_id]
        history_len = len(session.get("history", []))

        self.assertGreater(history_len, 0)

    async def test_e2e_minimal_context_for_reasoning_loop(self):
        """E2E: Minimal Context für schnelle Reasoning Loops"""
        session_id = "reasoning_test"

        await self.manager.initialize_session(session_id)

        for i in range(5):
            await self.manager.add_interaction(session_id, "user", f"Step {i}")

            minimal = self.manager.get_minimal_context_for_reasoning(session_id)

            self.assertIsInstance(minimal, str)
            self.assertLess(len(minimal), 500)

    async def test_e2e_formatted_output_quality(self):
        """E2E: Qualität des formatierten Outputs"""
        session_id = "format_test"

        await self.manager.initialize_session(session_id)

        await self.manager.add_interaction(session_id, "user", "Short question")
        await self.manager.add_interaction(session_id, "assistant", "A" * 200)
        await self.manager.add_interaction(session_id, "user", "Follow-up")

        context = await self.manager.build_unified_context(session_id, "test query")
        formatted = self.manager.get_formatted_context_for_llm(context)

        self.assertIn("##", formatted)
        self.assertIn("...", formatted)
        lines = formatted.split("\n")
        self.assertLess(len(lines), 30)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedContextManagerUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedContextManagerAsyncUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedContextManagerE2E))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
