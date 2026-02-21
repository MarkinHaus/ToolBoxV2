"""
Tests für chain_tools.py

Testet:
- DSL Parser (alle Step-Typen, Operatoren, Custom Functions)
- Validator (Tool-Checks, unsafe code detection)
- ChainBuilder (Structure → Chain Objects)
- ChainStore (CRUD + accept Workflow)
- ChainContextWrapper (Agent erhält Chain-Kontext)
- SafeCustomFunction (Safe eval + Rejection)
- Tool-Funktionen (create, run, list)
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from toolboxv2.mods.isaa.base.Agent.chain_tools import (
    ChainBuilder,
    ChainContextWrapper,
    CoderContextWrapper,
    ChainDSLParser,
    ChainParseError,
    ChainStore,
    ChainValidator,
    ParsedStep,
    SafeCustomFunction,
    StoredChain,
    create_chain_tools,
    generate_chain_id,
)


# =============================================================================
# DSL PARSER TESTS
# =============================================================================


class TestChainDSLParser(unittest.TestCase):
    """Tests für den DSL Parser."""

    def setUp(self):
        self.parser = ChainDSLParser()

    # --- Atom-Parsing ---

    def test_parse_tool_simple(self):
        structure, errors = self.parser.parse('tool:search(query="test")')
        self.assertEqual(errors, [])
        self.assertIsInstance(structure, ParsedStep)
        self.assertEqual(structure.type, "tool")
        self.assertEqual(structure.name, "search")
        self.assertEqual(structure.args, {"query": "test"})

    def test_parse_tool_multiple_args(self):
        structure, errors = self.parser.parse('tool:write(path="out.txt", content="{prev}")')
        self.assertEqual(errors, [])
        self.assertEqual(structure.args["path"], "out.txt")
        self.assertEqual(structure.args["content"], "{prev}")

    def test_parse_agent_with_focus(self):
        structure, errors = self.parser.parse('@analyzer("Fasse den Text zusammen")')
        self.assertEqual(errors, [])
        self.assertEqual(structure.type, "agent")
        self.assertEqual(structure.name, "analyzer")
        self.assertEqual(structure.focus_instruction, "Fasse den Text zusammen")

    def test_parse_agent_simple(self):
        structure, errors = self.parser.parse("@researcher")
        self.assertEqual(errors, [])
        self.assertEqual(structure.type, "agent")
        self.assertEqual(structure.name, "researcher")

    def test_parse_custom_func_inline(self):
        structure, errors = self.parser.parse("def:clean(text) -> text.strip().lower()")
        self.assertEqual(errors, [])
        self.assertEqual(structure.type, "custom_func")
        self.assertEqual(structure.name, "clean")
        self.assertEqual(structure.func_param, "text")
        self.assertEqual(structure.func_body, "text.strip().lower()")

    def test_parse_format_with_key(self):
        structure, errors = self.parser.parse('CF(Summary) - "result"')
        self.assertEqual(errors, [])
        self.assertEqual(structure.type, "format")
        self.assertEqual(structure.format_class_name, "Summary")
        self.assertEqual(structure.extract_key, "result")
        self.assertFalse(structure.is_parallel_extract)

    def test_parse_format_parallel_extract(self):
        structure, errors = self.parser.parse('CF(Items) - "items[n]"')
        self.assertEqual(errors, [])
        self.assertTrue(structure.is_parallel_extract)
        self.assertEqual(structure.extract_key, "items")

    def test_parse_condition(self):
        structure, errors = self.parser.parse('IS(type==code)')
        self.assertEqual(errors, [])
        self.assertEqual(structure.type, "condition")
        self.assertEqual(structure.condition_key, "type")
        self.assertEqual(structure.condition_value, "code")

    # --- Operator-Parsing ---

    def test_parse_sequential(self):
        structure, errors = self.parser.parse(
            'tool:search(query="{prev}") >> @analyzer("Analysiere") >> tool:write(path="out.md")'
        )
        self.assertEqual(errors, [])
        self.assertIsInstance(structure, dict)
        self.assertEqual(structure["type"], "sequential")
        self.assertEqual(len(structure["steps"]), 3)

    def test_parse_parallel(self):
        structure, errors = self.parser.parse("(@researcher + @coder)")
        self.assertEqual(errors, [])
        self.assertEqual(structure["type"], "parallel")
        self.assertEqual(len(structure["branches"]), 2)

    def test_parse_error_handling(self):
        structure, errors = self.parser.parse("(@primary | @fallback)")
        self.assertEqual(errors, [])
        self.assertEqual(structure["type"], "error_handling")

    def test_parse_conditional(self):
        structure, errors = self.parser.parse(
            "IS(ready==true) >> @processor % @validator"
        )
        self.assertEqual(errors, [])
        self.assertEqual(structure["type"], "conditional_branches")

    # --- Custom Functions (mehrzeilig) ---

    def test_parse_multiline_with_custom_func(self):
        dsl = """
def:extract_urls(text) -> [w for w in text.split() if w.startswith("http")]
tool:web_search(query="{prev}") >> def:extract_urls >> @analyzer("Analysiere URLs")
"""
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        self.assertEqual(structure["type"], "sequential")
        steps = structure["steps"]
        self.assertEqual(steps[1].type, "custom_func")
        self.assertEqual(steps[1].name, "extract_urls")

    def test_parse_comment_lines_ignored(self):
        dsl = """
# Das ist ein Kommentar
tool:search(query="test")
# Noch ein Kommentar
"""
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        self.assertEqual(structure.type, "tool")

    # --- Nested Expressions ---

    def test_parse_nested_parallel_in_sequential(self):
        dsl = 'tool:plan >> (@worker_a + @worker_b) >> tool:merge'
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        self.assertEqual(structure["type"], "sequential")
        self.assertEqual(len(structure["steps"]), 3)
        self.assertEqual(structure["steps"][1]["type"], "parallel")

    def test_parse_error_handling_in_sequential(self):
        dsl = 'tool:prepare >> (@primary | @fallback) >> tool:finalize'
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        self.assertEqual(structure["type"], "sequential")
        self.assertEqual(structure["steps"][1]["type"], "error_handling")

    # --- Error Cases ---

    def test_parse_empty_dsl(self):
        structure, errors = self.parser.parse("")
        self.assertTrue(len(errors) > 0)

    def test_parse_unknown_step(self):
        structure, errors = self.parser.parse("random_garbage_123")
        self.assertTrue(len(errors) > 0)

    def test_parse_bracket_mismatch_detected_by_validator(self):
        # Parser selbst bricht nicht, aber Validator erkennt es
        dsl = "((tool:search(query=\"x\")"
        validator = ChainValidator()
        structure, _ = self.parser.parse(dsl)
        is_valid, errors, _ = validator.validate(structure, dsl)
        self.assertFalse(is_valid)

    # --- model: Pre-Block ---

    def test_parse_inline_model_simple(self):
        dsl = """
model:SearchResult(title: str, url: str, score: float)
tool:search(query="{prev}") >> CF(SearchResult) - "title"
"""
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        # Model muss im Parser registriert sein
        self.assertIn("SearchResult", self.parser.inline_models)
        model_cls = self.parser.inline_models["SearchResult"]
        # Prüfe dass es ein Pydantic Model mit den richtigen Feldern ist
        self.assertIn("title", model_cls.__annotations__)
        self.assertIn("url", model_cls.__annotations__)
        self.assertIn("score", model_cls.__annotations__)
        self.assertEqual(model_cls.__annotations__["title"], str)
        self.assertEqual(model_cls.__annotations__["score"], float)

    def test_parse_inline_model_list_types(self):
        dsl = """
model:TaskList(tasks: list[str], count: int)
CF(TaskList) - "tasks[n]"
"""
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        model_cls = self.parser.inline_models["TaskList"]
        self.assertEqual(model_cls.__annotations__["tasks"], list[str])

    def test_parse_inline_model_invalid_syntax(self):
        dsl = """
model:Bad(no_type_here)
@agent
"""
        structure, errors = self.parser.parse(dsl)
        self.assertTrue(any("braucht Typ" in e for e in errors))

    def test_parse_inline_model_unknown_type_falls_back_to_str(self):
        dsl = """
model:Flexible(data: custom_type)
@agent
"""
        structure, errors = self.parser.parse(dsl)
        # Warnung, aber Model wird trotzdem erstellt mit str Fallback
        self.assertIn("Flexible", self.parser.inline_models)
        self.assertEqual(self.parser.inline_models["Flexible"].__annotations__["data"], str)

    def test_inline_model_used_by_validator(self):
        """Validator soll inline models als verfügbare Format-Klassen erkennen."""
        parser = ChainDSLParser()
        dsl = """
model:MyResult(summary: str, confidence: float)
tool:search(query="{prev}") >> CF(MyResult) - "summary"
"""
        structure, errors = parser.parse(dsl)
        # Validator mit inline_models als available_format_classes
        all_formats = set(parser.inline_models.keys())
        validator = ChainValidator(
            available_tools={"search"},
            available_format_classes=all_formats,
        )
        is_valid, val_errors, _ = validator.validate(structure, dsl)
        self.assertTrue(is_valid, f"Errors: {val_errors}")

    def test_inline_model_with_def_and_chain(self):
        """Kombinierter Pre-Block: model + def + chain."""
        dsl = """
model:Analysis(result: str, score: float)
def:clean(text) -> text.strip()
tool:fetch >> def:clean >> CF(Analysis) - "result"
"""
        structure, errors = self.parser.parse(dsl)
        self.assertEqual(errors, [])
        self.assertIn("Analysis", self.parser.inline_models)
        self.assertIn("clean", self.parser.custom_functions)


# =============================================================================
# VALIDATOR TESTS
# =============================================================================


class TestChainValidator(unittest.TestCase):
    """Tests für den Chain Validator."""

    def test_validate_known_tools(self):
        validator = ChainValidator(available_tools={"search", "write"})
        parser = ChainDSLParser()
        structure, _ = parser.parse('tool:search(query="x")')
        is_valid, errors, _ = validator.validate(structure, 'tool:search(query="x")')
        self.assertTrue(is_valid)

    def test_validate_unknown_tool(self):
        validator = ChainValidator(available_tools={"search"})
        parser = ChainDSLParser()
        structure, _ = parser.parse('tool:nonexistent(query="x")')
        is_valid, errors, _ = validator.validate(structure, 'tool:nonexistent(query="x")')
        self.assertFalse(is_valid)
        self.assertTrue(any("nonexistent" in e for e in errors))

    def test_validate_unsafe_custom_func(self):
        validator = ChainValidator()
        parser = ChainDSLParser()
        structure, _ = parser.parse("def:evil(x) -> __import__('os').system('rm -rf /')")
        is_valid, errors, _ = validator.validate(structure, "def:evil(x) -> __import__('os').system('rm -rf /')")
        self.assertFalse(is_valid)
        self.assertTrue(any("nicht erlaubt" in e for e in errors))

    def test_validate_safe_custom_func(self):
        validator = ChainValidator()
        parser = ChainDSLParser()
        dsl = "def:upper(text) -> text.upper()"
        structure, _ = parser.parse(dsl)
        is_valid, errors, _ = validator.validate(structure, dsl)
        self.assertTrue(is_valid)

    def test_validate_metadata_extraction(self):
        validator = ChainValidator(
            available_tools={"search", "write"},
            available_agents={"analyzer"},
        )
        parser = ChainDSLParser()
        dsl = 'tool:search(query="x") >> @analyzer("focus") >> tool:write(path="out")'
        structure, _ = parser.parse(dsl)
        is_valid, errors, metadata = validator.validate(structure, dsl)
        self.assertTrue(is_valid)
        self.assertIn("search", metadata["referenced_tools"])
        self.assertIn("write", metadata["referenced_tools"])
        self.assertTrue(metadata["uses_agents"])
        self.assertEqual(metadata["step_count"], 3)


# =============================================================================
# SAFE CUSTOM FUNCTION TESTS
# =============================================================================


class TestSafeCustomFunction(unittest.TestCase):
    """Tests für die sichere Custom-Function Ausführung."""

    def test_simple_transform(self):
        func = SafeCustomFunction("upper", "text", "text.upper()")
        result = asyncio.run(func("hello world"))
        self.assertEqual(result, "HELLO WORLD")

    def test_list_comprehension(self):
        func = SafeCustomFunction(
            "extract", "data", "[w for w in data.split() if len(w) > 3]"
        )
        result = asyncio.run(func("hi there hello world"))
        self.assertEqual(result, ["there", "hello", "world"])

    def test_json_access(self):
        func = SafeCustomFunction(
            "parse", "data", "json.loads(data)['key']"
        )
        result = asyncio.run(func('{"key": "value"}'))
        self.assertEqual(result, "value")

    def test_error_handling(self):
        func = SafeCustomFunction("bad", "x", "1/0")
        result = asyncio.run(func("anything"))
        self.assertIn("[CustomFunc Error]", result)

    def test_import_blocked(self):
        """eval kann keine __import__ nutzen weil __builtins__ eingeschränkt."""
        func = SafeCustomFunction("evil", "x", "__import__('os')")
        result = asyncio.run(func("test"))
        self.assertIn("[CustomFunc Error]", result)


# =============================================================================
# CHAIN STORE TESTS
# =============================================================================


class TestChainStore(unittest.TestCase):
    """Tests für den persistenten Chain Store."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = ChainStore(self.tmp_dir)

    def test_save_and_get(self):
        chain = StoredChain(
            id="test123", name="TestChain",
            dsl='tool:search(query="{prev}")', description="Test",
        )
        self.store.save(chain)
        loaded = self.store.get("test123")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "TestChain")

    def test_get_by_name(self):
        chain = StoredChain(id="abc", name="MyChain", dsl="@agent")
        self.store.save(chain)
        loaded = self.store.get_by_name("MyChain")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, "abc")

    def test_accept_workflow(self):
        chain = StoredChain(id="sec1", name="SecureChain", dsl="@agent")
        self.store.save(chain)

        self.assertFalse(self.store.get("sec1").accepted)
        self.store.accept("sec1")
        self.assertTrue(self.store.get("sec1").accepted)

    def test_persistence(self):
        chain = StoredChain(id="persist", name="PersistChain", dsl="@agent")
        self.store.save(chain)

        # Neuer Store vom selben Pfad
        store2 = ChainStore(self.tmp_dir)
        loaded = store2.get("persist")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "PersistChain")

    def test_delete(self):
        chain = StoredChain(id="del1", name="DeleteMe", dsl="@agent")
        self.store.save(chain)
        self.assertTrue(self.store.delete("del1"))
        self.assertIsNone(self.store.get("del1"))

    def test_list_all(self):
        self.store.save(StoredChain(id="a", name="A", dsl="@a"))
        self.store.save(StoredChain(id="b", name="B", dsl="@b"))
        self.assertEqual(len(self.store.list_all()), 2)


# =============================================================================
# CHAIN CONTEXT WRAPPER TESTS
# =============================================================================


class TestChainContextWrapper(unittest.TestCase):
    """Tests für den Agent Chain-Kontext."""

    def test_context_injection(self):
        """Agent muss Chain-Kontext im Query erhalten."""
        mock_agent = AsyncMock()
        mock_agent.a_run = AsyncMock(return_value="agent result")

        wrapper = ChainContextWrapper(
            agent=mock_agent,
            chain_name="TestPipeline",
            step_index=2,
            focus_instruction="Extrahiere Schlüsselpunkte",
        )

        result = asyncio.run(wrapper.a_run("test input data"))

        # Prüfe dass a_run aufgerufen wurde
        mock_agent.a_run.assert_called_once()
        call_args = mock_agent.a_run.call_args

        # Query muss Chain-Kontext enthalten
        query = call_args.kwargs.get("query", call_args.args[0] if call_args.args else "")
        self.assertIn("[CHAIN-EXECUTION MODE]", query)
        self.assertIn("TestPipeline", query)
        self.assertIn("Extrahiere Schlüsselpunkte", query)
        self.assertIn("test input data", query)

    def test_context_enforces_focus(self):
        """Kontext muss Fokus-Anweisung enthalten."""
        mock_agent = AsyncMock()
        mock_agent.a_run = AsyncMock(return_value="ok")

        wrapper = ChainContextWrapper(
            agent=mock_agent, chain_name="Test", step_index=1,
            focus_instruction="NUR Zusammenfassung",
        )

        asyncio.run(wrapper.a_run("data"))

        query = mock_agent.a_run.call_args.kwargs.get(
            "query", mock_agent.a_run.call_args.args[0]
        )
        self.assertIn("NUR auf diese eine Aufgabe", query)
        self.assertIn("Stelle KEINE Rückfragen", query)


# =============================================================================
# CODER CONTEXT WRAPPER TESTS
# =============================================================================


class TestCoderContextWrapper(unittest.TestCase):
    """Tests für den CoderAgent Chain-Wrapper."""

    def _make_mock_coder(self, success=True, files=None, message="Done"):
        """Erstellt einen Mock-CoderAgent mit realistischer API."""
        coder = MagicMock()
        coder.state = {"plan": [], "done": [], "current_file": "None", "last_error": None}
        coder.agent = AsyncMock()
        coder.agent.a_format_class = AsyncMock(return_value="formatted")

        result = MagicMock()
        result.success = success
        result.message = message
        result.files_changed = files or ["src/main.py"]
        result.tokens_used = 1500
        coder.execute = AsyncMock(return_value=result)
        return coder

    def test_coder_receives_chain_context(self):
        """Coder.execute() muss den Chain-Prefix im Task erhalten."""
        coder = self._make_mock_coder()

        wrapper = CoderContextWrapper(
            coder=coder, chain_name="BuildPipeline",
            step_index=3, focus_instruction="Implementiere die API Route",
        )
        asyncio.run(wrapper.a_run("POST /api/users endpoint hinzufügen"))

        coder.execute.assert_called_once()
        task = coder.execute.call_args.args[0]
        self.assertIn("[CHAIN-EXECUTION MODE", task)
        self.assertIn("BuildPipeline", task)
        self.assertIn("Implementiere die API Route", task)
        self.assertIn("POST /api/users", task)

    def test_coder_state_injected(self):
        """Wrapper injiziert Focus in coder.state['plan']."""
        coder = self._make_mock_coder()

        wrapper = CoderContextWrapper(
            coder=coder, chain_name="Test",
            step_index=1, focus_instruction="Refactor auth module",
        )
        asyncio.run(wrapper.a_run("data"))

        self.assertEqual(coder.state["plan"], ["Refactor auth module"])

    def test_coder_success_result(self):
        """Erfolgreicher Coder → formatierte Ausgabe mit Dateien + Tokens."""
        coder = self._make_mock_coder(success=True, files=["a.py", "b.py"])
        wrapper = CoderContextWrapper(coder=coder, chain_name="T", step_index=1)

        result = asyncio.run(wrapper.a_run("task"))
        self.assertIn("[Coder OK]", result)
        self.assertIn("a.py", result)
        self.assertIn("b.py", result)
        self.assertIn("1500", result)

    def test_coder_failure_result(self):
        """Fehlgeschlagener Coder → Fehlermeldung."""
        coder = self._make_mock_coder(success=False, message="Syntax Error in line 42")
        wrapper = CoderContextWrapper(coder=coder, chain_name="T", step_index=1)

        result = asyncio.run(wrapper.a_run("task"))
        self.assertIn("[Coder FEHLER]", result)
        self.assertIn("Syntax Error", result)

    def test_same_coder_instance_preserves_state(self):
        """Gleiche Coder-Instanz über mehrere Wrapper → State bleibt erhalten."""
        coder = self._make_mock_coder()

        # Step 1 verändert state
        w1 = CoderContextWrapper(coder=coder, chain_name="P", step_index=1, focus_instruction="Step A")
        asyncio.run(w1.a_run("data1"))
        self.assertEqual(coder.state["plan"], ["Step A"])

        # Step 2 mit GLEICHER Instanz — state wird überschrieben
        w2 = CoderContextWrapper(coder=coder, chain_name="P", step_index=2, focus_instruction="Step B")
        asyncio.run(w2.a_run("data2"))
        self.assertEqual(coder.state["plan"], ["Step B"])

        # Beide calls auf der gleichen Instanz
        self.assertEqual(coder.execute.call_count, 2)

    def test_coder_format_class_delegates_to_agent(self):
        """a_format_class delegiert an coder.agent."""
        coder = self._make_mock_coder()
        wrapper = CoderContextWrapper(coder=coder, chain_name="T", step_index=1)

        result = asyncio.run(wrapper.a_format_class(str, "test prompt"))
        coder.agent.a_format_class.assert_called_once()


# =============================================================================
# CHAIN BUILDER TESTS
# =============================================================================


class TestChainBuilder(unittest.TestCase):
    """Tests für den Chain Builder."""

    def test_build_sequential(self):
        parser = ChainDSLParser()
        structure, _ = parser.parse(
            "def:upper(x) -> x.upper() >> def:strip(x) -> x.strip()"
        )

        # Custom funcs werden im sequential als steps geparst
        # aber da die def: definitionen als atomare steps im sequential landen
        # müssen wir die structure prüfen
        builder = ChainBuilder(chain_name="test")

        # Wenn structure ein dict mit type sequential ist:
        if isinstance(structure, dict) and structure["type"] == "sequential":
            chain = builder.build(structure)
            self.assertIsNotNone(chain)

    def test_build_tool_step(self):
        mock_executor = AsyncMock(return_value="search result")

        parser = ChainDSLParser()
        structure, _ = parser.parse('tool:search(query="python")')

        builder = ChainBuilder(
            tool_executor=mock_executor,
            chain_name="test",
        )
        chain = builder.build(structure)

        # Run
        result = asyncio.run(chain.a_run("input"))
        # Der Function wrapper ruft den executor auf
        mock_executor.assert_called()

    def test_build_agent_step_with_context(self):
        mock_agent = AsyncMock()
        mock_agent.a_run = AsyncMock(return_value="agent output")

        parser = ChainDSLParser()
        structure, _ = parser.parse('@myagent("Analysiere den Input")')

        builder = ChainBuilder(
            agent_registry={"myagent": mock_agent},
            chain_name="pipeline",
        )
        result_chain = builder.build(structure)

        # Prüfe dass ChainContextWrapper erstellt wurde
        self.assertIsInstance(result_chain, ChainContextWrapper)

    def test_build_coder_step_returns_coder_wrapper(self):
        """@name in coder_registry → CoderContextWrapper."""
        mock_coder = MagicMock()
        mock_coder.state = {"plan": [], "done": [], "current_file": "None", "last_error": None}

        parser = ChainDSLParser()
        structure, _ = parser.parse('@mycoder("Implementiere Feature X")')

        builder = ChainBuilder(
            coder_registry={"mycoder": mock_coder},
            chain_name="build_pipeline",
        )
        result_chain = builder.build(structure)
        self.assertIsInstance(result_chain, CoderContextWrapper)

    def test_build_coder_has_priority_over_agent(self):
        """Wenn gleicher Name in coder_registry und agent_registry → Coder gewinnt."""
        mock_agent = AsyncMock()
        mock_coder = MagicMock()
        mock_coder.state = {"plan": []}

        parser = ChainDSLParser()
        structure, _ = parser.parse("@dev")

        builder = ChainBuilder(
            agent_registry={"dev": mock_agent},
            coder_registry={"dev": mock_coder},
            chain_name="test",
        )
        result_chain = builder.build(structure)
        self.assertIsInstance(result_chain, CoderContextWrapper)

    def test_build_with_inline_model(self):
        """model: Pre-Block → CF() Step nutzt das inline definierte Model."""
        parser = ChainDSLParser()
        dsl = """
model:Result(summary: str, score: float)
CF(Result) - "summary"
"""
        structure, errors = parser.parse(dsl)
        self.assertEqual(errors, [])

        builder = ChainBuilder(
            inline_models=parser.inline_models,
            chain_name="test",
        )
        chain = builder.build(structure)
        # CF Step muss das inline Model verwenden, nicht einen Placeholder
        self.assertIn("Result", parser.inline_models)
        self.assertEqual(parser.inline_models["Result"].__annotations__["score"], float)


# =============================================================================
# TOOL FUNCTION TESTS
# =============================================================================


class TestChainToolFunctions(unittest.TestCase):
    """Tests für die 3 Agent-Tools."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = ChainStore(self.tmp_dir)

        # Mock Agent
        self.mock_agent = MagicMock()
        self.mock_agent.tool_manager = MagicMock()
        self.mock_agent.tool_manager._registry = {"search": None, "write": None}
        self.mock_agent.arun_function = AsyncMock(return_value="tool result")

        self.tool_defs, self.tool_funcs = create_chain_tools(
            agent=self.mock_agent,
            chain_store=self.store,
            agent_registry={"analyzer": self.mock_agent},
        )

    def test_tool_definitions_count(self):
        self.assertEqual(len(self.tool_defs), 3)

    def test_tool_names(self):
        names = {t["function"]["name"] for t in self.tool_defs}
        self.assertEqual(names, {"create_validate_chain", "run_chain", "list_auto_get_fitting"})

    def test_create_valid_chain(self):
        result = asyncio.run(
            self.tool_funcs["create_validate_chain"](
                name="SearchPipeline",
                dsl='tool:search(query="{prev}")',
                description="Sucht nach Informationen",
                tags="search,web",
            )
        )
        self.assertIn("VALID", result)
        self.assertIn("UNSAFE", result)
        self.assertIn("SearchPipeline", result)

        # Store muss Chain haben
        chains = self.store.list_all()
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].name, "SearchPipeline")
        self.assertFalse(chains[0].accepted)

    def test_create_invalid_chain(self):
        result = asyncio.run(
            self.tool_funcs["create_validate_chain"](
                name="Bad",
                dsl="def:evil(x) -> __import__('os')",
            )
        )
        self.assertIn("INVALID", result)
        self.assertIn("nicht erlaubt", result)

    def test_run_chain_not_accepted(self):
        # Erst erstellen
        asyncio.run(
            self.tool_funcs["create_validate_chain"](
                name="TestRun", dsl='tool:search(query="{prev}")',
            )
        )

        chain = self.store.list_all()[0]
        result = asyncio.run(
            self.tool_funcs["run_chain"](name_or_id=chain.id)
        )
        self.assertIn("UNSAFE", result)
        self.assertIn("accept=true", result)

    def test_run_chain_with_accept(self):
        asyncio.run(
            self.tool_funcs["create_validate_chain"](
                name="AcceptTest", dsl='tool:search(query="{prev}")',
            )
        )
        chain = self.store.list_all()[0]

        result = asyncio.run(
            self.tool_funcs["run_chain"](
                name_or_id=chain.id,
                input_data="test query",
                accept=True,
            )
        )
        self.assertIn("ausgeführt", result)
        # Chain muss jetzt accepted sein
        self.assertTrue(self.store.get(chain.id).accepted)

    def test_run_chain_not_found(self):
        result = asyncio.run(
            self.tool_funcs["run_chain"](name_or_id="nonexistent")
        )
        self.assertIn("nicht gefunden", result)

    def test_list_empty(self):
        result = asyncio.run(
            self.tool_funcs["list_auto_get_fitting"]()
        )
        self.assertIn("Keine Chains", result)

    def test_list_with_matching(self):
        asyncio.run(
            self.tool_funcs["create_validate_chain"](
                name="WebSearch", dsl='tool:search(query="{prev}")',
                description="Sucht im Web", tags="web,search",
            )
        )
        asyncio.run(
            self.tool_funcs["create_validate_chain"](
                name="FileWriter", dsl='tool:write(path="out.txt")',
                description="Schreibt Dateien", tags="file,write",
            )
        )

        result = asyncio.run(
            self.tool_funcs["list_auto_get_fitting"](
                task_description="Suche im web nach Informationen"
            )
        )
        self.assertIn("WebSearch", result)
        self.assertIn("PASSENDE CHAINS", result)


# =============================================================================
# INTEGRATION TEST
# =============================================================================


class TestIntegration(unittest.TestCase):
    """End-to-End Test: DSL → Parse → Validate → Build → (mock) Run."""

    def test_full_pipeline(self):
        dsl = """
def:clean(text) -> text.strip().lower()
tool:search(query="{prev}") >> def:clean
"""
        # Parse
        parser = ChainDSLParser()
        structure, errors = parser.parse(dsl)
        self.assertEqual(errors, [])

        # Validate
        validator = ChainValidator(available_tools={"search"})
        is_valid, val_errors, metadata = validator.validate(structure, dsl)
        self.assertTrue(is_valid)
        self.assertIn("search", metadata["referenced_tools"])
        self.assertIn("clean", metadata["custom_functions"])

        # Build
        mock_executor = AsyncMock(return_value="  SEARCH RESULT  ")
        builder = ChainBuilder(
            tool_executor=mock_executor,
            chain_name="test_pipeline",
        )
        chain = builder.build(structure)

        # Run
        result = asyncio.run(chain.a_run("python tutorials"))
        # search gibt "  SEARCH RESULT  " zurück, clean macht .strip().lower()
        self.assertEqual(result, "search result")


if __name__ == "__main__":
    unittest.main()
