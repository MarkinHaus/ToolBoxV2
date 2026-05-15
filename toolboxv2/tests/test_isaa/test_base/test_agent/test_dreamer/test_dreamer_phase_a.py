"""
Tests for Dreamer V3 — Phase A: Foundation

Tests cover:
- harvest.py: Log parsing, record filtering, cutoff logic
- prompts.py: System prompt building with context, cluster task template
- tools.py: Tool registry completeness, budget calculation
- agent.py: DreamerAgent creation, tool registration, VFS setup

Run:
    python -m unittest test_dreamer_v3_phase_a -v
"""

import json
import os
import re
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, List


# ═══════════════════════════════════════════════════════════════════
# Minimal stubs — just enough to test WITHOUT importing the real project
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DreamConfig:
    max_budget: int = 160000
    max_history_time: Optional[float] = None
    do_skill_split: bool = True
    do_skill_evolve: bool = True
    do_persona_evolve: bool = True
    do_create_new: bool = True
    hard_stop: bool = False
    publish_threshold: float = 0.8
    publish_min_version: int = 3


@dataclass
class RunRecord:
    run_id: str = ""
    query: str = ""
    tools_used: list = field(default_factory=list)
    success: bool = True
    error_traces: list = field(default_factory=list)
    summary: str = ""
    timestamp: str = ""
    log_path: str = ""


# ═══════════════════════════════════════════════════════════════════
# Import the modules under test (will be created after tests)
# ═══════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, "/home/claude")

from toolboxv2.mods.isaa.base.dreamer.harvest import (
    parse_log,
    filter_records,
    get_cutoff,
    harvest_from_vfs,
)
from toolboxv2.mods.isaa.base.dreamer.prompts import (
    build_dreamer_system_prompt,
    build_cluster_analysis_task,
    DREAMER_SYSTEM_PROMPT_TEMPLATE,
)
from toolboxv2.mods.isaa.base.dreamer.tools import (
    get_all_dream_tool_definitions,
    get_dream_tool_names,
    calculate_sub_agent_budget,
    DREAM_DATA_TOOLS,
    DREAM_CLUSTER_TOOLS,
    DREAM_SKILL_TOOLS,
    DREAM_RULE_TOOLS,
    DREAM_PERSONA_TOOLS,
    DREAM_CLEANUP_TOOLS,
    DREAM_PERSIST_TOOLS,
)
from toolboxv2.mods.isaa.base.dreamer.agent import (
    create_dreamer_agent_config,
    build_dream_query,
    prepare_dreamer_vfs,
)


# ═══════════════════════════════════════════════════════════════════
# TEST: harvest.py
# ═══════════════════════════════════════════════════════════════════

class TestParseLog(unittest.TestCase):
    """Test the section-based log parser."""

    def _make_log(self, query="Fix the login bug", tools=None, has_error=False):
        """Helper: build a realistic log file content."""
        tools_section = ""
        if tools:
            for t in tools:
                tools_section += f"### TOOL\n{{'name': '{t}', 'success': True}}\n"

        error_section = ""
        if has_error:
            error_section = "### SYSTEM\n❌ Error: Connection timeout\nTraceback: line 42\n"

        return (
            f"# Execution Log: run_abc123\n"
            f"Query: {query}\n"
            f"----------------------------------------\n"
            f"### SYSTEM\n"
            f"IDENTITY: FlowAgent v3\n"
            f"### USER\n"
            f"{query}\n"
            f"{tools_section}"
            f"### ASSISTANT\n"
            f"Here is the solution...\n"
            f"{error_section}"
            f"### SYSTEM\n"
            f"⚡ RUN SUMMARY [run_abc123]: Task completed\n"
        )

    def test_parse_basic_log(self):
        """Records should extract query, run_id from well-formed log."""
        content = self._make_log("Analyze the server config")
        record = parse_log(content, "/logs/20250410_120000_run1.md")

        self.assertIsNotNone(record)
        self.assertEqual(record.query, "Analyze the server config")
        self.assertEqual(record.run_id, "run_abc123")
        self.assertTrue(record.success)

    def test_parse_extracts_tools(self):
        """Tool names should be extracted from TOOL sections."""
        content = self._make_log("Debug it", tools=["vfs_read", "analyze_codebase"])
        record = parse_log(content, "/logs/20250410_120000_r.md")

        self.assertIsNotNone(record)
        self.assertIn("vfs_read", record.tools_used)
        self.assertIn("analyze_codebase", record.tools_used)

    def test_parse_detects_errors(self):
        """Error traces should be captured, success should be False for real errors."""
        content = self._make_log("Deploy", has_error=True)
        record = parse_log(content, "/logs/20250410_120000_r.md")

        self.assertIsNotNone(record)
        self.assertTrue(len(record.error_traces) > 0)
        # Real errors (with ❌/Traceback) should mark as failure
        self.assertFalse(record.success)

    def test_parse_skips_trivial_queries(self):
        """Logs with only 'hi'/'hello' as query should return None."""
        content = (
            "# Execution Log: run_trivial\n"
            "Query: hi\n"
            "----------------------------------------\n"
            "### USER\nhello\n"
            "### ASSISTANT\nHi!\n"
        )
        record = parse_log(content, "/logs/20250410_120000_r.md")
        self.assertIsNone(record)

    def test_parse_fallback_to_filename_id(self):
        """When header has no run_id, extract from filename."""
        content = (
            "# Execution Log\n"
            "----------------------------------------\n"
            "### USER\nDo something complex\n"
            "### ASSISTANT\nDone.\n"
        )
        record = parse_log(content, "/logs/20250410_143022_myrun.md")

        self.assertIsNotNone(record)
        self.assertEqual(record.run_id, "myrun")
        self.assertEqual(record.timestamp, "20250410_143022")

    def test_parse_deduplicates_tools(self):
        """Same tool appearing multiple times should be deduplicated."""
        content = self._make_log("Test", tools=["vfs_read", "vfs_read", "think"])
        record = parse_log(content, "/logs/20250410_120000_r.md")

        self.assertIsNotNone(record)
        self.assertEqual(len([t for t in record.tools_used if t == "vfs_read"]), 1)

    def test_parse_caps_query_length(self):
        """Very long queries should be capped at 500 chars."""
        long_query = "A" * 1000
        content = self._make_log(long_query)
        record = parse_log(content, "/logs/20250410_120000_r.md")

        self.assertIsNotNone(record)
        self.assertLessEqual(len(record.query), 500)

    def test_parse_empty_content_returns_none(self):
        """Empty or whitespace-only content should return None."""
        self.assertIsNone(parse_log("", "/logs/empty.md"))
        self.assertIsNone(parse_log("   \n\n  ", "/logs/ws.md"))


class TestFilterRecords(unittest.TestCase):
    """Test record filtering logic."""

    def setUp(self):
        self.records = [
            RunRecord(run_id="r1", query="fix python bug", success=True,
                      tools_used=["vfs_read", "think"]),
            RunRecord(run_id="r2", query="discord send message", success=False,
                      tools_used=["discord_send"], error_traces=["timeout"]),
            RunRecord(run_id="r3", query="analyze python file", success=True,
                      tools_used=["analyze_codebase"]),
            RunRecord(run_id="r4", query="create discord bot", success=True,
                      tools_used=["discord_create"]),
        ]

    def test_filter_no_args_returns_all(self):
        result = filter_records(self.records)
        self.assertEqual(len(result), 4)

    def test_filter_by_query_keyword(self):
        result = filter_records(self.records, query_filter="python")
        self.assertEqual(len(result), 2)
        self.assertTrue(all("python" in r.query for r in result))

    def test_filter_success_only(self):
        result = filter_records(self.records, success_only=True)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r.success for r in result))

    def test_filter_failure_only(self):
        result = filter_records(self.records, failure_only=True)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].success)

    def test_filter_with_limit(self):
        result = filter_records(self.records, limit=2)
        self.assertEqual(len(result), 2)

    def test_filter_combined(self):
        """query + success filter should AND together."""
        result = filter_records(self.records, query_filter="discord", success_only=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].run_id, "r4")


class TestGetCutoff(unittest.TestCase):
    """Test harvest time window calculation."""

    def test_explicit_hours(self):
        cutoff = get_cutoff(max_history_time=48.0, last_run_ts=None)
        expected = datetime.now() - timedelta(hours=48)
        # Allow 2 seconds tolerance
        self.assertAlmostEqual(
            cutoff.timestamp(), expected.timestamp(), delta=2.0
        )

    def test_last_run_timestamp(self):
        last = "2025-04-10T12:00:00"
        cutoff = get_cutoff(max_history_time=None, last_run_ts=last)
        self.assertEqual(cutoff, datetime.fromisoformat(last))

    def test_default_fallback(self):
        """No explicit time and no last_run → default 72h."""
        cutoff = get_cutoff(max_history_time=None, last_run_ts=None)
        expected = datetime.now() - timedelta(hours=72)
        self.assertAlmostEqual(
            cutoff.timestamp(), expected.timestamp(), delta=2.0
        )


class TestHarvestFromVFS(unittest.TestCase):
    """Test the VFS log reading orchestration."""

    def test_harvest_skips_non_md_files(self):
        """Only .md files should be processed."""
        mock_vfs = MagicMock()
        mock_vfs.ls.return_value = {
            "success": True,
            "contents": [
                {"name": "20250410_120000_run1.md"},
                {"name": "20250410_120000_run2.txt"},  # not .md
                {"name": "20250410_120000_run3.json"},  # not .md
            ]
        }
        mock_vfs.read.return_value = {
            "success": True,
            "content": (
                "# Execution Log: run1\n"
                "### USER\nDo something\n"
                "### ASSISTANT\nDone\n"
            )
        }

        records = harvest_from_vfs(mock_vfs, "/global/.memory/logs", cutoff=None)
        # Only 1 .md file processed
        self.assertEqual(mock_vfs.read.call_count, 1)

    def test_harvest_respects_cutoff(self):
        """Files older than cutoff should be skipped."""
        mock_vfs = MagicMock()
        mock_vfs.ls.return_value = {
            "success": True,
            "contents": [
                {"name": "20250101_120000_old.md"},  # Jan 1 — old
                {"name": "20250414_120000_recent.md"},  # Apr 14 — recent
            ]
        }
        mock_vfs.read.return_value = {
            "success": True,
            "content": "# Execution Log: recent\n### USER\nRecent task\n### ASSISTANT\nDone\n"
        }

        cutoff = datetime(2025, 4, 1)
        records = harvest_from_vfs(mock_vfs, "/global/.memory/logs", cutoff=cutoff)

        # Only the recent file should be read
        self.assertEqual(mock_vfs.read.call_count, 1)
        call_path = mock_vfs.read.call_args[0][0]
        self.assertIn("recent", call_path)

    def test_harvest_handles_empty_dir(self):
        mock_vfs = MagicMock()
        mock_vfs.ls.return_value = {"success": True, "contents": []}
        records = harvest_from_vfs(mock_vfs, "/global/.memory/logs", cutoff=None)
        self.assertEqual(records, [])

    def test_harvest_handles_ls_failure(self):
        mock_vfs = MagicMock()
        mock_vfs.ls.return_value = {"success": False}
        records = harvest_from_vfs(mock_vfs, "/global/.memory/logs", cutoff=None)
        self.assertEqual(records, [])


# ═══════════════════════════════════════════════════════════════════
# TEST: prompts.py
# ═══════════════════════════════════════════════════════════════════

class TestBuildDreamerSystemPrompt(unittest.TestCase):
    """Test system prompt construction with context variables."""

    def test_all_placeholders_filled(self):
        prompt = build_dreamer_system_prompt(
            parent_agent_name="myagent",
            budget=5000,
            harvest_window="last 72h",
            record_count=47,
            skill_count=23,
            active_count=20,
            rule_count=12,
            persona_count=5,
        )

        self.assertIn("myagent", prompt)
        self.assertIn("5000", prompt)
        self.assertIn("47", prompt)
        self.assertIn("23", prompt)
        self.assertIn("20", prompt)
        self.assertIn("12", prompt)
        self.assertIn("5", prompt)
        self.assertNotIn("{", prompt)  # No unfilled placeholders

    def test_prompt_contains_cleanup_phase(self):
        """Cleanup phase must be present and marked as critical."""
        prompt = build_dreamer_system_prompt(
            parent_agent_name="x", budget=1000, harvest_window="1h",
            record_count=1, skill_count=1, active_count=1,
            rule_count=1, persona_count=1,
        )
        self.assertIn("CLEANUP", prompt.upper())
        self.assertIn("PRUNING", prompt.upper())
        # Must mention deletion is important
        self.assertIn("LÖSCHEN", prompt.upper())

    def test_prompt_contains_all_workflow_phases(self):
        prompt = build_dreamer_system_prompt(
            parent_agent_name="x", budget=1, harvest_window="1h",
            record_count=0, skill_count=0, active_count=0,
            rule_count=0, persona_count=0,
        )
        required_phases = [
            "DATEN SICHTEN", "CLUSTERING", "SKILL-EVOLUTION",
            "REGEL-EXTRAKTION", "PERSONA-EVOLUTION", "MEMORY",
            "CLEANUP", "PERSISTIERUNG", "ABSCHLUSS",
        ]
        for phase in required_phases:
            self.assertIn(phase, prompt.upper(), f"Missing phase: {phase}")


class TestBuildClusterAnalysisTask(unittest.TestCase):
    """Test cluster analysis task template for sub-agents."""

    def test_task_contains_all_context(self):
        task = build_cluster_analysis_task(
            cluster_id="c_0",
            record_count=5,
            success_count=3,
            queries=["fix bug", "debug code", "analyze error"],
            success_tools=["vfs_read", "think", "analyze_codebase"],
            failure_info="Query: deploy failed | Error: timeout",
            existing_skill_context="Skill 'error_recovery' (conf=0.7)",
        )

        self.assertIn("c_0", task)
        self.assertIn("5", task)
        self.assertIn("3", task)
        self.assertIn("fix bug", task)
        self.assertIn("vfs_read", task)
        self.assertIn("timeout", task)
        self.assertIn("error_recovery", task)

    def test_task_references_dreamer_skills_guide(self):
        task = build_cluster_analysis_task(
            cluster_id="c_0", record_count=1, success_count=1,
            queries=["test"], success_tools=[], failure_info="",
            existing_skill_context="",
        )
        self.assertIn("dreamer_skills_guide", task)

    def test_task_requests_json_output(self):
        task = build_cluster_analysis_task(
            cluster_id="c_0", record_count=1, success_count=1,
            queries=["test"], success_tools=[], failure_info="",
            existing_skill_context="",
        )
        # Must ask for structured JSON analysis
        self.assertIn("dominant_intent", task)
        self.assertIn("success_pattern", task)
        self.assertIn("failure_patterns", task)
        self.assertIn("suggested_rules", task)
        self.assertIn("suggested_persona", task)


# ═══════════════════════════════════════════════════════════════════
# TEST: tools.py
# ═══════════════════════════════════════════════════════════════════

class TestToolDefinitions(unittest.TestCase):
    """Test tool registry completeness and structure."""

    def test_all_tool_groups_exist(self):
        """Every tool group constant must be a non-empty list."""
        for group_name, group in [
            ("DREAM_DATA_TOOLS", DREAM_DATA_TOOLS),
            ("DREAM_CLUSTER_TOOLS", DREAM_CLUSTER_TOOLS),
            ("DREAM_SKILL_TOOLS", DREAM_SKILL_TOOLS),
            ("DREAM_RULE_TOOLS", DREAM_RULE_TOOLS),
            ("DREAM_PERSONA_TOOLS", DREAM_PERSONA_TOOLS),
            ("DREAM_CLEANUP_TOOLS", DREAM_CLEANUP_TOOLS),
            ("DREAM_PERSIST_TOOLS", DREAM_PERSIST_TOOLS),
        ]:
            self.assertIsInstance(group, list, f"{group_name} must be list")
            self.assertTrue(len(group) > 0, f"{group_name} must not be empty")

    def test_all_tools_have_valid_schema(self):
        """Every tool must have type, function.name, function.parameters."""
        all_tools = get_all_dream_tool_definitions()
        for tool in all_tools:
            self.assertEqual(tool["type"], "function")
            func = tool["function"]
            self.assertIn("name", func)
            self.assertIn("description", func)
            self.assertIn("parameters", func)
            self.assertTrue(len(func["name"]) > 0)
            self.assertTrue(len(func["description"]) > 10)

    def test_no_duplicate_tool_names(self):
        """Tool names must be unique across all groups."""
        names = get_dream_tool_names()
        self.assertEqual(len(names), len(set(names)),
                         f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}")

    def test_expected_tools_present(self):
        """Critical tools must exist."""
        names = set(get_dream_tool_names())
        required = {
            "dream_get_records", "dream_get_skills", "dream_get_rules",
            "dream_get_personas", "dream_cluster_records",
            "dream_evolve_skill", "dream_create_skill", "dream_merge_skills",
            "dream_split_skill", "dream_compress_skill",
            "dream_extract_rules", "dream_learn_pattern",
            "dream_evolve_persona", "dream_prune_personas",
            "dream_cleanup_skills", "dream_cleanup_rules",
            "dream_delete_skill", "dream_delete_rule",
            "dream_extract_memories", "dream_persist_checkpoint",
        }
        missing = required - names
        self.assertEqual(missing, set(), f"Missing tools: {missing}")

    def test_cleanup_tools_exist_separately(self):
        """Cleanup must have delete + bulk cleanup tools."""
        cleanup_names = {t["function"]["name"] for t in DREAM_CLEANUP_TOOLS}
        self.assertIn("dream_cleanup_skills", cleanup_names)
        self.assertIn("dream_cleanup_rules", cleanup_names)
        self.assertIn("dream_prune_personas", cleanup_names)
        self.assertIn("dream_delete_skill", cleanup_names)
        self.assertIn("dream_delete_rule", cleanup_names)

    def test_evolve_skill_requires_evidence_fields(self):
        """dream_evolve_skill must require cluster_size and success_ratio."""
        evolve_tool = None
        for t in DREAM_SKILL_TOOLS:
            if t["function"]["name"] == "dream_evolve_skill":
                evolve_tool = t
                break
        self.assertIsNotNone(evolve_tool)
        required = evolve_tool["function"]["parameters"].get("required", [])
        self.assertIn("skill_id", required)
        self.assertIn("cluster_size", required)
        self.assertIn("success_ratio", required)


class TestSubAgentBudget(unittest.TestCase):
    """Test automatic budget calculation for cluster analysis sub-agents."""

    def test_single_record_budget(self):
        budget = calculate_sub_agent_budget(1)
        self.assertEqual(budget, 1600)  # 800 + 1*800

    def test_three_records_budget(self):
        budget = calculate_sub_agent_budget(3)
        self.assertEqual(budget, 3200)  # 800 + 3*800

    def test_large_cluster_capped(self):
        """Budget must be capped at 8000."""
        budget = calculate_sub_agent_budget(50)
        self.assertEqual(budget, 8000)

    def test_cap_boundary(self):
        """Exactly at cap threshold."""
        budget = calculate_sub_agent_budget(9)  # 800 + 9*800 = 8000
        self.assertEqual(budget, 8000)
        budget = calculate_sub_agent_budget(10)  # would be 8800, capped
        self.assertEqual(budget, 8000)

    def test_zero_records(self):
        budget = calculate_sub_agent_budget(0)
        self.assertEqual(budget, 800)  # base only


# ═══════════════════════════════════════════════════════════════════
# TEST: agent.py
# ═══════════════════════════════════════════════════════════════════

class TestCreateDreamerAgentConfig(unittest.TestCase):
    """Test DreamerAgent configuration building."""

    def test_config_has_dreamer_name(self):
        config = create_dreamer_agent_config(parent_name="myagent")
        self.assertIn("dreamer", config["name"])

    def test_config_uses_env_model(self):
        with patch.dict(os.environ, {"DREAMER_FAST_MODEL": "test/model-x"}):
            config = create_dreamer_agent_config(parent_name="a")
            self.assertEqual(config["fast_llm_model"], "test/model-x")

    def test_config_fallback_model(self):
        """Without env var, should use default model."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("DREAMER_FAST_MODEL", None)
            config = create_dreamer_agent_config(
                parent_name="a", parent_fast_model="parent/model"
            )
            # Should fall back to parent model or a sensible default
            self.assertTrue(len(config["fast_llm_model"]) > 0)

    def test_config_includes_persona(self):
        """Config must include the dreamer_analyst persona."""
        config = create_dreamer_agent_config(parent_name="test")
        self.assertIn("persona", config)
        persona = config["persona"]
        self.assertEqual(persona["name"], "dreamer_analyst")

    def test_persona_is_not_product_owner(self):
        """Dreamer must NOT get product_owner persona."""
        config = create_dreamer_agent_config(parent_name="test")
        persona = config["persona"]
        self.assertNotEqual(persona["name"], "product_owner")
        self.assertNotIn("Product Owner", persona.get("prompt_modifier", ""))

    def test_persona_has_low_temperature(self):
        """Dreamer should be deterministic (low temperature)."""
        config = create_dreamer_agent_config(parent_name="test")
        temp = config["persona"]["temperature"]
        self.assertLessEqual(temp, 0.3)

    def test_persona_has_high_iterations_factor(self):
        """Dreamer needs more iterations than normal agents."""
        config = create_dreamer_agent_config(parent_name="test")
        factor = config["persona"]["max_iterations_factor"]
        self.assertGreaterEqual(factor, 1.3)

    def test_persona_keywords_include_dream_tools(self):
        """Keywords must cover dream_* tool names."""
        config = create_dreamer_agent_config(parent_name="test")
        keywords = config.get("persona_keywords", [])
        kw_text = " ".join(keywords).lower()
        for tool_prefix in ["dream_get", "dream_evolve", "dream_cleanup", "dream_persist"]:
            self.assertIn(tool_prefix.replace("_", "_"), kw_text,
                          f"Persona keywords should match {tool_prefix}")

    def test_persona_prompt_mentions_cleanup(self):
        """Persona modifier must emphasize cleanup as core duty."""
        config = create_dreamer_agent_config(parent_name="test")
        modifier = config["persona"]["prompt_modifier"].upper()
        self.assertIn("LÖSCH", modifier)  # "Lösche KONSEQUENT"


class TestBuildDreamQuery(unittest.TestCase):
    """Test the query string built for the DreamerAgent."""

    def test_query_contains_record_count(self):
        query = build_dream_query(
            config=DreamConfig(max_budget=5000),
            record_count=42,
            skill_count=15,
            rule_count=8,
        )
        self.assertIn("42", query)

    def test_query_contains_budget(self):
        query = build_dream_query(
            config=DreamConfig(max_budget=3000),
            record_count=10, skill_count=5, rule_count=3,
        )
        self.assertIn("3000", query)

    def test_query_mentions_config_flags(self):
        """Disabled features should be mentioned so agent knows."""
        query = build_dream_query(
            config=DreamConfig(do_skill_split=False, do_persona_evolve=False),
            record_count=10, skill_count=5, rule_count=3,
        )
        # Agent needs to know what's disabled
        self.assertIn("split", query.lower())
        self.assertIn("persona", query.lower())


class TestPrepareDreamerVFS(unittest.TestCase):
    """Test VFS setup for dreamer session."""

    def test_writes_all_required_files(self):
        mock_vfs = MagicMock()

        harvest_data = {
            "records": [
                RunRecord(run_id="r1", query="test", success=True),
            ],
            "skill_snapshot": {"skills": {}, "tool_groups": {}},
            "rule_snapshot": {"situation_rules": {}},
            "persona_snapshot": {},
        }

        prepare_dreamer_vfs(mock_vfs, harvest_data)

        # Check all required files were written
        written_paths = [call[0][0] for call in mock_vfs.write.call_args_list]
        self.assertIn("/harvest/records.json", written_paths)
        self.assertIn("/harvest/skills_snapshot.json", written_paths)
        self.assertIn("/harvest/rules_snapshot.json", written_paths)
        self.assertIn("/harvest/personas_snapshot.json", written_paths)
        self.assertIn("/reference/dreamer_skills_guide.md", written_paths)

    def test_records_serialized_as_json(self):
        mock_vfs = MagicMock()

        records = [
            RunRecord(run_id="r1", query="fix bug", success=True, tools_used=["think"]),
            RunRecord(run_id="r2", query="deploy", success=False),
        ]

        prepare_dreamer_vfs(mock_vfs, {
            "records": records,
            "skill_snapshot": {},
            "rule_snapshot": {},
            "persona_snapshot": {},
        })

        # Find the records.json write call
        for call_args in mock_vfs.write.call_args_list:
            if call_args[0][0] == "/harvest/records.json":
                content = call_args[0][1]
                parsed = json.loads(content)
                self.assertEqual(len(parsed), 2)
                self.assertEqual(parsed[0]["run_id"], "r1")
                self.assertEqual(parsed[1]["success"], False)
                return

        self.fail("records.json was not written")

    def test_dreamer_skills_guide_is_not_empty(self):
        """The guide file must have substantial content."""
        mock_vfs = MagicMock()

        prepare_dreamer_vfs(mock_vfs, {
            "records": [],
            "skill_snapshot": {},
            "rule_snapshot": {},
            "persona_snapshot": {},
        })

        for call_args in mock_vfs.write.call_args_list:
            if call_args[0][0] == "/reference/dreamer_skills_guide.md":
                content = call_args[0][1]
                # Must be substantial (the real SKILL.md is ~500 lines)
                self.assertTrue(len(content) > 100,
                                "dreamer_skills_guide.md should have substantial content")
                return

        self.fail("dreamer_skills_guide.md was not written")


if __name__ == "__main__":
    unittest.main()
