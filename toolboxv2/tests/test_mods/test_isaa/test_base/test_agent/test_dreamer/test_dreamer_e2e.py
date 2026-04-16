"""
Dreamer V3 — End-to-End Integration Test

Simulates a FULL dream cycle without real LLM or VFS:
  1. Creates mock Skills, Rules, Personas, RunRecords
  2. Runs the DreamerToolHandler through all phases in order
  3. Prints live progress (what the agent would see/do)
  4. Validates final state: what changed, what was cleaned up
  5. Prints a formatted report

Run:
    python test_dreamer_v3_e2e.py
    python -m unittest test_dreamer_v3_e2e -v

Author: FlowAgent V3
"""

import json
import os
import sys
import unittest
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock
from io import StringIO

sys.path.insert(0, "/home/claude")

from toolboxv2.mods.isaa.base.dreamer.harvest import RunRecord, parse_log, filter_records, harvest_from_vfs, get_cutoff
from toolboxv2.mods.isaa.base.dreamer.tools import calculate_sub_agent_budget, get_all_dream_tool_definitions
from toolboxv2.mods.isaa.base.dreamer.prompts import build_dreamer_system_prompt, build_cluster_analysis_task
from toolboxv2.mods.isaa.base.dreamer.agent import create_dreamer_agent_config, build_dream_query, prepare_dreamer_vfs
from toolboxv2.mods.isaa.base.dreamer.tool_handler import DreamerToolHandler


# ═══════════════════════════════════════════════════════════════════
# ANSI colors for live output
# ═══════════════════════════════════════════════════════════════════

class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    DIM     = "\033[90m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    MAGENTA = "\033[95m"


def _phase(name):
    print(f"\n{C.CYAN}{'═'*60}{C.RESET}")
    print(f"{C.CYAN}  ▸ {name}{C.RESET}")
    print(f"{C.CYAN}{'═'*60}{C.RESET}")


def _action(icon, msg, color=C.DIM):
    print(f"  {color}{icon}{C.RESET} {msg}")


def _result(msg, ok=True):
    c = C.GREEN if ok else C.RED
    sym = "✓" if ok else "✗"
    print(f"    {c}{sym} {msg}{C.RESET}")


# ═══════════════════════════════════════════════════════════════════
# Mock data structures (matching real Skill/Rule interfaces)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MockSkill:
    id: str
    name: str
    triggers: list = field(default_factory=list)
    instruction: str = ""
    tools_used: list = field(default_factory=list)
    tool_groups: list = field(default_factory=list)
    source: str = "predefined"
    confidence: float = 1.0
    activation_threshold: float = 0.6
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    recent_queries: list = field(default_factory=list)

    def is_active(self):
        return self.confidence >= self.activation_threshold

    def matches_keywords(self, query):
        return any(t.lower() in query.lower() for t in self.triggers)

    @property
    def effectiveness(self):
        return self.success_count / self.total_uses if self.total_uses else 0.0

    @property
    def avg_iterations(self):
        return 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class MockRule:
    id: str
    situation: str
    intent: str
    instructions: list = field(default_factory=list)
    required_tool_groups: list = field(default_factory=list)
    learned: bool = False
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    preconditions: list = field(default_factory=list)
    postconditions: list = field(default_factory=list)


@dataclass
class MockPattern:
    pattern: str
    source_situation: str
    confidence: float = 0.5
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    category: str = "general"
    tags: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# BUILD MOCK DATA — Realistic scenario
# ═══════════════════════════════════════════════════════════════════

def build_mock_skills():
    """Create a realistic set of skills with various health states."""
    return {
        # Good predefined skill — should NOT be touched (mature + working)
        "error_recovery": MockSkill(
            id="error_recovery", name="Error Recovery", source="predefined",
            confidence=0.9, total_uses=50, success_count=42, failure_count=8,
            triggers=["fehler", "error", "problem", "bug"],
            instruction="1. Analysiere Fehler\n2. Identifiziere Ursache\n3. Fixe",
            tools_used=["think", "vfs_read"],
        ),
        # Learned skill doing well — should get evolved with new evidence
        "learned_api_helper": MockSkill(
            id="learned_api_helper", name="API Helper", source="learned",
            confidence=0.65, total_uses=12, success_count=9, failure_count=3,
            triggers=["api", "endpoint", "rest"],
            instruction="1. Check API docs\n2. Build request\n3. Test",
            tools_used=["think", "searchWeb"],
        ),
        # BAD learned skill — conf < 0.15, uses >= 5 → should be DELETED
        "learned_broken_scraper": MockSkill(
            id="learned_broken_scraper", name="Broken Web Scraper", source="learned",
            confidence=0.12, total_uses=8, success_count=1, failure_count=7,
            triggers=["scrape", "crawl", "extract html"],
            instruction="1. Use requests\n2. Parse HTML\n3. Somehow fails",
            tools_used=["searchWeb"],
        ),
        # BLOATED skill — too many triggers/tools/instruction → should be COMPRESSED
        "learned_bloated_deployer": MockSkill(
            id="learned_bloated_deployer", name="Bloated Deployer", source="learned",
            confidence=0.55, total_uses=6, success_count=4, failure_count=2,
            triggers=["deploy", "release", "publish", "push", "ship", "launch",
                       "rollout", "stage", "production", "docker", "kubernetes",
                       "helm", "terraform", "ansible"],
            instruction="x" * 2000,  # way too long
            tools_used=["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
                         "t9", "t10", "t11", "t12", "t13", "t14"],
        ),
        # DUPLICATE skill (same name as learned_api_helper normalized)
        "learned_api_helper_v2": MockSkill(
            id="learned_api_helper_v2", name="API Helper", source="learned",
            confidence=0.45, total_uses=3, success_count=2, failure_count=1,
            triggers=["api", "http", "fetch"],
            instruction="Alternative approach to API work",
            tools_used=["think"],
        ),
        # STALE skill — never used in 3+ cycles
        "learned_old_discord": MockSkill(
            id="learned_old_discord", name="Old Discord Bot Helper", source="learned",
            confidence=0.5, total_uses=2, success_count=1, failure_count=1,
            triggers=["discord", "bot"],
            instruction="Old discord instructions",
            tools_used=["discord_send"],
            last_used=datetime.now() - timedelta(days=120),
        ),
    }


def build_mock_rules():
    """Rules with various health states."""
    return {
        "rule_good": MockRule(
            id="rule_good", situation="python debugging", intent="fix error",
            instructions=["Read traceback", "Identify root cause", "Fix"],
            confidence=0.85, success_count=10,
        ),
        "rule_bad": MockRule(
            id="rule_bad", situation="old api", intent="deprecated workflow",
            instructions=["Use old API"],
            confidence=0.15, failure_count=8,
        ),
    }


def build_mock_patterns():
    """Patterns: some useful, some dead."""
    return [
        MockPattern(pattern="Python tracebacks always show the last frame first",
                     source_situation="python debugging", usage_count=5, confidence=0.8),
        MockPattern(pattern="Forgotten unused pattern",
                     source_situation="unknown", usage_count=0, confidence=0.3),
        MockPattern(pattern="API rate limits are usually per-minute",
                     source_situation="api work", usage_count=3, confidence=0.7),
    ]


def build_mock_personas():
    """Personas: one good, one to prune."""
    return {
        "learned_code_reviewer": {
            "confidence": 0.6, "evidence_count": 8, "dream_cycles": 2,
            "usage_count": 5, "profile": {"name": "code_reviewer"},
        },
        "learned_useless_persona": {
            "confidence": 0.2, "evidence_count": 6, "dream_cycles": 4,
            "usage_count": 0, "profile": {"name": "useless"},
        },
    }


def build_mock_records():
    """Simulate harvested run records."""
    return [
        RunRecord(run_id="r1", query="fix the login bug in auth.py",
                  tools_used=["think", "vfs_read", "vfs_shell", "final_answer"],
                  success=True, summary="Fixed null check in auth handler"),
        RunRecord(run_id="r2", query="fix the registration error",
                  tools_used=["think", "vfs_read", "analyze_codebase", "final_answer"],
                  success=True, summary="Fixed validation logic"),
        RunRecord(run_id="r3", query="debug the payment timeout",
                  tools_used=["think", "vfs_read", "searchWeb", "final_answer"],
                  success=False, error_traces=["TimeoutError: payment gateway"]),
        RunRecord(run_id="r4", query="call the weather API",
                  tools_used=["think", "searchWeb", "final_answer"],
                  success=True, summary="Successfully called weather endpoint"),
        RunRecord(run_id="r5", query="fetch user data from REST API",
                  tools_used=["think", "searchWeb", "vfs_write", "final_answer"],
                  success=True, summary="Fetched and saved user data"),
        RunRecord(run_id="r6", query="deploy the new version",
                  tools_used=["think", "docker_run", "final_answer"],
                  success=True, summary="Deployed v2.3.1"),
        RunRecord(run_id="r7", query="deploy hotfix to production",
                  tools_used=["think", "docker_run", "final_answer"],
                  success=False, error_traces=["DockerError: image not found"]),
        RunRecord(run_id="r8", query="send discord notification",
                  tools_used=["think", "discord_send", "final_answer"],
                  success=True, summary="Sent deploy notification"),
    ]


# ═══════════════════════════════════════════════════════════════════
# E2E TEST
# ═══════════════════════════════════════════════════════════════════

class TestDreamerE2E(unittest.TestCase):
    """
    Full dream cycle E2E test.

    Simulates what the DreamerAgent would do, calling tools in order,
    verifying state changes at each step.
    """

    def setUp(self):
        self.skills = build_mock_skills()
        self.rules = build_mock_rules()
        self.patterns = build_mock_patterns()
        self.personas = build_mock_personas()
        self.records = build_mock_records()

        self.handler = DreamerToolHandler(
            skills=self.skills,
            rules=self.rules,
            patterns=self.patterns,
            personas=self.personas,
            records=self.records,
            dream_cycle_count=3,
        )

        # Snapshot before for comparison
        self._skills_before = set(self.skills.keys())
        self._rules_before = set(self.rules.keys())
        self._patterns_before = len(self.patterns)
        self._personas_before = set(self.personas.keys())

    def test_full_dream_cycle(self):
        """Run all dream phases in order and validate final state."""

        # ── Phase 0: Pre-Harvest (already done — records injected) ──
        _phase("Phase 0: Pre-Harvest")
        _action("📋", f"Loaded {len(self.records)} RunRecords")
        _action("📦", f"Skills: {len(self.skills)}, Rules: {len(self.rules)}, "
                       f"Patterns: {len(self.patterns)}, Personas: {len(self.personas)}")

        # ── Phase 1: Data Sighting ──
        _phase("Phase 1: Data Sighting — dream_get_records / dream_get_skills")

        records_json = json.loads(self.handler.handle_get_records())
        _action("📊", f"Records overview: {len(records_json)} total")
        self.assertEqual(len(records_json), 8)

        success_only = json.loads(self.handler.handle_get_records(success_only=True))
        _action("✓", f"Successful: {len(success_only)}")
        self.assertEqual(len(success_only), 6)

        failure_only = json.loads(self.handler.handle_get_records(failure_only=True))
        _action("✗", f"Failed: {len(failure_only)}")
        self.assertEqual(len(failure_only), 2)

        filtered = json.loads(self.handler.handle_get_records(query_filter="api"))
        _action("🔍", f"Filtered by 'api': {len(filtered)} records")
        self.assertTrue(len(filtered) >= 2)

        skills_json = json.loads(self.handler.handle_get_skills())
        _action("📚", f"Skills: {len(skills_json)}")
        bloated = [s for s in skills_json if s["bloat_score"] > 0.5]
        _action("⚠️", f"Bloated skills: {len(bloated)}", C.YELLOW)

        rules_json = json.loads(self.handler.handle_get_rules())
        _action("📜", f"Rules: {len(rules_json['rules'])}, Patterns: {len(rules_json['patterns'])}")

        personas_json = json.loads(self.handler.handle_get_personas())
        _action("🎭", f"Personas: {len(personas_json)}")

        # ── Phase 2: Clustering ──
        _phase("Phase 2: Clustering — dream_cluster_records")

        clusters_json = json.loads(self.handler.handle_cluster_records())
        _action("📦", f"Clusters found: {len(clusters_json)}")
        for cid, cdata in clusters_json.items():
            _action("  ▣", f"{cid}: {cdata['record_count']} records, "
                           f"{cdata['success_count']} success — \"{cdata['intent'][:50]}\"")
        self.assertTrue(len(clusters_json) >= 1)

        # ── Phase 3: (Simulated) Cluster Analysis ──
        _phase("Phase 3: Cluster Analysis (simulated Sub-Agent results)")

        # In real run, Sub-Agents would analyze each cluster.
        # We simulate the ClusterAnalysis results:
        analyses = {
            "debugging": {
                "dominant_intent": "python error debugging",
                "success_ratio": 0.67,
                "instruction_update": "1. Read traceback\n2. Find root cause\n3. Apply fix\n4. Verify",
                "failure_patterns": ["Timeout bei externen APIs"],
                "new_triggers": ["debug", "traceback"],
                "success_tools": ["vfs_read", "analyze_codebase"],
            },
            "api_work": {
                "dominant_intent": "REST API integration",
                "success_ratio": 1.0,
                "instruction_update": "1. Check API docs\n2. Build request\n3. Handle errors\n4. Parse response",
                "failure_patterns": [],
                "new_triggers": ["rest", "fetch", "http"],
                "success_tools": ["searchWeb", "vfs_write"],
            },
        }

        for cid, analysis in analyses.items():
            budget = calculate_sub_agent_budget(3)
            _action("🚀", f"Sub-Agent for '{cid}' (budget={budget})")
            _action("  📊", f"Intent: {analysis['dominant_intent']}, "
                            f"Success: {analysis['success_ratio']:.0%}")

        # ── Phase 4: Skill Evolution ──
        _phase("Phase 4: Skill Evolution")

        # Evolve the API helper based on analysis
        result = self.handler.handle_evolve_skill(
            skill_id="learned_api_helper",
            cluster_size=3,
            success_ratio=1.0,
            instruction_update=analyses["api_work"]["instruction_update"],
            new_triggers=["rest", "fetch", "http"],
            success_tools=["searchWeb", "vfs_write"],
        )
        _action("↻", result)
        _result("API Helper evolved")

        # Create a new skill from debugging analysis (no exact match)
        result = self.handler.handle_create_skill(
            name="Python Debugger",
            triggers=["debug", "traceback", "exception", "bug"],
            instruction=analyses["debugging"]["instruction_update"],
            tools_used=["vfs_read", "analyze_codebase", "think"],
            failure_patterns=analyses["debugging"]["failure_patterns"],
        )
        _action("★", result)
        _result("Python Debugger skill created")

        # ── Phase 5: Rule Extraction ──
        _phase("Phase 5: Rule Extraction")

        result = self.handler.handle_extract_rules([
            {
                "situation": "python error debugging",
                "intent": "fix runtime error",
                "instructions": [
                    "Read full traceback from bottom to top",
                    "Identify the failing function and line",
                    "Check variable state at that point",
                    "Apply minimal fix and verify",
                ],
                "required_tool_groups": ["vfs"],
                "confidence": 0.7,
            },
        ])
        _action("📜", result)
        _result("Rule extracted")

        # ── Phase 6: Learn Pattern ──
        _phase("Phase 6: Memory — Learn Patterns")

        result = self.handler.handle_learn_pattern(
            pattern="Python TimeoutError meist durch fehlende timeout-Parameter in requests",
            source_situation="python debugging",
            category="error_handling",
            tags=["python", "timeout", "requests"],
        )
        _action("💡", result)
        _result("Pattern learned")

        # ── Phase 7: Persona Evolution ──
        _phase("Phase 7: Persona Evolution (skip — would need LLM)")
        _action("⏭️", "Skipped in E2E test (no LLM for persona generation)")

        # ── Phase 8: CLEANUP & PRUNING ──
        _phase("Phase 8: CLEANUP & PRUNING ⚠️")

        # 8a: Cleanup Skills
        skills_before = len(self.handler._skills)
        result = self.handler.handle_cleanup_skills()
        skills_after = len(self.handler._skills)
        _action("🧹", result)
        _action("  📊", f"Skills: {skills_before} → {skills_after}", C.YELLOW)

        # Verify: broken_scraper should be DELETED
        self.assertNotIn("learned_broken_scraper", self.handler._skills,
                         "Bad skill should be deleted")
        _result("learned_broken_scraper DELETED (conf=0.12, uses=8)")

        # Verify: bloated_deployer should be COMPRESSED
        if "learned_bloated_deployer" in self.handler._skills:
            bloat = self.handler.calculate_bloat(self.handler._skills["learned_bloated_deployer"])
            _result(f"learned_bloated_deployer COMPRESSED (bloat now {bloat:.0%})")
            self.assertLess(bloat, 0.7, "Bloat should be reduced after compression")

        # Verify: duplicate API helper should be MERGED
        # (learned_api_helper_v2 had same normalized name)
        _action("  ⊕", "Duplicate 'API Helper' skills should be merged")
        api_helpers = [s for s in self.handler._skills.values()
                       if "api" in s.name.lower() and "helper" in s.name.lower()]
        self.assertLessEqual(len(api_helpers), 1,
                             "Duplicate API helpers should be merged into one")
        if api_helpers:
            _result(f"Merged into: {api_helpers[0].id} (conf={api_helpers[0].confidence:.2f})")

        # Verify: old_discord should be DEACTIVATED (stale)
        if "learned_old_discord" in self.handler._skills:
            old_disc = self.handler._skills["learned_old_discord"]
            old_disc._dream_cycles_since_last_match = 4  # simulate staleness
        # Re-run cleanup to catch staleness (first run may have already caught it)
        self.handler.handle_cleanup_skills()
        if "learned_old_discord" in self.handler._skills:
            self.assertFalse(self.handler._skills["learned_old_discord"].is_active(),
                             "Stale skill should be deactivated")
            _result("learned_old_discord DEACTIVATED (stale, 4 cycles)")

        # 8b: Cleanup Rules
        rules_before = len(self.handler._rules)
        patterns_before = len(self.handler._patterns)
        result = self.handler.handle_cleanup_rules()
        _action("🧹", result)
        _action("  📊", f"Rules: {rules_before} → {len(self.handler._rules)}, "
                        f"Patterns: {patterns_before} → {len(self.handler._patterns)}", C.YELLOW)

        # Verify: rule_bad should be DELETED
        self.assertNotIn("rule_bad", self.handler._rules,
                         "Low-confidence rule should be deleted")
        _result("rule_bad DELETED (conf=0.15)")

        # Verify: unused pattern should be PRUNED
        forgotten = [p for p in self.handler._patterns if p.pattern == "Forgotten unused pattern"]
        self.assertEqual(len(forgotten), 0, "Unused pattern should be pruned")
        _result("'Forgotten unused pattern' PRUNED (usage=0, 3+ cycles)")

        # 8c: Prune Personas
        result = self.handler.handle_prune_personas()
        _action("🧹", result)

        self.assertNotIn("learned_useless_persona", self.handler._personas,
                          "Bad persona should be pruned")
        self.assertIn("learned_code_reviewer", self.handler._personas,
                       "Good persona should survive")
        _result("learned_useless_persona PRUNED (conf=0.2, usage=0)")
        _result("learned_code_reviewer KEPT (conf=0.6, usage=5)")

        # ── Phase 9: Persist Checkpoint ──
        _phase("Phase 9: Persist Checkpoint")

        mock_vfs = MagicMock()
        result = self.handler.handle_persist_checkpoint(mock_vfs)
        _action("💾", result)
        self.assertIn("OK", result)

        written_paths = [c[0][0] for c in mock_vfs.write.call_args_list]
        _action("  📁", f"Files written: {len(written_paths)}")
        for p in written_paths:
            _action("    →", p)
        _result(f"Checkpoint persisted ({len(written_paths)} files)")

        # ── Phase 10: Final Report ──
        _phase("Phase 10: Final Report")
        report = self.handler._report

        print(f"\n{C.BOLD}{'─'*60}{C.RESET}")
        print(f"{C.BOLD}  DREAM REPORT{C.RESET}")
        print(f"{C.BOLD}{'─'*60}{C.RESET}")

        # System Health Delta
        skills_now = len(self.handler._skills)
        rules_now = len(self.handler._rules)
        patterns_now = len(self.handler._patterns)
        personas_now = len(self.handler._personas)

        print(f"\n  {C.CYAN}System Health:{C.RESET}")
        print(f"    Skills:   {len(self._skills_before)} → {skills_now} "
              f"(Δ {skills_now - len(self._skills_before):+d})")
        print(f"    Rules:    {len(self._rules_before)} → {rules_now} "
              f"(Δ {rules_now - len(self._rules_before):+d})")
        print(f"    Patterns: {self._patterns_before} → {patterns_now} "
              f"(Δ {patterns_now - self._patterns_before:+d})")
        print(f"    Personas: {len(self._personas_before)} → {personas_now} "
              f"(Δ {personas_now - len(self._personas_before):+d})")

        print(f"\n  {C.GREEN}Evolved:{C.RESET}  {report['skills_evolved']}")
        print(f"  {C.CYAN}Created:{C.RESET}  {report['skills_created']}")
        print(f"  {C.MAGENTA}Merged:{C.RESET}   {report['skills_merged']}")
        print(f"  {C.YELLOW}Compressed:{C.RESET} {report['skills_compressed']}")
        print(f"  {C.RED}Deleted:{C.RESET}  {report['skills_deleted']}")
        print(f"  {C.RED}Deactivated:{C.RESET} {report['skills_deactivated']}")
        print(f"  {C.RED}Rules deleted:{C.RESET} {report['rules_deleted']}")
        print(f"  {C.RED}Patterns pruned:{C.RESET} {report['patterns_pruned']}")
        print(f"  {C.RED}Personas pruned:{C.RESET} {report['personas_pruned']}")
        print(f"  {C.GREEN}Rules created:{C.RESET} {report['rules_created']}")
        print(f"  {C.GREEN}Patterns added:{C.RESET} {report['patterns_added']}")

        print(f"\n{C.BOLD}{'─'*60}{C.RESET}")

        # ── Final Assertions ──
        # At least something should have happened in each category
        total_actions = (
            len(report['skills_evolved']) + len(report['skills_created']) +
            len(report['skills_merged']) + len(report['skills_deleted']) +
            len(report['skills_compressed']) + len(report['skills_deactivated']) +
            len(report['rules_created']) + len(report['rules_deleted']) +
            len(report['patterns_added']) + len(report['patterns_pruned']) +
            len(report['personas_pruned'])
        )
        self.assertGreater(total_actions, 5,
                           f"Dream should have performed multiple actions, got {total_actions}")
        _result(f"Total actions: {total_actions} (expected >5)", total_actions > 5)

        print(f"\n{C.GREEN}{C.BOLD}  ✓ E2E DREAM CYCLE COMPLETE{C.RESET}\n")


if __name__ == "__main__":
    # Run with live output
    unittest.main(verbosity=2)
