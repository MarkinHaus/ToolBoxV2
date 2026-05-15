"""
Tests for Dreamer V3 — Phase B: Tool Handler

Tests the DreamerToolHandler logic:
- Skill evolution (evidence gate, merge vs replace, rollback)
- Skill creation (id generation, initial confidence)
- Skill merging (primary/secondary, trigger dedup)
- Skill splitting (parent deactivation, sub-skill creation)
- Skill compression (bloat reduction)
- Cleanup skills (delete bad, deactivate stale, merge dupes, compress bloated)
- Cleanup rules (delete low-conf, prune patterns)
- Delete skill/rule (explicit, predefined protection)
- Persona pruning
- Persist checkpoint
- Bloat calculation

Run:
    python -m unittest test_dreamer_v3_phase_b -v
"""

import json
import os
import unittest
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, "/home/claude")

from toolboxv2.mods.isaa.base.dreamer.harvest import RunRecord


# ═══════════════════════════════════════════════════════════════════
# Minimal Skill/RuleSet stubs matching real interfaces
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Skill:
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

    def is_active(self) -> bool:
        return self.confidence >= self.activation_threshold

    def matches_keywords(self, query: str) -> bool:
        return any(t.lower() in query.lower() for t in self.triggers)

    @property
    def effectiveness(self):
        return self.success_count / self.total_uses if self.total_uses else 0.0

    @property
    def avg_iterations(self):
        logged = [e["iters"] for e in self.recent_queries if e.get("iters")]
        return sum(logged) / len(logged) if logged else 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SituationRule:
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
class LearnedPattern:
    pattern: str
    source_situation: str
    confidence: float = 0.5
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    category: str = "general"
    tags: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Import tool_handler under test
# ═══════════════════════════════════════════════════════════════════

from toolboxv2.mods.isaa.base.dreamer.tool_handler import DreamerToolHandler


# ═══════════════════════════════════════════════════════════════════
# HELPER: Build a handler with test data
# ═══════════════════════════════════════════════════════════════════

def _make_handler(skills=None, rules=None, patterns=None, personas=None):
    """Create a DreamerToolHandler with injected test data."""
    handler = DreamerToolHandler.__new__(DreamerToolHandler)

    # Skills
    handler._skills = {}
    for s in (skills or []):
        handler._skills[s.id] = s

    # Rules
    handler._rules = {}
    for r in (rules or []):
        handler._rules[r.id] = r

    # Patterns
    handler._patterns = list(patterns or [])

    # Personas
    handler._personas = dict(personas or {})

    # Records (empty for most tests)
    handler._records = []

    # Report tracking
    handler._report = {
        "skills_evolved": [], "skills_created": [], "skills_merged": [],
        "skills_split": [], "skills_compressed": [], "skills_deleted": [],
        "skills_deactivated": [], "rules_created": [], "rules_deleted": [],
        "patterns_added": [], "patterns_pruned": [], "personas_evolved": [],
        "personas_pruned": [], "memories_added": [],
    }

    handler._dream_cycle_count = 3  # Simulate 3rd dream cycle

    return handler


# ═══════════════════════════════════════════════════════════════════
# TEST: Bloat Calculation
# ═══════════════════════════════════════════════════════════════════

class TestBloatCalculation(unittest.TestCase):

    def test_lean_skill(self):
        h = _make_handler()
        s = Skill(id="s1", name="test", triggers=["a", "b"],
                  instruction="Short instruction", tools_used=["think"])
        bloat = h.calculate_bloat(s)
        self.assertLess(bloat, 0.2)

    def test_bloated_skill(self):
        h = _make_handler()
        s = Skill(id="s1", name="test",
                  triggers=["a"] * 15,
                  instruction="x" * 2000,
                  tools_used=["t"] * 15)
        bloat = h.calculate_bloat(s)
        self.assertGreater(bloat, 0.7)

    def test_bloat_range(self):
        """Bloat should always be 0.0-1.0."""
        h = _make_handler()
        for trig_n, inst_len, tool_n in [(0, 0, 0), (100, 10000, 100)]:
            s = Skill(id="x", name="x", triggers=["t"] * trig_n,
                      instruction="x" * inst_len, tools_used=["t"] * tool_n)
            bloat = h.calculate_bloat(s)
            self.assertGreaterEqual(bloat, 0.0)
            self.assertLessEqual(bloat, 1.0)


# ═══════════════════════════════════════════════════════════════════
# TEST: Skill Evolution
# ═══════════════════════════════════════════════════════════════════

class TestEvolveSkill(unittest.TestCase):

    def test_mature_skill_small_cluster_no_instruction_change(self):
        """Predefined skill with conf≥0.7 and cluster<3 → instruction unchanged."""
        original_instruction = "Original predefined instruction"
        s = Skill(id="s1", name="Test", source="predefined",
                  confidence=0.8, instruction=original_instruction)
        h = _make_handler(skills=[s])

        result = h.handle_evolve_skill(
            skill_id="s1", cluster_size=2, success_ratio=0.8,
            instruction_update="New improved instruction",
        )

        self.assertIn("OK", result)
        # Instruction should NOT be replaced (mature + small cluster)
        self.assertEqual(h._skills["s1"].instruction, original_instruction)

    def test_mature_skill_large_cluster_merges(self):
        """Predefined skill with cluster≥3 → instruction merged, not replaced."""
        s = Skill(id="s1", name="Test", source="predefined",
                  confidence=0.8, instruction="Old instruction with steps")
        h = _make_handler(skills=[s])

        h.handle_evolve_skill(
            skill_id="s1", cluster_size=5, success_ratio=0.9,
            instruction_update="New insight to add",
        )

        inst = h._skills["s1"].instruction
        # Should contain BOTH old and new content
        self.assertIn("Old instruction", inst)
        self.assertIn("New insight", inst)

    def test_learned_skill_gets_replaced(self):
        """Learned skill → instruction can be fully replaced."""
        s = Skill(id="s1", name="Test", source="learned",
                  confidence=0.4, instruction="Old learned instruction")
        h = _make_handler(skills=[s])

        h.handle_evolve_skill(
            skill_id="s1", cluster_size=5, success_ratio=0.7,
            instruction_update="Completely new instruction",
        )

        self.assertEqual(h._skills["s1"].instruction, "Completely new instruction")

    def test_failure_patterns_appended(self):
        s = Skill(id="s1", name="Test", source="learned",
                  confidence=0.5, instruction="Base")
        h = _make_handler(skills=[s])

        h.handle_evolve_skill(
            skill_id="s1", cluster_size=3, success_ratio=0.5,
            failure_patterns=["Timeout bei großen Dateien", "API rate limit"],
        )

        inst = h._skills["s1"].instruction
        self.assertIn("FALLSTRICKE", inst)
        self.assertIn("Timeout", inst)

    def test_triggers_capped_at_8(self):
        s = Skill(id="s1", name="Test", source="learned",
                  triggers=["a", "b", "c"], confidence=0.5)
        h = _make_handler(skills=[s])

        h.handle_evolve_skill(
            skill_id="s1", cluster_size=3, success_ratio=0.7,
            new_triggers=["d", "e", "f", "g", "h", "i", "j"],
        )

        self.assertLessEqual(len(h._skills["s1"].triggers), 8)

    def test_confidence_updated(self):
        s = Skill(id="s1", name="Test", source="learned",
                  confidence=0.5, success_count=3, failure_count=2)
        h = _make_handler(skills=[s])

        h.handle_evolve_skill(
            skill_id="s1", cluster_size=5, success_ratio=0.9,
        )

        # Confidence should increase toward success_ratio
        self.assertGreater(h._skills["s1"].confidence, 0.5)

    def test_nonexistent_skill_returns_error(self):
        h = _make_handler()
        result = h.handle_evolve_skill(
            skill_id="nonexistent", cluster_size=3, success_ratio=0.5,
        )
        self.assertIn("ERROR", result)

    def test_rollback_stored(self):
        s = Skill(id="s1", name="Test", source="learned",
                  confidence=0.4, instruction="V1 instruction")
        h = _make_handler(skills=[s])

        h.handle_evolve_skill(
            skill_id="s1", cluster_size=5, success_ratio=0.7,
            instruction_update="V2 instruction",
        )

        self.assertTrue(hasattr(h._skills["s1"], '_instruction_history'))
        self.assertEqual(h._skills["s1"]._instruction_history[-1]["instruction"], "V1 instruction")


# ═══════════════════════════════════════════════════════════════════
# TEST: Skill Creation
# ═══════════════════════════════════════════════════════════════════

class TestCreateSkill(unittest.TestCase):

    def test_create_basic_skill(self):
        h = _make_handler()
        result = h.handle_create_skill(
            name="API Debugger",
            triggers=["api", "debug", "endpoint"],
            instruction="1. Check endpoint\n2. Analyze response\n3. Fix headers",
            tools_used=["vfs_read", "think"],
        )

        self.assertIn("OK", result)
        self.assertEqual(len(h._skills), 1)
        skill = list(h._skills.values())[0]
        self.assertEqual(skill.source, "learned")
        self.assertAlmostEqual(skill.confidence, 0.3)

    def test_create_generates_unique_id(self):
        existing = Skill(id="learned_api_debugger", name="API Debugger",
                         triggers=["api"], source="learned")
        h = _make_handler(skills=[existing])

        h.handle_create_skill(
            name="API Debugger",
            triggers=["api", "rest"],
            instruction="New version",
        )

        # Should have 2 skills with different IDs
        self.assertEqual(len(h._skills), 2)
        ids = list(h._skills.keys())
        self.assertNotEqual(ids[0], ids[1])

    def test_create_with_failure_patterns(self):
        h = _make_handler()
        h.handle_create_skill(
            name="Deploy Helper",
            triggers=["deploy"],
            instruction="Steps...",
            failure_patterns=["Don't deploy on Friday"],
        )

        skill = list(h._skills.values())[0]
        self.assertIn("FALLSTRICKE", skill.instruction)


# ═══════════════════════════════════════════════════════════════════
# TEST: Skill Merging
# ═══════════════════════════════════════════════════════════════════

class TestMergeSkills(unittest.TestCase):

    def test_merge_keeps_higher_confidence(self):
        s1 = Skill(id="s1", name="A", confidence=0.8, triggers=["a"],
                    tools_used=["t1"], source="learned")
        s2 = Skill(id="s2", name="A", confidence=0.5, triggers=["b"],
                    tools_used=["t2"], source="learned")
        h = _make_handler(skills=[s1, s2])

        result = h.handle_merge_skills("s1", "s2")

        self.assertIn("OK", result)
        # s1 should survive (higher confidence), s2 deleted
        self.assertIn("s1", h._skills)
        self.assertNotIn("s2", h._skills)

    def test_merge_combines_triggers(self):
        s1 = Skill(id="s1", name="A", triggers=["a", "b"], source="learned")
        s2 = Skill(id="s2", name="A", triggers=["b", "c"], source="learned")
        h = _make_handler(skills=[s1, s2])

        h.handle_merge_skills("s1", "s2")

        triggers = h._skills["s1"].triggers
        self.assertIn("a", triggers)
        self.assertIn("c", triggers)
        # "b" should appear only once
        self.assertEqual(triggers.count("b"), 1)

    def test_merge_combines_tools(self):
        s1 = Skill(id="s1", name="A", tools_used=["t1"], source="learned")
        s2 = Skill(id="s2", name="A", tools_used=["t1", "t2"], source="learned")
        h = _make_handler(skills=[s1, s2])

        h.handle_merge_skills("s1", "s2")

        tools = h._skills["s1"].tools_used
        self.assertIn("t2", tools)
        self.assertEqual(len([t for t in tools if t == "t1"]), 1)

    def test_merge_nonexistent_returns_error(self):
        h = _make_handler()
        result = h.handle_merge_skills("nope", "nada")
        self.assertIn("ERROR", result)


# ═══════════════════════════════════════════════════════════════════
# TEST: Skill Splitting
# ═══════════════════════════════════════════════════════════════════

class TestSplitSkill(unittest.TestCase):

    def test_split_creates_sub_skills(self):
        parent = Skill(id="sp", name="Big Skill", confidence=0.7,
                       triggers=["big"], tools_used=["t1", "t2", "t3"],
                       instruction="Big long instruction", source="learned")
        h = _make_handler(skills=[parent])

        result = h.handle_split_skill("sp", ["Sub Intent A", "Sub Intent B"])

        self.assertIn("OK", result)
        # Should have parent + 2 sub-skills
        self.assertGreaterEqual(len(h._skills), 3)

    def test_split_deactivates_parent(self):
        parent = Skill(id="sp", name="Big", confidence=0.7, source="learned")
        h = _make_handler(skills=[parent])

        h.handle_split_skill("sp", ["A", "B"])

        # Parent should be deactivated (threshold > 1.0)
        self.assertFalse(h._skills["sp"].is_active())

    def test_split_sub_skills_inherit_confidence(self):
        parent = Skill(id="sp", name="Big", confidence=0.8, source="learned")
        h = _make_handler(skills=[parent])

        h.handle_split_skill("sp", ["A", "B"])

        for sid, skill in h._skills.items():
            if sid != "sp":
                # Sub-skills should have parent_conf * 0.8
                self.assertAlmostEqual(skill.confidence, 0.64, places=2)


# ═══════════════════════════════════════════════════════════════════
# TEST: Skill Compression
# ═══════════════════════════════════════════════════════════════════

class TestCompressSkill(unittest.TestCase):

    def test_compress_reduces_bloat(self):
        s = Skill(id="s1", name="Bloated",
                  triggers=["a"] * 15,
                  instruction="x" * 2000,
                  tools_used=["t"] * 15,
                  source="learned")
        h = _make_handler(skills=[s])

        bloat_before = h.calculate_bloat(s)
        h.handle_compress_skill("s1")
        bloat_after = h.calculate_bloat(h._skills["s1"])

        self.assertLess(bloat_after, bloat_before)

    def test_compress_caps_triggers(self):
        s = Skill(id="s1", name="T", triggers=["t"] * 20, source="learned")
        h = _make_handler(skills=[s])

        h.handle_compress_skill("s1")

        self.assertLessEqual(len(h._skills["s1"].triggers), 6)

    def test_compress_caps_tools(self):
        s = Skill(id="s1", name="T", tools_used=["t"] * 20, source="learned")
        h = _make_handler(skills=[s])

        h.handle_compress_skill("s1")

        self.assertLessEqual(len(h._skills["s1"].tools_used), 8)


# ═══════════════════════════════════════════════════════════════════
# TEST: Cleanup Skills
# ═══════════════════════════════════════════════════════════════════

class TestCleanupSkills(unittest.TestCase):

    def test_deletes_bad_skills(self):
        """Skills with conf<0.15 and ≥5 uses should be deleted."""
        bad = Skill(id="bad", name="Bad", source="learned",
                    confidence=0.12, total_uses=7)
        good = Skill(id="good", name="Good", source="learned",
                     confidence=0.8, total_uses=10)
        h = _make_handler(skills=[bad, good])

        result = h.handle_cleanup_skills()

        self.assertNotIn("bad", h._skills)
        self.assertIn("good", h._skills)

    def test_never_deletes_predefined(self):
        """Predefined skills must never be deleted, even with low confidence."""
        pred = Skill(id="pred", name="Predefined", source="predefined",
                     confidence=0.1, total_uses=10)
        h = _make_handler(skills=[pred])

        h.handle_cleanup_skills()

        self.assertIn("pred", h._skills)

    def test_deactivates_stale_skills(self):
        """Skills with no match in 3+ cycles → deactivated."""
        stale = Skill(id="stale", name="Stale", source="learned",
                      confidence=0.5,
                      last_used=datetime.now() - timedelta(days=90))
        stale._dream_cycles_since_last_match = 4
        h = _make_handler(skills=[stale])

        h.handle_cleanup_skills()

        # Stale skill should be deactivated
        self.assertFalse(h._skills["stale"].is_active())

    def test_compresses_bloated(self):
        """Skills with bloat>0.7 should be compressed."""
        bloated = Skill(id="bloated", name="Bloated", source="learned",
                        triggers=["a"] * 15,
                        instruction="x" * 2000,
                        tools_used=["t"] * 15)
        h = _make_handler(skills=[bloated])

        bloat_before = h.calculate_bloat(bloated)
        h.handle_cleanup_skills()
        bloat_after = h.calculate_bloat(h._skills["bloated"])

        self.assertLess(bloat_after, bloat_before)


# ═══════════════════════════════════════════════════════════════════
# TEST: Cleanup Rules
# ═══════════════════════════════════════════════════════════════════

class TestCleanupRules(unittest.TestCase):

    def test_deletes_low_confidence_rules(self):
        bad_rule = SituationRule(id="bad", situation="x", intent="y",
                                 confidence=0.15)
        good_rule = SituationRule(id="good", situation="a", intent="b",
                                  confidence=0.8)
        h = _make_handler(rules=[bad_rule, good_rule])

        h.handle_cleanup_rules()

        self.assertNotIn("bad", h._rules)
        self.assertIn("good", h._rules)

    def test_prunes_unused_patterns(self):
        used = LearnedPattern(pattern="useful fact", source_situation="x",
                              usage_count=5)
        unused = LearnedPattern(pattern="forgotten fact", source_situation="x",
                                usage_count=0)
        h = _make_handler(patterns=[used, unused])

        h.handle_cleanup_rules()

        # Unused pattern should be pruned (after 3 cycles)
        patterns_text = [p.pattern for p in h._patterns]
        self.assertIn("useful fact", patterns_text)
        self.assertNotIn("forgotten fact", patterns_text)

    def test_caps_patterns_at_50(self):
        patterns = [
            LearnedPattern(pattern=f"pattern_{i}", source_situation="x",
                           usage_count=1)
            for i in range(60)
        ]
        h = _make_handler(patterns=patterns)

        h.handle_cleanup_rules()

        self.assertLessEqual(len(h._patterns), 50)


# ═══════════════════════════════════════════════════════════════════
# TEST: Delete Skill/Rule
# ═══════════════════════════════════════════════════════════════════

class TestDeleteSkill(unittest.TestCase):

    def test_delete_learned_skill(self):
        s = Skill(id="s1", name="Learned", source="learned")
        h = _make_handler(skills=[s])

        result = h.handle_delete_skill("s1", "Kontraproduktiv")

        self.assertIn("OK", result)
        self.assertNotIn("s1", h._skills)

    def test_delete_predefined_deactivates(self):
        """Predefined can't be deleted, only deactivated."""
        s = Skill(id="pred", name="Predefined", source="predefined")
        h = _make_handler(skills=[s])

        result = h.handle_delete_skill("pred", "Bad performance")

        self.assertIn("pred", h._skills)  # Still exists
        self.assertFalse(h._skills["pred"].is_active())  # But deactivated


class TestDeleteRule(unittest.TestCase):

    def test_delete_rule(self):
        r = SituationRule(id="r1", situation="x", intent="y")
        h = _make_handler(rules=[r])

        result = h.handle_delete_rule("r1", "Widersprüchlich")

        self.assertIn("OK", result)
        self.assertNotIn("r1", h._rules)

    def test_delete_nonexistent(self):
        h = _make_handler()
        result = h.handle_delete_rule("nope", "reason")
        self.assertIn("ERROR", result)


# ═══════════════════════════════════════════════════════════════════
# TEST: Persona Pruning
# ═══════════════════════════════════════════════════════════════════

class TestPrunePersonas(unittest.TestCase):

    def test_prunes_bad_confidence(self):
        personas = {
            "bad_persona": {
                "confidence": 0.2, "evidence_count": 6, "dream_cycles": 3,
                "usage_count": 2, "profile": {},
            },
            "good_persona": {
                "confidence": 0.8, "evidence_count": 10, "dream_cycles": 5,
                "usage_count": 8, "profile": {},
            },
        }
        h = _make_handler(personas=personas)

        result = h.handle_prune_personas()

        self.assertNotIn("bad_persona", h._personas)
        self.assertIn("good_persona", h._personas)

    def test_prunes_zero_usage(self):
        personas = {
            "unused": {
                "confidence": 0.5, "evidence_count": 3, "dream_cycles": 4,
                "usage_count": 0, "profile": {},
            },
        }
        h = _make_handler(personas=personas)

        h.handle_prune_personas()

        self.assertNotIn("unused", h._personas)

    def test_keeps_good_personas(self):
        personas = {
            "active": {
                "confidence": 0.6, "evidence_count": 5, "dream_cycles": 2,
                "usage_count": 3, "profile": {},
            },
        }
        h = _make_handler(personas=personas)

        h.handle_prune_personas()

        self.assertIn("active", h._personas)


# ═══════════════════════════════════════════════════════════════════
# TEST: Rule/Pattern Extraction
# ═══════════════════════════════════════════════════════════════════

class TestExtractRules(unittest.TestCase):

    def test_extract_creates_rules(self):
        h = _make_handler()

        rules_data = [
            {"situation": "discord api work", "intent": "send message",
             "instructions": ["Step 1", "Step 2"]},
        ]

        result = h.handle_extract_rules(rules_data)

        self.assertIn("OK", result)
        self.assertEqual(len(h._rules), 1)
        rule = list(h._rules.values())[0]
        self.assertEqual(rule.situation, "discord api work")


class TestLearnPattern(unittest.TestCase):

    def test_learn_adds_pattern(self):
        h = _make_handler()

        result = h.handle_learn_pattern(
            pattern="Discord needs hex color",
            source_situation="discord api",
            category="api",
            tags=["discord", "color"],
        )

        self.assertIn("OK", result)
        self.assertEqual(len(h._patterns), 1)
        self.assertEqual(h._patterns[0].pattern, "Discord needs hex color")


# ═══════════════════════════════════════════════════════════════════
# TEST: Persist Checkpoint
# ═══════════════════════════════════════════════════════════════════

class TestPersistCheckpoint(unittest.TestCase):

    def test_persist_writes_to_vfs(self):
        s = Skill(id="s1", name="Test", source="learned")
        r = SituationRule(id="r1", situation="x", intent="y")
        h = _make_handler(skills=[s], rules=[r])

        mock_vfs = MagicMock()
        result = h.handle_persist_checkpoint(mock_vfs)

        self.assertIn("OK", result)
        # Should have written multiple files
        self.assertTrue(mock_vfs.write.called)
        written_paths = [c[0][0] for c in mock_vfs.write.call_args_list]
        # At minimum: skills checkpoint, rules checkpoint, report
        self.assertTrue(any("skills" in p for p in written_paths))
        self.assertTrue(any("rules" in p or "rule" in p for p in written_paths))


if __name__ == "__main__":
    unittest.main()
