"""
Unit tests for RuleSet system.

Compatible with unittest and pytest.
"""

import unittest


from toolboxv2.mods.isaa.base.Agent import rule_set
import sys
import unittest
import importlib
from unittest.mock import MagicMock

# --- FIX START ---
# Wir müssen sicherstellen, dass wir die ECHTE RuleSet Klasse testen,
# nicht den Mock aus conftest.py.
# Wir entfernen den Mock aus sys.modules für diesen Test.

TARGET_MODULE = 'toolboxv2.mods.isaa.base.Agent.rule_set'

# 1. Entferne den Mock aus sys.modules falls vorhanden
if TARGET_MODULE in sys.modules:
    del sys.modules[TARGET_MODULE]

# 2. Importiere das lokale Modul 'rule_set' (das echte File)
# Falls es bereits als 'rule_set' geladen wurde (ggf. als Mock), reloaden wir es.
# importlib.reload(rule_set)


# --- FIX END ---

class TestRuleSet(unittest.TestCase):
    def setUp(self):
        # Stelle sicher, dass wir eine echte Instanz haben
        from toolboxv2.mods.isaa.base.Agent.rule_set import (
            RuleSet
        )
        self.rs = RuleSet(auto_sync_vfs=False)

    def test_01_register_group(self):
        # Testet das Registrieren einer ToolGroup
        grp = self.rs.register_tool_group("g1", "Display G1", ["t1", "t2"], ["kw1"])

        # Verifiziere, dass das zurückgegebene Objekt korrekt ist
        self.assertEqual(grp.name, "g1")
        self.assertEqual(grp.display_name, "Display G1")

        # Verifiziere State im RuleSet
        self.assertIn("g1", self.rs.tool_groups)
        self.assertEqual(self.rs.tool_groups["g1"], grp)

    def test_02_group_matching(self):
        grp = self.rs.register_tool_group("discord_tools", "Discord", [], ["discord"])
        self.assertTrue(grp.matches_intent("Setup a discord bot"))
        self.assertFalse(grp.matches_intent("Write a file"))

    def test_03_add_rule(self):
        rule = self.rs.add_rule(
            situation="coding",
            intent="refactor",
            instructions=["Do X"],
            rule_id="r1"
        )
        self.assertEqual(rule.id, "r1")
        self.assertEqual(rule.situation, "coding")
        self.assertIn("r1", self.rs.situation_rules)

    def test_04_match_rules_exact(self):
        self.rs.add_rule("A", "B", [], rule_id="r1")
        matches = self.rs.match_rules("A", "B")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].id, "r1")

    def test_05_match_rules_fuzzy(self):
        self.rs.add_rule("work on api", "create endpoint", [])
        matches = self.rs.match_rules("working api", "creating endpoint")
        self.assertEqual(len(matches), 1)

    def test_06_set_situation(self):
        self.rs.register_tool_group("g1", "G1", [], ["kw"])
        self.rs.set_situation("ctx", "use kw")

        self.assertEqual(self.rs.current_intent, "use kw")
        self.assertEqual(self.rs.current_situation, "ctx")
        # Should activate group based on keyword match in intent
        self.assertIn("g1", self.rs._active_tool_groups)

    def test_07_rule_on_action_allow(self):
        res = self.rs.rule_on_action("print")
        self.assertTrue(res.allowed)

    def test_08_rule_on_action_restrict(self):
        # Default safety rule simulation manually added for test
        # (Standard RuleSet might have defaults, but we want to be explicit)
        self.rs.situation_rules.clear()
        self.rs.add_rule("any", "save", [], preconditions=["validated"])

        # Activate the rule
        self.rs.set_situation("any", "save")

        # Action 'save' matches the context active rules intent?
        # Actually rule_on_action logic in RuleSet implementation checks active rules.
        # It checks if ANY active rule has preconditions for the current context.

        res = self.rs.rule_on_action("save")

        # Da preconditions=["validated"] gesetzt ist und context leer ist -> False
        self.assertFalse(res.allowed)
        self.assertIn("validated", res.required_steps[0])

    def test_09_rule_on_action_pass(self):
        self.rs.situation_rules.clear()
        self.rs.add_rule("any", "save", [], preconditions=["validated"])
        self.rs.set_situation("any", "save")

        res = self.rs.rule_on_action("save", context={"validated": True})
        self.assertTrue(res.allowed)

    def test_10_learn_pattern(self):
        pat = self.rs.learn_pattern("Sky is blue")
        self.assertEqual(pat.pattern, "Sky is blue")
        self.assertIn(pat, self.rs.learned_patterns)

    def test_11_prune_patterns(self):
        self.rs.learn_pattern("Bad", confidence=0.1)
        self.rs.learn_pattern("Good", confidence=0.9)
        removed = self.rs.prune_low_confidence_patterns(0.5)
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.rs.learned_patterns), 1)
        self.assertEqual(self.rs.learned_patterns[0].pattern, "Good")

    def test_12_vfs_content(self):
        self.rs.register_tool_group("g1", "G1", [], ["k"])
        content = self.rs.build_vfs_content()
        self.assertIn("# Active Rules", content)
        self.assertIn("G1", content)

    def test_13_suggestion_flow(self):
        # Prepare state for suggestion
        self.rs.add_rule("Sit", "Int", [])

        sugg = self.rs.suggest_situation("Sit", "Int")
        self.assertIsNotNone(self.rs._pending_suggestion)
        self.assertEqual(sugg['intent'], "Int")

        self.rs.confirm_suggestion()
        self.assertEqual(self.rs.current_situation, "Sit")
        self.assertIsNone(self.rs._pending_suggestion)

    def test_14_reject_suggestion(self):
        self.rs.suggest_situation("Sit", "Int")
        self.rs.reject_suggestion()
        self.assertIsNone(self.rs._pending_suggestion)
        self.assertIsNone(self.rs.current_situation)

    def test_15_update_rule(self):
        self.rs.add_rule("A", "B", [], rule_id="r1")
        self.rs.update_rule("r1", confidence=0.5)
        self.assertEqual(self.rs.get_rule("r1").confidence, 0.5)

    def test_16_remove_rule(self):
        self.rs.add_rule("A", "B", [], rule_id="r1")
        self.rs.remove_rule("r1")
        self.assertIsNone(self.rs.get_rule("r1"))

    def test_17_checkpoint(self):
        self.rs.set_situation("S", "I")
        self.rs.register_tool_group("g1", "G1", [], ["k"])
        self.rs.activate_tool_group("g1")

        cp = self.rs.to_checkpoint()
        self.assertEqual(cp['current_situation'], "S")
        self.assertIn("g1", cp['active_tool_groups'])

    def test_18_restore(self):
        # Mock ToolGroup creation during restore implicitly via from_checkpoint logic
        # Data needs to match ToolGroup dataclass fields
        data = {
            'current_situation': 'Restored',
            'current_intent': 'I',
            'active_tool_groups': ['g1'],
            'tool_groups': {
                'g1': {
                    'name': 'g1', 'display_name': 'G1', 'description': 'd',
                    'tool_names': [], 'trigger_keywords': [], 'priority': 5,
                    'icon': 'x', 'auto_generated': False
                }
            },
            'situation_rules': {},
            'learned_patterns': []
        }
        self.rs.from_checkpoint(data)
        self.assertEqual(self.rs.current_situation, "Restored")
        self.assertIn("g1", self.rs._active_tool_groups)
        self.assertIn("g1", self.rs.tool_groups)
