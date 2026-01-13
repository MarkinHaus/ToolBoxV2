"""
RuleSet - Dynamic Skill/Behavior System for FlowAgent

Provides:
- Tool grouping with categories (instead of showing 50 tools, show "Discord Tools available")
- Situation-aware instructions based on intent + context
- Runtime learning of patterns and behaviors
- Live VFS integration (always visible after system_context)

Author: FlowAgent V2
"""

import json
import os
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ToolGroup:
    """
    Groups multiple tools under a single display name.
    Instead of showing 50 Discord tools, show "discord_tools: Discord Server APIs"
    """
    name: str                          # "discord_tools"
    display_name: str                  # "Discord Server APIs"
    description: str                   # Short description for agent
    tool_names: list[str]              # Actual tool names in registry
    trigger_keywords: list[str]        # ["discord", "server", "bot", "webhook"]
    priority: int = 5                  # Sorting priority (1=highest)
    icon: str = "🔧"                   # Display icon
    auto_generated: bool = False       # True if from ToolManager category

    def matches_intent(self, intent: str) -> bool:
        """Check if this group matches the given intent"""
        intent_lower = intent.lower()
        return any(kw.lower() in intent_lower for kw in self.trigger_keywords)

    def to_display_line(self, active: bool = False) -> str:
        """Generate display line for VFS"""
        marker = "⭐ ACTIVE" if active else ""
        return f"- {self.name}: {self.display_name} {marker}".strip()


@dataclass
class SituationRule:
    """
    Defines behavior rules for specific situation + intent combinations.

    Example:
        situation: "working on discord server api"
        intent: "create welcome message"
        instructions: [
            "First gather info about message formatting requirements",
            "Create draft and test once in sandbox",
            "Ask human for validation before proceeding",
            "Only after explicit approval: save permanently"
        ]
    """
    id: str
    situation: str                     # Context description
    intent: str                        # What user wants to achieve
    instructions: list[str]            # Step-by-step guidance
    required_tool_groups: list[str]    # Tool groups needed

    # Learning metadata
    learned: bool = False              # Was this learned at runtime?
    success_count: int = 0             # How often successfully used
    failure_count: int = 0             # How often failed
    confidence: float = 1.0            # Confidence in this rule (0.0-1.0)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None

    # Optional conditions
    preconditions: list[str] = field(default_factory=list)  # Must be true
    postconditions: list[str] = field(default_factory=list) # Expected after

    def matches(self, situation: str, intent: str) -> float:
        """
        Calculate match score for given situation and intent.
        Returns 0.0-1.0 match score.
        """
        score = 0.0

        # Exact match is best
        if self.situation.lower() == situation.lower():
            score += 0.5
        elif self._fuzzy_match(self.situation, situation):
            score += 0.3

        if self.intent.lower() == intent.lower():
            score += 0.5
        elif self._fuzzy_match(self.intent, intent):
            score += 0.3

        return min(score * self.confidence, 1.0)

    def _fuzzy_match(self, pattern: str, text: str) -> bool:
        """Simple fuzzy matching - check if key words overlap"""
        pattern_words = set(pattern.lower().split())
        text_words = set(text.lower().split())
        overlap = pattern_words & text_words
        return len(overlap) >= min(2, len(pattern_words) // 2)

    def record_usage(self, success: bool):
        """Record usage for learning"""
        self.last_used = datetime.now()
        if success:
            self.success_count += 1
            # Increase confidence on success
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.failure_count += 1
            # Decrease confidence on failure
            self.confidence = max(0.1, self.confidence - 0.1)


@dataclass
class LearnedPattern:
    """
    Patterns learned during runtime that provide helpful context.

    Example:
        pattern: "Discord embeds require: title, description, color (hex format)"
        source_situation: "discord api work"
        confidence: 0.85
    """
    pattern: str                       # The learned information
    source_situation: str              # Where it was learned
    confidence: float = 0.5            # How confident (0.0-1.0)
    usage_count: int = 0               # How often referenced
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None

    # Optional categorization
    category: str = "general"          # "api", "formatting", "workflow", etc.
    tags: list[str] = field(default_factory=list)

    def is_relevant_to(self, situation: str) -> bool:
        """Check if pattern is relevant to situation"""
        situation_lower = situation.lower()
        source_lower = self.source_situation.lower()

        # Check word overlap
        situation_words = set(situation_lower.split())
        source_words = set(source_lower.split())

        return bool(situation_words & source_words) or \
               any(tag.lower() in situation_lower for tag in self.tags)

    def use(self):
        """Mark pattern as used"""
        self.usage_count += 1
        self.last_used = datetime.now()
        # Slight confidence boost on use
        self.confidence = min(1.0, self.confidence + 0.01)


@dataclass
class RuleResult:
    """Result of rule evaluation for an action"""
    allowed: bool                      # Can the action proceed?
    instructions: list[str]            # Additional instructions to follow
    warnings: list[str]                # Warnings to consider
    required_steps: list[str]          # Steps that must be done first
    suggested_tool_group: str | None   # Recommended tool group
    matched_rule: SituationRule | None = None  # The rule that matched
    confidence: float = 1.0            # Confidence in this result


# =============================================================================
# MAIN RULESET CLASS
# =============================================================================

class RuleSet:
    """
    Dynamic skill/behavior system that provides:
    - Tool grouping for cleaner agent context
    - Situation-aware instructions
    - Runtime learning capabilities
    - Live VFS integration
    """

    def __init__(
        self,
        config_path: str | None = None,
        auto_sync_vfs: bool = True
    ):
        """
        Initialize RuleSet.

        Args:
            config_path: Path to YAML/JSON config file (optional)
            auto_sync_vfs: Automatically mark dirty when changes occur
        """
        # Tool Groups
        self.tool_groups: dict[str, ToolGroup] = {}

        # Situation Rules
        self.situation_rules: dict[str, SituationRule] = {}

        # Learned Patterns
        self.learned_patterns: list[LearnedPattern] = []

        # Current State
        self.current_situation: str | None = None
        self.current_intent: str | None = None
        self._active_tool_groups: set[str] = set()

        # VFS Integration
        self._dirty: bool = True  # Needs VFS update
        self._auto_sync = auto_sync_vfs
        self._vfs_filename = "active_rules"

        # Suggestion system (for L1: Hybrid approach)
        self._pending_suggestion: dict[str, Any] | None = None

        # Load config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    # =========================================================================
    # TOOL GROUP MANAGEMENT
    # =========================================================================

    def register_tool_group(
        self,
        name: str,
        display_name: str,
        tool_names: list[str],
        trigger_keywords: list[str],
        description: str = "",
        priority: int = 5,
        icon: str = "🔧",
        auto_generated: bool = False
    ) -> ToolGroup:
        """
        Register a new tool group.

        Args:
            name: Internal name (e.g., "discord_tools")
            display_name: Display name (e.g., "Discord Server APIs")
            tool_names: List of actual tool names in registry
            trigger_keywords: Keywords that activate this group
            description: Short description
            priority: Sort priority (1=highest)
            icon: Display icon
            auto_generated: True if from ToolManager category

        Returns:
            Created ToolGroup
        """
        group = ToolGroup(
            name=name,
            display_name=display_name,
            description=description or f"Tools for {display_name}",
            tool_names=tool_names,
            trigger_keywords=trigger_keywords,
            priority=priority,
            icon=icon,
            auto_generated=auto_generated
        )

        self.tool_groups[name] = group
        self._mark_dirty()

        return group

    def register_tool_groups_from_categories(
        self,
        category_tools: dict[str, list[str]],
        category_descriptions: dict[str, str] | None = None
    ):
        """
        Auto-generate tool groups from ToolManager categories.

        Args:
            category_tools: Dict mapping category -> list of tool names
            category_descriptions: Optional descriptions per category
        """
        descriptions = category_descriptions or {}

        for category, tools in category_tools.items():
            if not tools:
                continue

            # Generate group name from category
            # "mcp_discord" -> "discord_tools"
            name_parts = category.replace("mcp_", "").replace("a2a_", "").split("_")
            group_name = f"{name_parts[0]}_tools" if name_parts else f"{category}_tools"

            # Generate display name
            display_name = " ".join(word.capitalize() for word in name_parts) + " Tools"

            # Generate trigger keywords from category
            triggers = name_parts + [category]

            self.register_tool_group(
                name=group_name,
                display_name=display_name,
                tool_names=tools,
                trigger_keywords=triggers,
                description=descriptions.get(category, f"Tools from {category}"),
                auto_generated=True
            )

    def unregister_tool_group(self, name: str) -> bool:
        """Remove a tool group"""
        if name in self.tool_groups:
            del self.tool_groups[name]
            self._active_tool_groups.discard(name)
            self._mark_dirty()
            return True
        return False

    def get_groups_for_intent(self, intent: str) -> list[ToolGroup]:
        """Get tool groups that match the given intent"""
        matching = []
        for group in self.tool_groups.values():
            if group.matches_intent(intent):
                matching.append(group)

        # Sort by priority
        return sorted(matching, key=lambda g: g.priority)

    def expand_group(self, group_name: str) -> list[str]:
        """
        Expand a tool group to its actual tool names.
        Used when agent decides to use a tool group.
        """
        if group_name in self.tool_groups:
            return self.tool_groups[group_name].tool_names.copy()
        return []

    def activate_tool_group(self, group_name: str):
        """Mark a tool group as active"""
        if group_name in self.tool_groups:
            self._active_tool_groups.add(group_name)
            self._mark_dirty()

    def deactivate_tool_group(self, group_name: str):
        """Mark a tool group as inactive"""
        self._active_tool_groups.discard(group_name)
        self._mark_dirty()

    # =========================================================================
    # SITUATION & INTENT MANAGEMENT
    # =========================================================================

    def set_situation(self, situation: str, intent: str):
        """
        Set current situation and intent.
        This updates the VFS file and activates relevant tool groups.
        """
        self.current_situation = situation
        self.current_intent = intent

        # Auto-activate relevant tool groups
        self._active_tool_groups.clear()
        for group in self.get_groups_for_intent(intent):
            self._active_tool_groups.add(group.name)

        # Also check situation keywords
        for group in self.tool_groups.values():
            if group.matches_intent(situation):
                self._active_tool_groups.add(group.name)

        self._mark_dirty()

    def suggest_situation(self, situation: str, intent: str) -> dict[str, Any]:
        """
        System suggests a situation/intent (L1: Hybrid approach).
        Agent must confirm before it takes effect.

        Returns suggestion dict that can be confirmed or rejected.
        """
        # Find matching rules
        matching_rules = self.match_rules(situation, intent)
        matching_groups = self.get_groups_for_intent(intent)

        self._pending_suggestion = {
            "situation": situation,
            "intent": intent,
            "matching_rules": [r.id for r in matching_rules],
            "suggested_groups": [g.name for g in matching_groups],
            "timestamp": datetime.now().isoformat()
        }

        return self._pending_suggestion.copy()

    def confirm_suggestion(self) -> bool:
        """Confirm pending suggestion and apply it"""
        if not self._pending_suggestion:
            return False

        self.set_situation(
            self._pending_suggestion["situation"],
            self._pending_suggestion["intent"]
        )
        self._pending_suggestion = None
        return True

    def reject_suggestion(self):
        """Reject pending suggestion"""
        self._pending_suggestion = None

    def clear_situation(self):
        """Clear current situation and intent"""
        self.current_situation = None
        self.current_intent = None
        self._active_tool_groups.clear()
        self._pending_suggestion = None
        self._mark_dirty()

    # =========================================================================
    # RULE MANAGEMENT
    # =========================================================================

    def add_rule(
        self,
        situation: str,
        intent: str,
        instructions: list[str],
        required_tool_groups: list[str] | None = None,
        preconditions: list[str] | None = None,
        postconditions: list[str] | None = None,
        rule_id: str | None = None,
        learned: bool = False,
        confidence: float = 1.0
    ) -> SituationRule:
        """
        Add a new situation rule.

        Args:
            situation: Context description
            intent: What user wants to achieve
            instructions: Step-by-step guidance
            required_tool_groups: Tool groups needed
            preconditions: Conditions that must be true
            postconditions: Expected results
            rule_id: Optional custom ID
            learned: True if learned at runtime
            confidence: Initial confidence

        Returns:
            Created SituationRule
        """
        import uuid

        rule_id = rule_id or f"rule_{uuid.uuid4().hex[:8]}"

        rule = SituationRule(
            id=rule_id,
            situation=situation,
            intent=intent,
            instructions=instructions,
            required_tool_groups=required_tool_groups or [],
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            learned=learned,
            confidence=confidence
        )

        self.situation_rules[rule_id] = rule
        self._mark_dirty()

        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID"""
        if rule_id in self.situation_rules:
            del self.situation_rules[rule_id]
            self._mark_dirty()
            return True
        return False

    def update_rule(self, rule_id: str, **updates) -> bool:
        """Update a rule's attributes"""
        if rule_id not in self.situation_rules:
            return False

        rule = self.situation_rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        self._mark_dirty()
        return True

    def get_rule(self, rule_id: str) -> SituationRule | None:
        """Get rule by ID"""
        return self.situation_rules.get(rule_id)

    def match_rules(
        self,
        situation: str,
        intent: str,
        min_score: float = 0.3
    ) -> list[SituationRule]:
        """
        Find rules that match the given situation and intent.

        Returns list of matching rules sorted by match score.
        """
        matches = []

        for rule in self.situation_rules.values():
            score = rule.matches(situation, intent)
            if score >= min_score:
                matches.append((score, rule))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)

        return [rule for _, rule in matches]

    def get_active_rules(self) -> list[SituationRule]:
        """Get rules matching current situation/intent"""
        if not self.current_situation or not self.current_intent:
            return []

        return self.match_rules(self.current_situation, self.current_intent)

    # =========================================================================
    # LEARNING SYSTEM
    # =========================================================================

    def record_rule_success(self, rule_id: str):
        """Record successful rule application"""
        if rule_id in self.situation_rules:
            self.situation_rules[rule_id].record_usage(success=True)
            self._mark_dirty()

    def record_rule_failure(self, rule_id: str):
        """Record failed rule application"""
        if rule_id in self.situation_rules:
            self.situation_rules[rule_id].record_usage(success=False)
            self._mark_dirty()

    def learn_pattern(
        self,
        pattern: str,
        source_situation: str | None = None,
        confidence: float = 0.5,
        category: str = "general",
        tags: list[str] | None = None
    ) -> LearnedPattern:
        """
        Learn a new pattern from runtime experience.

        Args:
            pattern: The information learned
            source_situation: Where it was learned (default: current)
            confidence: Initial confidence
            category: Pattern category
            tags: Optional tags for matching

        Returns:
            Created LearnedPattern
        """
        source = source_situation or self.current_situation or "unknown"

        learned = LearnedPattern(
            pattern=pattern,
            source_situation=source,
            confidence=confidence,
            category=category,
            tags=tags or []
        )

        self.learned_patterns.append(learned)
        self._mark_dirty()

        return learned

    def get_relevant_patterns(
        self,
        situation: str | None = None,
        min_confidence: float = 0.3,
        limit: int = 10
    ) -> list[LearnedPattern]:
        """
        Get patterns relevant to the given or current situation.
        """
        target_situation = situation or self.current_situation or ""

        relevant = []
        for pattern in self.learned_patterns:
            if pattern.confidence >= min_confidence:
                if pattern.is_relevant_to(target_situation):
                    relevant.append(pattern)

        # Sort by confidence and usage
        relevant.sort(
            key=lambda p: (p.confidence, p.usage_count),
            reverse=True
        )

        return relevant[:limit]

    def prune_low_confidence_patterns(self, threshold: float = 0.2) -> int:
        """
        Remove patterns below confidence threshold.
        Returns count of removed patterns.
        """
        before_count = len(self.learned_patterns)
        self.learned_patterns = [
            p for p in self.learned_patterns
            if p.confidence >= threshold
        ]
        removed = before_count - len(self.learned_patterns)

        if removed > 0:
            self._mark_dirty()

        return removed

    # =========================================================================
    # CORE EXPOSED METHODS
    # =========================================================================

    def get_current_rule_set(self) -> dict[str, Any]:
        """
        Get complete current rule set state.
        Used for inspection and debugging.

        Returns:
            Dict with:
            - tool_groups: All groups with active status
            - situation: Current situation
            - intent: Current intent
            - active_rules: Currently matching rules
            - patterns: Relevant learned patterns
            - pending_suggestion: If any
        """
        active_rules = self.get_active_rules()
        relevant_patterns = self.get_relevant_patterns()

        return {
            "tool_groups": [
                {
                    "name": g.name,
                    "display_name": g.display_name,
                    "description": g.description,
                    "tool_count": len(g.tool_names),
                    "active": g.name in self._active_tool_groups,
                    "priority": g.priority
                }
                for g in sorted(self.tool_groups.values(), key=lambda x: x.priority)
            ],
            "situation": self.current_situation,
            "intent": self.current_intent,
            "active_rules": [
                {
                    "id": r.id,
                    "instructions": r.instructions,
                    "required_groups": r.required_tool_groups,
                    "confidence": r.confidence,
                    "success_count": r.success_count
                }
                for r in active_rules
            ],
            "patterns": [
                {
                    "pattern": p.pattern,
                    "confidence": p.confidence,
                    "category": p.category
                }
                for p in relevant_patterns
            ],
            "pending_suggestion": self._pending_suggestion
        }

    def rule_on_action(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> RuleResult:
        """
        Evaluate if an action is allowed based on current rules.

        Args:
            action: The action being attempted (e.g., "save_permanent", "delete")
            context: Additional context (e.g., {"tool": "discord_save", "validated": False})

        Returns:
            RuleResult with allowed status and instructions
        """
        context = context or {}
        active_rules = self.get_active_rules()

        # Default: allowed with no special instructions
        if not active_rules:
            return RuleResult(
                allowed=True,
                instructions=[],
                warnings=[],
                required_steps=[],
                suggested_tool_group=None
            )

        # Check rules for restrictions
        all_instructions = []
        all_warnings = []
        required_steps = []
        suggested_group = None

        best_match: SituationRule | None = None
        best_confidence = 0.0

        for rule in active_rules:
            # Collect instructions
            all_instructions.extend(rule.instructions)

            # Check preconditions
            for precond in rule.preconditions:
                if not self._evaluate_precondition(precond, context):
                    required_steps.append(precond)

            # Suggest tool group from rule
            if rule.required_tool_groups and not suggested_group:
                suggested_group = rule.required_tool_groups[0]

            # Track best matching rule
            if rule.confidence > best_confidence:
                best_confidence = rule.confidence
                best_match = rule

        # Check specific action restrictions
        action_lower = action.lower()

        # Common restriction patterns
        if "save" in action_lower or "permanent" in action_lower:
            if not context.get("validated", False):
                all_warnings.append("Permanent save without validation detected")
                required_steps.append("Request human validation before permanent save")

        if "delete" in action_lower:
            all_warnings.append("Destructive action detected - ensure confirmation")

        # Determine if allowed
        allowed = len(required_steps) == 0

        return RuleResult(
            allowed=allowed,
            instructions=list(dict.fromkeys(all_instructions)),  # Remove duplicates
            warnings=all_warnings,
            required_steps=required_steps,
            suggested_tool_group=suggested_group,
            matched_rule=best_match,
            confidence=best_confidence if best_match else 1.0
        )

    def _evaluate_precondition(self, precondition: str, context: dict[str, Any]) -> bool:
        """
        Evaluate a precondition string against context.
        Simple implementation - can be extended.
        """
        precond_lower = precondition.lower()

        # Check for validation requirement
        if "validation" in precond_lower or "validated" in precond_lower:
            return context.get("validated", False)

        # Check for confirmation requirement
        if "confirm" in precond_lower:
            return context.get("confirmed", False)

        # Check for test requirement
        if "test" in precond_lower:
            return context.get("tested", False)

        # Default: assume met
        return True

    # =========================================================================
    # VFS INTEGRATION
    # =========================================================================

    def get_vfs_filename(self) -> str:
        """Get VFS filename for this rule set"""
        return self._vfs_filename

    def is_dirty(self) -> bool:
        """Check if VFS content needs update"""
        return self._dirty

    def _mark_dirty(self):
        """Mark as needing VFS update"""
        if self._auto_sync:
            self._dirty = True

    def mark_clean(self):
        """Mark as synced with VFS"""
        self._dirty = False

    def build_vfs_content(self) -> str:
        """
        Build VFS file content for agent visibility.
        This is what the agent sees in the context window.
        """
        lines = []

        # Header
        lines.append("# Active Rules & Tool Groups")
        lines.append("")

        # Tool Groups Section
        lines.append("## Available Tool Groups")

        if self.tool_groups:
            sorted_groups = sorted(
                self.tool_groups.values(),
                key=lambda g: (0 if g.name in self._active_tool_groups else 1, g.priority)
            )

            for group in sorted_groups:
                is_active = group.name in self._active_tool_groups
                marker = " ⭐ ACTIVE" if is_active else ""
                triggers = ", ".join(group.trigger_keywords[:3])
                lines.append(f"- {group.icon} {group.name}: {group.display_name}{marker}")
                lines.append(f"  └─ Triggers: {triggers}")
        else:
            lines.append("(No tool groups registered)")

        lines.append("")

        # Current Situation Section
        lines.append("## Current Situation")

        if self.current_intent or self.current_situation:
            lines.append(f"Intent: {self.current_intent or 'unknown'}")
            lines.append(f"Context: {self.current_situation or 'none'}")
        else:
            lines.append("Intent: unknown")
            lines.append("Context: none")

        # Pending suggestion
        if self._pending_suggestion:
            lines.append("")
            lines.append("⚠️ PENDING SUGGESTION (confirm or reject):")
            lines.append(f"  Suggested Intent: {self._pending_suggestion['intent']}")
            lines.append(f"  Suggested Context: {self._pending_suggestion['situation']}")

        lines.append("")

        # Active Rules Section
        lines.append("## Active Rules")

        active_rules = self.get_active_rules()

        if active_rules:
            for i, rule in enumerate(active_rules[:5], 1):  # Max 5 rules shown
                confidence_indicator = "●" * int(rule.confidence * 5) + "○" * (5 - int(rule.confidence * 5))
                lines.append(f"### Rule {i}: {rule.intent[:50]} [{confidence_indicator}]")

                for j, instruction in enumerate(rule.instructions, 1):
                    lines.append(f"   {j}. {instruction}")

                if rule.required_tool_groups:
                    groups_str = ", ".join(rule.required_tool_groups)
                    lines.append(f"   └─ Required tools: {groups_str}")

                lines.append("")
        else:
            lines.append("(No specific rules active - general operation mode)")

        lines.append("")

        # Learned Patterns Section
        lines.append("## Learned Patterns")

        patterns = self.get_relevant_patterns(limit=5)

        if patterns:
            for pattern in patterns:
                conf = f"[{pattern.confidence:.0%}]"
                lines.append(f"- {pattern.pattern} {conf}")
        else:
            lines.append("(No learned patterns yet)")

        return "\n".join(lines)

    # =========================================================================
    # CONFIG & SERIALIZATION
    # =========================================================================

    def load_config(self, path: str) -> bool:
        """
        Load configuration from YAML or JSON file.

        Expected format:
        ```yaml
        tool_groups:
          - name: discord_tools
            display_name: Discord Server APIs
            tool_names: [discord_send, discord_create, ...]
            trigger_keywords: [discord, server, bot]
            priority: 3

        rules:
          - situation: working on discord server api
            intent: create welcome message
            instructions:
              - First gather info about message formatting
              - Create draft and test once
              - Ask human for validation
              - Only after approval: save permanently
            required_tool_groups: [discord_tools]

        patterns:
          - pattern: Discord embeds need title, description, color
            category: api
            confidence: 0.8
        ```
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            # Load tool groups
            for group_data in config.get('tool_groups', []):
                self.register_tool_group(
                    name=group_data['name'],
                    display_name=group_data.get('display_name', group_data['name']),
                    tool_names=group_data.get('tool_names', []),
                    trigger_keywords=group_data.get('trigger_keywords', []),
                    description=group_data.get('description', ''),
                    priority=group_data.get('priority', 5),
                    icon=group_data.get('icon', '🔧')
                )

            # Load rules
            for rule_data in config.get('rules', []):
                self.add_rule(
                    situation=rule_data['situation'],
                    intent=rule_data['intent'],
                    instructions=rule_data.get('instructions', []),
                    required_tool_groups=rule_data.get('required_tool_groups', []),
                    preconditions=rule_data.get('preconditions', []),
                    postconditions=rule_data.get('postconditions', []),
                    rule_id=rule_data.get('id'),
                    confidence=rule_data.get('confidence', 1.0)
                )

            # Load patterns
            for pattern_data in config.get('patterns', []):
                self.learn_pattern(
                    pattern=pattern_data['pattern'],
                    source_situation=pattern_data.get('source_situation', 'config'),
                    confidence=pattern_data.get('confidence', 0.8),
                    category=pattern_data.get('category', 'general'),
                    tags=pattern_data.get('tags', [])
                )

            self._mark_dirty()
            return True

        except Exception as e:
            print(f"[RuleSet] Failed to load config from {path}: {e}")
            return False

    def save_config(self, path: str) -> bool:
        """Save current configuration to file"""
        try:
            config = {
                'tool_groups': [
                    {
                        'name': g.name,
                        'display_name': g.display_name,
                        'description': g.description,
                        'tool_names': g.tool_names,
                        'trigger_keywords': g.trigger_keywords,
                        'priority': g.priority,
                        'icon': g.icon
                    }
                    for g in self.tool_groups.values()
                    if not g.auto_generated  # Don't save auto-generated
                ],
                'rules': [
                    {
                        'id': r.id,
                        'situation': r.situation,
                        'intent': r.intent,
                        'instructions': r.instructions,
                        'required_tool_groups': r.required_tool_groups,
                        'preconditions': r.preconditions,
                        'postconditions': r.postconditions,
                        'learned': r.learned,
                        'confidence': r.confidence,
                        'success_count': r.success_count
                    }
                    for r in self.situation_rules.values()
                ],
                'patterns': [
                    {
                        'pattern': p.pattern,
                        'source_situation': p.source_situation,
                        'confidence': p.confidence,
                        'category': p.category,
                        'tags': p.tags,
                        'usage_count': p.usage_count
                    }
                    for p in self.learned_patterns
                ]
            }

            with open(path, 'w', encoding='utf-8') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"[RuleSet] Failed to save config to {path}: {e}")
            return False

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize for checkpoint"""
        return {
            'tool_groups': {
                name: asdict(group)
                for name, group in self.tool_groups.items()
            },
            'situation_rules': {
                rule_id: {
                    **asdict(rule),
                    'created_at': rule.created_at.isoformat(),
                    'last_used': rule.last_used.isoformat() if rule.last_used else None
                }
                for rule_id, rule in self.situation_rules.items()
            },
            'learned_patterns': [
                {
                    **asdict(p),
                    'created_at': p.created_at.isoformat(),
                    'last_used': p.last_used.isoformat() if p.last_used else None
                }
                for p in self.learned_patterns
            ],
            'current_situation': self.current_situation,
            'current_intent': self.current_intent,
            'active_tool_groups': list(self._active_tool_groups)
        }

    def from_checkpoint(self, data: dict[str, Any]):
        """Restore from checkpoint"""
        # Restore tool groups
        self.tool_groups.clear()
        for name, group_data in data.get('tool_groups', {}).items():
            self.tool_groups[name] = ToolGroup(**group_data)

        # Restore rules
        self.situation_rules.clear()
        for rule_id, rule_data in data.get('situation_rules', {}).items():
            # Convert datetime strings back
            if isinstance(rule_data.get('created_at'), str):
                rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
            if rule_data.get('last_used') and isinstance(rule_data['last_used'], str):
                rule_data['last_used'] = datetime.fromisoformat(rule_data['last_used'])

            self.situation_rules[rule_id] = SituationRule(**rule_data)

        # Restore patterns
        self.learned_patterns.clear()
        for pattern_data in data.get('learned_patterns', []):
            if isinstance(pattern_data.get('created_at'), str):
                pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
            if pattern_data.get('last_used') and isinstance(pattern_data['last_used'], str):
                pattern_data['last_used'] = datetime.fromisoformat(pattern_data['last_used'])

            self.learned_patterns.append(LearnedPattern(**pattern_data))

        # Restore state
        self.current_situation = data.get('current_situation')
        self.current_intent = data.get('current_intent')
        self._active_tool_groups = set(data.get('active_tool_groups', []))

        self._mark_dirty()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_ruleset(config_path: str | None = None) -> RuleSet:
    """
    Create a RuleSet with sensible defaults.
    """
    ruleset = RuleSet(config_path=config_path)

    if not ruleset.situation_rules:

        # =========================
        # GENERAL RULES (1)
        # =========================

        ruleset.add_rule(
            situation="any",
            intent="insufficient information",
            instructions=[
                "Detect missing, ambiguous, or contradictory information",
                "Explicitly ask the user for the missing details using kernel_ask_user",
                "Do not assume defaults for critical parameters",
                "Pause execution until required information is provided or timeout occurs"
            ],
            required_tool_groups=["communication"],
            rule_id="general_missing_information",
            confidence=1.0
        )

        # =========================
        # SPECIFIC RULES (5)
        # =========================

        # 1. Task scheduling
        ruleset.add_rule(
            situation="task scheduling",
            intent="schedule reminder or job",
            instructions=[
                "Verify task_type and content are provided",
                "Check whether delay_seconds or scheduled_time is specified",
                "If neither is provided, ask the user when the task should run",
                "Schedule the task using kernel_schedule_task",
                "Confirm scheduling success to the user"
            ],
            required_tool_groups=["scheduling", "communication"],
            preconditions=[
                "Task description is understandable"
            ],
            postconditions=[
                "Task is scheduled and task_id is returned"
            ],
            rule_id="schedule_task_rule"
        )

        # 2. Long-running processing
        ruleset.add_rule(
            situation="long running operation",
            intent="process data or perform multi-step reasoning",
            instructions=[
                "Send an initial intermediate response indicating start",
                "Provide periodic status updates via kernel_send_intermediate",
                "If processing stalls or blocks, notify the user",
                "Send final confirmation when finished"
            ],
            required_tool_groups=["communication"],
            rule_id="long_running_feedback"
        )

        # 3. Memory injection
        ruleset.add_rule(
            situation="user preference or fact detected",
            intent="store memory",
            instructions=[
                "Evaluate whether the information is stable and reusable",
                "If importance or memory_type is unclear, ask the user for confirmation",
                "Inject memory using kernel_inject_memory",
                "Avoid storing temporary or speculative information"
            ],
            required_tool_groups=["memory", "communication"],
            preconditions=[
                "Information is explicitly stated or clearly implied by the user"
            ],
            postconditions=[
                "Memory entry is persisted"
            ],
            rule_id="memory_injection_rule"
        )

        # 4. Personalized response generation
        ruleset.add_rule(
            situation="response generation",
            intent="personalize answer",
            instructions=[
                "Retrieve user preferences via kernel_get_preferences",
                "Adapt tone, verbosity, and structure accordingly",
                "If preferences conflict with the request, ask the user which to prioritize"
            ],
            required_tool_groups=["memory", "communication"],
            rule_id="preference_application_rule"
        )

        # 5. Feedback handling
        ruleset.add_rule(
            situation="user feedback received",
            intent="learn from feedback",
            instructions=[
                "Interpret feedback sentiment and intent",
                "If feedback is unclear, ask the user to clarify",
                "Record feedback using kernel_record_feedback",
                "Adjust future behavior implicitly based on feedback score"
            ],
            required_tool_groups=["learning", "communication"],
            rule_id="feedback_learning_rule"
        )

    return ruleset



def auto_group_tools_by_name_pattern(
    tool_manager: 'ToolManager',
    rule_set: 'RuleSet',
    min_group_size: int = 2,
    separator: str = "_",
    ignore_prefixes: list[str] = None,
    ignore_suffixes: list[str] = None
) -> dict[str, list[str]]:
    """
    Automatically create tool groups based on repeating patterns in tool names.

    Analyzes all registered tools and groups them by common prefixes/patterns.
    Creates RuleSet tool groups for each discovered pattern.

    Args:
        tool_manager: ToolManager instance with registered tools
        rule_set: RuleSet instance for group registration
        min_group_size: Minimum tools needed to form a group (default: 2)
        separator: Separator character in tool names (default: "_")
        ignore_prefixes: Prefixes to ignore when grouping (e.g., ["mcp", "a2a"])
        ignore_suffixes: Suffixes to ignore (e.g., ["tool", "helper"])

    Returns:
        Dict mapping group_name -> list of tool names

    Example:
        Tools: discord_send, discord_edit, discord_delete, github_clone, github_push
        Result: {
            "discord_tools": ["discord_send", "discord_edit", "discord_delete"],
            "github_tools": ["github_clone", "github_push"]
        }
    """
    from collections import defaultdict

    ignore_prefixes = ignore_prefixes or ["mcp", "a2a", "local"]
    ignore_suffixes = ignore_suffixes or ["tool", "helper", "util", "utils"]

    # Get all tool names
    all_tools = tool_manager.list_names()

    if not all_tools:
        return {}

    # Step 1: Extract potential group prefixes from tool names
    prefix_tools: dict[str, list[str]] = defaultdict(list)

    for tool_name in all_tools:
        parts = tool_name.lower().split(separator)

        if len(parts) < 2:
            continue

        # Try different prefix lengths (1 part, 2 parts, etc.)
        for prefix_len in range(1, min(3, len(parts))):
            prefix_parts = parts[:prefix_len]

            # Skip ignored prefixes
            if prefix_parts[0] in ignore_prefixes:
                if len(prefix_parts) > 1:
                    prefix_parts = prefix_parts[1:]
                else:
                    continue

            # Skip if prefix is just an ignored suffix
            if prefix_parts[-1] in ignore_suffixes:
                continue

            prefix = separator.join(prefix_parts)

            # Only add if prefix is meaningful (not too short)
            if len(prefix) >= 2:
                prefix_tools[prefix].append(tool_name)

    # Step 2: Filter to groups with enough tools and resolve overlaps
    valid_groups: dict[str, list[str]] = {}
    assigned_tools: set[str] = set()

    # Sort by prefix length (longer = more specific) then by group size
    sorted_prefixes = sorted(
        prefix_tools.items(),
        key=lambda x: (-len(x[0].split(separator)), -len(x[1]))
    )

    for prefix, tools in sorted_prefixes:
        # Filter out already assigned tools
        available_tools = [t for t in tools if t not in assigned_tools]

        # Remove duplicates while preserving order
        unique_tools = list(dict.fromkeys(available_tools))

        if len(unique_tools) >= min_group_size:
            group_name = f"{prefix}_tools"
            valid_groups[group_name] = unique_tools
            assigned_tools.update(unique_tools)

    # Step 3: Register groups in RuleSet
    for group_name, tool_names in valid_groups.items():
        # Extract display name from group name
        display_parts = group_name.replace("_tools", "").split(separator)
        display_name = " ".join(word.capitalize() for word in display_parts) + " Tools"

        # Generate trigger keywords
        trigger_keywords = list(set(
            part for part in group_name.replace("_tools", "").split(separator)
            if part and len(part) > 1
        ))

        # Add common action words from tool names as triggers
        for tool_name in tool_names:
            tool_parts = tool_name.lower().split(separator)
            for part in tool_parts:
                if part not in trigger_keywords and len(part) > 2:
                    if part not in ignore_prefixes + ignore_suffixes:
                        trigger_keywords.append(part)

        # Limit trigger keywords
        trigger_keywords = trigger_keywords[:10]

        # Get tool descriptions for better group description
        tool_entries = [tool_manager.get(name) for name in tool_names]
        valid_entries = [e for e in tool_entries if e]

        # Build group description
        if valid_entries:
            sample_descs = [e.description[:50] for e in valid_entries[:3]]
            description = f"{len(tool_names)} tools: {', '.join(sample_descs)}..."
        else:
            description = f"Auto-grouped {len(tool_names)} tools with '{group_name.replace('_tools', '')}' pattern"

        # Register in RuleSet
        rule_set.register_tool_group(
            name=group_name,
            display_name=display_name,
            tool_names=tool_names,
            trigger_keywords=trigger_keywords,
            description=description,
            priority=5,
            icon="🔧",
            auto_generated=True
        )

        # Also update tool categories in ToolManager
        for tool_name in tool_names:
            entry = tool_manager.get(tool_name)
            if entry and group_name.replace("_tools", "") not in entry.category:
                entry.category.append(group_name.replace("_tools", ""))

    return valid_groups

"""
# Nach dem Laden aller Tools
groups = auto_group_tools_by_name_pattern(
    tool_manager=agent.tool_manager,
    rule_set=session.rule_set,
    min_group_size=2,
    separator="_"
)

print(f"Created {len(groups)} auto-groups:")
for group, tools in groups.items():
    print(f"  {group}: {len(tools)} tools")
"""
