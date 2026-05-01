"""
Dreamer V3 — Agent-Based Meta-Learning

Replaces the batch pipeline with a standalone FlowAgent
that uses Tools, Sub-Agents, and the Narrator system.
"""

from toolboxv2.mods.isaa.base.dreamer.harvest import RunRecord, parse_log, filter_records, get_cutoff, harvest_from_vfs
from toolboxv2.mods.isaa.base.dreamer.tools import (
    get_all_dream_tool_definitions,
    get_dream_tool_names,
    calculate_sub_agent_budget,
)
from toolboxv2.mods.isaa.base.dreamer.agent import (
    create_dreamer_agent_config,
    build_dream_query,
    prepare_dreamer_vfs,
)
from toolboxv2.mods.isaa.base.dreamer.prompts import (
    build_dreamer_system_prompt,
    build_cluster_analysis_task,
)
from  toolboxv2.mods.isaa.base.dreamer.tool_handler import DreamerToolHandler
