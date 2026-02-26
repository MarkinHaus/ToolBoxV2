"""
FlowAgent V2 - Production-ready Agent System

Components:
- FlowAgent: Main agent class
- AgentSession: Session-isolated context
- SessionManager: Session lifecycle
- ToolManager: Unified tool registry
- CheckpointManager: State persistence
- BindManager: Agent-to-agent binding
- RuleSet: Dynamic skill/behavior system
- ExecutionEngine: MAKER/RLM orchestration

Author: FlowAgent V2
"""

## Core agent
#from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

## Session management
#from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSession,AgentSessionV2
#from toolboxv2.mods.isaa.base.Agent.vfs_v2 import  VirtualFileSystemV2, VirtualFileSystem, VFSFile
#from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager

## Tool management
#from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager, ToolEntry

## Checkpoint
#from toolboxv2.mods.isaa.base.Agent.checkpoint_manager import CheckpointManager, AgentCheckpoint

## Binding
#from toolboxv2.mods.isaa.base.Agent.bind_manager import BindManager, BindConfig, SyncEntry

## Execution Engine
#from toolboxv2.mods.isaa.base.Agent.execution_engine import (
#    ExecutionEngine,
#)

## RuleSet
#from toolboxv2.mods.isaa.base.Agent.rule_set import (
#    RuleSet,
#    RuleResult,
#    SituationRule,
#    ToolGroup,
#    LearnedPattern,
#    create_default_ruleset
#)

## Docker & Execution
#from toolboxv2.mods.isaa.base.Agent.docker_vfs import (
#    DockerConfig,
#    DockerVFS,
#    CommandResult,
#    create_docker_vfs_tool,
#)

## Code Execution (minimal)
#from toolboxv2.mods.isaa.base.Agent.executors import (
#    LocalCodeExecutor,
#    DockerCodeExecutor,
#    DockerConfig as CodeExecutorDockerConfig,
#    create_local_code_exec_tool,
#    create_docker_code_exec_tool,
#    register_code_exec_tools,
#    AgentToolProxy,
#)

## Types (from existing types.py)
#from toolboxv2.mods.isaa.base.Agent.types import (
#    AgentModelData,
#    PersonaConfig,
#    FormatConfig,
#    ResponseFormat,
#    TextLength,
#    ProgressEvent,
#    ProgressTracker,
#    NodeStatus,
#    Task,
#    TaskPlan,
#    LLMTask,
#    ToolTask,
#    DecisionTask,
#    CheckpointConfig,
#)
"""
FlowAgent V2 - Lazy-Loading __init__.py

Alle Imports werden erst bei erstem Zugriff geladen.
Spart ~3s Startup (execution_engine → litellm → SSL chain).
"""

import importlib

# Mapping: ExportName → (submodule, attribute)
_LAZY_MAP = {
    # Core
    'FlowAgent': ('.flow_agent', 'FlowAgent'),

    # Session
    'AgentSession': ('.agent_session_v2', 'AgentSession'),
    'AgentSessionV2': ('.agent_session_v2', 'AgentSessionV2'),
    'VirtualFileSystemV2': ('.vfs_v2', 'VirtualFileSystemV2'),
    'VirtualFileSystem': ('.vfs_v2', 'VirtualFileSystem'),
    'VFSFile': ('.vfs_v2', 'VFSFile'),
    'SessionManager': ('.session_manager', 'SessionManager'),

    # Tools
    'ToolManager': ('.tool_manager', 'ToolManager'),
    'ToolEntry': ('.tool_manager', 'ToolEntry'),

    # Checkpoint
    'CheckpointManager': ('.checkpoint_manager', 'CheckpointManager'),
    'AgentCheckpoint': ('.checkpoint_manager', 'AgentCheckpoint'),

    # Binding
    'BindManager': ('.bind_manager', 'BindManager'),
    'BindConfig': ('.bind_manager', 'BindConfig'),
    'SyncEntry': ('.bind_manager', 'SyncEntry'),

    # Execution Engine
    'ExecutionEngine': ('.execution_engine', 'ExecutionEngine'),

    # RuleSet
    'RuleSet': ('.rule_set', 'RuleSet'),
    'RuleResult': ('.rule_set', 'RuleResult'),
    'SituationRule': ('.rule_set', 'SituationRule'),
    'ToolGroup': ('.rule_set', 'ToolGroup'),
    'LearnedPattern': ('.rule_set', 'LearnedPattern'),
    'create_default_ruleset': ('.rule_set', 'create_default_ruleset'),

    # Docker & Execution
    'DockerConfig': ('.docker_vfs', 'DockerConfig'),
    'DockerVFS': ('.docker_vfs', 'DockerVFS'),
    'CommandResult': ('.docker_vfs', 'CommandResult'),
    'create_docker_vfs_tool': ('.docker_vfs', 'create_docker_vfs_tool'),

    # Code Execution
    'LocalCodeExecutor': ('.executors', 'LocalCodeExecutor'),
    'DockerCodeExecutor': ('.executors', 'DockerCodeExecutor'),
    'CodeExecutorDockerConfig': ('.executors', 'DockerConfig'),
    'create_local_code_exec_tool': ('.executors', 'create_local_code_exec_tool'),
    'create_docker_code_exec_tool': ('.executors', 'create_docker_code_exec_tool'),
    'register_code_exec_tools': ('.executors', 'register_code_exec_tools'),
    'AgentToolProxy': ('.executors', 'AgentToolProxy'),

    # Types
    'AgentModelData': ('.types', 'AgentModelData'),
    'PersonaConfig': ('.types', 'PersonaConfig'),
    'FormatConfig': ('.types', 'FormatConfig'),
    'ResponseFormat': ('.types', 'ResponseFormat'),
    'TextLength': ('.types', 'TextLength'),
    'ProgressEvent': ('.types', 'ProgressEvent'),
    'ProgressTracker': ('.types', 'ProgressTracker'),
    'NodeStatus': ('.types', 'NodeStatus'),
    'Task': ('.types', 'Task'),
    'TaskPlan': ('.types', 'TaskPlan'),
    'LLMTask': ('.types', 'LLMTask'),
    'ToolTask': ('.types', 'ToolTask'),
    'DecisionTask': ('.types', 'DecisionTask'),
    'CheckpointConfig': ('.types', 'CheckpointConfig'),
}

# Cache für bereits geladene Module
_loaded = {}


def __getattr__(name):
    if name in _LAZY_MAP:
        submodule, attr = _LAZY_MAP[name]
        if submodule not in _loaded:
            _loaded[submodule] = importlib.import_module(submodule, __package__)
        val = getattr(_loaded[submodule], attr)
        globals()[name] = val  # Cache im Modul-Namespace
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_MAP.keys()) + list(globals().keys())


__all__ = list(_LAZY_MAP.keys())
"""__all__ = [
    # Core
    'FlowAgent',

    # Session
    'AgentSession',
    'SessionManager',
    'VirtualFileSystem',
    'VFSFile',

    # Tools
    'ToolManager',
    'ToolEntry',

    # Checkpoint
    'CheckpointManager',
    'AgentCheckpoint',

    # Binding
    'BindManager',
    'BindConfig',
    'SyncEntry',

    # Execution Engine
    'ExecutionEngine',

    # RuleSet
    'RuleSet',
    'RuleResult',
    'SituationRule',
    'ToolGroup',
    'LearnedPattern',
    'create_default_ruleset',

    # Docker & Execution (VFS)
    'DockerConfig',
    'DockerVFS',
    'CommandResult',
    'create_docker_vfs_tool',

    # Code Execution
    'LocalCodeExecutor',
    'DockerCodeExecutor',
    'CodeExecutorDockerConfig',
    'create_local_code_exec_tool',
    'create_docker_code_exec_tool',
    'register_code_exec_tools',
    'AgentToolProxy',

    # Types
    'AgentModelData',
    'PersonaConfig',
    'FormatConfig',
    'ResponseFormat',
    'TextLength',
    'ProgressEvent',
    'ProgressTracker',
    'NodeStatus',
    'Task',
    'TaskPlan',
    'LLMTask',
    'ToolTask',
    'DecisionTask',
    'CheckpointConfig',
]"""
