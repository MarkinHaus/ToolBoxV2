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

# Core agent
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

# Session management
from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSession,AgentSessionV2
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import  VirtualFileSystemV2, VirtualFileSystem, VFSFile
from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager

# Tool management
from toolboxv2.mods.isaa.base.Agent.tool_manager import ToolManager, ToolEntry

# Checkpoint
from toolboxv2.mods.isaa.base.Agent.checkpoint_manager import CheckpointManager, AgentCheckpoint

# Binding
from toolboxv2.mods.isaa.base.Agent.bind_manager import BindManager, BindConfig, SyncEntry

# Execution Engine
from toolboxv2.mods.isaa.base.Agent.execution_engine import (
    ExecutionEngine,
)

# RuleSet
from toolboxv2.mods.isaa.base.Agent.rule_set import (
    RuleSet,
    RuleResult,
    SituationRule,
    ToolGroup,
    LearnedPattern,
    create_default_ruleset
)

# Docker & Execution
from toolboxv2.mods.isaa.base.Agent.docker_vfs import (
    DockerConfig,
    DockerVFS,
    CommandResult,
    create_docker_vfs_tool,
)

# Code Execution (minimal)
from toolboxv2.mods.isaa.base.Agent.code_executor import (
    LocalCodeExecutor,
    DockerCodeExecutor,
    DockerConfig as CodeExecutorDockerConfig,
    create_local_code_exec_tool,
    create_docker_code_exec_tool,
    register_code_exec_tools,
    AgentToolProxy,
)

# Types (from existing types.py)
from toolboxv2.mods.isaa.base.Agent.types import (
    AgentModelData,
    PersonaConfig,
    FormatConfig,
    ResponseFormat,
    TextLength,
    ProgressEvent,
    ProgressTracker,
    NodeStatus,
    Task,
    TaskPlan,
    LLMTask,
    ToolTask,
    DecisionTask,
    CheckpointConfig,
)

__all__ = [
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
]
