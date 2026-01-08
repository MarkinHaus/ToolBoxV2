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
from toolboxv2.mods.isaa.base.Agent.agent_session import AgentSession, VirtualFileSystem, VFSFile
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
    ExecutionState,
    ExecutionResult,
    ExecutionPhase,
    MicroagentConfig,
    MicroagentResult,
    IntentClassification,
    CategorySelection,
    ToolSelection,
    TaskDecomposition,
    ThoughtAction,
    ValidationResult,
    VFS_TOOLS_LITELLM,
    REACT_SYSTEM_PROMPT,
    MICROAGENT_SYSTEM_PROMPT,
)

# Voice Streaming
from toolboxv2.mods.isaa.base.Agent.voice_stream import (
    VoiceStreamEngine,
    BackgroundTaskManager,
    BackgroundTask,
    QueryComplexity,
    CapabilityDetection,
    voice_stream,
    VOICE_PHRASES,
)

# Default Tools
from toolboxv2.mods.isaa.base.Agent.default_tools import (
    DefaultToolsHandler,
    DefaultToolDef,
    DefaultToolCategory,
    DEFAULT_TOOLS_DEFS,
    get_default_tools_litellm,
    get_default_tool_names,
    is_default_tool,
    create_default_tools_handler,
)

# Execution Engine Patch (optional import)
try:
    from toolboxv2.mods.isaa.base.Agent.execution_engine_patch import (
        patch_execution_engine,
        DefaultToolsMixin,
    )
except ImportError:
    patch_execution_engine = None
    DefaultToolsMixin = None

# RuleSet
from toolboxv2.mods.isaa.base.Agent.rule_set import (
    RuleSet,
    RuleResult,
    SituationRule,
    ToolGroup,
    LearnedPattern,
    create_default_ruleset
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
    'ExecutionState',
    'ExecutionResult',
    'ExecutionPhase',
    'MicroagentConfig',
    'MicroagentResult',
    'IntentClassification',
    'CategorySelection',
    'ToolSelection',
    'TaskDecomposition',
    'ThoughtAction',
    'ValidationResult',
    'VFS_TOOLS_LITELLM',
    'REACT_SYSTEM_PROMPT',
    'MICROAGENT_SYSTEM_PROMPT',
    
    # Voice Streaming
    'VoiceStreamEngine',
    'BackgroundTaskManager',
    'BackgroundTask',
    'QueryComplexity',
    'CapabilityDetection',
    'voice_stream',
    'VOICE_PHRASES',
    
    # Default Tools
    'DefaultToolsHandler',
    'DefaultToolDef',
    'DefaultToolCategory',
    'DEFAULT_TOOLS_DEFS',
    'get_default_tools_litellm',
    'get_default_tool_names',
    'is_default_tool',
    'create_default_tools_handler',
    'patch_execution_engine',
    'DefaultToolsMixin',
    
    # RuleSet
    'RuleSet',
    'RuleResult',
    'SituationRule',
    'ToolGroup',
    'LearnedPattern',
    'create_default_ruleset',
    
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
