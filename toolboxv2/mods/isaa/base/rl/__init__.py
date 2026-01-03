"""
ToolBoxV2 RL Training Pipeline for FlowAgent

Complete lifecycle for training local LLMs via GRPO/KTO with LoRA,
converting to GGUF, and deploying via Ollama.

Modules:
- hardware_config: Hardware detection and optimization profiles
- data_collection: Trace collection from FlowAgent checkpoints
- reward_functions: Verifiable rewards for code/tool execution
- dataset_builder: KTO/GRPO dataset preparation
- training: LoRA-based GRPO/KTO training
- export: GGUF conversion and Ollama deployment

Quick Start:
    from toolboxv2.mods.isaa.base.rl import TrainingPipeline

    pipeline = TrainingPipeline(
        agent_name="isaa",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        method="grpo"
    )
    results = pipeline.run_full_pipeline(deploy_ollama=True)
"""

from .hardware_config import (
    HardwareConfig,
    HardwareProfile,
    detect_hardware,
    get_ryzen_optimized_config,
)

from .data_collection import (
    ExecutionTrace,
    ToolCallTrace,
    ReasoningStep,
    TraceCollector,
    CheckpointLoader,
    hook_into_agent,
)

from .reward_functions import (
    BaseReward,
    RewardResult,
    RewardEngine,
    CodeExecutionReward,
    SyntaxValidationReward,
    ToolSuccessReward,
    TaskCompletionReward,
    EfficiencyReward,
    FormatComplianceReward,
    LearnedReward,
)

from .dataset_builder import (
    KTOExample,
    GRPOExample,
    KTODatasetBuilder,
    GRPODatasetBuilder,
    DatasetPipeline,
)

from .training import (
    TrainingConfig,
    RLTrainer,
    TrainingPipeline,
)

from .export import (
    GGUFExporter,
    GGUFQuantization,
    OllamaDeployer,
    OllamaHostingProfile,
    ExportPipeline,
    quick_export,
)

from .agent_tools import (
    RLTrainingManager,
    TrainingSession,
    TrainingState,
    start_rl_training,
    stop_rl_training,
    check_training_status,
    switch_rl_model,
    list_rl_models,
    get_rl_training_tools,
    register_rl_tools,
)

__all__ = [
    # Hardware
    "HardwareConfig",
    "HardwareProfile",
    "detect_hardware",
    "get_ryzen_optimized_config",
    # Data Collection
    "ExecutionTrace",
    "ToolCallTrace",
    "ReasoningStep",
    "TraceCollector",
    "CheckpointLoader",
    "hook_into_agent",
    # Rewards
    "BaseReward",
    "RewardResult",
    "RewardEngine",
    "CodeExecutionReward",
    "SyntaxValidationReward",
    "ToolSuccessReward",
    "TaskCompletionReward",
    "EfficiencyReward",
    "FormatComplianceReward",
    "LearnedReward",
    # Dataset
    "KTOExample",
    "GRPOExample",
    "KTODatasetBuilder",
    "GRPODatasetBuilder",
    "DatasetPipeline",
    # Training
    "TrainingConfig",
    "RLTrainer",
    "TrainingPipeline",
    # Export
    "GGUFExporter",
    "GGUFQuantization",
    "OllamaDeployer",
    "OllamaHostingProfile",
    "ExportPipeline",
    "quick_export",
    # Agent Tools
    "RLTrainingManager",
    "TrainingSession",
    "TrainingState",
    "start_rl_training",
    "stop_rl_training",
    "check_training_status",
    "switch_rl_model",
    "list_rl_models",
    "get_rl_training_tools",
    "register_rl_tools",
]

__version__ = "1.0.0"
