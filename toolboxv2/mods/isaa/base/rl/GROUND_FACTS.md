# ToolBoxV2 RL Training Pipeline - Ground Facts Documentation

## Overview

Complete RL training pipeline for FlowAgent models using GRPO/KTO with LoRA.
Designed for CPU-first training on Ryzen hardware with GPU acceleration when available.

**Location**: `toolboxv2/mods/isaa/base/rl/`

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RL Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐        │
│  │   Data      │───▶│   Dataset    │───▶│    Training     │        │
│  │ Collection  │    │   Builder    │    │   (GRPO/KTO)    │        │
│  └─────────────┘    └──────────────┘    └────────┬────────┘        │
│        │                   │                      │                 │
│        ▼                   ▼                      ▼                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐        │
│  │   Trace     │    │    Reward    │    │     Export      │        │
│  │  Collector  │───▶│   Functions  │    │  (GGUF/Ollama)  │        │
│  └─────────────┘    └──────────────┘    └─────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Modules

### 1. hardware_config.py
**Purpose**: Hardware detection and optimization profiles

**Key Components**:
- `HardwareConfig`: Dataclass with all hardware settings
- `detect_hardware()`: Auto-detect CPU, RAM, GPU
- `get_ryzen_optimized_config()`: Preset for Ryzen 9 5950X

**Hardware Profiles**:
- `RYZEN_OPTIMIZED`: Optimized for Ryzen 9 5950X (16 cores, 32 threads)
- `AUTO_DETECT`: Dynamic detection of available resources
- `GPU_ENABLED`: When CUDA/ROCm GPU is available
- `CPU_ONLY`: Fallback for CPU-only systems

**Derived Settings** (based on 40GB RAM):
```python
recommended_batch_size = 2
recommended_model_size = "1.5B"
lora_r = 16
lora_alpha = 32
num_generations = 6  # GRPO group size
use_bf16 = True  # Zen3 AVX2 support
```

### 2. data_collection.py
**Purpose**: Collect training traces from FlowAgent execution

**Key Components**:
- `ExecutionTrace`: Complete trace with tool calls, reasoning, outcomes
- `ToolCallTrace`: Individual tool call with success/failure
- `ReasoningStep`: Reasoning step from LLMReasonerNode
- `TraceCollector`: Hooks into agent to capture traces
- `CheckpointLoader`: Extracts traces from FlowAgent checkpoints

**Trace Data Captured**:
```python
ExecutionTrace:
    - user_query: str
    - tool_calls: list[ToolCallTrace]  # What tools were called
    - reasoning_steps: list[ReasoningStep]  # How agent reasoned
    - tasks_created/completed/failed: list[dict]
    - final_response: str
    - total_tokens_in/out: int
    - total_cost: float
    - label: Optional[bool]  # Manual label
    - reward_score: Optional[float]  # Computed reward
```

**Integration**:
```python
from toolboxv2.mods.isaa.base.rl import TraceCollector, hook_into_agent

collector = TraceCollector()
hook_into_agent(agent, collector)
# Now all agent runs are traced
```

### 3. reward_functions.py
**Purpose**: Verifiable rewards for RL training

**Reward Types**:

| Reward | Type | Weight | Description |
|--------|------|--------|-------------|
| `CodeExecutionReward` | Binary | 2.0 | Actually executes code and checks success |
| `SyntaxValidationReward` | Binary | 1.0 | AST parsing validation |
| `ToolSuccessReward` | Binary | 2.0 | Checks actual tool call outcomes |
| `TaskCompletionReward` | Binary | 1.5 | Task completion rate |
| `EfficiencyReward` | Soft | 0.5 | Token/call efficiency |
| `FormatComplianceReward` | Binary | 1.0 | No XML, plain text focus |
| `LearnedReward` | Soft | 1.0 | Learned from manual labels |

**Key Principle**: Rewards examine what agent ACTUALLY did, not just final output.

```python
# Example: CodeExecutionReward
- Extracts code blocks from response
- Actually executes Python/Shell code
- Returns binary success/failure
- Sandboxed with timeout
```

**RewardEngine**:
```python
engine = RewardEngine()
combined = engine.compute_combined(trace)  # Weighted average
normalized = engine.compute_for_group(traces)  # For GRPO
label = engine.get_binary_label(trace)  # For KTO
```

### 4. dataset_builder.py
**Purpose**: Build KTO/GRPO datasets from traces

**KTO Dataset Format**:
```python
{
    "prompt": "User: <query>",
    "completion": "<agent response>",
    "label": True/False  # Good/bad
}
```

**GRPO Dataset Format**:
```python
{
    "prompt": "User: <query>",
    "completions": ["response1", "response2", ...],
    "rewards": [0.8, -0.3, ...]  # Normalized within group
}
```

**Building Pipeline**:
```python
from toolboxv2.mods.isaa.base.rl import DatasetPipeline

pipeline = DatasetPipeline("agent_name")
kto_examples = pipeline.build_kto_dataset("kto_data.jsonl")
grpo_examples = pipeline.build_grpo_dataset("grpo_data.jsonl")
```

### 5. training.py
**Purpose**: LoRA-based GRPO/KTO training with TRL

**TrainingConfig** (from 40GB RAM):
```python
TrainingConfig(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    method="grpo",
    lora_r=16,
    lora_alpha=32,
    per_device_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=6,
    learning_rate=5e-5,
    num_epochs=3,
    use_bf16=True,
    gradient_checkpointing=True
)
```

**Training Flow**:
```python
trainer = RLTrainer(config)
trainer.setup()  # Load model + LoRA
trainer.train(dataset)  # GRPO or KTO
trainer.save_model(merge_lora=True)  # Merged weights
```

**Full Pipeline**:
```python
pipeline = TrainingPipeline(
    agent_name="my_agent",
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    method="grpo"
)
results = pipeline.run_full_pipeline(deploy_ollama=True)
```

### 6. export.py
**Purpose**: GGUF conversion and Ollama deployment

**GGUF Quantization Options**:
| Type | Bits | Use Case |
|------|------|----------|
| Q2_K | 2.5 | Very limited RAM |
| Q4_K_M | 4.5 | **Recommended** for general use |
| Q8_0 | 8.0 | Maximum quality |

**Export Flow**:
```python
exporter = GGUFExporter("./merged-model")
gguf_path = exporter.convert("Q4_K_M")

deployer = OllamaDeployer()
deployer.create_model("toolbox-agent", gguf_path)
# ollama run toolbox-agent
```

**Hosting Profiles**:

**Ryzen Optimized** (for 5950X):
```python
OllamaHostingProfile(
    num_parallel=4,
    num_ctx=4096,
    num_thread=14  # 16 - 2 for system
)
```

**Auto-Detect**:
- Checks CPU cores, RAM, GPU
- Adjusts parallel requests, context size
- Enables flash attention if GPU available

## Usage Examples

### Real-World Example: Extract Training Data from Saved Checkpoints

The most powerful feature is automatic extraction of training data from how users
actually interact with the agent. FlowAgent saves checkpoints with conversation
history, and CheckpointLoader extracts these as training examples.

```python
from toolboxv2.mods.isaa.base.rl import CheckpointLoader

# Discover all agents with saved checkpoints
loader = CheckpointLoader()  # No agent_name = discover all
agents = loader.discover_all_agents()
print(f"Found agents: {agents}")
# Output: ['DiscordKernelAssistant', 'TelegramKernelAssistant', 'Coder', 'self']

# Load traces from a specific agent
loader = CheckpointLoader(agent_name="DiscordKernelAssistant")
traces = loader.load_all_traces()
print(f"Extracted {len(traces)} conversation traces")

# Get comprehensive statistics
stats = loader.get_training_statistics()
print(f"Total traces: {stats['total_traces']}")
print(f"Unique sessions: {stats['unique_sessions']}")
print(f"Tools used: {stats['tools_list']}")

# Load traces from ALL agents at once
all_traces = loader.load_all_agents_traces()
for agent_name, agent_traces in all_traces.items():
    print(f"{agent_name}: {len(agent_traces)} traces")
```

**Checkpoint Data Structure** (what gets extracted):
```
AgentCheckpoint:
├── session_data: {session_id: {history: [{role, content}, ...]}}
│   └── Primary source: user/assistant conversation pairs
├── variable_scopes: {scope_name: {var_name: value}}
│   └── Contains reasoning results, delegation info
├── task_state: {task_id: {status, result, error}}
│   └── Task completion/failure info
├── agent_state: {tokens_in, tokens_out, cost, llm_calls}
│   └── Metrics for efficiency rewards
└── tool_capabilities: {tool_name: capability_info}
    └── Available tools for context
```

**Example extracted trace:**
```python
ExecutionTrace(
    session_id="268830485889810432",  # Discord user ID
    user_query="Hey bist du online?",
    final_response="Ja, ich bin online und bereit zu helfen!",
    tool_calls=[],  # Any tools called between query and response
    reasoning_steps=[ReasoningStep(step_type="final_result", ...)],
    total_tokens_in=150,
    total_tokens_out=45,
    total_cost=0.0002
)
```

### Complete Training Run

```python
from toolboxv2.mods.isaa.base.rl import TrainingPipeline

# Initialize
pipeline = TrainingPipeline(
    agent_name="isaa",
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    method="grpo"
)

# Run complete pipeline
results = pipeline.run_full_pipeline(deploy_ollama=True)

print(f"Model deployed: ollama run {results['ollama_model']}")
```

### Collecting Traces from Running Agent

```python
from toolboxv2.mods.isaa.base.Agent import FlowAgent
from toolboxv2.mods.isaa.base.rl import TraceCollector, hook_into_agent

# Create agent
agent = FlowAgent(amd)

# Hook trace collection
collector = TraceCollector()
hook_into_agent(agent, collector)

# Run agent normally - traces are captured
await agent.a_run("Write a Python function...")

# Get statistics
stats = collector.get_statistics()
print(f"Collected {stats['total']} traces")
```

### Manual Labeling Workflow

```python
from toolboxv2.mods.isaa.base.rl import TraceCollector

collector = TraceCollector()

# Get unlabeled traces
unlabeled = collector.get_unlabeled_traces(limit=10)

for trace in unlabeled:
    print(f"Query: {trace.user_query}")
    print(f"Response: {trace.final_response[:200]}...")
    print(f"Tools used: {[tc.tool_name for tc in trace.tool_calls]}")

    # Manual review
    label = input("Good? (y/n): ").lower() == "y"
    collector.label_trace(trace.trace_id, label)
```

### Custom Reward Functions

```python
from toolboxv2.mods.isaa.base.rl import BaseReward, RewardResult, RewardEngine

class CustomReward(BaseReward):
    name = "my_custom_reward"
    weight = 1.5

    def compute(self, trace) -> RewardResult:
        # Check something specific
        has_feature = "specific_pattern" in trace.final_response
        return RewardResult(
            score=1.0 if has_feature else 0.0,
            is_binary=True,
            details={"has_feature": has_feature}
        )

# Use with engine
engine = RewardEngine([
    CodeExecutionReward(),
    ToolSuccessReward(),
    CustomReward()
])
```

## Dependencies

**Required**:
```
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
datasets>=2.18.0
torch>=2.0.0
```

**Optional**:
```
psutil  # Hardware detection
llama-cpp-python  # Alternative GGUF conversion
```

**System Requirements**:
- llama.cpp (auto-installed)
- Ollama (https://ollama.ai)
- Python 3.10+

## File Paths

**Default Locations** (via `get_app().data_dir`):
- Traces: `{data_dir}/rl_traces/YYYY-MM-DD/*.json`
- Checkpoints: `{data_dir}/Agents/checkpoint/{agent_name}/*.pkl`
- Training Output: `{data_dir}/rl_training/{agent_name}/`
- Models: `{data_dir}/models/`

## Hardware Optimization Notes

**Ryzen 9 5950X + 40GB RAM**:
- Batch size 2, gradient accumulation 4 = effective batch 8
- LoRA r=16, alpha=32 for good adaptation
- BF16 enabled (Zen3 AVX2)
- 6 generations per GRPO group
- Context length 2048 for training

**GPU Acceleration** (if available):
- Automatic detection via CUDA/ROCm
- FP16 instead of BF16
- Higher batch sizes
- Flash attention enabled

## Error Handling

**Common Issues**:

1. **Out of Memory**: Reduce `per_device_batch_size`, enable `gradient_checkpointing`
2. **llama.cpp not found**: Auto-installs on first run
3. **Ollama not responding**: Ensure `ollama serve` is running
4. **Conversion fails**: Check model architecture support in llama.cpp

## Next Steps for Development

1. **Auto-Labeler**: Train classifier on manual labels to auto-label new traces
2. **Curriculum Learning**: Start with simple tasks, increase difficulty
3. **Online GRPO**: Generate completions on-the-fly during training
4. **Multi-Agent**: Collect traces from multiple agent configurations
5. **Benchmark Integration**: Connect with existing bench/ module
