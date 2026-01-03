"""
Dataset Builder for KTO and GRPO Training

Converts ExecutionTraces into training datasets suitable for
TRL's KTOTrainer and GRPOTrainer.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

from .data_collection import ExecutionTrace, TraceCollector, CheckpointLoader
from .reward_functions import RewardEngine


@dataclass
class KTOExample:
    """Single example for KTO training"""
    prompt: str
    completion: str
    label: bool  # True = desirable, False = undesirable

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "label": self.label
        }


@dataclass
class GRPOExample:
    """Single example for GRPO training with multiple completions"""
    prompt: str
    completions: list[str]
    rewards: list[float]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completions": self.completions,
            "rewards": self.rewards
        }


class KTODatasetBuilder:
    """
    Builds KTO (Kahneman-Tversky Optimization) datasets from traces.

    KTO uses binary feedback (good/bad) rather than preference pairs.
    Better suited for FlowAgent where we have verifiable outcomes.
    """

    def __init__(
        self,
        reward_engine: Optional[RewardEngine] = None,
        reward_threshold: float = 0.6,
        system_prompt: str = ""
    ):
        """
        Args:
            reward_engine: For computing rewards (uses default if None)
            reward_threshold: Score threshold for positive label
            system_prompt: System prompt to prepend to all prompts
        """
        self.reward_engine = reward_engine or RewardEngine()
        self.reward_threshold = reward_threshold
        self.system_prompt = system_prompt

    def trace_to_example(self, trace: ExecutionTrace) -> KTOExample:
        """Convert single trace to KTO example"""

        # Build prompt with context
        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)
        prompt_parts.append(f"User: {trace.user_query}")

        prompt = "\n\n".join(prompt_parts)

        # Completion is the agent's response
        completion = trace.final_response

        # Determine label
        if trace.label is not None:
            # Use manual label if available
            label = trace.label
        else:
            # Compute from rewards
            label = self.reward_engine.get_binary_label(trace, self.reward_threshold)

        return KTOExample(
            prompt=prompt,
            completion=completion,
            label=label
        )

    def build_dataset(
        self,
        traces: list[ExecutionTrace],
        balance: bool = True,
        max_examples: int = None
    ) -> list[KTOExample]:
        """
        Build KTO dataset from traces.

        Args:
            traces: List of ExecutionTrace objects
            balance: Balance positive/negative examples
            max_examples: Maximum total examples

        Returns:
            List of KTOExample objects
        """
        examples = [self.trace_to_example(t) for t in traces]

        if balance:
            positives = [e for e in examples if e.label]
            negatives = [e for e in examples if not e.label]

            min_count = min(len(positives), len(negatives))
            if min_count > 0:
                random.shuffle(positives)
                random.shuffle(negatives)
                examples = positives[:min_count] + negatives[:min_count]

        random.shuffle(examples)

        if max_examples:
            examples = examples[:max_examples]

        return examples

    def build_from_collector(
        self,
        collector: TraceCollector,
        include_unlabeled: bool = True,
        **kwargs
    ) -> list[KTOExample]:
        """Build dataset from TraceCollector"""
        traces = collector.load_traces(labeled_only=not include_unlabeled)
        return self.build_dataset(traces, **kwargs)

    def build_from_checkpoints(
        self,
        loader: CheckpointLoader,
        **kwargs
    ) -> list[KTOExample]:
        """Build dataset from checkpoints"""
        traces = loader.load_all_traces()
        return self.build_dataset(traces, **kwargs)

    def save_dataset(
        self,
        examples: list[KTOExample],
        output_path: str,
        format: str = "jsonl"
    ):
        """
        Save dataset to file.

        Args:
            examples: KTO examples
            output_path: Output file path
            format: "jsonl" or "json"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([ex.to_dict() for ex in examples], f, indent=2, ensure_ascii=False)

        print(f"Saved {len(examples)} KTO examples to {output_path}")

    def load_dataset(self, input_path: str) -> list[KTOExample]:
        """Load dataset from file"""
        input_path = Path(input_path)

        if input_path.suffix == ".jsonl":
            examples = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(KTOExample(**data))
            return examples
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [KTOExample(**d) for d in data]

    def to_hf_dataset(self, examples: list[KTOExample]):
        """Convert to HuggingFace Dataset format"""
        try:
            from datasets import Dataset

            data = {
                "prompt": [e.prompt for e in examples],
                "completion": [e.completion for e in examples],
                "label": [e.label for e in examples]
            }

            return Dataset.from_dict(data)
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

    def get_statistics(self, examples: list[KTOExample]) -> dict:
        """Get dataset statistics"""
        positives = sum(1 for e in examples if e.label)
        negatives = len(examples) - positives

        avg_prompt_len = sum(len(e.prompt) for e in examples) / len(examples) if examples else 0
        avg_completion_len = sum(len(e.completion) for e in examples) / len(examples) if examples else 0

        return {
            "total": len(examples),
            "positives": positives,
            "negatives": negatives,
            "balance_ratio": positives / negatives if negatives > 0 else float("inf"),
            "avg_prompt_length": avg_prompt_len,
            "avg_completion_length": avg_completion_len
        }


class GRPODatasetBuilder:
    """
    Builds GRPO (Group Relative Policy Optimization) datasets.

    GRPO requires multiple completions per prompt with rewards,
    enabling contrastive learning within groups.
    """

    def __init__(
        self,
        reward_engine: Optional[RewardEngine] = None,
        num_completions: int = 4,
        system_prompt: str = ""
    ):
        """
        Args:
            reward_engine: For computing rewards
            num_completions: Target completions per prompt
            system_prompt: System prompt for all examples
        """
        self.reward_engine = reward_engine or RewardEngine()
        self.num_completions = num_completions
        self.system_prompt = system_prompt

    def group_traces_by_query(
        self,
        traces: list[ExecutionTrace]
    ) -> dict[str, list[ExecutionTrace]]:
        """Group traces by similar queries"""
        groups = {}

        for trace in traces:
            # Normalize query for grouping
            key = self._normalize_query(trace.user_query)

            if key not in groups:
                groups[key] = []
            groups[key].append(trace)

        return groups

    def _normalize_query(self, query: str) -> str:
        """Normalize query for grouping similar ones"""
        # Simple normalization - can be enhanced with embeddings
        normalized = query.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized[:200]  # Limit length for key

    def build_example_from_group(
        self,
        prompt: str,
        traces: list[ExecutionTrace]
    ) -> Optional[GRPOExample]:
        """Build GRPO example from a group of traces with same prompt"""

        if len(traces) < 2:
            return None  # Need at least 2 for contrastive learning

        # Compute rewards for each trace
        completions = []
        rewards = []

        for trace in traces[:self.num_completions]:
            completions.append(trace.final_response)
            reward = self.reward_engine.compute_combined(trace)
            rewards.append(reward)

        # Normalize rewards within group (GRPO requirement)
        if len(rewards) > 1:
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = variance ** 0.5 if variance > 0 else 1.0
            rewards = [(r - mean) / std for r in rewards]

        # Build prompt
        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)
        prompt_parts.append(f"User: {prompt}")

        full_prompt = "\n\n".join(prompt_parts)

        return GRPOExample(
            prompt=full_prompt,
            completions=completions,
            rewards=rewards
        )

    def build_dataset(
        self,
        traces: list[ExecutionTrace],
        min_group_size: int = 2,
        max_examples: int = None,
        include_singles: bool = True
    ) -> list[GRPOExample]:
        """
        Build GRPO dataset from traces.

        Groups traces by query and creates examples with multiple
        completions per prompt.

        Args:
            traces: List of ExecutionTrace objects
            min_group_size: Minimum traces per group for contrastive learning
            max_examples: Maximum total examples
            include_singles: Include single traces with synthetic variations
        """
        # Group by query
        groups = self.group_traces_by_query(traces)

        examples = []
        for query, group_traces in groups.items():
            if len(group_traces) >= min_group_size:
                example = self.build_example_from_group(query, group_traces)
                if example:
                    examples.append(example)
            elif include_singles and len(group_traces) == 1:
                # Create example from single trace with synthetic variation
                example = self._build_single_trace_example(query, group_traces[0])
                if example:
                    examples.append(example)

        random.shuffle(examples)

        if max_examples:
            examples = examples[:max_examples]

        return examples

    def _build_single_trace_example(
        self,
        prompt: str,
        trace: ExecutionTrace
    ) -> Optional[GRPOExample]:
        """
        Build GRPO example from a single trace by creating synthetic variations.

        Creates a second completion by slightly modifying the original response
        to enable contrastive learning even with single-trace data.

        The original response gets a positive reward, the synthetic "worse"
        response gets a negative reward, enabling the model to learn preferences.
        """
        original_response = trace.final_response

        # Create a synthetic "worse" variation by truncating or adding noise
        # This allows GRPO to learn to prefer the original
        if len(original_response) > 100:
            # Truncate to create a worse version
            truncated = original_response[:len(original_response) // 2] + "..."
            completions = [original_response, truncated]
        else:
            # Add a generic worse response
            completions = [original_response, "I cannot help with that request."]

        # Compute rewards for original trace
        original_reward = self.reward_engine.compute_combined(trace)

        # Create synthetic trace for worse completion
        synthetic_trace = ExecutionTrace(
            user_query=trace.user_query,
            final_response=completions[1],
            tool_calls=[],  # No tool calls for synthetic
            tasks_completed=[]  # No tasks completed for synthetic
        )
        synthetic_reward = self.reward_engine.compute_combined(synthetic_trace)

        # Ensure there's a meaningful difference in rewards
        # If rewards are too similar, apply a penalty to the synthetic one
        if abs(original_reward - synthetic_reward) < 0.1:
            # Apply length-based penalty to synthetic (shorter = worse)
            length_ratio = len(completions[1]) / max(len(original_response), 1)
            synthetic_reward = synthetic_reward * length_ratio * 0.5

        rewards = [original_reward, synthetic_reward]

        # Normalize rewards to have mean 0 and std 1
        if len(rewards) > 1:
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = variance ** 0.5 if variance > 0 else 0.5  # Use 0.5 as default std
            if std > 0:
                rewards = [(r - mean) / std for r in rewards]
            else:
                # If no variance, assign fixed contrastive rewards
                rewards = [1.0, -1.0]

        # Build prompt
        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)
        prompt_parts.append(f"User: {prompt}")

        full_prompt = "\n\n".join(prompt_parts)

        return GRPOExample(
            prompt=full_prompt,
            completions=completions,
            rewards=rewards
        )

    def build_synthetic_groups(
        self,
        traces: list[ExecutionTrace],
        agent_generate_func: Callable,
        num_generations: int = 4
    ) -> list[GRPOExample]:
        """
        Build GRPO dataset by generating multiple completions per prompt.

        Uses the agent to generate additional completions for each
        unique query, enabling GRPO even with single-trace data.

        Args:
            traces: Existing traces (one per query)
            agent_generate_func: async func(prompt) -> str
            num_generations: Completions per prompt
        """
        import asyncio

        examples = []
        unique_queries = list(set(t.user_query for t in traces))

        async def generate_group(query: str) -> Optional[GRPOExample]:
            completions = []

            # Generate multiple completions
            for _ in range(num_generations):
                try:
                    completion = await agent_generate_func(query)
                    completions.append(completion)
                except Exception as e:
                    print(f"Generation failed for query: {e}")

            if len(completions) < 2:
                return None

            # Create synthetic traces for reward computation
            rewards = []
            for completion in completions:
                synthetic_trace = ExecutionTrace(
                    user_query=query,
                    final_response=completion
                )
                reward = self.reward_engine.compute_combined(synthetic_trace)
                rewards.append(reward)

            # Normalize
            if len(rewards) > 1:
                mean = sum(rewards) / len(rewards)
                std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
                std = std if std > 0 else 1.0
                rewards = [(r - mean) / std for r in rewards]

            prompt = f"{self.system_prompt}\n\nUser: {query}" if self.system_prompt else f"User: {query}"

            return GRPOExample(
                prompt=prompt,
                completions=completions,
                rewards=rewards
            )

        # Run generations
        loop = asyncio.get_event_loop()
        tasks = [generate_group(q) for q in unique_queries]
        results = loop.run_until_complete(asyncio.gather(*tasks))

        examples = [r for r in results if r is not None]
        return examples

    def save_dataset(
        self,
        examples: list[GRPOExample],
        output_path: str,
        format: str = "jsonl"
    ):
        """Save GRPO dataset to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([ex.to_dict() for ex in examples], f, indent=2, ensure_ascii=False)

        print(f"Saved {len(examples)} GRPO examples to {output_path}")

    def load_dataset(self, input_path: str) -> list[GRPOExample]:
        """Load GRPO dataset from file"""
        input_path = Path(input_path)

        if input_path.suffix == ".jsonl":
            examples = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(GRPOExample(**data))
            return examples
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [GRPOExample(**d) for d in data]

    def to_hf_dataset(self, examples: list[GRPOExample]):
        """Convert to HuggingFace Dataset format for TRL GRPOTrainer"""
        try:
            from datasets import Dataset

            # Flatten for GRPO format
            data = {
                "prompt": [e.prompt for e in examples],
                "completions": [e.completions for e in examples],
                "rewards": [e.rewards for e in examples]
            }

            return Dataset.from_dict(data)
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

    def get_statistics(self, examples: list[GRPOExample]) -> dict:
        """Get dataset statistics"""
        total_completions = sum(len(e.completions) for e in examples)
        avg_completions = total_completions / len(examples) if examples else 0

        all_rewards = [r for e in examples for r in e.rewards]
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

        return {
            "total_examples": len(examples),
            "total_completions": total_completions,
            "avg_completions_per_prompt": avg_completions,
            "avg_reward": avg_reward,
            "reward_range": (min(all_rewards), max(all_rewards)) if all_rewards else (0, 0)
        }


class DatasetPipeline:
    """
    Complete pipeline for building training datasets from FlowAgent.

    Combines trace collection, reward computation, and dataset building.
    """

    def __init__(
        self,
        agent_name: str,
        storage_path: Optional[str] = None,
        system_prompt: str = ""
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt

        # Initialize components
        self.trace_collector = TraceCollector(storage_path)
        self.checkpoint_loader = CheckpointLoader(agent_name)
        self.reward_engine = RewardEngine()

        self.kto_builder = KTODatasetBuilder(
            reward_engine=self.reward_engine,
            system_prompt=system_prompt
        )
        self.grpo_builder = GRPODatasetBuilder(
            reward_engine=self.reward_engine,
            system_prompt=system_prompt
        )

    def collect_all_traces(self) -> list[ExecutionTrace]:
        """Collect traces from all sources"""
        traces = []

        # From trace collector
        collector_traces = self.trace_collector.load_traces()
        traces.extend(collector_traces)

        # From checkpoints
        checkpoint_traces = self.checkpoint_loader.load_all_traces(deduplicate=True)
        traces.extend(checkpoint_traces)

        # Deduplicate
        seen_ids = set()
        unique_traces = []
        for trace in traces:
            if trace.trace_id not in seen_ids:
                seen_ids.add(trace.trace_id)
                unique_traces.append(trace)

        print(f"Collected {len(unique_traces)} unique traces")
        return unique_traces

    def build_kto_dataset(self, output_path: str, **kwargs) -> list[KTOExample]:
        """Build and save KTO dataset"""
        traces = self.collect_all_traces()
        examples = self.kto_builder.build_dataset(traces, **kwargs)
        self.kto_builder.save_dataset(examples, output_path)

        stats = self.kto_builder.get_statistics(examples)
        print(f"KTO Dataset: {stats}")

        return examples

    def build_grpo_dataset(self, output_path: str, **kwargs) -> list[GRPOExample]:
        """Build and save GRPO dataset"""
        traces = self.collect_all_traces()
        examples = self.grpo_builder.build_dataset(traces, **kwargs)
        self.grpo_builder.save_dataset(examples, output_path)

        stats = self.grpo_builder.get_statistics(examples)
        print(f"GRPO Dataset: {stats}")

        return examples

    def get_unlabeled_for_review(self, limit: int = 50) -> list[ExecutionTrace]:
        """Get traces that need manual review"""
        return self.trace_collector.get_unlabeled_traces(limit)

    def label_trace(self, trace_id: str, label: bool, notes: str = ""):
        """Apply manual label"""
        self.trace_collector.label_trace(trace_id, label, notes)

    def get_pipeline_statistics(self) -> dict:
        """Get comprehensive pipeline statistics"""
        collector_stats = self.trace_collector.get_statistics()
        checkpoints = self.checkpoint_loader.list_checkpoints()

        return {
            "collector": collector_stats,
            "checkpoints": {
                "count": len(checkpoints),
                "total_size_mb": sum(c["size_mb"] for c in checkpoints)
            }
        }
