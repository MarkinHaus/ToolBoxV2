"""
RL Training Module for FlowAgent

LoRA-based GRPO and KTO training with TRL library.
Supports CPU and GPU training with automatic hardware detection.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from pathlib import Path
from datetime import datetime


@dataclass
class TrainingConfig:
    """Configuration for RL training"""

    # Model settings
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./rl_output"

    # Training method
    method: str = "grpo"  # "grpo" or "kto"

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048

    # GRPO specific
    num_generations: int = 4
    max_completion_length: int = 512
    beta: float = 0.1  # KL penalty coefficient

    # KTO specific
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    # Hardware settings
    use_cpu: bool = True
    use_bf16: bool = False
    use_fp16: bool = False
    gradient_checkpointing: bool = True

    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50

    # Callbacks
    early_stopping_patience: int = 3

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "method": self.method,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "beta": self.beta,
            "use_cpu": self.use_cpu,
            "use_bf16": self.use_bf16,
            "use_fp16": self.use_fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    @classmethod
    def from_hardware_config(cls, hw_config, **overrides) -> "TrainingConfig":
        """Create TrainingConfig from HardwareConfig"""
        config = cls(
            lora_r=hw_config.lora_r,
            lora_alpha=hw_config.lora_alpha,
            per_device_batch_size=hw_config.recommended_batch_size,
            gradient_accumulation_steps=max(1, 8 // hw_config.recommended_batch_size),
            num_generations=hw_config.num_generations,
            use_cpu=not hw_config.has_gpu,
            use_bf16=hw_config.use_bf16,
            use_fp16=hw_config.use_fp16,
            gradient_checkpointing=hw_config.gradient_checkpointing,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def save(self, path: str):
        """Save config to JSON"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class RLTrainer:
    """
    Main trainer class for GRPO/KTO training with LoRA.

    Handles the complete training lifecycle:
    1. Load base model
    2. Apply LoRA adapters
    3. Train with GRPO or KTO
    4. Save merged model
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: TrainingConfig with all settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_stats = {}

        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(os.path.join(config.output_dir, "training_config.json"))

    def setup(self):
        """Setup model, tokenizer, and LoRA"""
        print(f"Setting up training for {self.config.base_model}")
        print(f"Method: {self.config.method.upper()}")
        print(f"Device: {'CPU' if self.config.use_cpu else 'GPU'}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise ImportError(
                "Required libraries not installed. Run:\n"
                "pip install transformers peft trl datasets --break-system-packages"
            ) from e

        # Determine device and dtype
        if self.config.use_cpu:
            device_map = "cpu"
            torch_dtype = "float32"
        else:
            device_map = "auto"
            if self.config.use_fp16:
                torch_dtype = "float16"
            elif self.config.use_bf16:
                torch_dtype = "bfloat16"
            else:
                torch_dtype = "float32"

        print(f"Loading model with dtype={torch_dtype}, device_map={device_map}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype_map.get(torch_dtype, "auto"),
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager"  # Avoid flash attention issues on CPU
        )

        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        return self

    def train_grpo(self, dataset, reward_funcs: list[Callable] = None):
        """
        Train with GRPO (Group Relative Policy Optimization).

        Args:
            dataset: HuggingFace Dataset with prompt, completions, rewards
            reward_funcs: Optional list of reward functions for online rewards
        """
        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError:
            raise ImportError("TRL library required: pip install trl --break-system-packages")

        print("Starting GRPO training...")

        # Create training arguments
        training_args = GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            bf16=self.config.use_bf16 and not self.config.use_cpu,
            fp16=self.config.use_fp16 and not self.config.use_cpu,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # GRPO specific
            num_generations=self.config.num_generations,
            max_completion_length=self.config.max_completion_length,
            beta=self.config.beta,
            # Disable vLLM for CPU training
            use_vllm=False,
        )

        # Create trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs or [],
        )

        # Train
        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time

        self.training_stats = {
            "method": "grpo",
            "training_time_seconds": training_time,
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "epochs": self.config.num_epochs,
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else None,
        }

        print(f"GRPO training completed in {training_time:.1f} seconds")

        return train_result

    def train_kto(self, dataset):
        """
        Train with KTO (Kahneman-Tversky Optimization).

        Args:
            dataset: HuggingFace Dataset with prompt, completion, label
        """
        try:
            from trl import KTOTrainer, KTOConfig
        except ImportError:
            raise ImportError("TRL library required: pip install trl --break-system-packages")

        print("Starting KTO training...")

        # Create training arguments
        training_args = KTOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            bf16=self.config.use_bf16 and not self.config.use_cpu,
            fp16=self.config.use_fp16 and not self.config.use_cpu,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # KTO specific
            max_length=self.config.max_seq_length,
            max_completion_length=self.config.max_completion_length,
            desirable_weight=self.config.desirable_weight,
            undesirable_weight=self.config.undesirable_weight,
            beta=self.config.beta,
        )

        # Create trainer
        self.trainer = KTOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        # Train
        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time

        self.training_stats = {
            "method": "kto",
            "training_time_seconds": training_time,
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "epochs": self.config.num_epochs,
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else None,
        }

        print(f"KTO training completed in {training_time:.1f} seconds")

        return train_result

    def train(self, dataset, reward_funcs: list[Callable] = None):
        """
        Train with configured method.

        Args:
            dataset: Training dataset
            reward_funcs: Reward functions for GRPO
        """
        if self.model is None:
            self.setup()

        if self.config.method == "grpo":
            return self.train_grpo(dataset, reward_funcs)
        elif self.config.method == "kto":
            return self.train_kto(dataset)
        else:
            raise ValueError(f"Unknown training method: {self.config.method}")

    def save_model(self, output_path: Optional[str] = None, merge_lora: bool = True):
        """
        Save trained model.

        Args:
            output_path: Output directory (default: config.output_dir/final)
            merge_lora: Merge LoRA weights into base model
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "final")

        os.makedirs(output_path, exist_ok=True)

        if merge_lora:
            print("Merging LoRA weights...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
            print(f"Merged model saved to {output_path}")
        else:
            # Save LoRA adapter only
            self.model.save_pretrained(output_path)
            print(f"LoRA adapter saved to {output_path}")

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        # Save training stats
        stats_path = os.path.join(output_path, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.training_stats, f, indent=2)

        return output_path

    def evaluate(self, eval_dataset, metrics: list[str] = None) -> dict:
        """
        Evaluate model on dataset.

        Args:
            eval_dataset: Evaluation dataset
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation results
        """
        if self.trainer is None:
            raise ValueError("No trainer available. Run training first.")

        print("Running evaluation...")
        results = self.trainer.evaluate(eval_dataset)

        return results

    def get_training_summary(self) -> str:
        """Get human-readable training summary"""
        lines = [
            "=" * 50,
            "Training Summary",
            "=" * 50,
            f"Method: {self.config.method.upper()}",
            f"Base Model: {self.config.base_model}",
            f"Output: {self.config.output_dir}",
            "-" * 50,
            f"LoRA r: {self.config.lora_r}, alpha: {self.config.lora_alpha}",
            f"Batch Size: {self.config.per_device_batch_size} x {self.config.gradient_accumulation_steps}",
            f"Learning Rate: {self.config.learning_rate}",
            f"Epochs: {self.config.num_epochs}",
        ]

        if self.training_stats:
            lines.extend([
                "-" * 50,
                "Results:",
                f"Training Time: {self.training_stats.get('training_time_seconds', 0):.1f}s",
                f"Final Loss: {self.training_stats.get('train_loss', 'N/A')}",
                f"Total Steps: {self.training_stats.get('total_steps', 'N/A')}",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)


class TrainingPipeline:
    """
    Complete training pipeline from traces to trained model.

    Combines data collection, dataset building, and training.
    """

    def __init__(
        self,
        agent_name: str,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        output_dir: str = None,
        method: str = "grpo"
    ):
        from .hardware_config import detect_hardware
        from .dataset_builder import DatasetPipeline

        self.agent_name = agent_name
        self.base_model = base_model
        self.method = method

        # Detect hardware
        self.hw_config = detect_hardware()
        print(self.hw_config.summary())

        # Setup paths
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            try:
                from toolboxv2 import get_app
                self.output_dir = Path(get_app().data_dir) / "rl_training" / agent_name
            except:
                self.output_dir = Path.home() / ".toolbox" / "rl_training" / agent_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline components
        self.data_pipeline = DatasetPipeline(agent_name)

        # Training config from hardware
        self.training_config = TrainingConfig.from_hardware_config(
            self.hw_config,
            base_model=base_model,
            output_dir=str(self.output_dir),
            method=method
        )

        self.trainer = None

    def prepare_data(self, min_examples: int = 2) -> Any:
        """
        Prepare training dataset from traces.

        Args:
            min_examples: Minimum number of examples required for training

        Returns:
            HuggingFace Dataset ready for training

        Raises:
            ValueError: If not enough training examples are available
        """
        print("Preparing training data...")

        dataset_path = self.output_dir / f"{self.method}_dataset.jsonl"

        if self.method == "grpo":
            examples = self.data_pipeline.build_grpo_dataset(str(dataset_path))
            if not examples:
                raise ValueError(
                    f"No GRPO training examples could be built. "
                    f"GRPO requires traces with similar queries or single traces with synthetic variations. "
                    f"Try using method='kto' instead, or collect more traces."
                )
            hf_dataset = self.data_pipeline.grpo_builder.to_hf_dataset(examples)
        else:
            examples = self.data_pipeline.build_kto_dataset(str(dataset_path))
            if not examples:
                raise ValueError(
                    f"No KTO training examples could be built. "
                    f"Check that checkpoint data contains valid user-assistant conversation pairs."
                )
            hf_dataset = self.data_pipeline.kto_builder.to_hf_dataset(examples)

        if len(hf_dataset) < min_examples:
            raise ValueError(
                f"Only {len(hf_dataset)} training examples available, but {min_examples} required. "
                f"Collect more traces or lower min_examples (not recommended for quality training)."
            )

        print(f"Prepared {len(hf_dataset)} training examples")
        return hf_dataset

    def train(self, dataset=None, reward_funcs: list[Callable] = None, min_examples: int = 2):
        """
        Run training.

        Args:
            dataset: Pre-prepared dataset (optional, will prepare if None)
            reward_funcs: Reward functions for GRPO
            min_examples: Minimum examples required for training
        """
        if dataset is None:
            dataset = self.prepare_data(min_examples=min_examples)

        # Adjust training config for small datasets
        if len(dataset) < 10:
            print(f"Warning: Small dataset ({len(dataset)} examples). Adjusting training parameters...")
            # Reduce epochs and increase logging for small datasets
            self.training_config.num_epochs = min(self.training_config.num_epochs, 1)
            self.training_config.logging_steps = 1
            self.training_config.save_steps = max(1, len(dataset) // 2)

        self.trainer = RLTrainer(self.training_config)
        self.trainer.setup()

        result = self.trainer.train(dataset, reward_funcs)

        print(self.trainer.get_training_summary())
        return result

    def save(self, merge_lora: bool = True) -> str:
        """Save trained model"""
        if self.trainer is None:
            raise ValueError("No training completed")

        return self.trainer.save_model(merge_lora=merge_lora)

    def export_to_gguf(self, quantization: str = "Q4_K_M") -> str:
        """Export to GGUF format"""
        from .export import GGUFExporter

        model_path = self.output_dir / "final"
        exporter = GGUFExporter(str(model_path))

        return exporter.convert(quantization=quantization)

    def deploy_to_ollama(self, model_name: str = None) -> str:
        """Deploy to Ollama"""
        from .export import OllamaDeployer

        gguf_path = self.export_to_gguf()

        deployer = OllamaDeployer()
        model_name = model_name or f"toolbox-{self.agent_name}"

        return deployer.create_model(model_name, gguf_path)

    def run_full_pipeline(
        self,
        reward_funcs: list[Callable] = None,
        deploy_ollama: bool = True
    ) -> dict:
        """
        Run complete pipeline: data -> train -> export -> deploy
        """
        results = {
            "start_time": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "base_model": self.base_model,
            "method": self.method
        }

        try:
            # Prepare data
            dataset = self.prepare_data()
            results["dataset_size"] = len(dataset)

            # Train
            train_result = self.train(dataset, reward_funcs)
            results["training"] = self.trainer.training_stats

            # Save
            model_path = self.save(merge_lora=True)
            results["model_path"] = model_path

            # Export and deploy
            if deploy_ollama:
                ollama_model = self.deploy_to_ollama()
                results["ollama_model"] = ollama_model

            results["success"] = True
            results["end_time"] = datetime.now().isoformat()

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            import traceback
            results["traceback"] = traceback.format_exc()

        # Save results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
