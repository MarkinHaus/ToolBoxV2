"""
RL Training Agent Tools

Provides 4 tools for FlowAgent to manage RL training:
1. start_training - Start non-blocking training
2. stop_training - Stop training with auto-save and deploy
3. check_training_status - Check current training status
4. switch_model - Switch to a different trained model
"""

import asyncio
import threading
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum


class TrainingState(Enum):
    """Training state enum"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingSession:
    """Represents an active training session"""
    session_id: str
    model_name: str
    base_model: str
    method: str  # "grpo" or "kto"
    state: TrainingState = TrainingState.IDLE
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    error_message: Optional[str] = None
    output_dir: str = ""
    deployed_model_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "base_model": self.base_model,
            "method": self.method,
            "state": self.state.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_loss": self.current_loss,
            "error_message": self.error_message,
            "output_dir": self.output_dir,
            "deployed_model_name": self.deployed_model_name,
        }


class RLTrainingManager:
    """
    Singleton manager for RL training sessions.
    Handles non-blocking training, status tracking, and model switching.
    """

    _instance: Optional["RLTrainingManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.current_session: Optional[TrainingSession] = None
        self.session_history: list[TrainingSession] = []
        self._training_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._pipeline = None
        self._active_models: Dict[str, str] = {}  # name -> ollama_model_name

        # Storage path
        try:
            from toolboxv2 import get_app
            self.storage_path = Path(get_app().data_dir) / "rl_training"
        except:
            self.storage_path = Path.home() / ".toolbox" / "rl_training"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load session history from disk"""
        history_file = self.storage_path / "session_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    for item in data.get("sessions", []):
                        item["state"] = TrainingState(item["state"])
                        self.session_history.append(TrainingSession(**item))
                    self._active_models = data.get("active_models", {})
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")

    def _save_history(self):
        """Save session history to disk"""
        history_file = self.storage_path / "session_history.json"
        try:
            data = {
                "sessions": [s.to_dict() for s in self.session_history[-50:]],
                "active_models": self._active_models,
            }
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def start_training(
        self,
        model_name: str,
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        method: str = "grpo",
        agent_name: str = "default",
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
    ) -> Dict[str, Any]:
        """
        Start a non-blocking training session.

        Args:
            model_name: Name for the trained model
            base_model: HuggingFace base model to fine-tune
            method: Training method ("grpo" or "kto")
            agent_name: Agent name for data collection
            num_epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Dict with session_id and status
        """
        if self.current_session and self.current_session.state == TrainingState.RUNNING:
            return {
                "success": False,
                "error": "Training already in progress",
                "session_id": self.current_session.session_id,
            }

        session_id = self._generate_session_id()
        output_dir = str(self.storage_path / model_name)

        self.current_session = TrainingSession(
            session_id=session_id,
            model_name=model_name,
            base_model=base_model,
            method=method,
            state=TrainingState.STARTING,
            start_time=datetime.now().isoformat(),
            output_dir=output_dir,
        )

        self._stop_requested = False

        # Start training in background thread
        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(agent_name, num_epochs, learning_rate),
            daemon=True
        )
        self._training_thread.start()

        return {
            "success": True,
            "session_id": session_id,
            "message": f"Training started for {model_name} using {base_model}",
            "method": method,
            "output_dir": output_dir,
        }

    def _run_training(self, agent_name: str, num_epochs: int, learning_rate: float):
        """Background training execution"""
        try:
            from .training import TrainingPipeline, TrainingConfig

            self.current_session.state = TrainingState.RUNNING

            # Create pipeline
            self._pipeline = TrainingPipeline(
                agent_name=agent_name,
                base_model=self.current_session.base_model,
                output_dir=self.current_session.output_dir,
                method=self.current_session.method,
            )

            # Override config
            self._pipeline.training_config.num_epochs = num_epochs
            self._pipeline.training_config.learning_rate = learning_rate

            # Prepare data
            dataset = self._pipeline.prepare_data()
            self.current_session.total_steps = len(dataset) * num_epochs

            # Check for stop request
            if self._stop_requested:
                self.current_session.state = TrainingState.STOPPING
                self._finalize_training(save=True)
                return

            # Train
            self._pipeline.train(dataset)

            # Save model
            model_path = self._pipeline.save(merge_lora=True)

            self.current_session.state = TrainingState.COMPLETED
            self.current_session.end_time = datetime.now().isoformat()

            self.session_history.append(self.current_session)
            self._save_history()

        except Exception as e:
            import traceback
            self.current_session.state = TrainingState.FAILED
            self.current_session.error_message = str(e)
            self.current_session.end_time = datetime.now().isoformat()
            print(f"Training failed: {e}")
            traceback.print_exc()

    def _finalize_training(self, save: bool = True, deploy: bool = False):
        """Finalize training session"""
        if not self._pipeline:
            return

        try:
            if save and self._pipeline.trainer:
                model_path = self._pipeline.save(merge_lora=True)
                self.current_session.output_dir = model_path

                if deploy:
                    ollama_model = self._pipeline.deploy_to_ollama(
                        model_name=f"toolbox-{self.current_session.model_name}"
                    )
                    self.current_session.deployed_model_name = ollama_model
                    self._active_models[self.current_session.model_name] = ollama_model

            self.current_session.end_time = datetime.now().isoformat()
            self.session_history.append(self.current_session)
            self._save_history()

        except Exception as e:
            print(f"Error finalizing training: {e}")

    def stop_training(self, deploy: bool = True, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop current training, save model, and optionally deploy to Ollama.

        Args:
            deploy: Whether to deploy the model to Ollama after saving
            model_name: Custom name for the deployed model (default: toolbox-{model_name})

        Returns:
            Dict with status and deployed model info
        """
        if not self.current_session:
            return {
                "success": False,
                "error": "No training session active",
            }

        if self.current_session.state not in [TrainingState.RUNNING, TrainingState.STARTING]:
            return {
                "success": False,
                "error": f"Training is not running (state: {self.current_session.state.value})",
                "session_id": self.current_session.session_id,
            }

        self._stop_requested = True
        self.current_session.state = TrainingState.STOPPING

        # Wait for training thread to finish (with timeout)
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=30)

        # Finalize with save and deploy
        self._finalize_training(save=True, deploy=deploy)

        result = {
            "success": True,
            "session_id": self.current_session.session_id,
            "message": "Training stopped and model saved",
            "output_dir": self.current_session.output_dir,
        }

        if deploy and self.current_session.deployed_model_name:
            result["deployed_model"] = self.current_session.deployed_model_name
            result["message"] += f". Deployed as: {self.current_session.deployed_model_name}"

        self.current_session.state = TrainingState.COMPLETED
        return result

    def check_training_status(self) -> Dict[str, Any]:
        """
        Check the status of current or last training session.

        Returns:
            Dict with detailed training status
        """
        if self.current_session:
            session = self.current_session

            # Calculate progress
            progress = 0.0
            if session.total_steps > 0:
                progress = (session.current_step / session.total_steps) * 100

            # Calculate elapsed time
            elapsed = None
            if session.start_time:
                start = datetime.fromisoformat(session.start_time)
                elapsed = (datetime.now() - start).total_seconds()

            return {
                "has_active_session": session.state in [TrainingState.RUNNING, TrainingState.STARTING],
                "session": session.to_dict(),
                "progress_percent": round(progress, 2),
                "elapsed_seconds": elapsed,
                "is_running": session.state == TrainingState.RUNNING,
            }

        # Return last session from history
        if self.session_history:
            last_session = self.session_history[-1]
            return {
                "has_active_session": False,
                "session": last_session.to_dict(),
                "message": "No active training. Showing last session.",
            }

        return {
            "has_active_session": False,
            "session": None,
            "message": "No training sessions found.",
        }

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switch to a different trained model for inference.

        Args:
            model_name: Name of the model to switch to

        Returns:
            Dict with status and model info
        """
        # Check if model exists in active models
        if model_name in self._active_models:
            ollama_model = self._active_models[model_name]
            return {
                "success": True,
                "model_name": model_name,
                "ollama_model": ollama_model,
                "message": f"Switched to model: {ollama_model}",
            }

        # Check if model exists in session history
        for session in reversed(self.session_history):
            if session.model_name == model_name:
                if session.deployed_model_name:
                    self._active_models[model_name] = session.deployed_model_name
                    return {
                        "success": True,
                        "model_name": model_name,
                        "ollama_model": session.deployed_model_name,
                        "message": f"Switched to model: {session.deployed_model_name}",
                    }
                else:
                    # Model exists but not deployed - try to deploy
                    try:
                        from .export import OllamaDeployer, GGUFExporter

                        model_path = Path(session.output_dir) / "final"
                        if not model_path.exists():
                            model_path = Path(session.output_dir)

                        exporter = GGUFExporter(str(model_path))
                        gguf_path = exporter.convert(quantization="Q4_K_M")

                        deployer = OllamaDeployer()
                        ollama_model = deployer.create_model(
                            f"toolbox-{model_name}",
                            gguf_path
                        )

                        self._active_models[model_name] = ollama_model
                        self._save_history()

                        return {
                            "success": True,
                            "model_name": model_name,
                            "ollama_model": ollama_model,
                            "message": f"Model deployed and switched to: {ollama_model}",
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Failed to deploy model: {e}",
                            "model_path": str(session.output_dir),
                        }

        # List available models
        available = list(self._active_models.keys())
        available.extend([s.model_name for s in self.session_history if s.model_name not in available])

        return {
            "success": False,
            "error": f"Model '{model_name}' not found",
            "available_models": available,
        }

    def list_models(self) -> Dict[str, Any]:
        """List all available trained models"""
        models = []

        for session in self.session_history:
            models.append({
                "name": session.model_name,
                "base_model": session.base_model,
                "method": session.method,
                "state": session.state.value,
                "deployed": session.deployed_model_name is not None,
                "ollama_model": session.deployed_model_name,
                "trained_at": session.end_time or session.start_time,
            })

        return {
            "models": models,
            "active_models": self._active_models,
            "total": len(models),
        }


# ===== AGENT TOOL FUNCTIONS =====
# These are the functions that can be registered as agent tools

_manager: Optional[RLTrainingManager] = None


def _get_manager() -> RLTrainingManager:
    """Get or create the singleton manager"""
    global _manager
    if _manager is None:
        _manager = RLTrainingManager()
    return _manager


async def start_rl_training(
    model_name: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    method: str = "grpo",
    agent_name: str = "default",
    num_epochs: int = 3,
) -> str:
    """
    Start RL training for a new model (non-blocking).

    This starts training in the background and returns immediately.
    Use check_training_status to monitor progress.

    Args:
        model_name: Name for the trained model (e.g., "my-assistant-v1")
        base_model: HuggingFace base model to fine-tune (default: Qwen2.5-0.5B)
        method: Training method - "grpo" or "kto" (default: grpo)
        agent_name: Agent name for collecting training data (default: default)
        num_epochs: Number of training epochs (default: 3)

    Returns:
        Status message with session ID
    """
    manager = _get_manager()
    result = manager.start_training(
        model_name=model_name,
        base_model=base_model,
        method=method,
        agent_name=agent_name,
        num_epochs=num_epochs,
    )

    if result["success"]:
        return f"✓ Training started!\nSession ID: {result['session_id']}\nModel: {model_name}\nBase: {base_model}\nMethod: {method}"
    else:
        return f"✗ Failed to start training: {result.get('error', 'Unknown error')}"


async def stop_rl_training(deploy: bool = True) -> str:
    """
    Stop current RL training, save the model, and optionally deploy to Ollama.

    This will save the current training progress and create a usable model.
    If deploy=True, the model will be converted to GGUF and deployed to Ollama.

    Args:
        deploy: Whether to deploy the model to Ollama (default: True)

    Returns:
        Status message with saved model info
    """
    manager = _get_manager()
    result = manager.stop_training(deploy=deploy)

    if result["success"]:
        msg = f"✓ Training stopped and saved!\nSession: {result['session_id']}\nOutput: {result['output_dir']}"
        if result.get("deployed_model"):
            msg += f"\nDeployed as: {result['deployed_model']}"
        return msg
    else:
        return f"✗ Failed to stop training: {result.get('error', 'Unknown error')}"


async def check_training_status() -> str:
    """
    Check the status of current or last RL training session.

    Returns detailed information about training progress, including:
    - Current state (running, completed, failed)
    - Progress percentage
    - Elapsed time
    - Current loss

    Returns:
        Formatted status report
    """
    manager = _get_manager()
    result = manager.check_training_status()

    if result.get("session"):
        session = result["session"]
        lines = [
            "=" * 40,
            "RL Training Status",
            "=" * 40,
            f"Session: {session['session_id']}",
            f"Model: {session['model_name']}",
            f"Base: {session['base_model']}",
            f"Method: {session['method'].upper()}",
            f"State: {session['state']}",
        ]

        if result.get("is_running"):
            lines.append(f"Progress: {result.get('progress_percent', 0):.1f}%")
            lines.append(f"Step: {session['current_step']}/{session['total_steps']}")
            if result.get("elapsed_seconds"):
                elapsed = result["elapsed_seconds"]
                lines.append(f"Elapsed: {elapsed/60:.1f} minutes")

        if session.get("current_loss"):
            lines.append(f"Loss: {session['current_loss']:.4f}")

        if session.get("deployed_model_name"):
            lines.append(f"Deployed: {session['deployed_model_name']}")

        if session.get("error_message"):
            lines.append(f"Error: {session['error_message']}")

        lines.append("=" * 40)
        return "\n".join(lines)
    else:
        return result.get("message", "No training sessions found.")


async def switch_rl_model(model_name: str) -> str:
    """
    Switch to a different trained RL model for inference.

    This will activate a previously trained model. If the model
    hasn't been deployed to Ollama yet, it will be deployed automatically.

    Args:
        model_name: Name of the model to switch to

    Returns:
        Status message with model info
    """
    manager = _get_manager()
    result = manager.switch_model(model_name)

    if result["success"]:
        return f"✓ Switched to model: {result['ollama_model']}\nUse 'ollama run {result['ollama_model']}' to test"
    else:
        msg = f"✗ {result.get('error', 'Unknown error')}"
        if result.get("available_models"):
            msg += f"\nAvailable models: {', '.join(result['available_models'])}"
        return msg


async def list_rl_models() -> str:
    """
    List all available trained RL models.

    Shows all models that have been trained, including their
    training method, state, and whether they're deployed.

    Returns:
        Formatted list of models
    """
    manager = _get_manager()
    result = manager.list_models()

    if not result["models"]:
        return "No trained models found. Use start_rl_training to train a model."

    lines = [
        "=" * 50,
        "Available RL Models",
        "=" * 50,
    ]

    for model in result["models"]:
        status = "✓" if model["deployed"] else "○"
        lines.append(f"{status} {model['name']}")
        lines.append(f"   Base: {model['base_model']}")
        lines.append(f"   Method: {model['method'].upper()}")
        lines.append(f"   State: {model['state']}")
        if model["ollama_model"]:
            lines.append(f"   Ollama: {model['ollama_model']}")
        lines.append("")

    lines.append("=" * 50)
    lines.append(f"Total: {result['total']} models")

    return "\n".join(lines)


def get_rl_training_tools() -> list[tuple[Callable, str, str]]:
    """
    Get all RL training tools for agent registration.

    Returns:
        List of (function, name, description) tuples
    """
    return [
        (start_rl_training, "start_rl_training",
         "Start RL training for a new model (non-blocking). Returns immediately."),
        (stop_rl_training, "stop_rl_training",
         "Stop current training, save model, and deploy to Ollama."),
        (check_training_status, "check_training_status",
         "Check the status of current or last RL training session."),
        (switch_rl_model, "switch_rl_model",
         "Switch to a different trained RL model for inference."),
        (list_rl_models, "list_rl_models",
         "List all available trained RL models."),
    ]


def register_rl_tools(agent) -> None:
    """
    Register all RL training tools with a FlowAgent.

    Args:
        agent: FlowAgent instance to register tools with

    Example:
        from toolboxv2.mods.isaa.base.rl.agent_tools import register_rl_tools
        register_rl_tools(my_agent)
    """
    import asyncio

    for func, name, description in get_rl_training_tools():
        asyncio.create_task(agent.add_tool(func, name=name, description=description))

    print(f"Registered {len(get_rl_training_tools())} RL training tools")
