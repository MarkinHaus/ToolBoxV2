"""
Hardware Configuration and Detection for RL Training

Detects system capabilities (CPU, RAM, GPU) and provides
optimized training configurations for Ryzen and auto-detection modes.
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class HardwareProfile(Enum):
    RYZEN_OPTIMIZED = "ryzen_optimized"
    AUTO_DETECT = "auto_detect"
    GPU_ENABLED = "gpu_enabled"
    CPU_ONLY = "cpu_only"


@dataclass
class HardwareConfig:
    """Hardware configuration for training optimization"""
    
    # CPU Info
    cpu_name: str = ""
    cpu_cores: int = 1
    cpu_threads: int = 1
    has_avx2: bool = False
    has_avx512: bool = False
    
    # Memory
    ram_gb: float = 8.0
    available_ram_gb: float = 8.0
    
    # GPU
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    cuda_available: bool = False
    
    # Storage
    storage_path: str = ""
    storage_free_gb: float = 0.0
    
    # Derived settings
    profile: HardwareProfile = HardwareProfile.AUTO_DETECT
    recommended_batch_size: int = 1
    recommended_model_size: str = "1.5B"
    use_fp16: bool = False
    use_bf16: bool = False
    gradient_checkpointing: bool = True
    num_workers: int = 4
    
    # LoRA settings based on hardware
    lora_r: int = 8
    lora_alpha: int = 16
    
    # GRPO settings
    num_generations: int = 4
    max_completion_length: int = 256
    
    def __post_init__(self):
        self._optimize_for_hardware()
    
    def _optimize_for_hardware(self):
        """Set optimal parameters based on detected hardware"""
        
        # CPU-based optimizations
        if "5950X" in self.cpu_name or "5900X" in self.cpu_name:
            self.profile = HardwareProfile.RYZEN_OPTIMIZED
            self.num_workers = min(8, self.cpu_threads // 2)
        
        # RAM-based optimizations
        if self.ram_gb >= 64:
            self.recommended_batch_size = 4
            self.recommended_model_size = "3B"
            self.lora_r = 16
            self.lora_alpha = 32
            self.num_generations = 8
        elif self.ram_gb >= 32:
            self.recommended_batch_size = 2
            self.recommended_model_size = "1.5B"
            self.lora_r = 16
            self.lora_alpha = 32
            self.num_generations = 6
        elif self.ram_gb >= 16:
            self.recommended_batch_size = 1
            self.recommended_model_size = "0.5B"
            self.lora_r = 8
            self.lora_alpha = 16
            self.num_generations = 4
        else:
            self.recommended_batch_size = 1
            self.recommended_model_size = "0.5B"
            self.lora_r = 4
            self.lora_alpha = 8
            self.num_generations = 2
        
        # GPU optimizations
        if self.has_gpu and self.cuda_available:
            self.profile = HardwareProfile.GPU_ENABLED
            self.use_fp16 = True
            
            if self.gpu_vram_gb >= 24:
                self.recommended_model_size = "7B"
                self.recommended_batch_size = 8
                self.lora_r = 32
            elif self.gpu_vram_gb >= 16:
                self.recommended_model_size = "3B"
                self.recommended_batch_size = 4
                self.lora_r = 16
            elif self.gpu_vram_gb >= 8:
                self.recommended_model_size = "1.5B"
                self.recommended_batch_size = 2
        
        # BF16 support (AMD Zen3+ / Intel 12th+)
        if self.has_avx512 or "5950X" in self.cpu_name:
            self.use_bf16 = not self.has_gpu
    
    def get_training_device(self) -> str:
        """Return the device string for training"""
        if self.has_gpu and self.cuda_available:
            return "cuda"
        return "cpu"
    
    def get_torch_dtype(self) -> str:
        """Return the appropriate torch dtype"""
        if self.use_fp16:
            return "float16"
        if self.use_bf16:
            return "bfloat16"
        return "float32"
    
    def to_training_args(self) -> dict:
        """Convert to training arguments dict"""
        return {
            "per_device_train_batch_size": self.recommended_batch_size,
            "gradient_accumulation_steps": max(1, 8 // self.recommended_batch_size),
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.use_bf16 and not self.has_gpu,
            "fp16": self.use_fp16 and self.has_gpu,
            "dataloader_num_workers": self.num_workers,
            "use_cpu": not self.has_gpu,
        }
    
    def to_lora_config(self) -> dict:
        """Convert to LoRA config dict"""
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
    
    def to_grpo_config(self) -> dict:
        """Convert to GRPO-specific config"""
        return {
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "use_vllm": False,  # CPU training
        }
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "=" * 50,
            "Hardware Configuration Summary",
            "=" * 50,
            f"Profile: {self.profile.value}",
            f"CPU: {self.cpu_name} ({self.cpu_cores} cores, {self.cpu_threads} threads)",
            f"RAM: {self.ram_gb:.1f} GB (available: {self.available_ram_gb:.1f} GB)",
            f"AVX2: {self.has_avx2}, AVX512: {self.has_avx512}",
        ]
        
        if self.has_gpu:
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)")
            lines.append(f"CUDA: {self.cuda_available}")
        else:
            lines.append("GPU: None detected")
        
        lines.extend([
            "-" * 50,
            "Recommended Settings:",
            f"  Model Size: {self.recommended_model_size}",
            f"  Batch Size: {self.recommended_batch_size}",
            f"  LoRA r: {self.lora_r}, alpha: {self.lora_alpha}",
            f"  Precision: {'FP16' if self.use_fp16 else 'BF16' if self.use_bf16 else 'FP32'}",
            f"  GRPO Generations: {self.num_generations}",
            f"  Workers: {self.num_workers}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


def detect_hardware(storage_path: Optional[str] = None) -> HardwareConfig:
    """
    Detect system hardware and return optimized configuration.
    
    Args:
        storage_path: Path for model storage (default: ~/.toolbox/models)
    
    Returns:
        HardwareConfig with detected and optimized settings
    """
    config = HardwareConfig()
    
    # Storage path
    if storage_path:
        config.storage_path = storage_path
    else:
        try:
            from toolboxv2 import get_app
            config.storage_path = str(get_app().data_dir) + '/models'
        except:
            config.storage_path = os.path.expanduser("~/.toolbox/models")
    
    os.makedirs(config.storage_path, exist_ok=True)
    
    # Detect CPU
    config.cpu_name = platform.processor() or "Unknown"
    
    try:
        import psutil
        config.cpu_cores = psutil.cpu_count(logical=False) or 1
        config.cpu_threads = psutil.cpu_count(logical=True) or 1
        
        # RAM
        mem = psutil.virtual_memory()
        config.ram_gb = mem.total / (1024 ** 3)
        config.available_ram_gb = mem.available / (1024 ** 3)
        
        # Storage
        disk = psutil.disk_usage(config.storage_path)
        config.storage_free_gb = disk.free / (1024 ** 3)
    except ImportError:
        # Fallback without psutil
        config.cpu_cores = os.cpu_count() or 1
        config.cpu_threads = os.cpu_count() or 1
        config.ram_gb = 16.0  # Conservative default
        config.available_ram_gb = 8.0
    
    # Detect AVX support (Linux/Windows)
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                config.has_avx2 = "avx2" in cpuinfo
                config.has_avx512 = "avx512" in cpuinfo
        elif platform.system() == "Windows":
            # Check via CPU name patterns
            cpu_lower = config.cpu_name.lower()
            if "5950x" in cpu_lower or "5900x" in cpu_lower or "7950x" in cpu_lower:
                config.has_avx2 = True
                # Zen3/4 don't have AVX-512
                config.has_avx512 = False
    except:
        pass
    
    # Detect GPU
    try:
        import torch
        config.cuda_available = torch.cuda.is_available()
        
        if config.cuda_available:
            config.has_gpu = True
            config.gpu_name = torch.cuda.get_device_name(0)
            config.gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        config.cuda_available = False
        config.has_gpu = False
    
    # Also check for ROCm (AMD GPUs)
    if not config.has_gpu:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                config.has_gpu = True
                config.gpu_name = "AMD ROCm GPU"
                # Parse VRAM from output if possible
        except:
            pass
    
    # Re-run optimization with detected values
    config._optimize_for_hardware()
    
    return config


def get_ryzen_optimized_config(storage_path: Optional[str] = None) -> HardwareConfig:
    """
    Get Ryzen-optimized configuration (for Ryzen 9 5950X specifically).
    
    This is a preset for the known hardware configuration.
    """
    config = HardwareConfig(
        cpu_name="AMD Ryzen 9 5950X 16-Core Processor",
        cpu_cores=16,
        cpu_threads=32,
        has_avx2=True,
        has_avx512=False,
        ram_gb=40.0,
        available_ram_gb=32.0,
        storage_path=storage_path or os.path.expanduser("~/.toolbox/models"),
        profile=HardwareProfile.RYZEN_OPTIMIZED,
    )
    
    # Check for GPU at runtime
    try:
        import torch
        if torch.cuda.is_available():
            config.has_gpu = True
            config.cuda_available = True
            config.gpu_name = torch.cuda.get_device_name(0)
            config.gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            config.profile = HardwareProfile.GPU_ENABLED
    except:
        pass
    
    config._optimize_for_hardware()
    return config
