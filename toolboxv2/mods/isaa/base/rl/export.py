"""
Export Module for RL-Trained Models

Handles GGUF conversion and Ollama deployment with
Ryzen-optimized and auto-detect hosting profiles.
"""

import os
import subprocess
import tempfile
import shutil
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime


@dataclass
class GGUFQuantization:
    """GGUF quantization options"""
    name: str
    description: str
    bits: float
    recommended_for: str

    @staticmethod
    def available() -> dict:
        return {
            "Q2_K": GGUFQuantization("Q2_K", "2-bit quantization, smallest", 2.5, "Very limited RAM"),
            "Q3_K_M": GGUFQuantization("Q3_K_M", "3-bit quantization, medium", 3.5, "Limited RAM"),
            "Q4_K_M": GGUFQuantization("Q4_K_M", "4-bit quantization, balanced", 4.5, "General use"),
            "Q5_K_M": GGUFQuantization("Q5_K_M", "5-bit quantization, quality", 5.5, "Quality focus"),
            "Q6_K": GGUFQuantization("Q6_K", "6-bit quantization, high quality", 6.5, "High quality"),
            "Q8_0": GGUFQuantization("Q8_0", "8-bit quantization, near-FP16", 8.0, "Maximum quality"),
            "F16": GGUFQuantization("F16", "FP16, no quantization", 16.0, "Full precision"),
        }


class GGUFExporter:
    """
    Export HuggingFace models to GGUF format for llama.cpp/Ollama.

    Requires llama.cpp to be installed or will clone it automatically.
    """

    def __init__(
        self,
        model_path: str,
        llama_cpp_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize exporter.

        Args:
            model_path: Path to HuggingFace model directory
            llama_cpp_path: Path to llama.cpp installation
            output_dir: Output directory for GGUF files
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.model_path.parent / "gguf"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find or setup llama.cpp
        if llama_cpp_path:
            self.llama_cpp_path = Path(llama_cpp_path)
        else:
            self.llama_cpp_path = self._find_or_install_llama_cpp()

        self.convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"

    def _find_or_install_llama_cpp(self) -> Path:
        """Find existing llama.cpp or install it"""
        # Check common locations
        common_paths = [
            Path.home() / "llama.cpp",
            Path.home() / ".local" / "llama.cpp",
            Path("/opt/llama.cpp"),
        ]

        # Also check toolboxv2 data dir
        try:
            from toolboxv2 import get_app
            common_paths.insert(0, Path(get_app().data_dir) / "llama.cpp")
        except:
            pass

        for path in common_paths:
            if (path / "convert_hf_to_gguf.py").exists():
                print(f"Found llama.cpp at {path}")
                return path

        # Need to install
        install_path = common_paths[0]
        print(f"llama.cpp not found. Installing to {install_path}...")

        self._install_llama_cpp(install_path)
        return install_path

    def _install_llama_cpp(self, install_path: Path):
        """Clone and build llama.cpp"""
        install_path.parent.mkdir(parents=True, exist_ok=True)

        # Clone repository
        print("Cloning llama.cpp...")
        subprocess.run([
            "git", "clone",
            "https://github.com/ggml-org/llama.cpp.git",
            str(install_path)
        ], check=True)

        # Install Python requirements
        requirements_path = install_path / "requirements.txt"
        if requirements_path.exists():
            print("Installing Python requirements...")
            subprocess.run([
                "pip", "install", "-r", str(requirements_path),
                "--break-system-packages"
            ], check=True)

        # Build (optional, for quantization)
        print("Building llama.cpp...")
        build_dir = install_path / "build"
        build_dir.mkdir(exist_ok=True)

        try:
            subprocess.run(
                ["cmake", ".."],
                cwd=str(build_dir),
                check=True
            )
            subprocess.run(
                ["cmake", "--build", ".", "--config", "Release"],
                cwd=str(build_dir),
                check=True
            )
        except Exception as e:
            print(f"Build failed (optional): {e}")
            print("Conversion will still work, but quantization may need manual setup")

        print("llama.cpp installed successfully")

    def convert(
        self,
        quantization: str = "Q4_K_M",
        output_name: Optional[str] = None
    ) -> str:
        """
        Convert HuggingFace model to GGUF.

        The conversion is a two-step process:
        1. Convert HF model to F16 GGUF using convert_hf_to_gguf.py
        2. Quantize to target format using llama-quantize (if not F16/F32)

        Args:
            quantization: Quantization type (Q4_K_M, Q8_0, F16, etc.)
            output_name: Output filename (default: model-{quantization}.gguf)

        Returns:
            Path to GGUF file
        """
        if not self.convert_script.exists():
            raise FileNotFoundError(f"Convert script not found at {self.convert_script}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        model_name = self.model_path.name

        # Determine if we need post-conversion quantization
        # convert_hf_to_gguf.py only supports: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
        direct_types = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
        quant_lower = quantization.lower()

        if quant_lower in direct_types:
            # Direct conversion
            if output_name:
                output_file = self.output_dir / output_name
            else:
                output_file = self.output_dir / f"{model_name}-{quantization}.gguf"

            return self._convert_direct(output_file, quant_lower)
        else:
            # Two-step: convert to F16, then quantize
            f16_file = self.output_dir / f"{model_name}-F16.gguf"

            if output_name:
                output_file = self.output_dir / output_name
            else:
                output_file = self.output_dir / f"{model_name}-{quantization}.gguf"

            # Step 1: Convert to F16
            if not f16_file.exists():
                print(f"Step 1: Converting to F16...")
                self._convert_direct(f16_file, "f16")
            else:
                print(f"Using existing F16 file: {f16_file}")

            # Step 2: Quantize
            print(f"Step 2: Quantizing to {quantization}...")
            return self._quantize(f16_file, output_file, quantization)

    def _convert_direct(self, output_file: Path, outtype: str) -> str:
        """Direct conversion using convert_hf_to_gguf.py"""
        print(f"Converting {self.model_path} to GGUF...")
        print(f"Output type: {outtype}")
        print(f"Output: {output_file}")

        import sys
        cmd = [
            sys.executable, str(self.convert_script),
            str(self.model_path),
            "--outfile", str(output_file),
            "--outtype", outtype
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Conversion failed: {result.stderr}")

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"Conversion successful! Size: {size_mb:.1f} MB")
            return str(output_file)
        else:
            raise RuntimeError("Conversion completed but output file not found")

    def _quantize(self, input_file: Path, output_file: Path, quantization: str) -> str:
        """Quantize GGUF file using llama-quantize"""
        # Find llama-quantize executable
        quantize_exe = None
        possible_paths = [
            self.llama_cpp_path / "build" / "bin" / "llama-quantize",
            self.llama_cpp_path / "build" / "bin" / "llama-quantize.exe",
            self.llama_cpp_path / "llama-quantize",
            self.llama_cpp_path / "llama-quantize.exe",
            self.llama_cpp_path / "build" / "Release" / "llama-quantize.exe",
            self.llama_cpp_path / "build" / "Release" / "bin" / "llama-quantize.exe",
        ]

        for path in possible_paths:
            if path.exists():
                quantize_exe = path
                break

        if quantize_exe is None:
            print("Warning: llama-quantize not found. Returning F16 file instead.")
            print("To enable quantization, build llama.cpp with: cmake --build build --config Release")
            return str(input_file)

        print(f"Quantizing with: {quantize_exe}")
        cmd = [str(quantize_exe), str(input_file), str(output_file), quantization]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Quantization failed: {result.stderr}")
            print("Returning F16 file instead.")
            return str(input_file)

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"Quantization successful! Size: {size_mb:.1f} MB")
            return str(output_file)
        else:
            print("Quantization completed but output file not found. Returning F16.")
            return str(input_file)

    def get_recommended_quantization(self, available_ram_gb: float = 8.0) -> str:
        """Get recommended quantization based on available RAM"""
        if available_ram_gb >= 32:
            return "Q8_0"
        elif available_ram_gb >= 16:
            return "Q6_K"
        elif available_ram_gb >= 8:
            return "Q4_K_M"
        elif available_ram_gb >= 4:
            return "Q3_K_M"
        else:
            return "Q2_K"


@dataclass
class OllamaHostingProfile:
    """Hosting profile for Ollama"""
    name: str
    num_parallel: int = 1
    num_ctx: int = 4096
    num_gpu: int = 0
    num_thread: int = 0  # 0 = auto
    main_gpu: int = 0
    flash_attn: bool = False

    def to_env(self) -> dict:
        """Convert to environment variables"""
        env = {
            "OLLAMA_NUM_PARALLEL": str(self.num_parallel),
            "OLLAMA_MAX_LOADED_MODELS": "2",
        }

        if self.num_thread > 0:
            env["OLLAMA_NUM_THREAD"] = str(self.num_thread)

        if self.flash_attn:
            env["OLLAMA_FLASH_ATTENTION"] = "1"

        return env


class OllamaDeployer:
    """
    Deploy GGUF models to Ollama with optimized hosting profiles.
    """

    def __init__(self, ollama_path: str = "ollama"):
        """
        Initialize deployer.

        Args:
            ollama_path: Path to ollama executable
        """
        self.ollama_path = ollama_path
        self._verify_ollama()

    def _verify_ollama(self):
        """Verify Ollama is installed"""
        try:
            result = subprocess.run(
                [self.ollama_path, "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"Ollama version: {result.stdout.strip()}")
            else:
                raise RuntimeError("Ollama not responding")
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not found. Install from https://ollama.ai\n"
                "Linux: curl -fsSL https://ollama.ai/install.sh | sh"
            )

    def create_modelfile(
        self,
        gguf_path: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        stop_tokens: list[str] = None
    ) -> str:
        """
        Create Ollama Modelfile content.

        Args:
            gguf_path: Path to GGUF file
            system_prompt: System prompt for the model
            temperature: Default temperature
            num_ctx: Context window size
            stop_tokens: Stop sequences

        Returns:
            Modelfile content as string
        """
        lines = [f"FROM {gguf_path}"]

        # Parameters
        lines.append(f"PARAMETER temperature {temperature}")
        lines.append(f"PARAMETER num_ctx {num_ctx}")

        if stop_tokens:
            for token in stop_tokens:
                lines.append(f'PARAMETER stop "{token}"')

        # System prompt
        if system_prompt:
            lines.append(f'SYSTEM """{system_prompt}"""')
        else:
            default_prompt = """You are a helpful AI assistant trained for ToolBoxV2.
You can execute code, use tools, and help with various tasks.
Be concise and accurate in your responses."""
            lines.append(f'SYSTEM """{default_prompt}"""')

        return "\n".join(lines)

    def create_model(
        self,
        model_name: str,
        gguf_path: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        num_ctx: int = 4096
    ) -> str:
        """
        Create Ollama model from GGUF file.

        Args:
            model_name: Name for the Ollama model
            gguf_path: Path to GGUF file
            system_prompt: System prompt
            temperature: Default temperature
            num_ctx: Context window

        Returns:
            Model name
        """
        # Create Modelfile
        modelfile_content = self.create_modelfile(
            gguf_path,
            system_prompt,
            temperature,
            num_ctx
        )

        # Write temporary Modelfile
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".Modelfile",
            delete=False
        ) as f:
            f.write(modelfile_content)
            modelfile_path = f.name

        try:
            print(f"Creating Ollama model: {model_name}")

            result = subprocess.run(
                [self.ollama_path, "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                raise RuntimeError(f"Model creation failed: {result.stderr}")

            print(f"Model '{model_name}' created successfully")
            print(f"Run with: ollama run {model_name}")

            return model_name

        finally:
            os.unlink(modelfile_path)

    def list_models(self) -> list[dict]:
        """List installed Ollama models"""
        result = subprocess.run(
            [self.ollama_path, "list"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            return []

        models = []
        lines = result.stdout.strip().split("\n")[1:]  # Skip header

        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                models.append({
                    "name": parts[0],
                    "id": parts[1],
                    "size": parts[2],
                    "modified": " ".join(parts[3:])
                })

        return models

    def delete_model(self, model_name: str) -> bool:
        """Delete an Ollama model"""
        result = subprocess.run(
            [self.ollama_path, "rm", model_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode == 0

    def run_model(self, model_name: str, prompt: str) -> str:
        """Run a prompt through the model"""
        result = subprocess.run(
            [self.ollama_path, "run", model_name, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            raise RuntimeError(f"Model run failed: {result.stderr}")

        return result.stdout

    def get_ryzen_profile(self, cpu_cores: int = 16) -> OllamaHostingProfile:
        """Get Ryzen-optimized hosting profile"""
        return OllamaHostingProfile(
            name="ryzen_optimized",
            num_parallel=min(4, cpu_cores // 4),
            num_ctx=4096,
            num_thread=cpu_cores - 2,  # Leave 2 cores for system
            flash_attn=False  # CPU doesn't support flash attention
        )

    def get_auto_profile(self) -> OllamaHostingProfile:
        """Auto-detect optimal hosting profile"""
        import platform

        # Detect CPU cores
        cpu_cores = os.cpu_count() or 4

        # Detect RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            ram_gb = 16  # Conservative default

        # Detect GPU
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        # Build profile
        if has_gpu:
            return OllamaHostingProfile(
                name="gpu_auto",
                num_parallel=4,
                num_ctx=8192,
                num_gpu=99,  # Use all GPU layers
                flash_attn=True
            )
        else:
            # CPU profile based on resources
            parallel = min(4, cpu_cores // 4)
            ctx = 4096 if ram_gb >= 16 else 2048

            return OllamaHostingProfile(
                name="cpu_auto",
                num_parallel=parallel,
                num_ctx=ctx,
                num_thread=cpu_cores - 2
            )

    def start_server_with_profile(self, profile: OllamaHostingProfile):
        """Start Ollama server with hosting profile"""
        env = os.environ.copy()
        env.update(profile.to_env())

        print(f"Starting Ollama with profile: {profile.name}")
        print(f"Environment: {profile.to_env()}")

        # Start server in background
        subprocess.Popen(
            [self.ollama_path, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("Ollama server started")


class ExportPipeline:
    """
    Complete export pipeline from trained model to deployed Ollama.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "toolbox-agent",
        output_dir: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.model_path.parent / "export"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gguf_exporter = GGUFExporter(str(self.model_path), output_dir=str(self.output_dir))
        self.ollama_deployer = OllamaDeployer()

    def run(
        self,
        quantization: str = "Q4_K_M",
        system_prompt: str = "",
        hosting_profile: str = "auto"
    ) -> dict:
        """
        Run complete export pipeline.

        Args:
            quantization: GGUF quantization type
            system_prompt: System prompt for Ollama model
            hosting_profile: "ryzen" or "auto"

        Returns:
            Pipeline results
        """
        results = {
            "start_time": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "quantization": quantization
        }

        try:
            # Convert to GGUF
            print("Step 1: Converting to GGUF...")
            gguf_path = self.gguf_exporter.convert(quantization)
            results["gguf_path"] = gguf_path
            results["gguf_size_mb"] = Path(gguf_path).stat().st_size / (1024 * 1024)

            # Create Ollama model
            print("Step 2: Creating Ollama model...")
            ollama_model = self.ollama_deployer.create_model(
                self.model_name,
                gguf_path,
                system_prompt
            )
            results["ollama_model"] = ollama_model

            # Setup hosting profile
            print("Step 3: Configuring hosting profile...")
            if hosting_profile == "ryzen":
                profile = self.ollama_deployer.get_ryzen_profile()
            else:
                profile = self.ollama_deployer.get_auto_profile()

            results["hosting_profile"] = {
                "name": profile.name,
                "num_parallel": profile.num_parallel,
                "num_ctx": profile.num_ctx,
                "num_thread": profile.num_thread
            }

            # Save profile for later use
            profile_path = self.output_dir / "hosting_profile.json"
            with open(profile_path, "w") as f:
                json.dump(results["hosting_profile"], f, indent=2)

            results["success"] = True
            results["end_time"] = datetime.now().isoformat()

            print("\n" + "=" * 50)
            print("Export Pipeline Complete!")
            print("=" * 50)
            print(f"GGUF: {gguf_path} ({results['gguf_size_mb']:.1f} MB)")
            print(f"Ollama Model: {ollama_model}")
            print(f"Profile: {profile.name}")
            print(f"\nRun with: ollama run {ollama_model}")
            print("=" * 50)

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            import traceback
            results["traceback"] = traceback.format_exc()
            print(f"Export failed: {e}")

        # Save results
        results_path = self.output_dir / "export_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results


def quick_export(
    model_path: str,
    model_name: str = "toolbox-agent",
    quantization: str = "Q4_K_M"
) -> str:
    """
    Quick export function for simple use cases.

    Args:
        model_path: Path to HuggingFace model
        model_name: Name for Ollama model
        quantization: GGUF quantization type

    Returns:
        Ollama model name
    """
    pipeline = ExportPipeline(model_path, model_name)
    results = pipeline.run(quantization)

    if results["success"]:
        return results["ollama_model"]
    else:
        raise RuntimeError(f"Export failed: {results.get('error', 'Unknown error')}")
