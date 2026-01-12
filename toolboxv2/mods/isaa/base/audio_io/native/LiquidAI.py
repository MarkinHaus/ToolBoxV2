"""
OmniCore Native Audio Model Runner
==================================

Standalone runner for native audio language models like LFM2.5-Audio.
Designed for local CPU execution on systems without GPU.

Supported Models:
- LiquidAI/LFM2.5-Audio-1.5B (recommended for CPU)
- LiquidAI/LFM2-Audio-1.5B

Features:
- CPU-optimized inference (llama.cpp GGUF support)
- Interleaved generation (audio + text simultaneously)
- Sequential generation (ASR or TTS mode)
- Tool calling support
- Real-time streaming output

Hardware Requirements (LFM2.5-Audio-1.5B):
- CPU: Any modern x86_64 or ARM64
- RAM: 4-8GB (Q4 quantized)
- Disk: ~3GB for model files

Author: OmniCore Team
Version: 1.0.0
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Optional,
    Union,
)

# =============================================================================
# CONFIGURATION
# =============================================================================


class GenerationMode(Enum):
    """Native audio model generation modes."""

    INTERLEAVED = "interleaved"  # Real-time: audio + text alternating
    SEQUENTIAL = "sequential"  # Complete one modality then switch


class NativeModelBackend(Enum):
    """Backend for running native audio models."""

    TRANSFORMERS = "transformers"  # HuggingFace transformers
    LLAMA_CPP = "llama_cpp"  # llama.cpp (CPU optimized)


@dataclass
class NativeAudioConfig:
    """
    Configuration for native audio model execution.

    Attributes:
        model_id: HuggingFace model ID or local path
        backend: Which inference backend to use
        device: Device for inference (cpu, cuda)
        quantization: Quantization level for GGUF (Q4_K_M, Q8_0, etc.)

    Generation settings:
        generation_mode: Interleaved or sequential
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature for audio
        audio_top_k: Top-k sampling for audio tokens

    System settings:
        system_prompt: Default system prompt
        num_threads: CPU threads for llama.cpp
        cache_dir: Directory for model cache
    """

    # Model selection
    model_id: str = "LiquidAI/LFM2.5-Audio-1.5B"
    backend: NativeModelBackend = NativeModelBackend.TRANSFORMERS
    device: str = "cpu"
    quantization: Optional[str] = "Q4_K_M"  # For llama.cpp

    # Generation
    generation_mode: GenerationMode = GenerationMode.INTERLEAVED
    max_tokens: int = 512
    temperature: float = 1.0
    audio_top_k: int = 4

    # System
    system_prompt: str = "Respond with interleaved text and audio."
    num_threads: Optional[int] = None  # Auto-detect if None
    cache_dir: str = "./models"

    # Tool calling
    tools: Optional[list[dict]] = None

    def __post_init__(self):
        if self.num_threads is None:
            self.num_threads = os.cpu_count() or 4


@dataclass
class NativeAudioOutput:
    """Output from native audio model generation."""

    text: str
    audio: Optional[bytes]
    tool_calls: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# NATIVE MODEL LOADER
# =============================================================================


class NativeAudioModel(ABC):
    """Abstract base class for native audio models."""

    @abstractmethod
    def generate(
        self,
        audio_input: bytes,
        text_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.INTERLEAVED,
        **kwargs,
    ) -> NativeAudioOutput:
        """Generate response from audio/text input."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        audio_input: bytes,
        text_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.INTERLEAVED,
        **kwargs,
    ) -> Generator[tuple[str, bytes], None, None]:
        """Stream generation, yielding (text_chunk, audio_chunk) tuples."""
        pass

    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        """ASR mode: audio → text."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """TTS mode: text → audio."""
        pass


class LFMAudioTransformers(NativeAudioModel):
    """
    LFM2.5-Audio model using HuggingFace transformers.

    Requires: pip install liquid-audio torch torchaudio
    """

    def __init__(self, config: NativeAudioConfig):
        self.config = config
        self._load_model()

    def _load_model(self):
        """Load model and processor."""
        try:
            import torch
            import torchaudio
            from liquid_audio import (
                ChatState,
                LFM2AudioModel,
                LFM2AudioProcessor,
                LFMModality,
            )
        except ImportError:
            raise ImportError(
                "Required packages not installed. Install with:\n"
                "pip install liquid-audio torch torchaudio"
            )

        print(f"Loading model: {self.config.model_id}")

        self.processor = LFM2AudioProcessor.from_pretrained(
            self.config.model_id, cache_dir=self.config.cache_dir
        ).eval()

        self.model = LFM2AudioModel.from_pretrained(
            self.config.model_id, cache_dir=self.config.cache_dir
        ).eval()

        if self.config.device == "cuda":
            import torch

            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("Model loaded on CUDA")
            else:
                print("CUDA not available, using CPU")
        else:
            print("Model loaded on CPU")

        self.ChatState = ChatState
        self.LFMModality = LFMModality
        self.torch = torch
        self.torchaudio = torchaudio

    def _audio_bytes_to_tensor(self, audio: bytes):
        """Convert audio bytes to tensor."""
        # Save to temp file for torchaudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            temp_path = f.name

        try:
            wav, sr = self.torchaudio.load(temp_path)
            return wav, sr
        finally:
            os.unlink(temp_path)

    def _tensor_to_audio_bytes(self, audio_tensor, sample_rate: int = 24000) -> bytes:
        """Convert audio tensor to WAV bytes."""
        import numpy as np

        audio_np = audio_tensor.squeeze().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    def _create_chat(
        self, audio_input: Optional[bytes] = None, text_input: Optional[str] = None
    ):
        """Create chat state with inputs."""
        chat = self.ChatState(self.processor)

        # System prompt
        chat.new_turn("system")
        if self.config.tools:
            tools_json = json.dumps(self.config.tools)
            chat.add_text(
                f"<|tool_list_start|>{tools_json}<|tool_list_end|>\n"
                f"{self.config.system_prompt}"
            )
        else:
            chat.add_text(self.config.system_prompt)
        chat.end_turn()

        # User input
        chat.new_turn("user")
        if audio_input:
            wav, sr = self._audio_bytes_to_tensor(audio_input)
            chat.add_audio(wav, sr)
        if text_input:
            chat.add_text(text_input)
        chat.end_turn()

        chat.new_turn("assistant")
        return chat

    def generate(
        self,
        audio_input: Optional[bytes] = None,
        text_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.INTERLEAVED,
        **kwargs,
    ) -> NativeAudioOutput:
        """Generate response."""
        chat = self._create_chat(audio_input, text_input)

        text_tokens = []
        audio_tokens = []

        if mode == GenerationMode.INTERLEAVED:
            gen_func = self.model.generate_interleaved
        else:
            gen_func = self.model.generate_sequential

        for token in gen_func(
            **chat,
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            audio_temperature=kwargs.get("temperature", self.config.temperature),
            audio_top_k=kwargs.get("audio_top_k", self.config.audio_top_k),
        ):
            if token.numel() == 1:
                text_tokens.append(self.processor.text.decode(token))
            else:
                audio_tokens.append(token)

        # Decode outputs
        text_output = "".join(text_tokens)

        if audio_tokens:
            audio_tensor = self.torch.cat(audio_tokens)
            audio_output = self.processor.audio.decode(audio_tensor)
            audio_bytes = self._tensor_to_audio_bytes(audio_output)
        else:
            audio_bytes = None

        # Extract tool calls
        tool_calls = self._extract_tool_calls(text_output)

        return NativeAudioOutput(
            text=self._clean_text(text_output),
            audio=audio_bytes,
            tool_calls=tool_calls,
            metadata={
                "text_tokens": len(text_tokens),
                "audio_tokens": len(audio_tokens),
                "mode": mode.value,
            },
        )

    def generate_stream(
        self,
        audio_input: Optional[bytes] = None,
        text_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.INTERLEAVED,
        **kwargs,
    ) -> Generator[tuple[str, Optional[bytes]], None, None]:
        """Stream generation."""
        chat = self._create_chat(audio_input, text_input)

        audio_buffer = []
        audio_buffer_size = 10  # Yield every N audio tokens

        if mode == GenerationMode.INTERLEAVED:
            gen_func = self.model.generate_interleaved
        else:
            gen_func = self.model.generate_sequential

        for token in gen_func(
            **chat,
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            audio_temperature=kwargs.get("temperature", self.config.temperature),
            audio_top_k=kwargs.get("audio_top_k", self.config.audio_top_k),
        ):
            if token.numel() == 1:
                # Text token - yield immediately
                text_chunk = self.processor.text.decode(token)
                yield (text_chunk, None)
            else:
                # Audio token - buffer and yield when ready
                audio_buffer.append(token)

                if len(audio_buffer) >= audio_buffer_size:
                    audio_tensor = self.torch.cat(audio_buffer)
                    audio_output = self.processor.audio.decode(audio_tensor)
                    audio_bytes = self._tensor_to_audio_bytes(audio_output)
                    yield ("", audio_bytes)
                    audio_buffer = []

        # Yield remaining audio
        if audio_buffer:
            audio_tensor = self.torch.cat(audio_buffer)
            audio_output = self.processor.audio.decode(audio_tensor)
            audio_bytes = self._tensor_to_audio_bytes(audio_output)
            yield ("", audio_bytes)

    def transcribe(self, audio: bytes) -> str:
        """ASR: audio → text."""
        # Use sequential mode with ASR prompt
        original_prompt = self.config.system_prompt
        self.config.system_prompt = "Perform ASR."

        result = self.generate(audio_input=audio, mode=GenerationMode.SEQUENTIAL)

        self.config.system_prompt = original_prompt
        return result.text

    def synthesize(self, text: str) -> bytes:
        """TTS: text → audio."""
        # Use sequential mode with TTS prompt
        original_prompt = self.config.system_prompt
        self.config.system_prompt = "Perform TTS."

        result = self.generate(text_input=text, mode=GenerationMode.SEQUENTIAL)

        self.config.system_prompt = original_prompt
        return result.audio or b""

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from text."""
        import ast
        import re

        tool_calls = []
        pattern = r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>"

        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                call_str = match.group(1).strip()
                calls = ast.literal_eval(call_str)
                for call in calls if isinstance(calls, list) else [calls]:
                    tool_calls.append(
                        {"name": call.get("name"), "arguments": call.get("arguments", {})}
                    )
            except (SyntaxError, ValueError):
                pass

        return tool_calls

    def _clean_text(self, text: str) -> str:
        """Remove special tokens from text."""
        import re

        patterns = [
            r"<\|tool_call_start\|>.*?<\|tool_call_end\|>",
            r"<\|tool_response_start\|>.*?<\|tool_response_end\|>",
            r"<\|im_start\|>.*?<\|im_end\|>",
            r"<\|startoftext\|>",
            r"<\|endoftext\|>",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)
        return text.strip()


class LFMAudioLlamaCpp(NativeAudioModel):
    """
    LFM2.5-Audio model using llama.cpp (GGUF).

    CPU-optimized for maximum performance on x86_64/ARM64.

    Requires: pip install llama-cpp-python huggingface_hub

    Note: Audio model support in llama.cpp may be limited.
    Check https://github.com/ggml-org/llama.cpp for latest updates.
    """

    def __init__(self, config: NativeAudioConfig):
        self.config = config
        self._load_model()

    def _load_model(self):
        """Load GGUF model via llama.cpp."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python"
            )

        # Get GGUF path
        gguf_path = self._get_gguf_path()

        print(f"Loading GGUF: {gguf_path}")

        self.model = Llama(
            model_path=str(gguf_path),
            n_ctx=8192,
            n_threads=self.config.num_threads,
            verbose=False,
        )

        print(f"Model loaded (threads={self.config.num_threads})")

    def _get_gguf_path(self) -> Path:
        """Get or download GGUF model file."""
        model_id = self.config.model_id

        # Check if already a local path
        if Path(model_id).exists():
            return Path(model_id)

        # Download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Install with:\n"
                "pip install huggingface_hub"
            )

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Find GGUF file matching quantization
        quant = self.config.quantization or "Q4_K_M"

        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            # Find matching quantization
            target_file = None
            for f in gguf_files:
                if quant.lower() in f.lower():
                    target_file = f
                    break

            if not target_file and gguf_files:
                target_file = gguf_files[0]

            if not target_file:
                raise FileNotFoundError(f"No GGUF files found in {model_id}")

            # Download
            print(f"Downloading: {target_file}")
            local_path = hf_hub_download(
                repo_id=model_id, filename=target_file, local_dir=str(cache_dir)
            )

            return Path(local_path)

        except Exception as e:
            raise RuntimeError(f"Failed to download GGUF: {e}")

    def generate(
        self,
        audio_input: Optional[bytes] = None,
        text_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.INTERLEAVED,
        **kwargs,
    ) -> NativeAudioOutput:
        """
        Generate response.

        Note: llama.cpp audio support is experimental.
        This implementation uses text-only mode as fallback.
        """
        # Build prompt
        prompt = self._build_prompt(text_input or "[Audio input]")

        # Generate
        output = self.model(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        text = output["choices"][0]["text"]

        return NativeAudioOutput(
            text=text.strip(),
            audio=None,  # Audio not yet supported in llama.cpp
            tool_calls=self._extract_tool_calls(text),
            metadata={"backend": "llama_cpp"},
        )

    def generate_stream(
        self,
        audio_input: Optional[bytes] = None,
        text_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.INTERLEAVED,
        **kwargs,
    ) -> Generator[tuple[str, Optional[bytes]], None, None]:
        """Stream generation (text only for llama.cpp)."""
        prompt = self._build_prompt(text_input or "[Audio input]")

        for output in self.model(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=True,
        ):
            text = output["choices"][0]["text"]
            yield (text, None)

    def transcribe(self, audio: bytes) -> str:
        """ASR mode - not fully supported in llama.cpp."""
        raise NotImplementedError(
            "Audio input not yet fully supported in llama.cpp. "
            "Use transformers backend for audio processing."
        )

    def synthesize(self, text: str) -> bytes:
        """TTS mode - not fully supported in llama.cpp."""
        raise NotImplementedError(
            "Audio output not yet fully supported in llama.cpp. "
            "Use transformers backend for audio synthesis."
        )

    def _build_prompt(self, user_input: str) -> str:
        """Build chat prompt."""
        return (
            f"<|startoftext|><|im_start|>system\n"
            f"{self.config.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from text."""
        import ast
        import re

        tool_calls = []
        pattern = r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>"

        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                calls = ast.literal_eval(match.group(1).strip())
                for call in calls if isinstance(calls, list) else [calls]:
                    tool_calls.append(call)
            except:
                pass

        return tool_calls


# =============================================================================
# FACTORY
# =============================================================================


def load_native_audio_model(config: NativeAudioConfig) -> NativeAudioModel:
    """
    Load native audio model with specified backend.

    Args:
        config: NativeAudioConfig with model settings

    Returns:
        NativeAudioModel instance

    Example:
        config = NativeAudioConfig(
            model_id="LiquidAI/LFM2.5-Audio-1.5B",
            backend=NativeModelBackend.TRANSFORMERS,
            device="cpu"
        )
        model = load_native_audio_model(config)

        result = model.generate(
            audio_input=audio_bytes,
            mode=GenerationMode.INTERLEAVED
        )
        print(result.text)
    """
    if config.backend == NativeModelBackend.TRANSFORMERS:
        return LFMAudioTransformers(config)
    elif config.backend == NativeModelBackend.LLAMA_CPP:
        return LFMAudioLlamaCpp(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# =============================================================================
# CLI / DEMO
# =============================================================================


def demo_conversation():
    """
    Demo: Interactive voice conversation.

    Requires microphone and speakers.
    """
    print("=" * 60)
    print("OmniCore Native Audio Model Demo")
    print("=" * 60)

    # Check dependencies
    try:
        import torch
        import torchaudio
        from liquid_audio import LFM2AudioModel

        print("✓ liquid_audio available")
    except ImportError:
        print("✗ liquid_audio not installed")
        print("  Install with: pip install liquid-audio torch torchaudio")
        return

    # Load model
    config = NativeAudioConfig(
        model_id="LiquidAI/LFM2.5-Audio-1.5B",
        backend=NativeModelBackend.TRANSFORMERS,
        device="cpu",
        tools=[
            {"name": "get_time", "description": "Get the current time", "parameters": {}}
        ],
    )

    print(f"\nLoading model: {config.model_id}")
    print("This may take a few minutes on first run...\n")

    model = load_native_audio_model(config)

    print("Model loaded! Ready for conversation.\n")

    # Demo with sample audio (if available)
    sample_audio = Path("assets/question.wav")
    if sample_audio.exists():
        print(f"Processing sample: {sample_audio}")
        audio_bytes = sample_audio.read_bytes()

        result = model.generate(audio_input=audio_bytes, mode=GenerationMode.INTERLEAVED)

        print(f"\nResponse text: {result.text}")
        print(f"Audio generated: {len(result.audio or b'')} bytes")

        if result.audio:
            output_path = Path("response.wav")
            output_path.write_bytes(result.audio)
            print(f"Saved to: {output_path}")
    else:
        print("No sample audio found. Demonstrating text-to-speech:")

        text = "Hello! The current time is fourteen thirty five."
        print(f"Input: {text}")

        audio = model.synthesize(text)
        if audio:
            output_path = Path("tts_output.wav")
            output_path.write_bytes(audio)
            print(f"Audio saved to: {output_path}")


def demo_tool_calling():
    """
    Demo: Tool calling with LFM2.5-Audio.

    Shows how the model can generate tool calls and incorporate results.
    """
    print("=" * 60)
    print("Tool Calling Demo")
    print("=" * 60)

    from datetime import datetime

    # Define tools
    tools = [
        {"name": "get_time", "description": "Get the current time", "parameters": {}},
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {"location": {"type": "string", "description": "City name"}},
        },
    ]

    # Tool executor
    def execute_tool(name: str, args: dict) -> str:
        if name == "get_time":
            return datetime.now().strftime("%H:%M")
        elif name == "get_weather":
            location = args.get("location", "Unknown")
            return f"Weather in {location}: 18°C, sunny"
        return "Unknown tool"

    config = NativeAudioConfig(
        model_id="LiquidAI/LFM2.5-Audio-1.5B",
        backend=NativeModelBackend.TRANSFORMERS,
        device="cpu",
        tools=tools,
    )

    print(f"Available tools: {[t['name'] for t in tools]}")
    print("\nLoading model...")

    try:
        model = load_native_audio_model(config)
    except ImportError as e:
        print(f"Cannot load model: {e}")
        return

    # Simulate text query (since we may not have audio)
    print("\nQuery: 'What time is it?'")

    result = model.generate(
        text_input="What time is it?", mode=GenerationMode.INTERLEAVED
    )

    print(f"\nModel response: {result.text}")

    if result.tool_calls:
        print("\nTool calls detected:")
        for call in result.tool_calls:
            print(f"  - {call['name']}({call.get('arguments', {})})")
            tool_result = execute_tool(call["name"], call.get("arguments", {}))
            print(f"    Result: {tool_result}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmniCore Native Audio Model Runner")
    parser.add_argument("--demo", choices=["conversation", "tools"], help="Run demo mode")
    parser.add_argument("--transcribe", type=str, help="Transcribe audio file (ASR)")
    parser.add_argument("--synthesize", type=str, help="Synthesize text to audio (TTS)")
    parser.add_argument(
        "--output", type=str, default="output.wav", help="Output file path"
    )
    parser.add_argument(
        "--model", type=str, default="LiquidAI/LFM2.5-Audio-1.5B", help="Model ID"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )

    args = parser.parse_args()

    if args.demo == "conversation":
        demo_conversation()
    elif args.demo == "tools":
        demo_tool_calling()
    elif args.transcribe:
        # ASR mode
        config = NativeAudioConfig(model_id=args.model, device=args.device)
        model = load_native_audio_model(config)

        audio = Path(args.transcribe).read_bytes()
        text = model.transcribe(audio)
        print(f"Transcription: {text}")

    elif args.synthesize:
        # TTS mode
        config = NativeAudioConfig(model_id=args.model, device=args.device)
        model = load_native_audio_model(config)

        audio = model.synthesize(args.synthesize)
        Path(args.output).write_bytes(audio)
        print(f"Audio saved to: {args.output}")

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Examples:")
        print("  python native_audio_runner.py --demo conversation")
        print("  python native_audio_runner.py --demo tools")
        print("  python native_audio_runner.py --transcribe audio.wav")
        print('  python native_audio_runner.py --synthesize "Hello world"')
