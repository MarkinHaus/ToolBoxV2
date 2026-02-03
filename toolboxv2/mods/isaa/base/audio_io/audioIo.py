"""
OmniCore Audio I/O Integration
==============================

High-level audio processing functions for the OmniCore Agent.
Provides two main entry points:
- process_audio_raw(): Complete audio file processing
- process_audio_stream(): Real-time audio stream processing

Supports multiple processing pipelines:
1. Pipeline Mode: STT → Agent → TTS (separate components)
2. Native Mode: End-to-end audio LLM (e.g., LFM2.5-Audio)

Author: OmniCore Team
Version: 1.0.0
"""
import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Optional,
    Union,
)

# Import our STT/TTS modules
from .Stt import (
    STTBackend,
    STTConfig,
    STTResult,
    transcribe,
    transcribe_stream,
)
from .Tts import (
    TTSBackend,
    TTSConfig,
    TTSResult,
    synthesize,
    synthesize_stream,
)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

AudioData = Union[bytes, Path, str]
TextProcessor = Callable[[str], str]  # Function that takes text, returns text
AsyncTextProcessor = Callable[[str], "AsyncGenerator[str, None]"]


class ProcessingMode(Enum):
    """Audio processing mode."""

    PIPELINE = "pipeline"  # STT → Process → TTS (separate components)
    NATIVE_AUDIO = "native"  # End-to-end audio model (LFM2.5-Audio, etc.)


class AudioQuality(Enum):
    """Output audio quality preset."""

    FAST = "fast"  # Prioritize speed
    BALANCED = "balanced"  # Balance speed/quality
    HIGH = "high"  # Prioritize quality


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AudioIOConfig:
    """
    Configuration for Audio I/O processing.

    Attributes:
        mode: Processing mode (pipeline or native)
        quality: Output quality preset
        language: Primary language (ISO 639-1)

    Pipeline mode settings:
        stt_config: Configuration for speech-to-text
        tts_config: Configuration for text-to-speech

    Native mode settings:
        native_model: Model identifier for native audio LLM
        native_backend: Backend for native model (transformers, llama_cpp)

    Common settings:
        sample_rate: Audio sample rate (default: 16000)
        channels: Audio channels (default: 1 mono)
        enable_vad: Voice Activity Detection
    """

    mode: ProcessingMode = ProcessingMode.PIPELINE
    quality: AudioQuality = AudioQuality.BALANCED
    language: str = "en"

    # Pipeline mode
    stt_config: Optional[STTConfig] = None
    tts_config: Optional[TTSConfig] = None

    # Native mode
    native_model: str = "LiquidAI/LFM2.5-Audio-1.5B"
    native_backend: str = "transformers"  # or "llama_cpp"
    native_device: str = "cpu"
    native_quantization: Optional[str] = None  # e.g., "Q4_K_M"

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    enable_vad: bool = True

    # Tool calling support
    tools: Optional[list[dict]] = None
    tool_executor: Optional[Callable[[str, dict], str]] = None

    def __post_init__(self):
        # Set default STT config if not provided
        if self.stt_config is None:
            self.stt_config = STTConfig(
                backend=STTBackend.FASTER_WHISPER,
                language=self.language,
                device="cpu",
                compute_type="int8",
            )

        # Set default TTS config if not provided
        if self.tts_config is None:
            self.tts_config = TTSConfig(backend=TTSBackend.PIPER, language=self.language)


@dataclass
class AudioIOResult:
    """
    Result from audio processing.

    Attributes:
        text_input: Transcribed user input
        text_output: Generated response text
        audio_output: Synthesized audio bytes
        tool_calls: List of tool calls made during processing
        metadata: Additional processing metadata
    """

    text_input: str
    text_output: str
    audio_output: Optional[bytes] = None
    tool_calls: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Estimated audio duration in seconds."""
        return self.metadata.get("duration")


# =============================================================================
# PIPELINE MODE PROCESSING
# =============================================================================


async def _process_pipeline_raw(
    audio: AudioData, processor: AsyncTextProcessor, config: AudioIOConfig
) -> AudioIOResult:
    """
    Process audio using STT → Processor → TTS pipeline.

    This is the traditional approach using separate components.
    """
    # Step 1: Speech-to-Text
    stt_result = transcribe(audio, config=config.stt_config)
    text_input = stt_result.text

    # Step 2: Process text (your agent logic)
    text_output = await processor(text_input)

    # Step 3: Text-to-Speech
    tts_result = synthesize(text_output, config=config.tts_config)

    return AudioIOResult(
        text_input=text_input,
        text_output=text_output,
        audio_output=tts_result.audio,
        metadata={
            "mode": "pipeline",
            "stt_language": stt_result.language,
            "stt_duration": stt_result.duration,
            "tts_duration": tts_result.duration,
            "duration": tts_result.duration,
        },
    )


async def _process_pipeline_stream(
    audio_chunks: Generator[bytes, None, None],
    st_processor: AsyncTextProcessor,
    config: AudioIOConfig,
) -> AsyncGenerator[bytes, None]:
    """
    Stream process audio using STT → Processor → TTS pipeline.

    Yields audio chunks as they become available.
    """
    # Accumulate transcribed text
    full_text = []

    for segment in transcribe_stream(audio_chunks, config=config.stt_config):
        full_text.append(segment.text)

    # Process accumulated text
    text_input = " ".join(full_text)
    i = 0
    text_output = ""
    async for res_text_chunk in st_processor(text_input):
        i += 1
        text_output += res_text_chunk
        if i % 3 == 0:
            # Stream TTS output
            for chunk in synthesize_stream(text_output, config=config.tts_config):
                yield chunk
            #yield from await synthesize_stream(text_output, config=config.tts_config)
            text_output = ""

    # Stream TTS output
    for chunk in synthesize_stream(text_output, config=config.tts_config):
        yield chunk


# =============================================================================
# NATIVE AUDIO MODEL PROCESSING
# =============================================================================

# Global model cache to avoid reloading
_native_model_cache: dict = {}


def _load_native_model(config: AudioIOConfig):
    """
    Load native audio model (LFM2.5-Audio or similar).

    Caches model for reuse across calls.
    """
    cache_key = f"{config.native_model}:{config.native_backend}:{config.native_device}"

    if cache_key in _native_model_cache:
        return _native_model_cache[cache_key]

    if config.native_backend == "transformers":
        model, processor = _load_lfm_audio_transformers(config)
    elif config.native_backend == "llama_cpp":
        model, processor = _load_lfm_audio_llama_cpp(config)
    else:
        raise ValueError(f"Unknown native backend: {config.native_backend}")

    _native_model_cache[cache_key] = (model, processor)
    return model, processor


def _load_lfm_audio_transformers(config: AudioIOConfig):
    """Load LFM2.5-Audio using Hugging Face transformers."""
    try:
        import torch
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    except ImportError:
        raise ImportError(
            "liquid_audio not installed. Install with: pip install liquid-audio"
        )

    processor = LFM2AudioProcessor.from_pretrained(config.native_model).eval()
    model = LFM2AudioModel.from_pretrained(config.native_model).eval()

    # Move to appropriate device
    if config.native_device == "cuda":
        import torch

        if torch.cuda.is_available():
            model = model.cuda()

    return model, processor


def _load_lfm_audio_llama_cpp(config: AudioIOConfig):
    """Load LFM2.5-Audio using llama.cpp (GGUF)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        )

    # Determine GGUF path
    gguf_path = config.native_model
    if not gguf_path.endswith(".gguf"):
        # Download from Hugging Face
        from huggingface_hub import hf_hub_download

        quant = config.native_quantization or "Q4_K_M"
        gguf_path = hf_hub_download(
            repo_id=config.native_model, filename=f"*{quant}.gguf", local_dir="./models"
        )

    model = Llama(
        model_path=gguf_path, n_ctx=8192, n_threads=os.cpu_count() or 4, verbose=False
    )

    return model, None  # llama.cpp handles tokenization internally


def _process_native_raw(
    audio: AudioData, processor: Optional[TextProcessor], config: AudioIOConfig
) -> AudioIOResult:
    """
    Process audio using native end-to-end audio model.

    Uses LFM2.5-Audio or similar model that handles
    audio input → understanding → audio output natively.
    """
    import torch
    import torchaudio
    from liquid_audio import ChatState, LFMModality

    model, audio_processor = _load_native_model(config)

    # Load audio
    if isinstance(audio, bytes):
        # Save bytes to temp file for torchaudio
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            audio_path = f.name
        wav, sampling_rate = torchaudio.load(audio_path)
        os.unlink(audio_path)
    else:
        wav, sampling_rate = torchaudio.load(str(audio))

    # Set up chat state
    chat = ChatState(audio_processor)

    # System prompt
    chat.new_turn("system")
    if config.tools:
        # Add tool definitions
        chat.add_text(_format_tools_prompt(config.tools))
    else:
        chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()

    # User input (audio)
    chat.new_turn("user")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()

    # Generate response
    chat.new_turn("assistant")

    text_tokens = []
    audio_tokens = []
    tool_calls = []

    for token in model.generate_interleaved(
        **chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4
    ):
        if token.numel() == 1:
            # Text token
            decoded = audio_processor.text.decode(token)
            text_tokens.append(decoded)

            # Check for tool calls
            full_text = "".join(text_tokens)
            if "<|tool_call_start|>" in full_text and "<|tool_call_end|>" in full_text:
                tool_call = _extract_tool_call(full_text)
                if tool_call and config.tool_executor:
                    result = config.tool_executor(tool_call["name"], tool_call["args"])
                    tool_calls.append({**tool_call, "result": result})
                    # Inject result back
                    chat.add_text(f"<|tool_response_start|>{result}<|tool_response_end|>")
        else:
            # Audio token
            audio_tokens.append(token)

    # Decode audio
    if audio_tokens:
        audio_tensor = torch.cat(audio_tokens)
        audio_output = audio_processor.audio.decode(audio_tensor)
        audio_bytes = _tensor_to_wav_bytes(audio_output, sample_rate=24000)
    else:
        audio_bytes = None

    text_output = "".join(text_tokens)
    # Clean up special tokens
    text_output = _clean_special_tokens(text_output)

    return AudioIOResult(
        text_input="[audio input]",  # Native model doesn't expose transcription
        text_output=text_output,
        audio_output=audio_bytes,
        tool_calls=tool_calls,
        metadata={
            "mode": "native",
            "model": config.native_model,
            "text_tokens": len(text_tokens),
            "audio_tokens": len(audio_tokens),
        },
    )


def _process_native_stream(
    audio_chunks: Generator[bytes, None, None],
    processor: Optional[TextProcessor],
    config: AudioIOConfig,
) -> Generator[bytes, None, None]:
    """
    Stream process audio using native audio model.

    Note: Native models like LFM2.5-Audio support real-time
    interleaved generation for low-latency streaming.
    """
    import torch
    from liquid_audio import ChatState

    model, audio_processor = _load_native_model(config)

    # Accumulate audio chunks
    all_audio = b"".join(audio_chunks)

    # Convert to tensor
    import tempfile

    import torchaudio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(all_audio)
        audio_path = f.name

    wav, sampling_rate = torchaudio.load(audio_path)
    os.unlink(audio_path)

    # Set up chat
    chat = ChatState(audio_processor)
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()

    chat.new_turn("user")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()

    chat.new_turn("assistant")

    # Stream generation
    audio_buffer = []
    buffer_size = 10  # Yield every N audio tokens

    for token in model.generate_interleaved(
        **chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4
    ):
        if token.numel() > 1:  # Audio token
            audio_buffer.append(token)

            if len(audio_buffer) >= buffer_size:
                # Decode and yield chunk
                audio_tensor = torch.cat(audio_buffer)
                audio_chunk = audio_processor.audio.decode(audio_tensor)
                yield _tensor_to_pcm_bytes(audio_chunk)
                audio_buffer = []

    # Yield remaining audio
    if audio_buffer:
        audio_tensor = torch.cat(audio_buffer)
        audio_chunk = audio_processor.audio.decode(audio_tensor)
        yield _tensor_to_pcm_bytes(audio_chunk)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _format_tools_prompt(tools: list[dict]) -> str:
    """Format tools for LFM2.5 tool calling."""
    import json

    return (
        "You are a helpful assistant with access to the following tools:\n"
        f"<|tool_list_start|>{json.dumps(tools)}<|tool_list_end|>\n"
        "When you need to use a tool, output the call between "
        "<|tool_call_start|> and <|tool_call_end|> tokens.\n"
        "Respond with interleaved text and audio."
    )


def _extract_tool_call(text: str) -> Optional[dict]:
    """Extract tool call from text with special tokens."""
    import ast
    import re

    pattern = r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        try:
            # LFM2 uses Python list format for tool calls
            call_str = match.group(1).strip()
            calls = ast.literal_eval(call_str)
            if calls and isinstance(calls, list):
                call = calls[0]
                return {"name": call.get("name"), "args": call.get("arguments", {})}
        except (SyntaxError, ValueError):
            pass

    return None


def _clean_special_tokens(text: str) -> str:
    """Remove LFM2 special tokens from text."""
    import re

    patterns = [
        r"<\|tool_call_start\|>.*?<\|tool_call_end\|>",
        r"<\|tool_response_start\|>.*?<\|tool_response_end\|>",
        r"<\|im_start\|>.*?<\|im_end\|>",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text.strip()


def _tensor_to_wav_bytes(audio_tensor, sample_rate: int = 24000) -> bytes:
    """Convert PyTorch audio tensor to WAV bytes."""
    import io
    import wave

    import numpy as np

    # Convert to numpy
    audio_np = audio_tensor.squeeze().cpu().numpy()

    # Normalize to int16
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Create WAV
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


def _tensor_to_pcm_bytes(audio_tensor) -> bytes:
    """Convert PyTorch audio tensor to raw PCM bytes."""
    import numpy as np

    audio_np = audio_tensor.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


# =============================================================================
# PUBLIC API
# =============================================================================


async def process_audio_raw(
    audio: AudioData,
    processor: AsyncTextProcessor,
    config: Optional[AudioIOConfig] = None,
    **kwargs,
) -> AudioIOResult:
    """
    Process a complete audio file/buffer through the agent.

    This function handles the full pipeline:
    1. Audio input (file, bytes, or path)
    2. Understanding (STT or native audio model)
    3. Processing (your agent logic via processor callback)
    4. Response generation (TTS or native audio model)

    Args:
        audio: Audio input (bytes, file path, or Path object)
        processor: Function that takes user text and returns response text
                   Signature: (str) -> str
        config: AudioIOConfig with all settings
        **kwargs: Override config settings

    Returns:
        AudioIOResult with text and audio outputs

    Examples:
        # Simple usage with pipeline mode
        def my_agent(text: str) -> str:
            if "time" in text.lower():
                return f"The time is {get_current_time()}"
            return "I don't understand"

        result = process_audio_raw(
            "question.wav",
            processor=my_agent
        )
        result.audio_output  # WAV bytes of response

        # Using native audio model (LFM2.5-Audio)
        result = process_audio_raw(
            audio_bytes,
            processor=my_agent,
            config=AudioIOConfig(
                mode=ProcessingMode.NATIVE_AUDIO,
                native_model="LiquidAI/LFM2.5-Audio-1.5B",
                native_device="cpu"
            )
        )

        # With tool calling
        tools = [{
            "name": "get_time",
            "description": "Get current time",
            "parameters": {}
        }]

        def execute_tool(name: str, args: dict) -> str:
            if name == "get_time":
                from datetime import datetime
                return datetime.now().strftime("%H:%M")
            return "Unknown tool"

        result = process_audio_raw(
            "user_question.wav",
            processor=my_agent,
            config=AudioIOConfig(
                mode=ProcessingMode.NATIVE_AUDIO,
                tools=tools,
                tool_executor=execute_tool
            )
        )
    """
    if config is None:
        config = AudioIOConfig(**kwargs)
    elif kwargs:
        # Merge kwargs
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = AudioIOConfig(**config_dict)

    if config.mode == ProcessingMode.PIPELINE:
        return await _process_pipeline_raw(audio, processor, config)
    elif config.mode == ProcessingMode.NATIVE_AUDIO:
        return _process_native_raw(audio, processor, config)
    else:
        raise ValueError(f"Unknown processing mode: {config.mode}")


async def process_audio_stream(
    audio_chunks: Generator[bytes, None, None],
    processor: AsyncTextProcessor,
    config: Optional[AudioIOConfig] = None,
    **kwargs,
) -> AsyncGenerator[bytes, None]:
    """
    Process a stream of audio chunks through the agent.

    Use this for real-time audio processing where you want
    to yield audio output as soon as possible.

    Args:
        audio_chunks: Generator yielding audio byte chunks
        processor: Function that processes transcribed text
        config: AudioIOConfig with all settings
        **kwargs: Override config settings

    Yields:
        Audio bytes chunks for immediate playback

    Examples:
        # Real-time processing with microphone
        def mic_stream():
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1,
                           rate=16000, input=True, frames_per_buffer=1024)
            while recording:
                yield stream.read(1024)

        for audio_chunk in process_audio_stream(
            mic_stream(),
            processor=my_agent
        ):
            speaker.write(audio_chunk)
    """
    if config is None:
        config = AudioIOConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = AudioIOConfig(**config_dict)

    if config.mode == ProcessingMode.PIPELINE:
        async for chunk in _process_pipeline_stream(audio_chunks, processor, config):
            yield chunk
    elif config.mode == ProcessingMode.NATIVE_AUDIO:
        for chunk in _process_native_stream(audio_chunks, None, config):
            yield chunk
    else:
        raise ValueError(f"Unknown processing mode: {config.mode}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def process_audio_pipeline(
    audio: AudioData,
    processor: TextProcessor,
    stt_backend: STTBackend = STTBackend.FASTER_WHISPER,
    tts_backend: TTSBackend = TTSBackend.PIPER,
    language: str = "en",
    **kwargs,
) -> AudioIOResult:
    """
    Process audio using STT → Processor → TTS pipeline.

    Convenience function for pipeline mode.
    """
    config = AudioIOConfig(
        mode=ProcessingMode.PIPELINE,
        language=language,
        stt_config=STTConfig(backend=stt_backend, language=language),
        tts_config=TTSConfig(backend=tts_backend, language=language),
        **kwargs,
    )
    return process_audio_raw(audio, processor, config)


def process_audio_native(
    audio: AudioData,
    processor: Optional[TextProcessor] = None,
    model: str = "LiquidAI/LFM2.5-Audio-1.5B",
    device: str = "cpu",
    tools: Optional[list[dict]] = None,
    tool_executor: Optional[Callable] = None,
    **kwargs,
) -> AudioIOResult:
    """
    Process audio using native end-to-end audio model.

    Convenience function for native audio model mode.
    """
    config = AudioIOConfig(
        mode=ProcessingMode.NATIVE_AUDIO,
        native_model=model,
        native_device=device,
        tools=tools,
        tool_executor=tool_executor,
        **kwargs,
    )
    return process_audio_raw(audio, processor or (lambda x: x), config)


# =============================================================================
# AUDIO STREAMING PLAYER
# =============================================================================

class AudioStreamPlayer:
    """Streamt Audio während der Agent-Antwort generiert wird"""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._playing = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # TTS Config
        self.tts_backend = "groq"  # "groq", "piper", "elevenlabs"
        self.tts_voice = "autumn" # for groq [autumn diana hannah austin daniel troy]
        self.language = "de"

    def _load_tts(self):
        """Lazy load TTS"""
        try:
            from toolboxv2.mods.isaa.base.audio_io.Tts import (
                synthesize, TTSConfig, TTSBackend
            )
            return synthesize, TTSConfig, TTSBackend
        except ImportError:
            return None, None, None

    async def start(self):
        """Startet den Audio Player Worker"""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._player_worker())

    async def stop(self):
        """Stoppt den Audio Player"""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Queue leeren
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except:
                break

    async def queue_text(self, text: str):
        """Fügt Text zur TTS Queue hinzu"""
        if text.strip():
            await self._queue.put(text)

    async def _player_worker(self):
        """Background Worker der Audio abspielt"""
        synthesize, TTSConfig, TTSBackend = self._load_tts()

        if synthesize is None:
            print("[Audio] TTS not available")
            return

        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            print("[Audio] sounddevice not installed: pip install sounddevice")
            return

        while not self._stop_event.is_set():
            try:
                # Warte auf Text (mit Timeout für Stop-Check)
                try:
                    text = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                self._playing = True

                # TTS Config
                if self.tts_backend == "groq":
                    config = TTSConfig(
                        backend=TTSBackend.GROQ_TTS,
                        voice=self.tts_voice,
                        language=self.language,
                    )
                elif self.tts_backend == "elevenlabs":
                    config = TTSConfig(
                        backend=TTSBackend.ELEVENLABS,
                        voice=self.tts_voice,
                    )
                else:
                    config = TTSConfig(
                        backend=TTSBackend.PIPER,
                        voice=self.tts_voice,
                        language=self.language,
                    )

                # Synthesize
                try:
                    result = synthesize(text, config=config)

                    if result and result.audio:
                        # WAV zu numpy
                        import io
                        import wave

                        with io.BytesIO(result.audio) as buf:
                            with wave.open(buf, 'rb') as wav:
                                frames = wav.readframes(wav.getnframes())
                                sample_rate = wav.getframerate()
                                audio_data = np.frombuffer(frames, dtype=np.int16)

                        # Abspielen (non-blocking)
                        sd.play(audio_data, sample_rate)
                        sd.wait()

                except Exception as e:
                    print(f"[Audio] TTS error: {e}")

                self._playing = False
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Audio] Player error: {e}")
                self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing or not self._queue.empty()

# =============================================================================
# MODULE INFO
# =============================================================================

if __name__ == "__main__":
    print("OmniCore Audio I/O Integration")
    print("=" * 50)
    print("\nProcessing modes:")
    for mode in ProcessingMode:
        print(f"  - {mode.value}")

    print("\nUsage example:")
    print("""
    from audio_io import process_audio_raw, AudioIOConfig, ProcessingMode

    # Define your agent processor
    def my_agent(text: str) -> str:
        return f"You said: {text}"

    # Pipeline mode (STT → Agent → TTS)
    result = process_audio_raw("audio.wav", processor=my_agent)

    # Native mode (LFM2.5-Audio)
    result = process_audio_raw(
        "audio.wav",
        processor=my_agent,
        config=AudioIOConfig(
            mode=ProcessingMode.NATIVE_AUDIO,
            native_model="LiquidAI/LFM2.5-Audio-1.5B"
        )
    )

    # Save response audio
    with open("response.wav", "wb") as f:
        f.write(result.audio_output)
    """)
