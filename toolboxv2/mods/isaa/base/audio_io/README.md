# OmniCore Audio Module

**Unified STT (Speech-to-Text) and TTS (Text-to-Speech) interface with multiple backend support.**

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      OMNICORE AUDIO MODULE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────┐         ┌─────────────────────┐          │
│   │        STT          │         │        TTS          │          │
│   │   transcribe()      │         │   synthesize()      │          │
│   │   transcribe_stream │         │   synthesize_stream │          │
│   └──────────┬──────────┘         └──────────┬──────────┘          │
│              │                               │                      │
│   ┌──────────┴──────────┐         ┌──────────┴──────────┐          │
│   │     STTConfig       │         │     TTSConfig       │          │
│   │   - backend         │         │   - backend         │          │
│   │   - model           │         │   - voice           │          │
│   │   - language        │         │   - speed           │          │
│   │   - ...             │         │   - quality         │          │
│   └──────────┬──────────┘         └──────────┬──────────┘          │
│              │                               │                      │
├──────────────┼───────────────────────────────┼──────────────────────┤
│              │       BACKEND LAYER           │                      │
├──────────────┼───────────────────────────────┼──────────────────────┤
│              ▼                               ▼                      │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                    LOCAL BACKENDS                        │      │
│   ├─────────────────────┬───────────────────────────────────┤      │
│   │   faster-whisper    │         Piper TTS                 │      │
│   │   (STT, CPU/GPU)    │         (TTS, CPU)                │      │
│   │                     │         VibeVoice                 │      │
│   │                     │         (TTS, GPU only)           │      │
│   └─────────────────────┴───────────────────────────────────┘      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                     API BACKENDS                         │      │
│   ├─────────────────────┬───────────────────────────────────┤      │
│   │   Groq Whisper      │         Groq TTS (Orpheus)        │      │
│   │   (STT, API)        │         (TTS, API)                │      │
│   │                     │         ElevenLabs                │      │
│   │                     │         (TTS, API)                │      │
│   └─────────────────────┴───────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core module
pip install omnicore-audio

# Local backends
pip install faster-whisper    # STT (local)
pip install piper-tts         # TTS (local, CPU)
pip install vibevoice         # TTS (local, GPU) - requires NVIDIA

# API backends
pip install groq              # Groq STT & TTS
pip install elevenlabs        # ElevenLabs TTS
```

## Quick Start

### Speech-to-Text (STT)

```python
from omnicore_audio import transcribe

# Simplest usage - uses local faster-whisper
result = transcribe("recording.wav")
print(result.text)
```

### Text-to-Speech (TTS)

```python
from omnicore_audio import synthesize

# Simplest usage - uses local Piper
audio = synthesize("Hello, world!")
audio.save("output.wav")
```

## Backend Comparison

### STT Backends

| Backend | Type | Speed | Quality | Privacy | Requirements |
|---------|------|-------|---------|---------|--------------|
| **faster-whisper** | Local | Medium | High | ✅ Full | CPU or CUDA GPU |
| **groq_whisper** | API | ⚡ Fast | High | ❌ Cloud | API Key |

### TTS Backends

| Backend | Type | Speed | Quality | Privacy | Requirements |
|---------|------|-------|---------|---------|--------------|
| **piper** | Local | ⚡ Fast | Good | ✅ Full | CPU |
| **vibevoice** | Local | Medium | Excellent | ✅ Full | NVIDIA GPU 8GB+ |
| **groq_tts** | API | ⚡ Fast | Good | ❌ Cloud | API Key |
| **elevenlabs** | API | Medium | Excellent | ❌ Cloud | API Key + Credits |

## Detailed Usage

### STT with Configuration

```python
from omnicore_audio import transcribe, STTConfig, STTBackend

# Local with specific model
result = transcribe(
    "audio.wav",
    config=STTConfig(
        backend=STTBackend.FASTER_WHISPER,
        model="medium",           # tiny, base, small, medium, large-v3
        language="de",            # German
        device="cpu",             # or "cuda"
        compute_type="int8"       # Quantization for CPU
    )
)

# Groq API (fastest)
result = transcribe(
    audio_bytes,
    config=STTConfig(
        backend=STTBackend.GROQ_WHISPER,
        model="whisper-large-v3-turbo",
        groq_api_key="your-key"   # or set GROQ_API_KEY env var
    )
)

# Access detailed results
print(f"Text: {result.text}")
print(f"Language: {result.language}")
print(f"Duration: {result.duration}s")
for seg in result.segments:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
```

### TTS with Configuration

```python
from omnicore_audio import synthesize, TTSConfig, TTSBackend, TTSQuality

# Piper (local, CPU)
audio = synthesize(
    "Guten Tag, wie geht es Ihnen?",
    config=TTSConfig(
        backend=TTSBackend.PIPER,
        voice="de_DE-thorsten-medium",
        speed=1.0
    )
)

# Groq TTS (API)
audio = synthesize(
    "Fast cloud synthesis.",
    config=TTSConfig(
        backend=TTSBackend.GROQ_TTS,
        voice="Fritz-PlayAI",
        groq_api_key="your-key"
    )
)

# ElevenLabs (highest quality)
audio = synthesize(
    "Professional narration quality.",
    config=TTSConfig(
        backend=TTSBackend.ELEVENLABS,
        voice="21m00Tcm4TlvDq8ikWAM",  # Rachel
        quality=TTSQuality.HIGH,
        elevenlabs_stability=0.7,
        elevenlabs_similarity_boost=0.8
    )
)

# Save or use audio
audio.save("output.wav")
numpy_audio = audio.to_numpy()  # For further processing
```

### Streaming

```python
# STT Streaming
from omnicore_audio import transcribe_stream

def microphone_stream():
    """Your microphone capture logic."""
    while recording:
        yield audio_chunk

for segment in transcribe_stream(microphone_stream()):
    print(f"[{segment.start:.1f}s] {segment.text}")

# TTS Streaming
from omnicore_audio import synthesize_stream

for chunk in synthesize_stream("Long text to speak..."):
    audio_player.write(chunk)
```

### Convenience Functions

```python
from omnicore_audio import (
    transcribe_local,     # faster-whisper shortcut
    transcribe_groq,      # Groq Whisper shortcut
    synthesize_piper,     # Piper shortcut
    synthesize_groq,      # Groq TTS shortcut
    synthesize_elevenlabs # ElevenLabs shortcut
)

# These are equivalent:
result = transcribe_local("audio.wav", model="small")
result = transcribe("audio.wav", config=STTConfig(backend=STTBackend.FASTER_WHISPER, model="small"))
```

## Environment Variables

```bash
# Groq (STT & TTS)
export GROQ_API_KEY="your-groq-api-key"

# ElevenLabs (TTS)
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
```

## Voice Models

### Piper Voices (Local TTS)

Format: `{lang}_{COUNTRY}-{name}-{quality}`

| Voice | Language | Quality |
|-------|----------|---------|
| `en_US-amy-medium` | English (US) | Medium |
| `en_GB-alan-medium` | English (UK) | Medium |
| `de_DE-thorsten-high` | German | High |
| `fr_FR-upmc-medium` | French | Medium |
| `es_ES-carlfm-medium` | Spanish | Medium |

[Full list on Hugging Face](https://huggingface.co/rhasspy/piper-voices)

### Groq TTS Voices

| Voice | Style |
|-------|-------|
| `Fritz-PlayAI` | Professional male |
| `Arsenio-PlayAI` | Casual male |
| `Ava-PlayAI` | Professional female |
| `Zola-PlayAI` | Warm female |
| `Celeste-PlayAI` | Youthful female |

### ElevenLabs Voices

| Voice ID | Name | Style |
|----------|------|-------|
| `21m00Tcm4TlvDq8ikWAM` | Rachel | Narrative |
| `EXAVITQu4vr4xnSDxMaL` | Bella | Young, energetic |
| `ErXwobaYiN019PkySvjV` | Antoni | Professional |
| `MF3mGyEYCl7XYWbV9V6O` | Elli | Emotional |

[Full voice library](https://elevenlabs.io/voice-library)

## CPU-Only Recommendations

For systems without GPU (like your Ryzen server with 48GB RAM):

**STT:**
- Use `faster-whisper` with `device="cpu"` and `compute_type="int8"`
- Model `small` offers best speed/quality balance on CPU
- For high speed: use `groq_whisper` (API)

**TTS:**
- Use `piper` - designed for CPU, very fast
- For higher quality: use `groq_tts` or `elevenlabs` (API)
- ⚠️ Avoid `vibevoice` - requires NVIDIA GPU

## Error Handling

```python
from omnicore_audio import transcribe, check_requirements

# Check if backend is available
status = check_requirements("faster_whisper")
if not status["available"]:
    print(f"Missing: {status['missing']}")

# Handle errors gracefully
try:
    result = transcribe("audio.wav")
except ImportError as e:
    print(f"Backend not installed: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Integration with OmniCore Agent

```python
# Example: Voice-enabled agent function
async def process_voice_input(audio_data: bytes) -> str:
    """STT → Agent → TTS pipeline."""

    # 1. Transcribe input
    stt_result = transcribe(
        audio_data,
        config=STTConfig(
            backend=STTBackend.GROQ_WHISPER,  # Fast API
            language="auto"
        )
    )

    # 2. Process with agent (your logic)
    agent_response = await agent.process(stt_result.text)

    # 3. Synthesize response
    tts_result = synthesize(
        agent_response,
        config=TTSConfig(
            backend=TTSBackend.PIPER,  # Local for privacy
            voice="en_US-amy-medium"
        )
    )

    return tts_result.audio
```

## License

MIT License - see LICENSE file for details.

## Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - SYSTRAN
- [Piper TTS](https://github.com/rhasspy/piper) - Rhasspy
- [VibeVoice](https://github.com/microsoft/VibeVoice) - Microsoft
- [Groq](https://groq.com) - Groq Inc.
- [ElevenLabs](https://elevenlabs.io) - ElevenLabs Inc.
