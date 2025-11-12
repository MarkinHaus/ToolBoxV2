#!/usr/bin/env python3
"""
ToolBox High-Quality Text-to-Speech (TTS) Module
Supports both local offline TTS and high-quality online TTS
API available at http://localhost:8080/api/TTS/{function_name}
"""

import base64
import io
import asyncio
from typing import Optional, Literal

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

from toolboxv2 import App, Result, get_app, RequestData

Name = "TTS"
export = get_app(from_="TTS.EXPORT").tb

# Voice mappings for edge-tts (high quality)
EDGE_VOICES = {
    'de': 'de-DE-KatjaNeural',
    'en': 'en-US-AriaNeural',
    'es': 'es-ES-ElviraNeural',
    'fr': 'fr-FR-DeniseNeural',
    'it': 'it-IT-ElsaNeural',
    'pt': 'pt-BR-FranciscaNeural',
    'ru': 'ru-RU-SvetlanaNeural',
    'ja': 'ja-JP-NanamiNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'ko': 'ko-KR-SunHiNeural',
    'ar': 'ar-SA-ZariyahNeural',
    'hi': 'hi-IN-SwaraNeural',
    'nl': 'nl-NL-ColetteNeural',
    'pl': 'pl-PL-ZofiaNeural',
    'tr': 'tr-TR-EmelNeural',
}


def _local_tts(text: str, lang: str, rate: int = 150) -> Optional[bytes]:
    """
    Generate TTS using local pyttsx3 engine.
    """
    if not PYTTSX3_AVAILABLE:
        return None

    try:
        engine = pyttsx3.init()

        # Set properties
        engine.setProperty('rate', rate)
        engine.setProperty('volume', 1.0)

        # Try to set voice based on language
        voices = engine.getProperty('voices')
        for voice in voices:
            if lang.lower() in voice.languages or lang.lower() in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break

        # Save to memory buffer
        audio_fp = io.BytesIO()
        temp_file = "temp_tts.mp3"

        engine.save_to_file(text, temp_file)
        engine.runAndWait()

        # Read the file and convert to bytes
        with open(temp_file, 'rb') as f:
            audio_bytes = f.read()

        # Clean up temp file
        import os
        os.remove(temp_file)

        return audio_bytes

    except Exception as e:
        return None


async def _edge_tts(text: str, lang: str) -> Optional[bytes]:
    """
    Generate high-quality TTS using edge-tts.
    """
    if not EDGE_TTS_AVAILABLE:
        return None

    try:
        # Get appropriate voice for language
        voice = EDGE_VOICES.get(lang, 'en-US-AriaNeural')

        # Create TTS object
        communicate = edge_tts.Communicate(text, voice)

        # Generate audio to memory
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])

        audio_data.seek(0)
        return audio_data.read()

    except Exception as e:
        return None


@export(mod_name=Name, api=True)
async def speak(
    app: App,
    text: str = "",
    lang: str = 'de',
    engine: Literal['auto', 'edge', 'local'] = 'auto',
    rate: int = 150
) -> Result:
    """
    Converts text to high-quality speech and returns it as a base64 encoded audio string.

    Args:
        app: Application instance
        text: Text to convert to speech
        lang: Language code (de, en, es, fr, it, pt, ru, ja, zh, ko, ar, hi, nl, pl, tr)
        engine: TTS engine to use ('auto', 'edge' for high quality online, 'local' for offline)
        rate: Speech rate for local engine (words per minute, default 150)

    Returns:
        Result object with base64 encoded audio
    """
    if not text:
        return Result.default_user_error("Text to speak cannot be empty.")

    audio_bytes = None
    used_engine = None

    try:
        # Auto mode: Try edge-tts first, fallback to local
        if engine == 'auto' or engine == 'edge':
            if EDGE_TTS_AVAILABLE:
                app.logger.info("Attempting to use edge-tts for high-quality output")
                audio_bytes = await _edge_tts(text, lang)

                if audio_bytes:
                    used_engine = 'edge-tts'
                    app.logger.info("Successfully generated speech using edge-tts")

            if not audio_bytes and engine == 'edge':
                return Result.default_internal_error(
                    "edge-tts not available or failed. Install with: pip install edge-tts"
                )

        # Fallback to local or explicit local request
        if not audio_bytes and (engine == 'auto' or engine == 'local'):
            if PYTTSX3_AVAILABLE:
                app.logger.info("Using local pyttsx3 engine")
                audio_bytes = _local_tts(text, lang, rate)
                if audio_bytes:
                    used_engine = 'pyttsx3'
                    app.logger.info("Successfully generated speech using pyttsx3")

            if not audio_bytes and engine == 'local':
                return Result.default_internal_error(
                    "pyttsx3 not available or failed. Install with: pip install pyttsx3"
                )

        # If no engine worked
        if not audio_bytes:
            return Result.default_internal_error(
                "No TTS engine available. Install edge-tts (pip install edge-tts) "
                "or pyttsx3 (pip install pyttsx3)"
            )

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return Result.ok(data={
            'audio_content': audio_base64,
            'format': 'mp3',
            'engine': used_engine,
            'language': lang,
            'text_length': len(text)
        })

    except Exception as e:
        app.logger.error(f"TTS generation failed: {e}")
        return Result.default_internal_error(f"Failed to generate speech: {e}")


@export(mod_name=Name, api=True)
def list_voices(app: App, engine: str = 'edge') -> Result:
    """
    Lists available voices for the specified engine.

    Args:
        app: Application instance
        engine: Engine to list voices for ('edge' or 'local')

    Returns:
        Result object with available voices
    """
    try:
        if engine == 'edge':
            if not EDGE_TTS_AVAILABLE:
                return Result.default_user_error(
                    "edge-tts not available. Install with: pip install edge-tts"
                )

            return Result.ok(data={
                'engine': 'edge-tts',
                'voices': EDGE_VOICES,
                'note': 'Language codes map to neural voices for natural speech'
            })

        elif engine == 'local':
            if not PYTTSX3_AVAILABLE:
                return Result.default_user_error(
                    "pyttsx3 not available. Install with: pip install pyttsx3"
                )

            engine_inst = pyttsx3.init()
            voices = engine_inst.getProperty('voices')

            voice_list = [
                {
                    'id': v.id,
                    'name': v.name,
                    'languages': v.languages
                }
                for v in voices
            ]

            return Result.ok(data={
                'engine': 'pyttsx3',
                'voices': voice_list,
                'count': len(voice_list)
            })

        else:
            return Result.default_user_error("Invalid engine. Use 'edge' or 'local'")

    except Exception as e:
        app.logger.error(f"Failed to list voices: {e}")
        return Result.default_internal_error(f"Failed to list voices: {e}")


@export(mod_name=Name, api=True)
def get_engine_status(app: App) -> Result:
    """
    Check which TTS engines are available.

    Returns:
        Result object with engine availability status
    """
    return Result.ok(data={
        'edge_tts': {
            'available': EDGE_TTS_AVAILABLE,
            'install_command': 'pip install edge-tts',
            'quality': 'High (neural voices)',
            'online': True
        },
        'pyttsx3': {
            'available': PYTTSX3_AVAILABLE,
            'install_command': 'pip install pyttsx3',
            'quality': 'Medium (system voices)',
            'online': False
        }
    })
