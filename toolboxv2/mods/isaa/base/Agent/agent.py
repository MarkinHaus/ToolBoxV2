import os
import random
import re
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import wraps
from typing import List, Callable, AsyncGenerator

import yaml

from toolboxv2.mods.isaa.base.IntelligentRateLimiter.intelligent_rate_limiter import (
    LiteLLMRateLimitHandler,
    load_handler_from_file,
    create_handler_from_config,
)
from toolboxv2.mods.isaa.base.tbpocketflow import AsyncFlow, AsyncNode

from pydantic import ValidationError

from toolboxv2.mods.isaa.base.Agent.chain import CF, IS, Chain, ConditionalChain

from toolboxv2.utils.extras.Style import Spinner, print_prompt

# Framework imports with graceful degradation
try:
    import litellm
    from litellm import BudgetManager, Usage
    from litellm.utils import get_max_tokens
    LITELLM_AVAILABLE = True
    # prin litllm version


    def get_litellm_version():
        version = None
        try:
            import importlib.metadata
            version = importlib.metadata.version("litellm")
        except importlib.metadata.PackageNotFoundError:
            version = None
        except Exception as e:
            version = None
        return version
    print(f"INFO: LiteLLM version {get_litellm_version()} found.")
except ImportError:
    LITELLM_AVAILABLE = False
    class BudgetManager: pass
    def get_max_tokens(*a, **kw): return 4096

try:
    from python_a2a import A2AClient, A2AServer, AgentCard
    from python_a2a import run_server as run_a2a_server_func
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    class A2AServer: pass
    class A2AClient: pass
    class AgentCard: pass

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    class FastMCP: pass

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    class TracerProvider: pass

from toolboxv2 import get_logger
from toolboxv2.mods.isaa.base.Agent.types import *

logger = get_logger()

AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"
rprint = print if AGENT_VERBOSE else lambda *a, **k: None
wprint = print if AGENT_VERBOSE else lambda *a, **k: None


def safe_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: safe_for_yaml(v) for k, v in obj.items()}
    # remove locks completely
    if hasattr(obj, 'acquire'):
        return "<RLock omitted>"
    # convert unknown objects to string
    try:
        yaml.dump(obj)
        return obj
    except Exception:
        return str(obj)

# ===== MEDIA PARSING UTILITIES =====

# Supported media prefixes and their types
MEDIA_PREFIXES = {
    'media': 'auto',           # Auto-detect from extension
    'image': 'image',          # Explicit image
    'audio': 'audio',          # Explicit audio
    'video': 'video',          # Explicit video
    'file': 'file',            # Generic file
    'pdf': 'pdf',              # PDF document
    'voice_message_transcription': 'transcription',  # Voice message transcription (text)
}

def parse_media_from_query(query: str, model: str | None = None) -> tuple[str, list[dict]]:
    """
    Parse [prefix:(path/url/content)] tags from query and convert to litellm vision format

    Supports multiple prefixes:
        - [media:path] - Auto-detect type from extension
        - [image:url] - Explicit image
        - [audio:url] - Explicit audio file
        - [video:url] - Explicit video file
        - [file:url] - Generic file attachment
        - [pdf:url] - PDF document
        - [voice_message_transcription: text] - Transcribed voice message (kept as text)

    Args:
        query: Text query that may contain [prefix:(path/url)] tags
        model: Optional model name for capability checks (e.g., "gpt-4-vision-preview")

    Returns:
        tuple: (cleaned_query, media_list)
            - cleaned_query: Query with media tags removed (transcriptions kept inline)
            - media_list: List of dicts in litellm vision format

    Examples:
        >>> parse_media_from_query("Analyze [image:photo.jpg] this image")
        ("Analyze  this image", [{"type": "image_url", "image_url": {"url": "photo.jpg", "format": "image/jpeg"}}])

        >>> parse_media_from_query("User said: [voice_message_transcription: Hello world]")
        ("User said: Hello world", [])

    Note:
        litellm uses the OpenAI vision format: {"type": "image_url", "image_url": {"url": "...", "format": "..."}}
        The "format" field is optional but recommended for explicit MIME type specification.
    """
    media_list = []
    cleaned_query = query

    # Check model capabilities if model is provided
    model_capabilities = _get_model_capabilities(model) if model else {}

    # Process each prefix type
    for prefix, prefix_type in MEDIA_PREFIXES.items():
        # Pattern for this prefix: [prefix:content]
        pattern = rf'\[{prefix}:\s*([^\]]+)\]'
        matches = re.findall(pattern, cleaned_query, re.IGNORECASE)

        for content in matches:
            content = content.strip()

            if prefix_type == 'transcription':
                # Voice message transcription - replace tag with the transcribed text inline
                # This keeps the transcription as part of the text, not as media
                tag_pattern = rf'\[{prefix}:\s*{re.escape(content)}\]'
                cleaned_query = re.sub(tag_pattern, content, cleaned_query, flags=re.IGNORECASE)
                continue

            # Determine actual media type
            if prefix_type == 'auto':
                media_type = _detect_media_type(content)
            else:
                media_type = prefix_type

            # Check model capability for this media type
            if model_capabilities:
                if media_type == 'image' and not model_capabilities.get('supports_vision', True):
                    wprint(f"Warning: Model '{model}' may not support vision/images.")
                elif media_type == 'audio' and not model_capabilities.get('supports_audio', False):
                    wprint(f"Warning: Model '{model}' may not support audio input.")
                elif media_type == 'video' and not model_capabilities.get('supports_video', False):
                    wprint(f"Warning: Model '{model}' may not support video input.")
                elif media_type == 'pdf' and not model_capabilities.get('supports_pdf', False):
                    wprint(f"Warning: Model '{model}' may not support PDF input.")

            # Build media entry in litellm format
            if media_type == "image":
                mime_type = _get_image_mime_type(content)
                image_obj = {"url": content}
                if mime_type:
                    image_obj["format"] = mime_type
                media_list.append({
                    "type": "image_url",
                    "image_url": image_obj
                })
            elif media_type == "audio":
                # Audio - some models support this via input_audio
                media_list.append({
                    "type": "input_audio",
                    "input_audio": {"url": content, "format": _get_audio_format(content)}
                })
            elif media_type == "video":
                # Video - limited model support
                media_list.append({
                    "type": "video_url",
                    "video_url": {"url": content}
                })
            elif media_type == "pdf":
                # PDF - some models support document input
                media_list.append({
                    "type": "document_url",
                    "document_url": {"url": content, "format": "application/pdf"}
                })
            elif media_type == "file":
                # Generic file - try to detect type
                detected = _detect_media_type(content)
                if detected == "image":
                    mime_type = _get_image_mime_type(content)
                    image_obj = {"url": content}
                    if mime_type:
                        image_obj["format"] = mime_type
                    media_list.append({"type": "image_url", "image_url": image_obj})
                else:
                    # Unknown file type - add as generic
                    media_list.append({
                        "type": "file_url",
                        "file_url": {"url": content}
                    })
            else:
                # Unknown type - try as image (most compatible)
                media_list.append({
                    "type": "image_url",
                    "image_url": {"url": content}
                })

        # Remove processed tags from query (except transcriptions which are already handled)
        if prefix_type != 'transcription':
            cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

    # Clean up extra whitespace
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    return cleaned_query, media_list


def _get_model_capabilities(model: str) -> dict:
    """
    Get capability flags for a model

    Args:
        model: Model name (e.g., "gpt-4-vision-preview", "gemini-1.5-pro")

    Returns:
        dict with capability flags:
            - supports_vision: bool
            - supports_audio: bool
            - supports_video: bool
            - supports_pdf: bool
    """
    model_lower = model.lower() if model else ""

    # Default capabilities
    capabilities = {
        'supports_vision': False,
        'supports_audio': False,
        'supports_video': False,
        'supports_pdf': False,
    }

    # Vision models
    vision_models = ['vision', 'gpt-4o', 'gpt-4-turbo', 'claude-3', 'gemini', 'llava', 'cogvlm'] # TODO: add config
    if any(vm in model_lower for vm in vision_models):
        capabilities['supports_vision'] = True

    # Audio models (Gemini 1.5+, GPT-4o audio)
    audio_models = ['gemini-1.5', 'gemini-2', 'gpt-4o-audio', 'whisper']
    if any(am in model_lower for am in audio_models):
        capabilities['supports_audio'] = True

    # Video models (Gemini 1.5+)
    video_models = ['gemini-1.5', 'gemini-2']
    if any(vm in model_lower for vm in video_models):
        capabilities['supports_video'] = True

    # PDF models (Claude 3, Gemini)
    pdf_models = ['claude-3', 'gemini']
    if any(pm in model_lower for pm in pdf_models):
        capabilities['supports_pdf'] = True

    return capabilities


def _get_audio_format(path: str) -> str:
    """Get audio format from file extension"""
    path_lower = path.lower()
    format_map = {
        '.mp3': 'mp3',
        '.wav': 'wav',
        '.ogg': 'ogg',
        '.m4a': 'm4a',
        '.flac': 'flac',
        '.aac': 'aac',
        '.webm': 'webm',
    }
    for ext, fmt in format_map.items():
        if path_lower.endswith(ext):
            return fmt
    return 'wav'  # Default


def _detect_media_type(path: str) -> str:
    """Detect media type from file extension or URL"""
    path_lower = path.lower()

    # Image extensions
    if any(path_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']):
        return "image"

    # Audio extensions
    if any(path_lower.endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']):
        return "audio"

    # Video extensions
    if any(path_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']):
        return "video"

    # PDF
    if path_lower.endswith('.pdf'):
        return "pdf"

    return "unknown"


def _get_image_mime_type(path: str) -> str:
    """
    Get MIME type for image based on file extension

    Args:
        path: Image file path or URL

    Returns:
        str: MIME type (e.g., "image/jpeg") or empty string if unknown
    """
    path_lower = path.lower()

    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.ico': 'image/x-icon'
    }

    for ext, mime in mime_map.items():
        if path_lower.endswith(ext):
            return mime

    return ""


eprint = print if AGENT_VERBOSE else lambda *a, **k: None
iprint = print if AGENT_VERBOSE else lambda *a, **k: None

TASK_TYPES = ["llm_call", "tool_call", "analysis", "generic"]


import functools

import json
import pickle
from typing import Any

# TODO Chapint maby deleat
def _is_json_serializable(obj: Any) -> bool:
    """Prüft, ob ein Objekt sicher nach JSON serialisiert werden kann."""
    if obj is None or isinstance(obj, (str, int, float, bool, list, dict)):
        try:
            # Der schnellste und sicherste Test
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False
    return False

def _clean_data_for_serialization(data: Any) -> Any:
    """
    Bereinigt rekursiv Dictionaries und Listen, um nur sicher serialisierbare
    Werte beizubehalten.
    """
    if isinstance(data, dict):
        clean_dict = {}
        for k, v in data.items():
            # Überspringe bekanntermaßen nicht serialisierbare Schlüssel und Instanzen
            if isinstance(v, (types.FunctionType, types.ModuleType, threading.Thread, FlowAgent, AsyncNode)):
                continue
            if _is_json_serializable(v):
                clean_dict[k] = _clean_data_for_serialization(v)
        return clean_dict
    elif isinstance(data, list):
        clean_list = []
        for item in data:
            if isinstance(item, (types.FunctionType, types.ModuleType, threading.Thread, FlowAgent, AsyncNode)):
                continue
            if _is_json_serializable(item):
                clean_list.append(_clean_data_for_serialization(item))
        return clean_list
    else:
        return data

# Annahme: Die folgenden Klassen sind bereits definiert
# from your_project import AsyncNode, ProgressEvent, NodeStatus

# --- Dies ist der wiederverwendbare "Autohook"-Dekorator ---
def with_progress_tracking(cls):
    """
    Ein Klassendekorator, der die Methoden run_async, prep_async, exec_async,
    und exec_fallback_async automatisch mit umfassendem Progress-Tracking umwickelt.
    """

    # --- Wrapper für run_async ---
    original_run = getattr(cls, 'run_async', None)
    if original_run:
        @functools.wraps(original_run)
        async def wrapped_run_async(self, shared):
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_run(self, shared)

            timer_key = f"{node_name}_total"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_enter",
                timestamp=time.time(),
                node_name=node_name,
                session_id=shared.get("session_id"),
                task_id=shared.get("current_task_id"),
                plan_id=shared.get("current_plan", TaskPlan(id="none", name="none", description="none")).id if shared.get("current_plan") else None,
                status=NodeStatus.RUNNING,
                success=None
            ))

            try:
                # Hier wird die ursprüngliche Methode aufgerufen
                result = await original_run(self, shared)

                total_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_exit",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    node_duration=total_duration,
                    routing_decision=result,
                    session_id=shared.get("session_id"),
                    task_id=shared.get("current_task_id"),
                    metadata={"success": True}
                ))

                return result
            except Exception as e:
                total_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    node_duration=total_duration,
                    session_id=shared.get("session_id"),
                    metadata={"error": str(e), "error_type": type(e).__name__}
                ))
                raise

        cls.run_async = wrapped_run_async

    # --- Wrapper für prep_async ---
    original_prep = getattr(cls, 'prep_async', None)
    if original_prep:
        @functools.wraps(original_prep)
        async def wrapped_prep_async(self, shared):
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_prep(self, shared)
            timer_key = f"{node_name}_total_p"
            progress_tracker.start_timer(timer_key)
            timer_key = f"{node_name}_prep"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.STARTING,
                node_phase="prep",
                session_id=shared.get("session_id")
            ))

            try:
                result = await original_prep(self, shared)

                prep_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_phase",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    success=True,
                    node_name=node_name,
                    node_phase="prep_complete",
                    node_duration=prep_duration,
                    session_id=shared.get("session_id")
                ))
                return result
            except Exception as e:
                progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    metadata={"error": str(e), "error_type": type(e).__name__},
                    node_phase="prep_failed"
                ))
                raise


        cls.prep_async = wrapped_prep_async

    # --- Wrapper für exec_async ---
    original_exec = getattr(cls, 'exec_async', None)
    if original_exec:
        @functools.wraps(original_exec)
        async def wrapped_exec_async(self, prep_res):
            progress_tracker = prep_res.get("progress_tracker") if isinstance(prep_res, dict) else None
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_exec(self, prep_res)

            timer_key = f"{node_name}_exec"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.RUNNING,
                node_phase="exec",
                session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None
            ))

            # In exec gibt es normalerweise keine Fehlerbehandlung, da diese von run_async übernommen wird
            result = await original_exec(self, prep_res)

            exec_duration = progress_tracker.end_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.RUNNING,
                success=True,
                node_phase="exec_complete",
                node_duration=exec_duration,
                session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None
            ))
            return result

        cls.exec_async = wrapped_exec_async

    # --- Wrapper für post_async ---
    original_post = getattr(cls, 'post_async', None)
    if original_post:
        @functools.wraps(original_post)
        async def wrapped_post_async(self, shared, prep_res, exec_res):
            if isinstance(exec_res, str):
                print("exec_res is string:", exec_res)
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_post(self, shared, prep_res, exec_res)

            timer_key_post = f"{node_name}_post"
            progress_tracker.start_timer(timer_key_post)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.COMPLETING,  # Neue Phase "completing"
                node_phase="post",
                session_id=shared.get("session_id")
            ))

            try:
                # Die eigentliche post_async Methode aufrufen
                result = await original_post(self, shared, prep_res, exec_res)

                post_duration = progress_tracker.end_timer(timer_key_post)
                total_duration = progress_tracker.end_timer(f"{node_name}_total_p")  # Gesamtdauer stoppen

                # Sende das entscheidende "node_exit" Event nach erfolgreicher post-Phase
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_exit",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    node_duration=total_duration,
                    routing_decision=result,
                    session_id=shared.get("session_id"),
                    task_id=shared.get("current_task_id"),
                    metadata={
                        "success": True,
                        "post_duration": post_duration
                    }
                ))

                return result
            except Exception as e:
                # Fehler in der post-Phase

                post_duration = progress_tracker.end_timer(timer_key_post)
                total_duration = progress_tracker.end_timer(f"{node_name}_total")
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    node_duration=total_duration,
                    metadata={"error": str(e), "error_type": type(e).__name__, "phase": "post"},
                    node_phase="post_failed"
                ))
                raise

        cls.post_async = wrapped_post_async

    # --- Wrapper für exec_fallback_async ---
    original_fallback = getattr(cls, 'exec_fallback_async', None)
    if original_fallback:
        @functools.wraps(original_fallback)
        async def wrapped_fallback_async(self, prep_res, exc):
            progress_tracker = prep_res.get("progress_tracker") if isinstance(prep_res, dict) else None
            node_name = self.__class__.__name__

            if progress_tracker:
                timer_key = f"{node_name}_exec"
                exec_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_phase",
                    timestamp=time.time(),
                    node_name=node_name,
                    node_phase="exec_fallback",
                    node_duration=exec_duration,
                    status=NodeStatus.FAILED,
                    success=False,
                    session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None,
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                ))

            return await original_fallback(self, prep_res, exc)

        cls.exec_fallback_async = wrapped_fallback_async

    return cls


# ============================================================================
# META-TOOL DEFINITIONS - Native Function Calling Format
# ============================================================================

def get_meta_tools_schema(available_tools: list[str] = None) -> list[dict]:
    """
    Generate meta-tools schema dynamically.
    This prevents tool name confusion by being explicit.
    """
    tools_hint = ""
    if available_tools:
        tools_hint = f" Available: {', '.join(available_tools[:15])}"

    return [
        {
            "type": "function",
            "function": {
                "name": "reason",
                "description": "Think through a problem step by step. Use this before taking action. Always follow with 'run_tools' or 'finish'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your step-by-step reasoning",
                        }
                    },
                    "required": ["thought"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tools",
                "description": f"Execute external tools to accomplish a task.{tools_hint}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "What needs to be done (be specific)",
                        },
                        "tool_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Which tools to use for this task",
                        },
                    },
                    "required": ["task_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Complete the task with a final answer. Call this when you have all information needed to respond to the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_answer": {
                            "type": "string",
                            "description": "Your complete response to the user",
                        }
                    },
                    "required": ["final_answer"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save",
                "description": "Save a value for later use.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Key name"},
                        "data": {"type": "string", "description": "Value to save"},
                    },
                    "required": ["name", "data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load",
                "description": "Load a previously saved value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Key name to load"}
                    },
                    "required": ["name"],
                },
            },
        },
    ]


# Legacy constant for backwards compatibility
META_TOOLS_SCHEMA = get_meta_tools_schema()


# ============================================================================
# SYSTEM PROMPT - Klar und direkt
# ============================================================================

SYSTEM_PROMPT = """You are the STRATEGIC REASONING CORE of an autonomous agent.
Your job is NOT to guess, but to ORCHESTRATE the solution.

## AVAILABLE TOOLS & CAPABILITIES:
{tool_descriptions}

## CRITICAL RULES:
1. **NO HALLUCINATIONS**: If the user asks for current data (weather, prices, news, docs) or specific calculations, you MUST use `run_tools`. Do NOT answer from your internal training data.
2. **DELEGATE**: If a tool in the list above matches the user's intent, use `run_tools` immediately. Do not try to simulate the tool.
3. **VERIFY**: Check the `PREVIOUS TOOL RESULT` in context. If it contains the answer, use `finish`. If it's an error, try a different approach.
4. **ITERATE**: If the task is complex, break it down. Use `reason` to plan, then `run_tools` for the first step.

## STATE:
Current Loop: {loop_count}/{max_loops}

{context}

User Query: {query}
"""


# ============================================================================
# REASONER NODE - Lean Implementation
# ============================================================================

from dataclasses import dataclass, field

@dataclass
class ReasoningState:
    """Minimal state tracking"""
    loop_count: int = 0
    results: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    final_answer: Optional[str] = None
    completed: bool = False

# ===== VOTING ======



# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the minimalist orchestrator."""
    max_loops: int = 25
    max_history_messages: int = 30
    summary_batch_size: int = 10
    vfs_max_window_lines: int = 250
    decision_temperature: float = 0.7
    tool_temperature: float = 0.5
    enable_vfs: bool = True
    enable_compression: bool = True
    fast_model_preference: str = "fast"
    complex_model_preference: str = "complex"


# =============================================================================
# 1. VIRTUAL FILE SYSTEM (VFS)
# =============================================================================

@dataclass
class VFSFile:
    """Represents a file in the Virtual File System."""
    filename: str
    content: str
    state: str = "closed"  # "open" or "closed"
    view_start: int = 0
    view_end: int = -1
    mini_summary: str = ""
    readonly: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class VirtualFileSystem:
    """
    Virtual File System for token-efficient file management.

    Features:
    - open/closed states (only open files show content in context)
    - Windowing (show only specific line ranges)
    - System files (read-only, auto-updated)
    - Auto-summary on close via LLM
    """

    def __init__(self, agent_instance: Any = None, max_window_lines: int = 50):
        self.files: typing.Dict[str, VFSFile] = {}
        self.agent = agent_instance
        self.max_window_lines = max_window_lines
        self._init_system_files()

    def _init_system_files(self):
        """Initialize read-only system files."""
        self.files["system_context"] = VFSFile(
            filename="system_context",
            content=self._get_system_context(),
            state="open",
            readonly=True
        )
        self.files["agent_memory.txt"] = VFSFile(
            filename="agent_memory.txt",
            content="# Agent Long-Term Memory\n\n[No memories stored yet]\n",
            state="closed",
            readonly=False
        )

    def _get_system_context(self) -> str:
        """Generate current system context."""
        now = datetime.now()
        agent_name = "FlowAgent"
        session = "default"
        user_id = "anonymous"

        if self.agent:
            if hasattr(self.agent, 'amd') and hasattr(self.agent.amd, 'name'):
                agent_name = self.agent.amd.name
            session = getattr(self.agent, 'active_session', 'default') or 'default'
            if hasattr(self.agent, 'shared'):
                user_id = self.agent.shared.get('user_id', 'anonymous')

        return f"""# System Context
Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
Agent: {agent_name}
Session: {session}
User: {user_id}
"""

    def _update_system_context(self):
        """Refresh system context file."""
        if "system_context" in self.files:
            self.files["system_context"].content = self._get_system_context()
            self.files["system_context"].updated_at = datetime.now().isoformat()

    # -------------------------------------------------------------------------
    # VFS Tool Methods
    # -------------------------------------------------------------------------

    def create(self, filename: str, content: str = "") -> dict:
        """Create a new file."""
        if filename in self.files and self.files[filename].readonly:
            return {"success": False, "error": f"Cannot overwrite system file: {filename}"}

        self.files[filename] = VFSFile(filename=filename, content=content, state="closed")
        return {"success": True, "message": f"Created '{filename}' ({len(content)} chars)"}

    def edit(self, filename: str, line_start: int, line_end: int, new_content: str) -> dict:
        """Edit file by replacing lines (1-indexed)."""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        lines = f.content.split('\n')
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)

        new_lines = new_content.split('\n')
        lines = lines[:start_idx] + new_lines + lines[end_idx:]

        f.content = '\n'.join(lines)
        f.updated_at = datetime.now().isoformat()

        return {"success": True, "message": f"Edited {filename} lines {line_start}-{line_end}"}

    def open(self, filename: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open file (make content visible)."""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        f.state = "open"
        f.view_start = max(0, line_start - 1)
        f.view_end = line_end

        lines = f.content.split('\n')
        end = line_end if line_end > 0 else len(lines)
        visible = lines[f.view_start:end]

        return {
            "success": True,
            "message": f"Opened '{filename}' (lines {line_start}-{end})",
            "preview": '\n'.join(visible[:5]) + ("..." if len(visible) > 5 else "")
        }

    async def close(self, filename: str) -> dict:
        """Close file (create summary, remove from context)."""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Cannot close system file: {filename}"}

        # Generate summary if agent available and content is substantial
        if self.agent and len(f.content) > 100:
            try:
                prompt = f"Summarize in 1-2 sentences:\n\n{f.content[:2000]}"
                summary = await self.agent.a_run_llm_completion(
                    model_preference="fast",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3,
                    with_context=False,
                    task_id=f"vfs_summarize_{filename}"
                )
                f.mini_summary = summary.strip()
            except Exception as e:
                f.mini_summary = f"[{len(f.content)} chars]"
        else:
            f.mini_summary = f"[{len(f.content)} chars]"

        f.state = "closed"
        return {"success": True, "summary": f.mini_summary}

    async def extract_info(self, filename: str, query: str) -> dict:
        """Query file without loading into context."""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        if not self.agent:
            return {"success": False, "error": "No agent for extraction"}

        try:
            prompt = f"File: {filename}\nContent:\n{self.files[filename].content[:4000]}\n\nQuery: {query}\n\nAnswer:"
            result = await self.agent.a_run_llm_completion(
                model_preference="fast",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                with_context=False,
                task_id=f"vfs_extract_{filename}"
            )
            return {"success": True, "result": result.strip()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def view(self, filename: str, line_start: int = 1, line_end: int = -1) -> dict:
        """View/adjust visible window."""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.state != "open":
            self.open(filename, line_start, line_end)
        else:
            f.view_start = max(0, line_start - 1)
            f.view_end = line_end

        lines = f.content.split('\n')
        end = line_end if line_end > 0 else len(lines)

        return {
            "success": True,
            "content": '\n'.join(lines[f.view_start:end])
        }

    def list_files(self) -> dict:
        """List all files."""
        listing = []
        for name, f in self.files.items():
            info = {
                "filename": name,
                "state": f.state,
                "readonly": f.readonly,
                "size": len(f.content),
                "lines": len(f.content.split('\n'))
            }
            if f.state == "closed" and f.mini_summary:
                info["summary"] = f.mini_summary
            listing.append(info)
        return {"success": True, "files": listing}

    def cp_local_to_vfs(self, local_path: str, vfs_path: str) -> dict:
        """Copy local file to VFS."""
        if not os.path.exists(local_path):
            return {"success": False, "error": f"Not found: {local_path}"}
        try:
            with open(local_path, 'r', encoding='utf-8', errors='replace') as f:
                return self.create(vfs_path, f.read())
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_vfs_to_local(self, vfs_path: str, local_path: str) -> dict:
        """Save VFS file to local filesystem."""
        if vfs_path not in self.files:
            return {"success": False, "error": f"Not found: {vfs_path}"}
        try:
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(self.files[vfs_path].content)
            return {"success": True, "message": f"Saved to {local_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------

    def build_context_string(self) -> str:
        """Build VFS context string for LLM."""
        self._update_system_context()

        parts = ["=== VFS (Virtual File System) ==="]

        for name, f in self.files.items():
            if f.state == "open":
                lines = f.content.split('\n')
                end = f.view_end if f.view_end > 0 else len(lines)
                visible = lines[f.view_start:end]

                # Limit window size
                if len(visible) > self.max_window_lines:
                    visible = visible[:self.max_window_lines]
                    parts.append(
                        f"\n[{name}] OPEN (lines {f.view_start + 1}-{f.view_start + self.max_window_lines}, truncated):")
                else:
                    parts.append(f"\n[{name}] OPEN (lines {f.view_start + 1}-{end}):")
                parts.append('\n'.join(visible))
            else:
                summary = f.mini_summary or f"[{len(f.content)} chars]"
                parts.append(f"\n• {name} [closed]: {summary}")

        return '\n'.join(parts)

    def get_vfs_tools_schema(self) -> List[dict]:
        """Get VFS tools in LiteLLM format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "vfs_create",
                    "description": "Create file in VFS",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string", "default": ""}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_edit",
                    "description": "Edit file lines",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "line_start": {"type": "integer"},
                            "line_end": {"type": "integer"},
                            "new_content": {"type": "string"}
                        },
                        "required": ["filename", "line_start", "line_end", "new_content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_open",
                    "description": "Open file to show in context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "line_start": {"type": "integer", "default": 1},
                            "line_end": {"type": "integer", "default": -1}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_close",
                    "description": "Close file (auto-summarizes)",
                    "parameters": {
                        "type": "object",
                        "properties": {"filename": {"type": "string"}},
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_extract",
                    "description": "Query file without loading to context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "query": {"type": "string"}
                        },
                        "required": ["filename", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_view",
                    "description": "View/adjust file window",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "line_start": {"type": "integer", "default": 1},
                            "line_end": {"type": "integer", "default": -1}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_list",
                    "description": "List all VFS files",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_save_local",
                    "description": "Save VFS file to local path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vfs_path": {"type": "string"},
                            "local_path": {"type": "string"}
                        },
                        "required": ["vfs_path", "local_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vfs_load_local",
                    "description": "Load local file into VFS",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "local_path": {"type": "string"},
                            "vfs_path": {"type": "string"}
                        },
                        "required": ["local_path", "vfs_path"]
                    }
                }
            }
        ]


# =============================================================================
# 2. CONTEXT COMPRESSION
# =============================================================================

async def _compress_history(
    agent: Any,
    history: List[dict],
    config: OrchestratorConfig
) -> List[dict]:
    """
    Compress conversation history using rolling window summarization.

    If history > max_messages:
    1. Take existing summary + oldest batch
    2. Create new summary via LLM
    3. Return [summary] + recent messages
    """
    if not config.enable_compression:
        return history

    if len(history) <= config.max_history_messages:
        return history

    # Find existing summary
    existing_summary = ""
    summary_idx = -1
    for i, msg in enumerate(history):
        if msg.get("role") == "system" and "[SUMMARY]" in msg.get("content", ""):
            existing_summary = msg["content"]
            summary_idx = i
            break

    start_idx = summary_idx + 1 if summary_idx >= 0 else 0
    messages_to_keep = config.max_history_messages - 5

    if len(history) - start_idx <= messages_to_keep:
        return history

    to_summarize = history[start_idx:start_idx + config.summary_batch_size]
    to_keep = history[start_idx + config.summary_batch_size:]

    if not to_summarize:
        return history

    # Build summary prompt
    parts = []
    if existing_summary:
        parts.append(f"Previous:\n{existing_summary}\n")

    parts.append("Messages to summarize:\n")
    for msg in to_summarize:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))[:300]
        if role == "tool":
            parts.append(f"- Tool[{msg.get('name', '?')}]: {content[:150]}...")
        else:
            parts.append(f"- {role}: {content}...")

    prompt = f"Create concise summary (key info, decisions, pending tasks):\n\n{''.join(parts)}\n\nSummary:"

    try:
        summary = await agent.a_run_llm_completion(
            model_preference="fast",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
            with_context=False,
            task_id="compress_history"
        )

        return [{"role": "system", "content": f"[SUMMARY]\n{summary.strip()}"}] + to_keep
    except Exception as e:
        print(f"[WARN] Compression failed: {e}")
        return history[-config.max_history_messages:]


# =============================================================================
# 3. TOOL PREPARATION
# =============================================================================

def _get_tools_for_litellm(
    tool_registry: dict,
    available_tools: List[str]
) -> List[dict]:
    """Convert agent tools to LiteLLM format."""
    tools = []

    for tool_name in available_tools:
        if tool_name not in tool_registry:
            continue

        info = tool_registry[tool_name]
        description = info.get("description", "No description")
        args_schema = info.get("args_schema", "()")

        # Parse args schema
        properties = {}
        required = []

        if args_schema and args_schema != "()":
            args_str = args_schema.strip("()")
            for arg in args_str.split(","):
                arg = arg.strip()
                if ":" in arg:
                    name_part = arg.split(":")[0].strip()
                    type_part = arg.split(":")[1].strip()

                    has_default = "=" in type_part
                    if has_default:
                        type_part = type_part.split("=")[0].strip()

                    type_map = {"str": "string", "int": "integer", "float": "number",
                                "bool": "boolean", "list": "array", "dict": "object"}

                    properties[name_part] = {"type": type_map.get(type_part, "string")}
                    if not has_default:
                        required.append(name_part)

        tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        })

    # Add direct_response tool
    tools.append({
        "type": "function",
        "function": {
            "name": "direct_response",
            "description": "Provide final answer. Use when task is complete.",
            "parameters": {
                "type": "object",
                "properties": {"final_answer": {"type": "string", "description": "Complete answer"}},
                "required": ["final_answer"]
            }
        }
    })

    return tools


# =============================================================================
# 4. CORE ORCHESTRATION
# =============================================================================

async def _orchestrate_execution_minimal(
    agent: Any,
    config: OrchestratorConfig = None
) -> str:
    """
    Minimalist orchestration replacing LLMReasonerNode and LLMToolNode.

    Flow:
    1. Compress history
    2. Build context (VFS + history)
    3. Decision: direct answer OR [NEED_TOOL]
    4. If tools needed: native function calling loop
    5. Return final response
    """
    config = config or OrchestratorConfig()
    loop_count = 0

    # Initialize VFS
    if config.enable_vfs:
        if "vfs" not in agent.shared:
            agent.shared["vfs"] = VirtualFileSystem(agent, config.vfs_max_window_lines)
        vfs: VirtualFileSystem = agent.shared["vfs"]
    else:
        vfs = None

    query = agent.shared.get("current_query", "")
    history = agent.shared.get("conversation_history", [])

    # Step 0: Compress history
    history = await _compress_history(agent, history, config)
    agent.shared["conversation_history"] = history

    # Step 1: Build context
    vfs_context = vfs.build_context_string() if vfs else ""

    system_msg = getattr(agent.amd, 'get_system_message_with_persona', lambda: agent.amd.system_message)()

    messages = [{
        "role": "system",
        "content": f"""{system_msg}

{vfs_context}

INSTRUCTION: Answer directly if possible. Start with [NEED_TOOL] only if tools are required."""
    }]

    # Add history
    for msg in history:
        if msg.get("role") in ["user", "assistant", "system"]:
            messages.append(msg)

    messages.append({"role": "user", "content": query})

    # Step 2: Decision phase
    decision = await agent.a_run_llm_completion(
        model_preference=config.fast_model_preference,
        messages=messages,
        temperature=config.decision_temperature,
        stream=True,
        with_context=False,
        task_id="decision"
    )

    if not decision.strip().startswith("[NEED_TOOL]"):
        # Direct answer
        agent.shared["conversation_history"].append({"role": "user", "content": query})
        agent.shared["conversation_history"].append({"role": "assistant", "content": decision})
        return decision

    # Step 3: Tool loop
    available_tools = agent.shared.get("available_tools", [])
    litellm_tools = _get_tools_for_litellm(agent._tool_registry, available_tools)

    if vfs:
        litellm_tools.extend(vfs.get_vfs_tools_schema())

    # Add acknowledgment
    ack = decision.replace("[NEED_TOOL]", "").strip()
    if ack:
        messages.append({"role": "assistant", "content": ack})

    final_response = None
    all_results = {}

    while loop_count < config.max_loops:
        loop_count += 1

        try:
            resp_msg = await agent.a_run_llm_completion(
                model_preference=config.fast_model_preference,
                messages=messages,
                tools=litellm_tools,
                tool_choice="auto",
                temperature=config.tool_temperature,
                stream=False,
                get_response_message=True,
                with_context=False,
                task_id=f"tool_loop_{loop_count}"
            )

            tool_calls = getattr(resp_msg, 'tool_calls', None) or []
            content = getattr(resp_msg, 'content', None) or ""

            if not tool_calls:
                final_response = content or "Done."
                break

            # Build assistant message
            asst_msg = {"role": "assistant", "content": content, "tool_calls": []}

            for tc in tool_calls:
                tc_id = tc.id if hasattr(tc, 'id') else f"call_{uuid.uuid4().hex[:8]}"
                tc_name = tc.function.name if hasattr(tc.function, 'name') else tc.function.get('name', '')
                tc_args = tc.function.arguments if hasattr(tc.function, 'arguments') else tc.function.get('arguments',
                                                                                                          '{}')

                asst_msg["tool_calls"].append({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": tc_name, "arguments": tc_args}
                })

            messages.append(asst_msg)

            # Execute tools
            for tc_info in asst_msg["tool_calls"]:
                tc_id = tc_info["id"]
                tool_name = tc_info["function"]["name"]

                try:
                    args = json.loads(tc_info["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                # Check direct_response
                if tool_name == "direct_response":
                    final_response = args.get("final_answer", "Done.")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tool_name,
                        "content": json.dumps({"status": "completed"})
                    })
                    break

                # Execute VFS tools
                if tool_name.startswith("vfs_") and vfs:
                    method_map = {
                        "vfs_create": lambda **a: vfs.create(**a),
                        "vfs_edit": lambda **a: vfs.edit(**a),
                        "vfs_open": lambda **a: vfs.open(**a),
                        "vfs_close": lambda **a: vfs.close(**a),
                        "vfs_extract": lambda **a: vfs.extract_info(**a),
                        "vfs_view": lambda **a: vfs.view(**a),
                        "vfs_list": lambda **a: vfs.list_files(),
                        "vfs_save_local": lambda **a: vfs.save_vfs_to_local(a.get("vfs_path"), a.get("local_path")),
                        "vfs_load_local": lambda **a: vfs.cp_local_to_vfs(a.get("local_path"), a.get("vfs_path"))
                    }

                    method = method_map.get(tool_name)
                    if method:
                        if asyncio.iscoroutinefunction(method) or tool_name in ["vfs_close", "vfs_extract"]:
                            result = await method(**args)
                        else:
                            result = method(**args)
                    else:
                        result = {"error": f"Unknown VFS method: {tool_name}"}

                # Execute registered tools
                elif tool_name in agent._tool_registry:
                    try:
                        result = await agent.arun_function(tool_name, **args)
                        if result is None:
                            result = {"success": True}
                        elif not isinstance(result, dict):
                            result = {"success": True, "result": str(result)}
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                all_results[f"{loop_count}_{tool_name}"] = result

                # Add tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": json.dumps(result, default=str, ensure_ascii=False)[:4000]
                })

            if final_response:
                break

        except Exception as e:
            print(f"[ERROR] Tool loop: {e}")
            import traceback
            traceback.print_exc()
            final_response = f"Error: {str(e)}"
            break

    if not final_response:
        final_response = f"Reached limit ({config.max_loops} loops). Results: {json.dumps(all_results, default=str)[:500]}"

    agent.shared["conversation_history"].append({"role": "user", "content": query})
    agent.shared["conversation_history"].append({"role": "assistant", "content": final_response})
    agent.shared["results"] = all_results
    agent.shared["tool_calls_made"] = loop_count

    return final_response


# ===== MAIN AGENT CLASS =====
class FlowAgent:
    """Production-ready agent system built on PocketFlow """
    def __init__(
        self,
        amd: AgentModelData,
        world_model: dict[str, Any] = None,
        verbose: bool = False,
        checkpoint_interval: int = 300,  # 5 minutes
        max_parallel_tasks: int = 3,
        vfs_max_window_lines: int = 250,
        progress_callback: callable = None,
        stream:bool=True,
        **kwargs
    ):
        self.amd = amd
        self.stream = stream
        self.world_model = world_model or {}
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_config = CheckpointConfig()
        self.max_parallel_tasks = max_parallel_tasks
        self.progress_tracker = ProgressTracker(progress_callback, agent_name=amd.name)

        self.vfs_in_memory = VirtualFileSystem(self, amd.vfs_max_window_lines)
        # Agent state
        self.is_running = False
        self.last_checkpoint = None

        # Token and cost tracking (persistent across runs)
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_accumulated = 0.0
        self.total_llm_calls = 0

        self.checkpoint_data = {}

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)
        self._shutdown_event = threading.Event()

        # Server components
        self.a2a_server: A2AServer = None
        self.mcp_server: FastMCP = None

        # Enhanced tool registry
        self._tool_registry = {}

        self.active_session = None
        self.active_chat_session = None
        # Tool analysis file path
        self.tool_analysis_file = self._get_tool_analysis_path()

        # Session-restricted tools: {tool_name: {session_id: allowed (bool), '*': default_allowed (bool)}}
        # All tools start as allowed (True) by default via '*' key
        self.session_tool_restrictions = {}
        self.resent_tools_called = []

        # LLM Rate Limiter (P1 - HOCH: Prevent cost explosions)
        if isinstance(amd.handler_path_or_dict, dict):
            self.llm_handler = create_handler_from_config(amd.handler_path_or_dict)
        elif isinstance(amd.handler_path_or_dict, str) and os.path.exists(amd.handler_path_or_dict):
            self.llm_handler = load_handler_from_file(amd.handler_path_or_dict)
        else:
            self.llm_handler = LiteLLMRateLimitHandler(max_retries=3)


        # MCP Session Health Tracking (P0 - KRITISCH: Circuit breaker pattern)
        self.mcp_session_health = {}  # server_name -> {"failures": int, "last_failure": float, "state": "CLOSED|OPEN|HALF_OPEN"}
        self.mcp_circuit_breaker_threshold = 3  # Failures before opening circuit
        self.mcp_circuit_breaker_timeout = 60.0  # Seconds before trying HALF_OPEN

        if self.amd.budget_manager:
            self.amd.budget_manager.load_data()

        rprint(f"FlowAgent initialized: {amd.name}")

    @property
    def progress_callback(self):
        return self.progress_tracker.progress_callback

    @progress_callback.setter
    def progress_callback(self, value):
        self.progress_tracker.progress_callback = value

    def set_progress_callback(self, progress_callback: callable = None):
        self.progress_callback = progress_callback

    # TODO Remove
    def sanitize_message_history(self, messages: list[dict]) -> list[dict]:
        """
        Sanitize message history - FIXED with intelligent matching by tool name/content
        """
        if not messages:
            return messages

        sanitized = []
        pending_tool_calls = {}  # tool_call_id -> {index, name, arguments}

        for msg in messages:
            role = msg.get('role', '')

            if role == 'assistant':
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = None
                        tc_name = None
                        tc_args = None

                        if hasattr(tc, 'id'):
                            tc_id = tc.id
                            tc_name = tc.function.name if hasattr(tc, 'function') else None
                            tc_args = tc.function.arguments if hasattr(tc, 'function') else None
                        elif isinstance(tc, dict):
                            tc_id = tc.get('id') or tc.get('tool_call_id')
                            func = tc.get('function', {})
                            tc_name = func.get('name')
                            tc_args = func.get('arguments')

                        if tc_id:
                            pending_tool_calls[tc_id] = {
                                'index': len(sanitized),
                                'name': tc_name,
                                'arguments': tc_args
                            }
                sanitized.append(msg)

            elif role == 'tool':
                tool_call_id = msg.get('tool_call_id')
                tool_content = msg.get('content', '')

                # Versuche Tool-Namen aus Content zu extrahieren
                tool_name_in_response = None
                try:
                    content_json = json.loads(tool_content)
                    tool_name_in_response = content_json.get('tool_name')
                except:
                    pass

                matched = False

                # 1. Exakter ID-Match
                if tool_call_id and tool_call_id in pending_tool_calls:
                    sanitized.append(msg)
                    del pending_tool_calls[tool_call_id]
                    matched = True

                # 2. Match by Tool-Name
                elif tool_name_in_response and pending_tool_calls:
                    for tc_id, tc_info in list(pending_tool_calls.items()):
                        if tc_info['name'] == tool_name_in_response:
                            msg_copy = msg.copy()
                            msg_copy['tool_call_id'] = tc_id
                            sanitized.append(msg_copy)
                            del pending_tool_calls[tc_id]
                            matched = True
                            print(f"✓ [SANITIZE] Matched tool response by name: {tool_name_in_response} -> {tc_id}")
                            break

                # 3. FIFO Fallback - weise erstem pending zu
                if not matched and pending_tool_calls:
                    first_pending_id = next(iter(pending_tool_calls.keys()))
                    msg_copy = msg.copy()
                    msg_copy['tool_call_id'] = first_pending_id
                    sanitized.append(msg_copy)
                    print(f"⚠️ [SANITIZE] FIFO assigned: {tool_call_id} -> {first_pending_id}")
                    del pending_tool_calls[first_pending_id]
                    matched = True

                if not matched:
                    # Kein pending call - als User-Message mit Ergebnis einfügen
                    print(f"⚠️ [SANITIZE] Converting orphaned tool response to user message")
                    sanitized.append({
                        "role": "user",
                        "content": f"[Previous tool result]: {tool_content[:1000]}"
                    })
            else:
                sanitized.append(msg)

        # Cleanup: Entferne tool_calls ohne Responses am Ende
        if pending_tool_calls:
            for tc_id, tc_info in pending_tool_calls.items():
                idx = tc_info['index']
                if idx < len(sanitized):
                    msg = sanitized[idx]
                    if msg.get('tool_calls'):
                        # Entferne tool_calls aber behalte content
                        msg_copy = {k: v for k, v in msg.items() if k != 'tool_calls'}
                        if not msg_copy.get('content'):
                            msg_copy['content'] = f"(Attempted to call {tc_info['name']} but no response received)"
                        sanitized[idx] = msg_copy
                        print(f"⚠️ [SANITIZE] Removed unmatched tool_call: {tc_id}")

        return sanitized

    def _process_media_in_messages(self, messages: list[dict]) -> list[dict]:
        """
        Process messages to extract and convert [media:(path/url)] tags to litellm format

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            list[dict]: Processed messages with media content properly formatted
        """
        processed_messages = []

        for msg in messages:
            if not isinstance(msg.get("content"), str):
                # Already processed or non-text content
                processed_messages.append(msg)
                continue

            content = msg["content"]

            if not content:
                continue

            # Check if content contains media tags
            cleaned_content, media_list = parse_media_from_query(content, self.amd.complex_llm_model)

            if media_list:
                # Convert to multi-modal message format for litellm
                # Format: content becomes a list with text and media items
                content_parts = []

                # Add text part if there's any text left
                if cleaned_content.strip():
                    content_parts.append({
                        "type": "text",
                        "text": cleaned_content
                    })

                # Add media parts
                content_parts.extend(media_list)

                processed_messages.append({
                    "role": msg["role"],
                    "content": content_parts
                })
            else:
                # No valid media found, keep original
                processed_messages.append(msg)
        return processed_messages

    async def a_run_llm_completion(self, node_name="FlowAgentLLMCall",task_id="unknown",model_preference="fast", with_context=True, llm_kwargs=None, get_response_message=False, skip_sanitize=False,**kwargs) -> str:
        """
        Run LLM completion with support for media inputs and custom kwargs

        Args:
            node_name: Name of the calling node for tracking
            task_id: Task identifier for tracking
            model_preference: "fast" or "complex" model preference
            with_context: Whether to include session context
            auto_fallbacks: Whether to use automatic fallback models
            llm_kwargs: Additional kwargs to pass to litellm (merged with **kwargs)
            **kwargs: Additional arguments for litellm.acompletion

        Returns:
            str: LLM response content
        """
        # Merge llm_kwargs if provided
        if llm_kwargs:
            kwargs.update(llm_kwargs)

        if "model" not in kwargs:
            kwargs["model"] = self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model

        if not 'stream' in kwargs:
            kwargs['stream'] = self.stream

        # Parse media from messages if present
        if "messages" in kwargs:
            kwargs["messages"] = self._process_media_in_messages(kwargs["messages"])
            # if not skip_sanitize:
            #     # Sanitize message history to prevent tool call/response pair corruption
            #     kwargs["messages"] = self.sanitize_message_history(kwargs["messages"])

        llm_start = time.perf_counter()

        if self.progress_tracker:
            await self.progress_tracker.emit_event(
                ProgressEvent(
                    event_type="llm_call",
                    node_name=node_name,
                    session_id=self.active_session,
                    task_id=task_id,
                    status=NodeStatus.RUNNING,
                    llm_model=kwargs["model"],
                    llm_temperature=kwargs.get("temperature", 0.7),
                    llm_input=kwargs.get("messages", [{}])[-1].get(
                        "content", ""
                    ),  # Prompt direkt erfassen
                    metadata={"model_preference": kwargs.get("model_preference", "fast")},
                )
            )

        # auto api key addition supports (google, openrouter, openai, anthropic, azure, aws, huggingface, replicate, togetherai, groq)
        if "api_key" not in kwargs:
            # litellm model-prefix apikey mapp
            prefix = kwargs['model'].split("/")[0]
            model_prefix_map = {
                "openrouter": os.getenv("OPENROUTER_API_KEY"),
                "openai": os.getenv("OPENAI_API_KEY"),
                "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                "google": os.getenv("GOOGLE_API_KEY"),
                "azure": os.getenv("AZURE_API_KEY"),
                "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
                "replicate": os.getenv("REPLICATE_API_KEY"),
                "togetherai": os.getenv("TOGETHERAI_API_KEY"),
                "groq": os.getenv("GROQ_API_KEY"),
            }
            kwargs["api_key"] = model_prefix_map.get(prefix)

        if self.active_session and with_context:
            # Add context to fist messages as system message
            # OPTIMIZED: Conditional context injection
            context_mode = kwargs.pop("context_mode", "auto" if with_context else "none")

            if context_mode == "full" and self.active_session:
                # Full context (original behavior)
                context_ = await self.get_context(self.active_session)
                kwargs["messages"] = [
                    {
                        "role": "system",
                        "content": self.amd.get_system_message_with_persona()
                        + "\n\nContext:\n\n"
                        + str(context_),
                    }
                ] + kwargs.get("messages", [])

            elif context_mode == "minimal" and self.active_session:
                # Minimal: Only persona + last interaction
                last = (
                    await self.context_manager.get_contextual_history(self.active_session, max_entries=3)
                    if self.context_manager
                    else []
                )
                kwargs["messages"] = [
                    {
                        "role": "system",
                        "content": self.amd.get_system_message_with_persona()
                    }
                ] +last+ kwargs.get("messages", [])

            elif context_mode == "persona" and self.active_session:
                # Persona only, no context
                kwargs["messages"] = [{"role": "system", "content": self.amd.get_system_message_with_persona()}] + kwargs.get("messages", [])

            # "none" or "auto" with format_ task = no context injection

            elif context_mode == "auto" and with_context and self.active_session:
                task_id = kwargs.get("task_id", "")
                if not task_id.startswith("format_") and not task_id.startswith("lean_"):
                    # Only inject for non-formatting tasks
                    context_ = await self.get_context(self.active_session)
                    kwargs["messages"] = [{"role": "system", "content": self.amd.get_system_message_with_persona()+'\n\nContext:\n\n'+context_}] + kwargs.get("messages", [])

        try:
            if kwargs.get("stream", False):
                kwargs["stream_options"] = {"include_usage": True}

            # detailed informations str
            with (Spinner(f"LLM Call {self.amd.name}@{node_name}#{task_id if task_id else model_preference}-{kwargs['model']}")):
                response = await self.llm_handler.completion_with_rate_limiting(
                                    litellm,**kwargs
                                )

            if not kwargs.get("stream", False):
                result = response.choices[0].message.content
                result_to_retun = result if not get_response_message else response.choices[0].message
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0

            else:
                result = ""
                final_chunk = None
                from litellm.types.utils import (
                    Message,
                    ChatCompletionMessageToolCall,
                    Function
                )
                tool_calls_acc = {}
                async for chunk in response:
                    delta = chunk.choices[0].delta

                    # 1. Text sammeln
                    content = delta.content or ""
                    result += content

                    if self.progress_tracker and content:
                        await self.progress_tracker.emit_event(ProgressEvent(
                            event_type="llm_stream_chunk",
                            node_name=node_name,
                            task_id=task_id,
                            session_id=self.active_session,
                            status=NodeStatus.RUNNING,
                            llm_model=kwargs["model"],
                            llm_output=content,
                        ))

                    # 2. Tool Calls sammeln
                    if getattr(delta, "tool_calls", None):
                        for tc in delta.tool_calls:
                            idx = tc.index

                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = ChatCompletionMessageToolCall(
                                    id=tc.id,
                                    type="function",
                                    function=Function(
                                        name="",
                                        arguments=""
                                    )
                                )

                            if tc.function:
                                if tc.function.name:
                                    tool_calls_acc[idx].function.name = tc.function.name

                                if tc.function.arguments:
                                    tool_calls_acc[idx].function.arguments += tc.function.arguments

                    final_chunk = chunk

                usage = final_chunk.usage if hasattr(final_chunk, "usage") else None
                output_tokens = usage.completion_tokens if usage else 0
                input_tokens = usage.prompt_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0
                result_to_retun = result
                if get_response_message:
                    result_to_retun = Message(
                        role="assistant",
                        content=result or None,
                        tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else []
                    )

            llm_duration = time.perf_counter() - llm_start

            if AGENT_VERBOSE and self.verbose:
                kwargs["messages"] += [{"role": "assistant", "content": result}]
                print_prompt(kwargs)
            # else:
            #     print_prompt([{"role": "assistant", "content": result}])

            # Extract token usage and cost


            call_cost = self.progress_tracker.calculate_llm_cost(kwargs["model"], input_tokens,
                                                            output_tokens, response) if self.progress_tracker else 0.0

            # Accumulate total tokens and cost
            self.total_tokens_in += input_tokens
            self.total_tokens_out += output_tokens
            self.total_cost_accumulated += call_cost
            self.total_llm_calls += 1

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="llm_call",
                    node_name=node_name,
                    task_id=task_id,
                    session_id=self.active_session,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    duration=llm_duration,
                    llm_model=kwargs["model"],
                    llm_prompt_tokens=input_tokens,
                    llm_completion_tokens=output_tokens,
                    llm_total_tokens=total_tokens,
                    llm_cost=call_cost,
                    llm_temperature=kwargs.get("temperature", 0.7),
                    llm_output=result,
                    llm_input="",
                ))

            return result_to_retun
        except Exception as e:
            llm_duration = time.perf_counter() - llm_start
            import traceback
            print(traceback.format_exc())
            # print(f"LLM call failed: {json.dumps(kwargs, indent=2)}")

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="llm_call",  # Event-Typ bleibt konsistent
                    node_name=node_name,
                    task_id=task_id,
                    session_id=self.active_session,
                    status=NodeStatus.FAILED,
                    success=False,
                    duration=llm_duration,
                    llm_model=kwargs["model"],
                    error_details={
                        "message": str(e),
                        "type": type(e).__name__
                    }
                ))

            raise

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        stream_callback: Callable = None,
        remember: bool = True,
        as_callback: Callable = None,
        **kwargs
    ) -> str:
        """Main entry point für Agent-Ausführung mit UnifiedContextManager

        Args:
            query: Die Benutzeranfrage (kann [media:(path/url)] Tags enthalten)
            session_id: Session-ID für Kontext-Management
            stream_callback: Callback für Streaming-Antworten
            remember: Ob die Interaktion gespeichert werden soll
            **kwargs: Zusätzliche Argumente (kann llm_kwargs enthalten)

        Note:
            Media-Tags im Format [media:(path/url)] werden automatisch geparst und
            an das LLM als Multi-Modal-Input übergeben.
        """

        execution_start = self.progress_tracker.start_timer("total_execution")
        self.active_session = session_id
        #TODO add caht chsession self.active_chat_session = xyz,manger.getsession(session_id)
        self.resent_tools_called = []
        result = None

        await self.progress_tracker.emit_event(ProgressEvent(
            event_type="execution_start",
            timestamp=time.time(),
            status=NodeStatus.RUNNING,
            node_name="FlowAgent",
            session_id=session_id,
            metadata={"query": query, "has_callback": as_callback is not None}
        ))

        try:
            #Initialize or get session über UnifiedContextManager

            if remember and self.active_chat_session:
                await self.active_chat_session.add_message(
                    {"role": "user", "content": query},
                )

            # Set user context variables
            timestamp = datetime.now()

            # --- Neu: as_callback behandeln ---
            if as_callback:
                callback_context = {
                    'callback_timestamp': datetime.now().isoformat(),
                    'callback_name': getattr(as_callback, '__name__', 'unnamed_callback'),
                    'initial_query': query
                }

            self.is_running = True

            result = await _orchestrate_execution_minimal(self, kwargs.get('orchestrator_config') or OrchestratorConfig())

            #Store assistant response in ChatSession wenn remember=True

            total_duration = self.progress_tracker.end_timer("total_execution")
            if remember and self.active_chat_session:
                await self.active_chat_session.add_message(
                    {"role": "assistant", "content": result},
                    execution_duration= total_duration
                )

            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="execution_complete",
                timestamp=time.time(),
                node_name="FlowAgent",
                status=NodeStatus.COMPLETED,
                node_duration=total_duration,
                session_id=session_id,
                metadata={
                    "result_length": len(result),
                    "summary": self.progress_tracker.get_summary(),
                    "remembered": remember
                }
            ))

            # TODO run save checkpoint in bg self.executor.submit()

            return result

        except Exception as e:
            eprint(f"Agent execution failed: {e}")
            error_response = f"I encountered an error: {str(e)}"
            result = error_response
            import traceback
            print(traceback.format_exc())

            total_duration = self.progress_tracker.end_timer("total_execution")

            if remember and self.active_chat_session:
                await self.active_chat_session.add_message(
                    {"role": "assistant", "content": result},
                    execution_duration=total_duration,
                        error=True,
                        error_type=type(e).__name__
                )

            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="error",
                timestamp=time.time(),
                node_name="FlowAgent",
                status=NodeStatus.FAILED,
                node_duration=total_duration,
                session_id=session_id,
                metadata={"error": str(e), "error_type": type(e).__name__}
            ))

            return error_response

        finally:
            self.is_running = False
            self.active_session = None
            self.active_chat_session = None

    # ===== save FUNCTIONALITY =====

    async def save(self) -> bool:
        """Pause agent execution"""

        # Create checkpoint
        await self._save_and_checkpoint(save_path=True)

        rprint("Agent state saved")
        return True

    # ===== CHECKPOINT MANAGEMENT =====

    async def _create_checkpoint(self) -> AgentCheckpoint:
        """
        Erstellt einen robusten, serialisierbaren Checkpoint, der nur reine Daten enthält.
        Laufzeitobjekte und nicht-serialisierbare Elemente werden explizit ausgeschlossen.
        """

    async def _save_and_checkpoint(self, save_path: str = None, filepath = None):
        """Vereinfachtes Checkpoint-Speichern - alles in eine Datei"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            checkpoint = await self._create_checkpoint()
            if not filepath:
                timestamp = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
                filepath = f"agent_checkpoint_{timestamp}.pkl"
            filepath = os.path.join(folder, filepath)
            with open(filepath, "wb") as f:
                pickle.dump(checkpoint, f)
            return True
        except Exception as e:
            eprint(f"Checkpoint-Speichern fehlgeschlagen: {e}")
            import traceback
            print(traceback.format_exc())
            return False


    async def load_latest_checkpoint(self, auto_restore_history: bool = True, max_age_hours: int = 24) -> dict[
        str, Any]:
        """Vereinfachtes Checkpoint-Laden mit automatischer History-Wiederherstellung"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name

            if not os.path.exists(folder):
                return {"success": False, "error": "Kein Checkpoint-Verzeichnis gefunden"}

            # Finde neuesten Checkpoint
            checkpoint_files = []
            for file in os.listdir(folder):
                if file.endswith('.pkl') and (file.startswith('agent_checkpoint_') or file == 'final_checkpoint.pkl'):
                    filepath = os.path.join(folder, file)
                    try:
                        timestamp_str = file.replace('agent_checkpoint_', '').replace('.pkl', '')
                        if timestamp_str == 'final_checkpoint':
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        else:
                            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                        age_hours = (datetime.now() - file_time).total_seconds() / 3600
                        if age_hours <= max_age_hours:
                            checkpoint_files.append((filepath, file_time, age_hours))
                    except Exception:
                        continue

            if not checkpoint_files:
                return {"success": False, "error": f"Keine gültigen Checkpoints in {max_age_hours} Stunden gefunden"}

            # Lade neuesten Checkpoint
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            latest_checkpoint_path, latest_timestamp, age_hours = checkpoint_files[0]

            rprint(f"Lade Checkpoint: {latest_checkpoint_path} (Alter: {age_hours:.1f}h)")

            with open(latest_checkpoint_path, 'rb') as f:
                checkpoint: AgentCheckpoint = pickle.load(f)

                print("Loaded Checkpoint: ", f.__sizeof__())
            # Stelle Agent-Status wieder her
            restore_stats = await self._restore_from_checkpoint_simplified(checkpoint, auto_restore_history)

            return {
                "success": True,
                "checkpoint_file": latest_checkpoint_path,
                "checkpoint_age_hours": age_hours,
                "checkpoint_timestamp": latest_timestamp.isoformat(),
                "available_checkpoints": len(checkpoint_files),
                "restore_stats": restore_stats
            }

        except Exception as e:
            eprint(f"Checkpoint-Laden fehlgeschlagen: {e}")
            import traceback
            print(traceback.format_exc())
            return {"success": False, "error": str(e)}

    # TODO rework
    async def _restore_from_checkpoint_simplified(self, checkpoint: AgentCheckpoint, auto_restore_history: bool) -> \
    dict[str, Any]:
        """
        Stellt den Agentenzustand aus einem bereinigten Daten-Checkpoint wieder her, indem Laufzeitobjekte
        neu initialisiert und mit den geladenen Daten hydriert werden.
        """
        restore_stats = {
            "agent_state_restored": False, "world_model_restored": False,
            "tasks_restored": 0, "sessions_restored": 0, "variables_restored": 0,
            "conversation_restored": 0, "errors": []
        }
        rprint("Starte Wiederherstellung aus Daten-Checkpoint...")

        try:
            # 1. Agent-Status wiederherstellen (einfache Daten)
            if checkpoint.agent_state:
                self.is_paused = checkpoint.agent_state.get("is_paused", False)
                self.active_session = checkpoint.agent_state.get("active_session")

                # Token and cost tracking wiederherstellen
                self.total_tokens_in = checkpoint.agent_state.get("total_tokens_in", 0)
                self.total_tokens_out = checkpoint.agent_state.get("total_tokens_out", 0)
                self.total_cost_accumulated = checkpoint.agent_state.get("total_cost_accumulated", 0.0)
                self.total_llm_calls = checkpoint.agent_state.get("total_llm_calls", 0)

                # AMD-Daten selektiv wiederherstellen
                amd_data = checkpoint.agent_state.get("amd_data", {})
                if amd_data:
                    # Nur sichere Felder wiederherstellen
                    safe_fields = ["name", "use_fast_response", "max_input_tokens"]
                    for field in safe_fields:
                        if field in amd_data and hasattr(self.amd, field):
                            setattr(self.amd, field, amd_data[field])

                    # Persona wiederherstellen falls vorhanden
                    if "persona" in amd_data and amd_data["persona"]:
                        try:
                            persona_data = amd_data["persona"]
                            if isinstance(persona_data, dict):
                                self.amd.persona = PersonaConfig(**persona_data)
                        except Exception as e:
                            restore_stats["errors"].append(f"Persona restore failed: {e}")

                restore_stats["agent_state_restored"] = True

            # 2. World Model wiederherstellen
            if checkpoint.world_model:
                self.shared["world_model"] = checkpoint.world_model.copy()
                self.world_model = self.shared["world_model"]
                restore_stats["world_model_restored"] = True

            # 6. Sessions und Conversation wiederherstellen
            if auto_restore_history:
                await self._restore_sessions_and_conversation_simplified(checkpoint, restore_stats)

            # 8. Session Tool Restrictions wiederherstellen
            if hasattr(checkpoint, 'session_tool_restrictions') and checkpoint.session_tool_restrictions:
                self.session_tool_restrictions = checkpoint.session_tool_restrictions.copy()
                restore_stats["tool_restrictions_restored"] = len(checkpoint.session_tool_restrictions)
                rprint(f"Tool restrictions wiederhergestellt: {len(checkpoint.session_tool_restrictions)} Tools mit Restrictions")

            restore_stats["restoration_timestamp"] = datetime.now().isoformat()

            rprint(
                f"Checkpoint-Wiederherstellung abgeschlossen: {restore_stats['tasks_restored']} Tasks, {restore_stats['sessions_restored']} Sessions, {len(restore_stats['errors'])} Fehler.")
            return restore_stats

        except Exception as e:
            eprint(f"FEHLER bei der Checkpoint-Wiederherstellung: {e}")
            import traceback
            print(traceback.format_exc())
            restore_stats["errors"].append(f"Kritischer Fehler bei der Wiederherstellung: {e}")
            return restore_stats

    async def _restore_sessions_and_conversation_simplified(self, checkpoint: AgentCheckpoint, restore_stats: dict):
        """Vereinfachte Session- und Conversation-Wiederherstellung"""
        try:
            # Context Manager sicherstellen
            # Sessions wiederherstellen
            if hasattr(checkpoint, 'session_data') and checkpoint.session_data:
                for session_id, session_info in checkpoint.session_data.items():
                    try:
                        # Session über Context Manager initialisieren
                        max_length = session_info.get("message_count", 200)
                        restore_stats["sessions_restored"] += 1
                    except Exception as e:
                        restore_stats["errors"].append(f"Session {session_id}: {e}")


        except Exception as e:
            restore_stats["errors"].append(f"Session/conversation restore failed: {e}")

    def list_available_checkpoints(self, max_age_hours: int = 168) -> list[dict[str, Any]]:  # Default 1 week
        """List all available checkpoints with metadata"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name

            if not os.path.exists(folder):
                return []

            checkpoints = []
            for file in os.listdir(folder):
                if file.endswith('.pkl') and file.startswith('agent_checkpoint_'):
                    filepath = os.path.join(folder, file)
                    try:
                        # Get file info
                        file_stat = os.stat(filepath)
                        file_size = file_stat.st_size
                        modified_time = datetime.fromtimestamp(file_stat.st_mtime)

                        # Extract timestamp from filename
                        timestamp_str = file.replace('agent_checkpoint_', '').replace('.pkl', '')
                        if timestamp_str == 'final_checkpoint':
                            checkpoint_time = modified_time
                            checkpoint_type = "final"
                        else:
                            checkpoint_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            checkpoint_type = "regular"

                        # Check age
                        age_hours = (datetime.now() - checkpoint_time).total_seconds() / 3600
                        if age_hours <= max_age_hours:

                            # Try to load checkpoint metadata without full loading
                            metadata = {}
                            try:
                                with open(filepath, 'rb') as f:
                                    checkpoint = pickle.load(f)
                                metadata = {
                                    "tasks_count": len(checkpoint.task_state) if checkpoint.task_state else 0,
                                    "world_model_entries": len(checkpoint.world_model) if checkpoint.world_model else 0,
                                    "session_id": checkpoint.metadata.get("session_id", "unknown") if hasattr(
                                        checkpoint, 'metadata') and checkpoint.metadata else "unknown",
                                    "last_query": checkpoint.metadata.get("last_query", "unknown")[:100] if hasattr(
                                        checkpoint, 'metadata') and checkpoint.metadata else "unknown"
                                }
                            except:
                                metadata = {"load_error": True}

                            checkpoints.append({
                                "filepath": filepath,
                                "filename": file,
                                "checkpoint_type": checkpoint_type,
                                "timestamp": checkpoint_time.isoformat(),
                                "age_hours": round(age_hours, 1),
                                "file_size_kb": round(file_size / 1024, 1),
                                "metadata": metadata
                            })

                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        wprint(f"Could not analyze checkpoint file {file}: {e}")
                        continue

            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

            return checkpoints

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            eprint(f"Failed to list checkpoints: {e}")
            return []

    async def delete_old_checkpoints(self, keep_count: int = 5, max_age_hours: int = 168) -> dict[str, Any]:
        """Delete old checkpoints, keeping the most recent ones"""
        try:
            checkpoints = self.list_available_checkpoints(
                max_age_hours=max_age_hours * 2)  # Look further back for deletion

            deleted_count = 0
            deleted_size_kb = 0
            errors = []

            if len(checkpoints) > keep_count:
                # Keep the newest, delete the rest (except final checkpoint)
                to_delete = checkpoints[keep_count:]

                for checkpoint in to_delete:
                    if checkpoint["checkpoint_type"] != "final":  # Never delete final checkpoint
                        try:
                            os.remove(checkpoint["filepath"])
                            deleted_count += 1
                            deleted_size_kb += checkpoint["file_size_kb"]
                            rprint(f"Deleted old checkpoint: {checkpoint['filename']}")
                        except Exception as e:
                            import traceback
                            print(traceback.format_exc())
                            errors.append(f"Failed to delete {checkpoint['filename']}: {e}")

            # Also delete checkpoints older than max_age_hours
            old_checkpoints = [cp for cp in checkpoints if
                               cp["age_hours"] > max_age_hours and cp["checkpoint_type"] != "final"]
            for checkpoint in old_checkpoints:
                if checkpoint not in checkpoints[keep_count:]:  # Don't double-delete
                    try:
                        os.remove(checkpoint["filepath"])
                        deleted_count += 1
                        deleted_size_kb += checkpoint["file_size_kb"]
                        rprint(f"Deleted aged checkpoint: {checkpoint['filename']}")
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        errors.append(f"Failed to delete {checkpoint['filename']}: {e}")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "freed_space_kb": round(deleted_size_kb, 1),
                "remaining_checkpoints": len(checkpoints) - deleted_count,
                "errors": errors
            }

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            eprint(f"Failed to delete old checkpoints: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0
            }

    # ===== TOOL AND NODE MANAGEMENT =====

    def add_first_class_tool(self, tool_func: Callable, name: str, description: str):
        """
        Add a first-class meta-tool that can be used by the LLMReasonerNode.
        These are different from regular tools - they control agent sub-systems.

        Args:
            tool_func: The function to register as a meta-tool
            name: Name of the meta-tool
            description: Description of when and how to use it
        """

        if not asyncio.iscoroutinefunction(tool_func):
            @wraps(tool_func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.to_thread(tool_func, *args, **kwargs)

            effective_func = async_wrapper
        else:
            effective_func = tool_func

        tool_name = name or effective_func.__name__
        tool_description = description or effective_func.__doc__ or "No description"

        # Validate the tool function
        if not callable(tool_func):
            raise ValueError("Tool function must be callable")

        # Register in the reasoner's meta-tool registry (if reasoner exists)
        # {
        #         "function": effective_func,
        #         "description": tool_description,
        #         "args_schema": get_args_schema(tool_func)
        # }


    async def add_tool(self, tool_func: Callable, name: str = None, description: str = None, is_new=False, skip_analysis=False):
        """Enhanced tool addition with intelligent analysis

        Args:
            tool_func: The tool function to add
            name: Optional name override
            description: Optional description override
            is_new: If True, always run LLM analysis
            skip_analysis: If True, skip LLM analysis entirely (use basic fallback)
        """
        if not asyncio.iscoroutinefunction(tool_func):
            @wraps(tool_func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.to_thread(tool_func, *args, **kwargs)

            effective_func = async_wrapper
        else:
            effective_func = tool_func

        tool_name = name or effective_func.__name__
        tool_description = description or effective_func.__doc__ or "No description"

        # Store in registry
        self._tool_registry[tool_name] = {
            "function": effective_func,
            "description": tool_description,
            "args_schema": get_args_schema(tool_func)
        }

        # Add to available tools list
        if tool_name not in self.shared["available_tools"]:
            self.shared["available_tools"].append(tool_name)

        # Intelligent tool analysis
        if skip_analysis:
            # Use basic fallback without LLM
            self._tool_capabilities[tool_name] = {
                "use_cases": [tool_description],
                "triggers": [tool_name.lower().replace('_', ' ')],
                "complexity": "unknown",
                "confidence": 0.5
            }
        rprint(f"Tool added: {tool_name}")

    def add_custom_flow(self, flow: AsyncFlow, name: str):
        """Add a custom flow for dynamic execution"""
        self.add_tool(flow.run_async, name=name, description=f"Custom flow: {flow.__class__.__name__}")
        rprint(f"Custom node added: {name}")

    def get_tool_by_name(self, tool_name: str) -> Callable | None:
        """Get tool function by name"""
        return self._tool_registry.get(tool_name, {}).get("function")

    # ===== SESSION TOOL RESTRICTIONS =====

    def _is_tool_allowed_in_session(self, tool_name: str, session_id: str) -> bool:
        """
        Check if a tool is allowed in a specific session.

        Logic:
        1. If tool not in restrictions map -> allowed (default True)
        2. If tool in map, check session_id key -> use that value
        3. If session_id not in tool's map, use '*' default value
        4. If '*' not set, default to True (allow)

        Args:
            tool_name: Name of the tool
            session_id: Session ID to check

        Returns:
            bool: True if tool is allowed, False if restricted
        """
        if tool_name not in self.session_tool_restrictions:
            # Tool not in restrictions -> allowed by default
            return True

        tool_restrictions = self.session_tool_restrictions[tool_name]

        # Check specific session restriction
        if session_id in tool_restrictions:
            return tool_restrictions[session_id]

        # Fall back to default '*' value
        return tool_restrictions.get('*', True)

    def set_tool_restriction(self, tool_name: str, session_id: str = '*', allowed: bool = True):
        """
        Set tool restriction for a specific session or as default.

        Args:
            tool_name: Name of the tool to restrict
            session_id: Session ID to restrict (use '*' for default)
            allowed: True to allow, False to restrict

        Examples:
            # Restrict tool in specific session
            agent.set_tool_restriction('dangerous_tool', 'session_123', allowed=False)

            # Set default to restricted, but allow in specific session
            agent.set_tool_restriction('admin_tool', '*', allowed=False)
            agent.set_tool_restriction('admin_tool', 'admin_session', allowed=True)
        """
        if tool_name not in self.session_tool_restrictions:
            self.session_tool_restrictions[tool_name] = {}

        self.session_tool_restrictions[tool_name][session_id] = allowed
        rprint(
            f"Tool restriction set: {tool_name} in session '{session_id}' -> {'allowed' if allowed else 'restricted'}"
        )

    def get_tool_restriction(self, tool_name: str, session_id: str = "*") -> bool:
        """
        Get tool restriction status for a session.

        Args:
            tool_name: Name of the tool
            session_id: Session ID (use '*' for default)

        Returns:
            bool: True if allowed, False if restricted
        """
        return self._is_tool_allowed_in_session(tool_name, session_id)

    def reset_tool_restrictions(self, tool_name: str = None):
        """
        Reset tool restrictions. If tool_name is None, reset all restrictions.

        Args:
            tool_name: Specific tool to reset, or None for all tools
        """
        if tool_name is None:
            self.session_tool_restrictions.clear()
            rprint("All tool restrictions cleared")
        elif tool_name in self.session_tool_restrictions:
            del self.session_tool_restrictions[tool_name]
            rprint(f"Tool restrictions cleared for: {tool_name}")

    def list_tool_restrictions(self) -> dict[str, dict[str, bool]]:
        """
        Get all current tool restrictions.

        Returns:
            dict: Copy of session_tool_restrictions map
        """
        return self.session_tool_restrictions.copy()

    # ===== TOOL EXECUTION =====

    async def arun_function(self, function_name: str, *args, **kwargs) -> Any:
        """
        Asynchronously finds a function by its string name, executes it with
        the given arguments, and returns the result.
        """
        rprint(
            f"Attempting to run function: {function_name} with args: {args}, kwargs: {kwargs}"
        )

        # Check session-based tool restrictions
        if self.active_session:
            if not self._is_tool_allowed_in_session(function_name, self.active_session):
                raise PermissionError(
                    f"Tool '{function_name}' is restricted in session '{self.active_session}'. "
                    f"Use set_tool_restriction() to allow it."
                )

        target_function = self.get_tool_by_name(function_name)

        start_time = time.perf_counter()
        if not target_function:
            raise ValueError(
                f"Function '{function_name}' not found in the {self.amd.name}'s registered tools."
            )
        result = None
        try:
            if asyncio.iscoroutinefunction(target_function):
                result = await target_function(*args, **kwargs)
            else:
                # If the function is not async, run it in a thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: target_function(*args, **kwargs)
                )

            if asyncio.iscoroutine(result):
                result = await result

            if self.progress_tracker:
                await self.progress_tracker.emit_event(
                    ProgressEvent(
                        event_type="tool_call",  # Vereinheitlicht zu tool_call
                        node_name="FlowAgent",
                        status=NodeStatus.COMPLETED,
                        success=True,
                        duration=time.perf_counter() - start_time,
                        tool_name=function_name,
                        tool_args=kwargs,
                        tool_result=result,
                        is_meta_tool=False,  # Klarstellen, dass es kein Meta-Tool ist
                        metadata={
                            "result_type": type(result).__name__,
                            "result_length": len(str(result)),
                        },
                    )
                )
            rprint(
                f"Function {function_name} completed successfully with result: {result}"
            )
            return result

        except Exception as e:
            eprint(f"Function {function_name} execution failed: {e}")
            raise

        finally:
            self.resent_tools_called.append([function_name, args, kwargs, result])

    # ===== FORMATTING =====

    async def a_format_class(self,
                             pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 1,  # Reduced from 2
                             auto_context=False,    # Changed default to False
                             session_id: str = None,
                             llm_kwargs=None,
                             model_preference="fast",  # Changed default to fast
                             lean_mode=True,  # NEW: Enable lean mode by default
                             **kwargs) -> dict[str, Any]:
        """
        Optimized LLM-based structured data formatting.
        lean_mode=True uses ~80% fewer tokens.
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM required")

        if session_id and self.active_session != session_id:
            self.active_session = session_id

        schema = pydantic_model.model_json_schema()
        model_name = pydantic_model.__name__

        if lean_mode:
            # LEAN MODE: Minimal schema, no examples
            props = schema.get("properties", {})
            required = set(schema.get("required", []))

            fields_desc = []
            for name, info in props.items():
                ftype = info.get("type", "string")
                req = "*" if name in required else ""
                fields_desc.append(f"  {name}{req}: {ftype}")

            enhanced_prompt = f"""{prompt}

Return YAML with fields:
{chr(10).join(fields_desc)}
"""
        else:
            # ORIGINAL: Full schema (fallback)
            enhanced_prompt = f"""
{prompt}

SCHEMA FOR {model_name}:
{yaml.dump(safe_for_yaml(schema), default_flow_style=False, indent=2)}

Respond in YAML format only:
"""

        messages = []
        if message_context:
            messages.extend(message_context)
        messages.append({"role": "user", "content": enhanced_prompt})

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                temperature = 0.1 + (attempt * 0.1)
                max_tokens = 500 if lean_mode else min(2000 + (attempt * 500), 4000)

                response = await self.a_run_llm_completion(
                    model_preference=model_preference,
                    messages=messages,
                    stream=False,
                    with_context=auto_context,  # Respect auto_context setting
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task_id=f"format_{model_name.lower()}_{attempt}",
                    llm_kwargs=llm_kwargs
                )

                if not response or not response.strip():
                    raise ValueError("Empty response")

                yaml_content = self._extract_yaml_content(response)
                if not yaml_content:
                    raise ValueError("No YAML found")

                try:
                    parsed_data = yaml.safe_load(yaml_content)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML: {e}")

                if not isinstance(parsed_data, dict):
                    raise ValueError(f"Expected dict, got {type(parsed_data)}")

                try:
                    validated_instance = pydantic_model.model_validate(parsed_data)
                    return validated_instance.model_dump()
                except ValidationError as e:
                    errors = [f"{' -> '.join(str(x) for x in err['loc'])}: {err['msg']}"
                              for err in e.errors()]
                    raise ValueError("Validation failed: " + "; ".join(errors))

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    messages[-1]["content"] = enhanced_prompt + f"\\n\\nFix error: {str(e)}"

        raise RuntimeError(f"Failed after {max_retries + 1} attempts: {last_error}")

    def _extract_yaml_content(self, response: str) -> str:
        """Extract YAML content from LLM response with multiple strategies"""
        # Strategy 1: Extract from code blocks
        if "```yaml" in response:
            try:
                yaml_content = response.split("```yaml")[1].split("```")[0].strip()
                if yaml_content:
                    return yaml_content
            except IndexError:
                pass

        # Strategy 2: Extract from generic code blocks
        if "```" in response:
            try:
                parts = response.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd indices are inside code blocks
                        # Skip if it starts with a language identifier
                        lines = part.strip().split('\n')
                        if lines and not lines[0].strip().isalpha():
                            return part.strip()
                        elif len(lines) > 1:
                            # Try without first line
                            return '\n'.join(lines[1:]).strip()
            except:
                pass

        # Strategy 3: Look for YAML-like patterns
        lines = response.split('\n')
        yaml_lines = []
        in_yaml = False

        for line in lines:
            stripped = line.strip()

            # Detect start of YAML-like content
            if ':' in stripped and not stripped.startswith('#'):
                in_yaml = True
                yaml_lines.append(line)
            elif in_yaml:
                if stripped == '' or stripped.startswith(' ') or stripped.startswith('-') or ':' in stripped:
                    yaml_lines.append(line)
                else:
                    # Potential end of YAML
                    break

        if yaml_lines:
            return '\n'.join(yaml_lines).strip()

        # Strategy 4: Return entire response if it looks like YAML
        if ':' in response and not response.strip().startswith('<'):
            return response.strip()

        return ""

    async def a_stream(
        self,
        query: str,
        session_id: str = None,
        user_id: str = None,
        agentic_actions: bool = False,  # Flag für Tool-Aufrufe im Streaming
        voice_output: int | str = 0,  # 0=normal, 1=verbalisiert, custom strings auch möglich
        context_mode: str = "auto",
        with_context: bool = True,
        remember: bool = True,
        model_preference: str = "fast",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response with optional agentic tool execution and voice formatting.

        Args:
            query: User input
            session_id: Session identifier
            user_id: user idedifaction
            remember: Whether to save to conversation history
            agentic_actions: If True, agent can decide to use tools during streaming
            voice_output: 0=normal text, 1=verbalized for TTS, 2=with [emotion] markers
            context_mode: "full" or "minimal" or "persona" or "none" or "auto"
            with_context: Include session context
            model_preference: fast or complex
            **kwargs: Additional arguments

        Yields:
            str: Response chunks
        """
        # Set active session
        if session_id:
            self.active_session = session_id

        # Build voice formatting instructions

        if voice_output == 0:
            voice_instructions = ""
        elif voice_output == 1:
            voice_instructions = """
IMPORTANT: Format your response for voice/speech output:
- Use natural, conversational language
- Avoid markdown, code blocks, or special formatting
- Spell out abbreviations and numbers naturally
- Keep sentences clear and flowing
- Avoid lists - use prose instead
"""
        else:
            voice_instructions = voice_output

        # Build agentic decision prompt addition
        agentic_prompt = ""
        if agentic_actions:
            agentic_prompt = f"""
You have access to tools. If the user's request would benefit from tool usage,
start your response with exactly: [TOOLS_NEEDED]
Then continue with a brief acknowledgment.

Only use [TOOLS_NEEDED] if tools are truly necessary for this request.
    """

        await self.initialize_session_context(session_id, max_history=200)
        user_content = query
        if voice_instructions:
            user_content = f"{voice_instructions}\n\nUser request: {query}"
        if agentic_prompt:
            user_content = f"{agentic_prompt}\n\n{user_content}"

        self.shared.update({
            "current_query": user_content,
            "session_id": session_id,
            "user_id": user_id,
            "stream_callback": None,
            "remember": remember,
            # CENTRAL: Context Manager ist die primäre Context-Quelle
            "context_manager": self.context_manager,
            "variable_manager": self.variable_manager,
            "fast_run": True,  # fast_run-Flag übergeben
        })
        self.active_sessio = session_id

        if remember:
            await self.context_manager.add_interaction(session_id, 'user', query)
        # Build messages
        messages = [{"role": "system", "content": "You are a helpful assistant. To use a tool, must start your response with [TOOLS_NEEDED] and text for the user the tool will be called automatically."}]

        if self.active_session and with_context:
            # Add context to fist messages as system message
            # OPTIMIZED: Conditional context injection
            context_mode = context_mode or "auto" if with_context else "none"

            if context_mode == "full" and self.active_session:
                # Full context (original behavior)
                context_ = await self.get_context(self.active_session)
                messages = [
                    {
                        "role": "system",
                        "content": self.amd.get_system_message_with_persona()
                        + "\n\nContext:\n\n"
                        + str(context_),
                    }
                ]

            elif context_mode == "minimal" and self.active_session:
                # Minimal: Only persona + last interaction
                last = (
                    await self.context_manager.get_contextual_history(self.active_session, max_entries=3)
                    if self.context_manager
                    else []
                )
                messages = [
                    {
                        "role": "system",
                        "content": self.amd.get_system_message_with_persona()
                    }
                ] + last

            elif context_mode == "persona" and self.active_session:
                # Persona only, no context
                messages = [{"role": "system", "content": self.amd.get_system_message_with_persona()}]

            # "none" or "auto" with format_ task = no context injection

            elif context_mode == "auto" and with_context and self.active_session:
                task_id = kwargs.get("task_id", "")
                if not task_id.startswith("format_") and not task_id.startswith("lean_"):
                    # Only inject for non-formatting tasks
                    context_ = await self.get_context(self.active_session)
                    messages = [{"role": "system", "content": self.amd.get_system_message_with_persona()+'\n\nContext:\n\n'+context_}] + kwargs.get("messages", [])

        # Build user message with instructions
        messages.append({"role": "user", "content": user_content})
        # Stream response
        full_response = ""
        tools_needed = False

        try:
            # Initial streaming call
            response = await self.llm_handler.completion_with_rate_limiting(
                litellm,
                model=self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_response += content

                # Check for tools marker in first chunks
                if agentic_actions and len(full_response) < 50:
                    if "[TOOLS_NEEDED]" in full_response:
                        tools_needed = True
                        # Don't yield the marker
                        content = content.replace("[TOOLS_NEEDED]", "").strip()

                if content:
                    yield content

                # Emit progress event
                if self.progress_tracker and content:
                    await self.progress_tracker.emit_event(ProgressEvent(
                        event_type="llm_stream_chunk",
                        node_name="a_stream",
                        session_id=session_id,
                        status=NodeStatus.RUNNING,
                        llm_model=self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model,
                        llm_output=content,
                    ))

            # Handle agentic tool execution if needed
            if tools_needed and agentic_actions:
                yield "\n\n"  # Separator

                # Use a_format_class to select tools
                class ToolSelection(BaseModel):
                    """Selection of tools to execute"""
                    tools: list[str] = Field(description="List of tool names to use")
                    reasoning: str = Field(description="Brief reasoning for tool selection")
                    execution_order: list[dict] = Field(
                        description="Ordered list of tool calls with args",
                        default_factory=list
                    )

                # Get tool selection from agent
                selection = await self.a_format_class(
                    pydantic_model=ToolSelection,
                    prompt=f"""Based on this user request, select the most appropriate tools:

User request: {query}

Available tools and their capabilities:
{json.dumps({k: v.get('description', '')[:100] for k, v in list(self._tool_capabilities.items())[:15]}, indent=2)}

Select tools and specify execution order with arguments.""",
                    model_preference="fast",
                    lean_mode=True,
                    auto_context=True
                )

                if selection and selection.get("tools"):
                    # Inform user about tool execution
                    if voice_output >= 1:
                        yield "[working] Let me use some tools to help with that. "
                    else:
                        yield "Using tools: "

                    # Execute selected tools
                    for tool_call in selection.get("execution_order", []):
                        tool_name = tool_call.get("tool") or tool_call.get("name")
                        tool_args = tool_call.get("args", {})

                        if tool_name and tool_name in self.tool_registry:
                            try:
                                # Notify about tool execution
                                if voice_output == 0:
                                    yield f"`{tool_name}`... "

                                # Execute tool
                                result = await self.arun_function(tool_name, **tool_args)

                                # Generate response based on tool result
                                tool_response_messages = messages + [
                                    {"role": "assistant", "content": full_response},
                                    {"role": "user",
                                     "content": f"Tool {tool_name} returned: {str(result)[:1000]}\n\nProvide a natural response incorporating this result.{voice_instructions}"}
                                ]

                                # Stream tool result explanation
                                tool_response = await self.llm_handler.completion_with_rate_limiting(
                                    litellm,
                                    model=self.amd.fast_llm_model,
                                    messages=tool_response_messages,
                                    stream=True,
                                )

                                async for chunk in tool_response:
                                    content = chunk.choices[0].delta.content or ""
                                    if content:
                                        full_response += content
                                        yield content

                            except Exception as e:
                                if voice_output >= 1:
                                    yield f"[concerned] I encountered an issue with {tool_name}. "
                                else:
                                    yield f"(Error with {tool_name}: {str(e)[:50]}) "

            # Save to conversation history
            if remember and session_id:
                await self.context_manager.add_interaction(session_id, 'assistant', full_response.replace("[TOOLS_NEEDED]", ""))

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if voice_output >= 1:
                yield f"[apologetic] I'm sorry, something went wrong. {error_msg}"
            else:
                yield error_msg
            raise


    # ===== SERVER SETUP =====

    def setup_a2a_server(self, host: str = "0.0.0.0", port: int = 5000, **kwargs):
        """Setup A2A server for bidirectional communication"""
        if not A2A_AVAILABLE:
            wprint("A2A not available, cannot setup server")
            return

        try:
            self.a2a_server = A2AServer(
                host=host,
                port=port,
                agent_card=AgentCard(
                    name=self.amd.name,
                    description="Production-ready PocketFlow agent",
                    version="1.0.0"
                ),
                **kwargs
            )

            # Register agent methods
            @self.a2a_server.route("/run")
            async def handle_run(request_data):
                query = request_data.get("query", "")
                session_id = request_data.get("session_id", "a2a_session")

                response = await self.a_run(query, session_id=session_id)
                return {"response": response}

            rprint(f"A2A server setup on {host}:{port}")

        except Exception as e:
            eprint(f"Failed to setup A2A server: {e}")

    def setup_mcp_server(self, host: str = "0.0.0.0", port: int = 8000, name: str = None, **kwargs):
        """Setup MCP server"""
        if not MCP_AVAILABLE:
            wprint("MCP not available, cannot setup server")
            return

        try:
            server_name = name or f"{self.amd.name}_MCP"
            self.mcp_server = FastMCP(server_name)

            # Register agent as MCP tool
            @self.mcp_server.tool()
            async def agent_run(query: str, session_id: str = "mcp_session") -> str:
                """Execute agent with given query"""
                return await self.a_run(query, session_id=session_id)

            rprint(f"MCP server setup: {server_name}")

        except Exception as e:
            eprint(f"Failed to setup MCP server: {e}")

    # ===== LIFECYCLE MANAGEMENT =====

    async def start_servers(self):
        """Start all configured servers"""
        tasks = []

        if self.a2a_server:
            tasks.append(asyncio.create_task(self.a2a_server.start()))

        if self.mcp_server:
            tasks.append(asyncio.create_task(self.mcp_server.run()))

        if tasks:
            rprint(f"Starting {len(tasks)} servers...")
            await asyncio.gather(*tasks, return_exceptions=True)

    def clear_context(self, session_id: str = None) -> bool:
        """Clear context über UnifiedContextManager mit Session-spezifischer Unterstützung"""
        try:
            return True

        except Exception as e:
            eprint(f"Failed to clear context: {e}")
            return False

    async def clean_memory(self, deep_clean: bool = False) -> bool:
        """Clean memory and context of the agent"""
        try:
            # Clear current context first
            self.clear_context()

            # Clean world model
            self.world_model = {}

            session_managers = {}
            # Deep clean session storage
            if session_managers:
                for _manager_name, manager in session_managers.items():
                    if hasattr(manager, 'clear_all_history'):
                        await manager.clear_all_history()
                    elif hasattr(manager, 'clear_history'):
                        manager.clear_history()


            # Clean checkpoint data
            self.checkpoint_data = {}
            self.last_checkpoint = None

            # Force garbage collection
            import gc
            gc.collect()

            rprint(f"Memory cleaned (deep_clean: {deep_clean})")
            return True

        except Exception as e:
            eprint(f"Failed to clean memory: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        self.is_running = False
        self._shutdown_event.set()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Close servers
        if self.a2a_server:
            await self.a2a_server.close()

        if self.mcp_server:
            await self.mcp_server.close()

        if hasattr(self, '_mcp_session_manager'):
            await self._mcp_session_manager.cleanup_all()

        rprint("Agent shutdown complete")

    # ===== MCP CIRCUIT BREAKER METHODS (P0 - KRITISCH) =====

    def _check_mcp_circuit_breaker(self, server_name: str) -> bool:
        """Check if MCP circuit breaker allows requests for this server"""
        if server_name not in self.mcp_session_health:
            self.mcp_session_health[server_name] = {
                "failures": 0,
                "last_failure": 0.0,
                "state": "CLOSED"
            }

        health = self.mcp_session_health[server_name]
        now = time.time()

        # Check circuit state
        if health["state"] == "OPEN":
            # Check if timeout has passed to try HALF_OPEN
            if now - health["last_failure"] > self.mcp_circuit_breaker_timeout:
                health["state"] = "HALF_OPEN"
                rprint(f"MCP Circuit Breaker for {server_name}: OPEN -> HALF_OPEN (retry)")
                return True
            else:
                # Circuit still open
                return False

        return True  # CLOSED or HALF_OPEN allows requests

    def _record_mcp_success(self, server_name: str):
        """Record successful MCP call"""
        if server_name in self.mcp_session_health:
            health = self.mcp_session_health[server_name]
            health["failures"] = 0
            if health["state"] == "HALF_OPEN":
                health["state"] = "CLOSED"
                rprint(f"MCP Circuit Breaker for {server_name}: HALF_OPEN -> CLOSED (recovered)")

    def _record_mcp_failure(self, server_name: str):
        """Record failed MCP call and update circuit breaker state"""
        if server_name not in self.mcp_session_health:
            self.mcp_session_health[server_name] = {
                "failures": 0,
                "last_failure": 0.0,
                "state": "CLOSED"
            }

        health = self.mcp_session_health[server_name]
        health["failures"] += 1
        health["last_failure"] = time.time()

        # Open circuit if threshold exceeded
        if health["failures"] >= self.mcp_circuit_breaker_threshold:
            if health["state"] != "OPEN":
                health["state"] = "OPEN"
                eprint(f"MCP Circuit Breaker for {server_name}: OPENED after {health['failures']} failures")

    # ===== VOTING METHOD FOR FLOWAGENT =====

    async def voting_as_tool(self):

        if "voting" in self._tool_registry:
            return

        async def a_voting(**kwargs):
            return await self.a_voting(session_id=self.active_session, **kwargs)

        await self.add_tool(
            a_voting,
            "voting",
            description="""Advanced AI voting system with First-to-ahead-by-k algorithm.

Args:
    prompt: Main prompt/question for voting
    k_margin: Required vote margin to declare winner
    num_voters: Number of voters (simple mode)
    votes_per_voter: Votes each voter can cast (simple mode)
    vote_on_parts: Vote on parts vs structures (unstructured mode)
    context: addtional context
    task_id: Task identifier for tracking

Returns:
    dict: Voting results with winner, votes, margin, and cost info"""
        )

    async def a_voting(
        self,
        prompt: str = None,
        k_margin: int = 2,
        num_voters: int = 3,
        votes_per_voter: int = 1,
        task_id: str = "voting_task",
        session_id: str = None,
        context: str = "",
        **kwargs
    ) -> dict[str, Any]:
        """
        Advanced AI voting system with First-to-ahead-by-k algorithm.

        Modes:
        - simple: Vote on predefined options with multiple voters
        - advanced: Thinkers analyze, then best/vote/recombine strategies
        - unstructured: Organize data, vote on parts/structures, optional final construction

        Args:
            mode: Voting mode (simple/advanced/unstructured)
            prompt: Main prompt/question for voting
            options: List of options (simple mode)
            k_margin: Required vote margin to declare winner
            num_voters: Number of voters (simple mode)
            votes_per_voter: Votes each voter can cast (simple mode)
            num_thinkers: Number of thinkers (advanced mode)
            strategy: Strategy for advanced mode (best/vote/recombine)
            num_organizers: Number of organizers (unstructured mode)
            vote_on_parts: Vote on parts vs structures (unstructured mode)
            final_construction: Create final output (unstructured mode)
            unstructured_data: Raw data to organize (unstructured mode)
            complex_data: Use complex model for thinking/organizing phases
            task_id: Task identifier for tracking
            session_id: Session ID
            **kwargs: Additional arguments

        Returns:
            dict: Voting results with winner, votes, margin, and cost info

        Example:
            # Simple voting
            result = await agent.a_voting(
                mode="simple",
                prompt="Which approach is best?",
                options=["Approach A", "Approach B", "Approach C"],
                k_margin=2,
                num_voters=5
            )

            # Advanced with thinking
            result = await agent.a_voting(
                mode="advanced",
                prompt="Analyze the problem and propose solutions",
                num_thinkers=3,
                strategy="recombine",
                complex_data=True
            )
        """

        # Get voting model from env or use fast model
        voting_model = os.getenv("VOTING_MODEL")

        # Track costs
        start_tokens_in = self.total_tokens_in
        start_tokens_out = self.total_tokens_out
        start_cost = self.total_cost_accumulated

        try:

            return "result"

        except Exception as e:
            print(f"[Voting Error] {e}")
            raise

    @property
    def total_cost(self) -> float:
        """Get total accumulated cost from LLM calls"""
        # Return accumulated cost from tracking, fallback to budget manager if available
        if self.total_cost_accumulated > 0:
            return self.total_cost_accumulated
        if hasattr(self.amd, 'budget_manager') and self.amd.budget_manager:
            return getattr(self.amd.budget_manager, 'total_cost', 0.0)
        return 0.0

    async def get_context_overview(self, session_id: str = None, display: bool = False) -> dict[str, Any]:
        """
        Detaillierte Übersicht des aktuellen Contexts mit Token-Counts und optionaler Display-Darstellung

        Args:
            session_id: Session ID für context (default: active_session)
            display: Ob die Übersicht im Terminal-Style angezeigt werden soll

        Returns:
            dict: Detaillierte Context-Übersicht mit Raw-Daten und Token-Counts
        """
        try:
            session_id = session_id or self.active_session or "default"

            # Token counting function
            def count_tokens(text: str) -> int:
                """Einfache Token-Approximation (4 chars ≈ 1 token für deutsche/englische Texte)"""
                try:
                    from litellm import token_counter
                    return token_counter(self.amd.fast_llm_model, text=text)
                except:
                    pass
                return max(1, len(str(text)) // 4)

            context_overview = {
                "session_info": {
                    "session_id": session_id,
                    "agent_name": self.amd.name,
                    "timestamp": datetime.now().isoformat(),
                    "active_session": self.active_session,
                    "is_running": self.is_running
                },
                "system_prompt": {},
                "meta_tools": {},
                "agent_tools": {},
                "mcp_tools": {},
                "variables": {},
                "system_history": {},
                "unified_context": {},
                "reasoning_context": {},
                "llm_tool_context": {},
                "token_summary": {}
            }

            # === SYSTEM PROMPT ANALYSIS ===
            system_message = self.amd.get_system_message_with_persona()
            context_overview["system_prompt"] = {
                "raw_data": system_message,
                "token_count": count_tokens(system_message),
                "components": {
                    "base_message": self.amd.system_message,
                    "persona_active": self.amd.persona is not None,
                    "persona_name": self.amd.persona.name if self.amd.persona else None,
                    "persona_integration": self.amd.persona.apply_method if (self.amd.persona and hasattr(self.amd.persona, 'apply_method')) else None
                }
            }

            # === META TOOLS ANALYSIS ===
            if hasattr(self.task_flow, 'llm_reasoner') and hasattr(self.task_flow.llm_reasoner, 'meta_tools_registry'):
                meta_tools = self.task_flow.llm_reasoner.meta_tools_registry
            else:
                meta_tools = {}

            meta_tools_info = ""
            for tool_name, tool_info in meta_tools.items():
                tool_desc = tool_info.get("description", "No description")
                meta_tools_info += f"{tool_name}: {tool_desc}\n"

            # Standard Meta-Tools
            standard_meta_tools = [
                "internal_reasoning", "manage_internal_task_stack", "delegate_to_llm_tool_node",
                "create_and_execute_plan", "advance_outline_step", "write_to_variables",
                "read_from_variables", "direct_response"
            ]

            for meta_tool in standard_meta_tools:
                meta_tools_info += f"{meta_tool}: Built-in meta-tool for agent orchestration\n"

            context_overview["meta_tools"] = {
                "raw_data": meta_tools_info,
                "token_count": count_tokens(meta_tools_info),
                "count": len(meta_tools) + len(standard_meta_tools),
                "custom_meta_tools": list(meta_tools.keys()),
                "standard_meta_tools": standard_meta_tools
            }

            # === AGENT TOOLS ANALYSIS ===
            tools_info = ""
            tool_capabilities_text = ""

            for tool_name in self.shared.get("available_tools", []):
                tool_data = self._tool_registry.get(tool_name, {})
                description = tool_data.get("description", "No description")
                args_schema = tool_data.get("args_schema", "()")
                tools_info += f"{tool_name}{args_schema}: {description}\n"

                # Tool capabilities if available
                if tool_name in self._tool_capabilities:
                    cap = self._tool_capabilities[tool_name]
                    primary_function = cap.get("primary_function", "Unknown")
                    use_cases = cap.get("use_cases", [])
                    tool_capabilities_text += f"{tool_name}: {primary_function}\n"
                    if use_cases:
                        tool_capabilities_text += f"  Use cases: {', '.join(use_cases[:3])}\n"

            context_overview["agent_tools"] = {
                "raw_data": tools_info,
                "capabilities_data": tool_capabilities_text,
                "token_count": count_tokens(tools_info + tool_capabilities_text),
                "count": len(self.shared.get("available_tools", [])),
                "analyzed_count": len(self._tool_capabilities),
                "tool_names": self.shared.get("available_tools", []),
                "intelligence_level": "high" if self._tool_capabilities else "basic"
            }

            # === MCP TOOLS ANALYSIS ===
            # Placeholder für MCP Tools (falls implementiert)
            mcp_tools_info = "No MCP tools currently active"
            if self.mcp_server:
                mcp_tools_info = f"MCP Server active: {getattr(self.mcp_server, 'name', 'Unknown')}"

            context_overview["mcp_tools"] = {
                "raw_data": mcp_tools_info,
                "token_count": count_tokens(mcp_tools_info),
                "server_active": bool(self.mcp_server),
                "server_name": getattr(self.mcp_server, 'name', None) if self.mcp_server else None
            }

            # === VARIABLES ANALYSIS ===
            variables_text = ""
            if self.variable_manager:
                variables_text = self.variable_manager.get_llm_variable_context()
            else:
                variables_text = "No variable manager available"

            context_overview["variables"] = {
                "raw_data": variables_text,
                "token_count": count_tokens(variables_text),
                "manager_available": bool(self.variable_manager),
                "total_scopes": len(self.variable_manager.scopes) if self.variable_manager else 0,
                "scope_names": list(self.variable_manager.scopes.keys()) if self.variable_manager else []
            }

            # === SYSTEM HISTORY ANALYSIS ===
            history_text = ""
            if self.context_manager and session_id in self.context_manager.session_managers:
                session = self.context_manager.session_managers[session_id]
                if hasattr(session, 'history'):
                    history_count = len(session.history)
                    history_text = f"Session History: {history_count} messages\n"

                    # Recent messages preview
                    for msg in session.history[-3:]:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:100] + "..." if len(
                            msg.get('content', '')) > 100 else msg.get('content', '')
                        timestamp = msg.get('timestamp', '')[:19]
                        history_text += f"[{timestamp}] {role}: {content}\n"
                elif isinstance(session, dict) and 'history' in session:
                    history_count = len(session['history'])
                    history_text = f"Fallback Session History: {history_count} messages"
            else:
                history_text = "No session history available"

            context_overview["system_history"] = {
                "raw_data": history_text,
                "token_count": count_tokens(history_text),
                "session_initialized": self.shared.get("session_initialized", False),
                "context_manager_available": bool(self.context_manager),
                "session_count": len(self.context_manager.session_managers) if self.context_manager else 0
            }

            # === UNIFIED CONTEXT ANALYSIS ===
            unified_context_text = ""
            try:
                unified_context = await self.context_manager.build_unified_context(session_id, "",
                                                                                   "full") if self.context_manager else {}
                if unified_context:
                    formatted_context = self.context_manager.get_formatted_context_for_llm(unified_context)
                    unified_context_text = formatted_context
                else:
                    unified_context_text = "No unified context available"
            except Exception as e:
                unified_context_text = f"Error building unified context: {str(e)}"

            context_overview["unified_context"] = {
                "raw_data": unified_context_text,
                "token_count": count_tokens(unified_context_text),
                "build_successful": "Error" not in unified_context_text,
                "manager_available": bool(self.context_manager)
            }

            # === REASONING CONTEXT ANALYSIS ===
            reasoning_context_text = ""
            if hasattr(self.task_flow, 'llm_reasoner') and hasattr(self.task_flow.llm_reasoner, 'reasoning_context'):
                reasoning_context = self.task_flow.llm_reasoner.reasoning_context
                reasoning_context_text = f"Reasoning Context: {len(reasoning_context)} entries\n"

                # Recent reasoning entries
                for entry in reasoning_context[-3:]:
                    entry_type = entry.get('type', 'unknown')
                    content = str(entry.get('content', ''))[:150] + "..." if len(
                        str(entry.get('content', ''))) > 150 else str(entry.get('content', ''))
                    reasoning_context_text += f"  {entry_type}: {content}\n"
            else:
                reasoning_context_text = "No reasoning context available"

            context_overview["reasoning_context"] = {
                "raw_data": reasoning_context_text,
                "token_count": count_tokens(reasoning_context_text),
                "reasoner_available": hasattr(self.task_flow, 'llm_reasoner'),
                "context_entries": len(self.task_flow.llm_reasoner.reasoning_context) if hasattr(self.task_flow,
                                                                                                 'llm_reasoner') and hasattr(
                    self.task_flow.llm_reasoner, 'reasoning_context') else 0
            }

            # === LLM TOOL CONTEXT ANALYSIS ===
            llm_tool_context_text = ""
            if hasattr(self.task_flow, 'llm_tool_node'):
                llm_tool_context_text = f"LLM Tool Node available with max {self.task_flow.llm_tool_node.max_tool_calls} tool calls\n"
                if hasattr(self.task_flow.llm_tool_node, 'call_log'):
                    call_log = self.task_flow.llm_tool_node.call_log
                    llm_tool_context_text += f"Call log: {len(call_log)} entries\n"
            else:
                llm_tool_context_text = "No LLM Tool Node available"

            context_overview["llm_tool_context"] = {
                "raw_data": llm_tool_context_text,
                "token_count": count_tokens(llm_tool_context_text),
                "node_available": hasattr(self.task_flow, 'llm_tool_node'),
                "max_tool_calls": getattr(self.task_flow.llm_tool_node, 'max_tool_calls', 0) if hasattr(self.task_flow,
                                                                                                        'llm_tool_node') else 0
            }

            # === TOKEN SUMMARY ===
            total_tokens = sum([
                context_overview["system_prompt"]["token_count"],
                context_overview["meta_tools"]["token_count"],
                context_overview["agent_tools"]["token_count"],
                context_overview["mcp_tools"]["token_count"],
                context_overview["variables"]["token_count"],
                context_overview["system_history"]["token_count"],
                context_overview["unified_context"]["token_count"],
                context_overview["reasoning_context"]["token_count"],
                context_overview["llm_tool_context"]["token_count"]
            ])

            context_overview["token_summary"] = {
                "total_tokens": total_tokens,
                "breakdown": {
                    "system_prompt": context_overview["system_prompt"]["token_count"],
                    "meta_tools": context_overview["meta_tools"]["token_count"],
                    "agent_tools": context_overview["agent_tools"]["token_count"],
                    "mcp_tools": context_overview["mcp_tools"]["token_count"],
                    "variables": context_overview["variables"]["token_count"],
                    "system_history": context_overview["system_history"]["token_count"],
                    "unified_context": context_overview["unified_context"]["token_count"],
                    "reasoning_context": context_overview["reasoning_context"]["token_count"],
                    "llm_tool_context": context_overview["llm_tool_context"]["token_count"]
                },
                "percentage_breakdown": {}
            }

            # Calculate percentages
            for component, token_count in context_overview["token_summary"]["breakdown"].items():
                percentage = (token_count / total_tokens * 100) if total_tokens > 0 else 0
                context_overview["token_summary"]["percentage_breakdown"][component] = round(percentage, 1)

            # === DISPLAY OUTPUT ===
            if display:
                await self._display_context_overview(context_overview)

            return context_overview

        except Exception as e:
            eprint(f"Error generating context overview: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }

    async def _display_context_overview(self, overview: dict[str, Any]):
        """Display context overview in terminal-style format similar to the image"""
        try:
            from toolboxv2.utils.extras.Style import Spinner

            print("\n" + "=" * 80)
            print("🔍 FLOW AGENT CONTEXT OVERVIEW")
            print("=" * 80)

            # Session Info
            session_info = overview["session_info"]
            print(f"📅 Session: {session_info['session_id']} | Agent: {session_info['agent_name']}")
            print(f"⏰ Generated: {session_info['timestamp'][:19]} | Running: {session_info['is_running']}")

            # Token Summary (like in the image)
            token_summary = overview["token_summary"]
            total_tokens = token_summary["total_tokens"]
            breakdown = token_summary["percentage_breakdown"]

            print(f"\n📊 CONTEXT USAGE")
            print(f"Total Context: ~{total_tokens:,} tokens")

            # Create visual bars like in the image
            bar_length = 50

            try:mf=get_max_tokens(self.amd.fast_llm_model.split('/')[-1]);self.amd.max_tokens = mf
            except:mf = self.amd.max_tokens
            try:mc=get_max_tokens(self.amd.complex_llm_model.split('/')[-1]);self.amd.max_tokens = mf
            except:mc = self.amd.max_tokens
            components = [
                ("System prompt", breakdown.get("system_prompt", 0), "🔧"),
                ("Agent tools", breakdown.get("agent_tools", 0), "🛠️"),
                ("Meta tools", breakdown.get("meta_tools", 0), "⚡"),
                ("Variables", breakdown.get("variables", 0), "📝"),
                ("History", breakdown.get("system_history", 0), "📚"),
                ("Unified ctx", breakdown.get("unified_context", 0), "🔗"),
                ("Reasoning", breakdown.get("reasoning_context", 0), "🧠"),
                ("LLM Tools", breakdown.get("llm_tool_context", 0), "🤖"),
                ("Free Space F", mf, "⬜"),
                ("Free Space C", mc, "⬜"),

            ]

            for name, percentage, icon in components:
                if percentage > 0:
                    filled_length = int(percentage * bar_length / 100)
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    tokens = int(total_tokens * percentage / 100)
                    print(f"{icon} {name:13}: {bar} {percentage:5.1f}% ({tokens:,} tokens)") if not name.startswith("Free") else print(f"{icon} {name:13}: ({tokens:,} tokens) used {total_tokens/tokens*100:.3f}%")

            # Detailed breakdowns
            sections = [
                ("🔧 SYSTEM PROMPT", "system_prompt"),
                ("⚡ META TOOLS", "meta_tools"),
                ("🛠️ AGENT TOOLS", "agent_tools"),
                ("📝 VARIABLES", "variables"),
                ("📚 SYSTEM HISTORY", "system_history"),
                ("🔗 UNIFIED CONTEXT", "unified_context"),
                ("🧠 REASONING CONTEXT", "reasoning_context"),
                ("🤖 LLM TOOL CONTEXT", "llm_tool_context")
            ]

            for title, key in sections:
                section_data = overview.get(key, {})
                token_count = section_data.get("token_count", 0)

                if token_count > 0:
                    print(f"\n{title} ({token_count:,} tokens)")
                    print("-" * 50)

                    # Show component-specific info
                    if key == "agent_tools":
                        print(f"  Available tools: {section_data.get('count', 0)}")
                        print(f"  Analyzed tools: {section_data.get('analyzed_count', 0)}")
                        print(f"  Intelligence: {section_data.get('intelligence_level', 'unknown')}")
                    elif key == "variables":
                        print(f"  Manager available: {section_data.get('manager_available', False)}")
                        print(f"  Total scopes: {section_data.get('total_scopes', 0)}")
                        print(f"  Scope names: {', '.join(section_data.get('scope_names', []))}")
                    elif key == "system_history":
                        print(f"  Session initialized: {section_data.get('session_initialized', False)}")
                        print(f"  Total sessions: {section_data.get('session_count', 0)}")
                    elif key == "reasoning_context":
                        print(f"  Reasoner available: {section_data.get('reasoner_available', False)}")
                        print(f"  Context entries: {section_data.get('context_entries', 0)}")
                    elif key == "meta_tools":
                        print(f"  Total meta tools: {section_data.get('count', 0)}")
                        custom = section_data.get('custom_meta_tools', [])
                        if custom:
                            print(f"  Custom tools: {', '.join(custom)}")

                    # Show raw data preview if reasonable size
                    raw_data = section_data.get('raw_data', '')
                    if len(raw_data) <= 200:
                        print(f"  Preview: {raw_data[:200]}...")

            print("\n" + "=" * 80)
            print(f"💾 Total Context Size: ~{total_tokens:,} tokens")
            print("=" * 80 + "\n")

        except Exception as e:
            eprint(f"Error displaying context overview: {e}")
            # Fallback to simple display
            print(f"\n=== CONTEXT OVERVIEW (Fallback) ===")
            print(f"Total Tokens: {overview.get('token_summary', {}).get('total_tokens', 0):,}")
            for key, data in overview.items():
                if isinstance(data, dict) and 'token_count' in data:
                    print(f"{key}: {data['token_count']:,} tokens")
            print("=" * 40)

    async def status(self, pretty_print: bool = False) -> dict[str, Any] | str:
        """Get comprehensive agent status with optional pretty printing"""

        # Core status information
        base_status = {
            "agent_info": {
                "name": self.amd.name,
                "version": "2.0",
                "type": "FlowAgent"
            },
            "runtime_status": {
                "status": self.shared.get("system_status", "idle"),
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "uptime_seconds": (datetime.now() - getattr(self, '_start_time', datetime.now())).total_seconds()
            },
            "task_execution": {
                "total_tasks": len(self.shared.get("tasks", {})),
                "active_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "running"]),
                "completed_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "completed"]),
                "failed_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "failed"]),
                "plan_adaptations": self.shared.get("plan_adaptations", 0)
            },
            "conversation": {
                "turns": len(self.shared.get("conversation_history", [])),
                "session_id": self.shared.get("session_id", self.active_session),
                "current_user": self.shared.get("user_id"),
                "last_query": self.shared.get("current_query", "")[:100] + "..." if len(
                    self.shared.get("current_query", "")) > 100 else self.shared.get("current_query", "")
            },
            "capabilities": {
                "available_tools": len(self.shared.get("available_tools", [])),
                "tool_names": list(self.shared.get("available_tools", [])),
                "analyzed_tools": len(self._tool_capabilities),
                "world_model_size": len(self.shared.get("world_model", {})),
                "intelligence_level": "high" if self._tool_capabilities else "basic"
            },
            "memory_context": {
                "session_initialized": self.shared.get("session_initialized", False),
                "session_managers": len(self.shared.get("session_managers", {})),
                "context_system": "advanced_session_aware" if self.shared.get("session_initialized") else "basic",
                "variable_scopes": len(self.variable_manager.get_scope_info()) if hasattr(self,
                                                                                          'variable_manager') else 0
            },
            "performance": {
                "total_cost": self.total_cost,
                "checkpoint_enabled": self.enable_pause_resume,
                "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
                "max_parallel_tasks": self.max_parallel_tasks
            },
            "servers": {
                "a2a_server": self.a2a_server is not None,
                "mcp_server": self.mcp_server is not None,
                "server_count": sum([self.a2a_server is not None, self.mcp_server is not None])
            },
            "configuration": {
                "fast_llm_model": self.amd.fast_llm_model,
                "complex_llm_model": self.amd.complex_llm_model,
                "use_fast_response": getattr(self.amd, 'use_fast_response', False),
                "max_input_tokens": getattr(self.amd, 'max_input_tokens', 8000),
                "persona_configured": self.amd.persona is not None,
                "format_config": bool(getattr(self.amd.persona, 'format_config', None)) if self.amd.persona else False
            }
        }

        # Add detailed execution summary if tasks exist
        tasks = self.shared.get("tasks", {})
        if tasks:
            task_types_used = {}
            tools_used = []
            execution_timeline = []

            for task_id, task in tasks.items():
                # Count task types
                task_type = getattr(task, 'type', 'unknown')
                task_types_used[task_type] = task_types_used.get(task_type, 0) + 1

                # Collect tools used
                if hasattr(task, 'tool_name') and task.tool_name:
                    tools_used.append(task.tool_name)

                # Timeline info
                if hasattr(task, 'started_at') and task.started_at:
                    timeline_entry = {
                        "task_id": task_id,
                        "type": task_type,
                        "started": task.started_at.isoformat(),
                        "status": getattr(task, 'status', 'unknown')
                    }
                    if hasattr(task, 'completed_at') and task.completed_at:
                        timeline_entry["completed"] = task.completed_at.isoformat()
                        timeline_entry["duration"] = (task.completed_at - task.started_at).total_seconds()
                    execution_timeline.append(timeline_entry)

            base_status["task_execution"].update({
                "task_types_used": task_types_used,
                "tools_used": list(set(tools_used)),
                "execution_timeline": execution_timeline[-5:]  # Last 5 tasks
            })

        # Add context statistics
        if hasattr(self.task_flow, 'context_manager'):
            context_manager = self.task_flow.context_manager
            base_status["memory_context"].update({
                "compression_threshold": context_manager.compression_threshold,
                "max_tokens": context_manager.max_tokens,
                "active_context_sessions": len(getattr(context_manager, 'session_managers', {}))
            })

        # Add variable system info
        if hasattr(self, 'variable_manager'):
            available_vars = self.variable_manager.get_available_variables()
            scope_info = self.variable_manager.get_scope_info()

            base_status["variable_system"] = {
                "total_scopes": len(scope_info),
                "scope_names": list(scope_info.keys()),
                "total_variables": sum(len(vars) for vars in available_vars.values()),
                "scope_details": {
                    scope: {"type": info["type"], "variables": len(available_vars.get(scope, {}))}
                    for scope, info in scope_info.items()
                }
            }

        # Add format quality info if available
        quality_assessment = self.shared.get("quality_assessment", {})
        if quality_assessment:
            quality_details = quality_assessment.get("quality_details", {})
            base_status["format_quality"] = {
                "overall_score": quality_details.get("total_score", 0.0),
                "format_adherence": quality_details.get("format_adherence", 0.0),
                "length_adherence": quality_details.get("length_adherence", 0.0),
                "content_quality": quality_details.get("base_quality", 0.0),
                "assessment": quality_assessment.get("quality_assessment", "unknown"),
                "has_suggestions": bool(quality_assessment.get("suggestions", []))
            }

        # Add LLM usage statistics
        llm_stats = self.shared.get("llm_call_stats", {})
        if llm_stats:
            base_status["llm_usage"] = {
                "total_calls": llm_stats.get("total_calls", 0),
                "context_compression_rate": llm_stats.get("context_compression_rate", 0.0),
                "average_context_tokens": llm_stats.get("context_tokens_used", 0) / max(llm_stats.get("total_calls", 1),
                                                                                        1),
                "total_tokens_used": llm_stats.get("total_tokens_used", 0)
            }

        # Add timestamp
        base_status["timestamp"] = datetime.now().isoformat()

        base_status["context_statistic"] = self.get_context_statistics()
        if not pretty_print:
            base_status["agent_context"] = await self.get_context_overview()
            return base_status

        # Pretty print using EnhancedVerboseOutput
        try:
            from toolboxv2.mods.isaa.extras.verbose_output import EnhancedVerboseOutput
            verbose_output = EnhancedVerboseOutput(verbose=True)

            # Header
            verbose_output.log_header(f"Agent Status: {base_status['agent_info']['name']}")

            # Runtime Status
            status_color = {
                "running": "SUCCESS",
                "paused": "WARNING",
                "idle": "INFO",
                "error": "ERROR"
            }.get(base_status["runtime_status"]["status"], "INFO")

            getattr(verbose_output, f"print_{status_color.lower()}")(
                f"Status: {base_status['runtime_status']['status'].upper()}"
            )

            # Task Execution Summary
            task_exec = base_status["task_execution"]
            if task_exec["total_tasks"] > 0:
                verbose_output.formatter.print_section(
                    "Task Execution",
                    f"Total: {task_exec['total_tasks']} | "
                    f"Completed: {task_exec['completed_tasks']} | "
                    f"Failed: {task_exec['failed_tasks']} | "
                    f"Active: {task_exec['active_tasks']}\n"
                    f"Adaptations: {task_exec['plan_adaptations']}"
                )

                if task_exec.get("tools_used"):
                    verbose_output.formatter.print_section(
                        "Tools Used",
                        ", ".join(task_exec["tools_used"])
                    )

            # Capabilities
            caps = base_status["capabilities"]
            verbose_output.formatter.print_section(
                "Capabilities",
                f"Intelligence Level: {caps['intelligence_level']}\n"
                f"Available Tools: {caps['available_tools']}\n"
                f"Analyzed Tools: {caps['analyzed_tools']}\n"
                f"World Model Size: {caps['world_model_size']}"
            )

            # Memory & Context
            memory = base_status["memory_context"]
            verbose_output.formatter.print_section(
                "Memory & Context",
                f"Context System: {memory['context_system']}\n"
                f"Session Managers: {memory['session_managers']}\n"
                f"Variable Scopes: {memory['variable_scopes']}\n"
                f"Session Initialized: {memory['session_initialized']}"
            )

            # Context Statistics
            stats = base_status["context_statistic"]
            verbose_output.formatter.print_section(
                "Context & Stats",
                f"Compression Stats: {stats['compression_stats']}\n"
                f"Session Usage: {stats['context_usage']}\n"
                f"Session Managers: {stats['session_managers']}\n"
            )

            # Configuration
            config = base_status["configuration"]
            verbose_output.formatter.print_section(
                "Configuration",
                f"Fast LLM: {config['fast_llm_model']}\n"
                f"Complex LLM: {config['complex_llm_model']}\n"
                f"Max Tokens: {config['max_input_tokens']}\n"
                f"Persona: {'Configured' if config['persona_configured'] else 'Default'}\n"
                f"Format Config: {'Active' if config['format_config'] else 'None'}"
            )

            # Performance
            perf = base_status["performance"]
            verbose_output.formatter.print_section(
                "Performance",
                f"Total Cost: ${perf['total_cost']:.4f}\n"
                f"Checkpointing: {'Enabled' if perf['checkpoint_enabled'] else 'Disabled'}\n"
                f"Max Parallel Tasks: {perf['max_parallel_tasks']}\n"
                f"Last Checkpoint: {perf['last_checkpoint'] or 'None'}"
            )

            # Variable System Details
            if "variable_system" in base_status:
                var_sys = base_status["variable_system"]
                scope_details = []
                for scope, details in var_sys["scope_details"].items():
                    scope_details.append(f"{scope}: {details['variables']} variables ({details['type']})")

                verbose_output.formatter.print_section(
                    "Variable System",
                    f"Total Scopes: {var_sys['total_scopes']}\n"
                    f"Total Variables: {var_sys['total_variables']}\n" +
                    "\n".join(scope_details)
                )

            # Format Quality
            if "format_quality" in base_status:
                quality = base_status["format_quality"]
                verbose_output.formatter.print_section(
                    "Format Quality",
                    f"Overall Score: {quality['overall_score']:.2f}\n"
                    f"Format Adherence: {quality['format_adherence']:.2f}\n"
                    f"Length Adherence: {quality['length_adherence']:.2f}\n"
                    f"Content Quality: {quality['content_quality']:.2f}\n"
                    f"Assessment: {quality['assessment']}"
                )

            # LLM Usage
            if "llm_usage" in base_status:
                llm = base_status["llm_usage"]
                verbose_output.formatter.print_section(
                    "LLM Usage Statistics",
                    f"Total Calls: {llm['total_calls']}\n"
                    f"Avg Context Tokens: {llm['average_context_tokens']:.1f}\n"
                    f"Total Tokens: {llm['total_tokens_used']}\n"
                    f"Compression Rate: {llm['context_compression_rate']:.2%}"
                )

            # Servers
            servers = base_status["servers"]
            if servers["server_count"] > 0:
                server_status = []
                if servers["a2a_server"]:
                    server_status.append("A2A Server: Active")
                if servers["mcp_server"]:
                    server_status.append("MCP Server: Active")

                verbose_output.formatter.print_section(
                    "Servers",
                    "\n".join(server_status)
                )

            verbose_output.print_separator()
            await self.get_context_overview(display=True)
            verbose_output.print_separator()
            verbose_output.print_info(f"Status generated at: {base_status['timestamp']}")

            return "Status printed above"

        except Exception:
            # Fallback to JSON if pretty print fails
            import json
            return json.dumps(base_status, indent=2, default=str)

    @property
    def tool_registry(self):
        return self._tool_registry

    def __rshift__(self, other):
        return Chain(self) >> other

    def __add__(self, other):
        return Chain(self) + other

    def __and__(self, other):
        return Chain(self) & other

    def __mod__(self, other):
        """Implements % operator for conditional branching"""
        return ConditionalChain(self, other)

    def bind(self, *agents, shared_scopes: list[str] = None, auto_sync: bool = True):
        """
        Bind two or more agents together with shared and private variable spaces.

        Args:
            *agents: FlowAgent instances to bind together
            shared_scopes: List of scope names to share (default: ['world', 'results', 'system'])
            auto_sync: Whether to automatically sync variables and context

        Returns:
            dict: Binding configuration with agent references
        """
        if shared_scopes is None:
            shared_scopes = ['world', 'results', 'system']

        return "binding_config"

    def unbind(self, preserve_shared_data: bool = False):
        """
        Unbind this agent from any binding configuration.

        Args:
            preserve_shared_data: Whether to preserve shared data in the agent after unbinding

        Returns:
            dict: Unbinding result with statistics
        """
        if not self.shared.get('is_bound', False):
            return {
                'success': False,
                'message': f"Agent {self.amd.name} is not currently bound to any other agents"
            }

        binding_config = self.shared.get('binding_config')
        if not binding_config:
            return {
                'success': False,
                'message': "No binding configuration found"
            }

        binding_id = binding_config['binding_id']
        bound_agents = binding_config['agents']

        unbind_stats = {
            'binding_id': binding_id,
            'agents_affected': [],
            'shared_data_preserved': preserve_shared_data,
            'private_data_restored': False,
            'unbind_timestamp': datetime.now().isoformat()
        }

        try:
            # Restore original managers for this agent
            if hasattr(self, '_original_managers'):
                original = self._original_managers

                if preserve_shared_data:
                    # Merge current shared data with original data
                    if isinstance(original['world_model'], dict):
                        original['world_model'].update(self.world_model)
                    if isinstance(original['shared'], dict):
                        original['shared'].update({k: v for k, v in self.shared.items()
                                                   if k not in ['binding_config', 'is_bound', 'binding_id',
                                                                'bound_agents']})

                # Restore original variable manager
                self.world_model = original['world_model']
                self.shared = original['shared']

                # Update context manager
                if hasattr(self, 'context_manager') and self.context_manager:
                    self.context_manager.variable_manager = self.variable_manager

                unbind_stats['private_data_restored'] = True
                del self._original_managers

            # Clean up binding state
            self.shared.pop('binding_config', None)
            self.shared.pop('is_bound', None)
            self.shared.pop('binding_id', None)
            self.shared.pop('bound_agents', None)

            # Update binding configuration to remove this agent
            remaining_agents = [agent for agent in bound_agents if agent != self]
            if remaining_agents:
                # Update binding config for remaining agents
                binding_config["agents"] = remaining_agents
                for agent in remaining_agents:
                    if hasattr(agent, "shared") and agent.shared.get("is_bound"):
                        agent.shared["bound_agents"] = [
                            a.amd.name for a in remaining_agents
                        ]

            unbind_stats["agents_affected"] = [agent.amd.name for agent in bound_agents]

            # Clean up sync handler if this was the last agent
            if len(remaining_agents) <= 1:
                sync_handler = binding_config.get("sync_handler")
                if sync_handler and hasattr(sync_handler, "cleanup"):
                    sync_handler.cleanup()

            rprint(
                f"Agent {self.amd.name} successfully unbound from binding {binding_id}"
            )
            rprint(f"Shared data preserved: {preserve_shared_data}")

            return {
                "success": True,
                "stats": unbind_stats,
                "message": f"Agent {self.amd.name} unbound successfully",
            }

        except Exception as e:
            eprint(f"Error during unbinding: {e}")
            return {"success": False, "error": str(e), "stats": unbind_stats}



class BindingSyncHandler:
    """Handles automatic synchronization between bound agents"""

    def __init__(self, binding_config: dict):
        self.binding_config = binding_config
        self.sync_queue = []
        self.last_sync = time.time()

    def cleanup(self):
        """Clean up sync handler resources"""
        self.sync_queue.clear()
        rprint(f"Binding sync handler for {self.binding_config['binding_id']} cleaned up")


def get_progress_summary(self) -> dict[str, Any]:
    """Get comprehensive progress summary from the agent"""
    if hasattr(self, "progress_tracker"):
        return self.progress_tracker.get_summary()
    return {"error": "No progress tracker available"}


import inspect
import typing
from collections.abc import Callable
from typing import Any


def get_args_schema(func: Callable) -> str:
    """
    Generate a string representation of a function's arguments and annotations.
    Keeps *args and **kwargs indicators and handles modern Python type hints.
    """
    sig = inspect.signature(func)
    parts = []

    for name, param in sig.parameters.items():
        ann = ""
        if param.annotation is not inspect._empty:
            ann = f": {_annotation_to_str(param.annotation)}"

        default = ""
        if param.default is not inspect._empty:
            default = f" = {repr(param.default)}"

        prefix = ""
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            prefix = "**"

        parts.append(f"{prefix}{name}{ann}{default}")

    return f"({', '.join(parts)})"


def _annotation_to_str(annotation: Any) -> str:
    """
    Convert any annotation to a nice string, including | union syntax (PEP 604),
    Optional[T], generics, and forward references.
    """
    if isinstance(annotation, str):
        return annotation  # Forward reference as-is

    # Handle typing.Optional and typing.Union
    if getattr(annotation, "__origin__", None) is typing.Union:
        args = annotation.__args__
        if len(args) == 2 and type(None) in args:
            non_none = args[0] if args[1] is type(None) else args[1]
            return f"Optional[{_annotation_to_str(non_none)}]"
        return " | ".join(_annotation_to_str(a) for a in args)

    # Handle built-in Union syntax (PEP 604)
    if (
        hasattr(annotation, "__args__")
        and getattr(annotation, "__origin__", None) is None
        and "|" in str(annotation)
    ):
        return str(annotation)

    # Handle generics like list[int], dict[str, Any]
    if getattr(annotation, "__origin__", None):
        origin = getattr(annotation.__origin__, "__name__", str(annotation.__origin__))
        args = getattr(annotation, "__args__", None)
        if args:
            return f"{origin}[{', '.join(_annotation_to_str(a) for a in args)}]"
        return origin

    # Handle normal classes and built-ins
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return repr(annotation)


from typing import Any


def _extract_meta_tool_calls(
    text: str, prefix="META_TOOL_CALL:"
) -> list[tuple[str, str]]:
    """Extract META_TOOL_CALL with proper bracket balance handling"""
    import re

    matches = []
    pattern = (
        r"META_TOOL_CALL:\s*(\w+)\("
        if prefix == "META_TOOL_CALL:"
        else r"TOOL_CALL:\s*(\w+)\("
    )

    for match in re.finditer(pattern, text):
        tool_name = match.group(1)
        start_pos = match.end() - 1  # Position of opening parenthesis

        # Find matching closing parenthesis with bracket balancing
        paren_count = 0
        pos = start_pos
        in_string = False
        string_char = None
        escape_next = False

        while pos < len(text):
            char = text[pos]

            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        # Found matching closing parenthesis
                        args_str = text[start_pos + 1 : pos]
                        matches.append((tool_name, args_str))
                        break
            else:  # in_string is True
                if char == string_char:
                    in_string = False
                    string_char = None

            pos += 1

    return matches


def _parse_tool_args(args_str: str) -> dict[str, Any]:
    """Parse tool arguments from string format with enhanced error handling"""
    import ast

    # Handle simple key=value format
    if "=" in args_str and not args_str.strip().startswith("{"):
        args = {}
        # Split by commas but handle nested structures
        parts = []
        current_part = ""
        bracket_count = 0
        in_quotes = False
        quote_char = None

        for char in args_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char in ["[", "{"] and not in_quotes:
                bracket_count += 1
            elif char in ["]", "}"] and not in_quotes:
                bracket_count -= 1
            elif char == "," and bracket_count == 0 and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
                continue

            current_part += char

        if current_part.strip():
            parts.append(current_part.strip())

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().strip('"').strip("'")
                value = value.strip()

                # Try to evaluate the value
                try:
                    if value.startswith("[") or value.startswith("{"):
                        args[key] = ast.literal_eval(value)
                    elif value.lower() in ["true", "false"]:
                        args[key] = value.lower() == "true"
                    elif value.replace(".", "").replace("-", "").isdigit():
                        args[key] = float(value) if "." in value else int(value)
                    else:
                        # Remove quotes if present
                        args[key] = value.strip('"').strip("'")
                except:
                    args[key] = value.strip('"').strip("'")

        return auto_unescape(args)

    # Handle JSON-like format
    try:
        return auto_unescape(ast.literal_eval(f"{{{args_str}}}"))
    except:
        return auto_unescape({"raw_args": args_str})


def unescape_string(text: str) -> str:
    """Universal string unescaping for any programming language."""
    if not isinstance(text, str) or len(text) < 2:
        return text

    # Remove outer quotes if wrapped
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1]

    # Universal escape sequences
    escapes = {
        "\\n": "\n",
        "\\t": "\t",
        "\\r": "\r",
        '\\"': '"',
        "\\'": "'",
        "\\\\": "\\",
    }

    for escaped, unescaped in escapes.items():
        text = text.replace(escaped, unescaped)

    return text


def needs_unescaping(text: str) -> bool:
    """Detect if string likely needs unescaping."""
    return bool(re.search(r'\\[ntr"\'\\]', text)) or len(text) > 50


def process_nested(data: Any, max_depth: int = 20) -> Any:
    """Recursively process nested structures, unescaping strings that need it."""
    if max_depth <= 0:
        return data

    if isinstance(data, dict):
        return {k: process_nested(v, max_depth - 1) for k, v in data.items()}

    elif isinstance(data, list | tuple):
        processed = [process_nested(item, max_depth - 1) for item in data]
        return type(data)(processed)

    elif isinstance(data, str) and needs_unescaping(data):
        return unescape_string(data)

    return data


def auto_unescape(args: Any) -> Any:
    """Automatically unescape all strings in nested data structure."""
    return process_nested(args)


from typing import Any, Callable
import inspect
import json


def convert_tool_to_litellm_format(
    tool_name: str,
    tool_func: Callable,
    description: str = None,
    tool_capabilities: dict = None
) -> dict:
    """
    Konvertiert ein einzelnes Tool in das LiteLLM/OpenAI Function Calling Format.

    Args:
        tool_name: Name des Tools
        tool_func: Die Tool-Funktion
        description: Optionale Beschreibung (überschreibt auto-generierte)
        tool_capabilities: Optionale erweiterte Capabilities aus der Analyse

    Returns:
        dict: Tool im LiteLLM Format
    """
    # Hole Signatur und Docstring
    sig = inspect.signature(tool_func)
    docstring = tool_func.__doc__ or ""

    # Beschreibung priorisieren
    final_description = description
    if not final_description and tool_capabilities:
        final_description = tool_capabilities.get("primary_function", "")
    if not final_description:
        final_description = docstring.split("\n")[0] if docstring else f"Execute {tool_name}"

    # Parameter extrahieren
    properties = {}
    required = []

    # LiteLLM Tool Format
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": final_description[:1024],  # Max 1024 chars für OpenAI
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


def convert_registry_to_litellm_tools(
    tool_registry: dict[str, dict],
    tool_capabilities: dict[str, dict] = None,
    filter_tools: list[str] = None
) -> list[dict]:
    """
    Konvertiert die gesamte Tool Registry in LiteLLM-kompatibles Format.

    Args:
        tool_registry: Die _tool_registry des Agents
        tool_capabilities: Die _tool_capabilities des Agents
        filter_tools: Optional - nur diese Tools konvertieren

    Returns:
        list[dict]: Liste von Tools im LiteLLM Format
    """
    tools = []
    tool_capabilities = tool_capabilities or {}

    for tool_name, tool_info in tool_registry.items():
        # Filter anwenden
        if filter_tools is not None and tool_name not in filter_tools:
            continue

        tool_func = tool_info.get("function")
        description = tool_info.get("description", "")
        capabilities = tool_capabilities.get(tool_name, {})

        if tool_func is None:
            continue

        try:
            litellm_tool = convert_tool_to_litellm_format(
                tool_name=tool_name,
                tool_func=tool_func,
                description=description,
                tool_capabilities=capabilities
            )
            tools.append(litellm_tool)
        except Exception as e:
            wprint(f"Failed to convert tool {tool_name}: {e}")

    return tools


def convert_meta_tools_to_litellm_format(
    meta_tools_registry: dict[str, dict] = None,
    include_standard: bool = True
) -> list[dict]:
    """
    Konvertiert Meta-Tools in LiteLLM-kompatibles Format.

    Args:
        meta_tools_registry: Custom Meta-Tools Registry
        include_standard: Standard Meta-Tools einschließen

    Returns:
        list[dict]: Liste von Meta-Tools im LiteLLM Format
    """
    tools = []

    # Standard Meta-Tools Definitionen
    standard_meta_tools = {
        "internal_reasoning": {
            "description": "Structure your thinking process with numbered thoughts, insights, and confidence levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Your current thought or analysis"},
                    "thought_number": {"type": "integer", "description": "Current thought number in sequence"},
                    "total_thoughts": {"type": "integer", "description": "Estimated total thoughts needed"},
                    "next_thought_needed": {"type": "boolean", "description": "Whether another thought is needed"},
                    "current_focus": {"type": "string", "description": "What aspect you're focusing on"},
                    "key_insights": {"type": "array", "items": {"type": "string"}, "description": "Key insights discovered"},
                    "potential_issues": {"type": "array", "items": {"type": "string"}, "description": "Potential issues identified"},
                    "confidence_level": {"type": "number", "description": "Confidence level 0.0-1.0"}
                },
                "required": ["thought", "thought_number", "total_thoughts", "next_thought_needed"]
            }
        },
        "manage_internal_task_stack": {
            "description": "Manage the internal task stack - add, complete, remove, or get current tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "complete", "remove", "get_current"],
                        "description": "Action to perform on task stack"
                    },
                    "task_description": {"type": "string", "description": "Description of the task"},
                    "outline_step_ref": {"type": "string", "description": "Reference to outline step"}
                },
                "required": ["action"]
            }
        },
        "delegate_to_llm_tool_node": {
            "description": "Delegate a task to the LLM Tool Node for external tool execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Description of what to accomplish"},
                    "tools_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tool names to use"
                    }
                },
                "required": ["task_description", "tools_list"]
            }
        },
        "advance_outline_step": {
            "description": "Mark current outline step as complete and advance to next step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_completed": {"type": "boolean", "description": "Whether current step is completed"},
                    "completion_evidence": {"type": "string", "description": "Evidence of completion"},
                    "next_step_focus": {"type": "string", "description": "Focus for the next step"}
                },
                "required": ["step_completed", "completion_evidence"]
            }
        },
        "write_to_variables": {
            "description": "Store data in the variable system for later use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Variable scope (results, delegation, user, files, reasoning)"},
                    "key": {"type": "string", "description": "Variable key/path"},
                    "value": {"type": "string", "description": "Value to store"},
                    "description": {"type": "string", "description": "Description of what this variable contains"}
                },
                "required": ["scope", "key", "value"]
            }
        },
        "read_from_variables": {
            "description": "Read data from the variable system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Variable scope to read from"},
                    "key": {"type": "string", "description": "Variable key/path to read"},
                    "purpose": {"type": "string", "description": "Why you need this data"}
                },
                "required": ["scope", "key"]
            }
        },
        "direct_response": {
            "description": "Provide the final answer when all outline steps are completed. ONLY use when finished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string", "description": "The complete final answer to the user's query"},
                    "outline_completion": {"type": "boolean", "description": "Confirm outline is complete"},
                    "steps_completed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of completed steps"
                    }
                },
                "required": ["final_answer", "outline_completion"]
            }
        }
    }

    # Standard Meta-Tools hinzufügen
    if include_standard:
        for tool_name, tool_def in standard_meta_tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_def["description"],
                    "parameters": tool_def["parameters"]
                }
            })

    # Custom Meta-Tools hinzufügen
    if meta_tools_registry:
        for tool_name, tool_info in meta_tools_registry.items():
            if tool_name in standard_meta_tools:
                continue  # Bereits hinzugefügt

            tool_func = tool_info.get("function")
            description = tool_info.get("description", f"Custom meta-tool: {tool_name}")

            if tool_func:
                try:
                    litellm_tool = convert_tool_to_litellm_format(
                        tool_name=tool_name,
                        tool_func=tool_func,
                        description=description
                    )
                    tools.append(litellm_tool)
                except Exception as e:
                    wprint(f"Failed to convert meta-tool {tool_name}: {e}")

    return tools


def get_selected_tools_litellm(
    tool_registry: dict[str, dict],
    tool_capabilities: dict[str, dict],
    selected_tools: list[str]
) -> list[dict]:
    """
    Gibt nur ausgewählte Tools im LiteLLM Format zurück.

    Args:
        tool_registry: Die _tool_registry des Agents
        tool_capabilities: Die _tool_capabilities des Agents
        selected_tools: Liste der Tool-Namen die konvertiert werden sollen

    Returns:
        list[dict]: Gefilterte Tools im LiteLLM Format
    """
    return convert_registry_to_litellm_tools(
        tool_registry=tool_registry,
        tool_capabilities=tool_capabilities,
        filter_tools=selected_tools
    )

# Add this method to FlowAgent class
FlowAgent.get_progress_summary = get_progress_summary

# Example usage and tests
async def tchains():
    class CustomFormat(BaseModel):
        value: str

    print("=== Testing Basic Chain ===")
    agent_a = FlowAgent(AgentModelData(name="A"))
    agent_b = FlowAgent(AgentModelData(name="B"))
    agent_c = FlowAgent(AgentModelData(name="C"))

    async def a_run(self, query: str):
        print(f"FlowAgent {self.amd.name} running query: {query}")
        return f"Answer from {self.amd.name}"

    async def a_format_class(self, pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 2):
        print(f"FlowAgent {self.amd.name} formatting class: {pydantic_model.__name__}")
        return {"value": 'yes' if random.random() < 0.5 else 'no'}
    agent_a.a_run = types.MethodType(a_run, agent_a)
    agent_a.a_format_class = types.MethodType(a_format_class, agent_a)
    agent_b.a_run = types.MethodType(a_run, agent_b)
    agent_b.a_format_class = types.MethodType(a_format_class, agent_b)
    agent_c.a_run = types.MethodType(a_run, agent_c)
    agent_c.a_format_class =types.MethodType(a_format_class, agent_c)

    # Basic sequential chain
    c = agent_a >> agent_b
    result = await c.a_run("Hello World")
    print(f"Result: {result}\n")
    c.print_graph()

    # Three agent chain
    c = agent_a >> agent_c >> agent_b
    result = await c.a_run("Hello World")
    print(f"Three agent result: {result}\n")
    c.print_graph()

    print("=== Testing Format Chain ===")
    # Chain with formatting
    c = CF(CustomFormat) >> agent_a >> CF(CustomFormat) >> agent_b
    result = await c.a_run(CustomFormat(value="Hello World"))
    print(f"Format chain result: {result}\n")
    c.print_graph()

    print("=== Testing Parallel Execution ===")
    # Parallel execution
    c = agent_a + agent_b
    result = await c.a_run("Hello World")
    print(f"Parallel result: {result}\n")
    c.print_graph()

    print("=== Testing Mixed Chain ===")
    # Mixed parallel and sequential
    c = (agent_a & agent_b) >> CF(CustomFormat)
    result = await c.a_run("Hello World")
    print(f"Mixed chain result: {result}\n")
    c.print_graph()

    print("=== Testing Mixed Chain v2 ===")
    # Mixed parallel and sequential
    c = (agent_a >> agent_c >> agent_b & agent_b) >> CF(CustomFormat)
    result = await c.a_run("Hello World")
    print(f"Mixed chain result: {result}\n")
    c.print_graph()

    i = 0
    c: Chain = agent_a >> agent_b
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> agent_c >> agent_b
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = CF(CustomFormat) >> agent_a >> CF(
        CustomFormat) >> agent_b  # using a_format_class intelligently defalt all
    result = await c.a_run(CustomFormat(value="Hello World"))
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(CustomFormat) >> agent_b  # using a_format_class intelligently same as above
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - '*' >> agent_b  # using a_format_class intelligently same as above
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - 'value' >> agent_b  # using a_format_class intelligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - '*value' >> agent_b  # using a_format_class intelligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - ('value', 'value2') >> agent_b  # using a_format_class intelligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(
        CustomFormat) - 'value[n]' >> agent_b  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(CustomFormat) - IS('value',
                                                'yes') >> agent_b % agent_c  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    print("Cinc")
    chain_x = agent_b >> CF(CustomFormat)
    chain_z = agent_c >> CF(CustomFormat)

    c: Chain = agent_a >> CF(CustomFormat) - IS('value',
                                                'yes') >> chain_x % chain_z  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World IS 12")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(CustomFormat) - IS('value', 'yes') >> agent_b + agent_c | CF(
        CustomFormat) - 'error_reson_val_from_agent_a' >> agent_c  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a + agent_b  # runs a and p in parallel combines output inteligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a & agent_b  # same as above
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a & agent_b >> CF(CustomFormat) - 'value[n]' >> agent_b  # runs agent b n times parallel with different input
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    print("=== Testing Done ===")


async def run_new_custom_chain():
    # --- Agenten-Setup ---

    class CustomFormat(BaseModel):
        value: str
    # (Wir gehen davon aus, dass die Agenten wie im Beispiel-Code definiert sind)
    supervisor_agent = FlowAgent(AgentModelData(name="Supervisor"))
    writer_agent = FlowAgent(AgentModelData(name="Writer"))
    reviewer_agent = FlowAgent(AgentModelData(name="Reviewer"))
    notifier_agent = FlowAgent(AgentModelData(name="Notifier"))

    # Weisen Sie den Agenten die beispielhaften a_run und a_format_class Methoden zu
    # (Dieser Code ist der gleiche wie in Ihrem Beispiel)
    async def a_run(self, query: str):
        print(f"FlowAgent {self.amd.name} running query: {query}")
        return f"Answer from {self.amd.name}"

    async def a_format_class(self, pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 2):
        print(f"FlowAgent {self.amd.name} formatting class: {pydantic_model.__name__}")
        # Simuliert eine zufällige Entscheidung
        decision = 'yes' if random.random() < 0.5 else 'no'
        print(f"--> Decision made: {decision}")
        return {"value": decision}

    for agent in [supervisor_agent, writer_agent, reviewer_agent, notifier_agent]:
        agent.a_run = types.MethodType(a_run, agent)
        agent.a_format_class = types.MethodType(a_format_class, agent)

    # --- Die neue übersichtliche Test-Chain ---
    # Logik: Supervisor -> Entscheidung -> (Writer + Reviewer) ODER nichts -> Notifier
    conditional_parallel_chain = (writer_agent + reviewer_agent)

    # Erstellen der vollständigen Kette
    # Wenn der Wert 'yes' ist, führe die conditional_parallel_chain aus.
    # % notifier_agent bedeutet: Wenn die Bedingung nicht erfüllt ist, gehe direkt zu diesem Agenten.
    # Der letzte >> notifier_agent stellt sicher, dass der Notifier immer am Ende läuft (sowohl für den 'yes'- als auch für den 'no'-Pfad).
    c: Chain = supervisor_agent >> CF(CustomFormat) - IS('value', 'yes') >> conditional_parallel_chain % notifier_agent >> notifier_agent

    print("--- Start: Neue Test-Chain ---")
    result = await c.a_run("Start the content creation process")
    print(f"\nFinal Result of the Chain: {result}\n")
    c.print_graph()
    print("--- Ende: Neue Test-Chain ---")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_new_custom_chain())

if __name__ == "__main__2":


    # Simple test
    async def _agent():
        amd = AgentModelData(
        name="TestAgent",
        fast_llm_model="groq/llama-3.3-70b-versatile",
        complex_llm_model="openrouter/qwen/qwen3-coder",
        persona=PersonaConfig(
            name="Isaa",
            style="light and perishes",
            tone="modern friendly",
            personality_traits=["intelligent", "autonomous", "duzen", "not formal"],
            custom_instructions="dos not like to Talk in to long sanitize and texts."
            )
        )
        agent = FlowAgent(amd, verbose=True)

        # Load latest checkpoint with full history restoration
        result = await agent.load_latest_checkpoint(auto_restore_history=True, max_age_hours=24)

        if result["success"]:
            print(f"Loaded checkpoint from {result['checkpoint_timestamp']}")
            print(f"Restored {result['restore_stats']['conversation_history_entries']} conversation entries")
            print(f"Restored {result['restore_stats']['tasks_restored']} tasks")
            print(f"Restored {result['restore_stats']['world_model_entries']} world model entries")
        else:
            print(f"Failed to load checkpoint: {result['error']}")

        # List available checkpoints
        checkpoints = agent.list_available_checkpoints(max_age_hours=168)  # 1 week
        for cp in checkpoints:
            print(f"Checkpoint: {cp['filename']} (age: {cp['age_hours']}h, size: {cp['file_size_kb']}kb)")

        # Clean up old checkpoints
        cleanup_result = await agent.delete_old_checkpoints(keep_count=5, max_age_hours=168)
        print(f"Deleted {cleanup_result['deleted_count']} old checkpoints, freed {cleanup_result['freed_space_kb']}kb")

        def get_user_name():
            return "Markin"

        print(agent.get_available_variables())
        await agent.add_tool(get_user_name, "get_user_name", "Get the user's name")
        print("online")
        import time
        t = time.perf_counter()
        response = await agent.a_run("is 1980 45 years ago?")
        print(f"Time: {time.perf_counter() - t}")
        print(f"Response: {response}")
        await agent.status(pretty_print=True)

        while True:
            query = input("Query: ")
            if query == "exit":
                break
            response = await agent.a_run(query)
            print(f"Response: {response}")

        await agent.close()

    asyncio.run(_agent())

"""
Agent V2

agent exposes.

a_run(task, session_id, remember, context, **kwargs) -> str

a_stream(task, session_id, remember, context, **kwargs) -> AsyncGenerator[str, None]

a_format_class(pydantic_model, prompt, context, **kwargs) -> dict[str, Any]

# Auto
save and load history and progress data (chapoints)

"""
